#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp2.common import RESULTS_ROOT, command_manifest, ensure_dir, timestamp_now, write_json  # noqa: E402
from reinforce_exp2.scripts._carrier_sets import (  # noqa: E402
    build_static_carrier_sets,
    dump_alignment_table,
    dump_carrier_diagnostics,
    sequence_hash,
)
from reinforce_exp2.scripts._shared import emit_core_artifacts, holm_adjust_dict, parse_models_arg, safe_float  # noqa: E402
from experiment2.execution import _evaluate_example_from_token_logits  # noqa: E402
from experiment2.tasks import TaskExample, build_token_pools, generate_task_examples  # noqa: E402
from experiment3.theory1_si_circuits import MODELS, HeadID, head_output_ablation  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402


def _all_heads(model) -> list[HeadID]:
    heads: list[HeadID] = []
    n_layers = int(len(model.model.layers))
    for l in range(n_layers):
        attn = model.model.layers[l].self_attn
        n_heads = None
        for attr in ("num_heads", "n_heads", "num_attention_heads"):
            if hasattr(attn, attr):
                try:
                    n_heads = int(getattr(attn, attr))
                    break
                except Exception:
                    pass
        if n_heads is None and hasattr(model, "config") and hasattr(model.config, "num_attention_heads"):
            try:
                n_heads = int(model.config.num_attention_heads)
            except Exception:
                n_heads = None
        if n_heads is None and hasattr(attn, "q_proj") and hasattr(attn.q_proj, "out_features"):
            try:
                head_dim = int(getattr(attn, "head_dim", 0) or 0)
                if head_dim <= 0 and hasattr(model, "config") and hasattr(model.config, "hidden_size") and hasattr(model.config, "num_attention_heads"):
                    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
                n_heads = int(attn.q_proj.out_features // max(head_dim, 1))
            except Exception:
                n_heads = None
        if n_heads is None or n_heads <= 0:
            raise RuntimeError(f"Unable to infer num heads for layer {l} ({type(attn).__name__})")
        for h in range(n_heads):
            heads.append(HeadID(int(l), int(h)))
    return heads


def _load_head_sets(
    *,
    model_name: str,
    tokenizer,
    pools,
    seq_len: int,
    seed: int,
    out_dir: Path,
    eval_sequence_hashes: set[str] | None = None,
) -> dict[str, Any]:
    carrier = build_static_carrier_sets(
        model_name=model_name,
        tokenizer=tokenizer,
        pools=pools,
        seq_len=max(128, int(seq_len)),
        seed=int(seed),
        eval_sequence_hashes=set(eval_sequence_hashes or set()),
        n_calibration=96,
        max_pairs_per_sequence=4000,
        n_offset_bins=8,
        data_quality_max_missing=0.05,
    )
    dump_alignment_table(out_dir / f"{model_name}_content_alignment.parquet", carrier["alignment_table"])
    dump_carrier_diagnostics(out_dir / f"{model_name}_carrier_set_diagnostics.json", carrier["diagnostics"])
    return {
        "SI": carrier["SI"],
        "LowSI": carrier["LowSI"],
        "ContentCond": carrier["ContentCond"],
        "_diag": carrier["diagnostics"],
    }


def _load_mean_si_r2(model_name: str) -> float:
    p = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_r2_summary.parquet"
    if not p.exists():
        return float("nan")
    df = pd.read_parquet(p)
    if "mean_r2" not in df.columns and "r2" in df.columns:
        df = df.rename(columns={"r2": "mean_r2"})
    if "mean_r2" not in df.columns:
        return float("nan")
    return float(np.nanmean(df["mean_r2"].to_numpy(dtype=np.float64)))


def _si_support_diagnostic(model_name: str, si_heads: list[HeadID]) -> float:
    import json

    p = ROOT / "results" / "experiment3" / "theory8_position_ablation" / model_name / "estimated_kernels.json"
    if not p.exists() or not si_heads:
        return float("nan")
    raw = json.loads(p.read_text(encoding="utf-8"))
    vals = []
    for h in si_heads:
        k = f"L{int(h.layer)}H{int(h.head)}"
        v = raw.get(k)
        if v is None:
            continue
        arr = np.abs(np.asarray(v, dtype=np.float64))
        if arr.size < 8:
            continue
        lo = int(0.75 * arr.size)
        vals.append(float(np.sum(arr[lo:]) / max(np.sum(arr), 1e-12)))
    return float(np.nanmean(np.asarray(vals, dtype=np.float64))) if vals else float("nan")


def _shuffle_prequery_fraction(ex: TaskExample, strength: float, rng: np.random.Generator) -> TaskExample:
    toks = list(ex.tokens)
    targets = sorted(int(x) for x in ex.target_positions)
    if not targets:
        return ex
    q = int(min(targets))
    if q <= 3 or strength <= 0:
        return ex
    idx = np.arange(0, q, dtype=np.int64)
    n_perm = int(round(float(strength) * len(idx)))
    n_perm = max(0, min(len(idx), n_perm))
    if n_perm <= 1:
        return ex
    perm_idx = rng.choice(idx, size=n_perm, replace=False)
    perm_src = perm_idx.copy()
    rng.shuffle(perm_src)
    new_toks = list(toks)
    for dst, src in zip(perm_idx.tolist(), perm_src.tolist()):
        new_toks[int(dst)] = toks[int(src)]
    return TaskExample(
        id=f"{ex.id}:shuffle:{strength:.2f}",
        task_name=str(ex.task_name),
        tokens=[int(x) for x in new_toks],
        target_positions=list(ex.target_positions),
        target_tokens=list(ex.target_tokens),
        dependency_span=int(ex.dependency_span),
        task_class=str(ex.task_class),
        seed=int(ex.seed),
        model=str(ex.model),
        length=int(ex.length),
        task_params=dict(ex.task_params),
        pair_count=ex.pair_count,
        query_key=ex.query_key,
        distractor_key=ex.distractor_key,
        match_rule=str(ex.match_rule),
        has_no_match=bool(ex.has_no_match),
    )


def _evaluate_examples(
    model,
    device: str,
    examples: list[TaskExample],
    pools,
    candidate_size: int,
    batch_size: int,
) -> np.ndarray:
    vals: list[float] = []
    bs = max(1, int(batch_size))
    pos = 0
    while pos < len(examples):
        batch = examples[pos : pos + bs]
        try:
            input_ids = torch.tensor([ex.tokens for ex in batch], dtype=torch.long, device=device)
            with torch.inference_mode():
                logits = model(input_ids=input_ids, use_cache=False).logits
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                continue
            raise

        for i, ex in enumerate(batch):
            met, _ = _evaluate_example_from_token_logits(
                logits[i],
                ex,
                split="synthetic",
                pools=pools,
                synthetic_eval_mode="restricted",
                candidate_size=int(candidate_size),
                candidate_policy_version="restricted_candidates_v1_structured_first",
            )
            vals.append(float(met["accuracy"]))

        pos += len(batch)
        del input_ids, logits
        torch.cuda.empty_cache()
    return np.asarray(vals, dtype=np.float64)


def _ols_beta_with_p(y: np.ndarray, x_cols: list[np.ndarray], beta_index: int) -> dict[str, float]:
    yy = np.asarray(y, dtype=np.float64)
    X = np.column_stack([np.asarray(c, dtype=np.float64) for c in x_cols])
    keep = np.isfinite(yy) & np.all(np.isfinite(X), axis=1)
    yy = yy[keep]
    X = X[keep]
    n, p = X.shape
    if n <= p + 1:
        return {"beta": float("nan"), "se": float("nan"), "t": float("nan"), "p_two": float("nan"), "p_one_gt0": float("nan")}

    try:
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        resid = yy - X @ beta
        dof = max(1, n - p)
        sigma2 = float(np.sum(resid**2) / dof)
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = float(math.sqrt(max(sigma2 * xtx_inv[beta_index, beta_index], 1e-12)))
        t = float(beta[beta_index] / max(se, 1e-12))
        p_two = float(2.0 * (1.0 - scipy_stats.t.cdf(abs(t), dof)))
        p_one = float(p_two / 2.0) if t > 0 else float(1.0 - p_two / 2.0)
        return {"beta": float(beta[beta_index]), "se": se, "t": t, "p_two": p_two, "p_one_gt0": p_one}
    except Exception:
        return {"beta": float("nan"), "se": float("nan"), "t": float("nan"), "p_two": float("nan"), "p_one_gt0": float("nan")}


def _fixed_effect_meta_gt0(betas: list[float], ses: list[float]) -> dict[str, float]:
    b = np.asarray(betas, dtype=np.float64)
    s = np.asarray(ses, dtype=np.float64)
    keep = np.isfinite(b) & np.isfinite(s) & (s > 0)
    if np.sum(keep) < 2:
        return {"beta": float("nan"), "se": float("nan"), "z": float("nan"), "p_one_gt0": float("nan"), "n_models": int(np.sum(keep))}
    w = 1.0 / (s[keep] ** 2)
    mu = float(np.sum(w * b[keep]) / np.sum(w))
    se = float(math.sqrt(1.0 / np.sum(w)))
    z = float(mu / max(se, 1e-12))
    p = float(1.0 - scipy_stats.norm.cdf(z))
    return {"beta": mu, "se": se, "z": z, "p_one_gt0": p, "n_models": int(np.sum(keep))}


def _sign_flip_one_sided_gt_zero(values: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n < 2:
        return float("nan")
    obs = float(np.mean(v))
    if n <= 15:
        stats = []
        for signs in itertools.product([-1.0, 1.0], repeat=n):
            s = np.asarray(signs, dtype=np.float64)
            stats.append(float(np.mean(s * v)))
        arr = np.asarray(stats, dtype=np.float64)
        return float(np.sum(arr >= obs) / len(arr))
    rng = np.random.default_rng(int(seed))
    m = max(2000, int(n_perm))
    hits = 0
    for _ in range(m):
        s = rng.choice(np.asarray([-1.0, 1.0]), size=n, replace=True)
        stat = float(np.mean(s * v))
        if np.isfinite(stat) and stat >= obs:
            hits += 1
    return float((hits + 1) / (m + 1))


def _spearman_assoc_with_exact_small_n(x: np.ndarray, y: np.ndarray) -> dict[str, float | str]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[keep]
    yy = yy[keep]
    n = int(xx.size)
    if n < 3:
        return {"spearman_rho": float("nan"), "p_two": float("nan"), "p_method": "insufficient_n"}

    rho_obs, p_asym = scipy_stats.spearmanr(xx, yy)
    rho_obs = safe_float(rho_obs)
    p_asym = safe_float(p_asym)
    if not np.isfinite(rho_obs):
        return {"spearman_rho": float("nan"), "p_two": float("nan"), "p_method": "undefined"}

    if n <= 8:
        idx = list(range(n))
        vals = []
        for perm in itertools.permutations(idx):
            yp = yy[np.asarray(perm, dtype=np.int64)]
            r, _ = scipy_stats.spearmanr(xx, yp)
            r = safe_float(r)
            if np.isfinite(r):
                vals.append(float(r))
        if not vals:
            return {"spearman_rho": rho_obs, "p_two": float("nan"), "p_method": "exact_failed"}
        arr = np.asarray(vals, dtype=np.float64)
        p_two = float(np.mean(np.abs(arr) >= abs(rho_obs)))
        return {"spearman_rho": rho_obs, "p_two": p_two, "p_method": "exact_permutation"}
    return {"spearman_rho": rho_obs, "p_two": p_asym, "p_method": "asymptotic"}


def _model_ctx_defaults(model_name: str) -> tuple[int, int]:
    # (L_trainlike, conservative max_context default)
    mapping = {
        "llama-3.1-8b": (8192, 8192),
        "olmo-2-7b": (2048, 4096),
        "mistral-7b-v0.1": (8192, 8192),
    }
    return mapping.get(model_name, (2048, 4096))


def _run_c21_for_model(
    *,
    model_name: str,
    device: str,
    seed: int,
    n_examples: int,
    seq_len: int,
    candidate_size: int,
    batch_size: int,
    shuffle_strengths: list[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if device == "auto":
        loaded = load_model(MODELS[model_name], device_map="auto")
        model = loaded.model
        device = str(next(model.parameters()).device)
    else:
        loaded = load_model(MODELS[model_name])
        model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(MODELS[model_name])

    special_ids = [getattr(tokenizer, attr, None) for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")]
    special_ids = [int(x) for x in special_ids if x is not None]
    pools = build_token_pools(model_name, int(tokenizer.vocab_size), special_ids)

    n_a = max(8, int(n_examples // 2))
    n_b = max(8, int(n_examples - n_a))
    ex_a = generate_task_examples(
        task_name="local_key_match",
        model_name=model_name,
        seq_len=max(128, int(seq_len)),
        seed=int(seed),
        count=n_a,
        pools=pools,
    )
    try:
        ex_b = generate_task_examples(
            task_name="long_range_retrieval",
            model_name=model_name,
            seq_len=max(128, int(seq_len)),
            seed=int(seed) + 91,
            count=n_b,
            pools=pools,
            span_override=max(32, int(seq_len // 8)),
            span_choices=(max(32, int(seq_len // 8)),),
        )
    except Exception:
        ex_b = generate_task_examples(
            task_name="local_key_match",
            model_name=model_name,
            seq_len=max(128, int(seq_len)),
            seed=int(seed) + 91,
            count=n_b,
            pools=pools,
        )
    examples = ex_a + ex_b

    clean = _evaluate_examples(model, device, examples, pools, candidate_size=candidate_size, batch_size=batch_size)
    lengths = np.asarray([int(ex.length) for ex in examples], dtype=np.float64)
    difficulty = 1.0 - clean

    rows = []
    rng = np.random.default_rng(int(seed) + 177)
    for s in shuffle_strengths:
        shuffled = [_shuffle_prequery_fraction(ex, float(s), rng) for ex in examples]
        acc = _evaluate_examples(model, device, shuffled, pools, candidate_size=candidate_size, batch_size=batch_size)
        d = (clean - acc) / np.maximum(clean, 1e-6)
        for i in range(len(examples)):
            rows.append(
                {
                    "model": model_name,
                    "example_idx": int(i),
                    "shuffle_strength": float(s),
                    "seq_length": float(lengths[i]),
                    "a_clean": float(clean[i]),
                    "a_shuffle": float(acc[i]),
                    "difficulty_proxy": float(difficulty[i]),
                    "degradation_norm": float(d[i]),
                }
            )
    df = pd.DataFrame(rows)

    # within-model slope
    y = df["degradation_norm"].to_numpy(dtype=np.float64)
    s = df["shuffle_strength"].to_numpy(dtype=np.float64)
    ones = np.ones_like(s)
    unadj = _ols_beta_with_p(y, [ones, s], beta_index=1)

    length_z = (df["seq_length"].to_numpy(dtype=np.float64) - np.nanmean(df["seq_length"])) / max(np.nanstd(df["seq_length"]), 1e-8)
    diff_z = (df["difficulty_proxy"].to_numpy(dtype=np.float64) - np.nanmean(df["difficulty_proxy"])) / max(np.nanstd(df["difficulty_proxy"]), 1e-8)
    adj = _ols_beta_with_p(y, [ones, s, length_z, diff_z], beta_index=1)

    summary = {
        "model": model_name,
        "n_rows": int(df.shape[0]),
        "n_examples": int(len(examples)),
        "shuffle_strengths": [float(x) for x in shuffle_strengths],
        "beta_shuffle": unadj,
        "beta_shuffle_adj": adj,
        "mean_si_r2": _load_mean_si_r2(model_name),
    }
    return df, summary


def _run_c22_for_model(
    *,
    model_name: str,
    device: str,
    seed: int,
    n_examples: int,
    seq_len: int,
    candidate_size: int,
    batch_size: int,
    max_seq_len_eval: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if device == "auto":
        loaded = load_model(MODELS[model_name], device_map="auto")
        model = loaded.model
        device = str(next(model.parameters()).device)
    else:
        loaded = load_model(MODELS[model_name])
        model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(MODELS[model_name])

    model_ctx = int(getattr(getattr(model, "config", None), "max_position_embeddings", 0) or 0)
    l_trainlike, fallback_max = _model_ctx_defaults(model_name)
    max_ctx = min(max(256, model_ctx if model_ctx > 0 else fallback_max), max(256, int(max_seq_len_eval)))

    multipliers = [0.5, 1.0, 1.25]
    if 2.0 * l_trainlike <= max_ctx:
        multipliers.append(2.0)

    special_ids = [getattr(tokenizer, attr, None) for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")]
    special_ids = [int(x) for x in special_ids if x is not None]
    pools = build_token_pools(model_name, int(tokenizer.vocab_size), special_ids)

    examples_by_multiplier: dict[float, list[TaskExample]] = {}
    eval_hashes: set[str] = set()
    # Pre-generate all evaluation examples first so carrier-set construction can
    # enforce calibration/evaluation disjointness.
    for midx, mult in enumerate(multipliers):
        target_len = int(round(l_trainlike * float(mult)))
        seq_len_local = min(max_ctx, max(128, target_len))
        try:
            examples = generate_task_examples(
                task_name="long_range_retrieval",
                model_name=model_name,
                seq_len=seq_len_local,
                seed=int(seed) + midx * 43,
                count=max(32, int(n_examples)),
                pools=pools,
                span_override=max(32, int(seq_len_local // 8)),
                span_choices=(max(32, int(seq_len_local // 8)),),
            )
        except Exception:
            examples = generate_task_examples(
                task_name="local_key_match",
                model_name=model_name,
                seq_len=seq_len_local,
                seed=int(seed) + midx * 43,
                count=max(32, int(n_examples)),
                pools=pools,
            )
        examples_by_multiplier[float(mult)] = examples
        for ex in examples:
            eval_hashes.add(sequence_hash([int(x) for x in ex.tokens]))

    settings = _load_head_sets(
        model_name=model_name,
        tokenizer=tokenizer,
        pools=pools,
        seq_len=max(128, int(seq_len)),
        seed=int(seed) + 4001,
        out_dir=out_dir,
        eval_sequence_hashes=eval_hashes,
    )
    all_heads = _all_heads(model)
    keep_sets = {
        "SI": settings.get("SI", []),
        "ContentCond": settings.get("ContentCond", []),
    }

    rows = []
    # per-context evaluations
    for mult in multipliers:
        examples = examples_by_multiplier.get(float(mult), [])
        if not examples:
            continue
        seq_len_local = int(examples[0].length)

        for carrier, keep in keep_sets.items():
            keep_keys = {f"L{int(h.layer)}H{int(h.head)}" for h in keep}
            ablate = [h for h in all_heads if f"L{int(h.layer)}H{int(h.head)}" not in keep_keys]
            with head_output_ablation(model, ablate):
                acc = _evaluate_examples(model, device, examples, pools, candidate_size=candidate_size, batch_size=batch_size)
            for i, ex in enumerate(examples):
                rows.append(
                    {
                        "model": model_name,
                        "carrier_class": carrier,
                        "context_multiplier": float(mult),
                        "context_length": int(seq_len_local),
                        "example_idx": int(i),
                        "accuracy": float(acc[i]),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"model": model_name, "status": "empty"}

    # normalized long-context drop per class relative to 0.5x
    drop_rows = []
    for carrier, cdf in df.groupby("carrier_class"):
        base = cdf[cdf["context_multiplier"] == 0.5]["accuracy"].to_numpy(dtype=np.float64)
        base_mu = float(np.nanmean(base)) if len(base) else float("nan")
        for mult, mdf in cdf.groupby("context_multiplier"):
            acc = mdf["accuracy"].to_numpy(dtype=np.float64)
            mu = float(np.nanmean(acc)) if len(acc) else float("nan")
            d = (base_mu - mu) / max(base_mu, 1e-6) if np.isfinite(base_mu) else float("nan")
            drop_rows.append(
                {
                    "model": model_name,
                    "carrier_class": str(carrier),
                    "context_multiplier": float(mult),
                    "base_accuracy_0_5x": base_mu,
                    "accuracy_mean": mu,
                    "normalized_drop": d,
                }
            )
    drop_df = pd.DataFrame(drop_rows)

    # interaction model on aggregated drops
    cls = np.where(drop_df["carrier_class"].astype(str).to_numpy() == "SI", 1.0, 0.0)
    cm = drop_df["context_multiplier"].to_numpy(dtype=np.float64)
    y = drop_df["normalized_drop"].to_numpy(dtype=np.float64)
    ones = np.ones_like(cm)
    inter = cls * cm
    fit = _ols_beta_with_p(y, [ones, cm, cls, inter], beta_index=3)

    # practical endpoint at target long point
    long_points = sorted([x for x in drop_df["context_multiplier"].unique().tolist() if float(x) > 1.0])
    target_point = 1.25 if 1.25 in long_points else (max(long_points) if long_points else float("nan"))
    endpoint_diff = float("nan")
    if np.isfinite(target_point):
        s = drop_df[(drop_df["carrier_class"] == "SI") & (drop_df["context_multiplier"] == target_point)]["normalized_drop"]
        c = drop_df[(drop_df["carrier_class"] == "ContentCond") & (drop_df["context_multiplier"] == target_point)]["normalized_drop"]
        if len(s) and len(c):
            endpoint_diff = float(np.nanmean(s.to_numpy(dtype=np.float64)) - np.nanmean(c.to_numpy(dtype=np.float64)))

    # diagnostics for failure assignment
    k_support = _si_support_diagnostic(model_name, settings.get("SI", []))
    content_curve = drop_df[drop_df["carrier_class"] == "ContentCond"].sort_values("context_multiplier")
    if content_curve.shape[0] >= 2:
        cc_x = content_curve["context_multiplier"].to_numpy(dtype=np.float64)
        cc_y = content_curve["accuracy_mean"].to_numpy(dtype=np.float64)
        beta_c = _ols_beta_with_p(cc_y, [np.ones_like(cc_x), cc_x], beta_index=1)["beta"]
    else:
        beta_c = float("nan")

    summary = {
        "model": model_name,
        "L_trainlike": int(l_trainlike),
        "max_context_used": int(max_ctx),
        "tested_multipliers": [float(x) for x in sorted(df["context_multiplier"].unique().tolist())],
        "confirmatory_eligible": bool(np.any(df["context_multiplier"].to_numpy(dtype=np.float64) > 1.0)),
        "carrier_set_diagnostics": settings.get("_diag", {}),
        "calibration_eval_disjoint": bool(settings.get("_diag", {}).get("calibration_eval_disjoint", True)),
        "interaction_beta_int": fit,
        "endpoint_target_multiplier": safe_float(target_point),
        "endpoint_drop_diff_si_minus_content": safe_float(endpoint_diff),
        "diagnostics": {
            "K_support": safe_float(k_support),
            "S_content": safe_float(beta_c),
        },
        "drop_table": drop_df,
    }
    return df, summary


def main() -> None:
    p = argparse.ArgumentParser(description="C2 Carrier-class conditional prediction tests", allow_abbrev=False)
    p.add_argument("--models", default="llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1")
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1,mistral-7b-v0.1:cuda:2")
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "C2_carrier_class_predictions"))
    p.add_argument("--seed", type=int, default=20260417)
    p.add_argument("--n-examples-c21", type=int, default=96)
    p.add_argument("--seq-len-c21", type=int, default=512)
    p.add_argument("--n-examples-c22", type=int, default=96)
    p.add_argument("--max-seq-len-eval", type=int, default=1536)
    p.add_argument("--candidate-size", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--shuffle-strengths", default="0.0,0.25,0.5,0.75,1.0")
    args = p.parse_args()

    models = parse_models_arg(args.models)
    out_dir = ensure_dir(Path(args.output_root))

    device_map: dict[str, str] = {}
    for tok in [x.strip() for x in str(args.device_map).split(",") if x.strip()]:
        if ":" in tok:
            m, d = tok.split(":", 1)
            device_map[m.strip()] = d.strip()

    strengths = []
    for x in str(args.shuffle_strengths).split(","):
        t = x.strip()
        if not t:
            continue
        strengths.append(float(t))
    strengths = sorted(set([max(0.0, min(1.0, s)) for s in strengths]))
    if 0.0 not in strengths:
        strengths = [0.0] + strengths

    # C2.1
    c21_tables = []
    c21_model = {}
    c21_errors = {}
    for i, model in enumerate(models):
        device = device_map.get(model, "cuda:0")
        try:
            tdf, summ = _run_c21_for_model(
                model_name=model,
                device=device,
                seed=int(args.seed) + i * 73,
                n_examples=max(32, int(args.n_examples_c21)),
                seq_len=max(128, int(args.seq_len_c21)),
                candidate_size=max(4, int(args.candidate_size)),
                batch_size=max(1, int(args.batch_size)),
                shuffle_strengths=strengths,
            )
            c21_tables.append(tdf)
            c21_model[model] = summ
        except Exception as exc:
            c21_errors[model] = str(exc)
        finally:
            load_model.cache_clear()
            torch.cuda.empty_cache()

    c21_df = pd.concat(c21_tables, ignore_index=True) if c21_tables else pd.DataFrame()
    c21_df.to_parquet(out_dir / "position_shuffle_results.parquet", index=False)

    # C2.1 acceptance
    p_unadj = {}
    p_adj = {}
    beta_unadj = []
    se_unadj = []
    beta_adj = []
    se_adj = []
    mean_r2 = []
    slope_for_r2 = []
    for m, s in c21_model.items():
        p_unadj[m] = safe_float(s["beta_shuffle"]["p_one_gt0"])
        p_adj[m] = safe_float(s["beta_shuffle_adj"]["p_one_gt0"])
        beta_unadj.append(safe_float(s["beta_shuffle"]["beta"]))
        se_unadj.append(safe_float(s["beta_shuffle"]["se"]))
        beta_adj.append(safe_float(s["beta_shuffle_adj"]["beta"]))
        se_adj.append(safe_float(s["beta_shuffle_adj"]["se"]))
        mean_r2.append(safe_float(s.get("mean_si_r2")))
        slope_for_r2.append(safe_float(s["beta_shuffle"]["beta"]))

    p_unadj_h = holm_adjust_dict({f"{m}::unadj": p for m, p in p_unadj.items()})
    p_adj_h = holm_adjust_dict({f"{m}::adj": p for m, p in p_adj.items()})

    meta_unadj = _fixed_effect_meta_gt0(beta_unadj, se_unadj)
    meta_adj = _fixed_effect_meta_gt0(beta_adj, se_adj)
    n_models = len(c21_model)
    n_primary = n_models

    if n_primary >= 3:
        need_dir = int(math.ceil((2.0 / 3.0) * n_primary))
        n_dir = sum(1 for m, s in c21_model.items() if safe_float(s["beta_shuffle"]["beta"]) > 0)
        n_sig_practical = sum(
            1
            for m, s in c21_model.items()
            if safe_float(s["beta_shuffle"]["beta"]) >= 0.10
            and np.isfinite(safe_float(p_unadj_h.get(f"{m}::unadj")))
            and safe_float(p_unadj_h.get(f"{m}::unadj")) < 0.05
        )
        within_model_rule = bool(n_dir >= need_dir and n_sig_practical >= 2)
    else:
        within_model_rule = False

    n_adj_sig = sum(
        1
        for m, s in c21_model.items()
        if safe_float(s["beta_shuffle_adj"]["beta"]) > 0
        and np.isfinite(safe_float(p_adj_h.get(f"{m}::adj")))
        and safe_float(p_adj_h.get(f"{m}::adj")) < 0.05
    )
    pooled_ratio = safe_float(meta_adj["beta"]) / max(safe_float(meta_unadj["beta"]), 1e-8) if np.isfinite(safe_float(meta_adj["beta"])) else float("nan")
    robust_rule = bool(n_adj_sig >= 2 and np.isfinite(pooled_ratio) and pooled_ratio >= 0.70)

    mean_r2_arr = np.asarray(mean_r2, dtype=np.float64)
    slope_arr = np.asarray(slope_for_r2, dtype=np.float64)
    keep = np.isfinite(mean_r2_arr) & np.isfinite(slope_arr)
    if np.sum(keep) >= 3:
        r2_assoc = _spearman_assoc_with_exact_small_n(mean_r2_arr[keep], slope_arr[keep])
    else:
        r2_assoc = {"spearman_rho": float("nan"), "p_two": float("nan"), "p_method": "insufficient_n"}

    c21_supported = bool(n_models >= 3 and within_model_rule and robust_rule)

    # C2.2
    c22_tables = []
    c22_drop_tables = []
    c22_model = {}
    c22_errors = {}
    for i, model in enumerate(models):
        device = device_map.get(model, "cuda:0")
        try:
            tdf, summ = _run_c22_for_model(
                model_name=model,
                device=device,
                seed=int(args.seed) + 1000 + i * 97,
                n_examples=max(24, int(args.n_examples_c22)),
                seq_len=max(128, int(args.seq_len_c21)),
                candidate_size=max(4, int(args.candidate_size)),
                batch_size=max(1, int(args.batch_size)),
                max_seq_len_eval=max(256, int(args.max_seq_len_eval)),
                out_dir=out_dir,
            )
            c22_tables.append(tdf)
            d = summ.pop("drop_table")
            c22_drop_tables.append(d)
            c22_model[model] = summ
        except Exception as exc:
            c22_errors[model] = str(exc)
        finally:
            load_model.cache_clear()
            torch.cuda.empty_cache()

    c22_df = pd.concat(c22_tables, ignore_index=True) if c22_tables else pd.DataFrame()
    c22_df.to_parquet(out_dir / "long_context_results.parquet", index=False)

    # C2.2 acceptance and failure assignment
    eligible_models = [m for m, s in c22_model.items() if bool(s.get("confirmatory_eligible", False))]
    beta_int = [safe_float(c22_model[m]["interaction_beta_int"]["beta"]) for m in eligible_models]
    se_int = [safe_float(c22_model[m]["interaction_beta_int"]["se"]) for m in eligible_models]
    p_int = [safe_float(c22_model[m]["interaction_beta_int"]["p_one_gt0"]) for m in eligible_models]
    endpoint_diffs = [safe_float(c22_model[m]["endpoint_drop_diff_si_minus_content"]) for m in eligible_models]

    pooled_int = _fixed_effect_meta_gt0(beta_int, se_int)
    # second confirmatory family member: endpoint contrast at target long point
    if len(endpoint_diffs) >= 2:
        p1_ep = _sign_flip_one_sided_gt_zero(np.asarray(endpoint_diffs, dtype=np.float64), seed=int(args.seed) + 907)
    else:
        p1_ep = float("nan")
    fam = holm_adjust_dict({"pooled_beta_int": safe_float(pooled_int["p_one_gt0"]), "endpoint_contrast": p1_ep})

    criterion1 = bool(
        len(eligible_models) >= 2
        and np.isfinite(safe_float(fam.get("pooled_beta_int")))
        and safe_float(fam.get("pooled_beta_int")) < 0.05
    )
    eta_vals = [safe_float(c22_model[m]["interaction_beta_int"].get("beta", float("nan"))) for m in eligible_models]
    eta_proxy = float(np.nanmean(np.asarray([x * x for x in eta_vals if np.isfinite(x)], dtype=np.float64))) if eta_vals else float("nan")
    criterion2 = bool(
        (len(eligible_models) >= 2 and any(np.isfinite(x) and abs(x) >= 0.05 for x in endpoint_diffs))
        or (np.isfinite(eta_proxy) and eta_proxy >= 0.01)
    )
    overlap_ok_c22 = all(bool(c22_model[m].get("calibration_eval_disjoint", True)) for m in eligible_models) if eligible_models else False

    c22_supported = bool(len(eligible_models) >= 2 and criterion1 and criterion2 and overlap_ok_c22)

    # failure assignment rules
    failure_mode = "ambiguous_failure"
    if len(eligible_models) >= 2:
        pooled_p = safe_float(fam.get("pooled_beta_int"))
        pooled_b = safe_float(pooled_int["beta"])
        k_support = float(np.nanmean(np.asarray([safe_float(c22_model[m]["diagnostics"]["K_support"]) for m in eligible_models], dtype=np.float64)))
        s_content = float(np.nanmean(np.asarray([safe_float(c22_model[m]["diagnostics"]["S_content"]) for m in eligible_models], dtype=np.float64)))
        if np.isfinite(pooled_p) and pooled_p < 0.05 and np.isfinite(pooled_b) and pooled_b < 0:
            failure_mode = "carrier_class_inconsistent"
        elif np.isfinite(k_support) and np.isfinite(s_content) and k_support < 0.20 and s_content > -0.02:
            failure_mode = "kernel_support_limited"
        elif np.isfinite(k_support) and np.isfinite(s_content) and k_support >= 0.20 and s_content <= -0.05:
            failure_mode = "content_alignment_degrades"
        else:
            failure_mode = "ambiguous_failure"

    carrier_report = {
        "timestamp": timestamp_now(),
        "experiment_id": "C2_carrier_class_predictions",
        "C2_1": {
            "n_models": n_models,
            "errors": c21_errors,
            "per_model": c21_model,
            "meta_unadjusted": meta_unadj,
            "meta_adjusted": meta_adj,
            "p_holm_unadjusted": p_unadj_h,
            "p_holm_adjusted": p_adj_h,
            "within_model_rule_pass": within_model_rule,
            "robustness_rule_pass": robust_rule,
            "r2_slope_association": r2_assoc,
            "supported": c21_supported,
        },
        "C2_2": {
            "errors": c22_errors,
            "eligible_models": eligible_models,
            "per_model": c22_model,
            "pooled_interaction": pooled_int,
            "endpoint_diffs_si_minus_content": endpoint_diffs,
            "holm_family": fam,
            "criterion1_interaction_direction": criterion1,
            "criterion2_practical_threshold": criterion2,
            "criterion3_calibration_eval_disjoint": overlap_ok_c22,
            "eta_proxy": safe_float(eta_proxy),
            "failure_mode_assignment": failure_mode,
            "supported": c22_supported,
        },
        "verdict": {
            "C2_supported": bool(c21_supported and c22_supported),
            "C2_1_supported": c21_supported,
            "C2_2_supported": c22_supported,
        },
    }
    write_json(out_dir / "carrier_class_prediction_tests.json", carrier_report)

    prereg = {
        "experiment_id": "C2_carrier_class_predictions",
        "question": "Do carrier-class predictions hold for shuffle robustness and long-context behavior?",
        "primary_hypothesis": "SI-linked carrier structure predicts stronger shuffle sensitivity and context-conditional interaction effects.",
        "primary_endpoints": ["C2.1 beta_shuffle", "C2.2 pooled interaction beta_int"],
        "secondary_endpoints": ["R2-slope association", "endpoint long-context contrast", "failure-mode diagnostics"],
        "model_list": models,
        "dataset_sources": ["synthetic position-diagnostic tasks (experiment2/tasks.py)", "theory1 SI head artifacts", "theory8 kernels"],
        "inclusion_exclusion_rules": [
            "C2.1 confirmatory requires n_models>=3",
            "C2.2 confirmatory requires >=2 confirmatory-eligible models with >1.0x context point",
            "Calibration/evaluation sequence overlap forces C2.2 exploratory interpretation",
        ],
        "sample_size_plan": {
            "n_examples_c21": int(args.n_examples_c21),
            "n_examples_c22": int(args.n_examples_c22),
            "shuffle_strengths": strengths,
        },
        "seed_plan": {"seed": int(args.seed)},
        "stopping_rule": "Stop after both C2.1 and C2.2 summaries and criteria are computed.",
        "multiplicity_family": [
            "C2.1 within-model Holm families for unadjusted/adjusted slopes",
            "C2.2 Holm family over pooled interaction + endpoint contrast",
        ],
        "acceptance_criteria": ["C2.1 and C2.2 criteria per TODO."],
        "fallback_interpretation_if_null": "Carrier-class predictions remain unsupported; keep C narrative downgraded.",
    }
    manifest = command_manifest(
        experiment_id="C2_carrier_class_predictions",
        command="run_c2_carrier_class_predictions.py",
        model="+".join(models),
        extras={
            "models": models,
            "device_map": device_map,
            "seed": int(args.seed),
            "n_examples_c21": int(args.n_examples_c21),
            "n_examples_c22": int(args.n_examples_c22),
            "max_seq_len_eval": int(args.max_seq_len_eval),
        },
    )
    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "C2_carrier_class_predictions",
        "analysis_tier": "exploratory" if (len(eligible_models) >= 1 and not overlap_ok_c22) else "confirmatory",
        "canonical_eligible": bool(overlap_ok_c22),
        "override_used": False,
        "verdict": carrier_report["verdict"],
        "criteria": {
            "C2_1_within_model_rule_pass": within_model_rule,
            "C2_1_robustness_rule_pass": robust_rule,
            "C2_2_criterion1": criterion1,
            "C2_2_criterion2": criterion2,
            "C2_2_criterion3_disjoint": overlap_ok_c22,
        },
    }
    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "C2_carrier_class_predictions",
        "claim_status": "supported" if carrier_report["verdict"]["C2_supported"] else "mixed",
        "supports_main_text": bool(carrier_report["verdict"]["C2_supported"]),
        "strict_only": True,
        "notes": [
            "If C2.2 fails, report preregistered failure-mode assignment.",
            "If n_models<5, C2.1 cross-model regression is exploratory-only by design.",
        ],
    }
    data_dictionary = {
        "experiment_id": "C2_carrier_class_predictions",
        "tables": [
            {
                "path": str(out_dir / "position_shuffle_results.parquet"),
                "description": "Per-example C2.1 shuffle degradation rows.",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model name."},
                    {"name": "shuffle_strength", "dtype": "float", "description": "Fraction of pre-query tokens shuffled."},
                    {"name": "degradation_norm", "dtype": "float", "description": "Normalized performance degradation."},
                    {"name": "difficulty_proxy", "dtype": "float", "description": "Difficulty proxy (1 - clean accuracy)."},
                ],
            },
            {
                "path": str(out_dir / "long_context_results.parquet"),
                "description": "Per-example C2.2 carrier-class long-context rows.",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model name."},
                    {"name": "carrier_class", "dtype": "str", "description": "SI or ContentCond carrier class."},
                    {"name": "context_multiplier", "dtype": "float", "description": "Multiplier of train-like context."},
                    {"name": "accuracy", "dtype": "float", "description": "Accuracy under carrier-class isolation."},
                ],
            },
        ],
    }

    emit_core_artifacts(
        experiment_id="C2_carrier_class_predictions",
        out_dir=out_dir,
        preregistration=prereg,
        manifest=manifest,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    write_json(out_dir / "C2_claim_impact.json", claim_impact)

    print(f"[C2] wrote {out_dir}")


if __name__ == "__main__":
    main()
