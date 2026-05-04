#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize as scipy_opt
from scipy import stats as scipy_stats

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp2.common import RESULTS_ROOT, command_manifest, ensure_dir, file_sha256, load_kernels, timestamp_now, write_json  # noqa: E402
from reinforce_exp2.scripts._shared import emit_core_artifacts, holm_adjust_dict, parse_models_arg, safe_float  # noqa: E402

try:
    import torch as _torch
    from experiment2.execution import _evaluate_example_from_token_logits  # noqa: E402
    from experiment2.tasks import TaskExample, build_token_pools, generate_task_examples  # noqa: E402
    from experiment3.theory1_si_circuits import MODELS as _MODELS, HeadID, head_output_ablation  # noqa: E402
    from shared.models.loading import load_model, load_tokenizer  # noqa: E402
    _MODEL_INFERENCE_AVAILABLE = True
except ImportError:
    _MODEL_INFERENCE_AVAILABLE = False

_B3_PARTIAL_SCHEMA_VERSION = 1


def _parse_fractions(raw: str) -> list[float]:
    vals: list[float] = []
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        try:
            v = float(tok)
        except Exception:
            continue
        if 0.0 <= v <= 1.0:
            vals.append(float(v))
    if not vals:
        vals = [float(x) for x in np.linspace(0.0, 1.0, 11)]
    vals = sorted(set(vals))
    if 0.0 not in vals:
        vals = [0.0] + vals
    if 1.0 not in vals:
        vals = vals + [1.0]
    return [float(x) for x in vals]


def _truthy(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _partial_paths(partial_root: Path, model: str) -> dict[str, Path]:
    safe = str(model).replace("/", "_")
    model_root = ensure_dir(Path(partial_root) / safe)
    return {
        "root": model_root,
        "manifest": model_root / "partial_manifest.json",
        "curve": model_root / "curve.parquet",
        "diag": model_root / "matching_diag_df.parquet",
        "report": model_root / "model_report.json",
        "matching": model_root / "matching_diag_model.json",
    }


def _partial_config_hash(
    *,
    model: str,
    model_index: int,
    models: list[str],
    b1_path: Path,
    b2a_path: Path,
    calibration_root: Path,
    fractions: list[float],
    n_triplets_per_cluster: int,
    seed: int,
    seq_len: int,
    n_examples: int,
    candidate_size: int,
    batch_size: int,
    max_calib_seqs: int,
) -> str:
    cal_manifest = calibration_root / "calibration_v1_manifest.json"
    cal_sha = file_sha256(cal_manifest) if cal_manifest.exists() else None
    payload = {
        "schema": "b3_partial_v1",
        "schema_version": int(_B3_PARTIAL_SCHEMA_VERSION),
        "model": str(model),
        "model_index": int(model_index),
        "models": [str(x) for x in models],
        "b1_sha256": file_sha256(b1_path),
        "b2a_sha256": file_sha256(b2a_path),
        "calibration_manifest_sha256": cal_sha,
        "fractions": [float(x) for x in fractions],
        "n_triplets_per_cluster": int(n_triplets_per_cluster),
        "seed": int(seed),
        "seq_len": int(seq_len),
        "n_examples": int(n_examples),
        "candidate_size": int(candidate_size),
        "batch_size": int(batch_size),
        "max_calib_seqs": int(max_calib_seqs),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _partial_ready(partial_root: Path, model: str, expected_hash: str) -> tuple[bool, list[str]]:
    paths = _partial_paths(partial_root, model)
    errs: list[str] = []
    for k in ("manifest", "curve", "diag", "report", "matching"):
        if not paths[k].exists():
            errs.append(f"missing {k}: {paths[k]}")
    if errs:
        return False, errs
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"invalid partial manifest: {exc}"]
    if int(manifest.get("schema_version", 0)) != int(_B3_PARTIAL_SCHEMA_VERSION):
        errs.append("schema_version mismatch")
    if str(manifest.get("model", "")) != str(model):
        errs.append("model mismatch")
    if str(manifest.get("config_hash", "")) != str(expected_hash):
        errs.append("config_hash mismatch")
    if str(manifest.get("status", "")) != "ok":
        errs.append("status != ok")
    return (len(errs) == 0), errs


def _write_partial(
    *,
    partial_root: Path,
    model: str,
    config_hash: str,
    curve_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    model_report: dict[str, Any],
    matching_diag_model: dict[str, Any],
) -> None:
    paths = _partial_paths(partial_root, model)
    curve_df.to_parquet(paths["curve"], index=False)
    diag_df.to_parquet(paths["diag"], index=False)
    write_json(paths["report"], model_report)
    write_json(paths["matching"], matching_diag_model)
    write_json(
        paths["manifest"],
        {
            "timestamp": timestamp_now(),
            "schema_version": int(_B3_PARTIAL_SCHEMA_VERSION),
            "status": "ok",
            "model": str(model),
            "config_hash": str(config_hash),
            "curve_rows": int(curve_df.shape[0]),
            "diag_rows": int(diag_df.shape[0]),
            "paths": {k: str(v) for k, v in paths.items()},
        },
    )


def _load_partial(partial_root: Path, model: str) -> dict[str, Any]:
    paths = _partial_paths(partial_root, model)
    return {
        "model": str(model),
        "curve_df": pd.read_parquet(paths["curve"]),
        "diag_df": pd.read_parquet(paths["diag"]),
        "model_report": json.loads(paths["report"].read_text(encoding="utf-8")),
        "matching_diag_model": json.loads(paths["matching"].read_text(encoding="utf-8")),
    }


def _fit_linear(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    beta = np.polyfit(x, y, deg=1)
    pred = np.polyval(beta, x)
    rss = float(np.sum((y - pred) ** 2))
    return {"a": float(beta[1]), "b": float(beta[0]), "rss": rss}


def _fit_piecewise_continuous(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 5:
        return {"a": float("nan"), "b1": float("nan"), "b2": float("nan"), "tau": float("nan"), "rss": float("nan")}

    interior = [float(t) for t in x if 0.1 <= float(t) <= 0.9]
    if len(interior) < 2:
        interior = [float(np.quantile(x, q)) for q in np.linspace(0.1, 0.9, 9)]

    def solve_given_tau(tau: float) -> tuple[np.ndarray, float]:
        z = np.maximum(0.0, x - float(tau))
        X = np.column_stack([np.ones_like(x), x, z])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        rss = float(np.sum((y - pred) ** 2))
        return beta, rss

    best_tau = interior[0]
    best_beta, best_rss = solve_given_tau(best_tau)
    for t in interior[1:]:
        beta, rss = solve_given_tau(t)
        if rss < best_rss:
            best_tau = float(t)
            best_beta = beta
            best_rss = rss

    def obj(tau_scalar: np.ndarray) -> float:
        tau = float(np.clip(tau_scalar[0], 0.1, 0.9))
        _, rss = solve_given_tau(tau)
        return rss

    try:
        res = scipy_opt.minimize(obj, x0=np.asarray([best_tau], dtype=np.float64), bounds=[(0.1, 0.9)], method="L-BFGS-B")
        tau_opt = float(res.x[0]) if res.success else float(best_tau)
    except Exception:
        tau_opt = float(best_tau)

    beta_opt, rss_opt = solve_given_tau(tau_opt)
    return {
        "a": float(beta_opt[0]),
        "b1": float(beta_opt[1]),
        "b2": float(beta_opt[2]),
        "tau": float(tau_opt),
        "rss": float(rss_opt),
    }


def _aicc(rss: float, n: int, k: int) -> float:
    if n <= k + 1:
        return float("nan")
    sigma2 = max(rss / max(n, 1), 1e-12)
    aic = float(n * np.log(sigma2) + 2 * k)
    corr = float((2 * k * (k + 1)) / max(n - k - 1, 1e-12))
    return aic + corr


def _bounded_logit(y: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    v = np.asarray(y, dtype=np.float64)
    v = np.clip(v, 0.0, 1.0)
    v = np.clip(v, float(eps), 1.0 - float(eps))
    return np.log(v / (1.0 - v))


def _collapse_fraction(frac: np.ndarray, y: np.ndarray, tau: float) -> float:
    if np.isfinite(tau):
        return float(tau)
    mx = float(np.nanmax(y)) if y.size else float("nan")
    if not np.isfinite(mx) or mx <= 0:
        return float("nan")
    thr = 0.5 * mx
    idx = np.where(y >= thr)[0]
    return float(frac[idx[0]]) if len(idx) else float("nan")


def _auc(frac: np.ndarray, y: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> float:
    x = np.asarray(frac, dtype=np.float64)
    v = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(v) & (x >= lo) & (x <= hi)
    if np.sum(keep) < 2:
        return float("nan")
    xs = x[keep]
    ys = v[keep]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    return float(np.trapz(ys, xs) / max(hi - lo, 1e-12))


def _sign_flip_one_sided_gt_zero(values: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
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
    m = max(1000, int(n_perm))
    stats = []
    for _ in range(m):
        s = rng.choice(np.asarray([-1.0, 1.0]), size=n, replace=True)
        stats.append(float(np.mean(s * v)))
    arr = np.asarray(stats, dtype=np.float64)
    return float((np.sum(arr >= obs) + 1) / (len(arr) + 1))


def _relative_diff(a: float, b: float) -> float:
    return float(abs(float(a) - float(b)) / max((abs(float(a)) + abs(float(b))) / 2.0, 1e-8))


def _compute_head_magnitudes_from_model(
    model,
    device: str,
    calibration_tokens: list[list[int]],
    max_seqs: int = 256,
) -> dict[str, float]:
    """Compute m_h ≈ E_s,i[||A_h(s,i,:)||_2] on calibration corpus via attention weights."""
    model.eval()
    head_norm_vals: dict[str, list[float]] = {}

    for seq in calibration_tokens[:max_seqs]:
        toks = _torch.tensor([seq], dtype=_torch.long, device=device)
        with _torch.inference_mode():
            out = model(input_ids=toks, output_attentions=True, use_cache=False)
        attentions = out.attentions  # tuple: each (1, n_heads, L, L)

        for layer_idx, attn_t in enumerate(attentions):
            attn_np = attn_t[0].cpu().float().numpy()  # (n_heads, L, L)
            n_heads, L, _ = attn_np.shape
            for h in range(n_heads):
                hk = f"L{layer_idx}H{h}"
                if hk not in head_norm_vals:
                    head_norm_vals[hk] = []
                # L2 norm of each query's attention weight vector (length i+1 for query pos i)
                for i in range(L):
                    row = attn_np[h, i, :i + 1]
                    head_norm_vals[hk].append(float(np.linalg.norm(row)))

        del toks, out

    out: dict[str, float] = {}
    for hk, vals in head_norm_vals.items():
        if vals:
            m = float(np.mean(vals))
            out[hk] = m if np.isfinite(m) and m > 0 else 1e-6
    return out


def _sanitize_calibration_tokens_for_vocab(
    calibration_tokens: list[list[int]],
    vocab_size: int,
    min_len: int = 16,
) -> list[list[int]]:
    out: list[list[int]] = []
    vmax = max(1, int(vocab_size))
    for seq in calibration_tokens:
        if not isinstance(seq, list) or not seq:
            continue
        ok = all(isinstance(x, (int, np.integer)) and 0 <= int(x) < vmax for x in seq)
        if not ok:
            continue
        arr = [int(x) for x in seq]
        if len(arr) < int(min_len):
            continue
        out.append(arr)
    return out


def _ensure_output_attentions_supported(model, device: str, probe_tokens: list[int]) -> None:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(cfg, attr):
                try:
                    setattr(cfg, attr, "eager")
                except Exception:
                    pass
    ids = _torch.tensor([probe_tokens[: min(32, len(probe_tokens))]], dtype=_torch.long, device=device)
    with _torch.inference_mode():
        out = model(input_ids=ids, output_attentions=True, use_cache=False)
    atts = getattr(out, "attentions", None)
    if atts is None or len(atts) == 0:
        raise RuntimeError("output_attentions returned empty/None; cannot run B3 strict path")
    del ids, out
    if str(device).startswith("cuda"):
        _torch.cuda.empty_cache()


def _compute_head_magnitudes_fallback(model_name: str) -> dict[str, float]:
    """Fallback proxy: kernel mean absolute value (not actual head output norm)."""
    kernels = load_kernels(model_name)
    out: dict[str, float] = {}
    for hk, vec in kernels.items():
        arr = np.asarray(vec, dtype=np.float64)
        mag = float(np.mean(np.abs(arr)))
        if not np.isfinite(mag) or mag <= 0:
            mag = 1e-6
        out[str(hk)] = mag
    return out


def _all_heads_b3(model) -> list["HeadID"]:
    heads = []
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
        if n_heads is None or n_heads <= 0:
            n_heads = 32  # sensible default
        for h in range(n_heads):
            heads.append(HeadID(int(l), int(h)))
    return heads


def _evaluate_examples_b3(
    model,
    device: str,
    examples: list["TaskExample"],
    pools,
    candidate_size: int,
    batch_size: int,
) -> np.ndarray:
    vals: list[float] = []
    bs = max(1, int(batch_size))
    pos = 0
    while pos < len(examples):
        batch = examples[pos: pos + bs]
        try:
            input_ids = _torch.tensor([ex.tokens for ex in batch], dtype=_torch.long, device=device)
            with _torch.inference_mode():
                logits = model(input_ids=input_ids, use_cache=False).logits
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and bs > 1:
                _torch.cuda.empty_cache()
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
    return np.asarray(vals, dtype=np.float64)


def _prepare_b3_eval_context(
    model_name: str,
    model,
    device: str,
    seq_len: int,
    n_examples: int,
    seed: int,
    candidate_size: int,
    batch_size: int,
) -> dict[str, Any]:
    """Pre-generate task examples and compute baseline accuracy."""
    tokenizer = load_tokenizer(_MODELS[model_name])
    special_ids = [getattr(tokenizer, attr, None) for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")]
    special_ids = [int(x) for x in special_ids if x is not None]
    pools = build_token_pools(model_name, int(tokenizer.vocab_size), special_ids)

    phenoms = [
        ("copy_offset", "long_range_retrieval", 64),
        ("indexed_retrieval", "local_key_match", None),
        ("controlled_permutation", "local_copy_offset", None),
    ]

    all_examples: list["TaskExample"] = []
    for pidx, (phenom, task_name, span_override) in enumerate(phenoms):
        try:
            exs = generate_task_examples(
                task_name=task_name,
                model_name=model_name,
                seq_len=max(128, int(seq_len)),
                seed=int(seed) + pidx * 101,
                count=max(8, int(n_examples)),
                pools=pools,
                span_override=span_override,
                span_choices=(int(span_override),) if span_override is not None else None,
            )
        except Exception:
            exs = generate_task_examples(
                task_name="local_key_match",
                model_name=model_name,
                seq_len=max(128, int(seq_len)),
                seed=int(seed) + pidx * 101,
                count=max(8, int(n_examples)),
                pools=pools,
            )
        all_examples.extend(exs)

    baseline_acc = _evaluate_examples_b3(model, device, all_examples, pools, candidate_size, batch_size)
    all_heads = _all_heads_b3(model)

    return {
        "examples": all_examples,
        "pools": pools,
        "baseline_acc": baseline_acc,
        "all_heads": all_heads,
        "degradation_cache": {},
    }


def _evaluate_degradation(
    model,
    device: str,
    head_keys: set[str],
    all_heads: list["HeadID"],
    examples: list["TaskExample"],
    pools,
    baseline_acc: np.ndarray,
    candidate_size: int,
    batch_size: int,
) -> float:
    """Ablate specified heads and return mean accuracy drop (degradation)."""
    ablate = [h for h in all_heads if f"L{int(h.layer)}H{int(h.head)}" in head_keys]
    if not ablate:
        return 0.0
    with head_output_ablation(model, ablate):
        ablated_acc = _evaluate_examples_b3(model, device, examples, pools, candidate_size, batch_size)
    drop = float(np.nanmean(baseline_acc - ablated_acc))
    return max(0.0, float(drop))


def _load_reliability_eligibility(b1_root: Path) -> dict[str, bool]:
    path = b1_root / "kernel_taxonomy_summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, bool] = {}
    for model, row in payload.get("models", {}).items():
        gate = row.get("reliability_gate", {}) if isinstance(row, dict) else {}
        val = bool(gate.get("reliability_eligible", False))
        out[str(model)] = val
    return out


def _hist_layer_counts(df: pd.DataFrame, layers: list[int]) -> np.ndarray:
    vc = df.groupby("layer").size().to_dict()
    arr = np.asarray([float(vc.get(int(l), 0.0)) for l in layers], dtype=np.float64)
    return arr


def _chi_square_distance(ha: np.ndarray, hb: np.ndarray) -> float:
    exp = (ha + hb) / 2.0
    return float(np.sum(((ha - hb) ** 2) / np.maximum(exp, 1e-8)))


def _sparse_layer_permutation_pvalue(ha: np.ndarray, hb: np.ndarray, n_perm: int = 1024) -> float:
    a = np.asarray(np.round(ha), dtype=np.int64)
    b = np.asarray(np.round(hb), dtype=np.int64)
    if a.shape != b.shape:
        return float("nan")

    n_a = int(np.sum(a))
    n_b = int(np.sum(b))
    if n_a <= 0 or n_b <= 0:
        return float("nan")

    pooled = a + b
    obs = _chi_square_distance(a.astype(np.float64), b.astype(np.float64))
    seed_src = ",".join(str(int(x)) for x in a.tolist()) + "|" + ",".join(str(int(x)) for x in b.tolist())
    seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)

    hits = 0
    m = max(256, int(n_perm))
    for _ in range(m):
        try:
            a_draw = scipy_stats.multivariate_hypergeom.rvs(m=pooled, n=n_a, random_state=rng)
        except Exception:
            probs = pooled.astype(np.float64)
            probs = probs / max(float(np.sum(probs)), 1.0)
            a_draw = rng.multinomial(n=n_a, pvals=probs)
        b_draw = pooled - np.asarray(a_draw, dtype=np.int64)
        stat = _chi_square_distance(np.asarray(a_draw, dtype=np.float64), np.asarray(b_draw, dtype=np.float64))
        if np.isfinite(stat) and stat >= obs:
            hits += 1
    return float((hits + 1) / (m + 1))


def _layer_match_metrics(a: pd.DataFrame, b: pd.DataFrame, layers: list[int]) -> dict[str, float]:
    ha = _hist_layer_counts(a, layers)
    hb = _hist_layer_counts(b, layers)
    pa = ha / max(float(np.sum(ha)), 1.0)
    pb = hb / max(float(np.sum(hb)), 1.0)
    tvd = float(0.5 * np.sum(np.abs(pa - pb)))

    if np.allclose(ha, hb):
        p = 1.0
    else:
        try:
            exp = (ha + hb) / 2.0
            if np.any(exp < 5.0):
                p = _sparse_layer_permutation_pvalue(ha, hb, n_perm=1024)
            else:
                chi = _chi_square_distance(ha, hb)
                dof = max(1, len(layers) - 1)
                p = float(1.0 - scipy_stats.chi2.cdf(chi, dof))
        except Exception:
            p = float("nan")
    return {"tvd": tvd, "p": p}


def _degradation_for_set_eval(
    set_df: pd.DataFrame,
    eval_ctx: dict[str, Any],
    device: str,
    model,
    candidate_size: int,
    batch_size: int,
) -> float:
    """Compute degradation as mean accuracy drop from ablating the specified head set."""
    if set_df.empty:
        return 0.0
    head_keys = {str(hk) for hk in set_df["head_key"].astype(str).tolist()}
    if not head_keys:
        return 0.0
    cache = eval_ctx.get("degradation_cache")
    if not isinstance(cache, dict):
        cache = {}
        eval_ctx["degradation_cache"] = cache
    cache_key = hashlib.sha256(("|".join(sorted(head_keys))).encode("utf-8")).hexdigest()
    cached = cache.get(cache_key)
    if cached is not None and np.isfinite(float(cached)):
        return float(cached)
    deg = _evaluate_degradation(
        model=model,
        device=device,
        head_keys=head_keys,
        all_heads=eval_ctx["all_heads"],
        examples=eval_ctx["examples"],
        pools=eval_ctx["pools"],
        baseline_acc=eval_ctx["baseline_acc"],
        candidate_size=candidate_size,
        batch_size=batch_size,
    )
    cache[cache_key] = float(deg)
    return float(deg)


def _select_within_set(model_df: pd.DataFrame, cluster_id: int, k: int) -> pd.DataFrame:
    if k <= 0:
        return model_df.iloc[0:0].copy()
    cluster_df = model_df[model_df["cluster"] == int(cluster_id)].copy()
    other_df = model_df[model_df["cluster"] != int(cluster_id)].copy()

    cluster_df = cluster_df.sort_values(["m_h", "layer", "head"], ascending=[False, True, True])
    other_df = other_df.sort_values(["m_h", "layer", "head"], ascending=[False, True, True])

    n_cluster = min(k, int(cluster_df.shape[0]))
    picked = [cluster_df.head(n_cluster)]
    rem = int(k - n_cluster)
    if rem > 0:
        picked.append(other_df.head(rem))
    return pd.concat(picked, ignore_index=True) if picked else model_df.iloc[0:0].copy()


def _sample_by_layer(
    model_df: pd.DataFrame,
    layer_counts: dict[int, int],
    rng: np.random.Generator,
    *,
    mode: str,
    focus_cluster: int,
) -> pd.DataFrame:
    rows = []
    for layer, n in sorted(layer_counts.items()):
        if int(n) <= 0:
            continue
        pool = model_df[model_df["layer"] == int(layer)].copy()
        if pool.empty:
            continue

        pool = pool.sort_values(["head_key"], ascending=True).reset_index(drop=True)
        if mode == "mixed":
            # discourage concentrated draws from the within-cluster target while preserving layer histogram.
            score_focus = (pool["cluster"].astype(int) == int(focus_cluster)).astype(int).to_numpy(dtype=np.int64)
            jitter = rng.random(pool.shape[0])
            order = np.lexsort((jitter, score_focus))
            chosen = pool.iloc[order[: int(n)]].copy()
        else:
            order = rng.permutation(pool.shape[0])
            chosen = pool.iloc[order[: int(n)]].copy()
        rows.append(chosen)

    if not rows:
        return model_df.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def _adjust_magnitude(
    set_df: pd.DataFrame,
    model_df: pd.DataFrame,
    layer_counts: dict[int, int],
    target_mass: float,
    max_iter: int = 16,
) -> pd.DataFrame:
    if set_df.empty:
        return set_df
    work = set_df.copy()

    for _ in range(max(1, int(max_iter))):
        cur = float(work["m_h"].sum())
        rel = _relative_diff(cur, target_mass)
        if rel <= 0.05:
            break

        need_up = cur < target_mass
        best_swap = None
        best_diff = abs(cur - target_mass)

        selected_keys = set(work["head_key"].astype(str).tolist())
        for layer, n in layer_counts.items():
            if int(n) <= 0:
                continue
            in_layer = work[work["layer"] == int(layer)]
            if in_layer.empty:
                continue
            pool_layer = model_df[(model_df["layer"] == int(layer)) & (~model_df["head_key"].astype(str).isin(selected_keys))]
            if pool_layer.empty:
                continue

            # bounded candidate search for speed.
            if need_up:
                cand = pool_layer.sort_values(["m_h"], ascending=False).head(24)
            else:
                cand = pool_layer.sort_values(["m_h"], ascending=True).head(24)

            for in_row in in_layer.itertuples():
                for out_row in cand.itertuples():
                    new_mass = cur - float(in_row.m_h) + float(out_row.m_h)
                    diff = abs(new_mass - target_mass)
                    if diff < best_diff:
                        best_diff = diff
                        best_swap = (str(in_row.head_key), str(out_row.head_key), float(out_row.m_h))

        if best_swap is None:
            break

        in_key, out_key, _ = best_swap
        work = work[work["head_key"].astype(str) != in_key].copy()
        add_row = model_df[model_df["head_key"].astype(str) == out_key].head(1)
        if not add_row.empty:
            work = pd.concat([work, add_row], ignore_index=True)

    return work


def _build_model_triplets(
    model_name: str,
    model_df: pd.DataFrame,
    fractions: list[float],
    seed: int,
    n_triplets_per_cluster: int,
    eval_ctx: dict[str, Any] | None = None,
    model=None,
    device: str = "",
    candidate_size: int = 12,
    batch_size: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    layers = sorted(int(x) for x in model_df["layer"].unique().tolist())
    total_heads = int(model_df.shape[0])
    total_mass = float(model_df["m_h"].sum())

    cluster_sizes = model_df.groupby("cluster").size().reset_index(name="n")
    clusters = [int(x) for x in cluster_sizes[cluster_sizes["n"] >= 5]["cluster"].tolist()]

    rows = []
    diag_rows = []

    triplet_specs = []
    for c in clusters:
        for t_local in range(max(1, int(n_triplets_per_cluster))):
            triplet_specs.append((c, t_local))

    if not triplet_specs:
        return pd.DataFrame(), {
            "model": model_name,
            "status": "no_clusters_ge_5",
            "n_clusters_ge_5": 0,
            "n_triplets": 0,
            "fraction_grid": [float(x) for x in fractions],
        }

    for triplet_idx, (cluster_id, t_local) in enumerate(triplet_specs):
        for frac in fractions:
            k = int(round(float(frac) * total_heads))
            k = max(0, min(total_heads, k))

            if k == 0:
                for cond in ["within", "mixed", "random"]:
                    rows.append(
                        {
                            "model": model_name,
                            "triplet_id": f"{model_name}:c{cluster_id}:t{t_local}",
                            "cluster_focus": int(cluster_id),
                            "condition": cond,
                            "ablation_fraction": float(frac * 100.0),
                            "fraction": float(frac),
                            "n_heads": 0,
                            "set_mass": 0.0,
                            "degradation": 0.0,
                            "confirmatory": True,
                            "layer_match_pass": True,
                            "magnitude_match_pass": True,
                        }
                    )
                continue

            within = _select_within_set(model_df, cluster_id=cluster_id, k=k)
            layer_counts = {int(l): int(c) for l, c in within.groupby("layer").size().to_dict().items()}
            target_mass = float(within["m_h"].sum())

            mixed = _sample_by_layer(model_df, layer_counts, rng, mode="mixed", focus_cluster=cluster_id)
            random = _sample_by_layer(model_df, layer_counts, rng, mode="random", focus_cluster=cluster_id)

            mixed = _adjust_magnitude(mixed, model_df, layer_counts, target_mass=target_mass)
            random = _adjust_magnitude(random, model_df, layer_counts, target_mass=target_mass)

            mass_w = float(within["m_h"].sum())
            mass_m = float(mixed["m_h"].sum())
            mass_r = float(random["m_h"].sum())

            mag_pair = {
                "within_mixed": _relative_diff(mass_w, mass_m),
                "within_random": _relative_diff(mass_w, mass_r),
                "mixed_random": _relative_diff(mass_m, mass_r),
            }
            mag_ok = bool(all(float(v) <= 0.05 for v in mag_pair.values()))

            lm_wm = _layer_match_metrics(within, mixed, layers)
            lm_wr = _layer_match_metrics(within, random, layers)
            lm_mr = _layer_match_metrics(mixed, random, layers)
            layer_ok = bool(
                lm_wm["tvd"] <= 0.10
                and lm_wr["tvd"] <= 0.10
                and lm_mr["tvd"] <= 0.10
                and lm_wm["p"] > 0.10
                and lm_wr["p"] > 0.10
                and lm_mr["p"] > 0.10
            )

            confirm = bool(mag_ok and layer_ok)
            t_id = f"{model_name}:c{cluster_id}:t{t_local}"

            for cond, s in [("within", within), ("mixed", mixed), ("random", random)]:
                if eval_ctx is not None and model is not None and device:
                    deg = _degradation_for_set_eval(
                        s, eval_ctx=eval_ctx, device=device, model=model,
                        candidate_size=candidate_size, batch_size=batch_size,
                    )
                else:
                    deg = float("nan")
                rows.append(
                    {
                        "model": model_name,
                        "triplet_id": t_id,
                        "cluster_focus": int(cluster_id),
                        "condition": cond,
                        "ablation_fraction": float(frac * 100.0),
                        "fraction": float(frac),
                        "n_heads": int(s.shape[0]),
                        "set_mass": float(s["m_h"].sum()),
                        "degradation": deg,
                        "confirmatory": confirm,
                        "layer_match_pass": layer_ok,
                        "magnitude_match_pass": mag_ok,
                    }
                )

            diag_rows.append(
                {
                    "triplet_id": t_id,
                    "fraction": float(frac),
                    "cluster_focus": int(cluster_id),
                    "k": int(k),
                    "mass_within": mass_w,
                    "mass_mixed": mass_m,
                    "mass_random": mass_r,
                    "reldiff_within_mixed": mag_pair["within_mixed"],
                    "reldiff_within_random": mag_pair["within_random"],
                    "reldiff_mixed_random": mag_pair["mixed_random"],
                    "layer_tvd_within_mixed": lm_wm["tvd"],
                    "layer_tvd_within_random": lm_wr["tvd"],
                    "layer_tvd_mixed_random": lm_mr["tvd"],
                    "layer_p_within_mixed": lm_wm["p"],
                    "layer_p_within_random": lm_wr["p"],
                    "layer_p_mixed_random": lm_mr["p"],
                    "confirmatory": confirm,
                }
            )

    curve_df = pd.DataFrame(rows)
    diag_df = pd.DataFrame(diag_rows)

    n_triplets = int(curve_df["triplet_id"].nunique()) if not curve_df.empty else 0
    frac_count_confirm = int(curve_df[curve_df["confirmatory"]]["fraction"].nunique()) if not curve_df.empty else 0

    diag = {
        "model": model_name,
        "status": "ok" if not curve_df.empty else "empty",
        "n_clusters_ge_5": int(len(clusters)),
        "n_triplets": n_triplets,
        "fraction_grid": [float(x) for x in fractions],
        "n_confirmatory_fractions": frac_count_confirm,
        "layer_match_pass_rate": float(np.mean(diag_df["layer_tvd_within_mixed"].to_numpy(dtype=np.float64) <= 0.10)) if not diag_df.empty else float("nan"),
        "magnitude_match_pass_rate": float(np.mean(diag_df[["reldiff_within_mixed", "reldiff_within_random", "reldiff_mixed_random"]].max(axis=1).to_numpy(dtype=np.float64) <= 0.05)) if not diag_df.empty else float("nan"),
    }
    return curve_df, {"diag": diag, "diag_df": diag_df}


def _condition_curve(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    sub = df[df["condition"] == condition].copy()
    if sub.empty:
        return pd.DataFrame(columns=["ablation_fraction", "degradation", "frac"])
    out = sub.groupby("ablation_fraction", as_index=False)["degradation"].median().sort_values("ablation_fraction")
    out["frac"] = out["ablation_fraction"].astype(float) / 100.0
    return out


def _triplet_auc_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t_id, tdf in df.groupby("triplet_id"):
        auc = {}
        for cond in ["within", "mixed", "random"]:
            cdf = tdf[tdf["condition"] == cond].copy().sort_values("ablation_fraction")
            if cdf.empty:
                continue
            frac = cdf["ablation_fraction"].to_numpy(dtype=np.float64) / 100.0
            y = cdf["degradation"].to_numpy(dtype=np.float64)
            auc[cond] = _auc(frac, y, lo=0.0, hi=1.0)
        if "within" in auc and "mixed" in auc:
            rows.append(
                {
                    "triplet_id": str(t_id),
                    "auc_within": safe_float(auc.get("within")),
                    "auc_mixed": safe_float(auc.get("mixed")),
                    "auc_random": safe_float(auc.get("random")),
                    "d_auc_within_minus_mixed": safe_float(auc.get("within")) - safe_float(auc.get("mixed")),
                }
            )
    return pd.DataFrame(rows)


def _build_contribution_report(
    *,
    model_name: str,
    model_df: pd.DataFrame,
    confirm_df: pd.DataFrame,
    fractions: list[float],
    eval_ctx: dict[str, Any] | None = None,
    model=None,
    device: str = "",
    candidate_size: int = 12,
    batch_size: int = 8,
) -> dict[str, Any]:
    total_mass = float(model_df["m_h"].sum())
    if confirm_df.empty:
        return {
            "model": model_name,
            "informative": False,
            "non_informative_reason": "insufficient_confirmatory_triplets",
            "n_confirmatory_eligible_clusters": 0,
            "clusters": [],
        }

    within = confirm_df[confirm_df["condition"] == "within"].copy()
    if within.empty:
        return {
            "model": model_name,
            "informative": False,
            "non_informative_reason": "missing_within_condition",
            "n_confirmatory_eligible_clusters": 0,
            "clusters": [],
        }

    budget_mass = within.groupby("fraction")["set_mass"].median().to_dict()
    budget_k = within.groupby("fraction")["n_heads"].median().to_dict()

    cluster_sizes = model_df.groupby("cluster").size().reset_index(name="n")
    clusters = [int(x) for x in cluster_sizes[cluster_sizes["n"] >= 5]["cluster"].tolist()]

    cluster_rows = []
    eligible_clusters = []
    feasible_grids: dict[int, set[float]] = {}

    for c in clusters:
        cdf = model_df[model_df["cluster"] == int(c)].copy().sort_values(["m_h", "layer", "head"], ascending=[False, True, True])
        if cdf.empty:
            continue

        frac_rows = []
        feasible = []
        for frac in fractions:
            target_m = float(budget_mass.get(float(frac), float("nan")))
            target_k = int(round(float(budget_k.get(float(frac), 0.0))))
            if not np.isfinite(target_m):
                continue

            best = None
            prefix_mass = 0.0
            for i, row in enumerate(cdf.itertuples(), start=1):
                prefix_mass += float(row.m_h)
                rel = abs(prefix_mass - target_m) / max(target_m, 1e-8)
                cand = (
                    rel,
                    abs(i - target_k),
                    i,
                )
                if best is None or cand < best[0]:
                    best = (cand, i, prefix_mass)

            if best is None:
                continue
            _, k_sel, mass_sel = best
            rel_diff = abs(float(mass_sel) - float(target_m)) / max(float(target_m), 1e-8)
            ok = bool(rel_diff <= 0.05)
            if ok:
                feasible.append(float(frac))

            set_sel = cdf.head(int(k_sel)).copy()
            if eval_ctx is not None and model is not None and device:
                deg = _degradation_for_set_eval(
                    set_sel, eval_ctx=eval_ctx, device=device, model=model,
                    candidate_size=candidate_size, batch_size=batch_size,
                )
            else:
                deg = float("nan")

            frac_rows.append(
                {
                    "fraction": float(frac),
                    "target_budget_mass": float(target_m),
                    "selected_mass": float(mass_sel),
                    "relative_mass_error": float(rel_diff),
                    "feasible_confirmatory": bool(ok),
                    "n_heads_selected": int(k_sel),
                    "degradation": float(deg),
                }
            )

        coverage = float(len(feasible) / max(len(fractions), 1))
        eligible = bool(coverage >= 0.80)
        if eligible:
            eligible_clusters.append(int(c))
        feasible_grids[int(c)] = set(feasible)

        cluster_rows.append(
            {
                "cluster": int(c),
                "size": int(cdf.shape[0]),
                "coverage_confirmatory": coverage,
                "confirmatory_eligible": eligible,
                "fraction_rows": frac_rows,
            }
        )

    if len(eligible_clusters) < 2:
        return {
            "model": model_name,
            "informative": False,
            "non_informative_reason": "insufficient_clusters",
            "n_confirmatory_eligible_clusters": int(len(eligible_clusters)),
            "clusters": cluster_rows,
        }

    common = None
    for c in eligible_clusters:
        s = feasible_grids.get(int(c), set())
        common = set(s) if common is None else (common & set(s))
    common_grid = sorted(float(x) for x in (common or set()))

    if len(common_grid) < 8:
        return {
            "model": model_name,
            "informative": False,
            "non_informative_reason": "noninformative_contribution",
            "n_confirmatory_eligible_clusters": int(len(eligible_clusters)),
            "common_confirmatory_grid": common_grid,
            "clusters": cluster_rows,
        }

    contrib_auc = {}
    for row in cluster_rows:
        c = int(row["cluster"])
        if c not in eligible_clusters:
            continue
        frac_rows = pd.DataFrame(row["fraction_rows"])
        if frac_rows.empty:
            continue
        sub = frac_rows[frac_rows["fraction"].isin(common_grid)].copy().sort_values("fraction")
        auc_val = _auc(sub["fraction"].to_numpy(dtype=np.float64), sub["degradation"].to_numpy(dtype=np.float64), lo=0.0, hi=1.0)
        contrib_auc[c] = float(auc_val)

    sum_pos = float(np.sum([max(v, 0.0) for v in contrib_auc.values()]))
    if not np.isfinite(sum_pos) or sum_pos < 0.05:
        return {
            "model": model_name,
            "informative": False,
            "non_informative_reason": "noninformative_contribution",
            "n_confirmatory_eligible_clusters": int(len(eligible_clusters)),
            "common_confirmatory_grid": common_grid,
            "clusters": cluster_rows,
            "contribution_auc": {str(k): safe_float(v) for k, v in contrib_auc.items()},
            "sum_positive_auc": sum_pos,
        }

    shares = {int(k): float(max(v, 0.0) / max(sum_pos, 1e-8)) for k, v in contrib_auc.items()}
    max_share = float(max(shares.values())) if shares else float("nan")

    return {
        "model": model_name,
        "informative": True,
        "non_informative_reason": "none",
        "n_confirmatory_eligible_clusters": int(len(eligible_clusters)),
        "common_confirmatory_grid": common_grid,
        "contribution_auc": {str(k): safe_float(v) for k, v in contrib_auc.items()},
        "cluster_share": {str(k): safe_float(v) for k, v in shares.items()},
        "sum_positive_auc": sum_pos,
        "max_cluster_share": max_share,
        "clusters": cluster_rows,
    }


def _read_calibration_manifest(calibration_root: Path) -> dict[str, Any]:
    p = calibration_root / "calibration_v1_manifest.json"
    if not p.exists():
        return {
            "split_id": "calibration_v1",
            "available": False,
            "sha256_ids_parquet": None,
            "disjoint_with_evaluation": True,
            "overlap_count": 0,
            "overlap_ratio": 0.0,
            "note": "calibration manifest missing",
        }
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    return {
        "split_id": str(payload.get("split_id", "calibration_v1")),
        "available": True,
        "sha256_ids_parquet": payload.get("sha256_ids_parquet"),
        "disjoint_with_evaluation": True,
        "overlap_count": 0,
        "overlap_ratio": 0.0,
        "note": "B3 uses calibration corpus for m_h estimation; evaluation uses separate synthetic task examples",
    }


def _compute_model_result(
    *,
    model: str,
    model_index: int,
    b1_df: pd.DataFrame,
    b2a_df: pd.DataFrame,
    rel_gate: dict[str, bool],
    calibration_tokens: list[list[int]],
    device_map: dict[str, str],
    fractions: list[float],
    n_triplets_per_cluster: int,
    seed: int,
    seq_len: int,
    n_examples: int,
    candidate_size: int,
    batch_size: int,
    max_calib_seqs: int,
) -> dict[str, Any]:
    mb1 = b1_df[b1_df["model"] == model].copy()
    mb2 = b2a_df[b2a_df["model"] == model].copy()
    if mb1.empty or mb2.empty:
        report = {
            "status": "missing_prerequisites",
            "reliability_eligible": bool(rel_gate.get(model, False)),
        }
        return {
            "model": model,
            "curve_df": pd.DataFrame(),
            "diag_df": pd.DataFrame(),
            "model_report": report,
            "matching_diag_model": {
                "model": model,
                "status": "missing_prerequisites",
                "reliability_eligible": bool(rel_gate.get(model, False)),
            },
        }

    cols = ["model", "layer", "head", "head_key", "cluster_descriptor_kmeans", "mean_r2", "boundary_attn_score"]
    work = mb1[cols].merge(
        mb2[["model", "layer", "head", "head_key", "delta_signed", "delta_absolute", "absolute_alignment", "signed_alignment"]],
        on=["model", "layer", "head", "head_key"],
        how="inner",
    )
    if work.empty:
        report = {
            "status": "empty_after_merge",
            "reliability_eligible": bool(rel_gate.get(model, False)),
        }
        return {
            "model": model,
            "curve_df": pd.DataFrame(),
            "diag_df": pd.DataFrame(),
            "model_report": report,
            "matching_diag_model": {
                "model": model,
                "status": "empty_after_merge",
                "reliability_eligible": bool(rel_gate.get(model, False)),
            },
        }

    device = device_map.get(model, "")
    model_obj = None
    loaded = None
    eval_ctx: dict[str, Any] | None = None
    strict_inference = bool(_MODEL_INFERENCE_AVAILABLE and device and model in _MODELS)
    inference_failure: str | None = None
    inference_trace: str | None = None

    if strict_inference:
        try:
            tok = load_tokenizer(_MODELS[model])
            vocab_size = int(getattr(tok, "vocab_size", 0) or 0)
            safe_tokens = _sanitize_calibration_tokens_for_vocab(calibration_tokens, vocab_size=vocab_size, min_len=16)
            if len(safe_tokens) < 8:
                raise RuntimeError(
                    f"Insufficient vocab-safe calibration sequences ({len(safe_tokens)}) for B3 strict path"
                )
            loaded = load_model(_MODELS[model])
            model_obj = loaded.model.to(device)
            model_obj.eval()
            _ensure_output_attentions_supported(model_obj, device, safe_tokens[0])
            m_h = _compute_head_magnitudes_from_model(
                model_obj,
                device=device,
                calibration_tokens=safe_tokens,
                max_seqs=int(max_calib_seqs),
            )
            eval_ctx = _prepare_b3_eval_context(
                model_name=model,
                model=model_obj,
                device=device,
                seq_len=int(seq_len),
                n_examples=int(n_examples),
                seed=int(seed) + int(model_index) * 100,
                candidate_size=int(candidate_size),
                batch_size=int(batch_size),
            )
        except Exception as exc:
            inference_failure = f"{type(exc).__name__}: {exc}"
            inference_trace = traceback.format_exc(limit=3)
            m_h = {}
            eval_ctx = None
            if model_obj is not None:
                try:
                    del model_obj
                except Exception:
                    pass
                model_obj = None
            try:
                if loaded is not None:
                    del loaded
            except Exception:
                pass
            try:
                load_model.cache_clear()
            except Exception:
                pass
            try:
                _torch.cuda.empty_cache()
            except Exception:
                pass
    else:
        m_h = _compute_head_magnitudes_fallback(model)

    if inference_failure is not None:
        report = {
            "status": "attention_inference_failed",
            "reliability_eligible": bool(rel_gate.get(model, False)),
            "failure_reason": inference_failure,
            "failure_traceback_tail": inference_trace,
        }
        return {
            "model": model,
            "curve_df": pd.DataFrame(),
            "diag_df": pd.DataFrame(),
            "model_report": report,
            "matching_diag_model": {
                "model": model,
                "status": "attention_inference_failed",
                "reliability_eligible": bool(rel_gate.get(model, False)),
                "failure_reason": inference_failure,
            },
        }

    work["m_h"] = work["head_key"].map(lambda x: safe_float(m_h.get(str(x), float("nan"))) if str(x) in m_h else float("nan"))
    fill_val = float(np.nanmedian(work["m_h"].to_numpy(dtype=np.float64))) if np.any(np.isfinite(work["m_h"].to_numpy(dtype=np.float64))) else 1e-6
    work["m_h"] = work["m_h"].fillna(fill_val).astype(float)
    work["cluster"] = work["cluster_descriptor_kmeans"].astype(int)

    curve_df, diag_bundle = _build_model_triplets(
        model_name=model,
        model_df=work,
        fractions=fractions,
        seed=int(seed) + int(model_index) * 211,
        n_triplets_per_cluster=max(1, int(n_triplets_per_cluster)),
        eval_ctx=eval_ctx,
        model=model_obj,
        device=device,
        candidate_size=int(candidate_size),
        batch_size=int(batch_size),
    )

    diag = diag_bundle["diag"]
    diag_df = diag_bundle.get("diag_df", pd.DataFrame())

    if curve_df.empty:
        report = {
            "status": "no_triplets",
            "reliability_eligible": bool(rel_gate.get(model, False)),
            "matching": diag,
        }
        return {
            "model": model,
            "curve_df": pd.DataFrame(),
            "diag_df": diag_df,
            "model_report": report,
            "matching_diag_model": {
                **diag,
                "reliability_eligible": bool(rel_gate.get(model, False)),
            },
        }

    confirm_df = curve_df[curve_df["confirmatory"]].copy()
    frac_n_confirm = int(confirm_df["fraction"].nunique()) if not confirm_df.empty else 0
    model_selection_confirmatory = bool(frac_n_confirm >= 11)

    cond_fits = {}
    cond_auc = {}
    cond_fc = {}
    for cond in ["within", "mixed", "random"]:
        cc = _condition_curve(confirm_df, cond)
        if cc.empty or cc.shape[0] < 5:
            cond_fits[cond] = {"status": "insufficient_points"}
            cond_auc[cond] = float("nan")
            cond_fc[cond] = float("nan")
            continue
        x = cc["frac"].to_numpy(dtype=np.float64)
        y = cc["degradation"].to_numpy(dtype=np.float64)
        y_fit = _bounded_logit(y)
        lin = _fit_linear(x, y_fit)
        pw = _fit_piecewise_continuous(x, y_fit)
        aicc_lin = _aicc(lin["rss"], n=len(x), k=2)
        aicc_pw = _aicc(pw["rss"], n=len(x), k=4)
        delta = float(aicc_lin - aicc_pw) if np.isfinite(aicc_lin) and np.isfinite(aicc_pw) else float("nan")
        cond_fits[cond] = {
            "n_points": int(len(x)),
            "fit_scale": "logit_bounded_degradation",
            "linear": {**lin, "aicc": aicc_lin},
            "threshold_piecewise": {**pw, "aicc": aicc_pw},
            "delta_aicc_linear_minus_threshold": delta,
            "preferred_model": "threshold_piecewise" if np.isfinite(delta) and delta > 0 else "linear",
        }
        cond_auc[cond] = _auc(x, y, lo=0.0, hi=1.0)
        cond_fc[cond] = _collapse_fraction(x, y, pw["tau"])

    trip_auc = _triplet_auc_table(confirm_df)
    p_group = _sign_flip_one_sided_gt_zero(trip_auc["d_auc_within_minus_mixed"].to_numpy(dtype=np.float64), seed=int(seed) + int(model_index) * 31)

    contrib = _build_contribution_report(
        model_name=model,
        model_df=work,
        confirm_df=confirm_df,
        fractions=fractions,
        eval_ctx=eval_ctx,
        model=model_obj,
        device=device,
        candidate_size=int(candidate_size),
        batch_size=int(batch_size),
    )

    d_auc = safe_float(cond_auc.get("within")) - safe_float(cond_auc.get("mixed"))
    d_fc_abs = abs(safe_float(cond_fc.get("within")) - safe_float(cond_fc.get("mixed")))
    d_fc_signed = safe_float(cond_fc.get("mixed")) - safe_float(cond_fc.get("within"))

    redundancy_pre = bool(
        model_selection_confirmatory
        and np.isfinite(cond_fits.get("mixed", {}).get("delta_aicc_linear_minus_threshold", float("nan")))
        and cond_fits["mixed"]["delta_aicc_linear_minus_threshold"] >= 6.0
        and np.isfinite(cond_fits.get("random", {}).get("delta_aicc_linear_minus_threshold", float("nan")))
        and cond_fits["random"]["delta_aicc_linear_minus_threshold"] >= 6.0
        and bool(contrib.get("informative", False))
        and np.isfinite(safe_float(contrib.get("max_cluster_share", float("nan"))))
        and safe_float(contrib.get("max_cluster_share", float("nan"))) < 0.50
        and np.isfinite(d_auc)
        and abs(d_auc) < 0.15
        and np.isfinite(d_fc_abs)
        and d_fc_abs <= 0.10
    )

    report = {
        "status": "ok",
        "reliability_eligible": bool(rel_gate.get(model, False)),
        "matching": diag,
        "model_selection_confirmatory": model_selection_confirmatory,
        "curve_fit": cond_fits,
        "auc_drop": cond_auc,
        "collapse_fraction": cond_fc,
        "triplet_auc_table": trip_auc.to_dict(orient="records"),
        "group_structure_test": {
            "n_triplets": int(trip_auc.shape[0]),
            "mean_d_auc_within_minus_mixed": safe_float(np.nanmean(trip_auc["d_auc_within_minus_mixed"].to_numpy(dtype=np.float64))) if not trip_auc.empty else float("nan"),
            "p_one_sided_within_gt_mixed": p_group,
        },
        "contribution": contrib,
        "d_auc_within_minus_mixed": d_auc,
        "d_fc_mixed_minus_within": d_fc_signed,
        "model_level_redundancy_preholm": redundancy_pre,
    }
    matching = {
        **diag,
        "model_selection_confirmatory": model_selection_confirmatory,
        "n_confirmatory_triplets": int(confirm_df["triplet_id"].nunique()) if not confirm_df.empty else 0,
        "n_confirmatory_rows": int(confirm_df.shape[0]),
        "reliability_eligible": bool(rel_gate.get(model, False)),
    }

    if model_obj is not None:
        del model_obj
        model_obj = None
        if _MODEL_INFERENCE_AVAILABLE and device.startswith("cuda"):
            _torch.cuda.empty_cache()
    try:
        if loaded is not None:
            del loaded
    except Exception:
        pass
    try:
        load_model.cache_clear()
    except Exception:
        pass

    return {
        "model": model,
        "curve_df": curve_df,
        "diag_df": diag_df,
        "model_report": report,
        "matching_diag_model": matching,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="B3 Cluster-wise ablation disambiguation (full matched-set path)",
        allow_abbrev=False,
    )
    p.add_argument("--execution-mode", choices=["full", "per_model", "aggregate"], default="full")
    p.add_argument("--single-model", default="")
    p.add_argument("--partial-root", default="")
    p.add_argument("--resume-partials", default="true")
    p.add_argument("--models", default="llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1")
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "B3_cluster_ablation"))
    p.add_argument("--b1-root", default=str(RESULTS_ROOT / "B1_kernel_taxonomy"))
    p.add_argument("--b2a-root", default=str(RESULTS_ROOT / "B2a_head_alignment"))
    p.add_argument("--calibration-root", default=str(RESULTS_ROOT / "calibration_splits"))
    p.add_argument("--fractions", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--n-triplets-per-cluster", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260417)
    p.add_argument("--device-map", default="")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-examples", type=int, default=64)
    p.add_argument("--candidate-size", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-calib-seqs", type=int, default=256)
    args = p.parse_args()

    execution_mode = str(args.execution_mode).strip().lower()
    if execution_mode == "per_model" and not str(args.single_model).strip():
        raise ValueError("--single-model is required for --execution-mode per_model")
    all_models = parse_models_arg(args.models)
    models = list(all_models)
    if execution_mode == "per_model":
        target = str(args.single_model).strip()
        if target not in models:
            raise ValueError(f"--single-model {target} is not in --models list: {models}")
        models = [target]

    out_dir = ensure_dir(Path(args.output_root))
    partial_root = ensure_dir(Path(args.partial_root)) if str(args.partial_root).strip() else ensure_dir(out_dir / "partials")
    resume_partials = _truthy(args.resume_partials)
    b1_root = Path(args.b1_root)
    b2a_root = Path(args.b2a_root)
    calibration_root = Path(args.calibration_root)
    fractions = _parse_fractions(args.fractions)

    device_map: dict[str, str] = {}
    for tok in [x.strip() for x in str(args.device_map).split(",") if x.strip()]:
        if ":" in tok:
            m, d = tok.split(":", 1)
            device_map[m.strip()] = d.strip()

    # Load calibration tokens for m_h computation
    calibration_tokens: list[list[int]] = []
    cal_tok_path = calibration_root / "calibration_v1_tokens.parquet"
    if cal_tok_path.exists():
        try:
            cal_df = pd.read_parquet(cal_tok_path)
            for row in cal_df.itertuples():
                toks = row.tokens
                if isinstance(toks, (list, np.ndarray)):
                    calibration_tokens.append([int(x) for x in toks])
        except Exception:
            pass

    b1_path = b1_root / "cluster_membership.parquet"
    b2a_path = b2a_root / "head_alignment_scores.parquet"
    if not b1_path.exists():
        raise FileNotFoundError(f"Missing B1 artifact: {b1_path}")
    if not b2a_path.exists():
        raise FileNotFoundError(f"Missing B2a artifact: {b2a_path}")

    b1_df = pd.read_parquet(b1_path)
    b2a_df = pd.read_parquet(b2a_path)
    rel_gate = _load_reliability_eligibility(b1_root)

    all_rows = []
    matching_diag_payload = {"timestamp": timestamp_now(), "models": {}, "calibration": _read_calibration_manifest(calibration_root)}
    model_reports: dict[str, Any] = {}
    raw_pvals_group: dict[str, float] = {}

    model_index_map = {str(m): int(i) for i, m in enumerate(all_models)}
    model_payloads: dict[str, dict[str, Any]] = {}

    if execution_mode == "aggregate":
        for model in models:
            idx = int(model_index_map.get(model, 0))
            cfg_hash = _partial_config_hash(
                model=model,
                model_index=idx,
                models=all_models,
                b1_path=b1_path,
                b2a_path=b2a_path,
                calibration_root=calibration_root,
                fractions=fractions,
                n_triplets_per_cluster=int(args.n_triplets_per_cluster),
                seed=int(args.seed),
                seq_len=int(args.seq_len),
                n_examples=int(args.n_examples),
                candidate_size=int(args.candidate_size),
                batch_size=int(args.batch_size),
                max_calib_seqs=int(args.max_calib_seqs),
            )
            ready, errs = _partial_ready(partial_root, model, cfg_hash)
            if not ready:
                raise RuntimeError(f"Missing/invalid partial for model={model}: {errs}")
            model_payloads[model] = _load_partial(partial_root, model)
    else:
        for model in models:
            idx = int(model_index_map.get(model, 0))
            cfg_hash = _partial_config_hash(
                model=model,
                model_index=idx,
                models=all_models,
                b1_path=b1_path,
                b2a_path=b2a_path,
                calibration_root=calibration_root,
                fractions=fractions,
                n_triplets_per_cluster=int(args.n_triplets_per_cluster),
                seed=int(args.seed),
                seq_len=int(args.seq_len),
                n_examples=int(args.n_examples),
                candidate_size=int(args.candidate_size),
                batch_size=int(args.batch_size),
                max_calib_seqs=int(args.max_calib_seqs),
            )
            ready, errs = _partial_ready(partial_root, model, cfg_hash)
            if resume_partials and ready:
                print(f"[B3 per_model] partial exists and valid for {model}; skipping")
                payload = _load_partial(partial_root, model)
            else:
                if resume_partials and (not ready) and errs:
                    print(f"[B3 per_model] partial invalid for {model}; recomputing")
                payload = _compute_model_result(
                    model=model,
                    model_index=idx,
                    b1_df=b1_df,
                    b2a_df=b2a_df,
                    rel_gate=rel_gate,
                    calibration_tokens=calibration_tokens,
                    device_map=device_map,
                    fractions=fractions,
                    n_triplets_per_cluster=int(args.n_triplets_per_cluster),
                    seed=int(args.seed),
                    seq_len=int(args.seq_len),
                    n_examples=int(args.n_examples),
                    candidate_size=int(args.candidate_size),
                    batch_size=int(args.batch_size),
                    max_calib_seqs=int(args.max_calib_seqs),
                )
                _write_partial(
                    partial_root=partial_root,
                    model=model,
                    config_hash=cfg_hash,
                    curve_df=payload["curve_df"],
                    diag_df=payload["diag_df"],
                    model_report=payload["model_report"],
                    matching_diag_model=payload["matching_diag_model"],
                )
                print(f"[B3 per_model] wrote partial for {model} at {partial_root}")
            model_payloads[model] = payload

    if execution_mode == "per_model":
        print(f"[B3 per_model] complete for {models[0]}")
        return

    for model in models:
        payload = model_payloads.get(model, {})
        curve_df = payload.get("curve_df", pd.DataFrame())
        diag_df = payload.get("diag_df", pd.DataFrame())
        report = payload.get("model_report", {})
        matching = payload.get("matching_diag_model", {})
        model_reports[model] = report
        matching_diag_payload["models"][model] = matching
        if isinstance(curve_df, pd.DataFrame) and not curve_df.empty:
            all_rows.append(curve_df)
        p_group = safe_float(report.get("group_structure_test", {}).get("p_one_sided_within_gt_mixed", float("nan")))
        if np.isfinite(p_group):
            raw_pvals_group[model] = float(p_group)
        if isinstance(diag_df, pd.DataFrame) and not diag_df.empty:
            diag_out = diag_df.copy()
            diag_out["model"] = model
            diag_path = out_dir / f"matching_diagnostics_{model}.parquet"
            diag_out.to_parquet(diag_path, index=False)

    curve_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not curve_all.empty:
        curve_all.to_parquet(out_dir / "cluster_ablation_curve.parquet", index=False)

    # Holm family for group-structure per-model tests (reliability-eligible models only).
    p_family = {
        m: p
        for m, p in raw_pvals_group.items()
        if bool(model_reports.get(m, {}).get("status") == "ok") and bool(model_reports.get(m, {}).get("reliability_eligible", False))
    }
    p_holm = holm_adjust_dict(p_family)

    eligible_models = [
        m
        for m in models
        if model_reports.get(m, {}).get("status") == "ok"
        and bool(model_reports.get(m, {}).get("reliability_eligible", False))
        and bool(model_reports.get(m, {}).get("model_selection_confirmatory", False))
    ]

    for m in models:
        rep = model_reports.get(m, {})
        if rep.get("status") != "ok":
            continue
        p_h = safe_float(p_holm.get(m)) if m in p_holm else float("nan")
        d_auc = safe_float(rep.get("d_auc_within_minus_mixed", float("nan")))
        d_fc = safe_float(rep.get("d_fc_mixed_minus_within", float("nan")))

        grp = bool(
            bool(rep.get("model_selection_confirmatory", False))
            and np.isfinite(p_h)
            and p_h < 0.05
            and np.isfinite(d_auc)
            and d_auc >= 0.15
            and np.isfinite(d_fc)
            and d_fc >= 0.10
        )

        red = bool(rep.get("model_level_redundancy_preholm", False))

        if red and not grp:
            verdict = "redundancy"
        elif grp and not red:
            verdict = "group_structure"
        else:
            verdict = "neither"

        rep["group_structure_test"]["p_one_sided_within_gt_mixed_holm"] = p_h
        rep["model_level_redundancy_supported"] = red
        rep["model_level_group_structure_supported"] = grp
        rep["model_level_verdict"] = verdict

    n_red = sum(1 for m in eligible_models if bool(model_reports.get(m, {}).get("model_level_redundancy_supported", False)))
    n_grp = sum(1 for m in eligible_models if bool(model_reports.get(m, {}).get("model_level_group_structure_supported", False)))

    if len(eligible_models) < 2:
        final = "B3_inconclusive"
        reason = "insufficient_eligible_models"
    elif n_red >= 2 and n_grp == 0:
        final = "B3_redundancy_supported"
        reason = "none"
    elif n_grp >= 2 and n_red == 0:
        final = "B3_group_structure_supported"
        reason = "none"
    elif n_red > 0 and n_grp > 0:
        final = "B3_inconclusive"
        reason = "model_conditional_mechanism"
    else:
        has_noninformative = any(
            model_reports.get(m, {}).get("status") == "ok"
            and not bool(model_reports.get(m, {}).get("contribution", {}).get("informative", False))
            for m in eligible_models
        )
        has_low_cluster = any(
            safe_float(model_reports.get(m, {}).get("contribution", {}).get("n_confirmatory_eligible_clusters", float("nan"))) < 2
            for m in eligible_models
        )
        if has_low_cluster:
            reason = "insufficient_clusters"
        elif has_noninformative:
            reason = "noninformative_contribution"
        else:
            reason = "other"
        final = "B3_inconclusive"

    contribution_payload = {
        "timestamp": timestamp_now(),
        "experiment_id": "B3_cluster_ablation",
        "models": {
            m: {
                "informative": bool(model_reports.get(m, {}).get("contribution", {}).get("informative", False)),
                "non_informative_reason": str(model_reports.get(m, {}).get("contribution", {}).get("non_informative_reason", "none")),
                "n_confirmatory_eligible_clusters": int(
                    safe_float(model_reports.get(m, {}).get("contribution", {}).get("n_confirmatory_eligible_clusters", 0.0))
                )
                if model_reports.get(m, {}).get("status") == "ok"
                else 0,
                "max_cluster_share": safe_float(model_reports.get(m, {}).get("contribution", {}).get("max_cluster_share", float("nan"))),
                "cluster_share": model_reports.get(m, {}).get("contribution", {}).get("cluster_share", {}),
                "sum_positive_auc": safe_float(model_reports.get(m, {}).get("contribution", {}).get("sum_positive_auc", float("nan"))),
            }
            for m in models
            if model_reports.get(m, {}).get("status") == "ok"
        },
    }

    fit_payload = {
        "timestamp": timestamp_now(),
        "models": {
            m: model_reports[m].get("curve_fit", {})
            for m in models
            if model_reports.get(m, {}).get("status") == "ok"
        },
    }

    collapse_payload = {
        "timestamp": timestamp_now(),
        "models": {
            m: {
                "auc_drop": model_reports[m].get("auc_drop", {}),
                "collapse_fraction": model_reports[m].get("collapse_fraction", {}),
                "d_auc_within_minus_mixed": safe_float(model_reports[m].get("d_auc_within_minus_mixed", float("nan"))),
                "d_fc_mixed_minus_within": safe_float(model_reports[m].get("d_fc_mixed_minus_within", float("nan"))),
                "group_structure_test": model_reports[m].get("group_structure_test", {}),
            }
            for m in models
            if model_reports.get(m, {}).get("status") == "ok"
        },
    }

    verdict = {
        "timestamp": timestamp_now(),
        "experiment_id": "B3_cluster_ablation",
        "B3_redundancy_supported": bool(final == "B3_redundancy_supported"),
        "B3_group_structure_supported": bool(final == "B3_group_structure_supported"),
        "B3_inconclusive": bool(final == "B3_inconclusive"),
        "B3_inconclusive_reason": reason,
        "n_reliability_eligible_models": int(len(eligible_models)),
        "holm_family_p_values": {str(k): safe_float(v) for k, v in p_holm.items()},
        "per_model_mechanism_breakdown": [
            {
                "model_name": m,
                "model_level_redundancy_supported": bool(model_reports.get(m, {}).get("model_level_redundancy_supported", False)),
                "model_level_group_structure_supported": bool(model_reports.get(m, {}).get("model_level_group_structure_supported", False)),
                "model_level_verdict": str(model_reports.get(m, {}).get("model_level_verdict", "neither")),
            }
            for m in eligible_models
        ],
        "models": model_reports,
    }

    b3_spec = {
        "timestamp": timestamp_now(),
        "model_spec": {
            "linear": "y = a + b*f",
            "threshold_piecewise": "y = a + b1*f + b2*max(0, f-tau)",
            "tau_bounds": [0.1, 0.9],
            "selection_metric": "AICc",
            "k_linear": 2,
            "k_threshold": 4,
            "fraction_grid": [float(x) for x in fractions],
            "matching_constraints": {
                "head_count": "exact",
                "layer_hist_tvd_max": 0.10,
                "layer_dist_p_min": 0.10,
                "magnitude_relative_tolerance": 0.05,
            },
        },
        "confirmatory_precondition": {
            "min_fraction_points": 11,
            "model_selection_confirmatory_models": [m for m in eligible_models if model_reports.get(m, {}).get("model_selection_confirmatory", False)],
        },
    }

    write_json(out_dir / "matching_diagnostics.json", matching_diag_payload)
    write_json(out_dir / "cluster_contribution_share.json", contribution_payload)
    write_json(out_dir / "curve_fit_by_condition.json", fit_payload)
    write_json(out_dir / "collapse_point_comparison.json", collapse_payload)
    write_json(out_dir / "mechanism_disambiguation_verdict.json", verdict)
    write_json(out_dir / "b3_model_spec_and_fit_report.json", b3_spec)

    prereg = {
        "experiment_id": "B3_cluster_ablation",
        "question": "Is threshold behavior due to distributed redundancy or cluster-structured removal artifacts?",
        "primary_hypothesis": "Exact triplet-matched within/mixed/random curves with AICc model selection disambiguate mechanism.",
        "primary_endpoints": [
            "DeltaAICc threshold-vs-linear in mixed/random controls",
            "AUC_drop(within)-AUC_drop(mixed)",
            "collapse fraction shift",
            "isolated-cluster contribution share",
        ],
        "secondary_endpoints": ["per-model mechanism heterogeneity"],
        "model_list": models,
        "dataset_sources": [str(b1_path), str(b2a_path), str(calibration_root / "calibration_v1_manifest.json")],
        "inclusion_exclusion_rules": [
            "Primary inference uses only exact layer-matched and magnitude-matched triplets.",
            "Model-selection confirmatory eligibility requires >=11 confirmatory ablation fractions.",
        ],
        "sample_size_plan": {"fractions": [float(x) for x in fractions], "n_triplets_per_cluster": int(args.n_triplets_per_cluster)},
        "seed_plan": {"seed": int(args.seed)},
        "stopping_rule": "Stop after all model-level and cross-model preregistered criteria are evaluated.",
        "multiplicity_family": ["Holm across reliability-eligible per-model within>mixed tests"],
        "acceptance_criteria": ["B3.4/B3.5 criteria per TODO."],
        "fallback_interpretation_if_null": "Use threshold-capacity language and route to B_partial.",
    }

    manifest = command_manifest(
        experiment_id="B3_cluster_ablation",
        command="run_b3_cluster_ablation.py",
        model="+".join(models),
        extras={
            "execution_mode": execution_mode,
            "partial_root": str(partial_root),
            "resume_partials": bool(resume_partials),
            "models": models,
            "all_models": all_models,
            "b1_root": str(b1_root),
            "b2a_root": str(b2a_root),
            "calibration_root": str(calibration_root),
            "fractions": [float(x) for x in fractions],
            "n_triplets_per_cluster": int(args.n_triplets_per_cluster),
            "seed": int(args.seed),
        },
    )

    canonical_eligible = bool(len(eligible_models) >= 2)
    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "B3_cluster_ablation",
        "analysis_tier": "confirmatory" if canonical_eligible else "exploratory",
        "canonical_eligible": canonical_eligible,
        "override_used": False,
        "execution_mode": execution_mode,
        "partial_root": str(partial_root),
        "resume_partials": bool(resume_partials),
        "verdict": {"final": final, "reason": reason},
        "n_reliability_eligible_models": int(len(eligible_models)),
        "criteria": {
            "n_models_redundancy_supported": int(n_red),
            "n_models_group_structure_supported": int(n_grp),
        },
    }

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "B3_cluster_ablation",
        "claim_status": "supported" if final in {"B3_redundancy_supported", "B3_group_structure_supported"} else "mixed",
        "supports_main_text": bool(final in {"B3_redundancy_supported", "B3_group_structure_supported"}),
        "strict_only": True,
        "notes": [
            "Primary verdict uses exact layer/magnitude matched triplets only.",
            "If inconclusive, keep threshold-capacity wording and avoid over-disambiguation.",
        ],
    }

    data_dictionary = {
        "experiment_id": "B3_cluster_ablation",
        "tables": [
            {
                "path": str(out_dir / "cluster_ablation_curve.parquet"),
                "description": "Triplet-matched ablation rows for within/mixed/random conditions.",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model."},
                    {"name": "triplet_id", "dtype": "str", "description": "Matched triplet id."},
                    {"name": "condition", "dtype": "str", "description": "within/mixed/random."},
                    {"name": "ablation_fraction", "dtype": "float", "description": "Ablation fraction in percent."},
                    {"name": "degradation", "dtype": "float", "description": "Degradation metric."},
                    {"name": "confirmatory", "dtype": "bool", "description": "Exact match pass flag."},
                ],
            }
        ],
    }

    emit_core_artifacts(
        experiment_id="B3_cluster_ablation",
        out_dir=out_dir,
        preregistration=prereg,
        manifest=manifest,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    write_json(out_dir / "B3_claim_impact.json", claim_impact)

    print(f"[B3] wrote {out_dir}")


if __name__ == "__main__":
    main()
