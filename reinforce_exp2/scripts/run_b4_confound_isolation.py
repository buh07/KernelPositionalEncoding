#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp2.common import RESULTS_ROOT, command_manifest, ensure_dir, load_head_groups, timestamp_now, write_json  # noqa: E402
from reinforce_exp2.scripts._shared import emit_core_artifacts, head_dataframe_from_kernels, holm_adjust_dict, parse_models_arg, safe_float  # noqa: E402

try:
    import torch as _torch
    from experiment3.theory1_si_circuits import MODELS as _MODELS  # noqa: E402
    from shared.models.loading import load_model, load_tokenizer  # noqa: E402
    _MODEL_INFERENCE_AVAILABLE = True
except ImportError:
    _MODEL_INFERENCE_AVAILABLE = False


def _compute_attention_entropy_per_head(
    model,
    device: str,
    calibration_tokens: list[list[int]],
    max_seqs: int = 256,
) -> dict[str, float]:
    """Compute E_h = median_{s,i}(H(h,s,i)) from actual attention weights.

    H(h,s,i) = -sum_{j<=i} a_{h,s,i,j} log(a_{h,s,i,j}) / log(i) for i>=2
    """
    model.eval()
    head_entropy_vals: dict[str, list[float]] = {}

    seqs = calibration_tokens[:max_seqs]
    for seq in seqs:
        toks = _torch.tensor([seq], dtype=_torch.long, device=device)
        with _torch.inference_mode():
            out = model(input_ids=toks, output_attentions=True, use_cache=False)
        attentions = out.attentions  # tuple: (n_layers,), each (1, n_heads, L, L)

        for layer_idx, attn_t in enumerate(attentions):
            attn_np = attn_t[0].cpu().float().numpy()  # (n_heads, L, L)
            n_heads, L, _ = attn_np.shape
            for h in range(n_heads):
                hk = f"L{layer_idx}H{h}"
                if hk not in head_entropy_vals:
                    head_entropy_vals[hk] = []
                for i in range(2, L):
                    row = attn_np[h, i, :i + 1]  # attention from pos i to pos 0..i
                    row = np.clip(row, 1e-30, 1.0)
                    row = row / max(float(np.sum(row)), 1e-12)
                    h_val = float(-np.sum(row * np.log(row)) / math.log(i))
                    head_entropy_vals[hk].append(h_val)

        del toks, out
        if device.startswith("cuda"):
            _torch.cuda.empty_cache()

    return {hk: float(np.median(np.asarray(vals, dtype=np.float64))) for hk, vals in head_entropy_vals.items() if vals}


def _kernel_entropy_fallback(vec: np.ndarray) -> float:
    """Fallback proxy: entropy of kernel weight distribution (not actual attention entropy)."""
    v = np.abs(np.asarray(vec, dtype=np.float64))
    p = v / max(np.sum(v), 1e-12)
    h = float(-np.sum(p * np.log(p + 1e-12)))
    n = len(p)
    return float(h / max(np.log(max(2, n)), 1e-12))


def _hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    n1, n2 = len(x), len(y)
    v1 = np.var(x, ddof=1)
    v2 = np.var(y, ddof=1)
    sp = math.sqrt(max(((n1 - 1) * v1 + (n2 - 1) * v2) / max(1, n1 + n2 - 2), 1e-12))
    d = float((np.mean(x) - np.mean(y)) / sp)
    j = 1.0 - 3.0 / max(1.0, 4.0 * (n1 + n2) - 9.0)
    return float(j * d)


def _effect_and_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    g = _hedges_g(x, y)
    if len(x) < 2 or len(y) < 2:
        return g, float("nan")
    t, p = scipy_stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return g, safe_float(p)


def _ivw_layer_meta(layer_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eff = []
    var = []
    for r in layer_rows:
        g = safe_float(r["hedges_g"])
        n1 = int(r["n_high"])
        n2 = int(r["n_low"])
        if not np.isfinite(g) or n1 < 2 or n2 < 2:
            continue
        vg = float((n1 + n2) / max(n1 * n2, 1) + (g * g) / max(2 * (n1 + n2 - 2), 1))
        eff.append(g)
        var.append(vg)
    if not eff:
        return {"ivw_hedges_g": float("nan"), "ivw_ci95": [float("nan"), float("nan")], "p_two": float("nan"), "n_layers": 0}
    w = 1.0 / np.asarray(var, dtype=np.float64)
    e = np.asarray(eff, dtype=np.float64)
    mu = float(np.sum(w * e) / np.sum(w))
    se = float(math.sqrt(1.0 / np.sum(w)))
    lo = float(mu - 1.96 * se)
    hi = float(mu + 1.96 * se)
    z = float(mu / max(se, 1e-12))
    p_two = float(2 * (1.0 - scipy_stats.norm.cdf(abs(z))))
    return {"ivw_hedges_g": mu, "ivw_ci95": [lo, hi], "p_two": p_two, "n_layers": int(len(eff))}


def _sign_flip_two_sided(values: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n < 2:
        return float("nan")
    obs = float(abs(np.mean(v)))
    if n <= 15:
        stats = []
        for signs in itertools.product([-1.0, 1.0], repeat=n):
            s = np.asarray(signs, dtype=np.float64)
            stats.append(abs(float(np.mean(s * v))))
        arr = np.asarray(stats, dtype=np.float64)
        return float(np.sum(arr >= obs) / len(arr))
    rng = np.random.default_rng(int(seed))
    m = max(2000, int(n_perm))
    hits = 0
    for _ in range(m):
        s = rng.choice(np.asarray([-1.0, 1.0]), size=n, replace=True)
        stat = abs(float(np.mean(s * v)))
        if np.isfinite(stat) and stat >= obs:
            hits += 1
    return float((hits + 1) / (m + 1))


def _match_entropy_pairs(df: pd.DataFrame, tol: float = 0.02) -> pd.DataFrame:
    # deterministic 1:1 match without replacement inside each layer
    rows = []
    for layer, ldf in df.groupby("layer"):
        high = ldf[ldf["group"] == "high"].copy().sort_values("head_key")
        low = ldf[ldf["group"] == "low"].copy().sort_values("head_key")
        used_low = set()
        for hr in high.itertuples():
            cands = []
            for lr in low.itertuples():
                if lr.head_key in used_low:
                    continue
                de = abs(float(hr.entropy) - float(lr.entropy))
                if de <= tol:
                    cands.append((de, str(lr.head_key), lr))
            if not cands:
                continue
            cands.sort(key=lambda x: (x[0], x[1]))
            best = cands[0][2]
            used_low.add(str(best.head_key))
            rows.append(
                {
                    "layer": int(layer),
                    "high_head_key": str(hr.head_key),
                    "low_head_key": str(best.head_key),
                    "high_entropy": safe_float(hr.entropy),
                    "low_entropy": safe_float(best.entropy),
                    "abs_delta_entropy": abs(safe_float(hr.entropy) - safe_float(best.entropy)),
                    "high_boundary_score": safe_float(hr.boundary_attn_score),
                    "low_boundary_score": safe_float(best.boundary_attn_score),
                    "delta_boundary_score": safe_float(hr.boundary_attn_score) - safe_float(best.boundary_attn_score),
                }
            )
    return pd.DataFrame(rows)


def _sanitize_calibration_tokens_for_vocab(
    calibration_tokens: list[list[int]],
    vocab_size: int,
    min_len: int = 16,
) -> list[list[int]]:
    out: list[list[int]] = []
    vmax = max(1, int(vocab_size))
    for seq in calibration_tokens:
        if not isinstance(seq, list):
            continue
        if not seq:
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
        raise RuntimeError("output_attentions returned empty/None; cannot run B4 strict path")
    del ids, out
    if str(device).startswith("cuda"):
        _torch.cuda.empty_cache()


def run_model(
    model_name: str,
    b1_root: Path,
    attention_entropy: dict[str, float] | None = None,
) -> dict[str, Any]:
    kernels = head_dataframe_from_kernels(model_name)
    groups = load_head_groups(model_name)
    high = {f"L{int(x['layer'])}H{int(x['head'])}" for x in groups.get("high_si", [])}
    low = {f"L{int(x['layer'])}H{int(x['head'])}" for x in groups.get("low_si", [])}

    # attach boundary scores from B1 membership if available (already includes boundary scores)
    b1_mem = Path(b1_root) / "cluster_membership.parquet"
    if b1_mem.exists():
        b1 = pd.read_parquet(b1_mem)
        b1 = b1[b1["model"] == model_name][["head_key", "layer", "boundary_attn_score"]].drop_duplicates()
    else:
        b1 = pd.DataFrame(columns=["head_key", "layer", "boundary_attn_score"])

    df = kernels.merge(b1, on=["head_key", "layer"], how="left")
    if df["boundary_attn_score"].isna().all():
        # fallback direct source
        path = ROOT / "results" / "experiment3" / "theory5b_boundary_detection" / model_name / "boundary_attention_scores.parquet"
        if path.exists():
            bs = pd.read_parquet(path)
            bs["head_key"] = bs.apply(lambda r: f"L{int(r['layer'])}H{int(r['head'])}", axis=1)
            df = df.drop(columns=["boundary_attn_score"], errors="ignore").merge(bs[["head_key", "boundary_attn_score"]], on="head_key", how="left")

    entropy_source = "attention_forward" if attention_entropy is not None else "kernel_proxy"
    if attention_entropy is not None:
        # Use actual attention entropy from model forward passes (E_h = median H(h,s,i))
        df["entropy"] = df["head_key"].map(lambda hk: attention_entropy.get(str(hk), float("nan")))
    else:
        # Fallback: kernel weight distribution entropy (proxy only)
        df["entropy"] = df["kernel"].apply(_kernel_entropy_fallback)

    df["group"] = np.where(df["head_key"].isin(high), "high", np.where(df["head_key"].isin(low), "low", "other"))
    df = df[df["group"].isin(["high", "low"])].copy()

    within_rows = []
    for layer, ldf in df.groupby("layer"):
        x = ldf[ldf["group"] == "high"]["boundary_attn_score"].to_numpy(dtype=np.float64)
        y = ldf[ldf["group"] == "low"]["boundary_attn_score"].to_numpy(dtype=np.float64)
        if len(x) < 1 or len(y) < 1:
            continue
        g, p = _effect_and_p(x, y)
        within_rows.append(
            {
                "layer": int(layer),
                "n_high": int(np.sum(np.isfinite(x))),
                "n_low": int(np.sum(np.isfinite(y))),
                "mean_high": float(np.nanmean(x)),
                "mean_low": float(np.nanmean(y)),
                "hedges_g": safe_float(g),
                "p_two": safe_float(p),
            }
        )

    within_df = pd.DataFrame(within_rows)
    within_meta = _ivw_layer_meta(within_rows)

    match_df = _match_entropy_pairs(df, tol=0.02)
    if not match_df.empty:
        g_match = _hedges_g(match_df["high_boundary_score"].to_numpy(dtype=np.float64), match_df["low_boundary_score"].to_numpy(dtype=np.float64))
        p_two = _sign_flip_two_sided(match_df["delta_boundary_score"].to_numpy(dtype=np.float64), seed=20260421 + int(len(match_df)))
    else:
        g_match, p_two = float("nan"), float("nan")

    return {
        "model": model_name,
        "within_layer_rows": within_rows,
        "within_layer_meta": within_meta,
        "entropy_match": {
            "n_pairs": int(match_df.shape[0]),
            "hedges_g": safe_float(g_match),
            "p_two": safe_float(p_two),
        },
        "entropy_source": entropy_source,
        "match_df": match_df,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="B4 Confound-isolated high-vs-low comparison", allow_abbrev=False)
    p.add_argument("--models", default="llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1")
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "B4_confound_isolation"))
    p.add_argument("--b1-root", default=str(RESULTS_ROOT / "B1_kernel_taxonomy"))
    p.add_argument("--calibration-root", default=str(RESULTS_ROOT / "calibration_splits"))
    p.add_argument("--device-map", default="")
    p.add_argument("--max-seqs", type=int, default=256)
    args = p.parse_args()

    models = parse_models_arg(args.models)
    out_dir = ensure_dir(Path(args.output_root))

    device_map: dict[str, str] = {}
    for tok in [x.strip() for x in str(args.device_map).split(",") if x.strip()]:
        if ":" in tok:
            m, d = tok.split(":", 1)
            device_map[m.strip()] = d.strip()

    # Load calibration tokens for entropy computation
    calibration_tokens: list[list[int]] = []
    cal_tok_path = Path(args.calibration_root) / "calibration_v1_tokens.parquet"
    if cal_tok_path.exists():
        try:
            cal_df = pd.read_parquet(cal_tok_path)
            for row in cal_df.itertuples():
                toks = row.tokens
                if isinstance(toks, (list, np.ndarray)):
                    calibration_tokens.append([int(x) for x in toks])
        except Exception:
            pass

    reports = {}
    within_tables = []
    match_tables = []
    pvals = {}

    b1_root = Path(args.b1_root)
    for model in models:
        attention_entropy: dict[str, float] | None = None
        entropy_failure_reason: str | None = None
        entropy_failure_traceback: str | None = None
        device = device_map.get(model, "")
        strict_inference = bool(_MODEL_INFERENCE_AVAILABLE and device and calibration_tokens and model in _MODELS)
        if strict_inference:
            loaded = None
            model_obj = None
            try:
                tok = load_tokenizer(_MODELS[model])
                vocab_size = int(getattr(tok, "vocab_size", 0) or 0)
                safe_tokens = _sanitize_calibration_tokens_for_vocab(calibration_tokens, vocab_size=vocab_size, min_len=16)
                if len(safe_tokens) < 8:
                    raise RuntimeError(
                        f"Insufficient vocab-safe calibration sequences ({len(safe_tokens)}) for B4 strict path"
                    )
                if device == "auto":
                    loaded = load_model(_MODELS[model], device_map="auto")
                    model_obj = loaded.model
                    _input_device = str(next(model_obj.parameters()).device)
                else:
                    loaded = load_model(_MODELS[model])
                    model_obj = loaded.model.to(device)
                    _input_device = device
                model_obj.eval()
                _ensure_output_attentions_supported(model_obj, _input_device, safe_tokens[0])
                attention_entropy = _compute_attention_entropy_per_head(
                    model_obj,
                    device=_input_device,
                    calibration_tokens=safe_tokens,
                    max_seqs=int(args.max_seqs),
                )
            except Exception as exc:
                entropy_failure_reason = f"{type(exc).__name__}: {exc}"
                entropy_failure_traceback = traceback.format_exc(limit=3)
                attention_entropy = None
            finally:
                try:
                    if model_obj is not None:
                        del model_obj
                except Exception:
                    pass
                try:
                    if loaded is not None:
                        del loaded
                except Exception:
                    pass
                try:
                    load_model.cache_clear()  # release GPU memory held by lru_cache
                except Exception:
                    pass
                try:
                    _torch.cuda.empty_cache()
                except Exception:
                    pass

        rep = run_model(model, b1_root=b1_root, attention_entropy=attention_entropy)
        if entropy_failure_reason is not None:
            rep["attention_entropy_failure"] = {
                "failed": True,
                "reason": entropy_failure_reason,
                "traceback_tail": entropy_failure_traceback,
            }
        else:
            rep["attention_entropy_failure"] = {"failed": False}
        reports[model] = rep
        if rep["within_layer_rows"]:
            wdf = pd.DataFrame(rep["within_layer_rows"])
            wdf["model"] = model
            within_tables.append(wdf)
        mdf = rep["match_df"]
        if not mdf.empty:
            mdf = mdf.copy()
            mdf["model"] = model
            match_tables.append(mdf)

        pvals[f"{model}::within"] = safe_float(rep["within_layer_meta"]["p_two"])
        pvals[f"{model}::entropy"] = safe_float(rep["entropy_match"]["p_two"])

    within_cols = ["model", "layer", "n_high", "n_low", "mean_high", "mean_low", "hedges_g", "p_two"]
    match_cols = [
        "model",
        "layer",
        "high_head_key",
        "low_head_key",
        "high_entropy",
        "low_entropy",
        "abs_delta_entropy",
        "high_boundary_score",
        "low_boundary_score",
        "delta_boundary_score",
    ]

    if within_tables:
        wout = pd.concat(within_tables, ignore_index=True)
    else:
        wout = pd.DataFrame(columns=within_cols)
    for c in within_cols:
        if c not in wout.columns:
            wout[c] = np.nan
    wout[within_cols].to_parquet(out_dir / "within_layer_contrast.parquet", index=False)

    if match_tables:
        mout = pd.concat(match_tables, ignore_index=True)
    else:
        mout = pd.DataFrame(columns=match_cols)
    for c in match_cols:
        if c not in mout.columns:
            mout[c] = np.nan
    mout[match_cols].to_parquet(out_dir / "entropy_matched_contrast.parquet", index=False)

    p_holm = holm_adjust_dict(pvals)

    within_ok = 0
    entropy_ok = 0
    both_ok = 0
    entropy_eligible_models = 0

    for model in models:
        rep = reports[model]
        w = rep["within_layer_meta"]
        e = rep["entropy_match"]

        p_w = safe_float(p_holm.get(f"{model}::within")) if f"{model}::within" in p_holm else float("nan")
        p_e = safe_float(p_holm.get(f"{model}::entropy")) if f"{model}::entropy" in p_holm else float("nan")
        g_w = safe_float(w["ivw_hedges_g"])
        g_e = safe_float(e["hedges_g"])

        within_sig = bool(np.isfinite(p_w) and p_w < 0.05 and np.isfinite(g_w) and abs(g_w) >= 0.20)
        direct_entropy = str(rep.get("entropy_source", "")) == "attention_forward"
        entropy_eligible = bool(direct_entropy and int(e["n_pairs"]) >= 5)
        entropy_sig = bool(entropy_eligible and np.isfinite(p_e) and p_e < 0.05 and np.isfinite(g_e) and abs(g_e) >= 0.20)

        if within_sig:
            within_ok += 1
        if entropy_eligible:
            entropy_eligible_models += 1
        if entropy_sig:
            entropy_ok += 1

        sign_same = bool(np.isfinite(g_w) and np.isfinite(g_e) and np.sign(g_w) == np.sign(g_e)) if entropy_eligible else True
        atten = abs(g_e) / max(abs(g_w), 1e-8) if entropy_eligible and np.isfinite(g_w) and np.isfinite(g_e) else float("nan")
        retention = bool((not entropy_eligible) or (sign_same and np.isfinite(atten) and atten >= 0.50))

        if within_sig and (entropy_sig or not entropy_eligible) and retention:
            both_ok += 1

        rep["acceptance"] = {
            "within_sig_holm": within_sig,
            "direct_entropy_available": direct_entropy,
            "entropy_eligible": entropy_eligible,
            "entropy_sig_holm": entropy_sig,
            "sign_same": sign_same,
            "attenuation_ratio": safe_float(atten),
            "retention_ok": retention,
            "p_within_holm": safe_float(p_w),
            "p_entropy_holm": safe_float(p_e),
        }
        rep.pop("match_df", None)

    criterion1 = bool(within_ok >= 2)
    criterion2 = bool(within_ok >= 2 and (entropy_ok >= 2 or entropy_eligible_models < 2))
    criterion3 = bool(both_ok >= 2)
    b4_supported = bool(criterion1 and criterion2 and criterion3 and entropy_eligible_models >= 2)

    summary_obj = {
        "timestamp": timestamp_now(),
        "experiment_id": "B4_confound_isolation",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "models": reports,
        "holm_family": p_holm,
        "criteria": {
            "criterion1_direction_retention": criterion1,
            "criterion2_statistical_retention": criterion2,
            "criterion3_practical_retention": criterion3,
            "entropy_eligible_models": int(entropy_eligible_models),
        },
        "verdict": {
            "B4_supported": b4_supported,
            "B4_mixed": bool(not b4_supported),
        },
        "limitations": [
            "Entropy metric is computed from kernel-distribution entropy as a proxy when direct attention-entropy traces are unavailable.",
            "Entropy matching is confirmatory only for models with >=5 matched pairs.",
        ],
    }

    prereg = {
        "experiment_id": "B4_confound_isolation",
        "question": "Do high-vs-low SI effects persist after depth and entropy controls?",
        "primary_hypothesis": "Within-layer and entropy-matched contrasts retain sign, significance, and practical magnitude.",
        "primary_endpoints": ["within-layer IVW effect", "entropy-matched effect", "Holm-corrected significance"],
        "secondary_endpoints": ["retention ratio vs unadjusted"],
        "model_list": models,
        "dataset_sources": [
            "results/experiment3/theory8_position_ablation/*/estimated_kernels.json",
            "results/experiment3/theory5b_boundary_detection/*/boundary_attention_scores.parquet",
            "results/experiment3/theory1_si_circuits/*/head_groups.json",
        ],
        "inclusion_exclusion_rules": ["Entropy confirmatory eligibility requires >=5 matched pairs per model."],
        "sample_size_plan": {"matching_tolerance": 0.02},
        "seed_plan": {"deterministic": True},
        "stopping_rule": "Stop when all model contrasts are computed.",
        "multiplicity_family": ["{contrast_type in [within-layer, entropy-matched]} x {models}"],
        "acceptance_criteria": ["B4.3 criteria per TODO."],
        "fallback_interpretation_if_null": "Downgrade SI-vs-low comparative language to caveated/descriptive.",
    }
    manifest = command_manifest(
        experiment_id="B4_confound_isolation",
        command="run_b4_confound_isolation.py",
        model="+".join(models),
        extras={"models": models, "b1_root": str(b1_root)},
    )
    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "B4_confound_isolation",
        "claim_status": "supported" if b4_supported else "mixed",
        "supports_main_text": bool(b4_supported),
        "strict_only": True,
        "notes": ["If mixed due entropy eligibility shortfall, report within-layer as confirmatory and entropy as exploratory/caveated."],
    }
    data_dictionary = {
        "experiment_id": "B4_confound_isolation",
        "tables": [
            {
                "path": str(out_dir / "within_layer_contrast.parquet"),
                "description": "Layer-stratified high-vs-low contrasts.",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model."},
                    {"name": "layer", "dtype": "int", "description": "Layer."},
                    {"name": "hedges_g", "dtype": "float", "description": "Within-layer effect size."},
                    {"name": "p_two", "dtype": "float", "description": "Per-layer two-sided p-value."},
                ],
            },
            {
                "path": str(out_dir / "entropy_matched_contrast.parquet"),
                "description": "Entropy-matched high-vs-low pairs.",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model."},
                    {"name": "layer", "dtype": "int", "description": "Layer."},
                    {"name": "high_head_key", "dtype": "str", "description": "Matched high-SI head."},
                    {"name": "low_head_key", "dtype": "str", "description": "Matched low-SI head."},
                    {"name": "delta_boundary_score", "dtype": "float", "description": "High minus low boundary score."},
                ],
            },
        ],
    }

    emit_core_artifacts(
        experiment_id="B4_confound_isolation",
        out_dir=out_dir,
        preregistration=prereg,
        manifest=manifest,
        summary=summary_obj,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    write_json(out_dir / "B4_claim_impact.json", claim_impact)

    print(f"[B4] wrote {out_dir}")


if __name__ == "__main__":
    main()
