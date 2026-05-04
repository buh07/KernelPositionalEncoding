#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp2.common import RESULTS_ROOT, command_manifest, ensure_dir, timestamp_now, write_json  # noqa: E402
from reinforce_exp2.scripts._shared import (  # noqa: E402
    emit_core_artifacts,
    head_dataframe_from_kernels,
    holm_adjust_dict,
    parse_models_arg,
    safe_float,
)
from shared.models.loading import load_tokenizer  # noqa: E402
from experiment3.theory1_si_circuits import MODELS  # noqa: E402
from experiment3.theory5b_boundary_detection import load_wiki_sequences  # noqa: E402


def _is_word_initial(token_str: str) -> bool:
    t = str(token_str or "")
    return bool(t.startswith("\u0120") or t.startswith("\u2581"))


def _build_boundary_profile(model_name: str, seq_len: int, n_sequences: int, seed: int) -> pd.DataFrame:
    tokenizer = load_tokenizer(MODELS[model_name])
    seqs = load_wiki_sequences(model_name=model_name, max_sequences=max(8, int(n_sequences)), seq_len=max(64, int(seq_len)))
    seqs = seqs[: int(n_sequences)]

    n_delta = int(seq_len)
    cnt = np.zeros(n_delta, dtype=np.float64)
    tot = np.zeros(n_delta, dtype=np.float64)

    for seq in seqs:
        toks = tokenizer.convert_ids_to_tokens([int(x) for x in seq])
        is_b = np.asarray([1.0 if _is_word_initial(t) else 0.0 for t in toks], dtype=np.float64)
        n = len(is_b)
        for i in range(n):
            max_d = min(n_delta - 1, i)
            if max_d <= 0:
                continue
            for d in range(1, max_d + 1):
                j = i - d
                cnt[d] += is_b[j]
                tot[d] += 1.0

    p = cnt / np.maximum(tot, 1.0)
    rows = []
    for d in range(1, n_delta):
        rows.append(
            {
                "model": model_name,
                "frame": "key_relative",
                "delta": int(d),
                "p_boundary": float(p[d]),
                "n_obs": int(tot[d]),
            }
        )
    return pd.DataFrame(rows)


def _alignment_metrics(kernel: np.ndarray, profile: np.ndarray) -> tuple[float, float]:
    k = np.asarray(kernel, dtype=np.float64)
    p = np.asarray(profile, dtype=np.float64)
    n = min(len(k), len(p))
    if n < 8:
        return float("nan"), float("nan")
    k = k[:n]
    p = p[:n]

    kz = k - np.nanmean(k)
    pz = p - np.nanmean(p)
    ak = np.abs(k) - np.nanmean(np.abs(k))

    den_signed = float(np.linalg.norm(kz) * np.linalg.norm(pz))
    den_abs = float(np.linalg.norm(ak) * np.linalg.norm(pz))
    signed = float(np.dot(kz, pz) / max(den_signed, 1e-12))
    abs_align = float(np.dot(ak, pz) / max(den_abs, 1e-12))
    return signed, abs_align


def _null_scores(kernel: np.ndarray, profile: np.ndarray, n_null: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(kernel, dtype=np.float64)
    p = np.asarray(profile, dtype=np.float64)
    n = min(len(k), len(p))
    k = k[:n]
    p = p[:n]
    signed_vals = []
    abs_vals = []

    for _ in range(max(8, int(n_null))):
        shift = int(rng.integers(1, max(2, n)))
        p_shift = np.roll(p, shift)
        s, a = _alignment_metrics(k, p_shift)
        signed_vals.append(s)
        abs_vals.append(a)

    return np.asarray(signed_vals, dtype=np.float64), np.asarray(abs_vals, dtype=np.float64)


def _hedges_g_vs_null(obs: np.ndarray, null_mean: np.ndarray) -> float:
    x = np.asarray(obs, dtype=np.float64)
    y = np.asarray(null_mean, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    n1, n2 = len(x), len(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = math.sqrt(max(((n1 - 1) * v1 + (n2 - 1) * v2) / max(1, n1 + n2 - 2), 1e-12))
    d = float((np.mean(x) - np.mean(y)) / sp)
    j = 1.0 - 3.0 / max(1.0, (4.0 * (n1 + n2) - 9.0))
    return float(j * d)


def _model_level_null_resampling_test(
    observed: np.ndarray,
    null_per_head: list[np.ndarray],
    *,
    n_draws: int,
    seed: int,
) -> dict[str, float]:
    obs = np.asarray(observed, dtype=np.float64)
    keep = np.isfinite(obs)
    obs = obs[keep]
    null_sets = [np.asarray(v, dtype=np.float64) for v in null_per_head]
    null_sets = [v[np.isfinite(v)] for v in null_sets]
    null_sets = [v for v, k in zip(null_sets, keep.tolist()) if k]
    if obs.size < 2 or not null_sets or any(v.size == 0 for v in null_sets):
        return {"p_one_gt": float("nan"), "z_score": float("nan"), "obs_mean": float(np.nanmean(obs)) if obs.size else float("nan"), "null_mean": float("nan")}

    rng = np.random.default_rng(int(seed))
    m = max(2000, int(n_draws))
    null_means = np.empty(m, dtype=np.float64)
    for i in range(m):
        vals = []
        for arr in null_sets:
            j = int(rng.integers(0, arr.size))
            vals.append(float(arr[j]))
        null_means[i] = float(np.mean(np.asarray(vals, dtype=np.float64)))

    obs_mean = float(np.mean(obs))
    p_one = float((np.sum(null_means >= obs_mean) + 1) / (m + 1))
    sd = float(np.std(null_means, ddof=1))
    z = float((obs_mean - float(np.mean(null_means))) / max(sd, 1e-12))
    return {"p_one_gt": p_one, "z_score": z, "obs_mean": obs_mean, "null_mean": float(np.mean(null_means))}


def run_model(model_name: str, seq_len: int, n_sequences: int, n_null: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mdf = head_dataframe_from_kernels(model_name)
    profile_df = _build_boundary_profile(model_name, seq_len=seq_len, n_sequences=n_sequences, seed=seed)
    profile = profile_df.sort_values("delta")["p_boundary"].to_numpy(dtype=np.float64)

    stable_offset = int(hashlib.sha256(str(model_name).encode("utf-8")).hexdigest()[:8], 16) % 100000
    rng = np.random.default_rng(int(seed) + stable_offset)
    rows = []
    null_signed_per_head: list[np.ndarray] = []
    null_absolute_per_head: list[np.ndarray] = []
    for row in mdf.itertuples():
        kernel = np.asarray(row.kernel, dtype=np.float64)
        signed, abs_align = _alignment_metrics(kernel, profile)
        null_s, null_a = _null_scores(kernel, profile, n_null=n_null, rng=rng)
        null_signed_per_head.append(null_s)
        null_absolute_per_head.append(null_a)
        rows.append(
            {
                "model": model_name,
                "layer": int(row.layer),
                "head": int(row.head),
                "head_key": str(row.head_key),
                "signed_alignment": float(signed),
                "absolute_alignment": float(abs_align),
                "null_signed_mean": float(np.nanmean(null_s)),
                "null_absolute_mean": float(np.nanmean(null_a)),
                "delta_signed": float(signed - np.nanmean(null_s)),
                "delta_absolute": float(abs_align - np.nanmean(null_a)),
                "null_signed_std": float(np.nanstd(null_s, ddof=1) if len(null_s) > 1 else 0.0),
                "null_absolute_std": float(np.nanstd(null_a, ddof=1) if len(null_a) > 1 else 0.0),
            }
        )

    hdf = pd.DataFrame(rows)

    # model-level tests
    del_s = hdf["delta_signed"].to_numpy(dtype=np.float64)
    del_a = hdf["delta_absolute"].to_numpy(dtype=np.float64)
    obs_signed = hdf["signed_alignment"].to_numpy(dtype=np.float64)
    obs_absolute = hdf["absolute_alignment"].to_numpy(dtype=np.float64)

    rs_signed = _model_level_null_resampling_test(
        obs_signed,
        null_signed_per_head,
        n_draws=4096,
        seed=int(seed) + 7919,
    )
    rs_absolute = _model_level_null_resampling_test(
        obs_absolute,
        null_absolute_per_head,
        n_draws=4096,
        seed=int(seed) + 1237,
    )

    # retained for continuity as descriptive moments only; confirmatory p-values use null-resampling tests above.
    t_s, p2_s = scipy_stats.ttest_1samp(del_s, popmean=0.0, nan_policy="omit")
    t_a, p2_a = scipy_stats.ttest_1samp(del_a, popmean=0.0, nan_policy="omit")
    p1_s = safe_float(rs_signed["p_one_gt"])
    p1_a = safe_float(rs_absolute["p_one_gt"])

    se_s = float(np.nanstd(del_s, ddof=1) / max(math.sqrt(np.sum(np.isfinite(del_s))), 1.0))
    se_a = float(np.nanstd(del_a, ddof=1) / max(math.sqrt(np.sum(np.isfinite(del_a))), 1.0))

    valid_var = bool(len(del_s[np.isfinite(del_s)]) >= 10 and np.isfinite(se_s) and se_s > 0 and len(del_a[np.isfinite(del_a)]) >= 10 and np.isfinite(se_a) and se_a > 0)

    effect_signed = _hedges_g_vs_null(hdf["signed_alignment"].to_numpy(dtype=np.float64), hdf["null_signed_mean"].to_numpy(dtype=np.float64))
    effect_abs = _hedges_g_vs_null(hdf["absolute_alignment"].to_numpy(dtype=np.float64), hdf["null_absolute_mean"].to_numpy(dtype=np.float64))

    summary = {
        "model": model_name,
        "n_heads": int(hdf.shape[0]),
        "valid_variance_estimate": bool(valid_var),
        "signed": {
            "delta_mean": float(np.nanmean(del_s)),
            "t_stat": safe_float(t_s),
            "t_stat_descriptive": safe_float(t_s),
            "z_stat_null_resampling": safe_float(rs_signed["z_score"]),
            "p_one": safe_float(p1_s),
            "hedges_g_vs_null": safe_float(effect_signed),
            "se": se_s,
            "test_method": "model_level_null_resampling",
        },
        "absolute": {
            "delta_mean": float(np.nanmean(del_a)),
            "t_stat": safe_float(t_a),
            "t_stat_descriptive": safe_float(t_a),
            "z_stat_null_resampling": safe_float(rs_absolute["z_score"]),
            "p_one": safe_float(p1_a),
            "hedges_g_vs_null": safe_float(effect_abs),
            "se": se_a,
            "test_method": "model_level_null_resampling",
        },
        "supports_directional_subclaim": bool(np.isfinite(effect_signed) and abs(effect_signed) >= 0.20 and np.isfinite(p1_s) and p1_s < 0.05),
        "supports_nondirectional_practical": bool(np.isfinite(effect_abs) and effect_abs >= 0.20 and np.isfinite(p1_a) and p1_a < 0.05),
    }
    return profile_df, hdf, summary


def main() -> None:
    p = argparse.ArgumentParser(description="B2a Head-level kernel-offset alignment", allow_abbrev=False)
    p.add_argument("--models", default="llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1")
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "B2a_head_alignment"))
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--num-sequences", type=int, default=32)
    p.add_argument("--n-null", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260417)
    args = p.parse_args()

    models = parse_models_arg(args.models)
    out_dir = ensure_dir(Path(args.output_root))

    profile_rows = []
    score_rows = []
    model_summaries = {}

    for i, model in enumerate(models):
        pdf, hdf, summ = run_model(
            model,
            seq_len=max(64, int(args.seq_len)),
            n_sequences=max(8, int(args.num_sequences)),
            n_null=max(8, int(args.n_null)),
            seed=int(args.seed) + i * 101,
        )
        profile_rows.append(pdf)
        score_rows.append(hdf)
        model_summaries[model] = summ

    profile_df = pd.concat(profile_rows, ignore_index=True)
    head_df = pd.concat(score_rows, ignore_index=True)

    profile_df.to_parquet(out_dir / "boundary_offset_profiles.parquet", index=False)
    head_df.to_parquet(out_dir / "head_alignment_scores.parquet", index=False)

    # corrected p-values
    p_family = {}
    for model, summ in model_summaries.items():
        p_family[f"{model}::signed"] = safe_float(summ["signed"]["p_one"])
        p_family[f"{model}::absolute"] = safe_float(summ["absolute"]["p_one"])
    p_holm = holm_adjust_dict(p_family)

    n_valid_models = 0
    n_abs_supported = 0
    n_signed_supported = 0
    n_p_supported = 0
    n_g_supported = 0
    for model, summ in model_summaries.items():
        valid = bool(summ["valid_variance_estimate"])
        if valid:
            n_valid_models += 1
        p_abs = safe_float(p_holm.get(f"{model}::absolute"))
        p_sgn = safe_float(p_holm.get(f"{model}::signed"))
        p_ok = bool(valid and np.isfinite(p_abs) and p_abs < 0.05)
        g_ok = bool(valid and safe_float(summ["absolute"]["hedges_g_vs_null"]) >= 0.20)
        abs_ok = bool(p_ok and g_ok)
        sgn_ok = bool(valid and np.isfinite(p_sgn) and p_sgn < 0.05 and abs(safe_float(summ["signed"]["hedges_g_vs_null"])) >= 0.20)
        if p_ok:
            n_p_supported += 1
        if g_ok:
            n_g_supported += 1
        if abs_ok:
            n_abs_supported += 1
        if sgn_ok:
            n_signed_supported += 1
        summ["signed"]["p_one_holm_family"] = p_sgn
        summ["absolute"]["p_one_holm_family"] = p_abs
        summ["supports_nondirectional_practical_holm"] = abs_ok
        summ["supports_directional_subclaim_holm"] = sgn_ok

    c1 = bool(n_valid_models >= 2)
    c2 = bool(n_p_supported >= 2)
    c3 = bool(n_g_supported >= 2)
    c4 = bool(n_signed_supported >= 2)
    b2a_supported = bool(c1 and c2 and c3)

    sign_report = {
        "timestamp": timestamp_now(),
        "experiment_id": "B2a_head_alignment",
        "reference_frame": "key_relative",
        "sign_conventions": {
            "positive_signed_alignment": "boundary-seeking",
            "negative_signed_alignment": "boundary-suppressing",
            "absolute_alignment": "magnitude-only boundary-offset coupling",
        },
        "models": model_summaries,
    }
    write_json(out_dir / "alignment_sign_convention_report.json", sign_report)
    write_json(out_dir / "alignment_null_tests.json", {"timestamp": timestamp_now(), "p_one_holm": p_holm, "models": model_summaries})

    prereg = {
        "experiment_id": "B2a_head_alignment",
        "question": "Do head kernels align with empirical boundary-offset profiles?",
        "primary_hypothesis": "Observed alignment exceeds smoothness-preserving null baselines across reliability-eligible primary models.",
        "primary_endpoints": ["signed alignment", "absolute alignment", "model-level delta vs null"],
        "secondary_endpoints": ["directional subclaim eligibility"],
        "model_list": models,
        "dataset_sources": [
            "results/experiment3/theory8_position_ablation/*/estimated_kernels.json",
            "wiki held-out sequences via experiment3/theory5b loader",
        ],
        "inclusion_exclusion_rules": ["Model-level aggregate requires n_heads>=10 and finite positive SE."],
        "sample_size_plan": {"num_sequences": int(args.num_sequences), "n_null": int(args.n_null)},
        "seed_plan": {"seed": int(args.seed)},
        "stopping_rule": "Stop after all models complete alignment and null tests.",
        "multiplicity_family": ["{alignment_metric in [signed, absolute]} x {model}"],
        "acceptance_criteria": ["B2a.6 criteria per TODO."],
        "fallback_interpretation_if_null": "Keep alignment claims descriptive/exploratory only.",
    }
    manifest = command_manifest(
        experiment_id="B2a_head_alignment",
        command="run_b2a_head_alignment.py",
        model="+".join(models),
        extras={"models": models, "num_sequences": int(args.num_sequences), "n_null": int(args.n_null), "seed": int(args.seed)},
    )
    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "B2a_head_alignment",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "criteria": {
            "criterion_1_estimable_valid_variance": c1,
            "criterion_2_corrected_significance": c2,
            "criterion_3_practical_effect": c3,
            "criterion_4_directional_subclaim": c4,
        },
        "verdict": {"B2a_supported": b2a_supported},
    }
    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "B2a_head_alignment",
        "claim_status": "supported" if b2a_supported else "mixed",
        "supports_main_text": bool(b2a_supported),
        "strict_only": True,
        "notes": ["Directional language should only be used if directional criterion passes."],
    }
    data_dictionary = {
        "experiment_id": "B2a_head_alignment",
        "tables": [
            {
                "path": str(out_dir / "boundary_offset_profiles.parquet"),
                "description": "Boundary probability by relative offset.",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model."},
                    {"name": "delta", "dtype": "int", "description": "Relative key offset."},
                    {"name": "p_boundary", "dtype": "float", "description": "Empirical boundary probability."},
                    {"name": "n_obs", "dtype": "int", "description": "Count of observations."},
                ],
            },
            {
                "path": str(out_dir / "head_alignment_scores.parquet"),
                "description": "Per-head alignment metrics and null deltas.",
                "columns": [
                    {"name": "head_key", "dtype": "str", "description": "Head identifier."},
                    {"name": "signed_alignment", "dtype": "float", "description": "Signed alignment."},
                    {"name": "absolute_alignment", "dtype": "float", "description": "Absolute alignment."},
                    {"name": "delta_signed", "dtype": "float", "description": "Signed delta vs circular-shift null."},
                    {"name": "delta_absolute", "dtype": "float", "description": "Absolute delta vs circular-shift null."},
                ],
            },
        ],
    }

    emit_core_artifacts(
        experiment_id="B2a_head_alignment",
        out_dir=out_dir,
        preregistration=prereg,
        manifest=manifest,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    write_json(out_dir / "B2a_claim_impact.json", claim_impact)

    print(f"[B2a] wrote {out_dir}")


if __name__ == "__main__":
    main()
