#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, command_manifest, ensure_dir, read_json, safe_float, timestamp_now, write_json  # noqa: E402

PRIMARY_MODELS = ("llama-3.1-8b", "olmo-2-7b")
DEFAULT_OUT = RESULTS_ROOT / "exp_r2_dependence_reanalysis"


def _one_sided_from_t_p(t_stat: float, p_two: float, positive: bool = True) -> float:
    if not np.isfinite(t_stat) or not np.isfinite(p_two):
        return float("nan")
    p_two = float(max(min(p_two, 1.0), 0.0))
    if positive:
        return float(p_two / 2.0) if t_stat >= 0 else float(1.0 - p_two / 2.0)
    return float(p_two / 2.0) if t_stat <= 0 else float(1.0 - p_two / 2.0)


def _cluster_bootstrap_mean(
    *,
    df: pd.DataFrame,
    cluster_col: str,
    value_col: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    clusters = sorted(df[cluster_col].dropna().unique().tolist())
    if not clusters:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_vals = []
    for _ in range(max(1, n_boot)):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        bdf = pd.concat([df[df[cluster_col] == c] for c in sampled], axis=0, ignore_index=True)
        boot_vals.append(float(np.nanmean(bdf[value_col].astype(float).values)))
    arr = np.asarray(boot_vals, dtype=float)
    return float(np.nanmean(arr)), float(np.nanquantile(arr, 0.025)), float(np.nanquantile(arr, 0.975))


def _sequence_level_boundary_test(model: str, n_boot: int, seed: int) -> dict[str, Any]:
    base = ROOT / "results" / "experiment3_phase2" / "exp3p2b_trivial_feature_control" / model
    adv_path = base / "adversarial_sequences.parquet"
    post_path = base / "post_ablation_t5b_a.json"
    synth_path = base / "synthetic_boundary_results.json"
    if not adv_path.exists() or not post_path.exists() or not synth_path.exists():
        return {"status": "missing_inputs", "paths": [str(adv_path), str(post_path), str(synth_path)]}

    adv = pd.read_parquet(adv_path)
    seq_cell = (
        adv.groupby(["sequence_idx", "cell"], as_index=False)["high_prev_attn"]
        .mean()
        .pivot(index="sequence_idx", columns="cell", values="high_prev_attn")
        .reset_index()
    )
    need_cols = ["fake_boundary_with_prefix", "real_boundary_with_prefix"]
    for c in need_cols:
        if c not in seq_cell.columns:
            seq_cell[c] = np.nan
    seq_cell = seq_cell.dropna(subset=need_cols)

    if len(seq_cell) < 3:
        return {"status": "insufficient_sequences", "n_sequences": int(len(seq_cell))}

    diff = seq_cell["fake_boundary_with_prefix"].astype(float).values - seq_cell["real_boundary_with_prefix"].astype(float).values
    t_stat, p_two = scipy_stats.ttest_1samp(diff, popmean=0.0, nan_policy="omit")
    p_one = _one_sided_from_t_p(float(t_stat), float(p_two), positive=True)

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(max(1000, n_boot), len(diff)))
    perm_means = (signs * diff.reshape(1, -1)).mean(axis=1)
    obs = float(np.mean(diff))
    p_perm_one = float((np.sum(perm_means >= obs) + 1) / (len(perm_means) + 1))

    post = read_json(post_path)
    synth = read_json(synth_path)

    d_post = safe_float(post.get("high_vs_low_attn_to_prev_last", {}).get("cohens_d"))
    d_synth = safe_float(synth.get("prefix_following_assessment", {}).get("cohens_d_fake_minus_real"))

    mean_boot, ci_lo, ci_hi = _cluster_bootstrap_mean(
        df=pd.DataFrame({"sequence_idx": seq_cell["sequence_idx"], "delta": diff}),
        cluster_col="sequence_idx",
        value_col="delta",
        n_boot=n_boot,
        seed=seed + 111,
    )

    return {
        "status": "ok",
        "n_sequences": int(len(diff)),
        "mean_fake_minus_real_with_prefix": obs,
        "bootstrap_cluster_mean": mean_boot,
        "bootstrap_cluster_ci95": [ci_lo, ci_hi],
        "t_stat": safe_float(t_stat),
        "p_one_ttest": safe_float(p_one),
        "p_one_permutation": safe_float(p_perm_one),
        "post_ablation_d": d_post,
        "synthetic_d": d_synth,
        "prefix_flag": bool(synth.get("prefix_following_artifact_flag", False)),
        "claim_direction_supported": bool(np.isfinite(obs) and obs > 0),
    }


def _fit_linear_piecewise(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    # Returns (sse_linear, sse_piecewise, best_threshold)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a, b = np.polyfit(x, y, 1)
    pred_lin = a * x + b
    sse_lin = float(np.sum((y - pred_lin) ** 2))

    best_sse = float("inf")
    best_thr = float("nan")
    uniq = sorted(set(float(v) for v in x.tolist()))
    if len(uniq) < 3:
        return sse_lin, sse_lin, float("nan")
    for thr in uniq[1:-1]:
        z = np.maximum(0.0, x - thr)
        X = np.column_stack([np.ones_like(x), x, z])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            sse = float(np.sum((y - pred) ** 2))
        except Exception:
            continue
        if sse < best_sse:
            best_sse = sse
            best_thr = float(thr)
    if not np.isfinite(best_sse):
        best_sse = sse_lin
    return sse_lin, best_sse, best_thr


def _seed_boot_piecewise(model: str, n_boot: int, seed: int) -> dict[str, Any]:
    curve_path = ROOT / "results" / "experiment3_phase2" / "exp3p2c_redundancy_quantification" / model / "cumulative_ablation_curve.parquet"
    fit_path = ROOT / "results" / "experiment3_phase2" / "exp3p2c_redundancy_quantification" / model / "curve_fit_comparison.json"
    if not curve_path.exists() or not fit_path.exists():
        return {"status": "missing_inputs", "paths": [str(curve_path), str(fit_path)]}

    df = pd.read_parquet(curve_path)
    df = df[df["metric_name"].astype(str) == "accuracy"].copy()
    if df.empty:
        return {"status": "empty_curve"}

    seed_ids = sorted(df["seed"].dropna().astype(int).unique().tolist())
    if not seed_ids:
        return {"status": "no_seed_ids"}

    def _compute_delta(work: pd.DataFrame) -> float:
        g = work.groupby("ablation_fraction", as_index=False)["degradation"].mean().sort_values("ablation_fraction")
        x = g["ablation_fraction"].astype(float).values
        y = g["degradation"].astype(float).values
        sse_lin, sse_pw, _ = _fit_linear_piecewise(x, y)
        return float(sse_lin - sse_pw)

    obs = _compute_delta(df)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(max(1000, n_boot)):
        sampled = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        bdf = pd.concat([df[df["seed"] == s] for s in sampled], axis=0, ignore_index=True)
        boot.append(_compute_delta(bdf))
    barr = np.asarray(boot, dtype=float)
    p_one = float((np.sum(barr <= 0.0) + 1) / (len(barr) + 1))

    fit = read_json(fit_path)
    summary = fit.get("summary", {})
    return {
        "status": "ok",
        "n_rows": int(len(df)),
        "n_seed_clusters": int(len(seed_ids)),
        "obs_delta_sse_linear_minus_piecewise": obs,
        "bootstrap_ci95": [float(np.quantile(barr, 0.025)), float(np.quantile(barr, 0.975))],
        "p_one_piecewise_better": p_one,
        "claim_direction_supported": bool(np.isfinite(obs) and obs > 0),
        "reported_vote_support": summary.get("supports_threshold_model"),
        "reported_votes": summary.get("criterion_votes"),
    }


def _j_permutation_interaction(model: str, n_perm: int, seed: int) -> dict[str, Any]:
    path = ROOT / "results" / "experiment3_phase2" / "exp3p2j_conditional_regimes_longspan_repair" / model / "conditional_effects.parquet"
    summary_path = ROOT / "results" / "experiment3_phase2" / "exp3p2j_conditional_regimes_longspan_repair" / model / "regime_summary.json"
    if not path.exists() or not summary_path.exists():
        return {"status": "missing_inputs", "paths": [str(path), str(summary_path)]}

    df = pd.read_parquet(path)
    df = df[df["effect_metric"].astype(str) == "mean_surprisal_increase"].copy()
    if df.empty:
        return {"status": "empty_effects"}

    piv = (
        df.pivot_table(index=["seed", "regime", "task"], columns="group", values="effect_value", aggfunc="mean")
        .reset_index()
        .dropna(subset=["high_si", "low_si"])
    )
    if piv.empty:
        return {"status": "no_paired_rows"}
    piv["delta_high_minus_low"] = piv["high_si"].astype(float) - piv["low_si"].astype(float)

    reg_means = piv.groupby("regime", as_index=False)["delta_high_minus_low"].mean()
    obs_var = float(np.var(reg_means["delta_high_minus_low"].astype(float).values, ddof=1)) if len(reg_means) > 1 else 0.0

    # Permutation: shuffle regime labels within each seed-task block.
    rng = np.random.default_rng(seed)
    perm_stats: list[float] = []
    grouped = list(piv.groupby(["seed", "task"], sort=False))
    regimes = sorted(piv["regime"].astype(str).unique().tolist())
    for _ in range(max(1000, n_perm)):
        parts = []
        for (_, _), block in grouped:
            b = block.copy()
            if len(b) > 1:
                b["regime"] = rng.permutation(b["regime"].astype(str).values)
            parts.append(b)
        perm = pd.concat(parts, axis=0, ignore_index=True)
        g = perm.groupby("regime", as_index=False)["delta_high_minus_low"].mean()
        if len(g) > 1:
            perm_stats.append(float(np.var(g["delta_high_minus_low"].astype(float).values, ddof=1)))
    parr = np.asarray(perm_stats, dtype=float)
    p_perm = float((np.sum(parr >= obs_var) + 1) / (len(parr) + 1)) if len(parr) else float("nan")

    # Cluster bootstrap by seed for mean overall delta.
    seed_ids = sorted(piv["seed"].astype(int).unique().tolist())
    boot = []
    for _ in range(max(1000, n_perm)):
        sampled = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        bdf = pd.concat([piv[piv["seed"] == s] for s in sampled], axis=0, ignore_index=True)
        boot.append(float(np.mean(bdf["delta_high_minus_low"].astype(float).values)))
    barr = np.asarray(boot, dtype=float)
    overall_mean = float(np.mean(piv["delta_high_minus_low"].astype(float).values))

    summary = read_json(summary_path)
    return {
        "status": "ok",
        "n_rows": int(len(piv)),
        "n_seed_clusters": int(len(seed_ids)),
        "n_regimes": int(len(regimes)),
        "overall_delta_mean": overall_mean,
        "overall_delta_cluster_ci95": [float(np.quantile(barr, 0.025)), float(np.quantile(barr, 0.975))],
        "regime_delta_variance": obs_var,
        "p_permutation_regime_variance": p_perm,
        "claim_direction_supported": bool(np.isfinite(obs_var) and obs_var > 0),
        "regime_means": {
            str(r.regime): safe_float(r.delta_high_minus_low)
            for r in reg_means.itertuples()
        },
        "reported_interaction": summary.get("interaction_model", {}),
    }


def _t8_summary(model: str) -> dict[str, Any]:
    path = ROOT / "results" / "experiment3" / "theory8_position_ablation" / model / "report.json"
    if not path.exists():
        return {"status": "missing_inputs", "path": str(path)}
    rep = read_json(path)
    comp = rep.get("analysis", {}).get("comparisons", {}).get("subtract_kernel_high_si", {})
    mean_inc = safe_float(comp.get("mean_loss_increase"))
    ci = comp.get("bootstrap_ci_95", [float("nan"), float("nan")])
    return {
        "status": "ok",
        "mean_loss_increase": mean_inc,
        "bootstrap_ci_95": [safe_float(ci[0]), safe_float(ci[1])],
        "cohens_d": safe_float(comp.get("cohens_d")),
        "p_two_sided": safe_float(comp.get("paired_t_pval")),
        "claim_direction_supported": bool(np.isfinite(mean_inc) and mean_inc > 0),
    }


def run_all(*, out_root: Path, n_boot: int, n_perm: int, seed: int) -> dict[str, Any]:
    ensure_dir(out_root)
    model_tables: dict[str, Any] = {}
    claim_rows: list[dict[str, Any]] = []

    for i, model in enumerate(PRIMARY_MODELS):
        offset = seed + i * 1000
        t8 = _t8_summary(model)
        b = _sequence_level_boundary_test(model, n_boot=n_boot, seed=offset + 1)
        c1 = _seed_boot_piecewise(model, n_boot=n_boot, seed=offset + 2)
        j = _j_permutation_interaction(model, n_perm=n_perm, seed=offset + 3)

        model_tables[model] = {
            "T8_kernel_relevance": t8,
            "B_boundary_nontriviality": b,
            "C1_threshold_preference": c1,
            "J_regime_interaction": j,
        }

        for claim_id, payload in model_tables[model].items():
            claim_rows.append(
                {
                    "model": model,
                    "claim_id": claim_id,
                    "status": payload.get("status", "unknown"),
                    "claim_direction_supported": bool(payload.get("claim_direction_supported", False)),
                    "notes": payload,
                }
            )

        write_json(out_root / "model_summaries" / f"{model}.json", model_tables[model])

    stable_count = sum(1 for r in claim_rows if r["status"] == "ok" and r["claim_direction_supported"])
    total_ok = sum(1 for r in claim_rows if r["status"] == "ok")

    summary = {
        "timestamp": timestamp_now(),
        "experiment": "EXP-R2",
        "objective": "dependence_aware_reanalysis",
        "settings": {
            "n_boot": int(n_boot),
            "n_perm": int(n_perm),
            "seed": int(seed),
            "models": list(PRIMARY_MODELS),
        },
        "claim_rows": claim_rows,
        "stability": {
            "n_ok_claims": int(total_ok),
            "n_direction_supported": int(stable_count),
            "fraction_supported": float(stable_count / max(total_ok, 1)),
        },
        "claim_impact": "strengthens_main_claim" if stable_count >= max(1, total_ok - 1) else "no_change",
    }

    write_json(out_root / "claim_stability_table.json", summary)
    write_json(
        out_root / "manifest.json",
        command_manifest(
            experiment_id="EXP-R2",
            command="dependence_reanalysis",
            model="llama-3.1-8b+olmo-2-7b",
            seed_set=[int(seed)],
            extras={
                "n_boot": int(n_boot),
                "n_perm": int(n_perm),
                "output_root": str(out_root),
            },
        ),
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-R2: dependence-aware statistical reanalysis")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--n-perm", type=int, default=4000)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_all(
        out_root=Path(args.output_root),
        n_boot=max(1000, int(args.n_boot)),
        n_perm=max(1000, int(args.n_perm)),
        seed=int(args.seed),
    )
    print(f"[EXP-R2] wrote {Path(args.output_root) / 'claim_stability_table.json'}")
    print(f"[EXP-R2] stability={summary['stability']}")


if __name__ == "__main__":
    main()
