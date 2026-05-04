#!/usr/bin/env python3
"""E24b — SI-vs-non-SI cumulative-ablation contrast quantification.

Purpose
-------
E24 established that both SI-ranked and matched non-SI cumulative ablations are
mostly non-linear. E24b quantifies *how much* stronger non-linearity is in
SI-ranked depletion, rather than only counting linear/non-linear votes.

Design
------
- Reuses completed 3P2-C.1 curve artifacts (same models/tasks/fractions).
- Compares `high_to_low` (SI-ranked) vs `low_to_high` (matched non-SI) arms.
- Per model x task x metric cell:
  1) Delta-BIC contrast:
       delta_bic = BIC(linear) - min(BIC(threshold), BIC(logistic))
     then compare high vs low.
  2) Seed-paired AUC contrast across ablation fractions.
- Bootstrap CIs are reported for both contrasts.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    PRIMARY_MODELS,
    RESULTS_ROOT,
    command_manifest,
    ensure_dir,
    read_json,
    safe_float,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    bootstrap_mean_ci,
    emit_core_artifacts,
    enforce_coverage_contract,
    fit_three_models,
    parse_models_arg,
)

EXPERIMENT_ID = "E24B"
DEFAULT_OUT = RESULTS_ROOT / "E24b_si_vs_non_si_contrast"
DEFAULT_SOURCE_ROOTS = [
    ROOT
    / "results"
    / "reinforce_exp"
    / "runs"
    / "neurips_ext_opt_20260427_134753"
    / "exp3p2c_redundancy_quantification",
    ROOT / "results" / "experiment3_phase2" / "exp3p2c_redundancy_quantification",
]


def _resolve_model_curve_path(model_name: str, source_roots: list[Path]) -> Path:
    for root in source_roots:
        p = root / model_name / "cumulative_ablation_curve.parquet"
        if p.exists():
            return p
    searched = "\n".join(str(root / model_name / "cumulative_ablation_curve.parquet") for root in source_roots)
    raise FileNotFoundError(
        f"[E24b] hard_fail_reason: missing cumulative curve for model={model_name}. "
        f"Searched:\n{searched}"
    )


def _parse_source_roots(raw: str) -> list[Path]:
    if not str(raw).strip():
        return list(DEFAULT_SOURCE_ROOTS)
    roots: list[Path] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if t:
            roots.append(Path(t))
    return roots or list(DEFAULT_SOURCE_ROOTS)


def _delta_bic_from_rows(rows: pd.DataFrame) -> float:
    agg = (
        rows.groupby("ablation_fraction", as_index=False)["degradation"]
        .mean()
        .sort_values("ablation_fraction")
        .dropna(subset=["degradation"])
    )
    if len(agg) < 4:
        return float("nan")
    x = agg["ablation_fraction"].to_numpy(dtype=float) / 100.0
    y = agg["degradation"].to_numpy(dtype=float)
    fit = fit_three_models(x, y)
    b_lin = safe_float(fit.get("linear", {}).get("bic", float("nan")))
    b_thr = safe_float(fit.get("threshold_piecewise", {}).get("bic", float("nan")))
    b_log = safe_float(fit.get("logistic_sigmoid", {}).get("bic", float("nan")))
    b_nl = np.nanmin(np.asarray([b_thr, b_log], dtype=float))
    if not (np.isfinite(b_lin) and np.isfinite(b_nl)):
        return float("nan")
    return float(b_lin - b_nl)


def _seed_auc_deltas(cell_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (sort_order, seed), gdf in cell_df.groupby(["sort_order", "seed"], sort=True):
        agg = (
            gdf.groupby("ablation_fraction", as_index=False)["degradation"]
            .mean()
            .sort_values("ablation_fraction")
            .dropna(subset=["degradation"])
        )
        if len(agg) < 4:
            continue
        x = agg["ablation_fraction"].to_numpy(dtype=float) / 100.0
        y = agg["degradation"].to_numpy(dtype=float)
        auc = float(np.trapz(y, x))
        rows.append({"sort_order": str(sort_order), "seed": int(seed), "auc": auc})
    if not rows:
        return pd.DataFrame(columns=["seed", "auc_high", "auc_low", "auc_delta_high_minus_low"])
    auc_df = pd.DataFrame(rows)
    hi = auc_df[auc_df["sort_order"] == "high_to_low"][["seed", "auc"]].rename(columns={"auc": "auc_high"})
    lo = auc_df[auc_df["sort_order"] == "low_to_high"][["seed", "auc"]].rename(columns={"auc": "auc_low"})
    merged = hi.merge(lo, on="seed", how="inner")
    merged["auc_delta_high_minus_low"] = merged["auc_high"] - merged["auc_low"]
    return merged.sort_values("seed").reset_index(drop=True)


def _jackknife_bic_diff(
    cell_df: pd.DataFrame,
) -> tuple[float, float, float]:
    seeds = sorted(int(s) for s in cell_df["seed"].dropna().unique().tolist())
    if len(seeds) == 0:
        return float("nan"), float("nan"), float("nan")
    hi_all = cell_df[cell_df["sort_order"] == "high_to_low"].copy()
    lo_all = cell_df[cell_df["sort_order"] == "low_to_high"].copy()
    if hi_all.empty or lo_all.empty:
        return float("nan"), float("nan"), float("nan")
    d_hi_full = _delta_bic_from_rows(hi_all)
    d_lo_full = _delta_bic_from_rows(lo_all)
    if not (np.isfinite(d_hi_full) and np.isfinite(d_lo_full)):
        return float("nan"), float("nan"), float("nan")
    full = float(d_hi_full - d_lo_full)

    # Seed-level jackknife pseudo-CI (fast + deterministic for small-n seed sets).
    jk: list[float] = []
    if len(seeds) >= 2:
        for s_drop in seeds:
            hi = hi_all[hi_all["seed"] != int(s_drop)]
            lo = lo_all[lo_all["seed"] != int(s_drop)]
            if hi.empty or lo.empty:
                continue
            d_hi = _delta_bic_from_rows(hi)
            d_lo = _delta_bic_from_rows(lo)
            if np.isfinite(d_hi) and np.isfinite(d_lo):
                jk.append(float(d_hi - d_lo))
    if len(jk) == 0:
        return full, float("nan"), float("nan")
    arr = np.asarray(jk, dtype=float)
    return full, float(np.nanmin(arr)), float(np.nanmax(arr))


def _run_model(
    *,
    model_name: str,
    out_dir: Path,
    source_roots: list[Path],
    n_boot: int,
    smoke: bool,
) -> dict[str, Any]:
    model_dir = ensure_dir(out_dir / model_name)
    curve_path = _resolve_model_curve_path(model_name, source_roots)
    df = pd.read_parquet(curve_path).copy()
    req = {"sort_order", "task", "metric_name", "ablation_fraction", "degradation", "seed"}
    missing = sorted(req - set(df.columns))
    if missing:
        raise RuntimeError(f"[E24b] hard_fail_reason: missing required columns for {model_name}: {missing}")

    df = df[df["sort_order"].isin(["high_to_low", "low_to_high"])].copy()
    if df.empty:
        raise RuntimeError(f"[E24b] hard_fail_reason: empty high/low rows for {model_name}")

    if smoke:
        keep_seeds = sorted(int(s) for s in df["seed"].dropna().unique().tolist())[:2]
        keep_tasks = sorted(str(t) for t in df["task"].dropna().unique().tolist())[:2]
        df = df[df["seed"].isin(keep_seeds) & df["task"].isin(keep_tasks)].copy()

    cell_rows: list[dict[str, Any]] = []
    seed_auc_rows: list[pd.DataFrame] = []

    for (task, metric), cdf in df.groupby(["task", "metric_name"], sort=True):
        hi = cdf[cdf["sort_order"] == "high_to_low"].copy()
        lo = cdf[cdf["sort_order"] == "low_to_high"].copy()
        if hi.empty or lo.empty:
            continue

        d_hi = _delta_bic_from_rows(hi)
        d_lo = _delta_bic_from_rows(lo)
        d_diff = float(d_hi - d_lo) if (np.isfinite(d_hi) and np.isfinite(d_lo)) else float("nan")
        d_boot_mean, d_boot_lo, d_boot_hi = _jackknife_bic_diff(cdf)

        auc_df = _seed_auc_deltas(cdf)
        if not auc_df.empty:
            auc_vals = auc_df["auc_delta_high_minus_low"].to_numpy(dtype=float)
            auc_mean, auc_lo, auc_hi = bootstrap_mean_ci(
                auc_vals,
                n_boot=min(n_boot, 2000) if smoke else n_boot,
                seed=20260503 + abs(hash((task, metric))) % 10000,
            )
            adf = auc_df.copy()
            adf["model"] = model_name
            adf["task"] = str(task)
            adf["metric_name"] = str(metric)
            seed_auc_rows.append(adf)
        else:
            auc_mean = auc_lo = auc_hi = float("nan")

        cell_rows.append(
            {
                "model": model_name,
                "task": str(task),
                "metric_name": str(metric),
                "delta_bic_high": safe_float(d_hi),
                "delta_bic_low": safe_float(d_lo),
                "delta_bic_diff_high_minus_low": safe_float(d_diff),
                "delta_bic_diff_boot_mean": safe_float(d_boot_mean),
                "delta_bic_diff_ci95_lo": safe_float(d_boot_lo),
                "delta_bic_diff_ci95_hi": safe_float(d_boot_hi),
                "auc_diff_boot_mean": safe_float(auc_mean),
                "auc_diff_ci95_lo": safe_float(auc_lo),
                "auc_diff_ci95_hi": safe_float(auc_hi),
                "n_unique_seeds": int(cdf["seed"].nunique()),
            }
        )

    if not cell_rows:
        raise RuntimeError(f"[E24b] hard_fail_reason: no fit-able cells for {model_name}")

    cells_df = pd.DataFrame(cell_rows).sort_values(["task", "metric_name"]).reset_index(drop=True)
    auc_seed_df = (
        pd.concat(seed_auc_rows, ignore_index=True).sort_values(["task", "metric_name", "seed"]).reset_index(drop=True)
        if seed_auc_rows
        else pd.DataFrame(columns=["model", "task", "metric_name", "seed", "auc_high", "auc_low", "auc_delta_high_minus_low"])
    )

    cells_df.to_parquet(model_dir / "cell_contrasts.parquet", index=False)
    auc_seed_df.to_parquet(model_dir / "seed_auc_deltas.parquet", index=False)

    valid_bic = cells_df["delta_bic_diff_high_minus_low"].to_numpy(dtype=float)
    valid_bic = valid_bic[np.isfinite(valid_bic)]
    valid_auc = cells_df["auc_diff_boot_mean"].to_numpy(dtype=float)
    valid_auc = valid_auc[np.isfinite(valid_auc)]
    n_pos = int(np.sum(valid_bic > 0))
    n_cells = int(valid_bic.size)
    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "model": model_name,
        "source_curve": str(curve_path),
        "n_rows_source": int(len(df)),
        "n_cells": n_cells,
        "n_cells_positive_delta_bic_diff": n_pos,
        "mean_delta_bic_diff_high_minus_low": safe_float(np.mean(valid_bic) if valid_bic.size else float("nan")),
        "mean_auc_diff_high_minus_low": safe_float(np.mean(valid_auc) if valid_auc.size else float("nan")),
        "smoke": bool(smoke),
    }
    write_json(model_dir / "summary.json", summary)
    return summary


def _finalize(
    *,
    models: list[str],
    out_dir: Path,
    source_roots: list[Path],
    n_boot: int,
    smoke: bool,
) -> dict[str, Any]:
    model_summaries: list[dict[str, Any]] = []
    cell_tables: list[pd.DataFrame] = []
    for m in models:
        sp = out_dir / m / "summary.json"
        cp = out_dir / m / "cell_contrasts.parquet"
        if not sp.exists() or not cp.exists():
            raise RuntimeError(f"[E24b] hard_fail_reason: missing shard artifact for {m}: {sp} / {cp}")
        model_summaries.append(read_json(sp))
        cdf = pd.read_parquet(cp).copy()
        cdf["model"] = m
        cell_tables.append(cdf)

    enforce_coverage_contract(
        experiment_id=EXPERIMENT_ID,
        observed_models=[m.get("model", "") for m in model_summaries],
        required_models=models,
        observed_counts={m: 1 for m in models},
        min_counts={m: 1 for m in models},
    )

    all_cells = pd.concat(cell_tables, ignore_index=True, sort=False)
    all_cells.to_parquet(out_dir / "all_model_cell_contrasts.parquet", index=False)
    pd.DataFrame(model_summaries).to_parquet(out_dir / "per_model_summary.parquet", index=False)

    bic_vals = all_cells["delta_bic_diff_high_minus_low"].to_numpy(dtype=float)
    bic_vals = bic_vals[np.isfinite(bic_vals)]
    auc_vals = all_cells["auc_diff_boot_mean"].to_numpy(dtype=float)
    auc_vals = auc_vals[np.isfinite(auc_vals)]

    bic_mean, bic_lo, bic_hi = bootstrap_mean_ci(
        bic_vals,
        n_boot=min(n_boot, 3000) if smoke else n_boot,
        seed=20260510,
    )
    auc_mean, auc_lo, auc_hi = bootstrap_mean_ci(
        auc_vals,
        n_boot=min(n_boot, 3000) if smoke else n_boot,
        seed=20260511,
    )

    n_cells = int(bic_vals.size)
    n_pos = int(np.sum(bic_vals > 0))
    frac_pos = float(n_pos / n_cells) if n_cells > 0 else float("nan")

    if n_cells == 0 or not np.isfinite(bic_mean):
        interpretation = "insufficient_data"
        status = "not_supported"
    elif frac_pos >= 0.8 and bic_lo > 0 and auc_lo > 0:
        interpretation = "si_enrichment_supported"
        status = "supported"
    elif frac_pos >= 0.5 and bic_mean > 0:
        interpretation = "si_enrichment_supported_with_caveat"
        status = "supported_with_caveat"
    elif frac_pos >= 0.5:
        interpretation = "mixed"
        status = "mixed"
    else:
        interpretation = "no_si_enrichment"
        status = "not_supported"

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": timestamp_now(),
        "models": models,
        "n_cells": n_cells,
        "n_cells_positive_delta_bic_diff": n_pos,
        "fraction_positive_delta_bic_diff": safe_float(frac_pos),
        "delta_bic_diff_boot_mean": safe_float(bic_mean),
        "delta_bic_diff_ci95": [safe_float(bic_lo), safe_float(bic_hi)],
        "auc_diff_boot_mean": safe_float(auc_mean),
        "auc_diff_ci95": [safe_float(auc_lo), safe_float(auc_hi)],
        "interpretation": interpretation,
        "claim_status": status,
        "note": "Positive delta means SI-ranked arm is more non-linear than matched non-SI arm.",
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "cross_model_contrast_summary.json", summary)

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "How much stronger is SI-ranked cumulative-ablation non-linearity than matched non-SI non-linearity?",
        "primary_metrics": [
            "delta_bic_diff_high_minus_low",
            "auc_diff_high_minus_low",
        ],
        "decision_rule": "Support requires mostly positive cell-level SI-minus-nonSI contrasts with positive bootstrap lower bounds.",
    }
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "source_roots": [str(p) for p in source_roots],
        "artifacts": [
            "per_model_summary.parquet",
            "all_model_cell_contrasts.parquet",
            "cross_model_contrast_summary.json",
        ],
    }
    claim_impact = {
        "experiment_id": EXPERIMENT_ID,
        "claims_tested": [
            "Result II residual SI enrichment after E24 matched baseline",
        ],
        "status": status,
        "interpretation": interpretation,
        "impact_on_main_text": (
            "If not supported, keep Result II as generic ranked-ablation nonlinearity with SI-specificity unresolved."
            if status == "not_supported"
            else "Retain limited SI-enrichment wording with explicit caveat."
        ),
    }
    data_dictionary = {
        "tables": [
            {
                "path": "all_model_cell_contrasts.parquet",
                "description": "Per model-task-metric SI-vs-nonSI contrast statistics.",
                "columns": [
                    {"name": "model", "dtype": "str"},
                    {"name": "task", "dtype": "str"},
                    {"name": "metric_name", "dtype": "str"},
                    {"name": "delta_bic_diff_high_minus_low", "dtype": "float"},
                    {"name": "auc_diff_boot_mean", "dtype": "float"},
                ],
            }
        ]
    }
    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=out_dir,
        preregistration=prereg,
        manifest=manifest,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E24b SI-vs-nonSI contrast quantification")
    parser.add_argument("--models", default=",".join(PRIMARY_MODELS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--source-roots", default="", help="Comma-separated source roots for cumulative curves")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    if args.no_finalize and args.finalize_only:
        raise RuntimeError("[E24b] hard_fail_reason: --no-finalize and --finalize-only are mutually exclusive")

    out_dir = Path(args.output_root).resolve()
    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    source_roots = _parse_source_roots(args.source_roots)
    ensure_dir(out_dir)
    write_json(
        out_dir / "command_manifest.json",
        command_manifest(
            experiment_id=EXPERIMENT_ID,
            command="run_e24b_si_vs_non_si_contrast.py",
            model="multi-model",
        ),
    )

    if args.finalize_only:
        summary = _finalize(
            models=models,
            out_dir=out_dir,
            source_roots=source_roots,
            n_boot=int(args.bootstrap_reps),
            smoke=bool(args.smoke),
        )
        print(f"[E24b] finalize-only: interpretation={summary['interpretation']} status={summary['claim_status']}")
        return

    for m in models:
        rec = _run_model(
            model_name=m,
            out_dir=out_dir,
            source_roots=source_roots,
            n_boot=int(args.bootstrap_reps),
            smoke=bool(args.smoke),
        )
        print(
            f"[E24b] model={m} n_cells={rec['n_cells']} "
            f"mean_delta_bic_diff={rec['mean_delta_bic_diff_high_minus_low']:.4f}"
        )

    if not args.no_finalize:
        summary = _finalize(
            models=models,
            out_dir=out_dir,
            source_roots=source_roots,
            n_boot=int(args.bootstrap_reps),
            smoke=bool(args.smoke),
        )
        print(f"[E24b] interpretation={summary['interpretation']} status={summary['claim_status']}")


if __name__ == "__main__":
    main()
