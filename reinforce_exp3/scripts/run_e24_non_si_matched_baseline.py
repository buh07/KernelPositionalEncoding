#!/usr/bin/env python3
"""E24 — Matched non-SI cumulative-ablation baseline adjudication.

Purpose
-------
Resolve the key Result-II specificity objection by testing whether non-linear
depletion under cumulative ablation is unique to SI-ranked heads.

Method
------
Uses completed 3P2-C.1 artifacts (same fraction grid, same tasks, same models)
that contain both:
  - high_to_low: SI-ranked cumulative ablation
  - low_to_high: matched-cardinality non-SI cumulative ablation

For each model x task x metric x sort_order cell:
  1) Aggregate degradation by ablation fraction.
  2) Fit linear / threshold-piecewise / logistic (BIC via shared fitter).
  3) Count linear votes.

Primary interpretation:
  - supports_si_specific_nonlinearity:
      linear rejected for SI-ranked cells and at least one non-SI cell prefers linear.
  - not_supported:
      linear rejected in all SI and all non-SI cells (generic architecture-level
      nonlinearity remains plausible).
"""
from __future__ import annotations

import argparse
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
    safe_float,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    emit_core_artifacts,
    fit_three_models,
    parse_models_arg,
)

EXPERIMENT_ID = "E24"
DEFAULT_OUT = RESULTS_ROOT / "E24_non_si_matched_baseline"

# Prefer the latest completed run artifacts used for the paper.
DEFAULT_SOURCE_ROOTS = [
    ROOT / "results" / "reinforce_exp" / "runs" / "neurips_ext_opt_20260427_134753" / "exp3p2c_redundancy_quantification",
    ROOT / "results" / "experiment3_phase2" / "exp3p2c_redundancy_quantification",
]


def _resolve_model_curve_path(model_name: str, source_roots: list[Path]) -> Path:
    for root in source_roots:
        p = root / model_name / "cumulative_ablation_curve.parquet"
        if p.exists():
            return p
    searched = "\n".join(str(root / model_name / "cumulative_ablation_curve.parquet") for root in source_roots)
    raise FileNotFoundError(
        f"[E24] hard_fail_reason: missing cumulative curve for model={model_name}. "
        f"Searched:\n{searched}"
    )


def _fit_cells(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    vote_counts = {
        "high_linear": 0,
        "high_threshold": 0,
        "high_logistic": 0,
        "low_linear": 0,
        "low_threshold": 0,
        "low_logistic": 0,
    }
    for (model, sort_order, task, metric), gdf in df.groupby(
        ["model", "sort_order", "task", "metric_name"], sort=True
    ):
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
        fit = fit_three_models(x, y)
        pref = str(fit.get("preferred_model", "insufficient_points"))
        arm = "high" if str(sort_order) == "high_to_low" else "low"
        key = f"{arm}_{'linear' if pref == 'linear' else 'threshold' if pref == 'threshold_piecewise' else 'logistic'}"
        if key in vote_counts:
            vote_counts[key] += 1

        rows.append(
            {
                "model": str(model),
                "sort_order": str(sort_order),
                "task": str(task),
                "metric_name": str(metric),
                "n_points": int(len(agg)),
                "preferred_model": pref,
                "linear_bic": safe_float(fit.get("linear", {}).get("bic", float("nan"))),
                "threshold_bic": safe_float(fit.get("threshold_piecewise", {}).get("bic", float("nan"))),
                "logistic_bic": safe_float(fit.get("logistic_sigmoid", {}).get("bic", float("nan"))),
                "mean_deg_25pct": safe_float(agg.loc[agg["ablation_fraction"] == 25, "degradation"].mean()),
                "mean_deg_50pct": safe_float(agg.loc[agg["ablation_fraction"] == 50, "degradation"].mean()),
            }
        )
    return pd.DataFrame(rows), vote_counts


def _arm_auc_by_seed(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, sort_order, task, metric, seed), gdf in df.groupby(
        ["model", "sort_order", "task", "metric_name", "seed"], sort=True
    ):
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
        rows.append(
            {
                "model": str(model),
                "sort_order": str(sort_order),
                "task": str(task),
                "metric_name": str(metric),
                "seed": int(seed),
                "auc": auc,
            }
        )
    return pd.DataFrame(rows)


def _paired_auc_deltas(auc_df: pd.DataFrame) -> pd.DataFrame:
    high = auc_df[auc_df["sort_order"] == "high_to_low"].rename(columns={"auc": "auc_high"})
    low = auc_df[auc_df["sort_order"] == "low_to_high"].rename(columns={"auc": "auc_low"})
    merged = high.merge(
        low[["model", "task", "metric_name", "seed", "auc_low"]],
        on=["model", "task", "metric_name", "seed"],
        how="inner",
    )
    merged["auc_delta_high_minus_low"] = merged["auc_high"] - merged["auc_low"]
    return merged


def run_e24(*, models: list[str], out_dir: Path, source_roots: list[Path]) -> dict[str, Any]:
    ensure_dir(out_dir)

    all_rows: list[pd.DataFrame] = []
    source_map: dict[str, str] = {}
    for model_name in models:
        curve_path = _resolve_model_curve_path(model_name, source_roots)
        source_map[model_name] = str(curve_path)
        df = pd.read_parquet(curve_path).copy()
        df["model"] = str(model_name)
        all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True, sort=False)
    req_cols = {"sort_order", "task", "metric_name", "ablation_fraction", "degradation", "seed"}
    missing = sorted(req_cols - set(full_df.columns))
    if missing:
        raise RuntimeError(f"[E24] hard_fail_reason: missing required columns in source data: {missing}")

    # Keep only the two matched-cardinality arms.
    full_df = full_df[full_df["sort_order"].isin(["high_to_low", "low_to_high"])].copy()
    if full_df.empty:
        raise RuntimeError("[E24] hard_fail_reason: no high_to_low/low_to_high rows found")

    cell_fits_df, vote_counts = _fit_cells(full_df)
    if cell_fits_df.empty:
        raise RuntimeError("[E24] hard_fail_reason: no fit-able cells")

    auc_df = _arm_auc_by_seed(full_df)
    delta_df = _paired_auc_deltas(auc_df)

    high_linear = int(vote_counts["high_linear"])
    low_linear = int(vote_counts["low_linear"])
    n_high_cells = int(vote_counts["high_linear"] + vote_counts["high_threshold"] + vote_counts["high_logistic"])
    n_low_cells = int(vote_counts["low_linear"] + vote_counts["low_threshold"] + vote_counts["low_logistic"])

    # Main adjudication rule.
    if (high_linear == 0) and (low_linear > 0):
        interpretation = "supports_si_specific_nonlinearity"
        status = "supported_with_caveat"
        note = (
            "SI-ranked cells reject linear while at least one matched non-SI cell prefers linear; "
            "supports SI-specific nonlinearity with caveats."
        )
    elif (high_linear == 0) and (low_linear == 0):
        interpretation = "generic_nonlinearity_plausible"
        status = "not_supported"
        note = (
            "Both SI-ranked and matched non-SI cells reject linear; "
            "nonlinearity is not specific to SI-ranked ablation in this control."
        )
    else:
        interpretation = "mixed"
        status = "mixed"
        note = "Control evidence is mixed; SI-specific nonlinearity not cleanly established."

    # Save tabular outputs.
    full_df.to_parquet(out_dir / "source_curves_all_rows.parquet", index=False)
    cell_fits_df.to_parquet(out_dir / "cell_fits_bic_votes.parquet", index=False)
    auc_df.to_parquet(out_dir / "auc_by_seed.parquet", index=False)
    delta_df.to_parquet(out_dir / "auc_delta_high_minus_low.parquet", index=False)

    per_model_vote_rows: list[dict[str, Any]] = []
    for model_name in models:
        m = cell_fits_df[cell_fits_df["model"] == model_name]
        high = m[m["sort_order"] == "high_to_low"]["preferred_model"].value_counts().to_dict()
        low = m[m["sort_order"] == "low_to_high"]["preferred_model"].value_counts().to_dict()
        per_model_vote_rows.append(
            {
                "model": model_name,
                "high_to_low_votes": {k: int(v) for k, v in high.items()},
                "low_to_high_votes": {k: int(v) for k, v in low.items()},
                "high_linear_votes": int(high.get("linear", 0)),
                "low_linear_votes": int(low.get("linear", 0)),
            }
        )

    paired_auc_means = (
        delta_df.groupby(["model", "task", "metric_name"], as_index=False)["auc_delta_high_minus_low"]
        .mean()
        .rename(columns={"auc_delta_high_minus_low": "mean_auc_delta_high_minus_low"})
    )
    paired_auc_means.to_parquet(out_dir / "auc_delta_means_by_model_task.parquet", index=False)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": timestamp_now(),
        "models": models,
        "source_map": source_map,
        "n_rows_source": int(len(full_df)),
        "n_fit_cells": int(len(cell_fits_df)),
        "n_high_cells": n_high_cells,
        "n_low_cells": n_low_cells,
        "vote_counts": {k: int(v) for k, v in vote_counts.items()},
        "interpretation": interpretation,
        "claim_status": status,
        "note": note,
        "per_model_votes": per_model_vote_rows,
    }
    write_json(out_dir / "summary.json", summary)

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Is nonlinear cumulative-depletion behavior specific to SI-ranked ablation versus matched non-SI ablation?",
        "primary_criterion": "At least one matched non-SI cell must prefer linear while SI-ranked cells reject linear for SI-specific nonlinearity support.",
        "decision_rules": [
            "supports_si_specific_nonlinearity: SI linear votes=0 and non-SI linear votes>0",
            "not_supported: SI linear votes=0 and non-SI linear votes=0",
            "mixed otherwise",
        ],
    }
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "artifacts": [
            "source_curves_all_rows.parquet",
            "cell_fits_bic_votes.parquet",
            "auc_by_seed.parquet",
            "auc_delta_high_minus_low.parquet",
            "auc_delta_means_by_model_task.parquet",
            "summary.json",
        ],
        "source_roots": [str(p) for p in source_roots],
    }
    claim_impact = {
        "experiment_id": EXPERIMENT_ID,
        "claims_tested": [
            "Result II SI-specific collective organization under cumulative ablation",
        ],
        "status": status,
        "interpretation": interpretation,
        "impact_on_main_text": (
            "Downgrade SI-specific redundancy language if non-SI matched arm also rejects linear."
            if status == "not_supported"
            else "SI-specific wording may be retained with caveats."
        ),
    }
    data_dictionary = {
        "tables": [
            {
                "path": "cell_fits_bic_votes.parquet",
                "description": "Per model x arm x task x metric BIC fit votes.",
                "columns": [
                    {"name": "model", "dtype": "str"},
                    {"name": "sort_order", "dtype": "str"},
                    {"name": "task", "dtype": "str"},
                    {"name": "metric_name", "dtype": "str"},
                    {"name": "preferred_model", "dtype": "str"},
                    {"name": "linear_bic", "dtype": "float"},
                    {"name": "threshold_bic", "dtype": "float"},
                    {"name": "logistic_bic", "dtype": "float"},
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


def _parse_source_roots(raw: str) -> list[Path]:
    if not str(raw).strip():
        return list(DEFAULT_SOURCE_ROOTS)
    out: list[Path] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if t:
            out.append(Path(t))
    return out or list(DEFAULT_SOURCE_ROOTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="E24 matched non-SI baseline adjudication")
    parser.add_argument("--models", default=",".join(PRIMARY_MODELS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--source-roots",
        default="",
        help="Comma-separated roots containing <model>/cumulative_ablation_curve.parquet",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_root).resolve()
    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    source_roots = _parse_source_roots(args.source_roots)
    ensure_dir(out_dir)
    write_json(
        out_dir / "command_manifest.json",
        command_manifest(
            experiment_id=EXPERIMENT_ID,
            command="run_e24_non_si_matched_baseline.py",
            model="multi-model",
        ),
    )

    summary = run_e24(models=models, out_dir=out_dir, source_roots=source_roots)
    print(f"[E24] interpretation={summary['interpretation']} status={summary['claim_status']}")
    print(f"[E24] vote_counts={summary['vote_counts']}")


if __name__ == "__main__":
    main()
