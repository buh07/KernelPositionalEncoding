#!/usr/bin/env python3
"""E25 — Cross-model comparability sensitivity for Result I metrics.

Purpose
-------
Address the cross-model comparability critique by summarizing Result-I effects
across three axes:
  1) absolute disruption (nats),
  2) baseline-normalized disruption (% increase),
  3) within-model control-normalized contrasts (true-permuted / true-normmatched).

This does not resolve all comparability limits, but makes the remaining scope
explicit with a single artifacted sensitivity panel.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

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
    emit_core_artifacts,
    enforce_coverage_contract,
    parse_models_arg,
)

EXPERIMENT_ID = "E25"
DEFAULT_OUT = RESULTS_ROOT / "E25_cross_model_comparability"


def _resolve_t8_report(model_name: str) -> Path:
    candidates = [
        ROOT / "results" / "experiment3" / "theory8_position_ablation" / model_name / "report.json",
        ROOT / "results" / "reinforce_exp" / "exp_r3_core_replication" / model_name / "theory8_position_ablation" / model_name / "report.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"[E25] hard_fail_reason: missing T8 report for {model_name}. candidates={candidates}")


def _resolve_e17_summary(model_name: str) -> Path:
    p = ROOT / "results" / "reinforce_exp3" / "E17_normmatched_specificity" / model_name / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"[E25] hard_fail_reason: missing E17 summary for {model_name}: {p}")
    return p


def _dense_rank_desc(values: np.ndarray) -> np.ndarray:
    # Highest value -> rank 1
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    r = 1.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and np.isclose(values[order[j + 1]], values[order[i]], atol=1e-12, rtol=1e-12):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        r += 1.0
        i = j + 1
    return ranks


def _run_model(*, model_name: str, out_dir: Path) -> dict[str, Any]:
    model_dir = ensure_dir(out_dir / model_name)
    t8_path = _resolve_t8_report(model_name)
    e17_path = _resolve_e17_summary(model_name)

    t8 = read_json(t8_path)
    e17 = read_json(e17_path)

    ana = t8.get("analysis", {})
    base = float(ana.get("baseline", {}).get("mean_loss", float("nan")))
    hi = ana.get("subtract_kernel_high_si", {})
    cmp_hi = ana.get("comparisons", {}).get("subtract_kernel_high_si", {})
    e17_headline = e17.get("headline", {})

    rec = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "model": model_name,
        "t8_report": str(t8_path),
        "e17_summary": str(e17_path),
        "baseline_loss": safe_float(base),
        "true_nat_delta": safe_float(float(hi.get("mean_loss_increase", float("nan")))),
        "true_pct_delta": safe_float(float(hi.get("relative_increase_pct", float("nan")))),
        "true_cohens_d": safe_float(float(cmp_hi.get("cohens_d", float("nan")))),
        "true_minus_permuted": safe_float(
            float(e17_headline.get("true_mean_loss_delta", float("nan")))
            - float(e17_headline.get("permuted_mean_loss_delta", float("nan")))
        ),
        "true_minus_normmatched": safe_float(
            float(e17_headline.get("true_mean_loss_delta", float("nan")))
            - float(e17_headline.get("normmatched_mean_loss_delta", float("nan")))
        ),
        "ratio_true_over_permuted": safe_float(float(e17_headline.get("ratio_true_over_permuted", float("nan")))),
        "ratio_true_over_normmatched": safe_float(float(e17_headline.get("ratio_true_over_normmatched", float("nan")))),
    }
    rec["nat_over_baseline"] = safe_float(
        float(rec["true_nat_delta"]) / max(1e-12, float(rec["baseline_loss"]))
        if np.isfinite(rec["true_nat_delta"]) and np.isfinite(rec["baseline_loss"])
        else float("nan")
    )
    write_json(model_dir / "summary.json", rec)
    return rec


def _finalize(*, models: list[str], out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for m in models:
        p = out_dir / m / "summary.json"
        if not p.exists():
            raise RuntimeError(f"[E25] hard_fail_reason: missing shard summary for {m}: {p}")
        rows.append(read_json(p))

    enforce_coverage_contract(
        experiment_id=EXPERIMENT_ID,
        observed_models=[r.get("model", "") for r in rows],
        required_models=models,
        observed_counts={m: 1 for m in models},
        min_counts={m: 1 for m in models},
    )

    df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    for col in [
        "true_nat_delta",
        "true_pct_delta",
        "nat_over_baseline",
        "true_minus_permuted",
        "true_minus_normmatched",
        "ratio_true_over_permuted",
        "ratio_true_over_normmatched",
    ]:
        df[f"rank_{col}"] = _dense_rank_desc(df[col].to_numpy(dtype=float))

    rank_cols = [
        "rank_true_nat_delta",
        "rank_true_pct_delta",
        "rank_nat_over_baseline",
        "rank_true_minus_permuted",
        "rank_ratio_true_over_permuted",
    ]
    rank_consistency_rows: list[dict[str, Any]] = []
    for i in range(len(rank_cols)):
        for j in range(i + 1, len(rank_cols)):
            a = df[rank_cols[i]].to_numpy(dtype=float)
            b = df[rank_cols[j]].to_numpy(dtype=float)
            rho, p = scipy_stats.spearmanr(a, b)
            rank_consistency_rows.append(
                {
                    "metric_a": rank_cols[i],
                    "metric_b": rank_cols[j],
                    "spearman_rho": safe_float(float(rho)),
                    "spearman_p": safe_float(float(p)),
                }
            )

    rank_df = pd.DataFrame(rank_consistency_rows)
    rank_df.to_parquet(out_dir / "rank_consistency.parquet", index=False)
    df.to_parquet(out_dir / "per_model_comparability_metrics.parquet", index=False)

    nat_rank = df["rank_true_nat_delta"].to_numpy(dtype=float)
    pct_rank = df["rank_true_pct_delta"].to_numpy(dtype=float)
    ctrl_rank = df["rank_true_minus_permuted"].to_numpy(dtype=float)
    same_nat_pct = bool(np.allclose(nat_rank, pct_rank, atol=1e-12, rtol=1e-12))
    same_nat_ctrl = bool(np.allclose(nat_rank, ctrl_rank, atol=1e-12, rtol=1e-12))

    if same_nat_pct and same_nat_ctrl:
        interpretation = "baseline_and_control_normalized_axes_align"
        status = "supported_with_caveat"
    elif same_nat_pct:
        interpretation = "baseline_normalized_axis_aligns_control_axis_mixed"
        status = "supported_with_caveat"
    else:
        interpretation = "cross_metric_ranking_mixed"
        status = "mixed"

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": timestamp_now(),
        "models": models,
        "nat_rank_order": df.sort_values("rank_true_nat_delta")["model"].tolist(),
        "pct_rank_order": df.sort_values("rank_true_pct_delta")["model"].tolist(),
        "control_rank_order_true_minus_permuted": df.sort_values("rank_true_minus_permuted")["model"].tolist(),
        "same_nat_vs_pct_rank": same_nat_pct,
        "same_nat_vs_control_rank": same_nat_ctrl,
        "interpretation": interpretation,
        "claim_status": status,
        "note": (
            "Cross-model comparability remains bounded; this sensitivity panel adds baseline-normalized "
            "and control-normalized axes alongside raw nats."
        ),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "cross_model_comparability_summary.json", summary)

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Do baseline-normalized and control-normalized axes materially change cross-model interpretation of Result I?",
        "decision_rule": "If nat, percent, and true-minus-permuted rankings align, interpret cross-model ordering as robust with caveat.",
    }
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "artifacts": [
            "per_model_comparability_metrics.parquet",
            "rank_consistency.parquet",
            "cross_model_comparability_summary.json",
        ],
    }
    claim_impact = {
        "experiment_id": EXPERIMENT_ID,
        "claims_tested": [
            "Cross-model comparability framing for Result I",
        ],
        "status": status,
        "interpretation": interpretation,
        "impact_on_main_text": (
            "Report raw nats together with baseline-normalized and control-normalized contrasts."
            if status != "mixed"
            else "Avoid strong cross-model ordering language; emphasize bounded comparability caveat."
        ),
    }
    data_dictionary = {
        "tables": [
            {
                "path": "per_model_comparability_metrics.parquet",
                "description": "Per-model raw, baseline-normalized, and control-normalized Result-I metrics.",
                "columns": [
                    {"name": "model", "dtype": "str"},
                    {"name": "baseline_loss", "dtype": "float"},
                    {"name": "true_nat_delta", "dtype": "float"},
                    {"name": "true_pct_delta", "dtype": "float"},
                    {"name": "nat_over_baseline", "dtype": "float"},
                    {"name": "true_minus_permuted", "dtype": "float"},
                    {"name": "ratio_true_over_permuted", "dtype": "float"},
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
    parser = argparse.ArgumentParser(description="E25 cross-model comparability sensitivity")
    parser.add_argument("--models", default=",".join(PRIMARY_MODELS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    if args.no_finalize and args.finalize_only:
        raise RuntimeError("[E25] hard_fail_reason: --no-finalize and --finalize-only are mutually exclusive")

    out_dir = Path(args.output_root).resolve()
    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    ensure_dir(out_dir)
    write_json(
        out_dir / "command_manifest.json",
        command_manifest(
            experiment_id=EXPERIMENT_ID,
            command="run_e25_cross_model_comparability.py",
            model="multi-model",
        ),
    )

    if args.finalize_only:
        summary = _finalize(models=models, out_dir=out_dir)
        print(f"[E25] finalize-only: interpretation={summary['interpretation']} status={summary['claim_status']}")
        return

    run_models = models
    for m in run_models:
        rec = _run_model(model_name=m, out_dir=out_dir)
        print(
            f"[E25] model={m} nat_delta={rec['true_nat_delta']:.4f} "
            f"pct_delta={rec['true_pct_delta']:.2f}%"
        )

    if not args.no_finalize:
        summary = _finalize(models=run_models, out_dir=out_dir)
        print(f"[E25] interpretation={summary['interpretation']} status={summary['claim_status']}")


if __name__ == "__main__":
    main()
