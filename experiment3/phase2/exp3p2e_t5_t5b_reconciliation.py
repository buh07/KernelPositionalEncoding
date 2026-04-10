#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODELS = ("llama-3.1-8b", "olmo-2-7b")
POSITION_TYPES = (
    "mid_continuation",
    "last_subword",
    "word_initial_after_multi",
    "word_initial_after_single",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _cohens_d_from_summary(
    mean_a: float,
    std_a: float,
    n_a: int,
    mean_b: float,
    std_b: float,
    n_b: int,
) -> float:
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled_var_num = (n_a - 1) * (std_a ** 2) + (n_b - 1) * (std_b ** 2)
    pooled_var_den = n_a + n_b - 2
    if pooled_var_den <= 0:
        return float("nan")
    pooled_std = np.sqrt(pooled_var_num / pooled_var_den)
    if pooled_std == 0 or not np.isfinite(pooled_std):
        return float("nan")
    return float((mean_a - mean_b) / pooled_std)


def _build_loss_increase_table(per_pos_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["sequence_idx", "position", "token_id", "token_type"]
    none_df = (
        per_pos_df.loc[per_pos_df["condition"] == "none", key_cols + ["loss"]]
        .rename(columns={"loss": "loss_none"})
    )
    ablated_df = per_pos_df.loc[per_pos_df["condition"] != "none", key_cols + ["condition", "loss"]]

    merged = ablated_df.merge(none_df, on=key_cols, how="inner")
    merged["loss_increase"] = merged["loss"] - merged["loss_none"]

    agg = merged.groupby(["condition", "token_type"], as_index=False).agg(
        mean_loss_increase=("loss_increase", "mean"),
        std_loss_increase=("loss_increase", "std"),
        median_loss_increase=("loss_increase", "median"),
        n=("loss_increase", "count"),
    )
    return agg


def _build_position_type_breakdown(t5b_report: dict[str, Any], model: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    approach_b = t5b_report.get("approach_b", {})
    baseline = approach_b.get("baseline", {})
    conditions = approach_b.get("conditions", {})

    for ptype in POSITION_TYPES:
        b = baseline.get(ptype, {})
        rows.append(
            {
                "model": model,
                "source": "baseline",
                "condition": "none",
                "position_type": ptype,
                "mean_loss": _safe_float(b.get("mean_loss")),
                "std_loss": _safe_float(b.get("std_loss")),
                "median_loss": _safe_float(b.get("median_loss")),
                "n": int(b.get("n", 0)),
                "mean_loss_increase": float("nan"),
                "ci_95_low": float("nan"),
                "ci_95_high": float("nan"),
            }
        )

    for condition, condition_stats in conditions.items():
        for ptype in POSITION_TYPES:
            s = condition_stats.get(ptype, {})
            ci = s.get("ci_95", [float("nan"), float("nan")])
            rows.append(
                {
                    "model": model,
                    "source": "approach_b_condition",
                    "condition": condition,
                    "position_type": ptype,
                    "mean_loss": float("nan"),
                    "std_loss": float("nan"),
                    "median_loss": float("nan"),
                    "n": int(baseline.get(ptype, {}).get("n", 0)),
                    "mean_loss_increase": _safe_float(s.get("mean_loss_increase")),
                    "ci_95_low": _safe_float(ci[0]),
                    "ci_95_high": _safe_float(ci[1]),
                }
            )

    return pd.DataFrame(rows)


def _compute_boundary_mediation_proxy(t5b_report: dict[str, Any]) -> dict[str, Any]:
    approach_b = t5b_report.get("approach_b", {})
    conditions = approach_b.get("conditions", {})
    baseline = approach_b.get("baseline", {})

    out: dict[str, Any] = {
        "method": "proxy_from_t5b_approach_b_taxonomy",
        "note": (
            "Direct near-vs-far continuation labels are not persisted in cached T5 artifacts; "
            "this proxy uses last_subword (near-boundary continuation) vs mid_continuation."
        ),
        "per_condition": {},
    }

    for cond_name, cond_stats in conditions.items():
        near = cond_stats.get("last_subword", {})
        far = cond_stats.get("mid_continuation", {})
        n_near = int(baseline.get("last_subword", {}).get("n", 0))
        n_far = int(baseline.get("mid_continuation", {}).get("n", 0))
        near_mean = _safe_float(near.get("mean_loss_increase"))
        far_mean = _safe_float(far.get("mean_loss_increase"))
        near_std = _safe_float(near.get("std_loss_increase"))
        far_std = _safe_float(far.get("std_loss_increase"))

        out["per_condition"][cond_name] = {
            "near_boundary_proxy": {
                "position_type": "last_subword",
                "mean_loss_increase": near_mean,
                "std_loss_increase": near_std,
                "n": n_near,
            },
            "far_boundary_proxy": {
                "position_type": "mid_continuation",
                "mean_loss_increase": far_mean,
                "std_loss_increase": far_std,
                "n": n_far,
            },
            "near_minus_far": near_mean - far_mean,
            "cohens_d_near_vs_far": _cohens_d_from_summary(
                mean_a=near_mean,
                std_a=near_std,
                n_a=n_near,
                mean_b=far_mean,
                std_b=far_std,
                n_b=n_far,
            ),
        }

    return out


def run(model: str, output_root: Path) -> None:
    t5_dir = Path("results/experiment3/theory5_subword_ablation") / model
    t5b_dir = Path("results/experiment3/theory5b_boundary_detection") / model

    t5_analysis_path = t5_dir / "analysis.json"
    t5_losses_path = t5_dir / "per_position_losses.parquet"
    t5b_report_path = t5b_dir / "report.json"

    missing = [p for p in (t5_analysis_path, t5_losses_path, t5b_report_path) if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required inputs for {model}: {missing_str}")

    out_dir = output_root / model
    out_dir.mkdir(parents=True, exist_ok=True)

    t5_analysis = _load_json(t5_analysis_path)
    t5b_report = _load_json(t5b_report_path)
    loss_df = pd.read_parquet(t5_losses_path)

    loss_increase_tbl = _build_loss_increase_table(loss_df)
    pos_type_tbl = _build_position_type_breakdown(t5b_report, model)
    mediation = _compute_boundary_mediation_proxy(t5b_report)

    pos_type_tbl.to_parquet(out_dir / "position_type_breakdown.parquet", index=False)
    _write_json(out_dir / "boundary_mediation_analysis.json", mediation)

    t5_unmatched = t5_analysis.get("analysis_unmatched", {}).get("two_way_interaction", {})
    t5_matched = t5_analysis.get("analysis_matched", {}).get("two_way_interaction", {})
    t5b_interaction = t5b_report.get("approach_b", {}).get("interaction", {})
    high_proxy = mediation.get("per_condition", {}).get("ablate_high_si", {})
    low_proxy = mediation.get("per_condition", {}).get("ablate_low_si", {})

    high_proxy_delta = _safe_float(high_proxy.get("near_minus_far"))
    low_proxy_delta = _safe_float(low_proxy.get("near_minus_far"))

    reconciliation = {
        "experiment": "3P2-E_t5_t5b_reconciliation",
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": "tier1_confirmatory_core",
        "primary_test_id": "3P2-E",
        "inputs": {
            "t5_analysis_json": str(t5_analysis_path),
            "t5_per_position_losses_parquet": str(t5_losses_path),
            "t5b_report_json": str(t5b_report_path),
        },
        "t5_two_way_interaction_unmatched": {
            "high_minus_low_interaction": _safe_float(t5_unmatched.get("high_minus_low_interaction")),
            "high_minus_random_interaction": _safe_float(t5_unmatched.get("high_minus_random_interaction")),
            "hypothesis_supported": bool(t5_unmatched.get("hypothesis_supported", False)),
        },
        "t5_two_way_interaction_matched": {
            "high_minus_low_interaction": _safe_float(t5_matched.get("high_minus_low_interaction")),
            "high_minus_random_interaction": _safe_float(t5_matched.get("high_minus_random_interaction")),
            "hypothesis_supported": bool(t5_matched.get("hypothesis_supported", False)),
        },
        "t5b_boundary_interaction": {
            "high_si_boundary_sensitivity": _safe_float(t5b_interaction.get("high_si_boundary_sensitivity")),
            "low_si_boundary_sensitivity": _safe_float(t5b_interaction.get("low_si_boundary_sensitivity")),
            "interaction_high_minus_low": _safe_float(t5b_interaction.get("interaction_high_minus_low")),
        },
        "boundary_mediation_proxy": {
            "high_si_near_minus_far": high_proxy_delta,
            "low_si_near_minus_far": low_proxy_delta,
            "high_minus_low_near_far_delta": high_proxy_delta - low_proxy_delta,
        },
        "loss_increase_summary_by_token_type": loss_increase_tbl.to_dict(orient="records"),
        "reconciliation_verdict": (
            "consistent_with_boundary_primary"
            if np.isfinite(high_proxy_delta)
            and np.isfinite(low_proxy_delta)
            and high_proxy_delta > low_proxy_delta
            else "mixed_or_inconclusive"
        ),
        "limitations": [
            "Near-vs-far continuation mediation uses taxonomy proxy from cached T5b artifacts.",
            "No new model execution is performed in 3P2-E; analysis is artifact-driven.",
        ],
    }
    _write_json(out_dir / "t5_t5b_reconciliation.json", reconciliation)
    print(f"[3P2-E] {model}: wrote artifacts to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3P2-E: T5 vs T5b reconciliation")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--output-root",
        default="results/experiment3_phase2/exp3p2e_t5_t5b_reconciliation",
        help="Output root directory for 3P2-E artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(model=args.model, output_root=Path(args.output_root))


if __name__ == "__main__":
    main()
