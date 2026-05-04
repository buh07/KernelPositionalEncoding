#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp2.common import RESULTS_ROOT, command_manifest, ensure_dir, timestamp_now, write_json  # noqa: E402
from reinforce_exp2.scripts._shared import emit_core_artifacts  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="A2 Prediction Layer preregistration", allow_abbrev=False)
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "A2_prediction_layer"))
    args = p.parse_args()

    out_dir = ensure_dir(Path(args.output_root))

    predictions = {
        "timestamp": timestamp_now(),
        "experiment_id": "A2_prediction_layer",
        "predictions": [
            {
                "id": "P1_shuffle_sensitivity",
                "statement": "Higher mean SI-R2 predicts greater degradation under position-shuffle perturbation.",
                "direction": "positive",
                "primary_endpoint": "beta_shuffle",
                "failure_interpretation": "Either SI is not the dominant carrier for position-sensitive robustness, or nuisance factors dominate under current model set.",
                "analysis_link": "C2.1",
            }
        ],
    }
    write_json(out_dir / "predictions_preregistered.json", predictions)

    prereg = {
        "experiment_id": "A2_prediction_layer",
        "question": "Can we pre-register falsifiable predictions before C-path testing?",
        "primary_hypothesis": "At least one falsifiable SI-linked prediction can be frozen before evaluation.",
        "primary_endpoints": ["predictions_preregistered.json"],
        "secondary_endpoints": ["None"],
        "model_list": ["paper-level"],
        "dataset_sources": ["none (design-time artifact)"],
        "inclusion_exclusion_rules": ["Prediction statements must include direction and failure interpretation."],
        "sample_size_plan": {"n_predictions": 1},
        "seed_plan": {"deterministic": True},
        "stopping_rule": "Stop once prediction artifact is frozen.",
        "multiplicity_family": ["Not applicable"],
        "acceptance_criteria": ["Prediction contains explicit endpoint, direction, and failure interpretation."],
        "fallback_interpretation_if_null": "A narrative only paper with no predictive layer."
    }
    manifest = command_manifest(
        experiment_id="A2_prediction_layer",
        command="run_a2_prediction_layer.py",
        model="paper-level",
        extras={"output_root": str(out_dir)},
    )
    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "A2_prediction_layer",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {"completed": True, "n_predictions": 1},
    }
    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "A2_prediction_layer",
        "claim_status": "supported",
        "supports_main_text": True,
        "strict_only": True,
        "notes": ["Prediction layer is frozen pre-run as required by TODO."],
    }
    data_dictionary = {
        "experiment_id": "A2_prediction_layer",
        "tables": [
            {
                "path": str(out_dir / "predictions_preregistered.json"),
                "description": "Pre-registered predictions.",
                "columns": [
                    {"name": "id", "dtype": "str", "description": "Prediction id."},
                    {"name": "statement", "dtype": "str", "description": "Prediction statement."},
                    {"name": "direction", "dtype": "str", "description": "Expected direction."},
                    {"name": "primary_endpoint", "dtype": "str", "description": "Primary endpoint."},
                    {"name": "failure_interpretation", "dtype": "str", "description": "Interpretation if prediction fails."},
                    {"name": "analysis_link", "dtype": "str", "description": "Downstream analysis id."},
                ],
            }
        ],
    }

    emit_core_artifacts(
        experiment_id="A2_prediction_layer",
        out_dir=out_dir,
        preregistration=prereg,
        manifest=manifest,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )

    print(f"[A2] wrote {out_dir}")


if __name__ == "__main__":
    main()
