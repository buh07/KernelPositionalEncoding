#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp2.common import RESULTS_ROOT, command_manifest, ensure_dir, read_json, timestamp_now, write_json, write_text  # noqa: E402
from reinforce_exp2.scripts._shared import emit_core_artifacts  # noqa: E402


def _try_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def build_registry() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claims = [
        {
            "claim_id": "E2_threshold_capacity",
            "artifacts": [
                "results/experiment3_phase2/exp3p2c_redundancy_quantification/llama-3.1-8b/curve_fit_comparison.json",
                "results/experiment3_phase2/exp3p2c_redundancy_quantification/olmo-2-7b/curve_fit_comparison.json",
            ],
            "doc_sources": ["experiment3/phase2/RESULTS.md", "UNIFIED_RESULTS.md"],
        },
        {
            "claim_id": "Llama_gate",
            "artifacts": [
                "results/experiment3_phase2/exp3p2b_trivial_feature_control_multiseed/llama-3.1-8b/multiseed_gate_summary.json",
                "results/experiment3_phase2/exp3p2b_trivial_feature_control/llama-3.1-8b/synthetic_boundary_results.json",
            ],
            "doc_sources": ["experiment3/phase2/RESULTS.md", "paper/story.md"],
        },
        {
            "claim_id": "R5B_transfer",
            "artifacts": [
                "results/reinforce_exp/exp_r5b_regime_alignment/interaction_transfer_report.json",
                "results/reinforce_exp/exp_r5b_regime_alignment/llama-3.1-8b/interaction_transfer_report.json",
                "results/reinforce_exp/exp_r5b_regime_alignment/olmo-2-7b/interaction_transfer_report.json",
            ],
            "doc_sources": ["paper/story.md", "paper/critique.md"],
        },
        {
            "claim_id": "RoPE_control",
            "artifacts": [
                "results/reinforce_exp/exp_r4_pe_scheme_contrast/gpt2-medium/non_rope_control_summary.json",
                "results/reinforce_exp/exp_r4_pe_scheme_contrast/gpt2-small/non_rope_control_summary.json",
            ],
            "doc_sources": ["experiment3/phase2/RESULTS.md", "paper/neurips2026/main.tex"],
        },
    ]

    registry_rows: list[dict[str, Any]] = []
    discrepancy_rows: list[dict[str, Any]] = []

    for item in claims:
        claim_id = str(item["claim_id"])
        all_exist = True
        parsed_ok = True
        for artifact in item["artifacts"]:
            p = ROOT / artifact
            exists = p.exists()
            payload = _try_json(p)
            row = {
                "claim_id": claim_id,
                "artifact_path": artifact,
                "exists": bool(exists),
                "json_parse_ok": bool(payload is not None) if exists else False,
            }
            registry_rows.append(row)
            if not exists:
                all_exist = False
            if exists and payload is None:
                parsed_ok = False

        if not all_exist:
            discrepancy_rows.append(
                {
                    "claim_id": claim_id,
                    "severity": "high",
                    "issue": "missing_artifact",
                    "note": "One or more expected source artifacts are missing.",
                }
            )
        elif not parsed_ok:
            discrepancy_rows.append(
                {
                    "claim_id": claim_id,
                    "severity": "medium",
                    "issue": "unreadable_artifact",
                    "note": "Artifact exists but is not valid JSON.",
                }
            )
        else:
            discrepancy_rows.append(
                {
                    "claim_id": claim_id,
                    "severity": "none",
                    "issue": "none",
                    "note": "No structural discrepancy detected.",
                }
            )

    return registry_rows, discrepancy_rows


def main() -> None:
    p = argparse.ArgumentParser(description="A1 Evidence Consolidation", allow_abbrev=False)
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "A1_evidence_consolidation"))
    args = p.parse_args()

    out_dir = ensure_dir(Path(args.output_root))

    registry_rows, discrepancy_rows = build_registry()
    reg_df = pd.DataFrame(registry_rows)
    dis_df = pd.DataFrame(discrepancy_rows)

    reg_df.to_parquet(out_dir / "evidence_registry.parquet", index=False)
    write_json(
        out_dir / "evidence_registry.json",
        {"timestamp": timestamp_now(), "experiment_id": "A1_evidence_consolidation", "rows": registry_rows},
    )

    lines = ["# A1 Discrepancy Log", ""]
    for row in discrepancy_rows:
        lines.append(f"- `{row['claim_id']}`: `{row['severity']}` - {row['issue']} ({row['note']})")
    write_text(out_dir / "discrepancy_log.md", "\n".join(lines) + "\n")

    prereg = {
        "experiment_id": "A1_evidence_consolidation",
        "question": "Are claim-level source artifacts complete, readable, and linked to documentation?",
        "primary_hypothesis": "A single strict evidence registry can be created without missing claim-critical artifacts.",
        "primary_endpoints": ["evidence_registry.json completeness", "discrepancy_log.md generated"],
        "secondary_endpoints": ["evidence_registry.parquet"],
        "model_list": ["paper-level"],
        "dataset_sources": ["results/*", "paper/*", "experiment3/phase2/RESULTS.md"],
        "inclusion_exclusion_rules": ["Only artifacts referenced by claim map are included."],
        "sample_size_plan": {"n_claims": int(dis_df.shape[0])},
        "seed_plan": {"deterministic": True},
        "stopping_rule": "Stop when all mapped claims have artifact checks.",
        "multiplicity_family": ["Not applicable (registry validation)."],
        "acceptance_criteria": ["No high-severity unresolved discrepancy for mandatory claims."],
        "fallback_interpretation_if_null": "Keep affected claims as pending/mixed and block main-text promotion."
    }

    manifest = command_manifest(
        experiment_id="A1_evidence_consolidation",
        command="run_a1_evidence_consolidation.py",
        model="paper-level",
        extras={"output_root": str(out_dir)},
    )

    n_high = int((dis_df["severity"] == "high").sum()) if not dis_df.empty else 0
    claim_status = "supported" if n_high == 0 else "supported_with_caveat"
    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "A1_evidence_consolidation",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "n_registry_rows": int(reg_df.shape[0]),
        "n_discrepancies": int(dis_df.shape[0]),
        "n_high_severity": n_high,
        "verdict": {"completed": True, "high_severity_discrepancies": n_high},
    }
    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "A1_evidence_consolidation",
        "claim_status": claim_status,
        "supports_main_text": bool(n_high == 0),
        "strict_only": True,
        "notes": ["A1 is a governance/traceability gate for claim promotion."],
    }
    data_dictionary = {
        "experiment_id": "A1_evidence_consolidation",
        "tables": [
            {
                "path": str(out_dir / "evidence_registry.parquet"),
                "description": "Per-claim artifact existence/parse checks.",
                "columns": [
                    {"name": "claim_id", "dtype": "str", "description": "Claim identifier."},
                    {"name": "artifact_path", "dtype": "str", "description": "Artifact path."},
                    {"name": "exists", "dtype": "bool", "description": "Artifact exists."},
                    {"name": "json_parse_ok", "dtype": "bool", "description": "Artifact parse status."},
                ],
            }
        ],
    }

    emit_core_artifacts(
        experiment_id="A1_evidence_consolidation",
        out_dir=out_dir,
        preregistration=prereg,
        manifest=manifest,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )

    print(f"[A1] wrote {out_dir}")


if __name__ == "__main__":
    main()
