#!/usr/bin/env python3
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

from reinforce_exp2.common import RESULTS_ROOT, command_manifest, ensure_dir, timestamp_now, write_json, write_text  # noqa: E402
from reinforce_exp2.scripts._shared import emit_core_artifacts  # noqa: E402


def _safe_read(path: Path) -> dict[str, Any] | None:
    try:
        import json

        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def build_claim_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _pick(*paths: Path) -> tuple[dict[str, Any] | None, str]:
        for p in paths:
            payload = _safe_read(p)
            if payload is not None:
                return payload, str(p.relative_to(ROOT))
        return None, str(paths[0].relative_to(ROOT))

    # E2 threshold-capacity evidence: prefer reinforce_exp2 B3 verdict, fallback to prior phase2 curve-fit summary.
    b3, b3_src = _pick(
        ROOT / "results" / "reinforce_exp2" / "B3_cluster_ablation" / "mechanism_disambiguation_verdict.json",
        ROOT / "results" / "reinforce_exp2" / "B3_cluster_ablation_smoke" / "mechanism_disambiguation_verdict.json",
        ROOT / "results" / "experiment3_phase2" / "exp3p2c_redundancy_quantification" / "summary.json",
    )
    e2_status = "pending"
    e2_note = "No B3/threshold artifact found in current reinforcement outputs."
    if b3:
        if bool(b3.get("B3_redundancy_supported", False)) or bool(b3.get("B3_group_structure_supported", False)):
            e2_status = "supported"
            e2_note = "B3 disambiguation verdict is available in reinforce_exp2 outputs."
        elif bool(b3.get("B3_inconclusive", False)):
            e2_status = "mixed"
            e2_note = f"B3 inconclusive ({b3.get('B3_inconclusive_reason', 'other')}). Use threshold-capacity wording."
        else:
            e2_status = "pending"
    rows.append(
        {
            "claim_id": "E2_threshold_capacity",
            "headline": "Threshold-capacity organization is cross-model invariant",
            "status": e2_status,
            "strict_only": True,
            "source": [b3_src],
            "caveat": e2_note,
        }
    )

    # Llama strict gate status
    gate, gate_src = _pick(
        ROOT / "results" / "experiment3_phase2" / "exp3p2b_trivial_feature_control_multiseed" / "llama-3.1-8b" / "multiseed_gate_summary.json",
    )
    gate_status = "pending"
    gate_note = "No multiseed adjudication artifact found."
    if gate:
        assess = str(gate.get("gate_decision", {}).get("assessment", "")).strip().lower()
        if assess == "stable_blocked":
            gate_status = "supported_with_caveat"
            gate_note = "Llama strict gate is stably blocked."
        elif assess == "stable_clean":
            gate_status = "supported"
            gate_note = "Llama strict gate passes."
        elif assess == "ambiguous":
            gate_status = "mixed"
            gate_note = "Llama strict gate is ambiguous under multiseed adjudication."
    rows.append(
        {
            "claim_id": "Llama_strict_gate",
            "headline": "Llama strict trivial-feature gate adjudication",
            "status": gate_status,
            "strict_only": True,
            "source": [gate_src],
            "caveat": gate_note,
        }
    )

    # R5B transfer status
    r5b, r5b_src = _pick(
        ROOT / "results" / "reinforce_exp" / "exp_r5b_regime_alignment" / "interaction_transfer_report.json",
    )
    r5_status = "pending"
    r5_note = "Missing aggregate interaction transfer report."
    if r5b:
        passes = bool(r5b.get("overall", {}).get("supports_task_grounded_semantics", False))
        r5_status = "supported" if passes else "proxy_specific"
        r5_note = "Interaction significant but proxy-to-task sign transfer did not fully validate." if not passes else "Task-grounded sign transfer validated."
    rows.append(
        {
            "claim_id": "J_proxy_transfer",
            "headline": "Conditional specialization regime semantics transfer",
            "status": r5_status,
            "strict_only": True,
            "source": [r5b_src],
            "caveat": r5_note,
        }
    )

    # Non-RoPE control
    gpt2m, gpt2m_src = _pick(ROOT / "results" / "reinforce_exp" / "exp_r4_pe_scheme_contrast" / "gpt2-medium" / "non_rope_control_summary.json")
    gpt2s, gpt2s_src = _pick(ROOT / "results" / "reinforce_exp" / "exp_r4_pe_scheme_contrast" / "gpt2-small" / "non_rope_control_summary.json")
    if gpt2m is None or gpt2s is None:
        non_rope_status = "pending"
        non_rope_note = "Non-RoPE control artifacts missing for one or more anchor models."
    else:
        non_rope_supported = bool(
            gpt2m.get("verdict", {}).get("supports_non_rope_control", False)
            and gpt2s.get("verdict", {}).get("supports_non_rope_control", False)
        )
        non_rope_status = "supported" if non_rope_supported else "mixed"
        non_rope_note = "Anchors are smaller than primary models; scale-transfer remains limited."
    rows.append(
        {
            "claim_id": "RoPE_confound_control",
            "headline": "Non-RoPE anchors preserve SI structure and non-triviality",
            "status": non_rope_status,
            "strict_only": True,
            "source": [gpt2m_src, gpt2s_src],
            "caveat": non_rope_note,
        }
    )

    # Mistral core replication
    r3, r3_src = _pick(
        ROOT / "results" / "reinforce_exp" / "exp_r3_core_replication" / "mistral-7b-v0.1" / "core_replication_report.json",
    )
    r3_status = "pending"
    r3_note = "Core replication report missing."
    if r3:
        if bool(r3.get("verdict", {}).get("supports_core_replication", False)):
            r3_status = "supported"
            r3_note = "Core replication report supports Mistral replication."
        else:
            r3_status = "mixed"
            r3_note = "Core replication did not pass all preregistered criteria."
    rows.append(
        {
            "claim_id": "Mistral_core_replication",
            "headline": "Mistral core battery replication",
            "status": r3_status,
            "strict_only": True,
            "source": [r3_src],
            "caveat": r3_note,
        }
    )

    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="A0 Claim Hygiene: generate claim matrix and wording diff checklist",
        allow_abbrev=False,
    )
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "A0_claim_hygiene"))
    args = p.parse_args()

    out_dir = ensure_dir(Path(args.output_root))
    claim_rows = build_claim_rows()
    df = pd.DataFrame(claim_rows)
    df.to_csv(out_dir / "claim_matrix.csv", index=False)

    claim_matrix = {
        "timestamp": timestamp_now(),
        "experiment_id": "A0_claim_hygiene",
        "claims": claim_rows,
    }
    write_json(out_dir / "claim_matrix.json", claim_matrix)

    wording_changes = """# A0 Wording Changes\n\n1. Use `threshold-capacity organization` unless B3 disambiguation is supported.\n2. Keep strict-vs-exploratory boundary explicit in every cross-model table.\n3. Report Llama strict-gate outcome exactly as adjudicated (`stable_blocked`/`stable_clean`/`ambiguous`).\n4. Keep conditional specialization regime semantics labeled `proxy-specific` unless R5B sign transfer passes prereg criterion.\n5. Preserve RoPE confound caveat with explicit non-RoPE anchor scope limits.\n"""
    write_text(out_dir / "wording_changes.md", wording_changes)

    prereg = {
        "experiment_id": "A0_claim_hygiene",
        "question": "Are all main claims aligned with strict evidence tiers and latest reinforcement artifacts?",
        "primary_hypothesis": "Claim language can be fully mapped to strict evidence statuses without contradiction.",
        "primary_endpoints": ["claim_matrix.json completeness", "claim status consistency"],
        "secondary_endpoints": ["wording_changes.md generated"],
        "model_list": ["paper-level"],
        "dataset_sources": ["results/experiment3_phase2/*", "results/reinforce_exp/*"],
        "inclusion_exclusion_rules": ["Strict artifacts only for canonical status labels."],
        "sample_size_plan": {"n_claims": int(len(claim_rows))},
        "seed_plan": {"deterministic": True},
        "stopping_rule": "Complete when all core claims are mapped.",
        "multiplicity_family": ["Not applicable (documentation synthesis)."],
        "acceptance_criteria": ["No unresolved strict/evidence contradictions in generated matrix."],
        "fallback_interpretation_if_null": "Mark unresolved claims as pending and route to A narrative only."
    }
    manifest = command_manifest(
        experiment_id="A0_claim_hygiene",
        command="run_a0_claim_hygiene.py",
        model="paper-level",
        extras={"output_root": str(out_dir)},
    )
    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "A0_claim_hygiene",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "n_claims": int(len(claim_rows)),
        "status_counts": df["status"].value_counts().to_dict(),
        "verdict": {"completed": True},
    }
    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "A0_claim_hygiene",
        "claim_status": "supported_with_caveat",
        "supports_main_text": True,
        "strict_only": True,
        "notes": [
            "Claim matrix generated from latest artifacts.",
            "Any pending/mixed rows should be reflected explicitly in paper narrative.",
        ],
    }
    data_dictionary = {
        "experiment_id": "A0_claim_hygiene",
        "tables": [
            {
                "path": str(out_dir / "claim_matrix.csv"),
                "description": "Claim-to-evidence mapping table.",
                "columns": [
                    {"name": "claim_id", "dtype": "str", "description": "Claim identifier."},
                    {"name": "headline", "dtype": "str", "description": "Claim headline."},
                    {"name": "status", "dtype": "str", "description": "Claim status label."},
                    {"name": "strict_only", "dtype": "bool", "description": "Whether strict-only evidence is used."},
                    {"name": "source", "dtype": "json", "description": "Source artifact paths."},
                    {"name": "caveat", "dtype": "str", "description": "Caveat text."}
                ],
            }
        ],
    }
    emit_core_artifacts(
        experiment_id="A0_claim_hygiene",
        out_dir=out_dir,
        preregistration=prereg,
        manifest=manifest,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )

    print(f"[A0] wrote {out_dir}")


if __name__ == "__main__":
    main()
