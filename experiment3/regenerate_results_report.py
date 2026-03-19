#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


MODELS = ["llama-3.1-8b", "olmo-2-7b"]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _mtime(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))


def _status_from_bool(value: bool | None) -> str:
    if value is True:
        return "SUPPORTED"
    if value is False:
        return "NOT SUPPORTED"
    return "DESCRIPTIVE"


def _fmt_float(value: Any) -> str:
    try:
        if value is None:
            return "N/A"
        v = float(value)
        if v != v:
            return "N/A"
        return f"{v:+.4f}"
    except Exception:
        return "N/A"


def _extract_t1(payload: dict[str, Any]) -> tuple[str, str]:
    agg = payload.get("aggregate", {})
    paired = agg.get("paired_high_vs_low")
    if isinstance(paired, dict):
        status = _status_from_bool(paired.get("supports_high_si_more_damage"))
        metric = (
            f"paired Δdrop={_fmt_float(paired.get('mean_delta_drop_raw'))}, "
            f"p={paired.get('paired_ttest', {}).get('p_value', 'N/A')}"
        )
        return status, metric
    status = _status_from_bool(agg.get("high_si_causes_more_damage"))
    metric = f"high-low Δdrop={_fmt_float(agg.get('high_minus_low_drop_raw'))}"
    return status, metric


def _extract_t3(payload: dict[str, Any]) -> tuple[str, str]:
    agg = payload.get("aggregate", {})
    rho = agg.get("mean_spearman_ct_vs_raw_fisher_z", agg.get("mean_spearman_ct_vs_raw"))
    metric = f"mean Spearman={_fmt_float(rho)}; n_task_span={agg.get('n_task_span_combos', 'N/A')}"
    return "DESCRIPTIVE", metric


def _extract_t5(payload: dict[str, Any]) -> tuple[str, str]:
    twi = payload.get("analysis_unmatched", {}).get("two_way_interaction", {})
    status = _status_from_bool(twi.get("hypothesis_supported"))
    metric = (
        f"high-low interaction={_fmt_float(twi.get('high_minus_low_interaction'))}, "
        f"high={_fmt_float(twi.get('high_si_interaction'))}, low={_fmt_float(twi.get('low_si_interaction'))}"
    )
    return status, metric


def _extract_t5b(payload: dict[str, Any]) -> tuple[str, str]:
    ov = payload.get("overall_verdict", {})
    status = _status_from_bool(ov.get("hypothesis_supported"))
    metric = ov.get("summary", "N/A")
    return status, metric


def _extract_t7(payload: dict[str, Any]) -> tuple[str, str]:
    unified = payload.get("unified_analysis", {})
    overall = unified.get("overall_conclusion", "")
    if isinstance(overall, dict):
        interpretation = str(overall.get("interpretation", ""))
        support_flag = overall.get("feeder_heads_support_si_hypothesis")
        if isinstance(support_flag, str):
            support_flag = support_flag.lower() == "true"
        status = _status_from_bool(bool(support_flag) if support_flag is not None else None)
        metric = interpretation or str(overall)
        return status, metric
    conclusion = str(overall)
    status = "DESCRIPTIVE"
    if "SUPPORT" in conclusion.upper():
        status = "SUPPORTED"
    elif "NOT SUPPORT" in conclusion.upper() or "DOES NOT" in conclusion.upper():
        status = "NOT SUPPORTED"
    metric = conclusion if conclusion else "N/A"
    return status, metric


def _extract_t7b(payload: dict[str, Any]) -> tuple[str, str]:
    analysis = payload.get("analysis", {})
    primary = analysis.get("primary_correlation", {})
    status = _status_from_bool(analysis.get("hypothesis_supported"))
    metric = (
        f"Spearman={_fmt_float(primary.get('spearman_rho'))}, "
        f"p={primary.get('spearman_p', 'N/A')}"
    )
    return status, metric


def _extract_t8(payload: dict[str, Any]) -> tuple[str, str]:
    verdict = str(payload.get("verdict", ""))
    status = "DESCRIPTIVE"
    if " IS FUNCTIONALLY RELEVANT" in verdict.upper():
        status = "SUPPORTED"
    elif "IS NOT FUNCTIONALLY RELEVANT" in verdict.upper():
        status = "NOT SUPPORTED"
    return status, verdict.splitlines()[0] if verdict else "N/A"


def _extract_t9(payload: dict[str, Any]) -> tuple[str, str]:
    interp = payload.get("interpretation", {})
    verdict = str(interp.get("frequency_band_verdict", ""))
    status = "DESCRIPTIVE"
    if "SUPPORT" in verdict.upper():
        status = "SUPPORTED"
    elif "NOT SUPPORT" in verdict.upper():
        status = "NOT SUPPORTED"
    metric = (
        f"rho_hf={_fmt_float(interp.get('rho_disruption_hf'))}, "
        f"rho_lf={_fmt_float(interp.get('rho_disruption_lf'))}"
    )
    return status, metric


def _extract_t10(payload: dict[str, Any]) -> tuple[str, str]:
    analysis = payload.get("analysis", {})
    comp = analysis.get("cross_group_comparison", {})
    status = _status_from_bool(comp.get("induction_specific"))
    metric = (
        f"rho_ind={_fmt_float(comp.get('rho_induction'))}, "
        f"rho_rand={_fmt_float(comp.get('rho_random_mid'))}, "
        f"rho_low={_fmt_float(comp.get('rho_low_si_late'))}"
    )
    return status, metric


def _extract_missing(_: dict[str, Any]) -> tuple[str, str]:
    return "MISSING", "artifact missing"


THEORY_SPECS: list[tuple[str, str, str, Any]] = [
    ("T1", "theory1_si_circuits", "analysis.json", _extract_t1),
    ("T3", "theory3_crossterm", "analysis.json", _extract_t3),
    ("T5", "theory5_subword_ablation", "analysis.json", _extract_t5),
    ("T5b", "theory5b_boundary_detection", "report.json", _extract_t5b),
    ("T7", "theory7_induction_feeders", "report.json", _extract_t7),
    ("T7b", "theory7b_activation_patching", "report.json", _extract_t7b),
    ("T8", "theory8_position_ablation", "report.json", _extract_t8),
    ("T9", "theory9_freq_band_feeder", "report.json", _extract_t9),
    ("T10", "theory10_feeder_specificity", "report.json", _extract_t10),
]


def build_report(results_root: Path, models: list[str]) -> str:
    lines: list[str] = []
    lines.append("# Experiment 3 Results")
    lines.append("")
    lines.append(f"Generated: **{time.strftime('%Y-%m-%d %H:%M:%S')}**")
    lines.append("")
    lines.append("Source policy: this file is generated strictly from artifact JSON files in `results/experiment3/`.")
    lines.append("")

    lines.append("## Artifact Table")
    lines.append("")
    lines.append("| Theory | Model | Artifact | Artifact Timestamp | Status | Key Metric |")
    lines.append("|---|---|---|---|---|---|")

    summary: dict[str, dict[str, str]] = {}

    for theory, folder, filename, extractor in THEORY_SPECS:
        summary.setdefault(theory, {})
        for model in models:
            artifact = results_root / folder / model / filename
            payload = _load_json(artifact)
            if payload is None:
                status, metric = _extract_missing({})
            else:
                status, metric = extractor(payload)
            summary[theory][model] = status
            lines.append(
                f"| {theory} | {model} | `{artifact}` | {_mtime(artifact)} | {status} | {metric} |"
            )

    lines.append("")
    lines.append("## Status Summary")
    lines.append("")
    lines.append("| Theory | " + " | ".join(models) + " |")
    lines.append("|---|" + "|".join(["---" for _ in models]) + "|")
    for theory, *_ in THEORY_SPECS:
        row = [summary[theory].get(model, "MISSING") for model in models]
        lines.append("| " + theory + " | " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `DESCRIPTIVE` means the theory output is correlation-oriented or lacks a binary support flag in artifacts.")
    lines.append("- `MISSING` means the expected artifact file is not present or unreadable.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate experiment3results.md from artifacts")
    parser.add_argument("--results-root", default="results/experiment3")
    parser.add_argument("--output", default="experiment3/experiment3results.md")
    parser.add_argument("--models", nargs="*", default=MODELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output = Path(args.output)
    text = build_report(results_root=results_root, models=list(args.models))
    output.write_text(text, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
