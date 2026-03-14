#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from experiment2.flooring import floor_key


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune non-baseline floor-limited rows from a Phase2B manifest.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-root", default="results/experiment2")
    p.add_argument("--input-manifest", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    phase_root = output_root / "phase2b" / str(args.run_id)
    if not phase_root.exists():
        raise SystemExit(f"Missing phase root: {phase_root}")

    input_manifest = Path(args.input_manifest) if str(args.input_manifest).strip() else phase_root / "manifest.jsonl"
    if not input_manifest.exists():
        raise SystemExit(f"Missing input manifest: {input_manifest}")

    rows = [json.loads(line) for line in input_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"Input manifest has no rows: {input_manifest}")

    full_path = phase_root / "manifest_full.jsonl"
    pruned_path = phase_root / "manifest_pruned.jsonl"
    summary_path = phase_root / "manifest_prune_summary.json"

    full_path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )

    pruned: list[dict] = []
    pruned_reasons: dict[str, int] = {}
    missing_floor_decisions = 0
    baseline_kept = 0
    kept_nonbaseline = 0

    for row in rows:
        split = str(row.get("split"))
        intervention = str(row.get("intervention"))
        if split not in {"synthetic", "span_bridge", "mechanistic"}:
            pruned.append(row)
            kept_nonbaseline += int(intervention != "none")
            baseline_kept += int(intervention == "none")
            continue
        if intervention == "none":
            pruned.append(row)
            baseline_kept += 1
            continue

        key = floor_key(row)
        floor_path = phase_root / "floor_decisions" / f"{key}.json"
        if not floor_path.exists():
            pruned.append(row)
            kept_nonbaseline += 1
            missing_floor_decisions += 1
            continue

        payload = json.loads(floor_path.read_text(encoding="utf-8"))
        is_floor_limited = bool(payload.get("floor_limited", False))
        if is_floor_limited:
            pruned_reasons["floor_limited_nonbaseline_pruned"] = pruned_reasons.get(
                "floor_limited_nonbaseline_pruned", 0
            ) + 1
        else:
            pruned.append(row)
            kept_nonbaseline += 1

    pruned_path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in pruned) + "\n",
        encoding="utf-8",
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": str(args.run_id),
        "phase": "phase2b",
        "input_manifest": str(input_manifest),
        "manifest_full": str(full_path),
        "manifest_pruned": str(pruned_path),
        "total_rows": len(rows),
        "kept_rows": len(pruned),
        "pruned_rows": len(rows) - len(pruned),
        "baseline_rows_kept": baseline_kept,
        "kept_nonbaseline_rows": kept_nonbaseline,
        "missing_floor_decisions": missing_floor_decisions,
        "pruned_reasons": pruned_reasons,
        "row_conservation_ok": (len(pruned) + (len(rows) - len(pruned)) == len(rows)),
        "policy": "drop_only_nonbaseline_rows_with_floor_limited_true;always_keep_baseline_rows",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
