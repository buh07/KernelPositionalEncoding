#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_int_csv(raw: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for part in str(raw).split(","):
        tok = part.strip()
        if not tok:
            continue
        val = int(tok)
        if val not in seen:
            seen.add(val)
            out.append(val)
    return out


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _row_sort_key(row: dict[str, Any]) -> tuple:
    return (
        str(row.get("phase", "")),
        str(row.get("split", "")),
        str(row.get("model", "")),
        str(row.get("task", "")),
        int(row.get("seq_len", 0) or 0),
        int(row.get("span", -1) or -1),
        str(row.get("intervention", "")),
        int(row.get("random_draw", -1) if row.get("random_draw") is not None else -1),
        int(row.get("seed", -1)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build expanded cross-model pair-sweep manifests.")
    ap.add_argument("--input-manifest", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    ap.add_argument("--run-id", type=str, required=True)
    ap.add_argument("--phase", type=str, default="quick_pair_expand")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--pair-indices", type=str, default="0,8,16,24,32,40,48,56")
    ap.add_argument("--random-draws", type=int, default=3)
    ap.add_argument("--families-json", type=Path, required=True)
    ap.add_argument("--source-intervention", type=str, default="none")
    ap.add_argument("--notes-tag", type=str, default="quick_pair_expanded")
    args = ap.parse_args()

    source_rows = _load_manifest(args.input_manifest)
    seeds = set(_parse_int_csv(args.seeds))
    pair_indices = _parse_int_csv(args.pair_indices)
    random_draws = max(1, int(args.random_draws))
    families = json.loads(args.families_json.read_text(encoding="utf-8"))
    if not isinstance(families, list) or not families:
        raise RuntimeError(f"families-json must contain a non-empty list: {args.families_json}")

    template_map: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in source_rows:
        if str(row.get("split")) != "synthetic":
            continue
        if str(row.get("intervention")) != str(args.source_intervention):
            continue
        seed = int(row.get("seed", -1))
        if seed not in seeds:
            continue
        key = (str(row.get("model")), str(row.get("task")), seed)
        template_map.setdefault(key, dict(row))

    out_rows: list[dict[str, Any]] = []
    family_summaries: list[dict[str, Any]] = []
    for family in families:
        model = str(family["model"])
        task = str(family["task"])
        spans = family.get("spans")
        # For local_key_match we keep the template span if no explicit override is provided.
        for seed in sorted(seeds):
            t_key = (model, task, seed)
            if t_key not in template_map:
                raise RuntimeError(f"Missing template row for family={model}/{task}, seed={seed}")
            template = template_map[t_key]
            span_values: list[int]
            if spans is None:
                span_values = [int(template.get("span", 0) or 0)]
            else:
                span_values = [int(v) for v in spans]
            for span in span_values:
                base = dict(template)
                base["phase"] = str(args.phase)
                base["run_id"] = str(args.run_id)
                base["intervention"] = "none"
                base["random_draw"] = None
                base["span"] = int(span)
                base_note = str(base.get("notes") or "").strip()
                suffix = f"{args.notes_tag}|model={model}|task={task}|span={span}"
                base["notes"] = suffix if not base_note else f"{base_note}|{suffix}"
                out_rows.append(base)

                for pair in pair_indices:
                    row = dict(base)
                    row["intervention"] = f"ablate_pair_{pair}"
                    out_rows.append(row)

                for draw in range(random_draws):
                    row = dict(base)
                    row["intervention"] = "random_pair"
                    row["random_draw"] = int(draw)
                    out_rows.append(row)

        family_summaries.append(
            {
                "model": model,
                "task": task,
                "spans": [int(v) for v in (spans if spans is not None else [])],
                "seeds": sorted(int(s) for s in seeds),
            }
        )

    out_rows.sort(key=_row_sort_key)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "input_manifest": str(args.input_manifest),
        "output_manifest": str(args.output_manifest),
        "run_id": str(args.run_id),
        "phase": str(args.phase),
        "families": family_summaries,
        "pair_indices": [int(p) for p in pair_indices],
        "random_draws": random_draws,
        "row_count": int(len(out_rows)),
        "notes_tag": str(args.notes_tag),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
