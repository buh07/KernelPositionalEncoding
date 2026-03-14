#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_int_csv(raw: str) -> list[int]:
    vals: list[int] = []
    seen: set[int] = set()
    for part in str(raw).split(","):
        tok = part.strip()
        if not tok:
            continue
        value = int(tok)
        if value not in seen:
            vals.append(value)
            seen.add(value)
    return vals


def _parse_str_csv(raw: str) -> list[str]:
    vals: list[str] = []
    seen: set[str] = set()
    for part in str(raw).split(","):
        tok = part.strip()
        if not tok:
            continue
        if tok not in seen:
            vals.append(tok)
            seen.add(tok)
    return vals


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _expand_conditions(interventions: list[str], *, random_draws: int) -> list[tuple[str, int | None]]:
    expanded: list[tuple[str, int | None]] = []
    for name in interventions:
        if name.startswith("random_"):
            for draw in range(max(1, int(random_draws))):
                expanded.append((name, draw))
        else:
            expanded.append((name, None))
    return expanded


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
    ap = argparse.ArgumentParser(description="Build fixed-span quick manifests from existing Option C core rows.")
    ap.add_argument("--input-manifest", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, default=None)
    ap.add_argument("--run-id", type=str, required=True)
    ap.add_argument("--phase", type=str, required=True)
    ap.add_argument("--split", type=str, default="synthetic")
    ap.add_argument("--model", type=str, default="llama-3.1-8b")
    ap.add_argument("--task", type=str, default="long_range_retrieval")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--force-spans", type=str, required=True, help="Comma-separated forced spans, e.g. 32,48,64")
    ap.add_argument(
        "--source-interventions",
        type=str,
        default="none",
        help="Source-row interventions to use as templates (usually none).",
    )
    ap.add_argument(
        "--expand-interventions",
        type=str,
        required=True,
        help="Target interventions for output rows (e.g. none,ablate_high_strong,ablate_low_strong,random_strong).",
    )
    ap.add_argument("--random-draws", type=int, default=3)
    ap.add_argument("--notes-tag", type=str, default="quick_stress_test")
    args = ap.parse_args()

    rows = _load_manifest(args.input_manifest)
    seeds = set(_parse_int_csv(args.seeds))
    spans = _parse_int_csv(args.force_spans)
    src_interventions = set(_parse_str_csv(args.source_interventions))
    target_interventions = _parse_str_csv(args.expand_interventions)
    expanded = _expand_conditions(target_interventions, random_draws=max(1, int(args.random_draws)))

    base_rows: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("split")) != str(args.split):
            continue
        if str(row.get("model")) != str(args.model):
            continue
        if str(row.get("task")) != str(args.task):
            continue
        if int(row.get("seq_len", 0) or 0) != int(args.seq_len):
            continue
        if int(row.get("seed", -1)) not in seeds:
            continue
        if str(row.get("intervention")) not in src_interventions:
            continue
        base_rows.append(dict(row))

    if not base_rows:
        raise RuntimeError(
            "No template rows matched filters. "
            f"model={args.model} task={args.task} split={args.split} seeds={sorted(seeds)} "
            f"source_interventions={sorted(src_interventions)}"
        )

    # Keep one template row per seed.
    template_by_seed: dict[int, dict[str, Any]] = {}
    for row in sorted(base_rows, key=_row_sort_key):
        seed = int(row["seed"])
        template_by_seed.setdefault(seed, row)

    out_rows: list[dict[str, Any]] = []
    for seed in sorted(template_by_seed):
        template = template_by_seed[seed]
        for span in spans:
            for intervention, random_draw in expanded:
                row = dict(template)
                row["phase"] = str(args.phase)
                row["run_id"] = str(args.run_id)
                row["intervention"] = str(intervention)
                row["random_draw"] = (None if random_draw is None else int(random_draw))
                row["span"] = int(span)
                note = str(row.get("notes") or "").strip()
                suffix = f"{args.notes_tag}|forced_span={span}"
                row["notes"] = suffix if not note else f"{note}|{suffix}"
                out_rows.append(row)

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
        "model": str(args.model),
        "task": str(args.task),
        "seq_len": int(args.seq_len),
        "seeds": sorted(int(x) for x in seeds),
        "forced_spans": [int(x) for x in spans],
        "target_interventions": target_interventions,
        "random_draws": int(args.random_draws),
        "row_count": int(len(out_rows)),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
