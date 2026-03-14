#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Experiment 2 phase manifests into model-specific shards."
    )
    parser.add_argument("--run-id", required=True, help="Experiment 2 run ID.")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["phase2a", "phase2b", "phase2c"],
        help="Phases to shard.",
    )
    parser.add_argument(
        "--root",
        default="results/experiment2",
        help="Experiment 2 results root (default: results/experiment2).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Optional inclusive lower seed bound for sharding.",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=None,
        help="Optional inclusive upper seed bound for sharding.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_phase(
    root: Path,
    run_id: str,
    phase: str,
    *,
    seed_start: int | None = None,
    seed_end: int | None = None,
) -> dict[str, object]:
    manifest_path = root / phase / run_id / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest for phase {phase}: {manifest_path}")

    src_lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines: list[str] = []
    for raw_line in src_lines:
        row = json.loads(raw_line)
        seed = row.get("seed")
        if seed is None:
            continue
        seed_i = int(seed)
        if seed_start is not None and seed_i < seed_start:
            continue
        if seed_end is not None and seed_i > seed_end:
            continue
        lines.append(raw_line)
    shards_dir = root / phase / run_id / "model_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    buffers: dict[str, list[str]] = defaultdict(list)
    for raw_line in lines:
        row = json.loads(raw_line)
        model = row["model"]
        buffers[model].append(raw_line)

    shard_rows_total = 0
    shard_info: dict[str, dict[str, object]] = {}
    for model, model_lines in sorted(buffers.items()):
        out_path = shards_dir / f"{model}.jsonl"
        out_path.write_text("\n".join(model_lines) + "\n", encoding="utf-8")
        shard_rows_total += len(model_lines)

        unique_models = {json.loads(line)["model"] for line in model_lines}
        if unique_models != {model}:
            raise RuntimeError(
                f"Shard validation failed for {phase}/{model}: unique models={sorted(unique_models)}"
            )
        shard_info[model] = {
            "rows": len(model_lines),
            "manifest": str(out_path),
            "sha256": _sha256(out_path),
        }

    if shard_rows_total != len(lines):
        raise RuntimeError(
            f"Shard row conservation failed for {phase}: source={len(lines)} shard_total={shard_rows_total}"
        )

    return {
        "phase": phase,
        "source_manifest": str(manifest_path),
        "source_rows": len(src_lines),
        "filtered_rows": len(lines),
        "source_sha256": _sha256(manifest_path),
        "shard_rows_total": shard_rows_total,
        "seed_start": seed_start,
        "seed_end": seed_end,
        "shards": shard_info,
    }


def main() -> None:
    args = parse_args()
    if (args.seed_start is not None) and (args.seed_end is not None) and (int(args.seed_start) > int(args.seed_end)):
        raise SystemExit("--seed-start cannot be greater than --seed-end")
    root = Path(args.root)
    payload = {
        "run_id": args.run_id,
        "root": str(root),
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "phases": {},
    }
    for phase in args.phases:
        payload["phases"][phase] = shard_phase(
            root=root,
            run_id=args.run_id,
            phase=phase,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
        )

    summary_path = root / "manifest_shards" / f"{args.run_id}_model_shards_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote shard summary: {summary_path}")


if __name__ == "__main__":
    main()
