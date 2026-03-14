#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?usage: scripts/experiment2_phase2b_queue_build.sh <run_id> [output_root] [queue_file] [chunk_size] [manifest_path]}"
OUT_ROOT="${2:-results/experiment2}"
QUEUE_FILE="${3:-logs/experiment2/${RUN_ID}_phase2b_queue.tsv}"
CHUNK_SIZE="${4:-1}"
PHASE="phase2b"
MANIFEST_PATH="${5:-$OUT_ROOT/$PHASE/$RUN_ID/manifest.jsonl}"

if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] || [[ "$CHUNK_SIZE" -lt 1 ]]; then
  echo "chunk_size must be a positive integer, got: $CHUNK_SIZE" >&2
  exit 2
fi

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Missing manifest: $MANIFEST_PATH" >&2
  exit 1
fi

PHASE_DIR="$OUT_ROOT/$PHASE/$RUN_ID"
if [[ ! -d "$PHASE_DIR" ]]; then
  echo "Missing phase directory: $PHASE_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$QUEUE_FILE")"

python - <<'PY' "$MANIFEST_PATH" "$PHASE_DIR" "$QUEUE_FILE" "$CHUNK_SIZE"
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

manifest_path = Path(sys.argv[1])
phase_dir = Path(sys.argv[2])
queue_path = Path(sys.argv[3])
chunk_size = int(sys.argv[4])

rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
expected_total = len(rows)
if expected_total == 0:
    raise SystemExit(f"Manifest has zero rows: {manifest_path}")

by_model: dict[str, list[dict]] = defaultdict(list)
for row in rows:
    by_model[str(row.get("model", "unknown"))].append(row)

manifest_tag = hashlib.sha1(str(manifest_path.resolve()).encode("utf-8")).hexdigest()[:12]
shard_dir = phase_dir / "queue_shards" / manifest_tag
shard_dir.mkdir(parents=True, exist_ok=True)
for stale in shard_dir.glob("*.jsonl"):
    stale.unlink()

model_jobs: dict[str, list[tuple[Path, int, int, int]]] = {}
covered_total = 0
for model in sorted(by_model):
    model_rows = by_model[model]
    shard_path = shard_dir / f"{model}.jsonl"
    shard_path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in model_rows) + "\n",
        encoding="utf-8",
    )
    seeds = sorted({int(r["seed"]) for r in model_rows if "seed" in r})
    if not seeds:
        raise SystemExit(f"Model shard has no seeds: model={model} shard={shard_path}")
    jobs: list[tuple[Path, int, int, int]] = []
    for i in range(0, len(seeds), chunk_size):
        s0 = seeds[i]
        s1 = seeds[min(i + chunk_size - 1, len(seeds) - 1)]
        count = sum(1 for r in model_rows if s0 <= int(r["seed"]) <= s1)
        if count <= 0:
            continue
        jobs.append((shard_path, s0, s1, count))
        covered_total += count
    model_jobs[model] = jobs

if covered_total != expected_total:
    raise SystemExit(
        f"Queue row conservation failed: queue_rows={covered_total} manifest_rows={expected_total}."
    )

ordered_jobs: list[tuple[str, Path, str, int, int, int]] = []
job_idx = 1
queues = {m: list(jobs) for m, jobs in model_jobs.items()}
while any(queues[m] for m in sorted(queues)):
    for model in sorted(queues):
        if not queues[model]:
            continue
        shard_path, s0, s1, count = queues[model].pop(0)
        job_id = f"job{job_idx:03d}_{model}_s{s0}-{s1}"
        ordered_jobs.append((job_id, shard_path, model, s0, s1, count))
        job_idx += 1

queue_path.write_text(
    "\n".join(
        "\t".join(
            [
                job_id,
                str(manifest),
                model,
                str(seed_start),
                str(seed_end),
                str(expected_rows),
            ]
        )
        for (job_id, manifest, model, seed_start, seed_end, expected_rows) in ordered_jobs
    )
    + "\n",
    encoding="utf-8",
)

meta_path = queue_path.with_suffix(".meta.json")
meta = {
    "manifest_used": str(manifest_path),
    "manifest_rows": expected_total,
    "queue_rows_covered": covered_total,
    "job_count": len(ordered_jobs),
    "chunk_size": chunk_size,
    "models": sorted(model_jobs.keys()),
    "interleaving": "round_robin_by_model",
    "shard_dir": str(shard_dir),
}
meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(meta, sort_keys=True))
PY

echo "Queue built: $QUEUE_FILE"
echo "Queue meta: ${QUEUE_FILE%.tsv}.meta.json"
