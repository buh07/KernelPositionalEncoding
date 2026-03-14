#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?usage: scripts/experiment2_phase2b_posthoc_queue_build.sh <run_id> [output_root] [queue_file]}"
OUT_ROOT="${2:-results/experiment2}"
QUEUE_FILE="${3:-logs/experiment2/${RUN_ID}_phase2b_posthoc_queue.tsv}"
PHASE="phase2b"
PHASE_ROOT="$OUT_ROOT/$PHASE/$RUN_ID"

if [[ ! -d "$PHASE_ROOT" ]]; then
  echo "Missing phase root: $PHASE_ROOT" >&2
  exit 1
fi

mkdir -p "$(dirname "$QUEUE_FILE")"

python - <<'PY' "$RUN_ID" "$PHASE_ROOT" "$QUEUE_FILE"
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

run_id = str(sys.argv[1])
phase_root = Path(sys.argv[2])
queue_path = Path(sys.argv[3])

cfg_paths = sorted(phase_root.glob("**/run_config.json"))
if not cfg_paths:
    raise SystemExit(f"No run_config.json files found under {phase_root}")

groups: dict[tuple[str, int, str], int] = defaultdict(int)
pending_total = 0
for cfg_path in cfg_paths:
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not bool(payload.get("centered_pending", False)):
        continue
    row = payload.get("row") or {}
    model = str(row.get("model", ""))
    if not model:
        continue
    seed = int(row.get("seed", 0))
    split = str(row.get("split", ""))
    groups[(model, seed, split)] += 1
    pending_total += 1

ordered_models = sorted({model for model, _, _ in groups.keys()})
ordered_by_model: dict[str, list[tuple[int, str, int]]] = {}
for model in ordered_models:
    rows = [(seed, split, groups[(model, seed, split)]) for (m, seed, split) in groups.keys() if m == model]
    rows.sort(key=lambda x: (x[0], x[1]))
    ordered_by_model[model] = rows

jobs: list[tuple[str, str, str, int, int, int]] = []
job_idx = 1
cursor = {m: 0 for m in ordered_models}
while any(cursor[m] < len(ordered_by_model[m]) for m in ordered_models):
    for model in ordered_models:
        pos = cursor[model]
        entries = ordered_by_model[model]
        if pos >= len(entries):
            continue
        seed, split, expected_rows = entries[pos]
        cursor[model] = pos + 1
        split_tag = re.sub(r"[^a-zA-Z0-9]+", "_", split).strip("_") or "nosplit"
        job_id = f"posthoc_job{job_idx:04d}_{model}_s{seed}_{split_tag}"
        jobs.append((job_id, model, split, seed, seed, expected_rows))
        job_idx += 1

covered = sum(j[5] for j in jobs)
if covered != pending_total:
    raise SystemExit(
        f"Posthoc queue row conservation failed: queue_rows={covered} pending_rows={pending_total}."
    )

queue_path.write_text(
    "\n".join(
        "\t".join([run_id, job_id, model, split, str(seed_start), str(seed_end), str(expected_rows)])
        for (job_id, model, split, seed_start, seed_end, expected_rows) in jobs
    )
    + ("\n" if jobs else ""),
    encoding="utf-8",
)

meta = {
    "run_id": run_id,
    "phase": "phase2b",
    "pending_centered_rows": int(pending_total),
    "queue_rows_covered": int(covered),
    "job_count": len(jobs),
    "chunk_policy": "model_seed_split",
    "interleaving": "round_robin_by_model",
    "queue_file": str(queue_path),
}
meta_path = queue_path.with_suffix(".meta.json")
meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(meta, sort_keys=True))
PY

echo "Posthoc queue built: $QUEUE_FILE"
echo "Posthoc queue meta: ${QUEUE_FILE%.tsv}.meta.json"
