#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:-exp2_gpu345_p2ab_fixv5_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${2:-results/experiment2}"

for phase in phase2a phase2b; do
  echo "Building manifest for ${phase} run_id=${RUN_ID}"
  .venv/bin/python experiment2/run.py \
    --mode build \
    --phase "$phase" \
    --device cuda \
    --run-id "$RUN_ID" \
    --output-root "$OUT_ROOT" \
    --print-summary
done

.venv/bin/python scripts/experiment2_shard_manifests_by_model.py \
  --run-id "$RUN_ID" \
  --phases phase2a phase2b \
  --root "$OUT_ROOT"

echo "Prepared phase2a/phase2b model shards for run_id=${RUN_ID}"
