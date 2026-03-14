#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:-exp2_gpu345_modelpin_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${2:-results/experiment2}"

echo "Preparing Experiment 2 model-pinned run: run_id=${RUN_ID}"

for phase in phase2a phase2b phase2c; do
  echo "Building manifest for ${phase}"
  .venv/bin/python experiment2/run.py \
    --mode build \
    --phase "$phase" \
    --device cuda \
    --run-id "$RUN_ID" \
    --output-root "$OUT_ROOT" \
    --print-summary
done

echo "Sharding manifests by model"
.venv/bin/python scripts/experiment2_shard_manifests_by_model.py \
  --run-id "$RUN_ID" \
  --phases phase2a phase2b phase2c \
  --root "$OUT_ROOT"

echo "Run prepared: ${RUN_ID}"
