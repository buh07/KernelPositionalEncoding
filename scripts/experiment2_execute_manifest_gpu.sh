#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MANIFEST_PATH="${1:?usage: scripts/experiment2_execute_manifest_gpu.sh <manifest.jsonl> [device] [max_cells]}"
DEVICE_ARG="${2:-cuda}"
MAX_CELLS="${3:-}"
BATCH_SIZE_SYNTH="${BATCH_SIZE_SYNTH:-8}"
BATCH_SIZE_TIER1="${BATCH_SIZE_TIER1:-8}"
CENTERED_COMPUTE="${CENTERED_COMPUTE:-defer}"

CMD=(
  .venv/bin/python experiment2/run.py
  --mode execute
  --manifest "$MANIFEST_PATH"
  --device "$DEVICE_ARG"
  --centered-compute "$CENTERED_COMPUTE"
  --batch-size-synth "$BATCH_SIZE_SYNTH"
  --batch-size-tier1 "$BATCH_SIZE_TIER1"
  --output-root results/experiment2
  --print-summary
)

if [[ -n "$MAX_CELLS" ]]; then
  CMD+=(--max-cells "$MAX_CELLS")
fi

echo "Executing Experiment 2 manifest on device=${DEVICE_ARG}: ${MANIFEST_PATH}"
"${CMD[@]}"
