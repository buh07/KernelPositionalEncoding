#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:-run_$(date +%Y%m%d_%H%M%S)_gpu}"
PHASE="${2:-all}"
DEVICE_ARG="${3:-cuda}"
PHASE2A_BASELINE_ONLY="${PHASE2A_BASELINE_ONLY:-0}"

if [[ "$DEVICE_ARG" == auto ]]; then
  EXTRA_ARGS=(--allow-cpu-fallback)
else
  EXTRA_ARGS=()
fi
if [[ "$PHASE2A_BASELINE_ONLY" == "1" ]]; then
  EXTRA_ARGS+=(--phase2a-baseline-only)
fi

echo "Building Experiment 2 manifest: run_id=${RUN_ID}, phase=${PHASE}, device=${DEVICE_ARG}"
.venv/bin/python experiment2/run.py \
  --phase "$PHASE" \
  --device "$DEVICE_ARG" \
  --run-id "$RUN_ID" \
  --output-root results/experiment2 \
  --print-summary \
  "${EXTRA_ARGS[@]}"
