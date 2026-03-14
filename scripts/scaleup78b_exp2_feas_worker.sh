#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <gpu_id> <run_tag> <feas_run_id>"
  exit 2
fi

GPU_ID="$1"
RUN_TAG="$2"
FEAS_RUN_ID="$3"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="logs/scaleup/${RUN_TAG}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/exp2_feas_g${GPU_ID}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[scaleup-exp2-feas] start gpu=${GPU_ID} run_id=${FEAS_RUN_ID} at $(date -Iseconds)"

LONG_TASK_FEASIBILITY_POLICY="${LONG_TASK_FEASIBILITY_POLICY:-retrieval_fallback}"

.venv/bin/python experiment2/run.py \
  --mode feasibility-sweep \
  --model-profile scaleup_78b \
  --h12-endpoint-policy co_primary_raw_headroom \
  --run-id "$FEAS_RUN_ID" \
  --output-root results/experiment2 \
  --device "cuda:${GPU_ID}" \
  --feasibility-offsets 8,12,16,24,32,48,64,96,128 \
  --lock-candidate-offsets 8,12,16,24,32,48,64,96,128 \
  --long-task-feasibility-policy "$LONG_TASK_FEASIBILITY_POLICY" \
  --floor-threshold 0.15 \
  --feasibility-task-only \
  --print-summary

echo "[scaleup-exp2-feas] done gpu=${GPU_ID} run_id=${FEAS_RUN_ID} at $(date -Iseconds)"
