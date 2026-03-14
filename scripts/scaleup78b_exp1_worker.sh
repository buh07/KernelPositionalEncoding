#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <gpu_id> <model_name> <run_tag>"
  exit 2
fi

GPU_ID="$1"
MODEL_NAME="$2"
RUN_TAG="$3"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="logs/scaleup/${RUN_TAG}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/exp1_${MODEL_NAME}_g${GPU_ID}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[scaleup-exp1] start model=${MODEL_NAME} gpu=${GPU_ID} run_tag=${RUN_TAG} at $(date -Iseconds)"

.venv/bin/python experiment1/run.py tokenize \
  --model-profile scaleup_78b \
  --model "$MODEL_NAME" \
  --dataset wiki40b_en_pre2019 \
  --seq-len 1024

.venv/bin/python experiment1/run.py track-a \
  --model-profile scaleup_78b \
  --model "$MODEL_NAME" \
  --dataset wiki40b_en_pre2019 \
  --seq-len 1024 \
  --device "cuda:${GPU_ID}"

.venv/bin/python experiment1/run.py track-b \
  --model-profile scaleup_78b \
  --model "$MODEL_NAME" \
  --dataset wiki40b_en_pre2019 \
  --seq-len 1024 \
  --device "cuda:${GPU_ID}" \
  --track-b-centering-mode legacy_per_position \
  --track-b-output-group track_b_scaleup78b_rawcheck

echo "[scaleup-exp1] done model=${MODEL_NAME} gpu=${GPU_ID} run_tag=${RUN_TAG} at $(date -Iseconds)"
