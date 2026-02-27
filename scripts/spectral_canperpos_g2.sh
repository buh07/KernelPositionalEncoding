#!/usr/bin/env bash
set -euo pipefail
cd "/scratch2/f004ndc/Kernel PE"
export CUDA_VISIBLE_DEVICES=2
LOG_DIR="logs/spectral"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/spectral_canperpos_g2.log"
exec > >(tee -a "$LOG_FILE") 2>&1

PY=".venv/bin/python"
COMMON_ARGS=(
  experiment1/run.py spectral
  --dataset all
  --seq-len all
  --spectral-gate-threshold 0.0
  --spectral-track-b-group track_b_canonical_perpos_v1
  --spectral-output-group spectral_canonical_perpos_v1_t0
)

"$PY" "${COMMON_ARGS[@]}" --model llama-3.2-1b
"$PY" "${COMMON_ARGS[@]}" --model gpt2-medium

echo "GPU2 spectral batch complete."
