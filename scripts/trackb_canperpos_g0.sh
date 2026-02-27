#!/usr/bin/env bash
set -euo pipefail

cd "/scratch2/f004ndc/Kernel PE"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

PY=".venv/bin/python"
COMMON_ARGS=(
  experiment1/run.py track-b
  --dataset all
  --seq-len all
  --device cuda
  --track-b-centering-mode canonical_per_position
  --track-b-output-group track_b_canonical_perpos_v1
)

"$PY" "${COMMON_ARGS[@]}" --model tinyllama-nope-1.1b
"$PY" "${COMMON_ARGS[@]}" --model gpt2-small

