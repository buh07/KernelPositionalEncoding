#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

GPU="${1:-2}"
MODEL="${2:-auto}"
NUM_SEQ="${3:-24}"
SYNTH="${4:-300}"

for s in exp3p2_nonrope; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

tmux new-session -d -s exp3p2_nonrope \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$GPU' PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' .venv/bin/python -u experiment3/phase2/exp3p2k_non_rope_control.py --model '$MODEL' --device cuda:0 --num-sequences '$NUM_SEQ' --seq-len 512 --top-k-dims 16 --synthetic-target-per-cell '$SYNTH' --seed 0 --output-root results/experiment3_phase2/exp3p2k_non_rope_control"

echo "[launch] started exp3p2_nonrope on GPU $GPU"
tmux ls | rg exp3p2_nonrope || true
