#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

SEEDS_GPU0="${1:-0,2,4,6}"
SEEDS_GPU1="${2:-1,3,5}"
NUM_SEQ="${3:-32}"
SYNTH="${4:-600}"

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

for s in exp3p2_b_multi_g0 exp3p2_b_multi_g1 exp3p2_b_multi_finalize; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

tmux new-session -d -s exp3p2_b_multi_g0 \
  "cd '$ROOT' && bash scripts/experiment3_phase2_b_multiseed_worker.sh 0 llama-3.1-8b '$SEEDS_GPU0' '$NUM_SEQ' '$SYNTH'"

tmux new-session -d -s exp3p2_b_multi_g1 \
  "cd '$ROOT' && bash scripts/experiment3_phase2_b_multiseed_worker.sh 1 llama-3.1-8b '$SEEDS_GPU1' '$NUM_SEQ' '$SYNTH'"

tmux new-session -d -s exp3p2_b_multi_finalize \
  "cd '$ROOT' && bash scripts/experiment3_phase2_b_multiseed_finalize.sh llama-3.1-8b '0,1,2,3,4,5,6' 120 true"

echo "[launch] started tmux sessions"
tmux ls

echo "Monitor:"
echo "  tmux capture-pane -p -t exp3p2_b_multi_g0 | tail -n 60"
echo "  tmux capture-pane -p -t exp3p2_b_multi_g1 | tail -n 60"
echo "  tmux capture-pane -p -t exp3p2_b_multi_finalize | tail -n 60"
