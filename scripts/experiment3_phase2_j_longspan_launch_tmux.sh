#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

NUM_SEQ="${1:-160}"
SHARDS="${2:-16}"
DIST="${3:-64}"
MIN_SAMPLES="${4:-256}"

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

for s in exp3p2_j_longspan_llama exp3p2_j_longspan_olmo; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

tmux new-session -d -s exp3p2_j_longspan_llama \
  "cd '$ROOT' && bash scripts/experiment3_phase2_j_longspan_worker.sh 0 llama-3.1-8b '$NUM_SEQ' '$SHARDS' '$DIST' '$MIN_SAMPLES'"

tmux new-session -d -s exp3p2_j_longspan_olmo \
  "cd '$ROOT' && bash scripts/experiment3_phase2_j_longspan_worker.sh 1 olmo-2-7b '$NUM_SEQ' '$SHARDS' '$DIST' '$MIN_SAMPLES'"

echo "[launch] started: exp3p2_j_longspan_llama, exp3p2_j_longspan_olmo"
tmux ls
