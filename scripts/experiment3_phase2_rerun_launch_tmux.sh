#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

for s in exp3p2_b_llama exp3p2_b_olmo exp3p2_d1_mistral; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s"
  fi
done

tmux new-session -d -s exp3p2_b_llama "cd '$ROOT' && bash scripts/experiment3_phase2_rerun_b_full_gpu0.sh"
tmux new-session -d -s exp3p2_b_olmo "cd '$ROOT' && bash scripts/experiment3_phase2_rerun_b_full_gpu1.sh"
tmux new-session -d -s exp3p2_d1_mistral "cd '$ROOT' && bash scripts/experiment3_phase2_rerun_d1_clean_gpu2.sh"

echo "[launch] tmux sessions started"
tmux ls

echo
echo "Monitor commands:"
echo "  tmux capture-pane -p -t exp3p2_b_llama | tail -n 80"
echo "  tmux capture-pane -p -t exp3p2_b_olmo | tail -n 80"
echo "  tmux capture-pane -p -t exp3p2_d1_mistral | tail -n 80"
