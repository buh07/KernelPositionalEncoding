#!/usr/bin/env bash
set -euo pipefail

SESSIONS=(
  reinforce2_r12_g2
  reinforce2_r12_g3
  reinforce2_r2b_g4
  reinforce2_r2b_g5_r14
  reinforce2_r2b_g6_r14
  reinforce2_r2b_finalize
  reinforce2_r5b_g7
)

echo "[tmux sessions]"
tmux ls 2>/dev/null | rg 'reinforce2_' || echo "(none)"
echo

for s in "${SESSIONS[@]}"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    echo "===== $s ====="
    tmux capture-pane -p -t "$s" | tail -n 50
    echo
  fi
done

echo "[gpu status]"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
