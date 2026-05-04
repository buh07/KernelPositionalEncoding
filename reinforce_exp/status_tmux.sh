#!/usr/bin/env bash
set -euo pipefail

SESSIONS=(
  reinforce_r1_g2
  reinforce_r1_g3
  reinforce_r1_g4
  reinforce_r1_finalize
  reinforce_r2
  reinforce_r3
  reinforce_r4
  reinforce_r5
)

echo "[tmux sessions]"
tmux ls 2>/dev/null | rg 'reinforce_' || echo "(none)"

echo
for s in "${SESSIONS[@]}"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    echo "===== $s ====="
    tmux capture-pane -p -t "$s" | tail -n 40
    echo
  fi
done

echo "[gpu status]"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
