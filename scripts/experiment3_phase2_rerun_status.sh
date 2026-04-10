#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "[status] $(date)"
tmux ls || true
echo
for s in exp3p2_b_llama exp3p2_b_olmo exp3p2_d1_mistral; do
  echo "===== $s ====="
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux capture-pane -p -t "$s" | tail -n 80
  else
    echo "(not running)"
  fi
  echo
done

echo "[gpu]"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
