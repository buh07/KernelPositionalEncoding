#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "== tmux sessions =="
tmux ls | rg 'rexp3_r20_r21' || true

echo ""
echo "== GPU snapshot =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "== log tails =="
for f in \
  reinforce_exp3/logs/rexp3_r20_r21_g4_llama.log \
  reinforce_exp3/logs/rexp3_r20_r21_g5_mistral.log \
  reinforce_exp3/logs/rexp3_r20_r21_g6_olmo.log \
  reinforce_exp3/logs/rexp3_r20_r21_g7_r21.log \
  reinforce_exp3/logs/rexp3_r20_r21_finalize.log
  do
  echo "--- $f ---"
  if [[ -f "$f" ]]; then
    tail -n 25 "$f"
  else
    echo "(missing)"
  fi
  echo ""
done
