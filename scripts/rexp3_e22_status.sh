#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "[tmux] sessions"
tmux ls 2>/dev/null | rg 'rexp3_e22' || echo "(none)"

echo
echo "[gpu] status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "[artifacts] shard summary presence"
for model in llama-3.1-8b mistral-7b-v0.1 olmo-2-7b; do
  p="results/reinforce_exp3/E22_task_conditional_specificity/${model}/summary.json"
  if [[ -f "$p" ]]; then
    echo "  OK  $p"
  else
    echo "  MISS $p"
  fi
done

echo
echo "[tail logs]"
for log in \
  reinforce_exp3/logs/rexp3_e22_g0_llama.log \
  reinforce_exp3/logs/rexp3_e22_g1_mistral.log \
  reinforce_exp3/logs/rexp3_e22_g2_olmo.log \
  reinforce_exp3/logs/rexp3_e22_g3_finalize.log; do
  if [[ -f "$log" ]]; then
    echo "--- $log"
    tail -n 20 "$log"
  fi
done
