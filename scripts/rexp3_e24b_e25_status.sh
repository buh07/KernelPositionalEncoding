#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "[tmux] sessions"
tmux ls 2>/dev/null | rg 'rexp3_e24b_e25' || echo "(none)"

echo
echo "[gpu] status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "[artifacts] shard summary presence"
for model in llama-3.1-8b mistral-7b-v0.1 olmo-2-7b; do
  p24="results/reinforce_exp3/E24b_si_vs_non_si_contrast/${model}/summary.json"
  p25="results/reinforce_exp3/E25_cross_model_comparability/${model}/summary.json"
  [[ -f "$p24" ]] && echo "  OK   $p24" || echo "  MISS $p24"
  [[ -f "$p25" ]] && echo "  OK   $p25" || echo "  MISS $p25"
done

echo
echo "[tail logs]"
for log in \
  reinforce_exp3/logs/rexp3_e24b_e25_g0_llama.log \
  reinforce_exp3/logs/rexp3_e24b_e25_g1_mistral.log \
  reinforce_exp3/logs/rexp3_e24b_e25_g2_olmo.log \
  reinforce_exp3/logs/rexp3_e24b_e25_g3_finalize.log; do
  if [[ -f "$log" ]]; then
    echo "--- $log"
    tail -n 30 "$log"
  fi
done

