#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "[tmux]"
tmux ls 2>/dev/null | rg 'rexp3_e29abc' || echo "(no rexp3_e29abc sessions found)"

echo
echo "[gpu]"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

check_three() {
  local root="$1"
  local fname="$2"
  local label="$3"
  local n=0
  [[ -f "$root/llama-3.1-8b/$fname" ]] && n=$((n+1))
  [[ -f "$root/mistral-7b-v0.1/$fname" ]] && n=$((n+1))
  [[ -f "$root/olmo-2-7b/$fname" ]] && n=$((n+1))
  local done="no"
  [[ -f "$root/summary.json" ]] && done="yes"
  echo "[$label] shards=$n/3 finalized=$done root=$root"
}

check_one() {
  local root="$1"
  local model="$2"
  local fname="$3"
  local label="$4"
  local n=0
  [[ -f "$root/$model/$fname" ]] && n=1
  local done="no"
  [[ -f "$root/summary.json" ]] && done="yes"
  echo "[$label] shards=$n/1 finalized=$done root=$root"
}

echo
echo "[artifacts]"
check_three "results/reinforce_exp3/E29a_kernel_transplant_specificity_smoke" "summary.json" "E29A smoke"
check_three "results/reinforce_exp3/E29a_kernel_transplant_specificity" "summary.json" "E29A full"
check_three "results/reinforce_exp3/E29b_naturaltext_longcontext_probe_smoke" "summary.json" "E29B smoke"
check_three "results/reinforce_exp3/E29b_naturaltext_longcontext_probe" "summary.json" "E29B full"
check_one "results/reinforce_exp3/E29c_qwen_anchor_quickcheck_smoke" "qwen2.5-7b" "summary.json" "E29C smoke"
check_one "results/reinforce_exp3/E29c_qwen_anchor_quickcheck" "qwen2.5-7b" "summary.json" "E29C full"

echo
echo "[logs]"
for f in reinforce_exp3/logs/rexp3_e29abc_g0.log reinforce_exp3/logs/rexp3_e29abc_g1.log reinforce_exp3/logs/rexp3_e29abc_g2.log reinforce_exp3/logs/rexp3_e29abc_g3.log; do
  if [[ -f "$f" ]]; then
    echo "--- $f (tail) ---"
    tail -n 8 "$f"
  else
    echo "--- $f missing ---"
  fi
  echo
done
