#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "[tmux]"
tmux ls 2>/dev/null | rg 'rexp3_e28abcd' || echo "(no rexp3_e28abcd sessions found)"

echo
echo "[gpu]"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

check_triplet() {
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

echo
echo "[artifacts]"
check_triplet "results/reinforce_exp3/E28a_e26_20bin_exact_smoke" "model_summary.json" "E28a smoke"
check_triplet "results/reinforce_exp3/E28a_e26_20bin_exact" "model_summary.json" "E28a full"
check_triplet "results/reinforce_exp3/E28b_importance_matched_control_smoke" "summary.json" "E28b smoke"
check_triplet "results/reinforce_exp3/E28b_importance_matched_control" "summary.json" "E28b full"
check_triplet "results/reinforce_exp3/E28d_r2_localbias_decomposition_smoke" "summary.json" "E28d smoke"
check_triplet "results/reinforce_exp3/E28d_r2_localbias_decomposition" "summary.json" "E28d full"
check_triplet "results/reinforce_exp3/E28c_probe_diversity_transfer_smoke" "summary.json" "E28c smoke"
check_triplet "results/reinforce_exp3/E28c_probe_diversity_transfer" "summary.json" "E28c full"

echo
echo "[logs]"
for f in reinforce_exp3/logs/rexp3_e28abcd_g0_llama.log reinforce_exp3/logs/rexp3_e28abcd_g1_mistral.log reinforce_exp3/logs/rexp3_e28abcd_g2_olmo.log reinforce_exp3/logs/rexp3_e28abcd_g3_finalize.log; do
  if [[ -f "$f" ]]; then
    echo "--- $f (tail) ---"
    tail -n 6 "$f"
  else
    echo "--- $f missing ---"
  fi
  echo
 done
