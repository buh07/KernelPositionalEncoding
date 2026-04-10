#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "[tmux sessions]"
tmux ls 2>/dev/null | rg 'exp3p2_b_multi' || echo "(none)"

echo
echo "[gpu status]"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "[seed artifacts]"
for s in 0 1 2 3 4 5 6; do
  p="results/experiment3_phase2/exp3p2b_trivial_feature_control_multiseed/seed${s}/llama-3.1-8b/synthetic_boundary_results.json"
  if [[ -f "$p" ]]; then
    echo "  seed${s}: complete"
  else
    echo "  seed${s}: pending"
  fi
done

echo
echo "[pooled summary]"
SUM="results/experiment3_phase2/exp3p2b_trivial_feature_control/llama-3.1-8b/multiseed_gate_summary.json"
if [[ -f "$SUM" ]]; then
  n_done="$(jq -r '.pooled.n_completed // 0' "$SUM" 2>/dev/null || echo 0)"
  if [[ "$n_done" -gt 0 ]]; then
    jq '{assessment, pooled, seeds_completed, missing_seeds}' "$SUM"
  else
    echo "  pending (summary file exists but no completed multiseed entries yet)"
  fi
else
  echo "  not yet generated"
fi
