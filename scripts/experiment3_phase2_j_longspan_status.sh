#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "[tmux sessions]"
tmux ls 2>/dev/null | rg 'exp3p2_j_longspan' || echo "(none)"

echo
echo "[gpu status]"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "[coverage summaries]"
for m in llama-3.1-8b olmo-2-7b; do
  p="results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair/${m}/regime_coverage_manifest.json"
  if [[ -f "$p" ]]; then
    echo "--- $m"
    jq '{regime_coverage_complete, coverage}' "$p"
  else
    echo "--- $m: pending"
  fi
done
