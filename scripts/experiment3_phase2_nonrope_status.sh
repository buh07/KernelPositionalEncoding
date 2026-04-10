#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "[tmux]"
tmux ls 2>/dev/null | rg exp3p2_nonrope || echo "(none)"

echo
echo "[summary]"
p="results/experiment3_phase2/exp3p2k_non_rope_control/non_rope_control_summary.json"
if [[ -f "$p" ]]; then
  jq '.' "$p"
else
  echo "pending"
fi
