#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
LOG_DIR="$ROOT/results/cs_fix_logs"

echo "== tmux sessions =="
tmux ls 2>/dev/null | rg 'csfix_g[0-3]_' || echo "No csfix sessions found"

echo
for log in csfix_g0_exp7b_single.log csfix_g1_exp7b_global.log csfix_g2_exp7a.log csfix_g3_e31b.log; do
  p="$LOG_DIR/$log"
  echo "== $log =="
  if [[ -f "$p" ]]; then
    tail -n 20 "$p"
    if rg -n "Traceback|ERROR|RuntimeError|CUDA out of memory" "$p" >/dev/null 2>&1; then
      echo "!! detected potential error markers in $log"
    fi
  else
    echo "(log not created yet)"
  fi
  echo
 done
