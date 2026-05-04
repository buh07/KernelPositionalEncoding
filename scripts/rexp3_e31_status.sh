#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
LAST_LOG_FILE="$ROOT/logs/reinforce_exp3/E31_LAST_LOG_DIR.txt"
if [[ -f "$LAST_LOG_FILE" ]]; then
  LOG_DIR="$(cat "$LAST_LOG_FILE")"
else
  LOG_DIR="$ROOT/logs/reinforce_exp3"
fi

echo "== tmux sessions =="
tmux ls 2>/dev/null | rg 'rexp3_e31_gpu' || echo "(no e31 tmux sessions)"

echo
for name in e31a e31b e31c e31d; do
  f="$LOG_DIR/${name}.log"
  if [[ -f "$f" ]]; then
    echo "== tail: $f =="
    tail -n 25 "$f"
    echo
  fi
done
