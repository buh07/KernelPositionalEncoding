#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-reinforce_exp2_full}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
  echo "Stopped session: $SESSION"
else
  echo "Session not running: $SESSION"
fi
