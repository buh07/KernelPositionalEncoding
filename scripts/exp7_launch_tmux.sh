#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION="exp7_gpu1"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "exp7" \
  "cd '$ROOT_DIR' && bash scripts/experiment7_gpu1.sh"

echo "tmux session '$SESSION' launched."
echo "  Attach: tmux attach -t $SESSION"
echo "  Check:  tmux capture-pane -t $SESSION:exp7 -p | tail -40"

