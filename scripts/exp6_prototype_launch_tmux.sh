#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION="exp6_proto"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "tinyllama_proto" \
  "cd '$ROOT_DIR' && bash scripts/experiment6_prototype_tinyllama_gpu0.sh"

echo "tmux session '$SESSION' launched."
echo "  Attach: tmux attach -t $SESSION"
echo "  Check:  tmux capture-pane -t $SESSION:tinyllama_proto -p | tail -40"
