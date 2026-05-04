#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_SESSION="exp6"
QUEUE_SESSION="exp6_queue"

# Replace any existing queue watcher.
tmux kill-session -t "$QUEUE_SESSION" 2>/dev/null || true

tmux new-session -d -s "$QUEUE_SESSION" -n "queue" \
  "cd '$ROOT_DIR' && RUN_SESSION='$RUN_SESSION' bash scripts/experiment6_queue_after_current.sh"

echo "tmux queue session '$QUEUE_SESSION' launched."
echo "It will auto-start '$RUN_SESSION' after current Experiment 4A/5A/5C runs finish."
echo "  Attach queue: tmux attach -t $QUEUE_SESSION"
echo "  Check queue:  tmux capture-pane -t $QUEUE_SESSION:queue -p | tail -20"
echo "  Later attach run: tmux attach -t $RUN_SESSION"
