#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-2}"
SESSION="exp8_gpu${GPU_ID}"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "exp8_gpu${GPU_ID}"
tmux send-keys -t "$SESSION:exp8_gpu${GPU_ID}" "cd '$ROOT_DIR' && GPU_ID=${GPU_ID} bash scripts/experiment8_gpu0.sh" C-m

echo "tmux session '$SESSION' launched."
echo "  Attach: tmux attach -t $SESSION"
echo "  Check:  tmux capture-pane -t $SESSION:exp8_gpu${GPU_ID} -p | tail -20"
