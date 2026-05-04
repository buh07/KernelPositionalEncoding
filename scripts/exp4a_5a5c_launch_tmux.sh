#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXP4A_SESSION="exp4a_fix_gpu0"
EXP5_SESSION="exp5_fix_gpu1"

tmux kill-session -t "$EXP4A_SESSION" 2>/dev/null || true
tmux kill-session -t "$EXP5_SESSION" 2>/dev/null || true

tmux new-session -d -s "$EXP4A_SESSION" "bash '$ROOT_DIR/scripts/experiment4a_rerun_gpu0.sh'"
tmux new-session -d -s "$EXP5_SESSION" "bash '$ROOT_DIR/scripts/experiment5_fix_rerun_gpu1.sh'"

echo "Started tmux sessions:"
tmux ls

echo ""
echo "Attach commands:"
echo "  tmux attach -t $EXP4A_SESSION"
echo "  tmux attach -t $EXP5_SESSION"
