#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXP4A_SESSION="exp4a_rerun_gpu0"
EXP5D_SESSION="exp5d_rerun_gpu1"

tmux kill-session -t "$EXP4A_SESSION" 2>/dev/null || true
tmux kill-session -t "$EXP5D_SESSION" 2>/dev/null || true

tmux new-session -d -s "$EXP4A_SESSION" "bash '$ROOT_DIR/scripts/experiment4a_rerun_gpu0.sh'"
tmux new-session -d -s "$EXP5D_SESSION" "bash '$ROOT_DIR/scripts/experiment5d_rerun_gpu1.sh'"

echo "Started tmux sessions:"
tmux ls

echo ""
echo "Attach commands:"
echo "  tmux attach -t $EXP4A_SESSION"
echo "  tmux attach -t $EXP5D_SESSION"
echo ""
echo "Exp4A: 5 conditions (a/b/c/d/e) x 3 seeds x 2 models on GPU0"
echo "Exp5D: 2 models x 5 conditions x 50 sequences on GPU1"
