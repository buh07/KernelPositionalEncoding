#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[$(date)] Waiting for exp8b_gpu1 tmux session to finish..."
while tmux has-session -t exp8b_gpu1 2>/dev/null; do
    # Check if the session still has a running process
    PANE_PID=$(tmux list-panes -t exp8b_gpu1 -F '#{pane_pid}' 2>/dev/null | head -1)
    if [ -z "$PANE_PID" ]; then
        break
    fi
    # Check if the shell's child (the actual script) is still running
    CHILDREN=$(pgrep -P "$PANE_PID" 2>/dev/null || true)
    if [ -z "$CHILDREN" ]; then
        echo "[$(date)] exp8b_gpu1 shell has no children — 8B likely done."
        break
    fi
    sleep 60
    echo "[$(date)] Still waiting for 8B to finish..."
done

echo "[$(date)] 8B done (or session gone). Launching 8A v2 large models on GPU 1."
GPU_ID=1 bash scripts/exp8a_v2_large.sh
