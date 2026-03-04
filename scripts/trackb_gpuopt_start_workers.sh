#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch2/f004ndc/Kernel PE"
cd "$ROOT"

QUEUE_FILE="logs/trackb/trackb_gpuopt_queue.tsv"
bash scripts/trackb_gpuopt_queue_build.sh "$QUEUE_FILE"

tmux kill-session -t trackb_gpuopt_worker_g4 2>/dev/null || true
tmux kill-session -t trackb_gpuopt_worker_g5 2>/dev/null || true

tmux new-session -d -s trackb_gpuopt_worker_g4 "bash scripts/trackb_gpuopt_queue_worker.sh 4 $QUEUE_FILE"
tmux new-session -d -s trackb_gpuopt_worker_g5 "bash scripts/trackb_gpuopt_queue_worker.sh 5 $QUEUE_FILE"

echo "Started workers:"
echo "  trackb_gpuopt_worker_g4 (GPU 4)"
echo "  trackb_gpuopt_worker_g5 (GPU 5)"
echo "Queue: $QUEUE_FILE"
