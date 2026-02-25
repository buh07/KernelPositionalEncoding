#!/usr/bin/env bash
# GPU 0 follow-up: TinyLlama-1.1B Track B tail requeued from killed GPU 4 session.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
DEVICE="cuda"
RUN="python experiment1/run.py"

echo "########## [GPU 0] TinyLlama-1.1B Track B Tail (requeued from GPU 4) ##########"

echo "===== Track B: tinyllama-1.1b / codesearchnet_python_snapshot / len=1024 ====="
$RUN track-b --model "tinyllama-1.1b" --dataset "codesearchnet_python_snapshot" --seq-len "1024" --device "$DEVICE"

echo "===== Track B: tinyllama-1.1b / synthetic_random / len=256 ====="
$RUN track-b --model "tinyllama-1.1b" --dataset "synthetic_random" --seq-len "256" --device "$DEVICE"

echo "===== Track B: tinyllama-1.1b / synthetic_random / len=1024 ====="
$RUN track-b --model "tinyllama-1.1b" --dataset "synthetic_random" --seq-len "1024" --device "$DEVICE"

echo ""
echo "=========================================="
echo "  [GPU 0] TinyLlama Track B Tail COMPLETE"
echo "=========================================="
