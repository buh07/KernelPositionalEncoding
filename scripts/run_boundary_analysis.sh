#!/usr/bin/env bash
# GPU 4: Boundary analysis for TinyLlama-1.1B and TinyLlama-NoPE-1.1B
# Run AFTER rerun_remaining.sh and gpu0_nope_and_spectral.sh finish
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4
DEVICE="cuda"
RUN="python experiment1/run.py"

MODELS=("tinyllama-1.1b" "tinyllama-nope-1.1b")
DATASETS=("wiki40b_en_pre2019" "codesearchnet_python_snapshot" "synthetic_random")
LENGTHS=("256" "1024")

echo "########## [GPU 4] Boundary Analysis (TinyLlama models) ##########"
for m in "${MODELS[@]}"; do
    for d in "${DATASETS[@]}"; do
        for l in "${LENGTHS[@]}"; do
            echo "===== Boundary: ${m} / ${d} / len=${l} ====="
            $RUN boundary --model "$m" --dataset "$d" --seq-len "$l" --device "$DEVICE"
        done
    done
done

echo ""
echo "=========================================="
echo "  [GPU 4] BOUNDARY PART 2 COMPLETE"
echo "=========================================="
