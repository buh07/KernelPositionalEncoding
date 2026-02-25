#!/usr/bin/env bash
# GPU 4: Boundary analysis for TinyLlama-1.1B and TinyLlama-NoPE-1.1B
# Original GPU 4 variant. For the Feb 24, 2026 redistribution run, use
# gpu5_boundary_part2_tinyllama.sh on GPU 5 instead.
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
