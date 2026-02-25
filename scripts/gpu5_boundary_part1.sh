#!/usr/bin/env bash
# GPU 5: Boundary analysis for GPT-2 small, GPT-2 medium, OLMo-1B, LLaMA-3.2-1B
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5
DEVICE="cuda"
RUN="python experiment1/run.py"

MODELS=("gpt2-small" "gpt2-medium" "olmo-1b" "llama-3.2-1b")
DATASETS=("wiki40b_en_pre2019" "codesearchnet_python_snapshot" "synthetic_random")
LENGTHS=("256" "1024")

echo "########## [GPU 5] Boundary Analysis (4 models) ##########"
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
echo "  [GPU 5] BOUNDARY PART 1 COMPLETE"
echo "=========================================="
