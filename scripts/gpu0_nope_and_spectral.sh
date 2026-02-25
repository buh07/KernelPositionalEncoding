#!/usr/bin/env bash
# GPU 0: TinyLlama-NoPE Track A + Track B, then spectral for all models
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
DEVICE="cuda"
RUN="python experiment1/run.py"

DATASETS=("wiki40b_en_pre2019" "codesearchnet_python_snapshot" "synthetic_random")
LENGTHS=("256" "1024")

echo "########## [GPU 0] TinyLlama-NoPE-1.1B Track A ##########"
for d in "${DATASETS[@]}"; do
    for l in "${LENGTHS[@]}"; do
        if [[ "$d" == "synthetic_random" && "$l" == "1024" ]]; then
            echo "Skipping Track A tinyllama-nope-1.1b/${d}/len_${l} (already done)"
            continue
        fi
        echo "===== Track A: tinyllama-nope-1.1b / ${d} / len=${l} ====="
        $RUN track-a --model "tinyllama-nope-1.1b" --dataset "$d" --seq-len "$l" --device "$DEVICE"
    done
done

echo "########## [GPU 0] TinyLlama-NoPE-1.1B Track B ##########"
for d in "${DATASETS[@]}"; do
    for l in "${LENGTHS[@]}"; do
        echo "===== Track B: tinyllama-nope-1.1b / ${d} / len=${l} ====="
        $RUN track-b --model "tinyllama-nope-1.1b" --dataset "$d" --seq-len "$l" --device "$DEVICE"
    done
done

echo "########## [GPU 0] Spectral Analysis (all models) ##########"
ALL_MODELS=("gpt2-small" "gpt2-medium" "olmo-1b" "llama-3.2-1b" "tinyllama-1.1b" "tinyllama-nope-1.1b")
for m in "${ALL_MODELS[@]}"; do
    for d in "${DATASETS[@]}"; do
        for l in "${LENGTHS[@]}"; do
            echo "===== Spectral: ${m} / ${d} / len=${l} ====="
            $RUN spectral --model "$m" --dataset "$d" --seq-len "$l"
        done
    done
done

echo ""
echo "=========================================="
echo "  [GPU 0] NoPE + SPECTRAL COMPLETE"
echo "=========================================="
