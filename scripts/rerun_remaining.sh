#!/usr/bin/env bash
# Resume reruns: TinyLlama-1.1B only (NoPE + spectral handled on GPU 0)
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4
DEVICE="cuda"
RUN="python experiment1/run.py"

DATASETS=("wiki40b_en_pre2019" "codesearchnet_python_snapshot" "synthetic_random")
LENGTHS=("256" "1024")

# ── TinyLlama-1.1B (RoPE + RMSNorm) ────────────────────────────────
# Track A: wiki40b/256 already done in previous partial run; need rest
echo "########## TinyLlama-1.1B ##########"
for d in "${DATASETS[@]}"; do
    for l in "${LENGTHS[@]}"; do
        if [[ "$d" == "wiki40b_en_pre2019" && "$l" == "256" ]]; then
            echo "Skipping Track A tinyllama-1.1b/${d}/len_${l} (already rerun)"
            continue
        fi
        echo "===== Track A: tinyllama-1.1b / ${d} / len=${l} ====="
        $RUN track-a --model "tinyllama-1.1b" --dataset "$d" --seq-len "$l" --device "$DEVICE"
    done
done
for d in "${DATASETS[@]}"; do
    for l in "${LENGTHS[@]}"; do
        echo "===== Track B: tinyllama-1.1b / ${d} / len=${l} ====="
        $RUN track-b --model "tinyllama-1.1b" --dataset "$d" --seq-len "$l" --device "$DEVICE"
    done
done

echo ""
echo "=========================================="
echo "  TinyLlama-1.1B RERUNS COMPLETE"
echo "=========================================="
