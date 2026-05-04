#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/experiment8/run_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/experiment8/run_latest"

exec > >(tee -a "$LOG_DIR/exp8_gpu0.log") 2>&1

GPU_ID="${GPU_ID:-2}"
echo "[$(date)] Starting Experiment 8 on GPU${GPU_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export HF_HOME="/jumbo/lisp/f004ndc/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

# --- 8A: SIBAS for all 4 models ---
for MODEL in gpt2-small tinyllama-1.1b llama-3.1-8b olmo-2-7b; do
    echo ""
    echo "================================================================"
    echo "[$(date)] Running 8A (SIBAS) for $MODEL"
    echo "================================================================"
    .venv/bin/python -m experiment8.exp8a_sibas \
        --model "$MODEL" --device cuda:0
done

# --- 8B: Pruning for 7B+ models ---
for MODEL in llama-3.1-8b olmo-2-7b; do
    echo ""
    echo "================================================================"
    echo "[$(date)] Running 8B (Pruning) for $MODEL"
    echo "================================================================"
    .venv/bin/python -m experiment8.exp8b_pruning \
        --model "$MODEL" --device cuda:0
done

echo "[$(date)] Experiment 8 finished."
