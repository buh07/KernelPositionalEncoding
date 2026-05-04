#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-3}"
LOG_DIR="$ROOT_DIR/logs/experiment8/run_8a_v2"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export HF_HOME="/jumbo/lisp/f004ndc/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

echo "[$(date)] Starting Exp8A v2 (BDI/BAC/KOA) — small models on GPU${GPU_ID}"

for MODEL in gpt2-small tinyllama-1.1b; do
    echo ""
    echo "================================================================"
    echo "[$(date)] Running 8A v2 for $MODEL"
    echo "================================================================"
    .venv/bin/python -m experiment8.exp8a_sibas \
        --model "$MODEL" --device cuda:0 2>&1 | tee "$LOG_DIR/exp8a_${MODEL}.log"
done

echo "[$(date)] Exp8A v2 small models finished."
