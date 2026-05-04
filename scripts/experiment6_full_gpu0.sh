#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/experiment6/run_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/experiment6/run_latest"

exec > >(tee -a "$LOG_DIR/exp6_gpu0.log") 2>&1

echo "[$(date)] Starting Experiment 6 (all sub-experiments) on GPU0"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME="/jumbo/lisp/f004ndc/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

MODELS=("llama-3.1-8b" "olmo-2-7b")
SEEDS="0,1,2"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "================================================================"
    echo "[$(date)] Running 6A (Gradient Routing) for $MODEL"
    echo "================================================================"
    .venv/bin/python -m experiment6.exp6a_gradient_routing \
        --model "$MODEL" --device cuda:0 --seeds "$SEEDS"

    echo "[$(date)] Running 6B (Anti-Localization) for $MODEL"
    .venv/bin/python -m experiment6.exp6b_anti_localization \
        --model "$MODEL" --device cuda:0 --seeds "$SEEDS"

    echo "[$(date)] Running 6C (Distillation) for $MODEL"
    .venv/bin/python -m experiment6.exp6c_distillation \
        --model "$MODEL" --device cuda:0 --seeds "$SEEDS"

    echo "[$(date)] Running 6D (Contrastive) for $MODEL"
    .venv/bin/python -m experiment6.exp6d_contrastive \
        --model "$MODEL" --device cuda:0 --seeds "$SEEDS"

    echo "[$(date)] Running 6E (Tokenizer) for $MODEL"
    .venv/bin/python -m experiment6.exp6e_tokenizer \
        --model "$MODEL" --device cuda:0 --seed 0
done

RC=$?
echo "[$(date)] Experiment 6 finished with exit code $RC"
exit $RC
