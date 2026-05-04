#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/experiment5/full_program_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/experiment5/full_program_latest"

exec > >(tee -a "$LOG_DIR/exp5_gpu1.log") 2>&1

echo "[$(date)] Starting Experiment 5 full program on GPU1"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export HF_HOME="/jumbo/lisp/f004ndc/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

.venv/bin/python -m experiment5.run all --device cuda:0

RC=$?
echo "[$(date)] Experiment 5 full program finished with exit code $RC"
exit $RC
