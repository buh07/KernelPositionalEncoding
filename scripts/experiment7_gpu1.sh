#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/experiment7/run_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/experiment7/run_latest"

exec > >(tee -a "$LOG_DIR/exp7_gpu1.log") 2>&1

echo "[$(date)] Starting Experiment 7 (A+B core) on GPU1"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export HF_HOME="/jumbo/lisp/f004ndc/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

.venv/bin/python -m experiment7.run all --device cuda:0
RC=$?
echo "[$(date)] Experiment 7 finished with exit code $RC"
exit "$RC"

