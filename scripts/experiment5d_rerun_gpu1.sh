#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/experiment5/rerun_5d_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/experiment5/rerun_5d_latest"

exec > >(tee -a "$LOG_DIR/exp5d_gpu1.log") 2>&1

echo "[$(date)] Starting Experiment 5D rerun (50 sequences, per-condition cleanup) on GPU1"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export HF_HOME="/jumbo/lisp/f004ndc/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

# Clear previous empty 5D output directory
rm -rf "$ROOT_DIR/results/experiment5/exp5d_distribution_shift_fragility"

.venv/bin/python -m experiment5.exp5d_distribution_shift_fragility \
  --device cuda:0 \
  --num-sequences 50 \
  --seq-len 256 \
  --seed 0

RC=$?
echo "[$(date)] Experiment 5D rerun finished with exit code $RC"
exit $RC
