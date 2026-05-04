#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/experiment5/rerun_fix_5a5c_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/experiment5/rerun_fix_5a5c_latest"

exec > >(tee -a "$LOG_DIR/exp5_fix_gpu1.log") 2>&1

echo "[$(date)] Starting Experiment 5 rerun for fixed paths (5A + 5C) on GPU1"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export HF_HOME="/jumbo/lisp/f004ndc/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

# Archive previous outputs to avoid mixing pre-fix and post-fix artifacts.
if [ -d "$ROOT_DIR/results/experiment5/exp5a_cross_tokenizer_si_profiling" ]; then
  mv "$ROOT_DIR/results/experiment5/exp5a_cross_tokenizer_si_profiling" \
     "$ROOT_DIR/results/experiment5/exp5a_cross_tokenizer_si_profiling_stale_${TS}"
fi
if [ -d "$ROOT_DIR/results/experiment5/exp5c_synthetic_tokenizer_perturbation" ]; then
  mv "$ROOT_DIR/results/experiment5/exp5c_synthetic_tokenizer_perturbation" \
     "$ROOT_DIR/results/experiment5/exp5c_synthetic_tokenizer_perturbation_stale_${TS}"
fi

echo "[$(date)] Running 5A cross-tokenizer profiling..."
.venv/bin/python -m experiment5.exp5a_cross_tokenizer_si_profiling \
  --device cuda:0

echo "[$(date)] Running 5C perturbation battery..."
.venv/bin/python -m experiment5.exp5c_synthetic_tokenizer_perturbation \
  --device cuda:0

RC=$?
echo "[$(date)] Experiment 5 fix rerun finished with exit code $RC"
exit $RC
