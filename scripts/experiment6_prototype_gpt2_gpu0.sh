#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/experiment6/prototype_gpt2_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/experiment6/prototype_latest"

exec > >(tee -a "$LOG_DIR/exp6_proto_gpu0.log") 2>&1

echo "[$(date)] Starting Experiment 6 prototype on gpt2-small (GPU0)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME="/jumbo/lisp/f004ndc/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

MODEL="gpt2-small"
SEEDS="0,1,2"
MAX_STEPS=120
MATH_PER_TASK=80

echo "[$(date)] 6A (Gradient Routing) ..."
.venv/bin/python -m experiment6.exp6a_gradient_routing \
  --model "$MODEL" --device cuda:0 --seeds "$SEEDS" \
  --max-steps "$MAX_STEPS" --math-per-task "$MATH_PER_TASK"

echo "[$(date)] 6B (Anti-Localization) ..."
.venv/bin/python -m experiment6.exp6b_anti_localization \
  --model "$MODEL" --device cuda:0 --seeds "$SEEDS" \
  --max-steps "$MAX_STEPS" --math-per-task "$MATH_PER_TASK" \
  --lambda-anti-loc 0.03

echo "[$(date)] 6C (Distillation) ..."
.venv/bin/python -m experiment6.exp6c_distillation \
  --model "$MODEL" --device cuda:0 --seeds "$SEEDS" \
  --max-steps "$MAX_STEPS" --math-per-task "$MATH_PER_TASK"

echo "[$(date)] 6D (Contrastive) ..."
.venv/bin/python -m experiment6.exp6d_contrastive \
  --model "$MODEL" --device cuda:0 --seeds "$SEEDS" \
  --max-steps "$MAX_STEPS" --math-per-task "$MATH_PER_TASK" \
  --lambda-contrast 0.03

echo "[$(date)] 6E (Tokenizer formatting) ..."
.venv/bin/python -m experiment6.exp6e_tokenizer \
  --model "$MODEL" --device cuda:0 --seed 0

RC=$?
echo "[$(date)] Experiment 6 prototype finished with exit code $RC"
exit "$RC"
