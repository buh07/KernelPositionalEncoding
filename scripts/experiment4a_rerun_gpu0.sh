#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/experiment4/rerun_4a_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/experiment4/rerun_4a_latest"

exec > >(tee -a "$LOG_DIR/exp4a_gpu0.log") 2>&1

echo "[$(date)] Starting Experiment 4A rerun (post-fix: b vs d no longer duplicate) on GPU0"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME="/jumbo/lisp/f004ndc/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

# Archive previous 4A outputs so post-fix results are cleanly separated.
if [ -d "$ROOT_DIR/results/experiment4/exp4a_si_aware_lora" ]; then
  mv "$ROOT_DIR/results/experiment4/exp4a_si_aware_lora" \
     "$ROOT_DIR/results/experiment4/exp4a_si_aware_lora_stale_${TS}"
fi

# Run 4A for both models sequentially (5 conditions x 3 seeds each)
echo "[$(date)] Running 4A for llama-3.1-8b..."
.venv/bin/python -m experiment4.exp4a_si_aware_lora \
  --model llama-3.1-8b \
  --device cuda:0 \
  --seeds 0,1,2 \
  --conditions a_si_protecting_lora,b_uniform_lora,c_si_only_lora,d_full_qlora_baseline,e_si_amplified_lora

echo "[$(date)] Running 4A for olmo-2-7b..."
.venv/bin/python -m experiment4.exp4a_si_aware_lora \
  --model olmo-2-7b \
  --device cuda:0 \
  --seeds 0,1,2 \
  --conditions a_si_protecting_lora,b_uniform_lora,c_si_only_lora,d_full_qlora_baseline,e_si_amplified_lora

RC=$?
echo "[$(date)] Experiment 4A rerun finished with exit code $RC"
exit $RC
