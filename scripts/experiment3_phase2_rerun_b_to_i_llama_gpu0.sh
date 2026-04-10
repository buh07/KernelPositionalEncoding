#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY="$ROOT/.venv/bin/python"
MODEL="llama-3.1-8b"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_phase2_rerun/${MODEL}_b_to_i_${TS}"
mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  echo "[$(date)] START $name" | tee -a "$LOG_DIR/_session.log"
  "$PY" -u "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date)] END $name" | tee -a "$LOG_DIR/_session.log"
}

echo "[$(date)] MODEL=$MODEL GPU=$CUDA_VISIBLE_DEVICES" | tee "$LOG_DIR/_session.log"

echo "[$(date)] Step 1: robust 3P2-B full rerun (Llama)" | tee -a "$LOG_DIR/_session.log"
run_step 3p2b_full_rerun \
  experiment3/phase2/exp3p2b_trivial_feature_control.py \
  --model "$MODEL" \
  --mode full \
  --device cuda:0 \
  --num-sequences 24 \
  --seq-len 512 \
  --top-k-dims 32 \
  --synthetic-target-per-cell 600 \
  --seed 17 \
  --output-root results/experiment3_phase2/exp3p2b_trivial_feature_control

echo "[$(date)] Step 2: strict-gate 3P2-I rerun (no override)" | tee -a "$LOG_DIR/_session.log"
run_step 3p2i_strict_after_b \
  experiment3/phase2/exp3p2i_tokenizer_corpus_invariance.py \
  --model "$MODEL" \
  --device cuda:0 \
  --seq-len 512 \
  --seq-per-corpus 16 \
  --adversarial-positions 200 \
  --opus-max-rows 20000 \
  --output-root results/experiment3_phase2/exp3p2i_tokenizer_corpus_rerun_after_b

echo "[$(date)] COMPLETE $MODEL B->I rerun pipeline" | tee -a "$LOG_DIR/_session.log"
