#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY="$ROOT/.venv/bin/python"
MODEL="olmo-2-7b"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_phase2_rerun/${MODEL}_3p2b_full_strict_${TS}"
mkdir -p "$LOG_DIR"

echo "[$(date)] MODEL=$MODEL GPU=$CUDA_VISIBLE_DEVICES" | tee "$LOG_DIR/_session.log"
echo "[$(date)] START 3p2b_full_strict" | tee -a "$LOG_DIR/_session.log"

"$PY" -u experiment3/phase2/exp3p2b_trivial_feature_control.py \
  --model "$MODEL" \
  --mode full \
  --device cuda:0 \
  --num-sequences 24 \
  --seq-len 512 \
  --top-k-dims 32 \
  --synthetic-target-per-cell 600 \
  --seed 17 \
  --output-root results/experiment3_phase2/exp3p2b_trivial_feature_control \
  2>&1 | tee "$LOG_DIR/3p2b_full_strict.log"

echo "[$(date)] END 3p2b_full_strict" | tee -a "$LOG_DIR/_session.log"
echo "[$(date)] COMPLETE ${MODEL} 3P2-B strict rerun" | tee -a "$LOG_DIR/_session.log"
