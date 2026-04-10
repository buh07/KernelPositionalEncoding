#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY="$ROOT/.venv/bin/python"
MODEL="mistral-7b-v0.1"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_phase2_rerun/${MODEL}_3p2d1_clean_${TS}"
mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  echo "[$(date)] START $name" | tee -a "$LOG_DIR/_session.log"
  "$PY" -u "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date)] END $name" | tee -a "$LOG_DIR/_session.log"
}

echo "[$(date)] MODEL=$MODEL GPU=$CUDA_VISIBLE_DEVICES" | tee "$LOG_DIR/_session.log"
run_step 3p2d_d1_clean \
  experiment3/phase2/exp3p2d_architecture_tiebreaker.py \
  --mode d1 \
  --device cuda:0 \
  --num-pairs 30 \
  --ensure-direct-prereqs

echo "[$(date)] COMPLETE $MODEL 3P2-D.1 clean" | tee -a "$LOG_DIR/_session.log"
