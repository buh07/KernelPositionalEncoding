#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/f004ndc/Kernel PE"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY="$ROOT/.venv/bin/python"
MODEL="llama-3.1-8b"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_phase2_stage1/${MODEL}_${TS}"

mkdir -p "$LOG_DIR"

echo "[$(date)] Starting Stage 1 (3P2-E/F/B) for $MODEL on GPU0" | tee "$LOG_DIR/_session.log"
echo "[$(date)] LOG_DIR=$LOG_DIR" | tee -a "$LOG_DIR/_session.log"
echo "[$(date)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG_DIR/_session.log"
echo "[$(date)] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" | tee -a "$LOG_DIR/_session.log"

run_step() {
  local name="$1"
  shift
  echo "[$(date)] START $name" | tee -a "$LOG_DIR/_session.log"
  "$PY" -u "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date)] END $name" | tee -a "$LOG_DIR/_session.log"
}

run_step 3p2e experiment3/phase2/exp3p2e_t5_t5b_reconciliation.py --model "$MODEL"
run_step 3p2f experiment3/phase2/exp3p2f_proxy_decomposition.py --model "$MODEL"
run_step 3p2b experiment3/phase2/exp3p2b_trivial_feature_control.py --model "$MODEL" --device cuda:0

echo "[$(date)] COMPLETE Stage 1 for $MODEL" | tee -a "$LOG_DIR/_session.log"
