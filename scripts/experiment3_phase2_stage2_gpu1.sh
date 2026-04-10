#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/f004ndc/Kernel PE"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY="$ROOT/.venv/bin/python"
MODEL="olmo-2-7b"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_phase2_stage2/${MODEL}_${TS}"

mkdir -p "$LOG_DIR"

echo "[$(date)] Starting Stage 2 (3P2-G/H) for $MODEL on GPU1" | tee "$LOG_DIR/_session.log"
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

run_step 3p2g experiment3/phase2/exp3p2g_dose_response.py --model "$MODEL"
run_step 3p2h experiment3/phase2/exp3p2h_source_target_spec.py --model "$MODEL" --device cuda:0

echo "[$(date)] COMPLETE Stage 2 for $MODEL" | tee -a "$LOG_DIR/_session.log"
