#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/f004ndc/Kernel PE"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY="$ROOT/.venv/bin/python"
MODEL="llama-3.1-8b"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_refresh/${MODEL}_${TS}"
PAIR_EFFECTS_PATH="$ROOT/results/experiment2/quick/quick_pair_expand_pivot_20260314_0043_v2/reports/pair_unified/pair_effects_merged.parquet"

mkdir -p "$LOG_DIR"

echo "[$(date)] Starting Experiment 3 hard-refresh for $MODEL on GPU0" | tee "$LOG_DIR/_session.log"
echo "[$(date)] LOG_DIR=$LOG_DIR" | tee -a "$LOG_DIR/_session.log"
echo "[$(date)] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" | tee -a "$LOG_DIR/_session.log"

run_step() {
  local name="$1"
  shift
  echo "[$(date)] START $name" | tee -a "$LOG_DIR/_session.log"
  "$PY" -u "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date)] END $name" | tee -a "$LOG_DIR/_session.log"
}

if [[ ! -f "$PAIR_EFFECTS_PATH" ]]; then
  echo "[$(date)] ERROR missing pair effects artifact at $PAIR_EFFECTS_PATH" | tee -a "$LOG_DIR/_session.log"
  exit 1
fi

run_step theory1 experiment3/theory1_si_circuits.py --model "$MODEL" --device cuda:0
run_step theory3 experiment3/theory3_crossterm_correlation.py --model "$MODEL" --device cuda:0 --force-recompute-gap-kernels --force-recompute-total-energy
run_step theory5 experiment3/theory5_subword_ablation.py --model "$MODEL" --device cuda:0
run_step theory5b experiment3/theory5b_boundary_detection.py --model "$MODEL" --device cuda:0
run_step theory7b experiment3/theory7b_activation_patching.py --model "$MODEL" --device cuda:0
run_step theory10 experiment3/theory10_feeder_specificity.py --model "$MODEL" --device cuda:0

echo "[$(date)] COMPLETE all steps for $MODEL" | tee -a "$LOG_DIR/_session.log"
