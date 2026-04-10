#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/f004ndc/Kernel PE"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0
PY="$ROOT/.venv/bin/python"
MODEL="llama-3.1-8b"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_repair/${MODEL}_${TS}"
mkdir -p "$LOG_DIR"
PAIR_EFFECTS_PATH="$ROOT/results/experiment2/quick/quick_pair_expand_pivot_20260314_0043_v2/reports/pair_unified/pair_effects_merged.parquet"

echo "[$(date)] Starting Experiment 3 repair reruns (resume from theory3) for $MODEL on GPU0" | tee "$LOG_DIR/_session.log"

run_step() {
  local name="$1"
  shift
  echo "[$(date)] START $name" | tee -a "$LOG_DIR/_session.log"
  "$PY" -u "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date)] END $name" | tee -a "$LOG_DIR/_session.log"
}

if [[ -f "$PAIR_EFFECTS_PATH" ]]; then
  run_step theory3 experiment3/theory3_crossterm_correlation.py --model "$MODEL" --device cuda:0
else
  echo "[$(date)] SKIP theory3 (missing pair effects artifact at $PAIR_EFFECTS_PATH)" | tee -a "$LOG_DIR/_session.log"
fi
run_step theory5 experiment3/theory5_subword_ablation.py --model "$MODEL" --device cuda:0
run_step theory5b experiment3/theory5b_boundary_detection.py --model "$MODEL" --device cuda:0
run_step theory7b experiment3/theory7b_activation_patching.py --model "$MODEL" --device cuda:0
run_step theory10 experiment3/theory10_feeder_specificity.py --model "$MODEL" --device cuda:0

echo "[$(date)] COMPLETE all steps for $MODEL" | tee -a "$LOG_DIR/_session.log"
