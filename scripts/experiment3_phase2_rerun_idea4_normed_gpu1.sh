#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY="$ROOT/.venv/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_phase2_rerun/idea4_normed_${TS}"
mkdir -p "$LOG_DIR"

ALLOWLIST="$ROOT/results/experiment3_phase2/idea4_structural_ambiguity/shared_normed_variant_allowlist.txt"

run_step() {
  local name="$1"
  shift
  echo "[$(date)] START $name" | tee -a "$LOG_DIR/_session.log"
  "$PY" -u "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date)] END $name" | tee -a "$LOG_DIR/_session.log"
}

echo "[$(date)] Idea4 normed rerun GPU=$CUDA_VISIBLE_DEVICES" | tee "$LOG_DIR/_session.log"
run_step idea4_normed_rerun \
  experiment3/phase2/idea4_structural_ambiguity.py \
  --model all \
  --device-map "llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0" \
  --include-random-control \
  --bootstrap-samples 10000 \
  --variant-allowlist "$ALLOWLIST" \
  --output-root results/experiment3_phase2/idea4_structural_ambiguity_normed

echo "[$(date)] COMPLETE Idea4 normed rerun" | tee -a "$LOG_DIR/_session.log"
