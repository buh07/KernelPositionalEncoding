#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY="$ROOT/.venv/bin/python"
MODEL="llama-3.1-8b"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_phase2_stage4/${MODEL}_${TS}"
mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  echo "[$(date)] START $name" | tee -a "$LOG_DIR/_session.log"
  "$PY" -u "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date)] END $name" | tee -a "$LOG_DIR/_session.log"
}

echo "[$(date)] MODEL=$MODEL GPU=$CUDA_VISIBLE_DEVICES" | tee "$LOG_DIR/_session.log"

echo "[$(date)] Preparing Stage4 A.2 tokenized data" | tee -a "$LOG_DIR/_session.log"
run_step prepare_a2_data \
  scripts/experiment3_phase2_stage4_prepare_a2_data.py \
  --model "$MODEL" \
  --probe-seq-len 512 \
  --probe-train-sequences 500 \
  --probe-eval-sequences 100 \
  --extra-eval-buffer 16 \
  --safety-margin 64 \
  --min-generate 500 \
  --tokenized-seq-len 1024

echo "[$(date)] Running 3P2-A" | tee -a "$LOG_DIR/_session.log"
run_step 3p2a \
  experiment3/phase2/exp3p2a_positional_broadcast.py \
  --model "$MODEL" \
  --device cuda:0 \
  --num-seeds 5 \
  --synthetic-count 100 \
  --choice-count 80 \
  --seq-len 512 \
  --batch-size-synth 8 \
  --probe-train-sequences 500 \
  --probe-eval-sequences 100 \
  --probe-seq-len 512 \
  --probe-positions-per-seq 64 \
  --probe-batch-size 4 \
  --pos-backend nltk \
  --allow-nltk-download \
  --bootstrap-samples 5000 \
  --transfer-pairs 50 \
  --output-root results/experiment3_phase2/exp3p2a_positional_broadcast

echo "[$(date)] Running 3P2-J" | tee -a "$LOG_DIR/_session.log"
run_step 3p2j \
  experiment3/phase2/exp3p2j_conditional_regimes.py \
  --model "$MODEL" \
  --device cuda:0 \
  --num-sequences 80 \
  --seq-len 512 \
  --batch-size 4 \
  --num-seed-shards 5 \
  --a-output-root results/experiment3_phase2/exp3p2a_positional_broadcast \
  --output-root results/experiment3_phase2/exp3p2j_conditional_regimes

echo "[$(date)] Running Idea 6" | tee -a "$LOG_DIR/_session.log"
run_step idea6 \
  experiment3/phase2/idea6_math_si_channels.py \
  --model "$MODEL" \
  --device cuda:0 \
  --count-per-task 120 \
  --num-seeds 5 \
  --output-root results/experiment3_phase2/idea6_math_si_channels

ALLOWLIST="$ROOT/experiment3/phase2/idea4_stage4_balanced_variant_allowlist.txt"
if [[ ! -f "$ALLOWLIST" ]]; then
  echo "[$(date)] ERROR missing Idea4 allowlist: $ALLOWLIST" | tee -a "$LOG_DIR/_session.log"
  exit 1
fi

echo "[$(date)] Running Idea 4 (normed)" | tee -a "$LOG_DIR/_session.log"
run_step idea4_normed \
  experiment3/phase2/idea4_structural_ambiguity.py \
  --model "$MODEL" \
  --device cuda:0 \
  --include-random-control \
  --bootstrap-samples 10000 \
  --variant-allowlist "$ALLOWLIST" \
  --fail-on-quality-gate \
  --output-root results/experiment3_phase2/idea4_structural_ambiguity_normed_stage4

echo "[$(date)] COMPLETE Stage 4 for $MODEL" | tee -a "$LOG_DIR/_session.log"
