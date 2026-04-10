#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY="$ROOT/.venv/bin/python"
MODEL="llama-3.1-8b"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_phase2_stage3/${MODEL}_${TS}"
mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  echo "[$(date)] START $name" | tee -a "$LOG_DIR/_session.log"
  "$PY" -u "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date)] END $name" | tee -a "$LOG_DIR/_session.log"
}

echo "[$(date)] MODEL=$MODEL GPU=$CUDA_VISIBLE_DEVICES" | tee "$LOG_DIR/_session.log"

echo "[$(date)] Running 3P2-I + Idea2 (cross-lingual)" | tee -a "$LOG_DIR/_session.log"
run_step 3p2i_idea2 \
  experiment3/phase2/exp3p2i_tokenizer_corpus_invariance.py \
  --model "$MODEL" \
  --device cuda:0 \
  --seq-len 512 \
  --seq-per-corpus 16 \
  --adversarial-positions 200 \
  --opus-max-rows 20000 \
  --force-gate-override

echo "[$(date)] Running Idea4 structural ambiguity" | tee -a "$LOG_DIR/_session.log"
run_step idea4_structural_ambiguity \
  experiment3/phase2/idea4_structural_ambiguity.py \
  --model "$MODEL" \
  --device cuda:0 \
  --include-random-control \
  --bootstrap-samples 10000

echo "[$(date)] Running optional 3P2-C.2 stress test" | tee -a "$LOG_DIR/_session.log"
run_step 3p2c2_stress \
  experiment3/phase2/exp3p2c2_simultaneous_ablation.py \
  --model "$MODEL" \
  --device cuda:0 \
  --fractions 0,25,50,75 \
  --num-seeds 3 \
  --synthetic-count 100 \
  --batch-size-synth 8 \
  --ntp-count-per-seed 100 \
  --ntp-seq-len 512 \
  --batch-size-ntp 4 \
  --bootstrap-samples 10000

echo "[$(date)] COMPLETE Stage 3 for $MODEL" | tee -a "$LOG_DIR/_session.log"
