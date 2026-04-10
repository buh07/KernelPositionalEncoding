#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <gpu_index> <model_name> [num_sequences] [num_seed_shards] [long_span_min_distance] [min_regime_sample_count] [batch_size]"
  exit 1
fi

GPU="$1"
MODEL="$2"
NUM_SEQUENCES="${3:-160}"
NUM_SEED_SHARDS="${4:-16}"
LONG_SPAN_MIN_DISTANCE="${5:-64}"
MIN_REGIME_SAMPLE_COUNT="${6:-256}"
BATCH_SIZE="${7:-4}"

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY="$ROOT/.venv/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_phase2_j_longspan/${MODEL}_gpu${GPU}_${TS}"
mkdir -p "$LOG_DIR"

echo "[$(date)] model=$MODEL gpu=$GPU num_sequences=$NUM_SEQUENCES shards=$NUM_SEED_SHARDS long_span_min_distance=$LONG_SPAN_MIN_DISTANCE batch_size=$BATCH_SIZE" | tee -a "$LOG_DIR/_session.log"

"$PY" -u experiment3/phase2/exp3p2j_conditional_regimes.py \
  --model "$MODEL" \
  --device cuda:0 \
  --num-sequences "$NUM_SEQUENCES" \
  --seq-len 512 \
  --batch-size "$BATCH_SIZE" \
  --num-seed-shards "$NUM_SEED_SHARDS" \
  --long-span-min-distance "$LONG_SPAN_MIN_DISTANCE" \
  --min-regime-sample-count "$MIN_REGIME_SAMPLE_COUNT" \
  --a-output-root results/experiment3_phase2/exp3p2a_positional_broadcast \
  --output-root results/experiment3_phase2/exp3p2j_conditional_regimes_longspan_repair \
  2>&1 | tee "$LOG_DIR/3p2j_longspan.log"

echo "[$(date)] COMPLETE 3P2-J long-span repair for $MODEL" | tee -a "$LOG_DIR/_session.log"
