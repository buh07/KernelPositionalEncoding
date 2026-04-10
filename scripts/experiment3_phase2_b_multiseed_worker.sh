#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <gpu_index> <model_name> <seed_csv> [num_sequences] [synthetic_target_per_cell]"
  exit 1
fi

GPU="$1"
MODEL="$2"
SEEDS_CSV="$3"
NUM_SEQUENCES="${4:-32}"
SYNTH_TARGET="${5:-600}"

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PY="$ROOT/.venv/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/experiment3_phase2_multiseed_b/${MODEL}_gpu${GPU}_${TS}"
mkdir -p "$LOG_DIR"

echo "[$(date)] model=$MODEL gpu=$GPU seeds=$SEEDS_CSV num_sequences=$NUM_SEQUENCES synthetic_target=$SYNTH_TARGET" | tee -a "$LOG_DIR/_session.log"

IFS=',' read -r -a SEED_ARR <<< "$SEEDS_CSV"
for seed in "${SEED_ARR[@]}"; do
  seed_trim="$(echo "$seed" | xargs)"
  [[ -z "$seed_trim" ]] && continue
  out_root="results/experiment3_phase2/exp3p2b_trivial_feature_control_multiseed/seed${seed_trim}"
  echo "[$(date)] START seed=${seed_trim} out_root=${out_root}" | tee -a "$LOG_DIR/_session.log"
  "$PY" -u experiment3/phase2/exp3p2b_trivial_feature_control.py \
    --model "$MODEL" \
    --mode full \
    --device cuda:0 \
    --num-sequences "$NUM_SEQUENCES" \
    --seq-len 512 \
    --top-k-dims 16 \
    --synthetic-target-per-cell "$SYNTH_TARGET" \
    --seed "$seed_trim" \
    --output-root "$out_root" \
    2>&1 | tee "$LOG_DIR/seed_${seed_trim}.log"
  echo "[$(date)] END seed=${seed_trim}" | tee -a "$LOG_DIR/_session.log"
done

echo "[$(date)] COMPLETE multiseed worker model=$MODEL gpu=$GPU" | tee -a "$LOG_DIR/_session.log"
