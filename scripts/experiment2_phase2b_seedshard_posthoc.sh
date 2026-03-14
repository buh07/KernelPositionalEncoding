#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?usage: scripts/experiment2_phase2b_seedshard_posthoc.sh <run_id> [output_root] [gpu_csv]}"
OUT_ROOT="${2:-results/experiment2}"
GPU_CSV="${3:-${POSTHOC_GPUS:-0,1,2}}"
LOG_DIR="logs/experiment2"
mkdir -p "$LOG_DIR"

PHASE_ROOT="${OUT_ROOT}/phase2b/${RUN_ID}"
if [[ ! -d "$PHASE_ROOT" ]]; then
  echo "Missing Phase 2B run root: $PHASE_ROOT" >&2
  exit 1
fi

BATCH_SIZE_SYNTH="${BATCH_SIZE_SYNTH:-16}"
BATCH_SIZE_TIER1="${BATCH_SIZE_TIER1:-16}"
STRICT_POSTHOC="${STRICT_POSTHOC:-1}"
MODEL_FILTER="${MODEL_FILTER:-}"

STRICT_ARG=()
if [[ "$STRICT_POSTHOC" == "1" ]]; then
  STRICT_ARG+=(--strict-posthoc)
fi
MODEL_FILTER_ARG=()
if [[ -n "$MODEL_FILTER" ]]; then
  MODEL_FILTER_ARG+=(--model-filter "$MODEL_FILTER")
fi

IFS=',' read -r -a RAW_GPUS <<< "$GPU_CSV"
GPUS=()
for g in "${RAW_GPUS[@]}"; do
  g="$(echo "$g" | xargs)"
  [[ -z "$g" ]] && continue
  if [[ ! "$g" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id in list: '$g'" >&2
    exit 1
  fi
  GPUS+=("$g")
done
if (( ${#GPUS[@]} == 0 )); then
  echo "No GPUs provided for posthoc launch (GPU_CSV='$GPU_CSV')." >&2
  exit 1
fi

SEED_MIN=0
SEED_MAX=6
TOTAL_SEEDS=$((SEED_MAX - SEED_MIN + 1))
WORKERS=${#GPUS[@]}
if (( WORKERS > TOTAL_SEEDS )); then
  WORKERS=$TOTAL_SEEDS
fi
CHUNK_SIZE=$(( (TOTAL_SEEDS + WORKERS - 1) / WORKERS ))

for ((idx=0; idx<WORKERS; idx++)); do
  gpu="${GPUS[$idx]}"
  sess="exp2b_posthoc_g${gpu}"
  tmux kill-session -t "$sess" 2>/dev/null || true
done

launched=0
for ((idx=0; idx<WORKERS; idx++)); do
  gpu="${GPUS[$idx]}"
  start=$((SEED_MIN + idx * CHUNK_SIZE))
  end=$((start + CHUNK_SIZE - 1))
  if (( start > SEED_MAX )); then
    continue
  fi
  if (( end > SEED_MAX )); then
    end=$SEED_MAX
  fi

  sess="exp2b_posthoc_g${gpu}"
  log_file="$LOG_DIR/${RUN_ID}_g${gpu}_posthoc_phase2b_seed${start}${end}.log"
  cmd=(
    "cd \"$ROOT_DIR\" && CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 .venv/bin/python experiment2/run.py"
    "--mode posthoc-centered"
    "--phase phase2b"
    "--run-id \"$RUN_ID\""
    "--device cuda"
    "--output-root \"$OUT_ROOT\""
    "--batch-size-synth \"$BATCH_SIZE_SYNTH\""
    "--batch-size-tier1 \"$BATCH_SIZE_TIER1\""
    "--seed-start $start"
    "--seed-end $end"
    "--print-summary"
  )
  if (( ${#STRICT_ARG[@]} > 0 )); then
    cmd+=("${STRICT_ARG[@]}")
  fi
  if (( ${#MODEL_FILTER_ARG[@]} > 0 )); then
    cmd+=("${MODEL_FILTER_ARG[@]}")
  fi
  tmux new-session -d -s "$sess" "$(printf '%s ' "${cmd[@]}") |& tee \"$log_file\""
  launched=$((launched + 1))
done

echo "Launched Phase 2B seed-sharded posthoc-centered workers=${launched} gpus=${GPU_CSV} run_id=${RUN_ID}"
