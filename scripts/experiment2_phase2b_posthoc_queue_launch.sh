#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?usage: scripts/experiment2_phase2b_posthoc_queue_launch.sh <run_id> [output_root] [queue_file] [gpu_csv]}"
OUT_ROOT="${2:-results/experiment2}"
QUEUE_FILE="${3:-logs/experiment2/${RUN_ID}_phase2b_posthoc_queue.tsv}"
GPU_CSV="${4:-${POSTHOC_GPUS:-0,1,2}}"

BATCH_SIZE_SYNTH="${BATCH_SIZE_SYNTH:-24}"
BATCH_SIZE_TIER1="${BATCH_SIZE_TIER1:-24}"
STRICT_POSTHOC="${STRICT_POSTHOC:-1}"
DISABLE_EXAMPLE_CACHE="${DISABLE_EXAMPLE_CACHE:-0}"
MODEL_PROFILE="${MODEL_PROFILE:-legacy_1b}"
MODEL_ALLOWLIST="${MODEL_ALLOWLIST:-}"

mkdir -p logs/experiment2

bash scripts/experiment2_phase2b_posthoc_queue_build.sh "$RUN_ID" "$OUT_ROOT" "$QUEUE_FILE"

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
  echo "No GPUs provided for posthoc queue launch (GPU_CSV='$GPU_CSV')." >&2
  exit 1
fi

for gpu in "${GPUS[@]}"; do
  tmux kill-session -t "exp2p_g${gpu}" 2>/dev/null || true
done

for gpu in "${GPUS[@]}"; do
  tmux new-session -d -s "exp2p_g${gpu}" \
    "cd \"$ROOT_DIR\" && BATCH_SIZE_SYNTH=$BATCH_SIZE_SYNTH BATCH_SIZE_TIER1=$BATCH_SIZE_TIER1 STRICT_POSTHOC=$STRICT_POSTHOC DISABLE_EXAMPLE_CACHE=$DISABLE_EXAMPLE_CACHE MODEL_PROFILE=$MODEL_PROFILE MODEL_ALLOWLIST=$MODEL_ALLOWLIST bash scripts/experiment2_phase2b_posthoc_queue_worker.sh $gpu \"$QUEUE_FILE\" \"$OUT_ROOT\""
done

echo "Launched Phase2B posthoc queue workers on GPUs ${GPU_CSV} for run_id=$RUN_ID"
echo "Posthoc queue: $QUEUE_FILE"
