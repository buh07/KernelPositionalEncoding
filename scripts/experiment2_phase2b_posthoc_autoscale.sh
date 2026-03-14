#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?usage: scripts/experiment2_phase2b_posthoc_autoscale.sh <run_id> [output_root] [queue_file] [gpu_csv] [poll_seconds]}"
OUT_ROOT="${2:-results/experiment2}"
QUEUE_FILE="${3:-logs/experiment2/${RUN_ID}_phase2b_posthoc_queue.tsv}"
GPU_CSV="${4:-${POSTHOC_GPUS:-0,1,2,3,4,5}}"
POLL_SECONDS="${5:-60}"

BATCH_SIZE_SYNTH="${BATCH_SIZE_SYNTH:-24}"
BATCH_SIZE_TIER1="${BATCH_SIZE_TIER1:-24}"
STRICT_POSTHOC="${STRICT_POSTHOC:-1}"
DISABLE_EXAMPLE_CACHE="${DISABLE_EXAMPLE_CACHE:-0}"
MODEL_PROFILE="${MODEL_PROFILE:-legacy_1b}"
MODEL_ALLOWLIST="${MODEL_ALLOWLIST:-}"
EXIT_ON_FAILURES="${EXIT_ON_FAILURES:-1}"
POSTHOC_MAX_WORKERS="${POSTHOC_MAX_WORKERS:-4}"

mkdir -p logs/experiment2

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
  echo "No GPUs provided for posthoc autoscale (GPU_CSV='$GPU_CSV')." >&2
  exit 1
fi

if [[ ! -f "$QUEUE_FILE" ]]; then
  bash scripts/experiment2_phase2b_posthoc_queue_build.sh "$RUN_ID" "$OUT_ROOT" "$QUEUE_FILE"
fi

PENDING_FILE="${QUEUE_FILE%.tsv}.pending.tsv"
DONE_FILE="${QUEUE_FILE%.tsv}.done.tsv"
FAIL_FILE="${QUEUE_FILE%.tsv}.failures.tsv"
touch "$DONE_FILE" "$FAIL_FILE"
if [[ ! -f "$PENDING_FILE" ]]; then
  cp "$QUEUE_FILE" "$PENDING_FILE"
fi

launch_worker() {
  local gpu="$1"
  local preferred_model="${2:-}"
  tmux kill-session -t "exp2p_g${gpu}" 2>/dev/null || true
  tmux new-session -d -s "exp2p_g${gpu}" \
    "cd \"$ROOT_DIR\" && BATCH_SIZE_SYNTH=$BATCH_SIZE_SYNTH BATCH_SIZE_TIER1=$BATCH_SIZE_TIER1 STRICT_POSTHOC=$STRICT_POSTHOC DISABLE_EXAMPLE_CACHE=$DISABLE_EXAMPLE_CACHE MODEL_PROFILE=$MODEL_PROFILE MODEL_ALLOWLIST=$MODEL_ALLOWLIST PREFERRED_MODEL=$preferred_model bash scripts/experiment2_phase2b_posthoc_queue_worker.sh $gpu \"$QUEUE_FILE\" \"$OUT_ROOT\" \"$preferred_model\""
}

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] posthoc-autoscale start run_id=$RUN_ID gpus=$GPU_CSV queue=$QUEUE_FILE max_workers=$POSTHOC_MAX_WORKERS"

while true; do
  pending=0
  done_count=0
  fail_count=0
  [[ -f "$PENDING_FILE" ]] && pending="$(wc -l < "$PENDING_FILE" | tr -d ' ')"
  [[ -f "$DONE_FILE" ]] && done_count="$(wc -l < "$DONE_FILE" | tr -d ' ')"
  [[ -f "$FAIL_FILE" ]] && fail_count="$(wc -l < "$FAIL_FILE" | tr -d ' ')"

  alive=0
  for gpu in "${GPUS[@]}"; do
    if tmux has-session -t "exp2p_g${gpu}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done

  for gpu in "${GPUS[@]}"; do
    if tmux has-session -t "exp2p_g${gpu}" 2>/dev/null; then
      continue
    fi
    # Never preempt active execute workers.
    if tmux has-session -t "exp2q_g${gpu}" 2>/dev/null; then
      continue
    fi
    if (( pending > 0 && alive < POSTHOC_MAX_WORKERS )); then
      launch_worker "$gpu" ""
      alive=$((alive + 1))
    fi
  done

  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] posthoc-autoscale-progress run_id=$RUN_ID pending=$pending done_jobs=$done_count fail_jobs=$fail_count workers_alive=$alive max_workers=$POSTHOC_MAX_WORKERS"

  if (( EXIT_ON_FAILURES == 1 && fail_count > 0 )); then
    echo "Posthoc queue has failures. Review: $FAIL_FILE" >&2
    exit 1
  fi

  if (( pending == 0 && alive == 0 )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] posthoc-autoscale complete run_id=$RUN_ID"
