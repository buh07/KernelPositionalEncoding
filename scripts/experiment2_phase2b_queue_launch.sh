#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?usage: scripts/experiment2_phase2b_queue_launch.sh <run_id> [output_root] [queue_file] [chunk_size] [manifest_path]}"
OUT_ROOT="${2:-results/experiment2}"
QUEUE_FILE="${3:-logs/experiment2/${RUN_ID}_phase2b_queue.tsv}"
CHUNK_SIZE="${4:-1}"
MANIFEST_PATH="${5:-$OUT_ROOT/phase2b/$RUN_ID/manifest.jsonl}"

mkdir -p logs/experiment2

bash scripts/experiment2_phase2b_queue_build.sh "$RUN_ID" "$OUT_ROOT" "$QUEUE_FILE" "$CHUNK_SIZE" "$MANIFEST_PATH"

# Reset queue runtime state for a clean relaunch. Without this, stale
# pending/done/failure files from prior attempts can cause workers to exit
# early or skip jobs.
PENDING_FILE="${QUEUE_FILE%.tsv}.pending.tsv"
DONE_FILE="${QUEUE_FILE%.tsv}.done.tsv"
FAIL_FILE="${QUEUE_FILE%.tsv}.failures.tsv"
LOCK_FILE="${QUEUE_FILE%.tsv}.lock"
cp "$QUEUE_FILE" "$PENDING_FILE"
: > "$DONE_FILE"
: > "$FAIL_FILE"
rm -f "$LOCK_FILE"

GPU_LIST="${GPU_LIST:-0 1 2}"

for gpu in $GPU_LIST; do
  tmux kill-session -t "exp2q_g${gpu}" 2>/dev/null || true
done

KERNEL_ENGINE="${KERNEL_ENGINE:-optimized}"
CENTERED_COMPUTE="${CENTERED_COMPUTE:-defer}"
SYNTHETIC_EVAL_MODE="${SYNTHETIC_EVAL_MODE:-restricted}"
CANDIDATE_SIZE="${CANDIDATE_SIZE:-10}"
FLOOR_THRESHOLD="${FLOOR_THRESHOLD:-0.15}"
BATCH_SIZE_SYNTH="${BATCH_SIZE_SYNTH:-24}"
BATCH_SIZE_TIER1="${BATCH_SIZE_TIER1:-24}"
SYNTHETIC_COUNT="${SYNTHETIC_COUNT:-200}"
TIER1_COUNT="${TIER1_COUNT:-100}"
PRUNE_FLOOR_LIMITED="${PRUNE_FLOOR_LIMITED:-1}"
H12_ENDPOINT_POLICY="${H12_ENDPOINT_POLICY:-raw_primary}"
MODEL_PROFILE="${MODEL_PROFILE:-legacy_1b}"
MODEL_ALLOWLIST="${MODEL_ALLOWLIST:-}"
TRACK_A_ENABLED="${TRACK_A_ENABLED:-true}"
INTERVENTION_PROFILE="${INTERVENTION_PROFILE:-full}"

for gpu in $GPU_LIST; do
  tmux new-session -d -s "exp2q_g${gpu}" \
    "cd \"$ROOT_DIR\" && KERNEL_ENGINE=$KERNEL_ENGINE CENTERED_COMPUTE=$CENTERED_COMPUTE SYNTHETIC_EVAL_MODE=$SYNTHETIC_EVAL_MODE CANDIDATE_SIZE=$CANDIDATE_SIZE FLOOR_THRESHOLD=$FLOOR_THRESHOLD BATCH_SIZE_SYNTH=$BATCH_SIZE_SYNTH BATCH_SIZE_TIER1=$BATCH_SIZE_TIER1 SYNTHETIC_COUNT=$SYNTHETIC_COUNT TIER1_COUNT=$TIER1_COUNT H12_ENDPOINT_POLICY=$H12_ENDPOINT_POLICY MODEL_PROFILE=$MODEL_PROFILE MODEL_ALLOWLIST=$MODEL_ALLOWLIST TRACK_A_ENABLED=$TRACK_A_ENABLED INTERVENTION_PROFILE=$INTERVENTION_PROFILE PRUNE_FLOOR_LIMITED=$PRUNE_FLOOR_LIMITED bash scripts/experiment2_phase2b_queue_worker.sh $gpu \"$QUEUE_FILE\" \"$OUT_ROOT\""
done

echo "Launched Phase2B queue workers on GPUs: $GPU_LIST for run_id=$RUN_ID"
echo "Queue: $QUEUE_FILE"
echo "Manifest used: $MANIFEST_PATH"
