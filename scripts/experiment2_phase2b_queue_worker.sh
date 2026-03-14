#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <gpu_id> <queue_file> [output_root]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_ID="$1"
QUEUE_FILE="$2"
OUT_ROOT="${3:-results/experiment2}"

PENDING_FILE="${QUEUE_FILE%.tsv}.pending.tsv"
DONE_FILE="${QUEUE_FILE%.tsv}.done.tsv"
FAIL_FILE="${QUEUE_FILE%.tsv}.failures.tsv"
LOCK_FILE="${QUEUE_FILE%.tsv}.lock"

if [[ ! -f "$QUEUE_FILE" ]]; then
  echo "Missing queue file: $QUEUE_FILE" >&2
  exit 1
fi

touch "$LOCK_FILE"
touch "$DONE_FILE"
touch "$FAIL_FILE"

if [[ ! -f "$PENDING_FILE" ]]; then
  cp "$QUEUE_FILE" "$PENDING_FILE"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

KERNEL_ENGINE="${KERNEL_ENGINE:-optimized}"
CENTERED_COMPUTE="${CENTERED_COMPUTE:-defer}"
SYNTHETIC_EVAL_MODE="${SYNTHETIC_EVAL_MODE:-restricted}"
CANDIDATE_SIZE="${CANDIDATE_SIZE:-10}"
FLOOR_THRESHOLD="${FLOOR_THRESHOLD:-0.15}"
BATCH_SIZE_SYNTH="${BATCH_SIZE_SYNTH:-24}"
BATCH_SIZE_TIER1="${BATCH_SIZE_TIER1:-24}"
SYNTHETIC_COUNT="${SYNTHETIC_COUNT:-200}"
TIER1_COUNT="${TIER1_COUNT:-100}"
H12_ENDPOINT_POLICY="${H12_ENDPOINT_POLICY:-raw_primary}"
MODEL_PROFILE="${MODEL_PROFILE:-legacy_1b}"
MODEL_ALLOWLIST="${MODEL_ALLOWLIST:-}"
TRACK_A_ENABLED="${TRACK_A_ENABLED:-true}"
INTERVENTION_PROFILE="${INTERVENTION_PROFILE:-full}"
PRUNE_FLOOR_LIMITED="${PRUNE_FLOOR_LIMITED:-1}"

PRUNE_ARGS=()
if [[ "$PRUNE_FLOOR_LIMITED" == "1" ]]; then
  PRUNE_ARGS+=(--prune-floor-limited-interventions)
fi

claim_next_job() {
  local preferred_model="$1"
  local line_no line
  exec 9>>"$LOCK_FILE"
  flock -x 9
  if [[ ! -s "$PENDING_FILE" ]]; then
    flock -u 9
    exec 9>&-
    return 1
  fi

  line_no=""
  if [[ -n "$preferred_model" ]]; then
    line_no="$(awk -F'\t' -v m="$preferred_model" '$3==m {print NR; exit}' "$PENDING_FILE")"
  fi
  if [[ -z "$line_no" ]]; then
    line_no=1
  fi

  line="$(sed -n "${line_no}p" "$PENDING_FILE")"
  awk -v n="$line_no" 'NR!=n {print}' "$PENDING_FILE" > "${PENDING_FILE}.tmp"
  mv "${PENDING_FILE}.tmp" "$PENDING_FILE"

  flock -u 9
  exec 9>&-
  printf "%s" "$line"
}

worker_log="logs/experiment2/$(basename "${QUEUE_FILE%.tsv}")_worker_g${GPU_ID}.log"
mkdir -p "$(dirname "$worker_log")"
exec > >(tee -a "$worker_log") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] worker start gpu=$GPU_ID queue=$QUEUE_FILE"

CURRENT_MODEL=""
START_TS="$(date +%s)"
DONE_COUNT=0
FAIL_COUNT=0

while true; do
  JOB_LINE="$(claim_next_job "$CURRENT_MODEL" || true)"
  if [[ -z "$JOB_LINE" ]]; then
    break
  fi

  IFS=$'\t' read -r JOB_ID MANIFEST_PATH MODEL SEED_START SEED_END EXPECTED_ROWS <<< "$JOB_LINE"
  CURRENT_MODEL="$MODEL"

  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] START job=$JOB_ID model=$MODEL seeds=${SEED_START}-${SEED_END} expected_rows=$EXPECTED_ROWS"
  cmd=(
    .venv/bin/python experiment2/run.py
    --mode execute
    --model-profile "$MODEL_PROFILE"
    --model-allowlist "$MODEL_ALLOWLIST"
    --manifest "$MANIFEST_PATH"
    --device cuda
    --output-root "$OUT_ROOT"
    --synthetic-count "$SYNTHETIC_COUNT"
    --tier1-count "$TIER1_COUNT"
    --kernel-engine "$KERNEL_ENGINE"
    --centered-compute "$CENTERED_COMPUTE"
    --synthetic-eval-mode "$SYNTHETIC_EVAL_MODE"
    --candidate-size "$CANDIDATE_SIZE"
    --h12-endpoint-policy "$H12_ENDPOINT_POLICY"
    --track-a-enabled "$TRACK_A_ENABLED"
    --intervention-profile "$INTERVENTION_PROFILE"
    --floor-threshold "$FLOOR_THRESHOLD"
    --batch-size-synth "$BATCH_SIZE_SYNTH"
    --batch-size-tier1 "$BATCH_SIZE_TIER1"
    --seed-start "$SEED_START"
    --seed-end "$SEED_END"
    --print-summary
  )
  cmd+=("${PRUNE_ARGS[@]}")

  if "${cmd[@]}"; then
    DONE_COUNT=$((DONE_COUNT + 1))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$GPU_ID" "$JOB_ID" "$MODEL" "$SEED_START" "$SEED_END" "ok" >> "$DONE_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] DONE job=$JOB_ID"
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$GPU_ID" "$JOB_ID" "$MODEL" "$SEED_START" "$SEED_END" "failed" >> "$FAIL_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] FAIL job=$JOB_ID"
  fi

done

ELAPSED=$(( $(date +%s) - START_TS ))
PENDING_LEFT=0
if [[ -f "$PENDING_FILE" ]]; then
  PENDING_LEFT="$(wc -l < "$PENDING_FILE" | tr -d ' ')"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] worker end gpu=$GPU_ID done=$DONE_COUNT failed=$FAIL_COUNT pending_left=$PENDING_LEFT elapsed_sec=$ELAPSED"

if (( FAIL_COUNT > 0 )); then
  exit 1
fi
