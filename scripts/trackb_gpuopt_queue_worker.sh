#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <gpu_id> [queue_file]"
  exit 1
fi

GPU_ID="$1"
QUEUE_FILE="${2:-logs/trackb/trackb_gpuopt_queue.tsv}"
STATE_FILE="${QUEUE_FILE%.tsv}.state"
LOCK_FILE="${QUEUE_FILE%.tsv}.lock"
FAIL_FILE="${QUEUE_FILE%.tsv}.failures.tsv"

ROOT="/scratch2/f004ndc/Kernel PE"
cd "$ROOT"
mkdir -p "$(dirname "$QUEUE_FILE")" "logs/trackb"

if [[ ! -f "$QUEUE_FILE" ]]; then
  bash scripts/trackb_gpuopt_queue_build.sh "$QUEUE_FILE"
fi
touch "$LOCK_FILE"
if [[ ! -s "$STATE_FILE" ]] || ! grep -Eq '^[0-9]+$' "$STATE_FILE"; then
  echo "1" > "$STATE_FILE"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

LOG_FILE="logs/trackb/trackb_gpuopt_worker_g${GPU_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] worker start gpu=$GPU_ID queue=$QUEUE_FILE"

claim_next_line() {
  local line_no total line
  exec 9>>"$LOCK_FILE"
  flock -x 9
  line_no="$(cat "$STATE_FILE" 2>/dev/null || echo 1)"
  total="$(wc -l < "$QUEUE_FILE")"
  if (( line_no > total )); then
    flock -u 9
    exec 9>&-
    return 1
  fi
  line="$(sed -n "${line_no}p" "$QUEUE_FILE")"
  echo $((line_no + 1)) > "$STATE_FILE"
  flock -u 9
  exec 9>&-
  printf "%s" "$line"
}

START_TS="$(date +%s)"
RAN=0
SKIPPED=0
FAILED=0

while true; do
  LINE="$(claim_next_line || true)"
  if [[ -z "$LINE" ]]; then
    break
  fi

  IFS=$'\t' read -r OUTPUT_GROUP CENTERING_MODE BUCKET_SIZE MODEL <<< "$LINE"
  model_root="results/${OUTPUT_GROUP}/${MODEL}"
  summary_count=0
  if [[ -d "$model_root" ]]; then
    summary_count="$(find "$model_root" -type f -name summary.parquet | wc -l | tr -d ' ')"
  fi
  if [[ "$summary_count" -ge 6 ]]; then
    SKIPPED=$((SKIPPED + 1))
    echo "SKIP existing ${OUTPUT_GROUP} model=${MODEL} summaries=${summary_count}/6"
    continue
  fi

  echo "START ${OUTPUT_GROUP} model=${MODEL} mode=${CENTERING_MODE} bucket=${BUCKET_SIZE}"
  cmd=(
    .venv/bin/python experiment1/run.py track-b
    --model "$MODEL"
    --dataset all
    --seq-len all
    --device cuda
    --track-b-centering-mode "$CENTERING_MODE"
    --track-b-output-group "$OUTPUT_GROUP"
    --track-b-artifact-level compact
  )
  if [[ "$CENTERING_MODE" == "bucketed_mean" ]]; then
    cmd+=(--track-b-bucket-size "$BUCKET_SIZE")
  fi

  if "${cmd[@]}"; then
    RAN=$((RAN + 1))
    done_count="$(find "$model_root" -type f -name summary.parquet | wc -l | tr -d ' ')"
    echo "DONE  ${OUTPUT_GROUP} model=${MODEL} summaries=${done_count}/6"
  else
    FAILED=$((FAILED + 1))
    printf "%s\n" "$LINE" >> "$FAIL_FILE"
    echo "FAIL  ${OUTPUT_GROUP} model=${MODEL} (recorded in $FAIL_FILE)"
  fi
done

END_TS="$(date +%s)"
ELAPSED=$((END_TS - START_TS))

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] worker end gpu=$GPU_ID ran=$RAN skipped=$SKIPPED failed=$FAILED elapsed_sec=$ELAPSED"
if (( FAILED > 0 )); then
  exit 1
fi
