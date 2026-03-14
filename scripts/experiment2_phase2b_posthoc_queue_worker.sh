#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <gpu_id> <queue_file> [output_root] [preferred_model]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_ID="$1"
QUEUE_FILE="$2"
OUT_ROOT="${3:-results/experiment2}"
PREFERRED_MODEL="${4:-${PREFERRED_MODEL:-}}"

if [[ ! -f "$QUEUE_FILE" ]]; then
  echo "Missing queue file: $QUEUE_FILE" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

BATCH_SIZE_SYNTH="${BATCH_SIZE_SYNTH:-24}"
BATCH_SIZE_TIER1="${BATCH_SIZE_TIER1:-24}"
STRICT_POSTHOC="${STRICT_POSTHOC:-1}"
DISABLE_EXAMPLE_CACHE="${DISABLE_EXAMPLE_CACHE:-0}"
MODEL_PROFILE="${MODEL_PROFILE:-legacy_1b}"
MODEL_ALLOWLIST="${MODEL_ALLOWLIST:-}"

STRICT_ARGS=()
if [[ "$STRICT_POSTHOC" == "1" ]]; then
  STRICT_ARGS+=(--strict-posthoc)
fi
CACHE_ARGS=()
if [[ "$DISABLE_EXAMPLE_CACHE" == "1" ]]; then
  CACHE_ARGS+=(--disable-example-cache)
fi
MODEL_ARGS=()
if [[ -n "$PREFERRED_MODEL" ]]; then
  MODEL_ARGS+=(--preferred-model "$PREFERRED_MODEL")
fi

worker_log="logs/experiment2/$(basename "${QUEUE_FILE%.tsv}")_worker_g${GPU_ID}.log"
mkdir -p "$(dirname "$worker_log")"
exec > >(tee -a "$worker_log") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] posthoc-worker(start) gpu=$GPU_ID queue=$QUEUE_FILE preferred_model=${PREFERRED_MODEL:-none}"

.venv/bin/python experiment2/run.py \
  --mode posthoc-queue-worker \
  --model-profile "$MODEL_PROFILE" \
  --model-allowlist "$MODEL_ALLOWLIST" \
  --phase phase2b \
  --queue-file "$QUEUE_FILE" \
  --device cuda \
  --output-root "$OUT_ROOT" \
  --batch-size-synth "$BATCH_SIZE_SYNTH" \
  --batch-size-tier1 "$BATCH_SIZE_TIER1" \
  --print-summary \
  "${STRICT_ARGS[@]}" \
  "${CACHE_ARGS[@]}" \
  "${MODEL_ARGS[@]}"

status=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] posthoc-worker(end) gpu=$GPU_ID status=$status"
exit "$status"
