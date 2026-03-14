#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 <gpu_id> <manifest1.jsonl> [manifest2.jsonl ...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_ID="$1"
shift

export CUDA_VISIBLE_DEVICES="$GPU_ID"
KERNEL_ENGINE="${KERNEL_ENGINE:-optimized}"
CENTERED_COMPUTE="${CENTERED_COMPUTE:-defer}"
BATCH_SIZE_SYNTH="${BATCH_SIZE_SYNTH:-8}"
BATCH_SIZE_TIER1="${BATCH_SIZE_TIER1:-8}"
SYNTHETIC_EVAL_MODE="${SYNTHETIC_EVAL_MODE:-restricted}"
CANDIDATE_SIZE="${CANDIDATE_SIZE:-10}"
FLOOR_THRESHOLD="${FLOOR_THRESHOLD:-0.15}"
ALLOW_PHASE2A_REUSE="${ALLOW_PHASE2A_REUSE:-0}"
PRUNE_FLOOR_LIMITED="${PRUNE_FLOOR_LIMITED:-0}"
SEED_START="${SEED_START:-}"
SEED_END="${SEED_END:-}"
EXAMPLE_CACHE_FLAG="${EXAMPLE_CACHE_FLAG:-}"
SEED_RANGE_ARGS=()
REUSE_ARGS=()
PRUNE_ARGS=()
if [[ -n "$SEED_START" ]]; then
  SEED_RANGE_ARGS+=(--seed-start "$SEED_START")
fi
if [[ -n "$SEED_END" ]]; then
  SEED_RANGE_ARGS+=(--seed-end "$SEED_END")
fi
if [[ "$ALLOW_PHASE2A_REUSE" == "1" ]]; then
  REUSE_ARGS+=(--allow-phase2a-reuse)
fi
if [[ "$PRUNE_FLOOR_LIMITED" == "1" ]]; then
  PRUNE_ARGS+=(--prune-floor-limited-interventions)
fi

for manifest in "$@"; do
  echo "[$(date +%F\ %T)] GPU=${GPU_ID} engine=${KERNEL_ENGINE} centered=${CENTERED_COMPUTE} starting manifest=${manifest}"
  PYTHONUNBUFFERED=1 .venv/bin/python experiment2/run.py \
    --mode execute \
    --manifest "$manifest" \
    --device cuda \
    --kernel-engine "$KERNEL_ENGINE" \
    --centered-compute "$CENTERED_COMPUTE" \
    --synthetic-eval-mode "$SYNTHETIC_EVAL_MODE" \
    --candidate-size "$CANDIDATE_SIZE" \
    --floor-threshold "$FLOOR_THRESHOLD" \
    --batch-size-synth "$BATCH_SIZE_SYNTH" \
    --batch-size-tier1 "$BATCH_SIZE_TIER1" \
    "${REUSE_ARGS[@]}" \
    "${PRUNE_ARGS[@]}" \
    "${SEED_RANGE_ARGS[@]}" \
    --output-root results/experiment2 \
    --print-summary \
    ${EXAMPLE_CACHE_FLAG}
  echo "[$(date +%F\ %T)] GPU=${GPU_ID} engine=${KERNEL_ENGINE} centered=${CENTERED_COMPUTE} finished manifest=${manifest}"
done

echo "[$(date +%F\ %T)] GPU=${GPU_ID} engine=${KERNEL_ENGINE} centered=${CENTERED_COMPUTE} all assigned manifests complete"
