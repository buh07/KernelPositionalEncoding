#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?usage: scripts/experiment2_phase2b_seedshard_launch.sh <run_id> [output_root]}"
OUT_ROOT="${2:-results/experiment2}"
MANIFEST="${OUT_ROOT}/phase2b/${RUN_ID}/manifest.jsonl"
LOG_DIR="logs/experiment2"
mkdir -p "$LOG_DIR"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing Phase 2B manifest: $MANIFEST" >&2
  exit 1
fi

KERNEL_ENGINE="${KERNEL_ENGINE:-optimized}"
CENTERED_COMPUTE="${CENTERED_COMPUTE:-defer}"
SYNTHETIC_EVAL_MODE="${SYNTHETIC_EVAL_MODE:-restricted}"
CANDIDATE_SIZE="${CANDIDATE_SIZE:-10}"
FLOOR_THRESHOLD="${FLOOR_THRESHOLD:-0.15}"
BATCH_SIZE_SYNTH="${BATCH_SIZE_SYNTH:-8}"
BATCH_SIZE_TIER1="${BATCH_SIZE_TIER1:-8}"
PRUNE_FLOOR_LIMITED="${PRUNE_FLOOR_LIMITED:-1}"

PRUNE_ARG=""
if [[ "$PRUNE_FLOOR_LIMITED" == "1" ]]; then
  PRUNE_ARG="--prune-floor-limited-interventions"
fi

for sess in exp2b_g0 exp2b_g1 exp2b_g2; do
  tmux kill-session -t "$sess" 2>/dev/null || true
done

tmux new-session -d -s exp2b_g0 \
  "cd \"$ROOT_DIR\" && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 .venv/bin/python experiment2/run.py --mode execute --manifest \"$MANIFEST\" --device cuda --output-root \"$OUT_ROOT\" --kernel-engine \"$KERNEL_ENGINE\" --centered-compute \"$CENTERED_COMPUTE\" --synthetic-eval-mode \"$SYNTHETIC_EVAL_MODE\" --candidate-size \"$CANDIDATE_SIZE\" --floor-threshold \"$FLOOR_THRESHOLD\" --batch-size-synth \"$BATCH_SIZE_SYNTH\" --batch-size-tier1 \"$BATCH_SIZE_TIER1\" --seed-start 0 --seed-end 1 --print-summary $PRUNE_ARG |& tee \"$LOG_DIR/${RUN_ID}_g0_phase2b_seed01.log\""

tmux new-session -d -s exp2b_g1 \
  "cd \"$ROOT_DIR\" && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python experiment2/run.py --mode execute --manifest \"$MANIFEST\" --device cuda --output-root \"$OUT_ROOT\" --kernel-engine \"$KERNEL_ENGINE\" --centered-compute \"$CENTERED_COMPUTE\" --synthetic-eval-mode \"$SYNTHETIC_EVAL_MODE\" --candidate-size \"$CANDIDATE_SIZE\" --floor-threshold \"$FLOOR_THRESHOLD\" --batch-size-synth \"$BATCH_SIZE_SYNTH\" --batch-size-tier1 \"$BATCH_SIZE_TIER1\" --seed-start 2 --seed-end 3 --print-summary $PRUNE_ARG |& tee \"$LOG_DIR/${RUN_ID}_g1_phase2b_seed23.log\""

tmux new-session -d -s exp2b_g2 \
  "cd \"$ROOT_DIR\" && CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 .venv/bin/python experiment2/run.py --mode execute --manifest \"$MANIFEST\" --device cuda --output-root \"$OUT_ROOT\" --kernel-engine \"$KERNEL_ENGINE\" --centered-compute \"$CENTERED_COMPUTE\" --synthetic-eval-mode \"$SYNTHETIC_EVAL_MODE\" --candidate-size \"$CANDIDATE_SIZE\" --floor-threshold \"$FLOOR_THRESHOLD\" --batch-size-synth \"$BATCH_SIZE_SYNTH\" --batch-size-tier1 \"$BATCH_SIZE_TIER1\" --seed-start 4 --seed-end 6 --print-summary $PRUNE_ARG |& tee \"$LOG_DIR/${RUN_ID}_g2_phase2b_seed46.log\""

echo "Launched Phase 2B seed-sharded execute on GPUs 0/1/2 for run_id=${RUN_ID}"
