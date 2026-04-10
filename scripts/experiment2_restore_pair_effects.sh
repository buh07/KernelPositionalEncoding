#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/f004ndc/Kernel PE"
cd "$ROOT"

# Keep this run tag fixed so Theory 3's default path resolves without code changes.
RUN_TAG="${RUN_TAG:-quick_pair_expand_pivot_20260314_0043_v2}"
GPU_LIST="${GPU_LIST:-0 1}"
SYNTHETIC_COUNT="${SYNTHETIC_COUNT:-100}"
PAIR_INDICES="${PAIR_INDICES:-0,8,16,24,32,40,48,56}"
SEEDS="${SEEDS:-0}"

read -r -a GPUS <<< "${GPU_LIST//,/ }"
if (( ${#GPUS[@]} != 2 )); then
  echo "Expected exactly 2 GPUs in GPU_LIST (got: '${GPU_LIST}')." >&2
  exit 1
fi
G0="${GPUS[0]}"
G1="${GPUS[1]}"

read -r -a SEED_LIST <<< "${SEEDS//,/ }"
if (( ${#SEED_LIST[@]} < 1 )); then
  echo "Expected at least one seed in SEEDS (got: '${SEEDS}')." >&2
  exit 1
fi
SEED_A="${SEED_LIST[0]}"
SEED_B="${SEED_LIST[1]:-}"

OUTPUT_ROOT="results/experiment2/quick/${RUN_TAG}"
LOG_ROOT="logs/experiment2/quick/${RUN_TAG}"
MANIFEST_DIR="${OUTPUT_ROOT}/manifests"
REPORT_EXPANDED="${OUTPUT_ROOT}/reports/pair_expanded"
REPORT_UNIFIED="${OUTPUT_ROOT}/reports/pair_unified"
BASE_RUN_ID="${RUN_TAG}_base"
PAIR_RUN_ID="${RUN_TAG}_pair"

BASE_MANIFEST="${OUTPUT_ROOT}/phase2b/${BASE_RUN_ID}/manifest.jsonl"
PAIR_MANIFEST="${MANIFEST_DIR}/${PAIR_RUN_ID}.jsonl"
FAMILIES_JSON="${MANIFEST_DIR}/${PAIR_RUN_ID}.families.json"

mkdir -p "$LOG_ROOT" "$MANIFEST_DIR" "$REPORT_EXPANDED" "$REPORT_UNIFIED"

cat > "$FAMILIES_JSON" <<'JSON'
[
  {"model": "llama-3.1-8b", "task": "long_range_retrieval", "spans": [32, 48, 64]},
  {"model": "olmo-2-7b", "task": "long_range_retrieval", "spans": [24, 32]},
  {"model": "llama-3.1-8b", "task": "local_key_match"},
  {"model": "olmo-2-7b", "task": "local_key_match"}
]
JSON

if [[ ! -f "$BASE_MANIFEST" ]]; then
  echo "[$(date)] [restore-pair-effects] build phase2b manifest..."
  ./.venv/bin/python experiment2/run.py \
    --mode build \
    --phase phase2b \
    --run-id "$BASE_RUN_ID" \
    --device cuda \
    --output-root "$OUTPUT_ROOT" \
    --model-profile scaleup_78b \
    --model-allowlist llama-3.1-8b,olmo-2-7b \
    --phase2b-core-synthetic-only \
    --intervention-profile strong_only \
    --random-draws-confirmatory 3 \
    --print-summary |& tee "$LOG_ROOT/build.log"
else
  echo "[$(date)] [restore-pair-effects] reusing existing base manifest: $BASE_MANIFEST"
fi

if [[ ! -f "$PAIR_MANIFEST" ]]; then
  echo "[$(date)] [restore-pair-effects] build expanded pair manifest..."
  ./.venv/bin/python scripts/experiment2_pair_sweep_manifest.py \
    --input-manifest "$BASE_MANIFEST" \
    --output-manifest "$PAIR_MANIFEST" \
    --summary-json "${MANIFEST_DIR}/${PAIR_RUN_ID}.summary.json" \
    --run-id "$PAIR_RUN_ID" \
    --phase quick_pair_expand \
    --seeds "$SEEDS" \
    --pair-indices "$PAIR_INDICES" \
    --random-draws 3 \
    --families-json "$FAMILIES_JSON" \
    --notes-tag quick_pair_expanded_matrix_restore |& tee "$LOG_ROOT/manifest.log"
else
  echo "[$(date)] [restore-pair-effects] reusing existing pair manifest: $PAIR_MANIFEST"
fi

echo "[$(date)] [restore-pair-effects] execute pair manifest on GPU ${G0} (seed ${SEED_A})..."
CUDA_VISIBLE_DEVICES="$G0" ./.venv/bin/python experiment2/run.py \
  --mode execute \
  --manifest "$PAIR_MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda:0 \
  --kernel-engine optimized \
  --centered-compute defer \
  --synthetic-eval-mode restricted \
  --candidate-size 10 \
  --h12-endpoint-policy co_primary_raw_headroom \
  --track-a-enabled false \
  --intervention-profile full \
  --synthetic-count "$SYNTHETIC_COUNT" \
  --batch-size-synth 24 \
  --seed-start "$SEED_A" \
  --seed-end "$SEED_A" \
  --print-summary |& tee "$LOG_ROOT/execute_g${G0}.log" &
PID0=$!

PID1=""
if [[ -n "$SEED_B" ]]; then
  echo "[$(date)] [restore-pair-effects] execute pair manifest on GPU ${G1} (seed ${SEED_B})..."
  CUDA_VISIBLE_DEVICES="$G1" ./.venv/bin/python experiment2/run.py \
    --mode execute \
    --manifest "$PAIR_MANIFEST" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda:0 \
    --kernel-engine optimized \
    --centered-compute defer \
    --synthetic-eval-mode restricted \
    --candidate-size 10 \
    --h12-endpoint-policy co_primary_raw_headroom \
    --track-a-enabled false \
    --intervention-profile full \
    --synthetic-count "$SYNTHETIC_COUNT" \
    --batch-size-synth 24 \
    --seed-start "$SEED_B" \
    --seed-end "$SEED_B" \
    --print-summary |& tee "$LOG_ROOT/execute_g${G1}.log" &
  PID1=$!
else
  echo "[$(date)] [restore-pair-effects] single-seed mode; skipping second GPU worker."
fi

wait "$PID0"
if [[ -n "$PID1" ]]; then
  wait "$PID1"
fi

echo "[$(date)] [restore-pair-effects] build pair report..."
./.venv/bin/python scripts/experiment2_pair_sweep_report.py \
  --aggregate-task-metrics "${OUTPUT_ROOT}/quick_pair_expand/${PAIR_RUN_ID}/aggregate_task_metrics.parquet" \
  --output-dir "$REPORT_EXPANDED" \
  --families-json "$FAMILIES_JSON" \
  --pair-indices "$PAIR_INDICES" \
  --floor-threshold 0.15 |& tee "$LOG_ROOT/report.log"

echo "[$(date)] [restore-pair-effects] merge to unified package..."
./.venv/bin/python scripts/experiment2_pair_sweep_merge.py \
  --inputs "$REPORT_EXPANDED" \
  --output-dir "$REPORT_UNIFIED" \
  --source-runs "$RUN_TAG" \
  --mc-method holm |& tee "$LOG_ROOT/merge.log"

PAIR_MERGED="${REPORT_UNIFIED}/pair_effects_merged.parquet"
if [[ ! -f "$PAIR_MERGED" ]]; then
  echo "FAILED: expected ${PAIR_MERGED} was not created." >&2
  exit 1
fi

echo "[$(date)] [restore-pair-effects] done."
echo "  pair_effects_merged: ${PAIR_MERGED}"
echo "  logs: ${LOG_ROOT}"
