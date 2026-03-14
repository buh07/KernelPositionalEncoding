#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${1:-quick_stage1_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="results/experiment2/quick/${RUN_TAG}"
LOG_ROOT="logs/experiment2/quick/${RUN_TAG}"
MANIFEST_DIR="${OUTPUT_ROOT}/manifests"
REPORT_DIR="${OUTPUT_ROOT}/reports"

BASE_RUN_ID="${RUN_TAG}_base"
SLOPE_RUN_ID="${RUN_TAG}_slope"
DOSE_RUN_ID="${RUN_TAG}_dose"

mkdir -p "${LOG_ROOT}" "${MANIFEST_DIR}" "${REPORT_DIR}/slope" "${REPORT_DIR}/dose"

echo "[stage1] run_tag=${RUN_TAG}"
echo "[stage1] output_root=${OUTPUT_ROOT}"
echo "[stage1] log_root=${LOG_ROOT}"

BASE_MANIFEST="${OUTPUT_ROOT}/phase2b/${BASE_RUN_ID}/manifest.jsonl"

.venv/bin/python experiment2/run.py \
  --mode build \
  --phase phase2b \
  --run-id "${BASE_RUN_ID}" \
  --device cuda \
  --output-root "${OUTPUT_ROOT}" \
  --model-profile scaleup_78b \
  --model-allowlist llama-3.1-8b \
  --phase2b-core-synthetic-only \
  --intervention-profile strong_only \
  --random-draws-confirmatory 3 \
  --print-summary

.venv/bin/python scripts/experiment2_quick_manifest_build.py \
  --input-manifest "${BASE_MANIFEST}" \
  --output-manifest "${MANIFEST_DIR}/${SLOPE_RUN_ID}.jsonl" \
  --summary-json "${MANIFEST_DIR}/${SLOPE_RUN_ID}.summary.json" \
  --run-id "${SLOPE_RUN_ID}" \
  --phase quick_slope \
  --split synthetic \
  --model llama-3.1-8b \
  --task long_range_retrieval \
  --seq-len 1024 \
  --seeds 0,1,2 \
  --force-spans 32,48,64 \
  --source-interventions none \
  --expand-interventions none,ablate_high_strong,ablate_low_strong,random_strong \
  --random-draws 3 \
  --notes-tag quick_slope_fixed_span

.venv/bin/python scripts/experiment2_quick_manifest_build.py \
  --input-manifest "${BASE_MANIFEST}" \
  --output-manifest "${MANIFEST_DIR}/${DOSE_RUN_ID}.jsonl" \
  --summary-json "${MANIFEST_DIR}/${DOSE_RUN_ID}.summary.json" \
  --run-id "${DOSE_RUN_ID}" \
  --phase quick_dose \
  --split synthetic \
  --model llama-3.1-8b \
  --task long_range_retrieval \
  --seq-len 1024 \
  --seeds 0,1,2 \
  --force-spans 32,48,64 \
  --source-interventions none \
  --expand-interventions none,ablate_high_medium,ablate_high_strong,ablate_low_medium,ablate_low_strong,random_strong \
  --random-draws 3 \
  --notes-tag quick_dose_fixed_span

if tmux has-session -t exp2_qslope_g0 2>/dev/null; then tmux kill-session -t exp2_qslope_g0; fi
if tmux has-session -t exp2_qslope_g1 2>/dev/null; then tmux kill-session -t exp2_qslope_g1; fi
if tmux has-session -t exp2_qdose_g0 2>/dev/null; then tmux kill-session -t exp2_qdose_g0; fi
if tmux has-session -t exp2_qdose_g1 2>/dev/null; then tmux kill-session -t exp2_qdose_g1; fi
if tmux has-session -t exp2_qslope_report 2>/dev/null; then tmux kill-session -t exp2_qslope_report; fi
if tmux has-session -t exp2_qdose_report 2>/dev/null; then tmux kill-session -t exp2_qdose_report; fi

SLOPE_MANIFEST="${MANIFEST_DIR}/${SLOPE_RUN_ID}.jsonl"
DOSE_MANIFEST="${MANIFEST_DIR}/${DOSE_RUN_ID}.jsonl"

tmux new-session -d -s exp2_qslope_g0 \
  "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiment2/run.py --mode execute --manifest '${SLOPE_MANIFEST}' --output-root '${OUTPUT_ROOT}' --device cuda:0 --kernel-engine optimized --centered-compute defer --synthetic-eval-mode restricted --candidate-size 10 --h12-endpoint-policy co_primary_raw_headroom --track-a-enabled false --intervention-profile strong_only --synthetic-count 100 --batch-size-synth 24 --feasibility-task-only --seed-start 0 --seed-end 1 --print-summary |& tee '${LOG_ROOT}/exp2_qslope_g0.log'"

tmux new-session -d -s exp2_qslope_g1 \
  "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiment2/run.py --mode execute --manifest '${SLOPE_MANIFEST}' --output-root '${OUTPUT_ROOT}' --device cuda:0 --kernel-engine optimized --centered-compute defer --synthetic-eval-mode restricted --candidate-size 10 --h12-endpoint-policy co_primary_raw_headroom --track-a-enabled false --intervention-profile strong_only --synthetic-count 100 --batch-size-synth 24 --feasibility-task-only --seed-start 2 --seed-end 2 --print-summary |& tee '${LOG_ROOT}/exp2_qslope_g1.log'"

tmux new-session -d -s exp2_qdose_g0 \
  "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiment2/run.py --mode execute --manifest '${DOSE_MANIFEST}' --output-root '${OUTPUT_ROOT}' --device cuda:0 --kernel-engine optimized --centered-compute defer --synthetic-eval-mode restricted --candidate-size 10 --h12-endpoint-policy co_primary_raw_headroom --track-a-enabled false --intervention-profile full --synthetic-count 100 --batch-size-synth 24 --feasibility-task-only --seed-start 0 --seed-end 1 --print-summary |& tee '${LOG_ROOT}/exp2_qdose_g0.log'"

tmux new-session -d -s exp2_qdose_g1 \
  "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiment2/run.py --mode execute --manifest '${DOSE_MANIFEST}' --output-root '${OUTPUT_ROOT}' --device cuda:0 --kernel-engine optimized --centered-compute defer --synthetic-eval-mode restricted --candidate-size 10 --h12-endpoint-policy co_primary_raw_headroom --track-a-enabled false --intervention-profile full --synthetic-count 100 --batch-size-synth 24 --feasibility-task-only --seed-start 2 --seed-end 2 --print-summary |& tee '${LOG_ROOT}/exp2_qdose_g1.log'"

tmux new-session -d -s exp2_qslope_report \
  "cd '${ROOT_DIR}' && while tmux has-session -t exp2_qslope_g0 2>/dev/null || tmux has-session -t exp2_qslope_g1 2>/dev/null; do sleep 30; done && \
   .venv/bin/python scripts/experiment2_quick_report.py --aggregate-task-metrics '${OUTPUT_ROOT}/quick_slope/${SLOPE_RUN_ID}/aggregate_task_metrics.parquet' --output-dir '${REPORT_DIR}/slope' --model llama-3.1-8b --task long_range_retrieval --spans 32,48,64 --seeds 0,1,2 --interventions none,ablate_high_strong,ablate_low_strong,random_strong --monotonic-interventions ablate_high_strong,ablate_low_strong |& tee '${LOG_ROOT}/exp2_qslope_report.log'"

tmux new-session -d -s exp2_qdose_report \
  "cd '${ROOT_DIR}' && while tmux has-session -t exp2_qdose_g0 2>/dev/null || tmux has-session -t exp2_qdose_g1 2>/dev/null; do sleep 30; done && \
   .venv/bin/python scripts/experiment2_quick_report.py --aggregate-task-metrics '${OUTPUT_ROOT}/quick_dose/${DOSE_RUN_ID}/aggregate_task_metrics.parquet' --output-dir '${REPORT_DIR}/dose' --model llama-3.1-8b --task long_range_retrieval --spans 32,48,64 --seeds 0,1,2 --interventions none,ablate_high_medium,ablate_high_strong,ablate_low_medium,ablate_low_strong,random_strong --monotonic-interventions ablate_high_medium,ablate_high_strong,ablate_low_medium,ablate_low_strong |& tee '${LOG_ROOT}/exp2_qdose_report.log'"

cat <<EOF
[stage1] launched tmux sessions on GPUs 0/1:
  - exp2_qslope_g0
  - exp2_qslope_g1
  - exp2_qdose_g0
  - exp2_qdose_g1

[stage1] manifests:
  - ${SLOPE_MANIFEST}
  - ${DOSE_MANIFEST}
EOF
