#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if tmux has-session -t exp2_qslope_g0 2>/dev/null || tmux has-session -t exp2_qslope_g1 2>/dev/null || \
   tmux has-session -t exp2_qdose_g0 2>/dev/null || tmux has-session -t exp2_qdose_g1 2>/dev/null; then
  echo "[stage2] Stage 1 sessions are still running. Finish Stage 1 before launching Stage 2."
  exit 1
fi

RUN_TAG="${1:-quick_stage2_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="results/experiment2/quick/${RUN_TAG}"
LOG_ROOT="logs/experiment2/quick/${RUN_TAG}"
MANIFEST_DIR="${OUTPUT_ROOT}/manifests"
REPORT_DIR="${OUTPUT_ROOT}/reports"
mkdir -p "${LOG_ROOT}" "${MANIFEST_DIR}" "${REPORT_DIR}/pair_sweep" "${REPORT_DIR}/natural"

BASE_RUN_ID="${RUN_TAG}_base"
PAIR_RUN_ID="${RUN_TAG}_pair"
NAT_RUN_ID="${RUN_TAG}_natural"

BASE_MANIFEST="${OUTPUT_ROOT}/phase2b/${BASE_RUN_ID}/manifest.jsonl"
PAIR_MANIFEST="${MANIFEST_DIR}/${PAIR_RUN_ID}.jsonl"

PAIRS=(0 8 16 24 32 40 48 56)
PAIR_INTERVENTIONS="none"
for p in "${PAIRS[@]}"; do
  PAIR_INTERVENTIONS="${PAIR_INTERVENTIONS},ablate_pair_${p}"
done
PAIR_INTERVENTIONS="${PAIR_INTERVENTIONS},random_pair"

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
  --output-manifest "${PAIR_MANIFEST}" \
  --summary-json "${MANIFEST_DIR}/${PAIR_RUN_ID}.summary.json" \
  --run-id "${PAIR_RUN_ID}" \
  --phase quick_pair \
  --split synthetic \
  --model llama-3.1-8b \
  --task long_range_retrieval \
  --seq-len 1024 \
  --seeds 0,1,2 \
  --force-spans 32,48,64 \
  --source-interventions none \
  --expand-interventions "${PAIR_INTERVENTIONS}" \
  --random-draws 3 \
  --notes-tag quick_pair_sparse_sweep

if tmux has-session -t exp2_qpair_g0 2>/dev/null; then tmux kill-session -t exp2_qpair_g0; fi
if tmux has-session -t exp2_qpair_g1 2>/dev/null; then tmux kill-session -t exp2_qpair_g1; fi
if tmux has-session -t exp2_qnat_g0 2>/dev/null; then tmux kill-session -t exp2_qnat_g0; fi
if tmux has-session -t exp2_qnat_g1 2>/dev/null; then tmux kill-session -t exp2_qnat_g1; fi

tmux new-session -d -s exp2_qpair_g0 \
  "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiment2/run.py --mode execute --manifest '${PAIR_MANIFEST}' --output-root '${OUTPUT_ROOT}' --device cuda:0 --kernel-engine optimized --centered-compute defer --synthetic-eval-mode restricted --candidate-size 10 --h12-endpoint-policy co_primary_raw_headroom --track-a-enabled false --intervention-profile strong_only --synthetic-count 100 --batch-size-synth 24 --feasibility-task-only --seed-start 0 --seed-end 1 --print-summary |& tee '${LOG_ROOT}/exp2_qpair_g0.log'"

tmux new-session -d -s exp2_qpair_g1 \
  "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiment2/run.py --mode execute --manifest '${PAIR_MANIFEST}' --output-root '${OUTPUT_ROOT}' --device cuda:0 --kernel-engine optimized --centered-compute defer --synthetic-eval-mode restricted --candidate-size 10 --h12-endpoint-policy co_primary_raw_headroom --track-a-enabled false --intervention-profile strong_only --synthetic-count 100 --batch-size-synth 24 --feasibility-task-only --seed-start 2 --seed-end 2 --print-summary |& tee '${LOG_ROOT}/exp2_qpair_g1.log'"

tmux new-session -d -s exp2_qnat_g0 \
  "cd '${ROOT_DIR}' && while tmux has-session -t exp2_qpair_g0 2>/dev/null || tmux has-session -t exp2_qpair_g1 2>/dev/null; do sleep 30; done && \
   .venv/bin/python scripts/experiment2_quick_report.py --aggregate-task-metrics '${OUTPUT_ROOT}/quick_pair/${PAIR_RUN_ID}/aggregate_task_metrics.parquet' --output-dir '${REPORT_DIR}/pair_sweep' --model llama-3.1-8b --task long_range_retrieval --spans 32,48,64 --seeds 0,1,2 --interventions '${PAIR_INTERVENTIONS}' --monotonic-interventions ablate_pair_0,ablate_pair_56,random_pair |& tee '${LOG_ROOT}/exp2_qpair_report.log' && \
   CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiment2_quick_seminatural_probe.py --run-tag '${NAT_RUN_ID}_part01' --output-root '${OUTPUT_ROOT}/quick_natural_shards' --logs-root '${LOG_ROOT}' --model llama-3.1-8b --dataset wiki40b_en_pre2019 --seq-len 1024 --spans 32,48,64 --seeds 0,1,2 --seed-start 0 --seed-end 1 --interventions none,ablate_high_strong,ablate_low_strong,random_strong --random-draws 3 --synthetic-count 100 --candidate-size 10 --batch-size 1 --device cuda:0 --data-root data |& tee '${LOG_ROOT}/exp2_qnat_g0.log' && \
   touch '${REPORT_DIR}/natural/part01.done' && \
   while [ ! -f '${REPORT_DIR}/natural/part2.done' ]; do sleep 30; done && \
   .venv/bin/python scripts/experiment2_quick_natural_merge.py --inputs '${OUTPUT_ROOT}/quick_natural_shards/${NAT_RUN_ID}_part01,${OUTPUT_ROOT}/quick_natural_shards/${NAT_RUN_ID}_part2' --output-dir '${REPORT_DIR}/natural' |& tee '${LOG_ROOT}/exp2_qnat_merge.log'"

tmux new-session -d -s exp2_qnat_g1 \
  "cd '${ROOT_DIR}' && while tmux has-session -t exp2_qpair_g0 2>/dev/null || tmux has-session -t exp2_qpair_g1 2>/dev/null; do sleep 30; done && \
   CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/experiment2_quick_seminatural_probe.py --run-tag '${NAT_RUN_ID}_part2' --output-root '${OUTPUT_ROOT}/quick_natural_shards' --logs-root '${LOG_ROOT}' --model llama-3.1-8b --dataset wiki40b_en_pre2019 --seq-len 1024 --spans 32,48,64 --seeds 0,1,2 --seed-start 2 --seed-end 2 --interventions none,ablate_high_strong,ablate_low_strong,random_strong --random-draws 3 --synthetic-count 100 --candidate-size 10 --batch-size 1 --device cuda:0 --data-root data |& tee '${LOG_ROOT}/exp2_qnat_g1.log' && \
   touch '${REPORT_DIR}/natural/part2.done'"

cat <<EOF
[stage2] launched tmux sessions on GPUs 0/1:
  - exp2_qpair_g0
  - exp2_qpair_g1
  - exp2_qnat_g0
  - exp2_qnat_g1

[stage2] artifacts root:
  ${OUTPUT_ROOT}
EOF
