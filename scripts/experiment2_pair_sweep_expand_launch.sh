#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${1:-quick_pair_expand_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0 1}"
OUTPUT_ROOT="results/experiment2/quick/${RUN_TAG}"
LOG_ROOT="logs/experiment2/quick/${RUN_TAG}"
MANIFEST_DIR="${OUTPUT_ROOT}/manifests"
REPORT_DIR="${OUTPUT_ROOT}/reports/pair_expanded"
BASE_RUN_ID="${RUN_TAG}_base"
PAIR_RUN_ID="${RUN_TAG}_pair"

mkdir -p "${LOG_ROOT}" "${MANIFEST_DIR}" "${REPORT_DIR}"

read -r -a GPUS <<< "${GPU_LIST//,/ }"
if (( ${#GPUS[@]} != 2 )); then
  echo "This launcher expects exactly 2 GPUs (received: ${GPU_LIST})." >&2
  exit 1
fi
G0="${GPUS[0]}"
G1="${GPUS[1]}"

BASE_MANIFEST="${OUTPUT_ROOT}/phase2b/${BASE_RUN_ID}/manifest.jsonl"
PAIR_MANIFEST="${MANIFEST_DIR}/${PAIR_RUN_ID}.jsonl"
FAMILIES_JSON="${MANIFEST_DIR}/${PAIR_RUN_ID}.families.json"

cat > "${FAMILIES_JSON}" <<'JSON'
[
  {"model": "olmo-2-7b", "task": "long_range_retrieval", "spans": [24, 32]},
  {"model": "gemma-7b", "task": "long_range_retrieval", "spans": [16, 24]},
  {"model": "llama-3.1-8b", "task": "local_key_match"},
  {"model": "olmo-2-7b", "task": "local_key_match"},
  {"model": "gemma-7b", "task": "local_key_match"}
]
JSON

.venv/bin/python experiment2/run.py \
  --mode build \
  --phase phase2b \
  --run-id "${BASE_RUN_ID}" \
  --device cuda \
  --output-root "${OUTPUT_ROOT}" \
  --model-profile scaleup_78b \
  --phase2b-core-synthetic-only \
  --intervention-profile strong_only \
  --random-draws-confirmatory 3 \
  --print-summary | tee "${LOG_ROOT}/build.log"

.venv/bin/python scripts/experiment2_pair_sweep_manifest.py \
  --input-manifest "${BASE_MANIFEST}" \
  --output-manifest "${PAIR_MANIFEST}" \
  --summary-json "${MANIFEST_DIR}/${PAIR_RUN_ID}.summary.json" \
  --run-id "${PAIR_RUN_ID}" \
  --phase quick_pair_expand \
  --seeds 0,1,2 \
  --pair-indices 0,8,16,24,32,40,48,56 \
  --random-draws 3 \
  --families-json "${FAMILIES_JSON}" \
  --notes-tag quick_pair_expanded_matrix | tee "${LOG_ROOT}/manifest.log"

for sess in exp2_xpair_g${G0} exp2_xpair_g${G1} exp2_xpair_report; do
  tmux kill-session -t "${sess}" 2>/dev/null || true
done

tmux new-session -d -s "exp2_xpair_g${G0}" \
  "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES=${G0} .venv/bin/python experiment2/run.py --mode execute --manifest '${PAIR_MANIFEST}' --output-root '${OUTPUT_ROOT}' --device cuda:0 --kernel-engine optimized --centered-compute defer --synthetic-eval-mode restricted --candidate-size 10 --h12-endpoint-policy co_primary_raw_headroom --track-a-enabled false --intervention-profile full --synthetic-count 100 --batch-size-synth 24 --seed-start 0 --seed-end 1 --print-summary |& tee '${LOG_ROOT}/exp2_xpair_g${G0}.log'"

tmux new-session -d -s "exp2_xpair_g${G1}" \
  "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES=${G1} .venv/bin/python experiment2/run.py --mode execute --manifest '${PAIR_MANIFEST}' --output-root '${OUTPUT_ROOT}' --device cuda:0 --kernel-engine optimized --centered-compute defer --synthetic-eval-mode restricted --candidate-size 10 --h12-endpoint-policy co_primary_raw_headroom --track-a-enabled false --intervention-profile full --synthetic-count 100 --batch-size-synth 24 --seed-start 2 --seed-end 2 --print-summary |& tee '${LOG_ROOT}/exp2_xpair_g${G1}.log'"

tmux new-session -d -s exp2_xpair_report \
  "cd '${ROOT_DIR}' && while tmux has-session -t exp2_xpair_g${G0} 2>/dev/null || tmux has-session -t exp2_xpair_g${G1} 2>/dev/null; do sleep 30; done && \
   .venv/bin/python scripts/experiment2_pair_sweep_report.py --aggregate-task-metrics '${OUTPUT_ROOT}/quick_pair_expand/${PAIR_RUN_ID}/aggregate_task_metrics.parquet' --output-dir '${REPORT_DIR}' --families-json '${FAMILIES_JSON}' --pair-indices 0,8,16,24,32,40,48,56 --floor-threshold 0.15 |& tee '${LOG_ROOT}/exp2_xpair_report.log'"

cat <<EOF
[pair-expand] launched on GPUs ${G0}/${G1}
  tmux sessions:
    - exp2_xpair_g${G0}
    - exp2_xpair_g${G1}
    - exp2_xpair_report
  manifest: ${PAIR_MANIFEST}
  report dir: ${REPORT_DIR}
EOF
