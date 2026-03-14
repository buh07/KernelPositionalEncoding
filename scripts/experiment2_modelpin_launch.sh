#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:?usage: scripts/experiment2_modelpin_launch.sh <run_id>}"
LOG_DIR="logs/experiment2"
mkdir -p "$LOG_DIR"

phase_shard() {
  local phase="$1"
  local model="$2"
  echo "results/experiment2/${phase}/${RUN_ID}/model_shards/${model}.jsonl"
}

G3_MANIFESTS=(
  "$(phase_shard phase2a llama-3.2-1b)"
  "$(phase_shard phase2b llama-3.2-1b)"
  "$(phase_shard phase2c llama-3.2-1b)"
  "$(phase_shard phase2c tinyllama-nope-1.1b)"
)
G4_MANIFESTS=(
  "$(phase_shard phase2b tinyllama-1.1b)"
  "$(phase_shard phase2c tinyllama-1.1b)"
  "$(phase_shard phase2c gpt2-medium)"
)
G5_MANIFESTS=(
  "$(phase_shard phase2b olmo-1b)"
  "$(phase_shard phase2c olmo-1b)"
  "$(phase_shard phase2c gpt2-small)"
)

for manifest in "${G3_MANIFESTS[@]}" "${G4_MANIFESTS[@]}" "${G5_MANIFESTS[@]}"; do
  if [[ ! -f "$manifest" ]]; then
    echo "Missing shard manifest: $manifest" >&2
    exit 1
  fi
done

tmux new-session -d -s exp2_g3 \
  "CENTERED_COMPUTE=defer BATCH_SIZE_SYNTH=8 BATCH_SIZE_TIER1=8 bash scripts/experiment2_modelpin_execute.sh 3 ${G3_MANIFESTS[*]} |& tee ${LOG_DIR}/${RUN_ID}_g3_modelpin.log"
tmux new-session -d -s exp2_g4 \
  "CENTERED_COMPUTE=defer BATCH_SIZE_SYNTH=8 BATCH_SIZE_TIER1=8 bash scripts/experiment2_modelpin_execute.sh 4 ${G4_MANIFESTS[*]} |& tee ${LOG_DIR}/${RUN_ID}_g4_modelpin.log"
tmux new-session -d -s exp2_g5 \
  "CENTERED_COMPUTE=defer BATCH_SIZE_SYNTH=8 BATCH_SIZE_TIER1=8 bash scripts/experiment2_modelpin_execute.sh 5 ${G5_MANIFESTS[*]} |& tee ${LOG_DIR}/${RUN_ID}_g5_modelpin.log"

echo "Launched tmux sessions for run_id=${RUN_ID}: exp2_g3, exp2_g4, exp2_g5"
