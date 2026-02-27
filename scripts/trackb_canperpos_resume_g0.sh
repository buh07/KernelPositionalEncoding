#!/usr/bin/env bash
set -euo pipefail

cd "/scratch2/f004ndc/Kernel PE"
export CUDA_VISIBLE_DEVICES=0

PY=".venv/bin/python"
GROUP="track_b_canonical_perpos_v1"
MODE="canonical_per_position"

run_combo() {
  local model="$1"
  local dataset="$2"
  local seqlen="$3"
  local outdir="results/${GROUP}/${model}/${dataset}/len_${seqlen}"
  local summary="${outdir}/summary.parquet"
  if [[ -f "${summary}" ]]; then
    echo "SKIP existing summary: ${model} / ${dataset} / len=${seqlen}"
    return 0
  fi
  echo "RUN  ${model} / ${dataset} / len=${seqlen}"
  "${PY}" experiment1/run.py track-b \
    --model "${model}" \
    --dataset "${dataset}" \
    --seq-len "${seqlen}" \
    --device cuda \
    --track-b-centering-mode "${MODE}" \
    --track-b-output-group "${GROUP}"
}

# GPU 0 assignment: tinyllama-nope-1.1b (remaining) + gpt2-small (all)
run_combo tinyllama-nope-1.1b codesearchnet_python_snapshot 1024
run_combo tinyllama-nope-1.1b synthetic_random 256
run_combo tinyllama-nope-1.1b synthetic_random 1024

run_combo gpt2-small wiki40b_en_pre2019 256
run_combo gpt2-small wiki40b_en_pre2019 1024
run_combo gpt2-small codesearchnet_python_snapshot 256
run_combo gpt2-small codesearchnet_python_snapshot 1024
run_combo gpt2-small synthetic_random 256
run_combo gpt2-small synthetic_random 1024

echo "GPU0 resume batch complete."
