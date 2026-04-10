#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
MODEL="${1:-llama-3.1-8b}"
SEEDS="${2:-0,1,2,3,4,5,6}"
POLL_SEC="${3:-120}"
AUTO_RUN_I_IF_CLEAN="${4:-true}"

LOG_DIR="$ROOT/logs/experiment3_phase2_multiseed_b/finalize_${MODEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

all_present() {
  local missing=0
  for seed in "${SEED_ARR[@]}"; do
    local s
    s="$(echo "$seed" | xargs)"
    [[ -z "$s" ]] && continue
    local p="results/experiment3_phase2/exp3p2b_trivial_feature_control_multiseed/seed${s}/${MODEL}/synthetic_boundary_results.json"
    if [[ ! -f "$p" ]]; then
      echo "[$(date)] waiting: missing $p" | tee -a "$LOG_DIR/finalize.log"
      missing=1
    fi
  done
  return "$missing"
}

echo "[$(date)] finalize watcher started model=$MODEL seeds=$SEEDS" | tee -a "$LOG_DIR/finalize.log"
while true; do
  if all_present; then
    break
  fi
  sleep "$POLL_SEC"
done

echo "[$(date)] all seed artifacts present; running pooled adjudication" | tee -a "$LOG_DIR/finalize.log"
"$PY" -u experiment3/phase2/exp3p2b_multiseed_gate_adjudication.py \
  --model "$MODEL" \
  --seeds "$SEEDS" \
  --input-root results/experiment3_phase2/exp3p2b_trivial_feature_control_multiseed \
  --canonical-output-root results/experiment3_phase2/exp3p2b_trivial_feature_control \
  2>&1 | tee "$LOG_DIR/multiseed_adjudication.log"

SUMMARY="results/experiment3_phase2/exp3p2b_trivial_feature_control/${MODEL}/multiseed_gate_summary.json"
ASSESSMENT=""
if [[ -f "$SUMMARY" ]]; then
  ASSESSMENT="$(jq -r '.assessment // ""' "$SUMMARY")"
fi

echo "[$(date)] pooled assessment=${ASSESSMENT}" | tee -a "$LOG_DIR/finalize.log"

if [[ "$AUTO_RUN_I_IF_CLEAN" == "true" && "$ASSESSMENT" == "stable_clean" ]]; then
  echo "[$(date)] assessment=stable_clean -> launching strict Llama 3P2-I" | tee -a "$LOG_DIR/finalize.log"
  "$PY" -u experiment3/phase2/exp3p2i_tokenizer_corpus_invariance.py \
    --model llama-3.1-8b \
    --device cuda:0 \
    --seq-len 512 \
    --seq-per-corpus 16 \
    --adversarial-positions 200 \
    --opus-max-rows 20000 \
    --output-root results/experiment3_phase2/exp3p2i_tokenizer_corpus \
    2>&1 | tee "$LOG_DIR/3p2i_llama_strict.log"
fi

echo "[$(date)] finalize complete" | tee -a "$LOG_DIR/finalize.log"
