#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${1:-scaleup78b_optc_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${2:-results/experiment2}"
FEAS_RUN_ID="${3:-exp2_feas_scaleup_optc_${RUN_TAG}}"

LOG_DIR="logs/scaleup/${RUN_TAG}"
mkdir -p "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] launch optionC run_tag=$RUN_TAG feas_run_id=$FEAS_RUN_ID out_root=$OUT_ROOT"

MODEL_PROFILE="scaleup_78b" \
MODEL_ALLOWLIST="llama-3.1-8b" \
H12_ENDPOINT_POLICY="co_primary_raw_headroom" \
LONG_TASK_FEASIBILITY_POLICY="retrieval_fallback" \
FEAS_OFFSETS="32,48,64" \
LOCK_CANDIDATES="32,48,64" \
FLOOR_THRESHOLD="0.15" \
BATCH_SIZE_SYNTH="${BATCH_SIZE_SYNTH:-24}" \
BATCH_SIZE_TIER1="${BATCH_SIZE_TIER1:-24}" \
POSTHOC_BATCH_SIZE_SYNTH="${POSTHOC_BATCH_SIZE_SYNTH:-24}" \
POSTHOC_BATCH_SIZE_TIER1="${POSTHOC_BATCH_SIZE_TIER1:-24}" \
P2B_INTERVENTION_PROFILE="${P2B_INTERVENTION_PROFILE:-strong_only}" \
P2B_TRACK_A_ENABLED="${P2B_TRACK_A_ENABLED:-false}" \
P2B_CORE_SYNTH_ONLY="${P2B_CORE_SYNTH_ONLY:-1}" \
P2B_EXEC_SYNTHETIC_COUNT="${P2B_EXEC_SYNTHETIC_COUNT:-100}" \
P2B_GPU_LIST="${P2B_GPU_LIST:-0 1}" \
P2A_SYNTHETIC_COUNT="${P2A_SYNTHETIC_COUNT:-50}" \
POLL_SECONDS="${POLL_SECONDS:-120}" \
P2A_PILOT_MODE="${P2A_PILOT_MODE:-floor-prepass}" \
PREPASS_PARALLEL="${PREPASS_PARALLEL:-0}" \
PHASE2B_SKIP_POSTHOC="${PHASE2B_SKIP_POSTHOC:-1}" \
PHASE2B_ALLOW_CENTERED_PENDING_REANALYZE="${PHASE2B_ALLOW_CENTERED_PENDING_REANALYZE:-1}" \
bash scripts/scaleup78b_exp2_full_pipeline.sh "$RUN_TAG" "$FEAS_RUN_ID" "$OUT_ROOT" | tee -a "${LOG_DIR}/optionc_pipeline.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] optionC complete run_tag=$RUN_TAG"
