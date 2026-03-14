#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${1:-results/experiment2}"
SUPERSEDE_RUN_TAG="${2:-}"
RUN_TAG_C="${3:-scaleup78b_optc_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG_B="${4:-scaleup78b_optb_$(date +%Y%m%d_%H%M%S)}"

FEAS_RUN_C="exp2_feas_scaleup_optc_${RUN_TAG_C}"
FEAS_RUN_B="exp2_feas_scaleup_optb_${RUN_TAG_B}"

mkdir -p "logs/scaleup/${RUN_TAG_C}" "logs/scaleup/${RUN_TAG_B}"

if [[ -n "$SUPERSEDE_RUN_TAG" ]] && [[ -d "logs/scaleup/${SUPERSEDE_RUN_TAG}" ]]; then
  cat > "logs/scaleup/${SUPERSEDE_RUN_TAG}/superseded_by_optionc_optionb_restart.json" <<JSON
{
  "superseded": true,
  "superseded_at": "$(date '+%Y-%m-%dT%H:%M:%S%z')",
  "superseded_by_run_tags": ["$RUN_TAG_C", "$RUN_TAG_B"],
  "note": "Superseded by span-overlap remediation restart (Option C then Option B on GPUs 0/1/2)."
}
JSON
fi

for session in \
  scaleup78b_g0 scaleup78b_g1 scaleup78b_g2 scaleup78b_exp2_pipe \
  scaleup78b_optc_pipe scaleup78b_optb_pipe scaleup78b_optcb_master \
  scaleup78b_feas_g0 scaleup78b_feas_g1 scaleup78b_feas_g2 \
  scaleup78b_p2a_pilot_g0 scaleup78b_p2a_pilot_g1 scaleup78b_p2a_pilot_g2 \
  scaleup78b_p2a_exec_g0 scaleup78b_p2a_exec_g1 scaleup78b_p2a_exec_g2 \
  scaleup78b_p2a_posthoc_g0 scaleup78b_p2a_posthoc_g1 scaleup78b_p2a_posthoc_g2 \
  exp2b_g0 exp2b_g1 exp2b_g2 exp2q_g0 exp2q_g1 exp2q_g2 \
  exp2b_prepass_g0 exp2b_prepass_g1 exp2b_prepass_g2 \
  exp2p_g0 exp2p_g1 exp2p_g2; do
  tmux kill-session -t "$session" 2>/dev/null || true
done

tmux new-session -d -s scaleup78b_optcb_master \
  "cd '$ROOT_DIR' && bash scripts/scaleup_optionc_llama_pipeline.sh '$RUN_TAG_C' '$OUT_ROOT' '$FEAS_RUN_C' && bash scripts/scaleup_optionb_crossmodel_pipeline.sh '$RUN_TAG_B' '$OUT_ROOT' '$FEAS_RUN_B'"

cat <<EOF
{
  "status": "launched",
  "session": "scaleup78b_optcb_master",
  "run_tags": {
    "option_c": "$RUN_TAG_C",
    "option_b": "$RUN_TAG_B"
  },
  "feas_run_ids": {
    "option_c": "$FEAS_RUN_C",
    "option_b": "$FEAS_RUN_B"
  },
  "out_root": "$OUT_ROOT"
}
EOF
