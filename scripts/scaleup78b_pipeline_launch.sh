#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${1:-scaleup78b_$(date +%Y%m%d_%H%M%S)}"
FEAS_RUN_ID="${2:-exp2_feas_scaleup_${RUN_TAG}}"
SUPERSEDE_RUN_TAG="${3:-}"

mkdir -p "logs/scaleup/${RUN_TAG}"

if [[ -n "$SUPERSEDE_RUN_TAG" ]] && [[ -d "logs/scaleup/${SUPERSEDE_RUN_TAG}" ]]; then
  cat > "logs/scaleup/${SUPERSEDE_RUN_TAG}/superseded_by_efficiency_relaunch.json" <<JSON
{
  "superseded": true,
  "superseded_at": "$(date '+%Y-%m-%dT%H:%M:%S%z')",
  "superseded_by_run_tag": "$RUN_TAG",
  "note": "Superseded by reduced-scope scaleup relaunch (GPUs 0/1/2 only, seed-sharded feasibility)."
}
JSON
fi

for session in scaleup78b_g0 scaleup78b_g1 scaleup78b_g2 scaleup78b_exp2_pipe; do
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux kill-session -t "$session"
  fi
done

tmux new-session -d -s scaleup78b_g0 "cd '$ROOT_DIR' && bash scripts/scaleup78b_exp1_worker.sh 0 olmo-2-7b '$RUN_TAG'"
tmux new-session -d -s scaleup78b_g1 "cd '$ROOT_DIR' && bash scripts/scaleup78b_exp1_worker.sh 1 llama-3.1-8b '$RUN_TAG'"
tmux new-session -d -s scaleup78b_g2 "cd '$ROOT_DIR' && bash scripts/scaleup78b_exp1_worker.sh 2 gemma-7b '$RUN_TAG'"
tmux new-session -d -s scaleup78b_exp2_pipe "cd '$ROOT_DIR' && bash scripts/scaleup78b_exp2_full_pipeline.sh '$RUN_TAG' '$FEAS_RUN_ID' results/experiment2"

cat <<EOF
{
  "status": "launched",
  "run_tag": "$RUN_TAG",
  "feas_run_id": "$FEAS_RUN_ID",
  "sessions": ["scaleup78b_g0", "scaleup78b_g1", "scaleup78b_g2", "scaleup78b_exp2_pipe"],
  "logs_dir": "logs/scaleup/$RUN_TAG"
}
EOF
