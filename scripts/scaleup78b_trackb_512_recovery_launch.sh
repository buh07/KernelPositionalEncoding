#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${1:-scaleup78b_tb512_$(date +%Y%m%d_%H%M%S)}"
SUPERSEDE_TAG="${2:-}"

mkdir -p "logs/scaleup/${RUN_TAG}"

if [[ -n "$SUPERSEDE_TAG" ]] && [[ -d "logs/scaleup/${SUPERSEDE_TAG}" ]]; then
  cat > "logs/scaleup/${SUPERSEDE_TAG}/superseded_trackb512_recovery.json" <<JSON
{
  "superseded": true,
  "superseded_at": "$(date '+%Y-%m-%dT%H:%M:%S%z')",
  "superseded_by_run_tag": "$RUN_TAG",
  "note": "Scale-up Track B recovery rerun at seq_len=512 for failed models only (llama-3.1-8b, olmo-2-7b)."
}
JSON
fi

for session in scaleup78b_tb512_g0 scaleup78b_tb512_g1; do
  tmux kill-session -t "$session" 2>/dev/null || true
done

tmux new-session -d -s scaleup78b_tb512_g0 \
  "cd '$ROOT_DIR' && bash scripts/scaleup78b_trackb_512_worker.sh 0 llama-3.1-8b '$RUN_TAG'"
tmux new-session -d -s scaleup78b_tb512_g1 \
  "cd '$ROOT_DIR' && bash scripts/scaleup78b_trackb_512_worker.sh 1 olmo-2-7b '$RUN_TAG'"

cat <<EOF
{
  "status": "launched",
  "run_tag": "$RUN_TAG",
  "sessions": ["scaleup78b_tb512_g0", "scaleup78b_tb512_g1"],
  "models": ["llama-3.1-8b", "olmo-2-7b"],
  "seq_len": 512,
  "output_group": "track_b_scaleup78b_rawcheck",
  "logs_dir": "logs/scaleup/$RUN_TAG"
}
EOF

