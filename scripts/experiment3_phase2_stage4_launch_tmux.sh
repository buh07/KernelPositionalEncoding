#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

for s in exp3p2_stage4_llama exp3p2_stage4_olmo; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

tmux new-session -d -s exp3p2_stage4_llama "cd '$ROOT' && bash scripts/experiment3_phase2_stage4_gpu0.sh"
tmux new-session -d -s exp3p2_stage4_olmo "cd '$ROOT' && bash scripts/experiment3_phase2_stage4_gpu1.sh"

echo "Launched stage4 tmux sessions: exp3p2_stage4_llama, exp3p2_stage4_olmo"
echo "Monitor with:"
echo "  tmux ls"
echo "  bash scripts/experiment3_phase2_stage4_status.sh"
