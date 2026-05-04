#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/Kernel PE"
RUN_ID="${1:?run_id}"
RUN_ROOT="results/reinforce_exp/runs/${RUN_ID}"
cd "$ROOT"

for s in g0 g1 g2 g3 g4 g5 g6 g7 r12_merge r2b_finalize regression postprocess; do
  tmux kill-session -t "n26opt_${RUN_ID}_${s}" 2>/dev/null || true
done

tmux new-session -d -s "n26opt_${RUN_ID}_g0" "bash '$ROOT/$RUN_ROOT/worker_g0.sh' '$RUN_ROOT'"
tmux new-session -d -s "n26opt_${RUN_ID}_g1" "bash '$ROOT/$RUN_ROOT/worker_g1.sh' '$RUN_ROOT'"
tmux new-session -d -s "n26opt_${RUN_ID}_g2" "bash '$ROOT/$RUN_ROOT/worker_g2.sh' '$RUN_ROOT'"
tmux new-session -d -s "n26opt_${RUN_ID}_g3" "bash '$ROOT/$RUN_ROOT/worker_g3.sh' '$RUN_ROOT'"
tmux new-session -d -s "n26opt_${RUN_ID}_g4" "bash '$ROOT/$RUN_ROOT/worker_g4.sh' '$RUN_ROOT'"
tmux new-session -d -s "n26opt_${RUN_ID}_g5" "bash '$ROOT/$RUN_ROOT/worker_g5.sh' '$RUN_ROOT'"
tmux new-session -d -s "n26opt_${RUN_ID}_g6" "bash '$ROOT/$RUN_ROOT/worker_g6.sh' '$RUN_ROOT'"
tmux new-session -d -s "n26opt_${RUN_ID}_g7" "bash '$ROOT/$RUN_ROOT/worker_g7.sh' '$RUN_ROOT'"
tmux new-session -d -s "n26opt_${RUN_ID}_r12_merge" "bash '$ROOT/$RUN_ROOT/worker_r12_merge.sh' '$RUN_ROOT'"
tmux new-session -d -s "n26opt_${RUN_ID}_r2b_finalize" "bash '$ROOT/$RUN_ROOT/worker_r2b_finalize.sh' '$RUN_ROOT'"
tmux new-session -d -s "n26opt_${RUN_ID}_regression" "bash '$ROOT/$RUN_ROOT/worker_regression.sh' '$RUN_ROOT'"
tmux new-session -d -s "n26opt_${RUN_ID}_postprocess" "bash '$ROOT/$RUN_ROOT/worker_postprocess.sh' '$RUN_ID'"

echo "Launched n26opt sessions for $RUN_ID"
tmux ls | rg "n26opt_${RUN_ID}_"
