#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
PY="$ROOT/.venv/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/reinforce_exp3/e31_$STAMP"
mkdir -p "$LOG_DIR"

SESSIONS=(rexp3_e31_gpu0 rexp3_e31_gpu1 rexp3_e31_gpu2 rexp3_e31_gpu3)
for s in "${SESSIONS[@]}"; do
  tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true
done

# GPU0: E31D smoke -> full (new inference)
CMD0=$(cat <<EOF
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
echo "[$(date)] E31D smoke start" | tee "$LOG_DIR/e31d.log"
"$PY" -u reinforce_exp3/scripts/run_e31d_llama_naturalistic_importance_control.py \
  --device cuda:0 \
  --output-root "$ROOT/results/reinforce_exp3/E31d_llama_naturalistic_importance_control_smoke" \
  --total-prompts 18 \
  --batch-size 2 \
  --n-control-trials 1 \
  --smoke 2>&1 | tee -a "$LOG_DIR/e31d.log"
echo "[$(date)] E31D full start" | tee -a "$LOG_DIR/e31d.log"
"$PY" -u reinforce_exp3/scripts/run_e31d_llama_naturalistic_importance_control.py \
  --device cuda:0 \
  --output-root "$ROOT/results/reinforce_exp3/E31d_llama_naturalistic_importance_control" \
  --total-prompts 60 \
  --batch-size 2 \
  --n-control-trials 2 2>&1 | tee -a "$LOG_DIR/e31d.log"
echo "[$(date)] E31D done" | tee -a "$LOG_DIR/e31d.log"
EOF
)

# GPU1: E31A breadth consolidation
CMD1=$(cat <<EOF
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=1
echo "[$(date)] E31A start" | tee "$LOG_DIR/e31a.log"
"$PY" -u reinforce_exp3/scripts/run_e31a_breadth_consolidation.py \
  --output-root "$ROOT/results/reinforce_exp3/E31a_breadth_consolidation" 2>&1 | tee -a "$LOG_DIR/e31a.log"
echo "[$(date)] E31A done" | tee -a "$LOG_DIR/e31a.log"
EOF
)

# GPU2: E31B waits for E31A table, then runs
CMD2=$(cat <<EOF
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=2
TABLE="$ROOT/results/reinforce_exp3/E31a_breadth_consolidation/model_breadth_r2_table.csv"
for i in {1..120}; do
  [[ -f "\$TABLE" ]] && break
  sleep 2
done
if [[ ! -f "\$TABLE" ]]; then
  echo "E31B hard_fail_reason: missing E31A table after wait" | tee "$LOG_DIR/e31b.log"
  exit 2
fi
echo "[$(date)] E31B start" | tee "$LOG_DIR/e31b.log"
"$PY" -u reinforce_exp3/scripts/run_e31b_coherence_refresh.py \
  --e31a-table "\$TABLE" \
  --output-root "$ROOT/results/reinforce_exp3/E31b_coherence_refresh" \
  --seq-len 512 2>&1 | tee -a "$LOG_DIR/e31b.log"
echo "[$(date)] E31B done" | tee -a "$LOG_DIR/e31b.log"
EOF
)

# GPU3: E31C waits for E31A table, then runs
CMD3=$(cat <<EOF
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=3
TABLE="$ROOT/results/reinforce_exp3/E31a_breadth_consolidation/model_breadth_r2_table.csv"
for i in {1..120}; do
  [[ -f "\$TABLE" ]] && break
  sleep 2
done
if [[ ! -f "\$TABLE" ]]; then
  echo "E31C hard_fail_reason: missing E31A table after wait" | tee "$LOG_DIR/e31c.log"
  exit 2
fi
echo "[$(date)] E31C start" | tee "$LOG_DIR/e31c.log"
"$PY" -u reinforce_exp3/scripts/run_e31c_functional_coupling_reanalysis.py \
  --e31a-table "\$TABLE" \
  --output-root "$ROOT/results/reinforce_exp3/E31c_functional_coupling_reanalysis" 2>&1 | tee -a "$LOG_DIR/e31c.log"
echo "[$(date)] E31C done" | tee -a "$LOG_DIR/e31c.log"
EOF
)

# Launch detached sessions
tmux new-session -d -s rexp3_e31_gpu0 "bash -lc '$CMD0'"
tmux new-session -d -s rexp3_e31_gpu1 "bash -lc '$CMD1'"
tmux new-session -d -s rexp3_e31_gpu2 "bash -lc '$CMD2'"
tmux new-session -d -s rexp3_e31_gpu3 "bash -lc '$CMD3'"

echo "Launched E31 sessions: ${SESSIONS[*]}"
echo "Logs: $LOG_DIR"
echo "$LOG_DIR" > "$ROOT/logs/reinforce_exp3/E31_LAST_LOG_DIR.txt"
