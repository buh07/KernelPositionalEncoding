#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_SESSION="${RUN_SESSION:-exp6}"
POLL_SEC="${POLL_SEC:-60}"

# Block until currently active Experiment 4A/5A/5C runs complete.
BLOCKER_PATTERN="${BLOCKER_PATTERN:-experiment4\\.exp4a_si_aware_lora|experiment5\\.exp5a_cross_tokenizer_si_profiling|experiment5\\.exp5c_synthetic_tokenizer_perturbation}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/experiment6/queue_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/experiment6/queue_latest"
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1

echo "[$(date)] Exp6 queue watcher started."
echo "[$(date)] Waiting for blockers matching pattern: $BLOCKER_PATTERN"

while pgrep -af "$BLOCKER_PATTERN" >/dev/null 2>&1; do
  echo "[$(date)] Blocking runs still active; sleeping ${POLL_SEC}s..."
  pgrep -af "$BLOCKER_PATTERN" || true
  sleep "$POLL_SEC"
done

echo "[$(date)] No blockers detected. Launching Experiment 6 session '$RUN_SESSION'."
tmux kill-session -t "$RUN_SESSION" 2>/dev/null || true
if ! tmux new-session -d -s "$RUN_SESSION" -n "exp6_gpu0" \
  "cd '$ROOT_DIR' && bash scripts/experiment6_full_gpu0.sh"; then
  rc=$?
  echo "[$(date)] ERROR: Failed to create tmux session '$RUN_SESSION' (rc=$rc)."
  tmux ls || true
  exit "$rc"
fi

sleep 1
if tmux has-session -t "$RUN_SESSION" 2>/dev/null; then
  echo "[$(date)] Experiment 6 launched in tmux session '$RUN_SESSION'."
  tmux ls || true
else
  echo "[$(date)] ERROR: Session '$RUN_SESSION' did not remain active after launch."
  tmux ls || true
  exit 1
fi
