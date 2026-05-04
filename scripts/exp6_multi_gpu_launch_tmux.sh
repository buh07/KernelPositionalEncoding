#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION="exp6_multi"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/experiment6/multi_$TS"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$ROOT_DIR/logs/experiment6/multi_latest"

# Reconcile existing prototype session to avoid overlapping writes.
if tmux has-session -t exp6_proto 2>/dev/null; then
  if pgrep -af "experiment6\\.exp6[abcde]_" >/dev/null 2>&1; then
    echo "[$(date)] stopping overlapping exp6_proto session"
    tmux kill-session -t exp6_proto || true
  fi
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

run_worker_cmd() {
  local gpu="$1"
  local worker_idx="$2"
  local num_workers="$3"
  local models="$4"
  local tag="$5"
  cat <<EOF
cd '$ROOT_DIR' && \
export CUDA_VISIBLE_DEVICES=$gpu && \
export PYTHONUNBUFFERED=1 && \
export HF_HOME='/jumbo/lisp/f004ndc/huggingface' && \
export HF_DATASETS_CACHE="\$HF_HOME/datasets" && \
export HUGGINGFACE_HUB_CACHE="\$HF_HOME/hub" && \
export TRANSFORMERS_CACHE="\$HF_HOME/transformers" && \
export XDG_CACHE_HOME="\$HF_HOME/xdg_cache" && \
mkdir -p "\$HF_DATASETS_CACHE" "\$HUGGINGFACE_HUB_CACHE" "\$TRANSFORMERS_CACHE" "\$XDG_CACHE_HOME" '$LOG_DIR' && \
.venv/bin/python -m experiment6.distributed run \
  --models '$models' \
  --experiments all \
  --seeds 0,1,2 \
  --gpu $gpu \
  --worker-index $worker_idx \
  --num-workers $num_workers \
  --retry-failed-once \
  --shard-root results/experiment6_shards \
  --canonical-root results/experiment6 \
  --worker-tag '$tag' \
  | tee -a '$LOG_DIR/$tag.log'
EOF
}

# GPU0: 7B models sequentially (safe memory policy).
tmux new-session -d -s "$SESSION" -n "gpu0_7b" \
  "$(run_worker_cmd 0 0 1 "llama-3.1-8b,olmo-2-7b" "gpu0_7b")"

# GPUs 2-7: tiny model sharded in parallel across 6 workers.
TINY_GPUS=(2 3 4 5 6 7)
for idx in "${!TINY_GPUS[@]}"; do
  gpu="${TINY_GPUS[$idx]}"
  win="gpu${gpu}_tiny"
  tag="gpu${gpu}_tiny_w${idx}"
  tmux new-window -t "$SESSION" -n "$win" \
    "$(run_worker_cmd "$gpu" "$idx" 6 "tinyllama-1.1b" "$tag")"
done

# Merge watcher window: merges shard outputs after all workers finish.
tmux new-window -t "$SESSION" -n "merge" \
  "cd '$ROOT_DIR' && POLL_SEC=60 bash scripts/experiment6_multi_gpu_merge_wait.sh | tee -a '$LOG_DIR/merge.log'"

echo "tmux session '$SESSION' launched."
echo "  Attach: tmux attach -t $SESSION"
echo "  Check:  tmux capture-pane -t $SESSION:gpu0_7b -p | tail -40"
echo "  Logs:   $LOG_DIR"

