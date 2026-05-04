#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

POLL_SEC="${POLL_SEC:-60}"
PATTERN="${PATTERN:-experiment6\\.distributed run}"

echo "[$(date)] exp6 merge watcher started (poll=${POLL_SEC}s)"
echo "[$(date)] waiting for workers pattern: $PATTERN"

while pgrep -af "$PATTERN" >/dev/null 2>&1; do
  echo "[$(date)] workers still running; sleeping ${POLL_SEC}s..."
  sleep "$POLL_SEC"
done

echo "[$(date)] no worker processes detected; running merge"
.venv/bin/python -m experiment6.distributed merge \
  --models all \
  --experiments all \
  --seeds 0,1,2 \
  --shard-root results/experiment6_shards \
  --canonical-root results/experiment6
RC=$?
echo "[$(date)] merge finished rc=$RC"
exit "$RC"

