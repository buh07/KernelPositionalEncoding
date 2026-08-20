#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 BATCH_ID [RUN_ROOT] [--execute]" >&2
  exit 2
fi

BATCH_ID="$1"
RUN_ROOT="${2:-rebuttal/runs/${BATCH_ID}}"
MODE="${3:---dry-run}"

if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--execute" ]]; then
  echo "mode must be --dry-run or --execute" >&2
  exit 2
fi

PYTHONPATH=rebuttal/src python -m si_rebuttal launch \
  --config rebuttal/configs/base.toml \
  --sweep rebuttal/configs/sweep.toml \
  --batch-id "${BATCH_ID}" \
  --run-root "${RUN_ROOT}" \
  "${MODE}"
