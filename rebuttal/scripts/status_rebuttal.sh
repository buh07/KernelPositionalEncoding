#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_ROOT" >&2
  exit 2
fi

PYTHONPATH=rebuttal/src python -m si_rebuttal status --run-root "$1"
