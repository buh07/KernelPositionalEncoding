#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "=== tmux sessions ==="
tmux ls | rg 'rexp3_e30ab' || echo "(none)"

echo
echo "=== recent logs (tail) ==="
for f in reinforce_exp3/logs/rexp3_e30ab_g{0,1,2,3}.log; do
  echo "--- $f"
  if [[ -f "$f" ]]; then
    tail -n 20 "$f"
  else
    echo "(missing)"
  fi
  echo
done

echo "=== artifact status ==="
for root in \
  results/reinforce_exp3/E30a_probe_boundary_grid_smoke \
  results/reinforce_exp3/E30a_probe_boundary_grid \
  results/reinforce_exp3/E30b_localbias_null_family
 do
  echo "-- $root"
  if [[ -d "$root" ]]; then
    find "$root" -maxdepth 2 -name summary.json -o -name cross_model_summary.json | sort
  else
    echo "(missing dir)"
  fi
 done
