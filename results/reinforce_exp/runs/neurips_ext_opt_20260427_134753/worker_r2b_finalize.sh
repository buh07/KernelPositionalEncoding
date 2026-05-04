#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/Kernel PE"
RUN_ROOT="${1:?run_root}"
cd "$ROOT"
echo "[r2b-finalize] waiting for 24 seed manifests"
while true; do
  n=$(ls "$RUN_ROOT/exp_r2b_olmo_boundary_power/manifests/seed_"*.json 2>/dev/null | wc -l || true)
  [ "$n" -ge 24 ] && break
  sleep 30
done
.venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py finalize-domain \
  --model olmo-2-7b --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
  --domains wiki,code,dialogue --output-root "$RUN_ROOT/exp_r2b_olmo_boundary_power" --n-boot 5000 --n-perm 5000 --seed 13
.venv/bin/python -u reinforce_exp/exp_r2b_olmo_boundary_power.py finalize-claim \
  --model olmo-2-7b --output-root "$RUN_ROOT/exp_r2b_olmo_boundary_power"
