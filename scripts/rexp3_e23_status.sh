#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

echo "== tmux sessions =="
tmux ls 2>/dev/null | rg 'rexp3_e23|session' || true

echo
echo "== gpu usage =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "== recent logs (tail 40) =="
for f in \
  reinforce_exp3/logs/rexp3_e23_g0_baseline.log \
  reinforce_exp3/logs/rexp3_e23_g1_early.log \
  reinforce_exp3/logs/rexp3_e23_g2_late.log \
  reinforce_exp3/logs/rexp3_e23_g3_full_finalize.log
 do
  echo "--- $f ---"
  if [[ -f "$f" ]]; then
    tail -n 40 "$f"
  else
    echo "(missing)"
  fi
 done

echo
echo "== artifact presence =="
E23="results/reinforce_exp3/E23_stage_sensitive_si_pretraining"
for arm in baseline early_si_aug late_si_aug full_si_aug; do
  echo "arm=$arm"
  ls -1 "$E23/$arm" 2>/dev/null || true
  [[ -f "$E23/$arm/arm_summary.json" ]] && echo "  arm_summary.json: present" || echo "  arm_summary.json: missing"
done

[[ -f "$E23/cross_arm_stage_sensitivity_summary.json" ]] && \
  echo "cross_arm_stage_sensitivity_summary.json: present" || \
  echo "cross_arm_stage_sensitivity_summary.json: missing"

echo
echo "== error sweep =="
rg -n "Traceback|CUDA out of memory|hard_fail_reason" reinforce_exp3/logs/rexp3_e23_*.log || true
