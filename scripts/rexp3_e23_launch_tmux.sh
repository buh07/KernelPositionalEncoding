#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

SESSIONS=(
  rexp3_e23_g0_baseline
  rexp3_e23_g1_early
  rexp3_e23_g2_late
  rexp3_e23_g3_full_finalize
)

for s in "${SESSIONS[@]}"; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

mkdir -p reinforce_exp3/logs

E23_FULL="results/reinforce_exp3/E23_stage_sensitive_si_pretraining"
E23_SMOKE="results/reinforce_exp3/E23_stage_sensitive_si_pretraining_smoke"

echo "[preflight] clearing prior E23 outputs to avoid stale finalize reads"
rm -rf "$E23_FULL" "$E23_SMOKE"

RUN_G0="/tmp/rexp3_e23_g0_baseline.sh"
RUN_G1="/tmp/rexp3_e23_g1_early.sh"
RUN_G2="/tmp/rexp3_e23_g2_late.sh"
RUN_G3="/tmp/rexp3_e23_g3_full_finalize.sh"

cat > "$RUN_G0" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e23_g0_baseline.log) 2>&1

echo "[g0] smoke E23 baseline"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e23_stage_sensitive_si_pretraining.py \
  --arms baseline --seed-list 0,1 --device cuda:0 \
  --output-root results/reinforce_exp3/E23_stage_sensitive_si_pretraining_smoke --smoke --no-finalize

echo "[g0] full E23 baseline"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e23_stage_sensitive_si_pretraining.py \
  --arms baseline --seed-list 0,1 --device cuda:0 \
  --output-root results/reinforce_exp3/E23_stage_sensitive_si_pretraining --no-finalize

echo "[g0] baseline shard complete"
EOF

cat > "$RUN_G1" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e23_g1_early.log) 2>&1

echo "[g1] smoke E23 early_si_aug"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e23_stage_sensitive_si_pretraining.py \
  --arms early_si_aug --seed-list 0,1 --device cuda:0 \
  --output-root results/reinforce_exp3/E23_stage_sensitive_si_pretraining_smoke --smoke --no-finalize

echo "[g1] full E23 early_si_aug"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e23_stage_sensitive_si_pretraining.py \
  --arms early_si_aug --seed-list 0,1 --device cuda:0 \
  --output-root results/reinforce_exp3/E23_stage_sensitive_si_pretraining --no-finalize

echo "[g1] early shard complete"
EOF

cat > "$RUN_G2" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e23_g2_late.log) 2>&1

echo "[g2] smoke E23 late_si_aug"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e23_stage_sensitive_si_pretraining.py \
  --arms late_si_aug --seed-list 0,1 --device cuda:0 \
  --output-root results/reinforce_exp3/E23_stage_sensitive_si_pretraining_smoke --smoke --no-finalize

echo "[g2] full E23 late_si_aug"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e23_stage_sensitive_si_pretraining.py \
  --arms late_si_aug --seed-list 0,1 --device cuda:0 \
  --output-root results/reinforce_exp3/E23_stage_sensitive_si_pretraining --no-finalize

echo "[g2] late shard complete"
EOF

cat > "$RUN_G3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e23_g3_full_finalize.log) 2>&1

echo "[g3] smoke E23 full_si_aug"
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e23_stage_sensitive_si_pretraining.py \
  --arms full_si_aug --seed-list 0,1 --device cuda:0 \
  --output-root results/reinforce_exp3/E23_stage_sensitive_si_pretraining_smoke --smoke --no-finalize

echo "[g3] full E23 full_si_aug"
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e23_stage_sensitive_si_pretraining.py \
  --arms full_si_aug --seed-list 0,1 --device cuda:0 \
  --output-root results/reinforce_exp3/E23_stage_sensitive_si_pretraining --no-finalize

E23_DIR="results/reinforce_exp3/E23_stage_sensitive_si_pretraining"
echo "[g3] waiting for all arm summaries"
while true; do
  N=0
  [[ -f "$E23_DIR/baseline/arm_summary.json" ]] && N=$((N+1))
  [[ -f "$E23_DIR/early_si_aug/arm_summary.json" ]] && N=$((N+1))
  [[ -f "$E23_DIR/late_si_aug/arm_summary.json" ]] && N=$((N+1))
  [[ -f "$E23_DIR/full_si_aug/arm_summary.json" ]] && N=$((N+1))
  echo "[g3] arm summary status: $N/4"
  [[ "$N" -eq 4 ]] && break
  sleep 60
done

echo "[g3] finalize-only E23"
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e23_stage_sensitive_si_pretraining.py \
  --arms all --seed-list 0,1 \
  --output-root results/reinforce_exp3/E23_stage_sensitive_si_pretraining --finalize-only

echo "[g3] full+finalize shard complete"
EOF

chmod +x "$RUN_G0" "$RUN_G1" "$RUN_G2" "$RUN_G3"

tmux new-session -d -s rexp3_e23_g0_baseline "bash '$RUN_G0'"
tmux new-session -d -s rexp3_e23_g1_early "bash '$RUN_G1'"
tmux new-session -d -s rexp3_e23_g2_late "bash '$RUN_G2'"
tmux new-session -d -s rexp3_e23_g3_full_finalize "bash '$RUN_G3'"

echo "[launch] started rexp3_e23 tmux sessions"
tmux ls | rg 'rexp3_e23' || true
echo "[monitor] bash scripts/rexp3_e23_status.sh"
