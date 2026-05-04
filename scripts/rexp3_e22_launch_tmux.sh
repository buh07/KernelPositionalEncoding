#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

SESSIONS=(
  rexp3_e22_g0_llama
  rexp3_e22_g1_mistral
  rexp3_e22_g2_olmo
  rexp3_e22_g3_finalize
)

for s in "${SESSIONS[@]}"; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

mkdir -p reinforce_exp3/logs

E22_FULL="results/reinforce_exp3/E22_task_conditional_specificity"
E22_SMOKE="results/reinforce_exp3/E22_task_conditional_specificity_smoke"

echo "[preflight] clearing prior E22 outputs to avoid stale finalize reads"
rm -rf "$E22_FULL" "$E22_SMOKE"

RUN_G0="/tmp/rexp3_e22_g0_llama.sh"
RUN_G1="/tmp/rexp3_e22_g1_mistral.sh"
RUN_G2="/tmp/rexp3_e22_g2_olmo.sh"
RUN_G3="/tmp/rexp3_e22_g3_finalize.sh"

cat > "$RUN_G0" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e22_g0_llama.log) 2>&1

echo "[g0] smoke E22 llama"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e22_task_conditional_specificity.py \
  --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 \
  --output-root results/reinforce_exp3/E22_task_conditional_specificity_smoke --smoke --no-finalize \
  --distortion-max-retry 6 --distortion-bisect-steps 8 --distortion-min-valid-trials 1 --distortion-max-extra-trials 4

echo "[g0] full E22 llama"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e22_task_conditional_specificity.py \
  --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 \
  --output-root results/reinforce_exp3/E22_task_conditional_specificity --no-finalize \
  --distortion-max-retry 12 --distortion-bisect-steps 12 --distortion-min-valid-trials 3 --distortion-max-extra-trials 20

echo "[g0] shard complete"
EOF

cat > "$RUN_G1" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e22_g1_mistral.log) 2>&1

echo "[g1] smoke E22 mistral"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e22_task_conditional_specificity.py \
  --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 \
  --output-root results/reinforce_exp3/E22_task_conditional_specificity_smoke --smoke --no-finalize \
  --distortion-max-retry 6 --distortion-bisect-steps 8 --distortion-min-valid-trials 1 --distortion-max-extra-trials 4

echo "[g1] full E22 mistral"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e22_task_conditional_specificity.py \
  --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 \
  --output-root results/reinforce_exp3/E22_task_conditional_specificity --no-finalize \
  --distortion-max-retry 12 --distortion-bisect-steps 12 --distortion-min-valid-trials 3 --distortion-max-extra-trials 20

echo "[g1] shard complete"
EOF

cat > "$RUN_G2" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e22_g2_olmo.log) 2>&1

echo "[g2] smoke E22 olmo"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e22_task_conditional_specificity.py \
  --models olmo-2-7b --device-map olmo-2-7b:cuda:0 \
  --output-root results/reinforce_exp3/E22_task_conditional_specificity_smoke --smoke --no-finalize \
  --distortion-max-retry 6 --distortion-bisect-steps 8 --distortion-min-valid-trials 1 --distortion-max-extra-trials 4

echo "[g2] full E22 olmo"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e22_task_conditional_specificity.py \
  --models olmo-2-7b --device-map olmo-2-7b:cuda:0 \
  --output-root results/reinforce_exp3/E22_task_conditional_specificity --no-finalize \
  --distortion-max-retry 12 --distortion-bisect-steps 12 --distortion-min-valid-trials 3 --distortion-max-extra-trials 20

echo "[g2] shard complete"
EOF

cat > "$RUN_G3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e22_g3_finalize.log) 2>&1

E22_DIR="results/reinforce_exp3/E22_task_conditional_specificity"

echo "[g3] waiting for full shard summaries"
while true; do
  N22=0
  [[ -f "$E22_DIR/llama-3.1-8b/summary.json" ]] && N22=$((N22+1))
  [[ -f "$E22_DIR/mistral-7b-v0.1/summary.json" ]] && N22=$((N22+1))
  [[ -f "$E22_DIR/olmo-2-7b/summary.json" ]] && N22=$((N22+1))
  echo "[g3] shard status: E22=$N22/3"
  [[ "$N22" -eq 3 ]] && break
  sleep 60
done

echo "[g3] running finalize-only pass"
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e22_task_conditional_specificity.py \
  --models llama-3.1-8b,mistral-7b-v0.1,olmo-2-7b \
  --output-root results/reinforce_exp3/E22_task_conditional_specificity --finalize-only

echo "[g3] finalize complete"
EOF

chmod +x "$RUN_G0" "$RUN_G1" "$RUN_G2" "$RUN_G3"

tmux new-session -d -s rexp3_e22_g0_llama "bash '$RUN_G0'"
tmux new-session -d -s rexp3_e22_g1_mistral "bash '$RUN_G1'"
tmux new-session -d -s rexp3_e22_g2_olmo "bash '$RUN_G2'"
tmux new-session -d -s rexp3_e22_g3_finalize "bash '$RUN_G3'"

echo "[launch] started rexp3_e22 tmux sessions"
tmux ls | rg 'rexp3_e22' || true
echo "[monitor] bash scripts/rexp3_e22_status.sh"
