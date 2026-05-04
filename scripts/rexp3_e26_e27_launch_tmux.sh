#!/usr/bin/env bash
# E26 + E27 — R²-stratified dose-response and absolute-threshold robustness
# Each GPU session loads one model and runs both experiments sequentially.
# GPU 3 handles the cross-model finalize once all shards are done.
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

SESSIONS=(
  rexp3_e26e27_g0_llama
  rexp3_e26e27_g1_mistral
  rexp3_e26e27_g2_olmo
  rexp3_e26e27_g3_finalize
)

for s in "${SESSIONS[@]}"; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

mkdir -p reinforce_exp3/logs

E26_OUT="results/reinforce_exp3/E26_r2_disruption_dose_response"
E27_OUT="results/reinforce_exp3/E27_absolute_threshold_robustness"

echo "[preflight] clearing prior E26/E27 outputs"
rm -rf "$E26_OUT" "$E27_OUT"

RUN_G0="/tmp/rexp3_e26e27_g0_llama.sh"
RUN_G1="/tmp/rexp3_e26e27_g1_mistral.sh"
RUN_G2="/tmp/rexp3_e26e27_g2_olmo.sh"
RUN_G3="/tmp/rexp3_e26e27_g3_finalize.sh"

# ---------------------------------------------------------------------------
# GPU 0 — Llama
# ---------------------------------------------------------------------------
cat > "$RUN_G0" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e26e27_g0_llama.log) 2>&1

echo "[g0] E26 smoke — llama"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e26_r2_disruption_dose_response.py \
  --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 \
  --output-root results/reinforce_exp3/E26_r2_disruption_dose_response_smoke \
  --smoke --no-finalize

echo "[g0] E26 full — llama"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e26_r2_disruption_dose_response.py \
  --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 \
  --output-root results/reinforce_exp3/E26_r2_disruption_dose_response \
  --no-finalize

echo "[g0] E27 full — llama"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e27_absolute_threshold_robustness.py \
  --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 \
  --output-root results/reinforce_exp3/E27_absolute_threshold_robustness \
  --no-finalize

echo "[g0] llama shard complete"
EOF

# ---------------------------------------------------------------------------
# GPU 1 — Mistral
# ---------------------------------------------------------------------------
cat > "$RUN_G1" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e26e27_g1_mistral.log) 2>&1

echo "[g1] E26 smoke — mistral"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e26_r2_disruption_dose_response.py \
  --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 \
  --output-root results/reinforce_exp3/E26_r2_disruption_dose_response_smoke \
  --smoke --no-finalize

echo "[g1] E26 full — mistral"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e26_r2_disruption_dose_response.py \
  --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 \
  --output-root results/reinforce_exp3/E26_r2_disruption_dose_response \
  --no-finalize

echo "[g1] E27 full — mistral"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e27_absolute_threshold_robustness.py \
  --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 \
  --output-root results/reinforce_exp3/E27_absolute_threshold_robustness \
  --no-finalize

echo "[g1] mistral shard complete"
EOF

# ---------------------------------------------------------------------------
# GPU 2 — OLMo
# ---------------------------------------------------------------------------
cat > "$RUN_G2" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e26e27_g2_olmo.log) 2>&1

echo "[g2] E26 smoke — olmo"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e26_r2_disruption_dose_response.py \
  --models olmo-2-7b --device-map olmo-2-7b:cuda:0 \
  --output-root results/reinforce_exp3/E26_r2_disruption_dose_response_smoke \
  --smoke --no-finalize

echo "[g2] E26 full — olmo"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e26_r2_disruption_dose_response.py \
  --models olmo-2-7b --device-map olmo-2-7b:cuda:0 \
  --output-root results/reinforce_exp3/E26_r2_disruption_dose_response \
  --no-finalize

echo "[g2] E27 full — olmo"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e27_absolute_threshold_robustness.py \
  --models olmo-2-7b --device-map olmo-2-7b:cuda:0 \
  --output-root results/reinforce_exp3/E27_absolute_threshold_robustness \
  --no-finalize

echo "[g2] olmo shard complete"
EOF

# ---------------------------------------------------------------------------
# GPU 3 — Finalize (waits for all 3 model shards)
# ---------------------------------------------------------------------------
cat > "$RUN_G3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e26e27_g3_finalize.log) 2>&1

E26_DIR="results/reinforce_exp3/E26_r2_disruption_dose_response"
E27_DIR="results/reinforce_exp3/E27_absolute_threshold_robustness"

echo "[g3] waiting for all E26 model shards"
while true; do
  N=0
  [[ -f "$E26_DIR/llama-3.1-8b/model_summary.json" ]] && N=$((N+1))
  [[ -f "$E26_DIR/mistral-7b-v0.1/model_summary.json" ]] && N=$((N+1))
  [[ -f "$E26_DIR/olmo-2-7b/model_summary.json" ]] && N=$((N+1))
  echo "[g3] E26 shard status: $N/3"
  [[ "$N" -eq 3 ]] && break
  sleep 30
done

echo "[g3] finalizing E26"
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e26_r2_disruption_dose_response.py \
  --models llama-3.1-8b,mistral-7b-v0.1,olmo-2-7b \
  --output-root "$E26_DIR" --finalize-only

echo "[g3] waiting for all E27 model shards"
while true; do
  N=0
  [[ -f "$E27_DIR/llama-3.1-8b/model_summary.json" ]] && N=$((N+1))
  [[ -f "$E27_DIR/mistral-7b-v0.1/model_summary.json" ]] && N=$((N+1))
  [[ -f "$E27_DIR/olmo-2-7b/model_summary.json" ]] && N=$((N+1))
  echo "[g3] E27 shard status: $N/3"
  [[ "$N" -eq 3 ]] && break
  sleep 30
done

echo "[g3] finalizing E27"
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e27_absolute_threshold_robustness.py \
  --models llama-3.1-8b,mistral-7b-v0.1,olmo-2-7b \
  --output-root "$E27_DIR" --finalize-only

echo "[g3] E26+E27 finalize complete"
EOF

chmod +x "$RUN_G0" "$RUN_G1" "$RUN_G2" "$RUN_G3"

tmux new-session -d -s rexp3_e26e27_g0_llama    "bash '$RUN_G0'"
tmux new-session -d -s rexp3_e26e27_g1_mistral   "bash '$RUN_G1'"
tmux new-session -d -s rexp3_e26e27_g2_olmo      "bash '$RUN_G2'"
tmux new-session -d -s rexp3_e26e27_g3_finalize  "bash '$RUN_G3'"

echo "[launch] started E26+E27 tmux sessions"
tmux ls | grep 'rexp3_e26e27' || true
echo "[monitor] tail -f reinforce_exp3/logs/rexp3_e26e27_g*.log"
echo "[ETA] ~45-60 min (model load ~5 min + E26 ~15 min + E27 ~15 min per model)"
