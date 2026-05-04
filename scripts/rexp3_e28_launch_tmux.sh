#!/usr/bin/env bash
# E28 — Probe-scope battery for Result III
# GPU0: llama, GPU1: mistral, GPU2: olmo, GPU3: finalize watcher
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

SESSIONS=(
  rexp3_e28_g0_llama
  rexp3_e28_g1_mistral
  rexp3_e28_g2_olmo
  rexp3_e28_g3_finalize
)

for s in "${SESSIONS[@]}"; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

mkdir -p reinforce_exp3/logs

E28_OUT="results/reinforce_exp3/E28_probe_scope_battery"
E28_SMOKE_OUT="results/reinforce_exp3/E28_probe_scope_battery_smoke"

echo "[preflight] clearing prior E28 outputs"
rm -rf "$E28_OUT" "$E28_SMOKE_OUT"

RUN_G0="/tmp/rexp3_e28_g0_llama.sh"
RUN_G1="/tmp/rexp3_e28_g1_mistral.sh"
RUN_G2="/tmp/rexp3_e28_g2_olmo.sh"
RUN_G3="/tmp/rexp3_e28_g3_finalize.sh"

cat > "$RUN_G0" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e28_g0_llama.log) 2>&1

echo "[g0] E28 smoke — llama"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e28_probe_scope_battery.py \
  --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 \
  --output-root results/reinforce_exp3/E28_probe_scope_battery_smoke \
  --n-prompts 40 --batch-size 8 --smoke --no-finalize

echo "[g0] E28 full — llama"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e28_probe_scope_battery.py \
  --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 \
  --output-root results/reinforce_exp3/E28_probe_scope_battery \
  --n-prompts 120 --batch-size 16 --no-finalize

echo "[g0] llama shard complete"
EOF

cat > "$RUN_G1" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e28_g1_mistral.log) 2>&1

echo "[g1] E28 smoke — mistral"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e28_probe_scope_battery.py \
  --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 \
  --output-root results/reinforce_exp3/E28_probe_scope_battery_smoke \
  --n-prompts 40 --batch-size 8 --smoke --no-finalize

echo "[g1] E28 full — mistral"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e28_probe_scope_battery.py \
  --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 \
  --output-root results/reinforce_exp3/E28_probe_scope_battery \
  --n-prompts 120 --batch-size 16 --no-finalize

echo "[g1] mistral shard complete"
EOF

cat > "$RUN_G2" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e28_g2_olmo.log) 2>&1

echo "[g2] E28 smoke — olmo"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e28_probe_scope_battery.py \
  --models olmo-2-7b --device-map olmo-2-7b:cuda:0 \
  --output-root results/reinforce_exp3/E28_probe_scope_battery_smoke \
  --n-prompts 40 --batch-size 8 --smoke --no-finalize

echo "[g2] E28 full — olmo"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e28_probe_scope_battery.py \
  --models olmo-2-7b --device-map olmo-2-7b:cuda:0 \
  --output-root results/reinforce_exp3/E28_probe_scope_battery \
  --n-prompts 120 --batch-size 16 --no-finalize

echo "[g2] olmo shard complete"
EOF

cat > "$RUN_G3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e28_g3_finalize.log) 2>&1

E28_DIR="results/reinforce_exp3/E28_probe_scope_battery"

echo "[g3] waiting for all E28 model shards"
while true; do
  N=0
  [[ -f "$E28_DIR/llama-3.1-8b/summary.json" ]] && N=$((N+1))
  [[ -f "$E28_DIR/mistral-7b-v0.1/summary.json" ]] && N=$((N+1))
  [[ -f "$E28_DIR/olmo-2-7b/summary.json" ]] && N=$((N+1))
  echo "[g3] E28 shard status: $N/3"
  [[ "$N" -eq 3 ]] && break
  sleep 30
done

echo "[g3] finalizing E28"
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u reinforce_exp3/scripts/run_e28_probe_scope_battery.py \
  --models llama-3.1-8b,mistral-7b-v0.1,olmo-2-7b \
  --output-root "$E28_DIR" --finalize-only

echo "[g3] E28 finalize complete"
EOF

chmod +x "$RUN_G0" "$RUN_G1" "$RUN_G2" "$RUN_G3"

tmux new-session -d -s rexp3_e28_g0_llama    "bash '$RUN_G0'"
tmux new-session -d -s rexp3_e28_g1_mistral   "bash '$RUN_G1'"
tmux new-session -d -s rexp3_e28_g2_olmo      "bash '$RUN_G2'"
tmux new-session -d -s rexp3_e28_g3_finalize  "bash '$RUN_G3'"

echo "[launch] started E28 tmux sessions"
tmux ls | grep 'rexp3_e28' || true
echo "[monitor] tail -f reinforce_exp3/logs/rexp3_e28_g*.log"
echo "[ETA] ~1-2h total depending on model throughput"
