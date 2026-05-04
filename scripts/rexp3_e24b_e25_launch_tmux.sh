#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

SESSIONS=(
  rexp3_e24b_e25_g0_llama
  rexp3_e24b_e25_g1_mistral
  rexp3_e24b_e25_g2_olmo
  rexp3_e24b_e25_g3_finalize
)

for s in "${SESSIONS[@]}"; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

mkdir -p reinforce_exp3/logs

E24B_FULL="results/reinforce_exp3/E24b_si_vs_non_si_contrast"
E25_FULL="results/reinforce_exp3/E25_cross_model_comparability"
E24B_SMOKE="results/reinforce_exp3/E24b_si_vs_non_si_contrast_smoke"
E25_SMOKE="results/reinforce_exp3/E25_cross_model_comparability_smoke"

echo "[preflight] clearing prior E24b/E25 outputs to avoid stale finalize reads"
rm -rf "$E24B_FULL" "$E25_FULL" "$E24B_SMOKE" "$E25_SMOKE"

RUN_G0="/tmp/rexp3_e24b_e25_g0_llama.sh"
RUN_G1="/tmp/rexp3_e24b_e25_g1_mistral.sh"
RUN_G2="/tmp/rexp3_e24b_e25_g2_olmo.sh"
RUN_G3="/tmp/rexp3_e24b_e25_g3_finalize.sh"

cat > "$RUN_G0" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e24b_e25_g0_llama.log) 2>&1

echo "[g0] smoke E24b/E25 llama"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u reinforce_exp3/scripts/run_e24b_si_vs_non_si_contrast.py \
  --models llama-3.1-8b --output-root results/reinforce_exp3/E24b_si_vs_non_si_contrast_smoke \
  --smoke --bootstrap-reps 500 --no-finalize
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u reinforce_exp3/scripts/run_e25_cross_model_comparability.py \
  --models llama-3.1-8b --output-root results/reinforce_exp3/E25_cross_model_comparability_smoke \
  --smoke --no-finalize

echo "[g0] full E24b/E25 llama"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u reinforce_exp3/scripts/run_e24b_si_vs_non_si_contrast.py \
  --models llama-3.1-8b --output-root results/reinforce_exp3/E24b_si_vs_non_si_contrast \
  --bootstrap-reps 5000 --no-finalize
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u reinforce_exp3/scripts/run_e25_cross_model_comparability.py \
  --models llama-3.1-8b --output-root results/reinforce_exp3/E25_cross_model_comparability \
  --no-finalize

echo "[g0] shard complete"
EOF

cat > "$RUN_G1" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e24b_e25_g1_mistral.log) 2>&1

echo "[g1] smoke E24b/E25 mistral"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u reinforce_exp3/scripts/run_e24b_si_vs_non_si_contrast.py \
  --models mistral-7b-v0.1 --output-root results/reinforce_exp3/E24b_si_vs_non_si_contrast_smoke \
  --smoke --bootstrap-reps 500 --no-finalize
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u reinforce_exp3/scripts/run_e25_cross_model_comparability.py \
  --models mistral-7b-v0.1 --output-root results/reinforce_exp3/E25_cross_model_comparability_smoke \
  --smoke --no-finalize

echo "[g1] full E24b/E25 mistral"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u reinforce_exp3/scripts/run_e24b_si_vs_non_si_contrast.py \
  --models mistral-7b-v0.1 --output-root results/reinforce_exp3/E24b_si_vs_non_si_contrast \
  --bootstrap-reps 5000 --no-finalize
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u reinforce_exp3/scripts/run_e25_cross_model_comparability.py \
  --models mistral-7b-v0.1 --output-root results/reinforce_exp3/E25_cross_model_comparability \
  --no-finalize

echo "[g1] shard complete"
EOF

cat > "$RUN_G2" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e24b_e25_g2_olmo.log) 2>&1

echo "[g2] smoke E24b/E25 olmo"
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u reinforce_exp3/scripts/run_e24b_si_vs_non_si_contrast.py \
  --models olmo-2-7b --output-root results/reinforce_exp3/E24b_si_vs_non_si_contrast_smoke \
  --smoke --bootstrap-reps 500 --no-finalize
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u reinforce_exp3/scripts/run_e25_cross_model_comparability.py \
  --models olmo-2-7b --output-root results/reinforce_exp3/E25_cross_model_comparability_smoke \
  --smoke --no-finalize

echo "[g2] full E24b/E25 olmo"
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u reinforce_exp3/scripts/run_e24b_si_vs_non_si_contrast.py \
  --models olmo-2-7b --output-root results/reinforce_exp3/E24b_si_vs_non_si_contrast \
  --bootstrap-reps 5000 --no-finalize
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u reinforce_exp3/scripts/run_e25_cross_model_comparability.py \
  --models olmo-2-7b --output-root results/reinforce_exp3/E25_cross_model_comparability \
  --no-finalize

echo "[g2] shard complete"
EOF

cat > "$RUN_G3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e24b_e25_g3_finalize.log) 2>&1

E24B_DIR="results/reinforce_exp3/E24b_si_vs_non_si_contrast"
E25_DIR="results/reinforce_exp3/E25_cross_model_comparability"

echo "[g3] waiting for full shard summaries"
while true; do
  N24=0; N25=0
  [[ -f "$E24B_DIR/llama-3.1-8b/summary.json" ]] && N24=$((N24+1))
  [[ -f "$E24B_DIR/mistral-7b-v0.1/summary.json" ]] && N24=$((N24+1))
  [[ -f "$E24B_DIR/olmo-2-7b/summary.json" ]] && N24=$((N24+1))
  [[ -f "$E25_DIR/llama-3.1-8b/summary.json" ]] && N25=$((N25+1))
  [[ -f "$E25_DIR/mistral-7b-v0.1/summary.json" ]] && N25=$((N25+1))
  [[ -f "$E25_DIR/olmo-2-7b/summary.json" ]] && N25=$((N25+1))
  echo "[g3] shard status: E24b=$N24/3 E25=$N25/3"
  [[ "$N24" -eq 3 && "$N25" -eq 3 ]] && break
  sleep 20
done

echo "[g3] running finalize-only passes"
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u reinforce_exp3/scripts/run_e24b_si_vs_non_si_contrast.py \
  --models llama-3.1-8b,mistral-7b-v0.1,olmo-2-7b \
  --output-root results/reinforce_exp3/E24b_si_vs_non_si_contrast --finalize-only
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u reinforce_exp3/scripts/run_e25_cross_model_comparability.py \
  --models llama-3.1-8b,mistral-7b-v0.1,olmo-2-7b \
  --output-root results/reinforce_exp3/E25_cross_model_comparability --finalize-only

echo "[g3] finalize complete"
EOF

chmod +x "$RUN_G0" "$RUN_G1" "$RUN_G2" "$RUN_G3"

tmux new-session -d -s rexp3_e24b_e25_g0_llama "bash '$RUN_G0'"
tmux new-session -d -s rexp3_e24b_e25_g1_mistral "bash '$RUN_G1'"
tmux new-session -d -s rexp3_e24b_e25_g2_olmo "bash '$RUN_G2'"
tmux new-session -d -s rexp3_e24b_e25_g3_finalize "bash '$RUN_G3'"

echo "[launch] started rexp3_e24b_e25 tmux sessions"
tmux ls | rg 'rexp3_e24b_e25' || true
echo "[monitor] bash scripts/rexp3_e24b_e25_status.sh"

