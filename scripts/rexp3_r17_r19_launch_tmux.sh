#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

SESSIONS=(
  rexp3_r17_r19_g0_llama
  rexp3_r17_r19_g1_mistral
  rexp3_r17_r19_g2_olmo
  rexp3_r17_r19_g3_finalize
)

for s in "${SESSIONS[@]}"; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

mkdir -p reinforce_exp3/logs

E17_FULL="results/reinforce_exp3/E17_normmatched_specificity"
E18_FULL="results/reinforce_exp3/E18_icl_format_confound"
E19_FULL="results/reinforce_exp3/E19_si_score_robustness"
E17_SMOKE="results/reinforce_exp3/E17_normmatched_specificity_smoke"
E18_SMOKE="results/reinforce_exp3/E18_icl_format_confound_smoke"
E19_SMOKE="results/reinforce_exp3/E19_si_score_robustness_smoke"

echo "[preflight] clearing prior E17/E18/E19 outputs to avoid stale finalize reads"
rm -rf "$E17_FULL" "$E18_FULL" "$E19_FULL" "$E17_SMOKE" "$E18_SMOKE" "$E19_SMOKE"

RUN_G0="/tmp/rexp3_r17_r19_g0_llama.sh"
RUN_G1="/tmp/rexp3_r17_r19_g1_mistral.sh"
RUN_G2="/tmp/rexp3_r17_r19_g2_olmo.sh"
RUN_G3="/tmp/rexp3_r17_r19_g3_finalize.sh"

cat > "$RUN_G0" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_r17_r19_g0_llama.log) 2>&1

echo "[g0] smoke E17/E18/E19 llama"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e17_normmatched_specificity.py \
  --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 \
  --output-root results/reinforce_exp3/E17_normmatched_specificity_smoke --smoke --no-finalize
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e18_icl_format_confound.py \
  --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 \
  --output-root results/reinforce_exp3/E18_icl_format_confound_smoke --smoke --no-finalize
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e19_si_score_robustness.py \
  --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 \
  --output-root results/reinforce_exp3/E19_si_score_robustness_smoke --smoke --no-finalize

echo "[g0] full E17/E18/E19 llama"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e17_normmatched_specificity.py \
  --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 \
  --output-root results/reinforce_exp3/E17_normmatched_specificity --no-finalize
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e18_icl_format_confound.py \
  --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 \
  --output-root results/reinforce_exp3/E18_icl_format_confound --no-finalize
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e19_si_score_robustness.py \
  --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 \
  --output-root results/reinforce_exp3/E19_si_score_robustness --no-finalize

echo "[g0] shard complete"
EOF

cat > "$RUN_G1" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_r17_r19_g1_mistral.log) 2>&1

echo "[g1] smoke E17/E18/E19 mistral"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e17_normmatched_specificity.py \
  --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 \
  --output-root results/reinforce_exp3/E17_normmatched_specificity_smoke --smoke --no-finalize
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e18_icl_format_confound.py \
  --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 \
  --output-root results/reinforce_exp3/E18_icl_format_confound_smoke --smoke --no-finalize
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e19_si_score_robustness.py \
  --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 \
  --output-root results/reinforce_exp3/E19_si_score_robustness_smoke --smoke --no-finalize

echo "[g1] full E17/E18/E19 mistral"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e17_normmatched_specificity.py \
  --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 \
  --output-root results/reinforce_exp3/E17_normmatched_specificity --no-finalize
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e18_icl_format_confound.py \
  --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 \
  --output-root results/reinforce_exp3/E18_icl_format_confound --no-finalize
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e19_si_score_robustness.py \
  --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 \
  --output-root results/reinforce_exp3/E19_si_score_robustness --no-finalize

echo "[g1] shard complete"
EOF

cat > "$RUN_G2" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_r17_r19_g2_olmo.log) 2>&1

echo "[g2] smoke E17/E18/E19 olmo"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e17_normmatched_specificity.py \
  --models olmo-2-7b --device-map olmo-2-7b:cuda:0 \
  --output-root results/reinforce_exp3/E17_normmatched_specificity_smoke --smoke --no-finalize
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e18_icl_format_confound.py \
  --models olmo-2-7b --device-map olmo-2-7b:cuda:0 \
  --output-root results/reinforce_exp3/E18_icl_format_confound_smoke --smoke --no-finalize
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e19_si_score_robustness.py \
  --models olmo-2-7b --device-map olmo-2-7b:cuda:0 \
  --output-root results/reinforce_exp3/E19_si_score_robustness_smoke --smoke --no-finalize

echo "[g2] full E17/E18/E19 olmo"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e17_normmatched_specificity.py \
  --models olmo-2-7b --device-map olmo-2-7b:cuda:0 \
  --output-root results/reinforce_exp3/E17_normmatched_specificity --no-finalize
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e18_icl_format_confound.py \
  --models olmo-2-7b --device-map olmo-2-7b:cuda:0 \
  --output-root results/reinforce_exp3/E18_icl_format_confound --no-finalize
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e19_si_score_robustness.py \
  --models olmo-2-7b --device-map olmo-2-7b:cuda:0 \
  --output-root results/reinforce_exp3/E19_si_score_robustness --no-finalize

echo "[g2] shard complete"
EOF

cat > "$RUN_G3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_r17_r19_g3_finalize.log) 2>&1

E17_DIR="results/reinforce_exp3/E17_normmatched_specificity"
E18_DIR="results/reinforce_exp3/E18_icl_format_confound"
E19_DIR="results/reinforce_exp3/E19_si_score_robustness"

echo "[g3] waiting for full shard summaries"
while true; do
  N17=0; N18=0; N19=0
  [[ -f "$E17_DIR/llama-3.1-8b/summary.json" ]] && N17=$((N17+1))
  [[ -f "$E17_DIR/mistral-7b-v0.1/summary.json" ]] && N17=$((N17+1))
  [[ -f "$E17_DIR/olmo-2-7b/summary.json" ]] && N17=$((N17+1))
  [[ -f "$E18_DIR/llama-3.1-8b/summary.json" ]] && N18=$((N18+1))
  [[ -f "$E18_DIR/mistral-7b-v0.1/summary.json" ]] && N18=$((N18+1))
  [[ -f "$E18_DIR/olmo-2-7b/summary.json" ]] && N18=$((N18+1))
  [[ -f "$E19_DIR/llama-3.1-8b/summary.json" ]] && N19=$((N19+1))
  [[ -f "$E19_DIR/mistral-7b-v0.1/summary.json" ]] && N19=$((N19+1))
  [[ -f "$E19_DIR/olmo-2-7b/summary.json" ]] && N19=$((N19+1))
  echo "[g3] shard status: E17=$N17/3 E18=$N18/3 E19=$N19/3"
  [[ "$N17" -eq 3 && "$N18" -eq 3 && "$N19" -eq 3 ]] && break
  sleep 60
done

echo "[g3] running finalize-only passes"
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e17_normmatched_specificity.py \
  --models llama-3.1-8b,mistral-7b-v0.1,olmo-2-7b \
  --output-root results/reinforce_exp3/E17_normmatched_specificity --finalize-only
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e18_icl_format_confound.py \
  --models llama-3.1-8b,mistral-7b-v0.1,olmo-2-7b \
  --output-root results/reinforce_exp3/E18_icl_format_confound --finalize-only
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e19_si_score_robustness.py \
  --models llama-3.1-8b,mistral-7b-v0.1,olmo-2-7b \
  --output-root results/reinforce_exp3/E19_si_score_robustness --finalize-only

echo "[g3] finalize complete"
EOF

chmod +x "$RUN_G0" "$RUN_G1" "$RUN_G2" "$RUN_G3"

tmux new-session -d -s rexp3_r17_r19_g0_llama "bash '$RUN_G0'"
tmux new-session -d -s rexp3_r17_r19_g1_mistral "bash '$RUN_G1'"
tmux new-session -d -s rexp3_r17_r19_g2_olmo "bash '$RUN_G2'"
tmux new-session -d -s rexp3_r17_r19_g3_finalize "bash '$RUN_G3'"

echo "[launch] started rexp3_r17_r19 tmux sessions"
tmux ls | rg 'rexp3_r17_r19' || true
echo "[monitor] bash scripts/rexp3_r17_r19_status.sh"
