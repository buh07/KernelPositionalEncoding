#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

SESSIONS=(
  rexp3_r20_r21_g4_llama
  rexp3_r20_r21_g5_mistral
  rexp3_r20_r21_g6_olmo
  rexp3_r20_r21_g7_r21
  rexp3_r20_r21_finalize
)

for s in "${SESSIONS[@]}"; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

mkdir -p reinforce_exp3/logs

RUN_G4="/tmp/rexp3_r20_r21_g4_llama.sh"
RUN_G5="/tmp/rexp3_r20_r21_g5_mistral.sh"
RUN_G6="/tmp/rexp3_r20_r21_g6_olmo.sh"
RUN_G7="/tmp/rexp3_r20_r21_g7_r21.sh"
RUN_F="/tmp/rexp3_r20_r21_finalize.sh"

cat > "$RUN_G4" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_r20_r21_g4_llama.log) 2>&1

echo "[g4] smoke E20 llama"
CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e20_heterogeneity_variance_decomp.py \
  --mode proxy --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 --seed-list 0,1,2 \
  --output-root results/reinforce_exp3/E20_heterogeneity_variance_decomp_smoke --smoke --no-finalize

echo "[g4] full E20 llama"
CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e20_heterogeneity_variance_decomp.py \
  --mode proxy --models llama-3.1-8b --device-map llama-3.1-8b:cuda:0 --seed-list 0,1,2 --no-finalize

echo "[g4] shard complete"
EOF

cat > "$RUN_G5" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_r20_r21_g5_mistral.log) 2>&1

echo "[g5] smoke E20 mistral"
CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e20_heterogeneity_variance_decomp.py \
  --mode proxy --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 --seed-list 0,1,2 \
  --output-root results/reinforce_exp3/E20_heterogeneity_variance_decomp_smoke --smoke --no-finalize

echo "[g5] full E20 mistral"
CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e20_heterogeneity_variance_decomp.py \
  --mode proxy --models mistral-7b-v0.1 --device-map mistral-7b-v0.1:cuda:0 --seed-list 0,1,2 --no-finalize

echo "[g5] shard complete"
EOF

cat > "$RUN_G6" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_r20_r21_g6_olmo.log) 2>&1

echo "[g6] smoke E20 olmo"
CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e20_heterogeneity_variance_decomp.py \
  --mode proxy --models olmo-2-7b --device-map olmo-2-7b:cuda:0 --seed-list 0,1,2 \
  --output-root results/reinforce_exp3/E20_heterogeneity_variance_decomp_smoke --smoke --no-finalize

echo "[g6] full E20 olmo"
CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e20_heterogeneity_variance_decomp.py \
  --mode proxy --models olmo-2-7b --device-map olmo-2-7b:cuda:0 --seed-list 0,1,2 --no-finalize

echo "[g6] shard complete"
EOF

cat > "$RUN_G7" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_r20_r21_g7_r21.log) 2>&1

echo "[g7] smoke E21 tinyllama rope/nope"
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e21_pe_family_training_contrast.py \
  --mode proxy --models tinyllama-1.1b,tinyllama-nope-1.1b \
  --device-map tinyllama-1.1b:cuda:0,tinyllama-nope-1.1b:cuda:0 --seed-list 0,1,2 \
  --output-root results/reinforce_exp3/E21_pe_family_training_contrast_smoke --smoke --no-finalize

echo "[g7] full E21 tinyllama rope/nope"
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u reinforce_exp3/scripts/run_e21_pe_family_training_contrast.py \
  --mode proxy --models tinyllama-1.1b,tinyllama-nope-1.1b \
  --device-map tinyllama-1.1b:cuda:0,tinyllama-nope-1.1b:cuda:0 --seed-list 0,1,2 --no-finalize

echo "[g7] shard complete"
EOF

cat > "$RUN_F" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_r20_r21_finalize.log) 2>&1

E20_DIR="results/reinforce_exp3/E20_heterogeneity_variance_decomp"
E21_DIR="results/reinforce_exp3/E21_pe_family_training_contrast"

echo "[finalize] waiting for full shard summaries"
while true; do
  N20=0; N21=0
  for m in llama-3.1-8b mistral-7b-v0.1 olmo-2-7b; do
    for s in 0 1 2; do
      [[ -f "$E20_DIR/$m/seed_${s}/summary.json" ]] && N20=$((N20+1))
    done
  done
  for m in tinyllama-1.1b tinyllama-nope-1.1b; do
    for s in 0 1 2; do
      [[ -f "$E21_DIR/$m/seed_${s}/summary.json" ]] && N21=$((N21+1))
    done
  done
  echo "[finalize] shard status: E20=$N20/9 E21=$N21/6"
  [[ "$N20" -eq 9 && "$N21" -eq 6 ]] && break
  sleep 60
done

echo "[finalize] running finalize-only passes"
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -u reinforce_exp3/scripts/run_e20_heterogeneity_variance_decomp.py \
  --mode proxy --models llama-3.1-8b,mistral-7b-v0.1,olmo-2-7b --seed-list 0,1,2 --finalize-only
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -u reinforce_exp3/scripts/run_e21_pe_family_training_contrast.py \
  --mode proxy --models tinyllama-1.1b,tinyllama-nope-1.1b --seed-list 0,1,2 --finalize-only

echo "[finalize] complete"
EOF

chmod +x "$RUN_G4" "$RUN_G5" "$RUN_G6" "$RUN_G7" "$RUN_F"

tmux new-session -d -s rexp3_r20_r21_g4_llama "bash '$RUN_G4'"
tmux new-session -d -s rexp3_r20_r21_g5_mistral "bash '$RUN_G5'"
tmux new-session -d -s rexp3_r20_r21_g6_olmo "bash '$RUN_G6'"
tmux new-session -d -s rexp3_r20_r21_g7_r21 "bash '$RUN_G7'"
tmux new-session -d -s rexp3_r20_r21_finalize "bash '$RUN_F'"

echo "[launch] started rexp3_r20_r21 tmux sessions"
tmux ls | rg 'rexp3_r20_r21' || true
echo "[monitor] bash scripts/rexp3_r20_r21_status.sh"
