#!/usr/bin/env bash
# E30a+b launcher
# Stage 1 (E30A): GPU0 llama, GPU1 mistral, GPU2 olmo, GPU3 finalize watcher
# Stage 2 (E30B): GPU3 finalize/full reanalysis
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

SESSIONS=(
  rexp3_e30ab_g0
  rexp3_e30ab_g1
  rexp3_e30ab_g2
  rexp3_e30ab_g3
)

for s in "${SESSIONS[@]}"; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

mkdir -p reinforce_exp3/logs

E30A_FULL="results/reinforce_exp3/E30a_probe_boundary_grid"
E30A_SMOKE="results/reinforce_exp3/E30a_probe_boundary_grid_smoke"
E30B_FULL="results/reinforce_exp3/E30b_localbias_null_family"

echo "[preflight] clearing prior E30 outputs"
rm -rf "$E30A_FULL" "$E30A_SMOKE" "$E30B_FULL"

RUN_G0="/tmp/rexp3_e30ab_g0.sh"
RUN_G1="/tmp/rexp3_e30ab_g1.sh"
RUN_G2="/tmp/rexp3_e30ab_g2.sh"
RUN_G3="/tmp/rexp3_e30ab_g3.sh"

cat > "$RUN_G0" <<'EOS0'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e30ab_g0.log) 2>&1

PY=".venv/bin/python"
GPU_ENV="CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
MODEL="llama-3.1-8b"
MAP="llama-3.1-8b:cuda:0"

echo "[g0] E30A smoke llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e30a_probe_boundary_grid.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E30a_probe_boundary_grid_smoke \
  --n-prompts 12 --batch-size 4 --smoke --no-finalize

echo "[g0] E30A full llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e30a_probe_boundary_grid.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E30a_probe_boundary_grid \
  --n-prompts 36 --batch-size 8 --n-control-trials 2 --no-finalize

echo "[g0] shard complete"
EOS0

cat > "$RUN_G1" <<'EOS1'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e30ab_g1.log) 2>&1

PY=".venv/bin/python"
GPU_ENV="CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
MODEL="mistral-7b-v0.1"
MAP="mistral-7b-v0.1:cuda:0"

echo "[g1] E30A smoke mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e30a_probe_boundary_grid.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E30a_probe_boundary_grid_smoke \
  --n-prompts 12 --batch-size 4 --smoke --no-finalize

echo "[g1] E30A full mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e30a_probe_boundary_grid.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E30a_probe_boundary_grid \
  --n-prompts 36 --batch-size 8 --n-control-trials 2 --no-finalize

echo "[g1] shard complete"
EOS1

cat > "$RUN_G2" <<'EOS2'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e30ab_g2.log) 2>&1

PY=".venv/bin/python"
GPU_ENV="CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
MODEL="olmo-2-7b"
MAP="olmo-2-7b:cuda:0"

echo "[g2] E30A smoke olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e30a_probe_boundary_grid.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E30a_probe_boundary_grid_smoke \
  --n-prompts 12 --batch-size 4 --smoke --no-finalize

echo "[g2] E30A full olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e30a_probe_boundary_grid.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E30a_probe_boundary_grid \
  --n-prompts 36 --batch-size 8 --n-control-trials 2 --no-finalize

echo "[g2] shard complete"
EOS2

cat > "$RUN_G3" <<'EOS3'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e30ab_g3.log) 2>&1

PY=".venv/bin/python"
GPU_ENV="CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
MODELS="llama-3.1-8b,mistral-7b-v0.1,olmo-2-7b"

wait_for_three() {
  local root="$1"
  local fname="$2"
  local label="$3"
  while true; do
    local n=0
    [[ -f "$root/llama-3.1-8b/$fname" ]] && n=$((n+1))
    [[ -f "$root/mistral-7b-v0.1/$fname" ]] && n=$((n+1))
    [[ -f "$root/olmo-2-7b/$fname" ]] && n=$((n+1))
    echo "[g3] $label shard status: $n/3"
    [[ "$n" -eq 3 ]] && break
    sleep 20
  done
}

# ----- Stage 1: E30A smoke/full finalization -----
wait_for_three "results/reinforce_exp3/E30a_probe_boundary_grid_smoke" "summary.json" "E30A smoke"
echo "[g3] finalize E30A smoke"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e30a_probe_boundary_grid.py \
  --models "$MODELS" --output-root results/reinforce_exp3/E30a_probe_boundary_grid_smoke --finalize-only

wait_for_three "results/reinforce_exp3/E30a_probe_boundary_grid" "summary.json" "E30A full"
echo "[g3] finalize E30A full"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e30a_probe_boundary_grid.py \
  --models "$MODELS" --output-root results/reinforce_exp3/E30a_probe_boundary_grid --finalize-only

# ----- Stage 2: E30B reanalysis (single run + finalize) -----
echo "[g3] run E30B full"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e30b_localbias_null_family.py \
  --models "$MODELS" --output-root results/reinforce_exp3/E30b_localbias_null_family

echo "[g3] E30A+B finalize complete"
EOS3

chmod +x "$RUN_G0" "$RUN_G1" "$RUN_G2" "$RUN_G3"

tmux new-session -d -s rexp3_e30ab_g0 "bash '$RUN_G0'"
tmux new-session -d -s rexp3_e30ab_g1 "bash '$RUN_G1'"
tmux new-session -d -s rexp3_e30ab_g2 "bash '$RUN_G2'"
tmux new-session -d -s rexp3_e30ab_g3 "bash '$RUN_G3'"

echo "[launch] started E30A+B tmux sessions"
tmux ls | rg 'rexp3_e30ab' || true
echo "[monitor] bash scripts/rexp3_e30ab_status.sh"
