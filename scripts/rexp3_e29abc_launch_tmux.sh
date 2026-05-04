#!/usr/bin/env bash
# E29a+b+c bundle launcher
# Stage 1 (E29A): GPU0 llama, GPU1 mistral, GPU2 olmo, GPU3 finalize watcher
# Stage 2 (E29B): GPU0 llama, GPU1 mistral, GPU2 olmo, GPU3 finalize watcher
# Stage 3 (E29C): GPU0 qwen, GPU3 finalize watcher
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

SESSIONS=(
  rexp3_e29abc_g0
  rexp3_e29abc_g1
  rexp3_e29abc_g2
  rexp3_e29abc_g3
)

for s in "${SESSIONS[@]}"; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

mkdir -p reinforce_exp3/logs

E29A_FULL="results/reinforce_exp3/E29a_kernel_transplant_specificity"
E29A_SMOKE="results/reinforce_exp3/E29a_kernel_transplant_specificity_smoke"
E29B_FULL="results/reinforce_exp3/E29b_naturaltext_longcontext_probe"
E29B_SMOKE="results/reinforce_exp3/E29b_naturaltext_longcontext_probe_smoke"
E29C_FULL="results/reinforce_exp3/E29c_qwen_anchor_quickcheck"
E29C_SMOKE="results/reinforce_exp3/E29c_qwen_anchor_quickcheck_smoke"

echo "[preflight] clearing prior E29 outputs"
rm -rf "$E29A_FULL" "$E29A_SMOKE" "$E29B_FULL" "$E29B_SMOKE" "$E29C_FULL" "$E29C_SMOKE"

RUN_G0="/tmp/rexp3_e29abc_g0.sh"
RUN_G1="/tmp/rexp3_e29abc_g1.sh"
RUN_G2="/tmp/rexp3_e29abc_g2.sh"
RUN_G3="/tmp/rexp3_e29abc_g3.sh"

cat > "$RUN_G0" <<'EOS0'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e29abc_g0.log) 2>&1

PY=".venv/bin/python"
GPU_ENV="CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# ----- Stage 1: E29A (llama shard) -----
MODEL="llama-3.1-8b"
MAP="llama-3.1-8b:cuda:0"

echo "[g0] E29A smoke llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29a_kernel_transplant_specificity.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E29a_kernel_transplant_specificity_smoke \
  --smoke --no-finalize

echo "[g0] E29A full llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29a_kernel_transplant_specificity.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E29a_kernel_transplant_specificity \
  --no-finalize

# ----- Stage 2: E29B (llama shard) -----
echo "[g0] E29B smoke llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29b_naturaltext_longcontext_probe.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E29b_naturaltext_longcontext_probe_smoke \
  --n-prompts 24 --batch-size 6 --n-control-trials 1 --smoke --no-finalize

echo "[g0] E29B full llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29b_naturaltext_longcontext_probe.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E29b_naturaltext_longcontext_probe \
  --n-prompts 72 --batch-size 12 --n-control-trials 2 --no-finalize

# ----- Stage 3: E29C (qwen quickcheck) -----
QMODEL="qwen2.5-7b"
QMAP="qwen2.5-7b:cuda:0"

echo "[g0] E29C smoke qwen"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29c_qwen_anchor_quickcheck.py \
  --models "$QMODEL" --device-map "$QMAP" \
  --output-root results/reinforce_exp3/E29c_qwen_anchor_quickcheck_smoke \
  --smoke --no-finalize

echo "[g0] E29C full qwen"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29c_qwen_anchor_quickcheck.py \
  --models "$QMODEL" --device-map "$QMAP" \
  --output-root results/reinforce_exp3/E29c_qwen_anchor_quickcheck \
  --no-finalize

echo "[g0] all assigned shards complete"
EOS0

cat > "$RUN_G1" <<'EOS1'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e29abc_g1.log) 2>&1

PY=".venv/bin/python"
GPU_ENV="CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
MODEL="mistral-7b-v0.1"
MAP="mistral-7b-v0.1:cuda:0"

echo "[g1] E29A smoke mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29a_kernel_transplant_specificity.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E29a_kernel_transplant_specificity_smoke \
  --smoke --no-finalize

echo "[g1] E29A full mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29a_kernel_transplant_specificity.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E29a_kernel_transplant_specificity \
  --no-finalize

echo "[g1] E29B smoke mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29b_naturaltext_longcontext_probe.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E29b_naturaltext_longcontext_probe_smoke \
  --n-prompts 24 --batch-size 6 --n-control-trials 1 --smoke --no-finalize

echo "[g1] E29B full mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29b_naturaltext_longcontext_probe.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E29b_naturaltext_longcontext_probe \
  --n-prompts 72 --batch-size 12 --n-control-trials 2 --no-finalize

echo "[g1] all assigned shards complete"
EOS1

cat > "$RUN_G2" <<'EOS2'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e29abc_g2.log) 2>&1

PY=".venv/bin/python"
GPU_ENV="CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
MODEL="olmo-2-7b"
MAP="olmo-2-7b:cuda:0"

echo "[g2] E29A smoke olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29a_kernel_transplant_specificity.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E29a_kernel_transplant_specificity_smoke \
  --smoke --no-finalize

echo "[g2] E29A full olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29a_kernel_transplant_specificity.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E29a_kernel_transplant_specificity \
  --no-finalize

echo "[g2] E29B smoke olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29b_naturaltext_longcontext_probe.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E29b_naturaltext_longcontext_probe_smoke \
  --n-prompts 24 --batch-size 6 --n-control-trials 1 --smoke --no-finalize

echo "[g2] E29B full olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29b_naturaltext_longcontext_probe.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E29b_naturaltext_longcontext_probe \
  --n-prompts 72 --batch-size 12 --n-control-trials 2 --no-finalize

echo "[g2] all assigned shards complete"
EOS2

cat > "$RUN_G3" <<'EOS3'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e29abc_g3.log) 2>&1

PY=".venv/bin/python"
GPU_ENV="CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
MODELS="llama-3.1-8b,mistral-7b-v0.1,olmo-2-7b"
QMODEL="qwen2.5-7b"

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
    sleep 30
  done
}

wait_for_one() {
  local root="$1"
  local model="$2"
  local fname="$3"
  local label="$4"
  while true; do
    local ok=0
    [[ -f "$root/$model/$fname" ]] && ok=1
    echo "[g3] $label shard status: $ok/1"
    [[ "$ok" -eq 1 ]] && break
    sleep 20
  done
}

# ----- Stage 1: E29A -----
wait_for_three "results/reinforce_exp3/E29a_kernel_transplant_specificity_smoke" "summary.json" "E29A smoke"
echo "[g3] finalize E29A smoke"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29a_kernel_transplant_specificity.py \
  --models "$MODELS" --output-root results/reinforce_exp3/E29a_kernel_transplant_specificity_smoke --finalize-only

wait_for_three "results/reinforce_exp3/E29a_kernel_transplant_specificity" "summary.json" "E29A full"
echo "[g3] finalize E29A full"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29a_kernel_transplant_specificity.py \
  --models "$MODELS" --output-root results/reinforce_exp3/E29a_kernel_transplant_specificity --finalize-only

# ----- Stage 2: E29B -----
wait_for_three "results/reinforce_exp3/E29b_naturaltext_longcontext_probe_smoke" "summary.json" "E29B smoke"
echo "[g3] finalize E29B smoke"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29b_naturaltext_longcontext_probe.py \
  --models "$MODELS" --output-root results/reinforce_exp3/E29b_naturaltext_longcontext_probe_smoke --finalize-only

wait_for_three "results/reinforce_exp3/E29b_naturaltext_longcontext_probe" "summary.json" "E29B full"
echo "[g3] finalize E29B full"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29b_naturaltext_longcontext_probe.py \
  --models "$MODELS" --output-root results/reinforce_exp3/E29b_naturaltext_longcontext_probe --finalize-only

# ----- Stage 3: E29C -----
wait_for_one "results/reinforce_exp3/E29c_qwen_anchor_quickcheck_smoke" "$QMODEL" "summary.json" "E29C smoke"
echo "[g3] finalize E29C smoke"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29c_qwen_anchor_quickcheck.py \
  --models "$QMODEL" --output-root results/reinforce_exp3/E29c_qwen_anchor_quickcheck_smoke --finalize-only

wait_for_one "results/reinforce_exp3/E29c_qwen_anchor_quickcheck" "$QMODEL" "summary.json" "E29C full"
echo "[g3] finalize E29C full"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e29c_qwen_anchor_quickcheck.py \
  --models "$QMODEL" --output-root results/reinforce_exp3/E29c_qwen_anchor_quickcheck --finalize-only

echo "[g3] E29a+b+c finalize complete"
EOS3

chmod +x "$RUN_G0" "$RUN_G1" "$RUN_G2" "$RUN_G3"

tmux new-session -d -s rexp3_e29abc_g0 "bash '$RUN_G0'"
tmux new-session -d -s rexp3_e29abc_g1 "bash '$RUN_G1'"
tmux new-session -d -s rexp3_e29abc_g2 "bash '$RUN_G2'"
tmux new-session -d -s rexp3_e29abc_g3 "bash '$RUN_G3'"

echo "[launch] started E29a+b+c tmux sessions"
tmux ls | rg 'rexp3_e29abc' || true
echo "[monitor] bash scripts/rexp3_e29abc_status.sh"
echo "[ETA] Stage-1 smoke ~30-90m, Stage-1 full ~2-4h; full bundle ~8-16h depending OLMo/Qwen throughput"
