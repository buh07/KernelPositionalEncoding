#!/usr/bin/env bash
# E28a+b+d+c bundle launcher
# GPU0: llama shard (smoke->full per experiment)
# GPU1: mistral shard (smoke->full per experiment)
# GPU2: olmo shard (smoke->full per experiment)
# GPU3: finalize watcher (smoke finalize + full finalize per experiment)
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/Kernel PE"
cd "$ROOT"

SESSIONS=(
  rexp3_e28abcd_g0_llama
  rexp3_e28abcd_g1_mistral
  rexp3_e28abcd_g2_olmo
  rexp3_e28abcd_g3_finalize
)

for s in "${SESSIONS[@]}"; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

echo "[preflight] GPU status"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

mkdir -p reinforce_exp3/logs

# Output roots
E28A_FULL="results/reinforce_exp3/E28a_e26_20bin_exact"
E28A_SMOKE="results/reinforce_exp3/E28a_e26_20bin_exact_smoke"
E28B_FULL="results/reinforce_exp3/E28b_importance_matched_control"
E28B_SMOKE="results/reinforce_exp3/E28b_importance_matched_control_smoke"
E28D_FULL="results/reinforce_exp3/E28d_r2_localbias_decomposition"
E28D_SMOKE="results/reinforce_exp3/E28d_r2_localbias_decomposition_smoke"
E28C_FULL="results/reinforce_exp3/E28c_probe_diversity_transfer"
E28C_SMOKE="results/reinforce_exp3/E28c_probe_diversity_transfer_smoke"

echo "[preflight] clearing prior E28a/b/c/d outputs"
rm -rf "$E28A_FULL" "$E28A_SMOKE" "$E28B_FULL" "$E28B_SMOKE" "$E28D_FULL" "$E28D_SMOKE" "$E28C_FULL" "$E28C_SMOKE"

RUN_G0="/tmp/rexp3_e28abcd_g0_llama.sh"
RUN_G1="/tmp/rexp3_e28abcd_g1_mistral.sh"
RUN_G2="/tmp/rexp3_e28abcd_g2_olmo.sh"
RUN_G3="/tmp/rexp3_e28abcd_g3_finalize.sh"

cat > "$RUN_G0" <<'EOS0'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e28abcd_g0_llama.log) 2>&1

PY=".venv/bin/python"
GPU_ENV="CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
MODEL="llama-3.1-8b"
MAP="llama-3.1-8b:cuda:0"

echo "[g0] E28a smoke llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28a_e26_20bin_exact.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28a_e26_20bin_exact_smoke \
  --smoke --no-finalize

echo "[g0] E28a full llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28a_e26_20bin_exact.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28a_e26_20bin_exact \
  --no-finalize

echo "[g0] E28b smoke llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28b_importance_matched_control.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28b_importance_matched_control_smoke \
  --smoke --no-finalize

echo "[g0] E28b full llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28b_importance_matched_control.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28b_importance_matched_control \
  --no-finalize

echo "[g0] E28d smoke llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28d_r2_localbias_decomposition.py \
  --models "$MODEL" \
  --output-root results/reinforce_exp3/E28d_r2_localbias_decomposition_smoke \
  --e28a-root results/reinforce_exp3/E28a_e26_20bin_exact_smoke \
  --smoke --no-finalize

echo "[g0] E28d full llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28d_r2_localbias_decomposition.py \
  --models "$MODEL" \
  --output-root results/reinforce_exp3/E28d_r2_localbias_decomposition \
  --e28a-root results/reinforce_exp3/E28a_e26_20bin_exact \
  --no-finalize

echo "[g0] E28c smoke llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28c_probe_diversity_transfer.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28c_probe_diversity_transfer_smoke \
  --n-prompts 40 --batch-size 8 --smoke --no-finalize

echo "[g0] E28c full llama"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28c_probe_diversity_transfer.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28c_probe_diversity_transfer \
  --n-prompts 120 --batch-size 16 --no-finalize

echo "[g0] all shards complete"
EOS0

cat > "$RUN_G1" <<'EOS1'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e28abcd_g1_mistral.log) 2>&1

PY=".venv/bin/python"
GPU_ENV="CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
MODEL="mistral-7b-v0.1"
MAP="mistral-7b-v0.1:cuda:0"

echo "[g1] E28a smoke mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28a_e26_20bin_exact.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28a_e26_20bin_exact_smoke \
  --smoke --no-finalize

echo "[g1] E28a full mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28a_e26_20bin_exact.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28a_e26_20bin_exact \
  --no-finalize

echo "[g1] E28b smoke mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28b_importance_matched_control.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28b_importance_matched_control_smoke \
  --smoke --no-finalize

echo "[g1] E28b full mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28b_importance_matched_control.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28b_importance_matched_control \
  --no-finalize

echo "[g1] E28d smoke mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28d_r2_localbias_decomposition.py \
  --models "$MODEL" \
  --output-root results/reinforce_exp3/E28d_r2_localbias_decomposition_smoke \
  --e28a-root results/reinforce_exp3/E28a_e26_20bin_exact_smoke \
  --smoke --no-finalize

echo "[g1] E28d full mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28d_r2_localbias_decomposition.py \
  --models "$MODEL" \
  --output-root results/reinforce_exp3/E28d_r2_localbias_decomposition \
  --e28a-root results/reinforce_exp3/E28a_e26_20bin_exact \
  --no-finalize

echo "[g1] E28c smoke mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28c_probe_diversity_transfer.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28c_probe_diversity_transfer_smoke \
  --n-prompts 40 --batch-size 8 --smoke --no-finalize

echo "[g1] E28c full mistral"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28c_probe_diversity_transfer.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28c_probe_diversity_transfer \
  --n-prompts 120 --batch-size 16 --no-finalize

echo "[g1] all shards complete"
EOS1

cat > "$RUN_G2" <<'EOS2'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e28abcd_g2_olmo.log) 2>&1

PY=".venv/bin/python"
GPU_ENV="CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
MODEL="olmo-2-7b"
MAP="olmo-2-7b:cuda:0"

echo "[g2] E28a smoke olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28a_e26_20bin_exact.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28a_e26_20bin_exact_smoke \
  --smoke --no-finalize

echo "[g2] E28a full olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28a_e26_20bin_exact.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28a_e26_20bin_exact \
  --no-finalize

echo "[g2] E28b smoke olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28b_importance_matched_control.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28b_importance_matched_control_smoke \
  --smoke --no-finalize

echo "[g2] E28b full olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28b_importance_matched_control.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28b_importance_matched_control \
  --no-finalize

echo "[g2] E28d smoke olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28d_r2_localbias_decomposition.py \
  --models "$MODEL" \
  --output-root results/reinforce_exp3/E28d_r2_localbias_decomposition_smoke \
  --e28a-root results/reinforce_exp3/E28a_e26_20bin_exact_smoke \
  --smoke --no-finalize

echo "[g2] E28d full olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28d_r2_localbias_decomposition.py \
  --models "$MODEL" \
  --output-root results/reinforce_exp3/E28d_r2_localbias_decomposition \
  --e28a-root results/reinforce_exp3/E28a_e26_20bin_exact \
  --no-finalize

echo "[g2] E28c smoke olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28c_probe_diversity_transfer.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28c_probe_diversity_transfer_smoke \
  --n-prompts 40 --batch-size 8 --smoke --no-finalize

echo "[g2] E28c full olmo"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28c_probe_diversity_transfer.py \
  --models "$MODEL" --device-map "$MAP" \
  --output-root results/reinforce_exp3/E28c_probe_diversity_transfer \
  --n-prompts 120 --batch-size 16 --no-finalize

echo "[g2] all shards complete"
EOS2

cat > "$RUN_G3" <<'EOS3'
#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/Kernel PE"
exec > >(tee -a reinforce_exp3/logs/rexp3_e28abcd_g3_finalize.log) 2>&1

PY=".venv/bin/python"
GPU_ENV="CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
MODELS="llama-3.1-8b,mistral-7b-v0.1,olmo-2-7b"

wait_for_shards() {
  local root="$1"
  local file_name="$2"
  local label="$3"
  while true; do
    local n=0
    [[ -f "$root/llama-3.1-8b/$file_name" ]] && n=$((n+1))
    [[ -f "$root/mistral-7b-v0.1/$file_name" ]] && n=$((n+1))
    [[ -f "$root/olmo-2-7b/$file_name" ]] && n=$((n+1))
    echo "[g3] $label shard status: $n/3"
    [[ "$n" -eq 3 ]] && break
    sleep 30
  done
}

# ---------- E28a ----------
wait_for_shards "results/reinforce_exp3/E28a_e26_20bin_exact_smoke" "model_summary.json" "E28a smoke"
echo "[g3] finalize E28a smoke"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28a_e26_20bin_exact.py \
  --models "$MODELS" --output-root results/reinforce_exp3/E28a_e26_20bin_exact_smoke --finalize-only

wait_for_shards "results/reinforce_exp3/E28a_e26_20bin_exact" "model_summary.json" "E28a full"
echo "[g3] finalize E28a full"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28a_e26_20bin_exact.py \
  --models "$MODELS" --output-root results/reinforce_exp3/E28a_e26_20bin_exact --finalize-only

# ---------- E28b ----------
wait_for_shards "results/reinforce_exp3/E28b_importance_matched_control_smoke" "summary.json" "E28b smoke"
echo "[g3] finalize E28b smoke"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28b_importance_matched_control.py \
  --models "$MODELS" --output-root results/reinforce_exp3/E28b_importance_matched_control_smoke --finalize-only

wait_for_shards "results/reinforce_exp3/E28b_importance_matched_control" "summary.json" "E28b full"
echo "[g3] finalize E28b full"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28b_importance_matched_control.py \
  --models "$MODELS" --output-root results/reinforce_exp3/E28b_importance_matched_control --finalize-only

# ---------- E28d ----------
wait_for_shards "results/reinforce_exp3/E28d_r2_localbias_decomposition_smoke" "summary.json" "E28d smoke"
echo "[g3] finalize E28d smoke"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28d_r2_localbias_decomposition.py \
  --models "$MODELS" \
  --output-root results/reinforce_exp3/E28d_r2_localbias_decomposition_smoke \
  --e28a-root results/reinforce_exp3/E28a_e26_20bin_exact_smoke \
  --finalize-only

wait_for_shards "results/reinforce_exp3/E28d_r2_localbias_decomposition" "summary.json" "E28d full"
echo "[g3] finalize E28d full"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28d_r2_localbias_decomposition.py \
  --models "$MODELS" \
  --output-root results/reinforce_exp3/E28d_r2_localbias_decomposition \
  --e28a-root results/reinforce_exp3/E28a_e26_20bin_exact \
  --finalize-only

# ---------- E28c ----------
wait_for_shards "results/reinforce_exp3/E28c_probe_diversity_transfer_smoke" "summary.json" "E28c smoke"
echo "[g3] finalize E28c smoke"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28c_probe_diversity_transfer.py \
  --models "$MODELS" --output-root results/reinforce_exp3/E28c_probe_diversity_transfer_smoke --finalize-only

wait_for_shards "results/reinforce_exp3/E28c_probe_diversity_transfer" "summary.json" "E28c full"
echo "[g3] finalize E28c full"
env $GPU_ENV $PY -u reinforce_exp3/scripts/run_e28c_probe_diversity_transfer.py \
  --models "$MODELS" --output-root results/reinforce_exp3/E28c_probe_diversity_transfer --finalize-only

echo "[g3] E28a+b+d+c finalize complete"
EOS3

chmod +x "$RUN_G0" "$RUN_G1" "$RUN_G2" "$RUN_G3"

tmux new-session -d -s rexp3_e28abcd_g0_llama "bash '$RUN_G0'"
tmux new-session -d -s rexp3_e28abcd_g1_mistral "bash '$RUN_G1'"
tmux new-session -d -s rexp3_e28abcd_g2_olmo "bash '$RUN_G2'"
tmux new-session -d -s rexp3_e28abcd_g3_finalize "bash '$RUN_G3'"

echo "[launch] started E28a+b+d+c tmux sessions"
tmux ls | rg 'rexp3_e28abcd' || true
echo "[monitor] bash scripts/rexp3_e28_abcd_status.sh"
echo "[ETA] ~12-22h end-to-end (smoke+full sequence), with first smoke signals typically in 1-3h"
