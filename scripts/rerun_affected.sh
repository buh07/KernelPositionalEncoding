#!/usr/bin/env bash
# Rerun all conditions affected by the RoPE hook fix and double-centering fix.
#
# Affected models:
#   olmo-1b          (RoPE bug)
#   llama-3.2-1b     (RoPE bug + double-centering bug)
#   tinyllama-1.1b   (RoPE bug + double-centering bug)
#   tinyllama-nope-1.1b (double-centering bug)
#
# GPT-2 small/medium are NOT affected (LayerNorm, no RoPE) — old results are valid.
#
# After all Track A/B runs, spectral analysis is run for ALL 6 models (self-gating).

set -euo pipefail

export CUDA_VISIBLE_DEVICES=4
DEVICE="cuda"
RUN="python experiment1/run.py"

DATASETS=("wiki40b_en_pre2019" "codesearchnet_python_snapshot" "synthetic_random")
LENGTHS=("256" "1024")

run_track_a() {
    local model="$1" dataset="$2" len="$3"
    echo "===== Track A: ${model} / ${dataset} / len=${len} ====="
    $RUN track-a --model "$model" --dataset "$dataset" --seq-len "$len" --device "$DEVICE"
}

run_track_b() {
    local model="$1" dataset="$2" len="$3"
    echo "===== Track B: ${model} / ${dataset} / len=${len} ====="
    $RUN track-b --model "$model" --dataset "$dataset" --seq-len "$len" --device "$DEVICE"
}

run_spectral() {
    local model="$1" dataset="$2" len="$3"
    echo "===== Spectral: ${model} / ${dataset} / len=${len} ====="
    $RUN spectral --model "$model" --dataset "$dataset" --seq-len "$len"
}

# ── OLMo-1B (RoPE + LayerNorm) ──────────────────────────────────────
# Track A: wiki40b/1024 already rerun; need the other 5
echo ""; echo "########## OLMo-1B ##########"
for d in "${DATASETS[@]}"; do
    for l in "${LENGTHS[@]}"; do
        # Skip the one already rerun
        if [[ "$d" == "wiki40b_en_pre2019" && "$l" == "1024" ]]; then
            echo "Skipping Track A olmo-1b/${d}/len_${l} (already rerun)"
            continue
        fi
        run_track_a "olmo-1b" "$d" "$l"
    done
done
for d in "${DATASETS[@]}"; do
    for l in "${LENGTHS[@]}"; do
        run_track_b "olmo-1b" "$d" "$l"
    done
done

# ── LLaMA-3.2-1B (RoPE + RMSNorm) ──────────────────────────────────
echo ""; echo "########## LLaMA-3.2-1B ##########"
for d in "${DATASETS[@]}"; do
    for l in "${LENGTHS[@]}"; do
        run_track_a "llama-3.2-1b" "$d" "$l"
    done
done
for d in "${DATASETS[@]}"; do
    for l in "${LENGTHS[@]}"; do
        run_track_b "llama-3.2-1b" "$d" "$l"
    done
done

# ── TinyLlama-1.1B (RoPE + RMSNorm) ────────────────────────────────
echo ""; echo "########## TinyLlama-1.1B ##########"
for d in "${DATASETS[@]}"; do
    for l in "${LENGTHS[@]}"; do
        run_track_a "tinyllama-1.1b" "$d" "$l"
    done
done
for d in "${DATASETS[@]}"; do
    for l in "${LENGTHS[@]}"; do
        run_track_b "tinyllama-1.1b" "$d" "$l"
    done
done

# ── TinyLlama-NoPE-1.1B (NoPE + RMSNorm) ───────────────────────────
# Track A: synthetic/1024 already rerun; need the other 5
echo ""; echo "########## TinyLlama-NoPE-1.1B ##########"
for d in "${DATASETS[@]}"; do
    for l in "${LENGTHS[@]}"; do
        if [[ "$d" == "synthetic_random" && "$l" == "1024" ]]; then
            echo "Skipping Track A tinyllama-nope-1.1b/${d}/len_${l} (already rerun)"
            continue
        fi
        run_track_a "tinyllama-nope-1.1b" "$d" "$l"
    done
done
for d in "${DATASETS[@]}"; do
    for l in "${LENGTHS[@]}"; do
        run_track_b "tinyllama-nope-1.1b" "$d" "$l"
    done
done

# ── Spectral analysis for ALL 6 models (self-gating) ────────────────
echo ""; echo "########## Spectral Analysis (all models) ##########"
ALL_MODELS=("gpt2-small" "gpt2-medium" "olmo-1b" "llama-3.2-1b" "tinyllama-1.1b" "tinyllama-nope-1.1b")
for m in "${ALL_MODELS[@]}"; do
    for d in "${DATASETS[@]}"; do
        for l in "${LENGTHS[@]}"; do
            run_spectral "$m" "$d" "$l"
        done
    done
done

echo ""
echo "=========================================="
echo "  ALL RERUNS COMPLETE"
echo "=========================================="
