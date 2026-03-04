#!/usr/bin/env bash
set -euo pipefail

QUEUE_FILE="${1:-logs/trackb/trackb_gpuopt_queue.tsv}"
mkdir -p "$(dirname "$QUEUE_FILE")"

VARIANTS=(
  "track_b_gpuopt_v1|legacy_per_position|na"
  "track_b_canonical_perpos_gpuopt_v1|canonical_per_position|na"
  "track_b_shared_mean_gpuopt_v1|shared_mean|na"
  "track_b_bucketed_mean_b32_gpuopt_v1|bucketed_mean|32"
  "track_b_bucketed_mean_b8_gpuopt_v1|bucketed_mean|8"
  "track_b_bucketed_mean_b16_gpuopt_v1|bucketed_mean|16"
  "track_b_bucketed_mean_b64_gpuopt_v1|bucketed_mean|64"
)

MODELS=(
  "tinyllama-nope-1.1b"
  "tinyllama-1.1b"
  "llama-3.2-1b"
  "gpt2-medium"
  "olmo-1b"
  "gpt2-small"
)

: > "$QUEUE_FILE"
for spec in "${VARIANTS[@]}"; do
  output_group="${spec%%|*}"
  rest="${spec#*|}"
  centering_mode="${rest%%|*}"
  bucket_size="${rest##*|}"
  for model in "${MODELS[@]}"; do
    printf "%s\t%s\t%s\t%s\n" \
      "$output_group" "$centering_mode" "$bucket_size" "$model" >> "$QUEUE_FILE"
  done
done

echo "Wrote $(wc -l < "$QUEUE_FILE") queued jobs to $QUEUE_FILE"
