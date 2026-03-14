#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SIZE_LIMIT_MB="${SIZE_LIMIT_MB:-20}"
SIZE_LIMIT_BYTES=$((SIZE_LIMIT_MB * 1024 * 1024))

mapfile -t staged_files < <(git diff --cached --name-only --diff-filter=ACMR)
if (( ${#staged_files[@]} == 0 )); then
  echo "[hygiene] no staged files."
  exit 0
fi

echo "[hygiene] staged files: ${#staged_files[@]}"
echo "[hygiene] size limit: ${SIZE_LIMIT_MB}MB"

oversized=0
for path in "${staged_files[@]}"; do
  if [[ -f "$path" ]]; then
    size=$(stat -c%s "$path")
    if (( size > SIZE_LIMIT_BYTES )); then
      echo "[hygiene][ERROR] oversized staged file: $path (${size} bytes)"
      oversized=1
    fi
  fi
done

if (( oversized != 0 )); then
  echo "[hygiene] aborting due to oversized staged files."
  exit 1
fi

if git diff --cached -U0 | rg -n "hf_[A-Za-z0-9]{20,}" >/tmp/prepush_hf_hits.txt 2>/dev/null; then
  echo "[hygiene][ERROR] potential Hugging Face token found in staged diff:"
  cat /tmp/prepush_hf_hits.txt
  exit 1
fi

echo "[hygiene] passed: no oversized files and no HF token patterns in staged diff."
