# Experiment 1 Setup Quickstart

This project keeps the Experiment 1 assets (models + datasets) in sync with the preregistered grid described in `experiment1/experiment1overview.md`. Follow the steps below whenever you need to refresh the environment or verify that downloads completed successfully.

## 1. Authenticate with Hugging Face

1. Create or reuse a Hugging Face token with read access to gated repos (https://huggingface.co/settings/tokens).
2. Request access to `meta-llama/Llama-3.2-1B` and any other gated models well in advance—the `scripts/download_assets.py` command will warn if the token is missing approvals.
3. Run the CLI login once per machine (token is stored under `~/.cache/huggingface`):

```bash
. .venv/bin/activate
huggingface-cli login --token <hf_xxx> --add-to-git-credential
```

The download script automatically reuses this credential, so you do not need to export extra environment variables afterwards.

## 2. Install dependencies / virtual environment

```bash
cd "Kernel PE"
./scripts/setup_env.sh
```

This creates `.venv/`, upgrades `pip`, installs the repository in editable mode, and ensures a consistent Python toolchain for all experiment stages.

## 3. Download models and datasets

Use the orchestrator to pull every asset defined in `experiment1/config.py`:

```bash
. .venv/bin/activate
python scripts/download_assets.py
```

Useful flags:

- `--models-only` or `--datasets-only` – limit the run to one asset type.
- `--names gpt2-small wiki40b_en_pre2019` – download a specific subset.
- `--force` – redownload even if sentinel files exist (useful after pruning caches).

Snapshot-backed datasets (e.g., CodeSearchNet parquet shards) are placed under `data/snapshots/<dataset>/`; Hugging Face streaming datasets are cached under `data/hf_cache/<dataset>/`. Each directory contains a `.download_complete` JSON with the exact config and patterns that were fetched, making it easy to audit what is on disk.

## 4. Tokenization sanity checks

After running `python experiment1/run.py tokenize ...`, validate that every `(model, dataset, length)` combination produced the preregistered 200 sequences (100 centering + 100 eval for natural corpora, 200 eval for synthetic):

```bash
python scripts/check_tokenized_manifest.py --model gpt2-small --dataset wiki40b_en_pre2019 --seq-len 256
```

The script scans `data/experiment1/**/len_*.manifest.json` and fails loudly if any counts drift from the preregistered split.

## 5. Verification checklist

Run the commands below to confirm that downloads, tokenization, and the Track A/B/Spectral pipelines remain aligned with `experiment1overview.md`.

1. **Environment smoke test**
   ```bash
   ./scripts/setup_env.sh
   . .venv/bin/activate
   python - <<'PY'
   import pandas, pyarrow
   print("parquet deps ok")
   PY
   ```
2. **Regenerate every `(model, dataset, length)` split**
   ```bash
   python experiment1/run.py tokenize --model all --dataset all --seq-len all --cleanup-legacy
   python scripts/check_tokenized_manifest.py
   ```
3. **Track A spot runs**
   ```bash
   CUDA_VISIBLE_DEVICES=3 python experiment1/run.py \
     track-a --model gpt2-small --dataset wiki40b_en_pre2019 --seq-len 256 --device cuda
   CUDA_VISIBLE_DEVICES=3 python experiment1/run.py \
     track-a --model olmo-1b --dataset wiki40b_en_pre2019 --seq-len 1024 --device cuda
   CUDA_VISIBLE_DEVICES=3 python experiment1/run.py \
     track-a --model tinyllama-nope-1.1b --dataset synthetic_random --seq-len 1024 --device cuda
   ```
4. **Track B + Spectral probes**
   ```bash
   CUDA_VISIBLE_DEVICES=3 python experiment1/run.py \
     track-b --model gpt2-small --dataset wiki40b_en_pre2019 --seq-len 256 --device cuda
   python experiment1/run.py spectral --model tinyllama-nope-1.1b --dataset synthetic_random --seq-len 256
   ```

--- 

Questions or conflicts between the code and `experiment1overview.md` should be raised immediately before collecting data. The tooling above is designed to surface these mismatches early so the full Track A/B pipelines only run on compliant assets.
