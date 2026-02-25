from __future__ import annotations

"""Experiment 1 configuration derived from experiment1overview.md."""

from shared.specs import DatasetSpec, ExperimentGrid, ModelSpec, SequenceLengthSpec

# Models span the preregistered 2×3 norm/PE grid.
MODELS = [
    ModelSpec(
        name="gpt2-small",
        hf_id="openai-community/gpt2",
        norm="LayerNorm",
        pe_scheme="learned-absolute",
        notes="Absolute PE + LayerNorm baseline (117M params).",
    ),
    ModelSpec(
        name="gpt2-medium",
        hf_id="openai-community/gpt2-medium",
        norm="LayerNorm",
        pe_scheme="learned-absolute",
        notes="Absolute PE + LayerNorm size-scaled baseline (345M params).",
    ),
    ModelSpec(
        name="olmo-1b",
        hf_id="allenai/OLMo-1B-hf",
        norm="LayerNorm",
        pe_scheme="RoPE",
        notes="RoPE + LayerNorm (16 layers, d_model=2048).",
        download_kwargs=(("torch_dtype", "bfloat16"),),
    ),
    ModelSpec(
        name="llama-3.2-1b",
        hf_id="meta-llama/Llama-3.2-1B",
        norm="RMSNorm",
        pe_scheme="RoPE",
        notes="RoPE + RMSNorm with grouped-query attention (32Q/8KV heads).",
    ),
    ModelSpec(
        name="tinyllama-1.1b",
        hf_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        norm="RMSNorm",
        pe_scheme="RoPE",
        notes="RoPE + RMSNorm architecture-matched pair for NoPE control.",
    ),
    ModelSpec(
        name="tinyllama-nope-1.1b",
        hf_id="AntNLP/TinyLlama-NoPE-1.1B",
        norm="RMSNorm",
        pe_scheme="None",
        notes="No positional encoding control (RoPE removed).",
    ),
]

# Dataset trio: natural text, structured code, and synthetic controls.
DATASETS = [
    DatasetSpec(
        name="wiki40b_en_pre2019",
        hf_id="wiki40b",
        config="en",
        split="train",
        kind="wikipedia",
        needs_center_split=True,
        max_sequences=200_000,
        notes=(
            "Using wiki40b English (curated dumps through 2019) to satisfy the "
            "pretraining-cutoff requirement from experiment1overview.md."
        ),
    ),
    DatasetSpec(
        name="codesearchnet_python_snapshot",
        hf_id="codeparrot/github-code",
        split="train",
        kind="code",
        download_strategy="snapshot",
        snapshot_files=tuple(
            f"data/train-{idx:05d}-of-01126.parquet" for idx in range(2)
        ),
        needs_center_split=True,
        max_sequences=60_000,
        notes=(
            "CodeSearchNet-style Python subset stored as deterministic parquet shards; "
            "snapshotting ensures reproducible sequence pools."
        ),
    ),
    DatasetSpec(
        name="synthetic_random",
        hf_id=None,
        split="",
        kind="synthetic",
        needs_center_split=False,
        notes="Uniform random tokens per model vocabulary; Track B centering skipped.",
    ),
]

SEQUENCE_LENGTHS = [
    SequenceLengthSpec(tokens=256),
    SequenceLengthSpec(tokens=1024),
]

EXPERIMENT_GRID = ExperimentGrid(
    models=MODELS,
    datasets=DATASETS,
    sequence_lengths=SEQUENCE_LENGTHS,
)
