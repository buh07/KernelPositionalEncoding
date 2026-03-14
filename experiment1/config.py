from __future__ import annotations

"""Experiment 1 configuration derived from experiment1overview.md."""

from shared.specs import DatasetSpec, ExperimentGrid, ModelSpec, SequenceLengthSpec

# Models span the preregistered 2×3 norm/PE grid.
LEGACY_1B_MODELS = [
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

# Scale-up RoPE profile for follow-on experiments.
SCALEUP_78B_MODELS = [
    ModelSpec(
        name="olmo-2-7b",
        hf_id="allenai/OLMo-2-1124-7B",
        norm="LayerNorm",
        pe_scheme="RoPE",
        notes="OLMo 2 7B scale-up RoPE + LayerNorm model.",
        download_kwargs=(("torch_dtype", "bfloat16"),),
    ),
    ModelSpec(
        name="llama-3.1-8b",
        hf_id="meta-llama/Meta-Llama-3.1-8B",
        norm="RMSNorm",
        pe_scheme="RoPE",
        notes="Llama 3.1 8B scale-up RoPE + RMSNorm model (GQA).",
    ),
    ModelSpec(
        name="gemma-7b",
        hf_id="google/gemma-7b",
        norm="RMSNorm",
        pe_scheme="RoPE",
        notes="Gemma 7B scale-up RoPE + RMSNorm model.",
    ),
]

DEFAULT_MODEL_PROFILE = "legacy_1b"
MODEL_PROFILES = {
    "legacy_1b": tuple(LEGACY_1B_MODELS),
    "scaleup_78b": tuple(SCALEUP_78B_MODELS),
}
SUPPORTED_MODEL_PROFILES = tuple(MODEL_PROFILES.keys())


def get_models_for_profile(model_profile: str = DEFAULT_MODEL_PROFILE) -> tuple[ModelSpec, ...]:
    try:
        return MODEL_PROFILES[str(model_profile)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model profile '{model_profile}'. "
            f"Supported profiles: {', '.join(SUPPORTED_MODEL_PROFILES)}"
        ) from exc


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

LEGACY_SEQUENCE_LENGTHS = [
    SequenceLengthSpec(tokens=256),
    SequenceLengthSpec(tokens=1024),
]
SCALEUP_SEQUENCE_LENGTHS = [
    SequenceLengthSpec(tokens=256),
    SequenceLengthSpec(tokens=512),
    SequenceLengthSpec(tokens=1024),
]
SEQUENCE_LENGTHS_BY_PROFILE = {
    "legacy_1b": tuple(LEGACY_SEQUENCE_LENGTHS),
    "scaleup_78b": tuple(SCALEUP_SEQUENCE_LENGTHS),
}

MODELS = list(get_models_for_profile(DEFAULT_MODEL_PROFILE))
SEQUENCE_LENGTHS = list(SEQUENCE_LENGTHS_BY_PROFILE[DEFAULT_MODEL_PROFILE])


def get_experiment_grid(model_profile: str = DEFAULT_MODEL_PROFILE) -> ExperimentGrid:
    try:
        lengths = SEQUENCE_LENGTHS_BY_PROFILE[str(model_profile)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model profile '{model_profile}'. "
            f"Supported profiles: {', '.join(SUPPORTED_MODEL_PROFILES)}"
        ) from exc
    return ExperimentGrid(
        models=get_models_for_profile(model_profile),
        datasets=DATASETS,
        sequence_lengths=lengths,
    )


EXPERIMENT_GRID = get_experiment_grid()
