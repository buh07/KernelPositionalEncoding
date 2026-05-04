from __future__ import annotations

from pathlib import Path

from shared.specs import ModelSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "experiment5"
LOG_ROOT = PROJECT_ROOT / "logs" / "experiment5"

MODELS: dict[str, ModelSpec] = {
    "pythia-410m": ModelSpec(
        name="pythia-410m",
        hf_id="EleutherAI/pythia-410m",
        norm="LayerNorm",
        pe_scheme="RoPE",
        notes="GPT-NeoX tokenizer family. Uses rotary positional embeddings.",
    ),
    "pythia-1.4b": ModelSpec(
        name="pythia-1.4b",
        hf_id="EleutherAI/pythia-1.4b",
        norm="LayerNorm",
        pe_scheme="RoPE",
        notes="GPT-NeoX tokenizer family. Uses rotary positional embeddings.",
    ),
    "gpt2-small": ModelSpec(
        name="gpt2-small",
        hf_id="openai-community/gpt2",
        norm="LayerNorm",
        pe_scheme="LearnedAbsolutePE",
        notes="GPT-2 tokenizer family.",
    ),
    "gpt2-medium": ModelSpec(
        name="gpt2-medium",
        hf_id="openai-community/gpt2-medium",
        norm="LayerNorm",
        pe_scheme="LearnedAbsolutePE",
        notes="GPT-2 tokenizer family.",
    ),
    "llama-2-7b": ModelSpec(
        name="llama-2-7b",
        hf_id="meta-llama/Llama-2-7b-hf",
        norm="RMSNorm",
        pe_scheme="RoPE",
        notes="Llama-family tokenizer comparison anchor.",
    ),
    "llama-3.1-8b": ModelSpec(
        name="llama-3.1-8b",
        hf_id="meta-llama/Meta-Llama-3.1-8B",
        norm="RMSNorm",
        pe_scheme="RoPE",
        notes="Existing baseline from Experiment 3.",
    ),
    "olmo-2-7b": ModelSpec(
        name="olmo-2-7b",
        hf_id="allenai/OLMo-2-1124-7B",
        norm="LayerNorm",
        pe_scheme="RoPE",
        notes="Existing baseline from Experiment 3.",
        download_kwargs=(("torch_dtype", "bfloat16"),),
    ),
}

TARGET_5A: tuple[str, ...] = (
    "pythia-410m",
    "pythia-1.4b",
    "gpt2-small",
    "gpt2-medium",
    "llama-3.1-8b",
    "olmo-2-7b",
)

TARGET_5B: tuple[str, ...] = ("llama-2-7b", "llama-3.1-8b")
TARGET_5C: tuple[str, ...] = ("llama-3.1-8b", "olmo-2-7b")
TARGET_5D: tuple[str, ...] = ("llama-3.1-8b", "olmo-2-7b")

TOKENIZER_FEATURES: tuple[str, ...] = (
    "space_prefix",
    "capitalization_marker",
    "punctuation_adjacency",
    "token_length_bucket",
)

