from __future__ import annotations

from pathlib import Path

from shared.specs import ModelSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "experiment4"
LOG_ROOT = PROJECT_ROOT / "logs" / "experiment4"

TARGET_MODELS: tuple[str, ...] = ("llama-3.1-8b", "olmo-2-7b")

MODELS: dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec(
        name="llama-3.1-8b",
        hf_id="meta-llama/Meta-Llama-3.1-8B",
        norm="RMSNorm",
        pe_scheme="RoPE",
        notes="Primary SI-guided FT target (Llama family).",
    ),
    "olmo-2-7b": ModelSpec(
        name="olmo-2-7b",
        hf_id="allenai/OLMo-2-1124-7B",
        norm="LayerNorm",
        pe_scheme="RoPE",
        notes="Primary SI-guided FT target (OLMo family).",
        download_kwargs=(("torch_dtype", "bfloat16"),),
    ),
}

MATH_TASK_MIX: tuple[str, ...] = (
    "digit_arithmetic",
    "counting_chain",
    "modular_arithmetic",
    "sequence_continuation",
)

CONTROL_MIX: tuple[str, ...] = ("wiki_instruction_control",)

EVAL_CHANNELS: tuple[str, ...] = (
    "math_accuracy",
    "wiki_perplexity",
    "post_ft_r2_profile",
    "post_ft_boundary_detection",
    "post_ft_c1_mini_curve",
)

