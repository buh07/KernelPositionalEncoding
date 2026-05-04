from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "experiment6"
LOG_ROOT = PROJECT_ROOT / "logs" / "experiment6"

TARGET_MODELS: tuple[str, ...] = ("tinyllama-1.1b", "llama-3.1-8b", "olmo-2-7b")

# Experiment 6 conditions (approaches).
CONDITIONS_6A: tuple[str, ...] = (
    "f_gradient_routed",        # Math gradients only through SI heads
    "d_full_qlora_baseline",    # Standard FT baseline (reused from exp4)
)

CONDITIONS_6B: tuple[str, ...] = (
    "g_anti_localization",      # Penalize math in non-SI channels
    "d_full_qlora_baseline",
)

CONDITIONS_6C: tuple[str, ...] = (
    "h_si_distillation",        # Two-phase: teacher -> SI-only student
)

CONDITIONS_6D: tuple[str, ...] = (
    "i_contrastive_channel",    # Contrastive channel assignment loss
    "d_full_qlora_baseline",
)

# Position-invariance evaluation: context positions at which to evaluate math.
POSITION_EVAL_SLOTS: tuple[int, ...] = (0, 64, 128, 192, 256, 384, 512)

# SI-optimized tokenizer experiment (6E).
TOKENIZER_MATH_OPERATORS: tuple[str, ...] = ("+", "-", "=", "*", "mod", "carry")
