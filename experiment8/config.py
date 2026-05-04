from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "experiment8"
LOG_ROOT = PROJECT_ROOT / "logs" / "experiment8"

# Models with existing R² profiles we can reuse.
TARGET_MODELS_8A: tuple[str, ...] = (
    "gpt2-small",
    "tinyllama-1.1b",
    "llama-3.1-8b",
    "olmo-2-7b",
)

# Models for the cross-tokenizer pruning test (8B).
TARGET_MODELS_8B: tuple[str, ...] = ("llama-3.1-8b", "olmo-2-7b")

# Ablation fractions for 8B progressive ablation.
ABLATION_FRACTIONS: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 35, 50, 75)

# Number of wiki sequences for SIBAS evaluation.
SIBAS_NUM_SEQUENCES: int = 24
SIBAS_SEQ_LEN: int = 512

# Number of wiki sequences for ablation evaluation.
ABLATION_EVAL_SEQUENCES: int = 50
ABLATION_SEQ_LEN: int = 256
