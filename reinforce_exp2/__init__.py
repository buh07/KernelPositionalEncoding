"""reinforce_exp2: A/B/C narrative reinforcement experiment suite."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent

__all__ = ["ROOT", "PACKAGE_ROOT"]
