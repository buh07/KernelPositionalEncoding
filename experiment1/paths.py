from __future__ import annotations

from pathlib import Path

from shared.specs import DatasetSpec, ModelSpec, SequenceLengthSpec

JSONL_SUFFIX = ".jsonl"


def tokenized_path(root: Path, model: ModelSpec, dataset: DatasetSpec, seq_len: SequenceLengthSpec) -> Path:
    return root / "experiment1" / dataset.name / model.name / f"len_{seq_len.tokens}{JSONL_SUFFIX}"


def track_a_dir(results_root: Path, model: ModelSpec, dataset: DatasetSpec, seq_len: SequenceLengthSpec) -> Path:
    return results_root / "track_a" / model.name / dataset.name / f"len_{seq_len.tokens}"


def track_b_dir(
    results_root: Path,
    model: ModelSpec,
    dataset: DatasetSpec,
    seq_len: SequenceLengthSpec,
    group: str = "track_b",
) -> Path:
    return results_root / group / model.name / dataset.name / f"len_{seq_len.tokens}"


def spectral_dir(
    results_root: Path,
    model: ModelSpec,
    dataset: DatasetSpec,
    seq_len: SequenceLengthSpec,
    group: str = "spectral",
) -> Path:
    return results_root / group / model.name / dataset.name / f"len_{seq_len.tokens}"
