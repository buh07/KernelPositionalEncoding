from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_id: str | None
    split: str
    kind: str  # e.g. "wikipedia", "code", "synthetic"
    config: str | None = None
    download_strategy: str = "datasets"  # "datasets" or "snapshot"
    snapshot_files: tuple[str, ...] | None = None
    snapshot_patterns: tuple[str, ...] | None = None
    needs_center_split: bool = True
    max_sequences: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class SequenceLengthSpec:
    tokens: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    norm: str
    pe_scheme: str
    notes: str = ""
    download_kwargs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ExperimentGrid:
    models: Sequence[ModelSpec]
    datasets: Sequence[DatasetSpec]
    sequence_lengths: Sequence[SequenceLengthSpec]

    def iter_runs(self) -> Iterable[tuple[ModelSpec, DatasetSpec, SequenceLengthSpec]]:
        for model in self.models:
            for dataset in self.datasets:
                for seq_len in self.sequence_lengths:
                    yield model, dataset, seq_len


@dataclass
class RunPaths:
    root: Path
    model: ModelSpec
    dataset: DatasetSpec
    seq_len: SequenceLengthSpec

    def artifact_dir(self) -> Path:
        path = (
            self.root
            / self.model.name
            / self.dataset.name
            / f"len_{self.seq_len.tokens}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class TrackOutputs:
    track_a_dir: Path
    track_b_dir: Path
    extras: dict[str, Path] = field(default_factory=dict)
