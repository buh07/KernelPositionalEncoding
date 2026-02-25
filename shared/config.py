from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    """Base directories shared across experiments."""

    root: Path
    data_dir: Path
    models_dir: Path
    results_dir: Path

    @staticmethod
    def from_root(root: Path) -> "PathConfig":
        data_dir = root / "data"
        models_dir = root / "models"
        results_dir = root / "results"
        return PathConfig(root=root, data_dir=data_dir, models_dir=models_dir, results_dir=results_dir)

    def ensure(self) -> None:
        for path in (self.data_dir, self.models_dir, self.results_dir):
            path.mkdir(parents=True, exist_ok=True)


def default_paths() -> PathConfig:
    root = Path(__file__).resolve().parent.parent
    cfg = PathConfig.from_root(root)
    cfg.ensure()
    return cfg
