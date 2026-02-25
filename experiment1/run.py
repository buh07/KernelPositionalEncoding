from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from shared.config import default_paths
from shared.data.tokenization import TokenizationRequest, tokenize_and_save
from shared.specs import DatasetSpec, ExperimentGrid, ModelSpec, SequenceLengthSpec
from shared.utils.logging import get_logger

from experiment1.config import EXPERIMENT_GRID
from experiment1.paths import tokenized_path
from experiment1.track_a import TrackAConfig, TrackARunner
from experiment1.track_b import TrackBConfig, TrackBRunner
from experiment1.spectral import SpectralConfig, SpectralRunner
from experiment1.boundary_analysis import BoundaryConfig, BoundaryRunner

LOGGER = get_logger("experiment1")
DEFAULT_CENTER_COUNT = 100
DEFAULT_EVAL_COUNT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 1 pipelines")
    parser.add_argument(
        "stage",
        choices=["tokenize", "download", "track-a", "track-b", "spectral", "boundary", "all"],
        help="Pipeline stage to execute",
    )
    parser.add_argument("--model", dest="model", default="all", help="Model name or 'all'")
    parser.add_argument("--dataset", dest="dataset", default="all", help="Dataset name or 'all'")
    parser.add_argument("--seq-len", dest="seq_len", default="all", help="Sequence length or 'all'")
    parser.add_argument("--force", action="store_true", help="Recompute even if artifacts exist")
    parser.add_argument(
        "--max-centering",
        type=int,
        default=None,
        help="Optional cap on centering sequences (for smoke tests)",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Optional cap on evaluation sequences (for smoke tests)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed override for tokenization order",
    )
    parser.add_argument(
        "--cleanup-legacy",
        action="store_true",
        help="Remove deprecated .json artifacts before writing new JSONL outputs",
    )
    parser.add_argument(
        "--limit-seqs",
        type=int,
        default=None,
        help="Optional cap on evaluation sequences for Track pipelines",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for Track pipelines (e.g., cuda:0, cpu). Use 'auto' to pick automatically.",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Disable optional Track A heatmap tensor dumps",
    )
    parser.add_argument(
        "--download-names",
        nargs="*",
        default=None,
        help="Asset names to download (applies to --stage download/all)",
    )
    parser.add_argument(
        "--download-models-only",
        action="store_true",
        help="Download stage: only fetch models",
    )
    parser.add_argument(
        "--download-datasets-only",
        action="store_true",
        help="Download stage: only fetch datasets",
    )
    return parser.parse_args()


def filter_specs(
    grid: ExperimentGrid,
    model_name: str,
    dataset_name: str,
    seq_len: str,
) -> Iterable[tuple[ModelSpec, DatasetSpec, SequenceLengthSpec]]:
    def match(value: str, target: str) -> bool:
        return value == "all" or value == target

    for spec in grid.iter_runs():
        model, dataset, length = spec
        if not match(model_name, model.name):
            continue
        if not match(dataset_name, dataset.name):
            continue
        if not match(seq_len, str(length.tokens)):
            continue
        yield spec


def run_tokenization(
    grid: ExperimentGrid,
    model_filter: str,
    dataset_filter: str,
    length_filter: str,
    *,
    force: bool = False,
    max_center: int | None = None,
    max_eval: int | None = None,
    seed: int | None = None,
    cleanup_legacy: bool = False,
) -> None:
    paths = default_paths()
    for model, dataset, seq_len in filter_specs(grid, model_filter, dataset_filter, length_filter):
        out_path = tokenized_path(paths.data_dir, model, dataset, seq_len)
        if cleanup_legacy:
            _cleanup_legacy_artifacts(out_path)
        if out_path.exists() and not force:
            LOGGER.info("Tokenized data exists for %s/%s/%s", model.name, dataset.name, seq_len.tokens)
            continue
        LOGGER.info(
            "Tokenizing dataset: model=%s dataset=%s len=%s", model.name, dataset.name, seq_len.tokens
        )
        if dataset.needs_center_split:
            center_goal = DEFAULT_CENTER_COUNT
            eval_goal = DEFAULT_EVAL_COUNT
            if max_center is not None:
                center_goal = min(center_goal, max_center)
        else:
            center_goal = 0
            eval_goal = DEFAULT_CENTER_COUNT + DEFAULT_EVAL_COUNT
        if max_eval is not None:
            eval_goal = min(eval_goal, max_eval)

        request = TokenizationRequest(
            model=model,
            dataset=dataset,
            seq_len=seq_len,
            center_count=center_goal,
            eval_count=eval_goal,
            data_root=paths.data_dir,
            seed=seed,
        )
        manifest = tokenize_and_save(request, out_path)
        LOGGER.info(
            "Saved tokenized sequences to %s (center=%s eval=%s)",
            out_path,
            manifest["center_count"],
            manifest["eval_count"],
        )
        LOGGER.info("Manifest version=%s total_records=%s", manifest.get("version"), manifest.get("total_records"))


def run_download(
    *,
    force: bool,
    names: list[str] | None,
    models_only: bool,
    datasets_only: bool,
) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "download_assets.py")]
    if force:
        cmd.append("--force")
    if models_only:
        cmd.append("--models-only")
    if datasets_only:
        cmd.append("--datasets-only")
    if names:
        cmd.append("--names")
        cmd.extend(names)
    LOGGER.info("Running download script: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_track_a(
    grid: ExperimentGrid,
    model_filter: str,
    dataset_filter: str,
    length_filter: str,
    *,
    limit_sequences: int | None,
    device: str,
    enable_heatmaps: bool,
) -> None:
    paths = default_paths()
    resolved_device = _resolve_device(device)
    LOGGER.info("Track A using device %s", resolved_device)
    for model, dataset, seq_len in filter_specs(grid, model_filter, dataset_filter, length_filter):
        jsonl = tokenized_path(paths.data_dir, model, dataset, seq_len)
        if not jsonl.exists():
            LOGGER.warning(
                "Skipping Track A; tokenized data missing for %s/%s len=%s",
                model.name,
                dataset.name,
                seq_len.tokens,
            )
            continue
        config = TrackAConfig(
            data_root=paths.data_dir,
            results_root=paths.results_dir,
            limit_sequences=limit_sequences,
            device=resolved_device,
            enable_heatmaps=enable_heatmaps,
        )
        runner = TrackARunner(model, dataset, seq_len, config)
        runner.run()


def run_track_b(
    grid: ExperimentGrid,
    model_filter: str,
    dataset_filter: str,
    length_filter: str,
    *,
    limit_eval_sequences: int | None,
    limit_center_sequences: int | None,
    device: str,
) -> None:
    paths = default_paths()
    resolved_device = _resolve_device(device)
    LOGGER.info("Track B using device %s", resolved_device)
    for model, dataset, seq_len in filter_specs(grid, model_filter, dataset_filter, length_filter):
        jsonl = tokenized_path(paths.data_dir, model, dataset, seq_len)
        if not jsonl.exists():
            LOGGER.warning(
                "Skipping Track B; tokenized data missing for %s/%s len=%s",
                model.name,
                dataset.name,
                seq_len.tokens,
            )
            continue
        config = TrackBConfig(
            data_root=paths.data_dir,
            results_root=paths.results_dir,
            limit_eval_sequences=limit_eval_sequences,
            limit_center_sequences=limit_center_sequences,
            device=resolved_device,
        )
        runner = TrackBRunner(model, dataset, seq_len, config)
        runner.run()


def run_spectral(
    grid: ExperimentGrid,
    model_filter: str,
    dataset_filter: str,
    length_filter: str,
) -> None:
    paths = default_paths()
    config = SpectralConfig(results_root=paths.results_dir)
    for model, dataset, seq_len in filter_specs(grid, model_filter, dataset_filter, length_filter):
        runner = SpectralRunner(model, dataset, seq_len, config)
        runner.run()


def run_boundary(
    grid: ExperimentGrid,
    model_filter: str,
    dataset_filter: str,
    length_filter: str,
    *,
    limit_sequences: int | None,
    device: str,
) -> None:
    paths = default_paths()
    resolved_device = _resolve_device(device)
    LOGGER.info("Boundary analysis using device %s", resolved_device)
    for model, dataset, seq_len in filter_specs(grid, model_filter, dataset_filter, length_filter):
        jsonl = tokenized_path(paths.data_dir, model, dataset, seq_len)
        if not jsonl.exists():
            LOGGER.warning(
                "Skipping boundary; tokenized data missing for %s/%s len=%s",
                model.name, dataset.name, seq_len.tokens,
            )
            continue
        config = BoundaryConfig(
            data_root=paths.data_dir,
            results_root=paths.results_dir,
            limit_sequences=limit_sequences,
            device=resolved_device,
        )
        runner = BoundaryRunner(model, dataset, seq_len, config)
        runner.run()


def _resolve_device(device_arg: str) -> str:
    if device_arg and device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cleanup_legacy_artifacts(jsonl_path: Path) -> None:
    legacy_candidates = [
        jsonl_path.with_suffix(".json"),
        jsonl_path.with_suffix(".json.gz"),
    ]
    for legacy in legacy_candidates:
        if legacy.exists():
            LOGGER.info("Removing legacy artifact %s", legacy)
            legacy.unlink()


def main() -> None:
    args = parse_args()
    if args.stage in ("download", "all"):
        run_download(
            force=args.force,
            names=args.download_names,
            models_only=args.download_models_only,
            datasets_only=args.download_datasets_only,
        )
    if args.stage in ("tokenize", "all"):
        run_tokenization(
            EXPERIMENT_GRID,
            args.model,
            args.dataset,
            args.seq_len,
            force=args.force,
            max_center=args.max_centering,
            max_eval=args.max_eval,
            seed=args.seed,
            cleanup_legacy=args.cleanup_legacy,
        )
    if args.stage in ("track-a", "all"):
        run_track_a(
            EXPERIMENT_GRID,
            args.model,
            args.dataset,
            args.seq_len,
            limit_sequences=args.limit_seqs,
            device=args.device,
            enable_heatmaps=not args.no_heatmap,
        )
    if args.stage in ("track-b", "all"):
        run_track_b(
            EXPERIMENT_GRID,
            args.model,
            args.dataset,
            args.seq_len,
            limit_eval_sequences=args.limit_seqs,
            limit_center_sequences=args.max_centering,
            device=args.device,
        )
    if args.stage in ("spectral", "all"):
        run_spectral(
            EXPERIMENT_GRID,
            args.model,
            args.dataset,
            args.seq_len,
        )
    if args.stage == "boundary":
        run_boundary(
            EXPERIMENT_GRID,
            args.model,
            args.dataset,
            args.seq_len,
            limit_sequences=args.limit_seqs,
            device=args.device,
        )


if __name__ == "__main__":
    main()
