#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from shared.config import default_paths
from shared.specs import DatasetSpec, ExperimentGrid, ModelSpec


def load_grid(module_name: str) -> ExperimentGrid:
    module = importlib.import_module(f"{module_name}.config")
    return module.EXPERIMENT_GRID


def download_models(
    grid: ExperimentGrid,
    models_dir: Path,
    force: bool,
    allowed_names: set[str] | None,
) -> None:
    for model in grid.models:
        if not _should_process(model.name, allowed_names):
            continue
        target_dir = models_dir / model.name
        target_dir.mkdir(parents=True, exist_ok=True)
        if target_dir.exists() and any(target_dir.iterdir()) and not force:
            print(f"[download] model {model.name} already present, skipping")
            continue
        print(f"[download] model {model.name} -> {target_dir}")
        snapshot_kwargs, runtime_only = _partition_model_kwargs(model)
        if runtime_only:
            print(f"[download] note: runtime-only kwargs for {model.name}: {runtime_only}")
        try:
            snapshot_download(
                repo_id=model.hf_id,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                cache_dir=target_dir,
                resume_download=not force,
                force_download=force,
                **snapshot_kwargs,
            )
        except (GatedRepoError, HfHubHTTPError) as exc:
            if isinstance(exc, HfHubHTTPError) and getattr(exc.response, "status_code", None) != 403:
                raise
            print(
                f"[download] warning: access denied for {model.hf_id}. "
                "Ensure huggingface-cli login + repository approval."
            )


def _download_dataset_snapshot(dataset: DatasetSpec, target_dir: Path, force: bool) -> None:
    sentinel = target_dir / ".download_complete"
    if sentinel.exists() and not force:
        print(f"[download] dataset {dataset.name} snapshot cached, skipping")
        return
    print(f"[download] dataset {dataset.name} snapshot -> {target_dir}")
    allow_patterns = None
    if dataset.snapshot_files:
        allow_patterns = dataset.snapshot_files
    elif dataset.snapshot_patterns:
        allow_patterns = dataset.snapshot_patterns
    snapshot_download(
        repo_id=dataset.hf_id,
        repo_type="dataset",
        local_dir=target_dir,
        cache_dir=target_dir,
        resume_download=not force,
        force_download=force,
        allow_patterns=allow_patterns,
    )
    _write_sentinel(
        sentinel,
        {
            "dataset": dataset.name,
            "hf_id": dataset.hf_id,
            "strategy": dataset.download_strategy,
            "allow_patterns": allow_patterns,
        },
    )


def download_datasets(
    grid: ExperimentGrid,
    data_dir: Path,
    force: bool,
    allowed_names: set[str] | None,
) -> None:
    for dataset in grid.datasets:
        if not _should_process(dataset.name, allowed_names):
            continue
        if dataset.kind == "synthetic":
            continue
        if dataset.download_strategy == "snapshot":
            target_dir = data_dir / "snapshots" / dataset.name
            target_dir.mkdir(parents=True, exist_ok=True)
            _download_dataset_snapshot(dataset, target_dir, force)
            continue
        target_cache = data_dir / "hf_cache" / dataset.name
        target_cache.mkdir(parents=True, exist_ok=True)
        sentinel = target_cache / ".download_complete"
        if sentinel.exists() and not force:
            print(f"[download] dataset {dataset.name} cached, skipping")
            continue
        name = dataset.config if dataset.config else None
        print(f"[download] dataset {dataset.name} ({dataset.hf_id})")
        load_dataset(
            dataset.hf_id,
            name=name,
            split=dataset.split,
            cache_dir=str(target_cache),
        )
        _write_sentinel(
            sentinel,
            {
                "dataset": dataset.name,
                "hf_id": dataset.hf_id,
                "split": dataset.split,
                "config": dataset.config,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download shared Experiment assets")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=["experiment1"],
        help="Experiment modules to pull specs from",
    )
    parser.add_argument("--force", action="store_true", help="Redownload assets even if cached")
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only download models (skip datasets)",
    )
    parser.add_argument(
        "--datasets-only",
        action="store_true",
        help="Only download datasets (skip models)",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional whitelist of asset names to download",
    )
    args = parser.parse_args()

    if args.models_only and args.datasets_only:
        parser.error("Cannot set both --models-only and --datasets-only.")

    allowed_names = set(args.names) if args.names else None

    paths = default_paths()
    for exp in args.experiments:
        grid = load_grid(exp)
        if not args.datasets_only:
            download_models(grid, paths.models_dir, args.force, allowed_names)
        if not args.models_only:
            download_datasets(grid, paths.data_dir, args.force, allowed_names)


def _should_process(name: str, allowed_names: set[str] | None) -> bool:
    if not allowed_names:
        return True
    return name in allowed_names


def _write_sentinel(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _partition_model_kwargs(model: ModelSpec) -> tuple[dict, dict]:
    allowed_keys = {"revision", "allow_patterns", "ignore_patterns"}
    kwargs = model.download_kwargs or {}
    snapshot_kwargs = {}
    runtime_only = {}
    for key, value in kwargs.items():
        if key in allowed_keys:
            snapshot_kwargs[key] = value
        else:
            runtime_only[key] = value
    return snapshot_kwargs, runtime_only


if __name__ == "__main__":
    main()
