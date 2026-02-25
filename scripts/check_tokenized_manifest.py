#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from experiment1.config import EXPERIMENT_GRID
from shared.config import default_paths
from shared.data.tokenization import MANIFEST_SUFFIX
from shared.specs import DatasetSpec, ExperimentGrid, ModelSpec, SequenceLengthSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Experiment 1 tokenization manifests")
    parser.add_argument("--model", default="all", help="Model name or 'all'")
    parser.add_argument("--dataset", default="all", help="Dataset name or 'all'")
    parser.add_argument("--seq-len", dest="seq_len", default="all", help="Sequence length or 'all'")
    return parser.parse_args()


def filter_specs(
    grid: ExperimentGrid,
    model_name: str,
    dataset_name: str,
    seq_len: str,
) -> Iterable[tuple[ModelSpec, DatasetSpec, SequenceLengthSpec]]:
    def match(value: str, target: str) -> bool:
        return value == "all" or value == target

    for model in grid.models:
        if not match(model_name, model.name):
            continue
        for dataset in grid.datasets:
            if not match(dataset_name, dataset.name):
                continue
            for length in grid.sequence_lengths:
                if not match(seq_len, str(length.tokens)):
                    continue
                yield model, dataset, length


def _base_paths(
    data_dir: Path, model: ModelSpec, dataset: DatasetSpec, seq_len: SequenceLengthSpec
) -> tuple[Path, Path]:
    base = data_dir / "experiment1" / dataset.name / model.name / f"len_{seq_len.tokens}"
    jsonl_path = Path(f"{base}.jsonl")
    manifest_path = Path(f"{base}{MANIFEST_SUFFIX}")
    return jsonl_path, manifest_path


def main() -> None:
    args = parse_args()
    paths = default_paths()
    issues: list[str] = []
    processed = 0

    for model, dataset, seq_len in filter_specs(EXPERIMENT_GRID, args.model, args.dataset, args.seq_len):
        jsonl_fp, manifest_fp = _base_paths(paths.data_dir, model, dataset, seq_len)
        if not jsonl_fp.exists():
            issues.append(f"Missing JSONL for {dataset.name}/{model.name}/len_{seq_len.tokens}")
            continue
        if not manifest_fp.exists():
            issues.append(f"Missing manifest for {dataset.name}/{model.name}/len_{seq_len.tokens}")
            continue
        with manifest_fp.open("r", encoding="utf-8") as mf:
            manifest = json.load(mf)
        issues.extend(_validate_manifest(manifest, dataset, model, seq_len))
        processed += 1

    if issues:
        print("[manifest-check] FAIL")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    print(f"[manifest-check] OK ({processed} manifests validated)")


def _validate_manifest(
    manifest: dict,
    dataset: DatasetSpec,
    model: ModelSpec,
    seq_len: SequenceLengthSpec,
) -> list[str]:
    errors: list[str] = []
    prefix = f"{dataset.name}/{model.name}/len_{seq_len.tokens}"
    needs_split = manifest.get("needs_center_split")
    if needs_split != dataset.needs_center_split:
        errors.append(f"{prefix}: manifest needs_center_split={needs_split} but spec={dataset.needs_center_split}")
    center = manifest.get("center_count")
    eval_count = manifest.get("eval_count")
    expected_center = manifest.get("expected_center")
    expected_eval = manifest.get("expected_eval")
    total_records = manifest.get("total_records")
    if total_records != (center or 0) + (eval_count or 0):
        errors.append(f"{prefix}: total_records mismatch (manifest={total_records}, computed={(center or 0) + (eval_count or 0)})")
    if dataset.needs_center_split:
        if center != expected_center or eval_count != expected_eval:
            errors.append(
                f"{prefix}: expected center/eval ({expected_center}, {expected_eval}) but observed ({center}, {eval_count})"
            )
        if center != 100 or eval_count != 100:
            errors.append(f"{prefix}: preregistered counts must be 100/100 but got {center}/{eval_count}")
    else:
        if center not in (0, None):
            errors.append(f"{prefix}: synthetic dataset should have 0 centering sequences (got {center})")
        if expected_eval is not None and eval_count != expected_eval:
            errors.append(f"{prefix}: expected eval {expected_eval} but observed {eval_count}")
        if eval_count != 200:
            errors.append(f"{prefix}: synthetic dataset eval count must be 200 (got {eval_count})")
    return errors


if __name__ == "__main__":
    main()
