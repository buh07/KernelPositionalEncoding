#!/usr/bin/env python3
"""Experiment 8 dispatcher.

Usage:
    python -m experiment8.run --list
    python -m experiment8.run 8a --model llama-3.1-8b --device cuda:0
    python -m experiment8.run all --model all --device cuda:0
"""

from __future__ import annotations

import argparse
import subprocess
import sys

EXPERIMENTS: dict[str, str] = {
    "8a": "experiment8.exp8a_sibas",
    "8b": "experiment8.exp8b_pruning",
}

ORDERED_ALL: tuple[str, ...] = ("8a", "8b")

TARGET_MODELS_8A: tuple[str, ...] = ("gpt2-small", "tinyllama-1.1b", "llama-3.1-8b", "olmo-2-7b")
TARGET_MODELS_8B: tuple[str, ...] = ("llama-3.1-8b", "olmo-2-7b")

MODELS_FOR_EXP: dict[str, tuple[str, ...]] = {
    "8a": TARGET_MODELS_8A,
    "8b": TARGET_MODELS_8B,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 8 sub-experiments")
    parser.add_argument("experiment", nargs="?", choices=[*EXPERIMENTS.keys(), "all"])
    parser.add_argument("--model", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def run_one(experiment_id: str, model: str, device: str) -> int:
    module = EXPERIMENTS[experiment_id]
    cmd = [sys.executable, "-m", module, "--model", model, "--device", device]
    print(f"\n{'=' * 72}")
    print(f"Running Experiment {experiment_id.upper()} for model={model} device={device}")
    print(f"{'=' * 72}\n")
    return subprocess.call(cmd)


def main() -> None:
    args = parse_args()
    if args.list or args.experiment is None:
        print("Available Experiment 8 entrypoints:")
        for key, val in EXPERIMENTS.items():
            models = MODELS_FOR_EXP[key]
            print(f"  {key:<4} {val}  (models: {', '.join(models)})")
        print(f"Recommended order: {' -> '.join(ORDERED_ALL)}")
        return

    queue = ORDERED_ALL if args.experiment == "all" else (args.experiment,)

    for exp_id in queue:
        valid_models = MODELS_FOR_EXP[exp_id]
        model_targets = list(valid_models) if args.model == "all" else [args.model]

        for model in model_targets:
            if model not in valid_models:
                print(f"  Skipping {model} for {exp_id} (not in target list)")
                continue
            rc = run_one(exp_id, model, args.device)
            if rc != 0:
                print(f"\n*** {exp_id} failed for {model} with exit code {rc} ***")
                sys.exit(rc)


if __name__ == "__main__":
    main()
