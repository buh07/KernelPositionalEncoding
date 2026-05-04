#!/usr/bin/env python3
"""Experiment 5 dispatcher.

Usage:
    python -m experiment5.run --list
    python -m experiment5.run 5a --device cuda:0
    python -m experiment5.run all --device cuda:0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENT5_DIR = Path(__file__).resolve().parent

EXPERIMENTS: dict[str, str] = {
    "5a": "experiment5.exp5a_cross_tokenizer_si_profiling",
    "5b": "experiment5.exp5b_same_family_tokenizer_variation",
    "5c": "experiment5.exp5c_synthetic_tokenizer_perturbation",
    "5d": "experiment5.exp5d_distribution_shift_fragility",
}

ORDERED_ALL: tuple[str, ...] = ("5a", "5b", "5c", "5d")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Experiment 5 scaffold entrypoints")
    p.add_argument("experiment", nargs="?", choices=[*EXPERIMENTS.keys(), "all"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--list", action="store_true")
    return p.parse_args()


def run_one(experiment_id: str, device: str) -> int:
    module = EXPERIMENTS[experiment_id]
    cmd = [sys.executable, "-m", module, "--device", device]
    print(f"\n{'=' * 72}")
    print(f"Running Experiment {experiment_id} scaffold on device={device}")
    print(f"{'=' * 72}\n")
    return subprocess.call(cmd)


def main() -> None:
    args = parse_args()
    if args.list or args.experiment is None:
        print("Available Experiment 5 entrypoints:")
        for key, val in EXPERIMENTS.items():
            print(f"  {key:<4} {val}")
        print("Recommended order: 5a -> 5b ; then 5c and 5d in parallel.")
        return

    queue = ORDERED_ALL if args.experiment == "all" else (args.experiment,)
    for exp_id in queue:
        rc = run_one(exp_id, args.device)
        if rc != 0:
            print(f"\n*** {exp_id} failed with exit code {rc} ***")
            sys.exit(rc)


if __name__ == "__main__":
    main()
