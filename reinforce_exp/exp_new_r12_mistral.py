#!/usr/bin/env python3
"""Run NEW-R12 random-order cumulative ablation for Mistral-7B-v0.1.

This is a thin wrapper around exp_new_r12_ordering_control.run_model
that adds mistral-7b-v0.1 support without modifying the original script.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, ensure_dir, timestamp_now, write_json  # noqa: E402
from reinforce_exp.exp_new_r12_ordering_control import run_model  # noqa: E402

DEFAULT_OUT = RESULTS_ROOT / "exp_new_r12_ordering_control"


def main() -> None:
    p = argparse.ArgumentParser(description="NEW-R12 for Mistral-7B-v0.1")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--fractions", default="0,1,2,5,10,15,20,25,50")
    p.add_argument("--num-orderings", type=int, default=10)
    p.add_argument("--ordering-seed-base", type=int, default=20260417)
    p.add_argument("--ordering-id-start", type=int, default=0)
    p.add_argument("--ordering-id-stop", type=int, default=None)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--synthetic-count", type=int, default=100)
    p.add_argument("--batch-size-synth", type=int, default=8)
    p.add_argument("--ntp-count-per-seed", type=int, default=100)
    p.add_argument("--ntp-seq-len", type=int, default=512)
    p.add_argument("--batch-size-ntp", type=int, default=4)
    p.add_argument("--skip-ntp", action="store_true")
    args = p.parse_args()

    fractions_str = str(args.fractions)
    fracs = tuple(int(x.strip()) for x in fractions_str.split(","))

    out_root = ensure_dir(Path(args.output_root))

    report = run_model(
        model_name="mistral-7b-v0.1",
        device=str(args.device),
        output_root=out_root,
        fractions=fracs,
        num_orderings=int(args.num_orderings),
        ordering_seed_base=int(args.ordering_seed_base),
        ordering_id_start=max(0, int(args.ordering_id_start)),
        ordering_id_stop=int(args.ordering_id_stop) if args.ordering_id_stop is not None else None,
        num_seeds=int(args.num_seeds),
        synthetic_count=int(args.synthetic_count),
        batch_size_synth=int(args.batch_size_synth),
        ntp_count_per_seed=int(args.ntp_count_per_seed),
        ntp_seq_len=int(args.ntp_seq_len),
        batch_size_ntp=int(args.batch_size_ntp),
        skip_ntp=bool(args.skip_ntp),
    )

    agg_path = out_root / "aggregate_summary.json"
    lock_path = out_root / "aggregate_summary.lock"
    with lock_path.open("a") as _lock_fh:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX)
        try:
            existing: dict = {}
            if agg_path.exists():
                try:
                    existing = json.loads(agg_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            models_section = existing.get("models", {})
            models_section["mistral-7b-v0.1"] = report
            payload = {
                "timestamp": timestamp_now(),
                "experiment": "NEW-R12",
                "models": models_section,
            }
            write_json(agg_path, payload)
        finally:
            fcntl.flock(_lock_fh, fcntl.LOCK_UN)
    print(f"[NEW-R12-Mistral] done. Summary written to {out_root / 'mistral-7b-v0.1' / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
