#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, command_manifest, ensure_dir, write_json  # noqa: E402
from experiment3.phase2.exp3p2c2_simultaneous_ablation import _parse_int_tuple, _run_model  # noqa: E402

DEFAULT_OUT = RESULTS_ROOT / "exp_r3_core_replication" / "c2_extended"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-R3 helper: run C.2 nonlinearity for arbitrary model")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fractions", default="0,25,50,75")
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--synthetic-count", type=int, default=100)
    p.add_argument("--batch-size-synth", type=int, default=8)
    p.add_argument("--ntp-count-per-seed", type=int, default=100)
    p.add_argument("--ntp-seq-len", type=int, default=512)
    p.add_argument("--batch-size-ntp", type=int, default=4)
    p.add_argument("--skip-ntp", action="store_true")
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    ensure_dir(out_root)
    fractions = _parse_int_tuple(args.fractions)

    rep = _run_model(
        model_name=str(args.model),
        device=str(args.device),
        output_root=out_root,
        fractions=fractions,
        num_seeds=max(1, int(args.num_seeds)),
        synthetic_count=max(1, int(args.synthetic_count)),
        batch_size_synth=max(1, int(args.batch_size_synth)),
        ntp_count_per_seed=max(1, int(args.ntp_count_per_seed)),
        ntp_seq_len=max(64, int(args.ntp_seq_len)),
        batch_size_ntp=max(1, int(args.batch_size_ntp)),
        skip_ntp=bool(args.skip_ntp),
        bootstrap_samples=max(500, int(args.bootstrap_samples)),
    )

    write_json(
        out_root / str(args.model) / "manifest.json",
        command_manifest(
            experiment_id="EXP-R3-C2",
            command="c2_extended",
            model=str(args.model),
            extras={
                "fractions": list(int(x) for x in fractions),
                "num_seeds": int(args.num_seeds),
                "output_root": str(out_root),
            },
        ),
    )

    write_json(out_root / f"{args.model}_nonlinearity_summary.json", rep)
    print(f"[EXP-R3-C2] wrote {out_root / f'{args.model}_nonlinearity_summary.json'}")


if __name__ == "__main__":
    main()
