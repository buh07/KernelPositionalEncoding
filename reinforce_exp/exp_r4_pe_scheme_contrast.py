#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, command_manifest, ensure_dir, safe_float, timestamp_now, write_json  # noqa: E402
from experiment3.phase2.exp3p2k_non_rope_control import run_model  # noqa: E402

DEFAULT_OUT = RESULTS_ROOT / "exp_r4_pe_scheme_contrast"


def run_all(
    *,
    output_root: Path,
    device: str,
    num_sequences: int,
    seq_len: int,
    top_k_dims: int,
    synthetic_target_per_cell: int,
    seed: int,
) -> dict[str, Any]:
    ensure_dir(output_root)

    models = ["gpt2-medium", "gpt2-small"]
    reports: list[dict[str, Any]] = []
    for model_name in models:
        rep = run_model(
            model_name=model_name,
            output_root=output_root,
            device=device,
            num_sequences=int(num_sequences),
            seq_len=int(seq_len),
            top_k_dims=int(top_k_dims),
            synthetic_target_per_cell=int(synthetic_target_per_cell),
            seed=int(seed),
        )
        reports.append(rep)

    def _verdict(rep: dict[str, Any]) -> dict[str, Any]:
        v = rep.get("verdict", {})
        return {
            "si_structure_detected": bool(v.get("si_structure_detected", False)),
            "boundary_non_trivial_after_control": bool(v.get("boundary_non_trivial_after_control", False)),
            "rope_confound_weakened_for_anchor": bool(v.get("rope_confound_weakened_for_anchor", False)),
        }

    compact = {str(r.get("model")): _verdict(r) for r in reports}
    n_pos = sum(
        1
        for v in compact.values()
        if v["si_structure_detected"] and v["boundary_non_trivial_after_control"]
    )

    high_r2 = [safe_float(r.get("r2_summary", {}).get("high_mean_r2")) for r in reports]
    low_r2 = [safe_float(r.get("r2_summary", {}).get("low_mean_r2")) for r in reports]
    d_vals = [safe_float(r.get("boundary_control", {}).get("post_ablation_high_vs_low_d")) for r in reports]

    summary = {
        "timestamp": timestamp_now(),
        "experiment": "EXP-R4",
        "models": models,
        "reports": reports,
        "aggregate": {
            "n_models": int(len(models)),
            "n_positive_replications": int(n_pos),
            "mean_high_r2": safe_float(np.nanmean(np.asarray(high_r2, dtype=float))),
            "mean_low_r2": safe_float(np.nanmean(np.asarray(low_r2, dtype=float))),
            "mean_post_ablation_d": safe_float(np.nanmean(np.asarray(d_vals, dtype=float))),
        },
        "verdict": {
            "pe_scheme_generality_supported": bool(n_pos >= 1),
            "claim_impact": "strengthens_main_claim" if n_pos >= 1 else "requires_claim_downgrade",
        },
    }

    write_json(output_root / "pe_scheme_comparison.json", summary)
    write_json(
        output_root / "manifest.json",
        command_manifest(
            experiment_id="EXP-R4",
            command="pe_scheme_contrast",
            model="gpt2-medium+gpt2-small",
            seed_set=[int(seed)],
            extras={
                "device": device,
                "num_sequences": int(num_sequences),
                "seq_len": int(seq_len),
                "top_k_dims": int(top_k_dims),
                "synthetic_target_per_cell": int(synthetic_target_per_cell),
                "output_root": str(output_root),
            },
        ),
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-R4: PE-scheme contrast extension")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-sequences", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--top-k-dims", type=int, default=16)
    p.add_argument("--synthetic-target-per-cell", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_all(
        output_root=Path(args.output_root),
        device=str(args.device),
        num_sequences=max(8, int(args.num_sequences)),
        seq_len=max(128, int(args.seq_len)),
        top_k_dims=max(1, int(args.top_k_dims)),
        synthetic_target_per_cell=max(64, int(args.synthetic_target_per_cell)),
        seed=int(args.seed),
    )
    print(f"[EXP-R4] wrote {Path(args.output_root) / 'pe_scheme_comparison.json'}")
    print(f"[EXP-R4] aggregate={summary['aggregate']}")


if __name__ == "__main__":
    main()
