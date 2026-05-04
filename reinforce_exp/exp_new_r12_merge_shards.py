#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, command_manifest, ensure_dir, safe_float, timestamp_now, write_json  # noqa: E402
from reinforce_exp.exp_new_r12_ordering_control import (  # noqa: E402
    _fit_curve_models,
    _load_canonical_fit,
    _mean_bic_delta,
    _mean_bic_delta_pairwise,
    _ordering_verdict,
)


def _parse_csv_paths(raw: str) -> list[Path]:
    paths: list[Path] = []
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        paths.append(Path(tok))
    if not paths:
        raise ValueError("At least one shard root is required")
    return paths


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_model_shards(
    *,
    model: str,
    shard_roots: list[Path],
    output_root: Path,
    expected_ordering_start: int,
    expected_ordering_stop: int,
) -> dict[str, Any]:
    out_dir = output_root / model
    ensure_dir(out_dir)

    shard_dfs: list[pd.DataFrame] = []
    ordering_payload_map: dict[int, dict[str, Any]] = {}
    shard_runtime_meta: list[dict[str, Any]] = []

    for shard_root in shard_roots:
        model_dir = shard_root / model
        curve_path = model_dir / "cumulative_ablation_curve_random.parquet"
        vote_path = model_dir / "ordering_vote_summary.json"
        payload_path = model_dir / "ordering_payloads.json"
        summary_path = model_dir / "summary.json"
        for req in [curve_path, vote_path, payload_path, summary_path]:
            if not req.exists():
                raise FileNotFoundError(f"Missing shard artifact: {req}")

        shard_dfs.append(pd.read_parquet(curve_path))
        payload = _load_json(payload_path)
        for row in payload.get("orderings", []):
            oid = int(row.get("ordering_id"))
            if oid in ordering_payload_map:
                raise RuntimeError(f"Duplicate ordering_id={oid} across shards")
            ordering_payload_map[oid] = row

        summary = _load_json(summary_path)
        shard_runtime_meta.append(
            {
                "shard_root": str(shard_root),
                "runtime_metadata": summary.get("runtime_metadata", {}),
            }
        )

    if not shard_dfs:
        raise RuntimeError("No shard dataframes loaded")

    full_df = pd.concat(shard_dfs, ignore_index=True, sort=False)
    ordering_ids = sorted(int(x) for x in full_df["ordering_id"].dropna().unique().tolist())
    expected_ids = list(range(int(expected_ordering_start), int(expected_ordering_stop)))
    if ordering_ids != expected_ids:
        raise RuntimeError(
            f"Merged ordering ids mismatch. got={ordering_ids} expected={expected_ids}"
        )

    out_curve = out_dir / "cumulative_ablation_curve_random.parquet"
    full_df.to_parquet(out_curve, index=False)

    per_ordering_fits: dict[str, dict[str, Any]] = {}
    ordering_rows: list[dict[str, Any]] = []
    ordering_bic_delta: list[float] = []
    ordering_bic_delta_logistic: list[float] = []

    for ordering_id, odf in full_df.groupby("ordering_id", sort=True):
        fits: dict[str, Any] = {}
        for (task, metric_name), gdf in odf.groupby(["task", "metric_name"], sort=True):
            key = f"{task}::{metric_name}"
            fits[key] = _fit_curve_models(gdf)
        per_ordering_fits[str(int(ordering_id))] = fits
        verdict = _ordering_verdict(fits)
        bic_delta = _mean_bic_delta(fits)
        bic_delta_log = _mean_bic_delta_pairwise(fits, left="logistic_sigmoid", right="threshold_piecewise")
        ordering_bic_delta.append(bic_delta)
        ordering_bic_delta_logistic.append(bic_delta_log)
        ordering_rows.append(
            {
                "ordering_id": int(ordering_id),
                "order_seed": int(ordering_payload_map[int(ordering_id)]["order_seed"])
                if int(ordering_id) in ordering_payload_map and ordering_payload_map[int(ordering_id)].get("order_seed") is not None
                else None,
                **verdict,
                "mean_bic_delta_linear_minus_threshold": safe_float(bic_delta),
                "mean_bic_delta_logistic_minus_threshold": safe_float(bic_delta_log),
            }
        )

    n_ord = len(ordering_rows)
    n_majority = sum(1 for r in ordering_rows if bool(r.get("threshold_majority", False)))
    frac_majority = float(n_majority / max(1, n_ord))
    if frac_majority >= 0.8:
        robustness = "strong_replication"
    elif frac_majority >= 0.5:
        robustness = "partial_replication"
    else:
        robustness = "failure"

    finite_bic = np.asarray([x for x in ordering_bic_delta if np.isfinite(x)], dtype=float)
    finite_bic_logistic = np.asarray([x for x in ordering_bic_delta_logistic if np.isfinite(x)], dtype=float)

    # Merge runtime metadata for auditability.
    synth_ceilings: dict[str, int] = {}
    eff_bs_ntp_vals: list[int] = []
    ntp_coverage_vals: list[dict[str, Any]] = []
    for row in shard_runtime_meta:
        meta = row.get("runtime_metadata", {}) or {}
        for k, v in (meta.get("effective_batch_size_synth") or {}).items():
            try:
                iv = int(v)
            except Exception:
                continue
            if k in synth_ceilings:
                synth_ceilings[k] = min(synth_ceilings[k], iv)
            else:
                synth_ceilings[k] = iv
        try:
            eff_bs_ntp_vals.append(int(meta.get("effective_batch_size_ntp")))
        except Exception:
            pass
        cov = meta.get("ntp_coverage")
        if isinstance(cov, dict):
            ntp_coverage_vals.append(cov)

    ntp_cov = ntp_coverage_vals[0] if ntp_coverage_vals else None
    if ntp_cov is not None:
        for other in ntp_coverage_vals[1:]:
            if other != ntp_cov:
                raise RuntimeError(f"Inconsistent ntp_coverage across shards: {ntp_coverage_vals}")

    aggregate = {
        "timestamp": timestamp_now(),
        "experiment": "NEW-R12",
        "model": model,
        "n_orderings": int(n_ord),
        "ordering_id_start": int(expected_ordering_start),
        "ordering_id_stop": int(expected_ordering_stop),
        "n_threshold_majority": int(n_majority),
        "fraction_threshold_majority": frac_majority,
        "robustness_assessment": robustness,
        "acceptance_thresholds": {
            "strong_replication_ge": 0.8,
            "partial_replication_ge": 0.5,
        },
        "mean_random_bic_delta_linear_minus_threshold": safe_float(np.mean(finite_bic) if finite_bic.size else float("nan")),
        "std_random_bic_delta_linear_minus_threshold": safe_float(np.std(finite_bic, ddof=1) if finite_bic.size >= 2 else float("nan")),
        "mean_random_bic_delta_logistic_minus_threshold": safe_float(
            np.mean(finite_bic_logistic) if finite_bic_logistic.size else float("nan")
        ),
        "std_random_bic_delta_logistic_minus_threshold": safe_float(
            np.std(finite_bic_logistic, ddof=1) if finite_bic_logistic.size >= 2 else float("nan")
        ),
        "effective_batch_size_synth": synth_ceilings,
        "effective_batch_size_ntp": min(eff_bs_ntp_vals) if eff_bs_ntp_vals else None,
        "ntp_coverage": ntp_cov,
    }

    canonical = _load_canonical_fit(model)
    canonical_vs_random: dict[str, Any] = {
        "timestamp": timestamp_now(),
        "experiment": "NEW-R12",
        "model": model,
        "canonical_curve_fit_path": None,
        "canonical_mean_bic_delta_linear_minus_threshold": float("nan"),
        "canonical_mean_bic_delta_logistic_minus_threshold": float("nan"),
        "random_mean_bic_delta_linear_minus_threshold": aggregate["mean_random_bic_delta_linear_minus_threshold"],
        "random_mean_bic_delta_logistic_minus_threshold": aggregate["mean_random_bic_delta_logistic_minus_threshold"],
        "random_distribution": [safe_float(x) for x in ordering_bic_delta],
        "random_distribution_logistic_minus_threshold": [safe_float(x) for x in ordering_bic_delta_logistic],
        "ranked_ordering_outlier": None,
    }
    if canonical is not None:
        canonical_fits = canonical.get("curve_fits", {})
        can_delta = _mean_bic_delta(canonical_fits)
        can_delta_log = _mean_bic_delta_pairwise(canonical_fits, left="logistic_sigmoid", right="threshold_piecewise")
        canonical_vs_random["canonical_mean_bic_delta_linear_minus_threshold"] = safe_float(can_delta)
        canonical_vs_random["canonical_mean_bic_delta_logistic_minus_threshold"] = safe_float(can_delta_log)
        canonical_vs_random["canonical_curve_fit_path"] = str(
            ROOT / "results" / "experiment3_phase2" / "exp3p2c_redundancy_quantification" / model / "curve_fit_comparison.json"
        )
        arr = np.asarray([x for x in ordering_bic_delta if np.isfinite(x)], dtype=float)
        if np.isfinite(can_delta) and arr.size >= 2:
            z = float((can_delta - float(np.mean(arr))) / max(float(np.std(arr, ddof=1)), 1e-8))
            canonical_vs_random["ranked_ordering_outlier"] = {
                "z_score": safe_float(z),
                "abs_z_gt_2": bool(abs(z) > 2.0),
            }

    write_json(out_dir / "ordering_vote_summary.json", {"rows": ordering_rows, "per_ordering_fits": per_ordering_fits})
    write_json(out_dir / "aggregate_robustness.json", aggregate)
    write_json(out_dir / "canonical_vs_random_comparison.json", canonical_vs_random)
    write_json(out_dir / "ordering_payloads.json", {"orderings": [ordering_payload_map[k] for k in sorted(ordering_payload_map.keys())]})

    report = {
        "timestamp": timestamp_now(),
        "experiment": "NEW-R12",
        "model": model,
        "output_dir": str(out_dir),
        "runtime_sec": float("nan"),
        "aggregate": aggregate,
        "canonical_vs_random": canonical_vs_random,
        "runtime_metadata": {
            "ordering_id_start": int(expected_ordering_start),
            "ordering_id_stop": int(expected_ordering_stop),
            "effective_batch_size_synth": aggregate.get("effective_batch_size_synth"),
            "effective_batch_size_ntp": aggregate.get("effective_batch_size_ntp"),
            "ntp_coverage": aggregate.get("ntp_coverage"),
            "source_shards": shard_runtime_meta,
        },
    }
    write_json(out_dir / "summary.json", report)

    write_json(
        out_dir / "manifest.json",
        command_manifest(
            experiment_id="NEW-R12",
            command="merge_shards",
            model=model,
            extras={
                "shard_roots": [str(x) for x in shard_roots],
                "output_root": str(output_root),
                "ordering_id_start": int(expected_ordering_start),
                "ordering_id_stop": int(expected_ordering_stop),
                "effective_batch_size_synth": aggregate.get("effective_batch_size_synth"),
                "effective_batch_size_ntp": aggregate.get("effective_batch_size_ntp"),
                "ntp_coverage": aggregate.get("ntp_coverage"),
            },
        ),
    )

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge NEW-R12 deterministic shard outputs")
    p.add_argument("--model", required=True, choices=["llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"])
    p.add_argument("--shard-roots", required=True, help="Comma-separated shard output roots. Each root must contain <model>/...")
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "exp_new_r12_ordering_control"))
    p.add_argument("--expected-ordering-start", type=int, default=0)
    p.add_argument("--expected-ordering-stop", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rep = merge_model_shards(
        model=str(args.model),
        shard_roots=_parse_csv_paths(args.shard_roots),
        output_root=Path(args.output_root),
        expected_ordering_start=int(args.expected_ordering_start),
        expected_ordering_stop=int(args.expected_ordering_stop),
    )
    print(f"[NEW-R12-MERGE] wrote {Path(args.output_root) / str(args.model) / 'summary.json'}")
    print(f"[NEW-R12-MERGE] robustness={rep['aggregate']['robustness_assessment']}")


if __name__ == "__main__":
    main()
