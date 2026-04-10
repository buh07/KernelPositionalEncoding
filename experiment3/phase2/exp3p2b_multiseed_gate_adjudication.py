#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _parse_seeds(raw: str) -> list[int]:
    out: list[int] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    if not out:
        raise ValueError("No seeds parsed from --seeds")
    return sorted(set(out))


def _seed_paths(input_root: Path, model: str, seed: int) -> dict[str, Path]:
    base = input_root / f"seed{seed}" / model
    return {
        "base": base,
        "post": base / "post_ablation_t5b_a.json",
        "synthetic": base / "synthetic_boundary_results.json",
        "adversarial": base / "adversarial_sequences.parquet",
    }


def _seed_row(
    *,
    input_root: Path,
    model: str,
    seed: int,
    min_abs_diff: float,
    min_d: float,
    alpha_one_sided: float,
) -> dict[str, Any]:
    paths = _seed_paths(input_root, model, seed)
    row: dict[str, Any] = {
        "seed": int(seed),
        "status": "missing",
        "paths": {k: str(v) for k, v in paths.items()},
    }
    if not paths["synthetic"].exists():
        return row

    synth = _load_json(paths["synthetic"])
    assess = synth.get("prefix_following_assessment", {})

    diff = _safe_float(synth.get("comparisons", {}).get("fake_minus_real_with_prefix"))
    if not np.isfinite(diff):
        fbp = _safe_float(synth.get("cells", {}).get("fake_boundary_with_prefix", {}).get("mean_high_prev_attn"))
        rbp = _safe_float(synth.get("cells", {}).get("real_boundary_with_prefix", {}).get("mean_high_prev_attn"))
        if np.isfinite(fbp) and np.isfinite(rbp):
            diff = float(fbp - rbp)

    d = _safe_float(assess.get("cohens_d_fake_minus_real"))
    p_one = _safe_float(assess.get("p_one_sided_fake_gt_real"))
    n_fake = int(max(0, _safe_float(assess.get("n_fake_with_prefix"))))
    n_real = int(max(0, _safe_float(assess.get("n_real_with_prefix"))))

    gate_flag = bool(
        np.isfinite(diff)
        and np.isfinite(d)
        and np.isfinite(p_one)
        and (diff >= float(min_abs_diff))
        and (d >= float(min_d))
        and (p_one < float(alpha_one_sided))
    )

    row.update(
        {
            "status": "complete",
            "diff_fake_minus_real_with_prefix": _safe_float(diff),
            "cohens_d_fake_minus_real": _safe_float(d),
            "p_one_sided_fake_gt_real": _safe_float(p_one),
            "n_fake_with_prefix": int(n_fake),
            "n_real_with_prefix": int(n_real),
            "seed_gate_flag": bool(gate_flag),
            "prefix_following_artifact_flag": bool(synth.get("prefix_following_artifact_flag", False)),
        }
    )
    return row


def _weighted_stouffer_p(seed_rows: list[dict[str, Any]]) -> float:
    z_terms: list[float] = []
    w_terms: list[float] = []
    for row in seed_rows:
        p = _safe_float(row.get("p_one_sided_fake_gt_real"))
        n_fake = int(max(0, _safe_float(row.get("n_fake_with_prefix"))))
        if not np.isfinite(p):
            continue
        p_clamped = float(min(max(p, 1e-12), 1.0 - 1e-12))
        z = float(scipy_stats.norm.isf(p_clamped))
        w = float(math.sqrt(max(1, n_fake)))
        z_terms.append(z)
        w_terms.append(w)
    if not z_terms:
        return float("nan")
    num = float(np.sum(np.asarray(w_terms, dtype=float) * np.asarray(z_terms, dtype=float)))
    den = float(np.sqrt(np.sum(np.square(np.asarray(w_terms, dtype=float)))))
    if den <= 0.0:
        return float("nan")
    z_meta = num / den
    return float(scipy_stats.norm.sf(z_meta))


def _weighted_mean(seed_rows: list[dict[str, Any]], value_key: str, weight_key: str) -> float:
    vals: list[float] = []
    wts: list[float] = []
    for row in seed_rows:
        v = _safe_float(row.get(value_key))
        w = _safe_float(row.get(weight_key))
        if np.isfinite(v) and np.isfinite(w) and w > 0:
            vals.append(float(v))
            wts.append(float(w))
    if not vals:
        return float("nan")
    return float(np.average(np.asarray(vals, dtype=float), weights=np.asarray(wts, dtype=float)))


def main() -> None:
    p = argparse.ArgumentParser(description="3P2-B multiseed strict-gate adjudication")
    p.add_argument("--model", default="llama-3.1-8b")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6")
    p.add_argument("--input-root", default="results/experiment3_phase2/exp3p2b_trivial_feature_control_multiseed")
    p.add_argument("--canonical-output-root", default="results/experiment3_phase2/exp3p2b_trivial_feature_control")
    p.add_argument("--min-abs-diff", type=float, default=0.005)
    p.add_argument("--min-cohens-d", type=float, default=0.2)
    p.add_argument("--alpha-one-sided", type=float, default=0.05)
    p.add_argument("--stable-blocked-min-flags", type=int, default=5)
    p.add_argument("--stable-clean-max-flags", type=int, default=1)
    p.add_argument("--stable-blocked-meta-p", type=float, default=0.01)
    p.add_argument("--stable-clean-meta-p", type=float, default=0.2)
    args = p.parse_args()

    input_root = Path(args.input_root)
    canonical_output_root = Path(args.canonical_output_root)
    seeds = _parse_seeds(args.seeds)

    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_rows.append(
            _seed_row(
                input_root=input_root,
                model=str(args.model),
                seed=int(seed),
                min_abs_diff=float(args.min_abs_diff),
                min_d=float(args.min_cohens_d),
                alpha_one_sided=float(args.alpha_one_sided),
            )
        )

    complete_rows = [r for r in seed_rows if str(r.get("status")) == "complete"]
    missing_seeds = [int(r["seed"]) for r in seed_rows if str(r.get("status")) != "complete"]

    flagged = int(sum(1 for r in complete_rows if bool(r.get("seed_gate_flag", False))))
    total_complete = int(len(complete_rows))
    meta_p_one = _weighted_stouffer_p(complete_rows)
    pooled_diff = _weighted_mean(complete_rows, "diff_fake_minus_real_with_prefix", "n_fake_with_prefix")
    pooled_d = _weighted_mean(complete_rows, "cohens_d_fake_minus_real", "n_fake_with_prefix")

    stable_blocked = bool(
        (total_complete >= len(seeds))
        and (flagged >= int(args.stable_blocked_min_flags))
        and np.isfinite(meta_p_one)
        and (meta_p_one < float(args.stable_blocked_meta_p))
        and np.isfinite(pooled_diff)
        and (pooled_diff >= float(args.min_abs_diff))
    )
    stable_clean = bool(
        (total_complete >= len(seeds))
        and (flagged <= int(args.stable_clean_max_flags))
        and np.isfinite(meta_p_one)
        and (meta_p_one > float(args.stable_clean_meta_p))
        and np.isfinite(pooled_diff)
        and (pooled_diff <= 0.0)
    )

    if stable_blocked:
        assessment = "stable_blocked"
    elif stable_clean:
        assessment = "stable_clean"
    else:
        assessment = "ambiguous"

    summary = {
        "model": str(args.model),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_root": str(input_root),
        "canonical_output_root": str(canonical_output_root),
        "decision_rule": {
            "per_seed_flag": {
                "diff_fake_minus_real_with_prefix_ge": float(args.min_abs_diff),
                "cohens_d_ge": float(args.min_cohens_d),
                "p_one_sided_lt": float(args.alpha_one_sided),
            },
            "meta_test": "weighted_stouffer_one_sided",
            "weight": "sqrt(n_fake_with_prefix)",
            "stable_blocked_rule": {
                "min_flagged_seeds": int(args.stable_blocked_min_flags),
                "meta_p_one_lt": float(args.stable_blocked_meta_p),
                "pooled_diff_ge": float(args.min_abs_diff),
            },
            "stable_clean_rule": {
                "max_flagged_seeds": int(args.stable_clean_max_flags),
                "meta_p_one_gt": float(args.stable_clean_meta_p),
                "pooled_diff_le": 0.0,
            },
        },
        "seeds_requested": seeds,
        "seeds_completed": [int(r["seed"]) for r in complete_rows],
        "missing_seeds": missing_seeds,
        "seed_rows": seed_rows,
        "pooled": {
            "n_completed": total_complete,
            "n_flagged": flagged,
            "meta_p_one": _safe_float(meta_p_one),
            "pooled_diff_fake_minus_real_with_prefix": _safe_float(pooled_diff),
            "pooled_cohens_d": _safe_float(pooled_d),
        },
        "assessment": str(assessment),
        "canonical_gate_recommendation": {
            "strict_gate_passes": bool(assessment == "stable_clean"),
            "strict_gate_blocked": bool(assessment == "stable_blocked"),
        },
    }

    run_summary_path = input_root / str(args.model) / "multiseed_gate_summary.json"
    canonical_summary_path = canonical_output_root / str(args.model) / "multiseed_gate_summary.json"
    _write_json(run_summary_path, summary)
    _write_json(canonical_summary_path, summary)

    print(f"[3P2-B-multiseed] wrote {run_summary_path}")
    print(f"[3P2-B-multiseed] wrote {canonical_summary_path}")
    print(f"[3P2-B-multiseed] assessment={assessment} completed={total_complete}/{len(seeds)} flagged={flagged}")


if __name__ == "__main__":
    main()
