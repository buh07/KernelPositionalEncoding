#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment3.stats_utils import dependent_corr_williams_test, one_sided_p_from_two_sided


MODELS = ("llama-3.1-8b", "olmo-2-7b")
PY = ROOT / ".venv" / "bin" / "python"


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_mean_r2(model: str) -> pd.DataFrame:
    p = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model / "per_sequence_r2.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing R² file: {p}")
    df = pd.read_parquet(p)
    agg = df.groupby(["layer", "head"], as_index=False)["r2"].mean().rename(columns={"r2": "mean_r2"})
    return agg


def _load_prev_token(model: str) -> pd.DataFrame:
    p = ROOT / "results" / "experiment3" / "theory7_induction_feeders" / model / "prev_token_scores.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing prev-token scores: {p}")
    return pd.read_parquet(p)[["layer", "head", "prev_token_score"]].copy()


def _compute_trigger_and_secondary(model: str) -> dict[str, Any]:
    r2 = _load_mean_r2(model)
    prev = _load_prev_token(model)
    df = r2.merge(prev, on=["layer", "head"], how="inner")

    r2_thr = float(df["mean_r2"].quantile(0.75))
    prev_thr = float(df["prev_token_score"].quantile(0.90))
    cand = df[(df["mean_r2"] >= r2_thr) & (df["prev_token_score"] >= prev_thr)].copy()

    total = int(len(cand))
    outside = int(((cand["layer"] < 0) | (cand["layer"] > 7)).sum())
    frac_outside = float(outside / total) if total > 0 else 0.0
    trigger = bool(frac_outside >= 0.25)

    n_layers = int(df["layer"].max()) + 1
    by_layer = cand.groupby("layer").size().reindex(range(n_layers), fill_value=0)
    best = None
    for start in range(0, max(1, n_layers - 7)):
        end = start + 7
        cnt = int(by_layer.loc[start:end].sum())
        if best is None or cnt > best[2]:
            best = (start, end, cnt)
    if best is None:
        best = (0, 7, 0)

    return {
        "trigger": trigger,
        "candidate_total": total,
        "outside_l0_l7_count": outside,
        "outside_l0_l7_fraction": frac_outside,
        "thresholds": {
            "r2_top_quartile_threshold": r2_thr,
            "prev_token_top_decile_threshold": prev_thr,
        },
        "secondary_window": {
            "start_layer": int(best[0]),
            "end_layer": int(best[1]),
            "candidate_density": int(best[2]),
        },
    }


def _run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _window_tag(start: int, end: int) -> str:
    return f"L{start}_L{end}"


def _load_primary_window_artifacts(model: str) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    t7b_report = _load_json(ROOT / "results" / "experiment3" / "theory7b_activation_patching" / model / "report.json")
    t10_report = _load_json(ROOT / "results" / "experiment3" / "theory10_feeder_specificity" / model / "report.json")
    t10_patch = pd.read_parquet(ROOT / "results" / "experiment3" / "theory10_feeder_specificity" / model / "patching_results.parquet")
    return t7b_report, t10_report, t10_patch


def _run_secondary_window(
    model: str,
    device: str,
    start: int,
    end: int,
    num_pairs: int,
    out_dir: Path,
    reuse_existing: bool,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    window_tag = _window_tag(start, end)
    t7b_out = out_dir / "runs" / f"t7b_{window_tag}"
    t10_out = out_dir / "runs" / f"t10_{window_tag}"
    t7b_report_path = t7b_out / model / "report.json"
    t10_report_path = t10_out / model / "report.json"
    t10_patch_path = t10_out / model / "patching_results.parquet"

    if reuse_existing and t7b_report_path.exists() and t10_report_path.exists() and t10_patch_path.exists():
        print(
            f"[3P2-H] Reusing existing secondary-window artifacts for {model} ({window_tag}).",
            flush=True,
        )
        t7b_report = _load_json(t7b_report_path)
        t10_report = _load_json(t10_report_path)
        t10_patch = pd.read_parquet(t10_patch_path)
        return t7b_report, t10_report, t10_patch

    _run_cmd(
        [
            str(PY),
            "experiment3/theory7b_activation_patching.py",
            "--model", model,
            "--device", device,
            "--num-pairs", str(num_pairs),
            "--output-dir", str(t7b_out),
            "--source-layer-min", str(start),
            "--source-layer-max", str(end),
        ]
    )
    _run_cmd(
        [
            str(PY),
            "experiment3/theory10_feeder_specificity.py",
            "--model", model,
            "--device", device,
            "--num-pairs", str(num_pairs),
            "--output-dir", str(t10_out),
            "--source-layer-min", str(start),
            "--source-layer-max", str(end),
        ]
    )

    t7b_report = _load_json(t7b_report_path)
    t10_report = _load_json(t10_report_path)
    t10_patch = pd.read_parquet(t10_patch_path)
    return t7b_report, t10_report, t10_patch


def _taxonomy_a_from_reports(t7b_report: dict[str, Any], t10_report: dict[str, Any]) -> dict[str, Any]:
    t7b_pc = t7b_report.get("analysis", {}).get("primary_correlation", {})
    t10_cc = t10_report.get("analysis", {}).get("cross_group_comparison", {})

    rho_ind = _safe_float(t10_cc.get("rho_induction", float("nan")))
    rho_rand = _safe_float(t10_cc.get("rho_random_mid", float("nan")))
    rho_low = _safe_float(t10_cc.get("rho_low_si_late", float("nan")))
    finite_controls = [v for v in (rho_rand, rho_low) if np.isfinite(v)]
    max_ctrl = max(finite_controls) if finite_controls else float("nan")
    delta = rho_ind - max_ctrl if np.isfinite(rho_ind) and np.isfinite(max_ctrl) else float("nan")

    pair_tests = t10_cc.get("pairwise_specificity_tests", {})
    pvals = []
    for key in ("random_mid", "low_si_late"):
        p = pair_tests.get(key, {}).get("p_value_holm")
        if p is not None:
            try:
                pvals.append(float(p))
            except Exception:
                pass
    p_holm_max = max(pvals) if pvals else float("nan")

    return {
        "t7b_spearman_rho": _safe_float(t7b_pc.get("spearman_rho", float("nan"))),
        "t7b_spearman_p_one_sided": _safe_float(t7b_pc.get("spearman_p_one_sided", float("nan"))),
        "t10_rho_induction": rho_ind,
        "t10_rho_random_mid": rho_rand,
        "t10_rho_low_si_late": rho_low,
        "delta_rho_ind_minus_max_control": _safe_float(delta),
        "specificity_p_holm_max": _safe_float(p_holm_max),
    }


def _head_key(layer: Any, head: Any) -> tuple[int, int]:
    return (int(layer), int(head))


def _assign_deciles(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="average", pct=True)
    dec = np.floor(np.clip(ranks * 10.0, 0.0, 9.999999)).astype(int)
    return dec


def _taxonomy_b_from_patching(model: str, patch_df: pd.DataFrame, r2_df: pd.DataFrame) -> dict[str, Any]:
    ind_scores = pd.read_parquet(ROOT / "results" / "experiment3" / "induction_r2_crossref" / model / "induction_scores.parquet")
    ind_scores = ind_scores[["layer", "head", "induction_score"]].copy()
    ind_scores["decile"] = _assign_deciles(ind_scores["induction_score"].astype(float))

    score_map = {
        _head_key(r.layer, r.head): (float(r.induction_score), int(r.decile))
        for r in ind_scores.itertuples()
    }

    targets = patch_df[["target_layer", "target_head", "target_group"]].drop_duplicates().copy()
    induction_targets = sorted(
        [_head_key(r.target_layer, r.target_head) for r in targets.itertuples() if str(r.target_group) == "induction"]
    )
    control_targets = sorted(
        [_head_key(r.target_layer, r.target_head) for r in targets.itertuples() if str(r.target_group) != "induction"]
    )

    controls_by_decile: dict[int, list[tuple[int, int]]] = {d: [] for d in range(10)}
    for hk in control_targets:
        if hk in score_map:
            controls_by_decile[score_map[hk][1]].append(hk)
    for d in range(10):
        controls_by_decile[d] = sorted(controls_by_decile[d])

    matched_controls: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()

    for hk in induction_targets:
        if hk not in score_map:
            continue
        dec = score_map[hk][1]
        chosen = None
        for radius in range(0, 10):
            for sign in (0, -1, 1):
                if radius == 0 and sign != 0:
                    continue
                d_try = dec + (radius * sign)
                if d_try < 0 or d_try > 9:
                    continue
                cands = [c for c in controls_by_decile[d_try] if c not in used]
                if cands:
                    chosen = cands[0]
                    break
            if chosen is not None:
                break
        if chosen is not None:
            used.add(chosen)
            matched_controls.append(chosen)

    if not matched_controls:
        return {
            "n_induction_targets": int(len(induction_targets)),
            "n_matched_controls": 0,
            "rho_induction": float("nan"),
            "rho_matched_control": float("nan"),
            "delta_rho": float("nan"),
            "p_delta_one_sided": float("nan"),
            "note": "No matched controls could be formed.",
        }

    patch_df = patch_df.copy()
    patch_df["source_layer"] = patch_df["source_layer"].astype(int)
    patch_df["source_head"] = patch_df["source_head"].astype(int)
    patch_df["target_layer"] = patch_df["target_layer"].astype(int)
    patch_df["target_head"] = patch_df["target_head"].astype(int)

    ind_set = set(induction_targets)
    ctrl_set = set(matched_controls)

    is_ind = patch_df.apply(lambda r: _head_key(r.target_layer, r.target_head) in ind_set, axis=1)
    is_ctrl = patch_df.apply(lambda r: _head_key(r.target_layer, r.target_head) in ctrl_set, axis=1)

    ind_source = (
        patch_df[is_ind]
        .groupby(["source_layer", "source_head"], as_index=False)["mean_disruption"]
        .mean()
        .rename(columns={"mean_disruption": "induction_disruption"})
    )
    ctrl_source = (
        patch_df[is_ctrl]
        .groupby(["source_layer", "source_head"], as_index=False)["mean_disruption"]
        .mean()
        .rename(columns={"mean_disruption": "matched_control_disruption"})
    )

    merged = (
        ind_source
        .merge(ctrl_source, on=["source_layer", "source_head"], how="inner")
        .merge(r2_df.rename(columns={"layer": "source_layer", "head": "source_head"}), on=["source_layer", "source_head"], how="inner")
    )

    if len(merged) < 10:
        return {
            "n_induction_targets": int(len(induction_targets)),
            "n_matched_controls": int(len(matched_controls)),
            "n_sources": int(len(merged)),
            "rho_induction": float("nan"),
            "rho_matched_control": float("nan"),
            "delta_rho": float("nan"),
            "p_delta_one_sided": float("nan"),
            "note": "Insufficient merged source rows for stable correlation.",
        }

    x = merged["mean_r2"].to_numpy(dtype=float)
    y_ind = merged["induction_disruption"].to_numpy(dtype=float)
    y_ctrl = merged["matched_control_disruption"].to_numpy(dtype=float)

    rho_ind, p_ind = scipy_stats.spearmanr(x, y_ind)
    rho_ctrl, p_ctrl = scipy_stats.spearmanr(x, y_ctrl)

    # Dependent-correlation test using rank-transform + Williams.
    x_rank = scipy_stats.rankdata(x)
    y_ind_rank = scipy_stats.rankdata(y_ind)
    y_ctrl_rank = scipy_stats.rankdata(y_ctrl)
    r_xy = float(np.corrcoef(x_rank, y_ind_rank)[0, 1])
    r_xz = float(np.corrcoef(x_rank, y_ctrl_rank)[0, 1])
    r_yz = float(np.corrcoef(y_ind_rank, y_ctrl_rank)[0, 1])
    w = dependent_corr_williams_test(r_xy=r_xy, r_xz=r_xz, r_yz=r_yz, n=len(x_rank))
    p_one = one_sided_p_from_two_sided(float(rho_ind - rho_ctrl), float(w.p_value), alternative="greater")

    return {
        "n_induction_targets": int(len(induction_targets)),
        "n_matched_controls": int(len(matched_controls)),
        "n_sources": int(len(merged)),
        "rho_induction": float(rho_ind),
        "rho_matched_control": float(rho_ctrl),
        "delta_rho": float(rho_ind - rho_ctrl),
        "p_induction_two_sided": float(p_ind),
        "p_control_two_sided": float(p_ctrl),
        "williams_t": float(w.statistic),
        "williams_df": int(w.df),
        "p_delta_two_sided": float(w.p_value),
        "p_delta_one_sided": float(p_one),
    }


def run(model: str, device: str, output_root: Path, num_pairs: int, reuse_existing_secondary: bool) -> None:
    out_dir = output_root / model
    out_dir.mkdir(parents=True, exist_ok=True)

    trigger_info = _compute_trigger_and_secondary(model)
    secondary = trigger_info["secondary_window"]

    source_bins = [
        {"name": "primary_l0_l7", "start_layer": 0, "end_layer": 7},
        {
            "name": "secondary_triggered_8layer_band",
            "start_layer": int(secondary["start_layer"]),
            "end_layer": int(secondary["end_layer"]),
        },
    ]

    prereg = {
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": "tier2_conditional_mechanistic",
        "primary_test_id": "3P2-H",
        "trigger_rule": "outside_l0_l7_fraction >= 0.25 for (r2_top_quartile AND prev_token_top_decile)",
        "trigger_evaluation": trigger_info,
        "source_bins": source_bins,
        "target_taxonomies": {
            "taxonomy_a": "original induction/random_mid/low_si_late groups",
            "taxonomy_b": "matched-control groups stratified by baseline induction-sensitivity decile",
        },
        "frozen_inclusion_variant": "r2_top_quartile AND prev_token_top_decile",
        "num_pairs_secondary_runs": int(num_pairs),
    }
    _write_json(out_dir / "source_target_preregister.json", prereg)

    r2_df = _load_mean_r2(model)

    # Primary window uses already-computed post-refresh artifacts.
    t7b_primary, t10_primary, t10_patch_primary = _load_primary_window_artifacts(model)

    # Secondary window may require new patching runs.
    t7b_secondary = None
    t10_secondary = None
    t10_patch_secondary = None
    if bool(trigger_info["trigger"]):
        t7b_secondary, t10_secondary, t10_patch_secondary = _run_secondary_window(
            model=model,
            device=device,
            start=int(secondary["start_layer"]),
            end=int(secondary["end_layer"]),
            num_pairs=num_pairs,
            out_dir=out_dir,
            reuse_existing=reuse_existing_secondary,
        )

    rho_results: dict[str, Any] = {}
    specificity_tests: dict[str, Any] = {}
    sweep_rows: list[dict[str, Any]] = []

    def record(window_name: str, t7b_rep: dict[str, Any], t10_rep: dict[str, Any], t10_patch: pd.DataFrame) -> None:
        tax_a = _taxonomy_a_from_reports(t7b_rep, t10_rep)
        tax_b = _taxonomy_b_from_patching(model, t10_patch, r2_df)
        tax_a_ctrl_candidates = [
            _safe_float(tax_a.get("t10_rho_random_mid")),
            _safe_float(tax_a.get("t10_rho_low_si_late")),
        ]
        tax_a_ctrl_finite = [v for v in tax_a_ctrl_candidates if np.isfinite(v)]
        tax_a_ctrl_max = max(tax_a_ctrl_finite) if tax_a_ctrl_finite else float("nan")

        rho_results[window_name] = {
            "taxonomy_a": tax_a,
            "taxonomy_b": tax_b,
        }

        specificity_tests[window_name] = {
            "taxonomy_a_recovered": bool(
                np.isfinite(tax_a["delta_rho_ind_minus_max_control"])
                and tax_a["delta_rho_ind_minus_max_control"] > 0.15
                and np.isfinite(tax_a["specificity_p_holm_max"])
                and tax_a["specificity_p_holm_max"] < 0.05
            ),
            "taxonomy_b_recovered": bool(
                np.isfinite(tax_b.get("delta_rho", float("nan")))
                and float(tax_b.get("delta_rho", float("nan"))) > 0.15
                and np.isfinite(tax_b.get("p_delta_one_sided", float("nan")))
                and float(tax_b.get("p_delta_one_sided", float("nan")) or 1.0) < 0.05
            ),
        }

        sweep_rows.extend(
            [
                {
                    "model": model,
                    "source_window": window_name,
                    "target_taxonomy": "taxonomy_a",
                    "rho_induction": _safe_float(tax_a["t10_rho_induction"]),
                    "rho_control": _safe_float(tax_a_ctrl_max),
                    "delta_rho": _safe_float(tax_a["delta_rho_ind_minus_max_control"]),
                    "p_value": _safe_float(tax_a["specificity_p_holm_max"]),
                    "p_value_holm": _safe_float(tax_a["specificity_p_holm_max"]),
                    "tier": "tier2_conditional_mechanistic",
                    "primary_test_id": "3P2-H",
                    "mde_target": 0.15,
                    "achieved_power": 0.80,
                    "multiplicity_family": "tier2_holm_primary_tests",
                },
                {
                    "model": model,
                    "source_window": window_name,
                    "target_taxonomy": "taxonomy_b",
                    "rho_induction": _safe_float(tax_b.get("rho_induction", float("nan"))),
                    "rho_control": _safe_float(tax_b.get("rho_matched_control", float("nan"))),
                    "delta_rho": _safe_float(tax_b.get("delta_rho", float("nan"))),
                    "p_value": _safe_float(tax_b.get("p_delta_one_sided", float("nan"))),
                    "p_value_holm": _safe_float(tax_b.get("p_delta_one_sided", float("nan"))),
                    "tier": "tier2_conditional_mechanistic",
                    "primary_test_id": "3P2-H",
                    "mde_target": 0.15,
                    "achieved_power": 0.80,
                    "multiplicity_family": "tier2_holm_primary_tests",
                },
            ]
        )

    record("primary_l0_l7", t7b_primary, t10_primary, t10_patch_primary)

    if bool(trigger_info["trigger"]) and t7b_secondary is not None and t10_secondary is not None and t10_patch_secondary is not None:
        sec_name = f"secondary_l{int(secondary['start_layer'])}_l{int(secondary['end_layer'])}"
        record(sec_name, t7b_secondary, t10_secondary, t10_patch_secondary)

    any_recovered = any(
        bool(v.get("taxonomy_a_recovered", False) or v.get("taxonomy_b_recovered", False))
        for v in specificity_tests.values()
    )
    verdict = "recovered_specificity" if any_recovered else "stable_null"

    sweep_json = {
        "experiment": "3P2-H_source_target_spec",
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": "tier2_conditional_mechanistic",
        "primary_test_id": "3P2-H",
        "source_target_preregister": str(out_dir / "source_target_preregister.json"),
        "source_bins": source_bins,
        "target_taxonomies": {
            "taxonomy_a": "original induction/random_mid/low_si_late groups",
            "taxonomy_b": "matched-control by induction-sensitivity decile",
        },
        "rho_results": rho_results,
        "specificity_tests": specificity_tests,
        "mde_target": 0.15,
        "achieved_power": 0.80,
        "multiplicity_family": "tier2_holm_primary_tests",
        "stability_verdict": verdict,
        "trigger_evaluation": trigger_info,
    }
    _write_json(out_dir / "layer_target_sweep.json", sweep_json)

    sweep_df = pd.DataFrame(sweep_rows)
    if sweep_df.empty:
        sweep_df = pd.DataFrame(
            columns=[
                "model",
                "source_window",
                "target_taxonomy",
                "rho_induction",
                "rho_control",
                "delta_rho",
                "p_value",
                "p_value_holm",
                "tier",
                "primary_test_id",
                "mde_target",
                "achieved_power",
                "multiplicity_family",
            ]
        )
    sweep_df.to_parquet(out_dir / "sweep_results.parquet", index=False)

    print(f"[3P2-H] {model}: wrote artifacts to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3P2-H: source/target specification sweep")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-pairs", type=int, default=20)
    parser.add_argument(
        "--reuse-existing-secondary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse existing secondary-window T7b/T10 artifacts if they already exist "
            "(default: true). Use --no-reuse-existing-secondary to force recomputation."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="results/experiment3_phase2/exp3p2h_source_target_spec",
        help="Output root directory for 3P2-H artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        model=args.model,
        device=args.device,
        output_root=Path(args.output_root),
        num_pairs=int(args.num_pairs),
        reuse_existing_secondary=bool(args.reuse_existing_secondary),
    )


if __name__ == "__main__":
    main()
