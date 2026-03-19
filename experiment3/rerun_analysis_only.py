#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment3 import theory1_si_circuits as t1
from experiment3 import theory3_crossterm_correlation as t3
from experiment3 import theory5_subword_ablation as t5
from experiment3 import theory7b_activation_patching as t7b
from experiment3 import theory10_feeder_specificity as t10


MODELS = ["llama-3.1-8b", "olmo-2-7b"]


def _artifact_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def rerun_t1(model: str) -> None:
    out = Path("results/experiment3/theory1_si_circuits") / model
    task_path = out / "task_results.parquet"
    r2_path = out / "head_r2_summary.parquet"
    if not (task_path.exists() and r2_path.exists()):
        print(f"[T1:{model}] skipped (missing parquet)")
        return

    results_df = pd.read_parquet(task_path)
    r2_summary = pd.read_parquet(r2_path)
    analysis = t1.analyze_results(results_df, r2_summary, model)
    _write_json(out / "analysis.json", analysis)

    report_path = out / "report.json"
    report = _load_json(report_path)
    report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["analysis"] = analysis
    report.setdefault("config", {})["analysis_only_repair"] = True
    report["artifact_timestamps"] = {
        "task_results_parquet": _artifact_timestamp(task_path),
        "head_r2_summary_parquet": _artifact_timestamp(r2_path),
    }
    _write_json(report_path, report)
    print(f"[T1:{model}] analysis refreshed")


def rerun_t3(model: str) -> None:
    out = Path("results/experiment3/theory3_crossterm") / model
    ct_path = out / "crossterm_per_pair_agg.parquet"
    if not ct_path.exists():
        print(f"[T3:{model}] skipped (missing crossterm parquet)")
        return

    crossterm_agg = pd.read_parquet(ct_path)
    total_energy_path = out / "total_pair_energy.parquet"
    total_energy_df = pd.read_parquet(total_energy_path) if total_energy_path.exists() else None
    pair_effects = t3.load_pair_effects(model)
    analysis = t3.run_correlation_analysis(crossterm_agg, pair_effects, total_energy_df, model)

    _write_json(out / "analysis.json", analysis)

    report_path = out / "report.json"
    report = _load_json(report_path)
    report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["correlation_analysis"] = analysis
    report.setdefault("config", {})["analysis_only_repair"] = True
    report["artifact_timestamps"] = {
        "crossterm_per_pair_agg_parquet": _artifact_timestamp(ct_path),
        "total_pair_energy_parquet": _artifact_timestamp(total_energy_path),
        "pair_effects_parquet": _artifact_timestamp(Path(t3.PAIR_EFFECTS_PATH)),
    }
    _write_json(report_path, report)
    print(f"[T3:{model}] analysis refreshed")


def _build_condition_losses(loss_df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    condition_losses: dict[str, dict[str, np.ndarray]] = {}
    for condition in sorted(loss_df["condition"].unique()):
        condition_losses[condition] = {
            "continuation": loss_df[
                (loss_df["condition"] == condition)
                & (loss_df["token_type"] == "continuation")
            ]["loss"].to_numpy(),
            "word_initial": loss_df[
                (loss_df["condition"] == condition)
                & (loss_df["token_type"] == "word_initial")
            ]["loss"].to_numpy(),
        }
    return condition_losses


def rerun_t5(model: str) -> None:
    out = Path("results/experiment3/theory5_subword_ablation") / model
    loss_path = out / "per_position_losses.parquet"
    if not loss_path.exists():
        print(f"[T5:{model}] skipped (missing loss parquet)")
        return

    loss_df = pd.read_parquet(loss_path)
    condition_losses = _build_condition_losses(loss_df)

    analysis_unmatched = t5.analyze_results(condition_losses)
    analysis_unmatched["matching"] = "unmatched"

    baseline_cont = condition_losses["none"]["continuation"]
    baseline_init = condition_losses["none"]["word_initial"]
    matched_cont_idx, matched_init_idx = t5.perplexity_matched_subsample(
        baseline_cont,
        baseline_init,
        [],
        [],
    )

    if len(matched_cont_idx) >= 50 and len(matched_init_idx) >= 50:
        n_matched = min(len(matched_cont_idx), len(matched_init_idx))
        matched_cont_idx = matched_cont_idx[:n_matched]
        matched_init_idx = matched_init_idx[:n_matched]

        matched_condition_losses: dict[str, dict[str, np.ndarray]] = {}
        for cond_name, cond_data in condition_losses.items():
            matched_condition_losses[cond_name] = {
                "continuation": cond_data["continuation"][matched_cont_idx],
                "word_initial": cond_data["word_initial"][matched_init_idx],
            }

        analysis_matched = t5.analyze_results(matched_condition_losses)
        analysis_matched["matching"] = "perplexity_matched"
        analysis_matched["n_matched_per_type"] = n_matched
        analysis_matched["matched_baseline_cont_mean"] = float(np.mean(baseline_cont[matched_cont_idx]))
        analysis_matched["matched_baseline_init_mean"] = float(np.mean(baseline_init[matched_init_idx]))
    else:
        analysis_matched = {
            "matching": "perplexity_matched",
            "skipped": True,
            "reason": "too_few_matched_samples",
        }

    report_path = out / "analysis.json"
    report = _load_json(report_path)
    report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["analysis_unmatched"] = analysis_unmatched
    report["analysis_matched"] = analysis_matched
    report.setdefault("config", {})["analysis_only_repair"] = True
    report["artifact_timestamps"] = {
        "per_position_losses_parquet": _artifact_timestamp(loss_path),
    }
    _write_json(report_path, report)
    print(f"[T5:{model}] analysis refreshed")


def rerun_t7b(model: str) -> None:
    out = Path("results/experiment3/theory7b_activation_patching") / model
    patch_path = out / "patching_results.parquet"
    if not patch_path.exists():
        print(f"[T7b:{model}] skipped (missing patching parquet)")
        return

    patch_df = pd.read_parquet(patch_path)
    r2_df = t7b.load_r2_data(model)
    prev_token_df = t7b.load_prev_token_scores(model)
    induction_df = t7b.load_induction_scores(model)
    top_induction = induction_df.sort_values("induction_score", ascending=False).head(t7b.TOP_INDUCTION_HEADS)
    induction_heads = [t7b.HeadID(int(r.layer), int(r.head)) for r in top_induction.itertuples()]

    num_pairs = int(patch_df["n_pairs"].max()) if "n_pairs" in patch_df.columns else t7b.NUM_SEQUENCE_PAIRS
    analysis = t7b.analyze_patching_results(
        patch_df=patch_df,
        r2_df=r2_df,
        prev_token_df=prev_token_df,
        induction_heads=induction_heads,
        model_name=model,
        num_pairs=num_pairs,
    )

    report_path = out / "report.json"
    report = _load_json(report_path)
    report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report.setdefault("config", {})["analysis_only_repair"] = True
    report.setdefault("config", {})["patching_parquet"] = str(patch_path)
    report["analysis"] = analysis
    report["artifact_timestamps"] = {
        "induction_scores_parquet": _artifact_timestamp(
            Path("results/experiment3/induction_r2_crossref") / model / "induction_scores.parquet"
        ),
        "r2_per_sequence_parquet": _artifact_timestamp(
            Path("results/experiment3/theory1_si_circuits") / model / "per_sequence_r2.parquet"
        ),
        "prev_token_scores_parquet": _artifact_timestamp(
            Path("results/experiment3/theory7_induction_feeders") / model / "prev_token_scores.parquet"
        ),
        "patching_results_parquet": _artifact_timestamp(patch_path),
    }
    _write_json(report_path, report)
    print(f"[T7b:{model}] analysis refreshed")


def rerun_t10(model: str) -> None:
    out = Path("results/experiment3/theory10_feeder_specificity") / model
    patch_path = out / "patching_results.parquet"
    if not patch_path.exists():
        print(f"[T10:{model}] skipped (missing patching parquet)")
        return

    patch_df = pd.read_parquet(patch_path)
    r2_df = t10.load_r2_data(model)
    prev_token_df = t10.load_prev_token_scores(model)
    target_groups = t10.infer_target_groups_from_patching_results(patch_df)
    num_pairs = int(patch_df["n_pairs"].max()) if "n_pairs" in patch_df.columns else t10.NUM_SEQUENCE_PAIRS

    analysis = t10.analyze_patching_results(
        patch_df=patch_df,
        r2_df=r2_df,
        prev_token_df=prev_token_df,
        target_groups=target_groups,
        model_name=model,
        num_pairs=num_pairs,
    )

    report_path = out / "report.json"
    report = _load_json(report_path)
    report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report.setdefault("config", {})["analysis_only_repair"] = True
    report.setdefault("config", {})["patching_parquet"] = str(patch_path)
    report["analysis"] = analysis
    report["artifact_timestamps"] = {
        "induction_scores_parquet": _artifact_timestamp(
            Path("results/experiment3/induction_r2_crossref") / model / "induction_scores.parquet"
        ),
        "r2_per_sequence_parquet": _artifact_timestamp(
            Path("results/experiment3/theory1_si_circuits") / model / "per_sequence_r2.parquet"
        ),
        "prev_token_scores_parquet": _artifact_timestamp(
            Path("results/experiment3/theory7_induction_feeders") / model / "prev_token_scores.parquet"
        ),
        "patching_results_parquet": _artifact_timestamp(patch_path),
    }
    _write_json(report_path, report)
    print(f"[T10:{model}] analysis refreshed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun Experiment 3 analysis-only repairs")
    parser.add_argument("--models", nargs="*", default=MODELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for model in args.models:
        rerun_t1(model)
        rerun_t3(model)
        rerun_t5(model)
        rerun_t7b(model)
        rerun_t10(model)


if __name__ == "__main__":
    main()
