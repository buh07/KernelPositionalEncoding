from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiment2.config import MODEL_BY_NAME
from experiment2.stats import bootstrap_bca_ci

PHASE_RE = re.compile(r"^phase2[abcd]$")


def _resolve_phase_name(phase_root: Path) -> str:
    for node in (phase_root, *phase_root.parents):
        if PHASE_RE.match(node.name):
            return node.name
    return "unknown_phase"


def _model_meta(model: str) -> dict[str, str]:
    spec = MODEL_BY_NAME.get(model)
    if spec is None:
        return {"norm": "unknown", "pe_scheme": "unknown"}
    norm = (spec.norm or "unknown").lower()
    norm_family = "RMSNorm" if norm.startswith("rms") else "LayerNorm" if norm.startswith("layer") else spec.norm or "unknown"
    return {
        "norm": norm_family,
        "pe_scheme": spec.pe_scheme,
    }


def _task_class(name: str) -> str:
    if name in {"local_copy_offset", "local_key_match"}:
        return "short"
    if name in {"delayed_copy", "long_range_retrieval"}:
        return "long"
    return "other"


def _h4_label(effect: float, ci_low: float) -> str:
    if not np.isfinite(effect) or not np.isfinite(ci_low):
        return "exploratory_unstable"
    if effect >= 0.03 and ci_low > 0:
        return "exploratory_signal"
    if effect > 0:
        return "exploratory_weak_positive"
    return "exploratory_null_or_negative"


def write_norm_family_decomposition(phase_root: Path) -> pd.DataFrame:
    task_path = phase_root / "aggregate_task_metrics.parquet"
    kernel_path = phase_root / "aggregate_kernel_metrics.parquet"
    task_df = pd.read_parquet(task_path) if task_path.exists() else pd.DataFrame()
    kernel_df = pd.read_parquet(kernel_path) if kernel_path.exists() else pd.DataFrame()

    rows = []
    if not task_df.empty:
        for model in sorted(task_df["model"].dropna().unique()):
            meta = _model_meta(model)
            sub = task_df[task_df["model"] == model]
            rows.append(
                {
                    "model": model,
                    "norm_family": meta["norm"],
                    "pe_scheme": meta["pe_scheme"],
                    "table": "task",
                    "mean_accuracy": float(sub["mean_accuracy"].mean()),
                    "mean_nll": float(sub["mean_nll"].mean()),
                    "rows": int(len(sub)),
                }
            )
    if not kernel_df.empty:
        for model in sorted(kernel_df["model"].dropna().unique()):
            meta = _model_meta(model)
            sub = kernel_df[kernel_df["model"] == model]
            for metric in sorted(sub["metric"].dropna().unique()):
                msub = sub[sub["metric"] == metric]
                rows.append(
                    {
                        "model": model,
                        "norm_family": meta["norm"],
                        "pe_scheme": meta["pe_scheme"],
                        "table": f"kernel_{metric}",
                        "mean_r2": float(msub["mean_r2"].mean()),
                        "rows": int(len(msub)),
                    }
                )

    out = pd.DataFrame(rows)
    out.to_parquet(phase_root / "norm_family_decomposition.parquet", engine="pyarrow", index=False)
    return out


def write_h4_interaction_exploratory(phase_root: Path) -> pd.DataFrame:
    phase_name = _resolve_phase_name(phase_root)
    task_path = phase_root / "aggregate_task_metrics.parquet"
    kernel_path = phase_root / "aggregate_kernel_metrics.parquet"
    task_df = pd.read_parquet(task_path) if task_path.exists() else pd.DataFrame()
    kernel_df = pd.read_parquet(kernel_path) if kernel_path.exists() else pd.DataFrame()
    rows: list[dict[str, object]] = []

    if not task_df.empty:
        work = task_df.copy()
        work["norm_family"] = work["model"].map(lambda m: _model_meta(str(m))["norm"])
        work["task_class"] = work["task"].map(_task_class)
        key_cols = ["model", "task", "seq_len", "seed", "split"]
        for col in ("dataset", "span"):
            if col in work.columns:
                key_cols.append(col)
        base = work[work["intervention"] == "none"][key_cols + ["mean_accuracy"]].rename(columns={"mean_accuracy": "none_accuracy"})
        eff = work.merge(base, on=key_cols, how="left", validate="m:1")
        eff["drop_acc"] = eff["none_accuracy"] - eff["mean_accuracy"]
        eff = eff[eff["task_class"].isin(["short", "long"])]
        for norm_family in sorted(eff["norm_family"].dropna().unique()):
            sub = eff[eff["norm_family"] == norm_family]
            for target, intervention in (("high", "ablate_high_strong"), ("low", "ablate_low_strong")):
                grp = sub[sub["intervention"] == intervention].groupby(["model", "seed", "task_class"], as_index=False)["drop_acc"].mean()
                piv = grp.pivot_table(index=["model", "seed"], columns="task_class", values="drop_acc").reset_index()
                if target == "high":
                    piv["contrast"] = piv.get("short", np.nan) - piv.get("long", np.nan)
                else:
                    piv["contrast"] = piv.get("long", np.nan) - piv.get("short", np.nan)
                vals = piv["contrast"].to_numpy(dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                effect = float(np.mean(vals)) if vals.size else float("nan")
                ci_low, ci_high = bootstrap_bca_ci(vals, b=20_000, seed=0) if vals.size else (float("nan"), float("nan"))
                rows.append(
                    {
                        "phase": phase_name,
                        "run_id": phase_root.name,
                        "domain": "task_class_contrast",
                        "norm_family": norm_family,
                        "target": target,
                        "metric": "drop_acc_class_contrast",
                        "effect_mean": effect,
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "n_effects": int(vals.size),
                        "label": _h4_label(effect, float(ci_low)),
                        "exploratory_only": True,
                    }
                )

    if not kernel_df.empty:
        work = kernel_df.copy()
        work = work[work["metric"] == "track_b_raw"].copy()
        work["norm_family"] = work["model"].map(lambda m: _model_meta(str(m))["norm"])
        key_cols = ["model", "task", "seq_len", "seed", "split", "metric", "layer", "head"]
        for col in ("dataset", "span"):
            if col in work.columns:
                key_cols.append(col)
        base = work[work["intervention"] == "none"][key_cols + ["mean_r2"]].rename(columns={"mean_r2": "baseline_r2"})
        merged = work.merge(base, on=key_cols, how="left", validate="m:1")
        merged["delta_abs"] = (merged["mean_r2"] - merged["baseline_r2"]).abs()
        for norm_family in sorted(merged["norm_family"].dropna().unique()):
            sub = merged[merged["norm_family"] == norm_family]
            for target, intervention in (("high", "ablate_high_strong"), ("low", "ablate_low_strong")):
                t = (
                    sub[sub["intervention"] == intervention]
                    .groupby(["model", "seed"], as_index=False)["delta_abs"]
                    .mean()
                    .rename(columns={"delta_abs": "targeted"})
                )
                r = (
                    sub[sub["intervention"] == "random_strong"]
                    .groupby(["model", "seed"], as_index=False)["delta_abs"]
                    .mean()
                    .rename(columns={"delta_abs": "random"})
                )
                pair = t.merge(r, on=["model", "seed"], how="inner")
                pair["diff"] = pair["targeted"] - pair["random"]
                vals = pair["diff"].to_numpy(dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                effect = float(np.mean(vals)) if vals.size else float("nan")
                ci_low, ci_high = bootstrap_bca_ci(vals, b=20_000, seed=0) if vals.size else (float("nan"), float("nan"))
                rows.append(
                    {
                        "phase": phase_name,
                        "run_id": phase_root.name,
                        "domain": "kernel_specificity",
                        "norm_family": norm_family,
                        "target": target,
                        "metric": "track_b_raw_targeted_minus_random",
                        "effect_mean": effect,
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "n_effects": int(vals.size),
                        "label": _h4_label(effect, float(ci_low)),
                        "exploratory_only": True,
                    }
                )

    out = pd.DataFrame(rows)
    out.to_parquet(phase_root / "h4_interaction_exploratory.parquet", engine="pyarrow", index=False)
    return out


def write_claim_guard(phase_root: Path) -> dict[str, object]:
    phase_name = _resolve_phase_name(phase_root)
    decision_path = phase_root / "decision_summary.json"
    training_variance_not_measured = True
    inference_scope = "task_sampling_variability_only"
    if decision_path.exists():
        try:
            payload = json.loads(decision_path.read_text(encoding="utf-8"))
            training_variance_not_measured = bool(payload.get("training_variance_not_measured", True))
            inference_scope = str(payload.get("inference_scope", inference_scope))
        except Exception:
            pass
    blocked = []
    if training_variance_not_measured:
        blocked = [
            "training_run_generalization",
            "retraining_robustness",
            "cross_retrain_causal_stability",
        ]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": phase_name,
        "run_id": phase_root.name,
        "training_variance_not_measured": training_variance_not_measured,
        "inference_scope": inference_scope,
        "blocked_claim_categories": blocked,
    }
    (phase_root / "claim_guard.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_protocol_revision_log(phase_root: Path) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "experiment2.execution",
        "version": "protocol_compliance_v3",
        "notes": [
            "Phase2D scoped head-group intervention enabled.",
            "Baseline floor gate + deterministic fallback enabled.",
            "Tier1 stratified bins and validity checks enabled.",
            "Hash-gated Phase2A->2B reuse enabled.",
            "Phase gate + inferential outputs enabled.",
            "Analysis-only advancement policy + promotion guard added.",
            "H4 exploratory interaction table added.",
            "Claim-scope guard written to claim_guard.json.",
        ],
    }
    (phase_root / "protocol_revision_log.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
