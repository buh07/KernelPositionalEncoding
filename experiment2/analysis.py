from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment2.stats import (
    bootstrap_bca_ci,
    bootstrap_bca_ci_sensitivity,
    holm_bonferroni,
    mde_two_sided_normal,
    paired_sign_flip_pvalue,
    precision_label,
)

H3_RATIO_EPS = 0.01
H3_RATIO_STABLE_MIN_RANDOM = 0.01
PHASE_RE = re.compile(r"^phase2[abcd]$")
SHORT_REGIME_MAX_OFFSET = 16
HEADROOM_IMBALANCE_RATIO_MIN = 0.50
HEADROOM_IMBALANCE_RATIO_MAX = 2.00
H12_ENDPOINT_POLICIES = {"raw_primary", "co_primary_raw_headroom"}


def task_class_for_name(name: str) -> str:
    if name in {"local_copy_offset", "local_key_match"}:
        return "short"
    if name in {"delayed_copy", "long_range_retrieval"}:
        return "long"
    if name in {"copy_offset_bridge", "retrieval_bridge"}:
        return "bridge"
    return "other"


@dataclass(frozen=True)
class PhaseAnalysis:
    gate_payload: dict[str, Any]
    specificity_df: pd.DataFrame
    decision_payload: dict[str, Any]


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _resolve_phase_name(phase_root: Path) -> str:
    for node in (phase_root, *phase_root.parents):
        if PHASE_RE.match(node.name):
            return node.name
    raise ValueError(f"Unable to infer phase name from path: {phase_root}")


def _normalize_h12_endpoint_policy(value: Any) -> str:
    policy = str(value or "raw_primary").strip()
    return policy if policy in H12_ENDPOINT_POLICIES else "raw_primary"


def _resolve_h12_endpoint_policy(phase_root: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for cfg_path in phase_root.glob("**/run_config.json"):
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        policy = _normalize_h12_endpoint_policy(payload.get("h12_endpoint_policy", "raw_primary"))
        counts[policy] = counts.get(policy, 0) + 1
    if counts:
        selected = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    else:
        selected = "raw_primary"
    return {
        "h12_endpoint_policy": selected,
        "h12_endpoint_policy_observed_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "h12_endpoint_policy_mixed": bool(len(counts) > 1),
    }


def _h12_endpoint_adjudication(
    *,
    endpoint_policy: str,
    raw_h1_label: str,
    raw_h2_label: str,
    headroom_diag_payload: dict[str, Any],
) -> dict[str, Any]:
    endpoint_policy = _normalize_h12_endpoint_policy(endpoint_policy)
    headroom_block = dict(headroom_diag_payload.get("headroom_normalized", {}))
    h1_headroom_label = str(headroom_block.get("h1", {}).get("label", "descriptive"))
    h2_headroom_label = str(headroom_block.get("h2", {}).get("label", "descriptive"))
    h1_raw_pass = bool(raw_h1_label == "pass")
    h2_raw_pass = bool(raw_h2_label == "pass")
    h1_headroom_pass = bool(h1_headroom_label == "pass")
    h2_headroom_pass = bool(h2_headroom_label == "pass")
    co_primary_h1_pass = bool(h1_raw_pass and h1_headroom_pass)
    co_primary_h2_pass = bool(h2_raw_pass and h2_headroom_pass)
    if endpoint_policy == "co_primary_raw_headroom":
        policy_h1_pass = co_primary_h1_pass
        policy_h2_pass = co_primary_h2_pass
    else:
        policy_h1_pass = h1_raw_pass
        policy_h2_pass = h2_raw_pass
    return {
        "policy": endpoint_policy,
        "raw_labels": {"h1": str(raw_h1_label), "h2": str(raw_h2_label)},
        "headroom_labels": {"h1": h1_headroom_label, "h2": h2_headroom_label},
        "raw_pass": {"h1": h1_raw_pass, "h2": h2_raw_pass},
        "headroom_pass": {"h1": h1_headroom_pass, "h2": h2_headroom_pass},
        "co_primary_pass": {"h1": co_primary_h1_pass, "h2": co_primary_h2_pass},
        "policy_pass": {"h1": policy_h1_pass, "h2": policy_h2_pass},
        "note": (
            "Endpoint policy is additive metadata for adjudication framing. "
            "Legacy confirmatory criteria remain unchanged unless policy-specific gates are explicitly enabled."
        ),
    }


def _conditional_key_cols(df: pd.DataFrame, *, required: list[str], optional: list[str]) -> list[str]:
    cols = list(required)
    for name in optional:
        if name in df.columns:
            cols.append(name)
    return cols


def _task_effects(task_df: pd.DataFrame) -> pd.DataFrame:
    df = task_df.copy()
    if df.empty:
        return df
    df["task_class"] = df["task"].map(task_class_for_name)
    key_cols = _conditional_key_cols(
        df,
        required=["model", "task", "seq_len", "seed", "split"],
        optional=["dataset", "span"],
    )
    base = df[df["intervention"] == "none"][key_cols + ["mean_accuracy", "mean_nll", "floor_limited"]].rename(
        columns={
            "mean_accuracy": "none_accuracy",
            "mean_nll": "none_nll",
            "floor_limited": "none_floor_limited",
        }
    )
    if not base.empty and base.duplicated(subset=key_cols).any():
        dup = int(base.duplicated(subset=key_cols).sum())
        raise RuntimeError(f"Baseline task rows are not unique for keys={key_cols}; duplicates={dup}.")
    eff = df.merge(base, on=key_cols, how="left", validate="m:1")
    if len(eff) != len(df):
        raise RuntimeError(f"Task effects merge expanded rows unexpectedly: in={len(df)} out={len(eff)}")
    eff["none_floor_limited"] = eff["none_floor_limited"].fillna(False).astype(bool)
    eff["drop_acc"] = eff["none_accuracy"] - eff["mean_accuracy"]
    eff["rel_nll_increase"] = (eff["mean_nll"] - eff["none_nll"]) / (eff["none_nll"].clip(lower=1e-9))
    eff["meaningful_effect"] = (eff["drop_acc"] >= 0.05) | (eff["rel_nll_increase"] >= 0.10)
    if "chance_accuracy" in eff.columns:
        chance = pd.to_numeric(eff["chance_accuracy"], errors="coerce")
    else:
        chance = pd.Series(np.nan, index=eff.index, dtype=np.float64)
    if "candidate_count" in eff.columns:
        candidate_count = pd.to_numeric(eff["candidate_count"], errors="coerce")
        chance_from_count = pd.Series(np.nan, index=eff.index, dtype=np.float64)
        valid = candidate_count > 0
        chance_from_count.loc[valid] = 1.0 / candidate_count.loc[valid]
    else:
        chance_from_count = pd.Series(np.nan, index=eff.index, dtype=np.float64)
    chance_effective = chance.fillna(chance_from_count).fillna(0.0).clip(lower=0.0, upper=1.0)
    eff["chance_accuracy_effective"] = chance_effective
    eff["baseline_headroom"] = eff["none_accuracy"] - eff["chance_accuracy_effective"]
    eff["drop_acc_over_baseline"] = eff["drop_acc"] / np.maximum(eff["none_accuracy"], 1e-6)
    eff["drop_acc_over_headroom"] = eff["drop_acc"] / np.maximum(eff["baseline_headroom"], 1e-6)
    return eff


def _class_contrast(eff: pd.DataFrame, intervention: str, direction: str, *, value_col: str = "drop_acc") -> pd.DataFrame:
    if value_col not in eff.columns:
        raise RuntimeError(f"Missing contrast value column: {value_col}")
    sub = eff[(eff["intervention"] == intervention) & (eff["task_class"].isin(["short", "long"]))].copy()
    if sub.empty:
        return pd.DataFrame(columns=["model", "seed", "contrast"])
    grp = sub.groupby(["model", "seed", "task_class"], as_index=False)[value_col].mean()
    piv = grp.pivot_table(index=["model", "seed"], columns="task_class", values=value_col).reset_index()
    if direction == "h1":
        piv["contrast"] = piv.get("short", 0.0) - piv.get("long", 0.0)
    elif direction == "h2":
        piv["contrast"] = piv.get("long", 0.0) - piv.get("short", 0.0)
    else:
        raise ValueError(direction)
    return piv[["model", "seed", "contrast"]]


def _effect_direction(value: float, *, eps: float = 1e-12) -> str:
    if not np.isfinite(value):
        return "nan"
    if value > eps:
        return "positive"
    if value < -eps:
        return "negative"
    return "zero"


def _contrast_inference_block(h1_vals: np.ndarray, h2_vals: np.ndarray, *, threshold: float = 0.05) -> dict[str, Any]:
    h1 = np.asarray(h1_vals, dtype=np.float64)
    h2 = np.asarray(h2_vals, dtype=np.float64)
    h1 = h1[np.isfinite(h1)]
    h2 = h2[np.isfinite(h2)]
    h1_effect = float(np.mean(h1)) if h1.size else float("nan")
    h2_effect = float(np.mean(h2)) if h2.size else float("nan")
    h1_ci = bootstrap_bca_ci(h1) if h1.size else (float("nan"), float("nan"))
    h2_ci = bootstrap_bca_ci(h2) if h2.size else (float("nan"), float("nan"))
    p_raw = {
        "H1": paired_sign_flip_pvalue(h1) if h1.size else float("nan"),
        "H2": paired_sign_flip_pvalue(h2) if h2.size else float("nan"),
    }
    p_adj = holm_bonferroni(p_raw)
    h1_label = precision_label(effect=h1_effect, ci_low=h1_ci[0], threshold=threshold)
    h2_label = precision_label(effect=h2_effect, ci_low=h2_ci[0], threshold=threshold)
    return {
        "h1": {
            "effect": h1_effect,
            "ci": [h1_ci[0], h1_ci[1]],
            "p": p_raw["H1"],
            "p_holm": p_adj["H1"],
            "label": h1_label,
            "n": int(h1.size),
            "direction": _effect_direction(h1_effect),
        },
        "h2": {
            "effect": h2_effect,
            "ci": [h2_ci[0], h2_ci[1]],
            "p": p_raw["H2"],
            "p_holm": p_adj["H2"],
            "label": h2_label,
            "n": int(h2.size),
            "direction": _effect_direction(h2_effect),
        },
    }


def _task_class_headroom_summary(task_eff: pd.DataFrame) -> dict[str, Any]:
    none = task_eff[(task_eff["intervention"] == "none") & (task_eff["task_class"].isin(["short", "long"]))].copy()
    if none.empty:
        return {
            "short": {},
            "long": {},
            "long_short_headroom_ratio": float("nan"),
            "imbalance_threshold_min": HEADROOM_IMBALANCE_RATIO_MIN,
            "imbalance_threshold_max": HEADROOM_IMBALANCE_RATIO_MAX,
            "severe_imbalance": False,
            "note": "No eligible short/long none-intervention rows available.",
        }

    grouped = none.groupby("task_class", as_index=False).agg(
        n_rows=("none_accuracy", "size"),
        mean_none_accuracy=("none_accuracy", "mean"),
        mean_chance_accuracy=("chance_accuracy_effective", "mean"),
        mean_baseline_headroom=("baseline_headroom", "mean"),
    )
    by_class = {str(r.task_class): r for r in grouped.itertuples(index=False)}

    def _cls_payload(name: str) -> dict[str, Any]:
        row = by_class.get(name)
        if row is None:
            return {"n_rows": 0, "mean_none_accuracy": float("nan"), "mean_chance_accuracy": float("nan"), "mean_baseline_headroom": float("nan")}
        return {
            "n_rows": int(row.n_rows),
            "mean_none_accuracy": float(row.mean_none_accuracy),
            "mean_chance_accuracy": float(row.mean_chance_accuracy),
            "mean_baseline_headroom": float(row.mean_baseline_headroom),
        }

    short = _cls_payload("short")
    long = _cls_payload("long")
    short_headroom = float(short["mean_baseline_headroom"])
    long_headroom = float(long["mean_baseline_headroom"])
    ratio = (
        float(long_headroom / short_headroom)
        if np.isfinite(short_headroom) and np.isfinite(long_headroom) and abs(short_headroom) > 1e-9
        else float("nan")
    )
    severe = bool(
        np.isfinite(ratio)
        and ((ratio < HEADROOM_IMBALANCE_RATIO_MIN) or (ratio > HEADROOM_IMBALANCE_RATIO_MAX))
    )
    return {
        "short": short,
        "long": long,
        "long_short_headroom_ratio": ratio,
        "imbalance_threshold_min": HEADROOM_IMBALANCE_RATIO_MIN,
        "imbalance_threshold_max": HEADROOM_IMBALANCE_RATIO_MAX,
        "severe_imbalance": severe,
    }


def _long_offset_overlap_diagnostic() -> dict[str, Any]:
    lock_path = Path(__file__).resolve().parent / "long_offset_lock.json"
    if not lock_path.exists():
        return {
            "available": False,
            "lock_path": str(lock_path),
            "short_regime_max_offset": SHORT_REGIME_MAX_OFFSET,
            "overlaps_short_regime": None,
            "note": "Long-offset lock artifact not found; span-overlap caveat cannot be auto-evaluated.",
        }
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "available": False,
            "lock_path": str(lock_path),
            "short_regime_max_offset": SHORT_REGIME_MAX_OFFSET,
            "overlaps_short_regime": None,
            "note": f"Failed to parse long-offset lock artifact: {exc}",
        }
    selected: list[int] = []
    for value in payload.get("selected_long_offsets", []):
        try:
            selected.append(int(value))
        except Exception:
            continue
    overlap = sorted([int(v) for v in selected if int(v) <= SHORT_REGIME_MAX_OFFSET])
    overlaps = bool(len(overlap) > 0)
    return {
        "available": True,
        "lock_path": str(lock_path),
        "short_regime_max_offset": SHORT_REGIME_MAX_OFFSET,
        "selected_long_offsets": selected,
        "overlap_offsets": overlap,
        "overlaps_short_regime": overlaps,
        "note": (
            "Long offsets overlap short regime; short/long contrasts combine task-family and span effects."
            if overlaps
            else "Long offsets do not overlap short regime."
        ),
    }


def _kernel_deltas(kernel_df: pd.DataFrame) -> pd.DataFrame:
    if kernel_df.empty:
        return kernel_df
    df = kernel_df.copy()
    df["task_class"] = df["task"].map(task_class_for_name)
    key_cols = _conditional_key_cols(
        df,
        required=["model", "task", "seq_len", "seed", "split", "metric", "layer", "head"],
        optional=["dataset", "span"],
    )
    base = df[df["intervention"] == "none"][key_cols + ["mean_r2"]].rename(columns={"mean_r2": "baseline_r2"})
    if not base.empty and base.duplicated(subset=key_cols).any():
        dup = int(base.duplicated(subset=key_cols).sum())
        raise RuntimeError(f"Baseline kernel rows are not unique for keys={key_cols}; duplicates={dup}.")
    out = df.merge(base, on=key_cols, how="left", validate="m:1")
    out["delta_r2_abs"] = (out["mean_r2"] - out["baseline_r2"]).abs()
    return out


def _eligible_task_rows(task_eff: pd.DataFrame) -> pd.DataFrame:
    if task_eff.empty:
        return task_eff
    if "none_floor_limited" not in task_eff.columns:
        return task_eff
    return task_eff[~task_eff["none_floor_limited"]].copy()


def _filter_kernel_to_eligible(kernel_delta: pd.DataFrame, task_eff_eligible: pd.DataFrame) -> pd.DataFrame:
    if kernel_delta.empty or task_eff_eligible.empty:
        return kernel_delta.iloc[0:0].copy()
    base_required = ["model", "task", "seq_len", "seed", "split"]
    missing_required = [c for c in base_required if c not in kernel_delta.columns or c not in task_eff_eligible.columns]
    if missing_required:
        raise RuntimeError(f"Missing required eligibility merge columns: {missing_required}")
    key_cols = list(base_required)
    for opt in ("dataset", "span"):
        if opt in task_eff_eligible.columns and opt in kernel_delta.columns:
            key_cols.append(opt)
    keys = task_eff_eligible[key_cols].drop_duplicates()
    return kernel_delta.merge(keys, on=key_cols, how="inner")


def _h3_specificity(
    kernel_delta: pd.DataFrame,
    *,
    allowed_task_classes: set[str] | None = None,
    allowed_metrics: set[str] | None = None,
    random_draw_mode: str = "seed_mean",
    compute_ci: bool = True,
) -> pd.DataFrame:
    if random_draw_mode not in {"seed_mean", "per_draw"}:
        raise ValueError(f"Unsupported random_draw_mode: {random_draw_mode}")
    if kernel_delta.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "task_class",
                "target",
                "severity",
                "metric",
                "delta_r2_targeted",
                "delta_r2_random",
                "paired_seed_count",
                "paired_pair_count",
                "random_draw_mode",
                "insufficient_pairing",
                "random_draws_per_seed_mean",
                "random_draws_per_seed_sd",
                "S_ratio",
                "S_diff",
                "ratio_stable",
                "severity_pass",
                "severity_pass_ci",
                "diff_ci_low",
                "diff_ci_high",
                "diff_p",
                "diff_label",
                "h3_group_pass",
                "h3_group_pass_ci",
            ]
        )
    work = kernel_delta.copy()
    if "metric" not in work.columns:
        work = work.assign(metric="unknown")
    if allowed_task_classes is not None:
        work = work[work["task_class"].isin(sorted(allowed_task_classes))].copy()
    if allowed_metrics is not None and "metric" in work.columns:
        work = work[work["metric"].isin(sorted(allowed_metrics))].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "task_class",
                "target",
                "severity",
                "metric",
                "delta_r2_targeted",
                "delta_r2_random",
                "paired_seed_count",
                "paired_pair_count",
                "random_draw_mode",
                "insufficient_pairing",
                "random_draws_per_seed_mean",
                "random_draws_per_seed_sd",
                "S_ratio",
                "S_diff",
                "ratio_stable",
                "severity_pass",
                "severity_pass_ci",
                "diff_ci_low",
                "diff_ci_high",
                "diff_p",
                "diff_label",
                "h3_group_pass",
                "h3_group_pass_ci",
            ]
        )

    rows = []
    for target, ablate_name in (("high", "ablate_high"), ("low", "ablate_low")):
        for sev in ("medium", "strong"):
            target_name = f"{ablate_name}_{sev}"
            random_name = f"random_{sev}"
            t = work[work["intervention"] == target_name]
            r = work[work["intervention"] == random_name]
            metric_values = sorted(set(t["metric"].dropna().unique().tolist()) | set(r["metric"].dropna().unique().tolist()))
            keys = sorted(set(zip(t["model"], t["task_class"])) | set(zip(r["model"], r["task_class"])))
            for metric in metric_values:
                t_metric = t[t["metric"] == metric]
                r_metric = r[r["metric"] == metric]
                metric_keys = sorted(set(zip(t_metric["model"], t_metric["task_class"])) | set(zip(r_metric["model"], r_metric["task_class"])))
                for model, task_class in metric_keys:
                    t_base = t_metric[(t_metric["model"] == model) & (t_metric["task_class"] == task_class)]
                    r_base = r_metric[(r_metric["model"] == model) & (r_metric["task_class"] == task_class)]
                    t_seed = (
                        t_base.groupby("seed", as_index=False)["delta_r2_abs"]
                        .mean()
                        .rename(columns={"delta_r2_abs": "targeted"})
                    )
                    if random_draw_mode == "per_draw" and ("random_draw" in r_base.columns):
                        r_seed = (
                            r_base.groupby(["seed", "random_draw"], as_index=False)["delta_r2_abs"]
                            .mean()
                            .rename(columns={"delta_r2_abs": "random"})
                        )
                        merged = t_seed.merge(r_seed, on="seed", how="inner")
                        paired_seed_count = int(merged["seed"].nunique()) if not merged.empty else 0
                        paired_pair_count = int(len(merged))
                        draw_counts = (
                            merged.groupby("seed")["random_draw"].nunique().to_numpy(dtype=np.float64)
                            if not merged.empty and "random_draw" in merged.columns
                            else np.asarray([], dtype=np.float64)
                        )
                    else:
                        r_seed = (
                            r_base.groupby("seed", as_index=False)["delta_r2_abs"]
                            .mean()
                            .rename(columns={"delta_r2_abs": "random"})
                        )
                        merged = t_seed.merge(r_seed, on="seed", how="inner")
                        paired_seed_count = int(len(merged))
                        paired_pair_count = int(len(merged))
                        draw_counts = (
                            r_base.groupby("seed")["random_draw"].nunique().to_numpy(dtype=np.float64)
                            if ("random_draw" in r_base.columns and not r_base.empty)
                            else np.asarray([], dtype=np.float64)
                        )
                    insufficient_pairing = paired_seed_count < 5
                    t_val = float(merged["targeted"].mean()) if paired_seed_count else float("nan")
                    r_val = float(merged["random"].mean()) if paired_seed_count else float("nan")
                    diff_vals = (merged["targeted"] - merged["random"]).to_numpy(dtype=np.float64) if paired_pair_count else np.asarray([], dtype=np.float64)
                    diff_vals = diff_vals[np.isfinite(diff_vals)]
                    ratio = t_val / (r_val + H3_RATIO_EPS) if np.isfinite(t_val) and np.isfinite(r_val) else float("nan")
                    diff = t_val - r_val if np.isfinite(t_val) and np.isfinite(r_val) else float("nan")
                    stable = bool(np.isfinite(r_val) and (r_val >= H3_RATIO_STABLE_MIN_RANDOM))
                    draw_mean = float(np.nanmean(draw_counts)) if draw_counts.size else float("nan")
                    draw_sd = float(np.nanstd(draw_counts, ddof=1)) if draw_counts.size >= 2 else (0.0 if draw_counts.size == 1 else float("nan"))
                    if compute_ci:
                        diff_ci = bootstrap_bca_ci(diff_vals) if diff_vals.size else (float("nan"), float("nan"))
                        diff_p = paired_sign_flip_pvalue(diff_vals) if diff_vals.size else float("nan")
                        diff_label = precision_label(effect=diff, ci_low=diff_ci[0], threshold=0.0)
                    else:
                        diff_ci = (float("nan"), float("nan"))
                        diff_p = float("nan")
                        diff_label = "not_computed"
                    sev_pass_point = bool((not insufficient_pairing) and np.isfinite(diff) and (diff > 0) and (ratio >= 1.5 if stable else True))
                    sev_pass_ci = bool(sev_pass_point and (diff_label == "pass")) if compute_ci else float("nan")
                    rows.append(
                        {
                            "model": model,
                            "task_class": task_class,
                            "target": target,
                            "severity": sev,
                            "metric": metric,
                            "delta_r2_targeted": t_val,
                            "delta_r2_random": r_val,
                            "paired_seed_count": paired_seed_count,
                            "paired_pair_count": paired_pair_count,
                            "random_draw_mode": random_draw_mode,
                            "insufficient_pairing": bool(insufficient_pairing),
                            "random_draws_per_seed_mean": draw_mean,
                            "random_draws_per_seed_sd": draw_sd,
                            "S_ratio": ratio,
                            "S_diff": diff,
                            "ratio_stable": bool(stable),
                            "severity_pass": bool(sev_pass_point),
                            "severity_pass_ci": sev_pass_ci,
                            "diff_ci_low": float(diff_ci[0]),
                            "diff_ci_high": float(diff_ci[1]),
                            "diff_p": float(diff_p),
                            "diff_label": str(diff_label),
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    grouped = out.groupby(["model", "task_class", "target", "metric"], as_index=False)
    both_pass = grouped["severity_pass"].all().rename(columns={"severity_pass": "h3_group_pass"})
    if compute_ci:
        both_pass_ci = grouped["severity_pass_ci"].all().rename(columns={"severity_pass_ci": "h3_group_pass_ci"})
    else:
        both_pass_ci = both_pass.copy()
        both_pass_ci["h3_group_pass_ci"] = float("nan")
        both_pass_ci = both_pass_ci.drop(columns=["h3_group_pass"])
    merged = out.merge(both_pass, on=["model", "task_class", "target", "metric"], how="left")
    merged = merged.merge(both_pass_ci, on=["model", "task_class", "target", "metric"], how="left")
    return merged


def _h3_summary_block(h3_df: pd.DataFrame, *, metric: str) -> dict[str, Any]:
    if h3_df.empty:
        return {
            "metric": metric,
            "rows_total": 0,
            "rows_evaluable": 0,
            "pass_rate": float("nan"),
            "pass_rate_ci": float("nan"),
        }
    evaluable = h3_df[~h3_df["insufficient_pairing"]].drop_duplicates(["model", "task_class", "target", "metric"])
    pass_rate = float(evaluable["h3_group_pass"].mean()) if not evaluable.empty else float("nan")
    pass_rate_ci = (
        float(evaluable["h3_group_pass_ci"].mean())
        if (not evaluable.empty and "h3_group_pass_ci" in evaluable.columns)
        else float("nan")
    )
    return {
        "metric": metric,
        "rows_total": int(len(h3_df)),
        "rows_evaluable": int(len(evaluable)),
        "pass_rate": pass_rate,
        "pass_rate_ci": pass_rate_ci,
    }


def _restricted_rank_diagnostic(task_eff: pd.DataFrame) -> dict[str, Any]:
    required = {"mean_accuracy_full_vocab", "mean_accuracy_restricted", "split"}
    if not required.issubset(set(task_eff.columns)):
        return {
            "rows": 0,
            "available": False,
            "note": "Restricted/full-vocab dual metrics unavailable in this run schema.",
        }
    work = task_eff[task_eff["split"].isin(["synthetic", "span_bridge", "mechanistic"])].copy()
    if "eval_mode" in work.columns:
        work = work[work["eval_mode"] == "restricted"].copy()
    work = work[np.isfinite(work["mean_accuracy_full_vocab"]) & np.isfinite(work["mean_accuracy_restricted"])].copy()
    if work.empty:
        return {
            "rows": 0,
            "available": True,
            "overall_spearman": float("nan"),
            "overall_rank_preserved": False,
            "rank_preservation_flag": True,
            "per_model": [],
        }

    overall = float(work["mean_accuracy_full_vocab"].corr(work["mean_accuracy_restricted"], method="spearman"))
    per_model: list[dict[str, Any]] = []
    for model, sub in work.groupby("model"):
        rho = float(sub["mean_accuracy_full_vocab"].corr(sub["mean_accuracy_restricted"], method="spearman"))
        per_model.append(
            {
                "model": str(model),
                "rows": int(len(sub)),
                "spearman": rho,
                "rank_preserved": bool(np.isfinite(rho) and rho >= 0.70),
            }
        )
    overall_ok = bool(np.isfinite(overall) and overall >= 0.70)
    return {
        "rows": int(len(work)),
        "available": True,
        "overall_spearman": overall,
        "overall_rank_preserved": overall_ok,
        "rank_preservation_flag": bool(not overall_ok),
        "per_model": per_model,
    }


def _phase2b_confirmatory_status(phase_root: Path) -> bool | None:
    run_id = phase_root.name
    output_root = phase_root.parent.parent
    phase2b_gate_path = output_root / "phase2b" / run_id / "gate_evaluation.json"
    if not phase2b_gate_path.exists():
        return None
    try:
        payload = json.loads(phase2b_gate_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("confirmatory_success")
    if isinstance(value, bool):
        return value
    return None


def _min_seed_count(task_eff: pd.DataFrame) -> int:
    if task_eff.empty or "seed" not in task_eff.columns:
        return 0
    key_cols = _conditional_key_cols(
        task_eff,
        required=["model", "task", "seq_len", "split", "intervention"],
        optional=["dataset", "span", "random_draw"],
    )
    grouped = task_eff.groupby(key_cols, as_index=False)["seed"].nunique()
    if grouped.empty:
        return 0
    return int(grouped["seed"].min())


def _degrade_label(label: str) -> str:
    order = ["no_directional_support", "directional_nonselective", "strong_directional_support"]
    try:
        idx = order.index(label)
    except ValueError:
        return label
    return order[max(0, idx - 1)]


def _selectivity_label(target_drop: float, non_target_drop: float, *, contrast_threshold: float = 0.05) -> str:
    contrast = target_drop - non_target_drop
    if not np.isfinite(contrast) or not np.isfinite(target_drop) or target_drop <= 0 or contrast < contrast_threshold:
        return "no_directional_support"
    if (non_target_drop <= 0.03) or (non_target_drop <= 0.5 * max(target_drop, 1e-9)):
        return "strong_directional_support"
    return "directional_nonselective"


def _power_block(values: np.ndarray, *, target_threshold: float = 0.05) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {
            "n": 0,
            "sd": float("nan"),
            "mde_at_80pct_power": float("nan"),
            "target_threshold": target_threshold,
            "underpowered": True,
            "alpha_two_sided": 0.025,
            "method": "normal_approximation_planning_only",
        }
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    # Conservative primary-family alpha proxy for Holm across H1/H2.
    alpha = 0.025
    mde = mde_two_sided_normal(sd=sd, n=n, alpha=alpha, power=0.80)
    return {
        "n": n,
        "sd": sd,
        "mde_at_80pct_power": mde,
        "target_threshold": target_threshold,
        "underpowered": bool(np.isfinite(mde) and (mde > target_threshold)),
        "alpha_two_sided": alpha,
        "method": "normal_approximation_planning_only",
    }


def _seed_correlation(task_eff: pd.DataFrame) -> dict[str, Any]:
    none = task_eff[(task_eff["intervention"] == "none") & (task_eff["split"].isin(["synthetic", "mechanistic", "span_bridge"]))].copy()
    if none.empty:
        return {
            "max_pairwise_corr": float("nan"),
            "correlation_sensitive": False,
            "matrix": {},
            "note": "No synthetic none-intervention rows available.",
        }
    seed_model = none.groupby(["model", "seed"], as_index=False)["mean_nll"].mean()
    piv = seed_model.pivot(index="seed", columns="model", values="mean_nll")
    cols = [c for c in piv.columns if piv[c].notna().sum() >= 2]
    piv = piv[cols]
    if len(cols) < 2:
        return {
            "max_pairwise_corr": float("nan"),
            "correlation_sensitive": False,
            "matrix": {},
            "note": "Insufficient model overlap for pairwise seed correlations.",
        }
    corr = piv.corr(min_periods=2)
    max_pair = float("nan")
    for i, c1 in enumerate(corr.columns):
        for c2 in corr.columns[i + 1 :]:
            val = corr.loc[c1, c2]
            if np.isfinite(val):
                max_pair = val if not np.isfinite(max_pair) else max(max_pair, float(val))
    matrix = {
        str(c1): {str(c2): float(corr.loc[c1, c2]) for c2 in corr.columns}
        for c1 in corr.columns
    }
    return {
        "max_pairwise_corr": max_pair,
        "correlation_sensitive": bool(np.isfinite(max_pair) and max_pair > 0.60),
        "matrix": matrix,
    }


def _h5_model_support(tier: pd.DataFrame, model: str) -> dict[str, Any]:
    m = tier[tier["model"] == model].copy()
    if m.empty:
        return {
            "model": model,
            "supported": False,
            "binning_validity_pass": False,
            "min_window_agreement": float("nan"),
            "quartile_conditioned": False,
            "targeted_max_abs_dependency_delta": float("nan"),
            "random_max_abs_dependency_delta": float("nan"),
            "high_directional_delta": float("nan"),
            "low_directional_delta": float("nan"),
            "directionally_coherent": False,
        }

    required_pairs = {(dep, q) for dep in ("local", "distal") for q in ("Q1", "Q2", "Q3", "Q4")}
    quartile_conditioned = True
    group_cols = ["intervention", "seed"]
    if "random_draw" in m.columns:
        group_cols.append("random_draw")
    for _, sub in m.groupby(group_cols, dropna=False):
        present = {(str(r.dependency_type), str(r.baseline_nll_quartile)) for r in sub.itertuples(index=False)}
        if not required_pairs.issubset(present):
            quartile_conditioned = False
            break

    robust_col = m["binning_robust_pass"].fillna(True) if "binning_robust_pass" in m.columns else pd.Series([True] * len(m))
    binning_valid = bool(m["h5_interpretable"].fillna(False).all() and robust_col.all())
    min_window_agreement = float(m["binning_window_agreement"].min()) if "binning_window_agreement" in m.columns and not m.empty else float("nan")
    tgt_df = m[m["intervention"].isin(["ablate_high_strong", "ablate_low_strong"])]
    if tgt_df.empty:
        tgt = float("nan")
    else:
        tgt_seed = tgt_df.groupby(["seed", "intervention"], as_index=False)["dependency_delta"].mean()
        tgt = float(tgt_seed["dependency_delta"].abs().max())
    high_df = m[m["intervention"] == "ablate_high_strong"].groupby("seed", as_index=False)["dependency_delta"].mean()
    low_df = m[m["intervention"] == "ablate_low_strong"].groupby("seed", as_index=False)["dependency_delta"].mean()
    high_dir = float(high_df["dependency_delta"].mean()) if not high_df.empty else float("nan")
    low_dir = float(low_df["dependency_delta"].mean()) if not low_df.empty else float("nan")
    directionally_coherent = bool(np.isfinite(high_dir) and np.isfinite(low_dir) and (high_dir >= 0.0) and (low_dir <= 0.0))
    rnd_df = m[m["intervention"] == "random_strong"]
    if rnd_df.empty:
        rnd = float("nan")
    else:
        rnd_seed = rnd_df.groupby("seed", as_index=False)["dependency_delta"].mean()
        rnd = float(rnd_seed["dependency_delta"].abs().max())
    supported = bool(
        quartile_conditioned
        and binning_valid
        and np.isfinite(tgt)
        and np.isfinite(rnd)
        and (tgt >= 0.5)
        and ((tgt - rnd) >= 0.2)
    )
    return {
        "model": model,
        "supported": supported,
        "binning_validity_pass": binning_valid,
        "min_window_agreement": min_window_agreement,
        "quartile_conditioned": quartile_conditioned,
        "targeted_max_abs_dependency_delta": tgt,
        "random_max_abs_dependency_delta": rnd,
        "high_directional_delta": high_dir,
        "low_directional_delta": low_dir,
        "directionally_coherent": directionally_coherent,
    }


def _phase2a_gate(
    task_eff: pd.DataFrame,
    kernel_delta: pd.DataFrame,
    *,
    endpoint_policy: str = "raw_primary",
) -> dict[str, Any]:
    eff = _eligible_task_rows(task_eff)
    ker = _filter_kernel_to_eligible(kernel_delta, eff)
    high = _class_contrast(eff, "ablate_high_strong", "h1")
    low = _class_contrast(eff, "ablate_low_strong", "h2")
    rnd_h1 = _class_contrast(eff, "random_strong", "h1")
    rnd_h2 = _class_contrast(eff, "random_strong", "h2")

    h1_vals = high["contrast"].to_numpy(dtype=np.float64)
    h2_vals = low["contrast"].to_numpy(dtype=np.float64)
    h1_mean = float(np.nanmean(h1_vals)) if h1_vals.size else float("nan")
    h2_mean = float(np.nanmean(h2_vals)) if h2_vals.size else float("nan")
    h1_ci = bootstrap_bca_ci(h1_vals) if h1_vals.size else (float("nan"), float("nan"))
    h2_ci = bootstrap_bca_ci(h2_vals) if h2_vals.size else (float("nan"), float("nan"))
    p_raw = {
        "H1": paired_sign_flip_pvalue(h1_vals) if h1_vals.size else float("nan"),
        "H2": paired_sign_flip_pvalue(h2_vals) if h2_vals.size else float("nan"),
    }
    p_adj = holm_bonferroni(p_raw)

    h1_label = precision_label(effect=h1_mean, ci_low=h1_ci[0], threshold=0.05)
    h2_label = precision_label(effect=h2_mean, ci_low=h2_ci[0], threshold=0.05)
    h1_sd = float(np.std(h1_vals, ddof=1)) if h1_vals.size > 1 else float("nan")
    h2_sd = float(np.std(h2_vals, ddof=1)) if h2_vals.size > 1 else float("nan")
    h1_d = float(h1_mean / h1_sd) if np.isfinite(h1_mean) and np.isfinite(h1_sd) and h1_sd > 0 else float("nan")
    h2_d = float(h2_mean / h2_sd) if np.isfinite(h2_mean) and np.isfinite(h2_sd) and h2_sd > 0 else float("nan")

    meaningful = bool(eff[(eff["intervention"].isin(["ablate_high_strong", "ablate_low_strong"])) & (eff["meaningful_effect"])].shape[0] > 0)
    rnd_h1_mean = float(np.nanmean(rnd_h1["contrast"])) if len(rnd_h1) else 0.0
    rnd_h2_mean = float(np.nanmean(rnd_h2["contrast"])) if len(rnd_h2) else 0.0
    specificity = max(h1_mean - rnd_h1_mean, h2_mean - rnd_h2_mean)

    ker_targeted = ker[
        ker["intervention"].isin(["ablate_high_strong", "ablate_low_strong"])
        & ker["metric"].isin(["track_a", "track_b_raw"])
        & ker["task_class"].isin(["short", "long"])
    ]
    kernel_shift = float(ker_targeted["delta_r2_abs"].max()) if not ker_targeted.empty else float("nan")
    audit_ok = bool((eff.get("norm_drift_pass", pd.Series([True]))).all())
    has_primary_pass = (h1_label == "pass") or (h2_label == "pass")
    has_imprecise_only = ((h1_label == "imprecise_pass") or (h2_label == "imprecise_pass")) and not has_primary_pass

    # Selectivity labels from strong targeted interventions.
    s_high = eff[eff["intervention"] == "ablate_high_strong"].groupby("task_class")["drop_acc"].mean()
    s_low = eff[eff["intervention"] == "ablate_low_strong"].groupby("task_class")["drop_acc"].mean()
    h1_target = float(s_high.get("short", float("nan")))
    h1_non_target = float(s_high.get("long", float("nan")))
    h2_target = float(s_low.get("long", float("nan")))
    h2_non_target = float(s_low.get("short", float("nan")))
    h1_select = _selectivity_label(h1_target, h1_non_target)
    h2_select = _selectivity_label(h2_target, h2_non_target)

    # Baseline-conditioned normalized selectivity sensitivity.
    none_cls = eff[eff["intervention"] == "none"].groupby("task_class")["none_accuracy"].mean()
    short_base = float(none_cls.get("short", float("nan")))
    long_base = float(none_cls.get("long", float("nan")))
    h1_target_norm = h1_target / max(short_base, 1e-6) if np.isfinite(h1_target) else float("nan")
    h1_non_target_norm = h1_non_target / max(long_base, 1e-6) if np.isfinite(h1_non_target) else float("nan")
    h2_target_norm = h2_target / max(long_base, 1e-6) if np.isfinite(h2_target) else float("nan")
    h2_non_target_norm = h2_non_target / max(short_base, 1e-6) if np.isfinite(h2_non_target) else float("nan")
    h1_select_norm = _selectivity_label(h1_target_norm, h1_non_target_norm)
    h2_select_norm = _selectivity_label(h2_target_norm, h2_non_target_norm)
    h1_floor_sensitive = h1_select != h1_select_norm
    h2_floor_sensitive = h2_select != h2_select_norm
    h1_select_final = _degrade_label(h1_select) if h1_floor_sensitive else h1_select
    h2_select_final = _degrade_label(h2_select) if h2_floor_sensitive else h2_select

    power = {
        "H1": _power_block(h1_vals, target_threshold=0.05),
        "H2": _power_block(h2_vals, target_threshold=0.05),
    }
    restricted_rank = _restricted_rank_diagnostic(task_eff)
    ci_sensitivity = {
        "H1": bootstrap_bca_ci_sensitivity(h1_vals),
        "H2": bootstrap_bca_ci_sensitivity(h2_vals),
    }
    h1_base = _class_contrast(eff, "ablate_high_strong", "h1", value_col="drop_acc_over_baseline")
    h2_base = _class_contrast(eff, "ablate_low_strong", "h2", value_col="drop_acc_over_baseline")
    h1_headroom = _class_contrast(eff, "ablate_high_strong", "h1", value_col="drop_acc_over_headroom")
    h2_headroom = _class_contrast(eff, "ablate_low_strong", "h2", value_col="drop_acc_over_headroom")
    base_diag = _contrast_inference_block(
        h1_base["contrast"].to_numpy(dtype=np.float64),
        h2_base["contrast"].to_numpy(dtype=np.float64),
        threshold=0.05,
    )
    headroom_diag = _contrast_inference_block(
        h1_headroom["contrast"].to_numpy(dtype=np.float64),
        h2_headroom["contrast"].to_numpy(dtype=np.float64),
        threshold=0.05,
    )
    raw_dirs = {"h1": _effect_direction(h1_mean), "h2": _effect_direction(h2_mean)}
    for block in (base_diag, headroom_diag):
        for hyp in ("h1", "h2"):
            block[hyp]["raw_direction"] = raw_dirs[hyp]
            block[hyp]["raw_direction_agree"] = bool(block[hyp]["direction"] == raw_dirs[hyp])
    agree_base = bool(base_diag["h1"]["raw_direction_agree"] and base_diag["h2"]["raw_direction_agree"])
    agree_headroom = bool(headroom_diag["h1"]["raw_direction_agree"] and headroom_diag["h2"]["raw_direction_agree"])
    class_headroom = _task_class_headroom_summary(eff)
    overlap_diag = _long_offset_overlap_diagnostic()
    severe_imbalance = bool(class_headroom.get("severe_imbalance", False))
    headroom_confound_risk = bool((not agree_base) or (not agree_headroom) or severe_imbalance)
    headroom_diag_payload = {
        "advisory_only": True,
        "policy": "raw_primary_non_gating",
        "note": "Headroom-normalized contrasts are diagnostic only; confirmatory criteria remain based on raw drops.",
        "raw_reference": {
            "h1": {"effect": h1_mean, "direction": raw_dirs["h1"], "n": int(h1_vals.size)},
            "h2": {"effect": h2_mean, "direction": raw_dirs["h2"], "n": int(h2_vals.size)},
        },
        "baseline_normalized": base_diag,
        "headroom_normalized": headroom_diag,
        "raw_vs_baseline_direction_agreement_all": agree_base,
        "raw_vs_headroom_direction_agreement_all": agree_headroom,
    }
    endpoint_adjudication = _h12_endpoint_adjudication(
        endpoint_policy=endpoint_policy,
        raw_h1_label=h1_label,
        raw_h2_label=h2_label,
        headroom_diag_payload=headroom_diag_payload,
    )

    h3_confirm = _h3_specificity(
        ker,
        allowed_task_classes={"short", "long"},
        allowed_metrics={"track_b_raw"},
        compute_ci=False,
    )
    h3_track_a = _h3_specificity(
        ker,
        allowed_task_classes={"short", "long"},
        allowed_metrics={"track_a"},
        compute_ci=False,
    )
    h3_centered = _h3_specificity(
        ker,
        allowed_task_classes={"short", "long"},
        allowed_metrics={"track_b_centered"},
        compute_ci=False,
    )

    criteria = {
        "audit_pass": audit_ok,
        "strong_meaningful_effect": meaningful,
        "class_contrast_threshold": bool((h1_mean >= 0.05) or (h2_mean >= 0.05)),
        "specificity_advantage_ge_0_02": bool(specificity >= 0.02),
        "kernel_shift_ge_0_03": bool(np.isfinite(kernel_shift) and kernel_shift >= 0.03),
        "ci_lower_gt_zero_for_primary": has_primary_pass,
        "imprecise_pass": has_imprecise_only,
    }
    auto_advance = all(
        criteria[k]
        for k in [
            "audit_pass",
            "strong_meaningful_effect",
            "class_contrast_threshold",
            "specificity_advantage_ge_0_02",
            "kernel_shift_ge_0_03",
            "ci_lower_gt_zero_for_primary",
        ]
    )

    return {
        "phase": "phase2a",
        "eligible_rows": int(len(eff)),
        "excluded_floor_rows": int(len(task_eff) - len(eff)),
        "criteria": criteria,
        "auto_advance": bool(auto_advance),
        "h1": {
            "effect": h1_mean,
            "ci": [h1_ci[0], h1_ci[1]],
            "p": p_raw["H1"],
            "p_holm": p_adj["H1"],
            "label": h1_label,
            "effect_size_d": h1_d,
            "selectivity_label": h1_select_final,
            "selectivity_raw_label": h1_select,
            "selectivity_normalized_label": h1_select_norm,
            "selectivity_floor_sensitive": h1_floor_sensitive,
        },
        "h2": {
            "effect": h2_mean,
            "ci": [h2_ci[0], h2_ci[1]],
            "p": p_raw["H2"],
            "p_holm": p_adj["H2"],
            "label": h2_label,
            "effect_size_d": h2_d,
            "selectivity_label": h2_select_final,
            "selectivity_raw_label": h2_select,
            "selectivity_normalized_label": h2_select_norm,
            "selectivity_floor_sensitive": h2_floor_sensitive,
        },
        "kernel_shift_max_abs": kernel_shift,
        "specificity_advantage": specificity,
        "power_detectability": power,
        "restricted_rank_diagnostic": restricted_rank,
        "ci_seed_sensitivity": ci_sensitivity,
        "headroom_normalized_diagnostic": headroom_diag_payload,
        "h12_endpoint_policy": _normalize_h12_endpoint_policy(endpoint_policy),
        "h12_endpoint_adjudication": endpoint_adjudication,
        "task_class_headroom_summary": class_headroom,
        "headroom_confound_risk": headroom_confound_risk,
        "normalized_contrast_policy": "advisory_non_gating_raw_primary",
        "span_overlap_diagnostic": overlap_diag,
        "h3_confirmatory_track_b_raw": _h3_summary_block(h3_confirm, metric="track_b_raw"),
        "h3_aux_track_a": _h3_summary_block(h3_track_a, metric="track_a"),
        "h3_diagnostic_centered": _h3_summary_block(h3_centered, metric="track_b_centered"),
    }


def _phase2b_gate(
    task_eff: pd.DataFrame,
    kernel_delta: pd.DataFrame,
    phase_root: Path,
    *,
    endpoint_policy: str = "raw_primary",
) -> dict[str, Any]:
    eff = _eligible_task_rows(task_eff)
    ker = _filter_kernel_to_eligible(kernel_delta, eff)
    active_long_tasks = sorted(
        {
            str(task)
            for task in eff[eff["task_class"] == "long"]["task"].dropna().unique().tolist()
        }
    )
    cross_task_sign_policy = "adaptive_by_active_long_task_count"
    cross_task_sign_rule_version = "v2_single_long_adaptive"
    ker_short_long = ker[ker["task_class"].isin(["short", "long"])].copy()
    ker_bridge = ker[ker["task_class"] == "bridge"].copy()
    ker_short_long_raw = ker_short_long[ker_short_long["metric"] == "track_b_raw"].copy()
    ker_short_long_track_a = ker_short_long[ker_short_long["metric"] == "track_a"].copy()
    ker_short_long_centered = ker_short_long[ker_short_long["metric"] == "track_b_centered"].copy()
    ker_bridge_raw = ker_bridge[ker_bridge["metric"] == "track_b_raw"].copy()
    h3_confirm = _h3_specificity(
        ker_short_long_raw,
    )
    h3_track_a = _h3_specificity(
        ker_short_long_track_a,
    )
    h3_centered = _h3_specificity(
        ker_short_long_centered,
    )
    h3_confirm_per_draw = _h3_specificity(
        ker_short_long_raw,
        random_draw_mode="per_draw",
    )
    bridge_specificity = _h3_specificity(
        ker_bridge_raw,
    )
    model_rows: list[dict[str, Any]] = []
    per_model_power: dict[str, dict[str, Any]] = {}
    h1_all_vals: list[float] = []
    h2_all_vals: list[float] = []
    for model in sorted(eff["model"].dropna().unique()):
        m = eff[eff["model"] == model]
        h1 = _class_contrast(m, "ablate_high_strong", "h1")
        h2 = _class_contrast(m, "ablate_low_strong", "h2")
        r1 = _class_contrast(m, "random_strong", "h1")
        r2 = _class_contrast(m, "random_strong", "h2")

        h1_mean = float(np.nanmean(h1["contrast"])) if len(h1) else float("nan")
        h2_mean = float(np.nanmean(h2["contrast"])) if len(h2) else float("nan")
        h1_all_vals.extend(h1["contrast"].to_numpy(dtype=np.float64).tolist())
        h2_all_vals.extend(h2["contrast"].to_numpy(dtype=np.float64).tolist())
        rnd_adv = max(
            h1_mean - (float(np.nanmean(r1["contrast"])) if len(r1) else 0.0),
            h2_mean - (float(np.nanmean(r2["contrast"])) if len(r2) else 0.0),
        )
        support = bool((h1_mean >= 0.05) or (h2_mean >= 0.05))
        per_model_power[model] = {
            "H1": _power_block(h1["contrast"].to_numpy(dtype=np.float64), target_threshold=0.05),
            "H2": _power_block(h2["contrast"].to_numpy(dtype=np.float64), target_threshold=0.05),
        }

        # Cross-task sign agreement in strong targeted conditions.
        # For single-long-task batteries, use a directional consistency rule instead of requiring >=2 long tasks.
        sign_ok = False
        model_long_tasks = sorted(
            {
                str(task)
                for task in m[m["task_class"] == "long"]["task"].dropna().unique().tolist()
            }
        )
        long_task_count = len(model_long_tasks)
        for intervention in ("ablate_high_strong", "ablate_low_strong"):
            s = m[(m["intervention"] == intervention) & (m["task_class"] == "short")].groupby("task")["drop_acc"].mean()
            l = m[(m["intervention"] == intervention) & (m["task_class"] == "long")].groupby("task")["drop_acc"].mean()
            if long_task_count >= 2 and len(s) >= 2 and len(l) >= 2:
                short_sign = np.sign(s.to_numpy())
                long_sign = np.sign(l.to_numpy())
                if (
                    (len(set(short_sign.tolist())) == 1)
                    and (len(set(long_sign.tolist())) == 1)
                    and (short_sign[0] != 0)
                    and (long_sign[0] != 0)
                ):
                    sign_ok = True
                    break
            elif long_task_count == 1 and len(s) >= 2 and len(l) >= 1:
                short_sign = np.sign(s.to_numpy())
                if (len(set(short_sign.tolist())) == 1) and (short_sign[0] != 0):
                    long_val = float(l.iloc[0])
                    long_sign = np.sign(long_val)
                    if long_sign != 0:
                        short_mean = float(s.mean())
                        long_mean = float(l.mean())
                        directional_consistency = (
                            (short_mean >= long_mean)
                            if intervention == "ablate_high_strong"
                            else (long_mean >= short_mean)
                        )
                        if directional_consistency:
                            sign_ok = True
                            break

        # Content-gated safeguard to avoid copy-family-only support.
        km_high = float(m[(m["intervention"] == "ablate_high_strong") & (m["task"] == "local_key_match")]["drop_acc"].mean())
        lr_high = float(m[(m["intervention"] == "ablate_high_strong") & (m["task"] == "long_range_retrieval")]["drop_acc"].mean())
        km_low = float(m[(m["intervention"] == "ablate_low_strong") & (m["task"] == "local_key_match")]["drop_acc"].mean())
        lr_low = float(m[(m["intervention"] == "ablate_low_strong") & (m["task"] == "long_range_retrieval")]["drop_acc"].mean())
        high_content_contrast = km_high - lr_high if np.isfinite(km_high) and np.isfinite(lr_high) else float("nan")
        low_content_contrast = lr_low - km_low if np.isfinite(km_low) and np.isfinite(lr_low) else float("nan")
        content_ok = bool(
            (np.isfinite(high_content_contrast) and high_content_contrast >= 0.03)
            or (np.isfinite(low_content_contrast) and low_content_contrast >= 0.03)
        )
        short_high_by_task = (
            m[(m["intervention"] == "ablate_high_strong") & (m["task_class"] == "short")]
            .groupby("task")["drop_acc"]
            .mean()
        )
        long_low_by_task = (
            m[(m["intervention"] == "ablate_low_strong") & (m["task_class"] == "long")]
            .groupby("task")["drop_acc"]
            .mean()
        )
        short_high_gap = (
            float(short_high_by_task.max() - short_high_by_task.min())
            if len(short_high_by_task) >= 2
            else float("nan")
        )
        long_low_gap = (
            float(long_low_by_task.max() - long_low_by_task.min())
            if len(long_low_by_task) >= 2
            else float("nan")
        )

        # Directional selectivity labels.
        high_cls = m[m["intervention"] == "ablate_high_strong"].groupby("task_class")["drop_acc"].mean()
        low_cls = m[m["intervention"] == "ablate_low_strong"].groupby("task_class")["drop_acc"].mean()
        h1_target = float(high_cls.get("short", float("nan")))
        h1_non_target = float(high_cls.get("long", float("nan")))
        h2_target = float(low_cls.get("long", float("nan")))
        h2_non_target = float(low_cls.get("short", float("nan")))
        h1_select = _selectivity_label(h1_target, h1_non_target)
        h2_select = _selectivity_label(h2_target, h2_non_target)

        none_cls = m[m["intervention"] == "none"].groupby("task_class")["none_accuracy"].mean()
        short_base = float(none_cls.get("short", float("nan")))
        long_base = float(none_cls.get("long", float("nan")))
        h1_target_norm = h1_target / max(short_base, 1e-6) if np.isfinite(h1_target) else float("nan")
        h1_non_target_norm = h1_non_target / max(long_base, 1e-6) if np.isfinite(h1_non_target) else float("nan")
        h2_target_norm = h2_target / max(long_base, 1e-6) if np.isfinite(h2_target) else float("nan")
        h2_non_target_norm = h2_non_target / max(short_base, 1e-6) if np.isfinite(h2_non_target) else float("nan")
        h1_select_norm = _selectivity_label(h1_target_norm, h1_non_target_norm)
        h2_select_norm = _selectivity_label(h2_target_norm, h2_non_target_norm)
        h1_floor_sensitive = h1_select != h1_select_norm
        h2_floor_sensitive = h2_select != h2_select_norm
        h1_select_final = _degrade_label(h1_select) if h1_floor_sensitive else h1_select
        h2_select_final = _degrade_label(h2_select) if h2_floor_sensitive else h2_select

        model_rows.append(
            {
                "model": model,
                "h1_effect": h1_mean,
                "h2_effect": h2_mean,
                "h1_or_h2_support": support,
                "cross_task_sign_agreement": sign_ok,
                "content_gated_support": content_ok,
                "content_contrast_high": high_content_contrast,
                "content_contrast_low": low_content_contrast,
                "short_high_task_gap": short_high_gap,
                "long_low_task_gap": long_low_gap,
                "specificity_advantage": rnd_adv,
                "h1_selectivity_label": h1_select_final,
                "h1_selectivity_raw_label": h1_select,
                "h1_selectivity_normalized_label": h1_select_norm,
                "h1_selectivity_floor_sensitive": h1_floor_sensitive,
                "h2_selectivity_label": h2_select_final,
                "h2_selectivity_raw_label": h2_select,
                "h2_selectivity_normalized_label": h2_select_norm,
                "h2_selectivity_floor_sensitive": h2_floor_sensitive,
                "active_long_tasks": model_long_tasks,
                "active_long_task_count": int(long_task_count),
            }
        )

    model_df = pd.DataFrame(model_rows)
    rep_ok = int(model_df["h1_or_h2_support"].sum()) >= 2 if not model_df.empty else False
    cross_ok = int(model_df["cross_task_sign_agreement"].sum()) >= 2 if not model_df.empty else False
    content_ok = int(model_df["content_gated_support"].sum()) >= 2 if not model_df.empty else False
    spec_ok = int((model_df["specificity_advantage"] >= 0.02).sum()) >= 2 if not model_df.empty else False
    short_gap_max = (
        float(model_df["short_high_task_gap"].dropna().max())
        if (not model_df.empty and "short_high_task_gap" in model_df.columns and model_df["short_high_task_gap"].notna().any())
        else float("nan")
    )
    long_gap_max = (
        float(model_df["long_low_task_gap"].dropna().max())
        if (not model_df.empty and "long_low_task_gap" in model_df.columns and model_df["long_low_task_gap"].notna().any())
        else float("nan")
    )
    task_family_gap_flag = bool(
        (np.isfinite(short_gap_max) and short_gap_max > 0.05) or (np.isfinite(long_gap_max) and long_gap_max > 0.05)
    )

    h1_arr = np.asarray(h1_all_vals, dtype=np.float64)
    h2_arr = np.asarray(h2_all_vals, dtype=np.float64)
    h1_arr = h1_arr[np.isfinite(h1_arr)]
    h2_arr = h2_arr[np.isfinite(h2_arr)]
    h1_effect = float(np.mean(h1_arr)) if h1_arr.size else float("nan")
    h2_effect = float(np.mean(h2_arr)) if h2_arr.size else float("nan")
    h1_ci = bootstrap_bca_ci(h1_arr) if h1_arr.size else (float("nan"), float("nan"))
    h2_ci = bootstrap_bca_ci(h2_arr) if h2_arr.size else (float("nan"), float("nan"))
    p_raw = {
        "H1": paired_sign_flip_pvalue(h1_arr) if h1_arr.size else float("nan"),
        "H2": paired_sign_flip_pvalue(h2_arr) if h2_arr.size else float("nan"),
    }
    p_adj = holm_bonferroni(p_raw)
    h1_label = precision_label(effect=h1_effect, ci_low=h1_ci[0], threshold=0.05)
    h2_label = precision_label(effect=h2_effect, ci_low=h2_ci[0], threshold=0.05)
    h1_sd = float(np.std(h1_arr, ddof=1)) if h1_arr.size > 1 else float("nan")
    h2_sd = float(np.std(h2_arr, ddof=1)) if h2_arr.size > 1 else float("nan")
    h1_d = float(h1_effect / h1_sd) if np.isfinite(h1_effect) and np.isfinite(h1_sd) and h1_sd > 0 else float("nan")
    h2_d = float(h2_effect / h2_sd) if np.isfinite(h2_effect) and np.isfinite(h2_sd) and h2_sd > 0 else float("nan")
    primary_sig = bool(
        (np.isfinite(p_adj["H1"]) and p_adj["H1"] <= 0.05 and h1_effect >= 0.05)
        or (np.isfinite(p_adj["H2"]) and p_adj["H2"] <= 0.05 and h2_effect >= 0.05)
    )

    h1_model_arr = (
        model_df["h1_effect"].to_numpy(dtype=np.float64)
        if ("h1_effect" in model_df.columns)
        else np.asarray([], dtype=np.float64)
    )
    h2_model_arr = (
        model_df["h2_effect"].to_numpy(dtype=np.float64)
        if ("h2_effect" in model_df.columns)
        else np.asarray([], dtype=np.float64)
    )
    h1_model_arr = h1_model_arr[np.isfinite(h1_model_arr)]
    h2_model_arr = h2_model_arr[np.isfinite(h2_model_arr)]
    h1_model_effect = float(np.mean(h1_model_arr)) if h1_model_arr.size else float("nan")
    h2_model_effect = float(np.mean(h2_model_arr)) if h2_model_arr.size else float("nan")
    h1_model_ci = bootstrap_bca_ci(h1_model_arr) if h1_model_arr.size else (float("nan"), float("nan"))
    h2_model_ci = bootstrap_bca_ci(h2_model_arr) if h2_model_arr.size else (float("nan"), float("nan"))
    p_cluster_raw = {
        "H1": paired_sign_flip_pvalue(h1_model_arr) if h1_model_arr.size else float("nan"),
        "H2": paired_sign_flip_pvalue(h2_model_arr) if h2_model_arr.size else float("nan"),
    }
    p_cluster_adj = holm_bonferroni(p_cluster_raw)
    h1_model_label = precision_label(effect=h1_model_effect, ci_low=h1_model_ci[0], threshold=0.05)
    h2_model_label = precision_label(effect=h2_model_effect, ci_low=h2_model_ci[0], threshold=0.05)
    min_exact_p_h1 = float(1.0 / (2 ** int(h1_model_arr.size))) if h1_model_arr.size else float("nan")
    min_exact_p_h2 = float(1.0 / (2 ** int(h2_model_arr.size))) if h2_model_arr.size else float("nan")
    cluster_h1_evaluable = bool(np.isfinite(min_exact_p_h1) and min_exact_p_h1 <= 0.05)
    cluster_h2_evaluable = bool(np.isfinite(min_exact_p_h2) and min_exact_p_h2 <= 0.05)
    cluster_any_evaluable = bool(cluster_h1_evaluable or cluster_h2_evaluable)
    cluster_evaluable = bool(cluster_any_evaluable)
    cluster_sig = (
        bool(
            (np.isfinite(p_cluster_adj["H1"]) and p_cluster_adj["H1"] <= 0.05 and h1_model_effect >= 0.05)
            or (np.isfinite(p_cluster_adj["H2"]) and p_cluster_adj["H2"] <= 0.05 and h2_model_effect >= 0.05)
        )
        if cluster_evaluable
        else None
    )
    cluster_sensitivity_flag = bool(cluster_evaluable and isinstance(cluster_sig, bool) and (primary_sig != cluster_sig))
    cluster_sensitivity_note = (
        "Cluster sensitivity is underpowered at current model count; treat as one-directional diagnostic only."
        if not cluster_evaluable
        else (
            "Primary fixed-effects pooled inference and model-cluster sensitivity disagree; downgrade claim strength."
            if cluster_sensitivity_flag
            else "Primary fixed-effects pooled inference and model-cluster sensitivity are aligned."
        )
    )

    h3_groups = h3_confirm[~h3_confirm["insufficient_pairing"]].drop_duplicates(["model", "task_class", "target", "metric"])
    h3_rate_point = float(h3_groups["h3_group_pass"].mean()) if not h3_groups.empty else 0.0
    h3_rate_ci = (
        float(h3_groups["h3_group_pass_ci"].mean())
        if (not h3_groups.empty and "h3_group_pass_ci" in h3_groups.columns)
        else float("nan")
    )
    h3_ok = bool(np.isfinite(h3_rate_point) and h3_rate_point >= 0.5)
    h3_ci_diagnostic_flag = bool(np.isfinite(h3_rate_ci) and (h3_rate_ci < 0.5))
    h3_insufficient = int(h3_confirm["insufficient_pairing"].sum()) if not h3_confirm.empty else 0
    bridge_groups = bridge_specificity[~bridge_specificity["insufficient_pairing"]].drop_duplicates(["model", "task_class", "target", "metric"])
    bridge_h3_rate = float(bridge_groups["h3_group_pass"].mean()) if not bridge_groups.empty else float("nan")

    transition_rows: list[dict[str, Any]] = []
    if "span" in eff.columns:
        span_panel = eff[(eff["split"] == "span_bridge") & (eff["span"].isin([32, 64, 96]))].copy()
    else:
        span_panel = eff.iloc[0:0].copy()
        span_panel["span"] = pd.Series(dtype="float64")
    for (model, span), sub in span_panel.groupby(["model", "span"]):
        sgrp = sub.groupby(["seed", "intervention", "task"], as_index=False)["drop_acc"].mean()
        piv = sgrp.pivot_table(index="seed", columns=["intervention", "task"], values="drop_acc").reset_index()
        high_vals = (
            piv[("ablate_high_strong", "copy_offset_bridge")] - piv[("ablate_high_strong", "retrieval_bridge")]
            if ("ablate_high_strong", "copy_offset_bridge") in piv.columns and ("ablate_high_strong", "retrieval_bridge") in piv.columns
            else pd.Series(dtype=float)
        )
        low_vals = (
            piv[("ablate_low_strong", "retrieval_bridge")] - piv[("ablate_low_strong", "copy_offset_bridge")]
            if ("ablate_low_strong", "copy_offset_bridge") in piv.columns and ("ablate_low_strong", "retrieval_bridge") in piv.columns
            else pd.Series(dtype=float)
        )
        rand_vals = (
            piv[("random_strong", "copy_offset_bridge")] - piv[("random_strong", "retrieval_bridge")]
            if ("random_strong", "copy_offset_bridge") in piv.columns and ("random_strong", "retrieval_bridge") in piv.columns
            else pd.Series(dtype=float)
        )
        high_arr = high_vals.to_numpy(dtype=np.float64)
        high_arr = high_arr[np.isfinite(high_arr)]
        low_arr = low_vals.to_numpy(dtype=np.float64)
        low_arr = low_arr[np.isfinite(low_arr)]
        rand_arr = rand_vals.to_numpy(dtype=np.float64)
        rand_arr = rand_arr[np.isfinite(rand_arr)]
        high_ci = bootstrap_bca_ci(high_arr) if high_arr.size else (float("nan"), float("nan"))
        low_ci = bootstrap_bca_ci(low_arr) if low_arr.size else (float("nan"), float("nan"))
        rand_ci = bootstrap_bca_ci(rand_arr) if rand_arr.size else (float("nan"), float("nan"))
        high_effect = float(np.mean(high_arr)) if high_arr.size else float("nan")
        low_effect = float(np.mean(low_arr)) if low_arr.size else float("nan")
        rand_effect = float(np.mean(rand_arr)) if rand_arr.size else float("nan")
        available_counts = [int(x.size) for x in (high_arr, low_arr, rand_arr) if x.size]
        seed_count = min(available_counts) if available_counts else 0
        transition_rows.append(
            {
                "model": str(model),
                "span": int(span),
                "high_target_minus_retrieval": high_effect,
                "high_target_minus_retrieval_ci": [high_ci[0], high_ci[1]],
                "high_target_minus_retrieval_p": paired_sign_flip_pvalue(high_arr) if high_arr.size else float("nan"),
                "low_retrieval_minus_target": low_effect,
                "low_retrieval_minus_target_ci": [low_ci[0], low_ci[1]],
                "low_retrieval_minus_target_p": paired_sign_flip_pvalue(low_arr) if low_arr.size else float("nan"),
                "random_copy_minus_retrieval": rand_effect,
                "random_copy_minus_retrieval_ci": [rand_ci[0], rand_ci[1]],
                "random_copy_minus_retrieval_p": paired_sign_flip_pvalue(rand_arr) if rand_arr.size else float("nan"),
                "seed_count": int(seed_count),
            }
        )

    h5_support_models = 0
    h5_directional_models = 0
    h5_rows: list[dict[str, Any]] = []
    tier1_files = list(phase_root.glob("**/tier1_stratified_metrics.parquet"))
    if tier1_files:
        tier = pd.concat([pd.read_parquet(p) for p in tier1_files], ignore_index=True)
        for model in sorted(tier["model"].dropna().unique()):
            info = _h5_model_support(tier, model)
            h5_rows.append(info)
            if info["supported"]:
                h5_support_models += 1
            if info.get("directionally_coherent"):
                h5_directional_models += 1
    h5_ok = h5_support_models >= 1

    def _worst_model_power(name: str) -> dict[str, Any]:
        mdes: list[float] = []
        ns: list[int] = []
        for model_name, blocks in per_model_power.items():
            block = dict(blocks.get(name, {}))
            mde = block.get("mde_at_80pct_power")
            n = block.get("n")
            if isinstance(n, int):
                ns.append(n)
            if isinstance(mde, float) and np.isfinite(mde):
                mdes.append(float(mde))
        worst = max(mdes) if mdes else float("nan")
        return {
            "worst_model_mde": worst,
            "target_threshold": 0.05,
            "underpowered": bool(np.isfinite(worst) and worst > 0.05),
            "min_seed_n_across_models": min(ns) if ns else 0,
        }

    power = {
        "per_model": per_model_power,
        "H1": _worst_model_power("H1"),
        "H2": _worst_model_power("H2"),
    }
    restricted_rank = _restricted_rank_diagnostic(task_eff)
    ci_sensitivity = {
        "H1": bootstrap_bca_ci_sensitivity(h1_arr),
        "H2": bootstrap_bca_ci_sensitivity(h2_arr),
    }
    h1_base = _class_contrast(eff, "ablate_high_strong", "h1", value_col="drop_acc_over_baseline")
    h2_base = _class_contrast(eff, "ablate_low_strong", "h2", value_col="drop_acc_over_baseline")
    h1_headroom = _class_contrast(eff, "ablate_high_strong", "h1", value_col="drop_acc_over_headroom")
    h2_headroom = _class_contrast(eff, "ablate_low_strong", "h2", value_col="drop_acc_over_headroom")
    base_diag = _contrast_inference_block(
        h1_base["contrast"].to_numpy(dtype=np.float64),
        h2_base["contrast"].to_numpy(dtype=np.float64),
        threshold=0.05,
    )
    headroom_diag = _contrast_inference_block(
        h1_headroom["contrast"].to_numpy(dtype=np.float64),
        h2_headroom["contrast"].to_numpy(dtype=np.float64),
        threshold=0.05,
    )
    raw_dirs = {"h1": _effect_direction(h1_effect), "h2": _effect_direction(h2_effect)}
    for block in (base_diag, headroom_diag):
        for hyp in ("h1", "h2"):
            block[hyp]["raw_direction"] = raw_dirs[hyp]
            block[hyp]["raw_direction_agree"] = bool(block[hyp]["direction"] == raw_dirs[hyp])
    agree_base = bool(base_diag["h1"]["raw_direction_agree"] and base_diag["h2"]["raw_direction_agree"])
    agree_headroom = bool(headroom_diag["h1"]["raw_direction_agree"] and headroom_diag["h2"]["raw_direction_agree"])
    class_headroom = _task_class_headroom_summary(eff)
    overlap_diag = _long_offset_overlap_diagnostic()
    severe_imbalance = bool(class_headroom.get("severe_imbalance", False))
    headroom_confound_risk = bool((not agree_base) or (not agree_headroom) or severe_imbalance)
    headroom_diag_payload = {
        "advisory_only": True,
        "policy": "raw_primary_non_gating",
        "note": "Headroom-normalized contrasts are diagnostic only; confirmatory criteria remain based on raw drops.",
        "raw_reference": {
            "h1": {"effect": h1_effect, "direction": raw_dirs["h1"], "n": int(h1_arr.size)},
            "h2": {"effect": h2_effect, "direction": raw_dirs["h2"], "n": int(h2_arr.size)},
        },
        "baseline_normalized": base_diag,
        "headroom_normalized": headroom_diag,
        "raw_vs_baseline_direction_agreement_all": agree_base,
        "raw_vs_headroom_direction_agreement_all": agree_headroom,
    }
    endpoint_adjudication = _h12_endpoint_adjudication(
        endpoint_policy=endpoint_policy,
        raw_h1_label=h1_label,
        raw_h2_label=h2_label,
        headroom_diag_payload=headroom_diag_payload,
    )
    seed_corr = _seed_correlation(task_eff)
    overlap_cols = {"overlap_high_jaccard", "overlap_low_jaccard", "overlap_high_fraction", "overlap_low_fraction"}
    if overlap_cols.issubset(set(eff.columns)):
        rand = eff[eff["intervention"].isin(["random_medium", "random_strong"])].copy()
        if rand.empty:
            random_overlap = {
                "rows": 0,
                "random_overlap_sensitive": False,
            }
        else:
            rand["max_overlap_jaccard"] = rand[["overlap_high_jaccard", "overlap_low_jaccard"]].max(axis=1)
            high_jacc = float((rand["max_overlap_jaccard"] >= 0.70).mean())
            random_overlap = {
                "rows": int(len(rand)),
                "mean_overlap_high_fraction": float(rand["overlap_high_fraction"].mean()),
                "mean_overlap_low_fraction": float(rand["overlap_low_fraction"].mean()),
                "mean_max_overlap_jaccard": float(rand["max_overlap_jaccard"].mean()),
                "frac_rows_jaccard_ge_0_70": high_jacc,
                "random_overlap_sensitive": bool(high_jacc > 0.10),
            }
    else:
        random_overlap = {
            "rows": 0,
            "random_overlap_sensitive": False,
            "note": "Overlap diagnostics unavailable in this run schema.",
        }
    task_family_difficulty_diagnostic = {
        "short_high_task_gap_max": short_gap_max,
        "long_low_task_gap_max": long_gap_max,
        "gap_threshold": 0.05,
        "task_family_gap_flag": task_family_gap_flag,
        "advisory_only": True,
        "recommended_action": "If flagged, report per-task class contrasts and avoid over-interpreting class-average magnitude.",
        "note": "Large within-class task-gap suggests heterogeneous restricted-candidate difficulty across tasks.",
    }

    criteria = {
        "replication_2_of_3": bool(rep_ok),
        "cross_task_agreement_2_of_3": bool(cross_ok),
        "content_gated_safeguard_2_of_3": bool(content_ok),
        "specificity_advantage_2_of_3": bool(spec_ok),
        "kernel_specificity_50pct": bool(h3_ok),
        "transfer_1_of_3": bool(h5_ok),
        "pooled_primary_inference_holm_0_05": bool(primary_sig),
    }
    confirmatory_success = all(
        criteria[k]
        for k in [
            "replication_2_of_3",
            "cross_task_agreement_2_of_3",
            "content_gated_safeguard_2_of_3",
            "specificity_advantage_2_of_3",
            "kernel_specificity_50pct",
            "pooled_primary_inference_holm_0_05",
        ]
    )

    return {
        "phase": "phase2b",
        "eligible_rows": int(len(eff)),
        "excluded_floor_rows": int(len(task_eff) - len(eff)),
        "criteria": criteria,
        "pooled_inference_assumption": "fixed_effects_model_seed_exchangeable",
        "pooled_inference_assumption_details": {
            "assumption": "fixed_effects_model_seed_exchangeable",
            "implication": "Model-seed contrasts are pooled for primary inference.",
            "limitation": "Within-model dependence may make pooled p-values anti-conservative.",
            "mitigation": "Model-cluster sensitivity block is reported as a robustness diagnostic.",
        },
        "h3_gate_policy": "point_estimate_confirmatory_ci_diagnostic",
        "cluster_sensitivity_flag": bool(cluster_sensitivity_flag),
        "cluster_sensitivity_downgrade_note": cluster_sensitivity_note,
        "confirmatory_success": bool(confirmatory_success),
        "advance_to_phase2c": bool(confirmatory_success),
        "primary_fixed_effects": {
            "unit": "model_seed_contrast",
            "family_alpha": 0.05,
            "h1": {
                "effect": h1_effect,
                "ci": [h1_ci[0], h1_ci[1]],
                "p": p_raw["H1"],
                "p_holm": p_adj["H1"],
                "label": h1_label,
                "n": int(h1_arr.size),
                "effect_size_d": h1_d,
            },
            "h2": {
                "effect": h2_effect,
                "ci": [h2_ci[0], h2_ci[1]],
                "p": p_raw["H2"],
                "p_holm": p_adj["H2"],
                "label": h2_label,
                "n": int(h2_arr.size),
                "effect_size_d": h2_d,
            },
            "pooled_primary_inference_holm_0_05": bool(primary_sig),
        },
        "cluster_sensitivity": {
            "unit": "model_mean_contrast",
            "family_alpha": 0.05,
            "evaluable": bool(cluster_evaluable),
            "min_exact_p_h1": min_exact_p_h1,
            "min_exact_p_h2": min_exact_p_h2,
            "h1": {
                "effect": h1_model_effect,
                "ci": [h1_model_ci[0], h1_model_ci[1]],
                "p": p_cluster_raw["H1"],
                "p_holm": p_cluster_adj["H1"],
                "label": h1_model_label,
                "n_models": int(h1_model_arr.size),
            },
            "h2": {
                "effect": h2_model_effect,
                "ci": [h2_model_ci[0], h2_model_ci[1]],
                "p": p_cluster_raw["H2"],
                "p_holm": p_cluster_adj["H2"],
                "label": h2_model_label,
                "n_models": int(h2_model_arr.size),
            },
            "cluster_sensitivity_flag": bool(cluster_sensitivity_flag),
            "cluster_sensitivity_note": cluster_sensitivity_note,
            "agrees_with_primary": (None if not cluster_evaluable else bool(not cluster_sensitivity_flag)),
        },
        "h1": {
            "effect": h1_effect,
            "ci": [h1_ci[0], h1_ci[1]],
            "p": p_raw["H1"],
            "p_holm": p_adj["H1"],
            "label": h1_label,
            "n_pooled": int(h1_arr.size),
            "effect_size_d": h1_d,
        },
        "h2": {
            "effect": h2_effect,
            "ci": [h2_ci[0], h2_ci[1]],
            "p": p_raw["H2"],
            "p_holm": p_adj["H2"],
            "label": h2_label,
            "n_pooled": int(h2_arr.size),
            "effect_size_d": h2_d,
        },
        "h3_rate_point": h3_rate_point,
        "h3_rate_ci": h3_rate_ci,
        "h3_ci_gated_pass_rate": h3_rate_ci,
        "h3_ci_gated_pass_rate_interpretation": (
            "pass rate using CI-gated group criterion (not CI of pass rate)"
        ),
        "h3_ci_diagnostic_flag": h3_ci_diagnostic_flag,
        "h3_pass_rate": h3_rate_point,
        "h3_pass_rate_ci": h3_rate_ci,
        "h3_rows_excluded_insufficient_pairing": h3_insufficient,
        "h3_bridge_pass_rate": bridge_h3_rate,
        "h3_diff_test": {
            "threshold": 0.0,
            "interpretation": "sign test on targeted-random difference",
        },
        "active_long_tasks": [str(t) for t in active_long_tasks],
        "cross_task_sign_policy": cross_task_sign_policy,
        "cross_task_sign_rule_version": cross_task_sign_rule_version,
        "cluster_h1_evaluable": bool(cluster_h1_evaluable),
        "cluster_h2_evaluable": bool(cluster_h2_evaluable),
        "cluster_any_evaluable": bool(cluster_any_evaluable),
        "cluster_evaluable_policy": "any_hypothesis_evaluable_or",
        "cluster_output_scope_note": (
            "non-evaluable hypothesis p-values are reported for completeness, not confirmatory decisioning"
        ),
        "h3_random_draw_handling": {
            "confirmatory_mode": "seed_mean",
            "diagnostic_mode": "per_draw",
            "note": "Confirmatory H3 compares targeted vs expected random baseline (mean across random draws per seed). Per-draw diagnostics are reported separately.",
        },
        "h3_confirmatory_track_b_raw": {
            "metric": "track_b_raw",
            "rows_total": int(len(h3_confirm)),
            "rows_evaluable": int(len(h3_groups)),
            "pass_rate_point": h3_rate_point,
            "pass_rate": h3_rate_point,
            "pass_rate_ci": h3_rate_ci,
            "ci_gated_pass_rate": h3_rate_ci,
            "ci_gated_pass_rate_interpretation": (
                "pass rate using CI-gated group criterion (not CI of pass rate)"
            ),
        },
        "h3_confirmatory_track_b_raw_per_draw_diagnostic": {
            "metric": "track_b_raw",
            "rows_total": int(len(h3_confirm_per_draw)),
            "rows_evaluable": int(
                len(
                    h3_confirm_per_draw[~h3_confirm_per_draw["insufficient_pairing"]]
                    .drop_duplicates(["model", "task_class", "target", "metric"])
                )
            )
            if not h3_confirm_per_draw.empty
            else 0,
            "pass_rate": float(
                h3_confirm_per_draw[~h3_confirm_per_draw["insufficient_pairing"]]
                .drop_duplicates(["model", "task_class", "target", "metric"])["h3_group_pass"]
                .mean()
            )
            if not h3_confirm_per_draw.empty
            else float("nan"),
            "paired_pair_count_mean": float(h3_confirm_per_draw["paired_pair_count"].mean())
            if not h3_confirm_per_draw.empty
            else float("nan"),
        },
        "h3_aux_track_a": {
            "metric": "track_a",
            "rows_total": int(len(h3_track_a)),
            "rows_evaluable": int(
                len(h3_track_a[~h3_track_a["insufficient_pairing"]].drop_duplicates(["model", "task_class", "target", "metric"]))
            )
            if not h3_track_a.empty
            else 0,
            "pass_rate": float(
                h3_track_a[~h3_track_a["insufficient_pairing"]]
                .drop_duplicates(["model", "task_class", "target", "metric"])["h3_group_pass"]
                .mean()
            )
            if not h3_track_a.empty
            else float("nan"),
        },
        "h3_diagnostic_centered": {
            "metric": "track_b_centered",
            "rows_total": int(len(h3_centered)),
            "rows_evaluable": int(
                len(h3_centered[~h3_centered["insufficient_pairing"]].drop_duplicates(["model", "task_class", "target", "metric"]))
            )
            if not h3_centered.empty
            else 0,
            "pass_rate": float(
                h3_centered[~h3_centered["insufficient_pairing"]]
                .drop_duplicates(["model", "task_class", "target", "metric"])["h3_group_pass"]
                .mean()
            )
            if not h3_centered.empty
            else float("nan"),
        },
        "transition_panel": {
            "spans": [32, 64, 96],
            "secondary_confirmatory_channel": True,
            "description": "Medium-range span bridge sensitivity panel; does not override primary H1/H2 labels.",
            "rows": transition_rows,
        },
        "h5_support_models": h5_support_models,
        "h5_directional_coherence_models": h5_directional_models,
        "h5_model_rows": h5_rows,
        "model_rows": model_rows,
        "power_detectability": power,
        "restricted_rank_diagnostic": restricted_rank,
        "ci_seed_sensitivity": ci_sensitivity,
        "headroom_normalized_diagnostic": headroom_diag_payload,
        "h12_endpoint_policy": _normalize_h12_endpoint_policy(endpoint_policy),
        "h12_endpoint_adjudication": endpoint_adjudication,
        "task_class_headroom_summary": class_headroom,
        "headroom_confound_risk": headroom_confound_risk,
        "normalized_contrast_policy": "advisory_non_gating_raw_primary",
        "span_overlap_diagnostic": overlap_diag,
        "seed_correlation": seed_corr,
        "random_overlap": random_overlap,
        "task_family_difficulty_diagnostic": task_family_difficulty_diagnostic,
    }


def _phase2b_feasibility_subset(
    task_eff: pd.DataFrame,
    kernel_delta: pd.DataFrame,
    *,
    subset_models: list[str],
    rationale: str,
) -> dict[str, Any]:
    del kernel_delta  # Included for interface symmetry/future extension.
    eff = _eligible_task_rows(task_eff)
    eff = eff[eff["model"].isin(subset_models)].copy()
    if eff.empty:
        return {
            "exploratory_only": True,
            "subset_models": [str(m) for m in subset_models],
            "subset_rationale": rationale,
            "status": "no_data",
            "note": "No eligible rows for feasibility-conditioned subset.",
        }

    h1 = _class_contrast(eff, "ablate_high_strong", "h1")
    h2 = _class_contrast(eff, "ablate_low_strong", "h2")
    h1_arr = h1["contrast"].to_numpy(dtype=np.float64)
    h2_arr = h2["contrast"].to_numpy(dtype=np.float64)
    h1_arr = h1_arr[np.isfinite(h1_arr)]
    h2_arr = h2_arr[np.isfinite(h2_arr)]
    h1_ci = bootstrap_bca_ci(h1_arr) if h1_arr.size else (float("nan"), float("nan"))
    h2_ci = bootstrap_bca_ci(h2_arr) if h2_arr.size else (float("nan"), float("nan"))
    h1_effect = float(np.mean(h1_arr)) if h1_arr.size else float("nan")
    h2_effect = float(np.mean(h2_arr)) if h2_arr.size else float("nan")
    h1_p = paired_sign_flip_pvalue(h1_arr) if h1_arr.size else float("nan")
    h2_p = paired_sign_flip_pvalue(h2_arr) if h2_arr.size else float("nan")
    h1_label = precision_label(effect=h1_effect, ci_low=h1_ci[0], threshold=0.05)
    h2_label = precision_label(effect=h2_effect, ci_low=h2_ci[0], threshold=0.05)

    h1_headroom = _class_contrast(eff, "ablate_high_strong", "h1", value_col="drop_acc_over_headroom")
    h2_headroom = _class_contrast(eff, "ablate_low_strong", "h2", value_col="drop_acc_over_headroom")
    h1_headroom_arr = h1_headroom["contrast"].to_numpy(dtype=np.float64)
    h2_headroom_arr = h2_headroom["contrast"].to_numpy(dtype=np.float64)
    h1_headroom_arr = h1_headroom_arr[np.isfinite(h1_headroom_arr)]
    h2_headroom_arr = h2_headroom_arr[np.isfinite(h2_headroom_arr)]
    h1_headroom_effect = float(np.mean(h1_headroom_arr)) if h1_headroom_arr.size else float("nan")
    h2_headroom_effect = float(np.mean(h2_headroom_arr)) if h2_headroom_arr.size else float("nan")

    per_model_effects: list[dict[str, Any]] = []
    for model in sorted(eff["model"].dropna().unique()):
        m = eff[eff["model"] == model]
        mh1 = _class_contrast(m, "ablate_high_strong", "h1")
        mh2 = _class_contrast(m, "ablate_low_strong", "h2")
        mh1_arr = mh1["contrast"].to_numpy(dtype=np.float64)
        mh2_arr = mh2["contrast"].to_numpy(dtype=np.float64)
        mh1_arr = mh1_arr[np.isfinite(mh1_arr)]
        mh2_arr = mh2_arr[np.isfinite(mh2_arr)]
        mh1_effect = float(np.mean(mh1_arr)) if mh1_arr.size else float("nan")
        mh2_effect = float(np.mean(mh2_arr)) if mh2_arr.size else float("nan")
        per_model_effects.append(
            {
                "model": str(model),
                "h1_effect": mh1_effect,
                "h2_effect": mh2_effect,
                "h1_direction": _effect_direction(mh1_effect),
                "h2_direction": _effect_direction(mh2_effect),
                "h1_n": int(mh1_arr.size),
                "h2_n": int(mh2_arr.size),
            }
        )

    return {
        "exploratory_only": True,
        "subset_models": [str(m) for m in subset_models],
        "subset_rationale": rationale,
        "status": "ok",
        "note": (
            "Feasibility-conditioned subset is exploratory. Primary confirmatory inference remains all-model pooled."
        ),
        "h1": {
            "effect": h1_effect,
            "ci": [h1_ci[0], h1_ci[1]],
            "p": float(h1_p),
            "label": h1_label,
            "direction": _effect_direction(h1_effect),
            "n": int(h1_arr.size),
        },
        "h2": {
            "effect": h2_effect,
            "ci": [h2_ci[0], h2_ci[1]],
            "p": float(h2_p),
            "label": h2_label,
            "direction": _effect_direction(h2_effect),
            "n": int(h2_arr.size),
        },
        "headroom_normalized": {
            "h1_effect": h1_headroom_effect,
            "h2_effect": h2_headroom_effect,
            "h1_direction": _effect_direction(h1_headroom_effect),
            "h2_direction": _effect_direction(h2_headroom_effect),
            "h1_n": int(h1_headroom_arr.size),
            "h2_n": int(h2_headroom_arr.size),
        },
        "n_contrasts": {
            "h1": int(h1_arr.size),
            "h2": int(h2_arr.size),
        },
        "per_model_effects": per_model_effects,
    }


def evaluate_phase(phase_root: Path) -> PhaseAnalysis:
    task_df = _load_parquet(phase_root / "aggregate_task_metrics.parquet")
    kernel_df = _load_parquet(phase_root / "aggregate_kernel_metrics.parquet")
    task_eff = _task_effects(task_df)
    kernel_delta = _kernel_deltas(kernel_df)
    task_eff_eligible = _eligible_task_rows(task_eff)
    kernel_delta_eligible = _filter_kernel_to_eligible(kernel_delta, task_eff_eligible)
    if "model" in task_eff_eligible.columns and not task_eff_eligible.empty:
        _observed_model_values = task_eff_eligible["model"].dropna().unique().tolist()
    elif "model" in task_eff.columns and not task_eff.empty:
        _observed_model_values = task_eff["model"].dropna().unique().tolist()
    else:
        _observed_model_values = []
    observed_models = sorted(str(m) for m in _observed_model_values)
    phase = _resolve_phase_name(phase_root)
    endpoint_policy_meta = _resolve_h12_endpoint_policy(phase_root)
    endpoint_policy = str(endpoint_policy_meta.get("h12_endpoint_policy", "raw_primary"))
    allowed_classes = {"short", "long"} if phase in {"phase2a", "phase2b"} else None
    allowed_metrics = {"track_b_raw"} if phase in {"phase2a", "phase2b"} else None
    specificity = _h3_specificity(
        kernel_delta_eligible,
        allowed_task_classes=allowed_classes,
        allowed_metrics=allowed_metrics,
    )
    if phase == "phase2a":
        gate = _phase2a_gate(task_eff, kernel_delta, endpoint_policy=endpoint_policy)
    elif phase == "phase2b":
        gate = _phase2b_gate(task_eff, kernel_delta, phase_root, endpoint_policy=endpoint_policy)
        subset_models = ["llama-3.2-1b", "olmo-1b"]
        subset_rationale = "feasibility_lock_delayed_copy_floor_exclusion"
        if all(name in observed_models for name in ["llama-3.1-8b", "olmo-2-7b"]):
            subset_models = ["llama-3.1-8b", "olmo-2-7b"]
            subset_rationale = "scaleup_feasibility_conditioned_subset"
        gate["feasibility_conditioned_subset"] = _phase2b_feasibility_subset(
            task_eff,
            kernel_delta,
            subset_models=subset_models,
            rationale=subset_rationale,
        )
    else:
        gate = {
            "phase": phase,
            "status": "descriptive_only",
            "note": "No confirmatory phase gate for this phase.",
            "h12_endpoint_policy": _normalize_h12_endpoint_policy(endpoint_policy),
        }
    gate.setdefault("h12_endpoint_policy", _normalize_h12_endpoint_policy(endpoint_policy))
    gate.setdefault(
        "h12_endpoint_policy_observed_counts",
        endpoint_policy_meta.get("h12_endpoint_policy_observed_counts", {}),
    )
    gate.setdefault("h12_endpoint_policy_mixed", bool(endpoint_policy_meta.get("h12_endpoint_policy_mixed", False)))

    phase2b_status = _phase2b_confirmatory_status(phase_root)
    if phase == "phase2b" and isinstance(gate.get("confirmatory_success"), bool):
        phase2b_observed: bool | str = bool(gate.get("confirmatory_success"))
    else:
        phase2b_observed = bool(phase2b_status) if phase2b_status is not None else "unknown"
    min_seed_count = _min_seed_count(task_eff)
    headline_7seed_ready = bool(min_seed_count >= 7) if phase in {"phase2c", "phase2d"} else True
    if phase in {"phase2c", "phase2d"}:
        promotion_eligible = bool((phase2b_status is True) and headline_7seed_ready)
    elif phase == "phase2b":
        promotion_eligible = bool(gate.get("confirmatory_success", False))
    else:
        promotion_eligible = False
    confirmatory_applicability = True
    exploratory_reason = None
    if phase == "phase2b" and len(observed_models) == 1:
        confirmatory_applicability = False
        exploratory_reason = "single_model_design"
        promotion_eligible = False
    gate["advancement_policy"] = "analysis_only"
    gate["phase2b_confirmatory_pass_required_for_promotion"] = True
    gate["phase2b_confirmatory_pass_observed"] = phase2b_observed
    gate["headline_claim_requires_7_seed_rerun"] = bool(phase in {"phase2c", "phase2d"})
    gate["headline_7_seed_ready"] = bool(headline_7seed_ready)
    gate["min_seed_count_observed"] = int(min_seed_count)
    gate["confirmatory_promotion_eligible"] = bool(promotion_eligible)
    gate["observed_models"] = [str(m) for m in observed_models]
    gate["observed_model_count"] = int(len(observed_models))
    gate["confirmatory_applicability"] = bool(confirmatory_applicability)
    gate["exploratory_reason"] = exploratory_reason

    if phase == "phase2b" and gate.get("model_rows"):
        h1_sel = {}
        h2_sel = {}
        for row in gate["model_rows"]:
            h1_sel[row["model"]] = row.get("h1_selectivity_label", "no_directional_support")
            h2_sel[row["model"]] = row.get("h2_selectivity_label", "no_directional_support")
        h1_major = max(set(h1_sel.values()), key=list(h1_sel.values()).count)
        h2_major = max(set(h2_sel.values()), key=list(h2_sel.values()).count)
    else:
        h1_major = gate.get("h1", {}).get("selectivity_label", "descriptive")
        h2_major = gate.get("h2", {}).get("selectivity_label", "descriptive")

    if phase == "phase2b":
        h3_block = gate.get("h3_confirmatory_track_b_raw", {})
        h3_val = h3_block.get("pass_rate_point", h3_block.get("pass_rate", 0.0))
        h3_pass = bool(np.isfinite(h3_val) and h3_val >= 0.5)
    else:
        h3_pass = bool((not specificity.empty) and (specificity["h3_group_pass"].mean() >= 0.5))

    decision_payload = {
        "phase": phase,
        "run_id": phase_root.name,
        "training_variance_not_measured": True,
        "inference_scope": "task_sampling_variability_only",
        "training_run_generalization_claim_allowed": False,
        "H4_status": "exploratory_interaction_reported",
        "H4_artifact": "h4_interaction_exploratory.parquet",
        "advancement_policy": "analysis_only",
        "phase2b_confirmatory_pass_required_for_promotion": True,
        "phase2b_confirmatory_pass_observed": phase2b_observed,
        "headline_claim_requires_7_seed_rerun": bool(phase in {"phase2c", "phase2d"}),
        "headline_7_seed_ready": bool(headline_7seed_ready),
        "min_seed_count_observed": int(min_seed_count),
        "confirmatory_promotion_eligible": bool(promotion_eligible),
        "confirmatory_applicability": bool(confirmatory_applicability),
        "exploratory_reason": exploratory_reason,
        "observed_models": [str(m) for m in observed_models],
        "observed_model_count": int(len(observed_models)),
        "h12_endpoint_policy": str(gate.get("h12_endpoint_policy", "raw_primary")),
        "h12_endpoint_policy_observed_counts": gate.get("h12_endpoint_policy_observed_counts", {}),
        "h12_endpoint_policy_mixed": bool(gate.get("h12_endpoint_policy_mixed", False)),
        "h12_endpoint_adjudication": gate.get("h12_endpoint_adjudication", {}),
        "reporting_interpretation_notes": [
            "h3_ci_gated_pass_rate is a pass rate computed with CI-gated group criteria; it is not a confidence interval over the pass rate itself.",
            "Cluster sensitivity uses any-hypothesis evaluability (OR); non-evaluable hypothesis p-values are retained for completeness, not confirmatory decisioning.",
            "Headroom-normalized H1/H2 diagnostics are advisory only; confirmatory inference remains raw-drop primary (see gate.headroom_normalized_diagnostic).",
        ],
        "hypotheses": {
            "H1": gate.get("h1", {}).get("label", "descriptive"),
            "H2": gate.get("h2", {}).get("label", "descriptive"),
            "H1_selectivity": h1_major,
            "H2_selectivity": h2_major,
            "H3": "pass" if h3_pass else "fail",
            "H3_co_occurrence_specificity": "pass" if h3_pass else "fail",
            "H4": "exploratory",
            "H5": "exploratory",
        },
        "power_detectability": gate.get("power_detectability", {}),
        "seed_correlation": gate.get("seed_correlation", {}),
        "gate": gate,
    }
    if phase == "phase2b":
        decision_payload["feasibility_conditioned_subset"] = gate.get("feasibility_conditioned_subset", {})
    if phase == "phase2b" and not bool(confirmatory_applicability):
        decision_payload["reporting_interpretation_notes"].append(
            "Phase 2B run is exploratory-only due to single-model design; confirmatory promotion is not applicable."
        )
    overlap_diag = gate.get("span_overlap_diagnostic", {})
    if str(decision_payload.get("h12_endpoint_policy")) == "co_primary_raw_headroom":
        decision_payload["reporting_interpretation_notes"].append(
            "Scale-up endpoint policy records co-primary raw+headroom adjudication in gate.h12_endpoint_adjudication."
        )
    if isinstance(overlap_diag, dict) and overlap_diag.get("overlaps_short_regime") is True:
        decision_payload["reporting_interpretation_notes"].append(
            "Locked long offsets overlap the short regime; short-vs-long contrasts should be interpreted as mixed task-family+span evidence."
        )

    return PhaseAnalysis(gate_payload=gate, specificity_df=specificity, decision_payload=decision_payload)


def write_phase_analysis(phase_root: Path) -> PhaseAnalysis:
    analysis = evaluate_phase(phase_root)
    analysis.specificity_df.to_parquet(phase_root / "specificity_metrics.parquet", engine="pyarrow", index=False)
    (phase_root / "gate_evaluation.json").write_text(json.dumps(analysis.gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (phase_root / "decision_summary.json").write_text(json.dumps(analysis.decision_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return analysis
