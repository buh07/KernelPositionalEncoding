from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class Tier1BinSummary:
    separation_gap: float
    interpretable: bool
    window_agreement: float
    robustness_pass: bool
    num_rows: int


def _window_agreement_from_scores(df: pd.DataFrame) -> tuple[float, bool]:
    if not {"local_score_w8", "local_score_w16", "local_score_w32"}.issubset(set(df.columns)):
        return float("nan"), True
    dep_cols: dict[str, pd.Series] = {}
    for window, col in (("w8", "local_score_w8"), ("w16", "local_score_w16"), ("w32", "local_score_w32")):
        q25 = float(df[col].quantile(0.25))
        q75 = float(df[col].quantile(0.75))
        dep = pd.Series(["middle"] * len(df), index=df.index, dtype="object")
        dep[df[col] <= q25] = "distal"
        dep[df[col] >= q75] = "local"
        dep_cols[window] = dep
    ref = dep_cols["w16"]
    mask = ref.isin(["local", "distal"]) & dep_cols["w8"].isin(["local", "distal"]) & dep_cols["w32"].isin(["local", "distal"])
    if not bool(mask.any()):
        return float("nan"), False
    agree = ((dep_cols["w8"][mask] == ref[mask]) & (dep_cols["w32"][mask] == ref[mask])).mean()
    agreement = float(agree)
    return agreement, bool(np.isfinite(agreement) and agreement >= 0.70)


def summarize_frozen_bins(bins: pd.DataFrame) -> Tier1BinSummary:
    local_mean = float(bins.loc[bins["dependency_type"] == "local", "local_score"].mean())
    distal_mean = float(bins.loc[bins["dependency_type"] == "distal", "local_score"].mean())
    gap = local_mean - distal_mean
    agreement, robust_pass = _window_agreement_from_scores(bins)
    interpretable = bool(np.isfinite(gap) and gap >= 0.15 and robust_pass)
    return Tier1BinSummary(
        separation_gap=gap,
        interpretable=interpretable,
        window_agreement=agreement,
        robustness_pass=robust_pass,
        num_rows=len(bins),
    )


def local_concentration_scores(
    attention_logits: torch.Tensor,
    *,
    target_positions: list[int],
    layers: tuple[int, ...] = (0, 1),
    window: int = 16,
) -> dict[int, float]:
    if attention_logits.ndim != 4:
        raise RuntimeError(f"Expected [layers,heads,seq,seq], got {tuple(attention_logits.shape)}")
    num_layers = attention_logits.shape[0]
    use_layers = [l for l in layers if 0 <= l < num_layers]
    if not use_layers:
        use_layers = [0]
    out: dict[int, float] = {}
    for pos in target_positions:
        q_idx = pos - 1
        if q_idx < 0 or q_idx >= attention_logits.shape[2]:
            continue
        vals = []
        for l in use_layers:
            layer_logits = attention_logits[l]  # [heads, seq, seq]
            for h in range(layer_logits.shape[0]):
                row = layer_logits[h, q_idx, : q_idx + 1]
                probs = torch.softmax(row.to(torch.float64), dim=-1)
                lo = max(0, q_idx - window + 1)
                vals.append(float(probs[lo : q_idx + 1].sum().item()))
        out[pos] = float(np.mean(vals)) if vals else float("nan")
    return out


def _quartile_labels(values: pd.Series) -> pd.Series:
    labels = ["Q1", "Q2", "Q3", "Q4"]
    try:
        return pd.qcut(values, q=4, labels=labels, duplicates="drop")
    except ValueError:
        rank = values.rank(method="average", pct=True)
        return pd.cut(rank, bins=[0.0, 0.25, 0.5, 0.75, 1.0], labels=labels, include_lowest=True)


def build_frozen_bins(records: pd.DataFrame, *, out_path: Path) -> Tier1BinSummary:
    # Required columns: example_id, position, local_score, baseline_nll
    if records.empty:
        raise RuntimeError("Cannot build tier1 bins from empty baseline records")
    df = records.copy()
    q25 = float(df["local_score"].quantile(0.25))
    q75 = float(df["local_score"].quantile(0.75))
    df["dependency_type"] = "middle"
    df.loc[df["local_score"] <= q25, "dependency_type"] = "distal"
    df.loc[df["local_score"] >= q75, "dependency_type"] = "local"
    df["baseline_nll_quartile"] = _quartile_labels(df["baseline_nll"]).astype(str)

    summary = summarize_frozen_bins(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    return summary


def load_frozen_bins(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def stratified_metrics(per_position: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    # per_position columns: example_id, position, nll
    merged = per_position.merge(
        bins[["example_id", "position", "dependency_type", "baseline_nll_quartile", "baseline_nll"]],
        on=["example_id", "position"],
        how="inner",
    )
    merged = merged[merged["dependency_type"].isin(["local", "distal"])].copy()
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "dependency_type",
                "baseline_nll_quartile",
                "mean_nll",
                "baseline_mean_nll",
                "delta_nll",
                "count",
            ]
        )
    agg = (
        merged.groupby(["dependency_type", "baseline_nll_quartile"], as_index=False)
        .agg(
            mean_nll=("nll", "mean"),
            baseline_mean_nll=("baseline_nll", "mean"),
            count=("nll", "size"),
        )
        .sort_values(["dependency_type", "baseline_nll_quartile"])
    )
    agg["delta_nll"] = agg["mean_nll"] - agg["baseline_mean_nll"]
    return agg


def dependency_delta(table: pd.DataFrame) -> float:
    if table.empty:
        return float("nan")
    # Equal weighting across quartiles for each dependency type.
    local = table.loc[table["dependency_type"] == "local", "delta_nll"].mean()
    distal = table.loc[table["dependency_type"] == "distal", "delta_nll"].mean()
    return float(local - distal)
