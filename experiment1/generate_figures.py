"""Generate all Experiment 1 result figures from summary parquet files."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Model display config ──────────────────────────────────────────────
MODEL_ORDER = [
    "llama-3.2-1b",
    "olmo-1b",
    "tinyllama-1.1b",
    "gpt2-small",
    "gpt2-medium",
    "tinyllama-nope-1.1b",
]
MODEL_LABELS = {
    "llama-3.2-1b": "LLaMA-3.2-1B\n(RoPE+RMSNorm)",
    "olmo-1b": "OLMo-1B\n(RoPE+LayerNorm)",
    "tinyllama-1.1b": "TinyLlama-1.1B\n(RoPE+RMSNorm)",
    "gpt2-small": "GPT-2 Small\n(AbsPE+LayerNorm)",
    "gpt2-medium": "GPT-2 Medium\n(AbsPE+LayerNorm)",
    "tinyllama-nope-1.1b": "TinyLlama-NoPE\n(None+RMSNorm)",
}
MODEL_COLORS = {
    "llama-3.2-1b": "#2196F3",
    "olmo-1b": "#4CAF50",
    "tinyllama-1.1b": "#FF9800",
    "gpt2-small": "#9C27B0",
    "gpt2-medium": "#E91E63",
    "tinyllama-nope-1.1b": "#607D8B",
}
MODEL_SHORT = {
    "llama-3.2-1b": "LLaMA-3.2",
    "olmo-1b": "OLMo-1B",
    "tinyllama-1.1b": "TinyLlama",
    "gpt2-small": "GPT-2 S",
    "gpt2-medium": "GPT-2 M",
    "tinyllama-nope-1.1b": "NoPE",
}
DATASET_LABELS = {
    "wiki40b_en_pre2019": "Wikipedia",
    "codesearchnet_python_snapshot": "Code",
    "synthetic_random": "Random",
}
DATASET_ORDER = ["wiki40b_en_pre2019", "codesearchnet_python_snapshot", "synthetic_random"]
ROPE_MODELS = ["llama-3.2-1b", "olmo-1b", "tinyllama-1.1b"]
NON_ROPE_MODELS = ["gpt2-small", "gpt2-medium", "tinyllama-nope-1.1b"]

# ── Data loading ───────────────────────────────────────────────────────

def load_all_track_a_summaries() -> pd.DataFrame:
    frames = []
    for p in sorted(RESULTS_ROOT.glob("track_a/*/*/*/summary.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError("No Track A summary parquets found")
    return pd.concat(frames, ignore_index=True)


def load_all_track_b_summaries(group: str = "track_b") -> pd.DataFrame:
    frames = []
    for p in sorted((RESULTS_ROOT / group).glob("*/*/*/summary.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(f"No Track B summary parquets found in group={group}")
    return pd.concat(frames, ignore_index=True)


def load_all_boundary_summaries() -> pd.DataFrame:
    frames = []
    for p in sorted(RESULTS_ROOT.glob("boundary/*/*/*/summary.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError("No boundary summary parquets found")
    return pd.concat(frames, ignore_index=True)


def load_all_track_a_per_sequence() -> pd.DataFrame:
    frames = []
    for p in sorted(RESULTS_ROOT.glob("track_a/*/*/*/per_sequence.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError("No Track A per-sequence parquets found")
    return pd.concat(frames, ignore_index=True)


def load_spectral_meta(group: str) -> pd.DataFrame:
    rows = []
    for p in sorted((RESULTS_ROOT / group).glob("*/*/*/spectral_run.json")):
        with p.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "model": p.parts[-4],
                "dataset": p.parts[-3],
                "seq_len": int(p.parts[-2].split("_")[1]),
                "gate_threshold": float(payload.get("gate_threshold", np.nan)),
                "gate_value": float(payload.get("gate_value", np.nan)),
                "ran": bool(payload.get("ran", False)),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No spectral_run.json files found in group={group}")
    return pd.DataFrame(rows)


def load_exploratory_spectral_rows(group: str) -> pd.DataFrame:
    frames = []
    for p in sorted((RESULTS_ROOT / group).glob("*/*/*/spectral.parquet")):
        df = pd.read_parquet(p).copy()
        df["model"] = p.parts[-4]
        df["dataset"] = p.parts[-3]
        df["seq_len"] = int(p.parts[-2].split("_")[1])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No spectral.parquet files found in group={group}")
    return pd.concat(frames, ignore_index=True)


# ── Figure 1: R² heatmaps per model (layer x head) ───────────────────

def fig1_r2_heatmaps(track_a: pd.DataFrame) -> None:
    """One heatmap per model showing pooled R² at layer x head, for wiki/256."""
    dataset = "synthetic_random"
    seq_len = 256
    subset = track_a[(track_a["dataset"] == dataset) & (track_a["seq_len"] == seq_len)]

    models_present = [m for m in MODEL_ORDER if m in subset["model"].unique()]
    n = len(models_present)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, model in enumerate(models_present):
        ax = axes[idx]
        mdf = subset[subset["model"] == model]
        n_layers = mdf["layer"].max() + 1
        n_heads = mdf["head"].max() + 1
        grid = np.full((n_layers, n_heads), np.nan)
        for _, row in mdf.iterrows():
            grid[int(row["layer"]), int(row["head"])] = row["pooled_r2"]

        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="upper")
        ax.set_title(MODEL_SHORT[model], fontsize=12, fontweight="bold")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        # Reduce tick clutter
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Figure 1: Track A Pooled R² by Layer and Head\n(dataset={DATASET_LABELS[dataset]}, len={seq_len})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.90, 0.93])
    # Place colorbar in a dedicated axes on the right margin, outside the subplots
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.75])
    fig.colorbar(im, cax=cbar_ax, label="Pooled R²")
    fig.savefig(FIGURES_DIR / "fig1_r2_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1_r2_heatmaps.png")


# ── Figure 2: Mean R² vs layer depth per model ──────────────────────

def fig2_r2_vs_depth(track_a: pd.DataFrame, track_b: pd.DataFrame) -> None:
    """Line plot of mean R² vs layer for each model, wiki/256 and wiki/1024."""
    for seq_len in [256, 1024]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        for didx, dataset in enumerate(DATASET_ORDER):
            ax = axes[didx]
            ta_sub = track_a[(track_a["dataset"] == dataset) & (track_a["seq_len"] == seq_len)]
            tb_sub = track_b[(track_b["dataset"] == dataset) & (track_b["seq_len"] == seq_len)]

            for model in MODEL_ORDER:
                color = MODEL_COLORS[model]
                # Track A
                mdf = ta_sub[ta_sub["model"] == model]
                if mdf.empty:
                    continue
                layer_means = mdf.groupby("layer")["mean_r2"].mean()
                ax.plot(layer_means.index, layer_means.values, "-o", color=color,
                        label=MODEL_SHORT[model], markersize=3, linewidth=1.5)

            ax.set_title(DATASET_LABELS[dataset], fontsize=12)
            ax.set_xlabel("Layer")
            if didx == 0:
                ax.set_ylabel("Mean R² (Track A)")
            ax.axhline(y=0.80, color="green", linestyle="--", alpha=0.4, linewidth=1)
            ax.axhline(y=0.40, color="red", linestyle="--", alpha=0.4, linewidth=1)
            ax.set_ylim(-0.02, 1.0)
            ax.grid(alpha=0.2)

        axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        fig.suptitle(
            f"Figure 2: Track A Mean R² vs Layer Depth (len={seq_len})\n"
            "Dashed lines: strong support (0.80) and falsification (0.40) thresholds",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 0.88, 0.90])
        fig.savefig(FIGURES_DIR / f"fig2_r2_vs_depth_len{seq_len}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig2_r2_vs_depth_len{seq_len}.png")


# ── Figure 3: Model comparison bar chart (early vs overall) ──────────

def fig3_model_comparison(track_a: pd.DataFrame) -> None:
    """Grouped bar chart comparing early-layer vs overall R² across models."""
    rows = []
    for model in MODEL_ORDER:
        mdf = track_a[track_a["model"] == model]
        if mdf.empty:
            continue
        early = mdf[mdf["layer"].isin([0, 1])]["mean_r2"].mean()
        overall = mdf["mean_r2"].mean()
        rows.append({"model": model, "early_r2": early, "overall_r2": overall})

    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, df["early_r2"], width, label="Early layers (0-1)",
                   color=[MODEL_COLORS[m] for m in df["model"]], alpha=0.85, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, df["overall_r2"], width, label="All layers",
                   color=[MODEL_COLORS[m] for m in df["model"]], alpha=0.45, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Mean R² (Track A, all datasets/lengths)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in df["model"]], fontsize=9)
    ax.axhline(y=0.80, color="green", linestyle="--", alpha=0.5, label="Strong support (0.80)")
    ax.axhline(y=0.40, color="red", linestyle="--", alpha=0.5, label="Falsification (0.40)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.set_title("Figure 3: Track A R² by Model — Early Layers vs All Layers", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_model_comparison.png")


# ── Figure 4: TinyLlama vs NoPE paired control ──────────────────────

def fig4_nope_paired(track_a: pd.DataFrame) -> None:
    """Side-by-side comparison of TinyLlama vs TinyLlama-NoPE."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for didx, dataset in enumerate(DATASET_ORDER):
        ax = axes[didx]
        for model, ls, lbl in [
            ("tinyllama-1.1b", "-o", "TinyLlama (RoPE)"),
            ("tinyllama-nope-1.1b", "-s", "TinyLlama (NoPE)"),
        ]:
            for seq_len, alpha in [(256, 0.6), (1024, 1.0)]:
                mdf = track_a[
                    (track_a["model"] == model) &
                    (track_a["dataset"] == dataset) &
                    (track_a["seq_len"] == seq_len)
                ]
                if mdf.empty:
                    continue
                layer_means = mdf.groupby("layer")["mean_r2"].mean()
                ax.plot(
                    layer_means.index, layer_means.values, ls,
                    color=MODEL_COLORS[model], alpha=alpha, markersize=3,
                    linewidth=1.5, label=f"{lbl} (len={seq_len})",
                )
        ax.set_title(DATASET_LABELS[dataset], fontsize=11)
        ax.set_xlabel("Layer")
        if didx == 0:
            ax.set_ylabel("Mean R² (Track A)")
        ax.set_ylim(-0.02, 0.85)
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Figure 4: TinyLlama RoPE vs NoPE Paired Control\n"
        "Architecture-matched comparison isolating the effect of RoPE",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIGURES_DIR / "fig4_nope_paired.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4_nope_paired.png")


# ── Figure 5: Track A vs Track B comparison ──────────────────────────

def fig5_track_a_vs_b(track_a: pd.DataFrame, track_b: pd.DataFrame) -> None:
    """Scatter plot comparing Track A mean R² vs Track B raw R² per (model, dataset, len, layer, head)."""
    # Merge on layer/head for each combo
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    datasets = DATASET_ORDER

    for didx, dataset in enumerate(datasets):
        ax = axes[didx]
        for model in MODEL_ORDER:
            ta = track_a[
                (track_a["model"] == model) & (track_a["dataset"] == dataset)
            ][["layer", "head", "seq_len", "mean_r2"]].copy()
            tb = track_b[
                (track_b["model"] == model) & (track_b["dataset"] == dataset)
            ][["layer", "head", "seq_len", "r2_raw"]].copy()
            if ta.empty or tb.empty:
                continue
            merged = ta.merge(tb, on=["layer", "head", "seq_len"])
            if merged.empty:
                continue
            ax.scatter(
                merged["mean_r2"], merged["r2_raw"],
                color=MODEL_COLORS[model], alpha=0.15, s=8,
                label=MODEL_SHORT[model] if didx == 0 else None,
            )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("Track A Mean R²")
        if didx == 0:
            ax.set_ylabel("Track B Raw R²")
        ax.set_title(DATASET_LABELS[dataset], fontsize=11)
        ax.set_xlim(-0.02, 1)
        ax.set_ylim(-0.02, 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=8, markerscale=3, loc="upper left")
    fig.suptitle(
        "Figure 5: Track A vs Track B (Raw) Agreement\n"
        "Points near the diagonal indicate the two tracks agree",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(FIGURES_DIR / "fig5_track_a_vs_b.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig5_track_a_vs_b.png")


# ── Figure 6: Track B centered vs raw collapse ──────────────────────

def fig6_centered_collapse(track_b: pd.DataFrame) -> None:
    """Bar chart showing the centered vs raw R² gap across models/datasets."""
    rows = []
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            mdf = track_b[(track_b["model"] == model) & (track_b["dataset"] == dataset)]
            if mdf.empty:
                continue
            rows.append({
                "model": model,
                "dataset": dataset,
                "r2_centered": mdf["r2_centered"].mean(),
                "r2_raw": mdf["r2_raw"].mean(),
            })

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for didx, dataset in enumerate(DATASET_ORDER):
        ax = axes[didx]
        ddf = df[df["dataset"] == dataset].copy()
        ddf = ddf.set_index("model").loc[[m for m in MODEL_ORDER if m in ddf["model"].values]]
        x = np.arange(len(ddf))
        width = 0.35
        ax.bar(x - width / 2, ddf["r2_raw"], width, label="Raw" if didx == 0 else None,
               color=[MODEL_COLORS[m] for m in ddf.index], alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, ddf["r2_centered"], width, label="Centered" if didx == 0 else None,
               color=[MODEL_COLORS[m] for m in ddf.index], alpha=0.3, edgecolor="black", linewidth=0.5,
               hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in ddf.index], fontsize=8, rotation=30)
        ax.set_title(DATASET_LABELS[dataset], fontsize=11)
        if didx == 0:
            ax.set_ylabel("Mean R² (Track B)")
        ax.grid(axis="y", alpha=0.2)

    axes[0].legend(fontsize=10)
    fig.suptitle(
        "Figure 6: Track B Centered vs Raw R² — Centering Collapse on Natural Text\n"
        "Solid = raw (no centering), Hatched = centered. Note collapse on Wiki/Code but not Random.",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(FIGURES_DIR / "fig6_centered_collapse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig6_centered_collapse.png")


# ── Figure 7: Boundary analysis ──────────────────────────────────────

def fig7_boundary(boundary: pd.DataFrame) -> None:
    """Grouped bar chart: full vs boundary vs interior R² per model."""
    rows = []
    for model in MODEL_ORDER:
        mdf = boundary[boundary["model"] == model]
        if mdf.empty:
            continue
        rows.append({
            "model": model,
            "full": mdf["mean_r2_full"].mean(),
            "boundary": mdf["mean_r2_boundary"].mean(),
            "interior": mdf["mean_r2_interior"].mean(),
        })

    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, df["boundary"], width, label="Boundary (pos 0-49)", color="#EF5350", alpha=0.8)
    ax.bar(x, df["full"], width, label="Full sequence", color="#42A5F5", alpha=0.8)
    ax.bar(x + width, df["interior"], width, label="Interior (pos 50+)", color="#66BB6A", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in df["model"]], fontsize=8)
    ax.set_ylabel("Mean R²")
    ax.legend(fontsize=10)
    ax.set_title(
        "Figure 7: Boundary Analysis — R² by Position Window\n"
        "Consistent with Proposition 4: causal masking depresses R² near sequence start",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_boundary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig7_boundary.png")


# ── Figure 8: Per-sequence R² distributions (violin) ────────────────

def fig8_per_seq_distributions(per_seq: pd.DataFrame) -> None:
    """Violin plots of per-sequence R² for early-layer heads, wiki/256."""
    dataset = "synthetic_random"
    seq_len = 256
    sub = per_seq[
        (per_seq["dataset"] == dataset) &
        (per_seq["seq_len"] == seq_len) &
        (per_seq["layer"].isin([0, 1]))
    ]

    models_present = [m for m in MODEL_ORDER if m in sub["model"].unique()]
    if not models_present:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    data_lists = []
    labels = []
    colors = []
    for model in models_present:
        vals = sub[sub["model"] == model]["r2_shift"].values
        if len(vals) == 0:
            continue
        data_lists.append(vals)
        labels.append(MODEL_SHORT[model])
        colors.append(MODEL_COLORS[model])

    parts = ax.violinplot(data_lists, showmeans=True, showmedians=True)
    for idx, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[idx])
        pc.set_alpha(0.6)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Per-Sequence R² (Track A, early layers)")
    ax.set_title(
        f"Figure 8: Per-Sequence R² Distribution — Early Layers (0-1)\n"
        f"(dataset={DATASET_LABELS[dataset]}, len={seq_len}). Narrow distributions = content-stable.",
        fontsize=12, fontweight="bold",
    )
    ax.axhline(y=0.80, color="green", linestyle="--", alpha=0.4, label="Strong support")
    ax.axhline(y=0.40, color="red", linestyle="--", alpha=0.4, label="Falsification")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_per_seq_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig8_per_seq_distributions.png")


# ── Figure 9: Pre-registration criteria summary ─────────────────────

def _merge_track_a_track_b(track_a: pd.DataFrame, track_b: pd.DataFrame) -> pd.DataFrame:
    return track_a[["model", "dataset", "layer", "head", "seq_len", "mean_r2"]].merge(
        track_b[["model", "dataset", "layer", "head", "seq_len", "r2_raw", "r2_centered"]],
        on=["model", "dataset", "layer", "head", "seq_len"],
    )


def _fmt_value(val: float) -> str:
    if np.isnan(val):
        return "nan"
    if abs(val) < 1e-3 and val != 0:
        return f"{val:.2e}"
    return f"{val:.3f}"


def fig9_preregistration_summary(
    track_a: pd.DataFrame,
    track_b_canonical: pd.DataFrame,
    spectral_hist_meta: pd.DataFrame,
) -> None:
    """Visual summary of current criteria outcomes with historical-vs-current split."""
    rope_early = track_a[
        (track_a["model"].isin(ROPE_MODELS)) &
        (track_a["layer"].isin([0, 1]))
    ]["mean_r2"].mean()

    nope_early = track_a[
        (track_a["model"] == "tinyllama-nope-1.1b") &
        (track_a["layer"].isin([0, 1]))
    ]["mean_r2"].mean()

    merged = _merge_track_a_track_b(track_a, track_b_canonical)
    natural = merged[merged["dataset"].isin(["wiki40b_en_pre2019", "codesearchnet_python_snapshot"])]
    synthetic = merged[merged["dataset"] == "synthetic_random"]

    agreement_natural_raw = (natural["mean_r2"] - natural["r2_raw"]).abs().mean()
    agreement_natural_centered = (natural["mean_r2"] - natural["r2_centered"]).abs().mean()
    agreement_synth_raw = (synthetic["mean_r2"] - synthetic["r2_raw"]).abs().mean()

    historical_gate_max = spectral_hist_meta["gate_value"].max()

    criteria = [
        ("RoPE early-layer R²", rope_early, 0.80, 0.40, "higher_graded"),
        ("NoPE early-layer R²", nope_early, 0.40, np.nan, "lower_threshold"),
        ("Track A≈B raw (natural)", agreement_natural_raw, 0.10, np.nan, "lower_threshold"),
        ("Track A≈B centered (natural)", agreement_natural_centered, 0.10, np.nan, "lower_threshold"),
        ("Track A≈B raw (synthetic)", agreement_synth_raw, 0.10, np.nan, "lower_threshold"),
        ("Historical spectral gate max", historical_gate_max, 0.60, np.nan, "higher_threshold"),
    ]

    labels = [c[0] for c in criteria]
    values = [c[1] for c in criteria]
    colors = []
    for _, val, upper, lower, mode in criteria:
        if mode == "higher_graded":
            if val >= upper:
                colors.append("#4CAF50")
            elif val < lower:
                colors.append("#F44336")
            else:
                colors.append("#FFC107")
        elif mode == "higher_threshold":
            colors.append("#4CAF50" if val >= upper else "#F44336")
        elif mode == "lower_threshold":
            colors.append("#4CAF50" if val < upper else "#F44336")
        else:
            colors.append("#FFC107")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(criteria))
    ax.barh(y, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Measured Value")

    for i, (_, val, upper, lower, mode) in enumerate(criteria):
        ax.text(min(val + 0.015, 0.97), i, _fmt_value(val), va="center", fontsize=9)
        if mode in {"higher_graded", "higher_threshold", "lower_threshold"} and not np.isnan(upper):
            ax.axvline(x=upper, color="#2E7D32", linestyle="--", alpha=0.35, linewidth=1)
        if mode == "higher_graded" and not np.isnan(lower):
            ax.axvline(x=lower, color="#C62828", linestyle="--", alpha=0.35, linewidth=1)

    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(
        "Figure 9: Criteria Snapshot (Canonical Track B + Historical Spectral Gate)\n"
        "Green=pass, Yellow=partial, Red=fail under stated threshold",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig9_preregistration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig9_preregistration.png")


# ── Figure 10: Legacy vs canonical Track B equivalence ─────────────

def fig10_trackb_variant_equivalence(
    track_a: pd.DataFrame,
    track_b_legacy: pd.DataFrame,
    track_b_canonical: pd.DataFrame,
) -> None:
    merged = track_b_legacy[[
        "model", "dataset", "layer", "head", "seq_len", "r2_centered", "r2_raw"
    ]].merge(
        track_b_canonical[[
            "model", "dataset", "layer", "head", "seq_len", "r2_centered", "r2_raw"
        ]],
        on=["model", "dataset", "layer", "head", "seq_len"],
        suffixes=("_legacy", "_canonical"),
    )
    mae_centered = (merged["r2_centered_legacy"] - merged["r2_centered_canonical"]).abs().mean()
    max_centered = (merged["r2_centered_legacy"] - merged["r2_centered_canonical"]).abs().max()
    mae_raw = (merged["r2_raw_legacy"] - merged["r2_raw_canonical"]).abs().mean()
    max_raw = (merged["r2_raw_legacy"] - merged["r2_raw_canonical"]).abs().max()

    ta_tb = _merge_track_a_track_b(track_a, track_b_canonical)
    gaps = (
        ta_tb.groupby("dataset", as_index=False)
        .agg(
            abs_a_minus_raw=("mean_r2", lambda s: np.nan),  # placeholder, overwritten below
            abs_a_minus_centered=("mean_r2", lambda s: np.nan),
        )
    )
    for idx, row in gaps.iterrows():
        ds = row["dataset"]
        sub = ta_tb[ta_tb["dataset"] == ds]
        gaps.loc[idx, "abs_a_minus_raw"] = (sub["mean_r2"] - sub["r2_raw"]).abs().mean()
        gaps.loc[idx, "abs_a_minus_centered"] = (sub["mean_r2"] - sub["r2_centered"]).abs().mean()
    gaps["dataset_label"] = gaps["dataset"].map(DATASET_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: centered legacy vs canonical
    ax = axes[0]
    ax.scatter(
        merged["r2_centered_legacy"],
        merged["r2_centered_canonical"],
        s=8,
        alpha=0.25,
        color="#1E88E5",
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_title("Centered: Legacy vs Canonical")
    ax.set_xlabel("Legacy centered R²")
    ax.set_ylabel("Canonical centered R²")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.text(
        0.03,
        0.93,
        f"MAE={mae_centered:.2e}\nMax={max_centered:.2e}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#90A4AE"),
    )

    # Panel B: raw legacy vs canonical
    ax = axes[1]
    ax.scatter(
        merged["r2_raw_legacy"],
        merged["r2_raw_canonical"],
        s=8,
        alpha=0.25,
        color="#43A047",
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_title("Raw: Legacy vs Canonical")
    ax.set_xlabel("Legacy raw R²")
    ax.set_ylabel("Canonical raw R²")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.text(
        0.03,
        0.93,
        f"MAE={mae_raw:.2e}\nMax={max_raw:.2e}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#90A4AE"),
    )

    # Panel C: |Track A - Track B| by dataset (canonical)
    ax = axes[2]
    x = np.arange(len(DATASET_ORDER))
    width = 0.36
    gaps = gaps.set_index("dataset").reindex(DATASET_ORDER).reset_index()
    raw_vals = gaps["abs_a_minus_raw"].values
    centered_vals = gaps["abs_a_minus_centered"].values
    ax.bar(x - width / 2, raw_vals, width, color="#1E88E5", alpha=0.85, label="|A - Track B raw|")
    ax.bar(
        x + width / 2,
        centered_vals,
        width,
        color="#FB8C00",
        alpha=0.45,
        hatch="//",
        edgecolor="black",
        linewidth=0.4,
        label="|A - Track B centered|",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=10)
    ax.set_ylabel("Mean absolute difference")
    ax.set_title("Canonical Track B vs Track A by dataset")
    ax.grid(axis="y", alpha=0.2)
    for i, (rv, cv) in enumerate(zip(raw_vals, centered_vals)):
        ax.text(i - width / 2, rv + 0.01, f"{rv:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, cv + 0.01, f"{cv:.3f}", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, max(centered_vals) * 1.25)

    fig.suptitle(
        "Figure 10: Canonical Track B Rerun Outcome\n"
        "Legacy and canonical per-position variants are near-identical; natural-text centered gap to Track A persists",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIGURES_DIR / "fig10_trackb_variant_equivalence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig10_trackb_variant_equivalence.png")


# ── Figure 11: Exploratory spectral summary ─────────────────────────

def fig11_spectral_exploratory_summary(spectral_rows: pd.DataFrame) -> None:
    model_agg = (
        spectral_rows.groupby("model", as_index=False)
        .agg(
            matched_mean=("matched_count", "mean"),
            mean_relative_error=("mean_relative_error", "mean"),
        )
        .set_index("model")
        .reindex(MODEL_ORDER)
        .reset_index()
    )
    md_agg = (
        spectral_rows.groupby(["model", "dataset"], as_index=False)
        .agg(matched_mean=("matched_count", "mean"))
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

    # Panel A: matched_count mean by model
    ax = axes[0]
    x = np.arange(len(MODEL_ORDER))
    vals = model_agg["matched_mean"].values
    bars = ax.bar(
        x,
        vals,
        color=[MODEL_COLORS[m] for m in MODEL_ORDER],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean matched_count (top-5)")
    ax.set_title("Exploratory spectral: peak matches")
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(0, 5.2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Panel B: mean relative error by model (RoPE-focused)
    ax = axes[1]
    mre_vals = []
    for m in MODEL_ORDER:
        row = model_agg[model_agg["model"] == m]
        mre = float(row["mean_relative_error"].iloc[0])
        mre_vals.append(mre if m in ROPE_MODELS else np.nan)
    bar_vals = [0 if np.isnan(v) else v for v in mre_vals]
    bars = ax.bar(
        x,
        bar_vals,
        color=[MODEL_COLORS[m] if m in ROPE_MODELS else "#B0BEC5" for m in MODEL_ORDER],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    for i, m in enumerate(MODEL_ORDER):
        if m not in ROPE_MODELS:
            bars[i].set_hatch("//")
            ax.text(i, 0.01, "N/A", ha="center", va="bottom", fontsize=8, color="#37474F")
        else:
            ax.text(i, bar_vals[i] + 0.006, f"{bar_vals[i]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean relative error")
    ax.set_title("RoPE models: alignment error\n(non-RoPE not comparable for this metric)")
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(0, max(bar_vals) * 1.25 if max(bar_vals) > 0 else 1)

    # Panel C: model x dataset heatmap (matched_mean)
    ax = axes[2]
    grid = np.zeros((len(MODEL_ORDER), len(DATASET_ORDER)))
    for i, m in enumerate(MODEL_ORDER):
        for j, d in enumerate(DATASET_ORDER):
            sub = md_agg[(md_agg["model"] == m) & (md_agg["dataset"] == d)]
            grid[i, j] = float(sub["matched_mean"].iloc[0]) if not sub.empty else np.nan
    im = ax.imshow(grid, aspect="auto", cmap="YlGnBu", vmin=0, vmax=5)
    ax.set_xticks(range(len(DATASET_ORDER)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], fontsize=9)
    ax.set_title("Matched-count heatmap\n(model × dataset)")
    for i in range(len(MODEL_ORDER)):
        for j in range(len(DATASET_ORDER)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean matched_count")

    fig.suptitle(
        "Figure 11: Exploratory Spectral Summary (gate=0.0)\n"
        "Preliminary but informative: strong RoPE-pattern matches, non-RoPE matching metric not directly comparable",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIGURES_DIR / "fig11_spectral_exploratory_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig11_spectral_exploratory_summary.png")


# ── Figure 12: Historical vs exploratory spectral gate drift ────────

def fig12_spectral_namespace_gate_drift(
    spectral_hist_meta: pd.DataFrame,
    spectral_expl_meta: pd.DataFrame,
) -> None:
    merged = spectral_hist_meta.merge(
        spectral_expl_meta,
        on=["model", "dataset", "seq_len"],
        suffixes=("_hist", "_expl"),
    )
    merged["gate_delta"] = merged["gate_value_expl"] - merged["gate_value_hist"]
    merged["abs_delta"] = merged["gate_delta"].abs()
    top = merged.sort_values("abs_delta", ascending=False).head(12).copy()
    large_count = int((merged["abs_delta"] > 0.1).sum())

    labels = [
        f"{MODEL_SHORT[m]} / {DATASET_LABELS[d]} / {int(s)}"
        for m, d, s in zip(top["model"], top["dataset"], top["seq_len"])
    ]
    vals = top["gate_delta"].values

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(vals))
    colors = ["#1E88E5" if v >= 0 else "#E53935" for v in vals]
    ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Exploratory - Historical gate_value")
    ax.set_title(
        "Figure 12: Spectral Namespace Gate Drift\n"
        f"Top absolute gate deltas by combo (abs(delta) > 0.1 count = {large_count})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.2)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.01 if v >= 0 else -0.015), f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig12_spectral_namespace_gate_drift.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig12_spectral_namespace_gate_drift.png")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    track_a = load_all_track_a_summaries()
    track_b_legacy = load_all_track_b_summaries("track_b")
    track_b_canonical = load_all_track_b_summaries("track_b_canonical_perpos_v1")
    boundary = load_all_boundary_summaries()
    per_seq = load_all_track_a_per_sequence()
    spectral_hist_meta = load_spectral_meta("spectral")
    spectral_expl_meta = load_spectral_meta("spectral_canonical_perpos_v1_t0")
    spectral_expl_rows = load_exploratory_spectral_rows("spectral_canonical_perpos_v1_t0")
    print(
        f"  Track A: {len(track_a)} rows, Track B legacy: {len(track_b_legacy)} rows, "
        f"Track B canonical: {len(track_b_canonical)} rows"
    )
    print(f"  Boundary: {len(boundary)} rows, Per-seq: {len(per_seq)} rows")
    print(
        f"  Spectral historical meta: {len(spectral_hist_meta)} rows, "
        f"exploratory meta: {len(spectral_expl_meta)} rows, "
        f"exploratory spectral rows: {len(spectral_expl_rows)}"
    )

    print("\nGenerating figures...")
    fig1_r2_heatmaps(track_a)
    fig2_r2_vs_depth(track_a, track_b_canonical)
    fig3_model_comparison(track_a)
    fig4_nope_paired(track_a)
    fig5_track_a_vs_b(track_a, track_b_canonical)
    fig6_centered_collapse(track_b_canonical)
    fig7_boundary(boundary)
    fig8_per_seq_distributions(per_seq)
    fig9_preregistration_summary(track_a, track_b_canonical, spectral_hist_meta)
    fig10_trackb_variant_equivalence(track_a, track_b_legacy, track_b_canonical)
    fig11_spectral_exploratory_summary(spectral_expl_rows)
    fig12_spectral_namespace_gate_drift(spectral_hist_meta, spectral_expl_meta)
    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
