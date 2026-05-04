#!/usr/bin/env python3
"""Generate NeurIPS main-paper figures from existing experiment artifacts.

Outputs:
- paper/neurips2026/figures/fig2_exp7a_coherence_vs_r2.png
- paper/neurips2026/figures/fig3_exp8b_llama_pruning.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def family_of(model: str) -> str:
    m = model.lower()
    if "nope" in m:
        return "NoPE"
    if m.startswith("gpt2"):
        return "Absolute PE + LayerNorm"
    if m.startswith("olmo"):
        return "RoPE + LayerNorm"
    if m.startswith("llama") or m.startswith("tinyllama") or m.startswith("mistral"):
        return "RoPE + RMSNorm"
    return "Other"


def make_fig2(repo_root: Path, out_dir: Path) -> None:
    df = pd.read_parquet(
        repo_root
        / "results"
        / "experiment7"
        / "exp7a_welch_r2"
        / "r2_prediction_table.parquet"
    ).copy()
    df["family"] = df["model"].map(family_of)

    colors = {
        "RoPE + RMSNorm": "#1f77b4",
        "RoPE + LayerNorm": "#d62728",
        "Absolute PE + LayerNorm": "#2ca02c",
        "NoPE": "#7f7f7f",
        "Other": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for fam, sub in df.groupby("family"):
        ax.scatter(
            sub["eta_welch_gap"],
            sub["mean_pooled_r2"],
            s=36,
            alpha=0.85,
            color=colors.get(fam, "#333333"),
            label=fam,
            edgecolor="white",
            linewidth=0.5,
        )

    # Power-law fit in log space: y = a * x^b
    valid = (df["eta_welch_gap"] > 0) & (df["mean_pooled_r2"] > 0)
    x = df.loc[valid, "eta_welch_gap"].to_numpy()
    y = df.loc[valid, "mean_pooled_r2"].to_numpy()
    b, loga = np.polyfit(np.log(x), np.log(y), 1)
    a = float(np.exp(loga))
    xx = np.linspace(float(x.min()) * 0.95, float(x.max()) * 1.02, 200)
    yy = a * np.power(xx, b)
    alpha = -b

    ax.plot(
        xx,
        yy,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"Power-law fit (alpha={alpha:.3f})",
    )

    ax.set_xlabel("PE coherence gap (mu/mu_W; lower is better)")
    ax.set_ylabel("Mean per-head R^2")
    ax.set_title("Exp 7A: Coherence gap tracks SI strength")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, frameon=True, loc="best")
    fig.tight_layout()

    out_path = out_dir / "fig2_exp7a_coherence_vs_r2.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_fig3(repo_root: Path, out_dir: Path) -> None:
    df = pd.read_parquet(
        repo_root
        / "results"
        / "experiment8"
        / "exp8b_pruning"
        / "llama-3.1-8b"
        / "pruning_curves.parquet"
    ).copy()

    order = ["si_matched", "si_mismatched", "entropy", "activation", "random"]
    labels = {
        "si_matched": "SI-matched",
        "si_mismatched": "SI-mismatched",
        "entropy": "Entropy",
        "activation": "Activation",
        "random": "Random",
    }
    styles = {
        "si_matched": dict(color="#1f77b4", linestyle="-", marker="o"),
        "si_mismatched": dict(color="#ff7f0e", linestyle="--", marker="o"),
        "entropy": dict(color="#d62728", linestyle=":", marker="s"),
        "activation": dict(color="#9467bd", linestyle="-.", marker="^"),
        "random": dict(color="#2ca02c", linestyle="--", marker="d"),
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for mask in order:
        sub = df[df["mask_type"] == mask].sort_values("fraction_ablated_pct")
        if sub.empty:
            continue
        ax.plot(
            sub["fraction_ablated_pct"],
            sub["math_accuracy"],
            label=labels.get(mask, mask),
            linewidth=1.8,
            markersize=4.5,
            **styles.get(mask, {}),
        )

    ax.set_xlabel("Ablation fraction (%)")
    ax.set_ylabel("4-shot MATH accuracy")
    ax.set_ylim(-0.02, 0.92)
    ax.set_xlim(-1, 76)
    ax.set_title("Exp 8B (Llama-3.1-8B): pruning trajectories by mask type")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, frameon=True, loc="upper right")
    fig.tight_layout()

    out_path = out_dir / "fig3_exp8b_llama_pruning.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "paper" / "neurips2026" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_fig2(repo_root, out_dir)
    make_fig3(repo_root, out_dir)


if __name__ == "__main__":
    main()
