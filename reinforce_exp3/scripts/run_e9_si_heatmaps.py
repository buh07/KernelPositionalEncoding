#!/usr/bin/env python3
"""E9 — Layer-Head SI Heatmaps.

Generates three sets of heatmaps (layer × head grid) from already-computed
artifacts: (1) R² values, (2) cluster membership, (3) boundary attention score.
No model runs required — pure visualization from parquet files.

Usage:
    python reinforce_exp3/scripts/run_e9_si_heatmaps.py \
        --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    B1_RESULTS,
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    emit_core_artifacts,
    parse_models_arg,
)

OUT_ROOT = RESULTS_ROOT / "E9_si_heatmaps"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_heatmap_data(models: list[str]) -> pd.DataFrame:
    """Load cluster_membership parquet; return filtered to requested models."""
    path = B1_RESULTS / "cluster_membership.parquet"
    if not path.exists():
        raise FileNotFoundError(f"cluster_membership.parquet not found at {path}")
    df = pd.read_parquet(path)
    if models:
        df = df[df["model"].isin(models)].copy()
    return df


# ---------------------------------------------------------------------------
# Heatmap matrix builders
# ---------------------------------------------------------------------------

def _build_r2_matrix(df_model: pd.DataFrame) -> np.ndarray:
    """Return (n_layers, n_heads) matrix of mean_r2 values."""
    n_layers = int(df_model["layer"].max()) + 1
    n_heads = int(df_model["head"].max()) + 1
    mat = np.full((n_layers, n_heads), float("nan"))
    for _, row in df_model.iterrows():
        mat[int(row["layer"]), int(row["head"])] = float(row["mean_r2"])
    return mat


def _build_cluster_matrix(df_model: pd.DataFrame) -> np.ndarray:
    """Return (n_layers, n_heads) matrix of cluster_descriptor_kmeans int codes."""
    n_layers = int(df_model["layer"].max()) + 1
    n_heads = int(df_model["head"].max()) + 1
    mat = np.full((n_layers, n_heads), -1.0)
    # Encode cluster labels as integers
    labels = df_model["cluster_descriptor_kmeans"].astype("category")
    for _, row in df_model.iterrows():
        code = labels.cat.categories.get_loc(row["cluster_descriptor_kmeans"])
        mat[int(row["layer"]), int(row["head"])] = float(code)
    return mat


def _build_boundary_matrix(df_model: pd.DataFrame) -> np.ndarray:
    """Return (n_layers, n_heads) matrix of boundary_attn_score values."""
    if "boundary_attn_score" not in df_model.columns:
        return np.full(
            (int(df_model["layer"].max()) + 1, int(df_model["head"].max()) + 1),
            float("nan"),
        )
    n_layers = int(df_model["layer"].max()) + 1
    n_heads = int(df_model["head"].max()) + 1
    mat = np.full((n_layers, n_heads), float("nan"))
    for _, row in df_model.iterrows():
        mat[int(row["layer"]), int(row["head"])] = float(row["boundary_attn_score"])
    return mat


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _save_heatmap_pdf(
    matrix: np.ndarray,
    title: str,
    out_path: Path,
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    discrete: bool = False,
    contour_matrix: np.ndarray | None = None,
    contour_threshold: float | None = None,
) -> None:
    """Save a single heatmap panel as PDF."""
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except ImportError:
        print(f"[E9] matplotlib not available; skipping PDF for {title}", flush=True)
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    if discrete:
        n_unique = int(np.nanmax(matrix)) + 1
        cmap = plt.get_cmap("tab10", n_unique)
        im = ax.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=n_unique - 0.5, aspect="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_ticks(range(n_unique))
        cbar.set_label("Cluster ID")
    else:
        im = ax.imshow(
            matrix,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        fig.colorbar(im, ax=ax)

    # Overlay high-SI contour if requested
    if contour_matrix is not None and contour_threshold is not None:
        binary = (contour_matrix >= contour_threshold).astype(float)
        ax.contour(binary, levels=[0.5], colors="white", linewidths=1.5, alpha=0.7)

    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_title(title)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(str(out_path), format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _generate_multipanel_heatmap(
    matrices: dict[str, np.ndarray],
    model_names: list[str],
    title_prefix: str,
    out_path: Path,
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    discrete: bool = False,
) -> None:
    """Save a 3-panel figure (one panel per model) as PDF."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[E9] matplotlib not available; skipping {out_path.name}", flush=True)
        return

    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        mat = matrices.get(model_name)
        if mat is None:
            ax.set_title(f"{model_name}\n(no data)")
            continue
        if discrete:
            n_unique = max(int(np.nanmax(mat[np.isfinite(mat)])) + 1, 1)
            cmap = plt.get_cmap("tab10", n_unique)
            im = ax.imshow(mat, cmap=cmap, vmin=-0.5, vmax=n_unique - 0.5, aspect="auto")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_ticks(range(n_unique))
        else:
            im = ax.imshow(mat, cmap=colormap, vmin=vmin, vmax=vmax, aspect="auto")
            fig.colorbar(im, ax=ax)
        ax.set_xlabel("Head index")
        ax.set_ylabel("Layer index")
        ax.set_title(f"{model_name}")

    fig.suptitle(title_prefix, fontsize=14)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(str(out_path), format="pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[E9] Saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Per-model processing
# ---------------------------------------------------------------------------

def process_model(
    model_name: str,
    df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:
    df_m = df[df["model"] == model_name].copy()
    if df_m.empty:
        print(f"[E9] No data for {model_name}", flush=True)
        return {"model": model_name, "status": "no_data"}

    r2_mat = _build_r2_matrix(df_m)
    cluster_mat = _build_cluster_matrix(df_m)
    boundary_mat = _build_boundary_matrix(df_m)

    # High-SI threshold: top quartile of r2
    finite_r2 = r2_mat[np.isfinite(r2_mat)]
    hi_si_thresh = float(np.nanpercentile(finite_r2, 75)) if len(finite_r2) > 0 else 0.5

    # Export data parquets
    model_data_out = ensure_dir(out_dir / "data")
    n_layers, n_heads = r2_mat.shape

    rows: list[dict[str, Any]] = []
    for li in range(n_layers):
        for hi in range(n_heads):
            rows.append({
                "model": model_name,
                "layer": li,
                "head": hi,
                "mean_r2": r2_mat[li, hi],
                "cluster_code": cluster_mat[li, hi],
                "boundary_attn_score": boundary_mat[li, hi],
            })
    pd.DataFrame(rows).to_parquet(
        model_data_out / f"heatmap_data_{model_name.replace('/', '_')}.parquet",
        index=False,
    )

    return {
        "model": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "hi_si_threshold": hi_si_thresh,
        "r2_matrix": r2_mat,
        "cluster_matrix": cluster_mat,
        "boundary_matrix": boundary_mat,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="E9: Generate layer-head R², cluster, and boundary heatmaps",
        allow_abbrev=False,
    )
    p.add_argument("--models", default=",".join(["llama-3.1-8b", "olmo-2-7b", "mistral-7b-v0.1"]))
    p.add_argument("--output-root", default=str(OUT_ROOT))
    args = p.parse_args()

    models = parse_models_arg(args.models)
    out_dir = ensure_dir(Path(args.output_root))
    fig_dir = ensure_dir(out_dir / "figures")
    data_dir = ensure_dir(out_dir / "data")
    start_ts = timestamp_now()

    print(f"[E9] Starting at {start_ts}", flush=True)

    try:
        df = _load_heatmap_data(models)
    except FileNotFoundError as exc:
        print(f"[E9] ERROR: {exc}", flush=True)
        sys.exit(1)

    model_results: list[dict[str, Any]] = []
    for model_name in models:
        result = process_model(model_name, df, out_dir)
        model_results.append(result)

    ok_results = [r for r in model_results if r.get("status") == "ok"]

    # Build multipanel figures
    r2_matrices = {r["model"]: r["r2_matrix"] for r in ok_results}
    cluster_matrices = {r["model"]: r["cluster_matrix"] for r in ok_results}
    boundary_matrices = {r["model"]: r["boundary_matrix"] for r in ok_results}
    ok_models = [r["model"] for r in ok_results]

    _generate_multipanel_heatmap(
        r2_matrices,
        ok_models,
        "Shift-Invariant R² by Layer and Head",
        fig_dir / "fig_r2_heatmap_all_models.pdf",
        colormap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    _generate_multipanel_heatmap(
        cluster_matrices,
        ok_models,
        "Cluster Membership by Layer and Head",
        fig_dir / "fig_cluster_heatmap_all_models.pdf",
        discrete=True,
    )

    _generate_multipanel_heatmap(
        boundary_matrices,
        ok_models,
        "Boundary Attention Score by Layer and Head",
        fig_dir / "fig_boundary_heatmap_all_models.pdf",
        colormap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )

    # Save combined parquet
    all_rows: list[dict[str, Any]] = []
    for r in ok_results:
        n_layers, n_heads = r["r2_matrix"].shape
        for li in range(n_layers):
            for hi in range(n_heads):
                all_rows.append({
                    "model": r["model"],
                    "layer": li,
                    "head": hi,
                    "mean_r2": r["r2_matrix"][li, hi],
                    "cluster_code": r["cluster_matrix"][li, hi],
                    "boundary_attn_score": r["boundary_matrix"][li, hi],
                    "is_high_si": r["r2_matrix"][li, hi] >= r["hi_si_threshold"],
                })
    pd.DataFrame(all_rows).to_parquet(data_dir / "r2_heatmap_data.parquet", index=False)

    # Emit governance artifacts
    claim_impact = {
        "experiment_id": "E9",
        "claim_addressed": "Visualization of SI structure, cluster membership, boundary attention in layer-head space",
        "claim_status": "supported",
        "supports_main_text": True,
        "outcome_summary": f"Generated 3 multipanel heatmaps for {len(ok_models)} models",
        "notes": [
            "Figures written to results/reinforce_exp3/E9_si_heatmaps/figures/",
            "Ready for inclusion in paper/neurips2026/figures/",
        ],
    }

    preregistration = {
        "experiment_id": "E9",
        "hypothesis": "Visualization only — no hypothesis tested",
        "models": models,
        "data_sources": [str(B1_RESULTS / "cluster_membership.parquet")],
        "timestamp": start_ts,
    }

    summary = {
        "experiment_id": "E9",
        "status": "complete",
        "models_processed": ok_models,
        "figures_generated": [
            "fig_r2_heatmap_all_models.pdf",
            "fig_cluster_heatmap_all_models.pdf",
            "fig_boundary_heatmap_all_models.pdf",
        ],
        "timestamp_start": start_ts,
        "timestamp_end": timestamp_now(),
    }

    data_dictionary = {
        "experiment_id": "E9",
        "tables": [
            {
                "path": "data/r2_heatmap_data.parquet",
                "description": "All-model heatmap data: R², cluster, boundary per (layer, head)",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model name"},
                    {"name": "layer", "dtype": "int", "description": "Transformer layer index"},
                    {"name": "head", "dtype": "int", "description": "Attention head index"},
                    {"name": "mean_r2", "dtype": "float", "description": "Shift-invariant R²"},
                    {"name": "cluster_code", "dtype": "float", "description": "K-means cluster code (int, -1=unassigned)"},
                    {"name": "boundary_attn_score", "dtype": "float", "description": "Boundary attention score"},
                    {"name": "is_high_si", "dtype": "bool", "description": "True if R² >= 75th percentile"},
                ],
            }
        ],
    }

    manifest_extra = {
        "models": models,
        "data_source": str(B1_RESULTS / "cluster_membership.parquet"),
        "figures_dir": str(fig_dir),
    }

    emit_core_artifacts(
        out_dir=out_dir,
        experiment_id="E9",
        preregistration=preregistration,
        manifest_extra=manifest_extra,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )

    print(f"[E9] Done. Figures in {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
