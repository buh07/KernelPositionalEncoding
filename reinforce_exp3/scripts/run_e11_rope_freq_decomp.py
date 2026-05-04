#!/usr/bin/env python3
"""E11 — High-SI RoPE-Frequency Decomposition.

Decomposes the estimated g_h(Δ) kernel for each high-SI head into its DFT
power spectrum and identifies which RoPE frequency bands dominate each head's
positional response. Tests the hypothesis that high-frequency SI heads are
boundary-sensitive (Δ=1) while low-frequency heads handle long-range structure.

Pure reanalysis — no GPU needed beyond what's already been run.

Usage:
    python reinforce_exp3/scripts/run_e11_rope_freq_decomp.py \
        --models llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    B1_RESULTS,
    EXP3_SI_CIRCUITS,
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    enforce_coverage_contract,
    emit_core_artifacts,
    parse_models_arg,
    read_json,
    spearman_with_ci,
)

OUT_ROOT = RESULTS_ROOT / "E11_rope_freq_decomposition"

MAX_OFFSET = 64   # must match the offset range used in g_h estimation
N_FREQ_BINS = 5   # number of frequency bands for clustering


# ---------------------------------------------------------------------------
# Kernel loading
# ---------------------------------------------------------------------------

def _load_kernel_data(model_name: str) -> pd.DataFrame | None:
    """Try to load g_h kernel data from experiment3 SI circuit artifacts.

    Expected: parquet with columns (layer, head, delta, g_h_value).
    Falls back to cluster_membership if kernels not stored separately.
    """
    # Preferred: dedicated kernel parquet
    kernel_path = EXP3_SI_CIRCUITS / model_name / "kernels_g_h.parquet"
    if kernel_path.exists():
        return pd.read_parquet(kernel_path)

    # Fallback: check for any parquet with kernel-like columns
    si_dir = EXP3_SI_CIRCUITS / model_name
    if si_dir.exists():
        for pq in si_dir.glob("*kernel*.parquet"):
            df = pd.read_parquet(pq)
            if "delta" in df.columns and "g_h_value" in df.columns:
                return df

    print(
        f"[E11] Warning: no kernel parquet found for {model_name}. "
        "Will synthesize kernels from mean attention data if available.",
        flush=True,
    )
    return None


def _load_cluster_and_boundary(model_name: str) -> pd.DataFrame:
    """Load cluster membership with boundary_attn_score for cross-referencing."""
    path = B1_RESULTS / "cluster_membership.parquet"
    if not path.exists():
        return pd.DataFrame()
    cm = pd.read_parquet(path)
    return cm[cm["model"] == model_name].copy()


# ---------------------------------------------------------------------------
# DFT power spectrum analysis
# ---------------------------------------------------------------------------

def _compute_power_spectrum(g_h: np.ndarray) -> np.ndarray:
    """DFT power spectrum of g_h(Δ). Returns array of shape (n_freqs,)."""
    n = len(g_h)
    if n < 4:
        return np.zeros(n)
    # Hanning window to reduce spectral leakage
    window = np.hanning(n)
    g_h_windowed = g_h * window
    fft = np.fft.rfft(g_h_windowed)
    power = np.abs(fft) ** 2
    return power


def _peak_frequency(power: np.ndarray) -> int:
    """Return index of peak frequency (0-based)."""
    return int(np.argmax(power))


def _frequency_band(freq_idx: int, n_fft_bins: int) -> str:
    """Assign a frequency band label."""
    frac = freq_idx / max(n_fft_bins - 1, 1)
    if frac < 0.2:
        return "very_low"
    elif frac < 0.4:
        return "low"
    elif frac < 0.6:
        return "medium"
    elif frac < 0.8:
        return "high"
    else:
        return "very_high"


# ---------------------------------------------------------------------------
# Per-head decomposition
# ---------------------------------------------------------------------------

def _decompose_head_kernel(
    g_h: np.ndarray,
    layer: int,
    head: int,
    max_offset: int,
) -> dict[str, Any]:
    """Full DFT decomposition for one head's g_h kernel."""
    # Handle NaN gaps in g_h
    g_clean = np.where(np.isfinite(g_h), g_h, 0.0)

    power = _compute_power_spectrum(g_clean)
    n_fft_bins = len(power)
    peak_freq_idx = _peak_frequency(power)
    freq_band = _frequency_band(peak_freq_idx, n_fft_bins)

    # Low-frequency fraction: fraction of power in bottom 20% of frequencies
    cutoff = max(1, int(0.2 * n_fft_bins))
    low_freq_power_frac = float(power[:cutoff].sum() / max(power.sum(), 1e-12))

    # Compute "local sensitivity" = g_h(0) + g_h(1) relative to g_h overall
    local_sensitivity = float(g_clean[0] + g_clean[1]) / max(float(np.abs(g_clean).sum()), 1e-12)

    return {
        "layer": layer,
        "head": head,
        "peak_freq_idx": peak_freq_idx,
        "peak_freq_normalized": float(peak_freq_idx) / max(n_fft_bins - 1, 1),
        "freq_band": freq_band,
        "low_freq_power_frac": low_freq_power_frac,
        "local_sensitivity": local_sensitivity,
        "power_spectrum_sum": float(power.sum()),
    }


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,
    out_dir: Path,
    allow_synthetic_fallback: bool = False,
) -> dict[str, Any]:
    kernel_df = _load_kernel_data(model_name)
    cm_df = _load_cluster_and_boundary(model_name)

    model_out = ensure_dir(out_dir / model_name)

    decomp_rows: list[dict[str, Any]] = []

    if kernel_df is not None and not kernel_df.empty:
        # Group by (layer, head) and extract g_h array
        for (layer, head), grp in kernel_df.groupby(["layer", "head"]):
            grp_sorted = grp.sort_values("delta")
            g_h = grp_sorted["g_h_value"].values[:MAX_OFFSET + 1]
            # Pad to MAX_OFFSET + 1 if shorter
            if len(g_h) < MAX_OFFSET + 1:
                g_h = np.pad(g_h, (0, MAX_OFFSET + 1 - len(g_h)), constant_values=float("nan"))
            rec = _decompose_head_kernel(g_h, int(layer), int(head), MAX_OFFSET)
            decomp_rows.append(rec)
    else:
        if not allow_synthetic_fallback:
            raise RuntimeError(
                f"[E11] hard_fail_reason: missing real g_h kernel artifacts for {model_name}; "
                "confirmatory mode forbids synthetic fallback"
            )
        print(f"[E11] {model_name}: using synthetic g_h placeholder (non-confirmatory mode)", flush=True)
        if cm_df.empty:
            raise RuntimeError(f"[E11] hard_fail_reason: no fallback cluster data for {model_name}")
        for _, row in cm_df.iterrows():
            layer = int(row["layer"])
            head = int(row["head"])
            mean_r2 = float(row.get("mean_r2", 0.5))
            g_h = np.array([mean_r2 * np.exp(-0.1 * d) for d in range(MAX_OFFSET + 1)])
            rec = _decompose_head_kernel(g_h, layer, head, MAX_OFFSET)
            decomp_rows.append(rec)

    decomp_df = pd.DataFrame(decomp_rows)

    # Merge with boundary attention score and cluster info
    if not cm_df.empty and not decomp_df.empty:
        want_cols = ["layer", "head", "mean_r2", "boundary_attn_score",
                     "cluster_descriptor_kmeans", "is_high_si"]
        merge_cols = [c for c in want_cols if c in cm_df.columns]
        merged = decomp_df.merge(cm_df[merge_cols], on=["layer", "head"], how="left")
    else:
        merged = decomp_df

    merged["model"] = model_name
    merged.to_parquet(model_out / "freq_decomposition.parquet", index=False)

    # Test hypothesis: high-freq SI heads have higher boundary_attn_score
    result: dict[str, Any] = {"model": model_name, "status": "ok"}
    if "boundary_attn_score" in merged.columns and "peak_freq_normalized" in merged.columns:
        valid = merged[merged["peak_freq_normalized"].notna() & merged["boundary_attn_score"].notna()]
        if len(valid) >= 10:
            spearman_result = spearman_with_ci(
                valid["peak_freq_normalized"].tolist(),
                valid["boundary_attn_score"].tolist(),
            )
            result["freq_boundary_spearman_rho"] = spearman_result.get("rho", float("nan"))
            result["freq_boundary_spearman_pval"] = spearman_result.get("pval", float("nan"))
            print(
                f"[E11] {model_name}: peak_freq ~ boundary_attn spearman "
                f"rho={spearman_result.get('rho', float('nan')):.4f} "
                f"p={spearman_result.get('pval', float('nan')):.4f}",
                flush=True,
            )

    # Frequency band distribution
    if "freq_band" in merged.columns and "is_high_si" in merged.columns:
        high_si_merged = merged[merged["is_high_si"] == True]
        band_counts = high_si_merged["freq_band"].value_counts().to_dict()
        result["high_si_freq_band_distribution"] = band_counts
        print(f"[E11] {model_name} high-SI freq band distribution: {band_counts}", flush=True)

    # Per-cluster mean peak frequency
    if "cluster_descriptor_kmeans" in merged.columns:
        cluster_freqs = merged.groupby("cluster_descriptor_kmeans")["peak_freq_normalized"].mean().to_dict()
        result["cluster_mean_peak_freq"] = cluster_freqs

    write_json(model_out / "decomposition_summary.json", result)
    return result


# ---------------------------------------------------------------------------
# Cross-model summary
# ---------------------------------------------------------------------------

def _cross_model_summary(
    model_results: list[dict[str, Any]],
    out_dir: Path,
    required_models: list[str],
) -> dict[str, Any]:
    observed = [r.get("model", "") for r in model_results if r.get("status") == "ok"]
    enforce_coverage_contract(
        experiment_id="E11",
        observed_models=observed,
        required_models=required_models,
        observed_counts={"n_models": len(observed)},
        min_counts={"n_models": len(required_models)},
    )
    rho_vals = [
        r.get("freq_boundary_spearman_rho", float("nan"))
        for r in model_results
        if r.get("status") == "ok"
    ]
    valid_rhos = [v for v in rho_vals if not np.isnan(v)]

    if not valid_rhos:
        interpretation = "no_data"
        note = "No valid frequency-boundary correlations computed."
    elif all(r > 0.3 for r in valid_rhos):
        interpretation = "high_freq_boundary_linked"
        note = (
            "Higher peak-frequency SI heads have higher boundary attention scores (ρ > 0.3) "
            "across all tested models. Supports frequency-band division of labor hypothesis "
            "(Barbero et al. 2024): high-frequency heads handle boundary detection."
        )
    elif all(r < -0.3 for r in valid_rhos):
        interpretation = "low_freq_boundary_linked"
        note = "Lower peak-frequency heads are boundary-sensitive — inverted from Barbero hypothesis."
    else:
        interpretation = "mixed_or_weak"
        note = (
            f"Mixed correlation signs or weak effects (|ρ| < 0.3) across models. "
            "No clear frequency-boundary division of labor detected."
        )

    summary = {
        "n_models": len(valid_rhos),
        "per_model_rho": [
            {"model": r["model"], "rho": r.get("freq_boundary_spearman_rho", float("nan"))}
            for r in model_results if r.get("status") == "ok"
        ],
        "interpretation": interpretation,
        "note": note,
    }
    write_json(out_dir / "cross_model_freq_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="E11: RoPE frequency decomposition of g_h kernels for high-SI heads",
        allow_abbrev=False,
    )
    p.add_argument("--models", default=",".join(["llama-3.1-8b", "olmo-2-7b", "mistral-7b-v0.1"]))
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--allow-synthetic-fallback", action="store_true",
                   help="Allow synthetic placeholder kernels when real g_h artifacts are missing.")
    p.add_argument("--finalize-only", action="store_true",
                   help="Emit cross-model artifacts from per-model decomposition summaries only.")
    p.add_argument("--no-finalize", action="store_true",
                   help="Run per-model decomposition but skip cross-model finalize emission.")
    args = p.parse_args()
    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E11] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models)
    out_dir = ensure_dir(Path(args.output_root))
    start_ts = timestamp_now()

    print(f"[E11] Starting at {start_ts}", flush=True)

    model_results: list[dict[str, Any]] = []
    if args.finalize_only:
        for model_name in models:
            summary_path = out_dir / model_name / "decomposition_summary.json"
            if not summary_path.exists():
                raise RuntimeError(f"[E11] hard_fail_reason: missing shard artifact {summary_path}")
            payload = read_json(summary_path)
            payload["model"] = model_name
            payload.setdefault("status", "ok")
            model_results.append(payload)
    else:
        for model_name in models:
            result = run_model(
                model_name,
                out_dir,
                allow_synthetic_fallback=bool(args.allow_synthetic_fallback),
            )
            model_results.append(result)

    if args.no_finalize:
        print("[E11] Shard run complete (no finalize).", flush=True)
        return

    cross_model = _cross_model_summary(model_results, out_dir, required_models=models)

    # Generate frequency-band visualization
    _save_freq_band_figure(model_results, out_dir / "figures" / "fig_freq_band_distribution.pdf")

    interp = cross_model.get("interpretation", "unknown")
    claim_status_map = {
        "high_freq_boundary_linked": "supported",
        "low_freq_boundary_linked": "mixed",
        "mixed_or_weak": "inconclusive",
        "no_data": "inconclusive",
    }
    claim_status = claim_status_map.get(interp, "inconclusive")

    claim_impact = {
        "experiment_id": "E11",
        "claim_addressed": (
            "High-SI heads with high-frequency RoPE dominance are preferentially "
            "boundary-sensitive, providing mechanistic grounding for the SI-boundary link"
        ),
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "mixed"),
        "outcome_summary": cross_model.get("note", ""),
        "notes": [cross_model.get("note", "")],
    }

    preregistration = {
        "experiment_id": "E11",
        "hypothesis": (
            "High-SI heads with higher peak DFT frequency in g_h(Δ) have higher boundary "
            "attention scores (Spearman ρ > 0 for peak_freq ~ boundary_attn_score)."
        ),
        "primary_criterion": "Spearman ρ > 0.3 in ≥ 2/3 models",
        "models": models,
        "max_offset": MAX_OFFSET,
        "reference": "Barbero et al. 2024, arXiv:2410.06205",
        "timestamp": start_ts,
    }

    summary = {
        "experiment_id": "E11",
        "status": "complete",
        "models_analyzed": models,
        "interpretation": interp,
        "timestamp_start": start_ts,
        "timestamp_end": timestamp_now(),
    }

    data_dictionary = {
        "experiment_id": "E11",
        "tables": [
            {
                "path": "<model>/freq_decomposition.parquet",
                "description": "Per-head DFT frequency decomposition of g_h kernel",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model name"},
                    {"name": "layer", "dtype": "int", "description": "Layer index"},
                    {"name": "head", "dtype": "int", "description": "Head index"},
                    {"name": "peak_freq_idx", "dtype": "int", "description": "Index of peak DFT frequency"},
                    {"name": "peak_freq_normalized", "dtype": "float",
                     "description": "Peak frequency normalized to [0, 1]"},
                    {"name": "freq_band", "dtype": "str",
                     "description": "Frequency band label (very_low/low/medium/high/very_high)"},
                    {"name": "low_freq_power_frac", "dtype": "float",
                     "description": "Fraction of DFT power in bottom 20% of frequencies"},
                    {"name": "local_sensitivity", "dtype": "float",
                     "description": "Normalized sum of g_h(0) + g_h(1) — local offset sensitivity"},
                    {"name": "boundary_attn_score", "dtype": "float",
                     "description": "Boundary attention score from B1 cluster membership"},
                    {"name": "is_high_si", "dtype": "bool", "description": "Top-quartile R²"},
                ],
            }
        ],
    }

    manifest_extra = {
        "models": models,
        "max_offset": MAX_OFFSET,
        "data_source": str(EXP3_SI_CIRCUITS),
        "fallback_source": str(B1_RESULTS),
        "allow_synthetic_fallback": bool(args.allow_synthetic_fallback),
    }

    emit_core_artifacts(
        out_dir=out_dir,
        experiment_id="E11",
        preregistration=preregistration,
        manifest_extra=manifest_extra,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )

    print(f"[E11] Done. Interpretation: {interp}", flush=True)


def _save_freq_band_figure(
    model_results: list[dict[str, Any]],
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    valid_models = [r for r in model_results if r.get("status") == "ok" and "high_si_freq_band_distribution" in r]
    if not valid_models:
        return

    bands = ["very_low", "low", "medium", "high", "very_high"]
    n = len(valid_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, valid_models):
        dist = r.get("high_si_freq_band_distribution", {})
        counts = [dist.get(b, 0) for b in bands]
        ax.bar(bands, counts, color="steelblue")
        ax.set_title(r["model"])
        ax.set_xlabel("Frequency band")
        ax.set_ylabel("Count of high-SI heads")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("RoPE Frequency Band Distribution of High-SI Heads", fontsize=13)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(str(out_path), format="pdf", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"[E11] Saved freq band figure: {out_path}", flush=True)


if __name__ == "__main__":
    main()
