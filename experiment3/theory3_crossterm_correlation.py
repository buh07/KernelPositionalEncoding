#!/usr/bin/env python3
"""
Experiment 3, Theory 3: Cross-term Correlation
================================================

Hypothesis
----------
The functionally important RoPE pairs are the ones with the strongest
content-position coupling (Hadamard product interaction), not the ones with
the "right" frequency for a given positional range.

Centering removes content-position coupling by subtracting per-query means
and per-key means (double centering).  The gap

    gap[Delta] = g_raw[Delta] - g_centered[Delta]

measures how much content-position interaction exists at each position
offset Delta.  If a RoPE pair is functionally important because of its
content-position coupling, the gap signal should have strong energy at
that pair's frequency — and that energy should correlate with the pair's
ablation effect measured in Experiment 2.

Protocol
--------
Phase 1 — Compute per-head raw and centered diagonal kernels from wiki text.
Phase 2 — Project the gap signal onto each RoPE pair's frequency.
Phase 3 — Load pair ablation effects from Experiment 2.
Phase 4 — Correlation analysis (Pearson, Spearman, partial).
Phase 5 — Report.

Usage
-----
    python scripts/experiment3_theory3_crossterm_correlation.py --model llama-3.1-8b --device cuda:0
    python scripts/experiment3_theory3_crossterm_correlation.py --model olmo-2-7b --device cuda:1
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import numpy as np
import pandas as pd

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment1.shift_kernels import RoPEEstimator, KernelFit
from experiment3.stats_utils import (
    fisher_z_mean,
    holm_adjust,
    partial_correlation_with_intercept,
)
# normalize_logits_for_norm removed: it pre-centers RMSNorm logits,
# which zeroed the cross-term we are trying to measure.  Raw logits
# are now used directly; double_center() handles the centered path.
from shared.specs import ModelSpec

try:
    from shared.attention.adapters import get_adapter
    from shared.models.loading import load_model, load_tokenizer
except Exception:  # pragma: no cover - optional for analysis-only environments
    get_adapter = None
    load_model = None
    load_tokenizer = None

# ── constants ────────────────────────────────────────────────────────────────
NUM_SEQUENCES = 50
SEQ_LEN = 512
ROPE_BASE = 10000.0
QUARTILE = 0.25  # top/bottom 25 % for high-SI / low-SI classification
PAIR_INDICES = [0, 8, 16, 24, 32, 40, 48, 56]  # pairs probed in Experiment 2

MODELS: dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec(
        name="llama-3.1-8b",
        hf_id="meta-llama/Meta-Llama-3.1-8B",
        norm="RMSNorm",
        pe_scheme="RoPE",
        notes="Llama 3.1 8B (GQA: 32Q/8KV heads).",
    ),
    "olmo-2-7b": ModelSpec(
        name="olmo-2-7b",
        hf_id="allenai/OLMo-2-1124-7B",
        norm="LayerNorm",
        pe_scheme="RoPE",
        notes="OLMo 2 7B.",
        download_kwargs=(("torch_dtype", "bfloat16"),),
    ),
}

PAIR_EFFECTS_PATH = (
    "results/experiment2/quick/quick_pair_expand_pivot_20260314_0043_v2"
    "/reports/pair_unified/pair_effects_merged.parquet"
)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: RAW AND CENTERED DIAGONAL KERNELS
# ═══════════════════════════════════════════════════════════════════════════════

def load_sequences(model_name: str, num_sequences: int, seq_len: int) -> list[list[int]]:
    """Load tokenized wiki sequences."""
    data_dir = Path("data/experiment1/wiki40b_en_pre2019") / model_name
    candidates = sorted(data_dir.rglob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No tokenized JSONL found in {data_dir}")
    sequences: list[list[int]] = []
    for jsonl_path in candidates:
        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                tokens = row.get("tokens", row.get("input_ids", []))
                if len(tokens) >= seq_len:
                    sequences.append(tokens[:seq_len])
                if len(sequences) >= num_sequences:
                    break
        if len(sequences) >= num_sequences:
            break
    if len(sequences) < num_sequences:
        print(f"  WARNING: Only found {len(sequences)} sequences (wanted {num_sequences})")
    return sequences


def double_center(logits: torch.Tensor) -> torch.Tensor:
    """Double-center a [seq, seq] logit matrix.

    Subtracts row means, column means, and adds back the global mean so that
    the resulting matrix has zero row sums and zero column sums.  This removes
    content-position coupling (the Hadamard cross-term).
    """
    # logits: [S, S], float
    row_mean = logits.mean(dim=1, keepdim=True)   # [S, 1]
    col_mean = logits.mean(dim=0, keepdim=True)   # [1, S]
    global_mean = logits.mean()                    # scalar
    return logits - row_mean - col_mean + global_mean


def extract_diagonal_means(matrix: torch.Tensor, max_delta: int) -> np.ndarray:
    """Extract the mean value along each lower diagonal Delta = 1 .. max_delta.

    For a causal attention matrix, diagonal Delta contains entries
    matrix[i, i - Delta] for i >= Delta.
    """
    S = matrix.shape[0]
    means = np.zeros(max_delta, dtype=np.float64)
    for delta in range(1, max_delta + 1):
        if delta >= S:
            break
        diag = torch.diagonal(matrix, offset=-delta)  # entries m[i, i-delta]
        means[delta - 1] = diag.float().mean().item()
    return means


def extract_diagonal_means_batched(logits: torch.Tensor, max_delta: int) -> torch.Tensor:
    """Vectorized diagonal means for batched logits [L, H, S, S].

    Returns [L, H, max_delta] on the same device as logits.
    """
    n_layers, n_heads, seq_len, _ = logits.shape
    out = torch.empty(
        (n_layers, n_heads, max_delta),
        dtype=torch.float32,
        device=logits.device,
    )
    for delta in range(1, max_delta + 1):
        if delta >= seq_len:
            out[:, :, delta - 1] = 0.0
            continue
        diag = torch.diagonal(logits, offset=-delta, dim1=-2, dim2=-1)  # [L, H, S-delta]
        out[:, :, delta - 1] = diag.mean(dim=-1)
    return out


def compute_gap_kernels(
    model,
    adapter,
    _model_spec: ModelSpec,
    device: str,
    sequences: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Compute per-head g_raw, g_centered, and gap diagonal profiles.

    Returns
    -------
    g_raw_all     : [n_layers, n_heads, max_delta]
    g_centered_all: [n_layers, n_heads, max_delta]
    gap_all       : [n_layers, n_heads, max_delta]  (= g_raw - g_centered)
    n_layers      : int
    n_heads       : int
    """
    max_delta = SEQ_LEN - 1

    # Accumulators — will be initialised on first sequence
    g_raw_accum: np.ndarray | None = None
    g_cent_accum: np.ndarray | None = None
    n_layers = n_heads = 0
    n_processed = 0
    phase_start = time.time()
    recent_seq_times: deque[float] = deque(maxlen=8)

    for seq_idx, tokens in enumerate(sequences):
        seq_start = time.time()
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        capture = adapter.capture(
            model,
            input_ids=input_ids,
            include_logits=True,
            return_token_logits=False,
            capture_attention=True,
            output_device=device,
        )

        if capture.logits is None:
            print(f"  Seq {seq_idx}: no logits captured, skipping")
            del capture, input_ids
            continue

        # capture.logits: [layers, heads, seq, seq] (raw pre-mask QK^T expected)
        logits = capture.logits.to(dtype=torch.float32)
        if not torch.isfinite(logits).all():
            n_bad = int((~torch.isfinite(logits)).sum().item())
            raise RuntimeError(
                "Non-finite attention logits captured in Theory 3. "
                f"Found {n_bad} non-finite entries. "
                "This typically means post-mask logits were captured; Theory 3 "
                "requires raw pre-mask attention logits."
            )
        n_layers, n_heads = logits.shape[0], logits.shape[1]

        if g_raw_accum is None:
            g_raw_accum = np.zeros((n_layers, n_heads, max_delta), dtype=np.float64)
            g_cent_accum = np.zeros((n_layers, n_heads, max_delta), dtype=np.float64)

        # Raw diagonal means — use truly raw logits (NO norm-specific centering)
        # so that the content-position cross-term is preserved.
        g_raw_seq = extract_diagonal_means_batched(logits, max_delta)

        # Centered diagonal means — always double-center explicitly.
        row_mean = logits.mean(dim=-1, keepdim=True)   # [L, H, S, 1]
        col_mean = logits.mean(dim=-2, keepdim=True)   # [L, H, 1, S]
        global_mean = logits.mean(dim=(-2, -1), keepdim=True)  # [L, H, 1, 1]
        centered = logits - row_mean - col_mean + global_mean
        g_cent_seq = extract_diagonal_means_batched(centered, max_delta)

        # Transfer only reduced [L, H, max_delta] to CPU accumulators.
        g_raw_accum += g_raw_seq.to(device="cpu", dtype=torch.float64).numpy()
        g_cent_accum += g_cent_seq.to(device="cpu", dtype=torch.float64).numpy()

        n_processed += 1
        seq_elapsed = time.time() - seq_start
        recent_seq_times.append(seq_elapsed)
        elapsed_total = time.time() - phase_start
        avg_total = elapsed_total / max(1, n_processed)
        remaining = len(sequences) - n_processed
        eta_sec = remaining * avg_total
        rolling = sum(recent_seq_times) / max(1, len(recent_seq_times))

        if (seq_idx + 1) % 5 == 0 or seq_idx == 0 or (seq_idx + 1) == len(sequences):
            print(
                f"  Gap kernels: {seq_idx + 1}/{len(sequences)} sequences | "
                f"last={seq_elapsed:.1f}s seq | rolling={rolling:.1f}s/seq | "
                f"avg={avg_total:.1f}s/seq | eta {eta_sec/60.0:.1f}m"
            )

        del capture, input_ids, logits, g_raw_seq, g_cent_seq, centered

    if n_processed == 0:
        raise RuntimeError("No sequences were processed — check data path.")

    g_raw_all = g_raw_accum / n_processed
    g_cent_all = g_cent_accum / n_processed
    gap_all = g_raw_all - g_cent_all

    print(f"  Processed {n_processed} sequences; {n_layers} layers x {n_heads} heads")
    return g_raw_all, g_cent_all, gap_all, n_layers, n_heads


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: PROJECT GAP ONTO ROPE PAIR FREQUENCIES
# ═══════════════════════════════════════════════════════════════════════════════

def rope_frequency(pair_idx: int, head_dim: int, base: float = ROPE_BASE) -> float:
    """Compute the RoPE angular frequency for pair index p: theta_p = 1 / base^(2p/d)."""
    return 1.0 / (base ** (2.0 * pair_idx / head_dim))


def project_gap_onto_rope_freqs(
    gap_all: np.ndarray,
    head_dim: int,
    n_layers: int,
    n_heads: int,
) -> pd.DataFrame:
    """Compute per-pair cross-term energy for each head.

    For each head and each RoPE pair p with frequency theta_p:
        energy[p] = |sum_Delta gap[Delta] * cos(theta_p * Delta)|^2
                  + |sum_Delta gap[Delta] * sin(theta_p * Delta)|^2

    Returns a DataFrame with columns:
        layer, head, pair_idx, crossterm_energy, theta
    """
    max_delta = gap_all.shape[2]
    deltas = np.arange(1, max_delta + 1, dtype=np.float64)
    n_pairs = head_dim // 2

    rows: list[dict[str, Any]] = []

    for p in range(n_pairs):
        theta = rope_frequency(p, head_dim)
        cos_basis = np.cos(theta * deltas)  # [max_delta]
        sin_basis = np.sin(theta * deltas)  # [max_delta]

        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                gap = gap_all[layer_idx, head_idx]  # [max_delta]
                proj_cos = np.dot(gap, cos_basis)
                proj_sin = np.dot(gap, sin_basis)
                energy = proj_cos ** 2 + proj_sin ** 2

                rows.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "pair_idx": p,
                    "crossterm_energy": float(energy),
                    "theta": float(theta),
                })

    return pd.DataFrame(rows)


def aggregate_crossterm_by_pair(
    crossterm_df: pd.DataFrame,
    r2_df: pd.DataFrame | None,
    pair_indices: list[int],
) -> pd.DataFrame:
    """Aggregate cross-term energy to per-pair summaries.

    Computes:
      - Overall mean cross-term energy per pair (across all heads)
      - Mean cross-term energy among high-SI heads (top 25 % R²)
      - Mean cross-term energy among low-SI heads (bottom 25 % R²)
    """
    # Filter to the pair indices actually ablated in Exp 2
    ct_sub = crossterm_df[crossterm_df["pair_idx"].isin(pair_indices)].copy()

    # Overall mean per pair
    overall = ct_sub.groupby("pair_idx")["crossterm_energy"].mean().reset_index()
    overall.columns = ["pair_idx", "crossterm_energy_mean"]

    if r2_df is not None and not r2_df.empty:
        # Merge R² into cross-term data
        ct_merged = pd.merge(ct_sub, r2_df, on=["layer", "head"], how="inner")

        r2_q75 = ct_merged["mean_r2"].quantile(1.0 - QUARTILE)
        r2_q25 = ct_merged["mean_r2"].quantile(QUARTILE)

        high_si = ct_merged[ct_merged["mean_r2"] >= r2_q75]
        low_si = ct_merged[ct_merged["mean_r2"] <= r2_q25]

        high_agg = high_si.groupby("pair_idx")["crossterm_energy"].mean().reset_index()
        high_agg.columns = ["pair_idx", "crossterm_energy_high_si"]

        low_agg = low_si.groupby("pair_idx")["crossterm_energy"].mean().reset_index()
        low_agg.columns = ["pair_idx", "crossterm_energy_low_si"]

        overall = pd.merge(overall, high_agg, on="pair_idx", how="left")
        overall = pd.merge(overall, low_agg, on="pair_idx", how="left")
    else:
        overall["crossterm_energy_high_si"] = np.nan
        overall["crossterm_energy_low_si"] = np.nan

    return overall.sort_values("pair_idx").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: LOAD PAIR ABLATION EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

def load_pair_effects(model_name: str) -> pd.DataFrame:
    """Load per-pair ablation effects from Experiment 2."""
    path = Path(PAIR_EFFECTS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Pair effects not found at {path}")
    df = pd.read_parquet(path)
    df_model = df[df["model"] == model_name].copy()
    if df_model.empty:
        raise ValueError(f"No pair effects for model '{model_name}' in {path}")
    print(f"  Loaded {len(df_model)} pair-effect rows for {model_name}")
    print(f"  Tasks: {sorted(df_model['task'].unique())}")
    print(f"  Spans: {sorted(df_model['span'].unique())}")
    print(f"  Pairs: {sorted(df_model['pair_idx'].unique())}")
    return df_model


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 (supplementary): LOAD TOTAL ENERGY FROM 3.2c
# ═══════════════════════════════════════════════════════════════════════════════

def load_freq_profile(model_name: str) -> pd.DataFrame:
    """Load freq_profile.parquet from the 3.2c frequency decomposition."""
    path = Path("results/experiment3/rope_freq_per_head") / model_name / "freq_profile.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Frequency profile not found at {path}")
    return pd.read_parquet(path)


def strict_causal_pair_energy_prefix(q_pair: torch.Tensor, k_pair: torch.Tensor) -> torch.Tensor:
    """Exact strict-causal pair energy per head via prefix covariance.

    Given q_pair, k_pair with shape [L, H, S, Dp], computes:
      E[l,h] = sum_i q_i^T (sum_{j<i} k_j k_j^T) q_i
    which is algebraically equal to:
      sum_{i>j} (q_i^T k_j)^2.
    """
    if q_pair.shape != k_pair.shape:
        raise ValueError(f"q/k shape mismatch: {q_pair.shape} vs {k_pair.shape}")
    if q_pair.ndim != 4:
        raise ValueError(f"Expected [L,H,S,D], got shape={q_pair.shape}")

    n_layers, n_heads, seq_len, pair_dim = q_pair.shape
    flat = n_layers * n_heads
    q_flat = q_pair.reshape(flat, seq_len, pair_dim)  # [N, S, D]
    k_flat = k_pair.reshape(flat, seq_len, pair_dim)  # [N, S, D]

    kk = k_flat.unsqueeze(-1) * k_flat.unsqueeze(-2)  # [N, S, D, D]
    prefix = torch.cumsum(kk, dim=1) - kk             # exclusive prefix sum
    per_pos = torch.einsum("nsd,nsde,nse->ns", q_flat, prefix, q_flat)  # [N, S]
    return per_pos.sum(dim=1).reshape(n_layers, n_heads)  # [L, H]


def compute_per_pair_total_energy(
    model,
    adapter,
    device: str,
    sequences: list[list[int]],
    head_dim: int,
    pair_indices: list[int],
) -> pd.DataFrame:
    """Compute total (not cross-term) per-pair energy from Q/K projections.

    This duplicates part of 3.2c's logic but restricted to the 8 ablated pairs,
    for use as a confound control variable.  If the 3.2c data already exists we
    derive it from there instead (see caller).
    """
    n_pairs_full = head_dim // 2
    energy_accum: dict[int, float] = {p: 0.0 for p in pair_indices}
    n_processed = 0
    n_layers: int | None = None
    n_heads: int | None = None
    phase_start = time.time()
    recent_seq_times: deque[float] = deque(maxlen=8)

    for seq_idx, tokens in enumerate(sequences):
        seq_start = time.time()
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        capture = adapter.capture(
            model,
            input_ids=input_ids,
            include_logits=False,
            return_token_logits=False,
            capture_attention=True,
            output_device=device,
        )

        q = capture.q.to(dtype=torch.float32)  # [L, H, S, D]
        k = capture.k.to(dtype=torch.float32)
        if not torch.isfinite(q).all() or not torch.isfinite(k).all():
            raise RuntimeError("Non-finite q/k captured in Theory 3 total-energy pass.")
        n_layers = int(q.shape[0])
        n_heads = int(q.shape[1])

        for p in pair_indices:
            if p >= n_pairs_full:
                continue
            q_pair = q[:, :, :, 2 * p: 2 * p + 2]  # [L, H, S, 2]
            k_pair = k[:, :, :, 2 * p: 2 * p + 2]
            per_head_energy = strict_causal_pair_energy_prefix(q_pair, k_pair)  # [L, H]
            energy_accum[p] += float(per_head_energy.sum().item())

        n_processed += 1
        seq_elapsed = time.time() - seq_start
        recent_seq_times.append(seq_elapsed)
        elapsed_total = time.time() - phase_start
        avg_total = elapsed_total / max(1, n_processed)
        remaining = len(sequences) - n_processed
        eta_sec = remaining * avg_total
        rolling = sum(recent_seq_times) / max(1, len(recent_seq_times))

        if (seq_idx + 1) % 5 == 0 or seq_idx == 0 or (seq_idx + 1) == len(sequences):
            print(
                f"  Total pair energy: {seq_idx + 1}/{len(sequences)} sequences | "
                f"last={seq_elapsed:.1f}s seq | rolling={rolling:.1f}s/seq | "
                f"avg={avg_total:.1f}s/seq | eta {eta_sec/60.0:.1f}m"
            )

        del capture, input_ids, q, k

    denom = max(1, n_processed * max(1, (n_layers or 1) * (n_heads or 1)))
    rows = [
        {
            "pair_idx": p,
            # Per-head-per-sequence mean energy; scale is now interpretable.
            "total_pair_energy": float(energy_accum[p] / denom),
            "total_pair_energy_sum": float(energy_accum[p]),
        }
        for p in pair_indices
    ]
    return pd.DataFrame(rows)


def resolve_artifact_actions(
    *,
    gap_exists: bool,
    total_exists: bool,
    skip_phase1: bool,
    force_recompute_gap_kernels: bool,
    force_recompute_total_energy: bool,
) -> dict[str, bool]:
    """Resolve resume/recompute actions for Theory 3 artifacts."""
    if skip_phase1 and force_recompute_gap_kernels:
        raise ValueError("Cannot use --skip-phase1 together with --force-recompute-gap-kernels")

    if skip_phase1:
        if not gap_exists:
            raise FileNotFoundError(
                "Requested --skip-phase1 but gap_kernels.npz does not exist."
            )
        load_gap = True
        compute_gap = False
    elif force_recompute_gap_kernels:
        load_gap = False
        compute_gap = True
    elif gap_exists:
        load_gap = True
        compute_gap = False
    else:
        load_gap = False
        compute_gap = True

    if force_recompute_total_energy:
        load_total = False
        compute_total = True
    elif total_exists:
        load_total = True
        compute_total = False
    else:
        load_total = False
        compute_total = True

    return {
        "load_gap": load_gap,
        "compute_gap": compute_gap,
        "load_total": load_total,
        "compute_total": compute_total,
        "need_model": (compute_gap or compute_total),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float, int]:
    """Compute partial correlation r(x, y | z) via residual regression.

    Returns (partial_r, p_value, df). Falls back to NaNs if degenerate.
    """
    res = partial_correlation_with_intercept(x, y, z)
    return float(res.r), float(res.p_value), int(res.df)


def run_correlation_analysis(
    crossterm_agg: pd.DataFrame,
    pair_effects: pd.DataFrame,
    total_energy_df: pd.DataFrame | None,
    model_name: str,
) -> dict[str, Any]:
    """Correlate per-pair cross-term energy with ablation effect.

    Runs separately for each (task, span) combination.
    """
    from scipy.stats import pearsonr, spearmanr

    results: dict[str, Any] = {"model": model_name, "per_task_span": [], "aggregate": {}}

    # Merge cross-term with total energy (if available)
    ct = crossterm_agg.copy()
    if total_energy_df is not None and not total_energy_df.empty:
        ct = pd.merge(ct, total_energy_df, on="pair_idx", how="left")
    else:
        ct["total_pair_energy"] = np.nan

    # Iterate over each (task, span) in the pair effects data
    task_span_groups = pair_effects.groupby(["task", "span"])

    all_pearson_raw = []
    all_spearman_raw = []

    for (task, span), effects_group in task_span_groups:
        # Merge on pair_idx
        merged = pd.merge(
            effects_group[["pair_idx", "effect_raw", "effect_over_headroom"]],
            ct,
            on="pair_idx",
            how="inner",
        )

        if len(merged) < 4:
            print(f"  WARN: task={task} span={span}: only {len(merged)} pairs after merge, skipping")
            continue

        x_ct = merged["crossterm_energy_mean"].values
        y_raw = merged["effect_raw"].values
        y_hdr = merged["effect_over_headroom"].values

        entry: dict[str, Any] = {
            "task": task,
            "span": int(span),
            "n_pairs": len(merged),
        }

        # Pearson: cross-term vs effect_raw
        r_p, p_p = pearsonr(x_ct, y_raw)
        entry["pearson_ct_vs_raw"] = float(r_p)
        entry["pearson_ct_vs_raw_p"] = float(p_p)

        # Spearman: cross-term vs effect_raw
        r_s, p_s = spearmanr(x_ct, y_raw)
        entry["spearman_ct_vs_raw"] = float(r_s)
        entry["spearman_ct_vs_raw_p"] = float(p_s)

        # Pearson: cross-term vs effect_over_headroom
        r_ph, p_ph = pearsonr(x_ct, y_hdr)
        entry["pearson_ct_vs_headroom"] = float(r_ph)
        entry["pearson_ct_vs_headroom_p"] = float(p_ph)

        # Spearman: cross-term vs effect_over_headroom
        r_sh, p_sh = spearmanr(x_ct, y_hdr)
        entry["spearman_ct_vs_headroom"] = float(r_sh)
        entry["spearman_ct_vs_headroom_p"] = float(p_sh)

        # Total energy vs effect_raw (confound check)
        if not np.all(np.isnan(merged["total_pair_energy"].values)):
            z_tot = merged["total_pair_energy"].values
            r_te, p_te = pearsonr(z_tot, y_raw)
            entry["pearson_total_vs_raw"] = float(r_te)
            entry["pearson_total_vs_raw_p"] = float(p_te)

            r_tes, p_tes = spearmanr(z_tot, y_raw)
            entry["spearman_total_vs_raw"] = float(r_tes)
            entry["spearman_total_vs_raw_p"] = float(p_tes)

            # Partial correlation: cross-term vs effect, controlling for total energy
            pr, pp, pdf = partial_correlation(x_ct, y_raw, z_tot)
            entry["partial_ct_vs_raw_ctrl_total"] = float(pr)
            entry["partial_ct_vs_raw_ctrl_total_p"] = float(pp)
            entry["partial_ct_vs_raw_ctrl_total_df"] = int(pdf)

            pr_h, pp_h, pdf_h = partial_correlation(x_ct, y_hdr, z_tot)
            entry["partial_ct_vs_headroom_ctrl_total"] = float(pr_h)
            entry["partial_ct_vs_headroom_ctrl_total_p"] = float(pp_h)
            entry["partial_ct_vs_headroom_ctrl_total_df"] = int(pdf_h)
        else:
            entry["pearson_total_vs_raw"] = float("nan")
            entry["pearson_total_vs_raw_p"] = float("nan")
            entry["spearman_total_vs_raw"] = float("nan")
            entry["spearman_total_vs_raw_p"] = float("nan")
            entry["partial_ct_vs_raw_ctrl_total"] = float("nan")
            entry["partial_ct_vs_raw_ctrl_total_p"] = float("nan")
            entry["partial_ct_vs_raw_ctrl_total_df"] = -1
            entry["partial_ct_vs_headroom_ctrl_total"] = float("nan")
            entry["partial_ct_vs_headroom_ctrl_total_p"] = float("nan")
            entry["partial_ct_vs_headroom_ctrl_total_df"] = -1

        # Per-SI-group cross-term vs effect
        for si_col, si_label in [
            ("crossterm_energy_high_si", "high_si"),
            ("crossterm_energy_low_si", "low_si"),
        ]:
            if si_col in merged.columns and not merged[si_col].isna().all():
                x_si = merged[si_col].values
                if np.std(x_si) > 1e-12:
                    rr, rp = pearsonr(x_si, y_raw)
                    entry[f"pearson_{si_label}_ct_vs_raw"] = float(rr)
                    entry[f"pearson_{si_label}_ct_vs_raw_p"] = float(rp)
                else:
                    entry[f"pearson_{si_label}_ct_vs_raw"] = float("nan")
                    entry[f"pearson_{si_label}_ct_vs_raw_p"] = float("nan")
            else:
                entry[f"pearson_{si_label}_ct_vs_raw"] = float("nan")
                entry[f"pearson_{si_label}_ct_vs_raw_p"] = float("nan")

        # Multiple-comparison correction within each (task, span) family.
        family_pvals = {
            "pearson_ct_vs_raw": entry["pearson_ct_vs_raw_p"],
            "spearman_ct_vs_raw": entry["spearman_ct_vs_raw_p"],
            "pearson_ct_vs_headroom": entry["pearson_ct_vs_headroom_p"],
            "spearman_ct_vs_headroom": entry["spearman_ct_vs_headroom_p"],
            "pearson_total_vs_raw": entry.get("pearson_total_vs_raw_p", float("nan")),
            "spearman_total_vs_raw": entry.get("spearman_total_vs_raw_p", float("nan")),
            "partial_ct_vs_raw_ctrl_total": entry.get("partial_ct_vs_raw_ctrl_total_p", float("nan")),
            "partial_ct_vs_headroom_ctrl_total": entry.get("partial_ct_vs_headroom_ctrl_total_p", float("nan")),
            "pearson_high_si_ct_vs_raw": entry.get("pearson_high_si_ct_vs_raw_p", float("nan")),
            "pearson_low_si_ct_vs_raw": entry.get("pearson_low_si_ct_vs_raw_p", float("nan")),
        }
        family_holm = holm_adjust(family_pvals)
        for test_name, p_adj in family_holm.items():
            entry[f"{test_name}_p_holm"] = float(p_adj)
        entry["multiplicity_policy"] = "holm within task/span family"

        results["per_task_span"].append(entry)
        all_pearson_raw.append(r_p)
        all_spearman_raw.append(r_s)

    # Aggregate across task/span
    if all_pearson_raw:
        min_pairs = int(min(e["n_pairs"] for e in results["per_task_span"]))
        max_pairs = int(max(e["n_pairs"] for e in results["per_task_span"]))
        results["aggregate"] = {
            "mean_pearson_ct_vs_raw_raw_mean": float(np.mean(all_pearson_raw)),
            "mean_pearson_ct_vs_raw_fisher_z": float(fisher_z_mean(all_pearson_raw)),
            "median_pearson_ct_vs_raw": float(np.median(all_pearson_raw)),
            "mean_spearman_ct_vs_raw_raw_mean": float(np.mean(all_spearman_raw)),
            "mean_spearman_ct_vs_raw_fisher_z": float(fisher_z_mean(all_spearman_raw)),
            "median_spearman_ct_vs_raw": float(np.median(all_spearman_raw)),
            "n_task_span_combos": len(all_pearson_raw),
            "n_pairs_per_combo_min": min_pairs,
            "n_pairs_per_combo_max": max_pairs,
            "small_n_limitation": (
                "Each task/span correlation is based on only 8 RoPE pairs; "
                "treat p-values as descriptive."
            ),
            "multiplicity_policy": "holm within task/span family",
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: LOAD R² DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_r2_data(model_name: str) -> pd.DataFrame | None:
    """Load per-head mean R² from Experiment 3.1."""
    path = Path("results/experiment3/theory1_si_circuits") / model_name / "per_sequence_r2.parquet"
    if not path.exists():
        print(f"  WARNING: No R² data at {path}, skipping SI-group analysis")
        return None
    r2_df = pd.read_parquet(path)
    mean_r2 = r2_df.groupby(["layer", "head"])["r2"].mean().reset_index()
    mean_r2.columns = ["layer", "head", "mean_r2"]
    return mean_r2


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def sig_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if np.isnan(p):
        return "  "
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "** "
    if p < 0.05:
        return "*  "
    return "ns "


def print_report(
    analysis: dict[str, Any],
    crossterm_agg: pd.DataFrame,
    model_name: str,
) -> None:
    """Print a comprehensive human-readable report to stdout."""

    print(f"\n{'=' * 80}")
    print(f"RESULTS — Experiment 3, Theory 3: Cross-term Correlation — {model_name}")
    print(f"{'=' * 80}")

    # Cross-term energy per pair
    print(f"\n  Per-pair cross-term energy (averaged across all heads):")
    print(f"    {'Pair':>4s}  {'CrosstermEnergy':>16s}  {'HighSI':>14s}  {'LowSI':>14s}")
    print(f"    {'-' * 56}")
    for _, row in crossterm_agg.iterrows():
        hi = f"{row['crossterm_energy_high_si']:.6f}" if not np.isnan(row.get("crossterm_energy_high_si", float("nan"))) else "N/A"
        lo = f"{row['crossterm_energy_low_si']:.6f}" if not np.isnan(row.get("crossterm_energy_low_si", float("nan"))) else "N/A"
        print(f"    {int(row['pair_idx']):4d}  {row['crossterm_energy_mean']:16.6f}  {hi:>14s}  {lo:>14s}")

    # Per task/span correlation results
    print(f"\n  Correlation: cross-term energy vs. pair ablation effect")
    print(f"    {'Task':<30s} {'Span':>4s}  {'r_P':>7s} {'p_P':>8s} {'sig':>3s}  {'r_S':>7s} {'p_S':>8s} {'sig':>3s}  {'r_part':>7s} {'p_part':>8s} {'df':>4s} {'sig':>3s}")
    print(f"    {'-' * 106}")

    for entry in analysis["per_task_span"]:
        r_p = entry["pearson_ct_vs_raw"]
        p_p = entry["pearson_ct_vs_raw_p"]
        p_p_sig = entry.get("pearson_ct_vs_raw_p_holm", p_p)
        r_s = entry["spearman_ct_vs_raw"]
        p_s = entry["spearman_ct_vs_raw_p"]
        p_s_sig = entry.get("spearman_ct_vs_raw_p_holm", p_s)
        r_part = entry.get("partial_ct_vs_raw_ctrl_total", float("nan"))
        p_part = entry.get("partial_ct_vs_raw_ctrl_total_p", float("nan"))
        p_part_sig = entry.get("partial_ct_vs_raw_ctrl_total_p_holm", p_part)
        df_part = entry.get("partial_ct_vs_raw_ctrl_total_df", -1)
        print(
            f"    {entry['task']:<30s} {entry['span']:4d}  "
            f"{r_p:+7.4f} {p_p:8.4f} {sig_stars(p_p_sig)}  "
            f"{r_s:+7.4f} {p_s:8.4f} {sig_stars(p_s_sig)}  "
            f"{r_part:+7.4f} {p_part:8.4f} {df_part:4d} {sig_stars(p_part_sig)}"
        )

    # Headroom-normalised
    print(f"\n  Correlation: cross-term energy vs. effect_over_headroom")
    print(f"    {'Task':<30s} {'Span':>4s}  {'r_P':>7s} {'p_P':>8s} {'sig':>3s}  {'r_S':>7s} {'p_S':>8s} {'sig':>3s}")
    print(f"    {'-' * 80}")
    for entry in analysis["per_task_span"]:
        r_p = entry["pearson_ct_vs_headroom"]
        p_p = entry["pearson_ct_vs_headroom_p"]
        p_p_sig = entry.get("pearson_ct_vs_headroom_p_holm", p_p)
        r_s = entry["spearman_ct_vs_headroom"]
        p_s = entry["spearman_ct_vs_headroom_p"]
        p_s_sig = entry.get("spearman_ct_vs_headroom_p_holm", p_s)
        print(
            f"    {entry['task']:<30s} {entry['span']:4d}  "
            f"{r_p:+7.4f} {p_p:8.4f} {sig_stars(p_p_sig)}  "
            f"{r_s:+7.4f} {p_s:8.4f} {sig_stars(p_s_sig)}"
        )

    # Confound check: total energy vs effect
    print(f"\n  Confound check: total pair energy vs. effect_raw")
    print(f"    {'Task':<30s} {'Span':>4s}  {'r_P':>7s} {'p_P':>8s} {'sig':>3s}")
    print(f"    {'-' * 60}")
    for entry in analysis["per_task_span"]:
        r_t = entry.get("pearson_total_vs_raw", float("nan"))
        p_t = entry.get("pearson_total_vs_raw_p", float("nan"))
        p_t_sig = entry.get("pearson_total_vs_raw_p_holm", p_t)
        print(
            f"    {entry['task']:<30s} {entry['span']:4d}  "
            f"{r_t:+7.4f} {p_t:8.4f} {sig_stars(p_t_sig)}"
        )

    # SI-group cross-term correlations
    print(f"\n  SI-group cross-term vs. effect_raw:")
    print(f"    {'Task':<30s} {'Span':>4s}  {'HighSI r_P':>10s} {'p':>8s}  {'LowSI r_P':>10s} {'p':>8s}")
    print(f"    {'-' * 80}")
    for entry in analysis["per_task_span"]:
        rh = entry.get("pearson_high_si_ct_vs_raw", float("nan"))
        ph = entry.get("pearson_high_si_ct_vs_raw_p", float("nan"))
        rl = entry.get("pearson_low_si_ct_vs_raw", float("nan"))
        pl = entry.get("pearson_low_si_ct_vs_raw_p", float("nan"))
        print(
            f"    {entry['task']:<30s} {entry['span']:4d}  "
            f"{rh:+10.4f} {ph:8.4f}  {rl:+10.4f} {pl:8.4f}"
        )

    # Aggregate summary
    if analysis.get("aggregate"):
        agg = analysis["aggregate"]
        print(f"\n  Aggregate across all task/span combinations (n={agg['n_task_span_combos']}):")
        print(f"    Mean  Pearson(crossterm, effect_raw) [Fisher-z]: {agg['mean_pearson_ct_vs_raw_fisher_z']:+.4f}")
        print(f"    Median Pearson(crossterm, effect_raw):  {agg['median_pearson_ct_vs_raw']:+.4f}")
        print(f"    Mean  Spearman(crossterm, effect_raw) [Fisher-z]: {agg['mean_spearman_ct_vs_raw_fisher_z']:+.4f}")
        print(f"    Median Spearman(crossterm, effect_raw): {agg['median_spearman_ct_vs_raw']:+.4f}")
        print(f"    Pair-count per combo: min={agg['n_pairs_per_combo_min']}, max={agg['n_pairs_per_combo_max']}")
        print(f"    NOTE: {agg['small_n_limitation']}")
        print(f"    MCC: {agg.get('multiplicity_policy', 'none')}; significance stars use adjusted p-values")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3, Theory 3: Cross-term Correlation"
    )
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/experiment3/theory3_crossterm")
    parser.add_argument("--num-sequences", type=int, default=NUM_SEQUENCES)
    parser.add_argument(
        "--skip-phase1", action="store_true",
        help="Skip Phase 1; load saved gap kernels from output dir",
    )
    parser.add_argument(
        "--force-recompute-gap-kernels",
        action="store_true",
        help="Recompute and overwrite gap_kernels.npz even if artifact exists.",
    )
    parser.add_argument(
        "--force-recompute-total-energy",
        action="store_true",
        help="Recompute and overwrite total_pair_energy.parquet even if artifact exists.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_spec = MODELS[args.model]

    print(f"{'=' * 80}")
    print(f"Experiment 3, Theory 3: Cross-term Correlation — {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 80}")

    # ── Phase 1: Compute gap kernels ─────────────────────────────────────────
    gap_npz = output_dir / "gap_kernels.npz"
    total_energy_parquet = output_dir / "total_pair_energy.parquet"
    actions = resolve_artifact_actions(
        gap_exists=gap_npz.exists(),
        total_exists=total_energy_parquet.exists(),
        skip_phase1=args.skip_phase1,
        force_recompute_gap_kernels=args.force_recompute_gap_kernels,
        force_recompute_total_energy=args.force_recompute_total_energy,
    )

    print(
        "  Artifact actions: "
        f"load_gap={actions['load_gap']} compute_gap={actions['compute_gap']} | "
        f"load_total={actions['load_total']} compute_total={actions['compute_total']}"
    )

    if actions["load_gap"]:
        print(f"\n[1/5] Phase 1: SKIPPED (loading saved gap kernels from {gap_npz})")
        data = np.load(gap_npz)
        g_raw_all = data["g_raw"]
        g_cent_all = data["g_centered"]
        gap_all = data["gap"]
        n_layers, n_heads = g_raw_all.shape[0], g_raw_all.shape[1]
        print(f"  Loaded: {n_layers} layers x {n_heads} heads, max_delta={g_raw_all.shape[2]}")
    else:
        print("\n[1/5] Phase 1: No reusable gap_kernels artifact; recomputation required.")
        g_raw_all = g_cent_all = gap_all = None
        n_layers = n_heads = 0

    loader = model = tokenizer = adapter = sequences = None
    if actions["need_model"]:
        if actions["compute_gap"]:
            print(f"\n[1/5] Phase 1: Computing per-head gap kernels ({args.num_sequences} sequences)...")
        else:
            print(
                f"\n[1/5] Phase 1: Reusing gap kernels; loading model only for total-energy "
                f"recompute ({args.num_sequences} sequences)..."
            )
        print("  Loading model and tokenizer...")
        loader = load_model(model_spec)
        model = loader.model.to(args.device)
        model.eval()
        tokenizer = load_tokenizer(model_spec)
        adapter = get_adapter(model_spec)
        adapter.register(model)
        print(f"  Model loaded: {model_spec.hf_id}")
        print(f"  Config: {model.config.num_hidden_layers} layers, "
              f"{model.config.num_attention_heads} query heads, "
              f"head_dim={model.config.hidden_size // model.config.num_attention_heads}")

        sequences = load_sequences(args.model, args.num_sequences, SEQ_LEN)
        print(f"  Loaded {len(sequences)} sequences (seq_len={SEQ_LEN})")

    if actions["compute_gap"]:
        t0 = time.time()
        g_raw_all, g_cent_all, gap_all, n_layers, n_heads = compute_gap_kernels(
            model, adapter, model_spec, args.device, sequences
        )
        elapsed = time.time() - t0
        print(f"  Phase 1 complete in {elapsed:.1f}s")

        np.savez_compressed(
            gap_npz,
            g_raw=g_raw_all,
            g_centered=g_cent_all,
            gap=gap_all,
        )
        print(f"  Saved gap kernels to {gap_npz}")

    if actions["compute_total"]:
        head_dim_for_total = model.config.hidden_size // model.config.num_attention_heads
        print(f"\n  Computing total per-pair energy (confound control)...")
        t0 = time.time()
        total_energy_df = compute_per_pair_total_energy(
            model, adapter, args.device, sequences, head_dim_for_total, PAIR_INDICES
        )
        total_energy_df.to_parquet(total_energy_parquet, index=False)
        print(f"  Total pair energy computed in {time.time() - t0:.1f}s")
        print(f"  Saved total pair energy to {total_energy_parquet}")

    if actions["need_model"]:
        del model, adapter, tokenizer, loader, sequences
        torch.cuda.empty_cache()

    # Determine head_dim from gap array shape — all models here use 128
    # but be safe: head_dim = 2 * n_pairs, n_pairs = ... we need actual head_dim
    # For the projection we need head_dim; derive from model config
    # Since model may have been freed, use known values
    HEAD_DIMS = {"llama-3.1-8b": 128, "olmo-2-7b": 128}
    head_dim = HEAD_DIMS[args.model]

    # ── Phase 2: Project gap onto RoPE frequencies ───────────────────────────
    print(f"\n[2/5] Phase 2: Projecting gap signal onto RoPE pair frequencies...")
    t0 = time.time()
    crossterm_df = project_gap_onto_rope_freqs(gap_all, head_dim, n_layers, n_heads)
    crossterm_df.to_parquet(output_dir / "crossterm_per_head_pair.parquet", index=False)
    print(f"  Projection complete: {len(crossterm_df)} (layer, head, pair) rows in {time.time() - t0:.1f}s")

    # Load R² data for SI-group analysis
    r2_df = load_r2_data(args.model)

    # Aggregate cross-term by pair (for the 8 ablated pairs)
    crossterm_agg = aggregate_crossterm_by_pair(crossterm_df, r2_df, PAIR_INDICES)
    crossterm_agg.to_parquet(output_dir / "crossterm_per_pair_agg.parquet", index=False)
    print(f"  Aggregated cross-term energy for {len(crossterm_agg)} pairs")

    # ── Phase 3: Load pair ablation effects ──────────────────────────────────
    print(f"\n[3/5] Phase 3: Loading pair ablation effects from Experiment 2...")
    pair_effects = load_pair_effects(args.model)

    # Load precomputed total energy (already refreshed above if compute_total=True)
    if total_energy_parquet.exists():
        total_energy_df = pd.read_parquet(total_energy_parquet)
        print(f"  Loaded total pair energy from {total_energy_parquet}")
    else:
        # Try to derive from 3.2c freq_profile (aggregate over heads)
        print(f"  Total pair energy not pre-computed; attempting derivation from 3.2c freq_profile...")
        try:
            freq_profile = load_freq_profile(args.model)
            # freq_profile has total_energy per head; we need per-pair.
            # The freq_profile doesn't have per-pair breakdown stored — only summary metrics.
            # We cannot derive per-pair total energy from it without raw per-pair data.
            print(f"  WARNING: Cannot derive per-pair total energy from summary metrics.")
            print(f"           Partial correlations will use NaN for total energy.")
            total_energy_df = None
        except FileNotFoundError:
            print(f"  WARNING: No freq_profile found either. Partial correlations will be NaN.")
            total_energy_df = None

    # ── Phase 4: Correlation analysis ────────────────────────────────────────
    print(f"\n[4/5] Phase 4: Correlation analysis...")
    analysis = run_correlation_analysis(crossterm_agg, pair_effects, total_energy_df, args.model)

    # ── Phase 5: Report ──────────────────────────────────────────────────────
    print(f"\n[5/5] Phase 5: Writing report...")

    print_report(analysis, crossterm_agg, args.model)

    # Save structured analysis
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, default=str), encoding="utf-8"
    )

    # Save full report
    report = {
        "experiment": "3_theory3_crossterm_correlation",
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_sequences": args.num_sequences,
            "seq_len": SEQ_LEN,
            "rope_base": ROPE_BASE,
            "head_dim": head_dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "pair_indices_ablated": PAIR_INDICES,
            "quartile_for_si_groups": QUARTILE,
        },
        "gap_kernel_summary": {
            "max_delta": int(gap_all.shape[2]),
            "mean_gap_magnitude": float(np.abs(gap_all).mean()),
            "max_gap_magnitude": float(np.abs(gap_all).max()),
            "mean_raw_magnitude": float(np.abs(g_raw_all).mean()),
            "mean_centered_magnitude": float(np.abs(g_cent_all).mean()),
            "gap_fraction_of_raw": float(np.abs(gap_all).mean() / max(np.abs(g_raw_all).mean(), 1e-30)),
        },
        "crossterm_per_pair": crossterm_agg.to_dict(orient="records"),
        "correlation_analysis": analysis,
    }

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n  Report saved to {output_dir / 'report.json'}")
    print(f"  Analysis saved to {output_dir / 'analysis.json'}")
    print(f"  Gap kernels: {gap_npz}")
    print(f"  Cross-term per (head, pair): {output_dir / 'crossterm_per_head_pair.parquet'}")
    print(f"  Cross-term aggregated: {output_dir / 'crossterm_per_pair_agg.parquet'}")
    print(f"\nDone. All artifacts in: {output_dir}")


if __name__ == "__main__":
    main()
