#!/usr/bin/env python3
"""
Experiment 3 Theory 7b: Activation patching test of induction circuit feeders
=============================================================================

Motivation
----------
Theory 7 Approach B (knockout analysis) had a critical flaw: zeroing a head's
output destroys the residual stream, causing catastrophic downstream disruption
rather than isolating circuit-specific effects.  A handful of "globally important"
early-layer heads dominated the results regardless of circuit membership.

This experiment replaces knockout with RESAMPLE PATCHING — the standard causal
technique in mechanistic interpretability:

    1. Generate pairs of repeated random sequences (clean, corrupt) with the
       same period but different tokens
    2. Clean run: cache every source head's o_proj input; measure induction scores
    3. Corrupt run: cache every source head's o_proj input
    4. Patch run (one per source head): run the CLEAN input but replace one
       source head's o_proj input with its CORRUPT activation
    5. Measure induction score disruption: clean_score - patched_score
    6. Correlate disruption with R² (shift-invariance)

The key difference: resample patching replaces a head's output with a NATURAL
activation from a different sequence (preserving magnitude and statistics),
rather than zeroing it.  This isolates "does this head carry sequence-specific
information that the induction head uses?" rather than "does the model work
without this head?"

Protocol
--------
Phase 1 — Load existing data (R², induction scores, top induction heads)
Phase 2 — Resample activation patching
Phase 3 — Correlate patching disruption with R²
Phase 4 — Comparison with Approach A (previous-token scores)

Usage
-----
    python scripts/experiment3_theory7b_activation_patching.py --model llama-3.1-8b --device cuda:2
    python scripts/experiment3_theory7b_activation_patching.py --model olmo-2-7b --device cuda:3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment3.stats_utils import partial_correlation_with_intercept
from shared.specs import ModelSpec

try:
    from shared.attention.adapters import get_adapter
    from shared.models.loading import load_model, load_tokenizer
except Exception:  # pragma: no cover - optional for analysis-only environments
    get_adapter = None
    load_model = None
    load_tokenizer = None

# ── constants ────────────────────────────────────────────────────────────────
NUM_SEQUENCE_PAIRS = 30        # pairs of (clean, corrupt) repeated sequences
KNOCKOUT_PERIOD = 32           # repeat period
KNOCKOUT_SEQ_LEN = 128         # 4 repeats of period
TOP_INDUCTION_HEADS = 10       # how many induction heads to test
MAX_SOURCE_LAYER = 7           # only test source heads in layers 0..7
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 42

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


# ── data structures ──────────────────────────────────────────────────────────
@dataclass
class HeadID:
    layer: int
    head: int

    def __hash__(self):
        return hash((self.layer, self.head))

    def __eq__(self, other):
        return isinstance(other, HeadID) and self.layer == other.layer and self.head == other.head

    def __repr__(self):
        return f"L{self.layer}H{self.head}"


# ═══════════════════════════════════════════════════════════════════════════════
# HOOKING INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_model_internals(model):
    """Extract model config and layer modules."""
    config = model.config
    num_query_heads = int(getattr(config, "num_attention_heads"))
    hidden_size = int(getattr(config, "hidden_size"))
    head_dim = hidden_size // num_query_heads
    layers = getattr(getattr(model, "model"), "layers")
    return num_query_heads, head_dim, layers


@contextmanager
def capture_head_outputs(model, layer_indices: set[int]):
    """Context manager that captures ALL heads' o_proj inputs for specified layers.

    Yields a dict: {layer_idx: tensor of shape [batch, seq, n_heads, head_dim]}.
    The tensors are detached and on CPU.
    """
    num_query_heads, head_dim, layers = _get_model_internals(model)

    captured: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        o_proj = layer.self_attn.o_proj

        def make_hook(l_idx: int, n_heads: int, h_dim: int):
            def hook(module, args):
                x = args[0]  # [batch, seq, hidden]
                batch, seq, hidden = x.shape
                view = x.view(batch, seq, n_heads, h_dim)
                captured[l_idx] = view.detach().cpu().clone()
                return args  # don't modify
            return hook

        handle = o_proj.register_forward_pre_hook(
            make_hook(layer_idx, num_query_heads, head_dim),
            with_kwargs=False,
        )
        handles.append(handle)

    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_head_output(model, head: HeadID, replacement: torch.Tensor):
    """Context manager that replaces one head's o_proj input with a cached value.

    replacement: tensor of shape [batch, seq, head_dim] on CPU.
    """
    num_query_heads, head_dim, layers = _get_model_internals(model)

    layer = layers[head.layer]
    o_proj = layer.self_attn.o_proj

    def make_hook(head_idx: int, n_heads: int, h_dim: int, repl: torch.Tensor):
        def hook(module, args):
            x = args[0]  # [batch, seq, hidden]
            batch, seq, hidden = x.shape
            view = x.view(batch, seq, n_heads, h_dim).clone()
            view[:, :, head_idx, :] = repl.to(x.device)
            return (view.reshape(batch, seq, hidden),) + args[1:]
        return hook

    handle = o_proj.register_forward_pre_hook(
        make_hook(head.head, num_query_heads, head_dim, replacement),
        with_kwargs=False,
    )

    try:
        yield
    finally:
        handle.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# INDUCTION SCORE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_induction_scores_from_capture(
    capture_logits: torch.Tensor,
    target_heads: list[HeadID],
    period: int,
) -> dict[tuple[int, int], float]:
    """Compute induction score for specific heads from captured attention logits.

    capture_logits: [layers, heads, seq, seq] (pre-softmax)
    Returns {(layer, head): induction_score}.
    """
    n_layers, n_heads, seq_len, _ = capture_logits.shape

    # Apply causal mask and softmax
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    logits = capture_logits.clone()
    logits[:, :, :, :][..., causal_mask] = float('-inf')
    attn = torch.softmax(logits, dim=-1)  # [L, H, S, S]

    scores = {}
    for h in target_heads:
        head_attn = attn[h.layer, h.head]  # [S, S]
        # Induction: in second repeat, attend to position i-period+1
        vals = []
        for i in range(period, min(seq_len, 2 * period)):
            target_pos = i - period + 1
            if 0 <= target_pos < seq_len:
                vals.append(head_attn[i, target_pos].item())
        scores[(h.layer, h.head)] = float(np.mean(vals)) if vals else 0.0

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# SEQUENCE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_repeated_random_sequence(
    vocab_size: int,
    period: int,
    total_len: int,
    seed: int,
    special_ids: set[int],
) -> list[int]:
    """Generate [A1..Ap A1..Ap ...] with random tokens."""
    rng = np.random.RandomState(seed)
    pool = [t for t in range(vocab_size) if t not in special_ids]
    base = rng.choice(pool, size=period, replace=True).tolist()
    n_repeats = (total_len // period) + 1
    full = (base * n_repeats)[:total_len]
    return full


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_induction_scores(model_name: str) -> pd.DataFrame:
    path = Path("results/experiment3/induction_r2_crossref") / model_name / "induction_scores.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return pd.read_parquet(path)


def load_r2_data(model_name: str) -> pd.DataFrame:
    path = Path("results/experiment3/theory1_si_circuits") / model_name / "per_sequence_r2.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_parquet(path)
    mean_r2 = df.groupby(["layer", "head"])["r2"].mean().reset_index()
    mean_r2.columns = ["layer", "head", "mean_r2"]
    return mean_r2


def load_prev_token_scores(model_name: str) -> pd.DataFrame | None:
    path = Path("results/experiment3/theory7_induction_feeders") / model_name / "prev_token_scores.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def _artifact_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: RESAMPLE ACTIVATION PATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def run_activation_patching(
    model,
    adapter,
    tokenizer,
    model_spec: ModelSpec,
    device: str,
    induction_heads: list[HeadID],
    num_pairs: int,
    period: int,
    total_len: int,
    max_source_layer: int,
) -> pd.DataFrame:
    """Run resample activation patching for all (source, target) pairs.

    Returns DataFrame with columns:
        source_layer, source_head, target_layer, target_head,
        mean_disruption, std_disruption, mean_recovery, n_pairs
    """
    vocab_size = tokenizer.vocab_size
    special_ids = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        sid = getattr(tokenizer, attr, None)
        if sid is not None:
            special_ids.add(sid)

    config = model.config
    num_query_heads = int(getattr(config, "num_attention_heads"))

    # Source layers to capture
    source_layers = set(range(min(max_source_layer + 1, config.num_hidden_layers)))

    # Build list of all source heads
    all_source_heads: list[HeadID] = []
    for l in sorted(source_layers):
        for h in range(num_query_heads):
            all_source_heads.append(HeadID(l, h))

    print(f"  Testing {len(all_source_heads)} source heads x "
          f"{len(induction_heads)} induction heads x {num_pairs} sequence pairs")
    print(f"  = {len(all_source_heads) * num_pairs + 2 * num_pairs} forward passes")

    # Accumulate per-(source, target) disruptions
    disruptions: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
    clean_scores_all: dict[tuple[int, int], list[float]] = defaultdict(list)
    corrupt_scores_all: dict[tuple[int, int], list[float]] = defaultdict(list)

    for pair_idx in range(num_pairs):
        t0 = time.time()

        # Generate clean and corrupt sequences (different tokens, same period)
        clean_tokens = generate_repeated_random_sequence(
            vocab_size, period, total_len,
            seed=pair_idx * 1000 + 111,
            special_ids=special_ids,
        )
        corrupt_tokens = generate_repeated_random_sequence(
            vocab_size, period, total_len,
            seed=pair_idx * 1000 + 999,
            special_ids=special_ids,
        )

        clean_input = torch.tensor([clean_tokens], dtype=torch.long, device=device)
        corrupt_input = torch.tensor([corrupt_tokens], dtype=torch.long, device=device)

        # ── Step 1: Clean run ─────────────────────────────────────────────
        with capture_head_outputs(model, source_layers) as clean_cached:
            with torch.no_grad():
                clean_capture = adapter.capture(
                    model, input_ids=clean_input,
                    include_logits=True, return_token_logits=False,
                    capture_attention=True, output_device="cpu",
                )
        if clean_capture.logits is None:
            print(f"  Pair {pair_idx}: clean capture failed, skipping")
            continue
        clean_scores = compute_induction_scores_from_capture(
            clean_capture.logits.float(), induction_heads, period,
        )
        for k, v in clean_scores.items():
            clean_scores_all[k].append(v)

        # ── Step 2: Corrupt run (cache source outputs) ────────────────────
        with capture_head_outputs(model, source_layers) as corrupt_cached:
            with torch.no_grad():
                corrupt_capture = adapter.capture(
                    model, input_ids=corrupt_input,
                    include_logits=True, return_token_logits=False,
                    capture_attention=True, output_device="cpu",
                )
        if corrupt_capture.logits is None:
            print(f"  Pair {pair_idx}: corrupt capture failed, skipping")
            continue
        corrupt_scores = compute_induction_scores_from_capture(
            corrupt_capture.logits.float(), induction_heads, period,
        )
        for k, v in corrupt_scores.items():
            corrupt_scores_all[k].append(v)

        del clean_capture, corrupt_capture, clean_input, corrupt_input
        torch.cuda.empty_cache()

        # ── Step 3: Patch runs (one per source head) ──────────────────────
        clean_input = torch.tensor([clean_tokens], dtype=torch.long, device=device)

        for src_idx, src_head in enumerate(all_source_heads):
            # Get the corrupt activation for this head
            corrupt_layer_cache = corrupt_cached.get(src_head.layer)
            if corrupt_layer_cache is None:
                continue
            # corrupt_layer_cache: [batch, seq, n_heads, head_dim]
            corrupt_head_act = corrupt_layer_cache[:, :, src_head.head, :]  # [batch, seq, head_dim]

            with patch_head_output(model, src_head, corrupt_head_act):
                with torch.no_grad():
                    patched_capture = adapter.capture(
                        model, input_ids=clean_input,
                        include_logits=True, return_token_logits=False,
                        capture_attention=True, output_device="cpu",
                    )

            if patched_capture.logits is None:
                continue

            patched_scores = compute_induction_scores_from_capture(
                patched_capture.logits.float(), induction_heads, period,
            )

            for tgt_head in induction_heads:
                tgt_key = (tgt_head.layer, tgt_head.head)
                disruption = clean_scores[tgt_key] - patched_scores[tgt_key]
                disruptions[
                    (src_head.layer, src_head.head, tgt_head.layer, tgt_head.head)
                ].append(disruption)

            del patched_capture
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  Pair {pair_idx + 1}/{num_pairs}: {elapsed:.1f}s "
              f"({len(all_source_heads)} patch runs)")

        del clean_input
        torch.cuda.empty_cache()

    # ── Build results DataFrame ───────────────────────────────────────────
    rows = []
    for (sl, sh, tl, th), disrupt_list in disruptions.items():
        arr = np.array(disrupt_list)
        # Compute recovery: disruption / (clean - corrupt baseline gap)
        tgt_key = (tl, th)
        clean_mean = np.mean(clean_scores_all[tgt_key])
        corrupt_mean = np.mean(corrupt_scores_all[tgt_key])
        gap = clean_mean - corrupt_mean
        recovery = float(np.mean(arr) / gap) if abs(gap) > 1e-6 else float("nan")

        rows.append({
            "source_layer": sl,
            "source_head": sh,
            "target_layer": tl,
            "target_head": th,
            "mean_disruption": float(np.mean(arr)),
            "std_disruption": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "mean_recovery": recovery,
            "n_pairs": len(arr),
            "clean_induction_mean": clean_mean,
            "corrupt_induction_mean": corrupt_mean,
        })

    df = pd.DataFrame(rows)
    print(f"\n  Patching complete: {len(df)} (source, target) pairs")

    # Print baseline induction score summary
    print(f"\n  Induction head baseline scores:")
    for h in induction_heads:
        cm = np.mean(clean_scores_all[(h.layer, h.head)])
        crm = np.mean(corrupt_scores_all[(h.layer, h.head)])
        print(f"    {h}: clean={cm:.4f}, corrupt={crm:.4f}, gap={cm - crm:.4f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_patching_results(
    patch_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    prev_token_df: pd.DataFrame | None,
    induction_heads: list[HeadID],
    model_name: str,
    num_pairs: int,
) -> dict[str, Any]:
    """Correlate patching disruption with R² and previous-token scores."""

    # Aggregate disruption per source head (mean across target induction heads)
    source_agg = patch_df.groupby(["source_layer", "source_head"]).agg(
        mean_disruption=("mean_disruption", "mean"),
        max_disruption=("mean_disruption", "max"),
        mean_recovery=("mean_recovery", "mean"),
    ).reset_index()

    # Merge with R²
    merged = pd.merge(
        source_agg,
        r2_df.rename(columns={"layer": "source_layer", "head": "source_head"}),
        on=["source_layer", "source_head"],
        how="inner",
    )

    # ── Primary correlation: disruption vs R² ────────────────────────────
    r_p_disrupt, p_p_disrupt = pearsonr(merged["mean_r2"], merged["mean_disruption"])
    r_s_disrupt, p_s_disrupt = spearmanr(merged["mean_r2"], merged["mean_disruption"])

    r_p_recovery, p_p_recovery = pearsonr(
        merged["mean_r2"].values,
        merged["mean_recovery"].values,
    ) if not merged["mean_recovery"].isna().all() else (float("nan"), float("nan"))
    r_s_recovery, p_s_recovery = spearmanr(
        merged["mean_r2"].values,
        merged["mean_recovery"].values,
    ) if not merged["mean_recovery"].isna().all() else (float("nan"), float("nan"))

    # Bootstrap CI for correlations
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    n = len(merged)
    boot_pearson = np.empty(BOOTSTRAP_N)
    boot_spearman = np.empty(BOOTSTRAP_N)
    r2_vals = merged["mean_r2"].values
    disrupt_vals = merged["mean_disruption"].values
    for i in range(BOOTSTRAP_N):
        idx = rng.randint(0, n, size=n)
        bp, _ = pearsonr(r2_vals[idx], disrupt_vals[idx])
        bs, _ = spearmanr(r2_vals[idx], disrupt_vals[idx])
        boot_pearson[i] = bp
        boot_spearman[i] = bs

    # ── Top disruptive heads ─────────────────────────────────────────────
    top20 = merged.nlargest(20, "mean_disruption")
    r2_median = float(r2_df["mean_r2"].median())
    n_above_median = int((top20["mean_r2"] >= r2_median).sum())

    # ── R² quartile distribution of top-20 ───────────────────────────────
    all_r2 = r2_df["mean_r2"].values
    p25, p50, p75 = np.percentile(all_r2, [25, 50, 75])
    top20_r2 = top20["mean_r2"].values
    q_dist = {
        "Q1_below_p25": int((top20_r2 < p25).sum()),
        "Q2_p25_to_p50": int(((top20_r2 >= p25) & (top20_r2 < p50)).sum()),
        "Q3_p50_to_p75": int(((top20_r2 >= p50) & (top20_r2 < p75)).sum()),
        "Q4_above_p75": int((top20_r2 >= p75).sum()),
    }

    # ── Layer distribution of top-20 ─────────────────────────────────────
    top20_layers = top20["source_layer"].values
    layer_counts = {}
    for l, c in zip(*np.unique(top20_layers, return_counts=True)):
        layer_counts[int(l)] = int(c)

    # ── Concentration: how many unique heads dominate? ────────────────────
    # For each target, get top-5 sources
    per_target_top5: list[dict] = []
    for h in induction_heads:
        target_df = patch_df[
            (patch_df["target_layer"] == h.layer) & (patch_df["target_head"] == h.head)
        ].nlargest(5, "mean_disruption")
        for _, r in target_df.iterrows():
            per_target_top5.append({
                "source_layer": int(r.source_layer),
                "source_head": int(r.source_head),
                "target": repr(h),
                "disruption": float(r.mean_disruption),
            })

    top5_df = pd.DataFrame(per_target_top5)
    if len(top5_df) > 0:
        unique_sources = top5_df.groupby(["source_layer", "source_head"]).size().reset_index(name="n_appearances")
        unique_sources = unique_sources.sort_values("n_appearances", ascending=False)
        n_unique = len(unique_sources)
        top3_appearances = int(unique_sources.head(3)["n_appearances"].sum())
        total_slots = len(top5_df)
    else:
        n_unique = 0
        top3_appearances = 0
        total_slots = 0

    # ── Merge with prev_token scores if available ────────────────────────
    pt_correlation = None
    if prev_token_df is not None:
        merged_pt = pd.merge(
            merged,
            prev_token_df.rename(columns={"layer": "source_layer", "head": "source_head"})[
                ["source_layer", "source_head", "prev_token_score"]
            ],
            on=["source_layer", "source_head"],
            how="inner",
        )
        if len(merged_pt) >= 5:
            r_pt_disrupt, p_pt_disrupt = spearmanr(
                merged_pt["prev_token_score"], merged_pt["mean_disruption"]
            )
            r_pt_r2, p_pt_r2 = spearmanr(
                merged_pt["prev_token_score"], merged_pt["mean_r2"]
            )
            partial = partial_correlation_with_intercept(
                merged_pt["mean_r2"].values,
                merged_pt["mean_disruption"].values,
                merged_pt["prev_token_score"].values,
            )

            pt_correlation = {
                "spearman_prev_token_vs_disruption": float(r_pt_disrupt),
                "spearman_prev_token_vs_disruption_p": float(p_pt_disrupt),
                "spearman_prev_token_vs_r2": float(r_pt_r2),
                "spearman_prev_token_vs_r2_p": float(p_pt_r2),
                "partial_r2_vs_disruption_ctrl_prev_token": float(partial.r),
                "partial_r2_vs_disruption_ctrl_prev_token_p": float(partial.p_value),
                "partial_df": int(partial.df),
                "n_heads": len(merged_pt),
            }

    # ── Build analysis dict ──────────────────────────────────────────────
    analysis = {
        "model": model_name,
        "n_source_heads": len(merged),
        "n_induction_heads": len(induction_heads),
        "n_sequence_pairs": int(num_pairs),
        "primary_correlation": {
            "description": "Patching disruption vs R² (shift-invariance)",
            "pearson_r": float(r_p_disrupt),
            "pearson_p": float(p_p_disrupt),
            "pearson_ci_95": [
                float(np.percentile(boot_pearson, 2.5)),
                float(np.percentile(boot_pearson, 97.5)),
            ],
            "spearman_rho": float(r_s_disrupt),
            "spearman_p": float(p_s_disrupt),
            "spearman_ci_95": [
                float(np.percentile(boot_spearman, 2.5)),
                float(np.percentile(boot_spearman, 97.5)),
            ],
        },
        "recovery_correlation": {
            "description": "Patching recovery (normalized disruption / clean-corrupt gap) vs R²",
            "pearson_r": float(r_p_recovery) if not np.isnan(r_p_recovery) else None,
            "pearson_p": float(p_p_recovery) if not np.isnan(p_p_recovery) else None,
            "spearman_rho": float(r_s_recovery) if not np.isnan(r_s_recovery) else None,
            "spearman_p": float(p_s_recovery) if not np.isnan(p_s_recovery) else None,
        },
        "top_20_disruptive_heads": {
            "n_above_r2_median": n_above_median,
            "fraction_above_median": n_above_median / 20,
            "r2_quartile_distribution": q_dist,
            "mean_r2": float(top20["mean_r2"].mean()),
            "overall_r2_median": float(r2_median),
            "layer_distribution": layer_counts,
            "heads": [
                {
                    "source": f"L{int(r.source_layer)}H{int(r.source_head)}",
                    "mean_disruption": float(r.mean_disruption),
                    "mean_recovery": float(r.mean_recovery),
                    "r2": float(r.mean_r2),
                }
                for r in top20.itertuples()
            ],
        },
        "concentration": {
            "description": "How concentrated are top-5 feeders across induction heads?",
            "n_unique_sources_in_top5": n_unique,
            "total_top5_slots": total_slots,
            "top3_sources_account_for": f"{top3_appearances}/{total_slots}",
            "top3_sources_fraction": top3_appearances / total_slots if total_slots > 0 else 0,
        },
        "prev_token_comparison": pt_correlation,
    }

    # ── Verdict ──────────────────────────────────────────────────────────
    analysis["hypothesis_supported"] = bool(
        r_s_disrupt > 0 and p_s_disrupt < 0.05
    )
    analysis["interpretation"] = (
        f"Resample patching {'SUPPORTS' if analysis['hypothesis_supported'] else 'DOES NOT SUPPORT'} "
        f"the feeder-SI hypothesis: Spearman rho={r_s_disrupt:+.4f} (p={p_s_disrupt:.2e}) "
        f"between patching disruption and R². "
        f"Top-20 most disruptive heads: {n_above_median}/20 above R² median."
    )

    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3 Theory 7b: Activation patching test of induction circuit feeders"
    )
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output-dir", default="results/experiment3/theory7b_activation_patching")
    parser.add_argument("--num-pairs", type=int, default=NUM_SEQUENCE_PAIRS)
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip model execution and recompute analysis from saved patching_results.parquet",
    )
    parser.add_argument(
        "--patching-parquet",
        default=None,
        help="Optional path to saved patching_results.parquet for --analysis-only",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_spec = MODELS[args.model]

    print(f"{'='*72}")
    print(f"Experiment 3 Theory 7b: Activation Patching — {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"{'='*72}")

    # ── Phase 1: Load data ────────────────────────────────────────────────
    print(f"\n[1/4] Loading existing data...")
    induction_df = load_induction_scores(args.model)
    r2_df = load_r2_data(args.model)
    prev_token_df = load_prev_token_scores(args.model)

    top_induction = induction_df.sort_values("induction_score", ascending=False).head(TOP_INDUCTION_HEADS)
    induction_heads = [HeadID(int(r.layer), int(r.head)) for r in top_induction.itertuples()]

    print(f"  Top {TOP_INDUCTION_HEADS} induction heads:")
    for i, h in enumerate(induction_heads):
        score = top_induction.iloc[i]["induction_score"]
        r2_val = r2_df[(r2_df.layer == h.layer) & (r2_df.head == h.head)]["mean_r2"].values
        r2_str = f"{r2_val[0]:.4f}" if len(r2_val) > 0 else "N/A"
        print(f"    {i+1:2d}. {h}  induction={score:.4f}  R²={r2_str}")

    patching_path = (
        Path(args.patching_parquet)
        if args.patching_parquet
        else (output_dir / "patching_results.parquet")
    )
    elapsed = 0.0

    if args.analysis_only:
        print(f"\n[2/4] Analysis-only mode: loading patching parquet...")
        if not patching_path.exists():
            raise FileNotFoundError(
                f"Analysis-only requested but patching parquet is missing: {patching_path}"
            )
        patch_df = pd.read_parquet(patching_path)
        print(f"  Loaded patching results: {len(patch_df)} rows from {patching_path}")
    else:
        if get_adapter is None or load_model is None or load_tokenizer is None:
            raise RuntimeError(
                "Model execution dependencies are unavailable. "
                "Install runtime deps or rerun with --analysis-only."
            )
        # ── Load model ────────────────────────────────────────────────────
        print(f"\n[2/4] Loading model...")
        loader = load_model(model_spec)
        model = loader.model.to(args.device)
        model.eval()
        tokenizer = load_tokenizer(model_spec)
        adapter = get_adapter(model_spec)
        adapter.register(model)
        print(f"  Model loaded: {model_spec.hf_id}")

        # ── Phase 2: Activation patching ─────────────────────────────────
        print(f"\n[3/4] Running resample activation patching...")
        t0 = time.time()
        patch_df = run_activation_patching(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            model_spec=model_spec,
            device=args.device,
            induction_heads=induction_heads,
            num_pairs=args.num_pairs,
            period=KNOCKOUT_PERIOD,
            total_len=KNOCKOUT_SEQ_LEN,
            max_source_layer=MAX_SOURCE_LAYER,
        )
        elapsed = time.time() - t0
        patch_df.to_parquet(patching_path, index=False)
        print(f"  Activation patching complete in {elapsed:.0f}s")

    # ── Phase 3: Analysis ────────────────────────────────────────────────
    print(f"\n[4/4] Analyzing results...")
    analysis = analyze_patching_results(
        patch_df, r2_df, prev_token_df, induction_heads, args.model, args.num_pairs,
    )

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"RESULTS — {args.model}")
    print(f"{'='*72}")

    pc = analysis["primary_correlation"]
    print(f"\n  PRIMARY: Patching disruption vs R²")
    print(f"    Pearson:  r={pc['pearson_r']:+.4f} (p={pc['pearson_p']:.2e})")
    print(f"              95% CI [{pc['pearson_ci_95'][0]:+.4f}, {pc['pearson_ci_95'][1]:+.4f}]")
    print(f"    Spearman: rho={pc['spearman_rho']:+.4f} (p={pc['spearman_p']:.2e})")
    print(f"              95% CI [{pc['spearman_ci_95'][0]:+.4f}, {pc['spearman_ci_95'][1]:+.4f}]")

    rc = analysis["recovery_correlation"]
    if rc.get("spearman_rho") is not None:
        print(f"\n  RECOVERY (normalized): rho={rc['spearman_rho']:+.4f} (p={rc['spearman_p']:.2e})")

    t20 = analysis["top_20_disruptive_heads"]
    print(f"\n  TOP-20 most disruptive source heads:")
    print(f"    {t20['n_above_r2_median']}/20 above R² median ({t20['overall_r2_median']:.4f})")
    print(f"    R² quartile distribution: {t20['r2_quartile_distribution']}")
    print(f"    Layer distribution: {t20['layer_distribution']}")
    for h in t20["heads"][:10]:
        print(f"      {h['source']:8s}  disruption={h['mean_disruption']:+.4f}  "
              f"recovery={h['mean_recovery']:+.4f}  R²={h['r2']:.4f}")

    conc = analysis["concentration"]
    print(f"\n  CONCENTRATION: {conc['n_unique_sources_in_top5']} unique heads in "
          f"{conc['total_top5_slots']} top-5 slots")
    print(f"    Top-3 sources: {conc['top3_sources_account_for']} ({conc['top3_sources_fraction']:.0%})")

    if analysis["prev_token_comparison"]:
        ptc = analysis["prev_token_comparison"]
        print(f"\n  PREV-TOKEN COMPARISON:")
        print(f"    prev_token vs disruption: rho={ptc['spearman_prev_token_vs_disruption']:+.4f} "
              f"(p={ptc['spearman_prev_token_vs_disruption_p']:.2e})")
        print(f"    R² vs disruption (partial, ctrl prev_token): r={ptc['partial_r2_vs_disruption_ctrl_prev_token']:+.4f} "
              f"(p={ptc['partial_r2_vs_disruption_ctrl_prev_token_p']:.2e})")

    print(f"\n  VERDICT: {analysis['interpretation']}")

    # ── Save report ──────────────────────────────────────────────────────
    report = {
        "experiment": "3_theory7b_activation_patching",
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_sequence_pairs": args.num_pairs,
            "period": KNOCKOUT_PERIOD,
            "seq_len": KNOCKOUT_SEQ_LEN,
            "top_induction_heads": TOP_INDUCTION_HEADS,
            "max_source_layer": MAX_SOURCE_LAYER,
            "bootstrap_n": BOOTSTRAP_N,
            "analysis_only": bool(args.analysis_only),
            "patching_parquet": str(patching_path),
        },
        "artifact_timestamps": {
            "induction_scores_parquet": _artifact_timestamp(
                Path("results/experiment3/induction_r2_crossref") / args.model / "induction_scores.parquet"
            ),
            "r2_per_sequence_parquet": _artifact_timestamp(
                Path("results/experiment3/theory1_si_circuits") / args.model / "per_sequence_r2.parquet"
            ),
            "prev_token_scores_parquet": _artifact_timestamp(
                Path("results/experiment3/theory7_induction_feeders") / args.model / "prev_token_scores.parquet"
            ),
            "patching_results_parquet": _artifact_timestamp(patching_path),
        },
        "analysis": analysis,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )
    print(f"\n  Report saved to {output_dir / 'report.json'}")
    print(f"  Runtime: {elapsed:.0f}s")
    print(f"  Done.")


if __name__ == "__main__":
    main()
