#!/usr/bin/env python3
"""
Experiment 3 Theory 10: Feeder Specificity — Are High-SI Heads Induction-Specific?
===================================================================================

Motivation
----------
Theory 7b showed that source head R² weakly predicts disruption to induction heads
(rho=+0.206).  But we don't know whether this effect is *induction-specific* or
whether high-SI early heads feed ALL downstream heads equally.

This experiment tests specificity: does patching high-SI early heads disrupt
non-induction late heads equally?  If high-SI heads are truly induction-circuit
feeders, their disruption effect should be STRONGER on induction targets than
on random or low-SI late targets.

Protocol
--------
1. Select three target groups (10 heads each):
   - Induction targets:     Top 10 induction heads
   - Random mid-layer:      10 randomly selected heads from layers 16-24
   - Low-SI late:           10 heads with lowest R² from layers 24-31

2. Use the SAME resample patching protocol as Theory 7b:
   - Generate 30 pairs of repeated random sequences (period=32, len=128)
   - For each source head in layers 0-7, replace output with corrupt activation
   - Measure disruption on ALL three target groups simultaneously

3. Analysis:
   - Per target group: correlate source disruption with source R²
   - Compare rho across groups
   - Partial correlation controlling for prev-token score, per group

Usage
-----
    python scripts/experiment3_theory10_feeder_specificity.py --model olmo-2-7b --device cuda:0
    python scripts/experiment3_theory10_feeder_specificity.py --model llama-3.1-8b --device cuda:1
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
from scipy import stats as scipy_stats
from scipy.stats import pearsonr, spearmanr

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment3.stats_utils import (
    dependent_corr_williams_test,
    holm_adjust,
    partial_correlation_with_intercept,
)
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
NUM_TARGET_PER_GROUP = 10      # heads per target group
MAX_SOURCE_LAYER = 7           # only test source heads in layers 0..7
RANDOM_MID_LAYER_RANGE = (16, 24)   # inclusive range for random mid-layer targets
LOW_SI_LATE_LAYER_RANGE = (24, 31)  # inclusive range for low-SI late targets
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 42
TARGET_SELECTION_SEED = 12345  # reproducible random mid-layer target selection

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
# HOOKING INFRASTRUCTURE  (reused from Theory 7b)
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
# INDUCTION SCORE COMPUTATION  (reused from Theory 7b)
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
# SEQUENCE GENERATION  (reused from Theory 7b)
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


# ═══════════════════════════════════════════════════════════════════════════════
# TARGET GROUP SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def select_target_groups(
    induction_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    num_query_heads: int,
    num_layers: int,
) -> dict[str, list[HeadID]]:
    """Select the three target groups for the specificity test.

    Returns:
        dict with keys "induction", "random_mid", "low_si_late",
        each mapping to a list of HeadID.
    """
    groups: dict[str, list[HeadID]] = {}

    # ── Group 1: Top induction heads ─────────────────────────────────────
    top_induction = induction_df.sort_values(
        "induction_score", ascending=False
    ).head(NUM_TARGET_PER_GROUP)
    groups["induction"] = [
        HeadID(int(r.layer), int(r.head)) for r in top_induction.itertuples()
    ]

    # ── Group 2: Random mid-layer heads (layers 16-24) ───────────────────
    lo, hi = RANDOM_MID_LAYER_RANGE
    # Clamp to actual model layer count
    hi = min(hi, num_layers - 1)
    all_mid_heads = [
        HeadID(l, h)
        for l in range(lo, hi + 1)
        for h in range(num_query_heads)
    ]
    # Exclude any that are already induction targets
    induction_set = set(groups["induction"])
    eligible_mid = [hid for hid in all_mid_heads if hid not in induction_set]

    rng = np.random.RandomState(TARGET_SELECTION_SEED)
    n_choose = min(NUM_TARGET_PER_GROUP, len(eligible_mid))
    chosen_indices = rng.choice(len(eligible_mid), size=n_choose, replace=False)
    groups["random_mid"] = [eligible_mid[i] for i in sorted(chosen_indices)]

    # ── Group 3: Low-SI late heads (layers 24-31, lowest R²) ─────────────
    lo_late, hi_late = LOW_SI_LATE_LAYER_RANGE
    hi_late = min(hi_late, num_layers - 1)
    late_r2 = r2_df[(r2_df["layer"] >= lo_late) & (r2_df["layer"] <= hi_late)].copy()
    # Exclude heads already selected in other groups
    selected_so_far = set(groups["induction"]) | set(groups["random_mid"])
    late_r2 = late_r2[
        ~late_r2.apply(
            lambda r: HeadID(int(r["layer"]), int(r["head"])) in selected_so_far,
            axis=1,
        )
    ]
    # Take the 10 with lowest mean_r2
    lowest_r2 = late_r2.sort_values("mean_r2", ascending=True).head(NUM_TARGET_PER_GROUP)
    groups["low_si_late"] = [
        HeadID(int(r.layer), int(r.head)) for r in lowest_r2.itertuples()
    ]

    return groups


def infer_target_groups_from_patching_results(
    patch_df: pd.DataFrame,
) -> dict[str, list[HeadID]]:
    """Recover target groups from saved patching results."""
    groups: dict[str, list[HeadID]] = {}
    for group_name, grp in patch_df.groupby("target_group"):
        uniq = (
            grp[["target_layer", "target_head"]]
            .drop_duplicates()
            .sort_values(["target_layer", "target_head"])
        )
        groups[group_name] = [
            HeadID(int(r.target_layer), int(r.target_head))
            for r in uniq.itertuples()
        ]
    return groups


def _artifact_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: RESAMPLE ACTIVATION PATCHING (multi-target)
# ═══════════════════════════════════════════════════════════════════════════════

def run_activation_patching(
    model,
    adapter,
    tokenizer,
    model_spec: ModelSpec,
    device: str,
    target_groups: dict[str, list[HeadID]],
    num_pairs: int,
    period: int,
    total_len: int,
    max_source_layer: int,
) -> pd.DataFrame:
    """Run resample activation patching for all (source, target) pairs.

    Targets are drawn from ALL three groups simultaneously.  Each patching
    forward pass measures disruption across all 30 target heads.

    Returns DataFrame with columns:
        source_layer, source_head, target_layer, target_head, target_group,
        mean_disruption, std_disruption, n_pairs
    """
    vocab_size = tokenizer.vocab_size
    special_ids = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        sid = getattr(tokenizer, attr, None)
        if sid is not None:
            special_ids.add(sid)

    config = model.config
    num_query_heads = int(getattr(config, "num_attention_heads"))

    # All target heads combined (with group labels)
    all_targets: list[tuple[HeadID, str]] = []
    for group_name, heads in target_groups.items():
        for h in heads:
            all_targets.append((h, group_name))

    all_target_heads = [h for h, _ in all_targets]

    # Source layers to capture
    source_layers = set(range(min(max_source_layer + 1, config.num_hidden_layers)))

    # Build list of all source heads
    all_source_heads: list[HeadID] = []
    for l in sorted(source_layers):
        for h in range(num_query_heads):
            all_source_heads.append(HeadID(l, h))

    total_targets = len(all_target_heads)
    print(f"  Testing {len(all_source_heads)} source heads x "
          f"{total_targets} target heads (3 groups) x {num_pairs} sequence pairs")
    print(f"  = {len(all_source_heads) * num_pairs + 2 * num_pairs} forward passes")

    # Accumulate per-(source, target) disruptions
    disruptions: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
    clean_scores_all: dict[tuple[int, int], list[float]] = defaultdict(list)

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
            clean_capture.logits.float(), all_target_heads, period,
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
                patched_capture.logits.float(), all_target_heads, period,
            )

            # Record disruption for ALL target heads across all 3 groups
            for tgt_head in all_target_heads:
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
    # Create a lookup for target group membership
    target_group_map: dict[tuple[int, int], str] = {}
    for h, group_name in all_targets:
        target_group_map[(h.layer, h.head)] = group_name

    rows = []
    for (sl, sh, tl, th), disrupt_list in disruptions.items():
        arr = np.array(disrupt_list)
        rows.append({
            "source_layer": sl,
            "source_head": sh,
            "target_layer": tl,
            "target_head": th,
            "target_group": target_group_map.get((tl, th), "unknown"),
            "mean_disruption": float(np.mean(arr)),
            "std_disruption": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n_pairs": len(arr),
        })

    df = pd.DataFrame(rows)
    print(f"\n  Patching complete: {len(df)} (source, target) pairs")

    # Print baseline score summary per group
    for group_name in ["induction", "random_mid", "low_si_late"]:
        group_heads = target_groups[group_name]
        mean_scores = [
            np.mean(clean_scores_all.get((h.layer, h.head), [0.0]))
            for h in group_heads
        ]
        print(f"  {group_name} group mean clean induction score: {np.mean(mean_scores):.4f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int):
    """Bootstrap 95% CI for Spearman rho."""
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        rhos[i], _ = spearmanr(x[idx], y[idx])
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def _dependent_spearman_difference(
    x: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
) -> dict[str, float]:
    """Difference test for two overlapping Spearman correlations.

    Uses rank-transform + Williams test.
    """
    x_rank = scipy_stats.rankdata(x)
    y_a_rank = scipy_stats.rankdata(y_a)
    y_b_rank = scipy_stats.rankdata(y_b)
    r_xy = float(np.corrcoef(x_rank, y_a_rank)[0, 1])
    r_xz = float(np.corrcoef(x_rank, y_b_rank)[0, 1])
    r_yz = float(np.corrcoef(y_a_rank, y_b_rank)[0, 1])
    test = dependent_corr_williams_test(
        r_xy=r_xy,
        r_xz=r_xz,
        r_yz=r_yz,
        n=len(x_rank),
    )
    return {
        "r_xy": r_xy,
        "r_xz": r_xz,
        "r_yz": r_yz,
        "statistic": float(test.statistic),
        "p_value": float(test.p_value),
        "df": int(test.df),
    }


def analyze_patching_results(
    patch_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    prev_token_df: pd.DataFrame | None,
    target_groups: dict[str, list[HeadID]],
    model_name: str,
    num_pairs: int,
) -> dict[str, Any]:
    """Correlate patching disruption with R² separately per target group.

    The key comparison: is the disruption-vs-R² correlation STRONGER for
    induction targets than for non-induction targets?
    """
    group_results: dict[str, dict[str, Any]] = {}
    source_disruption_by_group: dict[str, pd.DataFrame] = {}

    for group_name in ["induction", "random_mid", "low_si_late"]:
        group_df = patch_df[patch_df["target_group"] == group_name].copy()

        if len(group_df) == 0:
            group_results[group_name] = {"error": "no data"}
            continue

        # Aggregate disruption per source head (mean across this group's targets)
        source_agg = group_df.groupby(["source_layer", "source_head"]).agg(
            mean_disruption=("mean_disruption", "mean"),
            max_disruption=("mean_disruption", "max"),
        ).reset_index()
        source_disruption_by_group[group_name] = source_agg.copy()

        # Merge with R²
        merged = pd.merge(
            source_agg,
            r2_df.rename(columns={"layer": "source_layer", "head": "source_head"}),
            on=["source_layer", "source_head"],
            how="inner",
        )

        if len(merged) < 5:
            group_results[group_name] = {"error": "too few heads after merge"}
            continue

        r2_vals = merged["mean_r2"].values
        disrupt_vals = merged["mean_disruption"].values

        # ── Primary correlation: disruption vs R² ────────────────────────
        r_p, p_p = pearsonr(r2_vals, disrupt_vals)
        r_s, p_s = spearmanr(r2_vals, disrupt_vals)
        ci_lo, ci_hi = _bootstrap_spearman(
            r2_vals, disrupt_vals, BOOTSTRAP_N, BOOTSTRAP_SEED,
        )

        # ── Top-20 most disruptive heads ─────────────────────────────────
        top20 = merged.nlargest(20, "mean_disruption")
        r2_median = float(r2_df["mean_r2"].median())
        n_above_median = int((top20["mean_r2"] >= r2_median).sum())

        # ── Partial correlation controlling for prev-token score ─────────
        partial_result = None
        if prev_token_df is not None:
            merged_pt = pd.merge(
                merged,
                prev_token_df.rename(
                    columns={"layer": "source_layer", "head": "source_head"}
                )[["source_layer", "source_head", "prev_token_score"]],
                on=["source_layer", "source_head"],
                how="inner",
            )
            if len(merged_pt) >= 5:
                r_pt_disrupt, p_pt_disrupt = spearmanr(
                    merged_pt["prev_token_score"].values,
                    merged_pt["mean_disruption"].values,
                )
                partial = partial_correlation_with_intercept(
                    merged_pt["mean_r2"].values,
                    merged_pt["mean_disruption"].values,
                    merged_pt["prev_token_score"].values,
                )
                partial_result = {
                    "spearman_prev_token_vs_disruption": float(r_pt_disrupt),
                    "spearman_prev_token_vs_disruption_p": float(p_pt_disrupt),
                    "partial_r2_vs_disruption_ctrl_prev_token": float(partial.r),
                    "partial_r2_vs_disruption_ctrl_prev_token_p": float(partial.p_value),
                    "partial_df": int(partial.df),
                    "n_heads": len(merged_pt),
                }

        group_results[group_name] = {
            "n_source_heads": len(merged),
            "n_target_heads": len(target_groups.get(group_name, [])),
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
            "spearman_rho": float(r_s),
            "spearman_p": float(p_s),
            "spearman_ci_95": [ci_lo, ci_hi],
            "mean_disruption_all_sources": float(disrupt_vals.mean()),
            "std_disruption_all_sources": float(disrupt_vals.std()),
            "top_20_disruptive": {
                "n_above_r2_median": n_above_median,
                "fraction_above_median": (
                    n_above_median / len(top20) if len(top20) > 0 else 0.0
                ),
                "mean_r2": float(top20["mean_r2"].mean()),
                "heads": [
                    {
                        "source": f"L{int(r.source_layer)}H{int(r.source_head)}",
                        "mean_disruption": float(r.mean_disruption),
                        "r2": float(r.mean_r2),
                    }
                    for r in top20.head(10).itertuples()
                ],
            },
            "prev_token_partial": partial_result,
        }

    # ── Cross-group comparison ────────────────────────────────────────────
    rho_induction = group_results.get("induction", {}).get("spearman_rho", float("nan"))
    rho_random = group_results.get("random_mid", {}).get("spearman_rho", float("nan"))
    rho_low_si = group_results.get("low_si_late", {}).get("spearman_rho", float("nan"))

    mean_disrupt_induction = group_results.get("induction", {}).get(
        "mean_disruption_all_sources", float("nan"),
    )
    mean_disrupt_random = group_results.get("random_mid", {}).get(
        "mean_disruption_all_sources", float("nan"),
    )
    mean_disrupt_low_si = group_results.get("low_si_late", {}).get(
        "mean_disruption_all_sources", float("nan"),
    )

    pairwise_tests: dict[str, dict[str, Any]] = {}
    raw_pvals: dict[str, float] = {}
    induction_source = source_disruption_by_group.get("induction")
    if induction_source is not None:
        for comp_name in ["random_mid", "low_si_late"]:
            comp_source = source_disruption_by_group.get(comp_name)
            if comp_source is None:
                continue

            merged_comp = pd.merge(
                induction_source.rename(columns={"mean_disruption": "disruption_induction"}),
                comp_source.rename(columns={"mean_disruption": "disruption_comparator"}),
                on=["source_layer", "source_head"],
                how="inner",
            )
            merged_comp = pd.merge(
                merged_comp,
                r2_df.rename(columns={"layer": "source_layer", "head": "source_head"})[
                    ["source_layer", "source_head", "mean_r2"]
                ],
                on=["source_layer", "source_head"],
                how="inner",
            )

            if len(merged_comp) < 5:
                pairwise_tests[comp_name] = {
                    "error": "too few matched source heads",
                    "n_sources": int(len(merged_comp)),
                }
                continue

            x = merged_comp["mean_r2"].values
            y_ind = merged_comp["disruption_induction"].values
            y_cmp = merged_comp["disruption_comparator"].values
            rho_ind_cmp, _ = spearmanr(x, y_ind)
            rho_cmp, _ = spearmanr(x, y_cmp)
            dep = _dependent_spearman_difference(x, y_ind, y_cmp)

            key = f"induction_vs_{comp_name}"
            raw_pvals[key] = dep["p_value"]
            pairwise_tests[comp_name] = {
                "n_sources": int(len(merged_comp)),
                "rho_induction": float(rho_ind_cmp),
                "rho_comparator": float(rho_cmp),
                "rho_delta": float(rho_ind_cmp - rho_cmp),
                "williams_t": dep["statistic"],
                "df": dep["df"],
                "p_value": dep["p_value"],
                "p_value_holm": float("nan"),
                "supports_induction_specificity": False,
            }

    if raw_pvals:
        holm = holm_adjust(raw_pvals)
        for comp_name, entry in pairwise_tests.items():
            if "error" in entry:
                continue
            key = f"induction_vs_{comp_name}"
            p_adj = holm.get(key, float("nan"))
            entry["p_value_holm"] = float(p_adj)
            entry["supports_induction_specificity"] = bool(
                entry["rho_delta"] > 0 and np.isfinite(p_adj) and p_adj < 0.05
            )

    valid_pairwise = [
        v for v in pairwise_tests.values()
        if "error" not in v and np.isfinite(v.get("p_value_holm", float("nan")))
    ]
    induction_specific = bool(valid_pairwise) and all(
        v["supports_induction_specificity"] for v in valid_pairwise
    )

    comparison = {
        "rho_induction": float(rho_induction) if not np.isnan(rho_induction) else None,
        "rho_random_mid": float(rho_random) if not np.isnan(rho_random) else None,
        "rho_low_si_late": float(rho_low_si) if not np.isnan(rho_low_si) else None,
        "mean_disruption_induction": (
            float(mean_disrupt_induction) if not np.isnan(mean_disrupt_induction) else None
        ),
        "mean_disruption_random_mid": (
            float(mean_disrupt_random) if not np.isnan(mean_disrupt_random) else None
        ),
        "mean_disruption_low_si_late": (
            float(mean_disrupt_low_si) if not np.isnan(mean_disrupt_low_si) else None
        ),
        "induction_specific": induction_specific,
        "pairwise_specificity_tests": pairwise_tests,
        "multiplicity_policy": (
            "holm across induction-vs-comparator dependent-correlation tests"
        ),
    }

    # ── Build interpretation ──────────────────────────────────────────────
    def _fmt_rho(val):
        return f"{val:+.4f}" if not np.isnan(val) else "N/A"

    if induction_specific:
        interpretation = (
            f"High-SI early heads are INDUCTION-SPECIFIC feeders. "
            f"Dependent-correlation tests support stronger disruption-R2 coupling for induction targets "
            f"(rho={_fmt_rho(rho_induction)}) than for random mid-layer "
            f"(rho={_fmt_rho(rho_random)}) or low-SI late targets "
            f"(rho={_fmt_rho(rho_low_si)})."
        )
    else:
        interpretation = (
            f"Evidence does not support induction-specific feeders under formal dependent-correlation tests. "
            f"Observed disruption-R2 correlations are induction={_fmt_rho(rho_induction)}, "
            f"random_mid={_fmt_rho(rho_random)}, low_si_late={_fmt_rho(rho_low_si)}."
        )

    analysis = {
        "model": model_name,
        "n_sequence_pairs": num_pairs,
        "per_group": group_results,
        "cross_group_comparison": comparison,
        "interpretation": interpretation,
        "target_groups": {
            group_name: [repr(h) for h in heads]
            for group_name, heads in target_groups.items()
        },
    }

    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3 Theory 10: Feeder Specificity -- "
                    "Are high-SI heads induction-specific feeders?"
    )
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/experiment3/theory10_feeder_specificity")
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
    print(f"Experiment 3 Theory 10: Feeder Specificity -- {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"{'='*72}")

    # ── Phase 1: Load data and select target groups ──────────────────────
    print(f"\n[1/4] Loading existing data...")
    induction_df = load_induction_scores(args.model)
    r2_df = load_r2_data(args.model)
    prev_token_df = load_prev_token_scores(args.model)

    print(f"  Induction scores: {len(induction_df)} heads")
    print(f"  R2 data: {len(r2_df)} heads")
    if prev_token_df is not None:
        print(f"  Previous-token scores: {len(prev_token_df)} heads")
    else:
        print(f"  Previous-token scores: not available "
              f"(partial correlations will be skipped)")

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
        target_groups = infer_target_groups_from_patching_results(patch_df)
        print(f"  Loaded patching results: {len(patch_df)} rows from {patching_path}")
    else:
        if get_adapter is None or load_model is None or load_tokenizer is None:
            raise RuntimeError(
                "Model execution dependencies are unavailable. "
                "Install runtime deps or rerun with --analysis-only."
            )
        # ── Load model ───────────────────────────────────────────────────
        print(f"\n[2/4] Loading model...")
        loader = load_model(model_spec)
        model = loader.model.to(args.device)
        model.eval()
        tokenizer = load_tokenizer(model_spec)
        adapter = get_adapter(model_spec)
        adapter.register(model)
        print(f"  Model loaded: {model_spec.hf_id}")

        config = model.config
        num_query_heads = int(getattr(config, "num_attention_heads"))
        num_layers = int(getattr(config, "num_hidden_layers"))

        # ── Select target groups ─────────────────────────────────────────
        print(f"\n  Selecting target groups...")
        target_groups = select_target_groups(
            induction_df, r2_df, num_query_heads, num_layers,
        )
        for group_name, heads in target_groups.items():
            print(f"\n  {group_name} targets ({len(heads)} heads):")
            for h in heads:
                ind_row = induction_df[
                    (induction_df.layer == h.layer) & (induction_df.head == h.head)
                ]
                r2_row = r2_df[
                    (r2_df.layer == h.layer) & (r2_df.head == h.head)
                ]
                ind_str = (
                    f"{ind_row.iloc[0]['induction_score']:.4f}" if len(ind_row) > 0 else "N/A"
                )
                r2_str = (
                    f"{r2_row.iloc[0]['mean_r2']:.4f}" if len(r2_row) > 0 else "N/A"
                )
                print(f"    {h}  induction={ind_str}  R2={r2_str}")

        # ── Phase 2: Activation patching ─────────────────────────────────
        print(f"\n[3/4] Running resample activation patching...")
        t0 = time.time()
        patch_df = run_activation_patching(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            model_spec=model_spec,
            device=args.device,
            target_groups=target_groups,
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
        patch_df, r2_df, prev_token_df, target_groups, args.model, args.num_pairs,
    )

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"RESULTS -- {args.model}")
    print(f"{'='*72}")

    for group_name in ["induction", "random_mid", "low_si_late"]:
        gr = analysis["per_group"].get(group_name, {})
        if "error" in gr:
            print(f"\n  {group_name.upper()}: ERROR - {gr['error']}")
            continue
        print(f"\n  {group_name.upper()}: disruption vs R2")
        print(f"    Pearson:  r={gr['pearson_r']:+.4f} (p={gr['pearson_p']:.2e})")
        print(f"    Spearman: rho={gr['spearman_rho']:+.4f} (p={gr['spearman_p']:.2e})")
        print(f"              95% CI [{gr['spearman_ci_95'][0]:+.4f}, "
              f"{gr['spearman_ci_95'][1]:+.4f}]")
        print(f"    Mean disruption: {gr['mean_disruption_all_sources']:.6f}")
        t20 = gr["top_20_disruptive"]
        print(f"    Top-20: {t20['n_above_r2_median']}/20 above R2 median")
        for h in t20["heads"][:5]:
            print(f"      {h['source']:8s}  disruption={h['mean_disruption']:+.6f}  "
                  f"R2={h['r2']:.4f}")
        if gr.get("prev_token_partial"):
            ptc = gr["prev_token_partial"]
            print(f"    Partial (ctrl prev_token): "
                  f"r={ptc['partial_r2_vs_disruption_ctrl_prev_token']:+.4f} "
                  f"(p={ptc['partial_r2_vs_disruption_ctrl_prev_token_p']:.2e})")

    comp = analysis["cross_group_comparison"]
    print(f"\n  CROSS-GROUP COMPARISON:")
    print(f"    rho(induction):    {comp['rho_induction']}")
    print(f"    rho(random_mid):   {comp['rho_random_mid']}")
    print(f"    rho(low_si_late):  {comp['rho_low_si_late']}")
    print(f"    Induction-specific: {comp['induction_specific']}")
    for comp_name, details in comp.get("pairwise_specificity_tests", {}).items():
        if "error" in details:
            print(f"    induction vs {comp_name}: {details['error']} (n={details.get('n_sources')})")
            continue
        print(
            f"    induction vs {comp_name}: delta={details['rho_delta']:+.4f}, "
            f"t={details['williams_t']:+.4f}, p={details['p_value']:.2e}, "
            f"p_holm={details['p_value_holm']:.2e}"
        )

    print(f"\n  Mean disruption by group:")
    print(f"    induction:    {comp['mean_disruption_induction']}")
    print(f"    random_mid:   {comp['mean_disruption_random_mid']}")
    print(f"    low_si_late:  {comp['mean_disruption_low_si_late']}")

    print(f"\n  VERDICT: {analysis['interpretation']}")

    # ── Save report ──────────────────────────────────────────────────────
    report = {
        "experiment": "3_theory10_feeder_specificity",
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_sequence_pairs": args.num_pairs,
            "period": KNOCKOUT_PERIOD,
            "seq_len": KNOCKOUT_SEQ_LEN,
            "num_target_per_group": NUM_TARGET_PER_GROUP,
            "max_source_layer": MAX_SOURCE_LAYER,
            "random_mid_layer_range": list(RANDOM_MID_LAYER_RANGE),
            "low_si_late_layer_range": list(LOW_SI_LATE_LAYER_RANGE),
            "bootstrap_n": BOOTSTRAP_N,
            "target_selection_seed": TARGET_SELECTION_SEED,
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
