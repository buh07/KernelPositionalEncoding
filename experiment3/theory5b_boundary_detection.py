#!/usr/bin/env python3
"""
Experiment 3.5b: Boundary detection hypothesis
===============================================

Hypothesis
----------
High-SI heads serve as word-boundary detectors -- their position-relative
attention patterns help the model detect word transitions.  This explains why
ablating them disproportionately damages word-initial predictions (Theory 5
result: interaction = -1.30 in Llama, -0.39 in OLMo).

Three complementary approaches:

Approach A -- Attention flow at boundaries:
    At word-boundary positions, measure how much high-SI vs low-SI heads
    attend back to the last token of the previous word.

Approach B -- Fine-grained position taxonomy ablation:
    4-way position classification:
      (1) mid_continuation    -- mid-word, next token also same word
      (2) last_subword        -- last token of a multi-subword word
      (3) word_initial_after_multi  -- first of new word, prev word had multiple subwords
      (4) word_initial_after_single -- first of new word, prev word was single token
    Ablation damage should follow: (3) > (4) > (2) > (1) under high-SI ablation.

Approach C -- Boundary attention score vs R2 correlation:
    Per-head "boundary attention score" = mean attn[t, t-1] at word
    boundaries minus mean attn[t, t-1] within words.
    Correlate with R2 (shift-invariance).

Usage
-----
    python scripts/experiment3_theory5b_boundary_detection.py --model llama-3.1-8b --device cuda:2
    python scripts/experiment3_theory5b_boundary_detection.py --model olmo-2-7b --device cuda:3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment3.stats_utils import holm_adjust
from shared.attention.adapters import get_adapter
from shared.specs import ModelSpec
from shared.models.loading import load_model, load_tokenizer

# ── constants ────────────────────────────────────────────────────────────────
TARGET_EXAMPLES_PER_TYPE = 500
EVAL_SEQ_LEN = 1024
MAX_SEQUENCES = 200
NUM_ATTN_SEQUENCES = 50
RANDOM_DRAWS = 3
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

POSITION_TYPES = [
    "mid_continuation",
    "last_subword",
    "word_initial_after_multi",
    "word_initial_after_single",
]


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


@dataclass
class TokenPosition:
    sequence_idx: int
    position: int
    token_type: str
    token_id: int


# ═══════════════════════════════════════════════════════════════════════════════
# HEAD-OUTPUT ABLATION (reused from theory5)
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def head_output_ablation(model, heads_to_zero: list[HeadID]):
    """Zero the attention output of specified heads via o_proj pre-hooks."""
    if not heads_to_zero:
        yield
        return

    heads_by_layer: dict[int, set[int]] = {}
    for h in heads_to_zero:
        heads_by_layer.setdefault(h.layer, set()).add(h.head)

    handles: list[torch.utils.hooks.RemovableHandle] = []
    config = model.config
    num_query_heads = int(getattr(config, "num_attention_heads"))
    hidden_size = int(getattr(config, "hidden_size"))
    head_dim = hidden_size // num_query_heads
    layers = getattr(getattr(model, "model"), "layers")

    for layer_idx, layer in enumerate(layers):
        if layer_idx not in heads_by_layer:
            continue
        target_heads = heads_by_layer[layer_idx]
        o_proj = layer.self_attn.o_proj

        def make_hook(target_set: set[int], n_heads: int, h_dim: int):
            def hook(module, args):
                x = args[0]
                batch, seq, hidden = x.shape
                view = x.view(batch, seq, n_heads, h_dim).clone()
                for h_idx in target_set:
                    if h_idx < n_heads:
                        view[:, :, h_idx, :] = 0.0
                return (view.reshape(batch, seq, hidden),) + args[1:]
            return hook

        handle = o_proj.register_forward_pre_hook(
            make_hook(target_heads, num_query_heads, head_dim),
            with_kwargs=False,
        )
        handles.append(handle)

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# WORD BOUNDARY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_word_boundaries(tokenizer, token_ids: list[int]) -> list[int]:
    """Assign a word index to each token using whitespace / Ġ / ▁ heuristics."""
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    word_ids: list[int] = []
    current_word = 0
    for i, ts in enumerate(token_strings):
        if ts is None:
            word_ids.append(current_word)
            continue
        if i == 0:
            word_ids.append(current_word)
        elif ts.startswith("\u0120") or ts.startswith("\u2581"):
            current_word += 1
            word_ids.append(current_word)
        else:
            word_ids.append(current_word)
    return word_ids


def compute_word_info(word_ids: list[int]) -> dict[int, dict[str, int]]:
    """Return {word_id: {'start': first_pos, 'end': last_pos, 'n_tokens': count}}."""
    info: dict[int, dict[str, int]] = {}
    for pos, wid in enumerate(word_ids):
        if wid not in info:
            info[wid] = {"start": pos, "end": pos, "n_tokens": 1}
        else:
            info[wid]["end"] = pos
            info[wid]["n_tokens"] += 1
    return info


def load_wiki_sequences(model_name: str, max_sequences: int, seq_len: int) -> list[list[int]]:
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
                if len(sequences) >= max_sequences:
                    break
        if len(sequences) >= max_sequences:
            break
    if len(sequences) < max_sequences:
        print(f"  WARNING: Only found {len(sequences)} sequences (wanted {max_sequences})")
    return sequences


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: BUILD 4-WAY POSITION TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════

def build_taxonomy_positions(
    tokenizer,
    sequences: list[list[int]],
    target_per_type: int,
) -> dict[str, list[TokenPosition]]:
    """Classify each position into one of 4 types.

    Returns {type_name: [TokenPosition, ...]}, truncated to target_per_type each.
    """
    positions: dict[str, list[TokenPosition]] = {t: [] for t in POSITION_TYPES}

    for seq_idx, tokens in enumerate(sequences):
        word_ids = compute_word_boundaries(tokenizer, tokens)
        word_info = compute_word_info(word_ids)

        for t in range(2, len(tokens)):
            token_id = tokens[t]
            if word_ids[t] == word_ids[t - 1]:
                # Continuation token
                if t + 1 < len(tokens) and word_ids[t + 1] == word_ids[t]:
                    ptype = "mid_continuation"
                else:
                    ptype = "last_subword"
            else:
                # Word-initial token
                prev_word_id = word_ids[t - 1]
                if word_info[prev_word_id]["n_tokens"] > 1:
                    ptype = "word_initial_after_multi"
                else:
                    ptype = "word_initial_after_single"

            positions[ptype].append(TokenPosition(
                sequence_idx=seq_idx, position=t,
                token_type=ptype, token_id=token_id,
            ))

        # Check if all types have enough
        if all(len(positions[t]) >= target_per_type for t in POSITION_TYPES):
            break

    # Truncate to target
    for ptype in POSITION_TYPES:
        positions[ptype] = positions[ptype][:target_per_type]

    return positions


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: LOAD HEAD CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_head_groups(model_name: str) -> dict[str, Any]:
    path = Path("results/experiment3/theory1_si_circuits") / model_name / "head_groups.json"
    if not path.exists():
        raise FileNotFoundError(f"Head groups not found at {path}")
    with open(path) as f:
        return json.load(f)


def parse_head_list(entries: list[dict]) -> list[HeadID]:
    return [HeadID(layer=e["layer"], head=e["head"]) for e in entries]


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH A + C: COMBINED ATTENTION CAPTURE PASS
# ═══════════════════════════════════════════════════════════════════════════════

def run_attention_analysis(
    model,
    adapter,
    model_spec: ModelSpec,
    tokenizer,
    device: str,
    sequences: list[list[int]],
    high_si: list[HeadID],
    low_si: list[HeadID],
    r2_df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Combined Approach A (boundary attention flow) and Approach C (boundary score).

    Returns (approach_a_results, approach_c_results).
    """
    high_si_set = set((h.layer, h.head) for h in high_si)
    low_si_set = set((h.layer, h.head) for h in low_si)

    n_seqs = min(len(sequences), NUM_ATTN_SEQUENCES)

    # Accumulators for Approach A.
    # We aggregate to per-head-per-sequence means, then test on per-head means.
    approach_a_high_prev_last: dict[tuple[int, int], list[float]] = defaultdict(list)
    approach_a_high_prev_first: dict[tuple[int, int], list[float]] = defaultdict(list)
    approach_a_high_cross_total: dict[tuple[int, int], list[float]] = defaultdict(list)
    approach_a_low_prev_last: dict[tuple[int, int], list[float]] = defaultdict(list)
    approach_a_low_prev_first: dict[tuple[int, int], list[float]] = defaultdict(list)
    approach_a_low_cross_total: dict[tuple[int, int], list[float]] = defaultdict(list)

    # Accumulators for Approach C: per-head boundary scores
    per_head_boundary_scores: dict[tuple[int, int], list[float]] = defaultdict(list)
    per_head_nonboundary_scores: dict[tuple[int, int], list[float]] = defaultdict(list)

    total_boundaries = 0

    for seq_idx in range(n_seqs):
        t0 = time.time()
        tokens = sequences[seq_idx]
        word_ids = compute_word_boundaries(tokenizer, tokens)
        word_info = compute_word_info(word_ids)

        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            capture = adapter.capture(
                model,
                input_ids=input_ids,
                include_logits=True,
                return_token_logits=False,
                capture_attention=True,
                output_device="cpu",
            )

        if capture.logits is None:
            print(f"  Seq {seq_idx}: no logits captured, skipping")
            del capture, input_ids
            torch.cuda.empty_cache()
            continue

        # capture.logits: [layers, heads, seq, seq] — pre-softmax
        logits = capture.logits.float()
        n_layers, n_heads, seq_len, _ = logits.shape

        # Apply causal mask and softmax
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        logits[:, :, :, :][..., causal_mask] = float('-inf')
        attn = torch.softmax(logits, dim=-1).numpy()  # [L, H, S, S]

        # Find boundary positions (word_ids[t] != word_ids[t-1])
        boundary_positions = [t for t in range(2, seq_len)
                              if word_ids[t] != word_ids[t - 1]]
        nonboundary_positions = [t for t in range(2, seq_len)
                                 if word_ids[t] == word_ids[t - 1]]
        total_boundaries += len(boundary_positions)

        # ── Approach A: attention flow at boundaries (vectorized) ──
        if boundary_positions:
            bp = np.array(boundary_positions)

            # For each boundary position t, the previous word's last token is t-1
            prev_last = bp - 1

            # Previous word's first token
            prev_first = np.array([
                word_info[word_ids[t - 1]]["start"] for t in boundary_positions
            ])

            for layer_idx in range(n_layers):
                for head_idx in range(n_heads):
                    head_attn = attn[layer_idx, head_idx]  # [S, S]

                    # Attention from boundary positions to previous word's last token
                    attn_prev_last = head_attn[bp, prev_last]  # [n_boundaries]
                    attn_prev_first = head_attn[bp, prev_first]

                    # Total cross-boundary attention (to all tokens of previous word)
                    cross_total = np.zeros(len(bp))
                    for i, t in enumerate(boundary_positions):
                        prev_wid = word_ids[t - 1]
                        wi = word_info[prev_wid]
                        cross_total[i] = head_attn[t, wi["start"]:wi["end"] + 1].sum()

                    if (layer_idx, head_idx) in high_si_set:
                        key = (layer_idx, head_idx)
                        approach_a_high_prev_last[key].append(float(np.mean(attn_prev_last)))
                        approach_a_high_prev_first[key].append(float(np.mean(attn_prev_first)))
                        approach_a_high_cross_total[key].append(float(np.mean(cross_total)))
                    elif (layer_idx, head_idx) in low_si_set:
                        key = (layer_idx, head_idx)
                        approach_a_low_prev_last[key].append(float(np.mean(attn_prev_last)))
                        approach_a_low_prev_first[key].append(float(np.mean(attn_prev_first)))
                        approach_a_low_cross_total[key].append(float(np.mean(cross_total)))

        # ── Approach C: boundary attention score (vectorized) ──
        if boundary_positions and nonboundary_positions:
            bp_arr = np.array(boundary_positions)
            nbp_arr = np.array(nonboundary_positions)

            for layer_idx in range(n_layers):
                for head_idx in range(n_heads):
                    head_attn = attn[layer_idx, head_idx]
                    # attn[t, t-1] at boundary vs non-boundary positions
                    boundary_prev = head_attn[bp_arr, bp_arr - 1].mean()
                    nonboundary_prev = head_attn[nbp_arr, nbp_arr - 1].mean()
                    per_head_boundary_scores[(layer_idx, head_idx)].append(float(boundary_prev))
                    per_head_nonboundary_scores[(layer_idx, head_idx)].append(float(nonboundary_prev))

        elapsed = time.time() - t0
        if (seq_idx + 1) % 5 == 0 or seq_idx == 0:
            print(f"  Attention analysis: {seq_idx + 1}/{n_seqs} sequences "
                  f"({len(boundary_positions)} boundaries, {elapsed:.1f}s)")

        del capture, input_ids, logits, attn
        torch.cuda.empty_cache()

    # ── Build Approach A results ──────────────────────────────────────────
    approach_a = _build_approach_a_results(
        approach_a_high_prev_last, approach_a_high_prev_first, approach_a_high_cross_total,
        approach_a_low_prev_last, approach_a_low_prev_first, approach_a_low_cross_total,
        total_boundaries,
    )

    # ── Build Approach C results ──────────────────────────────────────────
    approach_c = _build_approach_c_results(
        per_head_boundary_scores, per_head_nonboundary_scores, r2_df,
        high_si_set, low_si_set,
    )

    return approach_a, approach_c


def _build_approach_a_results(
    high_prev_last, high_prev_first, high_cross_total,
    low_prev_last, low_prev_first, low_cross_total,
    total_boundaries: int,
) -> dict[str, Any]:
    """Statistical tests comparing high-SI vs low-SI attention at word boundaries."""

    def _per_head_means(values_by_head: dict[tuple[int, int], list[float]]) -> np.ndarray:
        vals = [
            float(np.mean(v))
            for v in values_by_head.values()
            if len(v) > 0
        ]
        return np.asarray(vals, dtype=np.float64)

    def group_stats(values: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "median": float("nan"),
                "ci_95": [float("nan"), float("nan")],
                "n": 0,
            }
        _, ci_lo, ci_hi = bootstrap_ci(arr)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "ci_95": [ci_lo, ci_hi],
            "n": len(arr),
        }

    def compare_groups(high_vals: np.ndarray, low_vals: np.ndarray) -> dict[str, Any]:
        h = np.asarray(high_vals, dtype=np.float64)
        l = np.asarray(low_vals, dtype=np.float64)
        if len(h) < 2 or len(l) < 2:
            return {
                "t_statistic": float("nan"),
                "t_p_value": float("nan"),
                "mannwhitney_u": float("nan"),
                "mannwhitney_p": float("nan"),
                "cohens_d": float("nan"),
                "high_si_mean": float(np.mean(h)) if len(h) else float("nan"),
                "low_si_mean": float(np.mean(l)) if len(l) else float("nan"),
                "difference": float(np.mean(h) - np.mean(l)) if len(h) and len(l) else float("nan"),
                "n_high": int(len(h)),
                "n_low": int(len(l)),
            }
        t_stat, t_p = scipy_stats.ttest_ind(h, l, equal_var=False)
        u_stat, u_p = scipy_stats.mannwhitneyu(h, l, alternative="two-sided")
        d = cohens_d(h, l)
        return {
            "t_statistic": float(t_stat),
            "t_p_value": float(t_p),
            "mannwhitney_u": float(u_stat),
            "mannwhitney_p": float(u_p),
            "cohens_d": d,
            "high_si_mean": float(np.mean(h)),
            "low_si_mean": float(np.mean(l)),
            "difference": float(np.mean(h) - np.mean(l)),
            "n_high": int(len(h)),
            "n_low": int(len(l)),
        }

    high_prev_last_arr = _per_head_means(high_prev_last)
    high_prev_first_arr = _per_head_means(high_prev_first)
    high_cross_total_arr = _per_head_means(high_cross_total)
    low_prev_last_arr = _per_head_means(low_prev_last)
    low_prev_first_arr = _per_head_means(low_prev_first)
    low_cross_total_arr = _per_head_means(low_cross_total)

    result: dict[str, Any] = {
        "unit_of_analysis": "per_head_mean_over_sequences",
        "n_boundary_positions_total": total_boundaries,
        "high_si_heads": {
            "attn_to_prev_last": group_stats(high_prev_last_arr),
            "attn_to_prev_first": group_stats(high_prev_first_arr),
            "total_cross_boundary": group_stats(high_cross_total_arr),
        },
        "low_si_heads": {
            "attn_to_prev_last": group_stats(low_prev_last_arr),
            "attn_to_prev_first": group_stats(low_prev_first_arr),
            "total_cross_boundary": group_stats(low_cross_total_arr),
        },
    }

    # Statistical comparisons
    result["high_vs_low_comparison"] = {
        "attn_to_prev_last": compare_groups(high_prev_last_arr, low_prev_last_arr),
        "attn_to_prev_first": compare_groups(high_prev_first_arr, low_prev_first_arr),
        "total_cross_boundary": compare_groups(high_cross_total_arr, low_cross_total_arr),
    }
    pvals = {
        metric: vals["t_p_value"]
        for metric, vals in result["high_vs_low_comparison"].items()
    }
    holm = holm_adjust(pvals)
    for metric, p_adj in holm.items():
        result["high_vs_low_comparison"][metric]["t_p_value_holm"] = float(p_adj)
    result["multiplicity_policy"] = "holm across Approach A metrics"

    # Verdict
    comp = result["high_vs_low_comparison"]["attn_to_prev_last"]
    result["hypothesis_supported"] = bool(
        comp["difference"] > 0
        and np.isfinite(comp.get("t_p_value_holm", float("nan")))
        and comp["t_p_value_holm"] < 0.05
    )

    return result


def _build_approach_c_results(
    boundary_scores: dict[tuple[int, int], list[float]],
    nonboundary_scores: dict[tuple[int, int], list[float]],
    r2_df: pd.DataFrame,
    high_si_set: set[tuple[int, int]],
    low_si_set: set[tuple[int, int]],
) -> dict[str, Any]:
    """Correlate per-head boundary attention score with R2."""
    from scipy.stats import pearsonr, spearmanr

    rows = []
    for (layer, head), b_scores in boundary_scores.items():
        nb_scores = nonboundary_scores.get((layer, head), [])
        if not b_scores or not nb_scores:
            continue
        boundary_score = float(np.mean(b_scores) - np.mean(nb_scores))
        rows.append({
            "layer": layer,
            "head": head,
            "boundary_attn_score": boundary_score,
            "mean_boundary_prev_attn": float(np.mean(b_scores)),
            "mean_nonboundary_prev_attn": float(np.mean(nb_scores)),
        })

    score_df = pd.DataFrame(rows)

    # Merge with R2
    merged = pd.merge(score_df, r2_df, on=["layer", "head"], how="inner")

    if len(merged) < 5:
        return {"error": "Too few heads for correlation", "n_heads": len(merged)}

    # Correlations
    r_p, p_p = pearsonr(merged["mean_r2"], merged["boundary_attn_score"])
    r_s, p_s = spearmanr(merged["mean_r2"], merged["boundary_attn_score"])

    # Bootstrap CI for correlations
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    n = len(merged)
    boot_pearson = np.empty(BOOTSTRAP_N)
    boot_spearman = np.empty(BOOTSTRAP_N)
    r2_vals = merged["mean_r2"].values
    score_vals = merged["boundary_attn_score"].values
    for i in range(BOOTSTRAP_N):
        idx = rng.randint(0, n, size=n)
        bp, _ = pearsonr(r2_vals[idx], score_vals[idx])
        bs, _ = spearmanr(r2_vals[idx], score_vals[idx])
        boot_pearson[i] = bp
        boot_spearman[i] = bs

    # Group comparison
    high_si_scores = merged[
        merged.apply(lambda r: (int(r["layer"]), int(r["head"])) in high_si_set, axis=1)
    ]["boundary_attn_score"]
    low_si_scores = merged[
        merged.apply(lambda r: (int(r["layer"]), int(r["head"])) in low_si_set, axis=1)
    ]["boundary_attn_score"]

    if len(high_si_scores) >= 2 and len(low_si_scores) >= 2:
        t_grp, p_grp = scipy_stats.ttest_ind(high_si_scores, low_si_scores)
        d_grp = cohens_d(high_si_scores.values, low_si_scores.values)
    else:
        t_grp, p_grp, d_grp = float("nan"), float("nan"), float("nan")

    # Top-20 highest R2 heads' boundary scores
    top20 = merged.sort_values("mean_r2", ascending=False).head(20)

    result = {
        "n_heads": len(merged),
        "correlation": {
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
            "pearson_ci_95": [float(np.percentile(boot_pearson, 2.5)),
                              float(np.percentile(boot_pearson, 97.5))],
            "spearman_rho": float(r_s),
            "spearman_p": float(p_s),
            "spearman_ci_95": [float(np.percentile(boot_spearman, 2.5)),
                                float(np.percentile(boot_spearman, 97.5))],
        },
        "group_comparison": {
            "high_si_mean_boundary_score": float(high_si_scores.mean()) if len(high_si_scores) > 0 else float("nan"),
            "low_si_mean_boundary_score": float(low_si_scores.mean()) if len(low_si_scores) > 0 else float("nan"),
            "n_high_si": len(high_si_scores),
            "n_low_si": len(low_si_scores),
            "t_statistic": float(t_grp),
            "p_value": float(p_grp),
            "cohens_d": d_grp,
        },
        "top_20_by_r2": [
            {"layer": int(r.layer), "head": int(r.head),
             "mean_r2": float(r.mean_r2),
             "boundary_attn_score": float(r.boundary_attn_score)}
            for r in top20.itertuples()
        ],
        "hypothesis_supported": bool(r_s > 0 and p_s < 0.05),
    }

    # Save the full per-head data
    result["_score_df_path"] = "boundary_attention_scores.parquet"

    return result, score_df


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH B: FINE-GRAINED TAXONOMY ABLATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_per_position_losses(
    model,
    device: str,
    sequences: list[list[int]],
    positions: list[TokenPosition],
    heads_to_zero: list[HeadID],
) -> np.ndarray:
    """Compute cross-entropy loss at each specified position under ablation."""
    pos_by_seq: dict[int, list[tuple[int, int]]] = {}
    for i, p in enumerate(positions):
        pos_by_seq.setdefault(p.sequence_idx, []).append((i, p.position))

    losses = np.full(len(positions), np.nan, dtype=np.float64)

    with head_output_ablation(model, heads_to_zero):
        for seq_idx in sorted(pos_by_seq.keys()):
            tokens = sequences[seq_idx]
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, use_cache=False)
                logits = outputs.logits[0]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            for result_idx, pos_t in pos_by_seq[seq_idx]:
                target_token = tokens[pos_t]
                ce_loss = -log_probs[pos_t - 1, target_token].item()
                losses[result_idx] = ce_loss

            del input_ids, outputs, logits, log_probs
            torch.cuda.empty_cache()

    return losses


def run_approach_b(
    model,
    device: str,
    sequences: list[list[int]],
    taxonomy_positions: dict[str, list[TokenPosition]],
    conditions: list[tuple[str, list[HeadID]]],
) -> dict[str, Any]:
    """Run ablation for all conditions and 4 position types."""

    # Combine all positions for efficient batch computation
    all_positions: list[TokenPosition] = []
    type_slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for ptype in POSITION_TYPES:
        positions = taxonomy_positions[ptype]
        all_positions.extend(positions)
        type_slices[ptype] = (offset, offset + len(positions))
        offset += len(positions)

    # Run each condition
    condition_losses: dict[str, dict[str, np.ndarray]] = {}

    for cond_name, heads in conditions:
        t0 = time.time()
        print(f"\n    Condition: {cond_name} ({len(heads)} heads ablated)")
        all_losses = compute_per_position_losses(
            model, device, sequences, all_positions, heads,
        )
        # Split by type
        type_losses: dict[str, np.ndarray] = {}
        for ptype in POSITION_TYPES:
            start, end = type_slices[ptype]
            type_losses[ptype] = all_losses[start:end]
        condition_losses[cond_name] = type_losses
        elapsed = time.time() - t0
        means = {t: f"{np.mean(type_losses[t]):.4f}" for t in POSITION_TYPES}
        print(f"      Mean losses: {means} ({elapsed:.0f}s)")

    return analyze_approach_b(condition_losses)


def analyze_approach_b(
    condition_losses: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Statistical analysis of the 4-way taxonomy ablation."""

    baseline = condition_losses["none"]
    analysis: dict[str, Any] = {}

    # Baseline stats per type
    analysis["baseline"] = {}
    for ptype in POSITION_TYPES:
        bl = baseline[ptype]
        analysis["baseline"][ptype] = {
            "mean_loss": float(np.mean(bl)),
            "std_loss": float(np.std(bl, ddof=1)),
            "median_loss": float(np.median(bl)),
            "n": len(bl),
        }

    # Per-condition, per-type loss increases
    analysis["conditions"] = {}
    for cond_name, cond_data in condition_losses.items():
        if cond_name == "none":
            continue
        cond_result: dict[str, Any] = {}
        for ptype in POSITION_TYPES:
            increase = cond_data[ptype] - baseline[ptype]
            mean_val, ci_lo, ci_hi = bootstrap_ci(increase)
            cond_result[ptype] = {
                "mean_loss_increase": float(np.mean(increase)),
                "std_loss_increase": float(np.std(increase, ddof=1)),
                "median_loss_increase": float(np.median(increase)),
                "ci_95": [ci_lo, ci_hi],
            }
        analysis["conditions"][cond_name] = cond_result

    # Pairwise comparisons within each condition
    analysis["pairwise_comparisons"] = {}
    type_pairs = [
        ("word_initial_after_multi", "mid_continuation"),
        ("word_initial_after_single", "mid_continuation"),
        ("word_initial_after_multi", "word_initial_after_single"),
        ("last_subword", "mid_continuation"),
        ("word_initial_after_multi", "last_subword"),
        ("word_initial_after_single", "last_subword"),
    ]

    for cond_name, cond_data in condition_losses.items():
        if cond_name == "none":
            continue
        cond_pairs: dict[str, Any] = {}
        for type_a, type_b in type_pairs:
            inc_a = cond_data[type_a] - baseline[type_a]
            inc_b = cond_data[type_b] - baseline[type_b]
            t_stat, t_p = scipy_stats.ttest_ind(inc_a, inc_b)
            u_stat, u_p = scipy_stats.mannwhitneyu(inc_a, inc_b, alternative="two-sided")
            d = cohens_d(inc_a, inc_b)
            diff_mean = float(np.mean(inc_a) - np.mean(inc_b))
            cond_pairs[f"{type_a}_vs_{type_b}"] = {
                "mean_difference": diff_mean,
                "t_statistic": float(t_stat),
                "t_p_value": float(t_p),
                "mannwhitney_u": float(u_stat),
                "mannwhitney_p": float(u_p),
                "cohens_d": d,
            }
        analysis["pairwise_comparisons"][cond_name] = cond_pairs

    # Monotonic trend test: does high-SI damage follow predicted ordering?
    # Predicted: mid_cont < last_subword < word_init_single < word_init_multi
    predicted_order = ["mid_continuation", "last_subword",
                       "word_initial_after_single", "word_initial_after_multi"]

    for cond_name in ["ablate_high_si", "ablate_low_si"]:
        if cond_name not in condition_losses or cond_name == "none":
            continue
        cond_data = condition_losses[cond_name]
        means = [float(np.mean(cond_data[t] - baseline[t])) for t in predicted_order]
        ranks = list(range(len(predicted_order)))
        rho, rho_p = scipy_stats.spearmanr(ranks, means)
        analysis.setdefault("monotonic_trend", {})[cond_name] = {
            "predicted_order": predicted_order,
            "observed_means": means,
            "spearman_rho_with_rank": float(rho),
            "spearman_p": float(rho_p),
            "ordering_matches_prediction": bool(
                means == sorted(means) or rho > 0.8
            ),
        }

    # Two-way interaction: (word_initial damage - continuation damage) for high vs low SI
    if "ablate_high_si" in condition_losses and "ablate_low_si" in condition_losses:
        high_data = condition_losses["ablate_high_si"]
        low_data = condition_losses["ablate_low_si"]

        # Aggregate word-initial and continuation damages
        high_init_damage = np.concatenate([
            high_data["word_initial_after_multi"] - baseline["word_initial_after_multi"],
            high_data["word_initial_after_single"] - baseline["word_initial_after_single"],
        ])
        high_cont_damage = np.concatenate([
            high_data["mid_continuation"] - baseline["mid_continuation"],
            high_data["last_subword"] - baseline["last_subword"],
        ])
        low_init_damage = np.concatenate([
            low_data["word_initial_after_multi"] - baseline["word_initial_after_multi"],
            low_data["word_initial_after_single"] - baseline["word_initial_after_single"],
        ])
        low_cont_damage = np.concatenate([
            low_data["mid_continuation"] - baseline["mid_continuation"],
            low_data["last_subword"] - baseline["last_subword"],
        ])

        high_boundary_sensitivity = float(np.mean(high_init_damage) - np.mean(high_cont_damage))
        low_boundary_sensitivity = float(np.mean(low_init_damage) - np.mean(low_cont_damage))
        interaction = high_boundary_sensitivity - low_boundary_sensitivity

        analysis["interaction"] = {
            "high_si_boundary_sensitivity": high_boundary_sensitivity,
            "low_si_boundary_sensitivity": low_boundary_sensitivity,
            "interaction_high_minus_low": interaction,
            "description": (
                "Positive boundary_sensitivity means word-initial damage > continuation damage. "
                "Positive interaction means high-SI has MORE boundary sensitivity than low-SI. "
                "Hypothesis predicts: high-SI boundary_sensitivity > 0 AND interaction > 0."
            ),
        }

    # Verdict
    if "ablate_high_si" in analysis.get("pairwise_comparisons", {}):
        primary = analysis["pairwise_comparisons"]["ablate_high_si"].get(
            "word_initial_after_multi_vs_mid_continuation", {}
        )
        analysis["hypothesis_supported"] = bool(
            primary.get("mean_difference", 0) > 0 and primary.get("t_p_value", 1) < 0.05
        )
    else:
        analysis["hypothesis_supported"] = False

    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    data: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = BOOTSTRAP_N,
    ci: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (point_estimate, ci_low, ci_high)."""
    rng = np.random.RandomState(seed)
    point = float(statistic_fn(data))
    boot_stats = np.empty(n_bootstrap)
    n = len(data)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_stats[i] = statistic_fn(data[idx])
    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(boot_stats, 100 * alpha))
    ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return point, ci_low, ci_high


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for two independent samples."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled_std = math.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    if pooled_std == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled_std)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3.5b: Boundary detection hypothesis"
    )
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output-dir", default="results/experiment3/theory5b_boundary_detection")
    parser.add_argument("--max-sequences", type=int, default=MAX_SEQUENCES)
    parser.add_argument("--target-examples", type=int, default=TARGET_EXAMPLES_PER_TYPE)
    parser.add_argument("--num-attn-sequences", type=int, default=NUM_ATTN_SEQUENCES)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_spec = MODELS[args.model]

    print(f"{'='*72}")
    print(f"Experiment 3.5b: Boundary Detection Hypothesis — {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"{'='*72}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\n[1/8] Loading model and tokenizer...")
    loader = load_model(model_spec)
    model = loader.model.to(args.device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)
    adapter = get_adapter(model_spec)
    adapter.register(model)
    print(f"  Model loaded: {model_spec.hf_id}")

    # ── Load wiki sequences ─────────────────────────────────────────────────
    print(f"\n[2/8] Loading wiki sequences...")
    sequences = load_wiki_sequences(args.model, args.max_sequences, EVAL_SEQ_LEN)
    print(f"  Loaded {len(sequences)} sequences (len={EVAL_SEQ_LEN})")

    # ── Build 4-way taxonomy ────────────────────────────────────────────────
    print(f"\n[3/8] Building 4-way position taxonomy...")
    taxonomy_positions = build_taxonomy_positions(
        tokenizer, sequences, args.target_examples,
    )
    for ptype in POSITION_TYPES:
        print(f"  {ptype}: {len(taxonomy_positions[ptype])} positions")

    # ── Load head groups ────────────────────────────────────────────────────
    print(f"\n[4/8] Loading head classification from Theory 1...")
    head_groups = load_head_groups(args.model)
    high_si = parse_head_list(head_groups["high_si"])
    low_si = parse_head_list(head_groups["low_si"])
    n_select = head_groups["n_select"]
    n_total = head_groups["n_total_heads"]
    print(f"  {n_select} heads per group ({n_total} total)")

    # Random draws
    random_draws_heads: list[list[HeadID]] = []
    for draw in range(RANDOM_DRAWS):
        key = f"random_draw{draw}"
        if key in head_groups:
            random_draws_heads.append(parse_head_list(head_groups[key]))
        else:
            n_layers = model.config.num_hidden_layers
            n_heads_per_layer = model.config.num_attention_heads
            all_heads = [HeadID(l, h) for l in range(n_layers) for h in range(n_heads_per_layer)]
            rng = random.Random(42 + draw * 1000)
            random_draws_heads.append(rng.sample(all_heads, k=n_select))

    # ── Load R2 data for Approach C ─────────────────────────────────────────
    print(f"\n[5/8] Loading R² data for correlation analysis...")
    r2_path = Path("results/experiment3/theory1_si_circuits") / args.model / "per_sequence_r2.parquet"
    r2_per_seq = pd.read_parquet(r2_path)
    r2_df = r2_per_seq.groupby(["layer", "head"])["r2"].mean().reset_index()
    r2_df.columns = ["layer", "head", "mean_r2"]
    print(f"  Loaded R² for {len(r2_df)} heads")

    # ══════════════════════════════════════════════════════════════════════════
    # APPROACHES A + C: ATTENTION ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[6/8] Running Approaches A + C (attention analysis, "
          f"{args.num_attn_sequences} sequences)...")
    t0 = time.time()
    approach_a, (approach_c, score_df) = run_attention_analysis(
        model=model,
        adapter=adapter,
        model_spec=model_spec,
        tokenizer=tokenizer,
        device=args.device,
        sequences=sequences[:args.num_attn_sequences],
        high_si=high_si,
        low_si=low_si,
        r2_df=r2_df,
    )
    elapsed_ac = time.time() - t0
    score_df.to_parquet(output_dir / "boundary_attention_scores.parquet", index=False)
    print(f"  Approaches A + C complete in {elapsed_ac:.0f}s")

    # Print Approach A summary
    print(f"\n  ── APPROACH A: Attention flow at word boundaries ──")
    for group in ("high_si_heads", "low_si_heads"):
        gdata = approach_a[group]
        print(f"  {group}:")
        for metric in ("attn_to_prev_last", "attn_to_prev_first", "total_cross_boundary"):
            m = gdata[metric]
            print(f"    {metric}: mean={m['mean']:.4f} "
                  f"(95% CI [{m['ci_95'][0]:.4f}, {m['ci_95'][1]:.4f}])")
    comp = approach_a["high_vs_low_comparison"]["attn_to_prev_last"]
    print(f"\n  High vs Low (attn_to_prev_last): diff={comp['difference']:+.4f}, "
          f"t={comp['t_statistic']:.3f}, p={comp['t_p_value']:.2e}, "
          f"p_holm={comp.get('t_p_value_holm', float('nan')):.2e}, d={comp['cohens_d']:.3f}")
    print(f"  Approach A supports hypothesis: {approach_a['hypothesis_supported']}")

    # Print Approach C summary
    print(f"\n  ── APPROACH C: Boundary attention score vs R² ──")
    corr = approach_c["correlation"]
    print(f"  Pearson:  r={corr['pearson_r']:+.4f} (p={corr['pearson_p']:.2e}), "
          f"95% CI [{corr['pearson_ci_95'][0]:+.4f}, {corr['pearson_ci_95'][1]:+.4f}]")
    print(f"  Spearman: rho={corr['spearman_rho']:+.4f} (p={corr['spearman_p']:.2e}), "
          f"95% CI [{corr['spearman_ci_95'][0]:+.4f}, {corr['spearman_ci_95'][1]:+.4f}]")
    gc = approach_c["group_comparison"]
    print(f"  High-SI mean boundary score: {gc['high_si_mean_boundary_score']:.4f}")
    print(f"  Low-SI  mean boundary score: {gc['low_si_mean_boundary_score']:.4f}")
    print(f"  Group t-test: t={gc['t_statistic']:.3f}, p={gc['p_value']:.2e}, d={gc['cohens_d']:.3f}")
    print(f"  Approach C supports hypothesis: {approach_c['hypothesis_supported']}")

    # ══════════════════════════════════════════════════════════════════════════
    # APPROACH B: TAXONOMY ABLATION
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[7/8] Running Approach B (4-way taxonomy ablation)...")
    t0 = time.time()

    conditions: list[tuple[str, list[HeadID]]] = [
        ("none", []),
        ("ablate_high_si", high_si),
        ("ablate_low_si", low_si),
    ]
    for draw in range(RANDOM_DRAWS):
        conditions.append((f"ablate_random_draw{draw}", random_draws_heads[draw]))

    approach_b = run_approach_b(
        model=model,
        device=args.device,
        sequences=sequences,
        taxonomy_positions=taxonomy_positions,
        conditions=conditions,
    )
    elapsed_b = time.time() - t0
    print(f"  Approach B complete in {elapsed_b:.0f}s")

    # Print Approach B summary
    print(f"\n  ── APPROACH B: 4-way taxonomy ablation ──")
    for cond_name in ["ablate_high_si", "ablate_low_si"]:
        if cond_name not in approach_b.get("conditions", {}):
            continue
        cond = approach_b["conditions"][cond_name]
        print(f"\n  {cond_name}:")
        for ptype in POSITION_TYPES:
            d = cond[ptype]
            print(f"    {ptype:30s}: increase={d['mean_loss_increase']:+.4f} "
                  f"(95% CI [{d['ci_95'][0]:+.4f}, {d['ci_95'][1]:+.4f}])")

    if "pairwise_comparisons" in approach_b and "ablate_high_si" in approach_b["pairwise_comparisons"]:
        print(f"\n  Pairwise comparisons (high-SI ablation):")
        for pair_name, pair_data in approach_b["pairwise_comparisons"]["ablate_high_si"].items():
            print(f"    {pair_name}: diff={pair_data['mean_difference']:+.4f}, "
                  f"t={pair_data['t_statistic']:.3f}, p={pair_data['t_p_value']:.2e}, "
                  f"d={pair_data['cohens_d']:.3f}")

    if "monotonic_trend" in approach_b:
        for cond_name, trend in approach_b["monotonic_trend"].items():
            print(f"\n  Monotonic trend ({cond_name}):")
            print(f"    Predicted order: {trend['predicted_order']}")
            print(f"    Observed means:  {[f'{m:+.4f}' for m in trend['observed_means']]}")
            print(f"    Spearman with rank: rho={trend['spearman_rho_with_rank']:+.4f}, "
                  f"p={trend['spearman_p']:.4f}")

    if "interaction" in approach_b:
        inter = approach_b["interaction"]
        print(f"\n  Two-way interaction (condition x position_category):")
        print(f"    High-SI boundary sensitivity: {inter['high_si_boundary_sensitivity']:+.4f}")
        print(f"    Low-SI  boundary sensitivity: {inter['low_si_boundary_sensitivity']:+.4f}")
        print(f"    Interaction (high - low):     {inter['interaction_high_minus_low']:+.4f}")

    print(f"\n  Approach B supports hypothesis: {approach_b.get('hypothesis_supported', False)}")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[8/8] Writing final report...")

    overall_a = approach_a.get("hypothesis_supported", False)
    overall_b = approach_b.get("hypothesis_supported", False)
    overall_c = approach_c.get("hypothesis_supported", False)

    overall = {
        "approach_a_supports": overall_a,
        "approach_b_supports": overall_b,
        "approach_c_supports": overall_c,
        "hypothesis_supported": bool(sum([overall_a, overall_b, overall_c]) >= 2),
        "summary": (
            f"Boundary detection hypothesis: "
            f"{'SUPPORTED' if sum([overall_a, overall_b, overall_c]) >= 2 else 'NOT SUPPORTED'}. "
            f"A={'Y' if overall_a else 'N'}, "
            f"B={'Y' if overall_b else 'N'}, "
            f"C={'Y' if overall_c else 'N'}."
        ),
    }

    report = {
        "experiment": "3.5b_boundary_detection",
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "eval_seq_len": EVAL_SEQ_LEN,
            "num_attn_sequences": args.num_attn_sequences,
            "max_sequences": args.max_sequences,
            "target_per_type": args.target_examples,
            "bootstrap_n": BOOTSTRAP_N,
            "random_draws": RANDOM_DRAWS,
        },
        "position_counts": {t: len(taxonomy_positions[t]) for t in POSITION_TYPES},
        "approach_a": approach_a,
        "approach_b": approach_b,
        "approach_c": approach_c,
        "overall_verdict": overall,
    }

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"  Report saved to {output_dir / 'report.json'}")

    # Final summary
    print(f"\n{'='*72}")
    print(f"EXPERIMENT 3.5b COMPLETE — {args.model}")
    print(f"{'='*72}")
    print(f"\n  {overall['summary']}")
    print(f"\n  Total runtime: {elapsed_ac + elapsed_b:.0f}s")
    print(f"  All artifacts in: {output_dir}")
    print(f"  Done.")


if __name__ == "__main__":
    main()
