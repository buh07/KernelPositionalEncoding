#!/usr/bin/env python3
"""
Experiment 3.5: Subword ablation interaction (causal test of the subword-assembly hypothesis)
=============================================================================================

Hypothesis
----------
If high-SI heads specialize in assembling multi-token words (as suggested by
rho = +0.55 adjacent-subword correlation from 3.2b), then ablating them should
disproportionately hurt the model's ability to predict tokens that are
continuations within a multi-token word, compared to tokens that begin new words.

Protocol
--------
Phase 1 — Construct evaluation sets:
    - Multi-token continuation set: positions where the model must predict a
      token that continues a multi-token word (~500 examples)
    - Word-initial set: positions where the model must predict a token that
      begins a new word (~500 examples, baseline-perplexity matched)

Phase 2 — Load head classification from Theory 1 results

Phase 3 — Ablation evaluation:
    Conditions: none, ablate_high_si, ablate_low_si, ablate_random (x3)
    Metric: per-token cross-entropy loss

Phase 4 — Analysis:
    Primary test: interaction (high-SI damage on continuation) minus
    (high-SI damage on word-initial) vs same interaction for low-SI and random.
    Statistical tests: paired t-test, Wilcoxon, Cohen's d, bootstrap CIs.

Usage
-----
    # GPU 0 (Llama)
    python scripts/experiment3_theory5_subword_ablation.py --model llama-3.1-8b --device cuda:0

    # GPU 1 (OLMo)
    python scripts/experiment3_theory5_subword_ablation.py --model olmo-2-7b --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
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
from shared.specs import ModelSpec

try:
    from shared.attention.adapters import get_adapter
    from shared.models.loading import load_model, load_tokenizer
except Exception:  # pragma: no cover - optional for analysis-only environments
    get_adapter = None
    load_model = None
    load_tokenizer = None

# ── constants ────────────────────────────────────────────────────────────────
TARGET_EXAMPLES_PER_TYPE = 500
EVAL_SEQ_LEN = 1024
MAX_SEQUENCES = 200          # max wiki sequences to scan
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


# ── data structures ──────────────────────────────────────────────────────────
@dataclass
class HeadID:
    layer: int
    head: int

    def __hash__(self):
        return hash((self.layer, self.head))

    def __eq__(self, other):
        return isinstance(other, HeadID) and self.layer == other.layer and self.head == other.head


@dataclass
class TokenPosition:
    """A single evaluation position from a wiki sequence."""
    sequence_idx: int
    position: int           # position t in the sequence (predicting token at t)
    token_type: str         # "continuation" or "word_initial"
    token_id: int           # ground-truth token at position t


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: CONSTRUCT EVALUATION SETS
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
            # Ġ (GPT/Llama style) or ▁ (SentencePiece style) => new word
            current_word += 1
            word_ids.append(current_word)
        else:
            word_ids.append(current_word)
    return word_ids


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


def build_evaluation_sets(
    tokenizer,
    sequences: list[list[int]],
    target_per_type: int,
) -> tuple[list[TokenPosition], list[TokenPosition]]:
    """
    Build continuation and word-initial evaluation sets from wiki sequences.

    For position t (t >= 1), we classify the *predicted* token (at position t)
    as either a multi-token word continuation or a word-initial token.
    - Continuation: token at t has the same word_id as token at t-1
    - Word-initial: token at t has a different word_id from token at t-1

    We skip position 0 (no preceding context to predict from) and position 1
    if it is a BOS token continuation.
    """
    continuation_positions: list[TokenPosition] = []
    word_initial_positions: list[TokenPosition] = []

    for seq_idx, tokens in enumerate(sequences):
        word_ids = compute_word_boundaries(tokenizer, tokens)

        # Start from position 2 to avoid BOS edge cases
        # (position 0 = BOS or first real token; position 1 may be special)
        for t in range(2, len(tokens)):
            if word_ids[t] == word_ids[t - 1]:
                # Token at t continues the same word as token at t-1
                pos = TokenPosition(
                    sequence_idx=seq_idx,
                    position=t,
                    token_type="continuation",
                    token_id=tokens[t],
                )
                continuation_positions.append(pos)
            else:
                # Token at t starts a new word
                pos = TokenPosition(
                    sequence_idx=seq_idx,
                    position=t,
                    token_type="word_initial",
                    token_id=tokens[t],
                )
                word_initial_positions.append(pos)

        if (len(continuation_positions) >= target_per_type
                and len(word_initial_positions) >= target_per_type):
            break

    print(f"  Raw counts: {len(continuation_positions)} continuation, "
          f"{len(word_initial_positions)} word-initial positions")

    # Truncate to target size (take first N for reproducibility)
    continuation_positions = continuation_positions[:target_per_type]
    word_initial_positions = word_initial_positions[:target_per_type]

    print(f"  Selected: {len(continuation_positions)} continuation, "
          f"{len(word_initial_positions)} word-initial positions")

    return continuation_positions, word_initial_positions


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: LOAD HEAD CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_head_groups(model_name: str) -> dict[str, Any]:
    """Load head classification from Theory 1 results."""
    path = Path("results/experiment3/theory1_si_circuits") / model_name / "head_groups.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Head groups not found at {path}. Run experiment3_theory1_si_circuits.py first."
        )
    with open(path) as f:
        data = json.load(f)
    return data


def parse_head_list(entries: list[dict]) -> list[HeadID]:
    """Convert JSON head entries to HeadID objects."""
    return [HeadID(layer=e["layer"], head=e["head"]) for e in entries]


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: HEAD-OUTPUT ABLATION
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def head_output_ablation(model, heads_to_zero: list[HeadID]):
    """
    Context manager that zeros the attention output of specified heads by
    hooking into each attention layer's o_proj input.

    The o_proj input has shape [batch, seq, num_query_heads * head_dim].
    We reshape to [batch, seq, num_query_heads, head_dim], zero targeted heads,
    and reshape back.
    """
    if not heads_to_zero:
        yield
        return

    # Build per-layer head sets
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
                x = args[0]  # [batch, seq, hidden]
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
# PHASE 3 (cont): LOSS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_per_position_losses(
    model,
    device: str,
    sequences: list[list[int]],
    positions: list[TokenPosition],
    heads_to_zero: list[HeadID],
) -> np.ndarray:
    """
    Compute cross-entropy loss at each specified position under the given
    ablation condition.

    Cross-entropy loss at position t: -log_softmax(logits[t-1])[tokens[t]]
    (the model predicts token t from the prefix up to t-1).

    Returns an array of per-position losses, aligned with `positions`.
    """
    # Group positions by sequence index for efficient batching
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
                logits = outputs.logits[0]  # [seq_len, vocab_size]

            # Compute log softmax once for all positions in this sequence
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            for result_idx, pos_t in pos_by_seq[seq_idx]:
                # Loss at position t: -log_softmax(logits[t-1])[tokens[t]]
                target_token = tokens[pos_t]
                ce_loss = -log_probs[pos_t - 1, target_token].item()
                losses[result_idx] = ce_loss

            del input_ids, outputs, logits, log_probs
            torch.cuda.empty_cache()

    return losses


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for two independent samples."""
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


def bootstrap_diff_means(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int = BOOTSTRAP_N,
    ci: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap CI for difference in means between independent samples."""
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    point = float(np.mean(x) - np.mean(y))
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        x_idx = rng.randint(0, nx, size=nx)
        y_idx = rng.randint(0, ny, size=ny)
        boot[i] = float(np.mean(x[x_idx]) - np.mean(y[y_idx]))
    alpha = (1 - ci) / 2.0
    return (
        point,
        float(np.percentile(boot, 100 * alpha)),
        float(np.percentile(boot, 100 * (1 - alpha))),
    )


def permutation_mean_diff_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int = 10000,
    seed: int = BOOTSTRAP_SEED,
) -> float:
    """Two-sided permutation test for independent mean difference."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    combined = np.concatenate([x, y])
    obs = abs(float(np.mean(x) - np.mean(y)))
    rng = np.random.RandomState(seed)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        diff = abs(float(np.mean(perm[:nx]) - np.mean(perm[nx:nx + ny])))
        ge += int(diff >= obs - 1e-15)
    return float((ge + 1) / (n_perm + 1))


def analyze_results(
    condition_losses: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """
    Compute all statistical tests for the subword ablation interaction.

    condition_losses: {condition_name: {"continuation": losses_array, "word_initial": losses_array}}
    """
    analysis: dict[str, Any] = {}

    # ── Baseline statistics ──────────────────────────────────────────────────
    baseline_cont = condition_losses["none"]["continuation"]
    baseline_init = condition_losses["none"]["word_initial"]
    analysis["baseline"] = {
        "continuation_mean_loss": float(np.mean(baseline_cont)),
        "continuation_std_loss": float(np.std(baseline_cont, ddof=1)),
        "continuation_median_loss": float(np.median(baseline_cont)),
        "word_initial_mean_loss": float(np.mean(baseline_init)),
        "word_initial_std_loss": float(np.std(baseline_init, ddof=1)),
        "word_initial_median_loss": float(np.median(baseline_init)),
        "n_continuation": len(baseline_cont),
        "n_word_initial": len(baseline_init),
    }

    # ── Per-condition loss increases ─────────────────────────────────────────
    condition_results: dict[str, dict[str, Any]] = {}
    interaction_ttest_pvals: dict[str, float] = {}
    for cond_idx, (cond_name, cond_data) in enumerate(condition_losses.items()):
        if cond_name == "none":
            continue

        cont_losses = cond_data["continuation"]
        init_losses = cond_data["word_initial"]

        # Loss increase = ablated loss - baseline loss (per position)
        cont_increase = cont_losses - baseline_cont
        init_increase = init_losses - baseline_init

        # Independent-sample tests: continuation and word-initial sets are not paired.
        t_stat, t_pval = scipy_stats.ttest_ind(
            cont_increase, init_increase, equal_var=False, nan_policy="omit",
        )
        u_stat, u_pval = scipy_stats.mannwhitneyu(
            cont_increase, init_increase, alternative="two-sided",
        )
        perm_pval = permutation_mean_diff_pvalue(
            cont_increase, init_increase, n_perm=10000, seed=BOOTSTRAP_SEED + 101 * cond_idx,
        )

        # Effect size for independent samples
        d_independent = cohens_d(cont_increase, init_increase)

        # Bootstrap CI for interaction (difference in independent means)
        interaction_point, interaction_ci_low, interaction_ci_high = bootstrap_diff_means(
            cont_increase, init_increase,
            seed=BOOTSTRAP_SEED + 101 * cond_idx,
        )

        # Bootstrap CI for continuation loss increase
        cont_inc_point, cont_inc_ci_low, cont_inc_ci_high = bootstrap_ci(cont_increase)

        # Bootstrap CI for word-initial loss increase
        init_inc_point, init_inc_ci_low, init_inc_ci_high = bootstrap_ci(init_increase)

        condition_results[cond_name] = {
            "continuation_loss_increase": {
                "mean": float(np.mean(cont_increase)),
                "std": float(np.std(cont_increase, ddof=1)),
                "median": float(np.median(cont_increase)),
                "ci_95": [cont_inc_ci_low, cont_inc_ci_high],
            },
            "word_initial_loss_increase": {
                "mean": float(np.mean(init_increase)),
                "std": float(np.std(init_increase, ddof=1)),
                "median": float(np.median(init_increase)),
                "ci_95": [init_inc_ci_low, init_inc_ci_high],
            },
            "interaction": {
                "mean_diff_independent": interaction_point,
                "ci_95": [interaction_ci_low, interaction_ci_high],
                "unit_assumption": "independent_samples",
            },
            "independent_ttest": {
                "t_statistic": float(t_stat),
                "p_value": float(t_pval),
                "equal_var_assumed": False,
            },
            "mannwhitney_test": {
                "u_statistic": float(u_stat),
                "p_value": float(u_pval),
            },
            "permutation_test": {
                "n_permutations": 10000,
                "p_value": float(perm_pval),
            },
            "effect_sizes": {
                "cohens_d_independent": d_independent,
            },
        }
        interaction_ttest_pvals[cond_name] = float(t_pval)

    if interaction_ttest_pvals:
        holm = holm_adjust(interaction_ttest_pvals)
        for cond_name, p_adj in holm.items():
            if cond_name in condition_results:
                condition_results[cond_name]["independent_ttest"]["p_value_holm"] = float(p_adj)

    analysis["conditions"] = condition_results

    # ── Two-way interaction: condition x token_type ──────────────────────────
    # Compare interaction effects across conditions
    if "ablate_high_si" in condition_results and "ablate_low_si" in condition_results:
        high_interaction = condition_results["ablate_high_si"]["interaction"]["mean_diff_independent"]
        low_interaction = condition_results["ablate_low_si"]["interaction"]["mean_diff_independent"]

        # Aggregate random interactions
        random_interactions = []
        for k, v in condition_results.items():
            if k.startswith("ablate_random"):
                random_interactions.append(v["interaction"]["mean_diff_independent"])
        random_interaction_mean = float(np.mean(random_interactions)) if random_interactions else float("nan")

        analysis["two_way_interaction"] = {
            "high_si_interaction": high_interaction,
            "low_si_interaction": low_interaction,
            "random_interaction_mean": random_interaction_mean,
            "high_minus_low_interaction": high_interaction - low_interaction,
            "high_minus_random_interaction": high_interaction - random_interaction_mean,
            "hypothesis_supported": bool(high_interaction > low_interaction and high_interaction > 0),
        }

    return analysis


def perplexity_matched_subsample(
    baseline_cont_losses: np.ndarray,
    baseline_init_losses: np.ndarray,
    cont_positions: list[TokenPosition],
    init_positions: list[TokenPosition],
    n_bins: int = 20,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Subsample positions to match baseline perplexity distributions between
    continuation and word-initial tokens.

    Uses histogram matching: bin both distributions, then subsample the larger
    group in each bin to match the smaller group.

    Returns indices into the original position lists.
    """
    rng = np.random.RandomState(seed)

    # Determine shared bin edges
    all_losses = np.concatenate([baseline_cont_losses, baseline_init_losses])
    lo, hi = np.percentile(all_losses, 1), np.percentile(all_losses, 99)
    bin_edges = np.linspace(lo, hi, n_bins + 1)

    cont_bins = np.digitize(baseline_cont_losses, bin_edges)
    init_bins = np.digitize(baseline_init_losses, bin_edges)

    selected_cont_idx: list[int] = []
    selected_init_idx: list[int] = []

    for b in range(1, n_bins + 2):
        c_idx = np.where(cont_bins == b)[0]
        i_idx = np.where(init_bins == b)[0]
        n_match = min(len(c_idx), len(i_idx))
        if n_match == 0:
            continue
        selected_cont_idx.extend(rng.choice(c_idx, size=n_match, replace=False).tolist())
        selected_init_idx.extend(rng.choice(i_idx, size=n_match, replace=False).tolist())

    return selected_cont_idx, selected_init_idx


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3.5: Subword ablation interaction (causal test of subword-assembly hypothesis)"
    )
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/experiment3/theory5_subword_ablation")
    parser.add_argument("--target-examples", type=int, default=TARGET_EXAMPLES_PER_TYPE)
    parser.add_argument("--max-sequences", type=int, default=MAX_SEQUENCES)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_spec = MODELS[args.model]

    print(f"{'='*72}")
    print(f"Experiment 3.5: Subword ablation interaction — {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"{'='*72}")

    # ── Load model ───────────────────────────────────────────────────────────
    if get_adapter is None or load_model is None or load_tokenizer is None:
        raise RuntimeError(
            "Model execution dependencies are unavailable in this environment."
        )
    print("\n[1/7] Loading model and tokenizer...")
    loader = load_model(model_spec)
    model = loader.model.to(args.device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)
    print(f"  Model loaded: {model_spec.hf_id}")
    print(f"  Config: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_attention_heads} query heads, "
          f"head_dim={model.config.hidden_size // model.config.num_attention_heads}")

    # ── Phase 1: Construct evaluation sets ───────────────────────────────────
    print(f"\n[2/7] Phase 1: Loading wiki sequences and building evaluation sets...")
    sequences = load_wiki_sequences(args.model, args.max_sequences, EVAL_SEQ_LEN)
    print(f"  Loaded {len(sequences)} wiki sequences (len={EVAL_SEQ_LEN})")

    continuation_positions, word_initial_positions = build_evaluation_sets(
        tokenizer, sequences, args.target_examples,
    )

    # Save position metadata
    pos_meta = {
        "n_continuation": len(continuation_positions),
        "n_word_initial": len(word_initial_positions),
        "n_sequences_scanned": len(sequences),
        "seq_len": EVAL_SEQ_LEN,
    }
    (output_dir / "position_metadata.json").write_text(
        json.dumps(pos_meta, indent=2), encoding="utf-8"
    )

    # ── Phase 2: Load head classification ────────────────────────────────────
    print(f"\n[3/7] Phase 2: Loading head classification from Theory 1...")
    head_groups = load_head_groups(args.model)

    high_si = parse_head_list(head_groups["high_si"])
    low_si = parse_head_list(head_groups["low_si"])
    n_select = head_groups["n_select"]
    n_total = head_groups["n_total_heads"]

    print(f"  Head groups loaded: {n_select} heads per group ({n_total} total)")
    print(f"  High-SI: {len(high_si)} heads")
    print(f"  Low-SI:  {len(low_si)} heads")

    # Reconstruct random draws from saved data
    random_draws_heads: list[list[HeadID]] = []
    for draw in range(RANDOM_DRAWS):
        key = f"random_draw{draw}"
        if key in head_groups:
            random_draws_heads.append(parse_head_list(head_groups[key]))
        else:
            # Fallback: generate random draws
            all_heads = []
            for entry in head_groups["high_si"] + head_groups["low_si"]:
                all_heads.append(HeadID(entry["layer"], entry["head"]))
            # Build full head list from model config
            n_layers = model.config.num_hidden_layers
            n_heads_per_layer = model.config.num_attention_heads
            all_heads = [HeadID(l, h) for l in range(n_layers) for h in range(n_heads_per_layer)]
            rng = random.Random(42 + draw * 1000)
            random_draws_heads.append(rng.sample(all_heads, k=n_select))
            print(f"  WARNING: random_draw{draw} not found in head_groups.json, generated fresh")

    for draw, rh in enumerate(random_draws_heads):
        print(f"  Random draw {draw}: {len(rh)} heads")

    # ── Phase 3: Ablation evaluation ─────────────────────────────────────────
    print(f"\n[4/7] Phase 3: Computing per-position losses under ablation conditions...")

    conditions: list[tuple[str, list[HeadID]]] = [
        ("none", []),
        ("ablate_high_si", high_si),
        ("ablate_low_si", low_si),
    ]
    for draw in range(RANDOM_DRAWS):
        conditions.append((f"ablate_random_draw{draw}", random_draws_heads[draw]))

    # Store all losses: {condition: {"continuation": array, "word_initial": array}}
    condition_losses: dict[str, dict[str, np.ndarray]] = {}
    all_loss_rows: list[dict[str, Any]] = []

    for cond_name, heads in conditions:
        t0 = time.time()
        n_ablated = len(heads)
        print(f"\n  --- Condition: {cond_name} ({n_ablated} heads ablated) ---")

        # Compute losses for continuation positions
        print(f"    Computing losses for {len(continuation_positions)} continuation positions...")
        cont_losses = compute_per_position_losses(
            model, args.device, sequences, continuation_positions, heads,
        )

        # Compute losses for word-initial positions
        print(f"    Computing losses for {len(word_initial_positions)} word-initial positions...")
        init_losses = compute_per_position_losses(
            model, args.device, sequences, word_initial_positions, heads,
        )

        condition_losses[cond_name] = {
            "continuation": cont_losses,
            "word_initial": init_losses,
        }

        # Store per-position results for saving
        for i, pos in enumerate(continuation_positions):
            all_loss_rows.append({
                "condition": cond_name,
                "token_type": "continuation",
                "sequence_idx": pos.sequence_idx,
                "position": pos.position,
                "token_id": pos.token_id,
                "loss": float(cont_losses[i]),
            })
        for i, pos in enumerate(word_initial_positions):
            all_loss_rows.append({
                "condition": cond_name,
                "token_type": "word_initial",
                "sequence_idx": pos.sequence_idx,
                "position": pos.position,
                "token_id": pos.token_id,
                "loss": float(init_losses[i]),
            })

        elapsed = time.time() - t0
        print(f"    Condition {cond_name}: continuation mean={np.mean(cont_losses):.4f}, "
              f"word-initial mean={np.mean(init_losses):.4f} ({elapsed:.0f}s)")

    # Save raw losses
    loss_df = pd.DataFrame(all_loss_rows)
    loss_df.to_parquet(output_dir / "per_position_losses.parquet", index=False)
    print(f"\n  Saved {len(loss_df)} loss measurements to per_position_losses.parquet")

    # ── Phase 4: Analysis (unmatched) ────────────────────────────────────────
    print(f"\n[5/7] Phase 4a: Statistical analysis (unmatched)...")
    analysis_unmatched = analyze_results(condition_losses)
    analysis_unmatched["matching"] = "unmatched"

    # Print unmatched results
    print_analysis_summary(analysis_unmatched, "UNMATCHED")

    # ── Phase 4b: Perplexity-matched analysis ────────────────────────────────
    print(f"\n[6/7] Phase 4b: Perplexity-matched analysis...")
    baseline_cont = condition_losses["none"]["continuation"]
    baseline_init = condition_losses["none"]["word_initial"]

    matched_cont_idx, matched_init_idx = perplexity_matched_subsample(
        baseline_cont, baseline_init,
        continuation_positions, word_initial_positions,
    )
    print(f"  Matched subsamples: {len(matched_cont_idx)} continuation, "
          f"{len(matched_init_idx)} word-initial")

    if len(matched_cont_idx) >= 50 and len(matched_init_idx) >= 50:
        # Build matched condition losses
        matched_condition_losses: dict[str, dict[str, np.ndarray]] = {}
        # Ensure both have same length (take minimum)
        n_matched = min(len(matched_cont_idx), len(matched_init_idx))
        matched_cont_idx = matched_cont_idx[:n_matched]
        matched_init_idx = matched_init_idx[:n_matched]

        for cond_name, cond_data in condition_losses.items():
            matched_condition_losses[cond_name] = {
                "continuation": cond_data["continuation"][matched_cont_idx],
                "word_initial": cond_data["word_initial"][matched_init_idx],
            }

        analysis_matched = analyze_results(matched_condition_losses)
        analysis_matched["matching"] = "perplexity_matched"
        analysis_matched["n_matched_per_type"] = n_matched
        analysis_matched["matched_baseline_cont_mean"] = float(
            np.mean(baseline_cont[matched_cont_idx])
        )
        analysis_matched["matched_baseline_init_mean"] = float(
            np.mean(baseline_init[matched_init_idx])
        )

        print_analysis_summary(analysis_matched, "PERPLEXITY-MATCHED")
    else:
        print(f"  WARNING: Too few matched samples ({len(matched_cont_idx)}, "
              f"{len(matched_init_idx)}), skipping matched analysis")
        analysis_matched = {"matching": "perplexity_matched", "skipped": True,
                           "reason": "too_few_matched_samples"}

    # ── Save results ─────────────────────────────────────────────────────────
    print(f"\n[7/7] Writing final report...")
    report = {
        "experiment": "3.5_subword_ablation_interaction",
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "eval_seq_len": EVAL_SEQ_LEN,
            "target_examples_per_type": args.target_examples,
            "max_sequences": args.max_sequences,
            "random_draws": RANDOM_DRAWS,
            "bootstrap_n": BOOTSTRAP_N,
        },
        "position_metadata": pos_meta,
        "head_groups": {
            "n_total": n_total,
            "n_selected_per_group": n_select,
            "source": f"results/experiment3/theory1_si_circuits/{args.model}/head_groups.json",
        },
        "analysis_unmatched": analysis_unmatched,
        "analysis_matched": analysis_matched,
    }

    (output_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"  Report saved to {output_dir / 'analysis.json'}")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"EXPERIMENT 3.5 COMPLETE — {args.model}")
    print(f"{'='*72}")

    if "two_way_interaction" in analysis_unmatched:
        twi = analysis_unmatched["two_way_interaction"]
        print(f"\n  PRIMARY RESULT (unmatched):")
        print(f"    High-SI interaction (cont - init damage):    {twi['high_si_interaction']:+.6f}")
        print(f"    Low-SI  interaction (cont - init damage):    {twi['low_si_interaction']:+.6f}")
        print(f"    Random  interaction (cont - init damage):    {twi['random_interaction_mean']:+.6f}")
        print(f"    High-SI minus Low-SI interaction gap:        {twi['high_minus_low_interaction']:+.6f}")
        print(f"    High-SI minus Random interaction gap:        {twi['high_minus_random_interaction']:+.6f}")
        print(f"    Hypothesis supported:                        {twi['hypothesis_supported']}")

    if isinstance(analysis_matched, dict) and "two_way_interaction" in analysis_matched:
        twi = analysis_matched["two_way_interaction"]
        print(f"\n  PRIMARY RESULT (perplexity-matched):")
        print(f"    High-SI interaction (cont - init damage):    {twi['high_si_interaction']:+.6f}")
        print(f"    Low-SI  interaction (cont - init damage):    {twi['low_si_interaction']:+.6f}")
        print(f"    Random  interaction (cont - init damage):    {twi['random_interaction_mean']:+.6f}")
        print(f"    Hypothesis supported:                        {twi['hypothesis_supported']}")

    print(f"\n  All artifacts in: {output_dir}")
    print(f"  Done.")


def print_analysis_summary(analysis: dict[str, Any], label: str) -> None:
    """Print a formatted summary of the analysis results."""
    print(f"\n  ── {label} RESULTS ────────────────────────────────────────────")

    if "baseline" in analysis:
        bl = analysis["baseline"]
        print(f"  Baseline continuation mean loss:  {bl['continuation_mean_loss']:.4f} "
              f"(std={bl['continuation_std_loss']:.4f}, n={bl['n_continuation']})")
        print(f"  Baseline word-initial mean loss:  {bl['word_initial_mean_loss']:.4f} "
              f"(std={bl['word_initial_std_loss']:.4f}, n={bl['n_word_initial']})")

    if "conditions" in analysis:
        for cond_name, cond in analysis["conditions"].items():
            cont_inc = cond["continuation_loss_increase"]
            init_inc = cond["word_initial_loss_increase"]
            inter = cond["interaction"]
            tests = cond["independent_ttest"]
            mw = cond["mannwhitney_test"]
            perm = cond["permutation_test"]
            eff = cond["effect_sizes"]

            print(f"\n  {cond_name}:")
            print(f"    Continuation loss increase:  {cont_inc['mean']:+.4f} "
                  f"(95% CI [{cont_inc['ci_95'][0]:+.4f}, {cont_inc['ci_95'][1]:+.4f}])")
            print(f"    Word-initial loss increase:  {init_inc['mean']:+.4f} "
                  f"(95% CI [{init_inc['ci_95'][0]:+.4f}, {init_inc['ci_95'][1]:+.4f}])")
            print(f"    Interaction (cont - init):   {inter['mean_diff_independent']:+.4f} "
                  f"(95% CI [{inter['ci_95'][0]:+.4f}, {inter['ci_95'][1]:+.4f}])")
            holm_str = (
                f", p_holm={tests['p_value_holm']:.2e}"
                if "p_value_holm" in tests
                else ""
            )
            print(f"    Welch t-test:   t={tests['t_statistic']:.3f}, p={tests['p_value']:.2e}{holm_str}")
            print(f"    Mann-Whitney:   U={mw['u_statistic']:.1f}, p={mw['p_value']:.2e}")
            print(f"    Permutation:    p={perm['p_value']:.2e} ({perm['n_permutations']} perms)")
            print(f"    Cohen's d (independent): {eff['cohens_d_independent']:.3f}")

    if "two_way_interaction" in analysis:
        twi = analysis["two_way_interaction"]
        print(f"\n  Two-way interaction (condition x token_type):")
        print(f"    High-SI interaction:           {twi['high_si_interaction']:+.6f}")
        print(f"    Low-SI  interaction:           {twi['low_si_interaction']:+.6f}")
        print(f"    Random  interaction (mean):    {twi['random_interaction_mean']:+.6f}")
        print(f"    High - Low gap:                {twi['high_minus_low_interaction']:+.6f}")
        print(f"    High - Random gap:             {twi['high_minus_random_interaction']:+.6f}")
        print(f"    Hypothesis supported:          {twi['hypothesis_supported']}")


if __name__ == "__main__":
    main()
