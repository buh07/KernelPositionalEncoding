#!/usr/bin/env python3
"""
Experiment 3.1: Shift-invariance as enabler of position-invariant circuits
==========================================================================

Hypothesis
----------
Heads with high shift-invariance R² are disproportionately responsible for
position-relative operations (copy, retrieval, induction-like pattern matching),
regardless of attention range.  The functional role of shift-invariance is not
"which frequency controls which range" but "which heads implement computations
that must work identically at any absolute position."

Protocol
--------
Phase 1 — R² profiling:   compute per-(layer, head) shift-invariance R²
Phase 2 — Classification: top-25% = high-SI, bottom-25% = low-SI
Phase 3 — Head ablation:  zero attention outputs of selected head groups
Phase 4 — Task battery:   run retrieval + mirror tasks under each ablation
Phase 5 — Analysis:       compare damage across conditions

Usage
-----
    # GPU 0 (Llama)
    python scripts/experiment3_theory1_si_circuits.py --model llama-3.1-8b --device cuda:0

    # GPU 1 (OLMo)
    python scripts/experiment3_theory1_si_circuits.py --model olmo-2-7b --device cuda:1
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

from experiment1.shift_kernels import RoPEEstimator, KernelFit
from experiment1.norm_utils import normalize_logits_for_norm
from experiment2.tasks import (
    TaskExample,
    TokenPools,
    build_token_pools,
    generate_task_examples,
)
from shared.specs import ModelSpec

try:
    from experiment2.execution import (
        _evaluate_example_from_token_logits,
        _build_restricted_candidates,
    )
except Exception:  # pragma: no cover - optional for analysis-only environments
    _evaluate_example_from_token_logits = None
    _build_restricted_candidates = None

try:
    from shared.attention.adapters import get_adapter
except Exception:  # pragma: no cover - optional for analysis-only environments
    get_adapter = None

# ── constants ────────────────────────────────────────────────────────────────
NUM_PROFILE_SEQUENCES = 50
PROFILE_SEQ_LEN = 512
TASK_SEQ_LEN = 512
SYNTHETIC_COUNT = 100
CANDIDATE_SIZE = 10
NUM_SEEDS = 7
QUARTILE = 0.25  # top/bottom 25% for head classification
RANDOM_DRAWS = 3

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

RETRIEVAL_SPANS: dict[str, tuple[int, ...]] = {
    "llama-3.1-8b": (32, 48, 64),
    "olmo-2-7b": (24, 32),
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


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: R² PROFILING
# ═══════════════════════════════════════════════════════════════════════════════

def load_profile_sequences(tokenizer, model_name: str, num_sequences: int, seq_len: int) -> list[list[int]]:
    """Load tokenized wiki sequences for R² profiling."""
    data_dir = Path("data/experiment1/wiki40b_en_pre2019") / model_name
    # Find any available tokenized JSONL
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


def compute_per_head_r2(
    model,
    adapter,
    tokenizer,
    model_spec: ModelSpec,
    device: str,
    sequences: list[list[int]],
) -> pd.DataFrame:
    """Compute per-(layer, head) shift-invariance R² on profiling sequences."""
    estimator = RoPEEstimator()
    rows: list[dict[str, Any]] = []
    n_seq = len(sequences)

    for seq_idx, tokens in enumerate(sequences):
        t0 = time.time()
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

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
            continue

        # capture.logits shape: [layers, heads, seq, seq]
        n_layers, n_heads = capture.logits.shape[0], capture.logits.shape[1]
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                head_logits = capture.logits[layer_idx, head_idx]  # [seq, seq]
                prepared = normalize_logits_for_norm(head_logits, model_spec.norm)
                fit: KernelFit = estimator.fit_logits(prepared)
                rows.append({
                    "sequence_id": seq_idx,
                    "layer": layer_idx,
                    "head": head_idx,
                    "r2": fit.r2,
                })

        elapsed = time.time() - t0
        if (seq_idx + 1) % 5 == 0 or seq_idx == 0:
            print(f"  R² profiling: {seq_idx + 1}/{n_seq} sequences ({elapsed:.1f}s)")

        # Free memory
        del capture, input_ids
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: HEAD CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_heads(
    r2_df: pd.DataFrame,
    quantile: float = QUARTILE,
) -> tuple[list[HeadID], list[HeadID], pd.DataFrame]:
    """Classify heads into high-SI (top quantile) and low-SI (bottom quantile)."""
    mean_r2 = r2_df.groupby(["layer", "head"])["r2"].mean().reset_index()
    mean_r2.columns = ["layer", "head", "mean_r2"]
    mean_r2 = mean_r2.sort_values("mean_r2", ascending=False).reset_index(drop=True)

    n_heads = len(mean_r2)
    n_select = max(1, int(n_heads * quantile))

    high_si = [HeadID(int(r.layer), int(r.head)) for r in mean_r2.head(n_select).itertuples()]
    low_si = [HeadID(int(r.layer), int(r.head)) for r in mean_r2.tail(n_select).itertuples()]

    print(f"\n  Head classification: {n_heads} total heads, selecting top/bottom {n_select}")
    print(f"  High-SI R² range: [{mean_r2.iloc[n_select-1]['mean_r2']:.4f}, {mean_r2.iloc[0]['mean_r2']:.4f}]")
    print(f"  Low-SI  R² range: [{mean_r2.iloc[-1]['mean_r2']:.4f}, {mean_r2.iloc[-n_select]['mean_r2']:.4f}]")

    # Layer distribution
    high_layers = [h.layer for h in high_si]
    low_layers = [h.layer for h in low_si]
    print(f"  High-SI layer distribution: mean={np.mean(high_layers):.1f}, "
          f"range=[{min(high_layers)}, {max(high_layers)}]")
    print(f"  Low-SI  layer distribution: mean={np.mean(low_layers):.1f}, "
          f"range=[{min(low_layers)}, {max(low_layers)}]")

    return high_si, low_si, mean_r2


def sample_random_heads(
    all_heads: list[HeadID],
    n_select: int,
    draw: int,
    seed_base: int = 42,
) -> list[HeadID]:
    """Sample n_select random heads for control condition."""
    rng = random.Random(seed_base + draw * 1000)
    return rng.sample(all_heads, k=n_select)


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
# PHASE 4: TASK EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_task_battery(
    model,
    tokenizer,
    model_spec: ModelSpec,
    device: str,
    heads_to_zero: list[HeadID],
    condition_name: str,
    seeds: range,
    retrieval_spans: tuple[int, ...],
    pools: TokenPools,
) -> list[dict[str, Any]]:
    """Run the full task battery under a head-ablation condition."""
    if _evaluate_example_from_token_logits is None:
        raise RuntimeError(
            "Task execution dependencies are unavailable in this environment."
        )
    results: list[dict[str, Any]] = []

    # Define task configurations
    tasks: list[tuple[str, int | None, tuple[int, ...] | None]] = []
    # Retrieval at each span
    for span in retrieval_spans:
        tasks.append(("long_range_retrieval", span, (span,)))
    # Mirror short task
    tasks.append(("local_key_match", None, None))

    with head_output_ablation(model, heads_to_zero):
        for seed in seeds:
            for task_name, span_override, span_choices in tasks:
                examples = generate_task_examples(
                    task_name=task_name,
                    model_name=model_spec.name,
                    seq_len=TASK_SEQ_LEN,
                    seed=seed,
                    count=SYNTHETIC_COUNT,
                    pools=pools,
                    span_override=span_override,
                    span_choices=span_choices,
                )

                correct_total = 0
                count_total = 0

                # Batch evaluation
                batch_size = 8
                for batch_start in range(0, len(examples), batch_size):
                    batch = examples[batch_start:batch_start + batch_size]
                    for ex in batch:
                        input_ids = torch.tensor(
                            [ex.tokens], dtype=torch.long, device=device
                        )
                        with torch.no_grad():
                            outputs = model(input_ids=input_ids, use_cache=False)
                            token_logits = outputs.logits[0]  # [seq, vocab]

                        metrics, _ = _evaluate_example_from_token_logits(
                            token_logits,
                            ex,
                            split="synthetic",
                            pools=pools,
                            synthetic_eval_mode="restricted",
                            candidate_size=CANDIDATE_SIZE,
                            candidate_policy_version="restricted_candidates_v1_structured_first",
                        )
                        correct_total += int(round(metrics["accuracy"] * metrics["num_targets"]))
                        count_total += int(metrics["num_targets"])

                        del input_ids, outputs, token_logits

                accuracy = correct_total / max(count_total, 1)
                span_val = span_override if span_override is not None else 0
                results.append({
                    "condition": condition_name,
                    "task": task_name,
                    "span": span_val,
                    "seed": seed,
                    "accuracy": accuracy,
                    "n_examples": len(examples),
                    "n_targets": count_total,
                    "n_correct": correct_total,
                })

                print(f"    {condition_name} | {task_name} span={span_val} seed={seed} | "
                      f"acc={accuracy:.4f} ({correct_total}/{count_total})")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_results(results_df: pd.DataFrame, r2_summary: pd.DataFrame, model_name: str) -> dict[str, Any]:
    """Compute the primary analysis: does high-SI ablation cause more damage?"""

    baseline = results_df[results_df["condition"] == "none"].copy()
    baseline_mean = baseline.groupby(["task", "span"])["accuracy"].mean().to_dict()

    analysis: dict[str, Any] = {"model": model_name, "comparisons": []}

    conditions = [c for c in results_df["condition"].unique() if c != "none"]

    for condition in conditions:
        sub = results_df[results_df["condition"] == condition].copy()
        rows = []
        for (task, span), group in sub.groupby(["task", "span"]):
            bl = baseline_mean.get((task, span), None)
            if bl is None:
                continue
            mean_acc = group["accuracy"].mean()
            std_acc = group["accuracy"].std()
            drop = bl - mean_acc
            headroom = bl - 0.10  # chance = 10% for 10-way
            drop_over_headroom = drop / max(headroom, 0.01)
            rows.append({
                "task": task,
                "span": span,
                "condition": condition,
                "baseline_accuracy": bl,
                "ablation_accuracy": mean_acc,
                "accuracy_std": std_acc,
                "drop_raw": drop,
                "headroom": headroom,
                "drop_over_headroom": drop_over_headroom,
                "n_seeds": len(group),
            })
        analysis["comparisons"].extend(rows)

    # Compute aggregate high-SI vs low-SI contrast
    comp_df = pd.DataFrame(analysis["comparisons"])
    if not comp_df.empty:
        high_si = comp_df[comp_df["condition"] == "ablate_high_si"]
        low_si = comp_df[comp_df["condition"] == "ablate_low_si"]

        if not high_si.empty and not low_si.empty:
            h_mean_drop = high_si["drop_raw"].mean()
            l_mean_drop = low_si["drop_raw"].mean()
            h_mean_hdr = high_si["drop_over_headroom"].mean()
            l_mean_hdr = low_si["drop_over_headroom"].mean()

            analysis["aggregate"] = {
                "high_si_mean_drop_raw": float(h_mean_drop),
                "low_si_mean_drop_raw": float(l_mean_drop),
                "high_minus_low_drop_raw": float(h_mean_drop - l_mean_drop),
                "high_si_mean_drop_headroom": float(h_mean_hdr),
                "low_si_mean_drop_headroom": float(l_mean_hdr),
                "high_minus_low_drop_headroom": float(h_mean_hdr - l_mean_hdr),
                "high_si_causes_more_damage": bool(h_mean_drop > l_mean_drop),
            }

            # Matched-cell paired inference (task/span/seed aligned).
            pivot = results_df.pivot_table(
                index=["task", "span", "seed"],
                columns="condition",
                values="accuracy",
                aggfunc="mean",
            )
            needed_cols = {"none", "ablate_high_si", "ablate_low_si"}
            if needed_cols.issubset(set(pivot.columns)):
                paired = pivot.dropna(subset=list(needed_cols))
                if not paired.empty:
                    high_drop = (paired["none"] - paired["ablate_high_si"]).values
                    low_drop = (paired["none"] - paired["ablate_low_si"]).values
                    delta = high_drop - low_drop
                    t_stat, t_p = scipy_stats.ttest_rel(high_drop, low_drop, nan_policy="omit")
                    try:
                        w_stat, w_p = scipy_stats.wilcoxon(delta)
                    except ValueError:
                        w_stat, w_p = float("nan"), float("nan")

                    rng = np.random.RandomState(42)
                    n = len(delta)
                    boot = np.empty(5000, dtype=np.float64)
                    for i in range(len(boot)):
                        idx = rng.randint(0, n, size=n)
                        boot[i] = float(np.mean(delta[idx]))

                    analysis["aggregate"]["paired_high_vs_low"] = {
                        "n_cells": int(len(delta)),
                        "mean_delta_drop_raw": float(np.mean(delta)),
                        "ci_95": [
                            float(np.percentile(boot, 2.5)),
                            float(np.percentile(boot, 97.5)),
                        ],
                        "paired_ttest": {
                            "t_statistic": float(t_stat),
                            "p_value": float(t_p),
                        },
                        "wilcoxon": {
                            "w_statistic": float(w_stat),
                            "p_value": float(w_p),
                        },
                        "supports_high_si_more_damage": bool(
                            np.mean(delta) > 0 and np.isfinite(t_p) and t_p < 0.05
                        ),
                    }

            # Random control comparison
            random_conds = comp_df[comp_df["condition"].str.startswith("ablate_random")]
            if not random_conds.empty:
                r_mean_drop = random_conds["drop_raw"].mean()
                r_mean_hdr = random_conds["drop_over_headroom"].mean()
                analysis["aggregate"]["random_mean_drop_raw"] = float(r_mean_drop)
                analysis["aggregate"]["random_mean_drop_headroom"] = float(r_mean_hdr)
                analysis["aggregate"]["high_si_exceeds_random"] = bool(h_mean_drop > r_mean_drop)

    # R² distribution summary
    analysis["r2_summary"] = {
        "mean_r2": float(r2_summary["mean_r2"].mean()),
        "std_r2": float(r2_summary["mean_r2"].std()),
        "min_r2": float(r2_summary["mean_r2"].min()),
        "max_r2": float(r2_summary["mean_r2"].max()),
        "median_r2": float(r2_summary["mean_r2"].median()),
    }

    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3.1: Shift-invariance as enabler of position-invariant circuits"
    )
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/experiment3/theory1_si_circuits")
    parser.add_argument("--profile-sequences", type=int, default=NUM_PROFILE_SEQUENCES)
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--synthetic-count", type=int, default=SYNTHETIC_COUNT)
    parser.add_argument("--skip-profiling", action="store_true",
                        help="Skip Phases 1-2; load saved R² data from output dir")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_spec = MODELS[args.model]
    retrieval_spans = RETRIEVAL_SPANS[args.model]

    print(f"{'='*72}")
    print(f"Experiment 3.1: Shift-invariance circuits — {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"{'='*72}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\n[1/6] Loading model and tokenizer...")
    from shared.models.loading import load_model, load_tokenizer

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

    # ── Phase 1: R² profiling ────────────────────────────────────────────────
    r2_parquet = output_dir / "per_sequence_r2.parquet"
    if args.skip_profiling and r2_parquet.exists():
        print(f"\n[2/6] Phase 1: SKIPPED (loading saved R² from {r2_parquet})")
        r2_df = pd.read_parquet(r2_parquet)
        print(f"  Loaded {len(r2_df)} per-(seq, layer, head) R² measurements")
    else:
        print(f"\n[2/6] Phase 1: R² profiling ({args.profile_sequences} sequences)...")
        sequences = load_profile_sequences(tokenizer, args.model, args.profile_sequences, PROFILE_SEQ_LEN)
        r2_df = compute_per_head_r2(model, adapter, tokenizer, model_spec, args.device, sequences)
        r2_df.to_parquet(r2_parquet, index=False)
        print(f"  Saved {len(r2_df)} per-(seq, layer, head) R² measurements")

    # ── Phase 2: Head classification ─────────────────────────────────────────
    print(f"\n[3/6] Phase 2: Head classification (top/bottom {int(QUARTILE*100)}%)...")
    high_si, low_si, r2_summary = classify_heads(r2_df)
    r2_summary.to_parquet(output_dir / "head_r2_summary.parquet", index=False)

    # Build all-heads list for random sampling
    all_heads = [HeadID(int(r.layer), int(r.head)) for r in r2_summary.itertuples()]
    n_select = len(high_si)

    # Save head groups
    head_groups = {
        "high_si": [{"layer": h.layer, "head": h.head} for h in high_si],
        "low_si": [{"layer": h.layer, "head": h.head} for h in low_si],
        "n_select": n_select,
        "n_total_heads": len(all_heads),
        "quartile": QUARTILE,
    }
    for draw in range(RANDOM_DRAWS):
        rand_heads = sample_random_heads(all_heads, n_select, draw)
        head_groups[f"random_draw{draw}"] = [{"layer": h.layer, "head": h.head} for h in rand_heads]

    (output_dir / "head_groups.json").write_text(
        json.dumps(head_groups, indent=2), encoding="utf-8"
    )
    print(f"  Head groups saved to {output_dir / 'head_groups.json'}")

    # ── Phase 3-4: Task evaluation under ablation ────────────────────────────
    print(f"\n[4/6] Phase 3-4: Task evaluation under head ablation...")

    # Build token pools
    vocab_size = int(tokenizer.vocab_size)
    special_ids = [
        getattr(tokenizer, attr, None)
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    ]
    special_ids = [sid for sid in special_ids if sid is not None]
    pools = build_token_pools(model_spec.name, vocab_size, special_ids)

    seeds = range(args.num_seeds)
    all_results: list[dict[str, Any]] = []

    # Define conditions
    conditions: list[tuple[str, list[HeadID]]] = [
        ("none", []),
        ("ablate_high_si", high_si),
        ("ablate_low_si", low_si),
    ]
    for draw in range(RANDOM_DRAWS):
        rand_heads = sample_random_heads(all_heads, n_select, draw)
        conditions.append((f"ablate_random_draw{draw}", rand_heads))

    for cond_name, heads in conditions:
        t0 = time.time()
        n_heads_ablated = len(heads)
        print(f"\n  --- Condition: {cond_name} ({n_heads_ablated} heads ablated) ---")
        cond_results = evaluate_task_battery(
            model=model,
            tokenizer=tokenizer,
            model_spec=model_spec,
            device=args.device,
            heads_to_zero=heads,
            condition_name=cond_name,
            seeds=seeds,
            retrieval_spans=retrieval_spans,
            pools=pools,
        )
        all_results.extend(cond_results)
        elapsed = time.time() - t0
        print(f"  Condition {cond_name} complete in {elapsed:.0f}s")

    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(output_dir / "task_results.parquet", index=False)
    print(f"\n  Saved {len(results_df)} result rows to {output_dir / 'task_results.parquet'}")

    # ── Phase 5: Analysis ────────────────────────────────────────────────────
    print(f"\n[5/6] Phase 5: Analysis...")
    analysis = analyze_results(results_df, r2_summary, args.model)
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, default=str), encoding="utf-8"
    )

    # Print summary
    print(f"\n{'='*72}")
    print(f"RESULTS SUMMARY — {args.model}")
    print(f"{'='*72}")

    if "aggregate" in analysis:
        agg = analysis["aggregate"]
        print(f"\n  High-SI ablation mean drop (raw):      {agg['high_si_mean_drop_raw']:+.4f}")
        print(f"  Low-SI  ablation mean drop (raw):      {agg['low_si_mean_drop_raw']:+.4f}")
        print(f"  High - Low gap (raw):                  {agg['high_minus_low_drop_raw']:+.4f}")
        print(f"  High-SI causes more damage:            {agg['high_si_causes_more_damage']}")
        if "paired_high_vs_low" in agg:
            paired = agg["paired_high_vs_low"]
            print(
                f"  Matched paired test (high-low drop):   "
                f"{paired['mean_delta_drop_raw']:+.4f} "
                f"(p={paired['paired_ttest']['p_value']:.2e}, n={paired['n_cells']})"
            )
        if "random_mean_drop_raw" in agg:
            print(f"  Random  ablation mean drop (raw):      {agg['random_mean_drop_raw']:+.4f}")
            print(f"  High-SI exceeds random:                {agg['high_si_exceeds_random']}")

    print(f"\n  Per-task breakdown:")
    for row in analysis["comparisons"]:
        if row["condition"] in ("ablate_high_si", "ablate_low_si"):
            print(f"    {row['condition']:20s} | {row['task']:25s} span={row['span']:3d} | "
                  f"drop={row['drop_raw']:+.4f}  headroom_norm={row['drop_over_headroom']:+.4f}")

    # ── Save final report ────────────────────────────────────────────────────
    print(f"\n[6/6] Writing final report...")
    report = {
        "experiment": "3.1_shift_invariance_circuits",
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "profile_sequences": args.profile_sequences,
            "profile_seq_len": PROFILE_SEQ_LEN,
            "task_seq_len": TASK_SEQ_LEN,
            "synthetic_count": args.synthetic_count,
            "candidate_size": CANDIDATE_SIZE,
            "num_seeds": args.num_seeds,
            "quartile": QUARTILE,
            "random_draws": RANDOM_DRAWS,
            "retrieval_spans": list(retrieval_spans),
        },
        "r2_summary": analysis.get("r2_summary", {}),
        "head_groups": {
            "n_total": len(all_heads),
            "n_selected_per_group": n_select,
            "high_si_layer_mean": float(np.mean([h.layer for h in high_si])),
            "low_si_layer_mean": float(np.mean([h.layer for h in low_si])),
        },
        "analysis": analysis,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"  Report saved to {output_dir / 'report.json'}")
    print(f"\nDone. All artifacts in: {output_dir}")


if __name__ == "__main__":
    main()
