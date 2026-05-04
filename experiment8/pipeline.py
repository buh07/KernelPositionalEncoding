"""Experiment 8: Tokenizer Analysis via SI Structure.

  8A  SI Boundary Alignment Score (SIBAS)
  8B  Tokenizer-Aware Pruning via SI Masks
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiment3.theory1_si_circuits import (
    MODELS as THEORY_MODELS,
    HeadID,
    classify_heads,
    compute_per_head_r2,
    head_output_ablation,
    load_profile_sequences,
    sample_random_heads,
)
from experiment3.theory5b_boundary_detection import (
    compute_word_boundaries,
    compute_word_info,
)
from experiment4.common import (
    clear_cuda,
    cohens_d,
    ensure_dir,
    mean_ci95,
    now_timestamp,
    safe_float,
    write_json,
    write_parquet,
)
from experiment4.math_data import build_math_eval_battery
from experiment4.pipeline import (
    _force_eager_attention,
    _prepare_tokenizer_for_training,
    _score_option_logprob,
    evaluate_math_accuracy,
    evaluate_wiki_perplexity,
)
from shared.attention.adapters import get_adapter
from shared.models.loading import load_model, load_tokenizer

from experiment8.config import (
    ABLATION_EVAL_SEQUENCES,
    ABLATION_FRACTIONS,
    ABLATION_SEQ_LEN,
    SIBAS_NUM_SEQUENCES,
    SIBAS_SEQ_LEN,
)


# ---------------------------------------------------------------------------
# Shared model loading
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_name: str, device: str):
    """Load model + tokenizer with eager attention."""
    model_spec = THEORY_MODELS[model_name]
    # Clear caches to free memory.
    for fn in (load_model, load_tokenizer):
        try:
            fn.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        loaded = load_model(model_spec, attn_implementation="eager")
    except Exception:
        loaded = load_model(model_spec)
    model = loaded.model.to(device)
    _force_eager_attention(model)
    tokenizer = load_tokenizer(model_spec)
    _prepare_tokenizer_for_training(tokenizer)
    return model, tokenizer, model_spec


def _load_r2_profile(model_name: str) -> pd.DataFrame:
    """Load precomputed R² profiles from Experiment 3."""
    from experiment8.config import RESULTS_ROOT as _  # noqa: F401
    project_root = Path(__file__).resolve().parents[1]
    r2_path = project_root / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_r2_summary.parquet"
    if not r2_path.exists():
        raise FileNotFoundError(f"R² profile not found: {r2_path}")
    return pd.read_parquet(r2_path)


def _get_ranked_heads(model_name: str) -> tuple[list[HeadID], pd.DataFrame]:
    """Return (ranked_heads_desc, mean_r2_df) from existing profiles.

    The summary parquet already has columns [layer, head, mean_r2].
    """
    mean_r2 = _load_r2_profile(model_name)
    mean_r2 = mean_r2.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    ranked = [HeadID(int(r.layer), int(r.head)) for r in mean_r2.itertuples()]
    return ranked, mean_r2


# ---------------------------------------------------------------------------
# 8A: Tokenizer–SI Alignment Metrics (redesigned from original SIBAS)
# ---------------------------------------------------------------------------
#
# Three complementary metrics replace the original SIBAS:
#
# 1. Boundary Discrimination Index (BDI)
#    Measures how differently a head attends to the immediately preceding token
#    at word-transition positions vs within-word positions.  At position t where
#    word_ids[t] != word_ids[t-1] (boundary), A(t, t-1) is the "cross-boundary
#    attention."  BDI(h) = mean_boundary A(t,t-1) − mean_continuation A(t,t-1).
#    Aggregated as BDI_SI − BDI_nonSI to control for model-wide patterns.
#    (This is the signal that 3P2-B measured with d=1.085 for Llama.)
#
# 2. Boundary Attention Contrast (BAC)
#    Ratio of SI-head boundary discrimination to non-SI-head discrimination:
#    BAC = mean_BDI(SI heads) / mean_BDI(non-SI heads).
#    BAC > 1 ⇒ SI heads are disproportionately boundary-sensitive.
#    BAC is comparable across models/tokenizers (ratio, not absolute).
#
# 3. Kernel Offset Alignment (KOA)
#    For each SI head, estimate the dominant relative offset Δ* (the distance
#    at which g(Δ) peaks, excluding Δ=0).  Then measure whether consecutive
#    word boundaries in the tokenized text are spaced at multiples of Δ*.
#    High KOA ⇒ the tokenizer creates boundaries that "resonate" with the
#    SI kernel's preferred offset.  This is a purely structural metric: it
#    depends on the kernel shape and tokenizer segmentation, not on any
#    downstream task.


def _get_boundary_and_continuation_positions(
    tokenizer, token_ids: list[int],
) -> tuple[list[int], list[int]]:
    """Split positions into word-boundary and within-word-continuation sets.

    Uses the same definition as 3P2-B:
      boundary:     word_ids[t] != word_ids[t-1]  (word transition)
      continuation: word_ids[t] == word_ids[t-1]  (same word)
    Positions 0 and 1 are excluded (no valid t-1 context).
    """
    word_ids = compute_word_boundaries(tokenizer, token_ids)
    boundary: list[int] = []
    continuation: list[int] = []
    for t in range(2, len(word_ids)):
        if word_ids[t] != word_ids[t - 1]:
            boundary.append(t)
        else:
            continuation.append(t)
    return boundary, continuation


def _compute_per_head_bdi(
    attn_weights: torch.Tensor,   # [layers, heads, seq, seq]
    boundary_pos: list[int],
    continuation_pos: list[int],
) -> dict[tuple[int, int], float]:
    """Compute Boundary Discrimination Index for every head.

    BDI(h) = mean A_h(t, t-1) at boundary positions
           − mean A_h(t, t-1) at continuation positions.
    """
    n_layers, n_heads = attn_weights.shape[:2]
    attn_np = attn_weights.numpy()

    bp = np.array(boundary_pos)
    cp = np.array(continuation_pos)

    result: dict[tuple[int, int], float] = {}
    for l in range(n_layers):
        for h in range(n_heads):
            head_attn = attn_np[l, h]  # [seq, seq]
            if len(bp) > 0:
                mean_b = float(head_attn[bp, bp - 1].mean())
            else:
                mean_b = 0.0
            if len(cp) > 0:
                mean_c = float(head_attn[cp, cp - 1].mean())
            else:
                mean_c = 0.0
            result[(l, h)] = mean_b - mean_c
    return result


def _estimate_dominant_offset(
    attn_weights: torch.Tensor,  # [seq, seq] for one head, softmaxed
) -> int:
    """Estimate the dominant relative offset Δ* for one head's attention.

    For each query position i, find the offset Δ = i − argmax_j A(i,j).
    Δ* = mode of these offsets, excluding Δ=0.
    Returns 1 as fallback if no clear non-zero mode.
    """
    seq_len = attn_weights.shape[0]
    offsets: list[int] = []
    attn_np = attn_weights.numpy()
    for i in range(1, seq_len):
        # Only causal positions: j <= i.
        row = attn_np[i, :i + 1]
        j_star = int(np.argmax(row))
        delta = i - j_star
        if delta > 0:
            offsets.append(delta)
    if not offsets:
        return 1
    # Mode of non-zero offsets.
    from collections import Counter
    counts = Counter(offsets)
    # Most common offset.
    dominant, _ = counts.most_common(1)[0]
    return dominant


def _compute_koa_for_head(
    dominant_offset: int,
    boundary_pos: list[int],
    seq_len: int,
    tolerance: int = 1,
) -> float:
    """Kernel Offset Alignment for one head.

    Measures the fraction of consecutive boundary pairs whose spacing
    is within `tolerance` of a multiple of the head's dominant offset Δ*.

    Returns a value in [0, 1].  Higher = better alignment.
    """
    if dominant_offset < 1 or len(boundary_pos) < 2:
        return 0.0
    bp = sorted(boundary_pos)
    aligned = 0
    total = 0
    for i in range(1, len(bp)):
        gap = bp[i] - bp[i - 1]
        # Check if gap is close to any multiple of Δ*.
        remainder = gap % dominant_offset
        near_multiple = min(remainder, dominant_offset - remainder)
        if near_multiple <= tolerance:
            aligned += 1
        total += 1
    return aligned / total if total > 0 else 0.0


def _extract_attention_weights(
    model, adapter, input_ids: torch.Tensor, device: str,
) -> torch.Tensor:
    """Run one forward pass and return softmaxed causal attention weights.

    Returns: [layers, heads, seq, seq] float32 tensor (on CPU).
    """
    capture = adapter.capture(
        model,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        include_logits=True,
        return_token_logits=False,
        capture_attention=True,
        output_device="cpu",
    )
    logits = capture.logits  # [layers, heads, seq, seq]
    if logits is None:
        raise RuntimeError("Adapter did not capture attention logits")

    # Apply strict causal mask (future keys get -inf before softmax).
    seq_len = logits.shape[-1]
    causal = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=logits.device),
        diagonal=1,
    )
    logits = logits.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Softmax over keys (last dim).
    attn = torch.softmax(logits.float(), dim=-1)
    return attn


def run_8a(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    num_sequences: int = SIBAS_NUM_SEQUENCES,
    seq_len: int = SIBAS_SEQ_LEN,
) -> dict[str, Any]:
    """Run Experiment 8A: tokenizer–SI alignment metrics.

    Computes three metrics per model:
      BDI  — Boundary Discrimination Index (attention-to-previous at transitions)
      BAC  — Boundary Attention Contrast (SI / non-SI ratio of BDI)
      KOA  — Kernel Offset Alignment (boundary spacing vs kernel peak offset)

    Reuses existing R² profiles from Experiment 3.
    """
    from scipy import stats as sp_stats

    model_root = ensure_dir(output_root / model_name)
    print(f"\n{'=' * 60}")
    print(f"[8A] {model_name}: Tokenizer–SI Alignment Metrics")
    print(f"{'=' * 60}")

    # Load model.
    model, tokenizer, model_spec = _load_model_and_tokenizer(model_name, device)
    model.eval()
    adapter = get_adapter(model_spec)
    adapter.register(model)

    # Load R² profile and classify heads (top/bottom quartile).
    ranked_heads, mean_r2_df = _get_ranked_heads(model_name)
    n_total = len(ranked_heads)
    n_quartile = max(1, int(n_total * 0.25))
    high_si = ranked_heads[:n_quartile]
    low_si = ranked_heads[-n_quartile:]
    si_head_set = {(h.layer, h.head) for h in high_si}
    low_si_set = {(h.layer, h.head) for h in low_si}
    print(f"  Heads: {n_total} total, {n_quartile} high-SI, {n_quartile} low-SI")

    # Load evaluation sequences.
    sequences = load_profile_sequences(
        tokenizer=tokenizer, model_name=model_name,
        num_sequences=num_sequences, seq_len=seq_len,
    )
    print(f"  Loaded {len(sequences)} sequences (len={seq_len})")

    # ── Accumulate per-sequence metrics ──
    rows: list[dict[str, Any]] = []

    # Per-head accumulators across sequences (for KOA and per-head BDI).
    per_head_bdi_accum: dict[tuple[int, int], list[float]] = defaultdict(list)
    per_head_koa_accum: dict[tuple[int, int], list[float]] = defaultdict(list)
    per_head_dominant_offsets: dict[tuple[int, int], list[int]] = defaultdict(list)

    for seq_idx, token_ids in enumerate(sequences):
        t0 = time.time()
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        # Extract softmaxed causal attention weights.
        with torch.no_grad():
            attn_weights = _extract_attention_weights(model, adapter, input_ids, device)

        # Get boundary / continuation positions (3P2-B definition).
        boundary_pos, continuation_pos = _get_boundary_and_continuation_positions(
            tokenizer, token_ids,
        )

        # ── Metric 1: BDI per head ──
        head_bdi = _compute_per_head_bdi(attn_weights, boundary_pos, continuation_pos)

        bdi_si_vals = [head_bdi[k] for k in head_bdi if k in si_head_set]
        bdi_lo_vals = [head_bdi[k] for k in head_bdi if k in low_si_set]
        bdi_si_mean = float(np.mean(bdi_si_vals)) if bdi_si_vals else 0.0
        bdi_lo_mean = float(np.mean(bdi_lo_vals)) if bdi_lo_vals else 0.0

        # ── Metric 2: BAC = BDI_SI / BDI_nonSI ──
        bac = bdi_si_mean / bdi_lo_mean if abs(bdi_lo_mean) > 1e-12 else float("nan")

        # ── Metric 3: KOA (dominant offset alignment) ──
        koa_si_vals: list[float] = []
        for l, h in si_head_set:
            if l >= attn_weights.shape[0] or h >= attn_weights.shape[1]:
                continue
            head_attn = attn_weights[l, h]
            dominant = _estimate_dominant_offset(head_attn)
            per_head_dominant_offsets[(l, h)].append(dominant)
            koa_val = _compute_koa_for_head(dominant, boundary_pos, len(token_ids))
            koa_si_vals.append(koa_val)
            per_head_koa_accum[(l, h)].append(koa_val)

        koa_mean = float(np.mean(koa_si_vals)) if koa_si_vals else 0.0

        # Accumulate per-head BDI for later per-head analysis.
        for k, v in head_bdi.items():
            per_head_bdi_accum[k].append(v)

        elapsed = time.time() - t0
        print(f"  seq {seq_idx+1}/{len(sequences)}: "
              f"BDI_si={bdi_si_mean:.4f}  BDI_lo={bdi_lo_mean:.4f}  "
              f"BAC={bac:.2f}  KOA={koa_mean:.3f}  "
              f"(#b={len(boundary_pos)} #c={len(continuation_pos)}) "
              f"({elapsed:.1f}s)")

        rows.append({
            "sequence_id": seq_idx,
            "bdi_high_si": safe_float(bdi_si_mean),
            "bdi_low_si": safe_float(bdi_lo_mean),
            "bdi_delta": safe_float(bdi_si_mean - bdi_lo_mean),
            "bac": safe_float(bac),
            "koa_high_si": safe_float(koa_mean),
            "n_boundary": len(boundary_pos),
            "n_continuation": len(continuation_pos),
            "boundary_fraction": safe_float(len(boundary_pos) / max(1, len(boundary_pos) + len(continuation_pos))),
        })

    df = pd.DataFrame(rows)
    write_parquet(model_root / "alignment_scores.parquet", df)

    # ── Per-head BDI × R² correlation (Approach C from 3P2-B) ──
    per_head_mean_bdi = {k: float(np.mean(v)) for k, v in per_head_bdi_accum.items()}
    r2_lookup = {(int(r.layer), int(r.head)): r.mean_r2 for r in mean_r2_df.itertuples()}
    common_keys = sorted(set(per_head_mean_bdi) & set(r2_lookup))
    bdi_arr = np.array([per_head_mean_bdi[k] for k in common_keys])
    r2_arr = np.array([r2_lookup[k] for k in common_keys])
    if len(bdi_arr) >= 3:
        bdi_r2_corr, bdi_r2_p = sp_stats.pearsonr(bdi_arr, r2_arr)
        bdi_r2_rho, bdi_r2_rho_p = sp_stats.spearmanr(bdi_arr, r2_arr)
    else:
        bdi_r2_corr = bdi_r2_p = bdi_r2_rho = bdi_r2_rho_p = float("nan")

    # ── Summary statistics ──
    bdi_hi = [r["bdi_high_si"] for r in rows]
    bdi_lo = [r["bdi_low_si"] for r in rows]
    bdi_delta = [r["bdi_delta"] for r in rows]
    bac_vals = [r["bac"] for r in rows if not np.isnan(r["bac"])]
    koa_vals = [r["koa_high_si"] for r in rows]

    # Statistical tests.
    # H1: BDI_SI > 0 (SI heads discriminate boundaries).
    if len(bdi_hi) >= 2 and float(np.std(bdi_hi)) > 0:
        t1, p1 = sp_stats.ttest_1samp(bdi_hi, 0.0)
        p1_one = p1 / 2 if t1 > 0 else 1.0 - p1 / 2
    else:
        t1 = p1_one = float("nan")

    # H2: BDI_SI > BDI_nonSI (SI heads discriminate more than non-SI).
    if len(bdi_delta) >= 2 and float(np.std(bdi_delta)) > 0:
        t2, p2 = sp_stats.ttest_1samp(bdi_delta, 0.0)
        p2_one = p2 / 2 if t2 > 0 else 1.0 - p2 / 2
    else:
        t2 = p2_one = float("nan")

    # H3: BAC > 1.0 (SI heads are disproportionately boundary-sensitive).
    if len(bac_vals) >= 2 and float(np.std(bac_vals)) > 0:
        t3, p3 = sp_stats.ttest_1samp(bac_vals, 1.0)
        p3_one = p3 / 2 if t3 > 0 else 1.0 - p3 / 2
    else:
        t3 = p3_one = float("nan")

    # H4: KOA > chance. Chance KOA depends on dominant offset; approximate as 2/Δ*.
    # Use a simple one-sample test against the median dominant offset's chance level.
    all_dom_offsets = [v for vals in per_head_dominant_offsets.values() for v in vals]
    median_offset = int(np.median(all_dom_offsets)) if all_dom_offsets else 1
    chance_koa = 2.0 / max(1, median_offset)  # tolerance=1, so ~2/Δ* positions per gap are "aligned"
    if len(koa_vals) >= 2 and float(np.std(koa_vals)) > 0:
        t4, p4 = sp_stats.ttest_1samp(koa_vals, chance_koa)
        p4_one = p4 / 2 if t4 > 0 else 1.0 - p4 / 2
    else:
        t4 = p4_one = float("nan")

    summary = {
        "model": model_name,
        "n_sequences": len(sequences),
        "seq_len": seq_len,
        "n_si_heads": len(si_head_set),
        "n_low_si_heads": len(low_si_set),
        "timestamp": now_timestamp(),

        # BDI: Boundary Discrimination Index.
        "bdi_high_si": mean_ci95(bdi_hi),
        "bdi_low_si": mean_ci95(bdi_lo),
        "bdi_delta": mean_ci95(bdi_delta),
        "bdi_cohens_d": safe_float(cohens_d(bdi_hi, bdi_lo)),
        "h1_bdi_si_gt_zero": {
            "t_statistic": safe_float(t1),
            "p_one_sided": safe_float(p1_one),
            "significant": bool(not np.isnan(p1_one) and p1_one < 0.05),
        },
        "h2_bdi_si_gt_nonsi": {
            "t_statistic": safe_float(t2),
            "p_one_sided": safe_float(p2_one),
            "significant": bool(not np.isnan(p2_one) and p2_one < 0.05),
        },

        # BAC: Boundary Attention Contrast.
        "bac": mean_ci95(bac_vals),
        "h3_bac_gt_1": {
            "t_statistic": safe_float(t3),
            "p_one_sided": safe_float(p3_one),
            "significant": bool(not np.isnan(p3_one) and p3_one < 0.05),
        },

        # KOA: Kernel Offset Alignment.
        "koa_high_si": mean_ci95(koa_vals),
        "koa_median_dominant_offset": median_offset,
        "koa_chance_level": safe_float(chance_koa),
        "h4_koa_gt_chance": {
            "t_statistic": safe_float(t4),
            "p_one_sided": safe_float(p4_one),
            "significant": bool(not np.isnan(p4_one) and p4_one < 0.05),
        },

        # BDI × R² correlation.
        "bdi_r2_correlation": {
            "pearson_r": safe_float(bdi_r2_corr),
            "pearson_p": safe_float(bdi_r2_p),
            "spearman_rho": safe_float(bdi_r2_rho),
            "spearman_p": safe_float(bdi_r2_rho_p),
            "n_heads": len(common_keys),
        },
    }

    write_json(model_root / "alignment_summary.json", summary)

    print(f"\n  [8A] BDI (high-SI): {summary['bdi_high_si']}")
    print(f"  [8A] BDI (low-SI):  {summary['bdi_low_si']}")
    print(f"  [8A] BDI delta:     {summary['bdi_delta']}")
    print(f"  [8A] BAC:           {summary['bac']}")
    print(f"  [8A] KOA:           {summary['koa_high_si']} (chance={chance_koa:.3f})")
    print(f"  [8A] BDI×R² corr:   r={bdi_r2_corr:.3f} (p={bdi_r2_p:.2e})")
    print(f"  [8A] Cohen's d:     {summary['bdi_cohens_d']}")

    adapter.cleanup()
    clear_cuda()
    return summary


# ---------------------------------------------------------------------------
# 8B: Tokenizer-Aware Pruning
# ---------------------------------------------------------------------------


def _heads_for_fraction(
    ranked_heads: list[HeadID], fraction_pct: int,
) -> list[HeadID]:
    """Select the top `fraction_pct`% of ranked heads (highest R² first)."""
    n = max(0, int(math.ceil(len(ranked_heads) * fraction_pct / 100)))
    return ranked_heads[:n]


def _compute_attention_entropy(
    model, adapter, tokenizer, model_name: str, device: str,
    num_sequences: int = 12, seq_len: int = 256,
) -> pd.DataFrame:
    """Compute mean attention entropy per head (lower entropy = more peaked).

    Returns DataFrame with columns [layer, head, mean_entropy].
    """
    sequences = load_profile_sequences(
        tokenizer=tokenizer, model_name=model_name,
        num_sequences=num_sequences, seq_len=seq_len,
    )

    entropy_accum: dict[tuple[int, int], list[float]] = defaultdict(list)

    for token_ids in sequences:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            attn = _extract_attention_weights(model, adapter, input_ids, device)
        # attn: [layers, heads, seq, seq]
        n_layers, n_heads = attn.shape[:2]
        for l in range(n_layers):
            for h in range(n_heads):
                # Entropy of attention distribution per query, averaged.
                a = attn[l, h]  # [seq, seq]
                # Avoid log(0).
                log_a = torch.log(a.clamp(min=1e-12))
                ent = -(a * log_a).sum(dim=-1).mean().item()
                entropy_accum[(l, h)].append(ent)

    rows = []
    for (l, h), vals in entropy_accum.items():
        rows.append({"layer": l, "head": h, "mean_entropy": float(np.mean(vals))})
    return pd.DataFrame(rows).sort_values("mean_entropy").reset_index(drop=True)


def _compute_activation_magnitude(
    model, tokenizer, model_name: str, device: str,
    num_sequences: int = 12, seq_len: int = 256,
) -> pd.DataFrame:
    """Compute mean activation magnitude per head.

    Returns DataFrame with columns [layer, head, mean_magnitude].
    """
    sequences = load_profile_sequences(
        tokenizer=tokenizer, model_name=model_name,
        num_sequences=num_sequences, seq_len=seq_len,
    )

    # Register hooks to capture attention output per layer.
    mag_accum: dict[tuple[int, int], list[float]] = defaultdict(list)

    for token_ids in sequences:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        layer_outputs: dict[int, torch.Tensor] = {}
        hooks = []

        # Hook into o_proj inputs (same as ablation mechanism).
        for layer_idx, layer_module in enumerate(_get_attention_layers(model)):
            o_proj = _get_o_proj(layer_module)
            if o_proj is None:
                continue

            def _make_hook(li: int):
                def hook_fn(module, args):
                    x = args[0] if isinstance(args, tuple) else args
                    layer_outputs[li] = x.detach().cpu()
                return hook_fn

            hooks.append(o_proj.register_forward_pre_hook(_make_hook(layer_idx)))

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

        for hk in hooks:
            hk.remove()

        # Parse per-head magnitudes.
        n_heads = _get_num_heads(model)
        for li, tensor in layer_outputs.items():
            # tensor: [batch, seq, hidden_dim]
            hidden = tensor.shape[-1]
            head_dim = hidden // n_heads
            reshaped = tensor.view(tensor.shape[0], tensor.shape[1], n_heads, head_dim)
            for h in range(n_heads):
                mag = reshaped[:, :, h, :].abs().mean().item()
                mag_accum[(li, h)].append(mag)

    rows = []
    for (l, h), vals in mag_accum.items():
        rows.append({"layer": l, "head": h, "mean_magnitude": float(np.mean(vals))})
    return pd.DataFrame(rows).sort_values("mean_magnitude", ascending=False).reset_index(drop=True)


def _get_attention_layers(model):
    """Yield attention layer modules."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Cannot find attention layers")


def _get_o_proj(layer_module):
    """Get the output projection module from an attention layer."""
    if hasattr(layer_module, "self_attn"):
        attn = layer_module.self_attn
        if hasattr(attn, "o_proj"):
            return attn.o_proj
    if hasattr(layer_module, "attn"):
        attn = layer_module.attn
        if hasattr(attn, "c_proj"):
            return attn.c_proj
    return None


def _get_num_heads(model) -> int:
    """Get number of attention heads from model config."""
    cfg = model.config
    if hasattr(cfg, "num_attention_heads"):
        return cfg.num_attention_heads
    if hasattr(cfg, "n_head"):
        return cfg.n_head
    raise ValueError("Cannot determine number of attention heads")


def _build_mask_from_ranking(
    ranked_df: pd.DataFrame, n_select: int, ascending: bool = True,
) -> list[HeadID]:
    """Build a mask selecting `n_select` heads from a ranked DataFrame.

    ascending=True: select lowest values (e.g., lowest entropy = most peaked).
    ascending=False: select highest values.
    """
    df = ranked_df.sort_values(ranked_df.columns[-1], ascending=ascending).reset_index(drop=True)
    n = max(0, min(int(n_select), len(df)))
    return [HeadID(int(r.layer), int(r.head)) for r in df.head(n).itertuples()]


def _evaluate_under_ablation(
    *,
    model,
    tokenizer,
    model_name: str,
    device: str,
    heads_to_ablate: list[HeadID],
    eval_battery: dict,
    eval_sequences: int = ABLATION_EVAL_SEQUENCES,
    seq_len: int = ABLATION_SEQ_LEN,
) -> dict[str, float]:
    """Evaluate math accuracy and wiki perplexity with heads ablated."""
    with head_output_ablation(model, heads_to_ablate):
        math_eval = evaluate_math_accuracy(
            model=model, tokenizer=tokenizer, device=device,
            eval_battery=eval_battery,
        )
        wiki_eval = evaluate_wiki_perplexity(
            model=model, tokenizer=tokenizer, model_name=model_name,
            device=device, count=eval_sequences, seq_len=seq_len,
        )
    return {
        "math_accuracy": safe_float(math_eval["overall_accuracy"]),
        "wiki_perplexity": safe_float(wiki_eval["perplexity"]),
    }


def run_8b(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    fractions: tuple[int, ...] = ABLATION_FRACTIONS,
    cross_model_name: str | None = None,
) -> dict[str, Any]:
    """Run Experiment 8B: tokenizer-aware pruning comparison.

    Compares 5 mask types at each ablation fraction:
      - si_matched: retain top SI heads by R² (from this model's profile)
      - si_mismatched: retain top SI heads from cross_model_name's profile
      - entropy: retain lowest-entropy heads
      - activation: retain highest-activation heads
      - random: retain random heads

    At each fraction, we *ablate the non-retained heads* and evaluate.
    """
    model_root = ensure_dir(output_root / model_name)
    print(f"\n{'=' * 60}")
    print(f"[8B] {model_name}: Tokenizer-Aware Pruning")
    print(f"{'=' * 60}")

    model, tokenizer, model_spec = _load_model_and_tokenizer(model_name, device)
    model.eval()
    adapter = get_adapter(model_spec)
    adapter.register(model)

    # Get all head IDs.
    ranked_heads, mean_r2_df = _get_ranked_heads(model_name)
    all_heads = list(ranked_heads)
    n_total = len(all_heads)

    # Build mask rankings.
    print("  Computing attention entropy...")
    entropy_df = _compute_attention_entropy(model, adapter, tokenizer, model_name, device)
    print("  Computing activation magnitudes...")
    activation_df = _compute_activation_magnitude(model, tokenizer, model_name, device)

    # Cross-model SI ranking (mismatched tokenizer).
    cross_ranked = None
    if cross_model_name:
        try:
            cross_ranked, _ = _get_ranked_heads(cross_model_name)
            print(f"  Loaded cross-model R² profile from {cross_model_name}")
        except FileNotFoundError:
            print(f"  WARNING: No R² profile for {cross_model_name}, skipping mismatch condition")

    # Eval battery.
    eval_battery = build_math_eval_battery(count_per_task=32, seed=42)

    # Run progressive ablation.
    rows: list[dict[str, Any]] = []
    mask_types = ["si_matched", "entropy", "activation", "random"]
    if cross_ranked is not None:
        mask_types.insert(1, "si_mismatched")

    all_head_pairs = {(h.layer, h.head) for h in all_heads}
    for frac in fractions:
        # Protocol uses ablation fractions (0-75%), not retained fractions.
        n_ablate = max(0, int(math.ceil(n_total * frac / 100)))
        n_retain = max(0, n_total - n_ablate)

        print(f"\n  --- Fraction {frac}% ablated (retain {n_retain}, ablate {n_ablate}) ---")

        for mask_type in mask_types:
            if mask_type == "si_matched":
                retained = set((h.layer, h.head) for h in ranked_heads[:n_retain])
            elif mask_type == "si_mismatched" and cross_ranked is not None:
                # Use cross-model ranking, but only heads that exist in this model.
                valid_cross = [h for h in cross_ranked if (h.layer, h.head) in all_head_pairs]
                retained_list = [(h.layer, h.head) for h in valid_cross[:n_retain]]
                # Keep cardinality matched even if cross-model index spaces diverge.
                if len(retained_list) < n_retain:
                    fallback = [
                        (h.layer, h.head) for h in ranked_heads
                        if (h.layer, h.head) not in set(retained_list)
                    ]
                    retained_list.extend(fallback[: (n_retain - len(retained_list))])
                retained = set(retained_list)
            elif mask_type == "entropy":
                # Retain lowest-entropy heads (most peaked attention).
                ent_heads = _build_mask_from_ranking(entropy_df, n_retain, ascending=True)
                retained = set((h.layer, h.head) for h in ent_heads)
            elif mask_type == "activation":
                # Retain highest-activation heads.
                act_heads = _build_mask_from_ranking(activation_df, n_retain, ascending=False)
                retained = set((h.layer, h.head) for h in act_heads)
            elif mask_type == "random":
                rng = random.Random(42 + frac)
                sample = rng.sample(all_heads, k=n_retain) if n_retain < n_total else all_heads
                retained = set((h.layer, h.head) for h in sample)
            else:
                continue

            # Ablate everything NOT retained.
            to_ablate = [h for h in all_heads if (h.layer, h.head) not in retained]

            t0 = time.time()
            metrics = _evaluate_under_ablation(
                model=model, tokenizer=tokenizer, model_name=model_name,
                device=device, heads_to_ablate=to_ablate, eval_battery=eval_battery,
            )
            elapsed = time.time() - t0

            row = {
                "mask_type": mask_type,
                "fraction_ablated_pct": frac,
                "fraction_retained_pct": 100 - frac,
                "n_retained": n_retain,
                "n_ablated": len(to_ablate),
                "math_accuracy": metrics["math_accuracy"],
                "wiki_perplexity": metrics["wiki_perplexity"],
                "elapsed_sec": round(elapsed, 1),
            }
            rows.append(row)
            print(f"    {mask_type:18s}: math={metrics['math_accuracy']:.3f}  "
                  f"ppl={metrics['wiki_perplexity']:.2f}  ({elapsed:.0f}s)")

    df = pd.DataFrame(rows)
    write_parquet(model_root / "pruning_curves.parquet", df)

    # Compute AUDC (area under degradation curve) for each mask type.
    audc: dict[str, dict[str, float]] = {}
    for mt in mask_types:
        mt_df = df[df["mask_type"] == mt].sort_values("fraction_ablated_pct")
        if len(mt_df) < 2:
            continue
        # AUDC for math accuracy (higher = better, so negate for "degradation").
        x = mt_df["fraction_ablated_pct"].values / 100.0
        y_math = mt_df["math_accuracy"].values
        y_ppl = mt_df["wiki_perplexity"].values
        audc[mt] = {
            "math_accuracy_audc": safe_float(float(np.trapz(y_math, x))),
            "wiki_perplexity_audc": safe_float(float(np.trapz(y_ppl, x))),
        }

    # Statistical comparisons.
    comparisons = {}
    from scipy import stats as sp_stats

    for metric_col in ["math_accuracy", "wiki_perplexity"]:
        for mt_a, mt_b in [("si_matched", "random"), ("si_matched", "entropy"),
                            ("si_matched", "activation")]:
            if mt_a not in audc or mt_b not in audc:
                continue
            key = f"{mt_a}_vs_{mt_b}_{metric_col}"
            # Paired comparison across fractions.
            a_vals = df[df["mask_type"] == mt_a].sort_values("fraction_ablated_pct")[metric_col].values
            b_vals = df[df["mask_type"] == mt_b].sort_values("fraction_ablated_pct")[metric_col].values
            if len(a_vals) == len(b_vals) and len(a_vals) > 2:
                t, p = sp_stats.ttest_rel(a_vals, b_vals)
                comparisons[key] = {
                    "t_statistic": safe_float(t),
                    "p_two_sided": safe_float(p),
                    "mean_diff": safe_float(float(np.mean(a_vals - b_vals))),
                }

    # Mismatch test (H_8B3).
    if "si_mismatched" in mask_types:
        for metric_col in ["math_accuracy", "wiki_perplexity"]:
            a_vals = df[df["mask_type"] == "si_matched"].sort_values("fraction_ablated_pct")[metric_col].values
            b_vals = df[df["mask_type"] == "si_mismatched"].sort_values("fraction_ablated_pct")[metric_col].values
            if len(a_vals) == len(b_vals) and len(a_vals) > 2:
                t, p = sp_stats.ttest_rel(a_vals, b_vals)
                comparisons[f"matched_vs_mismatched_{metric_col}"] = {
                    "t_statistic": safe_float(t),
                    "p_two_sided": safe_float(p),
                    "mean_diff": safe_float(float(np.mean(a_vals - b_vals))),
                }

    summary = {
        "model": model_name,
        "cross_model": cross_model_name,
        "n_total_heads": n_total,
        "fractions": list(fractions),
        "mask_types": mask_types,
        "audc": audc,
        "comparisons": comparisons,
        "timestamp": now_timestamp(),
    }

    write_json(model_root / "pruning_summary.json", summary)
    print(f"\n  [8B] AUDC summary:")
    for mt, vals in audc.items():
        print(f"    {mt:18s}: math_audc={vals['math_accuracy_audc']:.4f}  ppl_audc={vals['wiki_perplexity_audc']:.2f}")

    adapter.cleanup()
    clear_cuda()
    return summary
