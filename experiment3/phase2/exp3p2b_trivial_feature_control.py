#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import torch

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    LogisticRegression = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment3.theory5b_boundary_detection import (  # noqa: E402
    MODELS,
    compute_word_boundaries,
    load_wiki_sequences,
    parse_head_list,
    run_attention_analysis,
)
from shared.attention.adapters import get_adapter  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402


MODEL_NAMES = tuple(MODELS.keys())
PREFIX_ARTIFACT_ALPHA = 0.05
PREFIX_ARTIFACT_MIN_COHENS_D = 0.20
PREFIX_ARTIFACT_MIN_ABS_DIFF = 0.005


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = np.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2))
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled)


def _prefix_artifact_assessment(
    *,
    fake_with_prefix: np.ndarray,
    real_with_prefix: np.ndarray,
    alpha: float = PREFIX_ARTIFACT_ALPHA,
    min_cohens_d: float = PREFIX_ARTIFACT_MIN_COHENS_D,
    min_abs_diff: float = PREFIX_ARTIFACT_MIN_ABS_DIFF,
) -> dict[str, Any]:
    fake = np.asarray(fake_with_prefix, dtype=float)
    real = np.asarray(real_with_prefix, dtype=float)
    fake = fake[np.isfinite(fake)]
    real = real[np.isfinite(real)]

    out: dict[str, Any] = {
        "n_fake_with_prefix": int(fake.size),
        "n_real_with_prefix": int(real.size),
        "mean_fake_with_prefix": _safe_float(np.nanmean(fake) if fake.size else float("nan")),
        "mean_real_with_prefix": _safe_float(np.nanmean(real) if real.size else float("nan")),
        "fake_minus_real_diff": float("nan"),
        "t_statistic": float("nan"),
        "p_two_sided": float("nan"),
        "p_one_sided_fake_gt_real": float("nan"),
        "cohens_d_fake_minus_real": float("nan"),
        "rule": {
            "alpha_one_sided": float(alpha),
            "min_cohens_d": float(min_cohens_d),
            "min_abs_diff": float(min_abs_diff),
            "decision": "artifact_if_diff_gt_0_and_p_one_lt_alpha_and_d_ge_min_and_abs_diff_ge_min",
        },
        "artifact_flag_statistical": False,
    }
    if fake.size < 2 or real.size < 2:
        return out

    t_stat, p_two = scipy_stats.ttest_ind(fake, real, equal_var=False, nan_policy="omit")
    diff = float(fake.mean() - real.mean())
    d = _cohens_d(fake, real)
    if np.isfinite(t_stat) and np.isfinite(p_two):
        if float(t_stat) > 0.0:
            p_one = float(p_two) / 2.0
        else:
            p_one = 1.0 - (float(p_two) / 2.0)
    else:
        p_one = float("nan")

    flag = bool(
        np.isfinite(diff)
        and np.isfinite(p_one)
        and np.isfinite(d)
        and (diff > 0.0)
        and (abs(diff) >= float(min_abs_diff))
        and (d >= float(min_cohens_d))
        and (p_one < float(alpha))
    )

    out.update(
        {
            "fake_minus_real_diff": diff,
            "t_statistic": _safe_float(t_stat),
            "p_two_sided": _safe_float(p_two),
            "p_one_sided_fake_gt_real": _safe_float(p_one),
            "cohens_d_fake_minus_real": _safe_float(d),
            "artifact_flag_statistical": flag,
        }
    )
    return out


def _token_has_prefix(token_str: str) -> bool:
    return token_str.startswith("\u0120") or token_str.startswith("\u2581")


def _prefix_lookup_from_tokenizer(tokenizer, token_ids: np.ndarray) -> dict[int, bool]:
    ids = [int(t) for t in np.unique(token_ids)]
    toks = tokenizer.convert_ids_to_tokens(ids)
    return {tid: _token_has_prefix(tok or "") for tid, tok in zip(ids, toks)}


def _load_prefix_lookup(model: str, token_ids: np.ndarray) -> tuple[dict[int, bool], str]:
    try:
        tokenizer = load_tokenizer(MODELS[model])
        lookup = _prefix_lookup_from_tokenizer(tokenizer, token_ids)
        return lookup, "tokenizer_prefix_markers"
    except Exception as exc:
        lookup = {int(t): False for t in np.unique(token_ids)}
        return lookup, f"fallback_no_tokenizer:{type(exc).__name__}"


def _space_prefix_classifier(none_df: pd.DataFrame, prefix_lookup: dict[int, bool], method_tag: str) -> dict[str, Any]:
    df = none_df.copy()
    df["is_word_initial"] = (df["token_type"] == "word_initial").astype(int)
    df["has_prefix_marker"] = df["token_id"].map(lambda t: int(prefix_lookup.get(int(t), False)))

    y = df["is_word_initial"].to_numpy(dtype=np.int32)
    pred = df["has_prefix_marker"].to_numpy(dtype=np.int32)
    acc = float(np.mean(pred == y))

    global_rate = float(df["is_word_initial"].mean())
    token_stats = df.groupby("token_id", as_index=False).agg(
        p_word_initial=("is_word_initial", "mean"),
        count=("is_word_initial", "count"),
    )
    token_stats["signal"] = np.abs(token_stats["p_word_initial"] - global_rate) * np.sqrt(token_stats["count"])
    top = token_stats.sort_values("signal", ascending=False).head(16)

    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    specificity = float(tn / max(1, tn + fp))

    return {
        "method": method_tag,
        "feature_type": "prefix_marker_surrogate",
        "classifier_accuracy": acc,
        "precision_word_initial": precision,
        "recall_word_initial": recall,
        "specificity_continuation": specificity,
        "n_samples": int(len(df)),
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "top_k_signal_token_ids": [
            {
                "token_id": int(r.token_id),
                "p_word_initial": float(r.p_word_initial),
                "count": int(r.count),
                "signal": float(r.signal),
            }
            for r in top.itertuples()
        ],
        "acceptance_threshold": 0.90,
        "meets_threshold": bool(acc >= 0.90),
    }


def _fit_embedding_prefix_classifier(
    *,
    model,
    tokenizer,
    token_ids: np.ndarray,
    top_k_dims: int,
) -> tuple[dict[str, Any], list[int], np.ndarray]:
    unique_ids = np.unique(token_ids.astype(np.int64))
    if len(unique_ids) < 4:
        raise RuntimeError("Not enough unique token ids for embedding prefix classifier.")

    toks = tokenizer.convert_ids_to_tokens([int(t) for t in unique_ids])
    y = np.array([1 if _token_has_prefix(tok or "") else 0 for tok in toks], dtype=np.int32)
    if int(y.min()) == int(y.max()):
        raise RuntimeError("Embedding prefix classifier found only one class.")

    emb = model.get_input_embeddings().weight
    ids_t = torch.tensor(unique_ids.tolist(), device=emb.device, dtype=torch.long)
    with torch.inference_mode():
        x = emb[ids_t].float().cpu().numpy()

    if LogisticRegression is None:
        pos = x[y == 1].mean(axis=0)
        neg = x[y == 0].mean(axis=0)
        coef = pos - neg
        logits = x @ coef
        thresh = float(np.median(logits))
        pred = (logits >= thresh).astype(np.int32)
        method = "embedding_linear_mean_diff_fallback"
    else:
        clf = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="liblinear",
            random_state=0,
        )
        clf.fit(x, y)
        pred = clf.predict(x).astype(np.int32)
        coef = clf.coef_[0]
        method = "embedding_logistic_regression"

    acc = float(np.mean(pred == y))
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    specificity = float(tn / max(1, tn + fp))

    top_k = max(1, int(top_k_dims))
    ranked = np.argsort(np.abs(coef))[::-1]
    top_dims = [int(i) for i in ranked[:top_k].tolist()]
    top_weights = [float(coef[i]) for i in top_dims]

    dim_idx = torch.tensor(top_dims, device=emb.device, dtype=torch.long)
    with torch.inference_mode():
        mean_vals = emb.index_select(dim=1, index=dim_idx).float().mean(dim=0).cpu().numpy()

    report = {
        "method": method,
        "feature_type": "embedding_dimensions",
        "classifier_accuracy": acc,
        "precision_word_initial": precision,
        "recall_word_initial": recall,
        "specificity_continuation": specificity,
        "n_samples": int(len(unique_ids)),
        "n_word_initial": int(y.sum()),
        "n_continuation": int((1 - y).sum()),
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "top_k_dims": top_dims,
        "top_k_weights": top_weights,
        "acceptance_threshold": 0.90,
        "meets_threshold": bool(acc >= 0.90),
    }
    return report, top_dims, mean_vals


def _post_ablation_t5b_a_proxy(t5b_report: dict[str, Any], classifier: dict[str, Any]) -> dict[str, Any]:
    comp = (
        t5b_report.get("approach_a", {})
        .get("high_vs_low_comparison", {})
        .get("attn_to_prev_last", {})
    )
    d = _safe_float(comp.get("cohens_d"))
    p_holm = _safe_float(comp.get("t_p_value_holm"))

    return {
        "method": "cached_approach_a_proxy_no_new_embedding_intervention",
        "note": (
            "This Stage 1 executable reuses post-refresh Approach A metrics as a proxy checkpoint; "
            "full embedding-dimension intervention rerun is deferred to dedicated heavy compute."
        ),
        "high_vs_low_attn_to_prev_last": {
            "difference": _safe_float(comp.get("difference")),
            "cohens_d": d,
            "t_p_value_holm": p_holm,
            "n_high": int(comp.get("n_high", 0)),
            "n_low": int(comp.get("n_low", 0)),
        },
        "space_prefix_classifier_summary": {
            "classifier_accuracy": _safe_float(classifier.get("classifier_accuracy")),
            "meets_90pct_threshold": bool(classifier.get("meets_threshold", False)),
        },
        "decision_rule": {
            "non_trivial_if_d_gt_0_5": bool(np.isfinite(d) and d > 0.5),
            "trivial_if_d_lt_0_2": bool(np.isfinite(d) and d < 0.2),
        },
    }


def _post_ablation_t5b_a_full(approach_a: dict[str, Any], classifier: dict[str, Any]) -> dict[str, Any]:
    comp = approach_a.get("high_vs_low_comparison", {}).get("attn_to_prev_last", {})
    high = approach_a.get("high_si_heads", {}).get("attn_to_prev_last", {})
    low = approach_a.get("low_si_heads", {}).get("attn_to_prev_last", {})
    d = _safe_float(comp.get("cohens_d"))

    return {
        "method": "full_forward_embedding_dimension_ablation",
        "note": (
            "Applied embedding-dimension mean ablation on top-k prefix-predictive dimensions and "
            "recomputed boundary attention (Approach A) via fresh forward passes."
        ),
        "high_vs_low_attn_to_prev_last": {
            "difference": _safe_float(comp.get("difference")),
            "cohens_d": d,
            "t_p_value": _safe_float(comp.get("t_p_value")),
            "t_p_value_holm": _safe_float(comp.get("t_p_value_holm")),
            "n_high": int(comp.get("n_high", 0)),
            "n_low": int(comp.get("n_low", 0)),
        },
        "group_estimates": {
            "high_si_mean": _safe_float(high.get("mean")),
            "high_si_ci_95": high.get("ci_95", [float("nan"), float("nan")]),
            "low_si_mean": _safe_float(low.get("mean")),
            "low_si_ci_95": low.get("ci_95", [float("nan"), float("nan")]),
        },
        "space_prefix_classifier_summary": {
            "classifier_accuracy": _safe_float(classifier.get("classifier_accuracy")),
            "meets_90pct_threshold": bool(classifier.get("meets_threshold", False)),
        },
        "decision_rule": {
            "non_trivial_if_d_gt_0_5": bool(np.isfinite(d) and d > 0.5),
            "trivial_if_d_lt_0_2": bool(np.isfinite(d) and d < 0.2),
        },
    }


def _synthetic_boundary_proxy(
    none_df: pd.DataFrame,
    high_loss_inc_df: pd.DataFrame,
    prefix_lookup: dict[int, bool],
    method_tag: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    df = none_df.copy()
    df["has_prefix_marker"] = df["token_id"].map(lambda t: bool(prefix_lookup.get(int(t), False)))
    df["is_word_initial"] = df["token_type"] == "word_initial"
    merged = df.merge(
        high_loss_inc_df[["sequence_idx", "position", "token_id", "token_type", "loss_increase"]],
        on=["sequence_idx", "position", "token_id", "token_type"],
        how="left",
    )

    def cell_name(row: pd.Series) -> str:
        if bool(row["is_word_initial"]) and bool(row["has_prefix_marker"]):
            return "real_boundary_with_prefix"
        if (not bool(row["is_word_initial"])) and bool(row["has_prefix_marker"]):
            return "fake_boundary_with_prefix"
        if bool(row["is_word_initial"]) and (not bool(row["has_prefix_marker"])):
            return "real_boundary_without_prefix"
        return "fake_boundary_without_prefix"

    merged["cell"] = merged.apply(cell_name, axis=1)
    cell_stats = merged.groupby("cell", as_index=False).agg(
        mean_loss_increase=("loss_increase", "mean"),
        std_loss_increase=("loss_increase", "std"),
        n=("loss_increase", "count"),
    )
    stats_by_cell = {r["cell"]: r for r in cell_stats.to_dict(orient="records")}

    rbp = _safe_float(stats_by_cell.get("real_boundary_with_prefix", {}).get("mean_loss_increase"))
    fbp = _safe_float(stats_by_cell.get("fake_boundary_with_prefix", {}).get("mean_loss_increase"))
    rbn = _safe_float(stats_by_cell.get("real_boundary_without_prefix", {}).get("mean_loss_increase"))
    fbn = _safe_float(stats_by_cell.get("fake_boundary_without_prefix", {}).get("mean_loss_increase"))

    legacy_prefix_following = bool(np.isfinite(fbp) and np.isfinite(rbp) and fbp >= rbp)
    prefix_assessment = _prefix_artifact_assessment(
        fake_with_prefix=merged[merged["cell"] == "fake_boundary_with_prefix"]["loss_increase"].to_numpy(dtype=float),
        real_with_prefix=merged[merged["cell"] == "real_boundary_with_prefix"]["loss_increase"].to_numpy(dtype=float),
    )
    prefix_following = bool(prefix_assessment.get("artifact_flag_statistical", False))
    result = {
        "method": f"{method_tag}_cached_sequence_proxy",
        "note": (
            "Proxy analysis from cached sequence positions; dedicated adversarial sequence generation is "
            "deferred to heavier follow-up runs."
        ),
        "cells": stats_by_cell,
        "prefix_following_artifact_flag": prefix_following,
        "prefix_following_artifact_flag_legacy_binary": legacy_prefix_following,
        "prefix_following_assessment": prefix_assessment,
        "comparisons": {
            "fake_minus_real_with_prefix": fbp - rbp if np.isfinite(fbp) and np.isfinite(rbp) else float("nan"),
            "real_minus_fake_without_prefix": rbn - fbn if np.isfinite(rbn) and np.isfinite(fbn) else float("nan"),
        },
        "acceptance_rule_inputs": {
            "fake_boundary_with_prefix": fbp,
            "real_boundary_with_prefix": rbp,
            "alpha_one_sided": PREFIX_ARTIFACT_ALPHA,
            "min_cohens_d": PREFIX_ARTIFACT_MIN_COHENS_D,
            "min_abs_diff": PREFIX_ARTIFACT_MIN_ABS_DIFF,
        },
    }
    return result, merged[
        [
            "sequence_idx",
            "position",
            "token_id",
            "token_type",
            "has_prefix_marker",
            "cell",
            "loss_increase",
        ]
    ]


def _build_token_replacement_maps(tokenizer) -> tuple[dict[int, int], dict[int, int]]:
    vocab = tokenizer.get_vocab()
    token_to_id: dict[str, int] = {str(tok): int(tid) for tok, tid in vocab.items()}

    to_prefix: dict[int, int] = {}
    to_noprefix: dict[int, int] = {}
    for tok, tid in token_to_id.items():
        if _token_has_prefix(tok):
            base = tok[1:]
            if base in token_to_id:
                to_noprefix[int(tid)] = int(token_to_id[base])
        else:
            pref = "\u0120" + tok
            us = "\u2581" + tok
            if pref in token_to_id:
                to_prefix[int(tid)] = int(token_to_id[pref])
            elif us in token_to_id:
                to_prefix[int(tid)] = int(token_to_id[us])
    return to_prefix, to_noprefix


def _cell_name(orig_boundary: bool, has_prefix_after: bool) -> str:
    if orig_boundary and has_prefix_after:
        return "real_boundary_with_prefix"
    if (not orig_boundary) and has_prefix_after:
        return "fake_boundary_with_prefix"
    if orig_boundary and (not has_prefix_after):
        return "real_boundary_without_prefix"
    return "fake_boundary_without_prefix"


@contextlib.contextmanager
def _embedding_dimension_mean_ablation(model, dims: list[int], mean_vals: np.ndarray):
    emb = model.get_input_embeddings()
    if not dims:
        yield
        return

    dim_idx_cpu = torch.tensor(dims, dtype=torch.long)
    mean_vals_cpu = torch.tensor(mean_vals, dtype=torch.float32)

    def _hook(_module, _inputs, output):
        if not torch.is_floating_point(output):
            return output
        out = output.clone()
        dim_idx = dim_idx_cpu.to(device=out.device)
        repl = mean_vals_cpu.to(device=out.device, dtype=out.dtype)
        out[..., dim_idx] = repl
        return out

    handle = emb.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def _generate_synthetic_sequences(
    *,
    tokenizer,
    sequences: list[list[int]],
    to_prefix: dict[int, int],
    to_noprefix: dict[int, int],
    target_per_transformed_cell: int,
    seed: int,
) -> tuple[list[list[int]], list[list[bool]], dict[str, int]]:
    rng = random.Random(int(seed))
    adv_sequences = [list(seq) for seq in sequences]
    orig_boundaries: list[list[bool]] = []

    boundary_candidates: list[tuple[int, int]] = []
    nonboundary_candidates: list[tuple[int, int]] = []
    for seq_idx, toks in enumerate(sequences):
        wids = compute_word_boundaries(tokenizer, toks)
        boundary_mask = [False] * len(toks)
        for pos in range(1, len(toks)):
            is_boundary = bool(wids[pos] != wids[pos - 1])
            boundary_mask[pos] = is_boundary
            tid = int(toks[pos])
            if is_boundary and tid in to_noprefix:
                boundary_candidates.append((seq_idx, pos))
            if (not is_boundary) and tid in to_prefix:
                nonboundary_candidates.append((seq_idx, pos))
        orig_boundaries.append(boundary_mask)

    rng.shuffle(boundary_candidates)
    rng.shuffle(nonboundary_candidates)
    n_boundary_replace = min(int(target_per_transformed_cell), len(boundary_candidates))
    n_nonboundary_replace = min(int(target_per_transformed_cell), len(nonboundary_candidates))

    for seq_idx, pos in boundary_candidates[:n_boundary_replace]:
        old_id = int(adv_sequences[seq_idx][pos])
        adv_sequences[seq_idx][pos] = int(to_noprefix.get(old_id, old_id))

    for seq_idx, pos in nonboundary_candidates[:n_nonboundary_replace]:
        old_id = int(adv_sequences[seq_idx][pos])
        adv_sequences[seq_idx][pos] = int(to_prefix.get(old_id, old_id))

    replacement_meta = {
        "boundary_candidates_available": int(len(boundary_candidates)),
        "nonboundary_candidates_available": int(len(nonboundary_candidates)),
        "boundary_replacements_applied": int(n_boundary_replace),
        "nonboundary_replacements_applied": int(n_nonboundary_replace),
    }
    return adv_sequences, orig_boundaries, replacement_meta


def _compute_prev_token_attention_by_group(
    *,
    logits: torch.Tensor,
    seq_len: int,
    high_heads: list[tuple[int, int]],
    low_heads: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    upper_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    row_idx = torch.arange(1, seq_len, dtype=torch.long)
    col_idx = torch.arange(0, seq_len - 1, dtype=torch.long)

    selected = sorted(set(high_heads + low_heads))
    per_head_prev: dict[tuple[int, int], np.ndarray] = {}
    for layer_idx, head_idx in selected:
        if layer_idx >= logits.shape[0] or head_idx >= logits.shape[1]:
            continue
        head_logits = logits[layer_idx, head_idx].float().clone()
        head_logits[upper_mask] = float("-inf")
        attn = torch.softmax(head_logits, dim=-1)
        prev = attn[row_idx, col_idx].cpu().numpy().astype(np.float64, copy=False)
        per_head_prev[(layer_idx, head_idx)] = prev

    def _mean_prev(heads: list[tuple[int, int]]) -> np.ndarray:
        rows = [per_head_prev[h] for h in heads if h in per_head_prev]
        if not rows:
            return np.full((seq_len - 1,), np.nan, dtype=np.float64)
        return np.mean(np.stack(rows, axis=0), axis=0)

    return _mean_prev(high_heads), _mean_prev(low_heads)


def _synthetic_boundary_full(
    *,
    model,
    adapter,
    tokenizer,
    device: str,
    sequences: list[list[int]],
    high_heads: list[tuple[int, int]],
    low_heads: list[tuple[int, int]],
    target_per_transformed_cell: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    to_prefix, to_noprefix = _build_token_replacement_maps(tokenizer)
    adv_sequences, orig_boundaries, repl_meta = _generate_synthetic_sequences(
        tokenizer=tokenizer,
        sequences=sequences,
        to_prefix=to_prefix,
        to_noprefix=to_noprefix,
        target_per_transformed_cell=target_per_transformed_cell,
        seed=seed,
    )

    flat_adv_ids = np.array([int(t) for seq in adv_sequences for t in seq], dtype=np.int64)
    prefix_lookup_adv = _prefix_lookup_from_tokenizer(tokenizer, flat_adv_ids)

    rows: list[dict[str, Any]] = []
    for seq_idx, (orig, adv, boundary_mask) in enumerate(zip(sequences, adv_sequences, orig_boundaries)):
        seq_len = len(adv)
        if seq_len < 3:
            continue
        input_ids = torch.tensor([adv], dtype=torch.long, device=device)
        with torch.inference_mode():
            capture = adapter.capture(
                model,
                input_ids=input_ids,
                include_logits=True,
                return_token_logits=False,
                capture_attention=True,
                output_device="cpu",
            )

        if capture.logits is None:
            continue

        high_prev, low_prev = _compute_prev_token_attention_by_group(
            logits=capture.logits,
            seq_len=seq_len,
            high_heads=high_heads,
            low_heads=low_heads,
        )

        for pos in range(1, seq_len):
            orig_boundary = bool(boundary_mask[pos])
            adv_tid = int(adv[pos])
            has_prefix_after = bool(prefix_lookup_adv.get(adv_tid, False))
            cell = _cell_name(orig_boundary, has_prefix_after)
            rows.append(
                {
                    "sequence_idx": int(seq_idx),
                    "position": int(pos),
                    "orig_token_id": int(orig[pos]),
                    "adv_token_id": adv_tid,
                    "orig_boundary": bool(orig_boundary),
                    "adv_has_prefix": bool(has_prefix_after),
                    "cell": cell,
                    "high_prev_attn": float(high_prev[pos - 1]),
                    "low_prev_attn": float(low_prev[pos - 1]),
                }
            )

        del capture
        del input_ids
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if df.empty:
        return (
            {
                "method": "full_adversarial_forward_pass",
                "error": "no_rows_collected",
                "replacement_meta": repl_meta,
            },
            df,
        )

    cell_stats = (
        df.groupby("cell", as_index=False)
        .agg(
            mean_high_prev_attn=("high_prev_attn", "mean"),
            std_high_prev_attn=("high_prev_attn", "std"),
            mean_low_prev_attn=("low_prev_attn", "mean"),
            std_low_prev_attn=("low_prev_attn", "std"),
            n=("high_prev_attn", "count"),
        )
        .sort_values("cell")
        .reset_index(drop=True)
    )
    stats_by_cell = {r["cell"]: r for r in cell_stats.to_dict(orient="records")}

    rbp = _safe_float(stats_by_cell.get("real_boundary_with_prefix", {}).get("mean_high_prev_attn"))
    fbp = _safe_float(stats_by_cell.get("fake_boundary_with_prefix", {}).get("mean_high_prev_attn"))
    rbn = _safe_float(stats_by_cell.get("real_boundary_without_prefix", {}).get("mean_high_prev_attn"))
    fbn = _safe_float(stats_by_cell.get("fake_boundary_without_prefix", {}).get("mean_high_prev_attn"))

    legacy_prefix_following = bool(np.isfinite(fbp) and np.isfinite(rbp) and fbp >= rbp)
    prefix_assessment = _prefix_artifact_assessment(
        fake_with_prefix=df[df["cell"] == "fake_boundary_with_prefix"]["high_prev_attn"].to_numpy(dtype=float),
        real_with_prefix=df[df["cell"] == "real_boundary_with_prefix"]["high_prev_attn"].to_numpy(dtype=float),
    )
    prefix_following = bool(prefix_assessment.get("artifact_flag_statistical", False))
    result = {
        "method": "full_adversarial_forward_pass",
        "note": (
            "Synthetic 2x2 boundary control computed from fresh attention forward passes on adversarially "
            "perturbed token sequences."
        ),
        "replacement_meta": repl_meta,
        "cells": stats_by_cell,
        "prefix_following_artifact_flag": prefix_following,
        "prefix_following_artifact_flag_legacy_binary": legacy_prefix_following,
        "prefix_following_assessment": prefix_assessment,
        "comparisons": {
            "fake_minus_real_with_prefix": fbp - rbp if np.isfinite(fbp) and np.isfinite(rbp) else float("nan"),
            "real_minus_fake_without_prefix": rbn - fbn if np.isfinite(rbn) and np.isfinite(fbn) else float("nan"),
        },
        "acceptance_rule_inputs": {
            "fake_boundary_with_prefix": fbp,
            "real_boundary_with_prefix": rbp,
            "alpha_one_sided": PREFIX_ARTIFACT_ALPHA,
            "min_cohens_d": PREFIX_ARTIFACT_MIN_COHENS_D,
            "min_abs_diff": PREFIX_ARTIFACT_MIN_ABS_DIFF,
        },
    }
    return result, df


def _offset_group_boundary_scores(
    boundary_df: pd.DataFrame,
    prev_df: pd.DataFrame,
    head_groups: dict[str, Any],
) -> dict[str, Any]:
    merged = boundary_df.merge(prev_df[["layer", "head", "prev_token_score"]], on=["layer", "head"], how="left")
    q25 = float(merged["prev_token_score"].quantile(0.25))
    q75 = float(merged["prev_token_score"].quantile(0.75))

    def group_name(x: float) -> str:
        if not np.isfinite(x):
            return "unknown"
        if x >= q75:
            return "t_minus_1_preferred_proxy"
        if x <= q25:
            return "diffuse_proxy"
        return "mid_offset_proxy"

    merged["offset_group"] = merged["prev_token_score"].astype(float).apply(group_name)

    hs = {(int(h["layer"]), int(h["head"])) for h in head_groups.get("high_si", [])}
    ls = {(int(h["layer"]), int(h["head"])) for h in head_groups.get("low_si", [])}

    def si_group(row: pd.Series) -> str:
        key = (int(row["layer"]), int(row["head"]))
        if key in hs:
            return "high_si"
        if key in ls:
            return "low_si"
        return "other"

    merged["si_group"] = merged.apply(si_group, axis=1)

    stats = merged.groupby(["offset_group", "si_group"], as_index=False).agg(
        mean_boundary_score=("boundary_attn_score", "mean"),
        std_boundary_score=("boundary_attn_score", "std"),
        n=("boundary_attn_score", "count"),
    )

    comparisons: dict[str, Any] = {}
    for grp in sorted(merged["offset_group"].dropna().unique().tolist()):
        high = merged[(merged["offset_group"] == grp) & (merged["si_group"] == "high_si")]["boundary_attn_score"].to_numpy()
        low = merged[(merged["offset_group"] == grp) & (merged["si_group"] == "low_si")]["boundary_attn_score"].to_numpy()
        if len(high) >= 2 and len(low) >= 2:
            t_stat, p_val = scipy_stats.ttest_ind(high, low, equal_var=False)
            d = _cohens_d(high, low)
        else:
            t_stat, p_val, d = float("nan"), float("nan"), float("nan")
        comparisons[grp] = {
            "n_high_si": int(len(high)),
            "n_low_si": int(len(low)),
            "mean_high_si": float(np.mean(high)) if len(high) else float("nan"),
            "mean_low_si": float(np.mean(low)) if len(low) else float("nan"),
            "difference": (float(np.mean(high) - np.mean(low)) if len(high) and len(low) else float("nan")),
            "t_statistic": _safe_float(t_stat),
            "p_value": _safe_float(p_val),
            "cohens_d": _safe_float(d),
        }

    return {
        "method": "prev_token_quantile_offset_proxy",
        "quantiles": {"q25": q25, "q75": q75},
        "group_stats": stats.to_dict(orient="records"),
        "high_vs_low_comparisons": comparisons,
    }


def run_proxy(model: str, output_root: Path) -> None:
    out_dir = output_root / model
    out_dir.mkdir(parents=True, exist_ok=True)

    t5_dir = Path("results/experiment3/theory5_subword_ablation") / model
    t5b_dir = Path("results/experiment3/theory5b_boundary_detection") / model
    t1_dir = Path("results/experiment3/theory1_si_circuits") / model
    t7_dir = Path("results/experiment3/theory7_induction_feeders") / model

    t5_losses_path = t5_dir / "per_position_losses.parquet"
    t5b_report_path = t5b_dir / "report.json"
    t5b_scores_path = t5b_dir / "boundary_attention_scores.parquet"
    head_groups_path = t1_dir / "head_groups.json"
    prev_path = t7_dir / "prev_token_scores.parquet"

    required = [t5_losses_path, t5b_report_path, t5b_scores_path, head_groups_path, prev_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs for 3P2-B ({model}): {missing}")

    t5_losses = pd.read_parquet(t5_losses_path)
    t5b_report = _load_json(t5b_report_path)
    boundary_scores = pd.read_parquet(t5b_scores_path)
    head_groups = _load_json(head_groups_path)
    prev_scores = pd.read_parquet(prev_path)

    none_df = t5_losses[t5_losses["condition"] == "none"].copy()
    high_df = t5_losses[t5_losses["condition"] == "ablate_high_si"].copy()
    baseline_df = none_df[["sequence_idx", "position", "token_id", "token_type", "loss"]].rename(
        columns={"loss": "baseline_loss"}
    )
    high_df = high_df.merge(
        baseline_df,
        on=["sequence_idx", "position", "token_id", "token_type"],
        how="left",
    )
    high_df["loss_increase"] = high_df["loss"] - high_df["baseline_loss"]

    prefix_lookup, method_tag = _load_prefix_lookup(model, none_df["token_id"].to_numpy(dtype=np.int64))

    space_classifier = _space_prefix_classifier(none_df, prefix_lookup, method_tag)
    _write_json(out_dir / "space_prefix_classifier.json", space_classifier)

    post_ablation_proxy = _post_ablation_t5b_a_proxy(t5b_report, space_classifier)
    _write_json(out_dir / "post_ablation_t5b_a.json", post_ablation_proxy)

    synthetic_result, synthetic_rows = _synthetic_boundary_proxy(
        none_df=none_df,
        high_loss_inc_df=high_df,
        prefix_lookup=prefix_lookup,
        method_tag=method_tag,
    )
    _write_json(out_dir / "synthetic_boundary_results.json", synthetic_result)
    synthetic_rows.to_parquet(out_dir / "adversarial_sequences.parquet", index=False)

    offset_scores = _offset_group_boundary_scores(boundary_scores, prev_scores, head_groups)
    _write_json(out_dir / "offset_group_boundary_scores.json", offset_scores)

    print(f"[3P2-B][proxy] {model}: wrote artifacts to {out_dir}")


def run_full(
    *,
    model_name: str,
    output_root: Path,
    device: str,
    num_sequences: int,
    seq_len: int,
    top_k_dims: int,
    target_per_transformed_cell: int,
    seed: int,
) -> None:
    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_dir = Path("results/experiment3/theory1_si_circuits") / model_name
    t5b_dir = Path("results/experiment3/theory5b_boundary_detection") / model_name
    t7_dir = Path("results/experiment3/theory7_induction_feeders") / model_name

    head_groups_path = t1_dir / "head_groups.json"
    r2_path = t1_dir / "per_sequence_r2.parquet"
    t5b_scores_path = t5b_dir / "boundary_attention_scores.parquet"
    prev_path = t7_dir / "prev_token_scores.parquet"
    required = [head_groups_path, r2_path, t5b_scores_path, prev_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs for 3P2-B full mode ({model_name}): {missing}")

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)
    adapter = get_adapter(model_spec)
    adapter.register(model)

    sequences = load_wiki_sequences(model_name, max_sequences=max(8, num_sequences), seq_len=seq_len)[:num_sequences]
    if len(sequences) < 4:
        raise RuntimeError(f"Need >=4 sequences for full 3P2-B, found {len(sequences)}")

    head_groups = _load_json(head_groups_path)
    high_si = parse_head_list(head_groups["high_si"])
    low_si = parse_head_list(head_groups["low_si"])
    high_heads = [(int(h.layer), int(h.head)) for h in high_si]
    low_heads = [(int(h.layer), int(h.head)) for h in low_si]

    token_ids = np.array([int(t) for seq in sequences for t in seq], dtype=np.int64)
    classifier, top_dims, mean_vals = _fit_embedding_prefix_classifier(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        top_k_dims=top_k_dims,
    )
    _write_json(out_dir / "space_prefix_classifier.json", classifier)

    r2_per_seq = pd.read_parquet(r2_path)
    r2_df = r2_per_seq.groupby(["layer", "head"], as_index=False)["r2"].mean().rename(columns={"r2": "mean_r2"})

    with _embedding_dimension_mean_ablation(model, top_dims, mean_vals):
        approach_a, _ = run_attention_analysis(
            model=model,
            adapter=adapter,
            model_spec=model_spec,
            tokenizer=tokenizer,
            device=device,
            sequences=sequences,
            high_si=high_si,
            low_si=low_si,
            r2_df=r2_df,
        )
    post_ablation = _post_ablation_t5b_a_full(approach_a, classifier)
    post_ablation["embedding_ablation"] = {
        "top_k_dims": [int(d) for d in top_dims],
        "n_dims": int(len(top_dims)),
    }
    _write_json(out_dir / "post_ablation_t5b_a.json", post_ablation)

    synth_result, synth_rows = _synthetic_boundary_full(
        model=model,
        adapter=adapter,
        tokenizer=tokenizer,
        device=device,
        sequences=sequences,
        high_heads=high_heads,
        low_heads=low_heads,
        target_per_transformed_cell=target_per_transformed_cell,
        seed=seed,
    )
    _write_json(out_dir / "synthetic_boundary_results.json", synth_result)
    synth_rows.to_parquet(out_dir / "adversarial_sequences.parquet", index=False)

    boundary_scores = pd.read_parquet(t5b_scores_path)
    prev_scores = pd.read_parquet(prev_path)
    offset_scores = _offset_group_boundary_scores(boundary_scores, prev_scores, head_groups)
    _write_json(out_dir / "offset_group_boundary_scores.json", offset_scores)

    print(f"[3P2-B][full] {model_name}: wrote artifacts to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3P2-B: Trivial feature control for boundary detection")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--mode", choices=["proxy", "full"], default="proxy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-sequences", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--top-k-dims", type=int, default=16)
    parser.add_argument("--synthetic-target-per-cell", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        default="results/experiment3_phase2/exp3p2b_trivial_feature_control",
        help="Output root directory for 3P2-B artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "proxy":
        run_proxy(model=args.model, output_root=Path(args.output_root))
        return
    run_full(
        model_name=args.model,
        output_root=Path(args.output_root),
        device=args.device,
        num_sequences=int(args.num_sequences),
        seq_len=int(args.seq_len),
        top_k_dims=int(args.top_k_dims),
        target_per_transformed_cell=int(args.synthetic_target_per_cell),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
