from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

from experiment1.norm_utils import normalize_logits_for_norm
from experiment1.shift_kernels import get_kernel_estimator
from experiment3.phase2.exp3p2b_tokenizer_audit import _feature_eval
from experiment3.phase2.exp3p2b_trivial_feature_control import _build_token_replacement_maps, _synthetic_boundary_full
from experiment3.stats_utils import one_sided_p_from_two_sided
from experiment3.theory1_si_circuits import classify_heads, compute_per_head_r2
from experiment3.theory5b_boundary_detection import run_attention_analysis
from experiment5.common import (
    build_adversarial_bpe_sequences,
    build_sequences_from_text_dataset,
    ensure_dir,
    load_cached_sequences,
    load_model_bundle,
    now_timestamp,
    safe_float,
    set_seed,
    write_json,
)
from experiment5.config import MODELS, TARGET_5A, TARGET_5B, TARGET_5C, TARGET_5D, TOKENIZER_FEATURES
from shared.attention.adapters import get_adapter


def _nan_comparison_payload(error: str) -> dict[str, Any]:
    return {
        "high_vs_low_comparison": {
            "attn_to_prev_last": {
                "cohens_d": float("nan"),
                "t_statistic": float("nan"),
                "t_p_value": float("nan"),
                "error": error,
            }
        }
    }


def _normalize_approach_c_output(raw: Any) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Normalize Approach-C output across legacy/new call signatures.

    Historically, ``run_attention_analysis`` returned ``(approach_a, (approach_c, score_df))``
    while some call-sites assumed ``approach_c`` was already a dict.  This helper
    accepts either form and returns ``(approach_c_dict, score_df_or_none)``.
    """
    if isinstance(raw, tuple):
        if len(raw) >= 2 and isinstance(raw[0], dict):
            c_dict = raw[0]
            c_df = raw[1] if isinstance(raw[1], pd.DataFrame) else None
            return c_dict, c_df
        if len(raw) == 1 and isinstance(raw[0], dict):
            return raw[0], None
    if isinstance(raw, dict):
        return raw, None
    return {"error": f"Unexpected Approach-C payload type: {type(raw).__name__}"}, None


def _extract_effect_metric(comp: dict[str, Any]) -> float:
    """Extract a finite boundary-effect scalar when possible.

    Prefer Cohen's d, but use raw high-minus-low difference as fallback for
    degenerate slices where d is undefined.
    """
    d = safe_float(comp.get("cohens_d"))
    if np.isfinite(d):
        return d
    diff = safe_float(comp.get("difference"))
    if np.isfinite(diff):
        return diff
    high = safe_float(comp.get("high_si_mean"))
    low = safe_float(comp.get("low_si_mean"))
    if np.isfinite(high) and np.isfinite(low):
        return float(high - low)
    return float("nan")


def _load_wiki_sequences(model_name: str, tokenizer, *, seq_len: int, max_sequences: int, seed: int) -> list[list[int]]:
    rows = load_cached_sequences(
        model_name=model_name,
        dataset_name="wiki40b_en_pre2019",
        seq_len=seq_len,
        max_sequences=max_sequences,
    )
    if rows:
        return rows
    try:
        rows = build_sequences_from_text_dataset(
            tokenizer=tokenizer,
            dataset_id="wikitext",
            config_name="wikitext-103-raw-v1",
            split="train",
            text_fields=("text",),
            seq_len=seq_len,
            max_sequences=max_sequences,
            max_rows=max(3000, max_sequences * 60),
            seed=seed,
        )
        if rows:
            return rows
    except Exception:
        pass
    return build_sequences_from_text_dataset(
        tokenizer=tokenizer,
        dataset_id="wiki40b",
        config_name="en",
        split="train",
        text_fields=("text",),
        seq_len=seq_len,
        max_sequences=max_sequences,
        max_rows=max(2000, max_sequences * 30),
        seed=seed,
        streaming=True,
    )


def _load_code_sequences(model_name: str, tokenizer, *, seq_len: int, max_sequences: int, seed: int) -> list[list[int]]:
    rows = load_cached_sequences(
        model_name=model_name,
        dataset_name="codesearchnet_python_snapshot",
        seq_len=seq_len,
        max_sequences=max_sequences,
    )
    if rows:
        return rows
    try:
        rows = build_sequences_from_text_dataset(
            tokenizer=tokenizer,
            dataset_id="mbpp",
            split="train",
            text_fields=("code", "text", "prompt"),
            seq_len=seq_len,
            max_sequences=max_sequences,
            max_rows=max(2000, max_sequences * 40),
            seed=seed,
        )
        if rows:
            return rows
    except Exception:
        pass
    return build_sequences_from_text_dataset(
        tokenizer=tokenizer,
        dataset_id="codeparrot/github-code",
        split="train",
        text_fields=("code", "content", "text"),
        seq_len=seq_len,
        max_sequences=max_sequences,
        max_rows=max(2000, max_sequences * 40),
        seed=seed,
        streaming=True,
    )


def _load_dialogue_sequences(tokenizer, *, seq_len: int, max_sequences: int, seed: int) -> list[list[int]]:
    # Prefer DailyDialog (small/stable), fallback to OpenAssistant streaming.
    try:
        return build_sequences_from_text_dataset(
            tokenizer=tokenizer,
            dataset_id="daily_dialog",
            split="train",
            text_fields=("dialog", "text"),
            seq_len=seq_len,
            max_sequences=max_sequences,
            max_rows=max(3000, max_sequences * 40),
            seed=seed,
        )
    except Exception:
        return build_sequences_from_text_dataset(
            tokenizer=tokenizer,
            dataset_id="OpenAssistant/oasst1",
            split="train",
            text_fields=("text", "message", "content"),
            seq_len=seq_len,
            max_sequences=max_sequences,
            max_rows=max(3000, max_sequences * 40),
            seed=seed,
            streaming=True,
        )


def _load_technical_sequences(tokenizer, *, seq_len: int, max_sequences: int, seed: int) -> list[list[int]]:
    try:
        return build_sequences_from_text_dataset(
            tokenizer=tokenizer,
            dataset_id="ccdv/arxiv-summarization",
            split="train",
            text_fields=("article", "abstract", "text"),
            seq_len=seq_len,
            max_sequences=max_sequences,
            max_rows=max(3000, max_sequences * 50),
            seed=seed,
            streaming=True,
        )
    except Exception:
        return build_sequences_from_text_dataset(
            tokenizer=tokenizer,
            dataset_id="scientific_papers",
            config_name="arxiv",
            split="train",
            text_fields=("article", "abstract", "text"),
            seq_len=seq_len,
            max_sequences=max_sequences,
            max_rows=max(3000, max_sequences * 50),
            seed=seed,
            streaming=True,
        )


def _compute_r2_from_attentions(
    *,
    model,
    tokenizer,
    model_name: str,
    seqs: list[list[int]],
    device: str,
) -> pd.DataFrame:
    spec = MODELS[model_name]
    estimator = get_kernel_estimator(spec.pe_scheme)
    rows: list[dict[str, Any]] = []
    for seq_idx, seq in enumerate(seqs):
        input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False, output_attentions=True)
        if out.attentions is None or len(out.attentions) == 0:
            continue
        for layer_idx, attn in enumerate(out.attentions):
            # [batch, heads, seq, seq]
            att = attn[0].float().detach().cpu()
            for head_idx in range(att.shape[0]):
                probs = torch.clamp(att[head_idx], min=1e-6)
                logits = torch.log(probs)
                prepared = normalize_logits_for_norm(logits, spec.norm)
                fit = estimator.fit_logits(prepared)
                rows.append(
                    {
                        "sequence_id": int(seq_idx),
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                        "r2": float(fit.r2),
                    }
                )
        del input_ids, out
    return pd.DataFrame(rows)


def _compute_r2_profile(
    *,
    model,
    tokenizer,
    model_name: str,
    sequences: list[list[int]],
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Any], list[Any], Any | None, str | None]:
    spec = MODELS[model_name]
    adapter = None
    adapter_error: str | None = None
    try:
        adapter = get_adapter(spec)
        adapter.register(model)
        r2_df = compute_per_head_r2(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            model_spec=spec,
            device=device,
            sequences=sequences,
        )
    except Exception as exc:
        adapter_error = f"{type(exc).__name__}: {exc}"
        r2_df = _compute_r2_from_attentions(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            seqs=sequences,
            device=device,
        )
        adapter = None

    high_si, low_si, mean_r2 = classify_heads(r2_df)
    if adapter is not None:
        try:
            adapter.cleanup()
        except Exception:
            pass
    return r2_df, mean_r2, high_si, low_si, adapter, adapter_error


def _r2_summary(mean_r2: pd.DataFrame) -> dict[str, Any]:
    if mean_r2.empty:
        return {
            "n_heads": 0,
            "mean_r2": float("nan"),
            "std_r2": float("nan"),
            "early_vs_late_ratio": float("nan"),
        }
    n_layers = int(mean_r2["layer"].max()) + 1
    cut = max(1, n_layers // 3)
    early = mean_r2[mean_r2["layer"] < cut]["mean_r2"].to_numpy(dtype=float)
    late = mean_r2[mean_r2["layer"] >= (n_layers - cut)]["mean_r2"].to_numpy(dtype=float)
    ratio = float(np.nanmean(early) / max(1e-8, np.nanmean(late))) if early.size and late.size else float("nan")
    return {
        "n_heads": int(len(mean_r2)),
        "mean_r2": safe_float(mean_r2["mean_r2"].mean()),
        "std_r2": safe_float(mean_r2["mean_r2"].std()),
        "min_r2": safe_float(mean_r2["mean_r2"].min()),
        "max_r2": safe_float(mean_r2["mean_r2"].max()),
        "early_vs_late_ratio": ratio,
    }


def _run_boundary_and_feature_controls(
    *,
    model,
    tokenizer,
    model_name: str,
    device: str,
    sequences: list[list[int]],
    mean_r2: pd.DataFrame,
    high_si,
    low_si,
    top_k_dims: int,
    synthetic_target_per_cell: int,
    seed: int,
) -> dict[str, Any]:
    spec = MODELS[model_name]
    adapter = get_adapter(spec)
    adapter.register(model)

    try:
        approach_a, approach_c_raw = run_attention_analysis(
            model=model,
            adapter=adapter,
            model_spec=spec,
            tokenizer=tokenizer,
            device=device,
            sequences=sequences,
            high_si=high_si,
            low_si=low_si,
            r2_df=mean_r2,
        )
        approach_c, approach_c_df = _normalize_approach_c_output(approach_c_raw)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        approach_a = _nan_comparison_payload(err)
        approach_c = {"error": err}
        approach_c_df = None
    comp = approach_a.get("high_vs_low_comparison", {}).get("attn_to_prev_last", {})
    baseline_d = _extract_effect_metric(comp)

    high_heads = [(int(h.layer), int(h.head)) for h in high_si]
    low_heads = [(int(h.layer), int(h.head)) for h in low_si]
    synth_result, _ = _synthetic_boundary_full(
        model=model,
        adapter=adapter,
        tokenizer=tokenizer,
        device=device,
        sequences=sequences,
        high_heads=high_heads,
        low_heads=low_heads,
        target_per_transformed_cell=int(synthetic_target_per_cell),
        seed=int(seed),
    )

    feature_results: list[dict[str, Any]] = []
    for feature_name in TOKENIZER_FEATURES:
        try:
            feature_results.append(
                _feature_eval(
                    model=model,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    model_spec=spec,
                    sequences=sequences,
                    high_si=high_si,
                    low_si=low_si,
                    mean_r2=mean_r2,
                    feature_name=feature_name,
                    baseline_d=baseline_d,
                    top_k_dims=top_k_dims,
                    synthetic_target_per_cell=synthetic_target_per_cell,
                    seed=seed,
                )
            )
        except Exception as exc:
            feature_results.append(
                {
                    "feature_name": feature_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    adapter.cleanup()

    return {
        "baseline": {
            "approach_a": approach_a,
            "approach_c": approach_c,
            "approach_c_score_rows": int(len(approach_c_df)) if isinstance(approach_c_df, pd.DataFrame) else 0,
            "post_ablation_d": baseline_d,
            "synthetic": synth_result,
        },
        "features": feature_results,
    }


def _top_quartile_set(mean_r2: pd.DataFrame) -> set[tuple[int, int]]:
    if mean_r2.empty:
        return set()
    sorted_df = mean_r2.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    n_sel = max(1, int(round(0.25 * len(sorted_df))))
    return {(int(r.layer), int(r.head)) for r in sorted_df.head(n_sel).itertuples()}


def run_5a(
    *,
    device: str,
    output_root: Path,
    seq_len: int,
    num_sequences: int,
    top_k_dims: int,
    synthetic_target_per_cell: int,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    out_root = ensure_dir(output_root)

    per_model_reports: dict[str, Any] = {}
    r2_profile_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []

    for model_name in TARGET_5A:
        model_dir = ensure_dir(out_root / model_name)
        model, tokenizer, load_error = load_model_bundle(model_name, device)
        if load_error is not None or model is None or tokenizer is None:
            report = {
                "timestamp": now_timestamp(),
                "model": model_name,
                "status": "skipped_unavailable",
                "error": load_error,
            }
            write_json(model_dir / "tokenizer_audit_report.json", report)
            per_model_reports[model_name] = report
            continue

        try:
            sequences = _load_wiki_sequences(model_name, tokenizer, seq_len=seq_len, max_sequences=num_sequences, seed=seed)
            if len(sequences) < max(4, num_sequences // 2):
                raise RuntimeError(f"Insufficient wiki sequences for {model_name}: {len(sequences)}")

            r2_df, mean_r2, high_si, low_si, _adapter, adapter_error = _compute_r2_profile(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                sequences=sequences,
                device=device,
            )
            r2_stats = _r2_summary(mean_r2)

            controls = _run_boundary_and_feature_controls(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                device=device,
                sequences=sequences,
                mean_r2=mean_r2,
                high_si=high_si,
                low_si=low_si,
                top_k_dims=top_k_dims,
                synthetic_target_per_cell=synthetic_target_per_cell,
                seed=seed,
            )

            mean_r2.to_parquet(model_dir / "head_r2_summary.parquet", index=False)
            r2_df.to_parquet(model_dir / "per_sequence_r2.parquet", index=False)

            report = {
                "timestamp": now_timestamp(),
                "model": model_name,
                "status": "completed",
                "n_sequences": int(len(sequences)),
                "seq_len": int(seq_len),
                "adapter_error": adapter_error,
                "r2_summary": r2_stats,
                "si_head_count": int(len(_top_quartile_set(mean_r2))),
                "boundary_d": safe_float(controls["baseline"].get("post_ablation_d")),
                "prefix_following_artifact_flag": bool(
                    controls["baseline"].get("synthetic", {}).get("prefix_following_artifact_flag", False)
                ),
                "boundary_controls": controls["baseline"],
                "feature_controls": controls["features"],
            }
            write_json(model_dir / "tokenizer_audit_report.json", report)
            per_model_reports[model_name] = report

            for r in mean_r2.itertuples():
                r2_profile_rows.append(
                    {
                        "model": model_name,
                        "layer": int(r.layer),
                        "head": int(r.head),
                        "mean_r2": float(r.mean_r2),
                    }
                )

            for feat in controls["features"]:
                dep = safe_float(feat.get("boundary_metric", {}).get("dependence_index_delta_d"))
                clf_acc = safe_float(feat.get("classifier", {}).get("classifier_accuracy"))
                matrix_rows.append(
                    {
                        "model": model_name,
                        "feature": str(feat.get("feature_name", "unknown")),
                        "dependence_index_delta_d": dep,
                        "classifier_accuracy": clf_acc,
                        "has_finite_dependence_index": bool(np.isfinite(dep)),
                        "has_finite_classifier_accuracy": bool(np.isfinite(clf_acc)),
                        "prefix_following_artifact_flag": bool(feat.get("synthetic_control", {}).get("prefix_following_artifact_flag", False)),
                    }
                )

        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    r2_df_out = pd.DataFrame(r2_profile_rows)
    if not r2_df_out.empty:
        r2_df_out.to_parquet(out_root / "cross_tokenizer_r2_profiles.parquet", index=False)

    completed_models = sorted([m for m, rep in per_model_reports.items() if str(rep.get("status")) == "completed"])
    skipped_models = sorted([m for m, rep in per_model_reports.items() if str(rep.get("status")) != "completed"])
    model_feature_cells = {
        (str(row.get("model")), str(row.get("feature")))
        for row in matrix_rows
        if row.get("model") is not None and row.get("feature") is not None
    }
    finite_dep_cells = {
        (str(row.get("model")), str(row.get("feature")))
        for row in matrix_rows
        if bool(row.get("has_finite_dependence_index", False))
    }
    finite_clf_cells = {
        (str(row.get("model")), str(row.get("feature")))
        for row in matrix_rows
        if bool(row.get("has_finite_classifier_accuracy", False))
    }
    models_with_finite_boundary_d = sorted(
        [
            m
            for m, rep in per_model_reports.items()
            if str(rep.get("status")) == "completed"
            and np.isfinite(safe_float(rep.get("boundary_d")))
        ]
    )
    required_cells = len(TARGET_5A) * len(TOKENIZER_FEATURES)
    acceptance_ok = (
        (len(completed_models) == len(TARGET_5A))
        and (len(model_feature_cells) >= required_cells)
        and (len(finite_dep_cells) >= required_cells)
        and (len(finite_clf_cells) >= required_cells)
        and (len(models_with_finite_boundary_d) == len(TARGET_5A))
    )
    overall_status = "completed" if acceptance_ok else "partial"

    matrix_report = {
        "timestamp": now_timestamp(),
        "status": overall_status,
        "n_models": int(len({row["model"] for row in matrix_rows})),
        "n_features": int(len({row["feature"] for row in matrix_rows})),
        "completed_models": completed_models,
        "skipped_models": skipped_models,
        "acceptance": {
            "required_models": int(len(TARGET_5A)),
            "required_features": int(len(TOKENIZER_FEATURES)),
            "required_model_feature_cells": int(required_cells),
            "observed_model_feature_cells": int(len(model_feature_cells)),
            "finite_dependence_index_cells": int(len(finite_dep_cells)),
            "finite_classifier_accuracy_cells": int(len(finite_clf_cells)),
            "models_with_finite_boundary_d": models_with_finite_boundary_d,
            "passed": bool(acceptance_ok),
        },
        "rows": matrix_rows,
        "per_model": per_model_reports,
    }
    write_json(out_root / "tokenizer_entanglement_matrix.json", matrix_report)
    return matrix_report


def run_5b(
    *,
    device: str,
    output_root: Path,
    seq_len: int,
    num_sequences: int,
    top_k_dims: int,
    synthetic_target_per_cell: int,
    seed: int,
) -> dict[str, Any]:
    out_root = ensure_dir(output_root)
    reports: dict[str, Any] = {}
    mean_r2_by_model: dict[str, pd.DataFrame] = {}

    for model_name in TARGET_5B:
        model, tokenizer, load_error = load_model_bundle(model_name, device)
        if load_error is not None or model is None or tokenizer is None:
            reports[model_name] = {
                "status": "skipped_unavailable",
                "error": load_error,
            }
            continue
        try:
            seqs = _load_wiki_sequences(model_name, tokenizer, seq_len=seq_len, max_sequences=num_sequences, seed=seed)
            r2_df, mean_r2, high_si, low_si, _adapter, adapter_error = _compute_r2_profile(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                sequences=seqs,
                device=device,
            )
            controls = _run_boundary_and_feature_controls(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                device=device,
                sequences=seqs,
                mean_r2=mean_r2,
                high_si=high_si,
                low_si=low_si,
                top_k_dims=top_k_dims,
                synthetic_target_per_cell=synthetic_target_per_cell,
                seed=seed,
            )

            mean_r2_by_model[model_name] = mean_r2
            reports[model_name] = {
                "status": "completed",
                "adapter_error": adapter_error,
                "r2_summary": _r2_summary(mean_r2),
                "boundary_d": safe_float(controls["baseline"].get("post_ablation_d")),
                "prefix_following_artifact_flag": bool(
                    controls["baseline"].get("synthetic", {}).get("prefix_following_artifact_flag", False)
                ),
                "feature_controls": controls["features"],
            }
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    overlap_report = {
        "timestamp": now_timestamp(),
        "status": "partial",
    }
    if "llama-2-7b" in mean_r2_by_model and "llama-3.1-8b" in mean_r2_by_model:
        a = mean_r2_by_model["llama-2-7b"].copy()
        b = mean_r2_by_model["llama-3.1-8b"].copy()
        merged = a.merge(b, on=["layer", "head"], how="inner", suffixes=("_l2", "_l31"))
        set_a = _top_quartile_set(a)
        set_b = _top_quartile_set(b)
        inter = len(set_a & set_b)
        union = max(1, len(set_a | set_b))
        rho, p = scipy_stats.spearmanr(merged["mean_r2_l2"], merged["mean_r2_l31"]) if not merged.empty else (float("nan"), float("nan"))
        overlap_report = {
            "timestamp": now_timestamp(),
            "status": "completed",
            "matched_heads": int(len(merged)),
            "top_quartile_jaccard": float(inter / union),
            "spearman_rank_correlation": safe_float(rho),
            "spearman_p_value": safe_float(p),
        }

    completed_models = sorted([m for m, rep in reports.items() if str(rep.get("status")) == "completed"])
    skipped_models = sorted([m for m, rep in reports.items() if str(rep.get("status")) != "completed"])
    acceptance_ok = (len(completed_models) == len(TARGET_5B)) and (str(overlap_report.get("status")) == "completed")
    overall_status = "completed" if acceptance_ok else "partial"

    write_json(out_root / "head_identity_overlap.json", overlap_report)
    write_json(
        out_root / "llama_tokenizer_comparison.json",
        {
            "timestamp": now_timestamp(),
            "status": overall_status,
            "reports": reports,
            "acceptance": {
                "required_models": int(len(TARGET_5B)),
                "completed_models": completed_models,
                "skipped_models": skipped_models,
                "overlap_status": str(overlap_report.get("status")),
                "passed": bool(acceptance_ok),
            },
        },
    )
    return {
        "status": overall_status,
        "reports": reports,
        "overlap": overlap_report,
        "acceptance": {
            "required_models": int(len(TARGET_5B)),
            "completed_models": completed_models,
            "skipped_models": skipped_models,
            "overlap_status": str(overlap_report.get("status")),
            "passed": bool(acceptance_ok),
        },
    }


def _capture_logits_for_sequence(model, adapter, seq: list[int], device: str) -> torch.Tensor | None:
    ids = torch.tensor([seq], dtype=torch.long, device=device)
    with torch.inference_mode():
        cap = adapter.capture(
            model,
            input_ids=ids,
            include_logits=True,
            return_token_logits=False,
            capture_attention=True,
            output_device="cpu",
        )
    if cap.logits is None:
        return None
    return cap.logits


def _head_prev_attention(logits: torch.Tensor, heads: list[tuple[int, int]], pos: int) -> float:
    if pos <= 0:
        return float("nan")
    vals: list[float] = []
    for layer, head in heads:
        if layer >= logits.shape[0] or head >= logits.shape[1]:
            continue
        l = logits[layer, head].float().clone()
        causal_mask = torch.triu(torch.ones(l.shape, dtype=torch.bool, device=l.device), diagonal=1)
        l = l.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(l, dim=-1)
        vals.append(float(attn[pos, pos - 1].item()))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _local_r2_mean(logits: torch.Tensor, model_name: str, heads: list[tuple[int, int]], pos: int, window: int = 48) -> float:
    spec = MODELS[model_name]
    estimator = get_kernel_estimator(spec.pe_scheme)
    s = max(0, int(pos - window))
    e = min(int(logits.shape[-1]), int(pos + window))
    if e - s < 8:
        return float("nan")
    vals: list[float] = []
    for layer, head in heads[:16]:
        if layer >= logits.shape[0] or head >= logits.shape[1]:
            continue
        mat = logits[layer, head, s:e, s:e]
        prepared = normalize_logits_for_norm(mat, spec.norm)
        fit = estimator.fit_logits(prepared)
        vals.append(float(fit.r2))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _is_word_initial_token(token_text: str) -> bool:
    tok = str(token_text or "")
    return tok.startswith("Ġ") or tok.startswith("▁") or tok.startswith(" ")


def _apply_perturbation(
    *,
    seq: list[int],
    pos: int,
    perturbation: str,
    tokenizer,
    to_prefix: dict[int, int],
    to_noprefix: dict[int, int],
    seed: int,
) -> list[int] | None:
    if pos <= 0 or pos >= len(seq):
        return None
    new_seq = list(seq)
    tok = int(new_seq[pos])

    if perturbation == "fake_boundary_with_prefix":
        rep = to_prefix.get(tok)
        if rep is None:
            return None
        new_seq[pos] = int(rep)
        return new_seq

    if perturbation == "real_boundary_no_prefix":
        rep = to_noprefix.get(tok)
        if rep is None:
            return None
        new_seq[pos] = int(rep)
        return new_seq

    if perturbation == "morpheme_internal_break":
        # Only apply at continuation subword positions (tokens that are NOT
        # word-initial).  Unlike fake_boundary_with_prefix, perform an actual
        # intra-token split (left + right) and re-tokenize the split text.
        tok_str = tokenizer.convert_ids_to_tokens([tok])[0] or ""
        if _is_word_initial_token(tok_str):
            return None
        stripped = tok_str.lstrip("Ġ▁ ").strip()
        if len(stripped) < 3:
            return None
        split = max(1, min(len(stripped) - 1, len(stripped) // 2))
        left = stripped[:split]
        right = stripped[split:]
        if not left or not right:
            return None
        rep_ids = tokenizer.encode(left + " " + right, add_special_tokens=False)
        if not rep_ids:
            return None
        if len(rep_ids) == 1 and int(rep_ids[0]) == tok:
            return None
        merged = new_seq[:pos] + [int(x) for x in rep_ids] + new_seq[pos + 1 :]
        if len(merged) < len(seq):
            merged.extend(seq[: len(seq) - len(merged)])
        return merged[: len(seq)]

    if perturbation == "character_decompose":
        tok_str = tokenizer.convert_ids_to_tokens([tok])[0] or ""
        stripped = tok_str.lstrip("Ġ▁")
        if len(stripped) < 3:
            return None
        rep_ids = tokenizer.encode(" ".join(list(stripped)), add_special_tokens=False)
        if not rep_ids:
            return None
        merged = new_seq[:pos] + [int(x) for x in rep_ids] + new_seq[pos + 1 :]
        if len(merged) < len(seq):
            merged.extend(seq[: len(seq) - len(merged)])
        return merged[: len(seq)]

    if perturbation == "merge_across_boundary":
        rep = to_noprefix.get(tok)
        if rep is not None:
            new_seq[pos] = int(rep)
            return new_seq
        if pos <= 0:
            return None
        txt = tokenizer.decode([new_seq[pos - 1], new_seq[pos]], skip_special_tokens=True).replace(" ", "")
        rep_ids = tokenizer.encode(txt, add_special_tokens=False)
        if not rep_ids:
            return None
        merged = new_seq[: pos - 1] + [int(rep_ids[0])] + new_seq[pos + 1 :]
        if len(merged) < len(seq):
            merged.extend(seq[: len(seq) - len(merged)])
        return merged[: len(seq)]

    if perturbation == "random_resegment":
        rng = random.Random(seed + pos)
        span = tokenizer.decode(seq[max(0, pos - 1) : pos + 2], skip_special_tokens=True)
        if len(span) < 2:
            return None
        cut = rng.randint(1, len(span) - 1)
        txt = span[:cut] + " " + span[cut:]
        rep_ids = tokenizer.encode(txt, add_special_tokens=False)
        if not rep_ids:
            return None
        merged = new_seq[: max(0, pos - 1)] + [int(x) for x in rep_ids] + new_seq[pos + 2 :]
        if len(merged) < len(seq):
            merged.extend(seq[: len(seq) - len(merged)])
        return merged[: len(seq)]

    return None


def run_5c(
    *,
    device: str,
    output_root: Path,
    seq_len: int,
    num_sequences: int,
    positions_per_type: int,
    seed: int,
) -> dict[str, Any]:
    out_root = ensure_dir(output_root)
    perturbations = [
        "fake_boundary_with_prefix",
        "real_boundary_no_prefix",
        "morpheme_internal_break",
        "character_decompose",
        "merge_across_boundary",
        "random_resegment",
    ]

    all_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []

    for model_name in TARGET_5C:
        model, tokenizer, load_error = load_model_bundle(model_name, device)
        if load_error is not None or model is None or tokenizer is None:
            for p in perturbations:
                matrix_rows.append({
                    "model": model_name,
                    "perturbation": p,
                    "status": "skipped_unavailable",
                    "error": load_error,
                })
            continue

        try:
            sequences = _load_wiki_sequences(model_name, tokenizer, seq_len=seq_len, max_sequences=num_sequences, seed=seed)
            r2_df, mean_r2, high_si, low_si, _adapter, _err = _compute_r2_profile(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                sequences=sequences,
                device=device,
            )
            high_heads = [(int(h.layer), int(h.head)) for h in high_si]

            adapter = get_adapter(MODELS[model_name])
            adapter.register(model)
            to_prefix, to_noprefix = _build_token_replacement_maps(tokenizer)

            for perturbation in perturbations:
                deltas: list[float] = []
                r2_deltas: list[float] = []
                n = 0
                for seq_idx, seq in enumerate(sequences):
                    if n >= positions_per_type:
                        break
                    base_logits = _capture_logits_for_sequence(model, adapter, seq, device)
                    if base_logits is None:
                        continue
                    for pos in range(2, len(seq) - 2):
                        if n >= positions_per_type:
                            break
                        pert = _apply_perturbation(
                            seq=seq,
                            pos=pos,
                            perturbation=perturbation,
                            tokenizer=tokenizer,
                            to_prefix=to_prefix,
                            to_noprefix=to_noprefix,
                            seed=seed + seq_idx,
                        )
                        if pert is None:
                            continue
                        pert_logits = _capture_logits_for_sequence(model, adapter, pert, device)
                        if pert_logits is None:
                            continue

                        base_prev = _head_prev_attention(base_logits, high_heads, pos)
                        pert_prev = _head_prev_attention(pert_logits, high_heads, pos)
                        base_r2 = _local_r2_mean(base_logits, model_name, high_heads, pos)
                        pert_r2 = _local_r2_mean(pert_logits, model_name, high_heads, pos)

                        if np.isfinite(base_prev) and np.isfinite(pert_prev):
                            deltas.append(float(pert_prev - base_prev))
                        if np.isfinite(base_r2) and np.isfinite(pert_r2):
                            r2_deltas.append(float(pert_r2 - base_r2))

                        all_rows.append(
                            {
                                "model": model_name,
                                "perturbation": perturbation,
                                "sequence_idx": int(seq_idx),
                                "position": int(pos),
                                "control_high_prev_attn": safe_float(base_prev),
                                "perturbed_high_prev_attn": safe_float(pert_prev),
                                "delta_high_prev_attn": safe_float(pert_prev - base_prev),
                                "control_local_r2": safe_float(base_r2),
                                "perturbed_local_r2": safe_float(pert_r2),
                                "delta_local_r2": safe_float(pert_r2 - base_r2),
                            }
                        )
                        n += 1

                arr = np.asarray(deltas, dtype=float)
                arr = arr[np.isfinite(arr)]
                arr_r2 = np.asarray(r2_deltas, dtype=float)
                arr_r2 = arr_r2[np.isfinite(arr_r2)]
                t_stat, p_two = scipy_stats.ttest_1samp(arr, popmean=0.0) if arr.size >= 2 else (float("nan"), float("nan"))
                mean_delta = float(arr.mean()) if arr.size else float("nan")
                ci_low = float(np.quantile(arr, 0.025)) if arr.size >= 4 else float("nan")
                ci_high = float(np.quantile(arr, 0.975)) if arr.size >= 4 else float("nan")

                matrix_rows.append(
                    {
                        "model": model_name,
                        "perturbation": perturbation,
                        "n_positions": int(arr.size),
                        "effect_delta_high_prev_attn": mean_delta,
                        "effect_ci95": [ci_low, ci_high],
                        "effect_p_two_sided": safe_float(p_two),
                        "effect_p_one_sided_gt0": safe_float(
                            one_sided_p_from_two_sided(safe_float(t_stat), safe_float(p_two), alternative="greater")
                        ),
                        "delta_local_r2_mean": safe_float(arr_r2.mean()) if arr_r2.size else float("nan"),
                        "prefix_following_flag_shift": bool(mean_delta > 0.0 and np.isfinite(mean_delta)),
                        "status": "completed",
                    }
                )

            adapter.cleanup()

        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows_df = pd.DataFrame(all_rows)
    if not rows_df.empty:
        rows_df.to_parquet(out_root / "perturbation_results.parquet", index=False)

    duplicate_pairs: list[dict[str, Any]] = []
    if not rows_df.empty:
        for model_name in TARGET_5C:
            fake = rows_df[
                (rows_df["model"] == model_name)
                & (rows_df["perturbation"] == "fake_boundary_with_prefix")
            ][["sequence_idx", "position", "delta_high_prev_attn"]].sort_values(["sequence_idx", "position"]).reset_index(drop=True)
            morph = rows_df[
                (rows_df["model"] == model_name)
                & (rows_df["perturbation"] == "morpheme_internal_break")
            ][["sequence_idx", "position", "delta_high_prev_attn"]].sort_values(["sequence_idx", "position"]).reset_index(drop=True)
            is_duplicate = False
            if len(fake) > 0 and len(fake) == len(morph):
                same_pairs = fake[["sequence_idx", "position"]].equals(morph[["sequence_idx", "position"]])
                same_deltas = np.allclose(
                    fake["delta_high_prev_attn"].to_numpy(dtype=float),
                    morph["delta_high_prev_attn"].to_numpy(dtype=float),
                    equal_nan=True,
                )
                is_duplicate = bool(same_pairs and same_deltas)
            duplicate_pairs.append(
                {
                    "model": model_name,
                    "fake_vs_morpheme_duplicate": bool(is_duplicate),
                }
            )

    expected_cells = len(TARGET_5C) * len(perturbations)
    completed_cells = [row for row in matrix_rows if str(row.get("status")) == "completed"]
    covered_cells = [
        row
        for row in completed_cells
        if int(row.get("n_positions", 0) or 0) >= int(positions_per_type)
    ]
    has_duplicate_morpheme = any(bool(x.get("fake_vs_morpheme_duplicate", False)) for x in duplicate_pairs)
    acceptance_ok = (len(covered_cells) == expected_cells) and (not has_duplicate_morpheme)
    overall_status = "completed" if acceptance_ok else "partial"

    matrix_report = {
        "timestamp": now_timestamp(),
        "status": overall_status,
        "acceptance": {
            "required_cells": int(expected_cells),
            "completed_cells": int(len(completed_cells)),
            "coverage_cells_at_target": int(len(covered_cells)),
            "target_positions_per_cell": int(positions_per_type),
            "fake_vs_morpheme_duplicate_detected": bool(has_duplicate_morpheme),
            "passed": bool(acceptance_ok),
        },
        "quality_checks": {
            "fake_vs_morpheme_duplicate": duplicate_pairs,
        },
        "rows": matrix_rows,
    }
    write_json(out_root / "perturbation_sensitivity_matrix.json", matrix_report)
    return matrix_report


def _evaluate_condition_metrics(
    *,
    model,
    tokenizer,
    model_name: str,
    device: str,
    sequences: list[list[int]],
    seed: int,
) -> dict[str, Any]:
    r2_df, mean_r2, high_si, low_si, _adapter, _err = _compute_r2_profile(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        sequences=sequences,
        device=device,
    )
    spec = MODELS[model_name]
    adapter = get_adapter(spec)
    adapter.register(model)
    try:
        approach_a, approach_c_raw = run_attention_analysis(
            model=model,
            adapter=adapter,
            model_spec=spec,
            tokenizer=tokenizer,
            device=device,
            sequences=sequences,
            high_si=high_si,
            low_si=low_si,
            r2_df=mean_r2,
        )
        approach_c, _approach_c_df = _normalize_approach_c_output(approach_c_raw)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        approach_a = _nan_comparison_payload(err)
        approach_c = {"error": err}
    adapter.cleanup()

    comp = approach_a.get("high_vs_low_comparison", {}).get("attn_to_prev_last", {})
    per_seq_r2 = r2_df.groupby("sequence_id", as_index=False)["r2"].mean()
    mean_r2_val = safe_float(per_seq_r2["r2"].mean()) if not per_seq_r2.empty else float("nan")
    ci_low = safe_float(np.quantile(per_seq_r2["r2"], 0.025)) if len(per_seq_r2) >= 4 else float("nan")
    ci_high = safe_float(np.quantile(per_seq_r2["r2"], 0.975)) if len(per_seq_r2) >= 4 else float("nan")

    return {
        "n_sequences": int(len(sequences)),
        "mean_r2": mean_r2_val,
        "mean_r2_ci95": [ci_low, ci_high],
        "boundary_d": _extract_effect_metric(comp),
        "boundary_t_p": safe_float(comp.get("t_p_value")),
        "spearman_rho": safe_float(approach_c.get("correlation", {}).get("spearman_rho")),
    }


def run_5d(
    *,
    device: str,
    output_root: Path,
    seq_len: int,
    num_sequences: int,
    seed: int,
) -> dict[str, Any]:
    out_root = ensure_dir(output_root)
    all_rows: list[dict[str, Any]] = []

    per_model_condition: dict[tuple[str, str], dict[str, Any]] = {}
    for model_name in TARGET_5D:
        model, tokenizer, load_error = load_model_bundle(model_name, device)
        if load_error is not None or model is None or tokenizer is None:
            for cond in ("wiki", "code", "dialogue", "technical_scientific", "adversarial_bpe"):
                all_rows.append(
                    {
                        "model": model_name,
                        "condition": cond,
                        "status": "skipped_unavailable",
                        "error": load_error,
                    }
                )
            continue
        try:
            wiki = _load_wiki_sequences(model_name, tokenizer, seq_len=seq_len, max_sequences=num_sequences, seed=seed)
            code = _load_code_sequences(model_name, tokenizer, seq_len=seq_len, max_sequences=num_sequences, seed=seed + 1)
            dialogue = _load_dialogue_sequences(tokenizer, seq_len=seq_len, max_sequences=num_sequences, seed=seed + 2)
            technical = _load_technical_sequences(tokenizer, seq_len=seq_len, max_sequences=num_sequences, seed=seed + 3)
            adv = build_adversarial_bpe_sequences(
                tokenizer=tokenizer,
                base_sequences=wiki,
                max_sequences=num_sequences,
                seq_len=seq_len,
                seed=seed + 4,
            )

            cond_map = {
                "wiki": wiki,
                "code": code,
                "dialogue": dialogue,
                "technical_scientific": technical,
                "adversarial_bpe": adv,
            }
            metrics: dict[str, dict[str, Any]] = {}
            for cond, seqs in cond_map.items():
                seqs_use = seqs[:num_sequences]
                if len(seqs_use) < max(8, num_sequences // 3):
                    all_rows.append(
                        {
                            "model": model_name,
                            "condition": cond,
                            "status": "insufficient_sequences",
                            "n_sequences": int(len(seqs_use)),
                        }
                    )
                    continue
                m = _evaluate_condition_metrics(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    device=device,
                    sequences=seqs_use,
                    seed=seed,
                )
                metrics[cond] = m
                per_model_condition[(model_name, cond)] = m
                # Free GPU memory between conditions to avoid OOM during
                # repeated R²-profiling passes on large models.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            base = metrics.get("wiki", {})
            base_r2 = safe_float(base.get("mean_r2"))
            base_d = safe_float(base.get("boundary_d"))
            for cond, m in metrics.items():
                all_rows.append(
                    {
                        "model": model_name,
                        "condition": cond,
                        "status": "completed",
                        "n_sequences": int(m.get("n_sequences", 0)),
                        "mean_r2": safe_float(m.get("mean_r2")),
                        "mean_r2_ci95_low": safe_float((m.get("mean_r2_ci95") or [float("nan"), float("nan")])[0]),
                        "mean_r2_ci95_high": safe_float((m.get("mean_r2_ci95") or [float("nan"), float("nan")])[1]),
                        "boundary_d": safe_float(m.get("boundary_d")),
                        "delta_r2_vs_wiki": safe_float(safe_float(m.get("mean_r2")) - base_r2),
                        "delta_boundary_d_vs_wiki": safe_float(safe_float(m.get("boundary_d")) - base_d),
                    }
                )

        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_parquet(out_root / "shift_results.parquet", index=False)

    interaction_rows: list[dict[str, Any]] = []
    for cond in ["code", "dialogue", "technical_scientific", "adversarial_bpe"]:
        l = per_model_condition.get(("llama-3.1-8b", cond))
        o = per_model_condition.get(("olmo-2-7b", cond))
        l_base = per_model_condition.get(("llama-3.1-8b", "wiki"))
        o_base = per_model_condition.get(("olmo-2-7b", "wiki"))
        if not l or not o or not l_base or not o_base:
            continue
        l_delta = safe_float(l.get("mean_r2")) - safe_float(l_base.get("mean_r2"))
        o_delta = safe_float(o.get("mean_r2")) - safe_float(o_base.get("mean_r2"))
        interaction_rows.append(
            {
                "condition": cond,
                "llama_delta_r2": safe_float(l_delta),
                "olmo_delta_r2": safe_float(o_delta),
                "interaction_delta_r2": safe_float(l_delta - o_delta),
                "llama_boundary_delta": safe_float(safe_float(l.get("boundary_d")) - safe_float(l_base.get("boundary_d"))),
                "olmo_boundary_delta": safe_float(safe_float(o.get("boundary_d")) - safe_float(o_base.get("boundary_d"))),
            }
        )

    conditions = ["wiki", "code", "dialogue", "technical_scientific", "adversarial_bpe"]
    required_cells = len(TARGET_5D) * len(conditions)
    completed_rows = [
        row
        for row in all_rows
        if str(row.get("status")) == "completed" and int(row.get("n_sequences", 0) or 0) >= int(num_sequences)
    ]
    interaction_complete = len(interaction_rows) == 4
    acceptance_ok = (len(completed_rows) == required_cells) and interaction_complete

    report = {
        "timestamp": now_timestamp(),
        "status": "completed" if acceptance_ok else "partial",
        "acceptance": {
            "required_cells": int(required_cells),
            "completed_cells_at_target_sequences": int(len(completed_rows)),
            "required_sequences_per_cell": int(num_sequences),
            "interaction_rows_required": 4,
            "interaction_rows_observed": int(len(interaction_rows)),
            "passed": bool(acceptance_ok),
        },
        "rows": all_rows,
        "interaction_test": interaction_rows,
    }
    write_json(out_root / "distribution_shift_fragility.json", report)
    return report
