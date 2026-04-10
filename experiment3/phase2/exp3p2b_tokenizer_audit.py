#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import string
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    LogisticRegression = None

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment3.stats_utils import holm_adjust, one_sided_p_from_two_sided  # noqa: E402
from experiment3.theory5b_boundary_detection import MODELS, load_wiki_sequences, parse_head_list, run_attention_analysis  # noqa: E402
from experiment3.phase2.exp3p2b_trivial_feature_control import (  # noqa: E402
    _embedding_dimension_mean_ablation,
    _synthetic_boundary_full,
)
from shared.attention.adapters import get_adapter  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402


TARGET_MODELS = ("llama-3.1-8b", "olmo-2-7b")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _token_has_prefix(token_str: str) -> bool:
    return token_str.startswith("\u0120") or token_str.startswith("\u2581")


def _strip_markers(tok: str) -> str:
    if tok is None:
        return ""
    out = str(tok)
    while out.startswith("\u0120") or out.startswith("\u2581"):
        out = out[1:]
    return out


def _is_punct_like(tok: str) -> bool:
    s = _strip_markers(tok)
    if not s:
        return False
    return all((ch in string.punctuation) for ch in s)


def _load_head_groups_and_r2(model_name: str) -> tuple[list[Any], list[Any], pd.DataFrame]:
    t1_root = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name
    groups = _load_json(t1_root / "head_groups.json")
    r2 = pd.read_parquet(t1_root / "per_sequence_r2.parquet")
    mean_r2 = r2.groupby(["layer", "head"], as_index=False)["r2"].mean().rename(columns={"r2": "mean_r2"})
    return parse_head_list(groups["high_si"]), parse_head_list(groups["low_si"]), mean_r2


def _feature_labels(
    *,
    feature_name: str,
    token_ids: np.ndarray,
    token_strs: dict[int, str],
    adjacency_rate: dict[int, float],
    length_median: float,
) -> np.ndarray:
    labels: list[int] = []
    for tid in token_ids:
        tok = token_strs.get(int(tid), "")
        stripped = _strip_markers(tok)
        if feature_name == "space_prefix":
            labels.append(1 if _token_has_prefix(tok) else 0)
        elif feature_name == "capitalization_marker":
            labels.append(1 if (len(stripped) > 0 and stripped[0].isalpha() and stripped[0].isupper()) else 0)
        elif feature_name == "punctuation_adjacency":
            labels.append(1 if adjacency_rate.get(int(tid), 0.0) >= 0.5 else 0)
        elif feature_name == "token_length_bucket":
            labels.append(1 if len(stripped) >= float(length_median) else 0)
        else:
            raise ValueError(f"Unknown feature: {feature_name}")
    return np.asarray(labels, dtype=np.int32)


def _fit_embedding_feature_classifier(
    *,
    model,
    token_ids: np.ndarray,
    labels: np.ndarray,
    top_k_dims: int,
    feature_name: str,
) -> tuple[dict[str, Any], list[int], np.ndarray]:
    if len(token_ids) < 8:
        raise RuntimeError(f"{feature_name}: not enough token ids ({len(token_ids)})")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise RuntimeError(f"{feature_name}: single-class labels; cannot fit classifier")

    emb = model.get_input_embeddings().weight
    ids_t = torch.tensor([int(x) for x in token_ids.tolist()], device=emb.device, dtype=torch.long)
    with torch.inference_mode():
        x = emb[ids_t].float().cpu().numpy()

    if LogisticRegression is None:
        pos = x[labels == 1].mean(axis=0)
        neg = x[labels == 0].mean(axis=0)
        coef = pos - neg
        logits = x @ coef
        thresh = float(np.median(logits))
        pred = (logits >= thresh).astype(np.int32)
        method = "embedding_linear_mean_diff_fallback"
    else:
        clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear", random_state=0)
        clf.fit(x, labels)
        pred = clf.predict(x).astype(np.int32)
        coef = clf.coef_[0]
        method = "embedding_logistic_regression"

    acc = float(np.mean(pred == labels))
    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    specificity = float(tn / max(1, tn + fp))

    top_k = max(1, int(top_k_dims))
    ranked = np.argsort(np.abs(coef))[::-1]
    top_dims = [int(i) for i in ranked[:top_k].tolist()]

    dim_idx = torch.tensor(top_dims, device=emb.device, dtype=torch.long)
    with torch.inference_mode():
        mean_vals = emb.index_select(dim=1, index=dim_idx).float().mean(dim=0).cpu().numpy()

    report = {
        "feature_name": feature_name,
        "method": method,
        "classifier_accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "n_samples": int(len(token_ids)),
        "n_positive": int(labels.sum()),
        "n_negative": int((1 - labels).sum()),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "top_k_dims": top_dims,
        "top_k_abs_weights": [float(abs(coef[i])) for i in top_dims],
    }
    return report, top_dims, mean_vals


def _build_adjacency_rate(tokenizer, sequences: list[list[int]]) -> dict[int, float]:
    punct_adj_counts: dict[int, int] = {}
    total_counts: dict[int, int] = {}
    for seq in sequences:
        toks = tokenizer.convert_ids_to_tokens([int(t) for t in seq])
        for i, tid in enumerate(seq):
            tok = toks[i] or ""
            prev_is_punct = _is_punct_like(toks[i - 1] or "") if i > 0 else False
            next_is_punct = _is_punct_like(toks[i + 1] or "") if i + 1 < len(seq) else False
            is_adj = bool(prev_is_punct or next_is_punct)
            key = int(tid)
            total_counts[key] = total_counts.get(key, 0) + 1
            punct_adj_counts[key] = punct_adj_counts.get(key, 0) + (1 if is_adj else 0)
    out: dict[int, float] = {}
    for tid, n in total_counts.items():
        out[int(tid)] = float(punct_adj_counts.get(int(tid), 0) / max(1, n))
    return out


def _feature_eval(
    *,
    model,
    adapter,
    tokenizer,
    model_name: str,
    model_spec,
    sequences: list[list[int]],
    high_si,
    low_si,
    mean_r2: pd.DataFrame,
    feature_name: str,
    baseline_d: float,
    top_k_dims: int,
    synthetic_target_per_cell: int,
    seed: int,
) -> dict[str, Any]:
    model_device = str(next(model.parameters()).device)
    unique_token_ids = np.unique(np.asarray([int(t) for seq in sequences for t in seq], dtype=np.int64))
    token_strs = {int(t): (s or "") for t, s in zip(unique_token_ids.tolist(), tokenizer.convert_ids_to_tokens(unique_token_ids.tolist()))}
    adjacency_rate = _build_adjacency_rate(tokenizer, sequences)
    lens = [len(_strip_markers(token_strs[int(t)])) for t in unique_token_ids.tolist()]
    length_median = float(np.median(np.asarray(lens, dtype=np.float64))) if lens else 1.0

    labels = _feature_labels(
        feature_name=feature_name,
        token_ids=unique_token_ids,
        token_strs=token_strs,
        adjacency_rate=adjacency_rate,
        length_median=length_median,
    )

    clf_report, dims, mean_vals = _fit_embedding_feature_classifier(
        model=model,
        token_ids=unique_token_ids,
        labels=labels,
        top_k_dims=int(top_k_dims),
        feature_name=feature_name,
    )

    with _embedding_dimension_mean_ablation(model, dims, mean_vals):
        approach_a, _ = run_attention_analysis(
            model=model,
            adapter=adapter,
            model_spec=model_spec,
            tokenizer=tokenizer,
            device=model_device,
            sequences=sequences,
            high_si=high_si,
            low_si=low_si,
            r2_df=mean_r2,
        )
        high_heads = [(int(h.layer), int(h.head)) for h in high_si]
        low_heads = [(int(h.layer), int(h.head)) for h in low_si]
        synth_result, _synth_rows = _synthetic_boundary_full(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            device=model_device,
            sequences=sequences,
            high_heads=high_heads,
            low_heads=low_heads,
            target_per_transformed_cell=int(synthetic_target_per_cell),
            seed=int(seed),
        )

    comp = approach_a.get("high_vs_low_comparison", {}).get("attn_to_prev_last", {})
    d_post = _safe_float(comp.get("cohens_d"))
    t_stat = _safe_float(comp.get("t_statistic"))
    p_two = _safe_float(comp.get("t_p_value"))
    p_one = _safe_float(one_sided_p_from_two_sided(t_stat, p_two, alternative="greater"))
    dep_idx = _safe_float(baseline_d - d_post)

    return {
        "feature_name": feature_name,
        "classifier": clf_report,
        "boundary_metric": {
            "baseline_d": _safe_float(baseline_d),
            "post_ablation_d": _safe_float(d_post),
            "dependence_index_delta_d": _safe_float(dep_idx),
            "t_statistic": _safe_float(t_stat),
            "p_two_sided": _safe_float(p_two),
            "p_one_sided_high_gt_low": _safe_float(p_one),
        },
        "synthetic_control": {
            "fake_minus_real_with_prefix": _safe_float(synth_result.get("comparisons", {}).get("fake_minus_real_with_prefix")),
            "prefix_following_artifact_flag": bool(synth_result.get("prefix_following_artifact_flag", False)),
            "prefix_following_assessment": synth_result.get("prefix_following_assessment", {}),
        },
    }


def run_model(
    *,
    model_name: str,
    device: str,
    num_sequences: int,
    seq_len: int,
    top_k_dims: int,
    synthetic_target_per_cell: int,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    high_si, low_si, mean_r2 = _load_head_groups_and_r2(model_name)

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)
    adapter = get_adapter(model_spec)
    adapter.register(model)

    sequences = load_wiki_sequences(model_name=model_name, max_sequences=max(8, int(num_sequences)), seq_len=max(128, int(seq_len)))
    sequences = sequences[: int(num_sequences)]

    b_root = ROOT / "results" / "experiment3_phase2" / "exp3p2b_trivial_feature_control" / model_name
    b_post = _load_json(b_root / "post_ablation_t5b_a.json")
    b_synth = _load_json(b_root / "synthetic_boundary_results.json")
    baseline_d = _safe_float(b_post.get("high_vs_low_attn_to_prev_last", {}).get("cohens_d"))

    features = [
        "space_prefix",
        "capitalization_marker",
        "punctuation_adjacency",
        "token_length_bucket",
    ]
    rows: list[dict[str, Any]] = []
    for feat in features:
        rep = _feature_eval(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            model_name=model_name,
            model_spec=model_spec,
            sequences=sequences,
            high_si=high_si,
            low_si=low_si,
            mean_r2=mean_r2,
            feature_name=feat,
            baseline_d=baseline_d,
            top_k_dims=top_k_dims,
            synthetic_target_per_cell=synthetic_target_per_cell,
            seed=seed,
        )
        rows.append(rep)
        _write_json(out_dir / f"feature_{feat}.json", rep)

    pvals = {
        r["feature_name"]: _safe_float(r.get("boundary_metric", {}).get("p_one_sided_high_gt_low"))
        for r in rows
    }
    pvals = {k: v for k, v in pvals.items() if np.isfinite(v)}
    p_adj = holm_adjust(pvals)

    summary_rows: list[dict[str, Any]] = []
    for r in rows:
        feat = str(r["feature_name"])
        p1 = _safe_float(r.get("boundary_metric", {}).get("p_one_sided_high_gt_low"))
        p1h = _safe_float(p_adj.get(feat)) if feat in p_adj else float("nan")
        d_post = _safe_float(r.get("boundary_metric", {}).get("post_ablation_d"))
        dep_idx = _safe_float(r.get("boundary_metric", {}).get("dependence_index_delta_d"))
        summary_rows.append(
            {
                "feature": feat,
                "dependence_index_delta_d": dep_idx,
                "post_ablation_d": d_post,
                "p_one_sided_high_gt_low": p1,
                "p_one_sided_high_gt_low_holm_features": p1h,
                "prefix_following_artifact_flag": bool(r.get("synthetic_control", {}).get("prefix_following_artifact_flag", False)),
                "classifier_accuracy": _safe_float(r.get("classifier", {}).get("classifier_accuracy")),
                "collapse_flag": bool((not np.isfinite(d_post)) or (d_post < 0.20) or (np.isfinite(p1h) and p1h >= 0.05)),
            }
        )

    dep = pd.DataFrame(summary_rows)
    dep.to_parquet(out_dir / "tokenizer_feature_dependence.parquet", index=False)

    robust = bool((~dep["collapse_flag"]).all()) if not dep.empty else False
    dominant = None
    if not dep.empty and np.isfinite(dep["dependence_index_delta_d"]).any():
        dominant = str(dep.sort_values("dependence_index_delta_d", ascending=False).iloc[0]["feature"])

    verdict = {
        "baseline_prefix_following_artifact_flag": bool(b_synth.get("prefix_following_artifact_flag", False)),
        "olmo_robust_to_feature_battery": bool(model_name == "olmo-2-7b" and robust),
        "llama_tokenizer_entangled": bool(model_name == "llama-3.1-8b" and (not robust)),
        "dominant_dependence_feature": dominant,
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "settings": {
            "num_sequences": int(len(sequences)),
            "seq_len": int(seq_len),
            "top_k_dims": int(top_k_dims),
            "synthetic_target_per_cell": int(synthetic_target_per_cell),
            "seed": int(seed),
        },
        "baseline": {
            "path": str(b_root),
            "post_ablation_d": _safe_float(baseline_d),
            "prefix_following_artifact_flag": bool(b_synth.get("prefix_following_artifact_flag", False)),
        },
        "feature_summaries": summary_rows,
        "verdict": verdict,
        "limitations": [
            "Tokenizer features are embedding-dimension surrogates and do not isolate all contextual confounds.",
            "Punctuation adjacency uses corpus-derived token-level rates rather than direct contextual interventions.",
        ],
    }
    _write_json(out_dir / "tokenizer_audit_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3P2-B tokenizer-entanglement feature audit")
    p.add_argument("--model", choices=["all", *TARGET_MODELS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1")
    p.add_argument("--num-sequences", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--top-k-dims", type=int, default=16)
    p.add_argument("--synthetic-target-per-cell", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default="results/experiment3_phase2/exp3p2b_tokenizer_audit")
    return p.parse_args()


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        model, dev = tok.split(":", 1)
        out[model.strip()] = dev.strip()
    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    models = list(TARGET_MODELS) if args.model == "all" else [str(args.model)]
    device_map = _parse_device_map(args.device_map) if args.model == "all" else {}

    all_reports: dict[str, Any] = {}
    for m in models:
        dev = device_map.get(m, str(args.device))
        print(f"[tokenizer-audit] model={m} device={dev}", flush=True)
        rep = run_model(
            model_name=m,
            device=dev,
            num_sequences=max(8, int(args.num_sequences)),
            seq_len=max(128, int(args.seq_len)),
            top_k_dims=max(1, int(args.top_k_dims)),
            synthetic_target_per_cell=max(64, int(args.synthetic_target_per_cell)),
            seed=int(args.seed),
            output_root=output_root,
        )
        all_reports[m] = rep

    aggregate = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": all_reports,
        "verdict": {
            "olmo_robust": bool(all_reports.get("olmo-2-7b", {}).get("verdict", {}).get("olmo_robust_to_feature_battery", False)),
            "llama_entangled": bool(all_reports.get("llama-3.1-8b", {}).get("verdict", {}).get("llama_tokenizer_entangled", False)),
        },
    }
    _write_json(output_root / "tokenizer_audit_report.json", aggregate)


if __name__ == "__main__":
    main()
