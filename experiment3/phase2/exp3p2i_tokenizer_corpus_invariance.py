#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment3.phase2.exp3p2b_trivial_feature_control import (  # noqa: E402
    _embedding_dimension_mean_ablation,
    _fit_embedding_prefix_classifier,
    _synthetic_boundary_full,
)
from experiment3.theory5b_boundary_detection import (  # noqa: E402
    MODELS,
    parse_head_list,
    run_attention_analysis,
)
from shared.attention.adapters import get_adapter  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402


TARGET_MODELS = ("llama-3.1-8b", "olmo-2-7b")


@dataclass(frozen=True)
class CorpusSpec:
    corpus_id: str
    domain: str
    language: str
    kind: str
    source: str


CORPORA: tuple[CorpusSpec, ...] = (
    CorpusSpec("wiki_en", "wikipedia", "en", "tokenized_jsonl", "data/experiment1/wiki40b_en_pre2019/{model}/len_1024.jsonl"),
    CorpusSpec("code_en", "code", "en", "tokenized_jsonl", "data/experiment1/codesearchnet_python_snapshot/{model}/len_1024.jsonl"),
    CorpusSpec("opus_tr", "translation", "tr", "opus100", "en-tr"),
    CorpusSpec("opus_zh", "translation", "zh", "opus100", "en-zh"),
    CorpusSpec("opus_ru", "translation", "ru", "opus100", "en-ru"),
)


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


def _load_tokenized_sequences(path: Path, *, seq_len: int, max_sequences: int) -> list[list[int]]:
    if not path.exists():
        raise FileNotFoundError(f"Tokenized JSONL missing: {path}")
    seqs: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            toks = row.get("tokens", row.get("input_ids", []))
            if len(toks) < seq_len:
                continue
            seqs.append([int(t) for t in toks[:seq_len]])
            if len(seqs) >= max_sequences:
                break
    if len(seqs) < max_sequences:
        raise RuntimeError(f"Only found {len(seqs)} sequences in {path}; need {max_sequences}")
    return seqs


def _opus_cache_path(model: str, lang: str, seq_len: int) -> Path:
    return ROOT / "data" / "experiment3_phase2_multilingual" / f"opus100_{lang}" / model / f"len_{seq_len}.jsonl"


def _load_opus_sequences_cached(path: Path, *, max_sequences: int) -> list[list[int]]:
    seqs: list[list[int]] = []
    if not path.exists():
        return seqs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            toks = row.get("tokens", [])
            if toks:
                seqs.append([int(t) for t in toks])
            if len(seqs) >= max_sequences:
                break
    return seqs


def _build_opus_sequences(
    *,
    tokenizer,
    lang: str,
    split_cfg: str,
    seq_len: int,
    max_sequences: int,
    max_rows: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    ds = load_dataset("opus100", split_cfg, split=f"train[:{max_rows}]")
    target_lang = lang

    sep_ids = tokenizer.encode("\n", add_special_tokens=False)
    sep = sep_ids[:1] if sep_ids else []

    seqs: list[list[int]] = []
    buffer: list[int] = []
    used_rows = 0

    for row in ds:
        trans = row.get("translation", {})
        text = str(trans.get(target_lang, "")).strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        used_rows += 1
        buffer.extend(ids)
        if sep:
            buffer.extend(sep)
        while len(buffer) >= seq_len and len(seqs) < max_sequences:
            seqs.append(buffer[:seq_len])
            buffer = buffer[seq_len:]
        if len(seqs) >= max_sequences:
            break

    meta = {
        "dataset": "opus100",
        "config": split_cfg,
        "language": lang,
        "rows_consumed": int(used_rows),
        "max_rows": int(max_rows),
        "sequences_emitted": int(len(seqs)),
    }
    return seqs, meta


def _ensure_opus_sequences(
    *,
    model: str,
    tokenizer,
    lang: str,
    split_cfg: str,
    seq_len: int,
    max_sequences: int,
    max_rows: int,
    force_rebuild: bool,
) -> tuple[list[list[int]], dict[str, Any]]:
    cache = _opus_cache_path(model, lang, seq_len)
    if not force_rebuild:
        cached = _load_opus_sequences_cached(cache, max_sequences=max_sequences)
        if len(cached) >= max_sequences:
            return cached[:max_sequences], {
                "source": "cache",
                "cache_path": str(cache),
                "sequences": int(len(cached[:max_sequences])),
            }

    seqs, meta = _build_opus_sequences(
        tokenizer=tokenizer,
        lang=lang,
        split_cfg=split_cfg,
        seq_len=seq_len,
        max_sequences=max_sequences,
        max_rows=max_rows,
    )
    if len(seqs) < max_sequences:
        raise RuntimeError(
            f"Failed to build enough opus sequences for {model}/{lang}: got {len(seqs)}, need {max_sequences}"
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", encoding="utf-8") as f:
        for i, toks in enumerate(seqs):
            f.write(json.dumps({"idx": i, "tokens": toks}, ensure_ascii=False) + "\n")

    meta.update({"source": "rebuilt", "cache_path": str(cache)})
    return seqs[:max_sequences], meta


def _load_corpus_sequences(
    *,
    model: str,
    tokenizer,
    spec: CorpusSpec,
    seq_len: int,
    max_sequences: int,
    opus_max_rows: int,
    force_rebuild_opus: bool,
) -> tuple[list[list[int]], dict[str, Any]]:
    if spec.kind == "tokenized_jsonl":
        rel = spec.source.format(model=model)
        path = ROOT / rel
        seqs = _load_tokenized_sequences(path, seq_len=seq_len, max_sequences=max_sequences)
        return seqs, {
            "source": "tokenized_jsonl",
            "path": str(path),
            "sequences": int(len(seqs)),
            "seq_len": int(seq_len),
        }

    if spec.kind == "opus100":
        seqs, meta = _ensure_opus_sequences(
            model=model,
            tokenizer=tokenizer,
            lang=spec.language,
            split_cfg=spec.source,
            seq_len=seq_len,
            max_sequences=max_sequences,
            max_rows=opus_max_rows,
            force_rebuild=force_rebuild_opus,
        )
        meta["seq_len"] = int(seq_len)
        return seqs, meta

    raise ValueError(f"Unknown corpus kind: {spec.kind}")


def _load_gate_inputs(model: str) -> dict[str, Any]:
    root = ROOT / "results" / "experiment3_phase2" / "exp3p2b_trivial_feature_control" / model
    post = _load_json(root / "post_ablation_t5b_a.json")
    synth = _load_json(root / "synthetic_boundary_results.json")
    d_val = _safe_float(post.get("high_vs_low_attn_to_prev_last", {}).get("cohens_d"))
    prefix_flag = bool(synth.get("prefix_following_artifact_flag", False))
    prefix_flag_legacy = bool(synth.get("prefix_following_artifact_flag_legacy_binary", prefix_flag))
    prefix_assessment = synth.get("prefix_following_assessment", {})
    return {
        "post_ablation_d": d_val,
        "prefix_following_artifact_flag": prefix_flag,
        "prefix_following_artifact_flag_legacy_binary": prefix_flag_legacy,
        "prefix_following_assessment": prefix_assessment,
        "passes_gate": bool(np.isfinite(d_val) and d_val > 0.50 and (not prefix_flag)),
        "source_files": {
            "post_ablation": str(root / "post_ablation_t5b_a.json"),
            "synthetic_boundary": str(root / "synthetic_boundary_results.json"),
        },
    }


def _condition_metrics_from_approach(
    *,
    model: str,
    corpus: CorpusSpec,
    perturbation: str,
    approach_a: dict[str, Any],
    approach_c: dict[str, Any],
    n_sequences: int,
) -> list[dict[str, Any]]:
    comp = approach_a.get("high_vs_low_comparison", {}).get("attn_to_prev_last", {})
    corr = approach_c.get("correlation", {})
    high = approach_a.get("high_si_heads", {}).get("attn_to_prev_last", {})
    low = approach_a.get("low_si_heads", {}).get("attn_to_prev_last", {})

    base = {
        "model": model,
        "corpus_id": corpus.corpus_id,
        "corpus_domain": corpus.domain,
        "language": corpus.language,
        "perturbation_condition": perturbation,
        "n_sequences": int(n_sequences),
    }

    rows = [
        {
            **base,
            "metric_name": "approach_a_cohens_d",
            "metric_value": _safe_float(comp.get("cohens_d")),
            "aux": {
                "difference": _safe_float(comp.get("difference")),
                "p_value": _safe_float(comp.get("t_p_value")),
                "p_value_holm": _safe_float(comp.get("t_p_value_holm")),
                "n_high": int(comp.get("n_high", 0)),
                "n_low": int(comp.get("n_low", 0)),
                "high_mean": _safe_float(high.get("mean")),
                "low_mean": _safe_float(low.get("mean")),
                "high_ci_95": high.get("ci_95", [float("nan"), float("nan")]),
                "low_ci_95": low.get("ci_95", [float("nan"), float("nan")]),
            },
        },
        {
            **base,
            "metric_name": "approach_c_spearman_rho",
            "metric_value": _safe_float(corr.get("spearman_rho")),
            "aux": {
                "pearson_r": _safe_float(corr.get("pearson_r")),
                "pearson_p": _safe_float(corr.get("pearson_p")),
                "spearman_p": _safe_float(corr.get("spearman_p")),
                "spearman_ci_95": corr.get("spearman_ci_95", [float("nan"), float("nan")]),
            },
        },
    ]
    return rows


def _run_approach_ac_condition(
    *,
    model_name: str,
    model,
    adapter,
    tokenizer,
    device: str,
    sequences: list[list[int]],
    high_si,
    low_si,
    r2_df: pd.DataFrame,
    perturbation: str,
    top_dims: list[int],
    mean_vals: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_spec = MODELS[model_name]

    cm: contextlib.AbstractContextManager
    if perturbation == "embedding_prefix_ablation":
        cm = _embedding_dimension_mean_ablation(model, top_dims, mean_vals)
    else:
        cm = contextlib.nullcontext()

    with cm:
        approach_a, (approach_c, _score_df) = run_attention_analysis(
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
    return approach_a, approach_c


def _rows_from_synthetic(
    *,
    model: str,
    corpus: CorpusSpec,
    synthetic_result: dict[str, Any],
    n_sequences: int,
) -> list[dict[str, Any]]:
    cells = synthetic_result.get("cells", {})
    rbp = _safe_float(cells.get("real_boundary_with_prefix", {}).get("mean_high_prev_attn"))
    fbp = _safe_float(cells.get("fake_boundary_with_prefix", {}).get("mean_high_prev_attn"))
    rbn = _safe_float(cells.get("real_boundary_without_prefix", {}).get("mean_high_prev_attn"))
    fbn = _safe_float(cells.get("fake_boundary_without_prefix", {}).get("mean_high_prev_attn"))

    base = {
        "model": model,
        "corpus_id": corpus.corpus_id,
        "corpus_domain": corpus.domain,
        "language": corpus.language,
        "perturbation_condition": "adversarial_boundary_variant",
        "n_sequences": int(n_sequences),
    }
    return [
        {
            **base,
            "metric_name": "adversarial_real_minus_fake_with_prefix",
            "metric_value": rbp - fbp if np.isfinite(rbp) and np.isfinite(fbp) else float("nan"),
            "aux": {
                "real_boundary_with_prefix": rbp,
                "fake_boundary_with_prefix": fbp,
                "prefix_following_artifact_flag": bool(synthetic_result.get("prefix_following_artifact_flag", False)),
            },
        },
        {
            **base,
            "metric_name": "adversarial_real_minus_fake_without_prefix",
            "metric_value": rbn - fbn if np.isfinite(rbn) and np.isfinite(fbn) else float("nan"),
            "aux": {
                "real_boundary_without_prefix": rbn,
                "fake_boundary_without_prefix": fbn,
            },
        },
    ]


def _compute_invariance(rows: pd.DataFrame) -> dict[str, Any]:
    d_rows = rows[rows["metric_name"] == "approach_a_cohens_d"].copy()
    adv_rows = rows[rows["metric_name"] == "adversarial_real_minus_fake_with_prefix"].copy()

    baseline = d_rows[(d_rows["corpus_id"] == "wiki_en") & (d_rows["perturbation_condition"] == "none")]
    if baseline.empty:
        return {
            "invariance_score": float("nan"),
            "invariance_verdict": "inconclusive_missing_baseline",
            "details": {},
        }

    baseline_d = float(baseline.iloc[0]["metric_value"])
    if not np.isfinite(baseline_d) or abs(baseline_d) < 1e-8:
        return {
            "invariance_score": float("nan"),
            "invariance_verdict": "inconclusive_invalid_baseline",
            "details": {"baseline_d": baseline_d},
        }

    comp = d_rows.copy()
    comp = comp[~((comp["corpus_id"] == "wiki_en") & (comp["perturbation_condition"] == "none"))]
    comp["retention"] = comp["metric_value"].astype(float) / baseline_d
    comp["retention_clipped"] = comp["retention"].clip(lower=-1.5, upper=1.5)
    comp["same_sign"] = np.sign(comp["metric_value"].astype(float)) == np.sign(baseline_d)

    if len(comp) == 0:
        effect_score = 1.0
        same_sign_frac = 1.0
    else:
        effect_score = float(np.clip(np.median(np.maximum(0.0, comp["retention_clipped"].to_numpy(dtype=float))), 0.0, 1.5) / 1.5)
        same_sign_frac = float(comp["same_sign"].mean())

    if len(adv_rows) == 0:
        adv_score = 0.5
    else:
        adv_score = float((adv_rows["metric_value"].astype(float) > 0).mean())

    invariance_score = float(0.55 * effect_score + 0.25 * same_sign_frac + 0.20 * adv_score)

    if invariance_score >= 0.75:
        verdict = "invariant"
    elif invariance_score >= 0.45:
        verdict = "partially_sensitive"
    else:
        verdict = "non_invariant"

    return {
        "invariance_score": invariance_score,
        "invariance_verdict": verdict,
        "details": {
            "baseline_d": baseline_d,
            "effect_retention_score": effect_score,
            "same_sign_fraction": same_sign_frac,
            "adversarial_consistency_score": adv_score,
            "n_effect_cells": int(len(comp)),
            "n_adversarial_cells": int(len(adv_rows)),
        },
    }


def _pooled_mixed_effects_proxy(rows: pd.DataFrame) -> dict[str, Any]:
    """
    Lightweight pooled summary used as a mixed-effects proxy.
    We report grouped stability statistics without over-claiming a full model fit.
    """
    payload: dict[str, Any] = {
        "note": (
            "Exploratory pooled summary (fixed-weight descriptive proxy), "
            "not a full hierarchical mixed-effects fit."
        ),
        "by_metric_and_perturbation": {},
        "by_metric_and_language": {},
    }

    for key_cols, field in [
        (["metric_name", "perturbation_condition"], "by_metric_and_perturbation"),
        (["metric_name", "language"], "by_metric_and_language"),
    ]:
        grouped = rows.groupby(key_cols, as_index=False)["metric_value"].agg(["mean", "std", "median", "min", "max", "count"]).reset_index()
        for r in grouped.itertuples(index=False):
            if len(key_cols) == 2:
                metric_name = str(getattr(r, key_cols[0]))
                key_val = str(getattr(r, key_cols[1]))
            else:
                continue
            payload[field][f"{metric_name}::{key_val}"] = {
                "n_cells": int(getattr(r, "count")),
                "mean": _safe_float(getattr(r, "mean")),
                "std": _safe_float(getattr(r, "std")),
                "median": _safe_float(getattr(r, "median")),
                "min": _safe_float(getattr(r, "min")),
                "max": _safe_float(getattr(r, "max")),
            }
    return payload


def run_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    seq_len: int,
    seq_per_corpus: int,
    adversarial_positions: int,
    opus_max_rows: int,
    force_gate_override: bool,
    force_rebuild_opus: bool,
) -> dict[str, Any]:
    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    gate = _load_gate_inputs(model_name)
    gate_status = {
        "rule": "d_gt_0.50_and_no_prefix_following_statistical_artifact",
        "measured": gate,
        "override": bool(force_gate_override),
        "effective_allow_run": bool(gate["passes_gate"] or force_gate_override),
    }
    if not gate_status["effective_allow_run"]:
        blocked = {
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tier": "tier3_publication_readiness",
            "execution_gate": gate_status,
            "corpora": [c.corpus_id for c in CORPORA],
            "perturbation_conditions": ["none", "embedding_prefix_ablation", "adversarial_boundary_variant"],
            "effect_sizes": {},
            "invariance_score": float("nan"),
            "invariance_verdict": "blocked_by_gate",
            "multiplicity_family": "tier3_exploratory_invariance",
            "status": "blocked",
        }
        _write_json(out_dir / "invariance_report.json", blocked)
        return blocked

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)
    adapter = get_adapter(model_spec)
    adapter.register(model)

    t1_dir = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name
    r2_path = t1_dir / "per_sequence_r2.parquet"
    head_groups_path = t1_dir / "head_groups.json"
    if not r2_path.exists() or not head_groups_path.exists():
        raise FileNotFoundError(f"Missing theory1 prerequisites for {model_name}: {r2_path}, {head_groups_path}")

    head_groups = _load_json(head_groups_path)
    high_si = parse_head_list(head_groups["high_si"])
    low_si = parse_head_list(head_groups["low_si"])
    r2_per_seq = pd.read_parquet(r2_path)
    r2_df = r2_per_seq.groupby(["layer", "head"], as_index=False)["r2"].mean().rename(columns={"r2": "mean_r2"})

    corpus_sequences: dict[str, list[list[int]]] = {}
    corpus_meta: dict[str, Any] = {}
    for spec in CORPORA:
        seqs, meta = _load_corpus_sequences(
            model=model_name,
            tokenizer=tokenizer,
            spec=spec,
            seq_len=seq_len,
            max_sequences=seq_per_corpus,
            opus_max_rows=opus_max_rows,
            force_rebuild_opus=force_rebuild_opus,
        )
        corpus_sequences[spec.corpus_id] = seqs
        corpus_meta[spec.corpus_id] = meta

    token_ids_ref = np.array([int(tok) for seq in corpus_sequences["wiki_en"] for tok in seq], dtype=np.int64)
    prefix_classifier, top_dims, mean_vals = _fit_embedding_prefix_classifier(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids_ref,
        top_k_dims=16,
    )

    rows: list[dict[str, Any]] = []

    high_heads = [(int(h.layer), int(h.head)) for h in high_si]
    low_heads = [(int(h.layer), int(h.head)) for h in low_si]

    for spec in CORPORA:
        seqs = corpus_sequences[spec.corpus_id]

        # Baseline and embedding-ablation conditions for Approach A/C.
        for perturb in ("none", "embedding_prefix_ablation"):
            approach_a, approach_c = _run_approach_ac_condition(
                model_name=model_name,
                model=model,
                adapter=adapter,
                tokenizer=tokenizer,
                device=device,
                sequences=seqs,
                high_si=high_si,
                low_si=low_si,
                r2_df=r2_df,
                perturbation=perturb,
                top_dims=top_dims,
                mean_vals=mean_vals,
            )
            rows.extend(
                _condition_metrics_from_approach(
                    model=model_name,
                    corpus=spec,
                    perturbation=perturb,
                    approach_a=approach_a,
                    approach_c=approach_c,
                    n_sequences=len(seqs),
                )
            )

        # Adversarial boundary variant control.
        synth_result, _synth_rows = _synthetic_boundary_full(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            device=device,
            sequences=seqs,
            high_heads=high_heads,
            low_heads=low_heads,
            target_per_transformed_cell=adversarial_positions,
            seed=0,
        )
        rows.extend(
            _rows_from_synthetic(
                model=model_name,
                corpus=spec,
                synthetic_result=synth_result,
                n_sequences=len(seqs),
            )
        )

    records = []
    for r in rows:
        aux = r.pop("aux", {})
        rec = {**r}
        for k, v in aux.items():
            rec[f"aux__{k}"] = v
        records.append(rec)

    breakdown = pd.DataFrame(records)
    breakdown.to_parquet(out_dir / "corpus_breakdown.parquet", index=False)

    inv = _compute_invariance(breakdown)
    pooled_summary = _pooled_mixed_effects_proxy(breakdown)

    # Compact effect-size payload by corpus/condition.
    effect_sizes: dict[str, Any] = {}
    for (cid, pert), grp in breakdown.groupby(["corpus_id", "perturbation_condition"], as_index=False):
        key = f"{cid}__{pert}"
        effect_sizes[key] = {
            row.metric_name: _safe_float(row.metric_value)
            for row in grp.itertuples()
        }

    report = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": "tier3_publication_readiness",
        "execution_gate": gate_status,
        "corpora": [
            {
                "corpus_id": c.corpus_id,
                "domain": c.domain,
                "language": c.language,
                "kind": c.kind,
                "meta": corpus_meta.get(c.corpus_id, {}),
            }
            for c in CORPORA
        ],
        "perturbation_conditions": ["none", "embedding_prefix_ablation", "adversarial_boundary_variant"],
        "effect_sizes": effect_sizes,
        "invariance_score": _safe_float(inv.get("invariance_score")),
        "invariance_verdict": str(inv.get("invariance_verdict", "unknown")),
        "invariance_details": inv.get("details", {}),
        "pooled_mixed_effects_summary_proxy": pooled_summary,
        "prefix_classifier": prefix_classifier,
        "multiplicity_family": "tier3_exploratory_invariance",
        "primary_test_id": "3P2-I+Idea2",
        "status": "complete",
    }
    _write_json(out_dir / "invariance_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3P2-I + Idea 2: tokenizer/corpus invariance with cross-lingual extension")
    p.add_argument("--model", choices=["all", *TARGET_MODELS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1",
        help="Comma-separated model:device map used when --model all",
    )
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--seq-per-corpus", type=int, default=16)
    p.add_argument("--adversarial-positions", type=int, default=200)
    p.add_argument("--opus-max-rows", type=int, default=20000)
    p.add_argument("--force-gate-override", action="store_true")
    p.add_argument("--force-rebuild-opus", action="store_true")
    p.add_argument(
        "--output-root",
        default="results/experiment3_phase2/exp3p2i_tokenizer_corpus",
    )
    return p.parse_args()


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [t.strip() for t in raw.split(",") if t.strip()]:
        model, device = tok.split(":", 1)
        out[model.strip()] = device.strip()
    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    models = list(TARGET_MODELS) if args.model == "all" else [args.model]
    device_map = _parse_device_map(args.device_map) if args.model == "all" else {}

    reports = {}
    for model_name in models:
        device = device_map.get(model_name, args.device)
        print(f"[3P2-I] model={model_name} device={device}", flush=True)
        rep = run_model(
            model_name=model_name,
            device=device,
            output_root=output_root,
            seq_len=int(args.seq_len),
            seq_per_corpus=int(args.seq_per_corpus),
            adversarial_positions=int(args.adversarial_positions),
            opus_max_rows=int(args.opus_max_rows),
            force_gate_override=bool(args.force_gate_override),
            force_rebuild_opus=bool(args.force_rebuild_opus),
        )
        reports[model_name] = rep

    _write_json(output_root / "invariance_summary.json", {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": reports,
    })


if __name__ == "__main__":
    main()
