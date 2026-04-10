#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment1.shift_kernels import RoPEEstimator  # noqa: E402
from experiment1.norm_utils import normalize_logits_for_norm  # noqa: E402
from experiment3.theory5b_boundary_detection import HeadID, load_wiki_sequences, run_attention_analysis  # noqa: E402
from experiment3.phase2.exp3p2b_trivial_feature_control import (  # noqa: E402
    _embedding_dimension_mean_ablation,
    _fit_embedding_prefix_classifier,
    _post_ablation_t5b_a_full,
    _synthetic_boundary_full,
)
from shared.attention.adapters import get_adapter  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402
from shared.specs import ModelSpec  # noqa: E402


NON_ROPE_MODELS: dict[str, ModelSpec] = {
    "gpt2-medium": ModelSpec(
        name="gpt2-medium",
        hf_id="openai-community/gpt2-medium",
        norm="LayerNorm",
        pe_scheme="LearnedAbsolutePE",
        notes="GPT-2 medium non-RoPE anchor.",
    ),
    "gpt2-small": ModelSpec(
        name="gpt2-small",
        hf_id="openai-community/gpt2",
        norm="LayerNorm",
        pe_scheme="LearnedAbsolutePE",
        notes="GPT-2 small non-RoPE fallback anchor.",
    ),
}


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _head_list_to_json(heads: list[HeadID]) -> list[dict[str, int]]:
    return [{"layer": int(h.layer), "head": int(h.head)} for h in heads]


def _classify_high_low(r2_df: pd.DataFrame, quantile: float = 0.25) -> tuple[list[HeadID], list[HeadID], pd.DataFrame]:
    mean_r2 = r2_df.groupby(["layer", "head"], as_index=False)["r2"].mean().rename(columns={"r2": "mean_r2"})
    mean_r2 = mean_r2.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    n_heads = int(len(mean_r2))
    n_select = max(1, int(n_heads * float(quantile)))
    high_si = [HeadID(int(r.layer), int(r.head)) for r in mean_r2.head(n_select).itertuples()]
    low_si = [HeadID(int(r.layer), int(r.head)) for r in mean_r2.tail(n_select).itertuples()]
    return high_si, low_si, mean_r2


def _compute_per_head_r2(
    *,
    model,
    adapter,
    model_spec: ModelSpec,
    device: str,
    sequences: list[list[int]],
) -> pd.DataFrame:
    estimator = RoPEEstimator()
    rows: list[dict[str, Any]] = []
    for seq_idx, tokens in enumerate(sequences):
        t0 = time.time()
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
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
        n_layers, n_heads = capture.logits.shape[0], capture.logits.shape[1]
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                head_logits = capture.logits[layer_idx, head_idx]
                prepared = normalize_logits_for_norm(head_logits, model_spec.norm)
                fit = estimator.fit_logits(prepared)
                rows.append(
                    {
                        "sequence_id": int(seq_idx),
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                        "r2": float(fit.r2),
                    }
                )
        del capture, input_ids
        torch.cuda.empty_cache()
        if (seq_idx + 1) % 4 == 0 or seq_idx == 0:
            print(
                f"[3P2-K][R2] {seq_idx + 1}/{len(sequences)} sequences ({time.time() - t0:.1f}s/seq)",
                flush=True,
            )
    if not rows:
        raise RuntimeError("No R2 rows were produced for non-RoPE control")
    return pd.DataFrame(rows)


def run_model(
    *,
    model_name: str,
    output_root: Path,
    device: str,
    num_sequences: int,
    seq_len: int,
    top_k_dims: int,
    synthetic_target_per_cell: int,
    seed: int,
) -> dict[str, Any]:
    spec = NON_ROPE_MODELS[model_name]
    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model(spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(spec)
    adapter = get_adapter(spec)
    adapter.register(model)

    sequences = load_wiki_sequences(model_name=model_name, max_sequences=max(8, int(num_sequences)), seq_len=int(seq_len))
    sequences = sequences[: int(num_sequences)]
    if len(sequences) < 4:
        raise RuntimeError(f"Need >=4 sequences for non-RoPE control, found {len(sequences)}")

    r2_df = _compute_per_head_r2(
        model=model,
        adapter=adapter,
        model_spec=spec,
        device=device,
        sequences=sequences,
    )
    print(f"[3P2-K] completed R2 profiling for {model_name}", flush=True)
    r2_df.to_parquet(out_dir / "per_sequence_r2.parquet", index=False)

    high_si, low_si, mean_r2 = _classify_high_low(r2_df)
    mean_r2.to_parquet(out_dir / "mean_r2_by_head.parquet", index=False)
    _write_json(
        out_dir / "head_groups.json",
        {
            "high_si": _head_list_to_json(high_si),
            "low_si": _head_list_to_json(low_si),
            "quantile": 0.25,
            "n_high": int(len(high_si)),
            "n_low": int(len(low_si)),
        },
    )

    token_ids = np.array([int(t) for seq in sequences for t in seq], dtype=np.int64)
    classifier, top_dims, mean_vals = _fit_embedding_prefix_classifier(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        top_k_dims=int(top_k_dims),
    )
    _write_json(out_dir / "space_prefix_classifier.json", classifier)
    print(f"[3P2-K] fitted feature classifier for {model_name}", flush=True)

    with _embedding_dimension_mean_ablation(model, top_dims, mean_vals):
        approach_a, _ = run_attention_analysis(
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
    post_ablation = _post_ablation_t5b_a_full(approach_a, classifier)
    post_ablation["embedding_ablation"] = {
        "top_k_dims": [int(d) for d in top_dims],
        "n_dims": int(len(top_dims)),
    }
    _write_json(out_dir / "post_ablation_t5b_a.json", post_ablation)
    print(f"[3P2-K] completed post-ablation boundary analysis for {model_name}", flush=True)

    high_heads = [(int(h.layer), int(h.head)) for h in high_si]
    low_heads = [(int(h.layer), int(h.head)) for h in low_si]
    synth_result, synth_rows = _synthetic_boundary_full(
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
    _write_json(out_dir / "synthetic_boundary_results.json", synth_result)
    synth_rows.to_parquet(out_dir / "adversarial_sequences.parquet", index=False)
    print(f"[3P2-K] completed synthetic control for {model_name}", flush=True)

    high_mean_r2 = _safe_float(mean_r2.head(len(high_si))["mean_r2"].mean())
    low_mean_r2 = _safe_float(mean_r2.tail(len(low_si))["mean_r2"].mean())
    d_post = _safe_float(post_ablation.get("high_vs_low_attn_to_prev_last", {}).get("cohens_d"))
    diff_fake_real = _safe_float(synth_result.get("comparisons", {}).get("fake_minus_real_with_prefix"))
    prefix_flag = bool(synth_result.get("prefix_following_artifact_flag", False))

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "model_spec": {
            "hf_id": spec.hf_id,
            "norm": spec.norm,
            "pe_scheme": spec.pe_scheme,
        },
        "settings": {
            "num_sequences": int(len(sequences)),
            "seq_len": int(seq_len),
            "top_k_dims": int(top_k_dims),
            "synthetic_target_per_cell": int(synthetic_target_per_cell),
            "seed": int(seed),
        },
        "r2_summary": {
            "n_heads": int(len(mean_r2)),
            "high_mean_r2": _safe_float(high_mean_r2),
            "low_mean_r2": _safe_float(low_mean_r2),
            "overall_mean_r2": _safe_float(mean_r2["mean_r2"].mean()),
            "overall_std_r2": _safe_float(mean_r2["mean_r2"].std(ddof=1)),
        },
        "boundary_control": {
            "post_ablation_high_vs_low_d": _safe_float(d_post),
            "fake_minus_real_with_prefix": _safe_float(diff_fake_real),
            "prefix_following_artifact_flag": bool(prefix_flag),
            "prefix_following_assessment": synth_result.get("prefix_following_assessment", {}),
        },
        "verdict": {
            "si_structure_detected": bool(np.isfinite(high_mean_r2) and np.isfinite(low_mean_r2) and high_mean_r2 > low_mean_r2),
            "boundary_non_trivial_after_control": bool(np.isfinite(d_post) and d_post > 0.50),
            "rope_confound_weakened_for_anchor": bool(
                np.isfinite(high_mean_r2)
                and np.isfinite(low_mean_r2)
                and (high_mean_r2 > low_mean_r2)
                and np.isfinite(d_post)
                and (d_post > 0.35)
                and (not prefix_flag)
            ),
        },
    }
    _write_json(out_dir / "non_rope_control_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase2 non-RoPE anchor control (3P2-K)")
    p.add_argument("--model", choices=["auto", "gpt2-medium", "gpt2-small"], default="auto")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-sequences", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--top-k-dims", type=int, default=16)
    p.add_argument("--synthetic-target-per-cell", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default="results/experiment3_phase2/exp3p2k_non_rope_control")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model_order = ["gpt2-medium", "gpt2-small"] if args.model == "auto" else [str(args.model)]
    attempts: list[dict[str, Any]] = []

    for m in model_order:
        t0 = time.time()
        try:
            rep = run_model(
                model_name=m,
                output_root=output_root,
                device=str(args.device),
                num_sequences=max(8, int(args.num_sequences)),
                seq_len=max(128, int(args.seq_len)),
                top_k_dims=max(1, int(args.top_k_dims)),
                synthetic_target_per_cell=max(64, int(args.synthetic_target_per_cell)),
                seed=int(args.seed),
            )
            attempts.append({"model": m, "status": "complete", "elapsed_sec": float(time.time() - t0)})
            summary = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "requested_model": str(args.model),
                "selected_model": m,
                "attempts": attempts,
                "result_path": str(output_root / m / "non_rope_control_summary.json"),
                "status": "complete",
                "verdict": rep.get("verdict", {}),
            }
            _write_json(output_root / "non_rope_control_summary.json", summary)
            print(f"[3P2-K] complete with {m}")
            return
        except Exception as exc:
            attempts.append(
                {
                    "model": m,
                    "status": "failed",
                    "elapsed_sec": float(time.time() - t0),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[3P2-K] {m} failed: {type(exc).__name__}: {exc}", flush=True)

    fail = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_model": str(args.model),
        "attempts": attempts,
        "status": "failed",
    }
    _write_json(output_root / "non_rope_control_summary.json", fail)
    raise RuntimeError("All non-RoPE anchor attempts failed")


if __name__ == "__main__":
    main()
