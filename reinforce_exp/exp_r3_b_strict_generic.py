#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, command_manifest, ensure_dir, timestamp_now, write_json  # noqa: E402
from experiment3.theory1_si_circuits import MODELS  # noqa: E402
from experiment3.theory5b_boundary_detection import load_wiki_sequences, parse_head_list, run_attention_analysis  # noqa: E402
from experiment3.phase2.exp3p2b_trivial_feature_control import (  # noqa: E402
    _embedding_dimension_mean_ablation,
    _fit_embedding_prefix_classifier,
    _offset_group_boundary_scores,
    _post_ablation_t5b_a_full,
    _synthetic_boundary_full,
)
from shared.attention.adapters import get_adapter  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402

DEFAULT_OUT = RESULTS_ROOT / "exp_r3_core_replication" / "exp3p2b_strict_generic"


def run_one(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    num_sequences: int,
    seq_len: int,
    top_k_dims: int,
    synthetic_target_per_cell: int,
    seed: int,
) -> dict[str, Any]:
    out_dir = output_root / model_name
    ensure_dir(out_dir)

    t1_dir = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name
    head_groups_path = t1_dir / "head_groups.json"
    r2_path = t1_dir / "per_sequence_r2.parquet"

    missing = [str(p) for p in [head_groups_path, r2_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs for strict B generic ({model_name}): {missing}")

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)
    adapter = get_adapter(model_spec)
    adapter.register(model)

    sequences = load_wiki_sequences(model_name, max_sequences=max(8, int(num_sequences)), seq_len=max(128, int(seq_len)))
    sequences = sequences[: int(num_sequences)]
    if len(sequences) < 4:
        raise RuntimeError(f"Need >=4 sequences for strict B generic, found {len(sequences)}")

    head_groups = json.loads(head_groups_path.read_text(encoding="utf-8"))
    high_si = parse_head_list(head_groups["high_si"])
    low_si = parse_head_list(head_groups["low_si"])
    high_heads = [(int(h.layer), int(h.head)) for h in high_si]
    low_heads = [(int(h.layer), int(h.head)) for h in low_si]

    token_ids = np.array([int(t) for seq in sequences for t in seq], dtype=np.int64)
    classifier, top_dims, mean_vals = _fit_embedding_prefix_classifier(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        top_k_dims=max(1, int(top_k_dims)),
    )
    write_json(out_dir / "space_prefix_classifier.json", classifier)

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
    write_json(out_dir / "post_ablation_t5b_a.json", post_ablation)

    synth_result, synth_rows = _synthetic_boundary_full(
        model=model,
        adapter=adapter,
        tokenizer=tokenizer,
        device=device,
        sequences=sequences,
        high_heads=high_heads,
        low_heads=low_heads,
        target_per_transformed_cell=max(64, int(synthetic_target_per_cell)),
        seed=int(seed),
    )
    write_json(out_dir / "synthetic_boundary_results.json", synth_result)
    synth_rows.to_parquet(out_dir / "adversarial_sequences.parquet", index=False)

    # Optional offset-group report: only when upstream artifacts exist for this model.
    t5b_scores_path = ROOT / "results" / "experiment3" / "theory5b_boundary_detection" / model_name / "boundary_attention_scores.parquet"
    prev_scores_path = ROOT / "results" / "experiment3" / "theory7_induction_feeders" / model_name / "prev_token_scores.parquet"
    if t5b_scores_path.exists() and prev_scores_path.exists():
        boundary_scores = pd.read_parquet(t5b_scores_path)
        prev_scores = pd.read_parquet(prev_scores_path)
        offset_scores = _offset_group_boundary_scores(boundary_scores, prev_scores, head_groups)
        write_json(out_dir / "offset_group_boundary_scores.json", offset_scores)
    else:
        write_json(
            out_dir / "offset_group_boundary_scores.json",
            {
                "status": "skipped",
                "reason": "Missing theory5b or prev_token per-head artifacts for this model.",
                "expected": [str(t5b_scores_path), str(prev_scores_path)],
            },
        )

    summary = {
        "timestamp": timestamp_now(),
        "model": model_name,
        "settings": {
            "num_sequences": int(len(sequences)),
            "seq_len": int(seq_len),
            "top_k_dims": int(top_k_dims),
            "synthetic_target_per_cell": int(synthetic_target_per_cell),
            "seed": int(seed),
        },
        "verdict": {
            "boundary_non_trivial_after_control": bool(
                float(post_ablation.get("high_vs_low_attn_to_prev_last", {}).get("cohens_d", 0.0)) > 0.5
            ),
            "prefix_following_artifact_flag": bool(synth_result.get("prefix_following_artifact_flag", False)),
        },
    }
    write_json(out_dir / "strict_b_generic_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-R3 helper: strict-B generic runner")
    p.add_argument("--model", required=True, choices=sorted(MODELS.keys()))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-sequences", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--top-k-dims", type=int, default=16)
    p.add_argument("--synthetic-target-per-cell", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    ensure_dir(out_root)
    rep = run_one(
        model_name=str(args.model),
        device=str(args.device),
        output_root=out_root,
        num_sequences=max(8, int(args.num_sequences)),
        seq_len=max(128, int(args.seq_len)),
        top_k_dims=max(1, int(args.top_k_dims)),
        synthetic_target_per_cell=max(64, int(args.synthetic_target_per_cell)),
        seed=int(args.seed),
    )

    write_json(
        out_root / str(args.model) / "manifest.json",
        command_manifest(
            experiment_id="EXP-R3-B",
            command="strict_b_generic",
            model=str(args.model),
            seed_set=[int(args.seed)],
            extras={
                "device": str(args.device),
                "num_sequences": int(args.num_sequences),
                "synthetic_target_per_cell": int(args.synthetic_target_per_cell),
                "output_root": str(out_root),
            },
        ),
    )

    print(f"[EXP-R3-B] wrote {out_root / str(args.model) / 'strict_b_generic_summary.json'}")
    print(f"[EXP-R3-B] verdict={rep['verdict']}")


if __name__ == "__main__":
    main()
