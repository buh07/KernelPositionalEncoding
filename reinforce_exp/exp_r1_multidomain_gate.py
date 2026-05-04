#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import (  # noqa: E402
    RESULTS_ROOT,
    command_manifest,
    ensure_dir,
    parse_csv_ints,
    parse_csv_strs,
    read_json,
    safe_float,
    timestamp_now,
    write_json,
)
from experiment3.theory5b_boundary_detection import MODELS, parse_head_list, run_attention_analysis  # noqa: E402
from experiment3.phase2.exp3p2b_trivial_feature_control import (  # noqa: E402
    _embedding_dimension_mean_ablation,
    _fit_embedding_prefix_classifier,
    _load_json,
    _offset_group_boundary_scores,
    _post_ablation_t5b_a_full,
    _synthetic_boundary_full,
)
from experiment5.pipeline import (  # noqa: E402
    _load_code_sequences,
    _load_dialogue_sequences,
    _load_wiki_sequences,
)
from shared.attention.adapters import get_adapter  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402

SUPPORTED_DOMAINS = ("wiki", "code", "dialogue")
DEFAULT_OUT = RESULTS_ROOT / "exp_r1_multidomain_gate"


def _load_required_inputs(model_name: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t1_dir = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name
    t5b_dir = ROOT / "results" / "experiment3" / "theory5b_boundary_detection" / model_name
    t7_dir = ROOT / "results" / "experiment3" / "theory7_induction_feeders" / model_name

    head_groups_path = t1_dir / "head_groups.json"
    r2_path = t1_dir / "per_sequence_r2.parquet"
    t5b_scores_path = t5b_dir / "boundary_attention_scores.parquet"
    prev_path = t7_dir / "prev_token_scores.parquet"

    required = [head_groups_path, r2_path, t5b_scores_path, prev_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs for EXP-R1 ({model_name}): {missing}")

    head_groups = _load_json(head_groups_path)
    r2_per_seq = pd.read_parquet(r2_path)
    r2_df = r2_per_seq.groupby(["layer", "head"], as_index=False)["r2"].mean().rename(columns={"r2": "mean_r2"})
    boundary_scores = pd.read_parquet(t5b_scores_path)
    prev_scores = pd.read_parquet(prev_path)
    return head_groups, r2_df, boundary_scores, prev_scores


def _load_domain_sequences(
    *,
    domain: str,
    model_name: str,
    tokenizer,
    seq_len: int,
    max_sequences: int,
    seed: int,
) -> list[list[int]]:
    if domain == "wiki":
        seqs = _load_wiki_sequences(model_name, tokenizer, seq_len=seq_len, max_sequences=max_sequences, seed=seed)
    elif domain == "code":
        seqs = _load_code_sequences(model_name, tokenizer, seq_len=seq_len, max_sequences=max_sequences, seed=seed + 17)
    elif domain == "dialogue":
        seqs = _load_dialogue_sequences(tokenizer, seq_len=seq_len, max_sequences=max_sequences, seed=seed + 31)
    else:
        raise ValueError(f"Unsupported domain={domain}; expected one of {SUPPORTED_DOMAINS}")

    if len(seqs) < 4:
        raise RuntimeError(f"Domain {domain} for {model_name} returned {len(seqs)} sequences (<4)")
    return seqs[:max_sequences]


def _run_one_domain(
    *,
    model_name: str,
    domain: str,
    seed: int,
    model,
    tokenizer,
    adapter,
    device: str,
    sequences: list[list[int]],
    head_groups: dict[str, Any],
    r2_df: pd.DataFrame,
    boundary_scores: pd.DataFrame,
    prev_scores: pd.DataFrame,
    top_k_dims: int,
    synthetic_target_per_cell: int,
    output_root: Path,
) -> dict[str, Any]:
    out_dir = output_root / f"domain_{domain}" / f"seed{seed}" / model_name
    ensure_dir(out_dir)

    high_si = parse_head_list(head_groups["high_si"])
    low_si = parse_head_list(head_groups["low_si"])
    high_heads = [(int(h.layer), int(h.head)) for h in high_si]
    low_heads = [(int(h.layer), int(h.head)) for h in low_si]

    token_ids = np.array([int(t) for seq in sequences for t in seq], dtype=np.int64)
    classifier, top_dims, mean_vals = _fit_embedding_prefix_classifier(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        top_k_dims=int(top_k_dims),
    )
    write_json(out_dir / "space_prefix_classifier.json", classifier)

    with _embedding_dimension_mean_ablation(model, top_dims, mean_vals):
        approach_a, _ = run_attention_analysis(
            model=model,
            adapter=adapter,
            model_spec=MODELS[model_name],
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
    post_ablation["domain"] = domain
    post_ablation["seed"] = int(seed)
    write_json(out_dir / "post_ablation_t5b_a.json", post_ablation)

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
    synth_result["domain"] = domain
    synth_result["seed"] = int(seed)
    write_json(out_dir / "synthetic_boundary_results.json", synth_result)
    synth_rows.to_parquet(out_dir / "adversarial_sequences.parquet", index=False)

    offset_scores = _offset_group_boundary_scores(boundary_scores, prev_scores, head_groups)
    write_json(out_dir / "offset_group_boundary_scores.json", offset_scores)

    diff = safe_float(synth_result.get("comparisons", {}).get("fake_minus_real_with_prefix"))
    d = safe_float(synth_result.get("prefix_following_assessment", {}).get("cohens_d_fake_minus_real"))
    p_one = safe_float(synth_result.get("prefix_following_assessment", {}).get("p_one_sided_fake_gt_real"))

    return {
        "domain": domain,
        "seed": int(seed),
        "n_sequences": int(len(sequences)),
        "fake_minus_real_with_prefix": diff,
        "cohens_d_fake_minus_real": d,
        "p_one_sided_fake_gt_real": p_one,
        "prefix_following_artifact_flag": bool(synth_result.get("prefix_following_artifact_flag", False)),
        "output_dir": str(out_dir),
    }


def cmd_run_seed(args: argparse.Namespace) -> None:
    domains = parse_csv_strs(args.domains)
    for d in domains:
        if d not in SUPPORTED_DOMAINS:
            raise ValueError(f"Unsupported domain={d}; expected one of {SUPPORTED_DOMAINS}")

    out_root = Path(args.output_root)
    ensure_dir(out_root)

    model_name = str(args.model)
    seed = int(args.seed)
    device = str(args.device)

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)
    adapter = get_adapter(model_spec)
    adapter.register(model)

    head_groups, r2_df, boundary_scores, prev_scores = _load_required_inputs(model_name)

    seed_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for domain in domains:
        d0 = time.time()
        seqs = _load_domain_sequences(
            domain=domain,
            model_name=model_name,
            tokenizer=tokenizer,
            seq_len=int(args.seq_len),
            max_sequences=int(args.num_sequences),
            seed=seed,
        )
        row = _run_one_domain(
            model_name=model_name,
            domain=domain,
            seed=seed,
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            device=device,
            sequences=seqs,
            head_groups=head_groups,
            r2_df=r2_df,
            boundary_scores=boundary_scores,
            prev_scores=prev_scores,
            top_k_dims=int(args.top_k_dims),
            synthetic_target_per_cell=int(args.synthetic_target_per_cell),
            output_root=out_root,
        )
        row["elapsed_sec"] = float(time.time() - d0)
        seed_rows.append(row)
        print(f"[EXP-R1][seed={seed}] domain={domain} complete in {row['elapsed_sec']:.1f}s", flush=True)

    manifest = command_manifest(
        experiment_id="EXP-R1",
        command="run-seed",
        model=model_name,
        seed_set=[seed],
        extras={
            "domains": domains,
            "num_sequences": int(args.num_sequences),
            "synthetic_target_per_cell": int(args.synthetic_target_per_cell),
            "output_root": str(out_root),
            "elapsed_sec": float(time.time() - t0),
            "seed_results": seed_rows,
        },
    )
    write_json(out_root / "manifests" / f"seed_{seed}.json", manifest)


def _run_subprocess(cmd: list[str]) -> None:
    print("[EXP-R1] exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def cmd_finalize_domain(args: argparse.Namespace) -> None:
    domains = parse_csv_strs(args.domains)
    seeds = parse_csv_ints(args.seeds)
    out_root = Path(args.output_root)

    py = str(ROOT / ".venv" / "bin" / "python")
    adjudicator = ROOT / "experiment3" / "phase2" / "exp3p2b_multiseed_gate_adjudication.py"

    domain_rows: list[dict[str, Any]] = []
    for domain in domains:
        in_root = out_root / f"domain_{domain}"
        canon_root = out_root / "canonical" / f"domain_{domain}"
        cmd = [
            py,
            str(adjudicator),
            "--model",
            str(args.model),
            "--seeds",
            ",".join(str(s) for s in seeds),
            "--input-root",
            str(in_root),
            "--canonical-output-root",
            str(canon_root),
            "--min-abs-diff",
            str(args.min_abs_diff),
            "--min-cohens-d",
            str(args.min_cohens_d),
            "--alpha-one-sided",
            str(args.alpha_one_sided),
            "--stable-blocked-min-flags",
            str(args.stable_blocked_min_flags),
            "--stable-clean-max-flags",
            str(args.stable_clean_max_flags),
            "--stable-blocked-meta-p",
            str(args.stable_blocked_meta_p),
            "--stable-clean-meta-p",
            str(args.stable_clean_meta_p),
        ]
        _run_subprocess(cmd)

        summary_path = canon_root / str(args.model) / "multiseed_gate_summary.json"
        summary = read_json(summary_path)
        domain_rows.append(
            {
                "domain": domain,
                "assessment": summary.get("assessment"),
                "meta_p_one": safe_float(summary.get("pooled", {}).get("meta_p_one")),
                "pooled_diff": safe_float(summary.get("pooled", {}).get("pooled_diff_fake_minus_real_with_prefix")),
                "n_flagged": int(summary.get("pooled", {}).get("n_flagged", 0)),
                "n_completed": int(summary.get("pooled", {}).get("n_completed", 0)),
                "summary_path": str(summary_path),
            }
        )

    report = {
        "timestamp": timestamp_now(),
        "experiment": "EXP-R1",
        "model": str(args.model),
        "domains": domains,
        "seeds": seeds,
        "domain_summaries": domain_rows,
        "output_root": str(out_root),
    }
    write_json(out_root / "domain_finalize_summary.json", report)
    print(f"[EXP-R1] wrote {out_root / 'domain_finalize_summary.json'}", flush=True)


def _weighted_stouffer(seed_rows: list[dict[str, Any]]) -> float:
    z_terms: list[float] = []
    w_terms: list[float] = []
    for row in seed_rows:
        p = safe_float(row.get("p_one_sided_fake_gt_real"))
        n_fake = max(1, int(row.get("n_fake_with_prefix", 1)))
        if not np.isfinite(p):
            continue
        p = min(max(float(p), 1e-12), 1.0 - 1e-12)
        z = float(scipy_stats.norm.isf(p))
        w = float(math.sqrt(n_fake))
        z_terms.append(z)
        w_terms.append(w)
    if not z_terms:
        return float("nan")
    num = float(np.sum(np.asarray(z_terms) * np.asarray(w_terms)))
    den = float(np.sqrt(np.sum(np.square(np.asarray(w_terms)))))
    if den <= 0:
        return float("nan")
    return float(scipy_stats.norm.sf(num / den))


def cmd_finalize_multidomain(args: argparse.Namespace) -> None:
    domains = parse_csv_strs(args.domains)
    out_root = Path(args.output_root)

    all_seed_rows: list[dict[str, Any]] = []
    domain_summaries: list[dict[str, Any]] = []

    for domain in domains:
        summary_path = out_root / "canonical" / f"domain_{domain}" / str(args.model) / "multiseed_gate_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing domain summary: {summary_path}")
        s = read_json(summary_path)
        domain_summaries.append(
            {
                "domain": domain,
                "assessment": s.get("assessment"),
                "summary_path": str(summary_path),
                "pooled": s.get("pooled", {}),
            }
        )
        for row in s.get("seed_rows", []):
            row2 = dict(row)
            row2["domain"] = domain
            all_seed_rows.append(row2)

    blocked_domains = sum(1 for d in domain_summaries if d.get("assessment") == "stable_blocked")
    clean_domains = sum(1 for d in domain_summaries if d.get("assessment") == "stable_clean")
    ambiguous_domains = sum(1 for d in domain_summaries if d.get("assessment") == "ambiguous")

    pooled_diff_vals = [
        safe_float(d.get("pooled", {}).get("pooled_diff_fake_minus_real_with_prefix"))
        for d in domain_summaries
    ]
    pooled_diff_vals = [x for x in pooled_diff_vals if np.isfinite(x)]
    pooled_diff_mean = float(np.mean(pooled_diff_vals)) if pooled_diff_vals else float("nan")

    meta_p_global = _weighted_stouffer([r for r in all_seed_rows if str(r.get("status", "complete")) == "complete"])

    if blocked_domains >= 2 and np.isfinite(meta_p_global) and meta_p_global < 0.01 and np.isfinite(pooled_diff_mean) and pooled_diff_mean >= float(args.min_abs_diff):
        assessment = "stable_blocked"
    elif clean_domains >= 2 and blocked_domains == 0 and np.isfinite(meta_p_global) and meta_p_global > 0.2 and np.isfinite(pooled_diff_mean) and pooled_diff_mean <= 0.0:
        assessment = "stable_clean"
    else:
        assessment = "ambiguous"

    summary = {
        "timestamp": timestamp_now(),
        "experiment": "EXP-R1",
        "model": str(args.model),
        "domains": domains,
        "domain_summaries": domain_summaries,
        "aggregate": {
            "n_domains": len(domains),
            "n_blocked": int(blocked_domains),
            "n_clean": int(clean_domains),
            "n_ambiguous": int(ambiguous_domains),
            "pooled_diff_mean": safe_float(pooled_diff_mean),
            "meta_p_one_weighted_stouffer": safe_float(meta_p_global),
        },
        "decision_rule": {
            "blocked": ">=2 domains stable_blocked and global meta_p<0.01 and pooled_diff_mean>=min_abs_diff",
            "clean": ">=2 domains stable_clean, none blocked, global meta_p>0.2, pooled_diff_mean<=0",
        },
        "assessment": assessment,
        "canonical_gate_recommendation": {
            "strict_gate_passes": bool(assessment == "stable_clean"),
            "strict_gate_blocked": bool(assessment == "stable_blocked"),
        },
    }

    out_path = out_root / "multidomain_gate_summary_v1.json"
    write_json(out_path, summary)
    print(f"[EXP-R1] wrote {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-R1: Llama strict-gate multidomain adjudication")
    sp = p.add_subparsers(dest="cmd", required=True)

    run_seed = sp.add_parser("run-seed", help="Run one seed across multiple domains")
    run_seed.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS.keys()))
    run_seed.add_argument("--seed", type=int, required=True)
    run_seed.add_argument("--domains", default="wiki,code,dialogue")
    run_seed.add_argument("--device", default="cuda:0")
    run_seed.add_argument("--num-sequences", type=int, default=48)
    run_seed.add_argument("--seq-len", type=int, default=512)
    run_seed.add_argument("--top-k-dims", type=int, default=16)
    run_seed.add_argument("--synthetic-target-per-cell", type=int, default=1000)
    run_seed.add_argument("--output-root", default=str(DEFAULT_OUT))

    fin_domain = sp.add_parser("finalize-domain", help="Run per-domain multiseed adjudication")
    fin_domain.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS.keys()))
    fin_domain.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9,10,11")
    fin_domain.add_argument("--domains", default="wiki,code,dialogue")
    fin_domain.add_argument("--output-root", default=str(DEFAULT_OUT))
    fin_domain.add_argument("--min-abs-diff", type=float, default=0.005)
    fin_domain.add_argument("--min-cohens-d", type=float, default=0.2)
    fin_domain.add_argument("--alpha-one-sided", type=float, default=0.05)
    fin_domain.add_argument("--stable-blocked-min-flags", type=int, default=9)
    fin_domain.add_argument("--stable-clean-max-flags", type=int, default=2)
    fin_domain.add_argument("--stable-blocked-meta-p", type=float, default=0.01)
    fin_domain.add_argument("--stable-clean-meta-p", type=float, default=0.2)

    fin_multi = sp.add_parser("finalize-multidomain", help="Aggregate domain adjudications")
    fin_multi.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS.keys()))
    fin_multi.add_argument("--domains", default="wiki,code,dialogue")
    fin_multi.add_argument("--output-root", default=str(DEFAULT_OUT))
    fin_multi.add_argument("--min-abs-diff", type=float, default=0.005)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "run-seed":
        cmd_run_seed(args)
        return
    if args.cmd == "finalize-domain":
        cmd_finalize_domain(args)
        return
    if args.cmd == "finalize-multidomain":
        cmd_finalize_multidomain(args)
        return
    raise ValueError(f"Unsupported cmd={args.cmd}")


if __name__ == "__main__":
    main()
