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
from reinforce_exp.exp_r1_multidomain_gate import (  # noqa: E402
    SUPPORTED_DOMAINS,
    _load_domain_sequences,
    _load_required_inputs,
    _run_one_domain,
)
from experiment3.theory5b_boundary_detection import MODELS  # noqa: E402
from shared.attention.adapters import get_adapter  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402

DEFAULT_OUT = RESULTS_ROOT / "exp_r2b_olmo_boundary_power"


def _run_subprocess(cmd: list[str]) -> None:
    print("[EXP-R2B] exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _weighted_stouffer(seed_rows: list[dict[str, Any]]) -> float:
    try:
        from scipy import stats as scipy_stats  # local import
    except Exception:
        return float("nan")
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


def _holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    finite = [(k, float(v)) for k, v in pvals.items() if np.isfinite(float(v))]
    if not finite:
        return {k: float("nan") for k in pvals}
    ordered = sorted(finite, key=lambda x: x[1])
    m = len(ordered)
    out: dict[str, float] = {k: float("nan") for k in pvals}
    running = 0.0
    for i, (k, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = float(min(1.0, running))
    return out


def _collect_domain_deltas(
    *,
    out_root: Path,
    model: str,
    domain: str,
    seeds: list[int],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        adv_path = out_root / f"domain_{domain}" / f"seed{seed}" / model / "adversarial_sequences.parquet"
        if not adv_path.exists():
            continue
        df = pd.read_parquet(adv_path)
        if df.empty:
            continue
        piv = (
            df.groupby(["sequence_idx", "cell"], as_index=False)["high_prev_attn"]
            .mean()
            .pivot(index="sequence_idx", columns="cell", values="high_prev_attn")
            .reset_index()
        )
        for col in ("fake_boundary_with_prefix", "real_boundary_with_prefix"):
            if col not in piv.columns:
                piv[col] = np.nan
        piv = piv.dropna(subset=["fake_boundary_with_prefix", "real_boundary_with_prefix"]).copy()
        if piv.empty:
            continue
        piv["delta"] = piv["fake_boundary_with_prefix"].astype(float) - piv["real_boundary_with_prefix"].astype(float)
        piv["seed"] = int(seed)
        piv["cluster_id"] = piv.apply(lambda r: f"s{int(r.seed)}_q{int(r.sequence_idx)}", axis=1)
        rows.append(piv[["seed", "sequence_idx", "cluster_id", "delta"]].copy())
    if not rows:
        return pd.DataFrame(columns=["seed", "sequence_idx", "cluster_id", "delta"])
    return pd.concat(rows, axis=0, ignore_index=True)


def _cluster_bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = int(values.size)
    boot = np.empty(max(1000, int(n_boot)), dtype=np.float64)
    for i in range(len(boot)):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(values[idx]))
    return float(np.mean(values)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _sign_permutation_p_one(values: np.ndarray, *, n_perm: int, seed: int) -> float:
    if values.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = float(np.mean(values))
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=(max(2000, int(n_perm)), values.size))
    perm_means = (signs * values.reshape(1, -1)).mean(axis=1)
    return float((np.sum(perm_means >= obs) + 1) / (len(perm_means) + 1))


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

        synth_path = Path(str(row["output_dir"])) / "synthetic_boundary_results.json"
        if synth_path.exists():
            synth = read_json(synth_path)
            assess = synth.get("prefix_following_assessment", {})
            row["n_fake_with_prefix"] = int(max(0, safe_float(assess.get("n_fake_with_prefix"))))
            row["n_real_with_prefix"] = int(max(0, safe_float(assess.get("n_real_with_prefix"))))
        else:
            row["n_fake_with_prefix"] = 0
            row["n_real_with_prefix"] = 0

        row["elapsed_sec"] = float(time.time() - d0)
        seed_rows.append(row)
        print(f"[EXP-R2B][seed={seed}] domain={domain} complete in {row['elapsed_sec']:.1f}s", flush=True)

    manifest = command_manifest(
        experiment_id="EXP-R2B",
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


def cmd_finalize_domain(args: argparse.Namespace) -> None:
    domains = parse_csv_strs(args.domains)
    seeds = parse_csv_ints(args.seeds)
    out_root = Path(args.output_root)
    ensure_dir(out_root)

    py = str(ROOT / ".venv" / "bin" / "python")
    adjudicator = ROOT / "experiment3" / "phase2" / "exp3p2b_multiseed_gate_adjudication.py"

    domain_rows: list[dict[str, Any]] = []
    perm_pvals: dict[str, float] = {}

    for idx, domain in enumerate(domains):
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
        adjudication = read_json(summary_path)
        pooled = adjudication.get("pooled", {})

        ddf = _collect_domain_deltas(out_root=out_root, model=str(args.model), domain=domain, seeds=seeds)
        delta = ddf["delta"].astype(float).to_numpy() if not ddf.empty else np.asarray([], dtype=float)
        mean_delta, ci_lo, ci_hi = _cluster_bootstrap_ci(delta, n_boot=int(args.n_boot), seed=int(args.seed) + idx * 17)
        p_perm = _sign_permutation_p_one(delta, n_perm=int(args.n_perm), seed=int(args.seed) + idx * 29)
        perm_pvals[domain] = p_perm
        cohens_d = float(np.mean(delta) / max(np.std(delta, ddof=1), 1e-8)) if len(delta) >= 2 else float("nan")

        artifact_positive = bool(int(pooled.get("n_flagged", 0)) > 0)
        practical_positive = bool(
            (np.isfinite(ci_lo) and ci_lo > 0.0) or (np.isfinite(mean_delta) and mean_delta >= float(args.min_abs_diff))
        )
        claim_direction_supported = bool(practical_positive and not artifact_positive)

        domain_rows.append(
            {
                "domain": domain,
                "adjudication_path": str(summary_path),
                "assessment": str(adjudication.get("assessment")),
                "n_completed": int(pooled.get("n_completed", 0)),
                "n_flagged": int(pooled.get("n_flagged", 0)),
                "meta_p_one": safe_float(pooled.get("meta_p_one")),
                "pooled_diff_fake_minus_real_with_prefix": safe_float(pooled.get("pooled_diff_fake_minus_real_with_prefix")),
                "cluster_n": int(len(delta)),
                "cluster_mean_delta": safe_float(mean_delta),
                "cluster_ci95": [safe_float(ci_lo), safe_float(ci_hi)],
                "cluster_cohens_d": safe_float(cohens_d),
                "p_one_permutation": safe_float(p_perm),
                "artifact_positive": artifact_positive,
                "practical_positive": practical_positive,
                "claim_direction_supported": claim_direction_supported,
            }
        )

    holm = _holm_adjust(perm_pvals)
    for row in domain_rows:
        dom = str(row["domain"])
        row["p_one_permutation_holm"] = safe_float(holm.get(dom))

    cluster_summary = {
        "timestamp": timestamp_now(),
        "experiment": "EXP-R2B",
        "model": str(args.model),
        "domains": domain_rows,
        "multiplicity": {
            "family": "domain_level_sequence_block_permutation",
            "method": "holm",
            "raw_p_one": {k: safe_float(v) for k, v in perm_pvals.items()},
            "holm_adjusted_p_one": {k: safe_float(v) for k, v in holm.items()},
        },
    }
    write_json(out_root / str(args.model) / "cluster_inference_summary.json", cluster_summary)

    n_supported = sum(1 for r in domain_rows if bool(r.get("claim_direction_supported", False)))
    n_artifact = sum(1 for r in domain_rows if bool(r.get("artifact_positive", False)))
    domain_summary = {
        "timestamp": timestamp_now(),
        "experiment": "EXP-R2B",
        "model": str(args.model),
        "domains": domain_rows,
        "summary": {
            "n_domains": int(len(domain_rows)),
            "n_domains_claim_supported": int(n_supported),
            "n_domains_with_artifact_signal": int(n_artifact),
            "claim_direction_supported": bool(n_supported >= 2),
        },
    }
    write_json(out_root / str(args.model) / "domain_summary.json", domain_summary)
    print(f"[EXP-R2B] wrote {out_root / str(args.model) / 'domain_summary.json'}", flush=True)


def cmd_finalize_claim(args: argparse.Namespace) -> None:
    out_root = Path(args.output_root)
    model = str(args.model)
    summary_path = out_root / model / "domain_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing domain summary: {summary_path}")
    summary = read_json(summary_path)
    domains = list(summary.get("domains", []))

    n_supported = sum(1 for r in domains if bool(r.get("claim_direction_supported", False)))
    n_artifact = sum(1 for r in domains if bool(r.get("artifact_positive", False)))
    pooled_diff_vals = [safe_float(r.get("pooled_diff_fake_minus_real_with_prefix")) for r in domains]
    pooled_diff_vals = [x for x in pooled_diff_vals if np.isfinite(x)]
    pooled_diff_mean = float(np.mean(pooled_diff_vals)) if pooled_diff_vals else float("nan")

    if n_supported >= 2 and n_artifact == 0:
        impact = "strengthens_olmo_boundary_nontriviality"
        boundary_claim_status = "supported_with_reinforcement"
    elif n_supported >= 1:
        impact = "mixed_reinforcement_keep_caveat"
        boundary_claim_status = "supported_with_caveat"
    else:
        impact = "does_not_reinforce_boundary_nontriviality"
        boundary_claim_status = "descriptive_caveated"

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment": "EXP-R2B",
        "model": model,
        "inputs": {
            "domain_summary_path": str(summary_path),
        },
        "headline": {
            "impact": impact,
            "boundary_claim_status": boundary_claim_status,
            "n_domains_supported": int(n_supported),
            "n_domains_artifact_positive": int(n_artifact),
            "pooled_diff_mean": safe_float(pooled_diff_mean),
        },
        "note": (
            "Claim status is derived from domain-level practical-positive direction under sequence-block "
            "inference and strict artifact checks."
        ),
    }
    write_json(out_root / model / "claim_impact.json", claim_impact)
    write_json(
        out_root / model / "manifest.json",
        command_manifest(
            experiment_id="EXP-R2B",
            command="finalize-claim",
            model=model,
            extras={
                "output_root": str(out_root),
                "boundary_claim_status": boundary_claim_status,
                "impact": impact,
            },
        ),
    )
    print(f"[EXP-R2B] wrote {out_root / model / 'claim_impact.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-R2B: OLMo strict boundary non-triviality power reinforcement")
    sp = p.add_subparsers(dest="cmd", required=True)

    run_seed = sp.add_parser("run-seed", help="Run one seed across wiki/code/dialogue")
    run_seed.add_argument("--model", default="olmo-2-7b", choices=sorted(MODELS.keys()))
    run_seed.add_argument("--seed", type=int, required=True)
    run_seed.add_argument("--domains", default="wiki,code,dialogue")
    run_seed.add_argument("--device", default="cuda:0")
    run_seed.add_argument("--num-sequences", type=int, default=64)
    run_seed.add_argument("--seq-len", type=int, default=512)
    run_seed.add_argument("--top-k-dims", type=int, default=16)
    run_seed.add_argument("--synthetic-target-per-cell", type=int, default=1200)
    run_seed.add_argument("--output-root", default=str(DEFAULT_OUT))

    fin_domain = sp.add_parser("finalize-domain", help="Run per-domain multiseed adjudication + cluster inference")
    fin_domain.add_argument("--model", default="olmo-2-7b", choices=sorted(MODELS.keys()))
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
    fin_domain.add_argument("--n-boot", type=int, default=5000)
    fin_domain.add_argument("--n-perm", type=int, default=5000)
    fin_domain.add_argument("--seed", type=int, default=13)

    fin_claim = sp.add_parser("finalize-claim", help="Produce claim impact from finalized domain summary")
    fin_claim.add_argument("--model", default="olmo-2-7b", choices=sorted(MODELS.keys()))
    fin_claim.add_argument("--output-root", default=str(DEFAULT_OUT))

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "run-seed":
        cmd_run_seed(args)
        return
    if args.cmd == "finalize-domain":
        cmd_finalize_domain(args)
        return
    if args.cmd == "finalize-claim":
        cmd_finalize_claim(args)
        return
    raise ValueError(f"Unsupported cmd={args.cmd}")


if __name__ == "__main__":
    main()
