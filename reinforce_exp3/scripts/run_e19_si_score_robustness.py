#!/usr/bin/env python3
"""E19 — SI Score Robustness to Offset Support and Corpus Composition.

Implements three robustness controls for Track-A SI scoring:
1) offset-support reweighting,
2) equal-count offset reweighting proxy,
3) cross-corpus kernel-transfer scoring.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    PRIMARY_MODELS,
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    enforce_coverage_contract,
    emit_core_artifacts,
    load_model_for_exp,
    parse_device_map,
    parse_models_arg,
    read_json,
)
from experiment1.shift_kernels import get_kernel_estimator  # noqa: E402
from experiment3.theory1_si_circuits import MODELS  # noqa: E402
from experiment5.pipeline import (  # noqa: E402
    _load_code_sequences,
    _load_dialogue_sequences,
    _load_wiki_sequences,
)
from shared.attention.adapters import get_adapter  # noqa: E402

OUT_ROOT = RESULTS_ROOT / "E19_si_score_robustness"
DOMAINS = ("wiki", "code", "dialogue")
MIN_DOMAIN_SEQS = 4


@dataclass
class HeadStats:
    sums: np.ndarray
    sq_sums: np.ndarray
    counts: np.ndarray


def _safe_r2(sse: float, sst: float) -> float:
    if not np.isfinite(sse) or not np.isfinite(sst) or sst <= 0.0:
        return float("nan")
    return float(max(0.0, 1.0 - (sse / sst)))


def _load_domain_sequences(
    *,
    domain: str,
    model_name: str,
    tokenizer: Any,
    seq_len: int,
    max_sequences: int,
    seed: int,
) -> list[list[int]]:
    if domain == "wiki":
        seqs = _load_wiki_sequences(
            model_name,
            tokenizer,
            seq_len=seq_len,
            max_sequences=max_sequences,
            seed=seed,
        )
    elif domain == "code":
        seqs = _load_code_sequences(
            model_name,
            tokenizer,
            seq_len=seq_len,
            max_sequences=max_sequences,
            seed=seed + 17,
        )
    elif domain == "dialogue":
        seqs = _load_dialogue_sequences(
            tokenizer,
            seq_len=seq_len,
            max_sequences=max_sequences,
            seed=seed + 31,
        )
    else:
        raise ValueError(f"Unsupported domain={domain}")
    return seqs[:max_sequences]


def _load_balanced_domain_sequences(
    *,
    model_name: str,
    tokenizer: Any,
    seq_len: int,
    max_sequences: int,
    seed: int,
) -> tuple[dict[str, list[list[int]]], dict[str, int], int]:
    """Load all domains, then truncate to a matched sequence count.

    Returns:
    - balanced sequences per domain,
    - raw loaded count per domain (before truncation),
    - matched count used for all domains.
    """
    raw: dict[str, list[list[int]]] = {}
    raw_counts: dict[str, int] = {}
    for didx, domain in enumerate(DOMAINS):
        seqs = _load_domain_sequences(
            domain=domain,
            model_name=model_name,
            tokenizer=tokenizer,
            seq_len=seq_len,
            max_sequences=max_sequences,
            seed=int(seed) + didx * 97,
        )
        raw[domain] = seqs
        raw_counts[domain] = int(len(seqs))
    n_balanced = min(raw_counts.values()) if raw_counts else 0
    if n_balanced < MIN_DOMAIN_SEQS:
        raise RuntimeError(
            f"[E19] insufficient matched domain sequences: "
            f"counts={raw_counts}, min_required={MIN_DOMAIN_SEQS}"
        )
    balanced = {domain: raw[domain][:n_balanced] for domain in DOMAINS}
    return balanced, raw_counts, int(n_balanced)


def _collect_offset_stats(
    *,
    model: Any,
    model_spec: Any,
    adapter: Any,
    sequences: list[list[int]],
    device: str,
    seq_len: int,
) -> tuple[dict[tuple[int, int], HeadStats], int, int]:
    dmax = max(1, int(seq_len) - 1)
    stats: dict[tuple[int, int], HeadStats] = {}
    n_layers = -1
    n_heads = -1

    for sidx, seq in enumerate(sequences):
        ids = torch.tensor([seq[:seq_len]], dtype=torch.long, device=device)
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
            continue
        logits = cap.logits.float().numpy()  # [L,H,S,S]
        n_layers, n_heads = int(logits.shape[0]), int(logits.shape[1])
        s = int(logits.shape[2])
        d_eff = min(dmax, s - 1)
        for l in range(n_layers):
            for h in range(n_heads):
                key = (l, h)
                if key not in stats:
                    stats[key] = HeadStats(
                        sums=np.zeros(dmax, dtype=np.float64),
                        sq_sums=np.zeros(dmax, dtype=np.float64),
                        counts=np.zeros(dmax, dtype=np.float64),
                    )
                arr = logits[l, h]
                hs = stats[key]
                for d in range(1, d_eff + 1):
                    diag = np.diagonal(arr, offset=-d).astype(np.float64)
                    if diag.size == 0:
                        continue
                    idx = d - 1
                    hs.sums[idx] += float(np.sum(diag))
                    hs.sq_sums[idx] += float(np.sum(diag * diag))
                    hs.counts[idx] += float(diag.size)
        del ids, cap
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (sidx + 1) % 10 == 0:
            print(f"[E19] collected {sidx+1}/{len(sequences)} sequences", flush=True)

    if n_layers <= 0 or n_heads <= 0:
        raise RuntimeError("[E19] failed to collect logits/offset stats")
    return stats, n_layers, n_heads


def _baseline_fit(stats: HeadStats, estimator: Any) -> tuple[float, np.ndarray]:
    t = estimator.fit_from_stats(
        sums=torch.tensor(stats.sums, dtype=torch.float64),
        sq_sums=torch.tensor(stats.sq_sums, dtype=torch.float64),
        counts=torch.tensor(stats.counts, dtype=torch.float64),
    )
    g = np.asarray(t.g_values, dtype=np.float64)
    return float(t.r2), g


def _weighted_stats_r2(stats: HeadStats, estimator: Any, *, mode: str) -> float:
    """Compute weighted SI-R2 under the same token-level SSE/SST objective.

    This keeps the scoring objective comparable to baseline fit_from_stats.
    """
    c = np.asarray(stats.counts, dtype=np.float64)
    mask = c > 0
    if not np.any(mask):
        return float("nan")

    if mode == "inverse_sqrt_support":
        c_ref = float(np.mean(c[mask]))
        w = np.zeros_like(c, dtype=np.float64)
        w[mask] = np.sqrt(c_ref / c[mask])
    elif mode == "equal_count":
        cmin = float(np.min(c[mask]))
        if not np.isfinite(cmin) or cmin <= 0:
            return float("nan")
        w = np.zeros_like(c, dtype=np.float64)
        w[mask] = cmin / c[mask]
    else:
        raise ValueError(f"unsupported mode={mode}")

    sums_w = stats.sums * w
    sq_w = stats.sq_sums * w
    cnt_w = stats.counts * w

    t = estimator.fit_from_stats(
        sums=torch.tensor(sums_w, dtype=torch.float64),
        sq_sums=torch.tensor(sq_w, dtype=torch.float64),
        counts=torch.tensor(cnt_w, dtype=torch.float64),
    )
    return float(t.r2)


def _offset_reweighted_r2(stats: HeadStats, estimator: Any) -> float:
    # Offset-support reweighting: soften frequency imbalance by inverse-sqrt support.
    return _weighted_stats_r2(stats, estimator, mode="inverse_sqrt_support")


def _equal_count_reweight_proxy_r2(stats: HeadStats, estimator: Any) -> float:
    # Equal-count proxy: full count balancing across occupied offset bins.
    return _weighted_stats_r2(stats, estimator, mode="equal_count")


def _transfer_r2(source_g: np.ndarray, target_stats: HeadStats) -> float:
    g = np.asarray(source_g, dtype=np.float64)
    d = min(len(g), len(target_stats.sums))
    if d <= 0:
        return float("nan")
    sums = target_stats.sums[:d]
    sq = target_stats.sq_sums[:d]
    cnt = target_stats.counts[:d]

    sse_terms = sq - 2.0 * g[:d] * sums + cnt * (g[:d] ** 2)
    sse = float(np.sum(np.where(cnt > 0, sse_terms, 0.0)))

    total_sum = float(np.sum(sums))
    total_sq = float(np.sum(sq))
    total_cnt = float(np.sum(cnt))
    if total_cnt <= 0:
        return float("nan")
    mu = total_sum / total_cnt
    sst = float(total_sq - total_cnt * (mu ** 2))
    return _safe_r2(sse, sst)


def run_model(
    *,
    model_name: str,
    device: str,
    out_root: Path,
    seq_len: int,
    num_sequences: int,
    seed: int,
) -> dict[str, Any]:
    t0 = time.time()
    print(f"[E19] Loading {model_name} on {device}", flush=True)
    model, tokenizer = load_model_for_exp(model_name, device, attn_implementation="eager")
    model_spec = MODELS[model_name]
    estimator = get_kernel_estimator(model_spec.pe_scheme)

    adapter = get_adapter(model_spec)
    adapter.register(model)

    domain_stats: dict[str, dict[tuple[int, int], HeadStats]] = {}
    domain_baseline: dict[str, dict[tuple[int, int], float]] = {}
    domain_reweighted: dict[str, dict[tuple[int, int], float]] = {}
    domain_equalcount: dict[str, dict[tuple[int, int], float]] = {}
    domain_g: dict[str, dict[tuple[int, int], np.ndarray]] = {}

    n_layers = 0
    n_heads = 0

    seq_len_use = max(64, int(seq_len))
    req_n = max(8, int(num_sequences))
    balanced_seqs, raw_counts, n_balanced = _load_balanced_domain_sequences(
        model_name=model_name,
        tokenizer=tokenizer,
        seq_len=seq_len_use,
        max_sequences=req_n,
        seed=int(seed),
    )
    print(f"[E19] {model_name}: domain raw counts={raw_counts}, balanced_n={n_balanced}", flush=True)

    for domain in DOMAINS:
        seqs = balanced_seqs[domain]

        stats_map, nl, nh = _collect_offset_stats(
            model=model,
            model_spec=model_spec,
            adapter=adapter,
            sequences=seqs,
            device=device,
            seq_len=seq_len_use,
        )
        n_layers = max(n_layers, nl)
        n_heads = max(n_heads, nh)
        domain_stats[domain] = stats_map

        bmap: dict[tuple[int, int], float] = {}
        rwmap: dict[tuple[int, int], float] = {}
        eqmap: dict[tuple[int, int], float] = {}
        gmap: dict[tuple[int, int], np.ndarray] = {}
        for key, hs in stats_map.items():
            r2_base, g = _baseline_fit(hs, estimator)
            bmap[key] = r2_base
            rwmap[key] = _offset_reweighted_r2(hs, estimator)
            eqmap[key] = _equal_count_reweight_proxy_r2(hs, estimator)
            gmap[key] = g

        domain_baseline[domain] = bmap
        domain_reweighted[domain] = rwmap
        domain_equalcount[domain] = eqmap
        domain_g[domain] = gmap
        print(f"[E19] {model_name} domain={domain} heads={len(bmap)}", flush=True)

    # Build per-head robustness table.
    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for key, r2 in domain_baseline[domain].items():
            rows.append(
                {
                    "model": model_name,
                    "domain": domain,
                    "layer": int(key[0]),
                    "head": int(key[1]),
                    "r2_baseline": float(r2),
                    "r2_offset_reweighted": float(domain_reweighted[domain].get(key, float("nan"))),
                    "r2_equalcount_reweight_proxy": float(domain_equalcount[domain].get(key, float("nan"))),
                }
            )
    df = pd.DataFrame(rows)

    # Cross-corpus kernel transfer: fit g on source, evaluate on target stats.
    trows: list[dict[str, Any]] = []
    for src in DOMAINS:
        for tgt in DOMAINS:
            if src == tgt:
                continue
            vals: list[float] = []
            tgt_base_vals: list[float] = []
            common = set(domain_g[src].keys()) & set(domain_stats[tgt].keys())
            for key in common:
                r2_t = _transfer_r2(domain_g[src][key], domain_stats[tgt][key])
                if np.isfinite(r2_t):
                    vals.append(float(r2_t))
                r2_base_t = domain_baseline[tgt].get(key, float("nan"))
                if np.isfinite(r2_base_t):
                    tgt_base_vals.append(float(r2_base_t))
            mean_transfer = float(np.nanmean(np.asarray(vals, dtype=float))) if vals else float("nan")
            mean_target_base = float(np.nanmean(np.asarray(tgt_base_vals, dtype=float))) if tgt_base_vals else float("nan")
            trows.append(
                {
                    "model": model_name,
                    "source_domain": src,
                    "target_domain": tgt,
                    "n_common_heads": int(len(common)),
                    "mean_transfer_r2": mean_transfer,
                    "mean_target_in_domain_r2": mean_target_base,
                    "transfer_minus_in_domain": float(mean_transfer - mean_target_base) if np.isfinite(mean_transfer) and np.isfinite(mean_target_base) else float("nan"),
                }
            )
    tdf = pd.DataFrame(trows)

    # Model-level summaries.
    per_domain = []
    for domain in DOMAINS:
        sub = df[df["domain"] == domain]
        per_domain.append(
            {
                "domain": domain,
                "n_sequences_used": int(len(balanced_seqs[domain])),
                "n_sequences_loaded_raw": int(raw_counts.get(domain, 0)),
                "mean_r2_baseline": float(sub["r2_baseline"].mean()),
                "mean_r2_offset_reweighted": float(sub["r2_offset_reweighted"].mean()),
                "mean_r2_equalcount_proxy": float(sub["r2_equalcount_reweight_proxy"].mean()),
                "delta_offset_reweighted": float(sub["r2_offset_reweighted"].mean() - sub["r2_baseline"].mean()),
                "delta_equalcount_proxy": float(sub["r2_equalcount_reweight_proxy"].mean() - sub["r2_baseline"].mean()),
            }
        )

    mean_abs_delta_reweight = float(np.nanmean(np.abs(np.asarray([x["delta_offset_reweighted"] for x in per_domain], dtype=float))))
    mean_abs_delta_equal = float(np.nanmean(np.abs(np.asarray([x["delta_equalcount_proxy"] for x in per_domain], dtype=float))))
    mean_abs_transfer_drop = float(np.nanmean(np.abs(tdf["transfer_minus_in_domain"].to_numpy(dtype=float)))) if len(tdf) else float("nan")

    model_dir = ensure_dir(out_root / model_name)
    df.to_parquet(model_dir / "si_r2_robustness_per_head.parquet", index=False)
    tdf.to_parquet(model_dir / "cross_corpus_transfer.parquet", index=False)

    # Stability heuristic: rank ordering robust if deltas are small.
    robust = bool(
        np.isfinite(mean_abs_delta_reweight)
        and np.isfinite(mean_abs_delta_equal)
        and np.isfinite(mean_abs_transfer_drop)
        and mean_abs_delta_reweight <= 0.08
        and mean_abs_delta_equal <= 0.08
        and mean_abs_transfer_drop <= 0.12
    )

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E19",
        "model": model_name,
        "runtime_sec": float(time.time() - t0),
        "num_domains": int(len(DOMAINS)),
        "num_sequences_per_domain_requested": int(num_sequences),
        "num_sequences_per_domain_actual": int(n_balanced),
        "domain_raw_counts": {k: int(v) for k, v in raw_counts.items()},
        "seq_len": int(seq_len_use),
        "mean_abs_delta_offset_reweight": mean_abs_delta_reweight,
        "mean_abs_delta_equalcount_proxy": mean_abs_delta_equal,
        "mean_abs_transfer_minus_indomain": mean_abs_transfer_drop,
        "robustness_pass": robust,
        "per_domain": per_domain,
    }
    write_json(model_dir / "summary.json", summary)

    try:
        adapter.cleanup()
    except Exception:
        pass

    print(
        f"[E19] {model_name}: reweight={mean_abs_delta_reweight:.4f} "
        f"equal_proxy={mean_abs_delta_equal:.4f} transfer={mean_abs_transfer_drop:.4f} pass={robust}",
        flush=True,
    )
    return summary


def _load_model_summary(model_name: str, out_root: Path) -> dict[str, Any]:
    p = out_root / model_name / "summary.json"
    if not p.exists():
        raise RuntimeError(f"[E19] hard_fail_reason: missing shard artifact {p}")
    return read_json(p)


def _finalize(models: list[str], out_root: Path, start_ts: str) -> dict[str, Any]:
    rows = [_load_model_summary(m, out_root) for m in models]
    enforce_coverage_contract(
        experiment_id="E19",
        observed_models=[r.get("model", "") for r in rows],
        required_models=models,
    )

    n_pass = int(sum(1 for r in rows if bool(r.get("robustness_pass", False))))
    mean_rew = float(np.nanmean(np.asarray([r.get("mean_abs_delta_offset_reweight", float("nan")) for r in rows], dtype=float)))
    mean_eq = float(np.nanmean(np.asarray([r.get("mean_abs_delta_equalcount_proxy", float("nan")) for r in rows], dtype=float)))
    mean_tr = float(np.nanmean(np.asarray([r.get("mean_abs_transfer_minus_indomain", float("nan")) for r in rows], dtype=float)))

    if n_pass == len(rows):
        claim_status = "supported"
        interp = "robust_cross_model"
        note = "Cross-model SI contrasts remain stable under offset-reweighting and transfer robustness checks."
    elif n_pass >= max(1, len(rows) - 1):
        claim_status = "supported_with_caveat"
        interp = "mostly_robust"
        note = "SI contrasts are mostly stable; one model shows larger sensitivity to support/corpus controls."
    elif n_pass >= 1:
        claim_status = "mixed"
        interp = "partial_robustness"
        note = "Robustness to offset/corpus controls is model-conditional."
    else:
        claim_status = "not_supported"
        interp = "fragile"
        note = "SI contrasts shift materially under offset-support and transfer controls."

    cross = {
        "timestamp": timestamp_now(),
        "experiment_id": "E19",
        "n_models": int(len(rows)),
        "n_pass": int(n_pass),
        "mean_abs_delta_offset_reweight": mean_rew,
        "mean_abs_delta_equalcount_proxy": mean_eq,
        "mean_abs_transfer_minus_indomain": mean_tr,
        "interpretation": interp,
        "note": note,
        "per_model": rows,
    }
    write_json(out_root / "cross_model_si_robustness_summary.json", cross)

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "E19",
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "supported_with_caveat", "mixed"),
        "notes": [note],
        "outcome_summary": note,
    }

    prereg = {
        "experiment_id": "E19",
        "question": "Are SI-score contrasts robust to offset-support imbalance and corpus-composition shifts?",
        "primary_hypothesis": "Model-level SI contrasts remain stable under offset reweighting and cross-corpus transfer.",
        "primary_endpoints": [
            "mean_abs_delta_offset_reweight",
            "mean_abs_delta_equalcount_proxy",
            "mean_abs_transfer_minus_indomain",
        ],
        "secondary_endpoints": ["per_domain_delta_profiles", "per_model_pass_flag"],
        "model_list": models,
        "dataset_sources": ["wiki", "code", "dialogue"],
        "inclusion_exclusion_rules": [
            "Use matched sequence count per domain (truncate to min loaded count)",
            "Require >=4 matched sequences per domain",
        ],
        "sample_size_plan": {
            "domains": list(DOMAINS),
            "sequences_per_domain": "configured by --num-sequences",
        },
        "seed_plan": {"base_seed": 20260501},
        "stopping_rule": "fixed sample size",
        "multiplicity_family": ["cross-domain robustness deltas"],
        "acceptance_criteria": [
            "mean absolute deltas below preregistered practical thresholds across models",
        ],
        "fallback_interpretation_if_null": "Cross-model SI contrasts may partly reflect support/corpus artifacts and should be treated cautiously.",
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E19",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {"interpretation": interp, "n_pass": int(n_pass), "n_models": int(len(rows))},
        "limitations": [
            "Equal-count component uses reweighting proxy rather than full raw-diagonal bootstrap resampling.",
            "Cross-corpus transfer uses fixed-kernel evaluation and does not isolate all tokenizer-domain confounds.",
        ],
    }

    data_dictionary = {
        "experiment_id": "E19",
        "tables": [
            {
                "path": "<model>/si_r2_robustness_per_head.parquet",
                "description": "Per-head baseline and reweighted SI R2 scores by domain.",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "model"},
                    {"name": "domain", "dtype": "str", "description": "domain"},
                    {"name": "layer", "dtype": "int", "description": "layer"},
                    {"name": "head", "dtype": "int", "description": "head"},
                    {"name": "r2_baseline", "dtype": "float", "description": "baseline SI R2"},
                    {"name": "r2_offset_reweighted", "dtype": "float", "description": "offset-reweighted SI R2"},
                    {"name": "r2_equalcount_reweight_proxy", "dtype": "float", "description": "equal-count reweight proxy SI R2"},
                ],
            },
            {
                "path": "<model>/cross_corpus_transfer.parquet",
                "description": "Source->target kernel-transfer SI scoring summary.",
                "columns": [
                    {"name": "source_domain", "dtype": "str", "description": "source domain"},
                    {"name": "target_domain", "dtype": "str", "description": "target domain"},
                    {"name": "n_common_heads", "dtype": "int", "description": "heads used"},
                    {"name": "mean_transfer_r2", "dtype": "float", "description": "transfer R2"},
                    {"name": "mean_target_in_domain_r2", "dtype": "float", "description": "in-domain R2"},
                    {"name": "transfer_minus_in_domain", "dtype": "float", "description": "transfer - in-domain"},
                ],
            },
        ],
    }

    emit_core_artifacts(
        experiment_id="E19",
        out_dir=out_root,
        preregistration=prereg,
        manifest_extra={"models": models, "domains": list(DOMAINS)},
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross


def main() -> None:
    p = argparse.ArgumentParser(description="E19: SI score robustness controls", allow_abbrev=False)
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0",
    )
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--num-sequences", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E19] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    device_map = parse_device_map(args.device_map)
    out_root = ensure_dir(Path(args.output_root))
    start_ts = timestamp_now()

    seq_len = int(args.seq_len)
    n_seq = int(args.num_sequences)
    if args.smoke:
        seq_len = min(seq_len, 256)
        n_seq = min(n_seq, 12)

    if args.finalize_only:
        cross = _finalize(models, out_root, start_ts)
        print(f"[E19] Finalized from shards. interpretation={cross['interpretation']}", flush=True)
        return

    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        run_model(
            model_name=model_name,
            device=device,
            out_root=out_root,
            seq_len=max(64, seq_len),
            num_sequences=max(8, n_seq),
            seed=int(args.seed),
        )

    if args.no_finalize:
        print("[E19] Shard run complete (no finalize).", flush=True)
        return

    cross = _finalize(models, out_root, start_ts)
    print(f"[E19] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
