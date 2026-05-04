#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp2.common import RESULTS_ROOT, command_manifest, ensure_dir, timestamp_now, write_json  # noqa: E402
from reinforce_exp2.scripts._carrier_sets import (  # noqa: E402
    build_static_carrier_sets,
    dump_alignment_table,
    dump_carrier_diagnostics,
    sequence_hash,
)
from reinforce_exp2.scripts._shared import emit_core_artifacts, holm_adjust_dict, parse_models_arg, safe_float  # noqa: E402
from experiment2.execution import _evaluate_example_from_token_logits  # noqa: E402
from experiment2.tasks import TaskExample, build_token_pools, generate_task_examples  # noqa: E402
from experiment3.theory1_si_circuits import MODELS, HeadID, head_output_ablation  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402

INTERACTION_ETA_MIN = 0.06


def _stable_hash_mod(text: str, mod: int) -> int:
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16], 16) % max(1, int(mod))


def _all_heads(model) -> list[HeadID]:
    heads = []
    n_layers = int(len(model.model.layers))
    for l in range(n_layers):
        attn = model.model.layers[l].self_attn
        n_heads = None
        for attr in ("num_heads", "n_heads", "num_attention_heads"):
            if hasattr(attn, attr):
                try:
                    n_heads = int(getattr(attn, attr))
                    break
                except Exception:
                    pass
        if n_heads is None and hasattr(model, "config") and hasattr(model.config, "num_attention_heads"):
            try:
                n_heads = int(model.config.num_attention_heads)
            except Exception:
                n_heads = None
        if n_heads is None and hasattr(attn, "q_proj") and hasattr(attn.q_proj, "out_features"):
            try:
                head_dim = int(getattr(attn, "head_dim", 0) or 0)
                if head_dim <= 0 and hasattr(model, "config") and hasattr(model.config, "hidden_size") and hasattr(model.config, "num_attention_heads"):
                    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
                n_heads = int(attn.q_proj.out_features // max(head_dim, 1))
            except Exception:
                n_heads = None
        if n_heads is None or n_heads <= 0:
            raise RuntimeError(f"Unable to infer num heads for layer {l} ({type(attn).__name__})")
        for h in range(n_heads):
            heads.append(HeadID(int(l), int(h)))
    return heads


def _permute_prequery_tokens(ex: TaskExample, rng: np.random.Generator) -> TaskExample:
    toks = list(ex.tokens)
    targets = sorted(int(x) for x in ex.target_positions)
    if not targets:
        return ex
    q = int(min(targets))
    if q <= 2:
        return ex
    idx = np.arange(0, q, dtype=np.int64)
    perm = rng.permutation(idx)
    prefix = [toks[int(i)] for i in perm.tolist()]
    new_toks = prefix + toks[q:]
    return TaskExample(
        id=str(ex.id) + ":perm",
        task_name=str(ex.task_name),
        tokens=[int(x) for x in new_toks],
        target_positions=list(ex.target_positions),
        target_tokens=list(ex.target_tokens),
        dependency_span=int(ex.dependency_span),
        task_class=str(ex.task_class),
        seed=int(ex.seed),
        model=str(ex.model),
        length=int(ex.length),
        task_params=dict(ex.task_params),
        pair_count=ex.pair_count,
        query_key=ex.query_key,
        distractor_key=ex.distractor_key,
        match_rule=str(ex.match_rule),
        has_no_match=bool(ex.has_no_match),
    )


def _evaluate_examples(
    model,
    device: str,
    examples: list[TaskExample],
    pools,
    candidate_size: int,
    batch_size: int,
) -> np.ndarray:
    vals: list[float] = []
    bs = max(1, int(batch_size))
    pos = 0
    while pos < len(examples):
        batch = examples[pos : pos + bs]
        try:
            input_ids = torch.tensor([ex.tokens for ex in batch], dtype=torch.long, device=device)
            with torch.inference_mode():
                logits = model(input_ids=input_ids, use_cache=False).logits
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                continue
            raise

        for i, ex in enumerate(batch):
            met, _ = _evaluate_example_from_token_logits(
                logits[i],
                ex,
                split="synthetic",
                pools=pools,
                synthetic_eval_mode="restricted",
                candidate_size=int(candidate_size),
                candidate_policy_version="restricted_candidates_v1_structured_first",
            )
            vals.append(float(met["accuracy"]))

        pos += len(batch)
        del input_ids, logits
        torch.cuda.empty_cache()

    return np.asarray(vals, dtype=np.float64)


def _rr(a_patch: np.ndarray, a_clean: np.ndarray, a_corrupt: np.ndarray) -> np.ndarray:
    num = a_patch - a_corrupt
    den = np.maximum(a_clean - a_corrupt, 1e-6)
    out = num / den
    out = np.clip(out, -1.0, 2.0)
    return out


def _bootstrap_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    m = max(2000, int(n_boot))
    boots = np.empty(m, dtype=np.float64)
    for i in range(m):
        idx = rng.integers(0, len(v), size=len(v))
        boots[i] = float(np.mean(v[idx]))
    return float(np.mean(v)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _interaction_report(df: pd.DataFrame) -> dict[str, Any]:
    work = df.copy()
    work = work[np.isfinite(work["rr"].astype(float).values)]
    if work.empty:
        return {"status": "empty"}

    models = sorted(work["model"].astype(str).unique().tolist())
    sets = sorted(work["head_set"].astype(str).unique().tolist())
    means_m = {m: float(work[work["model"] == m]["rr"].mean()) for m in models}
    means_s = {s: float(work[work["head_set"] == s]["rr"].mean()) for s in sets}
    grand = float(work["rr"].mean())

    ss_ab = 0.0
    ss_w = 0.0
    for m in models:
        for s in sets:
            cell = work[(work["model"] == m) & (work["head_set"] == s)]
            if len(cell) == 0:
                continue
            cm = float(cell["rr"].mean())
            ss_ab += len(cell) * ((cm - means_m[m] - means_s[s] + grand) ** 2)
            ss_w += float(np.sum((cell["rr"].to_numpy(dtype=np.float64) - cm) ** 2))

    a, b = len(models), len(sets)
    n = len(work)
    df_ab = max(1, (a - 1) * (b - 1))
    df_w = max(1, n - a * b)
    ms_ab = ss_ab / df_ab
    ms_w = ss_w / df_w
    f = float(ms_ab / ms_w) if ms_w > 0 else float("nan")
    p = float(1.0 - scipy_stats.f.cdf(f, df_ab, df_w)) if np.isfinite(f) else float("nan")
    eta = float(ss_ab / max(ss_ab + ss_w, 1e-12))

    return {
        "f_interaction": f,
        "p_value": p,
        "partial_eta_squared": eta,
        "supports_interaction": bool(np.isfinite(p) and p < 0.05 and np.isfinite(eta) and abs(eta) >= INTERACTION_ETA_MIN),
        "interaction_eta_threshold": float(INTERACTION_ETA_MIN),
    }


def _welch_ttest_one_sided_gt(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    if aa.size < 2 or bb.size < 2:
        return float("nan"), float("nan")
    ma = float(np.mean(aa))
    mb = float(np.mean(bb))
    va = float(np.var(aa, ddof=1))
    vb = float(np.var(bb, ddof=1))
    se2 = float(va / max(int(aa.size), 1) + vb / max(int(bb.size), 1))
    if not np.isfinite(se2) or se2 <= 0:
        return float("nan"), float("nan")
    t = float((ma - mb) / math.sqrt(se2))
    num = se2 * se2
    den = (va * va) / (max(int(aa.size), 1) ** 2 * max(int(aa.size) - 1, 1)) + (vb * vb) / (max(int(bb.size), 1) ** 2 * max(int(bb.size) - 1, 1))
    dof = float(num / max(den, 1e-12))
    p_one = float(scipy_stats.t.sf(t, dof)) if np.isfinite(t) and np.isfinite(dof) and dof > 0 else float("nan")
    return t, p_one


def run_model(
    *,
    model_name: str,
    device: str,
    seq_len: int,
    n_examples: int,
    seed: int,
    batch_size: int,
    candidate_size: int,
    n_boot: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], bool]:
    if device == "auto":
        loaded = load_model(MODELS[model_name], device_map="auto")
        model = loaded.model
        device = str(next(model.parameters()).device)
    else:
        loaded = load_model(MODELS[model_name])
        model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(MODELS[model_name])

    special_ids = [getattr(tokenizer, attr, None) for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")]
    special_ids = [int(x) for x in special_ids if x is not None]
    pools = build_token_pools(model_name, int(tokenizer.vocab_size), special_ids)

    phenoms = [
        ("copy_offset", "long_range_retrieval", 64),
        ("indexed_retrieval", "local_key_match", None),
        ("controlled_permutation", "local_copy_offset", None),
    ]

    exs_by_phenom: list[tuple[str, list[TaskExample]]] = []
    eval_hashes: set[str] = set()
    for pidx, (phenom, task_name, span_override) in enumerate(phenoms):
        try:
            exs = generate_task_examples(
                task_name=task_name,
                model_name=model_name,
                seq_len=max(128, int(seq_len)),
                seed=int(seed) + pidx * 101,
                count=max(8, int(n_examples)),
                pools=pools,
                span_override=span_override,
                span_choices=(int(span_override),) if span_override is not None else None,
            )
        except Exception:
            exs = generate_task_examples(
                task_name="local_key_match",
                model_name=model_name,
                seq_len=max(128, int(seq_len)),
                seed=int(seed) + pidx * 101,
                count=max(8, int(n_examples)),
                pools=pools,
            )
        for ex in exs:
            eval_hashes.add(sequence_hash([int(x) for x in ex.tokens]))
        exs_by_phenom.append((phenom, exs))

    carrier = build_static_carrier_sets(
        model_name=model_name,
        tokenizer=tokenizer,
        pools=pools,
        seq_len=max(128, int(seq_len)),
        seed=int(seed) + 5003,
        eval_sequence_hashes=eval_hashes,
        n_calibration=96,
        max_pairs_per_sequence=4000,
        n_offset_bins=8,
        data_quality_max_missing=0.05,
    )

    dump_alignment_table(out_dir / f"{model_name}_content_alignment.parquet", carrier["alignment_table"])
    dump_carrier_diagnostics(out_dir / f"{model_name}_carrier_set_diagnostics.json", carrier["diagnostics"])

    overlap_forces_exploratory = not bool(carrier["diagnostics"].get("calibration_eval_disjoint", True))

    all_heads = _all_heads(model)
    set_map = {
        "SI": carrier["SI"],
        "LowSI": carrier["LowSI"],
        "ContentCond": carrier["ContentCond"],
    }

    rows = []
    rr_summary = {
        "model": model_name,
        "sets": {},
        "phenomena": {},
        "carrier_set_diagnostics": carrier["diagnostics"],
    }

    for pidx, (phenom, exs) in enumerate(exs_by_phenom):
        rng = np.random.default_rng(int(seed) + pidx * 1009)
        corr = [_permute_prequery_tokens(ex, rng) for ex in exs]

        a_clean = _evaluate_examples(model, device, exs, pools, candidate_size=candidate_size, batch_size=batch_size)
        a_cor = _evaluate_examples(model, device, corr, pools, candidate_size=candidate_size, batch_size=batch_size)
        gap = a_clean - a_cor
        confirm_mask = np.asarray(gap >= 0.05, dtype=bool)

        rr_summary["phenomena"][phenom] = {
            "n_examples": int(len(exs)),
            "n_confirmatory_gap_examples": int(np.sum(confirm_mask)),
            "clean_mean": float(np.nanmean(a_clean)),
            "corrupt_mean": float(np.nanmean(a_cor)),
            "clean_minus_corrupt": float(np.nanmean(gap)),
        }

        for sname, keep_heads in set_map.items():
            keep_keys = {f"L{int(h.layer)}H{int(h.head)}" for h in keep_heads}
            ablate = [h for h in all_heads if f"L{int(h.layer)}H{int(h.head)}" not in keep_keys]
            with head_output_ablation(model, ablate):
                a_patch = _evaluate_examples(model, device, corr, pools, candidate_size=candidate_size, batch_size=batch_size)
            rr_vals = _rr(a_patch, a_clean, a_cor)

            for i in range(len(rr_vals)):
                rows.append(
                    {
                        "model": model_name,
                        "phenomenon": phenom,
                        "head_set": sname,
                        "example_idx": int(i),
                        "a_clean": safe_float(a_clean[i]),
                        "a_corrupt": safe_float(a_cor[i]),
                        "a_patch": safe_float(a_patch[i]),
                        "gap_clean_minus_corrupt": safe_float(gap[i]),
                        "confirmatory_gap_eligible": bool(confirm_mask[i]),
                        "rr": safe_float(rr_vals[i]),
                    }
                )

    rdf = pd.DataFrame(rows)

    set_stats = {}
    for sname, sdf in rdf.groupby("head_set"):
        cdf = sdf[sdf["confirmatory_gap_eligible"]].copy()
        vals = cdf["rr"].to_numpy(dtype=np.float64)
        mu, lo, hi = _bootstrap_ci(vals, n_boot=n_boot, seed=seed + _stable_hash_mod(sname, 1000))
        set_stats[sname] = {
            "rr_mean": mu,
            "rr_ci95": [lo, hi],
            "n": int(len(vals)),
            "n_all": int(sdf.shape[0]),
        }

    rr_summary["sets"] = set_stats

    head_set_report = {
        "model": model_name,
        "set_sizes": {k: int(len(v)) for k, v in set_map.items()},
        "content_conditional_definition": "mean_r2<median, entropy_proxy>layer_median, alignment>T (T=Q75 low_r2 alignment)",
    }

    return rdf, rr_summary, head_set_report, overlap_forces_exploratory


def main() -> None:
    p = argparse.ArgumentParser(description="C1 OLMo causal trace / two-carrier-class test", allow_abbrev=False)
    p.add_argument("--models", default="llama-3.1-8b,olmo-2-7b")
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-examples", type=int, default=1024)
    p.add_argument("--min-confirmatory-rows-per-phenomenon", type=int, default=2000)
    p.add_argument("--candidate-size", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260417)
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "C1_olmo_causal_trace"))
    args = p.parse_args()

    models = parse_models_arg(args.models, default=("llama-3.1-8b", "olmo-2-7b"))
    device_map = {}
    for tok in [x.strip() for x in str(args.device_map).split(",") if x.strip()]:
        if ":" in tok:
            m, d = tok.split(":", 1)
            device_map[m.strip()] = d.strip()

    out_dir = ensure_dir(Path(args.output_root))

    dfs = []
    model_reports = {}
    set_reports = {}
    overlap_flags = {}
    for i, model in enumerate(models):
        device = device_map.get(model, "cuda:0")
        rdf, rr_summary, hs, overlap_flag = run_model(
            model_name=model,
            device=device,
            seq_len=max(128, int(args.seq_len)),
            n_examples=max(64, int(args.n_examples)),
            seed=int(args.seed) + i * 100,
            batch_size=max(1, int(args.batch_size)),
            candidate_size=max(4, int(args.candidate_size)),
            n_boot=max(1000, int(args.n_boot)),
            out_dir=out_dir,
        )
        dfs.append(rdf)
        model_reports[model] = rr_summary
        set_reports[model] = hs
        overlap_flags[model] = bool(overlap_flag)
        load_model.cache_clear()
        torch.cuda.empty_cache()

    full = pd.concat(dfs, ignore_index=True)
    full.to_parquet(out_dir / "causal_trace_matrix.parquet", index=False)

    rr_summary = {"timestamp": timestamp_now(), "models": model_reports, "head_set_reports": set_reports}
    write_json(out_dir / "restoration_ratio_summary.json", rr_summary)

    # Confirmatory rows require corruption-gap eligibility.
    conf = full[full["confirmatory_gap_eligible"] == True].copy()  # noqa: E712

    inter = _interaction_report(conf if not conf.empty else full)
    pvals = {"interaction": safe_float(inter.get("p_value"))}

    olmo = conf[conf["model"] == "olmo-2-7b"] if not conf.empty else full[full["model"] == "olmo-2-7b"]
    p_olmo = float("nan")
    d_olmo = float("nan")
    if not olmo.empty:
        cc = olmo[olmo["head_set"] == "ContentCond"]["rr"].to_numpy(dtype=np.float64)
        lo = olmo[olmo["head_set"] == "LowSI"]["rr"].to_numpy(dtype=np.float64)
        if len(cc) >= 3 and len(lo) >= 3:
            _, p_olmo = _welch_ttest_one_sided_gt(cc, lo)
            cc_var = float(np.nanvar(cc, ddof=1))
            lo_var = float(np.nanvar(lo, ddof=1))
            pooled = np.sqrt(max(((len(cc) - 1) * cc_var + (len(lo) - 1) * lo_var) / max(len(cc) + len(lo) - 2, 1), 1e-12))
            d_olmo = float((np.nanmean(cc) - np.nanmean(lo)) / pooled)
    pvals["olmo_content_gt_low"] = p_olmo

    p_holm = holm_adjust_dict(pvals)

    def _model_set_mean(df: pd.DataFrame, m: str, s: str) -> float:
        sub = df[(df["model"] == m) & (df["head_set"] == s)]["rr"].to_numpy(dtype=np.float64)
        return float(np.nanmedian(sub)) if len(sub) else float("nan")

    use_df = conf if not conf.empty else full

    llama_si = _model_set_mean(use_df, "llama-3.1-8b", "SI")
    llama_low = _model_set_mean(use_df, "llama-3.1-8b", "LowSI")
    llama_cc = _model_set_mean(use_df, "llama-3.1-8b", "ContentCond")
    olmo_si = _model_set_mean(use_df, "olmo-2-7b", "SI")
    olmo_low = _model_set_mean(use_df, "olmo-2-7b", "LowSI")
    olmo_cc = _model_set_mean(use_df, "olmo-2-7b", "ContentCond")

    share_denom = max(max(llama_si, 0.0) + max(llama_low, 0.0) + max(llama_cc, 0.0), 1e-8)
    share_si = max(llama_si, 0.0) / share_denom

    c1_1 = bool(np.isfinite(llama_si) and llama_si >= 0.30 and np.isfinite(llama_low) and llama_si / max(llama_low, 0.05) >= 1.50 and np.isfinite(share_si) and share_si >= 0.50)
    if np.isfinite(olmo_low) and olmo_low >= 0.05:
        c1_2a = bool(np.isfinite(olmo_si) and olmo_si / max(olmo_low, 1e-8) <= 1.20)
        c1_2b = bool(np.isfinite(olmo_cc) and olmo_cc / max(olmo_low, 1e-8) >= 1.50)
    else:
        c1_2a = bool(np.isfinite(olmo_si) and np.isfinite(olmo_low) and (olmo_si - olmo_low) <= 0.20)
        c1_2b = bool(np.isfinite(olmo_cc) and np.isfinite(olmo_low) and (olmo_cc - olmo_low) >= 0.10)

    c1_2c = bool(np.isfinite(safe_float(p_holm.get("olmo_content_gt_low"))) and safe_float(p_holm.get("olmo_content_gt_low")) < 0.05)
    c1_2 = bool(c1_2a and c1_2b and c1_2c and np.isfinite(olmo_cc) and olmo_cc >= 0.20)

    p_int_h = safe_float(p_holm.get("interaction")) if "interaction" in p_holm else float("nan")
    delta_delta = (llama_si - llama_cc) - (olmo_si - olmo_cc) if np.isfinite(llama_si) and np.isfinite(llama_cc) and np.isfinite(olmo_si) and np.isfinite(olmo_cc) else float("nan")
    c1_3 = bool(np.isfinite(p_int_h) and p_int_h < 0.05 and np.isfinite(delta_delta) and abs(delta_delta) >= 0.10 and bool(inter.get("supports_interaction", False)))

    # strict confirmatory gate from TODO C1.2
    phenomenon_counts = {}
    for model in models:
        mdf = full[full["model"] == model]
        phenomenon_counts[model] = {}
        for phenom in ["copy_offset", "indexed_retrieval", "controlled_permutation"]:
            n = int(mdf[(mdf["phenomenon"] == phenom) & (mdf["confirmatory_gap_eligible"] == True)].shape[0])  # noqa: E712
            phenomenon_counts[model][phenom] = n

    min_confirm_rows = max(1, int(args.min_confirmatory_rows_per_phenomenon))
    coverage_ok = all(
        phenomenon_counts.get(m, {}).get(pn, 0) >= min_confirm_rows
        for m in models
        for pn in ["copy_offset", "indexed_retrieval", "controlled_permutation"]
    )
    overlap_ok = all(not overlap_flags.get(m, False) for m in models)

    c1_supported = bool(c1_1 and c1_2 and c1_3 and coverage_ok and overlap_ok)
    forced_exploratory = bool((not coverage_ok) or (not overlap_ok))

    inter_report = {
        "timestamp": timestamp_now(),
        "interaction": inter,
        "p_holm": p_holm,
        "delta_delta_rr": safe_float(delta_delta),
        "criteria": {
            "criterion1_llama_si_concentration": c1_1,
            "criterion2_olmo_non_si_recovery": c1_2,
            "criterion3_cross_model_interaction": c1_3,
            "criterion4_coverage_min_rows_per_phenomenon": coverage_ok,
            "criterion4_min_confirmatory_rows_threshold": int(min_confirm_rows),
            "criterion5_calibration_eval_disjoint": overlap_ok,
        },
        "phenomenon_confirmatory_counts": phenomenon_counts,
        "overlap_flags": overlap_flags,
        "verdict": {"C1_two_carrier_class_supported": c1_supported},
    }
    write_json(out_dir / "head_set_interaction_model.json", inter_report)

    prereg = {
        "experiment_id": "C1_olmo_causal_trace",
        "question": "Which head sets carry position-sensitive restoration under explicit positional corruption?",
        "primary_hypothesis": "Llama restoration is SI-concentrated while OLMo uses non-SI content-conditional carriers.",
        "primary_endpoints": ["restoration ratio by head set", "cross-model interaction", "OLMo ContentCond > LowSI"],
        "secondary_endpoints": ["set-size diagnostics"],
        "model_list": models,
        "dataset_sources": ["synthetic position-diagnostic tasks generated via experiment2/tasks.py", "fastText cc.en.300 static embeddings"],
        "inclusion_exclusion_rules": [
            "Confirmatory phenomena: copy_offset/indexed_retrieval/controlled_permutation.",
            "Require (A_clean - A_corrupt) >= 0.05 for confirmatory RR rows.",
            "Calibration/evaluation sequence overlap forces exploratory tier.",
        ],
        "sample_size_plan": {"n_examples_per_phenomenon": int(args.n_examples)},
        "seed_plan": {"seed": int(args.seed)},
        "stopping_rule": "Stop after all model/head-set RR metrics and interaction fit are computed.",
        "multiplicity_family": ["{interaction, olmo_content_gt_low}"],
        "acceptance_criteria": ["C1.6 criteria per TODO (operationalized in script)."],
        "fallback_interpretation_if_null": "Downgrade C-path and retain B/A narrative.",
    }
    manifest = command_manifest(
        experiment_id="C1_olmo_causal_trace",
        command="run_c1_olmo_causal_trace.py",
        model="+".join(models),
        extras={
            "models": models,
            "device_map": device_map,
            "n_examples": int(args.n_examples),
            "seq_len": int(args.seq_len),
            "candidate_size": int(args.candidate_size),
            "seed": int(args.seed),
        },
    )
    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "C1_olmo_causal_trace",
        "analysis_tier": "exploratory" if forced_exploratory else "confirmatory",
        "canonical_eligible": bool(not forced_exploratory),
        "override_used": False,
        "verdict": inter_report["verdict"],
        "criteria": inter_report["criteria"],
        "limitations": [
            f"C1 confirmatory tier requires >={int(min_confirm_rows)} gap-eligible rows per phenomenon per model.",
            "Static content alignment uses fastText external lexical embeddings and offset-bin partial Spearman control.",
        ],
    }
    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "C1_olmo_causal_trace",
        "claim_status": "supported" if c1_supported else "mixed",
        "supports_main_text": bool(c1_supported),
        "strict_only": True,
        "notes": [
            "If mixed, keep C narrative exploratory and avoid two-carrier-class headline.",
            "content_similarity_data_quality_failure hard-fails run when missing token ratio >5%.",
        ],
    }
    data_dictionary = {
        "experiment_id": "C1_olmo_causal_trace",
        "tables": [
            {
                "path": str(out_dir / "causal_trace_matrix.parquet"),
                "description": "Per-example clean/corrupt/patch restoration data.",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model."},
                    {"name": "phenomenon", "dtype": "str", "description": "Position-diagnostic phenomenon."},
                    {"name": "head_set", "dtype": "str", "description": "SI/LowSI/ContentCond."},
                    {"name": "a_clean", "dtype": "float", "description": "Clean accuracy."},
                    {"name": "a_corrupt", "dtype": "float", "description": "Corrupt accuracy."},
                    {"name": "a_patch", "dtype": "float", "description": "Patch/channel-isolation accuracy."},
                    {"name": "rr", "dtype": "float", "description": "Restoration ratio."},
                    {"name": "confirmatory_gap_eligible", "dtype": "bool", "description": "Gap eligibility flag."},
                ],
            }
        ],
    }

    emit_core_artifacts(
        experiment_id="C1_olmo_causal_trace",
        out_dir=out_dir,
        preregistration=prereg,
        manifest=manifest,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    write_json(out_dir / "C1_claim_impact.json", claim_impact)

    print(f"[C1] wrote {out_dir}")


if __name__ == "__main__":
    main()
