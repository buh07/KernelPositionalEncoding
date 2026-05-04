#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp.common import RESULTS_ROOT, command_manifest, ensure_dir, safe_float, timestamp_now, write_json  # noqa: E402
from experiment2.execution import _evaluate_example_from_token_logits  # noqa: E402
from experiment2.tasks import TaskExample, build_token_pools, generate_task_examples  # noqa: E402
from experiment3.stats_utils import holm_adjust, one_sided_p_from_two_sided  # noqa: E402
from experiment3.theory1_si_circuits import MODELS, HeadID, head_output_ablation  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402

TARGET_MODELS = ("llama-3.1-8b", "olmo-2-7b")
DEFAULT_OUT = RESULTS_ROOT / "exp_r5b_regime_alignment"

REGIME_CONFIGS = [
    {
        "regime": "boundary_dense",
        "proxy_regime": "boundary_dense",
        "task_name": "local_key_match",
        "span_override": None,
        "candidate_size": 12,
        "count": 240,
        "rare_context": False,
        "description": "Dense local dependency regime for boundary-heavy proxy mapping.",
    },
    {
        "regime": "high_uncertainty",
        "proxy_regime": "high_uncertainty_tokens",
        "task_name": "local_key_match",
        "span_override": None,
        "candidate_size": 64,
        "count": 240,
        "rare_context": False,
        "description": "High candidate uncertainty for uncertainty-proxy mapping.",
    },
    {
        "regime": "long_span_retrieval",
        "proxy_regime": "long_span_retrieval",
        "task_name": "delayed_copy",
        "span_override": 128,
        "candidate_size": 12,
        "count": 240,
        "rare_context": False,
        "description": "Long-span delayed-copy regime aligned with long-span proxy.",
    },
    {
        "regime": "rare_token_context",
        "proxy_regime": "rare_token_context",
        "task_name": "retrieval_bridge",
        "span_override": 64,
        "candidate_size": 12,
        "count": 240,
        "rare_context": True,
        "description": "Bridge retrieval under rare-context token replacement.",
    },
]


def _resilient_generate_examples(
    *,
    task_name: str,
    model_name: str,
    seq_len: int,
    seed: int,
    count: int,
    pools,
    span_override: int | None,
) -> tuple[list[TaskExample], int | None]:
    span_candidates: list[int | None] = [span_override]
    if span_override is not None:
        fallback = [int(span_override), 96, 64, 48, 32, 24]
        span_candidates = []
        for s in fallback:
            if int(s) + 8 < int(seq_len) and int(s) not in span_candidates:
                span_candidates.append(int(s))
        if not span_candidates:
            span_candidates = [None]
    else:
        span_candidates = [None]

    last_exc: Exception | None = None
    for sp in span_candidates:
        try:
            exs = generate_task_examples(
                task_name=str(task_name),
                model_name=model_name,
                seq_len=max(128, int(seq_len)),
                seed=int(seed),
                count=int(count),
                pools=pools,
                span_override=sp,
                span_choices=(int(sp),) if sp is not None else None,
            )
            return exs, sp
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unexpected generation failure without captured exception.")


def _load_head_groups(model_name: str) -> dict[str, list[HeadID]]:
    path = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_groups.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing head_groups.json: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    def parse(entries: list[dict[str, Any]]) -> list[HeadID]:
        return [HeadID(int(e["layer"]), int(e["head"])) for e in entries]

    return {
        "none": [],
        "ablate_high_si": parse(raw["high_si"]),
        "ablate_low_si": parse(raw["low_si"]),
    }


def _inject_rare_context(
    *,
    examples: list[TaskExample],
    rare_pool: tuple[int, ...],
    seed: int,
    max_replace_frac: float = 0.20,
) -> tuple[list[TaskExample], dict[str, Any]]:
    if not examples:
        return [], {"rare_pool_size": 0, "mean_replace_rate": 0.0}
    rng = np.random.default_rng(int(seed))
    out: list[TaskExample] = []
    replace_rates: list[float] = []
    pool = list(rare_pool)
    if not pool:
        return examples, {"rare_pool_size": 0, "mean_replace_rate": 0.0}

    for ex in examples:
        tokens = list(ex.tokens)
        target_pos = set(int(x) for x in ex.target_positions)
        candidate_pos = [i for i in range(len(tokens)) if i not in target_pos]
        if candidate_pos:
            n_rep = max(1, int(round(float(max_replace_frac) * len(candidate_pos))))
            n_rep = min(n_rep, len(candidate_pos))
            chosen = rng.choice(np.asarray(candidate_pos, dtype=np.int64), size=n_rep, replace=False)
            for p in chosen.tolist():
                tokens[int(p)] = int(pool[int(rng.integers(0, len(pool)))])
            replace_rates.append(float(n_rep / max(1, len(candidate_pos))))
        else:
            replace_rates.append(0.0)

        out.append(
            TaskExample(
                id=str(ex.id),
                task_name=str(ex.task_name),
                tokens=tokens,
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
        )

    return out, {
        "rare_pool_size": int(len(pool)),
        "mean_replace_rate": safe_float(np.mean(np.asarray(replace_rates, dtype=np.float64))),
    }


def _evaluate_examples(
    *,
    model,
    device: str,
    examples: list[TaskExample],
    pools,
    candidate_size: int,
    batch_size: int,
) -> dict[str, Any]:
    total_targets = 0
    total_correct = 0
    total_nll = 0.0
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

        for idx, ex in enumerate(batch):
            metrics, _ = _evaluate_example_from_token_logits(
                logits[idx],
                ex,
                split="synthetic",
                pools=pools,
                synthetic_eval_mode="restricted",
                candidate_size=int(candidate_size),
                candidate_policy_version="restricted_candidates_v1_structured_first",
            )
            total_targets += int(metrics["num_targets"])
            total_correct += int(round(float(metrics["accuracy"]) * int(metrics["num_targets"])))
            total_nll += float(metrics["mean_nll"]) * int(metrics["num_targets"])

        pos += len(batch)
        del input_ids, logits
        torch.cuda.empty_cache()

    return {
        "accuracy": float(total_correct / max(1, total_targets)),
        "mean_nll": float(total_nll / max(1, total_targets)),
        "n_targets": int(total_targets),
        "n_correct": int(total_correct),
    }


def _paired_test(rows_df: pd.DataFrame, regime: str) -> dict[str, Any]:
    sub = rows_df[rows_df["regime"] == regime].copy()
    hi = sub[sub["group"] == "high_si"][["seed", "drop"]].rename(columns={"drop": "high"})
    lo = sub[sub["group"] == "low_si"][["seed", "drop"]].rename(columns={"drop": "low"})
    pair = hi.merge(lo, on="seed", how="inner")
    if len(pair) < 2:
        return {
            "n_pairs": int(len(pair)),
            "mean_delta_high_minus_low": float("nan"),
            "p_one_sided": float("nan"),
            "cohens_d": float("nan"),
        }
    delta = pair["high"].astype(float).values - pair["low"].astype(float).values
    t_stat, p_two = scipy_stats.ttest_1samp(delta, popmean=0.0, nan_policy="omit")
    p_one = one_sided_p_from_two_sided(float(t_stat), float(p_two), alternative="greater")
    d = float(np.mean(delta) / max(np.std(delta, ddof=1), 1e-8))
    return {
        "n_pairs": int(len(pair)),
        "mean_delta_high_minus_low": float(np.mean(delta)),
        "p_one_sided": float(p_one),
        "cohens_d": d,
    }


def _interaction(rows_df: pd.DataFrame) -> dict[str, Any]:
    work = rows_df.copy()
    work = work[np.isfinite(work["drop"].astype(float).values)]
    if work.empty:
        return {
            "supports_conditional_specialization": False,
            "f_interaction": float("nan"),
            "p_value": float("nan"),
            "partial_eta_squared": float("nan"),
        }

    work["drop_z"] = (work["drop"] - work["drop"].mean()) / max(work["drop"].std(ddof=1), 1e-8)
    regimes = sorted(work["regime"].astype(str).unique().tolist())
    groups = sorted(work["group"].astype(str).unique().tolist())

    means_a = {r: float(work[work["regime"] == r]["drop_z"].mean()) for r in regimes}
    means_b = {g: float(work[work["group"] == g]["drop_z"].mean()) for g in groups}
    grand = float(work["drop_z"].mean())

    ss_ab = 0.0
    ss_within = 0.0
    for r in regimes:
        for g in groups:
            cell = work[(work["regime"] == r) & (work["group"] == g)]
            if len(cell) == 0:
                continue
            m = float(cell["drop_z"].mean())
            ss_ab += len(cell) * ((m - means_a[r] - means_b[g] + grand) ** 2)
            ss_within += float(np.sum((cell["drop_z"].to_numpy(dtype=float) - m) ** 2))

    a = len(regimes)
    b = len(groups)
    n_total = len(work)
    df_ab = max(1, (a - 1) * (b - 1))
    df_within = max(1, n_total - (a * b))
    ms_ab = ss_ab / df_ab
    ms_within = ss_within / df_within
    f_ab = float(ms_ab / ms_within) if ms_within > 0 else float("nan")
    p_ab = float(1.0 - scipy_stats.f.cdf(f_ab, df_ab, df_within)) if np.isfinite(f_ab) else float("nan")
    eta = float(ss_ab / (ss_ab + ss_within)) if (ss_ab + ss_within) > 0 else float("nan")

    return {
        "f_interaction": f_ab,
        "p_value": p_ab,
        "partial_eta_squared": eta,
        "supports_conditional_specialization": bool(np.isfinite(p_ab) and p_ab < 0.05 and np.isfinite(eta) and eta >= 0.02),
    }


def _load_proxy_signs(model_name: str) -> dict[str, float]:
    path = ROOT / "results" / "experiment3_phase2" / "exp3p2j_conditional_regimes_longspan_repair" / model_name / "regime_summary.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for k, v in raw.get("regime_tests", {}).items():
        out[str(k)] = safe_float(v.get("mean_delta_high_minus_low"))
    return out


def run_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    num_seeds: int,
    seq_len: int,
    batch_size: int,
    count_scale: float,
) -> dict[str, Any]:
    t0 = time.time()
    out_dir = output_root / model_name
    ensure_dir(out_dir)

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)

    special_ids = [getattr(tokenizer, a, None) for a in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")]
    special_ids = [int(x) for x in special_ids if x is not None]
    pools = build_token_pools(model_name, int(tokenizer.vocab_size), special_ids)
    groups = _load_head_groups(model_name)
    rare_pool = tuple(sorted(pools.filler)[-2048:])

    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    construct_rows: list[dict[str, Any]] = []

    for seed in range(max(2, int(num_seeds))):
        regime_examples: dict[str, list[TaskExample]] = {}
        regime_audit: dict[str, Any] = {}
        for cfg in REGIME_CONFIGS:
            count = max(96, int(round(float(cfg["count"]) * float(count_scale))))
            exs, used_span = _resilient_generate_examples(
                task_name=str(cfg["task_name"]),
                model_name=model_name,
                seq_len=max(128, int(seq_len)),
                seed=int(seed),
                count=count,
                pools=pools,
                span_override=cfg["span_override"],
            )
            rare_meta = {"mean_replace_rate": 0.0, "rare_pool_size": 0}
            if bool(cfg.get("rare_context", False)):
                exs, rare_meta = _inject_rare_context(
                    examples=exs,
                    rare_pool=rare_pool,
                    seed=int(seed) + 701,
                    max_replace_frac=0.20,
                )
            regime_examples[str(cfg["regime"])] = exs
            dep = np.asarray([int(e.dependency_span) for e in exs], dtype=np.float64) if exs else np.asarray([], dtype=np.float64)
            tcnt = np.asarray([len(e.target_positions) for e in exs], dtype=np.float64) if exs else np.asarray([], dtype=np.float64)
            regime_audit[str(cfg["regime"])] = {
                "n_examples": int(len(exs)),
                "mean_dependency_span": safe_float(np.mean(dep) if dep.size else float("nan")),
                "std_dependency_span": safe_float(np.std(dep, ddof=1) if dep.size > 1 else float("nan")),
                "mean_targets_per_example": safe_float(np.mean(tcnt) if tcnt.size else float("nan")),
                "candidate_size": int(cfg["candidate_size"]),
                "configured_span_override": int(cfg["span_override"]) if cfg["span_override"] is not None else None,
                "used_span_override": int(used_span) if used_span is not None else None,
                "rare_context_enabled": bool(cfg.get("rare_context", False)),
                "rare_context_mean_replace_rate": safe_float(rare_meta.get("mean_replace_rate")),
            }

        for cfg in REGIME_CONFIGS:
            regime = str(cfg["regime"])
            exs = regime_examples[regime]
            cond_scores: dict[str, dict[str, Any]] = {}
            for group_name in ("none", "ablate_high_si", "ablate_low_si"):
                heads = groups[group_name]
                cm: contextlib.AbstractContextManager
                if not heads:
                    cm = contextlib.nullcontext()
                else:
                    cm = head_output_ablation(model, heads)
                with cm:
                    ev = _evaluate_examples(
                        model=model,
                        device=device,
                        examples=exs,
                        pools=pools,
                        candidate_size=int(cfg["candidate_size"]),
                        batch_size=max(1, int(batch_size)),
                    )
                cond_scores[group_name] = ev
                raw_rows.append(
                    {
                        "model": model_name,
                        "seed": int(seed),
                        "regime": regime,
                        "proxy_regime": str(cfg["proxy_regime"]),
                        "group": group_name,
                        "candidate_size": int(cfg["candidate_size"]),
                        "task_name": str(cfg["task_name"]),
                        "span_override": int(cfg["span_override"]) if cfg["span_override"] is not None else None,
                        "accuracy": safe_float(ev["accuracy"]),
                        "mean_nll": safe_float(ev["mean_nll"]),
                        "n_targets": int(ev["n_targets"]),
                    }
                )

            baseline_acc = safe_float(cond_scores["none"]["accuracy"])
            for group_name in ("ablate_high_si", "ablate_low_si"):
                acc = safe_float(cond_scores[group_name]["accuracy"])
                drop = float(baseline_acc - acc)
                rows.append(
                    {
                        "model": model_name,
                        "seed": int(seed),
                        "regime": regime,
                        "proxy_regime": str(cfg["proxy_regime"]),
                        "group": "high_si" if group_name == "ablate_high_si" else "low_si",
                        "drop": drop,
                        "baseline_accuracy": baseline_acc,
                        "ablated_accuracy": acc,
                        "candidate_size": int(cfg["candidate_size"]),
                        "task_name": str(cfg["task_name"]),
                        "span_override": int(cfg["span_override"]) if cfg["span_override"] is not None else None,
                        "example_count": int(len(exs)),
                        "description": str(cfg["description"]),
                    }
                )

            construct_rows.append(
                {
                    "model": model_name,
                    "seed": int(seed),
                    "regime": regime,
                    **regime_audit[regime],
                }
            )

            print(
                f"[R5B][{model_name}] seed={seed} regime={regime} "
                f"none={baseline_acc:.4f} high_drop={rows[-2]['drop']:.4f} low_drop={rows[-1]['drop']:.4f}",
                flush=True,
            )

    effects = pd.DataFrame(rows)
    raw_df = pd.DataFrame(raw_rows)
    construct_df = pd.DataFrame(construct_rows)
    effects.to_parquet(out_dir / "task_aligned_effects.parquet", index=False)
    raw_df.to_parquet(out_dir / "task_aligned_raw_eval.parquet", index=False)
    construct_df.to_parquet(out_dir / "regime_construct_audit.parquet", index=False)

    regime_tests: dict[str, Any] = {}
    pvals = {}
    for regime in sorted(effects["regime"].astype(str).unique().tolist()):
        rep = _paired_test(effects, regime)
        regime_tests[regime] = rep
        pvals[regime] = safe_float(rep.get("p_one_sided"))
    adj = holm_adjust(pvals)
    for regime in regime_tests:
        regime_tests[regime]["p_one_sided_holm"] = safe_float(adj.get(regime))
        regime_tests[regime]["supports_high_gt_low"] = bool(
            np.isfinite(regime_tests[regime]["p_one_sided_holm"])
            and regime_tests[regime]["p_one_sided_holm"] < 0.05
            and safe_float(regime_tests[regime].get("mean_delta_high_minus_low")) > 0
        )

    interaction = _interaction(effects)
    interaction["p_value_holm"] = safe_float(holm_adjust({"interaction": safe_float(interaction.get("p_value"))}).get("interaction"))

    proxy_signs = _load_proxy_signs(model_name)
    sign_rows: list[dict[str, Any]] = []
    n_matches = 0
    n_compared = 0
    for cfg in REGIME_CONFIGS:
        regime = str(cfg["regime"])
        preg = str(cfg["proxy_regime"])
        task_s = safe_float(regime_tests.get(regime, {}).get("mean_delta_high_minus_low"))
        proxy_s = safe_float(proxy_signs.get(preg))
        sign_match = bool(np.isfinite(task_s) and np.isfinite(proxy_s) and ((task_s >= 0 and proxy_s >= 0) or (task_s < 0 and proxy_s < 0)))
        if np.isfinite(task_s) and np.isfinite(proxy_s):
            n_compared += 1
            if sign_match:
                n_matches += 1
        sign_rows.append(
            {
                "regime": regime,
                "proxy_regime": preg,
                "task_sign_delta": task_s,
                "proxy_sign_delta": proxy_s,
                "sign_match": sign_match if np.isfinite(task_s) and np.isfinite(proxy_s) else None,
            }
        )

    sign_fraction = float(n_matches / max(1, n_compared))
    transfer_ok = bool(n_compared >= 4 and sign_fraction >= 0.75)

    interaction_transfer = {
        "timestamp": timestamp_now(),
        "experiment": "R5B",
        "model": model_name,
        "interaction_model": interaction,
        "regime_tests": regime_tests,
        "sign_transfer": {
            "n_matches": int(n_matches),
            "n_compared": int(n_compared),
            "fraction": sign_fraction,
            "criterion": ">=3/4",
            "passes": transfer_ok,
        },
        "verdict": {
            "interaction_significant": bool(np.isfinite(interaction.get("p_value_holm", float("nan"))) and interaction["p_value_holm"] < 0.05),
            "supports_task_grounded_semantics": bool(
                np.isfinite(interaction.get("p_value_holm", float("nan")))
                and interaction["p_value_holm"] < 0.05
                and transfer_ok
            ),
        },
        "runtime_sec": float(time.time() - t0),
    }

    write_json(out_dir / "interaction_transfer_report.json", interaction_transfer)
    write_json(
        out_dir / "regime_construct_audit.json",
        {
            "timestamp": timestamp_now(),
            "experiment": "R5B",
            "model": model_name,
            "rows": construct_df.to_dict(orient="records"),
        },
    )
    write_json(
        out_dir / "proxy_vs_task_sign_matrix.json",
        {
            "timestamp": timestamp_now(),
            "experiment": "R5B",
            "model": model_name,
            "rows": sign_rows,
        },
    )
    return interaction_transfer


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        model, dev = tok.split(":", 1)
        out[model.strip()] = dev.strip()
    return out


def run_all(
    *,
    output_root: Path,
    device_map: dict[str, str],
    num_seeds: int,
    seq_len: int,
    batch_size: int,
    count_scale: float,
) -> dict[str, Any]:
    ensure_dir(output_root)
    reports: dict[str, Any] = {}
    for model in TARGET_MODELS:
        dev = device_map.get(model, "cuda:0")
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            "-u",
            "reinforce_exp/exp_r5b_regime_alignment.py",
            "--model",
            str(model),
            "--device",
            str(dev),
            "--num-seeds",
            str(max(2, int(num_seeds))),
            "--seq-len",
            str(max(128, int(seq_len))),
            "--batch-size",
            str(max(1, int(batch_size))),
            "--count-scale",
            str(float(count_scale)),
            "--output-root",
            str(output_root),
        ]
        print("[R5B] exec:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        summary_path = output_root / model / "interaction_transfer_report.json"
        reports[model] = json.loads(summary_path.read_text(encoding="utf-8"))

    aggregate = {
        "timestamp": timestamp_now(),
        "experiment": "R5B",
        "models": reports,
        "promotion_gate": {
            "requires_interaction_significant_each_model": True,
            "requires_sign_concordance_ge_3_of_4_each_model": True,
            "all_models_pass": bool(
                all(
                    bool(v.get("verdict", {}).get("interaction_significant", False))
                    and bool(v.get("sign_transfer", {}).get("passes", False))
                    for v in reports.values()
                )
            ),
        },
    }
    write_json(output_root / "interaction_transfer_report.json", aggregate)
    return aggregate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-R5B: one-to-one regime transfer replication")
    p.add_argument("--model", choices=["all", *TARGET_MODELS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1")
    p.add_argument("--num-seeds", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--count-scale", type=float, default=1.0)
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    ensure_dir(out_root)

    if args.model != "all":
        rep = run_model(
            model_name=str(args.model),
            device=str(args.device),
            output_root=out_root,
            num_seeds=max(2, int(args.num_seeds)),
            seq_len=max(128, int(args.seq_len)),
            batch_size=max(1, int(args.batch_size)),
            count_scale=max(0.1, float(args.count_scale)),
        )
        write_json(
            out_root / str(args.model) / "manifest.json",
            command_manifest(
                experiment_id="R5B",
                command="single_model",
                model=str(args.model),
                extras={
                    "num_seeds": int(args.num_seeds),
                    "seq_len": int(args.seq_len),
                    "batch_size": int(args.batch_size),
                    "count_scale": float(args.count_scale),
                    "output_root": str(out_root),
                },
            ),
        )
        print(f"[R5B] wrote {out_root / str(args.model) / 'interaction_transfer_report.json'}")
        print(f"[R5B] verdict={rep['verdict']}")
        return

    device_map = _parse_device_map(args.device_map)
    aggregate = run_all(
        output_root=out_root,
        device_map=device_map,
        num_seeds=max(2, int(args.num_seeds)),
        seq_len=max(128, int(args.seq_len)),
        batch_size=max(1, int(args.batch_size)),
        count_scale=max(0.1, float(args.count_scale)),
    )

    write_json(
        out_root / "manifest.json",
        command_manifest(
            experiment_id="R5B",
            command="all_models",
            model="+".join(TARGET_MODELS),
            extras={
                "num_seeds": int(args.num_seeds),
                "seq_len": int(args.seq_len),
                "batch_size": int(args.batch_size),
                "count_scale": float(args.count_scale),
                "output_root": str(out_root),
                "device_map": device_map,
            },
        ),
    )
    print(f"[R5B] wrote {out_root / 'interaction_transfer_report.json'}")
    print(f"[R5B] all_models_pass={aggregate['promotion_gate']['all_models_pass']}")


if __name__ == "__main__":
    main()
