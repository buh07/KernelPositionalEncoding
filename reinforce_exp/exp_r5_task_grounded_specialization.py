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
from experiment2.tasks import build_token_pools, generate_task_examples  # noqa: E402
from experiment3.stats_utils import holm_adjust, one_sided_p_from_two_sided  # noqa: E402
from experiment3.theory1_si_circuits import MODELS, HeadID, head_output_ablation  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402

TARGET_MODELS = ("llama-3.1-8b", "olmo-2-7b")
DEFAULT_OUT = RESULTS_ROOT / "exp_r5_task_grounded"


REGIME_CONFIGS = [
    {
        "regime": "long_span_64",
        "task_name": "long_range_retrieval",
        "span_override": 64,
        "candidate_size": 10,
        "count": 200,
        "description": "Controlled long-span retrieval, 64-token dependency.",
    },
    {
        "regime": "long_span_128",
        "task_name": "long_range_retrieval",
        "span_override": 128,
        "candidate_size": 10,
        "count": 200,
        "description": "Controlled longer-span retrieval, 128-token dependency.",
    },
    {
        "regime": "uncertainty_low",
        "task_name": "local_key_match",
        "span_override": None,
        "candidate_size": 10,
        "count": 200,
        "description": "Low uncertainty setting via 10-way restricted evaluation.",
    },
    {
        "regime": "uncertainty_high",
        "task_name": "local_key_match",
        "span_override": None,
        "candidate_size": 50,
        "count": 200,
        "description": "High uncertainty setting via 50-way restricted evaluation.",
    },
]


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


def _evaluate_examples(
    *,
    model,
    device: str,
    examples,
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

    acc = float(total_correct / max(total_targets, 1))
    mean_nll = float(total_nll / max(total_targets, 1))
    return {
        "accuracy": acc,
        "mean_nll": mean_nll,
        "n_targets": int(total_targets),
        "n_correct": int(total_correct),
    }


def _paired_test(rows_df: pd.DataFrame, regime: str) -> dict[str, Any]:
    sub = rows_df[rows_df["regime"] == regime].copy()
    hi = sub[sub["group"] == "high_si"][ ["seed", "drop"] ].rename(columns={"drop": "high"})
    lo = sub[sub["group"] == "low_si"][ ["seed", "drop"] ].rename(columns={"drop": "low"})
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

    rows: list[dict[str, Any]] = []
    raw_eval_rows: list[dict[str, Any]] = []

    for seed in range(max(1, int(num_seeds))):
        # Pre-generate the task datasets per regime to keep conditions matched.
        regime_examples: dict[str, list[Any]] = {}
        for cfg in REGIME_CONFIGS:
            count = max(24, int(round(int(cfg["count"]) * float(count_scale))))
            regime_examples[cfg["regime"]] = generate_task_examples(
                task_name=cfg["task_name"],
                model_name=model_name,
                seq_len=max(128, int(seq_len)),
                seed=seed,
                count=count,
                pools=pools,
                span_override=cfg["span_override"],
                span_choices=(int(cfg["span_override"]),) if cfg["span_override"] is not None else None,
            )

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
                    eval_res = _evaluate_examples(
                        model=model,
                        device=device,
                        examples=exs,
                        pools=pools,
                        candidate_size=int(cfg["candidate_size"]),
                        batch_size=max(1, int(batch_size)),
                    )
                cond_scores[group_name] = eval_res
                raw_eval_rows.append(
                    {
                        "model": model_name,
                        "seed": int(seed),
                        "regime": regime,
                        "group": group_name,
                        "candidate_size": int(cfg["candidate_size"]),
                        "task_name": str(cfg["task_name"]),
                        "span_override": int(cfg["span_override"]) if cfg["span_override"] is not None else None,
                        "accuracy": safe_float(eval_res["accuracy"]),
                        "mean_nll": safe_float(eval_res["mean_nll"]),
                        "n_targets": int(eval_res["n_targets"]),
                        "n_correct": int(eval_res["n_correct"]),
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
                        "group": "high_si" if group_name == "ablate_high_si" else "low_si",
                        "drop": drop,
                        "baseline_accuracy": baseline_acc,
                        "ablated_accuracy": acc,
                        "candidate_size": int(cfg["candidate_size"]),
                        "task_name": str(cfg["task_name"]),
                        "span_override": int(cfg["span_override"]) if cfg["span_override"] is not None else None,
                        "description": str(cfg["description"]),
                        "example_count": int(len(exs)),
                    }
                )

            print(
                f"[EXP-R5][{model_name}] seed={seed} regime={regime} "
                f"none={baseline_acc:.4f} high_drop={rows[-2]['drop']:.4f} low_drop={rows[-1]['drop']:.4f}",
                flush=True,
            )

    effects = pd.DataFrame(rows)
    raw_eval = pd.DataFrame(raw_eval_rows)
    effects.to_parquet(out_dir / "task_grounded_effects.parquet", index=False)
    raw_eval.to_parquet(out_dir / "task_grounded_raw_eval.parquet", index=False)

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

    # Compare task-grounded regime signs against proxy 3P2-J signs.
    proxy_path = ROOT / "results" / "experiment3_phase2" / "exp3p2j_conditional_regimes_longspan_repair" / model_name / "regime_summary.json"
    proxy_signs: dict[str, float] = {}
    if proxy_path.exists():
        proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
        for regime, data in proxy.get("regime_tests", {}).items():
            proxy_signs[str(regime)] = safe_float(data.get("mean_delta_high_minus_low"))

    task_signs = {k: safe_float(v.get("mean_delta_high_minus_low")) for k, v in regime_tests.items()}
    sign_matches = 0
    sign_total = 0
    map_regime = {
        "long_span_64": "long_span_retrieval",
        "long_span_128": "long_span_retrieval",
        "uncertainty_low": "high_uncertainty_tokens",
        "uncertainty_high": "high_uncertainty_tokens",
    }
    for tr, pr in map_regime.items():
        ts = task_signs.get(tr)
        ps = proxy_signs.get(pr)
        if np.isfinite(ts) and np.isfinite(ps):
            sign_total += 1
            if (ts >= 0 and ps >= 0) or (ts < 0 and ps < 0):
                sign_matches += 1

    summary = {
        "timestamp": timestamp_now(),
        "experiment": "EXP-R5",
        "model": model_name,
        "n_rows": int(len(effects)),
        "n_raw_rows": int(len(raw_eval)),
        "regime_configs": REGIME_CONFIGS,
        "regime_tests": regime_tests,
        "interaction_model": interaction,
        "proxy_vs_task_sign_match": {
            "n_matches": int(sign_matches),
            "n_compared": int(sign_total),
            "fraction": float(sign_matches / max(sign_total, 1)),
        },
        "verdict": {
            "interaction_significant": bool(np.isfinite(interaction.get("p_value_holm", float("nan")) ) and interaction["p_value_holm"] < 0.05),
            "supports_task_grounded_specialization": bool(
                interaction.get("supports_conditional_specialization", False)
                or any(bool(v.get("supports_high_gt_low", False)) for v in regime_tests.values())
            ),
        },
    }

    write_json(out_dir / "regime_summary.json", summary)
    return summary


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
    models = list(TARGET_MODELS)
    reports: dict[str, Any] = {}
    for model in models:
        device = device_map.get(model, "cuda:0")
        print(f"[EXP-R5] model={model} device={device}", flush=True)
        # Run each model in a separate process so cached model objects do not
        # retain GPU memory between models on 16GB cards.
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            "-u",
            "reinforce_exp/exp_r5_task_grounded_specialization.py",
            "--model",
            str(model),
            "--device",
            str(device),
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
        print(f"[EXP-R5] exec: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        summary_path = output_root / model / "regime_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"Expected per-model summary missing after subprocess run: {summary_path}")
        reports[model] = json.loads(summary_path.read_text(encoding="utf-8"))

    aggregate = {
        "timestamp": timestamp_now(),
        "experiment": "EXP-R5",
        "models": reports,
    }

    write_json(output_root / "interaction_comparison_proxy_vs_task.json", aggregate)
    return aggregate


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        model, dev = tok.split(":", 1)
        out[model.strip()] = dev.strip()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-R5: task-grounded conditional specialization validation")
    p.add_argument("--model", choices=["all", *TARGET_MODELS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1")
    p.add_argument("--num-seeds", type=int, default=6)
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
        device_map = {str(args.model): str(args.device)}
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
                experiment_id="EXP-R5",
                command="task_grounded_specialization_single",
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
        print(f"[EXP-R5] wrote {out_root / str(args.model) / 'regime_summary.json'}")
        print(f"[EXP-R5] verdict={rep['verdict']}")
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
            experiment_id="EXP-R5",
            command="task_grounded_specialization_all",
            model="llama-3.1-8b+olmo-2-7b",
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

    print(f"[EXP-R5] wrote {out_root / 'interaction_comparison_proxy_vs_task.json'}")
    print(f"[EXP-R5] models={list(aggregate['models'].keys())}")


if __name__ == "__main__":
    main()
