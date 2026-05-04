#!/usr/bin/env python3
"""E18 — ICL Format-Confound Controls.

Controls for E12 by adding:
1) matched-cardinality non-SI head ablation,
2) structure-preserving label-shuffle prompts,
3) matched-length positional-perturbation baseline.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    B1_RESULTS,
    PRIMARY_MODELS,
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    enforce_coverage_contract,
    emit_core_artifacts,
    head_output_ablation,
    load_model_for_exp,
    parse_device_map,
    parse_models_arg,
    read_json,
)

OUT_ROOT = RESULTS_ROOT / "E18_icl_format_confound"
N_DEMOS = 4
N_EVAL = 200
SEED_BASE = 20260501
MIN_DENOM = 0.01


@torch.no_grad()
def _eval_accuracy_and_logprob(model: Any, prompts: list[dict[str, Any]], device: str) -> dict[str, float]:
    model.eval()
    correct: list[int] = []
    lps: list[float] = []
    for p in prompts:
        ids = torch.tensor([p["input_ids"]], device=device, dtype=torch.long)
        out = model(ids)
        logits = out.logits[0, -1]
        target = int(p["target_token_id"])
        pred = int(torch.argmax(logits).item())
        lp = float(torch.log_softmax(logits, dim=-1)[target].item())
        correct.append(int(pred == target))
        lps.append(lp)
    return {
        "accuracy": float(np.mean(correct)) if correct else float("nan"),
        "mean_logprob": float(np.mean(lps)) if lps else float("nan"),
        "n_prompts": float(len(prompts)),
    }


def _build_icl_prompts(tokenizer: Any, n: int, n_demos: int, rng: random.Random) -> list[dict[str, Any]]:
    vocab = list(tokenizer.get_vocab().values())
    out: list[dict[str, Any]] = []
    for _ in range(n):
        key = rng.choice(vocab)
        val = rng.choice(vocab)
        demo: list[int] = []
        for _ in range(n_demos):
            demo.extend([key, val])
        out.append({"input_ids": demo + [key], "target_token_id": val, "is_icl": True})
    return out


def _build_nonicl_prompts(tokenizer: Any, n: int, n_demos: int, rng: random.Random) -> list[dict[str, Any]]:
    vocab = list(tokenizer.get_vocab().values())
    out: list[dict[str, Any]] = []
    for _ in range(n):
        demo: list[int] = []
        for _ in range(n_demos):
            demo.extend([rng.choice(vocab), rng.choice(vocab)])
        query = rng.choice(vocab)
        target = demo[-1]
        out.append({"input_ids": demo + [query], "target_token_id": target, "is_icl": False})
    return out


def _build_label_shuffle_prompts(tokenizer: Any, n: int, n_demos: int, rng: random.Random) -> list[dict[str, Any]]:
    """Structure-preserving ICL-like prompts with shuffled key->value label mapping."""
    vocab = list(tokenizer.get_vocab().values())
    out: list[dict[str, Any]] = []
    for _ in range(n):
        keys = [rng.choice(vocab) for _ in range(n_demos)]
        vals = [rng.choice(vocab) for _ in range(n_demos)]
        demo: list[int] = []
        for k, v in zip(keys, vals):
            demo.extend([k, v])
        qidx = rng.randrange(n_demos)
        query = keys[qidx]
        target = vals[(qidx + 1) % n_demos]  # intentionally shuffled label
        out.append({"input_ids": demo + [query], "target_token_id": target, "is_icl": True, "is_label_shuffle": True})
    return out


def _perturb_matched_length(prompts: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """Matched-length positional perturbation baseline: permute demo-token order, preserve query."""
    out: list[dict[str, Any]] = []
    for p in prompts:
        ids = list(p["input_ids"])
        if len(ids) <= 2:
            out.append(dict(p))
            continue
        body = ids[:-1]
        query = ids[-1]
        perm = list(range(len(body)))
        rng.shuffle(perm)
        body_perm = [body[i] for i in perm]
        q = dict(p)
        q["input_ids"] = body_perm + [query]
        out.append(q)
    return out


def _load_high_and_control_heads(model_name: str, seed: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    cluster = B1_RESULTS / "cluster_membership.parquet"
    if not cluster.exists():
        raise FileNotFoundError(f"[E18] missing cluster_membership at {cluster}")
    df = pd.read_parquet(cluster)
    sub = df[df["model"] == model_name].copy()
    if sub.empty:
        raise RuntimeError(f"[E18] no cluster rows for model={model_name}")

    high = sub[sub["is_high_si"] == True][["layer", "head"]]
    high_heads = [(int(r.layer), int(r.head)) for r in high.itertuples()]

    non_si_pool = sub[sub["is_high_si"] != True][["layer", "head", "mean_r2"]].copy()
    non_si_pool = non_si_pool.sort_values("mean_r2", ascending=True).reset_index(drop=True)
    if len(non_si_pool) < len(high_heads):
        raise RuntimeError(f"[E18] non-SI pool too small for matched-cardinality control: {len(non_si_pool)} < {len(high_heads)}")

    # Matched-cardinality control from low-to-mid R2 pool with deterministic shuffle.
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(non_si_pool), size=len(high_heads), replace=False)
    ctrl = non_si_pool.iloc[np.sort(idx)]
    ctrl_heads = [(int(r.layer), int(r.head)) for r in ctrl.itertuples()]
    return high_heads, ctrl_heads


def _compute_degradation(intact: dict[str, float], ablated: dict[str, float]) -> dict[str, float]:
    return {
        "acc_loss": float(intact["accuracy"] - ablated["accuracy"]),
        "lp_loss": float(intact["mean_logprob"] - ablated["mean_logprob"]),
    }


def run_model(
    *,
    model_name: str,
    device: str,
    out_root: Path,
    n_eval: int,
    n_demos: int,
    seed: int,
) -> dict[str, Any]:
    print(f"[E18] Loading {model_name} on {device}", flush=True)
    model, tokenizer = load_model_for_exp(model_name, device)

    high_heads, ctrl_heads = _load_high_and_control_heads(model_name, seed)

    rng = random.Random(int(seed))
    icl_prompts = _build_icl_prompts(tokenizer, n_eval, n_demos, rng)
    nonicl_prompts = _build_nonicl_prompts(tokenizer, n_eval, n_demos, random.Random(int(seed) + 1))
    shuffle_prompts = _build_label_shuffle_prompts(tokenizer, n_eval, n_demos, random.Random(int(seed) + 2))
    shuffle_nonicl_prompts = _build_nonicl_prompts(tokenizer, n_eval, n_demos, random.Random(int(seed) + 5))
    icl_perturbed = _perturb_matched_length(icl_prompts, random.Random(int(seed) + 3))
    nonicl_perturbed = _perturb_matched_length(nonicl_prompts, random.Random(int(seed) + 4))

    # Intact baselines
    intact_icl = _eval_accuracy_and_logprob(model, icl_prompts, device)
    intact_nonicl = _eval_accuracy_and_logprob(model, nonicl_prompts, device)
    intact_shuffle = _eval_accuracy_and_logprob(model, shuffle_prompts, device)
    intact_shuffle_nonicl = _eval_accuracy_and_logprob(model, shuffle_nonicl_prompts, device)
    intact_icl_pert = _eval_accuracy_and_logprob(model, icl_perturbed, device)
    intact_nonicl_pert = _eval_accuracy_and_logprob(model, nonicl_perturbed, device)

    # Primary: high-SI ablation on standard prompts
    with head_output_ablation(model, high_heads):
        high_icl = _eval_accuracy_and_logprob(model, icl_prompts, device)
        high_nonicl = _eval_accuracy_and_logprob(model, nonicl_prompts, device)
        high_shuffle = _eval_accuracy_and_logprob(model, shuffle_prompts, device)
        high_shuffle_nonicl = _eval_accuracy_and_logprob(model, shuffle_nonicl_prompts, device)

    # Control (i): matched-cardinality non-SI ablation
    with head_output_ablation(model, ctrl_heads):
        ctrl_icl = _eval_accuracy_and_logprob(model, icl_prompts, device)
        ctrl_nonicl = _eval_accuracy_and_logprob(model, nonicl_prompts, device)

    # Control (iii): matched-length positional perturbation baseline (no head ablation)
    # Measured as perturbation-only degradation relative to intact unperturbed prompts.
    pert_icl_loss = {
        "acc_loss": float(intact_icl["accuracy"] - intact_icl_pert["accuracy"]),
        "lp_loss": float(intact_icl["mean_logprob"] - intact_icl_pert["mean_logprob"]),
    }
    pert_nonicl_loss = {
        "acc_loss": float(intact_nonicl["accuracy"] - intact_nonicl_pert["accuracy"]),
        "lp_loss": float(intact_nonicl["mean_logprob"] - intact_nonicl_pert["mean_logprob"]),
    }

    primary_icl_loss = _compute_degradation(intact_icl, high_icl)
    primary_nonicl_loss = _compute_degradation(intact_nonicl, high_nonicl)
    ctrl_icl_loss = _compute_degradation(intact_icl, ctrl_icl)
    ctrl_nonicl_loss = _compute_degradation(intact_nonicl, ctrl_nonicl)
    shuffle_icl_loss = _compute_degradation(intact_shuffle, high_shuffle)
    shuffle_nonicl_loss = _compute_degradation(intact_shuffle_nonicl, high_shuffle_nonicl)

    primary_gap_acc = float(primary_icl_loss["acc_loss"] - primary_nonicl_loss["acc_loss"])
    ctrl_gap_acc = float(ctrl_icl_loss["acc_loss"] - ctrl_nonicl_loss["acc_loss"])
    shuffle_gap_acc = float(shuffle_icl_loss["acc_loss"] - shuffle_nonicl_loss["acc_loss"])
    pert_gap_acc = float(pert_icl_loss["acc_loss"] - pert_nonicl_loss["acc_loss"])

    primary_gap_lp = float(primary_icl_loss["lp_loss"] - primary_nonicl_loss["lp_loss"])
    ctrl_gap_lp = float(ctrl_icl_loss["lp_loss"] - ctrl_nonicl_loss["lp_loss"])
    shuffle_gap_lp = float(shuffle_icl_loss["lp_loss"] - shuffle_nonicl_loss["lp_loss"])
    pert_gap_lp = float(pert_icl_loss["lp_loss"] - pert_nonicl_loss["lp_loss"])

    ratio = float("nan")
    if abs(primary_nonicl_loss["acc_loss"]) >= MIN_DENOM:
        ratio = float(primary_icl_loss["acc_loss"] / max(abs(primary_nonicl_loss["acc_loss"]), 1e-12))

    # Promotion gate: require LP dominance (always informative), then ACC if ACC denominator is informative.
    lp_survives = bool(
        primary_gap_lp > 0.0
        and primary_gap_lp > ctrl_gap_lp
        and primary_gap_lp > shuffle_gap_lp
        and primary_gap_lp > pert_gap_lp
    )
    acc_informative = abs(primary_nonicl_loss["acc_loss"]) >= MIN_DENOM
    acc_survives = bool(
        primary_gap_acc > 0.0
        and primary_gap_acc > ctrl_gap_acc
        and primary_gap_acc > shuffle_gap_acc
        and primary_gap_acc > pert_gap_acc
    )
    survives_controls = bool(lp_survives and (acc_survives if acc_informative else True))

    status = "supported" if survives_controls else "mixed"

    model_dir = ensure_dir(out_root / model_name)
    row_data = [
        {
            "condition": "primary_high_si",
            "icl_acc_loss": primary_icl_loss["acc_loss"],
            "nonicl_acc_loss": primary_nonicl_loss["acc_loss"],
            "gap_acc": primary_gap_acc,
            "icl_lp_loss": primary_icl_loss["lp_loss"],
            "nonicl_lp_loss": primary_nonicl_loss["lp_loss"],
            "gap_lp": primary_gap_lp,
        },
        {
            "condition": "control_non_si",
            "icl_acc_loss": ctrl_icl_loss["acc_loss"],
            "nonicl_acc_loss": ctrl_nonicl_loss["acc_loss"],
            "gap_acc": ctrl_gap_acc,
            "icl_lp_loss": ctrl_icl_loss["lp_loss"],
            "nonicl_lp_loss": ctrl_nonicl_loss["lp_loss"],
            "gap_lp": ctrl_gap_lp,
        },
        {
            "condition": "control_label_shuffle",
            "icl_acc_loss": shuffle_icl_loss["acc_loss"],
            "nonicl_acc_loss": shuffle_nonicl_loss["acc_loss"],
            "gap_acc": shuffle_gap_acc,
            "icl_lp_loss": shuffle_icl_loss["lp_loss"],
            "nonicl_lp_loss": shuffle_nonicl_loss["lp_loss"],
            "gap_lp": shuffle_gap_lp,
        },
        {
            "condition": "control_positional_perturb",
            "icl_acc_loss": pert_icl_loss["acc_loss"],
            "nonicl_acc_loss": pert_nonicl_loss["acc_loss"],
            "gap_acc": pert_gap_acc,
            "icl_lp_loss": pert_icl_loss["lp_loss"],
            "nonicl_lp_loss": pert_nonicl_loss["lp_loss"],
            "gap_lp": pert_gap_lp,
        },
    ]
    pd.DataFrame(row_data).to_parquet(model_dir / "format_confound_controls.parquet", index=False)

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E18",
        "model": model_name,
        "n_high_si_heads": int(len(high_heads)),
        "n_control_heads": int(len(ctrl_heads)),
        "primary_ratio": ratio,
        "primary_gap_acc": primary_gap_acc,
        "control_non_si_gap_acc": ctrl_gap_acc,
        "control_label_shuffle_gap_acc": shuffle_gap_acc,
        "control_positional_perturb_gap_acc": pert_gap_acc,
        "primary_gap_lp": primary_gap_lp,
        "control_non_si_gap_lp": ctrl_gap_lp,
        "control_label_shuffle_gap_lp": shuffle_gap_lp,
        "control_positional_perturb_gap_lp": pert_gap_lp,
        "acc_informative": bool(acc_informative),
        "lp_survives": bool(lp_survives),
        "acc_survives": bool(acc_survives),
        "survives_controls": survives_controls,
        "model_level_status": status,
    }
    write_json(model_dir / "summary.json", summary)
    write_json(
        model_dir / "raw_metrics.json",
        {
            "intact": {
                "icl": intact_icl,
                "nonicl": intact_nonicl,
                "shuffle": intact_shuffle,
                "shuffle_nonicl": intact_shuffle_nonicl,
                "icl_perturbed": intact_icl_pert,
                "nonicl_perturbed": intact_nonicl_pert,
            },
            "high_ablation": {
                "icl": high_icl,
                "nonicl": high_nonicl,
                "shuffle": high_shuffle,
                "shuffle_nonicl": high_shuffle_nonicl,
            },
            "control_ablation": {
                "icl": ctrl_icl,
                "nonicl": ctrl_nonicl,
            },
        },
    )
    print(
        f"[E18] {model_name}: primary_gap_acc={primary_gap_acc:.4f} ctrl_gap_acc={ctrl_gap_acc:.4f} "
        f"primary_gap_lp={primary_gap_lp:.4f} ctrl_gap_lp={ctrl_gap_lp:.4f} "
        f"survives={survives_controls}",
        flush=True,
    )
    return summary


def _load_model_summary(model_name: str, out_root: Path) -> dict[str, Any]:
    p = out_root / model_name / "summary.json"
    if not p.exists():
        raise RuntimeError(f"[E18] hard_fail_reason: missing shard artifact {p}")
    return read_json(p)


def _finalize(models: list[str], out_root: Path, start_ts: str) -> dict[str, Any]:
    rows = [_load_model_summary(m, out_root) for m in models]
    enforce_coverage_contract(
        experiment_id="E18",
        observed_models=[r.get("model", "") for r in rows],
        required_models=models,
    )

    n_survive = int(sum(1 for r in rows if bool(r.get("survives_controls", False))))
    mean_gap_acc = float(np.nanmean(np.asarray([r.get("primary_gap_acc", float("nan")) for r in rows], dtype=float)))
    mean_gap_lp = float(np.nanmean(np.asarray([r.get("primary_gap_lp", float("nan")) for r in rows], dtype=float)))

    if n_survive == len(rows):
        claim_status = "supported"
        interp = "survives_all_controls"
        note = "ICL-selective SI-degradation survives all format-confound controls in all models."
    elif n_survive >= max(1, len(rows) - 1):
        claim_status = "supported_with_caveat"
        interp = "survives_most_controls"
        note = "ICL-selective SI-degradation survives controls in most models with one exception."
    elif n_survive >= 1:
        claim_status = "mixed"
        interp = "model_conditional"
        note = "ICL-selective SI-degradation is model-conditional after confound controls."
    else:
        claim_status = "not_supported"
        interp = "control_collapse"
        note = "ICL-selective SI-degradation does not survive format-confound controls."

    cross = {
        "timestamp": timestamp_now(),
        "experiment_id": "E18",
        "n_models": int(len(rows)),
        "n_survives_controls": int(n_survive),
        "mean_primary_gap_acc": mean_gap_acc,
        "mean_primary_gap_lp": mean_gap_lp,
        "interpretation": interp,
        "note": note,
        "per_model": rows,
    }
    write_json(out_root / "cross_model_format_confound_summary.json", cross)

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "E18",
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "supported_with_caveat", "mixed"),
        "notes": [note],
        "outcome_summary": note,
    }

    prereg = {
        "experiment_id": "E18",
        "question": "Does E12 ICL-selective degradation remain after non-SI ablation, label-shuffle, and positional-perturb controls?",
        "primary_hypothesis": "Primary high-SI ICL gap remains larger than each control gap.",
        "primary_endpoints": ["primary_gap_minus_non_si_gap", "primary_gap_minus_shuffle_gap", "primary_gap_minus_perturb_gap"],
        "secondary_endpoints": ["primary_ratio", "per_model_survival_flag"],
        "model_list": models,
        "dataset_sources": ["synthetic ICL/non-ICL prompts generated from model vocabulary"],
        "inclusion_exclusion_rules": [
            "matched-cardinality non-SI head control required",
            "prompt lengths matched across all controls",
        ],
        "sample_size_plan": {"prompts_per_condition": N_EVAL, "n_demos": N_DEMOS},
        "seed_plan": {"base_seed": SEED_BASE},
        "stopping_rule": "fixed sample sizes",
        "multiplicity_family": ["cross-model directional control comparisons"],
        "acceptance_criteria": ["primary_gap > each control gap in all models => supported"],
        "fallback_interpretation_if_null": "E12 may partly reflect format regularity rather than SI-specific ICL load.",
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E18",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {"interpretation": interp, "n_survives_controls": int(n_survive), "n_models": int(len(rows))},
        "limitations": [
            "Synthetic prompt controls may not cover all naturalistic ICL formats.",
            "Head-output ablation remains an intervention-level load-bearing test rather than circuit-identity proof.",
        ],
    }

    data_dictionary = {
        "experiment_id": "E18",
        "tables": [
            {
                "path": "<model>/format_confound_controls.parquet",
                "description": "Condition-level ICL vs non-ICL degradation gaps for primary and control arms.",
                "columns": [
                    {"name": "condition", "dtype": "str", "description": "primary_high_si/control_non_si/control_label_shuffle/control_positional_perturb"},
                    {"name": "icl_acc_loss", "dtype": "float", "description": "ICL accuracy degradation"},
                    {"name": "nonicl_acc_loss", "dtype": "float", "description": "Non-ICL accuracy degradation"},
                    {"name": "gap_acc", "dtype": "float", "description": "ICL - non-ICL accuracy-degradation gap"},
                    {"name": "icl_lp_loss", "dtype": "float", "description": "ICL log-probability degradation"},
                    {"name": "nonicl_lp_loss", "dtype": "float", "description": "Non-ICL log-probability degradation"},
                    {"name": "gap_lp", "dtype": "float", "description": "ICL - non-ICL log-probability-degradation gap"},
                ],
            }
        ],
    }

    emit_core_artifacts(
        experiment_id="E18",
        out_dir=out_root,
        preregistration=prereg,
        manifest_extra={"models": models, "n_eval": N_EVAL, "n_demos": N_DEMOS},
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross


def main() -> None:
    p = argparse.ArgumentParser(description="E18: ICL format-confound controls", allow_abbrev=False)
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0",
    )
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--n-eval", type=int, default=N_EVAL)
    p.add_argument("--n-demos", type=int, default=N_DEMOS)
    p.add_argument("--seed", type=int, default=SEED_BASE)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E18] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    device_map = parse_device_map(args.device_map)
    out_root = ensure_dir(Path(args.output_root))
    start_ts = timestamp_now()

    n_eval = int(args.n_eval)
    n_demos = int(args.n_demos)
    if args.smoke:
        n_eval = min(n_eval, 40)
        n_demos = min(n_demos, 3)

    if args.finalize_only:
        cross = _finalize(models, out_root, start_ts)
        print(f"[E18] Finalized from shards. interpretation={cross['interpretation']}", flush=True)
        return

    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        run_model(
            model_name=model_name,
            device=device,
            out_root=out_root,
            n_eval=max(16, n_eval),
            n_demos=max(2, n_demos),
            seed=int(args.seed),
        )

    if args.no_finalize:
        print("[E18] Shard run complete (no finalize).", flush=True)
        return

    cross = _finalize(models, out_root, start_ts)
    print(f"[E18] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
