#!/usr/bin/env python3
"""E28 — Probe-Scope Battery for Result III.

Purpose
-------
Address the critique that E12's 9-token ICL probe is structurally aligned with
SI by design. E28 tests whether preferential disruption under SI-kernel
subtraction extends across a broader family of offset-structured retrieval probes
that vary sequence length and structural regularity.

Design
------
- Models: primary 7-8B panel (llama, mistral, olmo)
- Intervention arms:
    1) true SI-kernel subtraction (high-SI heads)
    2) offset-permuted kernel subtraction (same heads)
- Probe families (4): short/medium/long/irregular offset structures
- Controls: matched-layout non-repetition retrieval prompts for each family

Primary inference
-----------------
For each model x family, compute preferential-gap in LP units:
    gap = (probe_lp_loss - control_lp_loss)
where loss = intact_lp - ablated_lp.

Primary family pass requires:
    gap_true_lp > 0 and (gap_true_lp - gap_perm_lp) > 0.

Model-level primary pass requires >= 3/4 family passes.
Cross-model interpretation:
    supported            : 3/3 model passes
    supported_with_caveat: 2/3
    mixed                : 1/3
    not_supported        : 0/3

This is a forward-pass-only scope-broadening experiment.
"""
from __future__ import annotations

import argparse
import json
import random
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
from experiment3.theory8_position_ablation import (  # noqa: E402
    subtract_positional_kernels,
    load_head_groups,
)

OUT_ROOT = RESULTS_ROOT / "E28_probe_scope_battery"
SEED_BASE = 20260503
N_PROMPTS_DEFAULT = 120
BATCH_SIZE_DEFAULT = 16


@dataclass(frozen=True)
class FamilySpec:
    name: str
    n_demos: int
    inpair_gap: tuple[int, int]  # fillers between key and value
    between_gap: tuple[int, int]  # fillers between demos


FAMILIES: list[FamilySpec] = [
    FamilySpec("short_fixed_offset", n_demos=4, inpair_gap=(0, 0), between_gap=(0, 0)),   # len=9
    FamilySpec("medium_spaced_offset", n_demos=8, inpair_gap=(0, 0), between_gap=(1, 1)), # len=25
    FamilySpec("long_fixed_offset", n_demos=8, inpair_gap=(2, 2), between_gap=(1, 1)),    # len=41
    FamilySpec("irregular_mixed_offset", n_demos=10, inpair_gap=(0, 2), between_gap=(0, 2)),
]


def _load_kernels(model_name: str) -> tuple[dict[tuple[int, int], np.ndarray], str]:
    candidates = [
        ROOT / "results" / "reinforce_exp" / "exp_r3_core_replication" / model_name / "theory8_position_ablation" / model_name / "estimated_kernels.json",
        ROOT / "results" / "experiment3" / "theory8_position_ablation" / model_name / "estimated_kernels.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        out: dict[tuple[int, int], np.ndarray] = {}
        for k, v in raw.items():
            if not (k.startswith("L") and "H" in k):
                continue
            left, right = k[1:].split("H", 1)
            out[(int(left), int(right))] = np.asarray(v, dtype=np.float32)
        if out:
            return out, str(p)
    raise FileNotFoundError(
        f"[E28] No estimated_kernels.json found for {model_name}; checked: {candidates}"
    )


def _permute_kernels(
    kernels: dict[tuple[int, int], np.ndarray],
    heads: list[tuple[int, int]],
    rng: np.random.Generator,
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    for head in heads:
        g = kernels.get(head)
        if g is None:
            continue
        idx = rng.permutation(len(g))
        out[head] = g[idx].copy()
    return out


def _candidate_vocab(tokenizer: Any) -> list[int]:
    vocab = tokenizer.get_vocab()
    ids = sorted(set(int(v) for v in vocab.values()))
    special = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)
    ids = [x for x in ids if x not in special]
    if len(ids) < 256:
        # fallback: include all ids if filtering is too aggressive
        ids = sorted(set(int(v) for v in vocab.values()))
    return ids


def _sample_span(low_high: tuple[int, int], rng: random.Random) -> int:
    lo, hi = int(low_high[0]), int(low_high[1])
    if hi < lo:
        lo, hi = hi, lo
    return int(rng.randint(lo, hi))


def _sample_token(vocab_ids: list[int], rng: random.Random) -> int:
    return int(vocab_ids[rng.randrange(len(vocab_ids))])


def _sample_distinct(vocab_ids: list[int], rng: random.Random, forbid: set[int]) -> int:
    for _ in range(64):
        t = _sample_token(vocab_ids, rng)
        if t not in forbid:
            return t
    return _sample_token(vocab_ids, rng)


def _build_prompt_pair(
    *,
    spec: FamilySpec,
    vocab_ids: list[int],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (probe_prompt, control_prompt) with matched layout."""
    # Probe mapping (repeated): same key->value across all demos
    key = _sample_token(vocab_ids, rng)
    val = _sample_distinct(vocab_ids, rng, {key})

    # Control mapping: independent key/value per demo, query intentionally remapped
    ctrl_keys = [_sample_token(vocab_ids, rng) for _ in range(spec.n_demos)]
    ctrl_vals = [_sample_distinct(vocab_ids, rng, {ctrl_keys[i]}) for i in range(spec.n_demos)]

    probe_ids: list[int] = []
    ctrl_ids: list[int] = []

    for d in range(spec.n_demos):
        inpair = _sample_span(spec.inpair_gap, rng)
        between = _sample_span(spec.between_gap, rng)

        # Key
        probe_ids.append(key)
        ctrl_ids.append(ctrl_keys[d])

        # Fillers inside pair
        fillers_inpair = [_sample_token(vocab_ids, rng) for _ in range(inpair)]
        probe_ids.extend(fillers_inpair)
        ctrl_ids.extend(fillers_inpair)

        # Value
        probe_ids.append(val)
        ctrl_ids.append(ctrl_vals[d])

        # Fillers between demos
        fillers_between = [_sample_token(vocab_ids, rng) for _ in range(between)]
        probe_ids.extend(fillers_between)
        ctrl_ids.extend(fillers_between)

    # Query+target
    probe_query = key
    probe_target = val

    q_idx = rng.randrange(spec.n_demos)
    ctrl_query = ctrl_keys[q_idx]
    ctrl_target = ctrl_vals[(q_idx + 1) % spec.n_demos]
    if ctrl_target == ctrl_vals[q_idx]:
        ctrl_target = ctrl_vals[(q_idx + 2) % spec.n_demos]

    probe_ids.append(probe_query)
    ctrl_ids.append(ctrl_query)

    probe = {
        "family": spec.name,
        "condition": "probe_repetition",
        "input_ids": probe_ids,
        "target_token_id": int(probe_target),
    }
    ctrl = {
        "family": spec.name,
        "condition": "control_nonrepetition",
        "input_ids": ctrl_ids,
        "target_token_id": int(ctrl_target),
    }
    return probe, ctrl


def _build_family_prompts(
    *,
    spec: FamilySpec,
    vocab_ids: list[int],
    n_prompts: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(int(seed))
    probe: list[dict[str, Any]] = []
    ctrl: list[dict[str, Any]] = []
    for _ in range(int(n_prompts)):
        p, c = _build_prompt_pair(spec=spec, vocab_ids=vocab_ids, rng=rng)
        probe.append(p)
        ctrl.append(c)
    return probe, ctrl


@torch.no_grad()
def _eval_prompts(
    *,
    model: Any,
    prompts: list[dict[str, Any]],
    device: str,
    batch_size: int,
    pad_token_id: int,
) -> dict[str, float]:
    model.eval()
    correct: list[int] = []
    lps: list[float] = []

    bs = max(1, int(batch_size))
    for i in range(0, len(prompts), bs):
        chunk = prompts[i : i + bs]
        lens = [len(x["input_ids"]) for x in chunk]
        mx = int(max(lens))

        input_ids = torch.full((len(chunk), mx), int(pad_token_id), dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(chunk), mx), dtype=torch.long, device=device)

        targets: list[int] = []
        for r, p in enumerate(chunk):
            ids = p["input_ids"]
            n = len(ids)
            input_ids[r, :n] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[r, :n] = 1
            targets.append(int(p["target_token_id"]))

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, T, V]

        for r, n in enumerate(lens):
            pos = int(n - 1)
            lg = logits[r, pos]
            pred = int(torch.argmax(lg).item())
            tgt = int(targets[r])
            lp = float(torch.log_softmax(lg, dim=-1)[tgt].item())
            correct.append(int(pred == tgt))
            lps.append(lp)

        del input_ids, attention_mask, out, logits

    return {
        "accuracy": float(np.mean(correct)) if correct else float("nan"),
        "mean_logprob": float(np.mean(lps)) if lps else float("nan"),
        "n_prompts": float(len(prompts)),
    }


def _compute_loss_delta(intact: dict[str, float], ablated: dict[str, float]) -> dict[str, float]:
    return {
        "acc_loss": float(intact["accuracy"] - ablated["accuracy"]),
        "lp_loss": float(intact["mean_logprob"] - ablated["mean_logprob"]),
    }


def _family_trial_rows(
    *,
    model_name: str,
    family: str,
    intact_probe: dict[str, float],
    intact_ctrl: dict[str, float],
    true_probe: dict[str, float],
    true_ctrl: dict[str, float],
    perm_probe: dict[str, float],
    perm_ctrl: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    true_probe_loss = _compute_loss_delta(intact_probe, true_probe)
    true_ctrl_loss = _compute_loss_delta(intact_ctrl, true_ctrl)
    perm_probe_loss = _compute_loss_delta(intact_probe, perm_probe)
    perm_ctrl_loss = _compute_loss_delta(intact_ctrl, perm_ctrl)

    gap_true_acc = float(true_probe_loss["acc_loss"] - true_ctrl_loss["acc_loss"])
    gap_true_lp = float(true_probe_loss["lp_loss"] - true_ctrl_loss["lp_loss"])
    gap_perm_acc = float(perm_probe_loss["acc_loss"] - perm_ctrl_loss["acc_loss"])
    gap_perm_lp = float(perm_probe_loss["lp_loss"] - perm_ctrl_loss["lp_loss"])

    delta_gap_acc = float(gap_true_acc - gap_perm_acc)
    delta_gap_lp = float(gap_true_lp - gap_perm_lp)

    family_pass_lp = bool(gap_true_lp > 0.0 and delta_gap_lp > 0.0)
    family_pass_acc = bool(gap_true_acc > 0.0 and delta_gap_acc > 0.0)

    rows = [
        {
            "model": model_name,
            "family": family,
            "arm": "true_si_subtraction",
            "probe_acc_loss": true_probe_loss["acc_loss"],
            "control_acc_loss": true_ctrl_loss["acc_loss"],
            "gap_acc": gap_true_acc,
            "probe_lp_loss": true_probe_loss["lp_loss"],
            "control_lp_loss": true_ctrl_loss["lp_loss"],
            "gap_lp": gap_true_lp,
        },
        {
            "model": model_name,
            "family": family,
            "arm": "permuted_subtraction",
            "probe_acc_loss": perm_probe_loss["acc_loss"],
            "control_acc_loss": perm_ctrl_loss["acc_loss"],
            "gap_acc": gap_perm_acc,
            "probe_lp_loss": perm_probe_loss["lp_loss"],
            "control_lp_loss": perm_ctrl_loss["lp_loss"],
            "gap_lp": gap_perm_lp,
        },
    ]

    fam = {
        "model": model_name,
        "family": family,
        "gap_true_acc": gap_true_acc,
        "gap_true_lp": gap_true_lp,
        "gap_perm_acc": gap_perm_acc,
        "gap_perm_lp": gap_perm_lp,
        "delta_gap_acc": delta_gap_acc,
        "delta_gap_lp": delta_gap_lp,
        "family_pass_lp": family_pass_lp,
        "family_pass_acc": family_pass_acc,
        "family_pass": bool(family_pass_lp),
    }
    return rows, fam


def run_model(
    *,
    model_name: str,
    device: str,
    out_root: Path,
    n_prompts: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    print(f"[E28] loading {model_name} on {device}", flush=True)
    model, tokenizer = load_model_for_exp(model_name, device, attn_implementation="eager")

    head_groups = load_head_groups(model_name)
    high_heads = [(int(l), int(h)) for (l, h) in head_groups.get("high_si", [])]
    if not high_heads:
        raise RuntimeError(f"[E28] no high_si heads available for {model_name}")

    kernels, kernel_path = _load_kernels(model_name)
    rng_perm = np.random.default_rng(int(seed) + 17)
    perm_kernels = _permute_kernels(kernels, high_heads, rng_perm)

    vocab_ids = _candidate_vocab(tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    trial_rows: list[dict[str, Any]] = []
    fam_rows: list[dict[str, Any]] = []

    for idx, spec in enumerate(FAMILIES):
        family_seed = int(seed) + 1000 * (idx + 1)
        probe_prompts, ctrl_prompts = _build_family_prompts(
            spec=spec,
            vocab_ids=vocab_ids,
            n_prompts=int(n_prompts),
            seed=family_seed,
        )
        seq_len = max(max(len(p["input_ids"]) for p in probe_prompts), max(len(p["input_ids"]) for p in ctrl_prompts))

        intact_probe = _eval_prompts(
            model=model,
            prompts=probe_prompts,
            device=device,
            batch_size=batch_size,
            pad_token_id=int(pad_id),
        )
        intact_ctrl = _eval_prompts(
            model=model,
            prompts=ctrl_prompts,
            device=device,
            batch_size=batch_size,
            pad_token_id=int(pad_id),
        )

        with subtract_positional_kernels(model, kernels, high_heads, int(seq_len)):
            true_probe = _eval_prompts(
                model=model,
                prompts=probe_prompts,
                device=device,
                batch_size=batch_size,
                pad_token_id=int(pad_id),
            )
            true_ctrl = _eval_prompts(
                model=model,
                prompts=ctrl_prompts,
                device=device,
                batch_size=batch_size,
                pad_token_id=int(pad_id),
            )

        with subtract_positional_kernels(model, perm_kernels, high_heads, int(seq_len)):
            perm_probe = _eval_prompts(
                model=model,
                prompts=probe_prompts,
                device=device,
                batch_size=batch_size,
                pad_token_id=int(pad_id),
            )
            perm_ctrl = _eval_prompts(
                model=model,
                prompts=ctrl_prompts,
                device=device,
                batch_size=batch_size,
                pad_token_id=int(pad_id),
            )

        rows, fam = _family_trial_rows(
            model_name=model_name,
            family=spec.name,
            intact_probe=intact_probe,
            intact_ctrl=intact_ctrl,
            true_probe=true_probe,
            true_ctrl=true_ctrl,
            perm_probe=perm_probe,
            perm_ctrl=perm_ctrl,
        )
        trial_rows.extend(rows)
        fam_rows.append(fam)

        print(
            f"[E28] {model_name} {spec.name}: gap_true_lp={fam['gap_true_lp']:.4f} "
            f"gap_perm_lp={fam['gap_perm_lp']:.4f} delta_lp={fam['delta_gap_lp']:.4f} "
            f"pass_lp={fam['family_pass_lp']}",
            flush=True,
        )

    fam_df = pd.DataFrame(fam_rows)
    n_family = int(len(fam_df))
    n_pass_lp = int(fam_df["family_pass_lp"].sum()) if n_family else 0
    n_pass_acc = int(fam_df["family_pass_acc"].sum()) if n_family else 0

    # model-level primary pass: >=3/4 LP family passes
    model_pass = bool(n_pass_lp >= 3)
    status = "supported" if model_pass else "mixed"

    model_dir = ensure_dir(out_root / model_name)
    pd.DataFrame(trial_rows).to_parquet(model_dir / "probe_scope_trials.parquet", index=False)
    fam_df.to_parquet(model_dir / "family_summary.parquet", index=False)

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E28",
        "model": model_name,
        "n_high_si_heads": int(len(high_heads)),
        "kernel_source": kernel_path,
        "n_families": n_family,
        "n_family_pass_lp": n_pass_lp,
        "n_family_pass_acc": n_pass_acc,
        "model_primary_pass": model_pass,
        "model_level_status": status,
        "mean_delta_gap_lp": float(fam_df["delta_gap_lp"].mean()) if n_family else float("nan"),
        "mean_delta_gap_acc": float(fam_df["delta_gap_acc"].mean()) if n_family else float("nan"),
    }
    write_json(model_dir / "summary.json", summary)
    return summary


def _load_model_summary(model_name: str, out_root: Path) -> dict[str, Any]:
    p = out_root / model_name / "summary.json"
    if not p.exists():
        raise RuntimeError(f"[E28] hard_fail_reason: missing shard artifact {p}")
    return read_json(p)


def _finalize(models: list[str], out_root: Path) -> dict[str, Any]:
    rows = [_load_model_summary(m, out_root) for m in models]
    enforce_coverage_contract(
        experiment_id="E28",
        observed_models=[r.get("model", "") for r in rows],
        required_models=models,
    )

    n_pass = int(sum(1 for r in rows if bool(r.get("model_primary_pass", False))))
    if n_pass == len(rows):
        claim_status = "supported"
        interp = "preferential_disruption_generalizes_across_probe_families"
        note = "All models pass the multi-family probe-scope gate (>=3/4 LP-family passes)."
    elif n_pass == len(rows) - 1:
        claim_status = "supported_with_caveat"
        interp = "generalizes_in_most_models"
        note = "Most models pass the multi-family gate; one model fails."
    elif n_pass >= 1:
        claim_status = "mixed"
        interp = "model_conditional_generalization"
        note = "Probe-family generalization is model-conditional across primary models."
    else:
        claim_status = "not_supported"
        interp = "no_generalization_beyond_short_probe"
        note = "No model passes the multi-family gate beyond the original short probe setting."

    cross = {
        "timestamp": timestamp_now(),
        "experiment_id": "E28",
        "n_models": int(len(rows)),
        "n_model_pass": int(n_pass),
        "interpretation": interp,
        "note": note,
        "per_model": rows,
    }
    write_json(out_root / "cross_model_probe_scope_summary.json", cross)

    prereg = {
        "experiment_id": "E28",
        "question": "Does SI preferential disruption persist beyond the short 9-token offset-repetition probe?",
        "primary_hypothesis": "True SI subtraction yields larger probe-vs-control LP degradation gaps than permuted subtraction across multiple probe families.",
        "primary_endpoints": [
            "family_gap_true_lp",
            "family_delta_gap_lp_true_minus_permuted",
            "model_primary_pass_geq_3_of_4_families",
        ],
        "secondary_endpoints": [
            "family_gap_true_acc",
            "family_delta_gap_acc_true_minus_permuted",
        ],
        "model_list": models,
        "probe_families": [
            {"name": f.name, "n_demos": f.n_demos, "inpair_gap": list(f.inpair_gap), "between_gap": list(f.between_gap)}
            for f in FAMILIES
        ],
        "sample_size_plan": {"prompts_per_family_per_condition": N_PROMPTS_DEFAULT},
        "seed_plan": {"base_seed": SEED_BASE},
        "acceptance_criteria": [
            "family pass: gap_true_lp>0 and delta_gap_lp>0",
            "model pass: >=3/4 families pass",
            "cross-model supported: all models pass",
        ],
        "fallback_interpretation_if_null": "Preferential SI disruption is likely specific to highly aligned short offset-repetition formats.",
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E28",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": interp,
            "n_model_pass": int(n_pass),
            "n_models": int(len(rows)),
        },
        "limitations": [
            "Probe families remain synthetic and are not naturalistic long-context ICL benchmarks.",
            "Kernel-subtraction intervention measures disruption cost, not training-time counterfactual necessity.",
        ],
    }

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "E28",
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "supported_with_caveat", "mixed"),
        "outcome_summary": note,
        "notes": [
            "Designed to test whether Result-III preferential disruption generalizes beyond a single short probe format.",
            note,
        ],
    }

    data_dictionary = {
        "experiment_id": "E28",
        "tables": [
            {
                "path": "<model>/probe_scope_trials.parquet",
                "description": "Per-family arm-level probe vs control degradation metrics for true and permuted SI-kernel subtraction.",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model identifier"},
                    {"name": "family", "dtype": "str", "description": "Probe-family identifier"},
                    {"name": "arm", "dtype": "str", "description": "true_si_subtraction or permuted_subtraction"},
                    {"name": "probe_acc_loss", "dtype": "float", "description": "Accuracy degradation on probe condition"},
                    {"name": "control_acc_loss", "dtype": "float", "description": "Accuracy degradation on matched control condition"},
                    {"name": "gap_acc", "dtype": "float", "description": "Probe-control accuracy degradation gap"},
                    {"name": "probe_lp_loss", "dtype": "float", "description": "Log-probability degradation on probe"},
                    {"name": "control_lp_loss", "dtype": "float", "description": "Log-probability degradation on control"},
                    {"name": "gap_lp", "dtype": "float", "description": "Probe-control LP degradation gap"},
                ],
            },
            {
                "path": "<model>/family_summary.parquet",
                "description": "Per-family pass/fail indicators and true-vs-permuted gap contrasts.",
                "columns": [
                    {"name": "family", "dtype": "str", "description": "Probe-family identifier"},
                    {"name": "gap_true_lp", "dtype": "float", "description": "Probe-control LP gap under true subtraction"},
                    {"name": "gap_perm_lp", "dtype": "float", "description": "Probe-control LP gap under permuted subtraction"},
                    {"name": "delta_gap_lp", "dtype": "float", "description": "(true gap) - (permuted gap) LP"},
                    {"name": "family_pass_lp", "dtype": "bool", "description": "Primary LP family pass"},
                ],
            },
        ],
    }

    emit_core_artifacts(
        experiment_id="E28",
        out_dir=out_root,
        preregistration=prereg,
        manifest_extra={
            "models": models,
            "n_prompts": N_PROMPTS_DEFAULT,
            "families": [f.name for f in FAMILIES],
        },
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross


def main() -> None:
    p = argparse.ArgumentParser(description="E28: probe-scope battery for Result III", allow_abbrev=False)
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0",
    )
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--n-prompts", type=int, default=N_PROMPTS_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_BASE)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E28] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    device_map = parse_device_map(args.device_map)
    out_root = ensure_dir(Path(args.output_root))

    n_prompts = int(args.n_prompts)
    batch_size = int(args.batch_size)
    if args.smoke:
        n_prompts = min(24, n_prompts)
        batch_size = min(8, batch_size)

    if args.finalize_only:
        cross = _finalize(models, out_root)
        print(f"[E28] Finalized from shards. interpretation={cross['interpretation']}", flush=True)
        return

    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        run_model(
            model_name=model_name,
            device=device,
            out_root=out_root,
            n_prompts=max(8, n_prompts),
            batch_size=max(1, batch_size),
            seed=int(args.seed),
        )

    if args.no_finalize:
        print("[E28] Shard run complete (no finalize).", flush=True)
        return

    cross = _finalize(models, out_root)
    print(f"[E28] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
