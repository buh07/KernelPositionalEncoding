#!/usr/bin/env python3
"""E30A — Boundary grid for Result III transfer scope.

2x2 grid:
- context length: short vs long
- offset regularity: high vs low

Goal
----
Disambiguate why E28C (broad synthetic family transfer) can fail while E29B
(semi-naturalistic long-context passkey) partially passes.

Interpretation focus
--------------------
Whether SI-linked preferential disruption is strongest in high-regularity
retrieval structure and whether context length modulates that signal.
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
    rng_for_stream,
)
from experiment3.theory8_position_ablation import (  # noqa: E402
    load_head_groups,
    subtract_positional_kernels,
)
from experiment5.pipeline import (  # noqa: E402
    _load_wiki_sequences,
)

EXPERIMENT_ID = "E30A"
OUT_ROOT = RESULTS_ROOT / "E30a_probe_boundary_grid"


@dataclass(frozen=True)
class FamilySpec:
    name: str
    context_band: str  # short | long
    regularity_band: str  # high | low
    context_len: int
    n_demos: int
    base_offset: int


FAMILIES: list[FamilySpec] = [
    FamilySpec("short_high", context_band="short", regularity_band="high", context_len=128, n_demos=4, base_offset=3),
    FamilySpec("short_low", context_band="short", regularity_band="low", context_len=128, n_demos=4, base_offset=3),
    FamilySpec("long_high", context_band="long", regularity_band="high", context_len=512, n_demos=6, base_offset=4),
    FamilySpec("long_low", context_band="long", regularity_band="low", context_len=512, n_demos=6, base_offset=4),
]

_KERNEL_CANDIDATES = [
    "results/reinforce_exp/exp_r3_core_replication/{model}/theory8_position_ablation/{model}/estimated_kernels.json",
    "results/experiment3/theory8_position_ablation/{model}/estimated_kernels.json",
]


def _load_kernels(model_name: str) -> dict[tuple[int, int], np.ndarray]:
    for template in _KERNEL_CANDIDATES:
        p = ROOT / template.format(model=model_name)
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
            return out
    raise FileNotFoundError(f"[E30A] No estimated_kernels.json for {model_name}")


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
        out[head] = g[rng.permutation(len(g))].copy()
    return out


def _candidate_vocab_ids(tokenizer: Any) -> list[int]:
    vocab = tokenizer.get_vocab()
    ids = sorted(set(int(v) for v in vocab.values()))
    special = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)
    keep = [x for x in ids if x not in special]
    return keep if len(keep) >= 256 else ids


def _sample_distinct_token(cands: list[int], forbid: set[int], rng: random.Random) -> int:
    for _ in range(64):
        t = int(cands[rng.randrange(len(cands))])
        if t not in forbid:
            return t
    return int(cands[rng.randrange(len(cands))])


def _load_sequences(
    *,
    model_name: str,
    tokenizer: Any,
    context_len: int,
    max_sequences: int,
    seed: int,
) -> list[list[int]]:
    return _load_wiki_sequences(
        model_name,
        tokenizer,
        seq_len=int(context_len - 1),
        max_sequences=max_sequences,
        seed=seed,
    )


def _build_prompt_pair(
    *,
    spec: FamilySpec,
    base_tokens: list[int],
    vocab_ids: list[int],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    ctx = list(base_tokens[: spec.context_len - 1])
    if len(ctx) < spec.context_len - 1:
        return None

    probe = list(ctx)
    control = list(ctx)

    uniq = list(dict.fromkeys(ctx))
    if len(uniq) < 16:
        return None

    key = int(uniq[rng.randrange(len(uniq))])
    val = int(_sample_distinct_token(vocab_ids, {key}, rng))

    start = 12
    end = max(start + spec.n_demos + spec.base_offset + 8, len(ctx) - 20)
    pos = np.linspace(start, end, spec.n_demos, dtype=int)

    used_ctrl: set[int] = set()
    ctrl_keys: list[int] = []
    ctrl_vals: list[int] = []

    for p in pos:
        key_pos = int(max(4, min(len(ctx) - 10, int(p))))

        if spec.regularity_band == "high":
            off = int(spec.base_offset)
        else:
            # Lower regularity: vary offsets per demo and add occasional extra shift.
            off = int(rng.choice([1, 2, 4, 6]))
            if rng.random() < 0.35:
                key_pos = int(max(4, min(len(ctx) - 10, key_pos + rng.choice([-2, -1, 1, 2]))))

        val_pos = int(max(key_pos + 1, min(len(ctx) - 3, key_pos + off)))

        # Probe: consistent key->value mapping, with offset variability in low-regularity arm.
        probe[key_pos] = key
        probe[val_pos] = val

        # Control: break retrievable mapping consistency.
        ck = _sample_distinct_token(vocab_ids, used_ctrl, rng)
        used_ctrl.add(ck)
        cv = _sample_distinct_token(vocab_ids, used_ctrl | {ck}, rng)
        used_ctrl.add(cv)
        ctrl_keys.append(int(ck))
        ctrl_vals.append(int(cv))
        control[key_pos] = int(ck)
        control[val_pos] = int(cv)

        # Add distractor mapping in low-regularity families to reduce periodic structure.
        if spec.regularity_band == "low" and rng.random() < 0.45:
            dkp = int(max(2, min(len(ctx) - 5, key_pos + rng.choice([-6, -4, 4, 6]))))
            dvp = int(max(dkp + 1, min(len(ctx) - 2, dkp + rng.choice([1, 3, 5]))))
            dk = _sample_distinct_token(vocab_ids, {key, val}, rng)
            dv = _sample_distinct_token(vocab_ids, {key, val, dk}, rng)
            probe[dkp] = dk
            probe[dvp] = dv
            control[dkp] = dk
            control[dvp] = dv

    if len(ctrl_keys) < 2:
        return None

    probe_query = key
    probe_target = val
    ctrl_query = ctrl_keys[0]
    ctrl_target = ctrl_vals[min(1, len(ctrl_vals) - 1)]

    probe_prompt = {
        "family": spec.name,
        "context_band": spec.context_band,
        "regularity_band": spec.regularity_band,
        "condition": "probe_retrievable",
        "input_ids": probe + [probe_query],
        "target_token_id": int(probe_target),
    }
    control_prompt = {
        "family": spec.name,
        "context_band": spec.context_band,
        "regularity_band": spec.regularity_band,
        "condition": "control_nonretrievable",
        "input_ids": control + [ctrl_query],
        "target_token_id": int(ctrl_target),
    }
    return probe_prompt, control_prompt


def _build_family_prompts(
    *,
    model_name: str,
    tokenizer: Any,
    spec: FamilySpec,
    n_prompts: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(int(seed))
    vocab_ids = _candidate_vocab_ids(tokenizer)
    seqs = _load_sequences(
        model_name=model_name,
        tokenizer=tokenizer,
        context_len=int(spec.context_len),
        max_sequences=max(4 * n_prompts, n_prompts + 32),
        seed=seed,
    )

    probe: list[dict[str, Any]] = []
    control: list[dict[str, Any]] = []
    for seq in seqs:
        out = _build_prompt_pair(spec=spec, base_tokens=seq, vocab_ids=vocab_ids, rng=rng)
        if out is None:
            continue
        p, c = out
        probe.append(p)
        control.append(c)
        if len(probe) >= n_prompts:
            break

    if len(probe) < max(8, n_prompts // 2):
        raise RuntimeError(
            f"[E30A] insufficient prompts for {model_name}/{spec.name}: {len(probe)} < {max(8, n_prompts // 2)}"
        )

    n = min(len(probe), len(control), n_prompts)
    return probe[:n], control[:n]


@torch.no_grad()
def _eval_target_metrics(
    *,
    model: Any,
    prompts: list[dict[str, Any]],
    device: str,
    pad_token_id: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    lp_vals: list[float] = []
    acc_vals: list[int] = []

    bs = max(1, int(batch_size))
    pos = 0
    while pos < len(prompts):
        chunk = prompts[pos : pos + bs]
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

        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                continue
            raise

        logits = out.logits
        for r, target in enumerate(targets):
            last_pos = int(lens[r] - 1)
            row_logits = logits[r, last_pos, :]
            log_probs = torch.log_softmax(row_logits, dim=-1)
            lp_vals.append(float(log_probs[target].detach().cpu().item()))
            pred = int(torch.argmax(row_logits).detach().cpu().item())
            acc_vals.append(1 if pred == target else 0)

        pos += len(chunk)
        del input_ids, attention_mask, out, logits
        torch.cuda.empty_cache()

    return {
        "lp": np.asarray(lp_vals, dtype=np.float64),
        "acc": np.asarray(acc_vals, dtype=np.int64),
    }


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_one_gt_zero": float("nan")}
    rng = np.random.default_rng(int(seed))
    n = int(arr.size)
    boot = np.empty(max(1500, int(n_boot)), dtype=np.float64)
    for i in range(boot.size):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(arr[idx]))
    return {
        "mean": float(np.mean(arr)),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "p_one_gt_zero": float((np.sum(boot <= 0.0) + 1) / (boot.size + 1)),
    }


def run_model(
    *,
    model_name: str,
    device: str,
    out_root: Path,
    n_prompts: int,
    batch_size: int,
    seq_len: int,
    n_control_trials: int,
    seed: int,
) -> dict[str, Any]:
    t0 = time.time()
    model_dir = ensure_dir(out_root / model_name)

    model, tokenizer = load_model_for_exp(model_name, device=device, attn_implementation="eager")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = int(tokenizer.pad_token_id)

    kernels = _load_kernels(model_name)
    high_heads = [
        (int(l), int(h))
        for (l, h) in load_head_groups(model_name).get("high_si", [])
        if (int(l), int(h)) in kernels
    ]
    if not high_heads:
        raise RuntimeError(f"[E30A] no high_si heads available for {model_name}")

    family_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    max_ctx_len = max(f.context_len for f in FAMILIES)

    for fidx, spec in enumerate(FAMILIES):
        probe_prompts, ctrl_prompts = _build_family_prompts(
            model_name=model_name,
            tokenizer=tokenizer,
            spec=spec,
            n_prompts=n_prompts,
            seed=seed + fidx * 97,
        )

        intact_probe = _eval_target_metrics(
            model=model,
            prompts=probe_prompts,
            device=device,
            pad_token_id=pad_id,
            batch_size=batch_size,
        )
        intact_ctrl = _eval_target_metrics(
            model=model,
            prompts=ctrl_prompts,
            device=device,
            pad_token_id=pad_id,
            batch_size=batch_size,
        )

        with subtract_positional_kernels(model, kernels, high_heads, int(max(seq_len, spec.context_len))):
            true_probe = _eval_target_metrics(
                model=model,
                prompts=probe_prompts,
                device=device,
                pad_token_id=pad_id,
                batch_size=batch_size,
            )
            true_ctrl = _eval_target_metrics(
                model=model,
                prompts=ctrl_prompts,
                device=device,
                pad_token_id=pad_id,
                batch_size=batch_size,
            )

        perm_probe_runs: list[np.ndarray] = []
        perm_ctrl_runs: list[np.ndarray] = []
        for trial in range(max(1, int(n_control_trials))):
            rng = rng_for_stream(seed, f"{model_name}:{spec.name}:perm:{trial}")
            k_perm = _permute_kernels(kernels, high_heads, rng)
            with subtract_positional_kernels(model, k_perm, high_heads, int(max(seq_len, spec.context_len))):
                pp = _eval_target_metrics(
                    model=model,
                    prompts=probe_prompts,
                    device=device,
                    pad_token_id=pad_id,
                    batch_size=batch_size,
                )
                pc = _eval_target_metrics(
                    model=model,
                    prompts=ctrl_prompts,
                    device=device,
                    pad_token_id=pad_id,
                    batch_size=batch_size,
                )
            perm_probe_runs.append(pp["lp"])
            perm_ctrl_runs.append(pc["lp"])

        perm_probe_lp = np.mean(np.vstack(perm_probe_runs), axis=0)
        perm_ctrl_lp = np.mean(np.vstack(perm_ctrl_runs), axis=0)

        loss_true_probe = intact_probe["lp"] - true_probe["lp"]
        loss_true_ctrl = intact_ctrl["lp"] - true_ctrl["lp"]
        gap_true = loss_true_probe - loss_true_ctrl

        loss_perm_probe = intact_probe["lp"] - perm_probe_lp
        loss_perm_ctrl = intact_ctrl["lp"] - perm_ctrl_lp
        gap_perm = loss_perm_probe - loss_perm_ctrl

        fam_ci = _bootstrap_mean_ci(gap_true, n_boot=2500, seed=seed + 4000 + fidx)
        fam_ci_delta = _bootstrap_mean_ci(gap_true - gap_perm, n_boot=2500, seed=seed + 5000 + fidx)

        fam_row = {
            "family": spec.name,
            "context_band": spec.context_band,
            "regularity_band": spec.regularity_band,
            "context_len": int(spec.context_len),
            "n_prompts": int(len(gap_true)),
            "mean_gap_true_lp": float(np.mean(gap_true)),
            "mean_gap_perm_lp": float(np.mean(gap_perm)),
            "mean_gap_true_minus_perm_lp": float(np.mean(gap_true - gap_perm)),
            "gap_true_ci": [float(fam_ci["ci_lo"]), float(fam_ci["ci_hi"])],
            "gap_true_minus_perm_ci": [float(fam_ci_delta["ci_lo"]), float(fam_ci_delta["ci_hi"])],
            "intact_probe_acc": float(np.mean(intact_probe["acc"])),
            "intact_ctrl_acc": float(np.mean(intact_ctrl["acc"])),
            "true_probe_acc": float(np.mean(true_probe["acc"])),
            "true_ctrl_acc": float(np.mean(true_ctrl["acc"])),
            "family_directional_pass": bool(np.mean(gap_true) > 0.0 and np.mean(gap_true - gap_perm) > 0.0),
        }
        family_rows.append(fam_row)

        for i in range(len(gap_true)):
            trial_rows.append(
                {
                    "family": spec.name,
                    "context_band": spec.context_band,
                    "regularity_band": spec.regularity_band,
                    "prompt_idx": int(i),
                    "gap_true_lp": float(gap_true[i]),
                    "gap_perm_lp": float(gap_perm[i]),
                    "gap_true_minus_perm_lp": float(gap_true[i] - gap_perm[i]),
                }
            )

        print(
            f"[E30A][{model_name}] {spec.name}: "
            f"gap_true={fam_row['mean_gap_true_lp']:.6f} "
            f"gap_true_minus_perm={fam_row['mean_gap_true_minus_perm_lp']:.6f}",
            flush=True,
        )

    fam_df = pd.DataFrame(family_rows)
    trials_df = pd.DataFrame(trial_rows)
    fam_df.to_parquet(model_dir / "family_summary.parquet", index=False)
    trials_df.to_parquet(model_dir / "prompt_level_gaps.parquet", index=False)

    pooled_true = trials_df["gap_true_lp"].to_numpy(dtype=np.float64)
    pooled_true_ci = _bootstrap_mean_ci(pooled_true, n_boot=4000, seed=seed + 9001)

    # Boundary effects.
    high = fam_df.loc[fam_df["regularity_band"] == "high", "mean_gap_true_lp"].to_numpy(dtype=np.float64)
    low = fam_df.loc[fam_df["regularity_band"] == "low", "mean_gap_true_lp"].to_numpy(dtype=np.float64)
    longv = fam_df.loc[fam_df["context_band"] == "long", "mean_gap_true_lp"].to_numpy(dtype=np.float64)
    shortv = fam_df.loc[fam_df["context_band"] == "short", "mean_gap_true_lp"].to_numpy(dtype=np.float64)

    reg_effect = float(np.mean(high) - np.mean(low)) if high.size and low.size else float("nan")
    len_effect = float(np.mean(longv) - np.mean(shortv)) if longv.size and shortv.size else float("nan")

    high_pass = bool(np.all(high > 0.0)) if high.size else False
    model_pass = bool(float(pooled_true_ci["ci_lo"]) > 0.0 and high_pass)

    rec = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "model": model_name,
        "n_high_si_heads": int(len(high_heads)),
        "n_families": int(len(family_rows)),
        "n_total_prompts": int(len(trials_df)),
        "mean_pooled_gap_true_lp": float(pooled_true_ci["mean"]),
        "pooled_gap_true_ci": [float(pooled_true_ci["ci_lo"]), float(pooled_true_ci["ci_hi"])],
        "p_one_gt_zero": float(pooled_true_ci["p_one_gt_zero"]),
        "regularity_effect_high_minus_low": reg_effect,
        "length_effect_long_minus_short": len_effect,
        "all_high_reg_positive": bool(high_pass),
        "model_directional_pass": bool(model_pass),
        "n_family_directional_pass": int(sum(1 for r in family_rows if bool(r.get("family_directional_pass", False)))),
        "family_results": family_rows,
        "elapsed_sec": float(time.time() - t0),
    }
    write_json(model_dir / "summary.json", rec)
    return rec


def _finalize(models: list[str], out_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for m in models:
        p = out_root / m / "summary.json"
        if not p.exists():
            raise RuntimeError(f"[E30A] hard_fail_reason: missing summary for {m}: {p}")
        rows.append(json.loads(p.read_text(encoding="utf-8")))

    enforce_coverage_contract(
        experiment_id=EXPERIMENT_ID,
        observed_models=[r.get("model", "") for r in rows],
        required_models=models,
    )

    n_pass = int(sum(1 for r in rows if bool(r.get("model_directional_pass", False))))
    if n_pass == len(rows):
        interp = "boundary_grid_supported"
        status = "supported"
    elif n_pass >= 2:
        interp = "boundary_grid_supported_with_caveat"
        status = "supported_with_caveat"
    elif n_pass >= 1:
        interp = "boundary_grid_mixed"
        status = "mixed"
    else:
        interp = "boundary_grid_not_supported"
        status = "not_supported"

    cross = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "interpretation": interp,
        "claim_status": status,
        "n_models": int(len(rows)),
        "n_model_pass": int(n_pass),
        "per_model": rows,
    }
    write_json(out_root / "cross_model_summary.json", cross)

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "What boundary (context length vs offset regularity) governs SI preferential disruption transfer?",
        "primary_hypothesis": "High-regularity families preserve positive preferential SI gaps more consistently than low-regularity families.",
        "primary_endpoints": [
            "regularity_effect_high_minus_low",
            "length_effect_long_minus_short",
            "model_directional_pass",
        ],
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "analysis_tier": "hardening",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": interp,
            "n_model_pass": int(n_pass),
            "n_models": int(len(rows)),
        },
        "limitations": [
            "Boundary grid uses semi-naturalistic synthetic constructions, not benchmark tasks.",
            "Intervention is evaluation-time SI subtraction, not a training counterfactual.",
        ],
    }

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "claim_status": status,
        "impacts": [
            "Refines the transfer boundary for Result III by separating context-length and regularity effects.",
            "Can resolve or narrow the E28C vs E29B scope inconsistency in main-text wording.",
        ],
    }

    data_dictionary = {
        "family_summary.parquet": {
            "description": "Per-family preferential-gap metrics across boundary-grid conditions.",
            "columns": {
                "family": "Boundary-grid family id",
                "context_band": "short/long",
                "regularity_band": "high/low",
                "mean_gap_true_lp": "Probe-control LP loss gap under true SI subtraction",
                "mean_gap_true_minus_perm_lp": "True minus permuted preferential gap",
            },
        },
        "prompt_level_gaps.parquet": {
            "description": "Prompt-level LP preferential gaps for pooled inference.",
        },
    }

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": timestamp_now(),
        "models": models,
        "output_root": str(out_root),
        "files": [
            "cross_model_summary.json",
            "summary.json",
            "manifest.json",
            "preregistration.json",
            "claim_impact.json",
            "data_dictionary.json",
        ],
    }

    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=out_root,
        preregistration=prereg,
        manifest=manifest,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross


def main() -> None:
    p = argparse.ArgumentParser(description="E30A: probe boundary 2x2 grid", allow_abbrev=False)
    p.add_argument("--models", default="all")
    p.add_argument("--device-map", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--n-prompts", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-control-trials", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260504)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E30A] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    device_map = parse_device_map(args.device_map)
    out_root = ensure_dir(Path(args.output_root))

    n_prompts = int(args.n_prompts)
    batch_size = int(args.batch_size)
    n_control_trials = int(args.n_control_trials)
    if args.smoke:
        n_prompts = min(n_prompts, 16)
        batch_size = min(batch_size, 4)
        n_control_trials = 1

    if args.finalize_only:
        cross = _finalize(models=models, out_root=out_root)
        print(f"[E30A] Finalized from shard outputs. interpretation={cross['interpretation']}", flush=True)
        return

    for model in models:
        device = device_map.get(model, args.device)
        run_model(
            model_name=model,
            device=device,
            out_root=out_root,
            n_prompts=n_prompts,
            batch_size=batch_size,
            seq_len=int(args.seq_len),
            n_control_trials=n_control_trials,
            seed=int(args.seed),
        )

    if args.no_finalize:
        print("[E30A] Shard run complete (no finalize).", flush=True)
        return

    cross = _finalize(models=models, out_root=out_root)
    print(
        f"[E30A] Done. interpretation={cross['interpretation']} "
        f"n_model_pass={cross['n_model_pass']}/{cross['n_models']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
