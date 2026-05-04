#!/usr/bin/env python3
"""E22 — Task-Conditional SI Specificity.

Extends E17 by evaluating SI-kernel specificity beyond baseline LM on a
multi-task battery:
1) LM domains (wiki/code/dialogue): per-sequence NLL degradation
2) ICL prompts: target-token log-probability degradation
3) Induction-style copy prompts: target-token log-probability degradation

Intervention arms per task:
- true SI subtraction
- offset-permuted control
- norm-matched random control
- distortion-matched random control (calibrated to match attention KL)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
    compute_per_token_loss,
    load_head_groups,
    subtract_positional_kernels,
)
from experiment5.pipeline import (  # noqa: E402
    _load_code_sequences,
    _load_dialogue_sequences,
    _load_wiki_sequences,
)

OUT_ROOT = RESULTS_ROOT / "E22_task_conditional_specificity"
LM_DOMAINS = ("wiki", "code", "dialogue")
SEED_BASE = 20260502
CONTROL_SEED_OFFSETS = {
    "permuted": 101,
    "normmatched": 202,
    "distortion_matched": 303,
}


def _set_eager_attention(model: Any) -> None:
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            try:
                layer.self_attn.config._attn_implementation = "eager"
            except Exception:
                continue


def _load_kernels(model_name: str) -> tuple[dict[tuple[int, int], np.ndarray], str]:
    candidates = [
        ROOT
        / "results"
        / "reinforce_exp"
        / "exp_r3_core_replication"
        / model_name
        / "theory8_position_ablation"
        / model_name
        / "estimated_kernels.json",
        ROOT
        / "results"
        / "experiment3"
        / "theory8_position_ablation"
        / model_name
        / "estimated_kernels.json",
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
        f"[E22] No estimated_kernels.json found for {model_name}; checked: {candidates}"
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


def _norm_matched_random_kernels(
    kernels: dict[tuple[int, int], np.ndarray],
    heads: list[tuple[int, int]],
    rng: np.random.Generator,
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    for head in heads:
        g = kernels.get(head)
        if g is None:
            continue
        n = int(len(g))
        if n <= 0:
            continue
        z = rng.standard_normal(n).astype(np.float32)
        z -= float(np.mean(z))
        zn = float(np.linalg.norm(z))
        gn = float(np.linalg.norm(g))
        if zn <= 1e-12 or gn <= 1e-12:
            out[head] = np.zeros_like(g, dtype=np.float32)
        else:
            out[head] = (z * (gn / zn)).astype(np.float32)
    return out


def _scale_kernels(
    kernels: dict[tuple[int, int], np.ndarray],
    scale: float,
) -> dict[tuple[int, int], np.ndarray]:
    s = float(scale)
    out: dict[tuple[int, int], np.ndarray] = {}
    for k, v in kernels.items():
        out[k] = (np.asarray(v, dtype=np.float32) * s).astype(np.float32)
    return out


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
    raw: dict[str, list[list[int]]] = {}
    raw_counts: dict[str, int] = {}
    for didx, domain in enumerate(LM_DOMAINS):
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
    if n_balanced < 4:
        raise RuntimeError(
            f"[E22] insufficient matched LM sequences: counts={raw_counts}, min_required=4"
        )
    balanced = {domain: raw[domain][:n_balanced] for domain in LM_DOMAINS}
    return balanced, raw_counts, int(n_balanced)


@torch.no_grad()
def _eval_sequence_mean_losses(
    model: Any,
    sequences: list[list[int]],
    device: str,
    batch_size: int,
) -> np.ndarray:
    vals: list[float] = []
    bs = max(1, int(batch_size))
    pos = 0
    while pos < len(sequences):
        batch = sequences[pos : pos + bs]
        try:
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            with torch.inference_mode():
                loss = compute_per_token_loss(model, input_ids)
            tok_per_seq = int(input_ids.shape[1] - 1)
            seq_loss = (
                loss.view(input_ids.shape[0], tok_per_seq)
                .mean(dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            vals.extend(float(x) for x in seq_loss.tolist())
            pos += len(batch)
            del input_ids, loss
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                continue
            raise
    return np.asarray(vals, dtype=np.float64)


def _build_icl_prompts(tokenizer: Any, n: int, n_demos: int, rng: random.Random) -> list[dict[str, Any]]:
    vocab = list(tokenizer.get_vocab().values())
    out: list[dict[str, Any]] = []
    for _ in range(n):
        key = rng.choice(vocab)
        val = rng.choice(vocab)
        demo: list[int] = []
        for _ in range(n_demos):
            demo.extend([key, val])
        out.append({"input_ids": demo + [key], "target_token_id": val, "task": "icl"})
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
        out.append({"input_ids": demo + [query], "target_token_id": target, "task": "nonicl"})
    return out


def _build_induction_prompts(tokenizer: Any, n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Needle/copy-style prompts: distractors + (key,value) + query key -> target value."""
    vocab = list(tokenizer.get_vocab().values())
    out: list[dict[str, Any]] = []
    for _ in range(n):
        n_distract = rng.randint(20, 60)
        key = rng.choice(vocab)
        val = rng.choice(vocab)
        distract = [rng.choice(vocab) for _ in range(n_distract)]
        ins = rng.randint(0, max(0, n_distract - 1))
        prefix = distract[:ins]
        suffix = distract[ins:]
        ids = prefix + [key, val] + suffix + [key]
        out.append({"input_ids": ids, "target_token_id": val, "task": "induction"})
    return out


@torch.no_grad()
def _eval_prompts_target_metrics(
    model: Any,
    prompts: list[dict[str, Any]],
    device: str,
) -> dict[str, np.ndarray]:
    acc: list[float] = []
    lp: list[float] = []
    for p in prompts:
        ids = torch.tensor([p["input_ids"]], device=device, dtype=torch.long)
        out = model(ids)
        logits = out.logits[0, -1]
        target = int(p["target_token_id"])
        pred = int(torch.argmax(logits).item())
        lpt = float(torch.log_softmax(logits, dim=-1)[target].item())
        acc.append(float(pred == target))
        lp.append(lpt)
    return {
        "acc": np.asarray(acc, dtype=np.float64),
        "lp": np.asarray(lp, dtype=np.float64),
    }


def _bootstrap_arm_diff_ci(
    true_delta: np.ndarray,
    ctrl_stack: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    td = np.asarray(true_delta, dtype=np.float64)
    cs = np.asarray(ctrl_stack, dtype=np.float64)
    if td.ndim != 1 or cs.ndim != 2:
        return {
            "mean": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_one_gt_zero": float("nan"),
        }
    n_seq = int(td.shape[0])
    n_trials = int(cs.shape[0])
    if n_seq == 0 or n_trials == 0 or cs.shape[1] != n_seq:
        return {
            "mean": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_one_gt_zero": float("nan"),
        }

    obs = float(np.mean(td) - np.mean(cs))
    rng = np.random.default_rng(int(seed))
    m = max(2000, int(n_boot))
    boot = np.empty(m, dtype=np.float64)
    for i in range(m):
        seq_idx = rng.integers(0, n_seq, size=n_seq)
        trial_idx = rng.integers(0, n_trials, size=n_trials)
        true_mean = float(np.mean(td[seq_idx]))
        ctrl_mean = float(np.mean(cs[trial_idx][:, seq_idx]))
        boot[i] = true_mean - ctrl_mean
    return {
        "mean": obs,
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "p_one_gt_zero": float((np.sum(boot <= 0.0) + 1) / (m + 1)),
    }


def _capture_attentions_for_input(model: Any, token_ids: list[int], device: str) -> list[np.ndarray]:
    ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        out = model(
            ids,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
        )
    if out.attentions is None:
        raise RuntimeError("[E22] output_attentions returned None during KL calibration")
    layers = [
        a[0].float().detach().cpu().numpy().astype(np.float64)
        for a in out.attentions
    ]
    return layers


def _head_map(heads: list[tuple[int, int]]) -> dict[int, list[int]]:
    hm: dict[int, list[int]] = {}
    for l, h in heads:
        hm.setdefault(int(l), []).append(int(h))
    for l in hm:
        hm[l] = sorted(set(hm[l]))
    return hm


def _mean_target_head_kl(
    baseline_attn: list[list[np.ndarray]],
    altered_attn: list[list[np.ndarray]],
    hm: dict[int, list[int]],
) -> float:
    vals: list[float] = []
    eps = 1e-9
    n = min(len(baseline_attn), len(altered_attn))
    for i in range(n):
        b_layers = baseline_attn[i]
        a_layers = altered_attn[i]
        n_layers = min(len(b_layers), len(a_layers))
        for l in range(n_layers):
            heads = hm.get(l)
            if not heads:
                continue
            b = b_layers[l]
            a = a_layers[l]
            hmax = min(b.shape[0], a.shape[0])
            use = [h for h in heads if 0 <= h < hmax]
            if not use:
                continue
            p = np.clip(b[use, :, :], eps, 1.0)
            q = np.clip(a[use, :, :], eps, 1.0)
            kl = np.sum(p * (np.log(p) - np.log(q)), axis=-1)
            vals.append(float(np.mean(kl)))
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _collect_altered_attn(
    *,
    model: Any,
    device: str,
    calib_inputs: list[list[int]],
    kernels: dict[tuple[int, int], np.ndarray],
    high_heads: list[tuple[int, int]],
    seq_len: int,
) -> list[list[np.ndarray]]:
    out: list[list[np.ndarray]] = []
    with subtract_positional_kernels(model, kernels, high_heads, int(seq_len)):
        for ids in calib_inputs:
            out.append(_capture_attentions_for_input(model, ids, device))
    return out


def _match_distortion_scaled_random(
    *,
    model: Any,
    device: str,
    calib_inputs: list[list[int]],
    baseline_attn: list[list[np.ndarray]],
    target_kl: float,
    source_kernels: dict[tuple[int, int], np.ndarray],
    high_heads: list[tuple[int, int]],
    rng: np.random.Generator,
    seq_len: int,
    alpha_grid: list[float],
    tol_rel: float,
    max_retry: int,
    bisect_steps: int,
    alpha_min: float,
    alpha_max: float,
) -> dict[str, Any]:
    hm = _head_map(high_heads)
    best_payload: dict[str, Any] | None = None

    if not np.isfinite(target_kl) or float(target_kl) <= 0.0:
        return {
            "attempt": 0,
            "alpha": 0.0,
            "kl": 0.0,
            "target_kl": float(target_kl),
            "rel_err": float("inf"),
            "matched": False,
            "kernels": {},
        }

    def _record_payload(
        *,
        attempt: int,
        alpha: float,
        kl: float,
        kernels_scaled: dict[tuple[int, int], np.ndarray],
    ) -> dict[str, Any]:
        rel_err = (
            abs(float(kl) - float(target_kl)) / max(abs(float(target_kl)), 1e-8)
            if np.isfinite(kl)
            else float("inf")
        )
        return {
            "attempt": int(attempt),
            "alpha": float(alpha),
            "kl": float(kl),
            "target_kl": float(target_kl),
            "rel_err": float(rel_err),
            "matched": bool(np.isfinite(rel_err) and rel_err <= tol_rel),
            "kernels": kernels_scaled,
        }

    def _eval_alpha(
        *,
        attempt: int,
        alpha: float,
        base_rand: dict[tuple[int, int], np.ndarray],
    ) -> tuple[dict[str, Any], float]:
        scaled = _scale_kernels(base_rand, float(alpha))
        altered = _collect_altered_attn(
            model=model,
            device=device,
            calib_inputs=calib_inputs,
            kernels=scaled,
            high_heads=high_heads,
            seq_len=seq_len,
        )
        kl = _mean_target_head_kl(baseline_attn, altered, hm)
        payload = _record_payload(
            attempt=attempt,
            alpha=float(alpha),
            kl=float(kl),
            kernels_scaled=scaled,
        )
        return payload, float(kl)

    def _consider(payload: dict[str, Any]) -> bool:
        nonlocal best_payload
        if best_payload is None or float(payload["rel_err"]) < float(best_payload["rel_err"]):
            best_payload = payload
        return bool(payload["matched"])

    def _find_bracket(
        vals: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        ordered = sorted(vals, key=lambda x: float(x[0]))
        for i in range(len(ordered) - 1):
            a0, k0 = ordered[i]
            a1, k1 = ordered[i + 1]
            if not (np.isfinite(k0) and np.isfinite(k1)):
                continue
            d0 = float(k0) - float(target_kl)
            d1 = float(k1) - float(target_kl)
            if d0 == 0.0:
                return (a0, k0), (a0, k0)
            if d1 == 0.0:
                return (a1, k1), (a1, k1)
            if d0 * d1 < 0.0:
                return (a0, k0), (a1, k1)
        return None

    alpha_seed = [float(a) for a in alpha_grid if np.isfinite(a) and float(a) > 0.0]
    alpha_seed.extend([0.0, float(alpha_min), float(alpha_max)])
    alpha_seed = sorted(set(max(0.0, float(a)) for a in alpha_seed))

    for attempt in range(max_retry):
        base_rand = _norm_matched_random_kernels(source_kernels, high_heads, rng)
        evaluated: list[tuple[float, float]] = []
        seen: set[float] = set()
        for alpha in alpha_seed:
            a = max(float(alpha_min), min(float(alpha_max), float(alpha)))
            if a in seen:
                continue
            seen.add(a)
            payload, kl = _eval_alpha(attempt=attempt, alpha=a, base_rand=base_rand)
            evaluated.append((a, float(kl)))
            if _consider(payload):
                return payload

        bracket = _find_bracket(evaluated)
        if bracket is None:
            continue

        (alo, klo), (ahi, khi) = bracket
        if float(alo) == float(ahi):
            continue
        lo = float(min(alo, ahi))
        hi = float(max(alo, ahi))
        for _ in range(max(0, int(bisect_steps))):
            mid = 0.5 * (lo + hi)
            payload, kmid = _eval_alpha(attempt=attempt, alpha=mid, base_rand=base_rand)
            if _consider(payload):
                return payload
            dlo = float(klo) - float(target_kl)
            dmid = float(kmid) - float(target_kl)
            if np.isfinite(dlo) and np.isfinite(dmid) and dlo * dmid <= 0.0:
                hi = float(mid)
                ahi, khi = float(mid), float(kmid)
            else:
                lo = float(mid)
                alo, klo = float(mid), float(kmid)

    if best_payload is None:
        return {
            "attempt": int(max_retry),
            "alpha": float("nan"),
            "kl": float("nan"),
            "target_kl": float(target_kl),
            "rel_err": float("inf"),
            "matched": False,
            "kernels": {},
        }
    best_payload["matched"] = False
    return best_payload


def _lm_task_eval(
    *,
    model: Any,
    device: str,
    sequences: list[list[int]],
    batch_size: int,
    kernels: dict[tuple[int, int], np.ndarray],
    high_heads: list[tuple[int, int]],
    seq_len: int,
) -> np.ndarray:
    with subtract_positional_kernels(model, kernels, high_heads, int(seq_len)):
        return _eval_sequence_mean_losses(
            model=model,
            sequences=sequences,
            device=device,
            batch_size=batch_size,
        )


def _prompt_task_eval(
    *,
    model: Any,
    device: str,
    prompts: list[dict[str, Any]],
    kernels: dict[tuple[int, int], np.ndarray],
    high_heads: list[tuple[int, int]],
    seq_len: int,
) -> dict[str, np.ndarray]:
    with subtract_positional_kernels(model, kernels, high_heads, int(seq_len)):
        return _eval_prompts_target_metrics(model, prompts, device)


def _max_input_len(items: list[list[int]]) -> int:
    if not items:
        return 64
    return max(64, max(len(x) for x in items))


def run_model(
    *,
    model_name: str,
    device: str,
    out_root: Path,
    seq_len: int,
    lm_num_sequences: int,
    icl_eval: int,
    icl_demos: int,
    induction_eval: int,
    n_control_trials: int,
    seed: int,
    batch_size: int,
    n_boot: int,
    calibration_count: int,
    calibration_seq_len: int,
    distortion_tol_rel: float,
    distortion_max_retry: int,
    alpha_grid: list[float],
    distortion_bisect_steps: int,
    distortion_alpha_min: float,
    distortion_alpha_max: float,
    distortion_min_valid_trials: int,
    distortion_max_extra_trials: int,
) -> dict[str, Any]:
    t0 = time.time()
    model_dir = ensure_dir(out_root / model_name)

    print(f"[E22] Loading {model_name} on {device}", flush=True)
    model, tokenizer = load_model_for_exp(model_name, device, attn_implementation="eager")
    _set_eager_attention(model)

    head_groups = load_head_groups(model_name)
    high_heads = [(int(l), int(h)) for (l, h) in head_groups["high_si"]]
    kernels, kernel_path = _load_kernels(model_name)

    lm_seqs, lm_raw_counts, lm_balanced_n = _load_balanced_domain_sequences(
        model_name=model_name,
        tokenizer=tokenizer,
        seq_len=max(64, int(seq_len)),
        max_sequences=max(8, int(lm_num_sequences)),
        seed=int(seed),
    )

    rng = random.Random(int(seed) + 101)
    icl_prompts = _build_icl_prompts(tokenizer, max(16, int(icl_eval)), max(2, int(icl_demos)), rng)
    nonicl_prompts = _build_nonicl_prompts(tokenizer, max(16, int(icl_eval)), max(2, int(icl_demos)), random.Random(int(seed) + 102))
    induction_prompts = _build_induction_prompts(tokenizer, max(16, int(induction_eval)), random.Random(int(seed) + 103))

    task_records: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    calib_rows: list[dict[str, Any]] = []

    summary_tasks: dict[str, Any] = {}

    def _process_task_lm(task_name: str, sequences: list[list[int]], rng_seed_base: int) -> None:
        baseline = _eval_sequence_mean_losses(
            model=model,
            sequences=sequences,
            device=device,
            batch_size=batch_size,
        )
        true_loss = _lm_task_eval(
            model=model,
            device=device,
            sequences=sequences,
            batch_size=batch_size,
            kernels=kernels,
            high_heads=high_heads,
            seq_len=max(64, int(seq_len)),
        )
        true_delta = true_loss - baseline

        calib_inputs = [seq[: max(32, int(calibration_seq_len))] for seq in sequences[: max(2, int(calibration_count))]]
        base_attn = [_capture_attentions_for_input(model, s, device) for s in calib_inputs]
        true_attn = _collect_altered_attn(
            model=model,
            device=device,
            calib_inputs=calib_inputs,
            kernels=kernels,
            high_heads=high_heads,
            seq_len=max(64, int(calibration_seq_len)),
        )
        target_kl = _mean_target_head_kl(base_attn, true_attn, _head_map(high_heads))

        ctrl_stacks: dict[str, list[np.ndarray]] = {"permuted": [], "normmatched": [], "distortion_matched": []}

        base_trials = max(1, int(n_control_trials))
        for t in range(base_trials):
            perm_seed = int(rng_seed_base) + t * 7919 + 11
            norm_seed = int(rng_seed_base) + t * 7919 + 29
            dist_seed = int(rng_seed_base) + t * 7919 + 47

            perm_k = _permute_kernels(kernels, high_heads, np.random.default_rng(perm_seed))
            perm_loss = _lm_task_eval(
                model=model,
                device=device,
                sequences=sequences,
                batch_size=batch_size,
                kernels=perm_k,
                high_heads=high_heads,
                seq_len=max(64, int(seq_len)),
            )
            perm_delta = perm_loss - baseline
            ctrl_stacks["permuted"].append(perm_delta)

            norm_k = _norm_matched_random_kernels(kernels, high_heads, np.random.default_rng(norm_seed))
            norm_loss = _lm_task_eval(
                model=model,
                device=device,
                sequences=sequences,
                batch_size=batch_size,
                kernels=norm_k,
                high_heads=high_heads,
                seq_len=max(64, int(seq_len)),
            )
            norm_delta = norm_loss - baseline
            ctrl_stacks["normmatched"].append(norm_delta)

            dist_match = _match_distortion_scaled_random(
                model=model,
                device=device,
                calib_inputs=calib_inputs,
                baseline_attn=base_attn,
                target_kl=target_kl,
                source_kernels=kernels,
                high_heads=high_heads,
                rng=np.random.default_rng(dist_seed),
                seq_len=max(64, int(calibration_seq_len)),
                alpha_grid=alpha_grid,
                tol_rel=float(distortion_tol_rel),
                max_retry=max(1, int(distortion_max_retry)),
                bisect_steps=max(0, int(distortion_bisect_steps)),
                alpha_min=max(1e-4, float(distortion_alpha_min)),
                alpha_max=max(float(distortion_alpha_max), float(distortion_alpha_min) + 1e-4),
            )
            matched = bool(dist_match.get("matched", False))
            dist_delta = None
            if matched:
                dist_k = dist_match["kernels"]
                dist_loss = _lm_task_eval(
                    model=model,
                    device=device,
                    sequences=sequences,
                    batch_size=batch_size,
                    kernels=dist_k,
                    high_heads=high_heads,
                    seq_len=max(64, int(seq_len)),
                )
                dist_delta = dist_loss - baseline
                ctrl_stacks["distortion_matched"].append(dist_delta)

            calib_rows.append(
                {
                    "task": task_name,
                    "trial": int(t),
                    "target_kl": float(target_kl),
                    "matched": bool(matched),
                    "alpha": float(dist_match.get("alpha", float("nan"))),
                    "matched_kl": float(dist_match.get("kl", float("nan"))),
                    "rel_err": float(dist_match.get("rel_err", float("nan"))),
                }
            )

            task_records.extend(
                [
                    {
                        "task": task_name,
                        "metric": "nll",
                        "arm": "true",
                        "trial": int(t),
                        "mean_damage": float(np.mean(true_delta)),
                        "std_damage": float(np.std(true_delta, ddof=1)) if len(true_delta) > 1 else float("nan"),
                        "n_units": int(len(true_delta)),
                        "valid": True,
                    },
                    {
                        "task": task_name,
                        "metric": "nll",
                        "arm": "permuted",
                        "trial": int(t),
                        "mean_damage": float(np.mean(perm_delta)),
                        "std_damage": float(np.std(perm_delta, ddof=1)) if len(perm_delta) > 1 else float("nan"),
                        "n_units": int(len(perm_delta)),
                        "valid": True,
                    },
                    {
                        "task": task_name,
                        "metric": "nll",
                        "arm": "normmatched",
                        "trial": int(t),
                        "mean_damage": float(np.mean(norm_delta)),
                        "std_damage": float(np.std(norm_delta, ddof=1)) if len(norm_delta) > 1 else float("nan"),
                        "n_units": int(len(norm_delta)),
                        "valid": True,
                    },
                    {
                        "task": task_name,
                        "metric": "nll",
                        "arm": "distortion_matched",
                        "trial": int(t),
                        "mean_damage": float(np.mean(dist_delta)) if dist_delta is not None else float("nan"),
                        "std_damage": float(np.std(dist_delta, ddof=1)) if dist_delta is not None and len(dist_delta) > 1 else float("nan"),
                        "n_units": int(len(dist_delta)) if dist_delta is not None else 0,
                        "valid": bool(matched),
                    },
                ]
            )

            for i in range(len(true_delta)):
                delta_rows.append(
                    {
                        "task": task_name,
                        "metric": "nll",
                        "unit_index": int(i),
                        "trial": int(t),
                        "true_delta": float(true_delta[i]),
                        "permuted_delta": float(perm_delta[i]),
                        "normmatched_delta": float(norm_delta[i]),
                        "distortion_delta": float(dist_delta[i]) if dist_delta is not None else float("nan"),
                        "distortion_valid": bool(matched),
                    }
                )

        min_valid = max(1, int(distortion_min_valid_trials))
        extra_budget = max(0, int(distortion_max_extra_trials))
        trial_cursor = int(base_trials)
        while len(ctrl_stacks["distortion_matched"]) < min_valid and trial_cursor < base_trials + extra_budget:
            dist_seed = int(rng_seed_base) + trial_cursor * 7919 + 47
            dist_match = _match_distortion_scaled_random(
                model=model,
                device=device,
                calib_inputs=calib_inputs,
                baseline_attn=base_attn,
                target_kl=target_kl,
                source_kernels=kernels,
                high_heads=high_heads,
                rng=np.random.default_rng(dist_seed),
                seq_len=max(64, int(calibration_seq_len)),
                alpha_grid=alpha_grid,
                tol_rel=float(distortion_tol_rel),
                max_retry=max(1, int(distortion_max_retry)),
                bisect_steps=max(0, int(distortion_bisect_steps)),
                alpha_min=max(1e-4, float(distortion_alpha_min)),
                alpha_max=max(float(distortion_alpha_max), float(distortion_alpha_min) + 1e-4),
            )
            matched = bool(dist_match.get("matched", False))
            dist_delta = None
            if matched:
                dist_k = dist_match["kernels"]
                dist_loss = _lm_task_eval(
                    model=model,
                    device=device,
                    sequences=sequences,
                    batch_size=batch_size,
                    kernels=dist_k,
                    high_heads=high_heads,
                    seq_len=max(64, int(seq_len)),
                )
                dist_delta = dist_loss - baseline
                ctrl_stacks["distortion_matched"].append(dist_delta)

            calib_rows.append(
                {
                    "task": task_name,
                    "trial": int(trial_cursor),
                    "target_kl": float(target_kl),
                    "matched": bool(matched),
                    "alpha": float(dist_match.get("alpha", float("nan"))),
                    "matched_kl": float(dist_match.get("kl", float("nan"))),
                    "rel_err": float(dist_match.get("rel_err", float("nan"))),
                    "extra_trial": True,
                }
            )
            task_records.append(
                {
                    "task": task_name,
                    "metric": "nll",
                    "arm": "distortion_matched",
                    "trial": int(trial_cursor),
                    "mean_damage": float(np.mean(dist_delta)) if dist_delta is not None else float("nan"),
                    "std_damage": float(np.std(dist_delta, ddof=1)) if dist_delta is not None and len(dist_delta) > 1 else float("nan"),
                    "n_units": int(len(dist_delta)) if dist_delta is not None else 0,
                    "valid": bool(matched),
                }
            )
            if dist_delta is not None:
                for i in range(len(dist_delta)):
                    delta_rows.append(
                        {
                            "task": task_name,
                            "metric": "nll",
                            "unit_index": int(i),
                            "trial": int(trial_cursor),
                            "true_delta": float(true_delta[i]),
                            "permuted_delta": float("nan"),
                            "normmatched_delta": float("nan"),
                            "distortion_delta": float(dist_delta[i]),
                            "distortion_valid": True,
                            "extra_trial": True,
                        }
                    )
            trial_cursor += 1

        cis: dict[str, dict[str, float]] = {}
        pass_flags: dict[str, bool] = {}
        for cname in ["permuted", "normmatched", "distortion_matched"]:
            stack = np.stack(ctrl_stacks[cname], axis=0) if ctrl_stacks[cname] else np.empty((0, len(true_delta)))
            ci = _bootstrap_arm_diff_ci(
                true_delta,
                stack,
                n_boot=max(1000, int(n_boot)),
                seed=int(rng_seed_base) + 901 + CONTROL_SEED_OFFSETS[cname],
            )
            cis[cname] = ci
            pass_flags[cname] = bool(np.isfinite(ci["ci_lo"]) and ci["ci_lo"] > 0.0)

        task_pass = bool(all(pass_flags.values()))
        summary_tasks[task_name] = {
            "task": task_name,
            "metric": "nll",
            "n_units": int(len(true_delta)),
            "true_mean_damage": float(np.mean(true_delta)),
            "control_cis": cis,
            "control_pass": pass_flags,
            "task_pass": task_pass,
            "distortion_valid_trials": int(sum(1 for r in calib_rows if r["task"] == task_name and r["matched"])),
            "distortion_total_trials": int(sum(1 for r in calib_rows if r["task"] == task_name),
            ),
        }

    def _process_task_prompt(
        *,
        task_name: str,
        prompts: list[dict[str, Any]],
        metric_key: str,
        rng_seed_base: int,
        include_secondary_acc: bool,
    ) -> None:
        baseline = _eval_prompts_target_metrics(model, prompts, device)
        maxlen = _max_input_len([p["input_ids"] for p in prompts])

        true_eval = _prompt_task_eval(
            model=model,
            device=device,
            prompts=prompts,
            kernels=kernels,
            high_heads=high_heads,
            seq_len=maxlen,
        )

        # degradation: higher means worse under intervention
        if metric_key == "lp":
            true_delta = baseline["lp"] - true_eval["lp"]
        else:
            true_delta = baseline["acc"] - true_eval["acc"]

        calib_prompts = prompts[: max(2, int(calibration_count))]
        calib_inputs = [p["input_ids"][: max(8, int(calibration_seq_len))] for p in calib_prompts]

        base_attn = [_capture_attentions_for_input(model, ids, device) for ids in calib_inputs]
        true_attn = _collect_altered_attn(
            model=model,
            device=device,
            calib_inputs=calib_inputs,
            kernels=kernels,
            high_heads=high_heads,
            seq_len=max(64, int(calibration_seq_len)),
        )
        target_kl = _mean_target_head_kl(base_attn, true_attn, _head_map(high_heads))

        ctrl_stacks: dict[str, list[np.ndarray]] = {"permuted": [], "normmatched": [], "distortion_matched": []}

        base_trials = max(1, int(n_control_trials))
        for t in range(base_trials):
            perm_seed = int(rng_seed_base) + t * 7919 + 11
            norm_seed = int(rng_seed_base) + t * 7919 + 29
            dist_seed = int(rng_seed_base) + t * 7919 + 47

            perm_k = _permute_kernels(kernels, high_heads, np.random.default_rng(perm_seed))
            perm_eval = _prompt_task_eval(
                model=model,
                device=device,
                prompts=prompts,
                kernels=perm_k,
                high_heads=high_heads,
                seq_len=maxlen,
            )
            perm_delta = (baseline[metric_key] - perm_eval[metric_key]).astype(np.float64)
            ctrl_stacks["permuted"].append(perm_delta)

            norm_k = _norm_matched_random_kernels(kernels, high_heads, np.random.default_rng(norm_seed))
            norm_eval = _prompt_task_eval(
                model=model,
                device=device,
                prompts=prompts,
                kernels=norm_k,
                high_heads=high_heads,
                seq_len=maxlen,
            )
            norm_delta = (baseline[metric_key] - norm_eval[metric_key]).astype(np.float64)
            ctrl_stacks["normmatched"].append(norm_delta)

            dist_match = _match_distortion_scaled_random(
                model=model,
                device=device,
                calib_inputs=calib_inputs,
                baseline_attn=base_attn,
                target_kl=target_kl,
                source_kernels=kernels,
                high_heads=high_heads,
                rng=np.random.default_rng(dist_seed),
                seq_len=max(64, int(calibration_seq_len)),
                alpha_grid=alpha_grid,
                tol_rel=float(distortion_tol_rel),
                max_retry=max(1, int(distortion_max_retry)),
                bisect_steps=max(0, int(distortion_bisect_steps)),
                alpha_min=max(1e-4, float(distortion_alpha_min)),
                alpha_max=max(float(distortion_alpha_max), float(distortion_alpha_min) + 1e-4),
            )
            matched = bool(dist_match.get("matched", False))
            dist_delta = None
            if matched:
                dist_k = dist_match["kernels"]
                dist_eval = _prompt_task_eval(
                    model=model,
                    device=device,
                    prompts=prompts,
                    kernels=dist_k,
                    high_heads=high_heads,
                    seq_len=maxlen,
                )
                dist_delta = (baseline[metric_key] - dist_eval[metric_key]).astype(np.float64)
                ctrl_stacks["distortion_matched"].append(dist_delta)

            calib_rows.append(
                {
                    "task": task_name,
                    "trial": int(t),
                    "target_kl": float(target_kl),
                    "matched": bool(matched),
                    "alpha": float(dist_match.get("alpha", float("nan"))),
                    "matched_kl": float(dist_match.get("kl", float("nan"))),
                    "rel_err": float(dist_match.get("rel_err", float("nan"))),
                }
            )

            task_records.extend(
                [
                    {
                        "task": task_name,
                        "metric": metric_key,
                        "arm": "true",
                        "trial": int(t),
                        "mean_damage": float(np.mean(true_delta)),
                        "std_damage": float(np.std(true_delta, ddof=1)) if len(true_delta) > 1 else float("nan"),
                        "n_units": int(len(true_delta)),
                        "valid": True,
                    },
                    {
                        "task": task_name,
                        "metric": metric_key,
                        "arm": "permuted",
                        "trial": int(t),
                        "mean_damage": float(np.mean(perm_delta)),
                        "std_damage": float(np.std(perm_delta, ddof=1)) if len(perm_delta) > 1 else float("nan"),
                        "n_units": int(len(perm_delta)),
                        "valid": True,
                    },
                    {
                        "task": task_name,
                        "metric": metric_key,
                        "arm": "normmatched",
                        "trial": int(t),
                        "mean_damage": float(np.mean(norm_delta)),
                        "std_damage": float(np.std(norm_delta, ddof=1)) if len(norm_delta) > 1 else float("nan"),
                        "n_units": int(len(norm_delta)),
                        "valid": True,
                    },
                    {
                        "task": task_name,
                        "metric": metric_key,
                        "arm": "distortion_matched",
                        "trial": int(t),
                        "mean_damage": float(np.mean(dist_delta)) if dist_delta is not None else float("nan"),
                        "std_damage": float(np.std(dist_delta, ddof=1)) if dist_delta is not None and len(dist_delta) > 1 else float("nan"),
                        "n_units": int(len(dist_delta)) if dist_delta is not None else 0,
                        "valid": bool(matched),
                    },
                ]
            )

            for i in range(len(true_delta)):
                delta_rows.append(
                    {
                        "task": task_name,
                        "metric": metric_key,
                        "unit_index": int(i),
                        "trial": int(t),
                        "true_delta": float(true_delta[i]),
                        "permuted_delta": float(perm_delta[i]),
                        "normmatched_delta": float(norm_delta[i]),
                        "distortion_delta": float(dist_delta[i]) if dist_delta is not None else float("nan"),
                        "distortion_valid": bool(matched),
                    }
                )

        min_valid = max(1, int(distortion_min_valid_trials))
        extra_budget = max(0, int(distortion_max_extra_trials))
        trial_cursor = int(base_trials)
        while len(ctrl_stacks["distortion_matched"]) < min_valid and trial_cursor < base_trials + extra_budget:
            dist_seed = int(rng_seed_base) + trial_cursor * 7919 + 47
            dist_match = _match_distortion_scaled_random(
                model=model,
                device=device,
                calib_inputs=calib_inputs,
                baseline_attn=base_attn,
                target_kl=target_kl,
                source_kernels=kernels,
                high_heads=high_heads,
                rng=np.random.default_rng(dist_seed),
                seq_len=max(64, int(calibration_seq_len)),
                alpha_grid=alpha_grid,
                tol_rel=float(distortion_tol_rel),
                max_retry=max(1, int(distortion_max_retry)),
                bisect_steps=max(0, int(distortion_bisect_steps)),
                alpha_min=max(1e-4, float(distortion_alpha_min)),
                alpha_max=max(float(distortion_alpha_max), float(distortion_alpha_min) + 1e-4),
            )
            matched = bool(dist_match.get("matched", False))
            dist_delta = None
            if matched:
                dist_k = dist_match["kernels"]
                dist_eval = _prompt_task_eval(
                    model=model,
                    device=device,
                    prompts=prompts,
                    kernels=dist_k,
                    high_heads=high_heads,
                    seq_len=maxlen,
                )
                dist_delta = (baseline[metric_key] - dist_eval[metric_key]).astype(np.float64)
                ctrl_stacks["distortion_matched"].append(dist_delta)

            calib_rows.append(
                {
                    "task": task_name,
                    "trial": int(trial_cursor),
                    "target_kl": float(target_kl),
                    "matched": bool(matched),
                    "alpha": float(dist_match.get("alpha", float("nan"))),
                    "matched_kl": float(dist_match.get("kl", float("nan"))),
                    "rel_err": float(dist_match.get("rel_err", float("nan"))),
                    "extra_trial": True,
                }
            )
            task_records.append(
                {
                    "task": task_name,
                    "metric": metric_key,
                    "arm": "distortion_matched",
                    "trial": int(trial_cursor),
                    "mean_damage": float(np.mean(dist_delta)) if dist_delta is not None else float("nan"),
                    "std_damage": float(np.std(dist_delta, ddof=1)) if dist_delta is not None and len(dist_delta) > 1 else float("nan"),
                    "n_units": int(len(dist_delta)) if dist_delta is not None else 0,
                    "valid": bool(matched),
                }
            )
            if dist_delta is not None:
                for i in range(len(true_delta)):
                    delta_rows.append(
                        {
                            "task": task_name,
                            "metric": metric_key,
                            "unit_index": int(i),
                            "trial": int(trial_cursor),
                            "true_delta": float(true_delta[i]),
                            "permuted_delta": float("nan"),
                            "normmatched_delta": float("nan"),
                            "distortion_delta": float(dist_delta[i]),
                            "distortion_valid": True,
                            "extra_trial": True,
                        }
                    )
            trial_cursor += 1

        cis: dict[str, dict[str, float]] = {}
        pass_flags: dict[str, bool] = {}
        for cname in ["permuted", "normmatched", "distortion_matched"]:
            stack = np.stack(ctrl_stacks[cname], axis=0) if ctrl_stacks[cname] else np.empty((0, len(true_delta)))
            ci = _bootstrap_arm_diff_ci(
                true_delta,
                stack,
                n_boot=max(1000, int(n_boot)),
                seed=int(rng_seed_base) + 901 + CONTROL_SEED_OFFSETS[cname],
            )
            cis[cname] = ci
            pass_flags[cname] = bool(np.isfinite(ci["ci_lo"]) and ci["ci_lo"] > 0.0)

        task_pass = bool(all(pass_flags.values()))
        secondary = {}
        if include_secondary_acc:
            secondary_true = baseline["acc"] - true_eval["acc"]
            secondary["accuracy_mean_damage"] = float(np.mean(secondary_true))
            secondary["accuracy_std_damage"] = float(np.std(secondary_true, ddof=1)) if len(secondary_true) > 1 else float("nan")

        summary_tasks[task_name] = {
            "task": task_name,
            "metric": metric_key,
            "n_units": int(len(true_delta)),
            "true_mean_damage": float(np.mean(true_delta)),
            "control_cis": cis,
            "control_pass": pass_flags,
            "task_pass": task_pass,
            "distortion_valid_trials": int(sum(1 for r in calib_rows if r["task"] == task_name and r["matched"])),
            "distortion_total_trials": int(sum(1 for r in calib_rows if r["task"] == task_name)),
            "secondary": secondary,
        }

    # LM domain tasks
    for didx, domain in enumerate(LM_DOMAINS):
        _process_task_lm(
            task_name=f"lm_{domain}",
            sequences=lm_seqs[domain],
            rng_seed_base=int(seed) + 1000 + didx * 100,
        )

    # ICL task (primary metric: lp)
    _process_task_prompt(
        task_name="icl",
        prompts=icl_prompts,
        metric_key="lp",
        rng_seed_base=int(seed) + 2000,
        include_secondary_acc=True,
    )

    # Non-ICL reference task (secondary only)
    _process_task_prompt(
        task_name="nonicl_reference",
        prompts=nonicl_prompts,
        metric_key="lp",
        rng_seed_base=int(seed) + 2100,
        include_secondary_acc=True,
    )

    # Induction-style copy task (primary metric: lp)
    _process_task_prompt(
        task_name="induction",
        prompts=induction_prompts,
        metric_key="lp",
        rng_seed_base=int(seed) + 2200,
        include_secondary_acc=True,
    )

    primary_task_pass = {
        "icl": bool(summary_tasks.get("icl", {}).get("task_pass", False)),
        "induction": bool(summary_tasks.get("induction", {}).get("task_pass", False)),
    }
    model_primary_pass = bool(primary_task_pass["icl"] and primary_task_pass["induction"])

    pd.DataFrame(task_records).to_parquet(model_dir / "task_arm_trials.parquet", index=False)
    pd.DataFrame(delta_rows).to_parquet(model_dir / "task_control_deltas.parquet", index=False)
    write_json(model_dir / "distortion_calibration.json", {"rows": calib_rows})

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E22",
        "model": model_name,
        "runtime_sec": float(time.time() - t0),
        "kernel_source": kernel_path,
        "n_high_si_heads": int(len(high_heads)),
        "lm_domain_raw_counts": {k: int(v) for k, v in lm_raw_counts.items()},
        "lm_sequences_balanced_per_domain": int(lm_balanced_n),
        "primary_task_pass": primary_task_pass,
        "model_primary_pass": bool(model_primary_pass),
        "tasks": summary_tasks,
    }
    write_json(model_dir / "summary.json", summary)

    print(
        f"[E22] {model_name}: primary_pass={model_primary_pass} "
        f"icl={primary_task_pass['icl']} induction={primary_task_pass['induction']}",
        flush=True,
    )
    return summary


def _load_model_summary(model_name: str, out_root: Path) -> dict[str, Any]:
    p = out_root / model_name / "summary.json"
    if not p.exists():
        raise RuntimeError(f"[E22] hard_fail_reason: missing shard artifact {p}")
    return read_json(p)


def _finalize(models: list[str], out_root: Path, start_ts: str) -> dict[str, Any]:
    rows = [_load_model_summary(m, out_root) for m in models]
    enforce_coverage_contract(
        experiment_id="E22",
        observed_models=[r.get("model", "") for r in rows],
        required_models=models,
    )

    n_primary = int(sum(1 for r in rows if bool(r.get("model_primary_pass", False))))

    if n_primary == len(rows):
        claim_status = "supported"
        interp = "supported"
        note = "Task-conditional SI specificity passes in all models (ICL + induction)."
    elif n_primary == max(1, len(rows) - 1):
        claim_status = "supported_with_caveat"
        interp = "supported_with_caveat"
        note = "Task-conditional SI specificity passes in most models with one exception."
    elif n_primary == 1:
        claim_status = "mixed"
        interp = "mixed"
        note = "Task-conditional SI specificity is present in one model only."
    else:
        claim_status = "not_supported"
        interp = "not_supported"
        note = "Task-conditional SI specificity does not pass primary gates in any model."

    cross = {
        "timestamp": timestamp_now(),
        "experiment_id": "E22",
        "n_models": int(len(rows)),
        "n_primary_pass": int(n_primary),
        "interpretation": interp,
        "note": note,
        "per_model": rows,
    }
    write_json(out_root / "cross_model_task_conditional_summary.json", cross)

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "E22",
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "supported_with_caveat", "mixed"),
        "notes": [note],
        "outcome_summary": note,
    }

    prereg = {
        "experiment_id": "E22",
        "question": "Does SI-kernel subtraction show task-conditional specificity beyond baseline LM under strict controls?",
        "primary_hypothesis": "True SI subtraction causes greater ICL and induction degradation than all controls.",
        "primary_endpoints": [
            "icl_true_minus_control_ci95_lo",
            "induction_true_minus_control_ci95_lo",
        ],
        "secondary_endpoints": [
            "lm_domain_true_minus_control_ci95_lo",
            "nonicl_reference_behavior",
            "accuracy_secondary_degradation",
        ],
        "model_list": models,
        "dataset_sources": ["wiki", "code", "dialogue", "synthetic ICL prompts", "synthetic induction/copy prompts"],
        "inclusion_exclusion_rules": [
            "Use top-quartile high-SI head set from existing head groups",
            "LM domains balanced by min loaded sequence count",
            "Distortion-matched trial marked invalid when KL target not matched within tolerance",
            "When initial distortion trials are insufficient, extra calibration trials are drawn up to a fixed cap",
        ],
        "sample_size_plan": {
            "lm_sequences_per_domain": "configured by --lm-num-sequences",
            "icl_prompts": "configured by --icl-eval",
            "induction_prompts": "configured by --induction-eval",
            "control_trials": "configured by --n-control-trials",
        },
        "seed_plan": {"base_seed": SEED_BASE},
        "stopping_rule": "fixed sample sizes and fixed trial counts",
        "multiplicity_family": ["per-task control comparisons"],
        "acceptance_criteria": [
            "Per-task pass requires ci95_lo(true-control) > 0 for permuted, normmatched, and distortion-matched controls",
            "Model-level primary pass requires ICL and induction task passes",
        ],
        "fallback_interpretation_if_null": "SI load-bearing may be broad perturbation sensitivity rather than task-conditional specificity.",
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E22",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": interp,
            "n_primary_pass": int(n_primary),
            "n_models": int(len(rows)),
        },
        "limitations": [
            "Distortion matching uses KL calibration on short held-out batches and may not match all downstream distortions.",
            "Prompt tasks are synthetic and probe positional dependence rather than full naturalistic ICL semantics.",
            "Distortion controls can remain underpowered in models with hard-to-match attention geometry if capped extra trials are exhausted.",
        ],
    }

    data_dictionary = {
        "experiment_id": "E22",
        "tables": [
            {
                "path": "<model>/task_arm_trials.parquet",
                "description": "Task/arm/trial mean damage summaries and validity flags.",
                "columns": [
                    {"name": "task", "dtype": "str", "description": "task id"},
                    {"name": "metric", "dtype": "str", "description": "primary metric used"},
                    {"name": "arm", "dtype": "str", "description": "true/permuted/normmatched/distortion_matched"},
                    {"name": "trial", "dtype": "int", "description": "trial index"},
                    {"name": "mean_damage", "dtype": "float", "description": "mean degradation relative to baseline"},
                    {"name": "valid", "dtype": "bool", "description": "whether the trial is admissible"},
                ],
            },
            {
                "path": "<model>/task_control_deltas.parquet",
                "description": "Per-unit true and control degradations across trials.",
                "columns": [
                    {"name": "task", "dtype": "str", "description": "task id"},
                    {"name": "unit_index", "dtype": "int", "description": "sequence/prompt index"},
                    {"name": "trial", "dtype": "int", "description": "trial index"},
                    {"name": "true_delta", "dtype": "float", "description": "true SI subtraction damage"},
                    {"name": "permuted_delta", "dtype": "float", "description": "permuted control damage"},
                    {"name": "normmatched_delta", "dtype": "float", "description": "norm-matched control damage"},
                    {"name": "distortion_delta", "dtype": "float", "description": "distortion-matched control damage"},
                ],
            },
            {
                "path": "<model>/distortion_calibration.json",
                "description": "Per-task, per-trial KL calibration diagnostics for distortion-matched controls.",
                "columns": [],
            },
        ],
    }

    emit_core_artifacts(
        experiment_id="E22",
        out_dir=out_root,
        preregistration=prereg,
        manifest_extra={"models": models},
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross


def main() -> None:
    p = argparse.ArgumentParser(description="E22: task-conditional SI specificity", allow_abbrev=False)
    p.add_argument("--models", default=",".join(PRIMARY_MODELS))
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0,mistral-7b-v0.1:cuda:0",
    )
    p.add_argument("--output-root", default=str(OUT_ROOT))

    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lm-num-sequences", type=int, default=50)
    p.add_argument("--icl-eval", type=int, default=200)
    p.add_argument("--icl-demos", type=int, default=4)
    p.add_argument("--induction-eval", type=int, default=200)
    p.add_argument("--n-control-trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=SEED_BASE)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=5000)

    p.add_argument("--calibration-count", type=int, default=6)
    p.add_argument("--calibration-seq-len", type=int, default=128)
    p.add_argument("--distortion-tol-rel", type=float, default=0.20)
    p.add_argument("--distortion-max-retry", type=int, default=8)
    p.add_argument("--distortion-bisect-steps", type=int, default=10)
    p.add_argument("--distortion-alpha-min", type=float, default=0.005)
    p.add_argument("--distortion-alpha-max", type=float, default=3.0)
    p.add_argument("--distortion-min-valid-trials", type=int, default=3)
    p.add_argument("--distortion-max-extra-trials", type=int, default=12)
    p.add_argument("--alpha-grid", default="0.005,0.01,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.25,0.33,0.5,0.75,1.0,1.5,2.0,3.0")

    p.add_argument("--smoke", action="store_true", help="Low-cost smoke mode")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E22] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = parse_models_arg(args.models, default=PRIMARY_MODELS)
    device_map = parse_device_map(args.device_map)
    out_root = ensure_dir(Path(args.output_root))
    start_ts = timestamp_now()

    alpha_grid = [float(x.strip()) for x in str(args.alpha_grid).split(",") if x.strip()]
    if not alpha_grid:
        raise RuntimeError("[E22] hard_fail_reason: --alpha-grid must contain at least one float")

    seq_len = int(args.seq_len)
    lm_num_sequences = int(args.lm_num_sequences)
    icl_eval = int(args.icl_eval)
    icl_demos = int(args.icl_demos)
    induction_eval = int(args.induction_eval)
    n_control_trials = int(args.n_control_trials)
    n_boot = int(args.n_boot)
    calibration_count = int(args.calibration_count)
    calibration_seq_len = int(args.calibration_seq_len)
    distortion_min_valid_trials = int(args.distortion_min_valid_trials)
    distortion_max_extra_trials = int(args.distortion_max_extra_trials)

    if args.smoke:
        seq_len = min(seq_len, 256)
        lm_num_sequences = min(lm_num_sequences, 12)
        icl_eval = min(icl_eval, 40)
        icl_demos = min(icl_demos, 3)
        induction_eval = min(induction_eval, 40)
        n_control_trials = min(n_control_trials, 2)
        n_boot = min(n_boot, 1000)
        calibration_count = min(calibration_count, 3)
        calibration_seq_len = min(calibration_seq_len, 96)
        distortion_min_valid_trials = min(max(1, distortion_min_valid_trials), 1)
        distortion_max_extra_trials = min(max(0, distortion_max_extra_trials), 4)

    if args.finalize_only:
        cross = _finalize(models, out_root, start_ts)
        print(f"[E22] Finalized from shards. interpretation={cross['interpretation']}", flush=True)
        return

    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        run_model(
            model_name=model_name,
            device=device,
            out_root=out_root,
            seq_len=max(64, seq_len),
            lm_num_sequences=max(8, lm_num_sequences),
            icl_eval=max(16, icl_eval),
            icl_demos=max(2, icl_demos),
            induction_eval=max(16, induction_eval),
            n_control_trials=max(1, n_control_trials),
            seed=int(args.seed),
            batch_size=max(1, int(args.batch_size)),
            n_boot=max(1000, n_boot),
            calibration_count=max(2, calibration_count),
            calibration_seq_len=max(32, calibration_seq_len),
            distortion_tol_rel=float(args.distortion_tol_rel),
            distortion_max_retry=max(1, int(args.distortion_max_retry)),
            alpha_grid=alpha_grid,
            distortion_bisect_steps=max(0, int(args.distortion_bisect_steps)),
            distortion_alpha_min=max(1e-6, float(args.distortion_alpha_min)),
            distortion_alpha_max=max(float(args.distortion_alpha_max), float(args.distortion_alpha_min) + 1e-6),
            distortion_min_valid_trials=max(1, distortion_min_valid_trials),
            distortion_max_extra_trials=max(0, distortion_max_extra_trials),
        )

    if args.no_finalize:
        print("[E22] Shard run complete (no finalize).", flush=True)
        return

    cross = _finalize(models, out_root, start_ts)
    print(f"[E22] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
