#!/usr/bin/env python3
"""E23 — Stage-Sensitive SI Pretraining (from-scratch tiny RoPE).

Tests whether SI utility must be shaped during pretraining-era dynamics.

Arms (equal token budget):
- baseline: no SI augmentation / no consistency
- early_si_aug: SI augmentation active in first 40% of steps
- late_si_aug: SI augmentation active in last 40% of steps
- full_si_aug: SI augmentation active all steps

Outputs:
- per arm+seed: train_summary.json, offset_eval.parquet, si_r2_summary.parquet, kernel_specificity.json
- per arm: arm_summary.json
- cross arm: cross_arm_stage_sensitivity_summary.json
- core artifacts: preregistration/manifest/summary/claim_impact/data_dictionary
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    emit_core_artifacts,
    enforce_coverage_contract,
    read_json,
)
from experiment3.theory1_si_circuits import (  # noqa: E402
    MODELS as THEORY_MODELS,
    classify_heads,
    compute_per_head_r2,
)
from shared.attention.adapters import get_adapter  # noqa: E402
from experiment3.theory8_position_ablation import (  # noqa: E402
    compute_per_token_loss,
    estimate_kernels,
    subtract_positional_kernels,
)

OUT_ROOT = RESULTS_ROOT / "E23_stage_sensitive_si_pretraining"
ARMS = ("baseline", "early_si_aug", "late_si_aug", "full_si_aug")

MODEL_CFG = {
    "hidden_size": 384,
    "num_hidden_layers": 10,
    "num_attention_heads": 6,
    "num_key_value_heads": 6,
    "intermediate_size": 1536,
    "max_position_embeddings": 256,
    "vocab_size": 192,
    "rope_theta": 10000.0,
}

SEQ_LEN = 128
MICRO_BATCH = 16
GRAD_ACCUM = 4
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_FRAC = 0.05
LAMBDA_CONSISTENCY = 0.2

TOKEN_BUDGET_FULL = 40_000_000
TOKEN_BUDGET_SMOKE = 2_000_000

SEEN_OFFSETS = (8, 16, 24, 32, 40)
UNSEEN_OFFSETS = (12, 20, 28, 36, 44)

TRAIN_TASK_PROBS = {
    "arith": 0.50,
    "copy": 0.50,
}

# Synthetic compact vocab (no tokenizer dependency)
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
TOK_ARITH = 3
TOK_COPY = 4
TOK_EQ = 5
TOK_PLUS = 6
TOK_QUERY = 7
TOK_ANS = 8
TOK_MAP = 9
TOK_SEP = 10

DIGIT_BASE = 16  # 16..25 => 0..9
KEY_BASE = 64    # 64..95 => 32 keys
VAL_BASE = 96    # 96..127 => 32 values
FILLER_BASE = 128
FILLER_TOP = 191


@dataclass(frozen=True)
class Sample:
    input_ids: list[int]
    answer_pos: int
    answer_token: int
    task: str
    offset: int


@dataclass(frozen=True)
class ArithSpec:
    a: int
    b: int


@dataclass(frozen=True)
class CopySpec:
    query_key: int
    distractors: tuple[int, int]


def _digit_token(d: int) -> int:
    return DIGIT_BASE + int(d)


def _key_token(k: int) -> int:
    return KEY_BASE + int(k)


def _val_token(v: int) -> int:
    return VAL_BASE + int(v)


def _value_for_key(k: int) -> int:
    # Fixed permutation-style mapping for copy probe
    return (int(k) * 7 + 3) % 32


def _rand_filler(rng: np.random.Generator, n: int) -> list[int]:
    if n <= 0:
        return []
    return rng.integers(FILLER_BASE, FILLER_TOP + 1, size=n).astype(int).tolist()


def _make_arith_spec(rng: np.random.Generator) -> ArithSpec:
    return ArithSpec(a=int(rng.integers(0, 10)), b=int(rng.integers(0, 10)))


def _make_copy_spec(rng: np.random.Generator) -> CopySpec:
    q = int(rng.integers(0, 32))
    d1 = int((q + int(rng.integers(1, 31))) % 32)
    d2 = int((q + int(rng.integers(1, 31))) % 32)
    if d2 == d1:
        d2 = (d2 + 1) % 32
    return CopySpec(query_key=q, distractors=(d1, d2))


def _render_arith_sample(spec: ArithSpec, offset: int, seq_len: int, rng: np.random.Generator) -> Sample:
    ans = int((spec.a + spec.b) % 10)
    core = [
        TOK_ARITH,
        _digit_token(spec.a),
        TOK_PLUS,
        _digit_token(spec.b),
        TOK_EQ,
        TOK_ANS,
        _digit_token(ans),
        EOS_ID,
    ]
    off = max(0, int(offset))
    if off + len(core) > seq_len:
        off = max(0, seq_len - len(core))
    prefix = _rand_filler(rng, off)
    suffix = _rand_filler(rng, max(0, seq_len - off - len(core)))
    seq = (prefix + core + suffix)[:seq_len]
    answer_pos = int(off + 6)  # index of answer token
    return Sample(
        input_ids=seq,
        answer_pos=answer_pos,
        answer_token=_digit_token(ans),
        task="arith",
        offset=int(off),
    )


def _render_copy_sample(spec: CopySpec, offset: int, seq_len: int, rng: np.random.Generator) -> Sample:
    q = int(spec.query_key)
    d1, d2 = int(spec.distractors[0]), int(spec.distractors[1])
    vq = _value_for_key(q)
    v1 = _value_for_key(d1)
    v2 = _value_for_key(d2)

    core = [
        TOK_COPY,
        _key_token(d1), TOK_MAP, _val_token(v1), TOK_SEP,
        _key_token(d2), TOK_MAP, _val_token(v2), TOK_SEP,
        TOK_QUERY, _key_token(q), TOK_ANS, _val_token(vq),
        EOS_ID,
    ]
    off = max(0, int(offset))
    if off + len(core) > seq_len:
        off = max(0, seq_len - len(core))
    prefix = _rand_filler(rng, off)
    suffix = _rand_filler(rng, max(0, seq_len - off - len(core)))
    seq = (prefix + core + suffix)[:seq_len]
    answer_pos = int(off + 12)  # index of target value token
    return Sample(
        input_ids=seq,
        answer_pos=answer_pos,
        answer_token=_val_token(vq),
        task="copy",
        offset=int(off),
    )


def _choose_task(rng: np.random.Generator) -> str:
    p = float(rng.random())
    return "arith" if p < TRAIN_TASK_PROBS["arith"] else "copy"


def _build_sample(
    *,
    rng: np.random.Generator,
    seq_len: int,
    task: str,
    offset: int,
    spec: ArithSpec | CopySpec | None = None,
) -> Sample:
    if task == "arith":
        sp = spec if isinstance(spec, ArithSpec) else _make_arith_spec(rng)
        return _render_arith_sample(sp, offset, seq_len, rng)
    if task == "copy":
        sp2 = spec if isinstance(spec, CopySpec) else _make_copy_spec(rng)
        return _render_copy_sample(sp2, offset, seq_len, rng)
    raise ValueError(f"Unsupported task={task}")


def _si_aug_active(arm: str, step_idx_1based: int, total_steps: int) -> bool:
    if arm == "baseline":
        return False
    frac = float(step_idx_1based / max(1, total_steps))
    if arm == "full_si_aug":
        return True
    if arm == "early_si_aug":
        return frac <= 0.40
    if arm == "late_si_aug":
        return frac > 0.60
    raise ValueError(f"Unsupported arm={arm}")


def _build_model(device: str) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=int(MODEL_CFG["vocab_size"]),
        hidden_size=int(MODEL_CFG["hidden_size"]),
        intermediate_size=int(MODEL_CFG["intermediate_size"]),
        num_hidden_layers=int(MODEL_CFG["num_hidden_layers"]),
        num_attention_heads=int(MODEL_CFG["num_attention_heads"]),
        num_key_value_heads=int(MODEL_CFG["num_key_value_heads"]),
        max_position_embeddings=int(MODEL_CFG["max_position_embeddings"]),
        rope_theta=float(MODEL_CFG["rope_theta"]),
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
        rms_norm_eps=1e-6,
        use_cache=False,
    )
    model = LlamaForCausalLM(cfg)
    model.to(device)
    model.train()
    return model


def _forward_ce_loss(model: LlamaForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    # Mask out filler tokens so the loss focuses on task-relevant positions.
    # Without this mask the ~90% filler positions dominate the CE signal and
    # the arithmetic task never receives meaningful gradient.
    mask = (labels < FILLER_BASE).view(-1)
    if not mask.any():
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="mean")
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1])[mask],
        labels.view(-1)[mask],
        reduction="mean",
    )


def _answer_logprob(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    answer_pos: torch.Tensor,
    answer_tok: torch.Tensor,
) -> torch.Tensor:
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits.float()  # [b, seq, vocab]
    pred_pos = torch.clamp(answer_pos - 1, min=0)
    gather_logits = logits[torch.arange(logits.shape[0], device=logits.device), pred_pos, :]
    lp = F.log_softmax(gather_logits, dim=-1)
    answer_lp = lp.gather(dim=-1, index=answer_tok.view(-1, 1)).squeeze(-1)
    return answer_lp


def _build_train_microbatch(
    *,
    rng: np.random.Generator,
    seq_len: int,
    micro_batch: int,
    offsets: tuple[int, ...],
    shifted_offsets: tuple[int, ...],
    with_si_aug: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, None, None, None]:
    base_samples: list[Sample] = []
    shift_samples: list[Sample] = []
    ans_pos_base: list[int] = []
    ans_pos_shift: list[int] = []
    ans_tok: list[int] = []

    for _ in range(int(micro_batch)):
        task = _choose_task(rng)
        off = int(offsets[int(rng.integers(0, len(offsets)))])
        if task == "arith":
            spec = _make_arith_spec(rng)
        else:
            spec = _make_copy_spec(rng)

        base = _build_sample(rng=rng, seq_len=seq_len, task=task, offset=off, spec=spec)
        base_samples.append(base)

        if with_si_aug:
            off2 = int(shifted_offsets[int(rng.integers(0, len(shifted_offsets)))])
            sh = _build_sample(rng=rng, seq_len=seq_len, task=task, offset=off2, spec=spec)
            shift_samples.append(sh)
            ans_pos_base.append(int(base.answer_pos))
            ans_pos_shift.append(int(sh.answer_pos))
            ans_tok.append(int(base.answer_token))

    base_ids = torch.tensor([s.input_ids for s in base_samples], dtype=torch.long)
    if not with_si_aug:
        return base_ids, None, None, None

    shift_ids = torch.tensor([s.input_ids for s in shift_samples], dtype=torch.long)
    pos_base = torch.tensor(ans_pos_base, dtype=torch.long)
    pos_shift = torch.tensor(ans_pos_shift, dtype=torch.long)
    tok = torch.tensor(ans_tok, dtype=torch.long)
    # pack shift pos/tok as [3, batch] for concise transport
    aux = torch.stack([pos_base, pos_shift, tok], dim=0)
    return base_ids, shift_ids, aux, torch.tensor([1], dtype=torch.long)


def _steps_for_budget(token_budget: int, seq_len: int, micro_batch: int, grad_accum: int) -> int:
    toks_per_step = int(seq_len) * int(micro_batch) * int(grad_accum)
    return max(1, int(math.ceil(float(token_budget) / float(toks_per_step))))


def _eval_samples(
    *,
    model: LlamaForCausalLM,
    samples: list[Sample],
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pos = 0
    bs = max(1, int(batch_size))
    model.eval()
    with torch.inference_mode():
        while pos < len(samples):
            chunk = samples[pos : pos + bs]
            ids = torch.tensor([s.input_ids for s in chunk], dtype=torch.long, device=device)
            ans_pos = torch.tensor([int(s.answer_pos) for s in chunk], dtype=torch.long, device=device)
            ans_tok = torch.tensor([int(s.answer_token) for s in chunk], dtype=torch.long, device=device)

            out = model(input_ids=ids, use_cache=False)
            logits = out.logits.float()
            pred_pos = torch.clamp(ans_pos - 1, min=0)
            gather_logits = logits[torch.arange(logits.shape[0], device=device), pred_pos, :]
            lp = F.log_softmax(gather_logits, dim=-1)
            target_lp = lp.gather(dim=-1, index=ans_tok.view(-1, 1)).squeeze(-1)
            pred = torch.argmax(gather_logits, dim=-1)
            correct = (pred == ans_tok).to(torch.int64)

            for i, s in enumerate(chunk):
                rows.append(
                    {
                        "task": s.task,
                        "offset": int(s.offset),
                        "target_logprob": float(target_lp[i].item()),
                        "correct": int(correct[i].item()),
                    }
                )

            pos += len(chunk)

    return pd.DataFrame(rows)


def _build_eval_set(
    *,
    rng: np.random.Generator,
    seq_len: int,
    per_offset_per_task: int,
    offsets: tuple[int, ...],
) -> list[Sample]:
    samples: list[Sample] = []
    for task in ("arith", "copy"):
        for off in offsets:
            for _ in range(int(per_offset_per_task)):
                samples.append(_build_sample(rng=rng, seq_len=seq_len, task=task, offset=int(off)))
    return samples


def _compute_si_r2_summary(
    *,
    model: LlamaForCausalLM,
    seqs: list[list[int]],
    device: str,
) -> pd.DataFrame:
    model.eval()
    spec = THEORY_MODELS["llama-3.1-8b"]
    adapter = get_adapter(spec)
    adapter.register(model)
    try:
        r2_df = compute_per_head_r2(
            model=model,
            adapter=adapter,
            tokenizer=None,
            model_spec=spec,
            device=device,
            sequences=seqs,
        )
    finally:
        try:
            adapter.cleanup()
        except Exception:
            pass

    _high, _low, mean_r2 = classify_heads(r2_df)
    return mean_r2


def _bootstrap_true_minus_control(true_delta: np.ndarray, ctrl_delta: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    td = np.asarray(true_delta, dtype=np.float64)
    cd = np.asarray(ctrl_delta, dtype=np.float64)
    if td.size == 0 or cd.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    n = min(td.size, cd.size)
    td = td[:n]
    cd = cd[:n]
    obs = float(np.mean(td - cd))
    rng = np.random.default_rng(int(seed))
    m = max(1000, int(n_boot))
    boot = np.empty(m, dtype=np.float64)
    for i in range(m):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(td[idx] - cd[idx]))
    return {
        "mean": obs,
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
    }


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


def _eval_seq_mean_nll(model: LlamaForCausalLM, seqs: list[list[int]], device: str, batch_size: int) -> np.ndarray:
    vals: list[float] = []
    pos = 0
    bs = max(1, int(batch_size))
    model.eval()
    while pos < len(seqs):
        batch = seqs[pos : pos + bs]
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
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and bs > 1:
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                continue
            raise
    return np.asarray(vals, dtype=np.float64)


def _kernel_specificity(
    *,
    model: LlamaForCausalLM,
    r2_mean_df: pd.DataFrame,
    eval_seqs: list[list[int]],
    seq_len: int,
    device: str,
    seed: int,
    n_boot: int,
) -> dict[str, Any]:
    model.eval()

    # top quartile high-SI heads from current trained model
    r2_sorted = r2_mean_df.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    n_heads = int(len(r2_sorted))
    n_sel = max(1, int(n_heads * 0.25))
    high = [
        (int(r.layer), int(r.head))
        for r in r2_sorted.head(n_sel).itertuples()
    ]

    # estimate kernels on model itself
    spec = THEORY_MODELS["llama-3.1-8b"]
    adapter = get_adapter(spec)
    adapter.register(model)
    try:
        kernels = estimate_kernels(
            model=model,
            adapter=adapter,
            sequences=eval_seqs,
            device=device,
            seq_len=seq_len,
        )
    finally:
        try:
            adapter.cleanup()
        except Exception:
            pass

    baseline = _eval_seq_mean_nll(model, eval_seqs, device, batch_size=8)

    with subtract_positional_kernels(model, kernels, high, int(seq_len)):
        true_nll = _eval_seq_mean_nll(model, eval_seqs, device, batch_size=8)
    true_delta = true_nll - baseline

    rng = np.random.default_rng(int(seed) + 9001)
    perm_k = _permute_kernels(kernels, high, rng)
    with subtract_positional_kernels(model, perm_k, high, int(seq_len)):
        perm_nll = _eval_seq_mean_nll(model, eval_seqs, device, batch_size=8)
    perm_delta = perm_nll - baseline

    rng2 = np.random.default_rng(int(seed) + 9173)
    norm_k = _norm_matched_random_kernels(kernels, high, rng2)
    with subtract_positional_kernels(model, norm_k, high, int(seq_len)):
        norm_nll = _eval_seq_mean_nll(model, eval_seqs, device, batch_size=8)
    norm_delta = norm_nll - baseline

    ci_perm = _bootstrap_true_minus_control(true_delta, perm_delta, n_boot=n_boot, seed=seed + 1)
    ci_norm = _bootstrap_true_minus_control(true_delta, norm_delta, n_boot=n_boot, seed=seed + 2)

    return {
        "n_sequences": int(len(eval_seqs)),
        "n_high_si_heads": int(len(high)),
        "mean_delta_true": float(np.mean(true_delta)),
        "mean_delta_permuted": float(np.mean(perm_delta)),
        "mean_delta_normmatched": float(np.mean(norm_delta)),
        "ratio_true_over_permuted": float(np.mean(true_delta) / max(abs(float(np.mean(perm_delta))), 1e-8)),
        "ratio_true_over_normmatched": float(np.mean(true_delta) / max(abs(float(np.mean(norm_delta))), 1e-8)),
        "true_minus_permuted_ci95": [float(ci_perm["ci_lo"]), float(ci_perm["ci_hi"])],
        "true_minus_normmatched_ci95": [float(ci_norm["ci_lo"]), float(ci_norm["ci_hi"])],
        "supports_specificity": bool(ci_perm["ci_lo"] > 0.0 and ci_norm["ci_lo"] > 0.0),
    }


def _model_profile_sequences(rng: np.random.Generator, n: int, seq_len: int) -> list[list[int]]:
    seqs: list[list[int]] = []
    offsets = tuple(sorted(set(SEEN_OFFSETS + UNSEEN_OFFSETS)))
    for _ in range(int(n)):
        task = _choose_task(rng)
        off = int(offsets[int(rng.integers(0, len(offsets)))])
        s = _build_sample(rng=rng, seq_len=seq_len, task=task, offset=off)
        seqs.append(s.input_ids)
    return seqs


def _run_seed(
    *,
    arm: str,
    seed: int,
    out_root: Path,
    device: str,
    smoke: bool,
    token_budget: int,
    seq_len: int,
    micro_batch: int,
    grad_accum: int,
    lr: float,
    warmup_frac: float,
    lambda_consistency: float,
) -> dict[str, Any]:
    t0 = time.time()
    seed_dir = ensure_dir(out_root / arm / f"seed_{seed}")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    rng = np.random.default_rng(int(seed))

    model = _build_model(device)
    optimizer = AdamW(model.parameters(), lr=float(lr), weight_decay=float(WEIGHT_DECAY))

    total_steps = _steps_for_budget(token_budget, seq_len, micro_batch, grad_accum)
    warmup_steps = max(1, int(round(total_steps * float(warmup_frac))))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    train_rows: list[dict[str, Any]] = []
    seen_offsets = tuple(int(x) for x in SEEN_OFFSETS)
    shifted_offsets = tuple(int(x) for x in (SEEN_OFFSETS + UNSEEN_OFFSETS))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, total_steps + 1):
        active = _si_aug_active(arm, step, total_steps)
        accum_loss = 0.0
        accum_ce = 0.0
        accum_symkl = 0.0

        for _ in range(int(grad_accum)):
            batch = _build_train_microbatch(
                rng=rng,
                seq_len=seq_len,
                micro_batch=micro_batch,
                offsets=seen_offsets,
                shifted_offsets=shifted_offsets,
                with_si_aug=active,
            )
            base_ids = batch[0].to(device)
            shift_ids = batch[1]
            aux = batch[2]

            ce = _forward_ce_loss(model, base_ids)
            loss = ce
            symkl_val = torch.tensor(0.0, dtype=ce.dtype, device=ce.device)

            if active and shift_ids is not None and aux is not None:
                shift_ids = shift_ids.to(device)
                aux = aux.to(device)
                pos_base = aux[0]
                pos_shift = aux[1]
                tok = aux[2]

                lp_base = _answer_logprob(model, base_ids, pos_base, tok)
                lp_shift = _answer_logprob(model, shift_ids, pos_shift, tok)
                p = torch.exp(lp_base)
                q = torch.exp(lp_shift)
                # Symmetric KL on Bernoulli event of correct target token mass
                p = torch.clamp(p, 1e-6, 1 - 1e-6)
                q = torch.clamp(q, 1e-6, 1 - 1e-6)
                kl_pq = p * (torch.log(p) - torch.log(q)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - q))
                kl_qp = q * (torch.log(q) - torch.log(p)) + (1 - q) * (torch.log(1 - q) - torch.log(1 - p))
                symkl_val = 0.5 * (kl_pq + kl_qp).mean()
                loss = ce + (float(lambda_consistency) * symkl_val)

            scaled = loss / float(grad_accum)
            scaled.backward()

            accum_loss += float(loss.detach().item())
            accum_ce += float(ce.detach().item())
            accum_symkl += float(symkl_val.detach().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), float(GRAD_CLIP))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        train_rows.append(
            {
                "step": int(step),
                "si_aug_active": bool(active),
                "loss": float(accum_loss / float(grad_accum)),
                "ce_loss": float(accum_ce / float(grad_accum)),
                "symkl": float(accum_symkl / float(grad_accum)),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if step == 1 or step % 100 == 0 or step == total_steps:
            elapsed = time.time() - t0
            eta = (elapsed / max(step, 1)) * max(0, total_steps - step)
            print(
                f"[E23] arm={arm} seed={seed} step={step}/{total_steps} "
                f"loss={train_rows[-1]['loss']:.4f} ce={train_rows[-1]['ce_loss']:.4f} "
                f"symkl={train_rows[-1]['symkl']:.4f} active={int(active)} eta={eta:.1f}s",
                flush=True,
            )

    train_df = pd.DataFrame(train_rows)
    train_df.to_parquet(seed_dir / "train_loss_curve.parquet", index=False)

    # Evaluation
    eval_rng = np.random.default_rng(int(seed) + 100_003)
    per_offset = 24 if smoke else 120
    seen_samples = _build_eval_set(
        rng=eval_rng,
        seq_len=seq_len,
        per_offset_per_task=per_offset,
        offsets=tuple(int(x) for x in SEEN_OFFSETS),
    )
    unseen_samples = _build_eval_set(
        rng=eval_rng,
        seq_len=seq_len,
        per_offset_per_task=per_offset,
        offsets=tuple(int(x) for x in UNSEEN_OFFSETS),
    )

    seen_df = _eval_samples(model=model, samples=seen_samples, device=device, batch_size=32)
    seen_df["bucket"] = "seen"
    unseen_df = _eval_samples(model=model, samples=unseen_samples, device=device, batch_size=32)
    unseen_df["bucket"] = "unseen"
    eval_df = pd.concat([seen_df, unseen_df], ignore_index=True)
    eval_df.to_parquet(seed_dir / "offset_eval.parquet", index=False)

    # SI R² summary
    r2_rng = np.random.default_rng(int(seed) + 200_003)
    r2_seqs = _model_profile_sequences(r2_rng, n=(12 if smoke else 32), seq_len=seq_len)
    r2_mean = _compute_si_r2_summary(model=model, seqs=r2_seqs, device=device)
    r2_mean.to_parquet(seed_dir / "si_r2_summary.parquet", index=False)

    # Kernel specificity
    k_rng = np.random.default_rng(int(seed) + 300_003)
    k_seqs = _model_profile_sequences(k_rng, n=(16 if smoke else 40), seq_len=seq_len)
    kspec = _kernel_specificity(
        model=model,
        r2_mean_df=r2_mean,
        eval_seqs=k_seqs,
        seq_len=seq_len,
        device=device,
        seed=int(seed),
        n_boot=(1000 if smoke else 3000),
    )
    write_json(seed_dir / "kernel_specificity.json", kspec)

    # Summaries used for cross-arm gates
    agg = eval_df.groupby(["task", "bucket"]).agg(
        accuracy=("correct", "mean"),
        mean_target_logprob=("target_logprob", "mean"),
    ).reset_index()

    def _metric(task: str, bucket: str, col: str) -> float:
        sub = agg[(agg["task"] == task) & (agg["bucket"] == bucket)]
        if sub.empty:
            return float("nan")
        return float(sub.iloc[0][col])

    arith_unseen_acc = _metric("arith", "unseen", "accuracy")
    copy_unseen_acc = _metric("copy", "unseen", "accuracy")
    joint_unseen = 0.5 * arith_unseen_acc + 0.5 * copy_unseen_acc

    train_summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E23",
        "arm": arm,
        "seed": int(seed),
        "runtime_sec": float(time.time() - t0),
        "smoke": bool(smoke),
        "token_budget": int(token_budget),
        "steps": int(total_steps),
        "tokens_per_step": int(seq_len * micro_batch * grad_accum),
        "effective_tokens": int(total_steps * seq_len * micro_batch * grad_accum),
        "train_config": {
            "seq_len": int(seq_len),
            "micro_batch": int(micro_batch),
            "grad_accum": int(grad_accum),
            "lr": float(lr),
            "warmup_frac": float(warmup_frac),
            "weight_decay": float(WEIGHT_DECAY),
            "grad_clip": float(GRAD_CLIP),
            "lambda_consistency": float(lambda_consistency),
            "seen_offsets": [int(x) for x in SEEN_OFFSETS],
            "unseen_offsets": [int(x) for x in UNSEEN_OFFSETS],
        },
        "model_config": dict(MODEL_CFG),
        "metrics": {
            "arith_unseen_acc": float(arith_unseen_acc),
            "copy_unseen_acc": float(copy_unseen_acc),
            "joint_unseen_score": float(joint_unseen),
            "arith_seen_acc": float(_metric("arith", "seen", "accuracy")),
            "copy_seen_acc": float(_metric("copy", "seen", "accuracy")),
            "arith_unseen_logprob": float(_metric("arith", "unseen", "mean_target_logprob")),
            "copy_unseen_logprob": float(_metric("copy", "unseen", "mean_target_logprob")),
            "si_mean_r2": float(r2_mean["mean_r2"].mean()),
            "si_median_r2": float(r2_mean["mean_r2"].median()),
            "si_q75_r2": float(r2_mean["mean_r2"].quantile(0.75)),
            "kernel_specificity_support": bool(kspec.get("supports_specificity", False)),
        },
    }
    write_json(seed_dir / "train_summary.json", train_summary)

    # cleanup
    try:
        del model
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_summary


def _arm_summary(out_root: Path, arm: str, seeds: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for s in seeds:
        p = out_root / arm / f"seed_{s}" / "train_summary.json"
        if not p.exists():
            raise RuntimeError(f"[E23] hard_fail_reason: missing seed summary {p}")
        rows.append(read_json(p))

    joint = [float(r.get("metrics", {}).get("joint_unseen_score", float("nan"))) for r in rows]
    arith = [float(r.get("metrics", {}).get("arith_unseen_acc", float("nan"))) for r in rows]
    copy = [float(r.get("metrics", {}).get("copy_unseen_acc", float("nan"))) for r in rows]
    si = [float(r.get("metrics", {}).get("si_mean_r2", float("nan"))) for r in rows]
    ksup = [bool(r.get("metrics", {}).get("kernel_specificity_support", False)) for r in rows]

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E23",
        "arm": arm,
        "n_seeds": int(len(rows)),
        "seed_summaries": rows,
        "arm_metrics": {
            "joint_unseen_score_mean": float(np.nanmean(np.asarray(joint, dtype=float))),
            "joint_unseen_score_std": float(np.nanstd(np.asarray(joint, dtype=float), ddof=1)) if len(joint) > 1 else float("nan"),
            "arith_unseen_acc_mean": float(np.nanmean(np.asarray(arith, dtype=float))),
            "copy_unseen_acc_mean": float(np.nanmean(np.asarray(copy, dtype=float))),
            "si_mean_r2_mean": float(np.nanmean(np.asarray(si, dtype=float))),
            "kernel_specificity_support_rate": float(np.mean(np.asarray(ksup, dtype=float))),
        },
    }
    write_json(out_root / arm / "arm_summary.json", summary)
    return summary


def _cross_arm_finalize(out_root: Path, arms: list[str], seeds: list[int], smoke: bool) -> dict[str, Any]:
    observed_arms = [a for a in arms if (out_root / a / "arm_summary.json").exists()]
    enforce_coverage_contract(
        experiment_id="E23",
        observed_models=observed_arms,
        required_models=arms,
    )

    arm_summaries = {a: read_json(out_root / a / "arm_summary.json") for a in arms}

    def m(arm: str) -> float:
        return float(arm_summaries[arm].get("arm_metrics", {}).get("joint_unseen_score_mean", float("nan")))

    baseline = m("baseline")
    early = m("early_si_aug")
    late = m("late_si_aug")
    full = m("full_si_aug")

    pass_early = bool((early - baseline) >= 0.04)
    pass_full = bool((full - baseline) >= 0.04)
    pass_late = bool(abs(late - baseline) <= 0.02)
    stage_pattern_pass = bool(pass_early and pass_full and pass_late)

    if stage_pattern_pass:
        claim_status = "supported"
        interpretation = "stage_sensitive_supported"
        note = "Observed pattern matches prediction: early/full improve unseen-offset joint score; late-only stays near baseline."
    elif (pass_early or pass_full) and pass_late:
        claim_status = "supported_with_caveat"
        interpretation = "partial_stage_signal"
        note = "Partial stage-sensitive signal: one of early/full clears threshold while late-only remains near baseline."
    elif pass_early or pass_full:
        claim_status = "mixed"
        interpretation = "ambiguous_stage_signal"
        note = "Some improvement exists but late-only constraint failed or improvements are inconsistent."
    else:
        claim_status = "not_supported"
        interpretation = "no_stage_signal"
        note = "Stage-sensitive pattern not observed under current budget/configuration."

    cross = {
        "timestamp": timestamp_now(),
        "experiment_id": "E23",
        "arms": arms,
        "seeds": [int(s) for s in seeds],
        "smoke": bool(smoke),
        "joint_unseen_score": {
            "baseline": float(baseline),
            "early_si_aug": float(early),
            "late_si_aug": float(late),
            "full_si_aug": float(full),
            "delta_early_minus_baseline": float(early - baseline),
            "delta_full_minus_baseline": float(full - baseline),
            "delta_late_minus_baseline": float(late - baseline),
        },
        "gate": {
            "pass_early_ge_0p04": bool(pass_early),
            "pass_full_ge_0p04": bool(pass_full),
            "pass_late_abs_le_0p02": bool(pass_late),
            "stage_pattern_pass": bool(stage_pattern_pass),
        },
        "interpretation": interpretation,
        "note": note,
        "arm_summaries": arm_summaries,
    }
    write_json(out_root / "cross_arm_stage_sensitivity_summary.json", cross)

    prereg = {
        "experiment_id": "E23",
        "question": "Is SI utility training-stage-sensitive during from-scratch tiny-RoPE pretraining?",
        "primary_hypothesis": "Early/full SI augmentation improves unseen-offset generalization while late-only remains near baseline.",
        "primary_endpoints": [
            "joint_unseen_score_by_arm",
            "delta_early_minus_baseline",
            "delta_full_minus_baseline",
            "delta_late_minus_baseline",
        ],
        "secondary_endpoints": [
            "si_mean_r2_by_arm",
            "kernel_specificity_support_rate_by_arm",
            "offset_bucket_target_logprob",
        ],
        "model_family": "from_scratch_tiny_llama_rope",
        "arm_definition": {
            "baseline": "no SI augmentation",
            "early_si_aug": "SI augmentation + consistency in first 40% steps",
            "late_si_aug": "SI augmentation + consistency in last 40% steps",
            "full_si_aug": "SI augmentation + consistency all steps",
        },
        "sample_size_plan": {
            "arms": arms,
            "seeds_per_arm": int(len(seeds)),
            "token_budget_per_arm_seed": int(TOKEN_BUDGET_SMOKE if smoke else TOKEN_BUDGET_FULL),
        },
        "acceptance_criteria": [
            "early-baseline >= +0.04",
            "full-baseline >= +0.04",
            "|late-baseline| <= 0.02",
        ],
        "fallback_interpretation_if_null": "Under this budget, SI shaping may require larger scale/longer training or different augmentation design.",
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "E23",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "claim_status": claim_status,
            "interpretation": interpretation,
            "stage_pattern_pass": bool(stage_pattern_pass),
        },
        "limitations": [
            "Tiny synthetic model/task setting; extrapolation to larger natural-corpus models requires follow-up.",
            "Kernel-specificity check still uses perturbational ablation and includes softmax renormalization effects.",
        ],
    }

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "E23",
        "claim_status": claim_status,
        "supports_main_text": claim_status in ("supported", "supported_with_caveat", "mixed"),
        "notes": [note],
        "outcome_summary": note,
    }

    data_dictionary = {
        "experiment_id": "E23",
        "tables": [
            {
                "path": "<arm>/seed_<k>/offset_eval.parquet",
                "description": "Per-example unseen/seen offset evaluation outcomes for arithmetic and copy tasks.",
                "columns": [
                    {"name": "task", "dtype": "str", "description": "arith/copy"},
                    {"name": "bucket", "dtype": "str", "description": "seen/unseen offset bucket"},
                    {"name": "offset", "dtype": "int", "description": "absolute answer offset"},
                    {"name": "target_logprob", "dtype": "float", "description": "log probability on target token"},
                    {"name": "correct", "dtype": "int", "description": "1 if argmax token equals target"},
                ],
            },
            {
                "path": "<arm>/seed_<k>/si_r2_summary.parquet",
                "description": "Per-head SI R² summary of trained model.",
                "columns": [
                    {"name": "layer", "dtype": "int", "description": "layer index"},
                    {"name": "head", "dtype": "int", "description": "head index"},
                    {"name": "mean_r2", "dtype": "float", "description": "mean SI R²"},
                ],
            },
        ],
    }

    emit_core_artifacts(
        experiment_id="E23",
        out_dir=out_root,
        preregistration=prereg,
        manifest_extra={
            "arms": arms,
            "seeds": [int(s) for s in seeds],
            "smoke": bool(smoke),
        },
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )

    return cross


def _parse_csv_ints(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise RuntimeError("[E23] hard_fail_reason: empty --seed-list")
    return sorted(set(vals))


def _parse_arms(raw: str) -> list[str]:
    if not str(raw).strip() or str(raw).strip().lower() == "all":
        return list(ARMS)
    out: list[str] = []
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        if tok not in ARMS:
            raise RuntimeError(f"[E23] hard_fail_reason: unknown arm '{tok}'")
        out.append(tok)
    if not out:
        raise RuntimeError("[E23] hard_fail_reason: no valid arms in --arms")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="E23: stage-sensitive SI pretraining", allow_abbrev=False)
    p.add_argument("--arms", default="all", help="Comma-separated arms or 'all'")
    p.add_argument("--seed-list", default="0,1", help="Comma-separated seeds")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--token-budget", type=int, default=TOKEN_BUDGET_FULL)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--micro-batch", type=int, default=MICRO_BATCH)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--warmup-frac", type=float, default=WARMUP_FRAC)
    p.add_argument("--lambda-consistency", type=float, default=LAMBDA_CONSISTENCY)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E23] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    arms = _parse_arms(args.arms)
    seeds = _parse_csv_ints(args.seed_list)
    out_root = ensure_dir(Path(args.output_root))

    token_budget = int(args.token_budget)
    if args.smoke:
        token_budget = min(token_budget, TOKEN_BUDGET_SMOKE)

    if args.finalize_only:
        cross = _cross_arm_finalize(out_root, arms=list(ARMS), seeds=seeds, smoke=bool(args.smoke))
        print(f"[E23] Finalized from shard outputs. interpretation={cross['interpretation']}", flush=True)
        return

    for arm in arms:
        for seed in seeds:
            _run_seed(
                arm=arm,
                seed=int(seed),
                out_root=out_root,
                device=str(args.device),
                smoke=bool(args.smoke),
                token_budget=int(token_budget),
                seq_len=max(32, int(args.seq_len)),
                micro_batch=max(1, int(args.micro_batch)),
                grad_accum=max(1, int(args.grad_accum)),
                lr=float(args.lr),
                warmup_frac=float(args.warmup_frac),
                lambda_consistency=float(args.lambda_consistency),
            )

        _arm_summary(out_root, arm=arm, seeds=seeds)

    if args.no_finalize:
        print("[E23] Shard run complete (no finalize).", flush=True)
        return

    cross = _cross_arm_finalize(out_root, arms=list(ARMS), seeds=seeds, smoke=bool(args.smoke))
    print(f"[E23] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
