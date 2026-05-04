"""Experiment 6: SI-Guided Math Knowledge Localization.

Six sub-experiments that test whether forcing math knowledge into shift-invariant
channels improves performance and position-invariance:

  6A  Gradient routing          – math gradients flow only through SI heads
  6B  Anti-localization loss    – penalize math capability in non-SI channels
  6C  Two-phase distillation    – teacher → SI-only student
  6D  Contrastive channel       – variance-based routing pressure
  6E  SI-optimized tokenizer    – tokenizer formatting to align with SI heads
  6F  Position-invariance eval  – shared evaluation across all approaches
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from experiment3.theory1_si_circuits import (
    MODELS as THEORY_MODELS,
    classify_heads,
    compute_per_head_r2,
    head_output_ablation,
    load_profile_sequences,
)
from experiment3.theory5b_boundary_detection import run_attention_analysis
from experiment4.common import (
    HeadIndex,
    clear_cuda,
    cohens_d,
    ensure_dir,
    format_eta,
    mean_ci95,
    now_timestamp,
    read_json,
    safe_float,
    set_global_seed,
    write_json,
    write_parquet,
)
from experiment4.lora import (
    LoRAApplyResult,
    LoRAHyperParams,
    apply_lora_policy,
    collect_trainable_parameter_names,
)
from experiment4.math_data import (
    MCExample,
    build_math_eval_battery,
    build_math_training_texts,
    load_control_texts,
)
from experiment4.pipeline import (
    AuxLossConfig,
    TrainConfig,
    _compute_attention_invariance,
    _critical_position_mask,
    _encode_texts,
    _force_eager_attention,
    _load_baseline_head_groups,
    _prepare_tokenizer_for_training,
    _score_option_logprob,
    _si_map_from_heads,
    evaluate_math_accuracy,
    evaluate_wiki_perplexity,
    post_ft_si_audit,
)
from experiment6.config import POSITION_EVAL_SLOTS, TOKENIZER_MATH_OPERATORS
from shared.attention.adapters import get_adapter
from shared.models.loading import load_model, load_tokenizer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_model_and_tokenizer(model_name: str, device: str):
    """Load model+tokenizer with eager attention and training-ready pad token."""
    model_spec = THEORY_MODELS[model_name]
    try:
        load_model.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        load_tokenizer.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        loaded = load_model(model_spec, attn_implementation="eager")
    except Exception:
        loaded = load_model(model_spec)
    model = loaded.model.to(device)
    _force_eager_attention(model)
    tokenizer = load_tokenizer(model_spec)
    _prepare_tokenizer_for_training(tokenizer)
    return model, tokenizer


def _ensure_baseline_head_groups(
    *,
    model_name: str,
    model,
    tokenizer,
    device: str,
    num_sequences: int = 24,
    seq_len: int = 256,
) -> tuple[list[HeadIndex], list[HeadIndex]]:
    """Load baseline SI head groups or compute/cache them on demand.

    Experiment 6 originally relied on precomputed Theory-1 head groups for 7B
    models only.  For prototype models (e.g., GPT-2 small), generate compatible
    head groups from cached profiling sequences and persist them under the
    canonical Theory-1 results path so downstream utilities can reuse them.
    """
    try:
        high, low = _load_baseline_head_groups(model_name)
        if high and low:
            return high, low
    except Exception:
        pass

    model_spec = THEORY_MODELS[model_name]
    seqs = load_profile_sequences(
        tokenizer=tokenizer,
        model_name=model_name,
        num_sequences=max(8, int(num_sequences)),
        seq_len=max(128, int(seq_len)),
    )
    if not seqs:
        raise RuntimeError(f"No profiling sequences available to build head groups for {model_name}.")

    adapter = get_adapter(model_spec)
    adapter.register(model)
    try:
        r2_df = compute_per_head_r2(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            model_spec=model_spec,
            device=device,
            sequences=seqs,
        )
    finally:
        try:
            adapter.cleanup()
        except Exception:
            pass

    high_si, low_si, mean_r2 = classify_heads(r2_df)
    high = [HeadIndex(layer=int(h.layer), head=int(h.head)) for h in high_si]
    low = [HeadIndex(layer=int(h.layer), head=int(h.head)) for h in low_si]
    if not high or not low:
        raise RuntimeError(f"Failed to derive non-empty head groups for {model_name}.")

    out_dir = ensure_dir(Path("results/experiment3/theory1_si_circuits") / model_name)
    write_json(
        out_dir / "head_groups.json",
        {
            "timestamp": now_timestamp(),
            "source": "experiment6_autogen",
            "high_si": [{"layer": int(h.layer), "head": int(h.head)} for h in high],
            "low_si": [{"layer": int(h.layer), "head": int(h.head)} for h in low],
        },
    )
    write_parquet(out_dir / "per_sequence_r2.parquet", r2_df)
    write_parquet(out_dir / "head_r2_summary.parquet", mean_r2)
    return high, low


def _build_separate_training_data(
    tokenizer,
    model_name: str,
    per_task: int,
    control_fraction: float,
    seq_len: int,
    seed: int,
    max_length: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return (math_encoded, control_encoded) as separate lists."""
    math_texts = build_math_training_texts(per_task=per_task, seed=seed)
    n_control = max(1, int(round((control_fraction / max(1e-8, 1.0 - control_fraction)) * len(math_texts))))
    control_texts = load_control_texts(
        tokenizer=tokenizer, model_name=model_name,
        count=n_control, seq_len=seq_len, seed=seed,
    )
    math_enc = _encode_texts(tokenizer, math_texts, max_length=max_length)
    ctrl_enc = _encode_texts(tokenizer, control_texts, max_length=max_length)
    return math_enc, ctrl_enc


def _make_batch(
    encoded: list[list[int]], *, start_idx: int, batch_size: int, pad_id: int, device: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build a padded batch from encoded sequences starting at start_idx (wrapping)."""
    n = len(encoded)
    rows: list[list[int]] = []
    idx = start_idx
    for _ in range(batch_size):
        rows.append(encoded[idx % n])
        idx += 1
    max_len = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        seq = torch.tensor(row, dtype=torch.long, device=device)
        input_ids[i, : len(row)] = seq
        attn_mask[i, : len(row)] = 1
    return input_ids, attn_mask, idx


def _classify_lora_params(model: nn.Module, si_map: dict[int, set[int]]) -> dict[str, list[nn.Parameter]]:
    """Classify LoRA parameters into si_attn, non_si_attn, and mlp groups."""
    import re
    layer_re = re.compile(r"model\.layers\.(\d+)\.")
    attn_keywords = ("self_attn",)
    mlp_keywords = ("mlp",)

    groups: dict[str, list[nn.Parameter]] = {"si_attn": [], "non_si_attn": [], "mlp": [], "other": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_a" not in name and "lora_b" not in name:
            continue
        m = layer_re.search(name)
        layer_idx = int(m.group(1)) if m else -1
        is_attn = any(kw in name for kw in attn_keywords)
        is_mlp = any(kw in name for kw in mlp_keywords)

        if is_attn:
            si_heads = si_map.get(layer_idx, set())
            if si_heads:
                # In layers with SI heads, both main and si adapters exist.
                # SI adapter params have "lora_a_si" or "lora_b_si" in name.
                if "_si" in name.split("lora_a")[-1] or "_si" in name.split("lora_b")[-1]:
                    groups["si_attn"].append(param)
                elif "_main" in name:
                    groups["non_si_attn"].append(param)
                else:
                    # Single LoRA (no composite) — treat as SI if layer has SI heads.
                    groups["si_attn"].append(param)
            else:
                groups["non_si_attn"].append(param)
        elif is_mlp:
            groups["mlp"].append(param)
        else:
            groups["other"].append(param)

    return groups


def _zero_grad_group(params: list[nn.Parameter]) -> None:
    """Zero gradients for a list of parameters."""
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


# ---------------------------------------------------------------------------
# Position-invariance evaluation (shared across all approaches)
# ---------------------------------------------------------------------------


def _build_positioned_math_prompt(
    problem_text: str, position_slot: int, tokenizer, filler_text: str,
) -> str:
    """Create a prompt where the math problem appears at approximately the
    given token position by prepending filler text."""
    if position_slot <= 0:
        return problem_text
    # Estimate filler length in tokens and trim to target position.
    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    n_filler = max(0, position_slot - 2)  # leave room for separator
    if n_filler > len(filler_ids):
        # Repeat filler if necessary.
        repeats = (n_filler // max(1, len(filler_ids))) + 1
        filler_ids = (filler_ids * repeats)[:n_filler]
    else:
        filler_ids = filler_ids[:n_filler]
    prefix = tokenizer.decode(filler_ids, skip_special_tokens=True)
    return prefix.strip() + "\n\n" + problem_text


def evaluate_position_invariance(
    *,
    model,
    tokenizer,
    device: str,
    eval_battery: dict[str, list[MCExample]],
    position_slots: tuple[int, ...],
    filler_text: str,
) -> dict[str, Any]:
    """Evaluate math accuracy at multiple context positions."""
    position_rows: list[dict[str, Any]] = []
    for slot in position_slots:
        correct = 0
        total = 0
        for task, examples in eval_battery.items():
            for ex in examples:
                prompt = _build_positioned_math_prompt(ex.prompt, slot, tokenizer, filler_text)
                scores = [
                    _score_option_logprob(model, tokenizer, device, prompt, opt)
                    for opt in ex.options
                ]
                pred = int(np.argmax(scores))
                correct += int(pred == int(ex.correct_index))
                total += 1
        acc = float(correct / max(1, total))
        position_rows.append({
            "position_slot": int(slot),
            "accuracy": acc,
            "n_total": int(total),
            "correct": int(correct),
        })
    # Compute flatness: std of accuracy across positions (lower = more position-invariant).
    accs = [r["accuracy"] for r in position_rows]
    return {
        "position_rows": position_rows,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "max_minus_min": float(max(accs) - min(accs)) if accs else float("nan"),
    }


# ---------------------------------------------------------------------------
# 6A: Gradient Routing
# ---------------------------------------------------------------------------


def _train_gradient_routed(
    *,
    model_name: str,
    seed: int,
    device: str,
    output_dir: Path,
    train_cfg: TrainConfig,
    lora_hp: LoRAHyperParams,
) -> tuple[Any, Any, dict[str, Any]]:
    """Train with gradient routing: math loss gradients only flow through
    SI attention LoRA parameters.  Control loss gradients flow through all
    parameters normally."""

    set_global_seed(seed)
    model, tokenizer = _load_model_and_tokenizer(model_name, device)
    model.train()

    high_heads, _ = _ensure_baseline_head_groups(
        model_name=model_name, model=model, tokenizer=tokenizer, device=device
    )
    si_map = _si_map_from_heads(high_heads)

    # Apply split LoRA (condition a) with equal SI/non-SI attention rank so
    # gradients can be routed at channel granularity rather than only by layer.
    routing_hp = LoRAHyperParams(
        rank=int(lora_hp.rank),
        alpha=float(lora_hp.alpha),
        dropout=float(lora_hp.dropout),
        si_rank=int(max(1, lora_hp.rank)),
        si_amplified_rank=int(lora_hp.si_amplified_rank),
        non_si_reduced_rank=int(lora_hp.non_si_reduced_rank),
    )
    apply_info = apply_lora_policy(
        model, condition="a_si_protecting_lora",
        si_heads_by_layer=si_map, hp=routing_hp,
    )

    # Classify parameters into SI-attn, non-SI-attn, MLP groups.
    param_groups = _classify_lora_params(model, si_map)
    non_si_params = param_groups["non_si_attn"] + param_groups["mlp"] + param_groups["other"]

    math_enc, ctrl_enc = _build_separate_training_data(
        tokenizer, model_name,
        per_task=train_cfg.math_per_task,
        control_fraction=train_cfg.control_fraction,
        seq_len=train_cfg.max_length, seed=seed,
        max_length=train_cfg.max_length,
    )
    if not math_enc:
        raise RuntimeError("No math training samples.")
    if not ctrl_enc:
        raise RuntimeError("No control training samples.")

    pad_id = int(tokenizer.pad_token_id)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters.")
    opt = AdamW(trainable, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    num_steps = max(1, int(train_cfg.max_steps))
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=min(int(train_cfg.warmup_steps), num_steps - 1),
        num_training_steps=num_steps,
    )

    losses: list[dict[str, Any]] = []
    math_idx, ctrl_idx = 0, 0
    start = time.time()

    for step in range(1, num_steps + 1):
        opt.zero_grad(set_to_none=True)

        # --- Math forward/backward (SI-only gradients) ---
        m_ids, m_mask, math_idx = _make_batch(
            math_enc, start_idx=math_idx, batch_size=train_cfg.batch_size,
            pad_id=pad_id, device=device,
        )
        m_labels = m_ids.clone().masked_fill(m_mask == 0, -100)
        m_out = model(input_ids=m_ids, attention_mask=m_mask, labels=m_labels, use_cache=False)
        math_loss = m_out.loss
        math_loss.backward()

        # Zero gradients on non-SI params from the math loss.
        _zero_grad_group(non_si_params)

        # --- Control forward/backward (all gradients) ---
        c_ids, c_mask, ctrl_idx = _make_batch(
            ctrl_enc, start_idx=ctrl_idx, batch_size=train_cfg.batch_size,
            pad_id=pad_id, device=device,
        )
        c_labels = c_ids.clone().masked_fill(c_mask == 0, -100)
        c_out = model(input_ids=c_ids, attention_mask=c_mask, labels=c_labels, use_cache=False)
        ctrl_loss = c_out.loss
        ctrl_loss.backward()

        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, train_cfg.grad_clip)
        opt.step()
        scheduler.step()

        row = {
            "step": int(step),
            "math_loss": float(math_loss.detach().item()),
            "ctrl_loss": float(ctrl_loss.detach().item()),
        }
        losses.append(row)

        if step == 1 or step % max(1, train_cfg.log_every) == 0 or step == num_steps:
            elapsed = time.time() - start
            eta = (elapsed / max(step, 1)) * max(0, num_steps - step)
            print(
                f"[6A|{model_name}|seed={seed}] step={step}/{num_steps} "
                f"math={row['math_loss']:.4f} ctrl={row['ctrl_loss']:.4f} eta={format_eta(eta)}",
                flush=True,
            )

        del m_ids, m_mask, m_labels, m_out, math_loss
        del c_ids, c_mask, c_labels, c_out, ctrl_loss
        clear_cuda()

    elapsed = time.time() - start
    model.eval()

    adapter_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if ("lora_a" in k or "lora_b" in k)}
    ensure_dir(output_dir)
    torch.save(adapter_state, output_dir / "adapter_state.pt")
    write_parquet(output_dir / "train_loss_curve.parquet", losses)

    summary = {
        "timestamp": now_timestamp(),
        "model": model_name,
        "condition": "f_gradient_routed",
        "seed": int(seed),
        "elapsed_sec": float(elapsed),
        "lora": {
            "trainable_params": int(apply_info.trainable_params),
            "total_params": int(apply_info.total_params),
            "si_attn_params": sum(p.numel() for p in param_groups["si_attn"]),
            "non_si_attn_params": sum(p.numel() for p in param_groups["non_si_attn"]),
            "mlp_params": sum(p.numel() for p in param_groups["mlp"]),
        },
    }
    write_json(output_dir / "train_summary.json", summary)
    return model, tokenizer, summary


# ---------------------------------------------------------------------------
# 6B: Anti-Localization Loss
# ---------------------------------------------------------------------------


def _ablate_si_heads(
    attentions: tuple[torch.Tensor, ...],
    si_map: dict[int, set[int]],
) -> tuple[torch.Tensor, ...]:
    """Zero out SI head attention weights to probe non-SI residual capability."""
    ablated: list[torch.Tensor] = []
    for layer_idx, attn in enumerate(attentions):
        if layer_idx in si_map:
            heads = sorted(si_map[layer_idx])
            a = attn.clone()
            # Set SI head attention to uniform (effectively removing their contribution).
            for h in heads:
                a[:, h, :, :] = 0.0
            ablated.append(a)
        else:
            ablated.append(attn)
    return tuple(ablated)


def _train_anti_localization(
    *,
    model_name: str,
    seed: int,
    device: str,
    output_dir: Path,
    train_cfg: TrainConfig,
    lora_hp: LoRAHyperParams,
    lambda_anti_loc: float = 0.1,
    probe_interval: int = 5,
    probe_batch_size: int = 4,
) -> tuple[Any, Any, dict[str, Any]]:
    """Train with anti-localization loss: penalize residual math capability
    when SI heads are ablated."""

    set_global_seed(seed)
    model, tokenizer = _load_model_and_tokenizer(model_name, device)
    model.train()

    high_heads, _ = _ensure_baseline_head_groups(
        model_name=model_name, model=model, tokenizer=tokenizer, device=device
    )
    si_map = _si_map_from_heads(high_heads)

    apply_info = apply_lora_policy(
        model, condition="d_full_qlora_baseline",
        si_heads_by_layer=si_map, hp=lora_hp,
    )

    math_enc, ctrl_enc = _build_separate_training_data(
        tokenizer, model_name,
        per_task=train_cfg.math_per_task,
        control_fraction=train_cfg.control_fraction,
        seq_len=train_cfg.max_length, seed=seed,
        max_length=train_cfg.max_length,
    )
    all_enc = math_enc + ctrl_enc
    rng_local = random.Random(seed + 777)
    rng_local.shuffle(all_enc)

    if not all_enc:
        raise RuntimeError("No training samples.")

    pad_id = int(tokenizer.pad_token_id)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters.")
    opt = AdamW(trainable, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    num_steps = max(1, int(train_cfg.max_steps))
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=min(int(train_cfg.warmup_steps), num_steps - 1),
        num_training_steps=num_steps,
    )

    # Build a small held-out math probe set for anti-localization loss.
    probe_math = math_enc[: min(probe_batch_size * 4, len(math_enc))]

    losses: list[dict[str, Any]] = []
    data_idx = 0
    probe_idx = 0
    start = time.time()

    for step in range(1, num_steps + 1):
        opt.zero_grad(set_to_none=True)

        # Standard training loss.
        ids, mask, data_idx = _make_batch(
            all_enc, start_idx=data_idx, batch_size=train_cfg.batch_size,
            pad_id=pad_id, device=device,
        )
        labels = ids.clone().masked_fill(mask == 0, -100)
        out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
        total_loss = out.loss
        anti_loc_loss = torch.tensor(0.0, device=device)

        # Anti-localization probe: every probe_interval steps.
        if step % max(1, probe_interval) == 0 and probe_math:
            p_ids, p_mask, probe_idx = _make_batch(
                probe_math, start_idx=probe_idx, batch_size=min(probe_batch_size, len(probe_math)),
                pad_id=pad_id, device=device,
            )
            p_labels = p_ids.clone().masked_fill(p_mask == 0, -100)

            si_head_ids = [
                HeadIndex(layer=int(layer_idx), head=int(head_idx))
                for layer_idx, heads in si_map.items()
                for head_idx in sorted(heads)
            ]
            with head_output_ablation(model, si_head_ids):
                p_out = model(input_ids=p_ids, attention_mask=p_mask, labels=p_labels, use_cache=False)

            # The anti-localization loss: we want this ablated loss to be HIGH
            # (math should fail without SI heads).  So we MINIMIZE the negative
            # of the ablated loss, i.e., penalize low ablated loss.
            # L_anti = -ablated_loss  (we add lambda * L_anti to total, so it
            # tries to push ablated_loss up, meaning non-SI channels can't do math).
            anti_loc_loss = -p_out.loss
            total_loss = total_loss + (lambda_anti_loc * anti_loc_loss)

            del p_ids, p_mask, p_labels, p_out

        total_loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, train_cfg.grad_clip)
        opt.step()
        scheduler.step()

        row = {
            "step": int(step),
            "base_loss": float(out.loss.detach().item()),
            "anti_loc_loss": float(anti_loc_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }
        losses.append(row)

        if step == 1 or step % max(1, train_cfg.log_every) == 0 or step == num_steps:
            elapsed = time.time() - start
            eta = (elapsed / max(step, 1)) * max(0, num_steps - step)
            print(
                f"[6B|{model_name}|seed={seed}] step={step}/{num_steps} "
                f"loss={row['total_loss']:.4f} anti_loc={row['anti_loc_loss']:.4f} eta={format_eta(eta)}",
                flush=True,
            )

        del ids, mask, labels, out, total_loss, anti_loc_loss
        clear_cuda()

    elapsed = time.time() - start
    model.eval()

    adapter_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if ("lora_a" in k or "lora_b" in k)}
    ensure_dir(output_dir)
    torch.save(adapter_state, output_dir / "adapter_state.pt")
    write_parquet(output_dir / "train_loss_curve.parquet", losses)

    summary = {
        "timestamp": now_timestamp(),
        "model": model_name,
        "condition": "g_anti_localization",
        "seed": int(seed),
        "elapsed_sec": float(elapsed),
        "lambda_anti_loc": float(lambda_anti_loc),
        "probe_interval": int(probe_interval),
        "lora": {
            "trainable_params": int(apply_info.trainable_params),
            "total_params": int(apply_info.total_params),
        },
    }
    write_json(output_dir / "train_summary.json", summary)
    return model, tokenizer, summary


# ---------------------------------------------------------------------------
# 6C: Two-Phase Distillation
# ---------------------------------------------------------------------------


def _train_baseline_teacher(
    *,
    model_name: str,
    seed: int,
    device: str,
    output_dir: Path,
    train_cfg: TrainConfig,
    lora_hp: LoRAHyperParams,
) -> tuple[Any, Any, dict[str, Any]]:
    """Phase 1: standard full-capacity math FT (same as condition d)."""
    from experiment4.pipeline import _train_single_run
    # Ensure baseline head groups exist for prototype models before delegating to
    # experiment4's trainer (which expects canonical Theory-1 artifacts).
    tmp_model, tmp_tokenizer = _load_model_and_tokenizer(model_name, device)
    try:
        _ensure_baseline_head_groups(
            model_name=model_name,
            model=tmp_model,
            tokenizer=tmp_tokenizer,
            device=device,
        )
    finally:
        del tmp_model
        clear_cuda()
    return _train_single_run(
        model_name=model_name,
        condition="d_full_qlora_baseline",
        seed=seed,
        device=device,
        output_dir=output_dir,
        train_cfg=train_cfg,
        lora_hp=lora_hp,
        aux_cfg=AuxLossConfig(mode="none", lambda_preserve=0.0, lambda_routing=0.0),
    )


def _train_si_student(
    *,
    teacher_model,
    model_name: str,
    seed: int,
    device: str,
    output_dir: Path,
    train_cfg: TrainConfig,
    lora_hp: LoRAHyperParams,
    temperature: float = 2.0,
    alpha_distill: float = 0.7,
) -> tuple[Any, Any, dict[str, Any]]:
    """Phase 2: train a student that only modifies SI head parameters to match
    the teacher's math output distribution."""

    set_global_seed(seed + 5000)
    model, tokenizer = _load_model_and_tokenizer(model_name, device)
    model.train()
    teacher_model.eval()

    high_heads, _ = _ensure_baseline_head_groups(
        model_name=model_name, model=model, tokenizer=tokenizer, device=device
    )
    si_map = _si_map_from_heads(high_heads)

    # Student only gets LoRA on SI heads (like condition c).
    apply_info = apply_lora_policy(
        model, condition="c_si_only_lora",
        si_heads_by_layer=si_map, hp=lora_hp,
    )

    math_enc, ctrl_enc = _build_separate_training_data(
        tokenizer, model_name,
        per_task=train_cfg.math_per_task,
        control_fraction=0.0,  # Student only trains on math.
        seq_len=train_cfg.max_length, seed=seed + 5000,
        max_length=train_cfg.max_length,
    )
    if not math_enc:
        raise RuntimeError("No math samples for student.")

    pad_id = int(tokenizer.pad_token_id)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable student parameters.")
    opt = AdamW(trainable, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    num_steps = max(1, int(train_cfg.max_steps))
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=min(int(train_cfg.warmup_steps), num_steps - 1),
        num_training_steps=num_steps,
    )

    losses: list[dict[str, Any]] = []
    data_idx = 0
    start = time.time()

    for step in range(1, num_steps + 1):
        opt.zero_grad(set_to_none=True)

        ids, mask, data_idx = _make_batch(
            math_enc, start_idx=data_idx, batch_size=train_cfg.batch_size,
            pad_id=pad_id, device=device,
        )
        labels = ids.clone().masked_fill(mask == 0, -100)

        # Student forward.
        s_out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)

        # Teacher forward (no grad).
        with torch.no_grad():
            t_out = teacher_model(input_ids=ids, attention_mask=mask, use_cache=False)

        # Distillation loss: KL(student_soft || teacher_soft) + hard label loss.
        s_logits = s_out.logits / temperature
        t_logits = t_out.logits / temperature

        kl_loss = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        hard_loss = s_out.loss
        total_loss = alpha_distill * kl_loss + (1.0 - alpha_distill) * hard_loss

        total_loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, train_cfg.grad_clip)
        opt.step()
        scheduler.step()

        row = {
            "step": int(step),
            "kl_loss": float(kl_loss.detach().item()),
            "hard_loss": float(hard_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }
        losses.append(row)

        if step == 1 or step % max(1, train_cfg.log_every) == 0 or step == num_steps:
            elapsed = time.time() - start
            eta = (elapsed / max(step, 1)) * max(0, num_steps - step)
            print(
                f"[6C-student|{model_name}|seed={seed}] step={step}/{num_steps} "
                f"kl={row['kl_loss']:.4f} hard={row['hard_loss']:.4f} eta={format_eta(eta)}",
                flush=True,
            )

        del ids, mask, labels, s_out, t_out, kl_loss, hard_loss, total_loss
        clear_cuda()

    elapsed = time.time() - start
    model.eval()

    adapter_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if ("lora_a" in k or "lora_b" in k)}
    ensure_dir(output_dir)
    torch.save(adapter_state, output_dir / "adapter_state.pt")
    write_parquet(output_dir / "train_loss_curve.parquet", losses)

    summary = {
        "timestamp": now_timestamp(),
        "model": model_name,
        "condition": "h_si_distillation_student",
        "seed": int(seed),
        "elapsed_sec": float(elapsed),
        "temperature": float(temperature),
        "alpha_distill": float(alpha_distill),
        "lora": {
            "trainable_params": int(apply_info.trainable_params),
            "total_params": int(apply_info.total_params),
        },
    }
    write_json(output_dir / "train_summary.json", summary)
    return model, tokenizer, summary


# ---------------------------------------------------------------------------
# 6D: Contrastive Channel Assignment
# ---------------------------------------------------------------------------


def _compute_channel_variance(
    attentions: tuple[torch.Tensor, ...],
    si_map: dict[int, set[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean activation variance for SI and non-SI heads."""
    si_vars: list[torch.Tensor] = []
    non_si_vars: list[torch.Tensor] = []
    for layer_idx, attn in enumerate(attentions):
        n_heads = attn.shape[1]
        si_heads = si_map.get(layer_idx, set())
        for h in range(n_heads):
            head_attn = attn[:, h, :, :]  # [batch, q, k]
            var = head_attn.var()
            if h in si_heads:
                si_vars.append(var)
            else:
                non_si_vars.append(var)

    dev = attentions[0].device if attentions else torch.device("cpu")
    si_var = torch.stack(si_vars).mean() if si_vars else torch.tensor(0.0, device=dev)
    non_si_var = torch.stack(non_si_vars).mean() if non_si_vars else torch.tensor(0.0, device=dev)
    return si_var, non_si_var


def _train_contrastive_channel(
    *,
    model_name: str,
    seed: int,
    device: str,
    output_dir: Path,
    train_cfg: TrainConfig,
    lora_hp: LoRAHyperParams,
    lambda_contrast: float = 0.05,
    contrast_interval: int = 5,
) -> tuple[Any, Any, dict[str, Any]]:
    """Train with contrastive channel assignment: for math inputs, encourage
    high SI variance and low non-SI variance."""

    set_global_seed(seed)
    model, tokenizer = _load_model_and_tokenizer(model_name, device)
    model.train()

    high_heads, _ = _ensure_baseline_head_groups(
        model_name=model_name, model=model, tokenizer=tokenizer, device=device
    )
    si_map = _si_map_from_heads(high_heads)

    apply_info = apply_lora_policy(
        model, condition="d_full_qlora_baseline",
        si_heads_by_layer=si_map, hp=lora_hp,
    )

    math_enc, ctrl_enc = _build_separate_training_data(
        tokenizer, model_name,
        per_task=train_cfg.math_per_task,
        control_fraction=train_cfg.control_fraction,
        seq_len=train_cfg.max_length, seed=seed,
        max_length=train_cfg.max_length,
    )
    all_enc = math_enc + ctrl_enc
    rng_local = random.Random(seed + 333)
    rng_local.shuffle(all_enc)
    if not all_enc:
        raise RuntimeError("No training samples.")

    pad_id = int(tokenizer.pad_token_id)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    num_steps = max(1, int(train_cfg.max_steps))
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=min(int(train_cfg.warmup_steps), num_steps - 1),
        num_training_steps=num_steps,
    )

    losses: list[dict[str, Any]] = []
    data_idx, math_idx = 0, 0
    start = time.time()

    for step in range(1, num_steps + 1):
        opt.zero_grad(set_to_none=True)

        # Standard training on mixed data.
        ids, mask, data_idx = _make_batch(
            all_enc, start_idx=data_idx, batch_size=train_cfg.batch_size,
            pad_id=pad_id, device=device,
        )
        labels = ids.clone().masked_fill(mask == 0, -100)
        out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False,
                     output_attentions=(step % max(1, contrast_interval) == 0))
        total_loss = out.loss
        contrast_loss = torch.tensor(0.0, device=device)

        # Contrastive channel loss on math batch.
        if step % max(1, contrast_interval) == 0 and math_enc and out.attentions is not None:
            m_ids, m_mask, math_idx = _make_batch(
                math_enc, start_idx=math_idx, batch_size=train_cfg.batch_size,
                pad_id=pad_id, device=device,
            )
            m_labels = m_ids.clone().masked_fill(m_mask == 0, -100)
            m_out = model(input_ids=m_ids, attention_mask=m_mask, labels=m_labels,
                          use_cache=False, output_attentions=True)

            if m_out.attentions is not None:
                si_var, non_si_var = _compute_channel_variance(m_out.attentions, si_map)
                # Minimize: non_si_var - si_var  (push non-SI variance down, SI up)
                contrast_loss = non_si_var - si_var
                total_loss = total_loss + (lambda_contrast * contrast_loss)

            del m_ids, m_mask, m_labels, m_out

        total_loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, train_cfg.grad_clip)
        opt.step()
        scheduler.step()

        row = {
            "step": int(step),
            "base_loss": float(out.loss.detach().item()),
            "contrast_loss": float(contrast_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }
        losses.append(row)

        if step == 1 or step % max(1, train_cfg.log_every) == 0 or step == num_steps:
            elapsed = time.time() - start
            eta = (elapsed / max(step, 1)) * max(0, num_steps - step)
            print(
                f"[6D|{model_name}|seed={seed}] step={step}/{num_steps} "
                f"loss={row['total_loss']:.4f} contrast={row['contrast_loss']:.4f} eta={format_eta(eta)}",
                flush=True,
            )

        del ids, mask, labels, out, total_loss, contrast_loss
        clear_cuda()

    elapsed = time.time() - start
    model.eval()

    adapter_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if ("lora_a" in k or "lora_b" in k)}
    ensure_dir(output_dir)
    torch.save(adapter_state, output_dir / "adapter_state.pt")
    write_parquet(output_dir / "train_loss_curve.parquet", losses)

    summary = {
        "timestamp": now_timestamp(),
        "model": model_name,
        "condition": "i_contrastive_channel",
        "seed": int(seed),
        "elapsed_sec": float(elapsed),
        "lambda_contrast": float(lambda_contrast),
        "contrast_interval": int(contrast_interval),
        "lora": {
            "trainable_params": int(apply_info.trainable_params),
            "total_params": int(apply_info.total_params),
        },
    }
    write_json(output_dir / "train_summary.json", summary)
    return model, tokenizer, summary


# ---------------------------------------------------------------------------
# 6E: SI-Optimized Tokenizer Design
# ---------------------------------------------------------------------------


def _insert_math_boundaries(text: str, operators: tuple[str, ...]) -> str:
    """Insert spaces around math operators to force the tokenizer to create
    word boundaries at math-critical positions, aligning them with SI head
    attention patterns."""
    result = text
    for op in operators:
        if op in result:
            # Ensure space before and after the operator.
            result = result.replace(op, f" {op} ")
    # Collapse multiple spaces.
    while "  " in result:
        result = result.replace("  ", " ")
    return result.strip()


def evaluate_tokenizer_formatting(
    *,
    model,
    tokenizer,
    device: str,
    eval_battery: dict[str, list[MCExample]],
    operators: tuple[str, ...],
) -> dict[str, Any]:
    """Compare math accuracy with standard vs SI-optimized formatting."""
    results: dict[str, dict[str, Any]] = {}

    for fmt_name, transform in [
        ("standard", lambda x: x),
        ("si_optimized", lambda x: _insert_math_boundaries(x, operators)),
    ]:
        correct = 0
        total = 0
        for task, examples in eval_battery.items():
            for ex in examples:
                prompt = transform(ex.prompt)
                options = [transform(opt) for opt in ex.options]
                scores = [
                    _score_option_logprob(model, tokenizer, device, prompt, opt)
                    for opt in options
                ]
                pred = int(np.argmax(scores))
                correct += int(pred == int(ex.correct_index))
                total += 1
        acc = float(correct / max(1, total))
        results[fmt_name] = {"accuracy": acc, "n_total": int(total), "correct": int(correct)}

    std_acc = results["standard"]["accuracy"]
    opt_acc = results["si_optimized"]["accuracy"]
    return {
        "standard": results["standard"],
        "si_optimized": results["si_optimized"],
        "delta_accuracy": float(opt_acc - std_acc),
    }


# ---------------------------------------------------------------------------
# Top-level runners (one per sub-experiment)
# ---------------------------------------------------------------------------


def _standard_eval_suite(
    *,
    model,
    tokenizer,
    model_name: str,
    device: str,
    seed: int,
    output_dir: Path,
    filler_text: str,
) -> dict[str, Any]:
    """Run the shared evaluation suite: math accuracy, wiki perplexity,
    position-invariance, and SI-optimized tokenizer formatting."""
    eval_battery = build_math_eval_battery(count_per_task=64, seed=seed + 10000)

    math_eval = evaluate_math_accuracy(
        model=model, tokenizer=tokenizer, device=device, eval_battery=eval_battery,
    )
    wiki_eval = evaluate_wiki_perplexity(
        model=model, tokenizer=tokenizer, model_name=model_name,
        device=device, count=24, seq_len=256,
    )
    pos_eval = evaluate_position_invariance(
        model=model, tokenizer=tokenizer, device=device,
        eval_battery=eval_battery,
        position_slots=POSITION_EVAL_SLOTS,
        filler_text=filler_text,
    )
    tok_eval = evaluate_tokenizer_formatting(
        model=model, tokenizer=tokenizer, device=device,
        eval_battery=eval_battery,
        operators=TOKENIZER_MATH_OPERATORS,
    )

    write_json(output_dir / "math_eval.json", math_eval)
    write_json(output_dir / "wiki_eval.json", wiki_eval)
    write_json(output_dir / "position_invariance.json", pos_eval)
    write_json(output_dir / "tokenizer_formatting.json", tok_eval)

    return {
        "math_overall_accuracy": safe_float(math_eval.get("overall_accuracy")),
        "wiki_perplexity": safe_float(wiki_eval.get("perplexity")),
        "position_std": safe_float(pos_eval.get("std_accuracy")),
        "position_range": safe_float(pos_eval.get("max_minus_min")),
        "tokenizer_delta": safe_float(tok_eval.get("delta_accuracy")),
    }


def run_6a(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    seeds: list[int],
    train_cfg: TrainConfig,
    lora_hp: LoRAHyperParams,
) -> dict[str, Any]:
    """Run Experiment 6A: gradient routing vs baseline."""
    from experiment4.pipeline import _train_single_run

    model_root = ensure_dir(output_root / model_name)
    filler_seqs = load_profile_sequences(
        tokenizer=load_tokenizer(THEORY_MODELS[model_name]),
        model_name=model_name, num_sequences=1, seq_len=512,
    )
    filler_text = load_tokenizer(THEORY_MODELS[model_name]).decode(
        filler_seqs[0] if filler_seqs else [], skip_special_tokens=True,
    )

    rows: list[dict[str, Any]] = []

    for seed in seeds:
        # Gradient-routed condition.
        run_dir = ensure_dir(model_root / "runs" / "f_gradient_routed" / f"seed_{seed}")
        model, tokenizer, summary = _train_gradient_routed(
            model_name=model_name, seed=seed, device=device,
            output_dir=run_dir, train_cfg=train_cfg, lora_hp=lora_hp,
        )
        evals = _standard_eval_suite(
            model=model, tokenizer=tokenizer, model_name=model_name,
            device=device, seed=seed, output_dir=run_dir, filler_text=filler_text,
        )
        rows.append({"condition": "f_gradient_routed", "seed": int(seed), **evals})
        del model
        clear_cuda()

        # Baseline condition (d).
        run_dir = ensure_dir(model_root / "runs" / "d_full_qlora_baseline" / f"seed_{seed}")
        model, tokenizer, summary = _train_single_run(
            model_name=model_name, condition="d_full_qlora_baseline",
            seed=seed, device=device, output_dir=run_dir,
            train_cfg=train_cfg, lora_hp=lora_hp,
            aux_cfg=AuxLossConfig(mode="none", lambda_preserve=0.0, lambda_routing=0.0),
        )
        evals = _standard_eval_suite(
            model=model, tokenizer=tokenizer, model_name=model_name,
            device=device, seed=seed, output_dir=run_dir, filler_text=filler_text,
        )
        rows.append({"condition": "d_full_qlora_baseline", "seed": int(seed), **evals})
        del model
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_parquet(model_root / "exp6a_runs.parquet", index=False)

    summary = _build_comparison_summary(df, "f_gradient_routed", "d_full_qlora_baseline", seeds)
    summary["experiment"] = "6A"
    summary["model"] = model_name
    write_json(model_root / "exp6a_comparison.json", summary)
    return summary


def run_6b(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    seeds: list[int],
    train_cfg: TrainConfig,
    lora_hp: LoRAHyperParams,
    lambda_anti_loc: float = 0.1,
) -> dict[str, Any]:
    """Run Experiment 6B: anti-localization loss vs baseline."""
    from experiment4.pipeline import _train_single_run

    model_root = ensure_dir(output_root / model_name)
    filler_seqs = load_profile_sequences(
        tokenizer=load_tokenizer(THEORY_MODELS[model_name]),
        model_name=model_name, num_sequences=1, seq_len=512,
    )
    filler_text = load_tokenizer(THEORY_MODELS[model_name]).decode(
        filler_seqs[0] if filler_seqs else [], skip_special_tokens=True,
    )

    rows: list[dict[str, Any]] = []

    for seed in seeds:
        run_dir = ensure_dir(model_root / "runs" / "g_anti_localization" / f"seed_{seed}")
        model, tokenizer, summary = _train_anti_localization(
            model_name=model_name, seed=seed, device=device,
            output_dir=run_dir, train_cfg=train_cfg, lora_hp=lora_hp,
            lambda_anti_loc=lambda_anti_loc,
        )
        evals = _standard_eval_suite(
            model=model, tokenizer=tokenizer, model_name=model_name,
            device=device, seed=seed, output_dir=run_dir, filler_text=filler_text,
        )
        rows.append({"condition": "g_anti_localization", "seed": int(seed), **evals})
        del model
        clear_cuda()

        run_dir = ensure_dir(model_root / "runs" / "d_full_qlora_baseline" / f"seed_{seed}")
        model, tokenizer, summary = _train_single_run(
            model_name=model_name, condition="d_full_qlora_baseline",
            seed=seed, device=device, output_dir=run_dir,
            train_cfg=train_cfg, lora_hp=lora_hp,
            aux_cfg=AuxLossConfig(mode="none", lambda_preserve=0.0, lambda_routing=0.0),
        )
        evals = _standard_eval_suite(
            model=model, tokenizer=tokenizer, model_name=model_name,
            device=device, seed=seed, output_dir=run_dir, filler_text=filler_text,
        )
        rows.append({"condition": "d_full_qlora_baseline", "seed": int(seed), **evals})
        del model
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_parquet(model_root / "exp6b_runs.parquet", index=False)

    summary = _build_comparison_summary(df, "g_anti_localization", "d_full_qlora_baseline", seeds)
    summary["experiment"] = "6B"
    summary["model"] = model_name
    write_json(model_root / "exp6b_comparison.json", summary)
    return summary


def run_6c(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    seeds: list[int],
    train_cfg: TrainConfig,
    lora_hp: LoRAHyperParams,
) -> dict[str, Any]:
    """Run Experiment 6C: two-phase distillation."""
    model_root = ensure_dir(output_root / model_name)
    filler_seqs = load_profile_sequences(
        tokenizer=load_tokenizer(THEORY_MODELS[model_name]),
        model_name=model_name, num_sequences=1, seq_len=512,
    )
    filler_text = load_tokenizer(THEORY_MODELS[model_name]).decode(
        filler_seqs[0] if filler_seqs else [], skip_special_tokens=True,
    )

    rows: list[dict[str, Any]] = []

    for seed in seeds:
        # Phase 1: train teacher.
        teacher_dir = ensure_dir(model_root / "runs" / "h_teacher" / f"seed_{seed}")
        teacher_model, tokenizer, teacher_summary = _train_baseline_teacher(
            model_name=model_name, seed=seed, device=device,
            output_dir=teacher_dir, train_cfg=train_cfg, lora_hp=lora_hp,
        )
        evals_teacher = _standard_eval_suite(
            model=teacher_model, tokenizer=tokenizer, model_name=model_name,
            device=device, seed=seed, output_dir=teacher_dir, filler_text=filler_text,
        )
        rows.append({"condition": "h_teacher_baseline", "seed": int(seed), **evals_teacher})

        # Phase 2: distill into SI-only student.
        student_dir = ensure_dir(model_root / "runs" / "h_si_student" / f"seed_{seed}")
        student_model, _, student_summary = _train_si_student(
            teacher_model=teacher_model,
            model_name=model_name, seed=seed, device=device,
            output_dir=student_dir, train_cfg=train_cfg, lora_hp=lora_hp,
        )
        evals_student = _standard_eval_suite(
            model=student_model, tokenizer=tokenizer, model_name=model_name,
            device=device, seed=seed, output_dir=student_dir, filler_text=filler_text,
        )
        rows.append({"condition": "h_si_student", "seed": int(seed), **evals_student})

        del teacher_model, student_model
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_parquet(model_root / "exp6c_runs.parquet", index=False)

    summary = _build_comparison_summary(df, "h_si_student", "h_teacher_baseline", seeds)
    summary["experiment"] = "6C"
    summary["model"] = model_name
    write_json(model_root / "exp6c_comparison.json", summary)
    return summary


def run_6d(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    seeds: list[int],
    train_cfg: TrainConfig,
    lora_hp: LoRAHyperParams,
    lambda_contrast: float = 0.05,
) -> dict[str, Any]:
    """Run Experiment 6D: contrastive channel assignment vs baseline."""
    from experiment4.pipeline import _train_single_run

    model_root = ensure_dir(output_root / model_name)
    filler_seqs = load_profile_sequences(
        tokenizer=load_tokenizer(THEORY_MODELS[model_name]),
        model_name=model_name, num_sequences=1, seq_len=512,
    )
    filler_text = load_tokenizer(THEORY_MODELS[model_name]).decode(
        filler_seqs[0] if filler_seqs else [], skip_special_tokens=True,
    )

    rows: list[dict[str, Any]] = []

    for seed in seeds:
        run_dir = ensure_dir(model_root / "runs" / "i_contrastive_channel" / f"seed_{seed}")
        model, tokenizer, summary = _train_contrastive_channel(
            model_name=model_name, seed=seed, device=device,
            output_dir=run_dir, train_cfg=train_cfg, lora_hp=lora_hp,
            lambda_contrast=lambda_contrast,
        )
        evals = _standard_eval_suite(
            model=model, tokenizer=tokenizer, model_name=model_name,
            device=device, seed=seed, output_dir=run_dir, filler_text=filler_text,
        )
        rows.append({"condition": "i_contrastive_channel", "seed": int(seed), **evals})
        del model
        clear_cuda()

        run_dir = ensure_dir(model_root / "runs" / "d_full_qlora_baseline" / f"seed_{seed}")
        model, tokenizer, summary = _train_single_run(
            model_name=model_name, condition="d_full_qlora_baseline",
            seed=seed, device=device, output_dir=run_dir,
            train_cfg=train_cfg, lora_hp=lora_hp,
            aux_cfg=AuxLossConfig(mode="none", lambda_preserve=0.0, lambda_routing=0.0),
        )
        evals = _standard_eval_suite(
            model=model, tokenizer=tokenizer, model_name=model_name,
            device=device, seed=seed, output_dir=run_dir, filler_text=filler_text,
        )
        rows.append({"condition": "d_full_qlora_baseline", "seed": int(seed), **evals})
        del model
        clear_cuda()

    df = pd.DataFrame(rows)
    df.to_parquet(model_root / "exp6d_runs.parquet", index=False)

    summary = _build_comparison_summary(df, "i_contrastive_channel", "d_full_qlora_baseline", seeds)
    summary["experiment"] = "6D"
    summary["model"] = model_name
    write_json(model_root / "exp6d_comparison.json", summary)
    return summary


def run_6e(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    seed: int,
) -> dict[str, Any]:
    """Run Experiment 6E: SI-optimized tokenizer formatting (zero-shot, no training)."""
    model_root = ensure_dir(output_root / model_name)

    model, tokenizer = _load_model_and_tokenizer(model_name, device)
    model.eval()

    eval_battery = build_math_eval_battery(count_per_task=64, seed=seed + 10000)
    tok_eval = evaluate_tokenizer_formatting(
        model=model, tokenizer=tokenizer, device=device,
        eval_battery=eval_battery,
        operators=TOKENIZER_MATH_OPERATORS,
    )

    # Also run position-invariance with both formatting styles.
    filler_seqs = load_profile_sequences(
        tokenizer=tokenizer, model_name=model_name, num_sequences=1, seq_len=512,
    )
    filler_text = tokenizer.decode(filler_seqs[0] if filler_seqs else [], skip_special_tokens=True)

    pos_standard = evaluate_position_invariance(
        model=model, tokenizer=tokenizer, device=device,
        eval_battery=eval_battery,
        position_slots=POSITION_EVAL_SLOTS,
        filler_text=filler_text,
    )

    # Re-format the battery for SI-optimized evaluation.
    reformatted_battery: dict[str, list[MCExample]] = {}
    for task, examples in eval_battery.items():
        reformatted_battery[task] = [
            MCExample(
                example_id=ex.example_id,
                task=ex.task,
                prompt=_insert_math_boundaries(ex.prompt, TOKENIZER_MATH_OPERATORS),
                options=tuple(_insert_math_boundaries(o, TOKENIZER_MATH_OPERATORS) for o in ex.options),
                correct_index=ex.correct_index,
                metadata=ex.metadata,
            )
            for ex in examples
        ]

    pos_optimized = evaluate_position_invariance(
        model=model, tokenizer=tokenizer, device=device,
        eval_battery=reformatted_battery,
        position_slots=POSITION_EVAL_SLOTS,
        filler_text=filler_text,
    )

    del model
    clear_cuda()

    summary = {
        "timestamp": now_timestamp(),
        "experiment": "6E",
        "model": model_name,
        "seed": int(seed),
        "tokenizer_formatting": tok_eval,
        "position_invariance_standard": pos_standard,
        "position_invariance_si_optimized": pos_optimized,
        "delta_position_std": float(
            safe_float(pos_optimized.get("std_accuracy"))
            - safe_float(pos_standard.get("std_accuracy"))
        ),
    }
    write_json(model_root / "exp6e_tokenizer_comparison.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Shared comparison builder
# ---------------------------------------------------------------------------


def _build_comparison_summary(
    df: pd.DataFrame,
    cond_x: str,
    cond_y: str,
    seeds: list[int],
) -> dict[str, Any]:
    """Build paired comparison summary across multiple metrics."""
    summary: dict[str, Any] = {
        "timestamp": now_timestamp(),
        "conditions": [cond_x, cond_y],
        "seeds": [int(s) for s in seeds],
        "n_runs": int(len(df)),
    }

    for metric in ["math_overall_accuracy", "wiki_perplexity", "position_std", "position_range", "tokenizer_delta"]:
        x = df[df["condition"] == cond_x].set_index("seed")
        y = df[df["condition"] == cond_y].set_index("seed")
        common = sorted(set(x.index.tolist()) & set(y.index.tolist()))
        if not common or metric not in x.columns:
            continue
        x_vals = x.loc[common, metric].to_numpy(dtype=float)
        y_vals = y.loc[common, metric].to_numpy(dtype=float)
        diffs = x_vals - y_vals
        if len(diffs) < 2:
            continue
        t_stat, p_two = scipy_stats.ttest_1samp(diffs, popmean=0.0)
        mean_diff, ci = mean_ci95(diffs.tolist())
        summary[f"comparison_{metric}"] = {
            "n_pairs": int(len(common)),
            "mean_diff": float(mean_diff),
            "ci95": [float(ci[0]), float(ci[1])],
            "t_statistic": safe_float(t_stat),
            "p_two_sided": safe_float(p_two),
            f"{cond_x.split('_')[0]}_values": [float(v) for v in x_vals.tolist()],
            f"{cond_y.split('_')[0]}_values": [float(v) for v in y_vals.tolist()],
        }

    return summary
