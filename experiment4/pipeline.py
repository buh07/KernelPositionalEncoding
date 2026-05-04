from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from experiment2.tasks import build_token_pools, generate_task_examples
from experiment3.phase2.exp3p2b_trivial_feature_control import _synthetic_boundary_full
from experiment3.stats_utils import one_sided_p_from_two_sided
from experiment3.theory1_si_circuits import (
    MODELS as THEORY_MODELS,
    RETRIEVAL_SPANS,
    classify_heads,
    compute_per_head_r2,
    evaluate_task_battery,
    load_profile_sequences,
)
from experiment3.theory5b_boundary_detection import parse_head_list, run_attention_analysis
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
from experiment4.lora import LoRAApplyResult, LoRAHyperParams, apply_lora_policy, collect_trainable_parameter_names
from experiment4.math_data import MCExample, build_math_eval_battery, build_training_mix
from shared.attention.adapters import get_adapter
from shared.models.loading import load_model, load_tokenizer


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    max_length: int
    log_every: int
    aux_interval: int
    math_per_task: int
    control_fraction: float


@dataclass(frozen=True)
class AuxLossConfig:
    mode: str  # none | si_preserve | si_routing | combined
    lambda_preserve: float
    lambda_routing: float


@contextmanager
def _temporary_eval_mode(model):
    was_training = bool(model.training)
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def _load_baseline_head_groups(model_name: str) -> tuple[list[HeadIndex], list[HeadIndex]]:
    path = Path("results/experiment3/theory1_si_circuits") / model_name / "head_groups.json"
    data = read_json(path)
    high = [HeadIndex(layer=int(x["layer"]), head=int(x["head"])) for x in data.get("high_si", [])]
    low = [HeadIndex(layer=int(x["layer"]), head=int(x["head"])) for x in data.get("low_si", [])]
    return high, low


def _si_map_from_heads(heads: list[HeadIndex]) -> dict[int, set[int]]:
    by_layer: dict[int, set[int]] = {}
    for h in heads:
        by_layer.setdefault(int(h.layer), set()).add(int(h.head))
    return by_layer


def _prepare_tokenizer_for_training(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def _force_eager_attention(model) -> None:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        if hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", "eager")
        if hasattr(cfg, "attn_implementation"):
            setattr(cfg, "attn_implementation", "eager")
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and hasattr(gen_cfg, "attn_implementation"):
        setattr(gen_cfg, "attn_implementation", "eager")


def _normalize_approach_c_output(raw: Any) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Normalize Approach-C output from theory5b helpers.

    Some call-sites receive ``(approach_c_dict, score_df)`` while others expect a dict.
    This returns a stable ``(dict, optional_df)`` pair.
    """
    if isinstance(raw, tuple):
        if len(raw) >= 2 and isinstance(raw[0], dict):
            c_dict = raw[0]
            c_df = raw[1] if isinstance(raw[1], pd.DataFrame) else None
            return c_dict, c_df
        if len(raw) == 1 and isinstance(raw[0], dict):
            return raw[0], None
    if isinstance(raw, dict):
        return raw, None
    return {"error": f"Unexpected Approach-C payload type: {type(raw).__name__}"}, None


def _encode_texts(tokenizer, texts: list[str], max_length: int) -> list[list[int]]:
    out: list[list[int]] = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=True, truncation=True, max_length=max_length)
        if len(ids) < 4:
            continue
        out.append([int(x) for x in ids])
    return out


def _iter_batches(encoded: list[list[int]], *, batch_size: int, pad_id: int, device: str):
    if not encoded:
        raise RuntimeError("No encoded training rows available")
    idx = 0
    n = len(encoded)
    while True:
        batch_rows: list[list[int]] = []
        for _ in range(batch_size):
            row = encoded[idx % n]
            batch_rows.append(row)
            idx += 1
        max_len = max(len(x) for x in batch_rows)
        input_ids = torch.full((len(batch_rows), max_len), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((len(batch_rows), max_len), dtype=torch.long, device=device)
        for i, row in enumerate(batch_rows):
            seq = torch.tensor(row, dtype=torch.long, device=device)
            input_ids[i, : len(row)] = seq
            attn_mask[i, : len(row)] = 1
        yield input_ids, attn_mask


def _critical_position_mask(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    # Mark digits/operators/mod tokens as routing-critical.
    bsz, seq = input_ids.shape
    mask = torch.zeros((bsz, seq), dtype=torch.bool, device=input_ids.device)
    for b in range(bsz):
        toks = tokenizer.convert_ids_to_tokens(input_ids[b].detach().cpu().tolist())
        for i, tok in enumerate(toks):
            s = (tok or "").lower()
            if any(ch.isdigit() for ch in s) or ("+" in s) or ("-" in s) or ("=" in s) or ("mod" in s) or ("carry" in s):
                mask[b, i] = True
    return mask


def _compute_attention_invariance(
    attentions: tuple[torch.Tensor, ...],
    si_heads_by_layer: dict[int, set[int]],
    fallback_device: torch.device | None = None,
) -> torch.Tensor:
    """Compute a *proxy* for shift-invariance preservation during training.

    NOTE: This is NOT the full R² metric used in post-hoc audits.  It measures
    the mean variance of attention weights along each relative-position diagonal
    for SI-classified heads.  Lower variance means the attention pattern is more
    consistent across positions at a fixed offset — a necessary (but not
    sufficient) condition for high R² under the kernel fit.

    The proxy is cheap enough to run inside the training loop and is used by the
    ``si_preserve`` auxiliary loss (Experiment 4B) to penalise large increases in
    diagonal variance relative to the pre-training baseline.  The actual R² is
    computed in ``post_ft_si_audit`` after training completes.
    """
    if not attentions:
        dev = fallback_device if fallback_device is not None else torch.device("cpu")
        return torch.tensor(0.0, dtype=torch.float32, device=dev)

    # Lower variance across each relative-position diagonal => more SI-like.
    vals: list[torch.Tensor] = []
    for layer_idx, layer_attn in enumerate(attentions):
        if layer_idx not in si_heads_by_layer:
            continue
        heads = sorted(si_heads_by_layer[layer_idx])
        if not heads:
            continue
        # [batch, heads, q, k]
        sel = layer_attn[:, heads, :, :]
        b, h, q, _k = sel.shape
        if q < 4:
            continue
        # aggregate per head over diagonals.
        for off in range(1, min(64, q)):
            idx_q = torch.arange(off, q, device=sel.device)
            idx_k = idx_q - off
            diag = sel[:, :, idx_q, idx_k]  # [b, h, n]
            if diag.numel() == 0:
                continue
            var = diag.var(dim=-1, unbiased=False).mean()  # scalar
            vals.append(var)
    if not vals:
        dev = attentions[0].device if attentions else (fallback_device if fallback_device is not None else torch.device("cpu"))
        return torch.tensor(0.0, dtype=torch.float32, device=dev)
    return torch.stack(vals).mean()


def _compute_routing_loss(
    attentions: tuple[torch.Tensor, ...],
    si_heads_by_layer: dict[int, set[int]],
    critical_mask: torch.Tensor,
) -> torch.Tensor:
    vals: list[torch.Tensor] = []
    for layer_idx, layer_attn in enumerate(attentions):
        heads = sorted(si_heads_by_layer.get(layer_idx, set()))
        if not heads:
            continue
        sel = layer_attn[:, heads, :, :]  # [b,h,q,k]
        b, h, q, _k = sel.shape
        if q < 2:
            continue
        idx_q = torch.arange(1, q, device=sel.device)
        idx_k = idx_q - 1
        prev_attn = sel[:, :, idx_q, idx_k]  # [b,h,q-1]
        crit = critical_mask[:, 1:q].float().unsqueeze(1)  # [b,1,q-1]
        denom = crit.sum()
        if float(denom.item()) <= 0.0:
            continue
        weighted = (prev_attn * crit).sum() / torch.clamp(denom, min=1.0)
        vals.append(weighted)
    if not vals:
        return torch.tensor(0.0, dtype=torch.float32, device=critical_mask.device)
    # Maximize weighted attention -> minimize negative.
    return -torch.stack(vals).mean()


def _compute_baseline_invariance(
    *,
    model,
    tokenizer,
    model_name: str,
    si_heads_by_layer: dict[int, set[int]],
    device: str,
    seq_len: int,
) -> float:
    seqs = load_profile_sequences(tokenizer=tokenizer, model_name=model_name, num_sequences=2, seq_len=seq_len)
    if not seqs:
        return 0.0
    vals: list[float] = []
    with torch.no_grad():
        for seq in seqs:
            ids = torch.tensor([seq], dtype=torch.long, device=device)
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_attentions=True)
            inv = _compute_attention_invariance(
                out.attentions,
                si_heads_by_layer=si_heads_by_layer,
                fallback_device=ids.device,
            )
            vals.append(float(inv.item()))
            del ids, out
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _train_single_run(
    *,
    model_name: str,
    condition: str,
    seed: int,
    device: str,
    output_dir: Path,
    train_cfg: TrainConfig,
    lora_hp: LoRAHyperParams,
    aux_cfg: AuxLossConfig,
    checkpoint_steps: tuple[int, ...] | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    set_global_seed(seed)
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
    model.train()

    tokenizer = load_tokenizer(model_spec)
    _prepare_tokenizer_for_training(tokenizer)

    high_heads, _low_heads = _load_baseline_head_groups(model_name)
    si_map = _si_map_from_heads(high_heads)
    apply_info: LoRAApplyResult = apply_lora_policy(
        model,
        condition=condition,
        si_heads_by_layer=si_map,
        hp=lora_hp,
    )

    training_texts = build_training_mix(
        tokenizer=tokenizer,
        model_name=model_name,
        per_task=train_cfg.math_per_task,
        control_fraction=train_cfg.control_fraction,
        seq_len=train_cfg.max_length,
        seed=seed,
    )
    encoded = _encode_texts(tokenizer, training_texts, max_length=train_cfg.max_length)
    if not encoded:
        raise RuntimeError("Training mix yielded zero encodable samples.")

    pad_id = int(tokenizer.pad_token_id)
    batch_iter = _iter_batches(encoded, batch_size=train_cfg.batch_size, pad_id=pad_id, device=device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters were enabled.")
    opt = AdamW(trainable, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    num_train_steps = max(1, int(train_cfg.max_steps))
    num_warmup = max(0, min(int(train_cfg.warmup_steps), num_train_steps - 1))
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=num_warmup,
        num_training_steps=num_train_steps,
    )

    baseline_inv = _compute_baseline_invariance(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        si_heads_by_layer=si_map,
        device=device,
        seq_len=min(256, train_cfg.max_length),
    )
    checkpoint_step_set = {int(x) for x in (checkpoint_steps or tuple())}
    if checkpoint_dir is not None and 0 in checkpoint_step_set:
        init_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if ("lora_a" in k or "lora_b" in k)}
        ckpt_path = ensure_dir(checkpoint_dir) / "checkpoint_step_0.pt"
        torch.save(init_state, ckpt_path)

    losses: list[dict[str, Any]] = []
    start = time.time()
    for step in range(1, int(train_cfg.max_steps) + 1):
        input_ids, attn_mask = next(batch_iter)
        labels = input_ids.clone()
        labels = labels.masked_fill(attn_mask == 0, -100)

        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
            use_cache=False,
            output_attentions=(aux_cfg.mode != "none"),
        )
        base_loss = out.loss
        total_loss = base_loss
        aux_preserve = torch.tensor(0.0, dtype=base_loss.dtype, device=base_loss.device)
        aux_route = torch.tensor(0.0, dtype=base_loss.dtype, device=base_loss.device)

        has_attn = out.attentions is not None and len(out.attentions) > 0
        if aux_cfg.mode != "none" and has_attn and (step % max(1, train_cfg.aux_interval) == 0):
            if aux_cfg.mode in {"si_preserve", "combined"} and aux_cfg.lambda_preserve > 0:
                cur_inv = _compute_attention_invariance(
                    out.attentions,
                    si_heads_by_layer=si_map,
                    fallback_device=base_loss.device,
                )
                baseline_t = torch.tensor(float(baseline_inv), device=cur_inv.device, dtype=cur_inv.dtype)
                aux_preserve = torch.relu(cur_inv - baseline_t)
                total_loss = total_loss + (aux_cfg.lambda_preserve * aux_preserve)

            if aux_cfg.mode in {"si_routing", "combined"} and aux_cfg.lambda_routing > 0:
                crit = _critical_position_mask(input_ids, tokenizer)
                aux_route = _compute_routing_loss(out.attentions, si_heads_by_layer=si_map, critical_mask=crit)
                total_loss = total_loss + (aux_cfg.lambda_routing * aux_route)

        total_loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, train_cfg.grad_clip)
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)

        row = {
            "step": int(step),
            "base_loss": float(base_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "aux_preserve": float(aux_preserve.detach().item()),
            "aux_route": float(aux_route.detach().item()),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        losses.append(row)

        if step == 1 or step % max(1, train_cfg.log_every) == 0 or step == int(train_cfg.max_steps):
            elapsed = time.time() - start
            eta = (elapsed / max(step, 1)) * max(0, int(train_cfg.max_steps) - step)
            print(
                f"[{model_name}|{condition}|seed={seed}] step={step}/{train_cfg.max_steps} "
                f"loss={row['total_loss']:.4f} base={row['base_loss']:.4f} "
                f"aux_p={row['aux_preserve']:.4f} aux_r={row['aux_route']:.4f} eta={format_eta(eta)}",
                flush=True,
            )

        # Save LoRA checkpoint at requested steps.
        if checkpoint_dir is not None and step in checkpoint_step_set:
            ckpt_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if ("lora_a" in k or "lora_b" in k)}
            ckpt_path = ensure_dir(checkpoint_dir) / f"checkpoint_step_{step}.pt"
            torch.save(ckpt_state, ckpt_path)

        del input_ids, attn_mask, labels, out, base_loss, total_loss
        clear_cuda()

    elapsed = time.time() - start
    model.eval()

    # Save only LoRA trainable tensors.
    adapter_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if ("lora_a" in k or "lora_b" in k)}
    ensure_dir(output_dir)
    torch.save(adapter_state, output_dir / "adapter_state.pt")
    write_parquet(output_dir / "train_loss_curve.parquet", losses)

    summary = {
        "timestamp": now_timestamp(),
        "model": model_name,
        "condition": condition,
        "seed": int(seed),
        "elapsed_sec": float(elapsed),
        "train_config": train_cfg.__dict__,
        "aux_config": aux_cfg.__dict__,
        "lora": {
            "rank": int(lora_hp.rank),
            "alpha": float(lora_hp.alpha),
            "dropout": float(lora_hp.dropout),
            "si_rank": int(lora_hp.si_rank),
            "si_amplified_rank": int(lora_hp.si_amplified_rank),
            "non_si_reduced_rank": int(lora_hp.non_si_reduced_rank),
            "replaced_modules": len(apply_info.replaced_modules),
            "trainable_params": int(apply_info.trainable_params),
            "total_params": int(apply_info.total_params),
            "trainable_param_fraction": float(apply_info.trainable_params / max(1, apply_info.total_params)),
            "sample_trainable_names": collect_trainable_parameter_names(model)[:32],
        },
        "optimizer": {
            "name": "AdamW",
            "warmup_steps_applied": int(num_warmup),
            "total_steps": int(num_train_steps),
        },
    }
    write_json(output_dir / "train_summary.json", summary)
    return model, tokenizer, summary


def _score_option_logprob(model, tokenizer, device: str, prompt: str, completion: str) -> float:
    p = tokenizer.encode(prompt, add_special_tokens=False)
    c = tokenizer.encode(completion, add_special_tokens=False)
    if not p or not c:
        return float("-inf")
    seq = p + c
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False)
        lp = torch.log_softmax(out.logits.float(), dim=-1)
    total = 0.0
    start = len(p)
    for j, tok in enumerate(c):
        pos = start + j - 1
        if pos < 0 or pos >= lp.shape[1]:
            continue
        total += float(lp[0, pos, int(tok)].item())
    return total


def evaluate_math_accuracy(
    *,
    model,
    tokenizer,
    device: str,
    eval_battery: dict[str, list[MCExample]],
) -> dict[str, Any]:
    task_rows: list[dict[str, Any]] = []
    all_correct = 0
    all_n = 0
    with _temporary_eval_mode(model):
        for task, examples in eval_battery.items():
            correct = 0
            for ex in examples:
                scores = [_score_option_logprob(model, tokenizer, device, ex.prompt, opt) for opt in ex.options]
                pred = int(np.argmax(scores))
                correct += int(pred == int(ex.correct_index))
            n = len(examples)
            acc = float(correct / max(1, n))
            task_rows.append({"task": task, "n": int(n), "correct": int(correct), "accuracy": acc})
            all_correct += int(correct)
            all_n += int(n)
    overall = float(all_correct / max(1, all_n))
    return {
        "overall_accuracy": overall,
        "n_total": int(all_n),
        "correct_total": int(all_correct),
        "by_task": task_rows,
    }


def evaluate_wiki_perplexity(
    *,
    model,
    tokenizer,
    model_name: str,
    device: str,
    count: int,
    seq_len: int,
) -> dict[str, Any]:
    seqs = load_profile_sequences(tokenizer=tokenizer, model_name=model_name, num_sequences=max(1, count), seq_len=seq_len)
    seqs = seqs[: max(1, count)]
    losses: list[float] = []
    with _temporary_eval_mode(model):
        with torch.inference_mode():
            for seq in seqs:
                ids = torch.tensor([seq], dtype=torch.long, device=device)
                out = model(input_ids=ids, labels=ids, use_cache=False)
                losses.append(float(out.loss.item()))
                del ids, out
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    ppl = float(math.exp(mean_loss)) if np.isfinite(mean_loss) else float("nan")
    return {
        "n_sequences": int(len(seqs)),
        "mean_loss": mean_loss,
        "perplexity": ppl,
    }


def post_ft_si_audit(
    *,
    model,
    tokenizer,
    model_name: str,
    device: str,
    output_dir: Path,
    r2_sequences: int,
    seq_len: int,
    synthetic_target_per_cell: int,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    model_spec = THEORY_MODELS[model_name]

    seqs = load_profile_sequences(tokenizer=tokenizer, model_name=model_name, num_sequences=max(8, r2_sequences), seq_len=seq_len)
    seqs = seqs[: max(8, r2_sequences)]

    with _temporary_eval_mode(model):
        adapter = get_adapter(model_spec)
        adapter.register(model)
        r2_df = compute_per_head_r2(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            model_spec=model_spec,
            device=device,
            sequences=seqs,
        )
        high_si, low_si, mean_r2 = classify_heads(r2_df)

        approach_a, approach_c_raw = run_attention_analysis(
            model=model,
            adapter=adapter,
            model_spec=model_spec,
            tokenizer=tokenizer,
            device=device,
            sequences=seqs,
            high_si=high_si,
            low_si=low_si,
            r2_df=mean_r2,
        )
        approach_c, approach_c_df = _normalize_approach_c_output(approach_c_raw)

        high_pairs = [(int(h.layer), int(h.head)) for h in high_si]
        low_pairs = [(int(h.layer), int(h.head)) for h in low_si]
        synth_result, synth_rows = _synthetic_boundary_full(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            device=device,
            sequences=seqs,
            high_heads=high_pairs,
            low_heads=low_pairs,
            target_per_transformed_cell=int(synthetic_target_per_cell),
            seed=0,
        )
        adapter.cleanup()

    baseline_high, _baseline_low = _load_baseline_head_groups(model_name)
    baseline_set = {(int(h.layer), int(h.head)) for h in baseline_high}
    post_set = {(int(h.layer), int(h.head)) for h in high_si}
    inter = len(baseline_set & post_set)
    union = max(1, len(baseline_set | post_set))

    comp = approach_a.get("high_vs_low_comparison", {}).get("attn_to_prev_last", {})
    d_val = safe_float(comp.get("cohens_d"))

    audit = {
        "timestamp": now_timestamp(),
        "model": model_name,
        "n_r2_rows": int(len(r2_df)),
        "n_profile_sequences": int(len(seqs)),
        "r2_summary": {
            "mean": safe_float(mean_r2["mean_r2"].mean()) if not mean_r2.empty else float("nan"),
            "std": safe_float(mean_r2["mean_r2"].std()) if not mean_r2.empty else float("nan"),
            "min": safe_float(mean_r2["mean_r2"].min()) if not mean_r2.empty else float("nan"),
            "max": safe_float(mean_r2["mean_r2"].max()) if not mean_r2.empty else float("nan"),
        },
        "boundary": {
            "post_ablation_d": d_val,
            "t_statistic": safe_float(comp.get("t_statistic")),
            "p_two_sided": safe_float(comp.get("t_p_value")),
            "p_one_sided_high_gt_low": safe_float(
                one_sided_p_from_two_sided(
                    safe_float(comp.get("t_statistic")),
                    safe_float(comp.get("t_p_value")),
                    alternative="greater",
                )
            ),
            "prefix_following_artifact_flag": bool(synth_result.get("prefix_following_artifact_flag", False)),
            "synthetic": synth_result,
        },
        "si_identity_shift": {
            "baseline_high_count": int(len(baseline_set)),
            "post_high_count": int(len(post_set)),
            "intersection": int(inter),
            "jaccard": float(inter / union),
        },
        "approach_c_score_rows": int(len(approach_c_df)) if isinstance(approach_c_df, pd.DataFrame) else 0,
    }

    write_parquet(output_dir / "post_ft_r2_per_sequence.parquet", r2_df)
    write_parquet(output_dir / "post_ft_r2_summary.parquet", mean_r2)
    write_json(output_dir / "post_ft_boundary_approach_a.json", approach_a)
    write_json(output_dir / "post_ft_boundary_approach_c.json", approach_c)
    write_json(output_dir / "post_ft_synthetic_boundary.json", synth_result)
    if isinstance(synth_rows, pd.DataFrame) and not synth_rows.empty:
        synth_rows.to_parquet(output_dir / "post_ft_synthetic_rows.parquet", index=False)
    write_json(output_dir / "post_ft_si_audit.json", audit)
    return audit


def run_c1_mini_ablation(
    *,
    model,
    tokenizer,
    model_name: str,
    device: str,
    output_path: Path,
    fractions: tuple[int, ...] = (0, 5, 10, 20, 25),
    num_seeds: int = 2,
    synthetic_count: int = 32,
    batch_size: int = 2,
) -> pd.DataFrame:
    model_spec = THEORY_MODELS[model_name]

    # Rank heads by post-FT R2 on a small sample.
    with _temporary_eval_mode(model):
        adapter = get_adapter(model_spec)
        adapter.register(model)
        seqs = load_profile_sequences(tokenizer=tokenizer, model_name=model_name, num_sequences=10, seq_len=256)
        r2_df = compute_per_head_r2(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            model_spec=model_spec,
            device=device,
            sequences=seqs,
        )
        adapter.cleanup()
    r2_summary = r2_df.groupby(["layer", "head"], as_index=False)["r2"].mean().rename(columns={"r2": "mean_r2"})
    r2_summary = r2_summary.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    ranked = [HeadIndex(layer=int(r.layer), head=int(r.head)) for r in r2_summary.itertuples()]

    vocab_size = int(tokenizer.vocab_size)
    special_ids = [
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "unk_token_id", None),
    ]
    special_ids = [int(x) for x in special_ids if x is not None]
    pools = build_token_pools(model_name=model_name, vocab_size=vocab_size, special_ids=special_ids)

    retrieval_candidates = RETRIEVAL_SPANS.get(model_name, (32,))
    retrieval_span = 48 if 48 in retrieval_candidates else int(retrieval_candidates[0])
    task_cfg = [
        ("long_range_retrieval", retrieval_span, (retrieval_span,)),
        ("local_key_match", None, None),
    ]

    prebuilt: dict[tuple[int, str, int], list[Any]] = {}
    for seed in range(num_seeds):
        for task_name, span_override, span_choices in task_cfg:
            span_val = span_override if span_override is not None else 0
            prebuilt[(seed, task_name, span_val)] = generate_task_examples(
                task_name=task_name,
                model_name=model_name,
                seq_len=512,
                seed=seed,
                count=synthetic_count,
                pools=pools,
                span_override=span_override,
                span_choices=span_choices,
            )

    all_rows: list[dict[str, Any]] = []
    for frac in fractions:
        n_total = len(ranked)
        n_sel = int(round((float(frac) / 100.0) * n_total))
        if frac > 0:
            n_sel = max(1, n_sel)
        heads = ranked[:n_sel]

        cond_name = f"ablate_top_{int(frac)}pct"
        rows = evaluate_task_battery(
            model=model,
            tokenizer=tokenizer,
            model_spec=model_spec,
            device=device,
            heads_to_zero=[type("H", (), {"layer": h.layer, "head": h.head}) for h in heads],
            condition_name=cond_name,
            seeds=range(num_seeds),
            retrieval_spans=(retrieval_span,),
            pools=pools,
            synthetic_count=synthetic_count,
            batch_size=batch_size,
            task_configs=task_cfg,
            prebuilt_examples=prebuilt,
        )
        for r in rows:
            r["ablation_fraction"] = int(frac)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        write_parquet(output_path, df)
        return df

    baseline = (
        df[df["ablation_fraction"] == 0]
        .groupby(["task", "span", "seed"], as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "baseline_accuracy"})
    )
    merged = df.merge(baseline, on=["task", "span", "seed"], how="left")
    merged["degradation"] = merged["baseline_accuracy"] - merged["accuracy"]
    write_parquet(output_path, merged)
    return merged


def run_4c_trajectory(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    train_cfg: TrainConfig,
    lora_hp: LoRAHyperParams,
    checkpoints: tuple[int, ...],
) -> dict[str, Any]:
    output_dir = ensure_dir(output_root / model_name)
    ckpt_dir = ensure_dir(output_dir / "ft_run" / "checkpoints")

    model, tokenizer, train_summary = _train_single_run(
        model_name=model_name,
        condition="d_full_qlora_baseline",
        seed=0,
        device=device,
        output_dir=output_dir / "ft_run",
        train_cfg=train_cfg,
        lora_hp=lora_hp,
        aux_cfg=AuxLossConfig(mode="none", lambda_preserve=0.0, lambda_routing=0.0),
        checkpoint_steps=checkpoints,
        checkpoint_dir=ckpt_dir,
    )

    # Load each saved checkpoint and audit SI structure at that training point.
    traj_rows: list[dict[str, Any]] = []
    baseline_high, _baseline_low = _load_baseline_head_groups(model_name)
    baseline_set = {(h.layer, h.head) for h in baseline_high}

    # Keep the final adapter state so we can restore it after checkpoint audits.
    final_state = {k: v.clone() for k, v in model.state_dict().items() if ("lora_a" in k or "lora_b" in k)}

    for idx, ckpt in enumerate(checkpoints):
        seq_seed = int(1000 + idx)
        set_global_seed(seq_seed)

        # Load this checkpoint's LoRA weights into the model.
        ckpt_path = ckpt_dir / f"checkpoint_step_{ckpt}.pt"
        if ckpt_path.exists():
            ckpt_state = torch.load(ckpt_path, map_location=device, weights_only=True)
            current_sd = model.state_dict()
            current_sd.update(ckpt_state)
            model.load_state_dict(current_sd, strict=False)
        else:
            print(f"[4C] Warning: checkpoint {ckpt_path} not found, auditing current model state", flush=True)

        audit = post_ft_si_audit(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=device,
            output_dir=output_dir / "checkpoint_audits" / f"step_{ckpt}",
            r2_sequences=8,
            seq_len=256,
            synthetic_target_per_cell=80,
        )
        post_r2 = pd.read_parquet(output_dir / "checkpoint_audits" / f"step_{ckpt}" / "post_ft_r2_summary.parquet")
        post_r2 = post_r2.sort_values("mean_r2", ascending=False).reset_index(drop=True)
        n_sel = max(1, int(round(0.25 * len(post_r2))))
        post_heads = {(int(r.layer), int(r.head)) for r in post_r2.head(n_sel).itertuples()}
        inter = len(baseline_set & post_heads)
        union = max(1, len(baseline_set | post_heads))
        traj_rows.append(
            {
                "checkpoint_step": int(ckpt),
                "mean_r2": safe_float(audit.get("r2_summary", {}).get("mean")),
                "boundary_d": safe_float(audit.get("boundary", {}).get("post_ablation_d")),
                "prefix_following_artifact_flag": bool(audit.get("boundary", {}).get("prefix_following_artifact_flag", False)),
                "jaccard_vs_step0": float(inter / union),
            }
        )

    # Restore the final adapter state.
    final_sd = model.state_dict()
    final_sd.update(final_state)
    model.load_state_dict(final_sd, strict=False)

    traj_df = pd.DataFrame(traj_rows)
    traj_df.to_parquet(output_dir / "si_trajectory_during_ft.parquet", index=False)

    summary = {
        "timestamp": now_timestamp(),
        "model": model_name,
        "train_summary": train_summary,
        "checkpoints": [int(x) for x in checkpoints],
        "n_points": int(len(traj_df)),
        "trajectory_mean_r2_delta": safe_float(traj_df["mean_r2"].iloc[-1] - traj_df["mean_r2"].iloc[0]) if len(traj_df) >= 2 else float("nan"),
        "trajectory_boundary_d_delta": safe_float(traj_df["boundary_d"].iloc[-1] - traj_df["boundary_d"].iloc[0]) if len(traj_df) >= 2 else float("nan"),
        "status": "completed",
    }
    write_json(output_dir / "si_trajectory_summary.json", summary)
    return summary


def run_4a_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    conditions: list[str],
    seeds: list[int],
    train_cfg: TrainConfig,
    lora_hp: LoRAHyperParams,
) -> dict[str, Any]:
    model_root = ensure_dir(output_root / model_name)
    seen_conditions: set[str] = set()
    canonical_conditions: list[str] = []
    for cond in conditions:
        if cond not in seen_conditions:
            seen_conditions.add(cond)
            canonical_conditions.append(cond)
    conditions = canonical_conditions

    run_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    ablation_written = False

    for condition in conditions:
        for seed in seeds:
            run_dir = ensure_dir(model_root / "runs" / condition / f"seed_{seed}")
            model, tokenizer, train_summary = _train_single_run(
                model_name=model_name,
                condition=condition,
                seed=seed,
                device=device,
                output_dir=run_dir,
                train_cfg=train_cfg,
                lora_hp=lora_hp,
                aux_cfg=AuxLossConfig(mode="none", lambda_preserve=0.0, lambda_routing=0.0),
            )

            eval_battery = build_math_eval_battery(count_per_task=64, seed=seed + 10000)
            math_eval = evaluate_math_accuracy(model=model, tokenizer=tokenizer, device=device, eval_battery=eval_battery)
            wiki_eval = evaluate_wiki_perplexity(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                device=device,
                count=24,
                seq_len=256,
            )
            write_json(run_dir / "math_eval.json", math_eval)
            write_json(run_dir / "wiki_eval.json", wiki_eval)

            audit = post_ft_si_audit(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                device=device,
                output_dir=run_dir / "post_ft_audit",
                r2_sequences=12,
                seq_len=256,
                synthetic_target_per_cell=120,
            )

            run_rows.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    "math_overall_accuracy": safe_float(math_eval.get("overall_accuracy")),
                    "wiki_perplexity": safe_float(wiki_eval.get("perplexity")),
                    "train_elapsed_sec": safe_float(train_summary.get("elapsed_sec")),
                    "boundary_d": safe_float(audit.get("boundary", {}).get("post_ablation_d")),
                    "si_jaccard": safe_float(audit.get("si_identity_shift", {}).get("jaccard")),
                }
            )

            audit_rows.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    "boundary_d": safe_float(audit.get("boundary", {}).get("post_ablation_d")),
                    "prefix_following_artifact_flag": bool(audit.get("boundary", {}).get("prefix_following_artifact_flag", False)),
                    "si_jaccard": safe_float(audit.get("si_identity_shift", {}).get("jaccard")),
                }
            )

            if (not ablation_written) and condition == conditions[0] and seed == seeds[0]:
                c1_df = run_c1_mini_ablation(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    device=device,
                    output_path=model_root / "post_ft_ablation_curve.parquet",
                    fractions=(0, 5, 10, 20, 25),
                    num_seeds=2,
                    synthetic_count=24,
                    batch_size=2,
                )
                ablation_written = True
                if c1_df.empty:
                    print(f"[4A] warning: empty C1 mini-run output for {model_name}", flush=True)

            del model
            clear_cuda()

    run_df = pd.DataFrame(run_rows)
    run_df.to_parquet(model_root / "si_lora_runs.parquet", index=False)
    write_json(model_root / "post_ft_si_audit.json", {"rows": audit_rows})

    summary: dict[str, Any] = {
        "timestamp": now_timestamp(),
        "model": model_name,
        "n_runs": int(len(run_df)),
        "conditions": conditions,
        "seeds": [int(x) for x in seeds],
    }

    def _paired_comparison(
        df: pd.DataFrame,
        cond_x: str,
        cond_y: str,
        metric: str = "math_overall_accuracy",
        alternative: str = "greater",
    ) -> dict[str, Any] | None:
        """Paired seed-level comparison: tests H_a that cond_x > cond_y."""
        x = df[df["condition"] == cond_x].set_index("seed")
        y = df[df["condition"] == cond_y].set_index("seed")
        common = sorted(set(x.index.tolist()) & set(y.index.tolist()))
        if not common:
            return None
        x_vals = x.loc[common, metric].to_numpy(dtype=float)
        y_vals = y.loc[common, metric].to_numpy(dtype=float)
        diffs = x_vals - y_vals
        t_stat, p_two = scipy_stats.ttest_1samp(diffs, popmean=0.0)
        mean_diff, ci = mean_ci95(diffs.tolist())
        return {
            "n_pairs": int(len(common)),
            "mean_diff_accuracy": float(mean_diff),
            "ci95": [float(ci[0]), float(ci[1])],
            "t_statistic": safe_float(t_stat),
            "p_two_sided": safe_float(p_two),
            f"p_one_sided_{cond_x.split('_')[0]}_gt_{cond_y.split('_')[0]}": safe_float(
                one_sided_p_from_two_sided(safe_float(t_stat), safe_float(p_two), alternative=alternative)
            ),
            "cohens_d_paired": safe_float(
                float(np.mean(diffs) / max(1e-12, np.std(diffs, ddof=1))) if len(diffs) > 1 else float("nan")
            ),
            "paired_seeds": [int(x) for x in common],
            f"{cond_x.split('_')[0]}_values": [float(v) for v in x_vals.tolist()],
            f"{cond_y.split('_')[0]}_values": [float(v) for v in y_vals.tolist()],
        }

    # Primary paired comparison: a (SI-protecting) vs b (uniform).
    comp_a_b = _paired_comparison(run_df, "a_si_protecting_lora", "b_uniform_lora")
    if comp_a_b is not None:
        summary["primary_comparison_a_vs_b"] = comp_a_b

    # Sanity comparison: uniform-vs-full baseline should now be meaningfully
    # different in implementation (condition d targets all Linear modules).
    comp_b_d = _paired_comparison(run_df, "b_uniform_lora", "d_full_qlora_baseline")
    if comp_b_d is not None:
        summary["comparison_b_vs_d"] = comp_b_d

    # Condition 'e' comparisons (if present in this run).
    if "e_si_amplified_lora" in conditions:
        # Main test: does SI amplification beat standard FT?
        comp_e_d = _paired_comparison(run_df, "e_si_amplified_lora", "d_full_qlora_baseline")
        if comp_e_d is not None:
            summary["comparison_e_vs_d"] = comp_e_d

        # Amplify vs protect: opposite rank allocations on SI heads.
        comp_e_a = _paired_comparison(run_df, "e_si_amplified_lora", "a_si_protecting_lora")
        if comp_e_a is not None:
            summary["comparison_e_vs_a"] = comp_e_a

        # Amplify vs exclusive: does adding reduced non-SI capacity help?
        comp_e_c = _paired_comparison(run_df, "e_si_amplified_lora", "c_si_only_lora")
        if comp_e_c is not None:
            summary["comparison_e_vs_c"] = comp_e_c

    write_json(model_root / "si_lora_comparison.json", summary)
    return summary


def run_4b_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    seeds: list[int],
    lambdas: list[float],
    train_cfg: TrainConfig,
    lora_hp: LoRAHyperParams,
) -> dict[str, Any]:
    model_root = ensure_dir(output_root / model_name)

    run_rows: list[dict[str, Any]] = []

    def _run(condition: str, seed: int, lam_p: float, lam_r: float) -> dict[str, Any]:
        run_tag = f"{condition}_seed{seed}_lp{lam_p}_lr{lam_r}"
        run_dir = ensure_dir(model_root / "runs" / run_tag)
        mode = "none"
        if condition == "a_standard_ft":
            mode = "none"
        elif condition == "b_si_preserve":
            mode = "si_preserve"
        elif condition == "c_si_routing":
            mode = "si_routing"
        elif condition == "d_combined":
            mode = "combined"

        model, tokenizer, train_summary = _train_single_run(
            model_name=model_name,
            condition="d_full_qlora_baseline",
            seed=seed,
            device=device,
            output_dir=run_dir,
            train_cfg=train_cfg,
            lora_hp=lora_hp,
            aux_cfg=AuxLossConfig(mode=mode, lambda_preserve=lam_p, lambda_routing=lam_r),
        )

        eval_battery = build_math_eval_battery(count_per_task=64, seed=seed + 20000)
        math_eval = evaluate_math_accuracy(model=model, tokenizer=tokenizer, device=device, eval_battery=eval_battery)
        wiki_eval = evaluate_wiki_perplexity(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=device,
            count=24,
            seq_len=256,
        )
        audit = post_ft_si_audit(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=device,
            output_dir=run_dir / "post_ft_audit",
            r2_sequences=10,
            seq_len=256,
            synthetic_target_per_cell=100,
        )

        row = {
            "condition": condition,
            "seed": int(seed),
            "lambda_preserve": float(lam_p),
            "lambda_routing": float(lam_r),
            "math_overall_accuracy": safe_float(math_eval.get("overall_accuracy")),
            "wiki_perplexity": safe_float(wiki_eval.get("perplexity")),
            "boundary_d": safe_float(audit.get("boundary", {}).get("post_ablation_d")),
            "si_jaccard": safe_float(audit.get("si_identity_shift", {}).get("jaccard")),
            "train_elapsed_sec": safe_float(train_summary.get("elapsed_sec")),
        }
        del model
        clear_cuda()
        return row

    # Baseline and sweeps.
    for seed in seeds:
        run_rows.append(_run("a_standard_ft", seed, 0.0, 0.0))
        for lam in lambdas:
            run_rows.append(_run("b_si_preserve", seed, lam, 0.0))
            run_rows.append(_run("c_si_routing", seed, 0.0, lam))

    df = pd.DataFrame(run_rows)
    df.to_parquet(model_root / "si_loss_runs.parquet", index=False)

    def _pick_best(sub_df: pd.DataFrame) -> tuple[float, float]:
        if sub_df.empty:
            return 0.0, 0.0
        g = sub_df.groupby(["lambda_preserve", "lambda_routing"], as_index=False)["math_overall_accuracy"].mean()
        top = g.sort_values("math_overall_accuracy", ascending=False).iloc[0]
        return float(top["lambda_preserve"]), float(top["lambda_routing"])

    best_preserve_lp, _ = _pick_best(df[df["condition"] == "b_si_preserve"])
    _, best_route_lr = _pick_best(df[df["condition"] == "c_si_routing"])

    combined_rows: list[dict[str, Any]] = []
    for seed in seeds:
        combined_rows.append(_run("d_combined", seed, best_preserve_lp, best_route_lr))
    if combined_rows:
        df = pd.concat([df, pd.DataFrame(combined_rows)], ignore_index=True)
        df.to_parquet(model_root / "si_loss_runs.parquet", index=False)

    # Summaries.
    sweep_summary = {
        "timestamp": now_timestamp(),
        "model": model_name,
        "lambdas": [float(x) for x in lambdas],
        "best_preserve_lambda": float(best_preserve_lp),
        "best_routing_lambda": float(best_route_lr),
        "n_rows": int(len(df)),
    }
    write_json(model_root / "si_loss_sweep.json", sweep_summary)

    routing_summary = {
        "rows": [
            {
                "condition": str(r.condition),
                "seed": int(r.seed),
                "lambda_preserve": float(r.lambda_preserve),
                "lambda_routing": float(r.lambda_routing),
                "math_overall_accuracy": safe_float(r.math_overall_accuracy),
                "wiki_perplexity": safe_float(r.wiki_perplexity),
                "boundary_d": safe_float(r.boundary_d),
                "si_jaccard": safe_float(r.si_jaccard),
            }
            for r in df.itertuples()
        ]
    }
    write_json(model_root / "si_routing_results.json", routing_summary)

    # Best-lambda comparison vs standard baseline.
    baseline = df[df["condition"] == "a_standard_ft"].set_index("seed")
    best_rows = df[(df["condition"] == "d_combined") & (df["lambda_preserve"] == best_preserve_lp) & (df["lambda_routing"] == best_route_lr)].set_index("seed")
    common = sorted(set(baseline.index.tolist()) & set(best_rows.index.tolist()))

    compare = {
        "n_pairs": 0,
        "mean_diff_accuracy": float("nan"),
        "ci95": [float("nan"), float("nan")],
        "t_statistic": float("nan"),
        "p_two_sided": float("nan"),
    }
    if common:
        base_vals = baseline.loc[common, "math_overall_accuracy"].to_numpy(dtype=float)
        best_vals = best_rows.loc[common, "math_overall_accuracy"].to_numpy(dtype=float)
        diffs = best_vals - base_vals
        t_stat, p_two = scipy_stats.ttest_1samp(diffs, popmean=0.0)
        mean_diff, ci = mean_ci95(diffs.tolist())
        compare = {
            "n_pairs": int(len(common)),
            "mean_diff_accuracy": float(mean_diff),
            "ci95": [float(ci[0]), float(ci[1])],
            "t_statistic": safe_float(t_stat),
            "p_two_sided": safe_float(p_two),
            "p_one_sided_best_gt_base": safe_float(one_sided_p_from_two_sided(safe_float(t_stat), safe_float(p_two), alternative="greater")),
            "paired_seeds": [int(x) for x in common],
        }

    post_ft_audit = {
        "timestamp": now_timestamp(),
        "model": model_name,
        "best_vs_baseline": compare,
        "best_lambdas": {
            "lambda_preserve": float(best_preserve_lp),
            "lambda_routing": float(best_route_lr),
        },
    }
    write_json(model_root / "post_ft_si_audit.json", post_ft_audit)
    return post_ft_audit
