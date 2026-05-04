#!/usr/bin/env python3
"""Softmax Redistribution Audit.

Tests whether SI-kernel subtraction disrupts performance through a
position-specific attention redistribution mechanism rather than
generic norm-matched perturbation.

Two controls:
  1. Entropy analysis: Compare attention entropy change under true SI
     subtraction vs permuted-kernel subtraction for high-SI heads.
     If true subtraction causes larger entropy increase, the SI kernel
     was concentrating attention at specific offsets.
  2. Constant-offset control: Shift all logits in each high-SI head by
     a scalar (the mean of g_h), preserving rank order.  If this
     causes much smaller LM-loss disruption than position-dependent
     subtraction, it confirms that position-specificity is the load-
     bearing property, not just the norm change.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import RESULTS_ROOT, ensure_dir, timestamp_now, write_json  # noqa: E402
from reinforce_exp3.scripts._shared import emit_core_artifacts, load_model_for_exp  # noqa: E402
from reinforce_exp3.scripts.run_e29b_naturaltext_longcontext_probe import (  # noqa: E402
    _load_kernels,
    _permute_kernels,
)
from reinforce_exp3.scripts.run_e31d_llama_naturalistic_importance_control import (  # noqa: E402
    _one_sided_signflip_p,
)

EXPERIMENT_ID = "SoftmaxRedistributionAudit"
MODELS_TO_RUN = ["llama-3.1-8b", "mistral-7b-v0.1", "olmo-2-7b"]
DEFAULT_OUT = RESULTS_ROOT / "softmax_redistribution_audit"

N_EVAL_SEQUENCES = 40
SEQ_LEN = 512
N_PERM_TRIALS = 3


# ---------------------------------------------------------------------------
# Hook infrastructure for entropy capture + constant-offset control
# ---------------------------------------------------------------------------

@contextmanager
def capture_attention_entropy(model, model_name: str, target_heads: list[tuple[int, int]], max_len: int):
    """Context manager capturing per-head softmax entropy for target heads.

    Inserts output hooks on each attention layer; records softmax entropy
    for specified (layer, head) pairs.  Results stored in `entropy_records`.
    """
    import types  # noqa: PLC0415
    entropy_records: dict[tuple[int, int], list[float]] = {h: [] for h in target_heads}
    head_set = set(target_heads)
    handles = []

    def _make_hook(layer_idx: int):
        def hook(module, args, output):
            attn_weights = None
            if isinstance(output, tuple):
                for x in output:
                    if isinstance(x, torch.Tensor) and x.ndim == 4:
                        attn_weights = x
                        break
            elif isinstance(output, torch.Tensor) and output.ndim == 4:
                attn_weights = output

            if attn_weights is None:
                return

            B, H, S, _ = attn_weights.shape
            for h_idx in range(H):
                if (layer_idx, h_idx) not in head_set:
                    continue
                w = attn_weights[:, h_idx, :, :].detach().float()
                # Only look at the causal part (lower triangle)
                for b in range(B):
                    for i in range(min(S, max_len)):
                        p = w[b, i, : i + 1]
                        p = p + 1e-12
                        p = p / p.sum()
                        H_val = float(-torch.sum(p * torch.log(p)).item())
                        entropy_records[(layer_idx, h_idx)].append(H_val)

        return hook

    # Register hooks on each attention module
    from shared.attention.adapters import get_adapter  # noqa: PLC0415
    spec = types.SimpleNamespace(name=model_name)
    adapter = get_adapter(spec)
    for layer_idx, attn_module in adapter._iter_attention_modules(model):
        handle = attn_module.register_forward_hook(_make_hook(layer_idx))
        handles.append(handle)

    try:
        yield entropy_records
    finally:
        for h in handles:
            h.remove()


@contextmanager
def subtract_constant_offset(model, model_name: str, kernels: dict, target_heads: list[tuple[int, int]], max_len: int):
    """Subtract only the scalar mean of g_h from each target head (position-independent).

    Because softmax is shift-invariant, this should produce near-zero LM-loss disruption,
    confirming that position-specificity is the load-bearing property of the true SI kernel.
    Uses the same mask-injection mechanism as subtract_positional_kernels.
    """
    from experiment3.theory8_position_ablation import get_model_config, get_attention_modules  # noqa: PLC0415

    if not target_heads:
        yield
        return

    cfg = get_model_config(model)
    num_query_heads = cfg["num_query_heads"]
    attn_modules = get_attention_modules(model)

    param = next(model.parameters())
    device = param.device
    mask_dtype = torch.float32

    # Build per-layer correction: constant scalar per target head (uniform across all positions)
    # Since softmax(x + c) = softmax(x), this has zero net effect and serves as the null control.
    layer_to_heads: dict[int, list[tuple[int, int]]] = {}
    for lh in target_heads:
        layer_to_heads.setdefault(lh[0], []).append(lh)

    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx, lh_list in layer_to_heads.items():
        if layer_idx not in attn_modules:
            continue
        attn_module = attn_modules[layer_idx]

        # Build a [1, num_query_heads, max_len, max_len] constant correction
        corr = torch.zeros(1, num_query_heads, max_len, max_len, dtype=mask_dtype, device=device)
        for lh in lh_list:
            h_idx = lh[1]
            k = kernels.get(lh)
            offset_val = float(np.mean(k)) if k is not None else 0.0
            corr[0, h_idx, :, :] = -offset_val  # uniform subtraction

        def make_hook(corr_tensor: torch.Tensor):
            def hook(module, args, kwargs):
                mask = kwargs.get("attention_mask")
                if mask is None:
                    return
                batch_size = mask.shape[0]
                actual_seq = mask.shape[-1]
                c = corr_tensor[:, :, :actual_seq, :actual_seq]
                if mask.shape[1] == 1:
                    expanded_mask = mask.expand(batch_size, c.shape[1], -1, -1).clone()
                else:
                    expanded_mask = mask.clone()
                expanded_mask = expanded_mask + c.to(expanded_mask.dtype)
                kwargs["attention_mask"] = expanded_mask
                return args, kwargs
            return hook

        handle = attn_module.register_forward_pre_hook(make_hook(corr), with_kwargs=True)
        handles.append(handle)

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_wiki_sequences(model_name: str, n: int, seq_len: int, seed: int) -> list[torch.Tensor]:
    """Load pre-tokenized Wikipedia sequences from local data directory."""
    from experiment3.theory8_position_ablation import load_wiki_sequences  # noqa: PLC0415
    token_lists = load_wiki_sequences(model_name, max_sequences=n * 4, seq_len=seq_len)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(token_lists))
    seqs: list[torch.Tensor] = []
    for i in indices:
        seqs.append(torch.tensor(token_lists[i], dtype=torch.long))
        if len(seqs) >= n:
            break
    while len(seqs) < n:
        seqs.append(seqs[0].clone())
    return seqs


def _eval_lm_loss(model, seqs: list[torch.Tensor], device: str, batch_size: int) -> np.ndarray:
    """Return per-sequence LM cross-entropy loss."""
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i : i + batch_size]
            max_len = max(s.shape[0] for s in batch)
            input_ids = torch.stack(
                [F.pad(s, (0, max_len - s.shape[0]), value=0) for s in batch]
            ).to(device)
            labels = input_ids.clone()
            labels[labels == 0] = -100
            out = model(input_ids=input_ids, labels=labels)
            for _ in batch:
                losses.append(float(out.loss.item()))
    return np.array(losses, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run_one_model(
    model_name: str,
    device: str,
    out_root: Path,
    n_sequences: int,
    seq_len: int,
    n_perm_trials: int,
    batch_size: int,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    t0 = time.time()
    if smoke:
        n_sequences = min(n_sequences, 8)
        n_perm_trials = 1

    model_dir = ensure_dir(out_root / model_name)
    model, tokenizer = load_model_for_exp(model_name, device=device, attn_implementation="eager")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kernels = _load_kernels(model_name)

    groups_path = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_groups.json"
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    high_heads = [
        (int(d["layer"]), int(d["head"]))
        for d in groups.get("high_si", [])
        if (int(d["layer"]), int(d["head"])) in kernels
    ][:64]  # cap to 64 for speed

    seqs = _load_wiki_sequences(model_name, n=n_sequences, seq_len=seq_len, seed=seed)

    # --- 1. Baseline LM loss ---
    baseline_loss = _eval_lm_loss(model, seqs, device, batch_size)

    # --- 2. True SI subtraction loss ---
    from experiment3.theory8_position_ablation import subtract_positional_kernels  # noqa: PLC0415
    with subtract_positional_kernels(model, kernels, high_heads, seq_len):
        si_loss = _eval_lm_loss(model, seqs, device, batch_size)

    si_delta = si_loss - baseline_loss

    # --- 3. Permuted-kernel subtraction losses (N trials) ---
    perm_deltas: list[np.ndarray] = []
    for trial in range(n_perm_trials):
        rng_np = np.random.default_rng(seed + 500 + trial)
        k_perm = _permute_kernels(kernels, high_heads, rng_np)
        with subtract_positional_kernels(model, k_perm, high_heads, seq_len):
            perm_loss = _eval_lm_loss(model, seqs, device, batch_size)
        perm_deltas.append(perm_loss - baseline_loss)

    perm_delta_mean = np.mean(np.stack(perm_deltas, axis=0), axis=0)

    # --- 4. Constant-offset (scalar) subtraction ---
    with subtract_constant_offset(model, model_name, kernels, high_heads, seq_len):
        const_loss = _eval_lm_loss(model, seqs, device, batch_size)

    const_delta = const_loss - baseline_loss

    # --- 5. Entropy analysis: true vs permuted ---
    # Capture attention entropy for a subset of sequences (first 10)
    n_entropy = min(10, len(seqs))
    entropy_seqs = seqs[:n_entropy]

    def _run_entropy_capture_ctx(ctx_fn, seqs_in):
        """Run forward passes under a context manager, capturing attention entropy.

        Uses output_attentions=True so that self_attn returns a [B,H,S,S] weight tensor
        that the hook can find regardless of attn_implementation.
        """
        ent_accum: dict[tuple[int, int], list[float]] = {h: [] for h in high_heads}
        with capture_attention_entropy(model, model_name, high_heads, seq_len) as rec:
            model.eval()
            with ctx_fn():
                with torch.no_grad():
                    for s in seqs_in:
                        ids = s.unsqueeze(0).to(device)
                        model(input_ids=ids, output_attentions=True)
                        for h in high_heads:
                            ent_accum[h].extend(rec[h])
                            rec[h].clear()
        return {h: float(np.mean(v)) if v else float("nan") for h, v in ent_accum.items()}

    from contextlib import nullcontext  # noqa: PLC0415
    entropy_baseline = _run_entropy_capture_ctx(nullcontext, entropy_seqs)

    entropy_si = _run_entropy_capture_ctx(
        lambda: subtract_positional_kernels(model, kernels, high_heads, seq_len),
        entropy_seqs,
    )

    rng_np_e = np.random.default_rng(seed + 700)
    k_perm_e = _permute_kernels(kernels, high_heads, rng_np_e)
    entropy_perm = _run_entropy_capture_ctx(
        lambda: subtract_positional_kernels(model, k_perm_e, high_heads, seq_len),
        entropy_seqs,
    )

    # Compute entropy deltas: SI - baseline, perm - baseline
    ent_delta_si = {h: entropy_si.get(h, float("nan")) - entropy_baseline.get(h, float("nan")) for h in high_heads}
    ent_delta_perm = {h: entropy_perm.get(h, float("nan")) - entropy_baseline.get(h, float("nan")) for h in high_heads}

    valid_heads = [h for h in high_heads if np.isfinite(ent_delta_si.get(h, float("nan"))) and np.isfinite(ent_delta_perm.get(h, float("nan")))]
    ent_si_arr = np.array([ent_delta_si[h] for h in valid_heads])
    ent_perm_arr = np.array([ent_delta_perm[h] for h in valid_heads])
    ent_si_gt_perm_frac = float(np.mean(ent_si_arr > ent_perm_arr)) if len(valid_heads) > 0 else float("nan")
    mean_ent_delta_si = float(np.nanmean(ent_si_arr))
    mean_ent_delta_perm = float(np.nanmean(ent_perm_arr))

    # Sign-flip test: true entropy increase > permuted?
    p_ent = _one_sided_signflip_p(ent_si_arr - ent_perm_arr, n_perm=10000, seed=seed + 999)

    # --- Aggregate loss results ---
    result = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "model": model_name,
        "smoke": bool(smoke),
        "n_sequences": int(len(seqs)),
        "n_high_si_heads": int(len(high_heads)),
        # LM loss disruption comparison
        "mean_si_delta_loss": float(np.mean(si_delta)),
        "mean_perm_delta_loss": float(np.mean(perm_delta_mean)),
        "mean_const_delta_loss": float(np.mean(const_delta)),
        "si_vs_const_ratio": float(np.mean(si_delta) / np.mean(const_delta)) if float(np.mean(const_delta)) != 0.0 else float("nan"),
        "si_vs_perm_ratio": float(np.mean(si_delta) / np.mean(perm_delta_mean)) if float(np.mean(perm_delta_mean)) != 0.0 else float("nan"),
        # Entropy analysis
        "n_entropy_heads_analyzed": int(len(valid_heads)),
        "mean_entropy_delta_si": mean_ent_delta_si,
        "mean_entropy_delta_perm": mean_ent_delta_perm,
        "frac_heads_si_entropy_gt_perm": ent_si_gt_perm_frac,
        "p_one_sided_ent_si_gt_perm": float(p_ent),
        # Interpretation flags
        "position_specific_confirmed_by_entropy": bool(mean_ent_delta_si > mean_ent_delta_perm and p_ent < 0.05),
        "position_specific_confirmed_by_loss_ratio": bool(float(np.mean(si_delta)) > 2.0 * float(np.mean(const_delta))),
        "elapsed_sec": float(time.time() - t0),
    }

    write_json(model_dir / "summary.json", result)

    # Save per-sequence arrays
    seq_df = pd.DataFrame({
        "seq_idx": list(range(len(seqs))),
        "si_delta_loss": si_delta.tolist(),
        "perm_delta_loss": perm_delta_mean.tolist(),
        "const_delta_loss": const_delta.tolist(),
    })
    seq_df.to_parquet(model_dir / "per_seq_loss_deltas.parquet", index=False)

    # Save per-head entropy table
    head_df = pd.DataFrame({
        "layer": [h[0] for h in high_heads],
        "head": [h[1] for h in high_heads],
        "ent_delta_si": [ent_delta_si.get(h, float("nan")) for h in high_heads],
        "ent_delta_perm": [ent_delta_perm.get(h, float("nan")) for h in high_heads],
    })
    head_df.to_parquet(model_dir / "per_head_entropy.parquet", index=False)

    print(
        f"[{model_name}] si_delta={result['mean_si_delta_loss']:.4f} "
        f"perm_delta={result['mean_perm_delta_loss']:.4f} "
        f"const_delta={result['mean_const_delta_loss']:.4f} "
        f"si/const_ratio={result['si_vs_const_ratio']:.2f} "
        f"ent_si_gt_perm_frac={ent_si_gt_perm_frac:.2f} "
        f"p_ent={p_ent:.4f}",
        flush=True,
    )
    print(json.dumps(result, indent=2))
    return result


def run(
    *,
    models: list[str],
    device: str,
    out_root: Path,
    n_sequences: int,
    seq_len: int,
    n_perm_trials: int,
    batch_size: int,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    t0 = time.time()
    out_root = ensure_dir(out_root)

    all_results: dict[str, Any] = {}
    for model_name in models:
        print(f"\n=== Running {model_name} ===", flush=True)
        try:
            r = run_one_model(
                model_name=model_name,
                device=device,
                out_root=out_root,
                n_sequences=n_sequences,
                seq_len=seq_len,
                n_perm_trials=n_perm_trials,
                batch_size=batch_size,
                seed=seed,
                smoke=smoke,
            )
            all_results[model_name] = r
        except Exception as e:
            print(f"[ERROR] {model_name}: {e}", flush=True)
            all_results[model_name] = {"error": str(e)}

    combined = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "models": models,
        "smoke": bool(smoke),
        "per_model": all_results,
        "summary": {
            m: {
                "si_vs_const_ratio": all_results[m].get("si_vs_const_ratio"),
                "frac_heads_si_entropy_gt_perm": all_results[m].get("frac_heads_si_entropy_gt_perm"),
                "p_ent": all_results[m].get("p_one_sided_ent_si_gt_perm"),
                "position_specific_confirmed": all_results[m].get("position_specific_confirmed_by_entropy"),
            }
            for m in models if "error" not in all_results.get(m, {})
        },
        "elapsed_sec": float(time.time() - t0),
    }
    write_json(out_root / "combined_summary.json", combined)

    prereg = {
        "experiment_id": EXPERIMENT_ID,
        "question": "Does SI-kernel subtraction disrupt attention through position-specific redistribution (not just norm change)?",
        "primary_endpoints": ["si_vs_const_ratio > 2", "frac_heads_si_entropy_gt_perm > 0.5"],
    }
    summary_out = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "verdict": combined["summary"],
        "elapsed_sec": combined["elapsed_sec"],
    }
    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "impact": "Provides mechanistic grounding for SI load-bearingness claim via attention redistribution analysis",
    }
    data_dict = {
        "combined_summary.json": "Cross-model summary of redistribution audit results.",
        "<model>/summary.json": "Per-model results.",
        "<model>/per_seq_loss_deltas.parquet": "Per-sequence loss deltas for SI/perm/const conditions.",
        "<model>/per_head_entropy.parquet": "Per-head attention entropy deltas.",
    }

    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=out_root,
        preregistration=prereg,
        summary=summary_out,
        claim_impact=claim_impact,
        data_dictionary=data_dict,
        manifest_extra={"models": models, "smoke": bool(smoke)},
    )

    print(json.dumps(combined, indent=2))
    return combined


def main() -> None:
    p = argparse.ArgumentParser(description="Softmax redistribution audit")
    p.add_argument("--models", default=",".join(MODELS_TO_RUN))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--n-sequences", type=int, default=40)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-perm-trials", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260504)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    run(
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        device=str(args.device),
        out_root=Path(args.output_root),
        n_sequences=int(args.n_sequences),
        seq_len=int(args.seq_len),
        n_perm_trials=int(args.n_perm_trials),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        smoke=bool(args.smoke),
    )


if __name__ == "__main__":
    main()
