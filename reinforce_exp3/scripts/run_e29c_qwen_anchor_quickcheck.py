#!/usr/bin/env python3
"""E29C — Extra-model anchor quickcheck on Qwen2.5-7B.

Quickcheck includes:
1) Per-head SI R² profile on wiki sequences.
2) One grouped high-SI kernel-subtraction disruption check with permuted control.

This is directional corroboration only (non-primary model).
"""
from __future__ import annotations

import argparse
import json
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

from reinforce_exp3.common import RESULTS_ROOT, ensure_dir, timestamp_now, write_json  # noqa: E402
from reinforce_exp3.scripts._shared import emit_core_artifacts, parse_device_map  # noqa: E402
from experiment3.theory1_si_circuits import compute_per_head_r2, classify_heads  # noqa: E402
from experiment3.theory8_position_ablation import compute_per_token_loss, subtract_positional_kernels  # noqa: E402
from experiment5.pipeline import _load_wiki_sequences  # noqa: E402
from experiment1.shift_kernels import get_kernel_estimator  # noqa: E402
from shared.attention.adapters import get_adapter  # noqa: E402
from shared.specs import ModelSpec  # noqa: E402
from shared.models.loading import load_model, load_tokenizer  # noqa: E402
from experiment1.norm_utils import normalize_logits_for_norm  # noqa: E402

EXPERIMENT_ID = "E29C"
DEFAULT_OUT = RESULTS_ROOT / "E29c_qwen_anchor_quickcheck"
DEFAULT_MODEL = "qwen2.5-7b"
QWEN_SPEC = ModelSpec(
    name=DEFAULT_MODEL,
    hf_id="Qwen/Qwen2.5-7B",
    norm="RMSNorm",
    pe_scheme="RoPE",
    notes="Directional quickcheck anchor for E29.",
)


def _eval_sequence_mean_losses(model: Any, sequences: list[list[int]], device: str, batch_size: int) -> np.ndarray:
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
            seq_loss = loss.view(input_ids.shape[0], tok_per_seq).mean(dim=1).detach().cpu().numpy().astype(np.float64)
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


def _permute_kernels(
    kernels: dict[tuple[int, int], np.ndarray],
    heads: list[tuple[int, int]],
    seed: int,
) -> dict[tuple[int, int], np.ndarray]:
    rng = np.random.default_rng(int(seed))
    out: dict[tuple[int, int], np.ndarray] = {}
    for head in heads:
        g = kernels.get(head)
        if g is None:
            continue
        out[head] = g[rng.permutation(len(g))].copy()
    return out


def _compute_r2_fallback(
    *,
    model: Any,
    spec: ModelSpec,
    sequences: list[list[int]],
    device: str,
) -> pd.DataFrame:
    estimator = get_kernel_estimator(spec.pe_scheme)
    rows: list[dict[str, Any]] = []
    for sidx, seq in enumerate(sequences):
        input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False, output_attentions=True)
        if out.attentions is None or len(out.attentions) == 0:
            continue
        for layer_idx, attn in enumerate(out.attentions):
            att = attn[0].float().detach().cpu()
            for head_idx in range(att.shape[0]):
                probs = torch.clamp(att[head_idx], min=1e-6)
                logits = torch.log(probs)
                prepared = normalize_logits_for_norm(logits, spec.norm)
                fit = estimator.fit_logits(prepared)
                rows.append(
                    {
                        "sequence_id": int(sidx),
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                        "r2": float(fit.r2),
                    }
                )
        del input_ids, out
        torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError("[E29C] fallback R² computation produced no rows")
    return pd.DataFrame(rows)


def _estimate_kernels_from_attn(
    *,
    model: Any,
    sequences: list[list[int]],
    device: str,
) -> dict[tuple[int, int], np.ndarray]:
    kernels: dict[tuple[int, int], np.ndarray] = {}
    counts: dict[tuple[int, int], int] = {}

    for seq in sequences:
        input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False, output_attentions=True)
        if out.attentions is None:
            continue
        for lidx, attn in enumerate(out.attentions):
            # attn: [batch, heads, S, S] probabilities
            probs = torch.clamp(attn[0].float().detach().cpu(), min=1e-6)
            logits = torch.log(probs).numpy()
            n_heads = logits.shape[0]
            s = logits.shape[1]
            for hidx in range(n_heads):
                mat = logits[hidx]
                diag_means = np.zeros(s, dtype=np.float64)
                for d in range(s):
                    vals = np.diag(mat, k=-d)
                    if vals.size > 0:
                        diag_means[d] = float(np.mean(vals))
                key = (int(lidx), int(hidx))
                if key not in kernels:
                    kernels[key] = np.zeros(s, dtype=np.float64)
                    counts[key] = 0
                kernels[key] += diag_means
                counts[key] += 1
        del input_ids, out
        torch.cuda.empty_cache()

    out_k: dict[tuple[int, int], np.ndarray] = {}
    for key, arr in kernels.items():
        c = max(1, int(counts.get(key, 1)))
        out_k[key] = (arr / c).astype(np.float32)
    if not out_k:
        raise RuntimeError("[E29C] kernel estimation produced no kernels")
    return out_k


def run_qwen(
    *,
    model_name: str,
    device: str,
    out_root: Path,
    profile_seqs: int,
    eval_seqs: int,
    seq_len: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    t0 = time.time()
    model_dir = ensure_dir(out_root / model_name)

    spec = QWEN_SPEC if model_name == DEFAULT_MODEL else ModelSpec(
        name=model_name,
        hf_id=f"Qwen/{model_name}",
        norm="RMSNorm",
        pe_scheme="RoPE",
    )

    tokenizer = load_tokenizer(spec)
    loaded = load_model(spec, device_map=device, attn_implementation="eager")
    model = loaded.model
    model.eval()

    seqs_profile = _load_wiki_sequences(
        model_name,
        tokenizer,
        seq_len=max(128, int(seq_len)),
        max_sequences=max(8, int(profile_seqs)),
        seed=int(seed),
    )
    seqs_eval = _load_wiki_sequences(
        model_name,
        tokenizer,
        seq_len=max(128, int(seq_len)),
        max_sequences=max(8, int(eval_seqs)),
        seed=int(seed) + 17,
    )
    if len(seqs_profile) < 4 or len(seqs_eval) < 4:
        raise RuntimeError(f"[E29C] insufficient profile/eval sequences for {model_name}")

    # R² profile (adapter path first, then fallback).
    adapter_error = None
    try:
        adapter = get_adapter(spec)
        adapter.register(model)
        r2_df = compute_per_head_r2(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            model_spec=spec,
            device=device,
            sequences=seqs_profile,
        )
        try:
            adapter.cleanup()
        except Exception:
            pass
    except Exception as exc:
        adapter_error = f"{type(exc).__name__}: {exc}"
        r2_df = _compute_r2_fallback(model=model, spec=spec, sequences=seqs_profile, device=device)

    high_si, low_si, mean_r2 = classify_heads(r2_df)
    mean_r2.to_parquet(model_dir / "head_r2_summary.parquet", index=False)

    # Kernel estimate from attention logits fallback path for portability.
    kernels = _estimate_kernels_from_attn(model=model, sequences=seqs_profile[: max(4, min(16, len(seqs_profile)))], device=device)

    high_heads = [(int(h.layer), int(h.head)) for h in high_si if (int(h.layer), int(h.head)) in kernels]
    low_heads = [(int(h.layer), int(h.head)) for h in low_si if (int(h.layer), int(h.head)) in kernels]
    if not high_heads:
        raise RuntimeError(f"[E29C] no high_si heads overlapped with estimated kernels for {model_name}")

    baseline = _eval_sequence_mean_losses(model, seqs_eval[:eval_seqs], device, batch_size)

    with subtract_positional_kernels(model, kernels, high_heads, int(seq_len)):
        loss_true = _eval_sequence_mean_losses(model, seqs_eval[:eval_seqs], device, batch_size)

    k_perm = _permute_kernels(kernels, high_heads, seed + 101)
    with subtract_positional_kernels(model, k_perm, high_heads, int(seq_len)):
        loss_perm = _eval_sequence_mean_losses(model, seqs_eval[:eval_seqs], device, batch_size)

    delta_true = loss_true - baseline
    delta_perm = loss_perm - baseline

    rec = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "model": model_name,
        "profile_sequences": int(len(seqs_profile)),
        "eval_sequences": int(min(eval_seqs, len(seqs_eval))),
        "n_total_heads": int(len(mean_r2)),
        "n_high_si_heads": int(len(high_heads)),
        "n_low_si_heads": int(len(low_heads)),
        "si_mean_r2": float(mean_r2["mean_r2"].mean()),
        "si_r2_std": float(mean_r2["mean_r2"].std()),
        "true_mean_delta": float(np.mean(delta_true)),
        "perm_mean_delta": float(np.mean(delta_perm)),
        "delta_true_minus_perm": float(np.mean(delta_true) - np.mean(delta_perm)),
        "true_over_perm": float(np.mean(delta_true) / np.mean(delta_perm)) if abs(float(np.mean(delta_perm))) > 1e-12 else float("nan"),
        "directional_support": bool(float(np.mean(delta_true)) > float(np.mean(delta_perm))),
        "adapter_fallback_used": bool(adapter_error is not None),
        "adapter_error": adapter_error,
        "elapsed_sec": float(time.time() - t0),
    }
    write_json(model_dir / "summary.json", rec)

    # Save estimated kernels for reproducibility.
    kernels_out = {f"L{l}H{h}": v.tolist() for (l, h), v in kernels.items()}
    write_json(model_dir / "estimated_kernels.json", kernels_out)

    print(
        f"[E29C][{model_name}] mean_r2={rec['si_mean_r2']:.4f} "
        f"true={rec['true_mean_delta']:.6f} perm={rec['perm_mean_delta']:.6f} "
        f"d={rec['delta_true_minus_perm']:.6f}",
        flush=True,
    )
    return rec


def _finalize(models: list[str], out_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for m in models:
        p = out_root / m / "summary.json"
        if not p.exists():
            raise RuntimeError(f"[E29C] hard_fail_reason: missing summary for {m}: {p}")
        rows.append(json.loads(p.read_text(encoding="utf-8")))

    n_pass = int(sum(1 for r in rows if bool(r.get("directional_support", False))))
    interp = "qwen_anchor_directional_corroboration" if n_pass >= 1 else "qwen_anchor_no_directional_corroboration"
    status = "supported_with_caveat" if n_pass >= 1 else "not_supported"

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
        "question": "Does an extra 7B RoPE model show directional SI amplitude + disruption corroboration?",
        "primary_hypothesis": "Qwen2.5-7B exhibits non-trivial SI R² and true>permuted disruption under high-SI subtraction.",
        "primary_endpoints": ["si_mean_r2", "true_mean_delta", "perm_mean_delta", "delta_true_minus_perm"],
    }

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "analysis_tier": "quick_anchor",
        "canonical_eligible": True,
        "override_used": False,
        "verdict": {
            "interpretation": interp,
            "n_model_pass": int(n_pass),
            "n_models": int(len(rows)),
        },
        "limitations": [
            "Single extra-model quickcheck; non-primary corroboration only.",
            "Kernel estimation is lightweight and not a full replication pipeline.",
        ],
    }

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": EXPERIMENT_ID,
        "claim_status": status,
        "supports_main_text": True,
        "notes": [
            "Directional extra-model anchor only; does not alter primary-model coverage contract.",
        ],
    }

    data_dictionary = {
        "experiment_id": EXPERIMENT_ID,
        "tables": [
            {
                "path": "<model>/head_r2_summary.parquet",
                "description": "Per-head SI R² profile for the Qwen anchor model.",
                "columns": [],
            },
            {
                "path": "<model>/estimated_kernels.json",
                "description": "Quick estimated kernels used for grouped subtraction checks.",
                "columns": [],
            },
        ],
    }

    emit_core_artifacts(
        experiment_id=EXPERIMENT_ID,
        out_dir=out_root,
        preregistration=prereg,
        manifest_extra={"models": models},
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )
    return cross


def main() -> None:
    p = argparse.ArgumentParser(description="E29C: Qwen2.5-7B anchor quickcheck", allow_abbrev=False)
    p.add_argument("--models", default=DEFAULT_MODEL)
    p.add_argument("--device-map", default=f"{DEFAULT_MODEL}:cuda:0")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--profile-seqs", type=int, default=24)
    p.add_argument("--eval-seqs", type=int, default=48)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260504)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize-only", action="store_true")
    p.add_argument("--no-finalize", action="store_true")
    args = p.parse_args()

    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E29C] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")

    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    device_map = parse_device_map(args.device_map)
    out_root = ensure_dir(Path(args.output_root))

    profile_seqs = int(args.profile_seqs)
    eval_seqs = int(args.eval_seqs)
    batch_size = int(args.batch_size)
    if args.smoke:
        profile_seqs = min(profile_seqs, 8)
        eval_seqs = min(eval_seqs, 12)
        batch_size = min(batch_size, 2)

    if args.finalize_only:
        cross = _finalize(models, out_root)
        print(f"[E29C] Finalized. interpretation={cross['interpretation']}", flush=True)
        return

    for model_name in models:
        device = device_map.get(model_name, "cuda:0")
        print(f"[E29C] Running {model_name} on {device}", flush=True)
        run_qwen(
            model_name=model_name,
            device=device,
            out_root=out_root,
            profile_seqs=max(4, profile_seqs),
            eval_seqs=max(4, eval_seqs),
            seq_len=max(128, int(args.seq_len)),
            batch_size=max(1, batch_size),
            seed=int(args.seed),
        )

    if args.no_finalize:
        print("[E29C] Shard complete (no finalize).", flush=True)
        return

    cross = _finalize(models, out_root)
    print(f"[E29C] Done. interpretation={cross['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
