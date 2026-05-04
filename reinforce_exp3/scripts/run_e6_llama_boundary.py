#!/usr/bin/env python3
"""E6 — Llama Strict-Boundary Control Expansion.

Two-phase experiment to resolve Llama's ambiguous boundary status:
  Phase 1: 24-seed replication (8 seeds × 3 domains) of the strict-gate design
  Phase 2: Expanded 5-feature artifact-ablation battery beyond space-prefix

Usage:
    python reinforce_exp3/scripts/run_e6_llama_boundary.py \
        --models llama-3.1-8b \
        --device-map llama-3.1-8b:cuda:0 \
        [--phase 1|2|both]
"""
from __future__ import annotations

import argparse
import math
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
    EXP3P2B_ROOT,
    RESULTS_ROOT,
    ensure_dir,
    read_json,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    cohen_d,
    enforce_coverage_contract,
    emit_core_artifacts,
    holm_adjust_dict,
    load_model_for_exp,
    parse_device_map,
    parse_models_arg,
)

OUT_ROOT = RESULTS_ROOT / "E6_llama_boundary_expansion"

# Phase 1 design
DOMAINS = ("wiki", "code", "dialogue")
SEEDS_PER_DOMAIN = 8   # 24 total (vs. original 7)
BOUNDARY_D_THRESHOLD = 0.20      # d > this = boundary effect present
ARTIFACT_DELTA_THRESHOLD = 0.20  # d < this = artifact absent (gate passes)
SEED_BASE_PHASE1 = 20260429

# Phase 2 feature battery
PHASE2_FEATURES = (
    "space_prefix",
    "bpe_merge_rank",
    "subword_length",
    "intraword_position",
    "capitalization",
)
PROBE_TOP_K_DIMS = 16   # top-k embedding dimensions from probe for each feature
SEED_BASE_PHASE2 = 20260500

# Nominal sequence parameters
SEQ_LEN = 256
N_SEQS_PER_SEED = 40


# ---------------------------------------------------------------------------
# Boundary-attention measurement helpers
# ---------------------------------------------------------------------------

def _is_boundary_token(token_str: str, prev_token_str: str | None) -> bool:
    """Space-prefix heuristic for word-initial boundary token."""
    return token_str.startswith(" ") or token_str.startswith("▁")


def _compute_boundary_attention_score(
    attn_weights: np.ndarray,   # shape (seq_len, seq_len)
    boundary_mask: np.ndarray,  # shape (seq_len,) bool: True = boundary token
) -> float:
    """Mean attention weight to boundary tokens vs non-boundary tokens.

    Returns d = (mean_attn_to_boundary - mean_attn_to_nonboundary) normalised
    by pooled std.  This is equivalent to Cohen's d treating attn-to-boundary
    and attn-to-nonboundary as two samples.
    """
    attn_to_boundary = attn_weights[:, boundary_mask].mean(axis=1)   # (seq,)
    attn_to_nonboundary = attn_weights[:, ~boundary_mask].mean(axis=1)
    return cohen_d(attn_to_boundary.tolist(), attn_to_nonboundary.tolist())


# ---------------------------------------------------------------------------
# Phase 1 — 24-seed strict-gate design
# ---------------------------------------------------------------------------

def _get_boundary_mask(token_ids: list[int], tokenizer: Any) -> np.ndarray:
    strs = [tokenizer.convert_ids_to_tokens(int(tid)) for tid in token_ids]
    mask = np.zeros(len(strs), dtype=bool)
    for i, s in enumerate(strs):
        if s is not None:
            mask[i] = s.startswith(" ") or s.startswith("▁") or s.startswith("Ġ")
    return mask


def _get_fake_boundary_mask(token_ids: list[int], tokenizer: Any, rng: random.Random) -> np.ndarray:
    """Construct a fake boundary mask matched in density to the real one."""
    real_mask = _get_boundary_mask(token_ids, tokenizer)
    n_boundary = int(real_mask.sum())
    fake_mask = np.zeros(len(token_ids), dtype=bool)
    idxs = rng.sample(range(len(token_ids)), k=min(n_boundary, len(token_ids)))
    fake_mask[idxs] = True
    return fake_mask


@torch.no_grad()
def _eval_head_boundary_d(
    model: Any,
    tokenizer: Any,
    sequences: list[list[int]],
    layer: int,
    head: int,
    boundary_mask_fn: Any,  # callable(token_ids, tokenizer) -> np.ndarray
    device: str,
) -> float:
    """Compute mean boundary attention d for a specific head across sequences."""
    d_values: list[float] = []
    model.eval()

    # Hook to capture attention weights for target layer/head
    captured: dict[str, Any] = {}

    def _hook(module: Any, inp: Any, out: Any) -> None:
        if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            captured["attn"] = out[1].detach().cpu()  # (1, n_heads, seq, seq)

    handles = []
    try:
        layer_module = model.model.layers[layer]
        h = layer_module.self_attn.register_forward_hook(_hook)
        handles.append(h)

        for seq in sequences:
            captured.clear()
            ids_tensor = torch.tensor([seq], device=device, dtype=torch.long)
            model(ids_tensor, output_attentions=True)
            if "attn" not in captured:
                continue
            _t = captured["attn"][0, head].float()
            attn_weights = np.array(_t.tolist(), dtype=np.float32)  # (seq_len, seq_len)
            bmask = boundary_mask_fn(seq, tokenizer)
            if bmask.sum() == 0 or (~bmask).sum() == 0:
                continue
            d = _compute_boundary_attention_score(attn_weights, bmask)
            d_values.append(d)
    finally:
        for h in handles:
            h.remove()

    return float(np.mean(d_values)) if d_values else float("nan")


def _load_sequences_for_domain(domain: str, n: int, seq_len: int, seed: int) -> list[list[int]]:
    """Load or generate token sequences for domain evaluation.

    Tries to load from reinforce_exp2 B multiseed data; falls back to synthetic.
    """
    # Try existing 3P2-B multiseed data
    candidate = EXP3P2B_ROOT / "llama-3.1-8b" / domain
    if candidate.exists():
        parquets = list(candidate.glob("*.parquet"))
        if parquets:
            df = pd.read_parquet(parquets[0])
            if "input_ids" in df.columns:
                rng = random.Random(seed)
                rows = df["input_ids"].tolist()
                rows = [r[:seq_len] for r in rows if len(r) >= 20]
                return rng.sample(rows, k=min(n, len(rows))) if rows else []

    # Fallback: return empty (caller handles gracefully)
    return []


def run_phase1(
    model: Any,
    tokenizer: Any,
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Phase 1: 24-seed strict-gate design for Llama."""
    print("[E6] Phase 1: 24-seed strict-gate boundary analysis", flush=True)
    phase1_out = ensure_dir(out_dir / "phase1_24seed" / "llama-3.1-8b")

    # Load cluster membership to identify high-SI heads for Llama
    cluster_path = B1_RESULTS / "cluster_membership.parquet"
    if not cluster_path.exists():
        raise RuntimeError("[E6] hard_fail_reason: missing B1 cluster_membership.parquet for high-SI head definition")
    cm = pd.read_parquet(cluster_path)
    llama_cm = cm[(cm["model"] == "llama-3.1-8b") & (cm["is_high_si"] == True)].copy()
    high_si_heads = list(zip(llama_cm["layer"].tolist(), llama_cm["head"].tolist()))
    if not high_si_heads:
        raise RuntimeError("[E6] hard_fail_reason: no high-SI heads found for llama-3.1-8b")

    print(f"[E6] Phase 1: testing {len(high_si_heads)} high-SI heads", flush=True)

    domain_outcomes: dict[str, Any] = {}
    all_seed_records: list[dict[str, Any]] = []

    for domain in DOMAINS:
        seed_results: list[dict[str, Any]] = []
        for seed_i in range(SEEDS_PER_DOMAIN):
            seed = SEED_BASE_PHASE1 + (DOMAINS.index(domain) * 100) + seed_i
            rng = random.Random(seed)
            sequences = _load_sequences_for_domain(domain, N_SEQS_PER_SEED, SEQ_LEN, seed)
            if not sequences:
                raise RuntimeError(
                    f"[E6] hard_fail_reason: missing required sequences for domain={domain} seed={seed}"
                )

            # Compute mean boundary d across high-SI heads
            d_vals_real: list[float] = []
            d_vals_fake: list[float] = []
            for layer, head in high_si_heads[:20]:  # limit for tractability
                d_real = _eval_head_boundary_d(
                    model, tokenizer, sequences, layer, head,
                    _get_boundary_mask, device,
                )
                d_fake = _eval_head_boundary_d(
                    model, tokenizer, sequences, layer, head,
                    lambda ids, tok, _rng=rng: _get_fake_boundary_mask(ids, tok, _rng),
                    device,
                )
                if not (np.isnan(d_real) or np.isnan(d_fake)):
                    d_vals_real.append(d_real)
                    d_vals_fake.append(d_fake)

            if not d_vals_real:
                raise RuntimeError(
                    f"[E6] hard_fail_reason: no valid boundary d-values for domain={domain} seed={seed}"
                )

            mean_real = float(np.mean(d_vals_real))
            mean_fake = float(np.mean(d_vals_fake))
            delta_d = mean_real - mean_fake

            # Gate evaluation
            boundary_present = mean_real > BOUNDARY_D_THRESHOLD
            artifact_absent = abs(delta_d) < ARTIFACT_DELTA_THRESHOLD
            gate_status = "clean" if (boundary_present and artifact_absent) else (
                "blocked" if (boundary_present and not artifact_absent) else "no_effect"
            )

            rec = {
                "domain": domain,
                "seed": seed,
                "mean_d_real": mean_real,
                "mean_d_fake": mean_fake,
                "delta_d": delta_d,
                "boundary_present": boundary_present,
                "artifact_absent": artifact_absent,
                "gate_status": gate_status,
            }
            seed_results.append(rec)
            all_seed_records.append(rec)
            print(
                f"[E6] Phase 1 {domain} seed={seed_i}: "
                f"d_real={mean_real:.3f} delta={delta_d:.3f} status={gate_status}",
                flush=True,
            )

        # Domain-level verdict
        if not seed_results:
            raise RuntimeError(f"[E6] hard_fail_reason: no valid seed results for domain={domain}")

        n_clean = sum(1 for r in seed_results if r["gate_status"] == "clean")
        n_blocked = sum(1 for r in seed_results if r["gate_status"] == "blocked")
        n_total = len(seed_results)
        frac_clean = n_clean / n_total

        if frac_clean >= 0.75:
            verdict = "stable_clean"
        elif (n_blocked / n_total) >= 0.75:
            verdict = "stable_blocked"
        else:
            verdict = "ambiguous"

        domain_outcomes[domain] = {
            "verdict": verdict,
            "n_seeds": n_total,
            "n_clean": n_clean,
            "n_blocked": n_blocked,
            "frac_clean": frac_clean,
        }
        if n_total < SEEDS_PER_DOMAIN:
            raise RuntimeError(
                f"[E6] hard_fail_reason: insufficient seed coverage for {domain}: "
                f"have={n_total}, required={SEEDS_PER_DOMAIN}"
            )
        print(f"[E6] Phase 1 {domain}: verdict={verdict} ({n_clean}/{n_total} clean)", flush=True)

    # Save outputs
    write_json(phase1_out / "domain_outcomes.json", domain_outcomes)
    if all_seed_records:
        pd.DataFrame(all_seed_records).to_parquet(
            phase1_out / "multiseed_gate_summary.parquet", index=False
        )

    # Pooled adjudication across domains
    verdicts = [v["verdict"] for v in domain_outcomes.values()]
    n_stable_clean = sum(1 for v in verdicts if v == "stable_clean")
    n_stable_blocked = sum(1 for v in verdicts if v == "stable_blocked")
    if n_stable_clean >= 2:
        pooled_verdict = "stable_clean"
    elif n_stable_blocked >= 2:
        pooled_verdict = "stable_blocked"
    else:
        pooled_verdict = "ambiguous"

    pooled = {
        "pooled_verdict": pooled_verdict,
        "domain_verdicts": domain_outcomes,
        "n_stable_clean_domains": n_stable_clean,
        "n_stable_blocked_domains": n_stable_blocked,
    }
    enforce_coverage_contract(
        experiment_id="E6",
        observed_models=["llama-3.1-8b"],
        required_models=["llama-3.1-8b"],
        observed_tasks=list(domain_outcomes.keys()),
        required_tasks=list(DOMAINS),
        observed_counts={
            "domains": len(domain_outcomes),
            "seed_records": len(all_seed_records),
        },
        min_counts={
            "domains": len(DOMAINS),
            "seed_records": len(DOMAINS) * SEEDS_PER_DOMAIN,
        },
    )
    write_json(phase1_out / "pooled_adjudication.json", pooled)
    print(f"[E6] Phase 1 pooled verdict: {pooled_verdict}", flush=True)
    return pooled


# ---------------------------------------------------------------------------
# Phase 2 — Expanded feature ablation battery
# ---------------------------------------------------------------------------

def _compute_feature_labels(
    token_ids: list[int],
    tokenizer: Any,
    feature: str,
) -> np.ndarray:
    """Return continuous or binary feature values for each token position."""
    strs = [tokenizer.convert_ids_to_tokens(int(tid)) or "" for tid in token_ids]
    n = len(strs)

    if feature == "space_prefix":
        return np.array([
            1.0 if (s.startswith(" ") or s.startswith("▁") or s.startswith("Ġ")) else 0.0
            for s in strs
        ])

    elif feature == "bpe_merge_rank":
        # Approximate merge rank by token id (lower id ~ higher frequency ~ earlier merge)
        ranks = np.array([float(tid) / tokenizer.vocab_size for tid in token_ids])
        return 1.0 - ranks  # invert: higher = more frequent / earlier merge

    elif feature == "subword_length":
        lengths = np.array([float(len(s.strip("▁ "))) for s in strs])
        max_len = max(lengths.max(), 1.0)
        return lengths / max_len  # normalize to [0,1]

    elif feature == "intraword_position":
        # Categorical: 0=word-initial, 0.33=word-medial, 0.67=word-final, 1=whole-word
        labels = np.zeros(n, dtype=float)
        for i, s in enumerate(strs):
            is_initial = s.startswith(" ") or s.startswith("▁") or s.startswith("Ġ")
            is_whole = is_initial and (i + 1 >= n or (
                strs[i + 1].startswith(" ") or strs[i + 1].startswith("▁")
            ))
            if is_whole:
                labels[i] = 1.0
            elif is_initial:
                labels[i] = 0.0
            else:
                # Check if next token starts new word (then this is final)
                if i + 1 >= n or strs[i + 1].startswith(" ") or strs[i + 1].startswith("▁"):
                    labels[i] = 0.67
                else:
                    labels[i] = 0.33
        return labels

    elif feature == "capitalization":
        return np.array([
            1.0 if s and s.strip(" ▁")[:1].isupper() else 0.0
            for s in strs
        ])

    return np.zeros(n, dtype=float)


def _fit_feature_probe(
    embeddings: np.ndarray,  # (n_tokens, hidden_dim)
    labels: np.ndarray,       # (n_tokens,)
) -> tuple[np.ndarray, float]:
    """Fit a linear probe. Return top-k dim indices and probe accuracy."""
    from sklearn.linear_model import LogisticRegression  # local import

    # Binarize for logistic probe
    binary_labels = (labels > 0.5).astype(int)
    if binary_labels.sum() < 5 or (1 - binary_labels).sum() < 5:
        return np.array([], dtype=int), float("nan")

    try:
        clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
        clf.fit(embeddings, binary_labels)
        coef = np.abs(clf.coef_[0])
        top_k_idxs = np.argsort(coef)[-PROBE_TOP_K_DIMS:]
        acc = float((clf.predict(embeddings) == binary_labels).mean())
        return top_k_idxs, acc
    except Exception:
        return np.array([], dtype=int), float("nan")


@torch.no_grad()
def _collect_embeddings(
    model: Any,
    sequences: list[list[int]],
    device: str,
) -> tuple[np.ndarray, list[list[int]]]:
    """Collect last-layer embeddings for all token positions across sequences."""
    model.eval()
    all_embeds: list[np.ndarray] = []
    all_ids: list[list[int]] = []
    for seq in sequences:
        ids_tensor = torch.tensor([seq], device=device, dtype=torch.long)
        out = model(ids_tensor, output_hidden_states=True)
        # last hidden state: (1, seq_len, hidden_dim)
        _t = out.hidden_states[-1][0].cpu().float()
        embeds = np.array(_t.tolist(), dtype=np.float32)
        all_embeds.append(embeds)
        all_ids.extend([seq])
    return np.concatenate(all_embeds, axis=0), all_ids


def _ablate_dimensions_and_eval(
    model: Any,
    tokenizer: Any,
    sequences: list[list[int]],
    top_k_dims: np.ndarray,
    high_si_heads: list[tuple[int, int]],
    device: str,
    rng: random.Random,
) -> dict[str, float]:
    """Zero out top-k embedding dimensions and re-evaluate boundary d.

    Returns {'d_real_ablated': ..., 'd_fake_ablated': ..., 'delta_d_ablated': ...}
    """
    if len(top_k_dims) == 0:
        return {"d_real_ablated": float("nan"), "d_fake_ablated": float("nan")}

    # Hook to zero out dimensions in input embeddings
    handles: list[Any] = []
    dims_set = set(int(d) for d in top_k_dims)

    def _embed_hook(module: Any, inp: Any, out: Any) -> torch.Tensor:
        out = out.clone()
        for d in dims_set:
            if d < out.shape[-1]:
                out[:, :, d] = 0.0
        return out

    try:
        h = model.model.embed_tokens.register_forward_hook(_embed_hook)
        handles.append(h)

        d_vals_real: list[float] = []
        d_vals_fake: list[float] = []
        for layer, head in high_si_heads[:10]:
            d_real = _eval_head_boundary_d(
                model, tokenizer, sequences, layer, head,
                _get_boundary_mask, device,
            )
            d_fake = _eval_head_boundary_d(
                model, tokenizer, sequences, layer, head,
                lambda ids, tok, _rng=rng: _get_fake_boundary_mask(ids, tok, _rng),
                device,
            )
            if not (np.isnan(d_real) or np.isnan(d_fake)):
                d_vals_real.append(d_real)
                d_vals_fake.append(d_fake)
    finally:
        for h in handles:
            h.remove()

    if not d_vals_real:
        return {"d_real_ablated": float("nan"), "d_fake_ablated": float("nan")}

    d_real_abl = float(np.mean(d_vals_real))
    d_fake_abl = float(np.mean(d_vals_fake))
    return {
        "d_real_ablated": d_real_abl,
        "d_fake_ablated": d_fake_abl,
        "delta_d_ablated": d_real_abl - d_fake_abl,
    }


def run_phase2(
    model: Any,
    tokenizer: Any,
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Phase 2: Expanded 5-feature artifact ablation battery."""
    print("[E6] Phase 2: Expanded feature ablation battery", flush=True)
    phase2_out = ensure_dir(out_dir / "phase2_expanded_battery")

    # Load high-SI heads
    cluster_path = B1_RESULTS / "cluster_membership.parquet"
    if not cluster_path.exists():
        raise RuntimeError("[E6] hard_fail_reason: missing B1 cluster_membership.parquet for phase2")
    cm = pd.read_parquet(cluster_path)
    llama_cm = cm[(cm["model"] == "llama-3.1-8b") & (cm["is_high_si"] == True)].copy()
    high_si_heads = list(zip(llama_cm["layer"].tolist(), llama_cm["head"].tolist()))
    if not high_si_heads:
        raise RuntimeError("[E6] hard_fail_reason: no high-SI heads for phase2")

    # Load wiki sequences for phase 2
    sequences = _load_sequences_for_domain("wiki", N_SEQS_PER_SEED * 2, SEQ_LEN, SEED_BASE_PHASE2)
    if not sequences:
        raise RuntimeError("[E6] hard_fail_reason: missing phase2 wiki sequences")

    rng = random.Random(SEED_BASE_PHASE2)

    # Collect embeddings once
    print("[E6] Phase 2: collecting embeddings", flush=True)
    all_embeds, all_ids_list = _collect_embeddings(model, sequences, device)

    feature_results: dict[str, Any] = {}
    probes_dir = ensure_dir(phase2_out / "feature_classifiers")

    for feature in PHASE2_FEATURES:
        print(f"[E6] Phase 2: fitting probe for {feature}", flush=True)
        # Build feature labels for all collected token positions
        all_labels: list[float] = []
        for seq in all_ids_list:
            feature_vals = _compute_feature_labels(seq, tokenizer, feature)
            all_labels.extend(feature_vals.tolist())
        label_arr = np.array(all_labels)

        # Trim to match embeddings shape
        min_n = min(len(label_arr), all_embeds.shape[0])
        embeds_trim = all_embeds[:min_n]
        labels_trim = label_arr[:min_n]

        top_k_dims, probe_acc = _fit_feature_probe(embeds_trim, labels_trim)
        if len(top_k_dims) == 0 or not np.isfinite(probe_acc):
            raise RuntimeError(
                f"[E6] hard_fail_reason: probe fitting failed for feature={feature}"
            )
        probe_info = {
            "feature": feature,
            "probe_accuracy": probe_acc,
            "top_k_dims": top_k_dims.tolist() if len(top_k_dims) > 0 else [],
            "probe_top_k": PROBE_TOP_K_DIMS,
        }
        write_json(probes_dir / f"{feature}_probe.json", probe_info)

        # Ablate feature dimensions and re-evaluate boundary d
        ablation_result = _ablate_dimensions_and_eval(
            model, tokenizer, sequences, top_k_dims, high_si_heads, device, rng
        )

        boundary_survives = (
            ablation_result.get("d_real_ablated", float("nan")) > BOUNDARY_D_THRESHOLD
            and abs(ablation_result.get("delta_d_ablated", float("nan"))) < ARTIFACT_DELTA_THRESHOLD
        )

        feature_results[feature] = {
            **probe_info,
            **ablation_result,
            "boundary_survives_ablation": boundary_survives,
        }
        print(
            f"[E6] Phase 2 {feature}: "
            f"probe_acc={probe_acc:.3f} "
            f"d_abl={ablation_result.get('d_real_ablated', float('nan')):.3f} "
            f"survives={boundary_survives}",
            flush=True,
        )

    n_surviving = sum(1 for r in feature_results.values() if r.get("boundary_survives_ablation"))

    if n_surviving >= 3:
        expanded_gate_verdict = "conditionally_clean"
        note = f"Boundary survives {n_surviving}/5 feature ablations"
    elif n_surviving == 0:
        expanded_gate_verdict = "fully_blocked"
        note = "Boundary blocked by all 5 feature ablations"
    else:
        expanded_gate_verdict = "partially_surviving"
        note = f"Boundary survives {n_surviving}/5 feature ablations (below conditionally_clean threshold)"

    expanded_summary = {
        "verdict": expanded_gate_verdict,
        "n_surviving_features": n_surviving,
        "n_features_tested": len(PHASE2_FEATURES),
        "note": note,
        "per_feature": feature_results,
    }
    enforce_coverage_contract(
        experiment_id="E6",
        observed_models=["llama-3.1-8b"],
        required_models=["llama-3.1-8b"],
        observed_tasks=list(feature_results.keys()),
        required_tasks=list(PHASE2_FEATURES),
        observed_counts={"features_tested": len(feature_results)},
        min_counts={"features_tested": len(PHASE2_FEATURES)},
    )
    write_json(phase2_out / "expanded_gate_summary.json", expanded_summary)

    # Write per-feature boundary ablation table
    rows = []
    for feat, r in feature_results.items():
        rows.append({
            "feature": feat,
            "d_real_before": float("nan"),  # not computed here, from phase1
            "d_real_after_ablation": r.get("d_real_ablated", float("nan")),
            "d_fake_after_ablation": r.get("d_fake_ablated", float("nan")),
            "boundary_survives": r.get("boundary_survives_ablation", False),
        })
    pd.DataFrame(rows).to_parquet(
        phase2_out / "boundary_per_feature_ablation.parquet", index=False
    )
    write_json(phase2_out / "boundary_per_feature_ablation.json", {"rows": rows})

    print(f"[E6] Phase 2 verdict: {expanded_gate_verdict}", flush=True)
    return expanded_summary


# ---------------------------------------------------------------------------
# Combined verdict
# ---------------------------------------------------------------------------

def _combined_verdict(
    phase1: dict[str, Any],
    phase2: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    p1_verdict = phase1.get("pooled_verdict", "not_run")
    p2_verdict = phase2.get("verdict", "not_run")

    if p1_verdict == "stable_clean":
        llama_boundary_status = "confirmed_clean"
        note = "Llama passes 24-seed strict gate in ≥2/3 domains."
        claim_status = "supported"
    elif p1_verdict == "stable_blocked" and p2_verdict in ("fully_blocked", "partially_surviving"):
        llama_boundary_status = "confirmed_blocked"
        note = "Llama fails strict gate consistently across 24 seeds and expanded features."
        claim_status = "not_supported"
    elif p1_verdict == "ambiguous" and p2_verdict == "conditionally_clean":
        llama_boundary_status = "conditionally_clean"
        note = (
            "Llama ambiguous on 24-seed gate but boundary survives ≥3/5 expanded feature ablations. "
            "Partial support — boundary not driven by tested artifact features."
        )
        claim_status = "mixed"
    else:
        llama_boundary_status = "ambiguous"
        note = (
            f"Phase 1: {p1_verdict}, Phase 2: {p2_verdict}. "
            "Outcome is inconclusive. Report with quantified uncertainty."
        )
        claim_status = "inconclusive"

    combined = {
        "llama_boundary_status": llama_boundary_status,
        "phase1_verdict": p1_verdict,
        "phase2_verdict": p2_verdict,
        "note": note,
        "claim_status": claim_status,
    }
    write_json(out_dir / "combined_verdict.json", combined)
    print(f"[E6] Combined verdict: {llama_boundary_status}", flush=True)
    return combined


# ---------------------------------------------------------------------------
# Artifact emission
# ---------------------------------------------------------------------------

def _emit_artifacts(
    out_dir: Path,
    combined: dict[str, Any],
    phase_run: str,
    start_ts: str,
) -> None:
    claim_status = combined.get("claim_status", "inconclusive")
    supports = claim_status in ("supported", "mixed")

    claim_impact = {
        "experiment_id": "E6",
        "claim_addressed": (
            "Llama-3.1-8B boundary attention status under strict tokenizer-feature gate "
            "(resolves current ambiguous verdict from reinforce_exp2)"
        ),
        "claim_status": claim_status,
        "supports_main_text": supports,
        "outcome_summary": combined.get("note", ""),
        "llama_boundary_status": combined.get("llama_boundary_status", "unknown"),
        "notes": [combined.get("note", "")],
    }

    preregistration = {
        "experiment_id": "E6",
        "hypothesis": (
            "Increasing seed count to 24 (Phase 1) or applying expanded 5-feature artifact "
            "battery (Phase 2) will resolve Llama's ambiguous boundary status to a stable verdict."
        ),
        "primary_criterion": (
            "Phase 1: stable_clean or stable_blocked in ≥2/3 domains with n=24 seeds. "
            "Phase 2: boundary survives ≥3/5 feature ablations = conditionally_clean."
        ),
        "model": "llama-3.1-8b",
        "phases": phase_run,
        "domains": list(DOMAINS),
        "seeds_per_domain": SEEDS_PER_DOMAIN,
        "boundary_d_threshold": BOUNDARY_D_THRESHOLD,
        "artifact_delta_threshold": ARTIFACT_DELTA_THRESHOLD,
        "phase2_features": list(PHASE2_FEATURES),
        "timestamp": start_ts,
    }

    summary = {
        "experiment_id": "E6",
        "status": "complete",
        "model": "llama-3.1-8b",
        "phase_run": phase_run,
        "llama_boundary_status": combined.get("llama_boundary_status", "unknown"),
        "claim_status": claim_status,
        "timestamp_start": start_ts,
        "timestamp_end": timestamp_now(),
    }

    data_dictionary = {
        "experiment_id": "E6",
        "tables": [
            {
                "path": "phase1_24seed/llama-3.1-8b/multiseed_gate_summary.parquet",
                "description": "Per-seed gate status across 24 seeds and 3 domains",
                "columns": [
                    {"name": "domain", "dtype": "str", "description": "Evaluation domain"},
                    {"name": "seed", "dtype": "int", "description": "Random seed"},
                    {"name": "mean_d_real", "dtype": "float", "description": "Mean boundary Cohen d (real boundary mask)"},
                    {"name": "mean_d_fake", "dtype": "float", "description": "Mean boundary Cohen d (shuffled/fake mask)"},
                    {"name": "delta_d", "dtype": "float", "description": "d_real - d_fake"},
                    {"name": "gate_status", "dtype": "str", "description": "clean / blocked / no_effect"},
                ],
            },
            {
                "path": "phase2_expanded_battery/boundary_per_feature_ablation.parquet",
                "description": "Boundary d values after ablating each feature's predictive dimensions",
                "columns": [
                    {"name": "feature", "dtype": "str", "description": "Feature name"},
                    {"name": "d_real_after_ablation", "dtype": "float", "description": "Boundary d after feature dims ablated"},
                    {"name": "d_fake_after_ablation", "dtype": "float", "description": "Fake-boundary d after feature dims ablated"},
                    {"name": "boundary_survives", "dtype": "bool", "description": "True if boundary effect persists after ablation"},
                ],
            },
        ],
    }

    manifest_extra = {
        "model": "llama-3.1-8b",
        "phase_run": phase_run,
        "domains": list(DOMAINS),
        "seeds_per_domain": SEEDS_PER_DOMAIN,
        "phase2_features": list(PHASE2_FEATURES),
        "seed_base_phase1": SEED_BASE_PHASE1,
        "seed_base_phase2": SEED_BASE_PHASE2,
    }

    emit_core_artifacts(
        out_dir=out_dir,
        experiment_id="E6",
        preregistration=preregistration,
        manifest_extra=manifest_extra,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="E6: Llama boundary expansion (24-seed + feature battery)",
        allow_abbrev=False,
    )
    p.add_argument("--models", default="llama-3.1-8b")
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0")
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument(
        "--phase",
        default="both",
        choices=["1", "2", "both"],
        help="Which phase(s) to run",
    )
    p.add_argument("--finalize-only", action="store_true",
                   help="Emit top-level governance artifacts from existing phase outputs only.")
    p.add_argument("--no-finalize", action="store_true",
                   help="Run phase computation but skip governance artifact finalize.")
    args = p.parse_args()
    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E6] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")
    requested_models = parse_models_arg(args.models, default=["llama-3.1-8b"])
    if requested_models != ["llama-3.1-8b"]:
        raise RuntimeError(
            f"[E6] hard_fail_reason: E6 only supports llama-3.1-8b, got {requested_models}"
        )

    device_map = parse_device_map(args.device_map)
    device = device_map.get("llama-3.1-8b", "cuda:0")
    out_dir = ensure_dir(Path(args.output_root))
    start_ts = timestamp_now()

    print(f"[E6] Starting at {start_ts}", flush=True)

    if args.finalize_only:
        phase1_result: dict[str, Any] = {}
        phase2_result: dict[str, Any] = {}
        phase1_path = out_dir / "phase1_24seed" / "llama-3.1-8b" / "pooled_adjudication.json"
        phase2_path = out_dir / "phase2_expanded_battery" / "expanded_gate_summary.json"
        if args.phase in ("1", "both"):
            if not phase1_path.exists():
                raise RuntimeError(f"[E6] hard_fail_reason: missing {phase1_path}")
            phase1_result = read_json(phase1_path)
        if args.phase in ("2", "both"):
            if not phase2_path.exists():
                raise RuntimeError(f"[E6] hard_fail_reason: missing {phase2_path}")
            phase2_result = read_json(phase2_path)
        combined = _combined_verdict(phase1_result, phase2_result, out_dir)
        _emit_artifacts(out_dir, combined, args.phase, start_ts)
        print(f"[E6] Finalized from existing phase artifacts.", flush=True)
        return

    print(f"[E6] Loading llama-3.1-8b on {device}", flush=True)
    model, tokenizer = load_model_for_exp("llama-3.1-8b", device, attn_implementation="eager")

    phase1_result: dict[str, Any] = {}
    phase2_result: dict[str, Any] = {}

    if args.phase in ("1", "both"):
        phase1_result = run_phase1(model, tokenizer, device, out_dir)

    if args.phase in ("2", "both"):
        phase2_result = run_phase2(model, tokenizer, device, out_dir)

    combined = _combined_verdict(phase1_result, phase2_result, out_dir)
    if not args.no_finalize:
        _emit_artifacts(out_dir, combined, args.phase, start_ts)
    else:
        print("[E6] Shard run complete (no finalize).", flush=True)

    print(f"[E6] Done. Llama boundary status: {combined['llama_boundary_status']}", flush=True)


if __name__ == "__main__":
    main()
