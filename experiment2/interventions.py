from __future__ import annotations

import math
import random
import re
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from scipy.fft import dct, idct


@dataclass(frozen=True)
class InterventionPlan:
    name: str
    kind: str  # none | rope_qk | abspe_dct
    severity: str  # none | medium | strong
    target: str  # none | high | low | random
    removed_indices: tuple[int, ...]
    reference_indices_high: tuple[int, ...]
    reference_indices_low: tuple[int, ...]
    overlap_high_count: int
    overlap_low_count: int
    overlap_high_fraction: float
    overlap_low_fraction: float
    overlap_high_jaccard: float
    overlap_low_jaccard: float
    nested_medium_in_strong: bool
    band_size: int
    component_count: int
    random_draw: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InterventionAudit:
    plan: InterventionPlan
    drift_median_q: float | None = None
    drift_median_k: float | None = None
    drift_median_abspe: float | None = None
    hook_calls_q: int = 0
    hook_calls_k: int = 0
    target_scope: str | None = None
    target_head_group: str | None = None
    target_layers: list[int] | None = None
    target_query_heads: int | None = None
    target_kv_heads: int | None = None
    num_query_heads: int | None = None
    num_key_value_heads: int | None = None
    gqa_repeat_factor: int | None = None
    kv_sharing_effect_note: str | None = None
    norm_drift_pass: bool | None = None
    notes: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plan"] = self.plan.to_json_dict()
        return payload


def build_rope_intervention_plan(
    head_dim: int,
    intervention: str,
    seed: int,
    *,
    random_draw: int | None = None,
) -> InterventionPlan:
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
    d_pairs = head_dim // 2
    high_band, low_band = _split_bands(d_pairs)
    high_medium, high_strong = _severity_sets(high_band, mode="high")
    low_medium, low_strong = _severity_sets(low_band, mode="low")

    pair_match = re.fullmatch(r"ablate_pair_(\d+)", str(intervention))

    if intervention == "none":
        removed = tuple()
        severity = "none"
        target = "none"
        ref_high = tuple()
        ref_low = tuple()
    elif pair_match is not None:
        pair_idx = int(pair_match.group(1))
        if not (0 <= pair_idx < d_pairs):
            raise ValueError(f"Pair index out of bounds for head_dim={head_dim}: {pair_idx}")
        removed = (pair_idx,)
        severity = "pair"
        target = "single_pair"
        ref_high = high_band
        ref_low = low_band
    elif intervention == "random_pair":
        severity = "pair"
        target = "random_pair"
        rng = random.Random(_stable_seed("random_rope_pair", seed, intervention, d_pairs, random_draw))
        removed = (int(rng.randrange(d_pairs)),)
        ref_high = high_band
        ref_low = low_band
    elif intervention == "ablate_high_medium":
        removed = high_medium
        severity = "medium"
        target = "high"
        ref_high = high_medium
        ref_low = low_medium
    elif intervention == "ablate_high_strong":
        removed = high_strong
        severity = "strong"
        target = "high"
        ref_high = high_strong
        ref_low = low_strong
    elif intervention == "ablate_low_medium":
        removed = low_medium
        severity = "medium"
        target = "low"
        ref_high = high_medium
        ref_low = low_medium
    elif intervention == "ablate_low_strong":
        removed = low_strong
        severity = "strong"
        target = "low"
        ref_high = high_strong
        ref_low = low_strong
    elif intervention in {"random_medium", "random_strong"}:
        severity = "medium" if intervention.endswith("medium") else "strong"
        target = "random"
        count = len(high_medium) if severity == "medium" else len(high_strong)
        rng = random.Random(_stable_seed("random_rope", seed, intervention, d_pairs, random_draw))
        removed = tuple(sorted(rng.sample(range(d_pairs), k=count)))
        ref_high = high_medium if severity == "medium" else high_strong
        ref_low = low_medium if severity == "medium" else low_strong
    else:
        raise ValueError(f"Unsupported intervention: {intervention}")

    overlap_high = len(set(removed) & set(ref_high)) if ref_high else 0
    overlap_low = len(set(removed) & set(ref_low)) if ref_low else 0
    overlap_high_fraction = overlap_high / max(len(ref_high), 1)
    overlap_low_fraction = overlap_low / max(len(ref_low), 1)
    overlap_high_jaccard = _jaccard(removed, ref_high)
    overlap_low_jaccard = _jaccard(removed, ref_low)
    return InterventionPlan(
        name=intervention,
        kind="none" if intervention == "none" else "rope_qk",
        severity=severity,
        target=target,
        removed_indices=removed,
        reference_indices_high=ref_high,
        reference_indices_low=ref_low,
        overlap_high_count=overlap_high,
        overlap_low_count=overlap_low,
        overlap_high_fraction=overlap_high_fraction,
        overlap_low_fraction=overlap_low_fraction,
        overlap_high_jaccard=overlap_high_jaccard,
        overlap_low_jaccard=overlap_low_jaccard,
        nested_medium_in_strong=(set(high_medium).issubset(set(high_strong)) and set(low_medium).issubset(set(low_strong))),
        band_size=len(high_band),
        component_count=d_pairs,
        random_draw=random_draw,
    )


def build_abspe_dct_plan(
    n_positions: int,
    intervention: str,
    seed: int,
    *,
    random_draw: int | None = None,
) -> InterventionPlan:
    high_band, low_band = _split_bands(n_positions)
    high_medium, high_strong = _severity_sets(high_band, mode="high")
    low_medium, low_strong = _severity_sets(low_band, mode="low")

    pair_match = re.fullmatch(r"ablate_pair_(\d+)", str(intervention))

    if intervention == "none":
        removed = tuple()
        severity = "none"
        target = "none"
        ref_high = tuple()
        ref_low = tuple()
    elif pair_match is not None:
        pair_idx = int(pair_match.group(1))
        if not (0 <= pair_idx < n_positions):
            raise ValueError(f"Pair index out of bounds for n_positions={n_positions}: {pair_idx}")
        removed = (pair_idx,)
        severity = "pair"
        target = "single_pair"
        ref_high = high_band
        ref_low = low_band
    elif intervention == "random_pair":
        severity = "pair"
        target = "random_pair"
        rng = random.Random(_stable_seed("random_abspe_pair", seed, intervention, n_positions, random_draw))
        removed = (int(rng.randrange(n_positions)),)
        ref_high = high_band
        ref_low = low_band
    elif intervention == "ablate_high_medium":
        removed = high_medium
        severity = "medium"
        target = "high"
        ref_high = high_medium
        ref_low = low_medium
    elif intervention == "ablate_high_strong":
        removed = high_strong
        severity = "strong"
        target = "high"
        ref_high = high_strong
        ref_low = low_strong
    elif intervention == "ablate_low_medium":
        removed = low_medium
        severity = "medium"
        target = "low"
        ref_high = high_medium
        ref_low = low_medium
    elif intervention == "ablate_low_strong":
        removed = low_strong
        severity = "strong"
        target = "low"
        ref_high = high_strong
        ref_low = low_strong
    elif intervention in {"random_medium", "random_strong"}:
        severity = "medium" if intervention.endswith("medium") else "strong"
        target = "random"
        count = len(high_medium) if severity == "medium" else len(high_strong)
        rng = random.Random(_stable_seed("random_abspe", seed, intervention, n_positions, random_draw))
        removed = tuple(sorted(rng.sample(range(n_positions), k=count)))
        ref_high = high_medium if severity == "medium" else high_strong
        ref_low = low_medium if severity == "medium" else low_strong
    else:
        raise ValueError(f"Unsupported intervention for AbsPE analog: {intervention}")

    overlap_high = len(set(removed) & set(ref_high)) if ref_high else 0
    overlap_low = len(set(removed) & set(ref_low)) if ref_low else 0
    return InterventionPlan(
        name=intervention,
        kind="none" if intervention == "none" else "abspe_dct",
        severity=severity,
        target=target,
        removed_indices=removed,
        reference_indices_high=ref_high,
        reference_indices_low=ref_low,
        overlap_high_count=overlap_high,
        overlap_low_count=overlap_low,
        overlap_high_fraction=overlap_high / max(len(ref_high), 1),
        overlap_low_fraction=overlap_low / max(len(ref_low), 1),
        overlap_high_jaccard=_jaccard(removed, ref_high),
        overlap_low_jaccard=_jaccard(removed, ref_low),
        nested_medium_in_strong=(set(high_medium).issubset(set(high_strong)) and set(low_medium).issubset(set(low_strong))),
        band_size=len(high_band),
        component_count=n_positions,
        random_draw=random_draw,
    )


class RopeQKInterventionContext(AbstractContextManager):
    def __init__(
        self,
        model,
        plan: InterventionPlan,
        *,
        target_query_heads_by_layer: dict[int, set[int]] | None = None,
        target_scope: str | None = None,
        target_head_group: str | None = None,
        norm_drift_tolerance: float = 0.05,
    ) -> None:
        self.model = model
        self.plan = plan
        self.target_query_heads_by_layer = target_query_heads_by_layer or {}
        self.target_scope = target_scope
        self.target_head_group = target_head_group
        self.norm_drift_tolerance = float(norm_drift_tolerance)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self._q_drift_medians: list[float] = []
        self._k_drift_medians: list[float] = []
        self._target_kv_heads_by_layer: dict[int, set[int]] = {}
        self._num_query_heads: int | None = None
        self._num_kv_heads: int | None = None
        self._repeat_factor: int | None = None

    def __enter__(self) -> "RopeQKInterventionContext":
        if self.plan.kind != "rope_qk":
            return self
        config = getattr(self.model, "config")
        q_heads = int(getattr(config, "num_attention_heads"))
        kv_heads = int(getattr(config, "num_key_value_heads", q_heads))
        if kv_heads <= 0:
            raise RuntimeError(f"Invalid num_key_value_heads={kv_heads} for model config.")
        if q_heads % kv_heads != 0:
            raise RuntimeError(
                f"GQA dimension mismatch: num_attention_heads={q_heads} is not divisible by "
                f"num_key_value_heads={kv_heads}."
            )
        repeat = max(q_heads // kv_heads, 1)
        self._num_query_heads = q_heads
        self._num_kv_heads = kv_heads
        self._repeat_factor = repeat
        scoped_mode = bool(self.target_query_heads_by_layer)
        for layer_idx, attn in _iter_decoder_attention(self.model):
            if scoped_mode:
                q_sel = set(self.target_query_heads_by_layer.get(layer_idx, set()))
            else:
                q_sel = None
            kv_sel: set[int] | None
            if q_sel is None:
                kv_sel = None
            elif not q_sel:
                kv_sel = set()
            else:
                kv_sel = {int(h // repeat) for h in q_sel}
            if kv_sel:
                self._target_kv_heads_by_layer[layer_idx] = kv_sel
            q_hook = attn.q_proj.register_forward_hook(self._build_proj_hook(q_heads, is_q=True, selected_heads=q_sel))
            k_hook = attn.k_proj.register_forward_hook(self._build_proj_hook(kv_heads, is_q=False, selected_heads=kv_sel))
            self.handles.extend([q_hook, k_hook])
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def audit(self) -> InterventionAudit:
        q_med = float(np.median(self._q_drift_medians)) if self._q_drift_medians else None
        k_med = float(np.median(self._k_drift_medians)) if self._k_drift_medians else None
        drift_vals = [x for x in (q_med, k_med) if x is not None]
        drift_ok = bool(max(drift_vals) <= self.norm_drift_tolerance) if drift_vals else None
        layer_ids = sorted(int(k) for k in self.target_query_heads_by_layer)
        query_head_count = sum(len(v) for v in self.target_query_heads_by_layer.values()) if self.target_query_heads_by_layer else None
        kv_head_count = sum(len(v) for v in self._target_kv_heads_by_layer.values()) if self._target_kv_heads_by_layer else None
        if self._repeat_factor is None or self._repeat_factor <= 1:
            kv_note = "No KV sharing expansion (num_query_heads == num_key_value_heads)."
        else:
            kv_note = (
                "Grouped-query sharing active: each K/V head services "
                f"{self._repeat_factor} query heads; K-side band masking propagates across those query groups."
            )
        return InterventionAudit(
            plan=self.plan,
            drift_median_q=q_med,
            drift_median_k=k_med,
            hook_calls_q=len(self._q_drift_medians),
            hook_calls_k=len(self._k_drift_medians),
            target_scope=self.target_scope,
            target_head_group=self.target_head_group,
            target_layers=layer_ids or None,
            target_query_heads=query_head_count,
            target_kv_heads=kv_head_count,
            num_query_heads=self._num_query_heads,
            num_key_value_heads=self._num_kv_heads,
            gqa_repeat_factor=self._repeat_factor,
            kv_sharing_effect_note=kv_note,
            norm_drift_pass=drift_ok,
            notes="RoPE q/k projection masking with per-token L2 norm rescaling",
        )

    def _build_proj_hook(self, num_heads: int, is_q: bool, selected_heads: set[int] | None):
        pair_indices = self.plan.removed_indices

        def hook(module, inputs, output):
            if not pair_indices:
                return output
            if output.ndim != 3:
                return output
            batch, seq_len, hidden = output.shape
            head_dim = hidden // num_heads
            if head_dim % 2 != 0:
                return output
            view = output.view(batch, seq_len, num_heads, head_dim)
            mask = torch.ones((num_heads, head_dim), dtype=output.dtype, device=output.device)
            if selected_heads is None:
                active_heads = range(num_heads)
            else:
                active_heads = [h for h in sorted(selected_heads) if 0 <= h < num_heads]
            if not active_heads:
                return output
            for h in active_heads:
                for idx in pair_indices:
                    base = 2 * idx
                    if base + 1 < head_dim:
                        mask[h, base : base + 2] = 0.0
            pre_norm = torch.linalg.vector_norm(view.float(), ord=2, dim=-1)
            masked = view * mask.view(1, 1, num_heads, head_dim)
            post_norm = torch.linalg.vector_norm(masked.float(), ord=2, dim=-1)
            safe_scale = torch.ones_like(post_norm)
            nonzero = post_norm > 0
            safe_scale[nonzero] = pre_norm[nonzero] / post_norm[nonzero]
            rescaled = masked * safe_scale.unsqueeze(-1)
            final_norm = torch.linalg.vector_norm(rescaled.float(), ord=2, dim=-1)
            ratio = torch.abs(final_norm / (pre_norm + 1e-12) - 1.0)
            drift = float(torch.median(ratio).item())
            if is_q:
                self._q_drift_medians.append(drift)
            else:
                self._k_drift_medians.append(drift)
            return rescaled.reshape(batch, seq_len, hidden).to(output.dtype)

        return hook


class AbsPEDCTInterventionContext(AbstractContextManager):
    def __init__(self, model, plan: InterventionPlan) -> None:
        self.model = model
        self.plan = plan
        self._original_weight: torch.Tensor | None = None
        self._drift: float | None = None

    def __enter__(self) -> "AbsPEDCTInterventionContext":
        if self.plan.kind != "abspe_dct":
            return self
        wpe = getattr(getattr(self.model, "transformer"), "wpe")
        original = wpe.weight.detach().float().cpu().numpy()
        self._original_weight = wpe.weight.detach().clone()
        transformed = dct(original, type=2, axis=0, norm="ortho")
        if self.plan.removed_indices:
            transformed[list(self.plan.removed_indices), :] = 0.0
        reconstructed = idct(transformed, type=3, axis=0, norm="ortho")
        pre_norm = np.linalg.norm(original, axis=1)
        post_norm = np.linalg.norm(reconstructed, axis=1)
        scale = np.ones_like(post_norm)
        nz = post_norm > 0
        scale[nz] = pre_norm[nz] / post_norm[nz]
        reconstructed = reconstructed * scale[:, None]
        final_norm = np.linalg.norm(reconstructed, axis=1)
        ratio = np.abs(final_norm / (pre_norm + 1e-12) - 1.0)
        self._drift = float(np.median(ratio))
        with torch.no_grad():
            wpe.weight.copy_(torch.tensor(reconstructed, device=wpe.weight.device, dtype=wpe.weight.dtype))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._original_weight is None:
            return
        wpe = getattr(getattr(self.model, "transformer"), "wpe")
        with torch.no_grad():
            wpe.weight.copy_(self._original_weight.to(device=wpe.weight.device, dtype=wpe.weight.dtype))

    def audit(self) -> InterventionAudit:
        drift_ok = (self._drift is not None) and (self._drift <= 0.05)
        return InterventionAudit(
            plan=self.plan,
            drift_median_abspe=self._drift,
            norm_drift_pass=drift_ok,
            notes="GPT-2 positional-embedding DCT masking analog with row-norm rescaling",
        )


def _iter_decoder_attention(model):
    layers = getattr(getattr(model, "model"), "layers")
    for idx, layer in enumerate(layers):
        yield idx, layer.self_attn


def _split_bands(size: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    half = size // 2
    if size % 2 == 0:
        high = tuple(range(0, half))
        low = tuple(range(half, size))
    else:
        high = tuple(range(0, half))
        low = tuple(range(half + 1, size))
    return high, low


def _severity_sets(band: tuple[int, ...], mode: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not band:
        return tuple(), tuple()
    m_medium = max(1, int(math.floor(0.4 * len(band))))
    m_strong = int(math.floor(0.6 * len(band)))
    if m_strong <= m_medium:
        m_strong = min(len(band), m_medium + 1)
    if mode == "high":
        # Highest frequency first: low index first for standard RoPE schedule.
        ordered = list(band)
    elif mode == "low":
        # Lowest frequency first: high index first for standard RoPE schedule.
        ordered = list(reversed(band))
    else:
        raise ValueError(f"Unsupported severity mode: {mode}")
    medium = tuple(sorted(ordered[:m_medium]))
    strong = tuple(sorted(ordered[:m_strong]))
    return medium, strong


def _jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _stable_seed(*parts: object) -> int:
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(key.encode("utf-8"), byteorder="little", signed=False) % (2**63 - 1)
