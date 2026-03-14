from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch

from experiment1.shift_kernels import get_kernel_estimator
from shared.specs import ModelSpec


@dataclass
class _ScalarAgg:
    count: int = 0
    total: float = 0.0
    sq_total: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.sq_total += value * value

    def mean(self) -> float:
        return self.total / max(self.count, 1)

    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        mean = self.mean()
        var = max(self.sq_total / self.count - mean * mean, 0.0)
        return float(math.sqrt(var))


class KernelMetricsAccumulator:
    def __init__(
        self,
        model_spec: ModelSpec,
        *,
        engine: str = "optimized",
        centered_mode: str = "shared_mean",
        head_chunk_size: int = 8,
    ) -> None:
        self.model_spec = model_spec
        self.estimator = get_kernel_estimator(model_spec.pe_scheme)
        if engine not in {"legacy", "optimized"}:
            raise ValueError(f"Unsupported kernel engine: {engine}")
        if centered_mode not in {"shared_mean", "legacy_per_sequence"}:
            raise ValueError(f"Unsupported centered_mode: {centered_mode}")
        self.engine = engine
        self.centered_mode = centered_mode
        self.head_chunk_size = max(1, int(head_chunk_size))
        self.track_a: dict[tuple[int, int], _ScalarAgg] = {}
        self.track_b_raw: dict[tuple[int, int], _ScalarAgg] = {}
        self.track_b_centered: dict[tuple[int, int], _ScalarAgg] = {}
        self._sum_q: torch.Tensor | None = None
        self._sum_k: torch.Tensor | None = None
        self._count_tokens: int = 0
        self._diag_count_cache: dict[tuple[int, str], torch.Tensor] = {}

    def update(self, capture: Any) -> None:
        self.update_track_a_raw(capture)
        self.update_centered(capture)

    @staticmethod
    def _to_batched_qk(
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Backward-compatible with single-sequence capture layout.
        if q.ndim == 4 and k.ndim == 4:
            return q.unsqueeze(1), k.unsqueeze(1)
        if q.ndim == 5 and k.ndim == 5:
            return q, k
        raise RuntimeError(f"Unexpected q/k capture shapes q={tuple(q.shape)} k={tuple(k.shape)}")

    @staticmethod
    def _to_batched_logits(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 4:
            return logits.unsqueeze(1)
        if logits.ndim == 5:
            return logits
        raise RuntimeError(f"Unexpected logits capture shape {tuple(logits.shape)}")

    def _get_diag_counts(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, str(device))
        cached = self._diag_count_cache.get(key)
        if cached is not None:
            return cached
        counts = torch.arange(seq_len - 1, 0, -1, device=device, dtype=torch.float64)
        self._diag_count_cache[key] = counts
        return counts

    def _diag_stats(self, mats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # mats: [N, S, S]
        n, seq_len, _ = mats.shape
        sums = torch.zeros((n, seq_len - 1), device=mats.device, dtype=torch.float64)
        sq_sums = torch.zeros_like(sums)
        counts = self._get_diag_counts(seq_len, mats.device).expand_as(sums)
        for delta in range(1, seq_len):
            diag = torch.diagonal(mats, offset=-delta, dim1=-2, dim2=-1).to(torch.float64)
            sums[:, delta - 1] = diag.sum(dim=-1)
            sq_sums[:, delta - 1] = (diag * diag).sum(dim=-1)
        return sums, sq_sums, counts

    def _normalize_for_model_norm(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [..., S, S]
        norm_name = (self.model_spec.norm or "").lower()
        if not norm_name.startswith("rms"):
            return logits
        row_mean = logits.mean(dim=-1, keepdim=True)
        col_mean = logits.mean(dim=-2, keepdim=True)
        global_mean = logits.mean(dim=(-1, -2), keepdim=True)
        return logits - row_mean - col_mean + global_mean

    def _update_metric_legacy(self, logits: torch.Tensor, store: dict[tuple[int, int], _ScalarAgg]) -> None:
        # logits: [L,B,H,S,S]
        num_layers, batch, num_heads = logits.shape[:3]
        for layer_idx in range(num_layers):
            for batch_idx in range(batch):
                for head_idx in range(num_heads):
                    key = (layer_idx, head_idx)
                    fit = self.estimator.fit_logits(logits[layer_idx, batch_idx, head_idx])
                    store.setdefault(key, _ScalarAgg()).update(fit.r2)

    def _update_metric_optimized(self, logits: torch.Tensor, store: dict[tuple[int, int], _ScalarAgg]) -> None:
        # logits: [L,B,H,S,S]
        num_layers, batch, num_heads, seq_len, _ = logits.shape
        if seq_len <= 1:
            return
        for layer_idx in range(num_layers):
            layer_logits = logits[layer_idx]  # [B,H,S,S]
            flat = layer_logits.reshape(batch * num_heads, seq_len, seq_len)
            chunk = max(self.head_chunk_size, 1) * max(batch, 1)
            for start in range(0, flat.shape[0], chunk):
                stop = min(start + chunk, flat.shape[0])
                part = flat[start:stop]
                sums, sq_sums, counts = self._diag_stats(part)
                for local_idx in range(part.shape[0]):
                    global_idx = start + local_idx
                    head_idx = global_idx % num_heads
                    key = (layer_idx, head_idx)
                    fit = self.estimator.fit_from_stats(
                        sums=sums[local_idx],
                        sq_sums=sq_sums[local_idx],
                        counts=counts[local_idx],
                    )
                    store.setdefault(key, _ScalarAgg()).update(fit.r2)

    def _update_metric(self, logits: torch.Tensor, store: dict[tuple[int, int], _ScalarAgg]) -> None:
        if self.engine == "legacy":
            self._update_metric_legacy(logits, store)
            return
        self._update_metric_optimized(logits, store)

    def update_track_a_raw(self, capture: Any) -> None:
        if capture.logits is None:
            raise RuntimeError("Capture must include logits to compute Track A metrics.")
        logits = self._to_batched_logits(capture.logits)
        q, k = self._to_batched_qk(capture.q, capture.k)
        if logits.shape[:3] != q.shape[:3] or q.shape[:3] != k.shape[:3]:
            raise RuntimeError(
                "Mismatched capture shapes for track-a/raw: "
                f"logits={tuple(logits.shape)} q={tuple(q.shape)} k={tuple(k.shape)}"
            )
        track_a_logits = self._normalize_for_model_norm(logits)
        raw_logits = self._raw_logits_from_qk(q=q, k=k)
        self._update_metric(track_a_logits, self.track_a)
        self._update_metric(raw_logits, self.track_b_raw)

    def _raw_logits_from_qk(self, *, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        head_dim = q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)
        raw_gram = torch.matmul(q, k.transpose(-1, -2)) * scale
        return self._normalize_for_model_norm(raw_gram)

    def update_track_b_raw_from_qk(self, capture: Any) -> None:
        q, k = self._to_batched_qk(capture.q, capture.k)
        if q.shape[:3] != k.shape[:3]:
            raise RuntimeError(
                "Mismatched capture shapes for track-b/raw: "
                f"q={tuple(q.shape)} k={tuple(k.shape)}"
            )
        raw_logits = self._raw_logits_from_qk(q=q, k=k)
        self._update_metric(raw_logits, self.track_b_raw)

    def accumulate_shared_means(self, capture: Any) -> None:
        q, k = self._to_batched_qk(capture.q, capture.k)
        # q/k: [L,B,H,S,D] -> [L,H,D]
        sum_q = q.to(torch.float64).sum(dim=(1, 3))
        sum_k = k.to(torch.float64).sum(dim=(1, 3))
        if self._sum_q is None:
            self._sum_q = sum_q
            self._sum_k = sum_k
        else:
            self._sum_q = self._sum_q + sum_q
            self._sum_k = self._sum_k + sum_k
        self._count_tokens += int(q.shape[1] * q.shape[3])

    def finalize_shared_means(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._sum_q is None or self._sum_k is None or self._count_tokens <= 0:
            raise RuntimeError("Shared means requested before accumulate_shared_means.")
        denom = float(self._count_tokens)
        return self._sum_q / denom, self._sum_k / denom

    def update_centered(
        self,
        capture: Any,
        *,
        shared_q_mean: torch.Tensor | None = None,
        shared_k_mean: torch.Tensor | None = None,
    ) -> None:
        q, k = self._to_batched_qk(capture.q, capture.k)

        if self.centered_mode == "shared_mean":
            if shared_q_mean is None or shared_k_mean is None:
                centered_q = q - q.mean(dim=-2, keepdim=True)
                centered_k = k - k.mean(dim=-2, keepdim=True)
            else:
                q_mean = shared_q_mean.to(q.device, dtype=q.dtype).unsqueeze(1).unsqueeze(3)
                k_mean = shared_k_mean.to(k.device, dtype=k.dtype).unsqueeze(1).unsqueeze(3)
                centered_q = q - q_mean
                centered_k = k - k_mean
        else:
            centered_q = q - q.mean(dim=-2, keepdim=True)
            centered_k = k - k.mean(dim=-2, keepdim=True)

        head_dim = q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)
        centered_gram = torch.matmul(centered_q, centered_k.transpose(-1, -2)) * scale
        centered_logits = self._normalize_for_model_norm(centered_gram)
        self._update_metric(centered_logits, self.track_b_centered)

    def to_dataframes(
        self,
        *,
        model: str,
        phase: str,
        split: str,
        task: str,
        intervention: str,
        seed: int,
        seq_len: int,
        dataset: str | None = None,
        span: int | None = None,
        random_draw: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        base_cols = {
            "model": model,
            "phase": phase,
            "split": split,
            "dataset": dataset,
            "task": task,
            "span": span,
            "intervention": intervention,
            "random_draw": random_draw,
            "seed": seed,
            "seq_len": seq_len,
        }
        a_rows = []
        raw_rows = []
        centered_rows = []
        for (layer, head), agg in sorted(self.track_a.items()):
            a_rows.append({**base_cols, "layer": layer, "head": head, "mean_r2": agg.mean(), "std_r2": agg.std(), "count": agg.count})
        for (layer, head), agg in sorted(self.track_b_raw.items()):
            raw_rows.append({**base_cols, "layer": layer, "head": head, "mean_r2": agg.mean(), "std_r2": agg.std(), "count": agg.count})
        for (layer, head), agg in sorted(self.track_b_centered.items()):
            centered_rows.append({**base_cols, "layer": layer, "head": head, "mean_r2": agg.mean(), "std_r2": agg.std(), "count": agg.count})
        return pd.DataFrame(a_rows), pd.DataFrame(raw_rows), pd.DataFrame(centered_rows)
