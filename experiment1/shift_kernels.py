from __future__ import annotations

"""Shift-kernel estimators aware of PE scheme differences."""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class KernelFit:
    g_values: list[float]
    r2: float
    sse: float
    sst: float


class KernelEstimator(Protocol):
    def transform(self, means: torch.Tensor) -> torch.Tensor:
        ...

    def fit_logits(self, logits: torch.Tensor) -> KernelFit:
        ...

    def fit_from_stats(self, *, sums: torch.Tensor, sq_sums: torch.Tensor, counts: torch.Tensor) -> KernelFit:
        ...


class BaseEstimator:
    def fit_logits(self, logits: torch.Tensor) -> KernelFit:
        seq_len = logits.size(-1)
        diagonals: list[torch.Tensor] = []
        diag_means = []
        for delta in range(1, seq_len):
            diag = torch.diagonal(logits, offset=-delta).to(torch.float64)
            diagonals.append(diag)
            diag_means.append(diag.mean())
        if not diagonals:
            return KernelFit(g_values=[], r2=0.0, sse=0.0, sst=0.0)
        mean_tensor = torch.stack(diag_means)
        targets = self.transform(mean_tensor)
        sse = 0.0
        for idx, diag in enumerate(diagonals):
            target = targets[idx]
            sse += torch.sum((diag - target) ** 2).item()
        stacked = torch.cat(diagonals)
        sst = torch.sum((stacked - stacked.mean()) ** 2).item()
        r2 = 1.0 if sst == 0.0 else max(0.0, 1.0 - (sse / sst))
        return KernelFit(g_values=targets.tolist(), r2=float(r2), sse=float(sse), sst=float(sst))

    def fit_from_stats(
        self,
        *,
        sums: torch.Tensor,
        sq_sums: torch.Tensor,
        counts: torch.Tensor,
    ) -> KernelFit:
        if counts.sum() == 0:
            return KernelFit(g_values=[], r2=0.0, sse=0.0, sst=0.0)
        means = torch.where(counts == 0, torch.zeros_like(sums), sums / torch.where(counts == 0, torch.ones_like(counts), counts))
        targets = self.transform(means)
        sse_terms = sq_sums - 2 * targets * sums + counts * (targets**2)
        sse = torch.sum(torch.where(counts == 0, torch.zeros_like(sse_terms), sse_terms)).item()
        total_sum = torch.sum(sums).item()
        total_sq_sum = torch.sum(sq_sums).item()
        total_count = torch.sum(counts).item()
        if total_count == 0:
            return KernelFit(g_values=targets.tolist(), r2=0.0, sse=sse, sst=0.0)
        global_mean = total_sum / total_count
        sst = total_sq_sum - total_count * (global_mean**2)
        r2 = 1.0 if sst <= 0 else max(0.0, 1.0 - (sse / sst))
        return KernelFit(g_values=targets.tolist(), r2=float(r2), sse=float(sse), sst=float(sst))

    def transform(self, means: torch.Tensor) -> torch.Tensor:
        return means


class RoPEEstimator(BaseEstimator):
    def __init__(self, max_components: int = 8) -> None:
        self.max_components = max_components

    def transform(self, means: torch.Tensor) -> torch.Tensor:
        if means.numel() == 0:
            return means
        freq = torch.fft.rfft(means)
        magnitudes = torch.abs(freq)
        if magnitudes.numel() > 1:
            keep = min(self.max_components, magnitudes.numel() - 1)
            if keep > 0:
                idx = torch.topk(magnitudes[1:], k=keep).indices + 1
                mask = torch.zeros_like(freq, dtype=torch.bool)
                mask[0] = True
                mask[idx] = True
                freq = freq * mask
        filtered = torch.fft.irfft(freq, n=means.numel())
        return filtered.real


class AbsolutePEEstimator(BaseEstimator):
    pass


class NoPEEstimator(BaseEstimator):
    def transform(self, means: torch.Tensor) -> torch.Tensor:
        if means.numel() == 0:
            return means
        centered = means - means.mean()
        return centered


def get_kernel_estimator(pe_scheme: str) -> BaseEstimator:
    scheme = (pe_scheme or "").lower()
    if "rope" in scheme:
        return RoPEEstimator()
    if "none" in scheme or "nope" in scheme:
        return NoPEEstimator()
    return AbsolutePEEstimator()
