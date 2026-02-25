from __future__ import annotations

"""Utilities for reconciling norm-dependent differences in attention logits."""

import torch


def normalize_logits_for_norm(logits: torch.Tensor, norm_name: str | None) -> torch.Tensor:
    """Align logits so LayerNorm and RMSNorm heads are comparable.

    LayerNorm already centers activations, but RMSNorm does not. To keep the
    shift-kernel estimator consistent across architectures, we explicitly remove
    the per-query and per-key means when the model uses RMSNorm. LayerNorm
    tensors pass through untouched.
    """

    if norm_name and norm_name.lower().startswith("rms"):
        logits = (logits
                  - logits.mean(dim=-1, keepdim=True)
                  - logits.mean(dim=-2, keepdim=True)
                  + logits.mean())
    return logits
