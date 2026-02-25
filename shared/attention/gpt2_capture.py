from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class HeadCapture:
    q: torch.Tensor  # [seq, head_dim]
    k: torch.Tensor  # [seq, head_dim]
    logits: torch.Tensor  # [seq, seq]


class GPT2AttentionCapture:
    """Monkey-patch GPT-2 attention to capture Q/K and logits per head."""

    def __init__(self) -> None:
        self.records: Dict[int, List[HeadCapture]] = {}
        self._patched = False
        self._original_attn = {}

    def attach(self, model) -> None:
        if self._patched:
            return
        blocks = getattr(model, "transformer").h
        for idx, block in enumerate(blocks):
            attn = block.attn
            original = attn._attn

            def patched(q, k, v, attention_mask=None, head_mask=None, *, _layer_idx=idx, _orig=original, _attn=attn):
                # q/k shapes: [batch, heads, seq, head_dim]
                logits = torch.matmul(q, k.transpose(-1, -2)) / _attn.scale
                self._store(_layer_idx, q.detach(), k.detach(), logits.detach())
                return _orig(q, k, v, attention_mask=attention_mask, head_mask=head_mask)

            attn._attn = patched
            self._original_attn[idx] = original
        self._patched = True

    def _store(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, logits: torch.Tensor) -> None:
        # Assume batch size 1 for now; extend later
        q = q[0]
        k = k[0]
        logits = logits[0]
        head_records: list[HeadCapture] = []
        for head in range(q.shape[0]):
            head_records.append(
                HeadCapture(
                    q=q[head].cpu(),
                    k=k[head].cpu(),
                    logits=logits[head].cpu(),
                )
            )
        self.records[layer_idx] = head_records

    def detach(self, model) -> None:
        if not self._patched:
            return
        blocks = getattr(model, "transformer").h
        for idx, block in enumerate(blocks):
            if idx in self._original_attn:
                block.attn._attn = self._original_attn[idx]
        self._original_attn.clear()
        self._patched = False

    def clear(self) -> None:
        self.records.clear()
