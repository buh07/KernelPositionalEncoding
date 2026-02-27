from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rope
from transformers.models.olmo.modeling_olmo import apply_rotary_pos_emb as olmo_apply_rope

from shared.specs import ModelSpec


@dataclass
class LayerCapture:
    q: torch.Tensor  # [heads, seq, head_dim]
    k: torch.Tensor
    logits: torch.Tensor  # [heads, seq, seq]
    rope_cos: torch.Tensor | None = None
    rope_sin: torch.Tensor | None = None


@dataclass
class CaptureRecord:
    q: torch.Tensor  # [layers, heads, seq, head_dim]
    k: torch.Tensor
    logits: torch.Tensor  # [layers, heads, seq, seq]
    rope_cos: torch.Tensor | None = None
    rope_sin: torch.Tensor | None = None


class AttentionCaptureAdapter(ABC):
    """Base class for attention instrumentation."""

    def __init__(self) -> None:
        self._handles: list[RemovableHandle] = []
        self._suspend_hooks = False

    def register(self, model) -> None:
        if self._handles:
            return
        self._handles = list(self._attach(model))

    def capture(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **forward_kwargs,
    ) -> CaptureRecord:
        self._reset()
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                **forward_kwargs,
            )
        return self._finalize_record()

    def cleanup(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._reset()

    @contextmanager
    def suspended(self) -> Iterator[None]:
        prev = self._suspend_hooks
        self._suspend_hooks = True
        try:
            yield
        finally:
            self._suspend_hooks = prev

    @abstractmethod
    def _attach(self, model) -> Iterable[RemovableHandle]:
        ...

    @abstractmethod
    def _reset(self) -> None:
        ...

    @abstractmethod
    def _finalize_record(self) -> CaptureRecord:
        ...


class GPT2Adapter(AttentionCaptureAdapter):
    """Capture adapter for GPT-2 attention (absolute positional encoding)."""

    def __init__(self) -> None:
        super().__init__()
        self._layers: dict[int, LayerCapture] = {}

    def _attach(self, model) -> Iterable[RemovableHandle]:
        handles: list[RemovableHandle] = []
        blocks = getattr(model, "transformer").h
        for idx, block in enumerate(blocks):
            attn = block.attn
            handle = attn.c_attn.register_forward_hook(self._make_hook(idx, attn))
            handles.append(handle)
        return handles

    def _reset(self) -> None:
        self._layers.clear()

    def _finalize_record(self) -> CaptureRecord:
        if not self._layers:
            raise RuntimeError("No GPT-2 attention activations were captured.")
        ordered = [self._layers[idx] for idx in sorted(self._layers)]
        q = torch.stack([layer.q for layer in ordered], dim=0)
        k = torch.stack([layer.k for layer in ordered], dim=0)
        logits = torch.stack([layer.logits for layer in ordered], dim=0)
        return CaptureRecord(q=q, k=k, logits=logits)

    def _make_hook(self, layer_idx: int, attn_module) -> callable:
        head_dim = attn_module.head_dim
        num_heads = attn_module.num_heads
        scale = 1.0 / math.sqrt(head_dim)

        def hook(module, inputs, output):
            if self._suspend_hooks:
                return
            q, k, _ = output.split(attn_module.split_size, dim=2)
            q_heads = self._reshape_heads(q, num_heads, head_dim)
            k_heads = self._reshape_heads(k, num_heads, head_dim)
            logits = torch.matmul(q_heads, k_heads.transpose(-1, -2)) * scale
            self._layers[layer_idx] = LayerCapture(
                q=q_heads.cpu(),
                k=k_heads.cpu(),
                logits=logits.cpu(),
            )

        return hook

    @staticmethod
    def _reshape_heads(tensor: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        view = tensor.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        if view.size(0) != 1:
            raise RuntimeError("Batch size > 1 is not supported for GPT-2 capture.")
        return view[0].to(torch.float32)


class ProjectionCaptureAdapter(AttentionCaptureAdapter):
    """Shared scaffolding for attention modules with explicit q_proj/k_proj layers."""

    def __init__(self, rope_fn=None) -> None:
        super().__init__()
        self._rope_fn = rope_fn
        self._raw: dict[int, dict[str, torch.Tensor]] = {}
        self._num_query_heads: int | None = None
        self._num_kv_heads: int | None = None
        self._head_dim: int | None = None

    def register(self, model) -> None:
        if self._handles:
            return
        self._configure_from_model(model)
        super().register(model)

    def _reset(self) -> None:
        self._raw.clear()

    def _finalize_record(self) -> CaptureRecord:
        if not self._raw:
            raise RuntimeError("No attention projections captured.")
        captures: list[LayerCapture] = []
        for layer_idx in sorted(self._raw):
            captures.append(self._finalize_layer(layer_idx, self._raw[layer_idx]))
        q = torch.stack([cap.q for cap in captures], dim=0)
        k = torch.stack([cap.k for cap in captures], dim=0)
        logits = torch.stack([cap.logits for cap in captures], dim=0)
        rope_cos = None
        rope_sin = None
        if all(cap.rope_cos is not None for cap in captures):
            rope_cos = torch.stack([cap.rope_cos for cap in captures if cap.rope_cos is not None], dim=0)
        if all(cap.rope_sin is not None for cap in captures):
            rope_sin = torch.stack([cap.rope_sin for cap in captures if cap.rope_sin is not None], dim=0)
        return CaptureRecord(q=q, k=k, logits=logits, rope_cos=rope_cos, rope_sin=rope_sin)

    def _attach(self, model) -> Iterable[RemovableHandle]:
        handles: list[RemovableHandle] = []
        for layer_idx, attn in self._iter_attention_modules(model):
            handles.append(attn.q_proj.register_forward_hook(self._make_proj_hook(layer_idx, "q")))
            handles.append(attn.k_proj.register_forward_hook(self._make_proj_hook(layer_idx, "k")))
            if self._rope_fn is not None:
                handles.append(attn.register_forward_pre_hook(self._make_position_hook(layer_idx), with_kwargs=True))
        return handles

    def _make_proj_hook(self, layer_idx: int, kind: str):
        def hook(module, inputs, output):
            if self._suspend_hooks:
                return
            self._raw.setdefault(layer_idx, {})[kind] = output.detach().to(torch.float32).cpu()

        return hook

    def _make_position_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            if self._suspend_hooks:
                return
            # transformers 5.x passes position_embeddings as a keyword argument;
            # fall back to second positional arg for older transformers versions.
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) >= 2:
                position_embeddings = args[1]
            if position_embeddings is None:
                return
            cos, sin = position_embeddings
            entry = self._raw.setdefault(layer_idx, {})
            entry["cos"] = cos.detach().to(torch.float32).cpu()
            entry["sin"] = sin.detach().to(torch.float32).cpu()

        return hook

    def _finalize_layer(self, layer_idx: int, data: dict[str, torch.Tensor]) -> LayerCapture:
        if "q" not in data or "k" not in data:
            raise RuntimeError(f"Layer {layer_idx} missing q/k projections.")
        q_heads = self._reshape_heads(data["q"], self._num_query_heads)
        k_heads = self._reshape_heads(data["k"], self._num_kv_heads)
        rope_cos = None
        rope_sin = None
        if self._rope_fn is not None and "cos" in data and "sin" in data:
            q_heads, k_heads = self._apply_rope(q_heads, k_heads, data)
            rope_cos = data["cos"].to(torch.float32).cpu()
            rope_sin = data["sin"].to(torch.float32).cpu()
        k_expanded = self._align_kv_heads(k_heads)
        logits = torch.matmul(q_heads, k_expanded.transpose(-1, -2)) / math.sqrt(self._head_dim or 1)
        return LayerCapture(
            q=q_heads.cpu(),
            k=k_expanded.cpu(),
            logits=logits.cpu(),
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

    def _reshape_heads(self, raw: torch.Tensor, num_heads: int | None) -> torch.Tensor:
        num_heads = num_heads or 1
        batch, seq_len, _ = raw.shape
        head_dim = self._head_dim or (raw.shape[-1] // num_heads)
        view = raw.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        if view.size(0) != 1:
            raise RuntimeError("Batch size > 1 is not supported for capture.")
        return view[0].to(torch.float32)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        data: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = data["cos"]
        sin = data["sin"]
        q_in = q.unsqueeze(0)
        k_in = k.unsqueeze(0)
        q_rot, k_rot = self._rope_fn(q_in, k_in, cos, sin)
        return q_rot[0].to(torch.float32), k_rot[0].to(torch.float32)

    def _align_kv_heads(self, k: torch.Tensor) -> torch.Tensor:
        num_query = self._num_query_heads or k.size(0)
        num_kv = self._num_kv_heads or k.size(0)
        if num_query == num_kv:
            return k
        repeat_factor = num_query // num_kv
        return k.repeat_interleave(repeat_factor, dim=0)

    @abstractmethod
    def _configure_from_model(self, model) -> None:
        ...

    @abstractmethod
    def _iter_attention_modules(self, model) -> Iterable[tuple[int, nn.Module]]:
        ...


class OLMoAdapter(ProjectionCaptureAdapter):
    def __init__(self) -> None:
        super().__init__(rope_fn=olmo_apply_rope)

    def _configure_from_model(self, model) -> None:
        config = getattr(model, "config")
        self._num_query_heads = config.num_attention_heads
        self._num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self._head_dim = config.hidden_size // config.num_attention_heads

    def _iter_attention_modules(self, model) -> Iterable[tuple[int, nn.Module]]:
        layers = getattr(model, "model").layers
        for idx, layer in enumerate(layers):
            yield idx, layer.self_attn


class LlamaAdapter(ProjectionCaptureAdapter):
    def __init__(self) -> None:
        super().__init__(rope_fn=llama_apply_rope)

    def _configure_from_model(self, model) -> None:
        config = getattr(model, "config")
        self._num_query_heads = config.num_attention_heads
        self._num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self._head_dim = config.hidden_size // config.num_attention_heads

    def _iter_attention_modules(self, model) -> Iterable[tuple[int, nn.Module]]:
        layers = getattr(model, "model").layers
        for idx, layer in enumerate(layers):
            yield idx, layer.self_attn


class TinyLlamaAdapter(LlamaAdapter):
    """TinyLlama shares the same attention implementation as LLaMA."""


class TinyLlamaNoPEAdapter(LlamaAdapter):
    """Adapter for the NoPE TinyLlama variant with validation."""

    def __init__(self) -> None:
        super().__init__()
        self._validated = False
        # No rotary embeddings applied.
        self._rope_fn = None

    def register(self, model) -> None:
        super().register(model)
        self._validate_nope(model)

    def _validate_nope(self, model) -> None:
        if self._validated:
            return
        device = next(model.parameters()).device
        vocab = model.config.vocab_size
        seq_len = 32
        with self.suspended():
            with torch.no_grad():
                tokens = torch.randint(0, vocab, (1, seq_len), device=device)
                pos_a = torch.arange(seq_len, device=device).unsqueeze(0)
                pos_b = torch.arange(seq_len - 1, -1, -1, device=device).unsqueeze(0)
                out_a = model(input_ids=tokens, position_ids=pos_a, use_cache=False).logits
                out_b = model(input_ids=tokens, position_ids=pos_b, use_cache=False).logits
        if not torch.allclose(out_a, out_b, atol=1e-4):
            diff = (out_a - out_b).abs().max().item()
            warnings.warn(
                f"TinyLlama-NoPE positional invariance check diff={diff:.4f}; proceeding but results may reflect residual PE effects."
            )
        self._validated = True


def get_adapter(model_spec: ModelSpec) -> AttentionCaptureAdapter:
    if model_spec.name.startswith("gpt2"):
        return GPT2Adapter()
    if model_spec.name == "olmo-1b":
        return OLMoAdapter()
    if model_spec.name == "llama-3.2-1b":
        return LlamaAdapter()
    if model_spec.name == "tinyllama-1.1b":
        return TinyLlamaAdapter()
    if model_spec.name == "tinyllama-nope-1.1b":
        return TinyLlamaNoPEAdapter()
    raise ValueError(f"No adapter registered for model {model_spec.name}")
