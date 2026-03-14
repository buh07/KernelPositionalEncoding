from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from experiment2.config import MODEL_BY_NAME
from shared.attention.adapters import GemmaAdapter, LlamaAdapter, OLMo2Adapter, get_adapter


class _ToySelfAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        resolved_head_dim = int(head_dim) if head_dim is not None else hidden_size // num_q_heads
        self.q_proj = nn.Linear(hidden_size, num_q_heads * resolved_head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * resolved_head_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, position_embeddings=None) -> torch.Tensor:  # noqa: ANN001
        _ = position_embeddings
        _ = self.q_proj(hidden_states)
        _ = self.k_proj(hidden_states)
        return hidden_states


class _ToyLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = _ToySelfAttention(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

    def forward(self, hidden_states: torch.Tensor, position_embeddings=None) -> torch.Tensor:  # noqa: ANN001
        return self.self_attn(hidden_states, position_embeddings=position_embeddings)


class _ToyBackbone(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _ToyLayer(
                    hidden_size=hidden_size,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
                for _ in range(num_layers)
            ]
        )


class _ToyProjectionModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 256,
        hidden_size: int = 64,
        num_layers: int = 3,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        resolved_head_dim = int(head_dim) if head_dim is not None else hidden_size // num_q_heads
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=resolved_head_dim,
        )
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = _ToyBackbone(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=resolved_head_dim,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, output_attentions, output_hidden_states, kwargs
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden, position_embeddings=None)
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits)


def test_get_adapter_scaleup_registrations() -> None:
    assert isinstance(get_adapter(MODEL_BY_NAME["llama-3.1-8b"]), LlamaAdapter)
    assert isinstance(get_adapter(MODEL_BY_NAME["olmo-2-7b"]), OLMo2Adapter)
    assert isinstance(get_adapter(MODEL_BY_NAME["gemma-7b"]), GemmaAdapter)


def test_projection_adapters_capture_exact_layer_count_for_scaleup_names() -> None:
    model = _ToyProjectionModel(num_layers=4, hidden_size=64, num_q_heads=8, num_kv_heads=2)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 12), dtype=torch.long)
    for name in ("llama-3.1-8b", "olmo-2-7b", "gemma-7b"):
        spec = MODEL_BY_NAME[name]
        adapter = get_adapter(spec)
        adapter._rope_fn = None
        adapter.register(model)
        record = adapter.capture(
            model,
            input_ids=input_ids,
            attention_mask=None,
            include_logits=True,
            return_token_logits=True,
            output_device="cpu",
        )
        assert int(record.q.shape[0]) == int(model.config.num_hidden_layers)
        assert int(record.k.shape[0]) == int(model.config.num_hidden_layers)
        assert int(record.logits.shape[0]) == int(model.config.num_hidden_layers)
        assert int(record.q.shape[1]) == int(model.config.num_attention_heads)
        assert int(record.k.shape[1]) == int(model.config.num_attention_heads)
        adapter.cleanup()


def test_gemma_adapter_uses_config_head_dim_when_hidden_size_ratio_differs() -> None:
    model = _ToyProjectionModel(
        num_layers=2,
        hidden_size=96,
        num_q_heads=4,
        num_kv_heads=4,
        head_dim=32,
    )
    input_ids = torch.randint(0, model.config.vocab_size, (1, 9), dtype=torch.long)
    spec = MODEL_BY_NAME["gemma-7b"]
    adapter = get_adapter(spec)
    adapter._rope_fn = None
    adapter.register(model)
    record = adapter.capture(
        model,
        input_ids=input_ids,
        attention_mask=None,
        include_logits=True,
        return_token_logits=True,
        output_device="cpu",
    )
    assert int(record.q.shape[-1]) == 32
    assert int(record.k.shape[-1]) == 32
    adapter.cleanup()


def test_projection_capture_can_skip_attention_capture_for_logits_only() -> None:
    model = _ToyProjectionModel(num_layers=2, hidden_size=64, num_q_heads=8, num_kv_heads=2)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 10), dtype=torch.long)
    spec = MODEL_BY_NAME["llama-3.1-8b"]
    adapter = get_adapter(spec)
    adapter._rope_fn = None
    adapter.register(model)
    record = adapter.capture(
        model,
        input_ids=input_ids,
        attention_mask=None,
        include_logits=False,
        return_token_logits=True,
        capture_attention=False,
        output_device="cpu",
    )
    assert record.token_logits is not None
    assert tuple(record.token_logits.shape) == (10, model.config.vocab_size)
    assert int(record.q.numel()) == 0
    assert int(record.k.numel()) == 0
    assert record.logits is None
    adapter.cleanup()
