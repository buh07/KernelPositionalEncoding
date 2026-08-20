from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import ClassVar, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from scipy import stats as scipy_stats
from torch.utils.hooks import RemovableHandle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import si_rebuttal.intervention as intervention_module
from si_rebuttal.controls import (
    ExpectedAttentionCallCounts,
    causal_toeplitz_correction,
    compact_ascii_json_seed,
    permute_offsets,
    zero_mean_l2_control,
)
from si_rebuttal.intervention import (
    ADR_LAYER_COUNT,
    ADR_OLMO_KV_HEADS,
    AttentionMaskForwardLayout,
    EagerAttentionIntervention,
    FamilyAdapter,
    ValidationProbeRecord,
    ValidationProbeRequest,
    attn_implementation_attr_name,
    create_llama_3_1_8b_intervention,
    create_mistral_7b_v0_1_intervention,
    create_olmo_2_7b_intervention,
    derive_supported_model_family_spec_for_capture,
    manual_attention_probs,
    mean_next_token_nll,
    repeat_kv_after_gqa,
)
from si_rebuttal.kernels import (
    SOURCE_OFFSET_START,
    SOURCE_OFFSET_STOP,
    TOP_NON_DC_COMPONENTS,
    HeadIndex,
    bin_mean_source_r2,
    build_causal_toeplitz_correction,
    double_center_matrix,
    estimate_raw_intervention_kernel,
    estimate_raw_intervention_kernels_by_head,
    estimate_source_scores,
    estimate_source_scores_by_head,
    sort_heads_into_bins,
)
from si_rebuttal.statistics import (
    DEFAULT_SPEARMAN_PERMUTATIONS,
    average_control_trials_per_sequence,
    monte_carlo_spearman_positive,
    paired_percentile_bootstrap,
)

Float64Array: TypeAlias = npt.NDArray[np.float64]
Int64Array: TypeAlias = npt.NDArray[np.int64]
ModelForward: TypeAlias = Callable[
    [torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    tuple[torch.Tensor, torch.Tensor | None],
]
ForwardPreHookRegistrar: TypeAlias = Callable[..., RemovableHandle]
TorchModuleInit: TypeAlias = Callable[[torch.nn.Module], None]
_test_family_registry: dict[str, FamilyAdapter] = {}
_TORCH_MODULE_INIT = cast(TorchModuleInit, torch.nn.Module.__init__)


class _AttentionForward(Protocol):
    def __call__(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


class _SciPySpearmanResult(Protocol):
    statistic: float | np.float64
    pvalue: float | np.float64


_SCIPY_SPEARMANR = cast(
    Callable[[Float64Array, Float64Array], _SciPySpearmanResult],
    scipy_stats.spearmanr,
)


class _IntegerGenerator(Protocol):
    def integers(
        self,
        low: int,
        high: int | None = None,
        *,
        size: tuple[int, int],
        endpoint: bool = False,
    ) -> Int64Array: ...


@dataclass(frozen=True)
class _SpearmanResult:
    statistic: float
    pvalue: float


def _coerce_float64_scalar(value: object, *, name: str) -> float:
    if isinstance(value, float):
        return float(value)
    if isinstance(value, np.floating):
        return float(value.item())
    raise TypeError(f"{name} must be a float-compatible scalar")


def _spearmanr(x: Float64Array, y: Float64Array) -> _SpearmanResult:
    result = _SCIPY_SPEARMANR(x, y)
    return _SpearmanResult(
        statistic=_coerce_float64_scalar(result.statistic, name="statistic"),
        pvalue=_coerce_float64_scalar(result.pvalue, name="pvalue"),
    )


def _rankdata(values: Float64Array) -> Float64Array:
    rankdata = cast(Callable[..., object], scipy_stats.rankdata)
    ranked = rankdata(values, method="average")
    if not isinstance(ranked, np.ndarray):
        raise TypeError("rankdata must return a NumPy array")
    return np.asarray(ranked, dtype=np.float64)


def _init_torch_module(module: torch.nn.Module) -> None:
    _TORCH_MODULE_INIT(module)


def _uninitialized_model_forward(
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del hidden_states, position_embeddings, attention_mask
    raise AssertionError("test model forward must be installed before use")


def _model_attn_implementation(model: _TestModelBase) -> str:
    if attn_implementation_attr_name(model) == "_attn_implementation":
        private_config = cast(_PrivateAttnImplementationConfig, model.config)
        return _config_private_attn_implementation(private_config)
    public_config = cast(_PublicAttnImplementationConfig, model.config)
    value = object.__getattribute__(public_config, "attn_implementation")
    if not isinstance(value, str):
        raise TypeError("attention implementation marker must be a string")
    return value


def _config_private_attn_implementation(config: _PrivateAttnImplementationConfig) -> str:
    value = object.__getattribute__(config, "_attn_implementation")
    if not isinstance(value, str):
        raise TypeError("private attention implementation marker must be a string")
    return value


def _manual_double_center(matrix: Float64Array) -> Float64Array:
    row_mean = matrix.mean(axis=1, keepdims=True)
    col_mean = matrix.mean(axis=0, keepdims=True)
    return matrix - row_mean - col_mean + matrix.mean()


def _manual_lower_diag_means(matrix: Float64Array, start: int, stop: int) -> Float64Array:
    values: list[np.float64] = []
    for offset in range(start, stop + 1):
        values.append(np.float64(np.diagonal(matrix, offset=-offset).mean()))
    return np.asarray(values, dtype=np.float64)


def _manual_fft_topk(means: Float64Array, keep_non_dc: int) -> Float64Array:
    freq = np.fft.rfft(means)
    mask = np.zeros(freq.shape[0], dtype=bool)
    mask[0] = True
    if freq.shape[0] > 1 and keep_non_dc > 0:
        idx = np.argsort(np.abs(freq[1:]))[-keep_non_dc:] + 1
        mask[idx] = True
    filtered = np.where(mask, freq, 0.0)
    return np.fft.irfft(filtered, n=means.shape[0]).real.astype(np.float64, copy=False)


def _manual_pair_weighted_r2(matrix: Float64Array, smoothed: Float64Array, start: int) -> float:
    diagonals: list[Float64Array] = []
    sse = 0.0
    for index, offset in enumerate(range(start, start + smoothed.shape[0])):
        diagonal = np.diagonal(matrix, offset=-offset).astype(np.float64, copy=False)
        diagonals.append(diagonal)
        sse += float(np.square(diagonal - smoothed[index]).sum())
    all_pairs = np.concatenate(diagonals)
    sst = float(np.square(all_pairs - all_pairs.mean()).sum())
    return 1.0 if sst == 0.0 else max(0.0, 1.0 - (sse / sst))


def _make_rms_sequence(
    row_bias: Float64Array, col_bias: Float64Array, diag_signal: Float64Array
) -> Float64Array:
    seq_len = row_bias.shape[0]
    if col_bias.shape[0] != seq_len:
        raise ValueError("row_bias and col_bias must share the same seq_len.")
    if diag_signal.shape[0] != seq_len - 1:
        raise ValueError("diag_signal must cover ADR source offsets 1..seq_len-1.")
    matrix = np.zeros((seq_len, seq_len), dtype=np.float64)
    for query_index in range(seq_len):
        for key_index in range(query_index + 1):
            delta = query_index - key_index
            matrix[query_index, key_index] = (
                row_bias[query_index]
                + col_bias[key_index]
                + (0.0 if delta == 0 else diag_signal[delta - 1])
            )
    return matrix


def _apply_test_rope_bfloat16(
    q: torch.Tensor,
    k: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    del position_embeddings
    return q.to(torch.bfloat16), k.to(torch.bfloat16)


class ToyAttention(torch.nn.Module):
    def __init__(self, *, num_heads: int, num_key_value_heads: int) -> None:
        _init_torch_module(self)
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hidden_states, position_embeddings
        if attention_mask is None:
            raise RuntimeError("mask required")
        return attention_mask


class _FakeAttentionInterface(dict[str, _AttentionForward]):
    def register(self, key: str, value: _AttentionForward) -> None:
        self[key] = value

    def get_interface(
        self, attn_implementation: str, default: _AttentionForward
    ) -> _AttentionForward:
        if attn_implementation == "eager":
            return default
        return self[attn_implementation]


class _NativeAttentionInterface:
    def __init__(self) -> None:
        self._entries: dict[str, _AttentionForward] = {}
        self.events: list[tuple[str, str]] = []

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def get(self, key: str, default: object | None = None) -> object | None:
        return self._entries.get(key, default)

    def pop(self, key: str, default: object | None = None) -> object | None:
        self.events.append(("pop", key))
        return self._entries.pop(key, default)

    def register(self, key: str, value: _AttentionForward) -> None:
        self.events.append(("register", key))
        self._entries[key] = value

    def get_interface(
        self, attn_implementation: str, default: _AttentionForward
    ) -> _AttentionForward:
        if attn_implementation == "eager":
            return default
        return self._entries[attn_implementation]


class _PinnedGeneralInterface:
    _global_mapping: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        *,
        label: str,
        event_log: list[tuple[str, str, str]] | None = None,
    ) -> None:
        self._local_mapping: dict[str, object] = {}
        self.events: list[tuple[str, str]] = []
        self._label = label
        self._event_log = event_log

    def _record(self, action: str, key: str) -> None:
        self.events.append((action, key))
        if self._event_log is not None:
            self._event_log.append((self._label, action, key))

    def has_local(self, key: str) -> bool:
        return key in self._local_mapping

    def has_global(self, key: str) -> bool:
        return key in type(self)._global_mapping

    def get_local(self, key: str) -> object | None:
        return self._local_mapping.get(key)

    def get_global(self, key: str) -> object | None:
        return type(self)._global_mapping.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._local_mapping or key in type(self)._global_mapping

    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for key in self._local_mapping:
            seen.add(key)
            yield key
        for key in type(self)._global_mapping:
            if key not in seen:
                yield key

    def get(self, key: str, default: object | None = None) -> object | None:
        if key in self._local_mapping:
            return self._local_mapping[key]
        return type(self)._global_mapping.get(key, default)

    def pop(self, key: str, default: object | None = None) -> object | None:
        self._record("pop_local", key)
        return self._local_mapping.pop(key, default)

    def __delitem__(self, key: str) -> None:
        self._record("delete_local", key)
        del self._local_mapping[key]

    def register(self, key: str, value: object) -> None:
        self._record("register_global", key)
        type(self)._global_mapping[key] = value

    def __setitem__(self, key: str, value: object) -> None:
        self._record("set_local", key)
        self._local_mapping[key] = value

    def get_interface(
        self, attn_implementation: str, default: _AttentionForward
    ) -> _AttentionForward:
        if attn_implementation == "eager":
            return default
        value = self.get(attn_implementation)
        if value is None:
            raise KeyError(attn_implementation)
        return cast(_AttentionForward, value)


class _FailingPinnedGeneralInterface(_PinnedGeneralInterface):
    def __init__(
        self,
        *,
        label: str,
        event_log: list[tuple[str, str, str]] | None = None,
        failure_message: str,
    ) -> None:
        super().__init__(label=label, event_log=event_log)
        self._failure_message = failure_message
        self._validation_failure_armed = False
        self._failed_validation_install = False

    def arm_validation_failure(self) -> None:
        self._validation_failure_armed = True

    def register(self, key: str, value: object) -> None:
        if (
            self._validation_failure_armed
            and key.startswith("si_rebuttal_validation_")
            and not self._failed_validation_install
        ):
            self._failed_validation_install = True
            self._record("register_global_fail", key)
            raise RuntimeError(self._failure_message)
        super().register(key, value)


class _ScaleModule(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        _init_torch_module(self)
        self.scale = scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale


def _fake_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    del dropout, kwargs
    repeated = repeat_kv_after_gqa(key, num_query_heads=query.shape[1])
    repeated_value = repeat_kv_after_gqa(value, num_query_heads=query.shape[1]).to(
        dtype=query.dtype
    )
    logits = torch.matmul(query, repeated.transpose(-1, -2))
    logits = logits * scaling
    if attention_mask is not None:
        logits = logits + attention_mask.to(torch.float32)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    probs = probs.to(dtype=query.dtype)
    output = torch.matmul(probs, repeated_value).transpose(1, 2).contiguous()
    return output, probs


def _runtime_faithful_expected_probabilities(
    full_q: torch.Tensor,
    full_k: torch.Tensor,
    full_post_mask: torch.Tensor,
    *,
    scaling: float,
) -> torch.Tensor:
    repeated = repeat_kv_after_gqa(full_k, num_query_heads=int(full_q.shape[1]))
    logits = torch.matmul(full_q, repeated.transpose(-1, -2))
    logits = logits * scaling
    logits = logits + full_post_mask
    return torch.softmax(logits, dim=-1, dtype=torch.float32).to(dtype=full_q.dtype)


class _FakeAttentionBase(torch.nn.Module):
    attention_interface = _FakeAttentionInterface()

    def __init__(
        self,
        *,
        num_heads: int,
        num_key_value_heads: int,
        q_scale: float = 1.0,
        k_scale: float = 1.0,
    ) -> None:
        _init_torch_module(self)
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.head_dim = 2
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.q_proj = torch.nn.Linear(
            num_heads * self.head_dim, num_heads * self.head_dim, bias=False
        )
        self.k_proj = torch.nn.Linear(
            num_heads * self.head_dim, num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = torch.nn.Linear(
            num_heads * self.head_dim, num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = torch.nn.Identity()
        self.q_norm = _ScaleModule(q_scale)
        self.k_norm = _ScaleModule(k_scale)
        self.config = _TestAttentionConfig(
            model_type="",
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            _attn_implementation="eager",
        )
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(num_heads * self.head_dim))
            self.k_proj.weight.copy_(
                torch.eye(num_key_value_heads * self.head_dim, num_heads * self.head_dim)
            )
            self.v_proj.weight.copy_(
                torch.eye(num_key_value_heads * self.head_dim, num_heads * self.head_dim)
            )
        self.q_proj = self.q_proj.to(dtype=torch.bfloat16)
        self.k_proj = self.k_proj.to(dtype=torch.bfloat16)
        self.v_proj = self.v_proj.to(dtype=torch.bfloat16)

    def _project_states(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projection_dtype = self.q_proj.weight.dtype
        hidden_states = hidden_states.to(dtype=projection_dtype)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = (
            self.k_proj(hidden_states)
            .view(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        return query_states, key_states, value_states

    def _dispatch(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if position_embeddings is None:
            raise RuntimeError("position_embeddings required")
        query_states, key_states, value_states = self._project_states(hidden_states)
        query_states, key_states = _apply_test_rope(query_states, key_states, position_embeddings)
        attention_interface = self.attention_interface.get_interface(
            _config_private_attn_implementation(self.config), _fake_eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(
            hidden_states.shape[0], hidden_states.shape[1], -1
        ).contiguous()
        return self.o_proj(attn_output), attn_weights


class LlamaAttentionLike(_FakeAttentionBase):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        cache_position: object | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del past_key_values, cache_position
        return self._dispatch(hidden_states, position_embeddings, attention_mask, **kwargs)


class MistralAttentionLike(_FakeAttentionBase):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: object | None = None,
        cache_position: object | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del past_key_values, cache_position
        return self._dispatch(hidden_states, position_embeddings, attention_mask, **kwargs)


class OlmoAttentionLike(_FakeAttentionBase):
    def __init__(self, *, num_heads: int, num_key_value_heads: int) -> None:
        super().__init__(
            num_heads=num_heads, num_key_value_heads=num_key_value_heads, q_scale=2.0, k_scale=3.0
        )
        self.config = _TestAttentionConfig(
            model_type="",
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            _attn_implementation="eager",
        )

    def _project_states(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projection_dtype = self.q_proj.weight.dtype
        hidden_states = hidden_states.to(dtype=projection_dtype)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        key_states = (
            self.k_norm(self.k_proj(hidden_states))
            .view(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.num_key_value_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: object | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del past_key_values
        return self._dispatch(hidden_states, position_embeddings, attention_mask, **kwargs)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _apply_test_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    q_dtype = q.dtype
    k_dtype = k.dtype
    cos, sin = position_embeddings
    aligned_cos = cos.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    aligned_sin = sin.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    return (
        (
            (q.to(torch.float32) * aligned_cos) + (_rotate_half(q.to(torch.float32)) * aligned_sin)
        ).to(dtype=q_dtype),
        (
            (k.to(torch.float32) * aligned_cos) + (_rotate_half(k.to(torch.float32)) * aligned_sin)
        ).to(dtype=k_dtype),
    )


class _CountingPermutationGenerator:
    def __init__(
        self,
        permutations: list[Int64Array] | None = None,
        *,
        repeated_permutation: Int64Array | None = None,
        max_calls: int | None = None,
    ) -> None:
        self._permutations = (
            None
            if permutations is None
            else [np.asarray(permutation, dtype=np.int64) for permutation in permutations]
        )
        self._repeated_permutation = (
            None
            if repeated_permutation is None
            else np.asarray(repeated_permutation, dtype=np.int64)
        )
        self._max_calls = max_calls
        self.call_count = 0

    def permutation(self, n: int) -> Int64Array:
        if self._max_calls is not None and self.call_count >= self._max_calls:
            raise AssertionError("permutation called more times than expected")
        if self._permutations is not None:
            if self.call_count >= len(self._permutations):
                raise AssertionError("permutation called more times than expected")
            permutation = self._permutations[self.call_count]
        elif self._repeated_permutation is not None:
            permutation = self._repeated_permutation
        else:
            raise AssertionError("no permutation fixture configured")
        self.call_count += 1
        assert permutation.shape == (n,)
        return permutation


def _protocol_kernel() -> Float64Array:
    return np.linspace(-2.5, 3.5, 512, dtype=np.float64)


def _probe_position_embeddings(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    freqs = torch.arange(head_dim, dtype=torch.float32).unsqueeze(0) + 1.0
    angles = positions / (freqs + 1.0)
    return torch.cos(angles), torch.sin(angles)


@dataclass
class _FakeParameter:
    device: torch.device


@dataclass
class _TestAttentionConfig:
    model_type: str
    num_attention_heads: int
    num_key_value_heads: int
    _attn_implementation: str


class _PrivateAttnImplementationConfig(Protocol):
    model_type: str
    num_attention_heads: int
    num_key_value_heads: int
    _attn_implementation: str


@dataclass
class _PublicAttnImplementationConfig:
    model_type: str
    num_attention_heads: int
    num_key_value_heads: int
    attn_implementation: str


@dataclass
class _TestLayer:
    self_attn: torch.nn.Module


@dataclass
class _TestBackbone:
    layers: list[_TestLayer]


class _TestModelBase:
    config: _PrivateAttnImplementationConfig | _PublicAttnImplementationConfig
    model: _TestBackbone
    device: torch.device
    forward: ModelForward

    def __init__(
        self,
        *,
        config: _PrivateAttnImplementationConfig | _PublicAttnImplementationConfig,
        model: _TestBackbone,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device
        self.forward = _uninitialized_model_forward

    def parameters(self) -> tuple[_FakeParameter, ...]:
        return (_FakeParameter(self.device),)


class _DeviceOnlyModule(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        _init_torch_module(self)
        self.device = device
        self._device_marker = torch.nn.Parameter(
            torch.empty(0, device=device, dtype=torch.float32),
            requires_grad=False,
        )


class LlamaAttention(_DeviceOnlyModule):
    pass


class MistralAttention(_DeviceOnlyModule):
    pass


class Olmo2Attention(_DeviceOnlyModule):
    pass


InterventionFactory: TypeAlias = Callable[..., EagerAttentionIntervention]
ModelInvoker: TypeAlias = Callable[
    [_TestModelBase],
    tuple[torch.Tensor, torch.Tensor | None],
]
AttentionModuleClass: TypeAlias = type[_DeviceOnlyModule] | type[_FakeAttentionBase]


def _expected_counts(layer_count: int, count: int = 1) -> ExpectedAttentionCallCounts:
    return ExpectedAttentionCallCounts({layer_index: count for layer_index in range(layer_count)})


def _factory_for_family(family: str) -> InterventionFactory:
    return {
        "llama-3.1-8b": create_llama_3_1_8b_intervention,
        "mistral-7b-v0.1": create_mistral_7b_v0_1_intervention,
        "olmo-2-7b": create_olmo_2_7b_intervention,
    }[family]


def _build_public_intervention(
    *,
    family: str,
    model: _TestModelBase,
    selected_heads_by_layer: dict[int, tuple[int, ...]] | None = None,
    kernels_by_layer_head: dict[tuple[int, int], torch.Tensor] | None = None,
    validation_probe: ValidationProbeRequest | None = None,
) -> EagerAttentionIntervention:
    if selected_heads_by_layer is None:
        selected_heads_by_layer = {0: (0,)}
    if kernels_by_layer_head is None:
        kernels_by_layer_head = {
            (layer_index, head_index): torch.ones(512)
            for layer_index, heads in selected_heads_by_layer.items()
            for head_index in heads
        }
    return _factory_for_family(family)(
        model,
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer=selected_heads_by_layer,
        kernels_by_layer_head=kernels_by_layer_head,
        validation_probe=validation_probe,
    )


def _patch_family_registry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake_interface: (
        _FakeAttentionInterface | _NativeAttentionInterface | _PinnedGeneralInterface | None
    ) = None,
    mask_registry: (
        _FakeAttentionInterface | _NativeAttentionInterface | _PinnedGeneralInterface | None
    ) = None,
) -> None:
    attention_registry = fake_interface or _FakeAttentionInterface()

    def family_adapter_kwargs() -> dict[str, object]:
        if "mask_attention_interface" not in inspect.signature(FamilyAdapter).parameters:
            return {}
        return {
            "mask_attention_interface": (
                mask_registry if mask_registry is not None else attention_registry
            )
        }

    registry = {
        "llama-3.1-8b": FamilyAdapter(
            family_name="llama-3.1-8b",
            model_class=type("LlamaForCausalLM", (_TestModelBase,), {}),
            config_class=type("LlamaConfig", (_TestAttentionConfig,), {}),
            config_model_type="llama",
            attention_class=LlamaAttentionLike,
            eager_attention_forward=_fake_eager_attention_forward,
            attention_interface=attention_registry,
            kv_head_count=8,
            gqa_repeat_factor=4,
            attention_mask_layout=AttentionMaskForwardLayout(
                arg_name="attention_mask",
                rank=4,
                axes=("batch", "head", "query_position", "key_position"),
            ),
            forward_signature=inspect.signature(LlamaAttentionLike.forward),
            **family_adapter_kwargs(),
        ),
        "mistral-7b-v0.1": FamilyAdapter(
            family_name="mistral-7b-v0.1",
            model_class=type("MistralForCausalLM", (_TestModelBase,), {}),
            config_class=type("MistralConfig", (_TestAttentionConfig,), {}),
            config_model_type="mistral",
            attention_class=MistralAttentionLike,
            eager_attention_forward=_fake_eager_attention_forward,
            attention_interface=attention_registry,
            kv_head_count=8,
            gqa_repeat_factor=4,
            attention_mask_layout=AttentionMaskForwardLayout(
                arg_name="attention_mask",
                rank=4,
                axes=("batch", "head", "query_position", "key_position"),
            ),
            forward_signature=inspect.signature(MistralAttentionLike.forward),
            **family_adapter_kwargs(),
        ),
        "olmo-2-7b": FamilyAdapter(
            family_name="olmo-2-7b",
            model_class=type("Olmo2ForCausalLM", (_TestModelBase,), {}),
            config_class=type("Olmo2Config", (_TestAttentionConfig,), {}),
            config_model_type="olmo2",
            attention_class=OlmoAttentionLike,
            eager_attention_forward=_fake_eager_attention_forward,
            attention_interface=attention_registry,
            kv_head_count=32,
            gqa_repeat_factor=1,
            attention_mask_layout=AttentionMaskForwardLayout(
                arg_name="attention_mask",
                rank=4,
                axes=("batch", "head", "query_position", "key_position"),
            ),
            forward_signature=inspect.signature(OlmoAttentionLike.forward),
            **family_adapter_kwargs(),
        ),
    }
    _test_family_registry.clear()
    _test_family_registry.update(registry)
    monkeypatch.setattr(intervention_module, "_FAMILY_REGISTRY", registry)


def _make_pinned_general_interface(
    *,
    label: str,
    event_log: list[tuple[str, str, str]] | None = None,
) -> _PinnedGeneralInterface:
    interface_cls = cast(
        type[_PinnedGeneralInterface],
        type(
            f"_{label.title().replace('_', '')}PinnedGeneralInterface",
            (_PinnedGeneralInterface,),
            {"_global_mapping": {}},
        ),
    )
    return interface_cls(label=label, event_log=event_log)


def _make_failing_pinned_general_interface(
    *,
    label: str,
    failure_message: str,
    event_log: list[tuple[str, str, str]] | None = None,
) -> _FailingPinnedGeneralInterface:
    interface_cls = cast(
        type[_FailingPinnedGeneralInterface],
        type(
            f"_{label.title().replace('_', '')}FailingPinnedGeneralInterface",
            (_FailingPinnedGeneralInterface,),
            {"_global_mapping": {}},
        ),
    )
    return interface_cls(
        label=label,
        event_log=event_log,
        failure_message=failure_message,
    )


def _patch_mask_attention_registry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mask_registry: _PinnedGeneralInterface,
) -> None:
    try:
        from transformers import masking_utils
    except ImportError:
        transformers_module = sys.modules.get("transformers")
        if transformers_module is None:
            transformers_module = types.ModuleType("transformers")
            monkeypatch.setitem(sys.modules, "transformers", transformers_module)
        masking_utils = types.ModuleType("transformers.masking_utils")
        monkeypatch.setitem(sys.modules, "transformers.masking_utils", masking_utils)
        monkeypatch.setattr(
            transformers_module,
            "masking_utils",
            masking_utils,
            raising=False,
        )
    monkeypatch.setattr(
        masking_utils,
        "ALL_MASK_ATTENTION_FUNCTIONS",
        mask_registry,
        raising=False,
    )
    cached_masking_utils = getattr(intervention_module, "masking_utils", None)
    if cached_masking_utils is not None:
        monkeypatch.setattr(
            cached_masking_utils,
            "ALL_MASK_ATTENTION_FUNCTIONS",
            mask_registry,
            raising=False,
        )


def _patch_validation_registry_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    attention_registry: _PinnedGeneralInterface,
    mask_registry: _PinnedGeneralInterface,
) -> None:
    monkeypatch.setattr(LlamaAttentionLike, "attention_interface", attention_registry)
    monkeypatch.setattr(MistralAttentionLike, "attention_interface", attention_registry)
    monkeypatch.setattr(OlmoAttentionLike, "attention_interface", attention_registry)
    _patch_family_registry(
        monkeypatch,
        fake_interface=attention_registry,
        mask_registry=mask_registry,
    )
    _patch_mask_attention_registry(monkeypatch, mask_registry=mask_registry)


def _validation_key(intervention: EagerAttentionIntervention) -> str:
    return f"si_rebuttal_validation_{id(intervention)}"


def _seed_validation_entry(
    registry: _PinnedGeneralInterface,
    *,
    validation_key: str,
    scope: str | None,
    value: object,
) -> None:
    if scope == "global":
        registry.register(validation_key, value)
        return
    if scope == "local":
        registry[validation_key] = value


def _assert_validation_entry_state(
    registry: _PinnedGeneralInterface,
    *,
    validation_key: str,
    scope: str | None,
    value: object | None,
) -> None:
    if scope == "global":
        assert registry.get_global(validation_key) is value
        assert not registry.has_local(validation_key)
        assert registry.get(validation_key) is value
        return
    if scope == "local":
        assert registry.get_local(validation_key) is value
        assert not registry.has_global(validation_key)
        assert registry.get(validation_key) is value
        return
    assert not registry.has_global(validation_key)
    assert not registry.has_local(validation_key)
    assert validation_key not in registry
    assert registry.get(validation_key) is None


def _set_family_adapter_mask_attention_interface(value: object) -> None:
    for adapter in _test_family_registry.values():
        object.__setattr__(adapter, "mask_attention_interface", value)


def _build_model(
    *,
    family: str,
    attn_implementation: str = "eager",
    layer_count: int = ADR_LAYER_COUNT,
    query_heads: int = 32,
    kv_heads: int = 8,
    wrong_attention_class: bool = False,
    wrong_config_class: bool = False,
    attention_cls: AttentionModuleClass | None = None,
) -> _TestModelBase:
    device = torch.device("cpu")
    adapter = _test_family_registry[family]
    model_cls = cast(type[_TestModelBase], adapter.model_class)
    config_cls = (
        cast(type[_TestAttentionConfig], type("WrongConfig", (_TestAttentionConfig,), {}))
        if wrong_config_class
        else cast(type[_TestAttentionConfig], adapter.config_class)
    )
    if attention_cls is None:
        attention_cls = cast(AttentionModuleClass, adapter.attention_class)
    if wrong_attention_class:
        attention_cls = type("WrongAttention", (_DeviceOnlyModule,), {})
    config = config_cls(
        model_type=adapter.config_model_type,
        num_attention_heads=query_heads,
        num_key_value_heads=kv_heads,
        _attn_implementation=attn_implementation,
    )
    layers: list[_TestLayer] = []
    for _ in range(layer_count):
        if issubclass(attention_cls, _DeviceOnlyModule):
            attn: torch.nn.Module = attention_cls(device)
        else:
            fake_attn = attention_cls(
                num_heads=query_heads,
                num_key_value_heads=kv_heads,
            )
            fake_attn.config = config
            attn = fake_attn
        layers.append(_TestLayer(self_attn=attn))
    model = model_cls(
        config=config,
        model=_TestBackbone(layers=layers),
        device=device,
    )

    def forward(
        self: _TestModelBase,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        current = hidden_states
        attn_weights: torch.Tensor | None = None
        for layer in self.model.layers:
            current, attn_weights = layer.self_attn(
                current,
                position_embeddings,
                attention_mask,
                **kwargs,
            )
        return current, attn_weights

    model.forward = MethodType(forward, model)
    return model


def _base_mask(seq_len: int) -> torch.Tensor:
    mask = torch.full((1, 32, seq_len, seq_len), float("-inf"), dtype=torch.float32)
    lower = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    mask = mask.masked_fill(lower.unsqueeze(0).unsqueeze(0), 0.0)
    return mask


def test_source_estimator_is_distinct_from_raw_kernel_estimator() -> None:
    seq_len = 512
    source_offsets = np.arange(SOURCE_OFFSET_START, SOURCE_OFFSET_STOP + 1, dtype=np.float64)
    diag_signal = 0.4 * np.cos(2.0 * np.pi * source_offsets / 17.0) + 0.2 * np.sin(
        2.0 * np.pi * source_offsets / 31.0
    )
    row_bias_a = np.linspace(-2.0, 3.0, seq_len, dtype=np.float64)
    col_bias_a = np.linspace(1.0, -1.5, seq_len, dtype=np.float64)
    row_bias_b = row_bias_a[::-1] * 0.5
    col_bias_b = col_bias_a[::-1] * -0.25
    matrices = np.stack(
        [
            _make_rms_sequence(row_bias_a, col_bias_a, diag_signal),
            _make_rms_sequence(row_bias_b, col_bias_b, diag_signal),
        ],
        axis=0,
    )

    result = estimate_source_scores(matrices, norm_name="RMSNorm")
    raw_kernel = estimate_raw_intervention_kernel(matrices)

    manual_centered = _manual_double_center(matrices[0])
    manual_means = _manual_lower_diag_means(
        manual_centered, SOURCE_OFFSET_START, SOURCE_OFFSET_STOP
    )
    manual_smoothed = _manual_fft_topk(manual_means, TOP_NON_DC_COMPONENTS)
    manual_r2 = _manual_pair_weighted_r2(manual_centered, manual_smoothed, SOURCE_OFFSET_START)
    manual_raw = np.mean(
        [
            _manual_lower_diag_means(matrices[0], 0, 511),
            _manual_lower_diag_means(matrices[1], 0, 511),
        ],
        axis=0,
    )

    assert np.isclose(result.per_sequence_r2[0], manual_r2, atol=1e-5)
    assert np.allclose(result.smoothed_means[0], manual_smoothed, atol=1e-5)
    assert np.allclose(raw_kernel, manual_raw, atol=1e-5)
    assert not np.allclose(raw_kernel[1:], result.smoothed_means.mean(axis=0), atol=1e-3)


def test_sort_heads_into_bins_uses_stable_tuple_order_and_array_split() -> None:
    scores = np.full((4, 10), 0.5, dtype=np.float64)
    scores[0, 0] = 0.1
    scores[3, 9] = 0.9
    bins = sort_heads_into_bins(scores)
    assert len(bins) == 20
    assert all(len(group) == 2 for group in bins)
    assert bins[0] == [HeadIndex(0, 0), HeadIndex(0, 1)]
    assert bins[-1] == [HeadIndex(3, 8), HeadIndex(3, 9)]
    means = bin_mean_source_r2(scores, bins)
    assert means.shape == (20,)
    assert np.isclose(means[0], np.mean([0.1, 0.5]))


def test_per_head_kernel_helpers_match_single_head_calls() -> None:
    base = np.zeros((2, 2, 512, 512), dtype=np.float64)
    for seq in range(2):
        for head in range(2):
            value = 0.1 * (seq + 1) * (head + 1)
            for query in range(512):
                for key in range(query + 1):
                    base[seq, head, query, key] = value + 0.01 * (query - key)
    per_sequence, mean_scores = estimate_source_scores_by_head(base, norm_name="LayerNorm")
    kernels = estimate_raw_intervention_kernels_by_head(base)
    ref_head0 = estimate_source_scores(base[:, 0], norm_name="LayerNorm")
    ref_head1 = estimate_source_scores(base[:, 1], norm_name="LayerNorm")
    raw_head0 = estimate_raw_intervention_kernel(base[:, 0])
    raw_head1 = estimate_raw_intervention_kernel(base[:, 1])
    assert np.allclose(per_sequence[:, 0], ref_head0.per_sequence_r2)
    assert np.allclose(per_sequence[:, 1], ref_head1.per_sequence_r2)
    assert np.allclose(mean_scores, np.array([ref_head0.mean_r2, ref_head1.mean_r2]))
    assert np.allclose(kernels[0], raw_head0)
    assert np.allclose(kernels[1], raw_head1)


def test_seed_derivation_and_controls_match_frozen_semantics() -> None:
    payload = [29039, "llama-3.1-8b", "wiki_to_code", "bin", 7, 2, "norm"]
    expected_seed = int.from_bytes(
        hashlib.sha256(
            json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode(
                "ascii"
            )
        ).digest()[:8],
        byteorder="big",
        signed=False,
    )
    kernel = _protocol_kernel()
    perm_seed = compact_ascii_json_seed(payload)
    manual_rng = np.random.Generator(np.random.PCG64(expected_seed))
    expected_perm = kernel[manual_rng.permutation(512)]

    assert perm_seed == expected_seed
    assert np.array_equal(permute_offsets(kernel, seed=perm_seed), expected_perm)

    norm_control = zero_mean_l2_control(kernel, seed=perm_seed)
    assert np.isclose(norm_control.mean(), 0.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(norm_control), np.linalg.norm(kernel), atol=1e-10)

    with pytest.raises(ValueError, match="shape \\(512,\\)"):
        permute_offsets(np.ones(511, dtype=np.float64), seed=perm_seed)

    with pytest.raises(ValueError, match="shape \\(512,\\)"):
        zero_mean_l2_control(np.ones(513, dtype=np.float64), seed=perm_seed)

    with pytest.raises(ValueError, match="zero kernel norm"):
        zero_mean_l2_control(np.zeros(512, dtype=np.float64), seed=perm_seed)


def test_bootstrap_and_small_budget_spearman_statistics_are_deterministic() -> None:
    baseline = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    condition = np.array([1.0, 3.0, 5.0, 8.0], dtype=np.float64)
    seed = 123456
    bootstrap = paired_percentile_bootstrap(baseline, condition, seed=seed, n_resamples=256)

    rng = np.random.Generator(np.random.PCG64(seed))
    indices = cast(_IntegerGenerator, rng).integers(
        0,
        baseline.shape[0],
        size=(256, baseline.shape[0]),
        endpoint=False,
    )
    manual_samples = (condition[indices] - baseline[indices]).mean(axis=1)
    manual_ci = cast(
        Float64Array,
        np.percentile(manual_samples, [2.5, 97.5], method="linear"),
    )

    assert np.isclose(bootstrap.point_estimate, np.mean(condition - baseline))
    assert np.allclose(bootstrap.samples, manual_samples)
    assert np.isclose(bootstrap.ci_low, manual_ci[0])
    assert np.isclose(bootstrap.ci_high, manual_ci[1])

    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0], dtype=np.float64)
    stat_seed = 77
    result = monte_carlo_spearman_positive(x, y, seed=stat_seed, n_permutations=200, chunk_size=25)

    observed = _spearmanr(x, y)
    rng = np.random.Generator(np.random.PCG64(stat_seed))
    x_ranks = _rankdata(x)
    y_ranks = _rankdata(y)
    x_centered = x_ranks - x_ranks.mean()
    x_norm = np.linalg.norm(x_centered)
    exceed = 0
    remaining = 200
    while remaining > 0:
        batch = min(25, remaining)
        perms = np.stack([rng.permutation(y.shape[0]) for _ in range(batch)], axis=0)
        permuted = y_ranks[perms]
        centered = permuted - permuted.mean(axis=1, keepdims=True)
        rho = (centered @ x_centered) / (np.linalg.norm(centered, axis=1) * x_norm)
        exceed += int(np.count_nonzero(rho >= observed.statistic))
        remaining -= batch
    expected_p = (1 + exceed) / 201

    assert np.isclose(result.rho_observed, observed.statistic)
    assert np.isclose(result.p_two_sided_scipy, observed.pvalue)
    assert result.permutation_count == 200
    assert result.permutation_exceedance_count == exceed
    assert np.isclose(result.p_one_sided, expected_p)


def test_spearman_permutation_uses_exact_stream_order_fixture() -> None:
    x = np.arange(20, dtype=np.float64)
    y = np.array([(3 * index + 5) % 20 for index in range(20)], dtype=np.float64)
    observed = _spearmanr(x, y)
    x_ranks = _rankdata(x)
    y_ranks = _rankdata(y)
    x_centered = x_ranks - x_ranks.mean()
    x_norm = np.linalg.norm(x_centered)
    perms = [
        np.array([(index + shift) % 20 for index in range(20)], dtype=np.int64)
        for shift in range(7)
    ]
    fake_rng = _CountingPermutationGenerator(perms)
    expected_exceed = 0
    for permutation in perms:
        permuted = y_ranks[permutation][None, :]
        centered = permuted - permuted.mean(axis=1, keepdims=True)
        rho = (centered @ x_centered) / (np.linalg.norm(centered, axis=1) * x_norm)
        expected_exceed += int(rho[0] >= observed.statistic)

    import si_rebuttal.statistics as statistics_module

    original_factory = statistics_module.make_pcg64

    def fake_make_pcg64(seed: int) -> _CountingPermutationGenerator:
        del seed
        return fake_rng

    statistics_module.make_pcg64 = fake_make_pcg64
    try:
        result = monte_carlo_spearman_positive(
            x, y, seed=123, n_permutations=len(perms), chunk_size=3
        )
    finally:
        statistics_module.make_pcg64 = original_factory

    assert fake_rng.call_count == len(perms)
    assert result.permutation_count == len(perms)
    assert result.permutation_exceedance_count == expected_exceed
    assert np.isclose(result.p_one_sided, (1 + expected_exceed) / (len(perms) + 1))


def test_spearman_positive_null_uses_plus_one_floor_when_exceedance_count_is_zero() -> None:
    x = np.arange(6, dtype=np.float64)
    y = np.arange(6, dtype=np.float64)
    fake_rng = _CountingPermutationGenerator(
        permutations=[np.arange(5, -1, -1, dtype=np.int64) for _ in range(5)]
    )

    import si_rebuttal.statistics as statistics_module

    original_factory = statistics_module.make_pcg64

    def fake_make_pcg64(seed: int) -> _CountingPermutationGenerator:
        del seed
        return fake_rng

    statistics_module.make_pcg64 = fake_make_pcg64
    try:
        result = monte_carlo_spearman_positive(x, y, seed=123, n_permutations=5, chunk_size=2)
    finally:
        statistics_module.make_pcg64 = original_factory

    assert fake_rng.call_count == 5
    assert np.isclose(result.rho_observed, 1.0)
    assert result.permutation_count == 5
    assert result.permutation_exceedance_count == 0
    assert np.isclose(result.p_one_sided, 1.0 / 6.0)


def test_spearman_positive_null_counts_every_ge_exceedance_once() -> None:
    x = np.arange(6, dtype=np.float64)
    y = np.array([0.0, 2.0, 1.0, 3.0, 5.0, 4.0], dtype=np.float64)
    perms = [
        np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        np.array([0, 2, 1, 3, 5, 4], dtype=np.int64),
        np.array([5, 4, 3, 2, 1, 0], dtype=np.int64),
        np.array([1, 0, 3, 2, 5, 4], dtype=np.int64),
    ]
    fake_rng = _CountingPermutationGenerator(perms)

    import si_rebuttal.statistics as statistics_module

    original_factory = statistics_module.make_pcg64

    def fake_make_pcg64(seed: int) -> _CountingPermutationGenerator:
        del seed
        return fake_rng

    statistics_module.make_pcg64 = fake_make_pcg64
    try:
        result = monte_carlo_spearman_positive(
            x, y, seed=456, n_permutations=len(perms), chunk_size=3
        )
    finally:
        statistics_module.make_pcg64 = original_factory

    observed = _spearmanr(x, y).statistic
    y_ranks = _rankdata(y)
    x_centered = _rankdata(x) - _rankdata(x).mean()
    x_norm = np.linalg.norm(x_centered)
    expected_exceed = 0
    for permutation in perms:
        permuted = y_ranks[permutation][None, :]
        centered = permuted - permuted.mean(axis=1, keepdims=True)
        rho = (centered @ x_centered) / (np.linalg.norm(centered, axis=1) * x_norm)
        expected_exceed += int(rho[0] >= observed)

    assert fake_rng.call_count == len(perms)
    assert expected_exceed == 1
    assert result.permutation_count == len(perms)
    assert result.permutation_exceedance_count == expected_exceed
    assert np.isclose(result.p_one_sided, (1 + expected_exceed) / (len(perms) + 1))


def test_spearman_default_budget_calls_permutation_exactly_200000_times() -> None:
    x = np.linspace(0.0, 1.0, 20, dtype=np.float64)
    y = np.linspace(1.0, 0.0, 20, dtype=np.float64)
    fake_rng = _CountingPermutationGenerator(
        repeated_permutation=np.arange(20, dtype=np.int64),
        max_calls=DEFAULT_SPEARMAN_PERMUTATIONS,
    )

    import si_rebuttal.statistics as statistics_module

    original_factory = statistics_module.make_pcg64

    def fake_make_pcg64(seed: int) -> _CountingPermutationGenerator:
        del seed
        return fake_rng

    statistics_module.make_pcg64 = fake_make_pcg64
    try:
        result = monte_carlo_spearman_positive(x, y, seed=321, chunk_size=8192)
    finally:
        statistics_module.make_pcg64 = original_factory

    assert fake_rng.call_count == DEFAULT_SPEARMAN_PERMUTATIONS
    assert result.permutation_count == DEFAULT_SPEARMAN_PERMUTATIONS
    assert 0 <= result.permutation_exceedance_count <= DEFAULT_SPEARMAN_PERMUTATIONS
    assert 0.0 <= result.p_one_sided <= 1.0


def test_average_control_trials_per_sequence() -> None:
    trials = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    assert np.allclose(average_control_trials_per_sequence(trials), np.array([3.0, 4.0]))


def test_direct_construction_is_rejected() -> None:
    with pytest.raises(TypeError, match="not publicly constructible"):
        EagerAttentionIntervention(
            attention_modules=[],
            kernels_by_layer={},
            _construction_token=object(),
        )


def test_intervention_hook_applies_selected_head_correction_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_family_registry(monkeypatch)
    model = _build_model(
        family="llama-3.1-8b", query_heads=32, kv_heads=8, attention_cls=LlamaAttentionLike
    )
    mask = _base_mask(4)[:, :1]
    hidden = torch.zeros((1, 4, 64), dtype=torch.float32)
    rope = _probe_position_embeddings(4, 2)
    kernel = np.array([0.5, -1.0, 2.0, -0.5] + [0.0] * 508, dtype=np.float64)
    intervention = _build_public_intervention(
        family="llama-3.1-8b",
        model=model,
        selected_heads_by_layer={1: (2,)},
        kernels_by_layer_head={(1, 2): torch.as_tensor(kernel)},
    )
    correction = intervention.build_layer_correction(
        layer_index=1,
        num_query_heads=32,
        seq_len=4,
        device=torch.device("cpu"),
    )
    with intervention:
        out, _ = model.forward(hidden, rope, mask)
        assert intervention.installed_hook_count == ADR_LAYER_COUNT
        assert out.dtype == torch.bfloat16
        assert intervention.mask_captures == []
        expected = torch.tensor(
            causal_toeplitz_correction(kernel[:4], seq_len=4), dtype=torch.float32
        )
        assert torch.allclose(correction[2], expected)
        assert torch.count_nonzero(correction) == torch.count_nonzero(correction[2])

    assert intervention.hook_call_counts == {
        layer_index: 1 for layer_index in range(ADR_LAYER_COUNT)
    }
    assert intervention.installed_hook_count == 0


@pytest.mark.parametrize("preseed_scope", [None, "global", "local"])
def test_validation_install_uses_mask_registry_and_restores_exact_state(
    monkeypatch: pytest.MonkeyPatch,
    preseed_scope: str | None,
) -> None:
    event_log: list[tuple[str, str, str]] = []
    attention_registry = _make_pinned_general_interface(label="attention", event_log=event_log)
    mask_registry = _make_pinned_general_interface(label="mask", event_log=event_log)
    eager_mask = object()
    previous_attention = object()
    previous_mask = object()
    mask_registry.register("eager", eager_mask)
    _patch_validation_registry_environment(
        monkeypatch,
        attention_registry=attention_registry,
        mask_registry=mask_registry,
    )
    model = _build_model(family="mistral-7b-v0.1", attention_cls=MistralAttentionLike)
    original_impl = _model_attn_implementation(model)
    intervention = create_mistral_7b_v0_1_intervention(
        model,
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (0,)},
        kernels_by_layer_head={(0, 0): torch.ones(512)},
        validation_probe=ValidationProbeRequest(layer_index=0, head_index=0),
    )
    validation_key = _validation_key(intervention)
    _seed_validation_entry(
        attention_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_attention,
    )
    _seed_validation_entry(
        mask_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_mask,
    )
    attention_registry.events.clear()
    mask_registry.events.clear()
    event_log.clear()

    with intervention:
        assert _model_attn_implementation(model) == validation_key
        assert validation_key in attention_registry
        assert validation_key in mask_registry
        assert mask_registry.get(validation_key) is eager_mask
        if preseed_scope == "local":
            assert not attention_registry.has_local(validation_key)
            assert not mask_registry.has_local(validation_key)
        _, probs = model.forward(
            torch.zeros((1, 64, 64), dtype=torch.bfloat16),
            _probe_position_embeddings(64, 2),
            _base_mask(64)[:, :1],
        )
        assert probs is not None
    assert _model_attn_implementation(model) == original_impl
    _assert_validation_entry_state(
        attention_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_attention if preseed_scope is not None else None,
    )
    _assert_validation_entry_state(
        mask_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_mask if preseed_scope is not None else None,
    )
    assert ("attention", "register_global", validation_key) in event_log
    assert ("mask", "register_global", validation_key) in event_log


def test_intervention_hook_fails_closed_on_missing_mask_and_bad_head_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_family_registry(monkeypatch)
    model = _build_model(family="llama-3.1-8b", attention_cls=LlamaAttentionLike)
    hidden = torch.zeros((1, 4, 64), dtype=torch.float32)
    rope = _probe_position_embeddings(4, 2)
    intervention = _build_public_intervention(family="llama-3.1-8b", model=model)
    with intervention:
        with pytest.raises(ValueError, match=r"attention_mask must be a torch\.Tensor"):
            model.model.layers[0].self_attn(hidden, rope, None)

    with pytest.raises(ValueError, match="out of range"):
        _build_public_intervention(
            family="llama-3.1-8b",
            model=_build_model(family="llama-3.1-8b", attention_cls=LlamaAttentionLike),
            selected_heads_by_layer={0: (32,)},
            kernels_by_layer_head={(0, 32): torch.ones(512)},
        )


def test_build_head_kernel_map_requires_every_selected_head() -> None:
    from si_rebuttal.intervention import build_head_kernel_map

    with pytest.raises(ValueError, match="missing kernel"):
        build_head_kernel_map(
            [HeadIndex(0, 0), HeadIndex(0, 1)], {HeadIndex(0, 0): np.ones(4, dtype=np.float64)}
        )


def test_repeat_kv_and_manual_attention_reconstruction_match_explicit_gqa() -> None:
    query = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 1.0], [1.0, 2.0]],
                [[-1.0, 0.5], [0.5, -1.0]],
                [[0.0, 2.0], [2.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    key = torch.tensor(
        [
            [
                [[1.0, 1.0], [0.0, 1.0]],
                [[-1.0, 2.0], [2.0, -1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[[[0.0, -10.0], [0.0, 0.0]]]], dtype=torch.float32)
    repeated = repeat_kv_after_gqa(key, num_query_heads=4)
    expected_repeated = key.repeat_interleave(2, dim=1)
    assert torch.allclose(repeated, expected_repeated)

    probs = manual_attention_probs(query, key, additive_mask=mask, num_query_heads=4)
    manual_logits = torch.matmul(query, expected_repeated.transpose(-1, -2)) / math.sqrt(2.0)
    manual_probs = torch.softmax(manual_logits + mask.expand(1, 4, 2, 2), dim=-1)
    assert torch.allclose(probs, manual_probs)


def test_zero_and_constant_corrections_preserve_expected_attention_behavior() -> None:
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    causal_mask = torch.tensor([[[[0.0, -1000.0], [0.0, 0.0]]]], dtype=torch.float32)
    zero_correction = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    constant_correction = torch.tensor([[[[-2.0, 0.0], [-2.0, -2.0]]]], dtype=torch.float32)
    baseline = manual_attention_probs(query, key, additive_mask=causal_mask)
    zero = manual_attention_probs(query, key, additive_mask=causal_mask + zero_correction)
    constant = manual_attention_probs(query, key, additive_mask=causal_mask + constant_correction)
    assert torch.allclose(zero, baseline)
    assert torch.allclose(constant, baseline, atol=1e-6, rtol=1e-6)


def test_negative_infinity_masks_are_accepted() -> None:
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    mask = torch.tensor([[[[0.0, float("-inf")], [0.0, 0.0]]]], dtype=torch.float32)
    probs = manual_attention_probs(query, key, additive_mask=mask)
    assert torch.isfinite(probs).all()

    with pytest.raises(ValueError, match="finite values or -inf"):
        manual_attention_probs(
            query, key, additive_mask=torch.tensor([[[[0.0, float("inf")], [0.0, 0.0]]]])
        )


def test_bfloat16_input_forwards_float32_mask_and_preserves_1e6_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_family_registry(monkeypatch)
    model = _build_model(family="llama-3.1-8b", attention_cls=LlamaAttentionLike)
    hidden = torch.zeros((1, 4, 64), dtype=torch.float32)
    kernel = np.array([1e-6, 2e-6, 3e-6, 4e-6], dtype=np.float64)
    full_kernel = np.concatenate([kernel, np.zeros(508, dtype=np.float64)])
    mask = torch.zeros((1, 1, 4, 4), dtype=torch.bfloat16)
    rope = _probe_position_embeddings(4, 2)
    intervention = _build_public_intervention(
        family="llama-3.1-8b",
        model=model,
        selected_heads_by_layer={0: (1,)},
        kernels_by_layer_head={(0, 1): torch.as_tensor(full_kernel)},
    )
    with intervention:
        out, _ = model.forward(hidden, rope, mask)
        corrected = intervention.apply_attention_mask_correction(mask, layer_index=0)
        expected = torch.tensor(causal_toeplitz_correction(kernel, seq_len=4), dtype=torch.float32)
        assert out.dtype == torch.bfloat16
        assert corrected.dtype == torch.float32
        assert intervention.mask_captures == []
        assert torch.max(torch.abs(corrected[0, 1] - expected)) <= 1e-6


def test_model_family_signature_layouts_support_keyword_and_positional_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_family_registry(monkeypatch)
    hidden = torch.zeros((1, 4, 64), dtype=torch.float32)
    mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    rope = _probe_position_embeddings(4, 2)
    model_specs: list[tuple[str, _TestModelBase, ModelInvoker]] = [
        (
            "llama-3.1-8b",
            _build_model(family="llama-3.1-8b", attention_cls=LlamaAttentionLike),
            lambda model: model.forward(hidden, rope, mask),
        ),
        (
            "mistral-7b-v0.1",
            _build_model(family="mistral-7b-v0.1", attention_cls=MistralAttentionLike),
            lambda model: model.forward(hidden, rope, mask),
        ),
        (
            "olmo-2-7b",
            _build_model(family="olmo-2-7b", kv_heads=32, attention_cls=OlmoAttentionLike),
            lambda model: model.forward(hidden, rope, mask),
        ),
    ]
    for family, model, invoke in model_specs:
        with _build_public_intervention(family=family, model=model) as intervention:
            out, _ = invoke(model)
            assert out.dtype == torch.bfloat16
            assert intervention.mask_captures == []


def test_context_exit_automatically_fails_on_missing_expected_hook_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_family_registry(monkeypatch)
    model = _build_model(family="llama-3.1-8b", attention_cls=LlamaAttentionLike)
    with pytest.raises(AssertionError, match="expected 1, found 0"):
        with _build_public_intervention(family="llama-3.1-8b", model=model):
            pass


def test_revalidation_fails_closed_on_module_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_family_registry(monkeypatch)
    model = _build_model(family="llama-3.1-8b", attention_cls=LlamaAttentionLike)
    intervention = _build_public_intervention(family="llama-3.1-8b", model=model)
    model.model.layers[0].self_attn = LlamaAttentionLike(num_heads=32, num_key_value_heads=8)
    with pytest.raises(RuntimeError, match="module identity changed"):
        intervention.__enter__()
    assert intervention.installed_hook_count == 0


def test_next_token_nll_and_nonfinite_errors() -> None:
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 3.0], [2.0, 1.0]],
            [[1.0, 2.0], [3.0, 0.0], [0.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    tokens = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.long)
    values = mean_next_token_nll(logits, tokens)
    expected: list[float] = []
    for batch_index in range(tokens.shape[0]):
        losses: list[float] = []
        for position in range(1, tokens.shape[1]):
            log_probs = torch.log_softmax(logits[batch_index, position - 1], dim=-1)
            losses.append(float(-log_probs[tokens[batch_index, position]].item()))
        expected.append(float(np.mean(losses)))
    assert np.allclose(np.asarray(values.detach().cpu(), dtype=np.float32), np.asarray(expected))

    bad = np.full((2, 512, 512), 0.0, dtype=np.float64)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        estimate_raw_intervention_kernel(bad)

    with pytest.raises(ValueError, match="non-finite"):
        manual_attention_probs(
            torch.full((1, 4, 2, 2), float("nan")),
            torch.zeros((1, 2, 2, 2)),
            num_query_heads=4,
        )


def test_double_center_matches_independent_formula() -> None:
    matrix = torch.tensor([[1.0, 2.0], [3.0, 6.0]], dtype=torch.float32)
    centered = double_center_matrix(matrix)
    expected = torch.tensor(
        _manual_double_center(np.asarray(matrix.detach().cpu(), dtype=np.float64)),
        dtype=torch.float32,
    )
    assert torch.allclose(centered, expected)


def test_family_factories_bind_exact_layout_and_adr_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_family_registry(monkeypatch)
    llama = _build_model(family="llama-3.1-8b", kv_heads=8)
    mistral = _build_model(family="mistral-7b-v0.1", kv_heads=8)
    olmo = _build_model(family="olmo-2-7b", kv_heads=ADR_OLMO_KV_HEADS)
    llama_spec = derive_supported_model_family_spec_for_capture(llama, "llama-3.1-8b")
    mistral_spec = derive_supported_model_family_spec_for_capture(mistral, "mistral-7b-v0.1")
    olmo_spec = derive_supported_model_family_spec_for_capture(olmo, "olmo-2-7b")
    assert llama_spec.layer_count == 32
    assert mistral_spec.query_head_count == 32
    assert llama_spec.kv_head_count == 8 and llama_spec.gqa_repeat_factor == 4
    assert mistral_spec.kv_head_count == 8 and mistral_spec.gqa_repeat_factor == 4
    assert olmo_spec.kv_head_count == 32 and olmo_spec.gqa_repeat_factor == 1
    assert llama_spec.attention_mask_layout == AttentionMaskForwardLayout(
        arg_name="attention_mask",
        rank=4,
        axes=("batch", "head", "query_position", "key_position"),
    )
    assert tuple(llama_spec.attention_forward_signature.parameters) == (
        "self",
        "hidden_states",
        "position_embeddings",
        "attention_mask",
        "past_key_values",
        "cache_position",
        "kwargs",
    )
    assert llama_spec.attention_forward_signature.parameters["position_embeddings"].default is None
    assert (
        mistral_spec.attention_forward_signature.parameters["position_embeddings"].default
        is inspect.Parameter.empty
    )
    assert (
        mistral_spec.attention_forward_signature.parameters["attention_mask"].default
        is inspect.Parameter.empty
    )
    assert tuple(olmo_spec.attention_forward_signature.parameters) == (
        "self",
        "hidden_states",
        "position_embeddings",
        "attention_mask",
        "past_key_values",
        "kwargs",
    )
    assert (
        olmo_spec.attention_forward_signature.parameters["attention_mask"].default
        is inspect.Parameter.empty
    )

    try:
        from transformers.models.llama.modeling_llama import LlamaAttention as RealLlamaAttention
        from transformers.models.mistral.modeling_mistral import (
            MistralAttention as RealMistralAttention,
        )
        from transformers.models.olmo2.modeling_olmo2 import Olmo2Attention as RealOlmo2Attention
    except ImportError:
        return

    assert inspect.signature(RealLlamaAttention.forward) == inspect.signature(
        LlamaAttentionLike.forward
    )
    assert inspect.signature(RealMistralAttention.forward) == inspect.signature(
        MistralAttentionLike.forward
    )
    assert tuple(inspect.signature(RealOlmo2Attention.forward).parameters) in (
        tuple(inspect.signature(OlmoAttentionLike.forward).parameters),
        (
            "self",
            "hidden_states",
            "position_embeddings",
            "attention_mask",
            "past_key_values",
            "cache_position",
            "kwargs",
        ),
    )


def test_family_factory_rejects_wrong_class_config_or_eager_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_family_registry(monkeypatch)
    with pytest.raises(TypeError, match="model class"):
        derive_supported_model_family_spec_for_capture(
            _build_model(family="llama-3.1-8b"), "mistral-7b-v0.1"
        )
    with pytest.raises(TypeError, match="config class"):
        derive_supported_model_family_spec_for_capture(
            _build_model(family="llama-3.1-8b", wrong_config_class=True),
            "llama-3.1-8b",
        )
    with pytest.raises(TypeError, match="attn_implementation='eager'"):
        derive_supported_model_family_spec_for_capture(
            _build_model(family="olmo-2-7b", kv_heads=32, attn_implementation="sdpa"),
            "olmo-2-7b",
        )
    with pytest.raises(TypeError, match="attention class"):
        derive_supported_model_family_spec_for_capture(
            _build_model(family="mistral-7b-v0.1", wrong_attention_class=True),
            "mistral-7b-v0.1",
        )


def test_public_wrappers_delegate_to_validated_family_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_family_registry(monkeypatch)
    model = _build_model(family="llama-3.1-8b", attention_cls=LlamaAttentionLike)
    spec = derive_supported_model_family_spec_for_capture(model, "llama-3.1-8b")
    assert spec.family_name == "llama-3.1-8b"
    assert attn_implementation_attr_name(model) == "_attn_implementation"

    public_config_model = _build_model(family="mistral-7b-v0.1", attention_cls=MistralAttentionLike)
    public_config_model.config = _PublicAttnImplementationConfig(
        model_type="mistral",
        num_attention_heads=32,
        num_key_value_heads=8,
        attn_implementation="eager",
    )
    assert attn_implementation_attr_name(public_config_model) == "attn_implementation"


def test_public_factory_requires_complete_per_layer_counts_and_has_no_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_family_registry(monkeypatch)
    model = _build_model(family="llama-3.1-8b")
    with pytest.raises(TypeError):
        create_llama_3_1_8b_intervention(
            model,
            expected_call_counts=None,
            selected_heads_by_layer={0: (0,)},
            kernels_by_layer_head={(0, 0): torch.ones(512)},
        )
    with pytest.raises(ValueError, match="complete per-layer attention call mapping"):
        create_llama_3_1_8b_intervention(
            model,
            expected_call_counts=ExpectedAttentionCallCounts(
                {layer_index: 1 for layer_index in range(31)}
            ),
            selected_heads_by_layer={0: (0,)},
            kernels_by_layer_head={(0, 0): torch.ones(512)},
        )
    mistral_intervention = create_mistral_7b_v0_1_intervention(
        _build_model(family="mistral-7b-v0.1"),
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (0,)},
        kernels_by_layer_head={(0, 0): torch.ones(512)},
    )
    assert mistral_intervention.validated_spec is not None
    assert mistral_intervention.validated_spec.family_name == "mistral-7b-v0.1"
    olmo_intervention = create_olmo_2_7b_intervention(
        _build_model(family="olmo-2-7b", kv_heads=32),
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (0,)},
        kernels_by_layer_head={(0, 0): torch.ones(512)},
    )
    assert olmo_intervention.validated_spec is not None
    assert olmo_intervention.validated_spec.family_name == "olmo-2-7b"
    assert not hasattr(intervention_module, "_private_test_only_create_attention_intervention")


def test_validation_probe_capture_retains_full_bf16_runtime_tensors_and_selected_head_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention_registry = _make_pinned_general_interface(label="attention")
    mask_registry = _make_pinned_general_interface(label="mask")
    mask_registry.register("eager", object())
    _patch_validation_registry_environment(
        monkeypatch,
        attention_registry=attention_registry,
        mask_registry=mask_registry,
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_apply_test_rope",
        _apply_test_rope_bfloat16,
    )
    hidden = torch.arange(1 * 64 * 64, dtype=torch.float32).reshape(1, 64, 64) / 100.0
    mask = _base_mask(64)[:, :1]
    position_embeddings = _probe_position_embeddings(64, 2)
    model = _build_model(
        family="mistral-7b-v0.1",
        query_heads=32,
        kv_heads=8,
        attention_cls=MistralAttentionLike,
    )
    intervention = create_mistral_7b_v0_1_intervention(
        model,
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (1,)},
        kernels_by_layer_head={(0, 1): torch.linspace(0.0, 1.0, 512)},
        validation_probe=ValidationProbeRequest(layer_index=0, head_index=1),
    )
    with intervention:
        _, probs = model.forward(
            hidden,
            position_embeddings,
            mask,
        )
        assert probs is not None
        assert probs.dtype == torch.bfloat16
    record = intervention.latest_validation_record()
    assert record is not None
    first_layer = cast(MistralAttentionLike, model.model.layers[0].self_attn)
    finite = torch.isfinite(record.pre_mask)
    assert record.full_q.shape == (1, 32, 64, 2)
    assert record.full_k.shape == (1, 8, 64, 2)
    assert record.full_post_mask.shape == (1, 32, 64, 64)
    assert record.full_q.dtype == torch.bfloat16
    assert record.full_k.dtype == torch.bfloat16
    assert record.full_post_mask.dtype == torch.float32
    assert record.scaling == first_layer.scaling
    assert record.q.dtype == torch.float32
    assert record.k.dtype == torch.float32
    assert record.attention_probabilities.dtype == torch.float32
    assert torch.equal(record.q, record.full_q[0, 1].to(torch.float32))
    repeated = repeat_kv_after_gqa(record.full_k, num_query_heads=record.full_q.shape[1])
    assert torch.equal(record.k, repeated[0, 1].to(torch.float32))
    assert torch.equal(record.post_mask, record.full_post_mask[0, 1])
    reconstructed = record.runtime_faithful_attention_probabilities()
    assert reconstructed.shape == (1, 32, 64, 64)
    assert reconstructed.dtype == torch.bfloat16
    assert torch.allclose(
        reconstructed[0, 1].to(torch.float32),
        record.attention_probabilities,
        atol=5e-3,
        rtol=5e-3,
    )
    assert not torch.allclose(
        record.manual_reconstruction(), record.attention_probabilities, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        record.correction[finite], (record.post_mask - record.pre_mask)[finite], atol=1e-6, rtol=0.0
    )
    assert torch.isneginf(record.pre_mask[~finite]).all()
    assert torch.isneginf(record.post_mask[~finite]).all()
    assert not torch.isnan(record.correction).any()


def test_validation_probe_capture_for_olmo_preserves_full_normed_qk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention_registry = _make_pinned_general_interface(label="attention")
    mask_registry = _make_pinned_general_interface(label="mask")
    mask_registry.register("eager", object())
    _patch_validation_registry_environment(
        monkeypatch,
        attention_registry=attention_registry,
        mask_registry=mask_registry,
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_apply_test_rope",
        _apply_test_rope_bfloat16,
    )
    hidden = torch.arange(1 * 64 * 64, dtype=torch.float32).reshape(1, 64, 64) / 50.0
    mask = _base_mask(64)[:, :1]
    position_embeddings = _probe_position_embeddings(64, 2)
    model = _build_model(
        family="olmo-2-7b",
        query_heads=32,
        kv_heads=32,
        attention_cls=OlmoAttentionLike,
    )
    intervention = create_olmo_2_7b_intervention(
        model,
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (0,)},
        kernels_by_layer_head={(0, 0): torch.linspace(-1.0, 1.0, 512)},
        validation_probe=ValidationProbeRequest(layer_index=0, head_index=0),
    )
    with intervention:
        _, probs = model.forward(hidden, position_embeddings, mask)
        assert probs is not None
    record = intervention.latest_validation_record()
    assert record is not None
    first_layer = cast(OlmoAttentionLike, model.model.layers[0].self_attn)
    hidden_bf16 = hidden.to(dtype=first_layer.q_proj.weight.dtype)
    q_raw = first_layer.q_proj(hidden_bf16).view(1, 64, 32, 2).transpose(1, 2)
    k_raw = first_layer.k_proj(hidden_bf16).view(1, 64, 32, 2).transpose(1, 2)
    q_normed, k_normed = _apply_test_rope(
        first_layer.q_norm(q_raw),
        first_layer.k_norm(k_raw),
        position_embeddings,
    )
    assert record.full_q.shape == (1, 32, 64, 2)
    assert record.full_k.shape == (1, 32, 64, 2)
    assert record.full_post_mask.shape == (1, 32, 64, 64)
    assert record.full_q.dtype == torch.bfloat16
    assert record.full_k.dtype == torch.bfloat16
    assert record.full_post_mask.dtype == torch.float32
    assert record.scaling == first_layer.scaling
    assert torch.equal(record.full_q, q_normed)
    assert torch.equal(record.full_k, k_normed)
    assert torch.equal(record.q, record.full_q[0, 0].to(torch.float32))
    assert torch.equal(record.k, record.full_k[0, 0].to(torch.float32))
    assert torch.equal(record.post_mask, record.full_post_mask[0, 0])
    reconstructed = record.runtime_faithful_attention_probabilities()
    assert reconstructed.shape == (1, 32, 64, 64)
    assert reconstructed.dtype == torch.bfloat16
    assert torch.allclose(
        reconstructed[0, 0].to(torch.float32),
        record.attention_probabilities,
        atol=5e-3,
        rtol=5e-3,
    )
    assert torch.allclose(
        record.manual_reconstruction(), record.attention_probabilities, atol=1e-6, rtol=1e-6
    )


def test_validation_probe_record_runtime_faithful_attention_probabilities_is_shape_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_q = (torch.arange(1 * 32 * 64 * 2, dtype=torch.float32).reshape(1, 32, 64, 2) / 64.0).to(
        torch.bfloat16
    )
    full_k = (
        torch.arange(1 * 8 * 64 * 2, dtype=torch.float32).reshape(1, 8, 64, 2).flip(-1) / 48.0
    ).to(torch.bfloat16)
    full_post_mask = _base_mask(64).expand(1, 32, 64, 64).to(torch.float32).clone()
    full_post_mask[0, :, 32:, :32] = full_post_mask[0, :, 32:, :32] + 0.125
    scaling = 0.625
    expected = _runtime_faithful_expected_probabilities(
        full_q,
        full_k,
        full_post_mask,
        scaling=scaling,
    )
    selected_head = 5
    record = ValidationProbeRecord(
        family_name="mistral-7b-v0.1",
        layer_index=0,
        head_index=selected_head,
        pre_mask=full_post_mask[0, selected_head].clone(),
        post_mask=full_post_mask[0, selected_head].clone(),
        correction=torch.zeros((64, 64), dtype=torch.float32),
        q=full_q[0, selected_head].to(torch.float32).clone(),
        k=repeat_kv_after_gqa(full_k, num_query_heads=full_q.shape[1])[0, selected_head]
        .to(torch.float32)
        .clone(),
        attention_probabilities=expected[0, selected_head].to(torch.float32).clone(),
        full_q=full_q.clone(),
        full_k=full_k.clone(),
        full_post_mask=full_post_mask.clone(),
        scaling=scaling,
    )

    original_matmul = torch.matmul
    original_softmax = intervention_module.F.softmax
    matmul_calls: list[tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.dtype]] = []
    softmax_inputs: list[torch.Tensor] = []
    softmax_dtypes: list[torch.dtype | None] = []

    def recording_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        matmul_calls.append((tuple(left.shape), tuple(right.shape), left.dtype, right.dtype))
        return original_matmul(left, right)

    def recording_softmax(
        value: torch.Tensor,
        dim: int | None = None,
        _stacklevel: int = 3,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del _stacklevel
        softmax_inputs.append(value.detach().clone())
        softmax_dtypes.append(dtype)
        return original_softmax(value, dim=dim, dtype=dtype)

    monkeypatch.setattr(torch, "matmul", recording_matmul)
    monkeypatch.setattr(intervention_module.F, "softmax", recording_softmax)

    reconstructed = record.runtime_faithful_attention_probabilities()

    assert matmul_calls == [((1, 32, 64, 2), (1, 32, 2, 64), torch.bfloat16, torch.bfloat16)]
    assert softmax_dtypes == [torch.float32]
    assert reconstructed.shape == (1, 32, 64, 64)
    assert reconstructed.dtype == torch.bfloat16
    assert torch.equal(reconstructed[0, selected_head], expected[0, selected_head])
    repeated = repeat_kv_after_gqa(full_k, num_query_heads=full_q.shape[1])
    expected_pre_softmax = (
        original_matmul(full_q, repeated.transpose(-1, -2)) * scaling
    ) + full_post_mask
    assert torch.equal(softmax_inputs[0], expected_pre_softmax)


@pytest.mark.parametrize(
    ("drifted_field", "full_q_dtype", "full_k_dtype"),
    [
        ("full_q", torch.float32, torch.bfloat16),
        ("full_k", torch.bfloat16, torch.float32),
    ],
)
def test_validation_probe_record_rejects_fp32_drift_in_full_q_or_full_k(
    drifted_field: str,
    full_q_dtype: torch.dtype,
    full_k_dtype: torch.dtype,
) -> None:
    record = ValidationProbeRecord(
        family_name="mistral-7b-v0.1",
        layer_index=0,
        head_index=0,
        pre_mask=_base_mask(64)[0, 0].to(torch.float32),
        post_mask=_base_mask(64)[0, 0].to(torch.float32),
        correction=torch.zeros((64, 64), dtype=torch.float32),
        q=torch.zeros((64, 2), dtype=torch.float32),
        k=torch.zeros((64, 2), dtype=torch.float32),
        attention_probabilities=torch.zeros((64, 64), dtype=torch.float32),
        full_q=torch.zeros((1, 32, 64, 2), dtype=full_q_dtype),
        full_k=torch.zeros((1, 8, 64, 2), dtype=full_k_dtype),
        full_post_mask=_base_mask(64).expand(1, 32, 64, 64).to(torch.float32),
        scaling=0.5,
    )
    with pytest.raises((TypeError, ValueError), match=drifted_field):
        record.runtime_faithful_attention_probabilities()


def test_validation_capture_rejects_singleton_full_post_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention_registry = _make_pinned_general_interface(label="attention")
    mask_registry = _make_pinned_general_interface(label="mask")
    mask_registry.register("eager", object())
    _patch_validation_registry_environment(
        monkeypatch,
        attention_registry=attention_registry,
        mask_registry=mask_registry,
    )
    model = _build_model(
        family="mistral-7b-v0.1",
        query_heads=32,
        kv_heads=8,
        attention_cls=MistralAttentionLike,
    )
    intervention = create_mistral_7b_v0_1_intervention(
        model,
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (0,)},
        kernels_by_layer_head={(0, 0): torch.ones(512)},
        validation_probe=ValidationProbeRequest(layer_index=0, head_index=0),
    )
    intervention._module_to_layer_index[id(model.model.layers[0].self_attn)] = 0  # pyright: ignore[reportPrivateUsage]
    intervention._pending_probe_records[(0, 0)] = (  # pyright: ignore[reportPrivateUsage]
        _base_mask(64)[0, 0].to(torch.float32),
        _base_mask(64)[0, 0].to(torch.float32),
        torch.zeros((64, 64), dtype=torch.float32),
    )
    full_q = torch.zeros((1, 32, 64, 2), dtype=torch.bfloat16)
    full_k = torch.zeros((1, 8, 64, 2), dtype=torch.bfloat16)
    full_probs = torch.zeros((1, 32, 64, 64), dtype=torch.bfloat16)
    with pytest.raises((TypeError, ValueError), match=r"full_post_mask|shape|32"):
        intervention._capture_validation_from_attention_interface(  # pyright: ignore[reportPrivateUsage]
            module=model.model.layers[0].self_attn,
            query=full_q,
            key=full_k,
            attention_mask=_base_mask(64)[:, :1].to(torch.float32),
            scaling=0.5,
            attention_probabilities=full_probs,
        )


def test_validation_probe_requires_exactly_64_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    attention_registry = _make_pinned_general_interface(label="attention")
    mask_registry = _make_pinned_general_interface(label="mask")
    mask_registry.register("eager", object())
    _patch_validation_registry_environment(
        monkeypatch,
        attention_registry=attention_registry,
        mask_registry=mask_registry,
    )
    model = _build_model(family="mistral-7b-v0.1", attention_cls=MistralAttentionLike)
    intervention = create_mistral_7b_v0_1_intervention(
        model,
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (0,)},
        kernels_by_layer_head={(0, 0): torch.ones(512)},
        validation_probe=ValidationProbeRequest(layer_index=0, head_index=0),
    )
    with pytest.raises(RuntimeError, match="exact sequence length 64"):
        with intervention:
            model.forward(
                torch.zeros((1, 63, 64), dtype=torch.float32),
                _probe_position_embeddings(63, 2),
                _base_mask(63)[:, :1],
            )


def test_production_mode_retains_zero_validation_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_family_registry(monkeypatch)
    model = _build_model(family="llama-3.1-8b", attention_cls=LlamaAttentionLike)
    intervention = create_llama_3_1_8b_intervention(
        model,
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (0,)},
        kernels_by_layer_head={(0, 0): torch.ones(512)},
    )
    with intervention:
        out, _ = model.forward(
            torch.zeros((1, 64, 64), dtype=torch.float32),
            _probe_position_embeddings(64, 2),
            _base_mask(64)[:, :1],
        )
        assert out.dtype == torch.bfloat16
        assert intervention.mask_captures == []
    assert intervention.latest_validation_record() is None
    assert intervention.latest_validation_record() is None


@pytest.mark.parametrize("preseed_scope", [None, "global", "local"])
def test_validation_forward_error_restores_attention_and_mask_registries_exactly(
    monkeypatch: pytest.MonkeyPatch,
    preseed_scope: str | None,
) -> None:
    attention_registry = _make_pinned_general_interface(label="attention")
    mask_registry = _make_pinned_general_interface(label="mask")
    previous_attention = object()
    previous_mask = object()
    eager_mask = object()
    mask_registry.register("eager", eager_mask)
    _patch_validation_registry_environment(
        monkeypatch,
        attention_registry=attention_registry,
        mask_registry=mask_registry,
    )
    model = _build_model(family="mistral-7b-v0.1", attention_cls=MistralAttentionLike)
    intervention = create_mistral_7b_v0_1_intervention(
        model,
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (0,)},
        kernels_by_layer_head={(0, 0): torch.ones(512)},
        validation_probe=ValidationProbeRequest(layer_index=0, head_index=0),
    )
    original_impl = _model_attn_implementation(model)
    validation_key = _validation_key(intervention)
    _seed_validation_entry(
        attention_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_attention,
    )
    _seed_validation_entry(
        mask_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_mask,
    )
    attention_registry.events.clear()
    mask_registry.events.clear()

    with pytest.raises(RuntimeError, match="exact sequence length 64"):
        with intervention:
            assert _model_attn_implementation(model) == validation_key
            assert validation_key in attention_registry
            assert validation_key in mask_registry
            assert mask_registry.get(validation_key) is eager_mask
            model.forward(
                torch.zeros((1, 63, 64), dtype=torch.float32),
                _probe_position_embeddings(63, 2),
                _base_mask(63)[:, :1],
            )

    assert intervention.installed_hook_count == 0
    assert _model_attn_implementation(model) == original_impl
    _assert_validation_entry_state(
        attention_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_attention if preseed_scope is not None else None,
    )
    _assert_validation_entry_state(
        mask_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_mask if preseed_scope is not None else None,
    )


@pytest.mark.parametrize("preseed_scope", [None, "global", "local"])
def test_validation_mask_registry_install_failure_restores_all_state(
    monkeypatch: pytest.MonkeyPatch,
    preseed_scope: str | None,
) -> None:
    event_log: list[tuple[str, str, str]] = []
    attention_registry = _make_pinned_general_interface(label="attention", event_log=event_log)
    mask_registry = _make_failing_pinned_general_interface(
        label="mask",
        failure_message="mask install boom",
        event_log=event_log,
    )
    previous_attention = object()
    previous_mask = object()
    mask_registry.register("eager", object())
    _patch_validation_registry_environment(
        monkeypatch,
        attention_registry=attention_registry,
        mask_registry=mask_registry,
    )
    model = _build_model(family="mistral-7b-v0.1", attention_cls=MistralAttentionLike)
    original_impl = _model_attn_implementation(model)
    intervention = create_mistral_7b_v0_1_intervention(
        model,
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (0,)},
        kernels_by_layer_head={(0, 0): torch.ones(512)},
        validation_probe=ValidationProbeRequest(layer_index=0, head_index=0),
    )
    validation_key = _validation_key(intervention)
    _seed_validation_entry(
        attention_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_attention,
    )
    _seed_validation_entry(
        mask_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_mask,
    )
    attention_registry.events.clear()
    mask_registry.events.clear()
    event_log.clear()
    mask_registry.arm_validation_failure()

    with pytest.raises(RuntimeError, match="mask install boom"):
        intervention.__enter__()

    assert intervention.installed_hook_count == 0
    assert _model_attn_implementation(model) == original_impl
    _assert_validation_entry_state(
        attention_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_attention if preseed_scope is not None else None,
    )
    _assert_validation_entry_state(
        mask_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_mask if preseed_scope is not None else None,
    )
    assert event_log[:2] == [
        ("attention", "register_global", validation_key),
        ("mask", "register_global_fail", validation_key),
    ]


@pytest.mark.parametrize("mask_registry_value", [None, object()])
@pytest.mark.parametrize("preseed_scope", [None, "global", "local"])
def test_invalid_mask_registry_fails_before_config_write_and_restores_attention(
    monkeypatch: pytest.MonkeyPatch,
    preseed_scope: str | None,
    mask_registry_value: object | None,
) -> None:
    attention_registry = _make_pinned_general_interface(label="attention")
    global_mask_registry = _make_pinned_general_interface(label="global_mask")
    previous_attention = object()
    global_mask_registry.register("eager", object())
    monkeypatch.setattr(LlamaAttentionLike, "attention_interface", attention_registry)
    monkeypatch.setattr(MistralAttentionLike, "attention_interface", attention_registry)
    monkeypatch.setattr(OlmoAttentionLike, "attention_interface", attention_registry)
    _patch_family_registry(monkeypatch, fake_interface=attention_registry)
    _patch_mask_attention_registry(monkeypatch, mask_registry=global_mask_registry)
    _set_family_adapter_mask_attention_interface(mask_registry_value)
    model = _build_model(family="mistral-7b-v0.1", attention_cls=MistralAttentionLike)
    if not isinstance(model.config, _TestAttentionConfig):
        raise TypeError("expected test attention config")
    tracking_config_base = type(model.config)

    class _TrackingAttnImplementationConfig(tracking_config_base):
        writes: list[object]

        def __init__(self, original: _TestAttentionConfig) -> None:
            self.writes = []
            super().__init__(
                model_type=original.model_type,
                num_attention_heads=original.num_attention_heads,
                num_key_value_heads=original.num_key_value_heads,
                _attn_implementation=_config_private_attn_implementation(original),
            )
            self.writes.clear()

        def __setattr__(self, name: str, value: object) -> None:
            if name == "_attn_implementation":
                self.writes.append(value)
            object.__setattr__(self, name, value)

    tracking_config = _TrackingAttnImplementationConfig(model.config)
    model.config = tracking_config
    original_impl = _model_attn_implementation(model)
    intervention = create_mistral_7b_v0_1_intervention(
        model,
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (0,)},
        kernels_by_layer_head={(0, 0): torch.ones(512)},
        validation_probe=ValidationProbeRequest(layer_index=0, head_index=0),
    )
    validation_key = _validation_key(intervention)
    _seed_validation_entry(
        attention_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_attention,
    )
    attention_registry.events.clear()
    global_mask_registry.events.clear()

    with pytest.raises(RuntimeError, match="mask attention-interface registry"):
        intervention.__enter__()

    assert intervention.installed_hook_count == 0
    assert _model_attn_implementation(model) == original_impl
    assert tracking_config.writes == []
    _assert_validation_entry_state(
        attention_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_attention if preseed_scope is not None else None,
    )
    assert validation_key not in global_mask_registry
    assert global_mask_registry.get(validation_key) is None


@pytest.mark.parametrize("preseed_scope", [None, "global", "local"])
def test_validation_hook_partial_failure_restores_attention_mask_and_config_state(
    monkeypatch: pytest.MonkeyPatch,
    preseed_scope: str | None,
) -> None:
    attention_registry = _make_pinned_general_interface(label="attention")
    mask_registry = _make_pinned_general_interface(label="mask")
    previous_attention = object()
    previous_mask = object()
    mask_registry.register("eager", object())
    _patch_validation_registry_environment(
        monkeypatch,
        attention_registry=attention_registry,
        mask_registry=mask_registry,
    )
    model = _build_model(family="mistral-7b-v0.1", attention_cls=MistralAttentionLike)
    first_register = cast(
        ForwardPreHookRegistrar,
        model.model.layers[1].self_attn.register_forward_pre_hook,
    )
    call_count = {"value": 0}

    def flaky_register(*args: object, **kwargs: object) -> RemovableHandle:
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise RuntimeError("boom")
        return first_register(*args, **kwargs)

    monkeypatch.setattr(
        model.model.layers[1].self_attn, "register_forward_pre_hook", flaky_register
    )
    intervention = create_mistral_7b_v0_1_intervention(
        model,
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (0,)},
        kernels_by_layer_head={(0, 0): torch.ones(512)},
        validation_probe=ValidationProbeRequest(layer_index=0, head_index=0),
    )
    original_impl = _model_attn_implementation(model)
    validation_key = _validation_key(intervention)
    _seed_validation_entry(
        attention_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_attention,
    )
    _seed_validation_entry(
        mask_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_mask,
    )
    attention_registry.events.clear()
    mask_registry.events.clear()

    with pytest.raises(RuntimeError, match="boom"):
        intervention.__enter__()
    assert intervention.installed_hook_count == 0
    assert _model_attn_implementation(model) == original_impl
    _assert_validation_entry_state(
        attention_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_attention if preseed_scope is not None else None,
    )
    _assert_validation_entry_state(
        mask_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_mask if preseed_scope is not None else None,
    )


@pytest.mark.parametrize("preseed_scope", [None, "global", "local"])
def test_validation_hook_baseexception_registration_failure_restores_state(
    monkeypatch: pytest.MonkeyPatch,
    preseed_scope: str | None,
) -> None:
    attention_registry = _make_pinned_general_interface(label="attention")
    mask_registry = _make_pinned_general_interface(label="mask")
    previous_attention = object()
    previous_mask = object()
    mask_registry.register("eager", object())
    _patch_validation_registry_environment(
        monkeypatch,
        attention_registry=attention_registry,
        mask_registry=mask_registry,
    )
    model = _build_model(family="mistral-7b-v0.1", attention_cls=MistralAttentionLike)

    def flaky_register(*args: object, **kwargs: object) -> RemovableHandle:
        del args, kwargs
        raise SystemExit("boom")

    monkeypatch.setattr(
        model.model.layers[1].self_attn, "register_forward_pre_hook", flaky_register
    )
    intervention = create_mistral_7b_v0_1_intervention(
        model,
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (0,)},
        kernels_by_layer_head={(0, 0): torch.ones(512)},
        validation_probe=ValidationProbeRequest(layer_index=0, head_index=0),
    )
    original_impl = _model_attn_implementation(model)
    validation_key = _validation_key(intervention)
    _seed_validation_entry(
        attention_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_attention,
    )
    _seed_validation_entry(
        mask_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_mask,
    )
    attention_registry.events.clear()
    mask_registry.events.clear()

    with pytest.raises(SystemExit, match="boom"):
        intervention.__enter__()
    assert intervention.installed_hook_count == 0
    assert _model_attn_implementation(model) == original_impl
    _assert_validation_entry_state(
        attention_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_attention if preseed_scope is not None else None,
    )
    _assert_validation_entry_state(
        mask_registry,
        validation_key=validation_key,
        scope=preseed_scope,
        value=previous_mask if preseed_scope is not None else None,
    )


def test_revalidate_installation_context_fails_closed_on_shortened_stored_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_family_registry(monkeypatch)
    model = _build_model(family="mistral-7b-v0.1", attention_cls=MistralAttentionLike)
    intervention = create_mistral_7b_v0_1_intervention(
        model,
        expected_call_counts=_expected_counts(ADR_LAYER_COUNT),
        selected_heads_by_layer={0: (0,)},
        kernels_by_layer_head={(0, 0): torch.ones(512)},
    )
    intervention.replace_attention_modules(intervention.attention_modules[:-1])
    with pytest.raises(
        RuntimeError,
        match=r"Stored intervention attention module count no longer matches the validated model\.",
    ):
        intervention.__enter__()


def test_build_causal_toeplitz_correction_matches_numpy_reference() -> None:
    kernel = torch.linspace(-1.0, 1.0, 64)
    correction = build_causal_toeplitz_correction(
        kernel, sequence_length=64, device=torch.device("cpu")
    )
    expected = torch.tensor(
        causal_toeplitz_correction(np.asarray(kernel.detach().cpu(), dtype=np.float64), seq_len=64),
        dtype=torch.float32,
    )
    assert torch.allclose(correction, expected)
