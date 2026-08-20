# ruff: noqa: N812
from __future__ import annotations

import inspect
import math
import types
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, TypeAlias, TypeGuard, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

from si_rebuttal.controls import ExpectedAttentionCallCounts
from si_rebuttal.kernels import HeadIndex, build_causal_toeplitz_correction
from si_rebuttal.statistics import reconstruct_attention_probabilities

ADR_LAYER_COUNT = 32
ADR_QUERY_HEAD_COUNT = 32
ADR_LLAMA_MISTRAL_KV_HEADS = 8
ADR_LLAMA_MISTRAL_REPEAT = 4
ADR_OLMO_KV_HEADS = 32
ADR_OLMO_REPEAT = 1
VALIDATION_MAX_SEQUENCE_LENGTH = 64
Float64Array: TypeAlias = npt.NDArray[np.float64]


class _ConfigWithPrivateAttnImplementation(Protocol):
    _attn_implementation: str


class _ConfigWithPublicAttnImplementation(Protocol):
    attn_implementation: str


class _ModelConfigProtocol(Protocol):
    model_type: str
    num_attention_heads: int
    num_key_value_heads: int


class _ParameterLike(Protocol):
    device: torch.device


class _LayerProtocol(Protocol):
    @property
    def self_attn(self) -> torch.nn.Module: ...


class _LayerContainerProtocol(Protocol):
    @property
    def layers(self) -> Sequence[_LayerProtocol]: ...


class _ModelContainerProtocol(Protocol):
    @property
    def config(self) -> object: ...

    @property
    def model(self) -> _LayerContainerProtocol: ...

    def parameters(self) -> Iterable[torch.Tensor | _ParameterLike]: ...


@dataclass(frozen=True)
class FrozenAttentionSpec:
    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    repeat_factor: int


@dataclass(frozen=True)
class HookMaskCapture:
    layer_index: int
    head_indices: tuple[int, ...]
    pre_mask: torch.Tensor
    post_mask: torch.Tensor
    correction: torch.Tensor


@dataclass(frozen=True)
class AttentionMaskForwardLayout:
    arg_name: str
    rank: int
    axes: tuple[str, ...]


@dataclass(frozen=True)
class ValidationProbeRequest:
    layer_index: int
    head_index: int
    max_sequence_length: int = VALIDATION_MAX_SEQUENCE_LENGTH

    def __post_init__(self) -> None:
        if not 0 <= self.layer_index < ADR_LAYER_COUNT:
            raise ValueError(f"Probe layer_index must be in [0, 31], got {self.layer_index}.")
        if not 0 <= self.head_index < ADR_QUERY_HEAD_COUNT:
            raise ValueError(f"Probe head_index must be in [0, 31], got {self.head_index}.")
        if self.max_sequence_length != VALIDATION_MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "The ADR fixes the validation probe to the exact 64-token prefix; "
                f"got max_sequence_length={self.max_sequence_length}."
            )


@dataclass(frozen=True)
class ValidationProbeRecord:
    family_name: str
    layer_index: int
    head_index: int
    pre_mask: torch.Tensor
    post_mask: torch.Tensor
    correction: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    attention_probabilities: torch.Tensor
    full_q: torch.Tensor
    full_k: torch.Tensor
    full_post_mask: torch.Tensor
    scaling: float

    def manual_reconstruction(self) -> torch.Tensor:
        return reconstruct_attention_probabilities(self.q, self.k, self.post_mask)

    def runtime_faithful_attention_probabilities(self) -> torch.Tensor:
        if self.full_q.ndim != 4:
            raise ValueError("full_q must have shape [batch, query_heads, seq_len, head_dim].")
        if self.full_k.ndim != 4:
            raise ValueError("full_k must have shape [batch, kv_heads, seq_len, head_dim].")
        if self.full_post_mask.ndim != 4:
            raise ValueError(
                "full_post_mask must have shape [batch, query_heads, seq_len, seq_len]."
            )
        if self.full_q.dtype != torch.bfloat16:
            raise ValueError(f"full_q must preserve eager BF16 values, got {self.full_q.dtype}.")
        if self.full_k.dtype != torch.bfloat16:
            raise ValueError(f"full_k must preserve eager BF16 values, got {self.full_k.dtype}.")
        if self.full_post_mask.dtype != torch.float32:
            raise ValueError(
                "full_post_mask must preserve the eager FP32 mask, "
                f"got {self.full_post_mask.dtype}."
            )
        if self.full_q.device != self.full_k.device:
            raise ValueError("full_q and full_k must share a device.")
        if self.full_post_mask.device != self.full_q.device:
            raise ValueError("full_post_mask must share the query device.")
        if (
            self.full_q.shape[0] != 1
            or self.full_k.shape[0] != 1
            or self.full_post_mask.shape[0] != 1
        ):
            raise ValueError("Validation capture supports exactly one bounded 64-token batch.")
        if self.full_q.shape[-2] != VALIDATION_MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "Validation capture requires exact query sequence length "
                f"{VALIDATION_MAX_SEQUENCE_LENGTH}, found {self.full_q.shape[-2]}."
            )
        if self.full_k.shape[-2] != VALIDATION_MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "Validation capture requires exact key sequence length "
                f"{VALIDATION_MAX_SEQUENCE_LENGTH}, found {self.full_k.shape[-2]}."
            )
        if self.full_q.shape[0] != self.full_k.shape[0]:
            raise ValueError("full_q and full_k batch axes must match.")
        if self.full_q.shape[2] != self.full_k.shape[2]:
            raise ValueError("full_q and full_k sequence axes must match.")
        if self.full_q.shape[3] != self.full_k.shape[3]:
            raise ValueError("full_q and full_k head_dim axes must match.")
        if self.full_k.shape[1] <= 0 or self.full_q.shape[1] <= 0:
            raise ValueError("Query and KV head counts must be positive.")
        if self.full_q.shape[1] % self.full_k.shape[1] != 0:
            raise ValueError("full_k head count must divide full_q head count for GQA repeat.")
        if self.full_post_mask.shape[1] != self.full_q.shape[1]:
            raise ValueError(
                "full_post_mask head axis must exactly match query heads; "
                "singleton masks are invalid."
            )
        if self.full_post_mask.shape[-2:] != (
            VALIDATION_MAX_SEQUENCE_LENGTH,
            VALIDATION_MAX_SEQUENCE_LENGTH,
        ):
            raise ValueError(
                "full_post_mask must preserve the exact [1, heads, 64, 64] eager mask shape."
            )
        if not torch.isfinite(self.full_q).all() or not torch.isfinite(self.full_k).all():
            raise ValueError("full_q or full_k contains non-finite values.")
        _require_valid_additive_mask(self.full_post_mask, name="full_post_mask")
        if not math.isfinite(self.scaling):
            raise ValueError("scaling must be finite.")
        repeated_key = repeat_kv_after_gqa(self.full_k, num_query_heads=int(self.full_q.shape[1]))
        logits = torch.matmul(self.full_q, repeated_key.transpose(-1, -2))
        logits = logits * self.scaling
        logits = logits + self.full_post_mask
        return F.softmax(logits, dim=-1, dtype=torch.float32).to(dtype=self.full_q.dtype)


@dataclass(frozen=True)
class FamilyAdapter:
    family_name: str
    model_class: type[object]
    config_class: type[object]
    config_model_type: str
    attention_class: type[torch.nn.Module]
    eager_attention_forward: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]
    attention_interface: object
    kv_head_count: int
    gqa_repeat_factor: int
    attention_mask_layout: AttentionMaskForwardLayout
    forward_signature: inspect.Signature
    mask_attention_interface: object | None = None


@dataclass(frozen=True)
class ForwardParameterContract:
    name: str
    kind: IntEnum
    has_default: bool
    default_is_none: bool


class _AttentionRegistryProtocol(Protocol):
    def __contains__(self, key: object, /) -> bool: ...

    def get(self, key: str, default: object | None = None, /) -> object | None: ...

    def pop(self, key: str, default: object | None = None, /) -> object | None: ...


class _RegisteringAttentionRegistryProtocol(_AttentionRegistryProtocol, Protocol):
    def register(self, key: str, value: object, /) -> None: ...


class _MutableAttentionRegistryProtocol(_AttentionRegistryProtocol, Protocol):
    def __setitem__(self, key: str, value: object, /) -> None: ...


ValidationAttentionRegistryHandle = (
    _RegisteringAttentionRegistryProtocol | _MutableAttentionRegistryProtocol
)


@dataclass(frozen=True)
class ValidationRegistrySnapshot:
    had_key: bool
    previous: object | None
    used_internal_mappings: bool
    local_had_key: bool
    local_previous: object | None
    global_had_key: bool
    global_previous: object | None


def _supports_registry_register(
    candidate: ValidationAttentionRegistryHandle,
) -> TypeGuard[_RegisteringAttentionRegistryProtocol]:
    return callable(getattr(candidate, "register", None))


def _supports_registry_setitem(
    candidate: ValidationAttentionRegistryHandle,
) -> TypeGuard[_MutableAttentionRegistryProtocol]:
    return callable(getattr(candidate, "__setitem__", None))


@dataclass(frozen=True)
class ValidationAttentionRegistry:
    registry: ValidationAttentionRegistryHandle

    def contains(self, key: str) -> bool:
        return bool(key in self.registry)

    def get(self, key: str) -> object | None:
        return self.registry.get(key)

    def snapshot(self, key: str) -> ValidationRegistrySnapshot:
        local_mapping = self._mapping_attr("_local_mapping")
        global_mapping = self._mapping_attr("_global_mapping")
        if local_mapping is not None or global_mapping is not None:
            local_had_key = local_mapping is not None and key in local_mapping
            global_had_key = global_mapping is not None and key in global_mapping
            previous = None
            if local_had_key and local_mapping is not None:
                previous = local_mapping[key]
            elif global_had_key and global_mapping is not None:
                previous = global_mapping[key]
            return ValidationRegistrySnapshot(
                had_key=local_had_key or global_had_key,
                previous=previous,
                used_internal_mappings=True,
                local_had_key=local_had_key,
                local_previous=(
                    local_mapping[key] if local_had_key and local_mapping is not None else None
                ),
                global_had_key=global_had_key,
                global_previous=(
                    global_mapping[key] if global_had_key and global_mapping is not None else None
                ),
            )

        had_key = self.contains(key)
        previous = self.get(key) if had_key else None
        return ValidationRegistrySnapshot(
            had_key=had_key,
            previous=previous,
            used_internal_mappings=False,
            local_had_key=False,
            local_previous=None,
            global_had_key=False,
            global_previous=None,
        )

    def install(self, key: str, value: object) -> None:
        local_mapping = self._mapping_attr("_local_mapping")
        if _supports_registry_register(self.registry):
            if local_mapping is not None:
                local_mapping.pop(key, None)
            self.registry.register(key, value)
            return
        if not _supports_registry_setitem(self.registry):
            raise TypeError("Attention registry is missing mutable mapping semantics.")
        self.registry[key] = value

    def restore(self, key: str, snapshot: ValidationRegistrySnapshot) -> None:
        if snapshot.used_internal_mappings:
            local_mapping = self._mapping_attr("_local_mapping")
            global_mapping = self._mapping_attr("_global_mapping")
            if local_mapping is None and global_mapping is None:
                raise TypeError("Registry lost its internal mapping state during restoration.")
            if local_mapping is not None:
                if snapshot.local_had_key:
                    local_mapping[key] = snapshot.local_previous
                else:
                    local_mapping.pop(key, None)
            elif snapshot.local_had_key:
                raise TypeError("Registry is missing _local_mapping during restoration.")
            if global_mapping is not None:
                if snapshot.global_had_key:
                    global_mapping[key] = snapshot.global_previous
                else:
                    global_mapping.pop(key, None)
            elif snapshot.global_had_key:
                raise TypeError("Registry is missing _global_mapping during restoration.")
            return

        if snapshot.had_key:
            self.install(key, snapshot.previous)
            return
        self.registry.pop(key, None)

    def _mapping_attr(self, name: str) -> MutableMapping[str, object] | None:
        mapping = getattr(self.registry, name, None)
        if isinstance(mapping, MutableMapping):
            return cast(MutableMapping[str, object], mapping)
        return None


def _is_attention_registry_protocol(candidate: object) -> TypeGuard[_AttentionRegistryProtocol]:
    if not hasattr(candidate, "__contains__"):
        return False
    getter = getattr(candidate, "get", None)
    pop = getattr(candidate, "pop", None)
    return callable(getter) and callable(pop)


def _coerce_validation_attention_registry(
    candidate: object,
) -> ValidationAttentionRegistry | None:
    if candidate is None:
        return None
    if not _is_attention_registry_protocol(candidate):
        return None
    register = getattr(candidate, "register", None)
    setitem = getattr(candidate, "__setitem__", None)
    if callable(register):
        return ValidationAttentionRegistry(cast(_RegisteringAttentionRegistryProtocol, candidate))
    if not callable(setitem):
        return None
    return ValidationAttentionRegistry(cast(_MutableAttentionRegistryProtocol, candidate))


def _raise_grouped_exceptions(message: str, errors: Sequence[BaseException]) -> None:
    if not errors:
        return
    if len(errors) == 1:
        raise errors[0]
    if all(isinstance(error, Exception) for error in errors):
        raise ExceptionGroup(message, cast(Sequence[Exception], errors))
    raise BaseExceptionGroup(message, list(errors))


@dataclass(frozen=True)
class SupportedModelFamilySpec:
    family_name: str
    model_class_name: str
    config_class_name: str
    config_model_type: str
    attention_class_name: str
    layer_count: int
    query_head_count: int
    kv_head_count: int
    gqa_repeat_factor: int
    attention_mask_layout: AttentionMaskForwardLayout
    device: torch.device
    attn_implementation: str
    attention_module_count: int
    attention_forward_signature: inspect.Signature
    model: object
    family_adapter: FamilyAdapter
    attention_modules: tuple[torch.nn.Module, ...]


_INTERNAL_INTERVENTION_CONSTRUCTOR_TOKEN = object()


def _signature_contract(signature: inspect.Signature) -> tuple[ForwardParameterContract, ...]:
    return tuple(
        ForwardParameterContract(
            name=parameter.name,
            kind=parameter.kind,
            has_default=parameter.default is not inspect.Parameter.empty,
            default_is_none=parameter.default is None,
        )
        for parameter in signature.parameters.values()
    )


def _contract_parameter(
    name: str,
    kind: IntEnum,
    *,
    has_default: bool,
    default_is_none: bool,
) -> ForwardParameterContract:
    return ForwardParameterContract(
        name=name,
        kind=kind,
        has_default=has_default,
        default_is_none=default_is_none,
    )


_LLAMA_FORWARD_CONTRACT = (
    _contract_parameter(
        "self", inspect.Parameter.POSITIONAL_OR_KEYWORD, has_default=False, default_is_none=False
    ),
    _contract_parameter(
        "hidden_states",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=False,
        default_is_none=False,
    ),
    _contract_parameter(
        "position_embeddings",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=True,
        default_is_none=True,
    ),
    _contract_parameter(
        "attention_mask",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=True,
        default_is_none=True,
    ),
    _contract_parameter(
        "past_key_values",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=True,
        default_is_none=True,
    ),
    _contract_parameter(
        "cache_position",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=True,
        default_is_none=True,
    ),
    _contract_parameter(
        "kwargs", inspect.Parameter.VAR_KEYWORD, has_default=False, default_is_none=False
    ),
)

_MISTRAL_FORWARD_CONTRACT = (
    _contract_parameter(
        "self", inspect.Parameter.POSITIONAL_OR_KEYWORD, has_default=False, default_is_none=False
    ),
    _contract_parameter(
        "hidden_states",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=False,
        default_is_none=False,
    ),
    _contract_parameter(
        "position_embeddings",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=False,
        default_is_none=False,
    ),
    _contract_parameter(
        "attention_mask",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=False,
        default_is_none=False,
    ),
    _contract_parameter(
        "past_key_values",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=True,
        default_is_none=True,
    ),
    _contract_parameter(
        "cache_position",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=True,
        default_is_none=True,
    ),
    _contract_parameter(
        "kwargs", inspect.Parameter.VAR_KEYWORD, has_default=False, default_is_none=False
    ),
)

_OLMO2_NO_CACHE_POSITION_FORWARD_CONTRACT = (
    _contract_parameter(
        "self", inspect.Parameter.POSITIONAL_OR_KEYWORD, has_default=False, default_is_none=False
    ),
    _contract_parameter(
        "hidden_states",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=False,
        default_is_none=False,
    ),
    _contract_parameter(
        "position_embeddings",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=False,
        default_is_none=False,
    ),
    _contract_parameter(
        "attention_mask",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=False,
        default_is_none=False,
    ),
    _contract_parameter(
        "past_key_values",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=True,
        default_is_none=True,
    ),
    _contract_parameter(
        "kwargs", inspect.Parameter.VAR_KEYWORD, has_default=False, default_is_none=False
    ),
)

_OLMO2_WITH_CACHE_POSITION_FORWARD_CONTRACT = (
    _contract_parameter(
        "self", inspect.Parameter.POSITIONAL_OR_KEYWORD, has_default=False, default_is_none=False
    ),
    _contract_parameter(
        "hidden_states",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=False,
        default_is_none=False,
    ),
    _contract_parameter(
        "position_embeddings",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=False,
        default_is_none=False,
    ),
    _contract_parameter(
        "attention_mask",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=False,
        default_is_none=False,
    ),
    _contract_parameter(
        "past_key_values",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=True,
        default_is_none=True,
    ),
    _contract_parameter(
        "cache_position",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        has_default=True,
        default_is_none=True,
    ),
    _contract_parameter(
        "kwargs", inspect.Parameter.VAR_KEYWORD, has_default=False, default_is_none=False
    ),
)


FROZEN_MODEL_SPECS: dict[str, FrozenAttentionSpec] = {
    "llama-3.1-8b": FrozenAttentionSpec(
        num_layers=32, num_query_heads=32, num_kv_heads=8, repeat_factor=4
    ),
    "mistral-7b-v0.1": FrozenAttentionSpec(
        num_layers=32, num_query_heads=32, num_kv_heads=8, repeat_factor=4
    ),
    "olmo-2-7b": FrozenAttentionSpec(
        num_layers=32, num_query_heads=32, num_kv_heads=32, repeat_factor=1
    ),
}


def frozen_attention_spec(model_name: str) -> FrozenAttentionSpec:
    try:
        return FROZEN_MODEL_SPECS[model_name]
    except KeyError as exc:
        raise ValueError(f"unsupported model_name={model_name!r}") from exc


def _has_private_attn_implementation(
    config: object,
) -> TypeGuard[_ConfigWithPrivateAttnImplementation]:
    return isinstance(getattr(config, "_attn_implementation", None), str)


def _has_public_attn_implementation(
    config: object,
) -> TypeGuard[_ConfigWithPublicAttnImplementation]:
    return isinstance(getattr(config, "attn_implementation", None), str)


def _supports_model_config(config: object) -> TypeGuard[_ModelConfigProtocol]:
    return (
        isinstance(getattr(config, "model_type", None), str)
        and isinstance(getattr(config, "num_attention_heads", None), int)
        and isinstance(getattr(config, "num_key_value_heads", None), int)
    )


def _read_attn_implementation(model: _ModelContainerProtocol) -> str:
    config = model.config
    attr_name = _attn_implementation_attr_name(model)
    value = getattr(config, attr_name, None)
    if not isinstance(value, str):
        raise TypeError("Model config must expose an attention implementation marker.")
    return value


def _attn_implementation_attr_name(model: _ModelContainerProtocol) -> str:
    config = model.config
    if _has_private_attn_implementation(config):
        return "_attn_implementation"
    if _has_public_attn_implementation(config):
        return "attn_implementation"
    raise TypeError("Model config must expose an attention implementation marker.")


def attn_implementation_attr_name(model: _ModelContainerProtocol) -> str:
    return _attn_implementation_attr_name(model)


def _attention_modules(model: _ModelContainerProtocol) -> tuple[torch.nn.Module, ...]:
    layers = model.model.layers
    modules: list[torch.nn.Module] = []
    for layer in layers:
        attn = layer.self_attn
        modules.append(attn)
    return tuple(modules)


def _infer_device(
    model: _ModelContainerProtocol, modules: Sequence[torch.nn.Module]
) -> torch.device:
    for source in (model, *modules):
        for parameter in source.parameters():
            return parameter.device
        device = getattr(source, "device", None)
        if isinstance(device, torch.device):
            return device
    return torch.device("cpu")


def _validate_llama_forward_signature(attention_class: type[torch.nn.Module]) -> inspect.Signature:
    signature = inspect.signature(attention_class.forward)
    actual_contract = _signature_contract(signature)
    if actual_contract != _LLAMA_FORWARD_CONTRACT:
        raise TypeError(
            "llama-3.1-8b attention forward signature drifted: "
            f"expected {_LLAMA_FORWARD_CONTRACT}, got {actual_contract}."
        )
    return signature


def _validate_mistral_forward_signature(
    attention_class: type[torch.nn.Module],
) -> inspect.Signature:
    signature = inspect.signature(attention_class.forward)
    actual_contract = _signature_contract(signature)
    if actual_contract != _MISTRAL_FORWARD_CONTRACT:
        raise TypeError(
            "mistral-7b-v0.1 attention forward signature drifted: "
            f"expected {_MISTRAL_FORWARD_CONTRACT}, got {actual_contract}."
        )
    return signature


def _validate_olmo2_forward_signature(attention_class: type[torch.nn.Module]) -> inspect.Signature:
    signature = inspect.signature(attention_class.forward)
    actual_contract = _signature_contract(signature)
    if actual_contract not in (
        _OLMO2_NO_CACHE_POSITION_FORWARD_CONTRACT,
        _OLMO2_WITH_CACHE_POSITION_FORWARD_CONTRACT,
    ):
        expected_contracts = (
            _OLMO2_NO_CACHE_POSITION_FORWARD_CONTRACT,
            _OLMO2_WITH_CACHE_POSITION_FORWARD_CONTRACT,
        )
        raise TypeError(
            "olmo-2-7b attention forward signature drifted: expected one of "
            f"{expected_contracts}, "
            f"got {actual_contract}."
        )
    return signature


def _build_family_registry() -> dict[str, FamilyAdapter]:
    try:
        from transformers import masking_utils, modeling_utils
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import (
            LlamaAttention,
            LlamaForCausalLM,
        )
        from transformers.models.llama.modeling_llama import (
            eager_attention_forward as llama_eager_attention_forward,
        )
        from transformers.models.mistral.configuration_mistral import MistralConfig
        from transformers.models.mistral.modeling_mistral import (
            MistralAttention,
            MistralForCausalLM,
        )
        from transformers.models.mistral.modeling_mistral import (
            eager_attention_forward as mistral_eager_attention_forward,
        )
        from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
        from transformers.models.olmo2.modeling_olmo2 import (
            Olmo2Attention,
            Olmo2ForCausalLM,
        )
        from transformers.models.olmo2.modeling_olmo2 import (
            eager_attention_forward as olmo2_eager_attention_forward,
        )
    except ImportError:
        return {}

    layout = AttentionMaskForwardLayout(
        arg_name="attention_mask",
        rank=4,
        axes=("batch", "head", "query_position", "key_position"),
    )
    return {
        "llama-3.1-8b": FamilyAdapter(
            family_name="llama-3.1-8b",
            model_class=LlamaForCausalLM,
            config_class=LlamaConfig,
            config_model_type="llama",
            attention_class=LlamaAttention,
            eager_attention_forward=llama_eager_attention_forward,
            attention_interface=modeling_utils.ALL_ATTENTION_FUNCTIONS,
            mask_attention_interface=masking_utils.ALL_MASK_ATTENTION_FUNCTIONS,
            kv_head_count=ADR_LLAMA_MISTRAL_KV_HEADS,
            gqa_repeat_factor=ADR_LLAMA_MISTRAL_REPEAT,
            attention_mask_layout=layout,
            forward_signature=_validate_llama_forward_signature(LlamaAttention),
        ),
        "mistral-7b-v0.1": FamilyAdapter(
            family_name="mistral-7b-v0.1",
            model_class=MistralForCausalLM,
            config_class=MistralConfig,
            config_model_type="mistral",
            attention_class=MistralAttention,
            eager_attention_forward=mistral_eager_attention_forward,
            attention_interface=modeling_utils.ALL_ATTENTION_FUNCTIONS,
            mask_attention_interface=masking_utils.ALL_MASK_ATTENTION_FUNCTIONS,
            kv_head_count=ADR_LLAMA_MISTRAL_KV_HEADS,
            gqa_repeat_factor=ADR_LLAMA_MISTRAL_REPEAT,
            attention_mask_layout=layout,
            forward_signature=_validate_mistral_forward_signature(MistralAttention),
        ),
        "olmo-2-7b": FamilyAdapter(
            family_name="olmo-2-7b",
            model_class=Olmo2ForCausalLM,
            config_class=Olmo2Config,
            config_model_type="olmo2",
            attention_class=Olmo2Attention,
            eager_attention_forward=olmo2_eager_attention_forward,
            attention_interface=modeling_utils.ALL_ATTENTION_FUNCTIONS,
            mask_attention_interface=masking_utils.ALL_MASK_ATTENTION_FUNCTIONS,
            kv_head_count=ADR_OLMO_KV_HEADS,
            gqa_repeat_factor=ADR_OLMO_REPEAT,
            attention_mask_layout=layout,
            forward_signature=_validate_olmo2_forward_signature(Olmo2Attention),
        ),
    }


_FAMILY_REGISTRY = _build_family_registry()


def _derive_supported_model_family_spec(
    model: _ModelContainerProtocol, family_name: str
) -> SupportedModelFamilySpec:
    adapter = _FAMILY_REGISTRY.get(family_name)
    if adapter is None:
        raise ValueError(f"Unsupported family {family_name!r}.")
    if not isinstance(model, adapter.model_class):
        raise TypeError(
            f"{family_name} requires model class {adapter.model_class.__name__}, "
            f"got {model.__class__.__name__}."
        )
    config = model.config
    if not isinstance(config, adapter.config_class):
        raise TypeError(
            f"{family_name} requires config class {adapter.config_class.__name__}, "
            f"got {config.__class__.__name__}."
        )
    if not _supports_model_config(config):
        raise TypeError(
            f"{family_name} config must expose typed model_type and attention-head counts."
        )
    if config.model_type != adapter.config_model_type:
        raise TypeError(
            f"{family_name} requires config.model_type={adapter.config_model_type!r}, "
            f"got {config.model_type!r}."
        )
    attn_implementation = _read_attn_implementation(model)
    if attn_implementation != "eager":
        raise TypeError(
            f"{family_name} requires attn_implementation='eager', got {attn_implementation!r}."
        )
    modules = _attention_modules(model)
    if len(modules) != ADR_LAYER_COUNT:
        raise TypeError(f"{family_name} requires exactly 32 attention modules, got {len(modules)}.")
    for index, module in enumerate(modules):
        if not isinstance(module, adapter.attention_class):
            raise TypeError(
                f"{family_name} layer {index} requires attention class "
                f"{adapter.attention_class.__name__}, "
                f"got {module.__class__.__name__}."
            )
    query_head_count = config.num_attention_heads
    kv_head_count = config.num_key_value_heads
    if query_head_count != ADR_QUERY_HEAD_COUNT:
        raise TypeError(f"{family_name} requires 32 query heads, got {query_head_count}.")
    if kv_head_count != adapter.kv_head_count:
        raise TypeError(
            f"{family_name} requires {adapter.kv_head_count} KV heads, got {kv_head_count}."
        )
    if query_head_count // max(kv_head_count, 1) != adapter.gqa_repeat_factor:
        raise TypeError(
            f"{family_name} requires GQA repeat {adapter.gqa_repeat_factor}, got "
            f"{query_head_count // max(kv_head_count, 1)}."
        )
    device = _infer_device(model, modules)
    return SupportedModelFamilySpec(
        family_name=family_name,
        model_class_name=adapter.model_class.__name__,
        config_class_name=adapter.config_class.__name__,
        config_model_type=adapter.config_model_type,
        attention_class_name=adapter.attention_class.__name__,
        layer_count=len(modules),
        query_head_count=query_head_count,
        kv_head_count=kv_head_count,
        gqa_repeat_factor=adapter.gqa_repeat_factor,
        attention_mask_layout=adapter.attention_mask_layout,
        device=device,
        attn_implementation=attn_implementation,
        attention_module_count=len(modules),
        attention_forward_signature=adapter.forward_signature,
        model=model,
        family_adapter=adapter,
        attention_modules=modules,
    )


def derive_supported_model_family_spec_for_capture(
    model: _ModelContainerProtocol, family_name: str
) -> SupportedModelFamilySpec:
    return _derive_supported_model_family_spec(model, family_name)


def derive_supported_model_family_spec(model: object, family_name: str) -> SupportedModelFamilySpec:
    del model, family_name
    raise RuntimeError(
        "derive_supported_model_family_spec is disabled for production use. "
        "Use the family-specific intervention factory on a real model instead."
    )


def _normalize_kernels(
    *,
    validated_spec: SupportedModelFamilySpec,
    selected_heads_by_layer: Mapping[int, Sequence[int]],
    kernels_by_layer_head: Mapping[tuple[int, int], torch.Tensor | Sequence[float]],
) -> dict[int, dict[int, Float64Array]]:
    kernels_by_layer: dict[int, dict[int, Float64Array]] = {}
    for layer_index, heads in selected_heads_by_layer.items():
        normalized_heads = tuple(int(head) for head in heads)
        if not normalized_heads:
            raise ValueError(f"Selected-head layer {layer_index} is empty.")
        if len(set(normalized_heads)) != len(normalized_heads):
            raise ValueError(f"Layer {layer_index} contains duplicate selected heads.")
        if not 0 <= int(layer_index) < validated_spec.layer_count:
            raise ValueError(f"Layer index out of range: {layer_index}.")
        for head_index in normalized_heads:
            if not 0 <= head_index < validated_spec.query_head_count:
                raise ValueError(f"Head index out of range: {head_index}.")
            key = (int(layer_index), head_index)
            if key not in kernels_by_layer_head:
                raise ValueError(f"Missing kernel for selected head {key}.")
            kernels_by_layer.setdefault(int(layer_index), {})[head_index] = np.asarray(
                torch.as_tensor(kernels_by_layer_head[key], dtype=torch.float32, device="cpu"),
                dtype=np.float64,
            )
    return kernels_by_layer


def _create_family_intervention(
    model: _ModelContainerProtocol,
    *,
    family_name: str,
    expected_call_counts: ExpectedAttentionCallCounts | None,
    selected_heads_by_layer: Mapping[int, Sequence[int]],
    kernels_by_layer_head: Mapping[tuple[int, int], torch.Tensor | Sequence[float]],
    validation_probe: ValidationProbeRequest | None,
) -> EagerAttentionIntervention:
    if expected_call_counts is None:
        raise TypeError("expected_call_counts is mandatory.")
    spec = _derive_supported_model_family_spec(model, family_name)
    per_layer_counts = expected_call_counts.require_complete(layer_count=spec.layer_count)
    kernels_by_layer = _normalize_kernels(
        validated_spec=spec,
        selected_heads_by_layer=selected_heads_by_layer,
        kernels_by_layer_head=kernels_by_layer_head,
    )
    if validation_probe is not None:
        selected = selected_heads_by_layer.get(validation_probe.layer_index, ())
        if validation_probe.head_index not in tuple(int(head) for head in selected):
            raise ValueError("Validation probe must target a configured selected head.")
    return EagerAttentionIntervention(
        _construction_token=_INTERNAL_INTERVENTION_CONSTRUCTOR_TOKEN,
        model=spec.model,
        family_name=family_name,
        family_adapter=spec.family_adapter,
        attention_modules=spec.attention_modules,
        kernels_by_layer=kernels_by_layer,
        expected_num_layers=spec.layer_count,
        expected_num_query_heads=spec.query_head_count,
        expected_num_kv_heads=spec.kv_head_count,
        expected_repeat_factor=spec.gqa_repeat_factor,
        expected_calls_per_layer=per_layer_counts,
        expected_device=spec.device,
        validated_spec=spec,
        validation_probe=validation_probe,
        expected_total_attention_calls=expected_call_counts.total_attention_calls,
    )


def create_llama_3_1_8b_intervention(
    model: _ModelContainerProtocol,
    *,
    expected_call_counts: ExpectedAttentionCallCounts | None,
    selected_heads_by_layer: Mapping[int, Sequence[int]],
    kernels_by_layer_head: Mapping[tuple[int, int], torch.Tensor | Sequence[float]],
    validation_probe: ValidationProbeRequest | None = None,
) -> EagerAttentionIntervention:
    return _create_family_intervention(
        model,
        family_name="llama-3.1-8b",
        expected_call_counts=expected_call_counts,
        selected_heads_by_layer=selected_heads_by_layer,
        kernels_by_layer_head=kernels_by_layer_head,
        validation_probe=validation_probe,
    )


def create_mistral_7b_v0_1_intervention(
    model: _ModelContainerProtocol,
    *,
    expected_call_counts: ExpectedAttentionCallCounts | None,
    selected_heads_by_layer: Mapping[int, Sequence[int]],
    kernels_by_layer_head: Mapping[tuple[int, int], torch.Tensor | Sequence[float]],
    validation_probe: ValidationProbeRequest | None = None,
) -> EagerAttentionIntervention:
    return _create_family_intervention(
        model,
        family_name="mistral-7b-v0.1",
        expected_call_counts=expected_call_counts,
        selected_heads_by_layer=selected_heads_by_layer,
        kernels_by_layer_head=kernels_by_layer_head,
        validation_probe=validation_probe,
    )


def create_olmo_2_7b_intervention(
    model: _ModelContainerProtocol,
    *,
    expected_call_counts: ExpectedAttentionCallCounts | None,
    selected_heads_by_layer: Mapping[int, Sequence[int]],
    kernels_by_layer_head: Mapping[tuple[int, int], torch.Tensor | Sequence[float]],
    validation_probe: ValidationProbeRequest | None = None,
) -> EagerAttentionIntervention:
    return _create_family_intervention(
        model,
        family_name="olmo-2-7b",
        expected_call_counts=expected_call_counts,
        selected_heads_by_layer=selected_heads_by_layer,
        kernels_by_layer_head=kernels_by_layer_head,
        validation_probe=validation_probe,
    )


def repeat_kv_after_gqa(keys: torch.Tensor, *, num_query_heads: int) -> torch.Tensor:
    if keys.ndim != 4:
        raise ValueError("keys must have shape [batch, kv_heads, seq_len, head_dim].")
    if not torch.isfinite(keys).all():
        raise ValueError("keys contain non-finite values.")
    kv_heads = keys.shape[1]
    if kv_heads <= 0 or num_query_heads <= 0:
        raise ValueError("head counts must be positive.")
    if num_query_heads % kv_heads != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} is not divisible by kv_heads={kv_heads}."
        )
    return keys.repeat_interleave(num_query_heads // kv_heads, dim=1)


def _is_valid_additive_mask(mask: torch.Tensor) -> bool:
    return not (torch.isnan(mask).any() or torch.isposinf(mask).any())


def _require_valid_additive_mask(mask: torch.Tensor, *, name: str) -> None:
    if not _is_valid_additive_mask(mask):
        raise ValueError(f"{name} must contain only finite values or -inf.")


def manual_attention_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    additive_mask: torch.Tensor | None = None,
    num_query_heads: int | None = None,
) -> torch.Tensor:
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, query_heads, seq_len, head_dim].")
    if key.ndim != 4:
        raise ValueError("key must have shape [batch, kv_or_query_heads, seq_len, head_dim].")
    if (
        query.shape[0] != key.shape[0]
        or query.shape[2] != key.shape[2]
        or query.shape[3] != key.shape[3]
    ):
        raise ValueError("query and key batch/sequence/head_dim must match.")
    if not torch.isfinite(query).all() or not torch.isfinite(key).all():
        raise ValueError("query or key contains non-finite values.")
    expanded_key = (
        key
        if key.shape[1] == query.shape[1]
        else repeat_kv_after_gqa(
            key,
            num_query_heads=num_query_heads or int(query.shape[1]),
        )
    )
    if expanded_key.shape[1] != query.shape[1]:
        raise ValueError("expanded key head count does not match query head count.")
    logits = torch.matmul(query.to(torch.float32), expanded_key.to(torch.float32).transpose(-1, -2))
    logits = logits * (1.0 / math.sqrt(query.shape[-1]))
    if additive_mask is not None:
        if additive_mask.ndim != 4:
            raise ValueError("additive_mask must have shape [batch, heads_or_1, seq_len, seq_len].")
        if (
            additive_mask.shape[0] != logits.shape[0]
            or additive_mask.shape[-2:] != logits.shape[-2:]
        ):
            raise ValueError("additive_mask shape does not match attention logits.")
        _require_valid_additive_mask(additive_mask, name="additive_mask")
        if additive_mask.shape[1] == 1:
            additive_mask = additive_mask.expand(
                logits.shape[0], logits.shape[1], logits.shape[2], logits.shape[3]
            )
        elif additive_mask.shape[1] != logits.shape[1]:
            raise ValueError("additive_mask head axis must be 1 or equal to the query head count.")
        logits = logits + additive_mask.to(dtype=torch.float32, device=logits.device)
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("attention logits contain invalid values.")
    return logits


def manual_attention_probs(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    additive_mask: torch.Tensor | None = None,
    num_query_heads: int | None = None,
) -> torch.Tensor:
    probs = torch.softmax(
        manual_attention_logits(
            query, key, additive_mask=additive_mask, num_query_heads=num_query_heads
        ),
        dim=-1,
    )
    if not torch.isfinite(probs).all():
        raise ValueError("attention probabilities contain non-finite values.")
    return probs


def mean_next_token_nll(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim not in (2, 3):
        raise ValueError("logits must have shape [seq_len, vocab] or [batch, seq_len, vocab].")
    if token_ids.ndim not in (1, 2):
        raise ValueError("token_ids must have shape [seq_len] or [batch, seq_len].")
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)
    if logits.shape[0] != token_ids.shape[0] or logits.shape[1] != token_ids.shape[1]:
        raise ValueError("logits and token_ids must match on batch and sequence length.")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain non-finite values.")
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_targets = token_ids[:, 1:].contiguous()
    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_targets.reshape(-1),
        reduction="none",
    ).reshape(shifted_targets.shape)
    per_sequence = losses.mean(dim=1)
    if not torch.isfinite(per_sequence).all():
        raise ValueError("next-token NLL contains non-finite values.")
    return per_sequence


def mean_next_token_nll_scalar(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    return float(mean_next_token_nll(logits, token_ids).mean().item())


class EagerAttentionIntervention(AbstractContextManager["EagerAttentionIntervention"]):
    def __init__(
        self,
        *,
        _construction_token: object,
        model: object | None = None,
        family_name: str | None = None,
        family_adapter: FamilyAdapter | None = None,
        attention_modules: Sequence[torch.nn.Module],
        kernels_by_layer: Mapping[int, Mapping[int, Sequence[float] | Float64Array | torch.Tensor]],
        expected_num_layers: int | None = None,
        expected_num_query_heads: int | None = None,
        expected_num_kv_heads: int | None = None,
        expected_repeat_factor: int | None = None,
        expected_calls_per_layer: int | Mapping[int, int] | None = None,
        expected_device: torch.device | str | None = None,
        capture_layers: Mapping[int, Sequence[int]] | None = None,
        validated_spec: SupportedModelFamilySpec | None = None,
        validation_probe: ValidationProbeRequest | None = None,
        expected_total_attention_calls: int | None = None,
    ) -> None:
        if _construction_token is not _INTERNAL_INTERVENTION_CONSTRUCTOR_TOKEN:
            raise TypeError(
                "EagerAttentionIntervention is not publicly constructible. "
                "Use one of the family-specific create_*_intervention() functions."
            )
        self._model = model
        self._family_name = family_name
        self._family_adapter = family_adapter
        self._attention_modules = list(attention_modules)
        self._module_to_layer_index = {
            id(module): index for index, module in enumerate(self._attention_modules)
        }
        self._kernels_by_layer = {
            int(layer): {
                int(head): np.asarray(kernel, dtype=np.float64) for head, kernel in per_head.items()
            }
            for layer, per_head in kernels_by_layer.items()
        }
        self._expected_num_layers = expected_num_layers
        self._expected_num_query_heads = expected_num_query_heads
        self._expected_num_kv_heads = expected_num_kv_heads
        self._expected_repeat_factor = expected_repeat_factor
        if isinstance(expected_calls_per_layer, Mapping):
            self._expected_calls_per_layer = {
                int(layer): int(count) for layer, count in expected_calls_per_layer.items()
            }
        elif expected_calls_per_layer is None:
            self._expected_calls_per_layer = None
        else:
            self._expected_calls_per_layer = {
                layer_index: int(expected_calls_per_layer)
                for layer_index in range(len(self._attention_modules))
            }
        self._expected_device = (
            torch.device(expected_device) if expected_device is not None else None
        )
        self._capture_layers = {
            int(layer): tuple(int(head) for head in heads)
            for layer, heads in (capture_layers or {}).items()
        }
        self.validated_spec = validated_spec
        self.validation_probe = validation_probe
        self._expected_total_attention_calls = expected_total_attention_calls
        self._handles: list[RemovableHandle] = []
        self.mask_captures: list[HookMaskCapture] = []
        self.hook_call_counts: dict[int, int] = {}
        self._finalized = False
        self._captured_validation_record: ValidationProbeRecord | None = None
        self._pending_probe_records: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._validation_attention_key: str | None = None
        self._validation_registry_restores: list[
            tuple[ValidationAttentionRegistry, ValidationRegistrySnapshot]
        ] = []
        self._original_attn_implementation: str | None = None
        self._attn_implementation_attr: str | None = None
        self._attn_implementation_restore_required = False
        self._encountered_forward_error = False
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if self.validated_spec is not None:
            if self.validated_spec.attn_implementation != "eager":
                raise TypeError("The production intervention path requires eager attention.")
            if len(self._attention_modules) != self.validated_spec.attention_module_count:
                raise ValueError(
                    "attention_modules count does not match validated family spec: "
                    f"{len(self._attention_modules)} vs "
                    f"{self.validated_spec.attention_module_count}."
                )
            self._expected_num_layers = self.validated_spec.layer_count
            self._expected_num_query_heads = self.validated_spec.query_head_count
            self._expected_num_kv_heads = self.validated_spec.kv_head_count
            self._expected_repeat_factor = self.validated_spec.gqa_repeat_factor
            self._expected_device = self.validated_spec.device
        if (
            self._expected_num_layers is not None
            and len(self._attention_modules) != self._expected_num_layers
        ):
            raise ValueError(
                f"expected {self._expected_num_layers} attention layers, "
                f"found {len(self._attention_modules)}."
            )
        max_layer = len(self._attention_modules) - 1
        for layer_index, per_head in self._kernels_by_layer.items():
            if layer_index < 0 or layer_index > max_layer:
                raise ValueError(f"selected layer {layer_index} is missing from attention_modules.")
            for head_index, kernel in per_head.items():
                if head_index < 0:
                    raise ValueError(f"head index must be non-negative, found {head_index}.")
                if kernel.ndim != 1:
                    raise ValueError(
                        f"kernel for layer {layer_index}, head {head_index} must be 1D."
                    )
                if self.validated_spec is not None and kernel.shape != (512,):
                    raise ValueError(
                        f"kernel for layer {layer_index}, head {head_index} "
                        "must have shape (512,), "
                        f"found {kernel.shape}."
                    )
                if not np.isfinite(kernel).all():
                    raise ValueError(
                        f"kernel for layer {layer_index}, head {head_index} "
                        "contains non-finite values."
                    )
        for layer_index, heads in self._capture_layers.items():
            if layer_index < 0 or layer_index > max_layer:
                raise ValueError(f"capture layer {layer_index} is missing from attention_modules.")
            if not heads:
                raise ValueError(f"capture layer {layer_index} must select at least one head.")
        if self._expected_calls_per_layer is not None:
            for layer_index in self._expected_calls_per_layer:
                if layer_index < 0 or layer_index > max_layer:
                    raise ValueError(
                        f"expected call layer {layer_index} is missing from attention_modules."
                    )
        if self.validation_probe is not None:
            if self.validated_spec is None or self._model is None or self._family_adapter is None:
                raise TypeError(
                    "validation_probe requires a validated real supported family model."
                )
            selected = self._kernels_by_layer.get(self.validation_probe.layer_index, {})
            if self.validation_probe.head_index not in selected:
                raise ValueError("Validation probe must target a configured selected head.")

    def __enter__(self) -> EagerAttentionIntervention:
        if self._handles:
            raise RuntimeError("intervention hooks are already installed.")
        super().__enter__()
        self._revalidate_installation_context()
        self.mask_captures.clear()
        self.hook_call_counts.clear()
        self._finalized = False
        self._captured_validation_record = None
        self._pending_probe_records.clear()
        self._encountered_forward_error = False
        try:
            if self.validation_probe is not None:
                self._install_validation_attention_interface()
            for layer_index, module in enumerate(self._attention_modules):
                self.hook_call_counts[layer_index] = 0
                self._handles.append(
                    module.register_forward_pre_hook(self._make_hook(layer_index), with_kwargs=True)
                )
        except BaseException as install_error:
            try:
                self.remove_hooks()
            except BaseException as cleanup_error:
                _raise_grouped_exceptions(
                    "Intervention hook installation failed and cleanup also failed.",
                    [install_error, cleanup_error],
                )
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        del exc, exc_tb
        try:
            if exc_type is None and not self._finalized and not self._encountered_forward_error:
                self.finalize()
        finally:
            self.remove_hooks()
        return None

    def _install_validation_attention_interface(self) -> None:
        if self._model is None or self._family_adapter is None or self.validated_spec is None:
            raise TypeError(
                "validation hook installation requires a validated real supported family model."
            )
        model = cast(_ModelContainerProtocol, self._model)
        family_adapter = self._family_adapter
        if _read_attn_implementation(model) != "eager":
            raise RuntimeError(
                "Validation hook installation requires the real model to begin in eager mode."
            )
        self._attn_implementation_attr = _attn_implementation_attr_name(model)
        self._original_attn_implementation = getattr(model.config, self._attn_implementation_attr)
        self._validation_attention_key = f"si_rebuttal_validation_{id(self)}"
        self._validation_registry_restores.clear()

        def validation_attention_forward(
            module: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor | None,
            scaling: float,
            dropout: float = 0.0,
            **kwargs: object,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            try:
                attn_output, attn_weights = family_adapter.eager_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    scaling,
                    dropout=dropout,
                    **kwargs,
                )
                self._capture_validation_from_attention_interface(
                    module=module,
                    query=query,
                    key=key,
                    attention_mask=attention_mask,
                    scaling=scaling,
                    attention_probabilities=attn_weights,
                )
                return attn_output, attn_weights
            except Exception:
                self._encountered_forward_error = True
                raise

        try:
            for registry, installed_value in self._validation_attention_registries(
                validation_attention_forward
            ):
                snapshot = registry.snapshot(self._validation_attention_key)
                self._validation_registry_restores.append((registry, snapshot))
                registry.install(self._validation_attention_key, installed_value)
            self._attn_implementation_restore_required = True
            setattr(model.config, self._attn_implementation_attr, self._validation_attention_key)
        except BaseException as install_error:
            try:
                self.remove_hooks()
            except BaseException as cleanup_error:
                _raise_grouped_exceptions(
                    "Validation attention interface installation failed and cleanup also failed.",
                    [install_error, cleanup_error],
                )
            raise

    def remove_hooks(self) -> None:
        cleanup_errors: list[BaseException] = []
        validation_attention_key = self._validation_attention_key
        try:
            for handle in self._handles:
                try:
                    handle.remove()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            while self._validation_registry_restores:
                registry, snapshot = self._validation_registry_restores.pop()
                if validation_attention_key is None:
                    continue
                try:
                    registry.restore(validation_attention_key, snapshot)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if (
                self._attn_implementation_restore_required
                and self._attn_implementation_attr is not None
                and self._model is not None
                and self._original_attn_implementation is not None
            ):
                model = cast(_ModelContainerProtocol, self._model)
                try:
                    setattr(
                        model.config,
                        self._attn_implementation_attr,
                        self._original_attn_implementation,
                    )
                except BaseException as exc:
                    cleanup_errors.append(exc)
                else:
                    self._attn_implementation_restore_required = False
        finally:
            self._handles.clear()
            self._validation_registry_restores.clear()
            self._validation_attention_key = None
            self._original_attn_implementation = None
            self._attn_implementation_attr = None
            self._attn_implementation_restore_required = False
        _raise_grouped_exceptions("Failed to fully remove intervention hooks.", cleanup_errors)

    @property
    def installed_hook_count(self) -> int:
        return len(self._handles)

    @property
    def attention_modules(self) -> tuple[torch.nn.Module, ...]:
        return tuple(self._attention_modules)

    def replace_attention_modules(self, attention_modules: Sequence[torch.nn.Module]) -> None:
        self._attention_modules = list(attention_modules)
        self._module_to_layer_index = {
            id(module): index for index, module in enumerate(self._attention_modules)
        }

    def finalize(self) -> None:
        if self._finalized:
            return
        if self._expected_calls_per_layer is not None:
            for layer_index, expected_calls in self._expected_calls_per_layer.items():
                actual_calls = self.hook_call_counts.get(layer_index, 0)
                if actual_calls != expected_calls:
                    raise AssertionError(
                        f"layer {layer_index} hook calls: expected {expected_calls}, "
                        f"found {actual_calls}."
                    )
        total_calls = sum(self.hook_call_counts.values())
        if (
            self._expected_total_attention_calls is not None
            and total_calls != self._expected_total_attention_calls
        ):
            raise AssertionError(
                f"attention hook calls: expected "
                f"{self._expected_total_attention_calls}, found {total_calls}."
            )
        if self.validation_probe is not None and self._captured_validation_record is None:
            raise RuntimeError("Validation mode requires exactly one captured probe record.")
        self._pending_probe_records.clear()
        self._finalized = True

    def assert_hook_call_count(self, *, layer_index: int, expected_calls: int) -> None:
        actual = self.hook_call_counts.get(layer_index)
        if actual != expected_calls:
            raise AssertionError(
                f"layer {layer_index} hook calls: expected {expected_calls}, found {actual}."
            )

    def cleanup(self) -> None:
        self.remove_hooks()
        self.mask_captures.clear()
        self.hook_call_counts.clear()
        self._finalized = False
        self._captured_validation_record = None
        self._pending_probe_records.clear()
        self._encountered_forward_error = False

    def latest_validation_record(self) -> ValidationProbeRecord | None:
        record = self._captured_validation_record
        self._captured_validation_record = None
        return record

    def _validation_attention_registries(
        self,
        validation_attention_forward: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    ) -> tuple[tuple[ValidationAttentionRegistry, object], ...]:
        registries: list[tuple[ValidationAttentionRegistry, object]] = []
        seen: set[int] = set()

        def add_registry(candidate: object, installed_value: object) -> None:
            registry = _coerce_validation_attention_registry(candidate)
            if registry is None:
                return
            registry_id = id(registry.registry)
            if registry_id in seen:
                return
            seen.add(registry_id)
            registries.append((registry, installed_value))

        if self._family_adapter is not None:
            add_registry(self._family_adapter.attention_interface, validation_attention_forward)
            mask_attention_interface = self._family_adapter.mask_attention_interface
            mask_registry = _coerce_validation_attention_registry(mask_attention_interface)
            if mask_attention_interface is None or mask_registry is None:
                raise RuntimeError(
                    "Validation hook installation requires a mutable mask attention-interface "
                    "registry with register/pop or mapping semantics."
                )
            mask_eager_handler = mask_registry.get("eager")
            if mask_eager_handler is None:
                raise RuntimeError(
                    "Validation hook installation requires mask_registry.get('eager')."
                )
            add_registry(mask_registry.registry, mask_eager_handler)
        for module in self._attention_modules:
            add_registry(getattr(module, "attention_interface", None), validation_attention_forward)
        if not registries:
            raise RuntimeError(
                "Validation hook installation requires an attention-interface registry with "
                "register/pop or mapping semantics."
            )
        return tuple(registries)

    def build_layer_correction(
        self,
        *,
        layer_index: int,
        num_query_heads: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        correction = torch.zeros(
            (num_query_heads, seq_len, seq_len), dtype=torch.float32, device=device
        )
        for head_index, kernel in self._kernels_by_layer.get(layer_index, {}).items():
            if head_index >= num_query_heads:
                raise ValueError(
                    f"selected head {head_index} is out of range for layer {layer_index} with "
                    f"{num_query_heads} query heads."
                )
            correction[head_index] = build_causal_toeplitz_correction(
                kernel.tolist(), sequence_length=seq_len, device=device
            )
        if not torch.isfinite(correction).all():
            raise ValueError(f"layer {layer_index} correction contains non-finite values.")
        return correction

    def apply_attention_mask_correction(
        self,
        attention_mask: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        query_heads, kv_heads = self._module_head_counts(self._attention_modules[layer_index])
        self._expected_head_shape(query_heads=query_heads, kv_heads=kv_heads)
        expanded = self._prepare_float32_mask(attention_mask, query_heads=query_heads)
        correction = self.build_layer_correction(
            layer_index=layer_index,
            num_query_heads=query_heads,
            seq_len=expanded.shape[-1],
            device=expanded.device,
        )
        post_mask = expanded + correction.unsqueeze(0)
        _require_valid_additive_mask(post_mask, name="corrected attention mask")
        self._record_probe_masks(
            layer_index=layer_index, pre_mask=expanded, post_mask=post_mask, correction=correction
        )
        return post_mask

    def _module_head_counts(self, module: torch.nn.Module) -> tuple[int, int]:
        query_heads = getattr(module, "num_heads", None)
        kv_heads = getattr(module, "num_key_value_heads", None)
        if query_heads is None:
            config = getattr(module, "config", None)
            query_heads = getattr(config, "num_attention_heads", None)
            kv_heads = getattr(config, "num_key_value_heads", kv_heads)
        if query_heads is None:
            raise ValueError(
                "attention module does not expose num_heads or config.num_attention_heads."
            )
        if kv_heads is None:
            kv_heads = query_heads
        return int(query_heads), int(kv_heads)

    def _expected_head_shape(self, *, query_heads: int, kv_heads: int) -> None:
        if (
            self._expected_num_query_heads is not None
            and query_heads != self._expected_num_query_heads
        ):
            raise ValueError(
                f"expected {self._expected_num_query_heads} query heads, found {query_heads}."
            )
        if self._expected_num_kv_heads is not None and kv_heads != self._expected_num_kv_heads:
            raise ValueError(f"expected {self._expected_num_kv_heads} KV heads, found {kv_heads}.")
        if query_heads % kv_heads != 0:
            raise ValueError(f"query_heads={query_heads} is not divisible by kv_heads={kv_heads}.")
        repeat_factor = query_heads // kv_heads
        if (
            self._expected_repeat_factor is not None
            and repeat_factor != self._expected_repeat_factor
        ):
            raise ValueError(
                f"expected repeat_factor={self._expected_repeat_factor}, found {repeat_factor}."
            )

    def _prepare_float32_mask(
        self, attention_mask: torch.Tensor, *, query_heads: int
    ) -> torch.Tensor:
        batch_size, mask_heads, query_len, key_len = attention_mask.shape
        if self._expected_device is not None and attention_mask.device != self._expected_device:
            raise ValueError(
                f"expected attention_mask device {self._expected_device}, "
                f"found {attention_mask.device}."
            )
        if mask_heads == 1:
            return (
                attention_mask.to(dtype=torch.float32)
                .expand(batch_size, query_heads, query_len, key_len)
                .clone()
            )
        if mask_heads == query_heads:
            return attention_mask.to(dtype=torch.float32).clone()
        raise ValueError(
            f"attention_mask head axis must be 1 or {query_heads}, found {mask_heads}."
        )

    def _record_probe_masks(
        self,
        *,
        layer_index: int,
        pre_mask: torch.Tensor,
        post_mask: torch.Tensor,
        correction: torch.Tensor,
    ) -> None:
        if self.validation_probe is None or layer_index != self.validation_probe.layer_index:
            return
        head_index = self.validation_probe.head_index
        for batch_index in range(pre_mask.shape[0]):
            self._pending_probe_records[(layer_index, batch_index)] = (
                pre_mask[batch_index, head_index].detach().clone(),
                post_mask[batch_index, head_index].detach().clone(),
                correction[head_index].detach().clone(),
            )

    def _capture_mask_slices(
        self,
        *,
        layer_index: int,
        pre_mask: torch.Tensor,
        post_mask: torch.Tensor,
        correction: torch.Tensor,
    ) -> None:
        head_indices = self._capture_layers.get(layer_index)
        if head_indices is None:
            return
        selected = torch.as_tensor(head_indices, dtype=torch.long, device=pre_mask.device)
        self.mask_captures.append(
            HookMaskCapture(
                layer_index=layer_index,
                head_indices=head_indices,
                pre_mask=pre_mask.index_select(1, selected)
                .detach()
                .to(dtype=torch.float32, device="cpu"),
                post_mask=post_mask.index_select(1, selected)
                .detach()
                .to(dtype=torch.float32, device="cpu"),
                correction=correction.index_select(0, selected)
                .detach()
                .to(dtype=torch.float32, device="cpu"),
            )
        )

    def _forward_signature(self, module: torch.nn.Module) -> inspect.Signature:
        if self.validated_spec is not None:
            return self.validated_spec.attention_forward_signature
        return inspect.signature(type(module).forward)

    def _revalidate_installation_context(self) -> None:
        if self.validated_spec is None:
            return
        if self._model is None or self._family_adapter is None or self._family_name is None:
            raise RuntimeError("Validated intervention lost its originating real-model context.")
        model = cast(_ModelContainerProtocol, self._model)
        refreshed = _derive_supported_model_family_spec(model, self._family_name)
        expected = self.validated_spec
        comparable_fields = (
            "family_name",
            "model_class_name",
            "config_class_name",
            "config_model_type",
            "attention_class_name",
            "layer_count",
            "query_head_count",
            "kv_head_count",
            "gqa_repeat_factor",
            "attention_mask_layout",
            "device",
            "attn_implementation",
            "attention_module_count",
            "attention_forward_signature",
        )
        for field_name in comparable_fields:
            if getattr(refreshed, field_name) != getattr(expected, field_name):
                raise RuntimeError(f"Validated model family spec drifted at {field_name}.")
        if refreshed.model is not expected.model:
            raise RuntimeError("Validated intervention model identity changed before installation.")
        if refreshed.family_adapter is not expected.family_adapter:
            raise RuntimeError("Validated intervention family adapter changed before installation.")
        if len(refreshed.attention_modules) != len(expected.attention_modules):
            raise RuntimeError(
                "Validated intervention attention module count changed before installation."
            )
        for index, (actual_module, expected_module) in enumerate(
            zip(refreshed.attention_modules, expected.attention_modules, strict=True)
        ):
            if actual_module is not expected_module:
                raise RuntimeError(
                    f"Validated intervention layer {index} attention module identity changed."
                )
        if len(self._attention_modules) != len(expected.attention_modules):
            raise RuntimeError(
                "Stored intervention attention module count no longer matches the validated model."
            )
        if any(
            actual_module is not expected_module
            for actual_module, expected_module in zip(
                self._attention_modules, expected.attention_modules, strict=True
            )
        ):
            raise RuntimeError(
                "Stored intervention attention module identities no longer "
                "match the validated model."
            )

    def _bind_forward_call(
        self,
        module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> inspect.BoundArguments:
        signature = self._forward_signature(module)
        try:
            bound = signature.bind(module, *args, **kwargs)
        except TypeError as exc:
            raise ValueError(f"{module.__class__.__name__}.forward binding failed: {exc}") from exc
        bound.apply_defaults()
        return bound

    def _rebuild_call(
        self, bound: inspect.BoundArguments
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        positional: list[object] = []
        keywords: dict[str, object] = {}
        for parameter in list(bound.signature.parameters.values())[1:]:
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional.append(bound.arguments[parameter.name])
            elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                positional.extend(bound.arguments.get(parameter.name, ()))
            elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                keywords[parameter.name] = bound.arguments[parameter.name]
            elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                keywords.update(bound.arguments.get(parameter.name, {}))
        return tuple(positional), keywords

    def _validate_mask_tensor(self, attention_mask: object) -> torch.Tensor:
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("attention_mask must be a torch.Tensor.")
        if attention_mask.ndim != 4:
            raise ValueError(
                "attention_mask must have shape [batch, heads_or_1, query_len, key_len]."
            )
        _require_valid_additive_mask(attention_mask, name="attention_mask")
        return attention_mask

    def _make_hook(self, layer_index: int):
        def hook(module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]):
            self.hook_call_counts[layer_index] = self.hook_call_counts.get(layer_index, 0) + 1
            try:
                bound = self._bind_forward_call(module, args, kwargs)
                attention_mask = self._validate_mask_tensor(bound.arguments["attention_mask"])
                query_heads, kv_heads = self._module_head_counts(module)
                self._expected_head_shape(query_heads=query_heads, kv_heads=kv_heads)
                expanded = self._prepare_float32_mask(attention_mask, query_heads=query_heads)
                if expanded.shape[-2] != expanded.shape[-1]:
                    raise ValueError(
                        "query_len and key_len must match for causal Toeplitz correction."
                    )
                correction = self.build_layer_correction(
                    layer_index=layer_index,
                    num_query_heads=query_heads,
                    seq_len=expanded.shape[-1],
                    device=expanded.device,
                )
                post_mask = expanded + correction.unsqueeze(0)
                _require_valid_additive_mask(post_mask, name="corrected attention mask")
                self._record_probe_masks(
                    layer_index=layer_index,
                    pre_mask=expanded,
                    post_mask=post_mask,
                    correction=correction,
                )
                bound.arguments["attention_mask"] = post_mask
                return self._rebuild_call(bound)
            except Exception:
                self._encountered_forward_error = True
                raise

        return hook

    def _capture_validation_from_attention_interface(
        self,
        *,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        attention_probabilities: torch.Tensor | None,
    ) -> None:
        if self.validation_probe is None:
            return
        layer_index = self._module_to_layer_index.get(id(module))
        if layer_index is None or layer_index != self.validation_probe.layer_index:
            return
        if self._captured_validation_record is not None:
            raise RuntimeError("Validation mode permits exactly one probe record per forward.")
        if attention_mask is None:
            raise RuntimeError("Validation mode requires the actual forwarded attention mask.")
        pending = self._pending_probe_records.get((layer_index, 0))
        if pending is None:
            raise RuntimeError(
                "Validation mode requires the actual hooked mask tensors "
                "from the same forward call."
            )
        if query.ndim != 4 or key.ndim != 4:
            raise RuntimeError(
                "Validation mode requires rank-4 query/key tensors from the actual eager interface."
            )
        if attention_probabilities is None or attention_probabilities.ndim != 4:
            raise RuntimeError(
                "Validation mode requires returned attention probabilities "
                "from the actual eager forward."
            )
        if query.shape[0] != 1 or key.shape[0] != 1 or attention_probabilities.shape[0] != 1:
            raise RuntimeError("Validation mode supports exactly one bounded probe batch.")
        if query.shape[-2] != VALIDATION_MAX_SEQUENCE_LENGTH:
            raise RuntimeError(
                "Validation capture requires exact sequence length "
                f"{VALIDATION_MAX_SEQUENCE_LENGTH}, "
                f"found {query.shape[-2]}."
            )
        full_query = self._validate_full_query_tensor(query, "full_q")
        full_key = self._validate_full_key_tensor(
            key,
            "full_k",
            query=full_query,
            num_query_heads=int(full_query.shape[1]),
        )
        full_post_mask = self._validate_full_post_mask(
            attention_mask,
            sequence_length=VALIDATION_MAX_SEQUENCE_LENGTH,
            device=full_query.device,
            query_head_count=int(full_query.shape[1]),
        )
        runtime_scaling = self._validate_runtime_scaling(scaling)
        repeated_key = repeat_kv_after_gqa(full_key, num_query_heads=int(full_query.shape[1]))
        head_index = self.validation_probe.head_index
        q_value = self._validate_head_tensor(full_query[0, head_index], "q")
        k_value = self._validate_head_tensor(repeated_key[0, head_index], "k_post_repeat")
        probs = self._validate_probability_matrix(
            attention_probabilities[0, head_index], VALIDATION_MAX_SEQUENCE_LENGTH
        )
        pre_mask = self._validate_mask_slice(
            pending[0], VALIDATION_MAX_SEQUENCE_LENGTH, q_value.device, "pre_mask"
        )
        post_mask = self._validate_mask_slice(
            pending[1], VALIDATION_MAX_SEQUENCE_LENGTH, q_value.device, "post_mask"
        )
        actual_correction = self._validate_mask_delta(
            pre_mask=pre_mask, post_mask=post_mask, expected=pending[2]
        )
        self._captured_validation_record = ValidationProbeRecord(
            family_name=self.validated_spec.family_name
            if self.validated_spec is not None
            else "unknown",
            layer_index=layer_index,
            head_index=head_index,
            pre_mask=pre_mask.detach().clone(),
            post_mask=post_mask.detach().clone(),
            correction=actual_correction.detach().clone(),
            q=q_value.detach().clone(),
            k=k_value.detach().clone(),
            attention_probabilities=probs.detach().clone(),
            full_q=full_query.detach().clone(),
            full_k=full_key.detach().clone(),
            full_post_mask=full_post_mask.detach().clone(),
            scaling=runtime_scaling,
        )

    def _validate_full_query_tensor(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(
                f"{name} must have shape [batch, query_heads, seq_len, head_dim], "
                f"got shape={tuple(value.shape)}."
            )
        if value.dtype != torch.bfloat16:
            raise ValueError(f"{name} must preserve eager BF16 values, got {value.dtype}.")
        if value.shape[0] != 1:
            raise ValueError(f"{name} batch axis must be 1, got {value.shape[0]}.")
        if value.shape[2] != VALIDATION_MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"{name} sequence length must be {VALIDATION_MAX_SEQUENCE_LENGTH}, "
                f"got {value.shape[2]}."
            )
        if (
            self._expected_num_query_heads is not None
            and value.shape[1] != self._expected_num_query_heads
        ):
            raise ValueError(
                f"{name} head count mismatch: expected {self._expected_num_query_heads}, "
                f"got {value.shape[1]}."
            )
        if self.validated_spec is not None and value.device != self.validated_spec.device:
            raise ValueError(
                f"{name} device mismatch: expected {self.validated_spec.device}, "
                f"got {value.device}."
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values.")
        return value

    def _validate_full_key_tensor(
        self,
        value: torch.Tensor,
        name: str,
        *,
        query: torch.Tensor,
        num_query_heads: int,
    ) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(
                f"{name} must have shape [batch, kv_heads, seq_len, head_dim], "
                f"got shape={tuple(value.shape)}."
            )
        if value.dtype != torch.bfloat16:
            raise ValueError(f"{name} must preserve eager BF16 values, got {value.dtype}.")
        if value.dtype != query.dtype:
            raise ValueError(f"{name} dtype mismatch: expected {query.dtype}, got {value.dtype}.")
        if value.device != query.device:
            raise ValueError(
                f"{name} device mismatch: expected {query.device}, got {value.device}."
            )
        if (
            value.shape[0] != query.shape[0]
            or value.shape[2] != query.shape[2]
            or value.shape[3] != query.shape[3]
        ):
            raise ValueError(f"{name} batch/sequence/head_dim must match full_q.")
        if (
            self._expected_num_kv_heads is not None
            and value.shape[1] != self._expected_num_kv_heads
        ):
            raise ValueError(
                f"{name} head count mismatch: expected {self._expected_num_kv_heads}, "
                f"got {value.shape[1]}."
            )
        if value.shape[1] <= 0 or num_query_heads % value.shape[1] != 0:
            raise ValueError(
                f"{name} GQA mismatch: query_heads={num_query_heads}, kv_heads={value.shape[1]}."
            )
        if (
            self._expected_repeat_factor is not None
            and num_query_heads // value.shape[1] != self._expected_repeat_factor
        ):
            raise ValueError(
                f"{name} repeat factor mismatch: expected {self._expected_repeat_factor}, "
                f"got {num_query_heads // value.shape[1]}."
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values.")
        return value

    def _validate_full_post_mask(
        self,
        value: torch.Tensor,
        *,
        sequence_length: int,
        device: torch.device,
        query_head_count: int,
    ) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(
                "full_post_mask must have shape [batch, query_heads, seq_len, seq_len], "
                f"got shape={tuple(value.shape)}."
            )
        if value.dtype != torch.float32:
            raise ValueError(
                f"full_post_mask must preserve the eager FP32 mask, got {value.dtype}."
            )
        if value.device != device:
            raise ValueError(
                f"full_post_mask device mismatch: expected {device}, got {value.device}."
            )
        if value.shape != (1, query_head_count, sequence_length, sequence_length):
            raise ValueError(
                "full_post_mask must preserve the exact eager mask shape "
                f"(1, {query_head_count}, {sequence_length}, {sequence_length}), "
                f"got {tuple(value.shape)}."
            )
        _require_valid_additive_mask(value, name="full_post_mask")
        return value

    def _validate_runtime_scaling(self, value: float) -> float:
        runtime_scaling = float(value)
        if not math.isfinite(runtime_scaling):
            raise ValueError("scaling must be finite.")
        return runtime_scaling

    def _validate_head_tensor(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if value.ndim != 2:
            raise ValueError(
                f"{name} must be rank-2 [seq, head_dim], got shape={tuple(value.shape)}."
            )
        output = value.to(torch.float32)
        if self.validated_spec is not None and output.device != self.validated_spec.device:
            raise ValueError(
                f"{name} device mismatch: expected {self.validated_spec.device}, "
                f"got {output.device}."
            )
        if not torch.isfinite(output).all():
            raise ValueError(f"{name} contains non-finite values.")
        return output

    def _validate_probability_matrix(
        self, probabilities: torch.Tensor, sequence_length: int
    ) -> torch.Tensor:
        if probabilities.shape != (sequence_length, sequence_length):
            raise ValueError(
                "attention_probabilities must be rank-2 [seq, seq], got "
                f"shape={tuple(probabilities.shape)}."
            )
        output = probabilities.to(torch.float32)
        if self.validated_spec is not None and output.device != self.validated_spec.device:
            raise ValueError(
                "attention_probabilities device mismatch: "
                f"expected {self.validated_spec.device}, got {output.device}."
            )
        if not torch.isfinite(output).all():
            raise ValueError("attention_probabilities contains non-finite values.")
        return output

    def _validate_mask_slice(
        self,
        value: torch.Tensor,
        sequence_length: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if value.shape != (sequence_length, sequence_length):
            raise ValueError(
                f"{name} must have shape {(sequence_length, sequence_length)}, "
                f"got {tuple(value.shape)}."
            )
        output = value.to(torch.float32)
        if output.device != device:
            raise ValueError(f"{name} device mismatch: expected {device}, got {output.device}.")
        _require_valid_additive_mask(output, name=name)
        return output

    def _validate_mask_delta(
        self,
        *,
        pre_mask: torch.Tensor,
        post_mask: torch.Tensor,
        expected: torch.Tensor,
    ) -> torch.Tensor:
        finite = torch.isfinite(pre_mask)
        if torch.isnan(post_mask).any():
            raise RuntimeError("Corrected attention mask contains NaNs.")
        if not torch.equal(torch.isneginf(pre_mask), torch.isneginf(post_mask)):
            raise RuntimeError("Masked entries must remain -inf after correction.")
        actual = torch.zeros_like(post_mask)
        actual[finite] = post_mask[finite] - pre_mask[finite]
        if not torch.allclose(actual[finite], expected[finite], atol=1e-6, rtol=0.0):
            raise RuntimeError(
                "Captured correction does not match the actual forwarded mask delta."
            )
        if torch.isnan(actual).any():
            raise RuntimeError("Captured correction contains NaNs.")
        return actual


def build_head_kernel_map(
    heads: Iterable[HeadIndex],
    kernels: Mapping[HeadIndex, Sequence[float] | Float64Array],
) -> dict[int, dict[int, Float64Array]]:
    by_layer: dict[int, dict[int, Float64Array]] = {}
    for head in heads:
        if head not in kernels:
            raise ValueError(f"missing kernel for head ({head.layer}, {head.head}).")
        by_layer.setdefault(head.layer, {})[head.head] = np.asarray(kernels[head], dtype=np.float64)
    return by_layer
