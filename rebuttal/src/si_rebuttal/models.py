from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeGuard, cast

import torch

from .config import FIXED_RUNTIME, ModelConfig, ResolvedConfig
from .controls import ExpectedAttentionCallCounts
from .intervention import (
    EagerAttentionIntervention,
    ValidationProbeRequest,
    create_llama_3_1_8b_intervention,
    create_mistral_7b_v0_1_intervention,
    create_olmo_2_7b_intervention,
)
from .provenance import sha256_tree


class ModelRegistryError(RuntimeError):
    """Raised when a model family or local checkpoint violates the frozen registry."""


class _LayerProtocol(Protocol):
    self_attn: torch.nn.Module


class _LayerContainerProtocol(Protocol):
    layers: Sequence[_LayerProtocol]


class ModelConfigLike(Protocol):
    _attn_implementation: str | None
    attn_implementation: str | None


class _ParameterLike(Protocol):
    device: torch.device


class _ModelContainerProtocol(Protocol):
    config: ModelConfigLike
    model: _LayerContainerProtocol

    def parameters(self) -> Iterable[torch.Tensor | _ParameterLike]: ...


class InterventionFactory(Protocol):
    def __call__(
        self,
        model: _ModelContainerProtocol,
        *,
        expected_call_counts: ExpectedAttentionCallCounts | None,
        selected_heads_by_layer: Mapping[int, Sequence[int]],
        kernels_by_layer_head: Mapping[tuple[int, int], torch.Tensor | Sequence[float]],
        validation_probe: ValidationProbeRequest | None = None,
    ) -> EagerAttentionIntervention: ...


class LoadedModelLike(_ModelContainerProtocol, Protocol):
    def to(self, device: torch.device) -> LoadedModelLike: ...
    def eval(self) -> object: ...


def _is_loaded_model_like(value: object) -> TypeGuard[LoadedModelLike]:
    if not hasattr(value, "config") or not hasattr(value, "model"):
        return False
    to_method = getattr(value, "to", None)
    eval_method = getattr(value, "eval", None)
    parameters_method = getattr(value, "parameters", None)
    return callable(to_method) and callable(eval_method) and callable(parameters_method)


def _resolve_attn_implementation(config: ModelConfigLike) -> str | None:
    private_value_raw = getattr(config, "_attn_implementation", None)
    public_value_raw = getattr(config, "attn_implementation", None)
    if private_value_raw is not None and not isinstance(private_value_raw, str):
        raise ModelRegistryError(
            "Model config private attention implementation marker must be a string or None."
        )
    if public_value_raw is not None and not isinstance(public_value_raw, str):
        raise ModelRegistryError(
            "Model config public attention implementation marker must be a string or None."
        )
    private_value = private_value_raw
    public_value = public_value_raw
    if private_value is not None:
        if public_value is not None and public_value != private_value:
            raise ModelRegistryError(
                "Model config attention implementation markers disagree: "
                f"private={private_value!r}, public={public_value!r}."
            )
        return private_value
    return public_value


class ModelLoader(Protocol):
    def __call__(
        self,
        path: str,
        *,
        local_files_only: bool,
        torch_dtype: torch.dtype,
        attn_implementation: str,
        device_map: None,
    ) -> LoadedModelLike: ...


class TokenizerLoader(Protocol):
    def __call__(self, path: str, *, local_files_only: bool) -> object: ...


def _llama_intervention_factory(
    model: _ModelContainerProtocol,
    *,
    expected_call_counts: ExpectedAttentionCallCounts | None,
    selected_heads_by_layer: Mapping[int, Sequence[int]],
    kernels_by_layer_head: Mapping[tuple[int, int], torch.Tensor | Sequence[float]],
    validation_probe: ValidationProbeRequest | None = None,
) -> EagerAttentionIntervention:
    return create_llama_3_1_8b_intervention(
        model,
        expected_call_counts=expected_call_counts,
        selected_heads_by_layer=selected_heads_by_layer,
        kernels_by_layer_head=kernels_by_layer_head,
        validation_probe=validation_probe,
    )


def _mistral_intervention_factory(
    model: _ModelContainerProtocol,
    *,
    expected_call_counts: ExpectedAttentionCallCounts | None,
    selected_heads_by_layer: Mapping[int, Sequence[int]],
    kernels_by_layer_head: Mapping[tuple[int, int], torch.Tensor | Sequence[float]],
    validation_probe: ValidationProbeRequest | None = None,
) -> EagerAttentionIntervention:
    return create_mistral_7b_v0_1_intervention(
        model,
        expected_call_counts=expected_call_counts,
        selected_heads_by_layer=selected_heads_by_layer,
        kernels_by_layer_head=kernels_by_layer_head,
        validation_probe=validation_probe,
    )


def _olmo_intervention_factory(
    model: _ModelContainerProtocol,
    *,
    expected_call_counts: ExpectedAttentionCallCounts | None,
    selected_heads_by_layer: Mapping[int, Sequence[int]],
    kernels_by_layer_head: Mapping[tuple[int, int], torch.Tensor | Sequence[float]],
    validation_probe: ValidationProbeRequest | None = None,
) -> EagerAttentionIntervention:
    return create_olmo_2_7b_intervention(
        model,
        expected_call_counts=expected_call_counts,
        selected_heads_by_layer=selected_heads_by_layer,
        kernels_by_layer_head=kernels_by_layer_head,
        validation_probe=validation_probe,
    )


@dataclass(frozen=True)
class ModelRegistryEntry:
    name: str
    family: str
    hf_id: str
    norm: str
    pe_scheme: str
    layers: int
    query_heads: int
    kv_heads: int
    kv_repetition: int
    intervention_factory: InterventionFactory


MODEL_REGISTRY: Mapping[str, ModelRegistryEntry] = {
    "llama-3.1-8b": ModelRegistryEntry(
        name="llama-3.1-8b",
        family="llama-3.1-8b",
        hf_id="meta-llama/Llama-3.1-8B",
        norm="RMSNorm",
        pe_scheme="RoPE",
        layers=32,
        query_heads=32,
        kv_heads=8,
        kv_repetition=4,
        intervention_factory=_llama_intervention_factory,
    ),
    "mistral-7b-v0.1": ModelRegistryEntry(
        name="mistral-7b-v0.1",
        family="mistral-7b-v0.1",
        hf_id="mistralai/Mistral-7B-v0.1",
        norm="RMSNorm",
        pe_scheme="RoPE",
        layers=32,
        query_heads=32,
        kv_heads=8,
        kv_repetition=4,
        intervention_factory=_mistral_intervention_factory,
    ),
    "olmo-2-7b": ModelRegistryEntry(
        name="olmo-2-7b",
        family="olmo-2-7b",
        hf_id="allenai/OLMo-2-7B",
        norm="LayerNorm",
        pe_scheme="RoPE",
        layers=32,
        query_heads=32,
        kv_heads=32,
        kv_repetition=1,
        intervention_factory=_olmo_intervention_factory,
    ),
}


@dataclass(frozen=True)
class FrozenModelBinding:
    model_name: str
    weights_path: Path
    tokenizer_path: Path
    weights_tree_sha256: str
    tokenizer_tree_sha256: str


@dataclass
class LoadedModelBundle:
    entry: ModelRegistryEntry
    binding: FrozenModelBinding
    model: LoadedModelLike
    tokenizer: object
    adapter: object | None
    logical_device: torch.device


def registry_entry_for_model(config: ModelConfig) -> ModelRegistryEntry:
    entry = MODEL_REGISTRY.get(config.name)
    if entry is None:
        raise ModelRegistryError(f"Unsupported frozen model: {config.name}")
    expected = (entry.layers, entry.query_heads, entry.kv_heads, entry.kv_repetition, entry.norm)
    actual = (config.layers, config.query_heads, config.kv_heads, config.kv_repetition, config.norm)
    if actual != expected:
        raise ModelRegistryError(
            "Config/model registry mismatch for "
            f"{config.name}: expected {expected}, found {actual}."
        )
    return entry


def freeze_model_binding(config: ModelConfig) -> FrozenModelBinding:
    entry = registry_entry_for_model(config)
    if not config.weights_path.exists():
        raise ModelRegistryError(
            f"Missing local weights tree for {entry.name}: {config.weights_path}"
        )
    if not config.tokenizer_path.exists():
        raise ModelRegistryError(
            f"Missing local tokenizer tree for {entry.name}: {config.tokenizer_path}"
        )
    return FrozenModelBinding(
        model_name=entry.name,
        weights_path=config.weights_path,
        tokenizer_path=config.tokenizer_path,
        weights_tree_sha256=sha256_tree(config.weights_path),
        tokenizer_tree_sha256=sha256_tree(config.tokenizer_path),
    )


def load_local_model_bundle(
    resolved_config: ResolvedConfig,
    *,
    model_name: str,
    local_files_only: bool = True,
    logical_device: str = "cuda:0",
    torch_dtype: torch.dtype = torch.bfloat16,
    auto_model_loader: ModelLoader | None = None,
    auto_tokenizer_loader: TokenizerLoader | None = None,
) -> LoadedModelBundle:
    if not local_files_only:
        raise ModelRegistryError(
            "ADR-0001 requires strictly offline local model/tokenizer loading."
        )
    entry = registry_entry_for_model(resolved_config.models[model_name])
    binding = freeze_model_binding(resolved_config.models[model_name])
    resolved_auto_model_loader = auto_model_loader
    resolved_auto_tokenizer_loader = auto_tokenizer_loader
    if resolved_auto_model_loader is None:
        from transformers import AutoModelForCausalLM

        transformers_model_loader = cast(ModelLoader, AutoModelForCausalLM.from_pretrained)

        def _default_model_loader(
            path: str,
            *,
            local_files_only: bool,
            torch_dtype: torch.dtype,
            attn_implementation: str,
            device_map: None,
        ) -> LoadedModelLike:
            loaded = transformers_model_loader(
                path,
                local_files_only=local_files_only,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                device_map=device_map,
            )
            if not _is_loaded_model_like(loaded):
                raise ModelRegistryError("Model loader returned an unsupported model container.")
            return loaded

        resolved_auto_model_loader = _default_model_loader

    if resolved_auto_tokenizer_loader is None:
        from transformers import AutoTokenizer

        transformers_tokenizer_loader = cast(TokenizerLoader, AutoTokenizer.from_pretrained)

        def _default_tokenizer_loader(path: str, *, local_files_only: bool) -> object:
            return transformers_tokenizer_loader(path, local_files_only=local_files_only)

        resolved_auto_tokenizer_loader = _default_tokenizer_loader

    attn_implementation = str(FIXED_RUNTIME["attn_implementation"])
    if resolved_auto_model_loader is None or resolved_auto_tokenizer_loader is None:
        raise AssertionError("Default loaders must be resolved before loading model artifacts.")

    tokenizer = resolved_auto_tokenizer_loader(
        str(binding.tokenizer_path),
        local_files_only=local_files_only,
    )
    model = resolved_auto_model_loader(
        str(binding.weights_path),
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=None,
    )
    device = torch.device(logical_device)
    model = model.to(device)
    _ = model.eval()
    config = model.config
    attn_impl = _resolve_attn_implementation(config)
    if attn_impl != attn_implementation:
        raise ModelRegistryError(
            f"{model_name} resolved attention implementation {attn_impl!r}, expected 'eager'."
        )
    first_parameter = next(iter(model.parameters()))
    if first_parameter.device != device:
        raise ModelRegistryError(
            f"{model_name} resolved device {first_parameter.device}, expected {device}."
        )
    return LoadedModelBundle(
        entry=entry,
        binding=binding,
        model=model,
        tokenizer=tokenizer,
        adapter=None,
        logical_device=device,
    )


def make_validation_probe(layer_index: int = 0, head_index: int = 0) -> ValidationProbeRequest:
    return ValidationProbeRequest(layer_index=layer_index, head_index=head_index)
