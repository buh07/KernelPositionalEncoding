from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pytest
import torch

from si_rebuttal.config import (
    FIXED_RUNTIME,
    ModelConfig,
    ResolvedConfig,
    ResolvedPaths,
)
from si_rebuttal.models import (
    LoadedModelLike,
    ModelConfigLike,
    ModelRegistryError,
    _resolve_attn_implementation,  # pyright: ignore[reportPrivateUsage]
    load_local_model_bundle,
)


class _LayerLike(Protocol):
    self_attn: torch.nn.Module


class _LayerContainerLike(Protocol):
    layers: Sequence[_LayerLike]


@dataclass
class _FakeLayer:
    self_attn: torch.nn.Module


@dataclass
class _FakeLayerContainer:
    layers: Sequence[_LayerLike]


@dataclass
class _FakeParameter:
    device: torch.device


class _FakeLoadedModel:
    config: ModelConfigLike
    model: _LayerContainerLike

    def __init__(self, config: ModelConfigLike) -> None:
        self.config = config
        self.model = _FakeLayerContainer(layers=(_FakeLayer(self_attn=torch.nn.Identity()),))
        self._parameters: tuple[_FakeParameter, ...] = (_FakeParameter(torch.device("cpu")),)

    def parameters(self) -> Sequence[_FakeParameter]:
        return self._parameters

    def to(self, device: torch.device) -> _FakeLoadedModel:
        normalized_device = torch.device(device)
        self._parameters = tuple(_FakeParameter(normalized_device) for _ in self._parameters)
        return self

    def eval(self) -> _FakeLoadedModel:
        return self


class _PrivateOnlyConfig:
    _attn_implementation: object

    def __init__(self, private_value: object) -> None:
        self._attn_implementation = private_value

    def __getattribute__(self, name: str) -> object:
        if name == "attn_implementation":
            raise AttributeError(name)
        return object.__getattribute__(self, name)


class _PublicOnlyConfig:
    attn_implementation: object

    def __init__(self, public_value: object) -> None:
        self.attn_implementation = public_value


class _DualConfig:
    _attn_implementation: object
    attn_implementation: object

    def __init__(self, private_value: object, public_value: object) -> None:
        self._attn_implementation = private_value
        self.attn_implementation = public_value


class _MissingBothConfig:
    def __getattribute__(self, name: str) -> object:
        if name in {"_attn_implementation", "attn_implementation"}:
            raise AttributeError(name)
        return object.__getattribute__(self, name)


class _Transformers53PrivateOnlyConfig(_PrivateOnlyConfig):
    pass


def _make_resolved_config(tmp_path: Path, *, model_name: str = "llama-3.1-8b") -> ResolvedConfig:
    weights_path = tmp_path / "weights"
    tokenizer_path = tmp_path / "tokenizer"
    weights_path.mkdir()
    tokenizer_path.mkdir()
    (weights_path / "config.json").write_text("{}", encoding="ascii")
    (tokenizer_path / "tokenizer.json").write_text("{}", encoding="ascii")

    model = ModelConfig(
        name=model_name,
        weight_env="UNUSED_WEIGHT_ENV",
        tokenizer_env="UNUSED_TOKENIZER_ENV",
        weights_path=weights_path,
        tokenizer_path=tokenizer_path,
        layers=32,
        query_heads=32,
        kv_heads=8,
        kv_repetition=4,
        norm="RMSNorm",
    )
    paths = ResolvedPaths(
        project_root=tmp_path,
        runs_root=tmp_path / "runs",
        data_root=tmp_path / "data",
        models_root=tmp_path / "models",
        logs_root=tmp_path / "logs",
        cache_root=tmp_path / "cache",
    )
    return ResolvedConfig(
        schema_version=1,
        seed_namespace=29039,
        paths=paths,
        runtime=cast(Mapping[str, object], dict(FIXED_RUNTIME)),
        datasets={},
        models={model_name: model},
        counts={},
        statistics={},
        validation={},
        benchmark={},
        placement={},
        identity_sha256="test-config",
        source_path=tmp_path / "base.toml",
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(_PrivateOnlyConfig("eager"), "eager", id="private-only"),
        pytest.param(_PublicOnlyConfig("eager"), "eager", id="public-only"),
        pytest.param(_DualConfig("eager", "eager"), "eager", id="equal-dual"),
        pytest.param(
            _Transformers53PrivateOnlyConfig("eager"),
            "eager",
            id="transformers-5.3-private-eager-public-absent",
        ),
        pytest.param(_MissingBothConfig(), None, id="both-missing"),
    ],
)
def test_resolve_attn_implementation_accepts_supported_marker_shapes(
    config: object, expected: str | None
) -> None:
    resolved = _resolve_attn_implementation(cast(ModelConfigLike, config))

    assert resolved == expected


def test_resolve_attn_implementation_rejects_conflicting_markers() -> None:
    config = cast(ModelConfigLike, _DualConfig("eager", "sdpa"))

    with pytest.raises(ModelRegistryError, match="markers disagree"):
        _resolve_attn_implementation(config)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_PrivateOnlyConfig(7), id="invalid-private-type"),
        pytest.param(_PublicOnlyConfig(7), id="invalid-public-type"),
    ],
)
def test_resolve_attn_implementation_rejects_non_string_markers(
    config: object,
) -> None:
    with pytest.raises(ModelRegistryError, match="must be a string or None"):
        _resolve_attn_implementation(cast(ModelConfigLike, config))


def test_load_local_model_bundle_stays_offline_and_rejects_non_eager_resolution(
    tmp_path: Path,
) -> None:
    resolved_config = _make_resolved_config(tmp_path)

    loader_calls: list[tuple[str, bool, torch.dtype, str, None]] = []
    tokenizer_calls: list[tuple[str, bool]] = []

    def auto_model_loader(
        path: str,
        *,
        local_files_only: bool,
        torch_dtype: torch.dtype,
        attn_implementation: str,
        device_map: None,
    ) -> LoadedModelLike:
        loader_calls.append((path, local_files_only, torch_dtype, attn_implementation, device_map))
        non_eager_config = cast(ModelConfigLike, cast(object, _PrivateOnlyConfig("sdpa")))
        return cast(LoadedModelLike, cast(object, _FakeLoadedModel(non_eager_config)))

    def auto_tokenizer_loader(path: str, *, local_files_only: bool) -> object:
        tokenizer_calls.append((path, local_files_only))
        return object()

    with pytest.raises(ModelRegistryError, match=r"resolved attention implementation 'sdpa'"):
        load_local_model_bundle(
            resolved_config,
            model_name="llama-3.1-8b",
            local_files_only=True,
            logical_device="cpu",
            auto_model_loader=auto_model_loader,
            auto_tokenizer_loader=auto_tokenizer_loader,
        )

    assert loader_calls == [
        (
            str(resolved_config.models["llama-3.1-8b"].weights_path),
            True,
            torch.bfloat16,
            "eager",
            None,
        )
    ]
    assert tokenizer_calls == [(str(resolved_config.models["llama-3.1-8b"].tokenizer_path), True)]


def test_load_local_model_bundle_rejects_non_offline_loading(tmp_path: Path) -> None:
    resolved_config = _make_resolved_config(tmp_path)

    with pytest.raises(ModelRegistryError, match="strictly offline"):
        load_local_model_bundle(
            resolved_config,
            model_name="llama-3.1-8b",
            local_files_only=False,
        )
