from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .provenance import canonical_json_bytes, sha256_bytes


class ConfigError(ValueError):
    """Raised when a rebuttal config drifts from ADR-0001."""


FIXED_SEED_NAMESPACE = 29039
FIXED_RUNTIME = {
    "torch_dtype": "bfloat16",
    "attn_implementation": "eager",
    "chunk_length": 512,
}
FIXED_COUNTS = {
    "fit_sequences": 50,
    "eval_sequences": 100,
    "head_bins": 20,
    "offset_permutation_trials": 3,
    "norm_trials": 3,
}
FIXED_STATISTICS = {
    "bootstrap_resamples": 10000,
    "bootstrap_percentiles": (2.5, 97.5),
    "bootstrap_method": "linear",
    "spearman_permutations": 200000,
}
FIXED_VALIDATION = {
    "zero_kernel_atol": 2e-2,
    "zero_kernel_rtol": 2e-2,
    "zero_kernel_nll_abs": 1e-3,
    "constant_kernel_atol": 5e-3,
    "constant_kernel_rtol": 5e-3,
    "constant_kernel_nll_abs": 1e-3,
    "toeplitz_max_abs": 1e-6,
    "attention_reconstruction_atol": 5e-3,
    "attention_reconstruction_rtol": 5e-3,
}
FIXED_DATASETS = {
    "wikipedia": {
        "domain": "wikipedia",
        "repository": "Salesforce/wikitext",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "field": "text",
        "document_title_regex": r"^ = [^=]+ = $",
    },
    "code": {
        "domain": "code",
        "repository": "code-search-net/code_search_net",
        "revision": "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
        "config": "python",
        "split": "train",
        "field": "whole_func_string",
        "repository_field": "repository_name",
    },
}
FIXED_MODELS = {
    "llama-3.1-8b": {
        "layers": 32,
        "query_heads": 32,
        "kv_heads": 8,
        "kv_repetition": 4,
        "norm": "RMSNorm",
    },
    "mistral-7b-v0.1": {
        "layers": 32,
        "query_heads": 32,
        "kv_heads": 8,
        "kv_repetition": 4,
        "norm": "RMSNorm",
    },
    "olmo-2-7b": {
        "layers": 32,
        "query_heads": 32,
        "kv_heads": 32,
        "kv_repetition": 1,
        "norm": "LayerNorm",
    },
}
FIXED_PLACEMENT = {
    "gpu0_models": ("llama-3.1-8b", "olmo-2-7b"),
    "gpu1_models": ("mistral-7b-v0.1",),
    "logical_device": "cuda:0",
}
BASE_TOP_LEVEL_KEYS = {
    "schema_version",
    "seed_namespace",
    "paths",
    "runtime",
    "datasets",
    "counts",
    "controls",
    "statistics",
    "validation",
    "benchmark",
    "placement",
    "models",
}
BASE_PATH_KEYS = {
    "project_root",
    "runs_root_env",
    "data_root_env",
    "models_root_env",
    "logs_root_env",
    "cache_root_env",
}
BASE_RUNTIME_KEYS = {"torch_dtype", "attn_implementation", "chunk_length"}
BASE_COUNTS_KEYS = {"fit_sequences", "eval_sequences", "head_bins"}
BASE_CONTROLS_KEYS = {"offset_permutation_trials", "norm_trials"}
BASE_STATISTICS_KEYS = {
    "bootstrap_resamples",
    "bootstrap_percentiles",
    "bootstrap_method",
    "spearman_permutations",
}
BASE_VALIDATION_KEYS = {
    "zero_kernel_atol",
    "zero_kernel_rtol",
    "zero_kernel_nll_abs",
    "constant_kernel_atol",
    "constant_kernel_rtol",
    "constant_kernel_nll_abs",
    "toeplitz_max_abs",
    "attention_reconstruction_atol",
    "attention_reconstruction_rtol",
}
BASE_BENCHMARK_KEYS = {
    "fit_sequences",
    "eval_sequences",
    "max_lane_hours",
    "min_free_space_gib",
    "artifact_margin_multiplier",
}
BASE_PLACEMENT_KEYS = {"gpu0_models", "gpu1_models", "logical_device"}
SWEEP_TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "models",
    "directions",
    "head_bins",
    "fit_sequences",
    "eval_sequences",
    "controls",
    "offset_permutation_trials",
    "norm_trials",
    "include_depth",
    "depth_direction",
    "depth_unit",
    "torch_dtype",
    "attn_implementation",
    "benchmark_fit_sequences",
    "benchmark_eval_sequences",
    "benchmark_max_lane_hours",
    "benchmark_min_free_space_gib",
    "placement_gpu0",
    "placement_gpu1",
}


@dataclass(frozen=True)
class DatasetConfig:
    domain: str
    repository: str
    revision: str
    config: str
    split: str
    field: str
    document_title_regex: str | None = None
    repository_field: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    weight_env: str
    tokenizer_env: str
    weights_path: Path
    tokenizer_path: Path
    layers: int
    query_heads: int
    kv_heads: int
    kv_repetition: int
    norm: str


@dataclass(frozen=True)
class ResolvedPaths:
    project_root: Path
    runs_root: Path
    data_root: Path
    models_root: Path
    logs_root: Path
    cache_root: Path


@dataclass(frozen=True)
class ResolvedConfig:
    schema_version: int
    seed_namespace: int
    paths: ResolvedPaths
    runtime: Mapping[str, Any]
    datasets: Mapping[str, DatasetConfig]
    models: Mapping[str, ModelConfig]
    counts: Mapping[str, int]
    statistics: Mapping[str, Any]
    validation: Mapping[str, float]
    benchmark: Mapping[str, float | int]
    placement: Mapping[str, Any]
    identity_sha256: str
    source_path: Path


@dataclass(frozen=True)
class SweepConfig:
    schema_version: int
    name: str
    models: tuple[str, ...]
    directions: tuple[str, ...]
    head_bins: int
    fit_sequences: int
    eval_sequences: int
    controls: tuple[str, ...]
    offset_permutation_trials: int
    norm_trials: int
    include_depth: bool
    depth_direction: str
    depth_unit: str
    torch_dtype: str
    attn_implementation: str
    benchmark_fit_sequences: int
    benchmark_eval_sequences: int
    benchmark_max_lane_hours: float
    benchmark_min_free_space_gib: float
    placement_gpu0: tuple[str, ...]
    placement_gpu1: tuple[str, ...]
    identity_sha256: str
    source_path: Path


def _expect_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ConfigError(f"{name} drifted: expected {expected!r}, found {actual!r}")


def _reject_unknown_keys(name: str, payload: Mapping[str, Any], expected_keys: set[str]) -> None:
    unknown = sorted(set(payload) - expected_keys)
    if unknown:
        raise ConfigError(f"{name} contains unknown keys: {unknown}")


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return payload


def _resolve_env_path(env_name: str, env: Mapping[str, str]) -> Path:
    raw = env.get(env_name, "").strip()
    if not raw:
        raise ConfigError(f"Environment variable {env_name} is required")
    return Path(raw).expanduser().resolve()


def _contains_parent_reference(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def _resolve_rebuttal_root(raw: str, *, config_dir: Path) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ConfigError(f"paths.project_root must be relative, found absolute path {raw!r}")
    return (config_dir / candidate).resolve()


def _resolve_mutable_root(env_name: str, env: Mapping[str, str], *, rebuttal_root: Path) -> Path:
    raw = env.get(env_name, "").strip()
    if not raw:
        raise ConfigError(f"Environment variable {env_name} is required")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        raise ConfigError(
            f"Environment variable {env_name} must be a relative path under {rebuttal_root}"
        )
    if _contains_parent_reference(candidate):
        raise ConfigError(f"Environment variable {env_name} may not contain '..': {raw!r}")
    resolved = (rebuttal_root / candidate).resolve()
    try:
        resolved.relative_to(rebuttal_root)
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable {env_name} escapes rebuttal root: {raw!r}"
        ) from exc
    return resolved


def _dataset_from_mapping(name: str, payload: Mapping[str, Any]) -> DatasetConfig:
    expected = FIXED_DATASETS[name]
    _reject_unknown_keys(f"datasets.{name}", payload, set(expected))
    for key, value in expected.items():
        _expect_equal(f"datasets.{name}.{key}", payload.get(key), value)
    return DatasetConfig(
        domain=str(payload["domain"]),
        repository=str(payload["repository"]),
        revision=str(payload["revision"]),
        config=str(payload["config"]),
        split=str(payload["split"]),
        field=str(payload["field"]),
        document_title_regex=payload.get("document_title_regex"),
        repository_field=payload.get("repository_field"),
    )


def _model_from_mapping(
    name: str, payload: Mapping[str, Any], env: Mapping[str, str]
) -> ModelConfig:
    expected = FIXED_MODELS[name]
    _reject_unknown_keys(f"models.{name}", payload, set(expected) | {"weight_env", "tokenizer_env"})
    for key, value in expected.items():
        _expect_equal(f"models.{name}.{key}", payload.get(key), value)
    weight_env = str(payload.get("weight_env", ""))
    tokenizer_env = str(payload.get("tokenizer_env", ""))
    if not weight_env or not tokenizer_env:
        raise ConfigError(f"models.{name} must define weight_env and tokenizer_env")
    return ModelConfig(
        name=name,
        weight_env=weight_env,
        tokenizer_env=tokenizer_env,
        weights_path=_resolve_env_path(weight_env, env),
        tokenizer_path=_resolve_env_path(tokenizer_env, env),
        layers=int(payload["layers"]),
        query_heads=int(payload["query_heads"]),
        kv_heads=int(payload["kv_heads"]),
        kv_repetition=int(payload["kv_repetition"]),
        norm=str(payload["norm"]),
    )


def _config_identity(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def load_base_config(path: str | Path, env: Mapping[str, str] | None = None) -> ResolvedConfig:
    """Load the frozen base config and reject any scientific drift."""

    source_path = Path(path).resolve()
    payload = _read_toml(source_path)
    env_map = os.environ if env is None else env
    _reject_unknown_keys("base", payload, BASE_TOP_LEVEL_KEYS)

    _expect_equal("schema_version", payload.get("schema_version"), 1)
    _expect_equal("seed_namespace", payload.get("seed_namespace"), FIXED_SEED_NAMESPACE)

    runtime = dict(payload.get("runtime", {}))
    _reject_unknown_keys("runtime", runtime, BASE_RUNTIME_KEYS)
    for key, value in FIXED_RUNTIME.items():
        _expect_equal(f"runtime.{key}", runtime.get(key), value)

    counts = dict(payload.get("counts", {}))
    _reject_unknown_keys("counts", counts, BASE_COUNTS_KEYS)
    _expect_equal(
        "counts.fit_sequences", counts.get("fit_sequences"), FIXED_COUNTS["fit_sequences"]
    )
    _expect_equal(
        "counts.eval_sequences", counts.get("eval_sequences"), FIXED_COUNTS["eval_sequences"]
    )
    _expect_equal("counts.head_bins", counts.get("head_bins"), FIXED_COUNTS["head_bins"])

    controls = dict(payload.get("controls", {}))
    _reject_unknown_keys("controls", controls, BASE_CONTROLS_KEYS)
    _expect_equal(
        "controls.offset_permutation_trials",
        controls.get("offset_permutation_trials"),
        FIXED_COUNTS["offset_permutation_trials"],
    )
    _expect_equal("controls.norm_trials", controls.get("norm_trials"), FIXED_COUNTS["norm_trials"])

    statistics = dict(payload.get("statistics", {}))
    _reject_unknown_keys("statistics", statistics, BASE_STATISTICS_KEYS)
    for key, value in FIXED_STATISTICS.items():
        actual = tuple(statistics.get(key, ())) if isinstance(value, tuple) else statistics.get(key)
        _expect_equal(f"statistics.{key}", actual, value)

    validation = dict(payload.get("validation", {}))
    _reject_unknown_keys("validation", validation, BASE_VALIDATION_KEYS)
    for key, value in FIXED_VALIDATION.items():
        _expect_equal(f"validation.{key}", validation.get(key), value)

    benchmark = dict(payload.get("benchmark", {}))
    _reject_unknown_keys("benchmark", benchmark, BASE_BENCHMARK_KEYS)
    _expect_equal("benchmark.fit_sequences", benchmark.get("fit_sequences"), 2)
    _expect_equal("benchmark.eval_sequences", benchmark.get("eval_sequences"), 4)
    _expect_equal("benchmark.max_lane_hours", benchmark.get("max_lane_hours"), 120.0)
    _expect_equal("benchmark.min_free_space_gib", benchmark.get("min_free_space_gib"), 20.0)
    _expect_equal(
        "benchmark.artifact_margin_multiplier", benchmark.get("artifact_margin_multiplier"), 2.0
    )

    placement = dict(payload.get("placement", {}))
    _reject_unknown_keys("placement", placement, BASE_PLACEMENT_KEYS)
    for key, value in FIXED_PLACEMENT.items():
        actual = tuple(placement.get(key, ())) if isinstance(value, tuple) else placement.get(key)
        _expect_equal(f"placement.{key}", actual, value)

    paths_payload = dict(payload.get("paths", {}))
    _reject_unknown_keys("paths", paths_payload, BASE_PATH_KEYS)
    config_dir = source_path.parent
    project_root = _resolve_rebuttal_root(
        str(paths_payload.get("project_root", "")), config_dir=config_dir
    )
    runs_root = _resolve_mutable_root(
        str(paths_payload.get("runs_root_env", "")), env_map, rebuttal_root=project_root
    )
    data_root = _resolve_mutable_root(
        str(paths_payload.get("data_root_env", "")), env_map, rebuttal_root=project_root
    )
    models_root = _resolve_env_path(str(paths_payload.get("models_root_env", "")), env_map)
    logs_root = _resolve_mutable_root(
        str(paths_payload.get("logs_root_env", "")), env_map, rebuttal_root=project_root
    )
    cache_root = _resolve_mutable_root(
        str(paths_payload.get("cache_root_env", "")), env_map, rebuttal_root=project_root
    )
    resolved_paths = ResolvedPaths(
        project_root=project_root,
        runs_root=runs_root,
        data_root=data_root,
        models_root=models_root,
        logs_root=logs_root,
        cache_root=cache_root,
    )

    datasets_payload = payload.get("datasets", {})
    models_payload = payload.get("models", {})
    _reject_unknown_keys("datasets", datasets_payload, set(FIXED_DATASETS))
    _reject_unknown_keys("models", models_payload, set(FIXED_MODELS))
    datasets = {
        name: _dataset_from_mapping(name, datasets_payload[name]) for name in FIXED_DATASETS
    }
    models = {
        name: _model_from_mapping(name, models_payload[name], env_map) for name in FIXED_MODELS
    }

    identity_payload = {
        "schema_version": payload["schema_version"],
        "seed_namespace": payload["seed_namespace"],
        "runtime": runtime,
        "datasets": {key: asdict(value) for key, value in datasets.items()},
        "models": {
            key: {
                "name": value.name,
                "weight_env": value.weight_env,
                "tokenizer_env": value.tokenizer_env,
                "layers": value.layers,
                "query_heads": value.query_heads,
                "kv_heads": value.kv_heads,
                "kv_repetition": value.kv_repetition,
                "norm": value.norm,
            }
            for key, value in models.items()
        },
        "counts": {**counts, **controls},
        "statistics": statistics,
        "validation": validation,
        "benchmark": benchmark,
        "placement": placement,
    }

    return ResolvedConfig(
        schema_version=1,
        seed_namespace=FIXED_SEED_NAMESPACE,
        paths=resolved_paths,
        runtime=runtime,
        datasets=datasets,
        models=models,
        counts={
            "fit_sequences": FIXED_COUNTS["fit_sequences"],
            "eval_sequences": FIXED_COUNTS["eval_sequences"],
            "head_bins": FIXED_COUNTS["head_bins"],
            "offset_permutation_trials": FIXED_COUNTS["offset_permutation_trials"],
            "norm_trials": FIXED_COUNTS["norm_trials"],
        },
        statistics=statistics,
        validation=validation,
        benchmark=benchmark,
        placement=placement,
        identity_sha256=_config_identity(identity_payload),
        source_path=source_path,
    )


def load_sweep_config(path: str | Path) -> SweepConfig:
    """Load the frozen sweep config and reject unsupported combinations."""

    source_path = Path(path).resolve()
    payload = _read_toml(source_path)
    _reject_unknown_keys("sweep", payload, SWEEP_TOP_LEVEL_KEYS)
    _expect_equal("schema_version", payload.get("schema_version"), 1)
    models = tuple(payload.get("models", ()))
    directions = tuple(payload.get("directions", ()))
    controls = tuple(payload.get("controls", ()))
    placement_gpu0 = tuple(payload.get("placement_gpu0", ()))
    placement_gpu1 = tuple(payload.get("placement_gpu1", ()))

    _expect_equal("models", models, tuple(FIXED_MODELS))
    _expect_equal("directions", directions, ("wikipedia_to_code", "code_to_wikipedia"))
    _expect_equal("head_bins", payload.get("head_bins"), FIXED_COUNTS["head_bins"])
    _expect_equal("fit_sequences", payload.get("fit_sequences"), FIXED_COUNTS["fit_sequences"])
    _expect_equal("eval_sequences", payload.get("eval_sequences"), FIXED_COUNTS["eval_sequences"])
    _expect_equal(
        "controls", controls, ("source_kernel", "target_kernel", "offset_permutation", "norm")
    )
    _expect_equal(
        "offset_permutation_trials",
        payload.get("offset_permutation_trials"),
        FIXED_COUNTS["offset_permutation_trials"],
    )
    _expect_equal("norm_trials", payload.get("norm_trials"), FIXED_COUNTS["norm_trials"])
    _expect_equal("include_depth", payload.get("include_depth"), True)
    _expect_equal("depth_direction", payload.get("depth_direction"), "wikipedia_to_code")
    _expect_equal("depth_unit", payload.get("depth_unit"), "layer")
    _expect_equal("torch_dtype", payload.get("torch_dtype"), FIXED_RUNTIME["torch_dtype"])
    _expect_equal(
        "attn_implementation",
        payload.get("attn_implementation"),
        FIXED_RUNTIME["attn_implementation"],
    )
    _expect_equal("benchmark_fit_sequences", payload.get("benchmark_fit_sequences"), 2)
    _expect_equal("benchmark_eval_sequences", payload.get("benchmark_eval_sequences"), 4)
    _expect_equal("benchmark_max_lane_hours", payload.get("benchmark_max_lane_hours"), 120.0)
    _expect_equal("benchmark_min_free_space_gib", payload.get("benchmark_min_free_space_gib"), 20.0)
    _expect_equal("placement_gpu0", placement_gpu0, FIXED_PLACEMENT["gpu0_models"])
    _expect_equal("placement_gpu1", placement_gpu1, FIXED_PLACEMENT["gpu1_models"])

    identity_sha256 = _config_identity({key: payload[key] for key in payload})
    return SweepConfig(
        schema_version=1,
        name=str(payload["name"]),
        models=models,
        directions=directions,
        head_bins=int(payload["head_bins"]),
        fit_sequences=int(payload["fit_sequences"]),
        eval_sequences=int(payload["eval_sequences"]),
        controls=controls,
        offset_permutation_trials=int(payload["offset_permutation_trials"]),
        norm_trials=int(payload["norm_trials"]),
        include_depth=bool(payload["include_depth"]),
        depth_direction=str(payload["depth_direction"]),
        depth_unit=str(payload["depth_unit"]),
        torch_dtype=str(payload["torch_dtype"]),
        attn_implementation=str(payload["attn_implementation"]),
        benchmark_fit_sequences=int(payload["benchmark_fit_sequences"]),
        benchmark_eval_sequences=int(payload["benchmark_eval_sequences"]),
        benchmark_max_lane_hours=float(payload["benchmark_max_lane_hours"]),
        benchmark_min_free_space_gib=float(payload["benchmark_min_free_space_gib"]),
        placement_gpu0=placement_gpu0,
        placement_gpu1=placement_gpu1,
        identity_sha256=identity_sha256,
        source_path=source_path,
    )
