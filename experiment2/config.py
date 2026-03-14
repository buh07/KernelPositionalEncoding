from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiment1.config import (
    DATASETS,
    DEFAULT_MODEL_PROFILE,
    SUPPORTED_MODEL_PROFILES,
    get_models_for_profile,
)


def _build_model_lookup() -> dict[str, object]:
    out: dict[str, object] = {}
    for profile in SUPPORTED_MODEL_PROFILES:
        for spec in get_models_for_profile(profile):
            out[spec.name] = spec
    return out


def model_names_for_profile(model_profile: str = DEFAULT_MODEL_PROFILE) -> tuple[str, ...]:
    models = get_models_for_profile(model_profile)
    return tuple(spec.name for spec in models)


def model_sets_for_profile(model_profile: str = DEFAULT_MODEL_PROFILE) -> dict[str, tuple[str, ...]]:
    rope: list[str] = []
    abspe: list[str] = []
    nope: list[str] = []
    for spec in get_models_for_profile(model_profile):
        scheme = str(spec.pe_scheme).lower()
        if scheme == "rope":
            rope.append(spec.name)
        elif "absolute" in scheme:
            abspe.append(spec.name)
        elif scheme in {"none", "nope"}:
            nope.append(spec.name)
        else:
            raise ValueError(f"Unsupported positional-encoding scheme '{spec.pe_scheme}' for model {spec.name}")
    return {
        "rope": tuple(rope),
        "abspe": tuple(abspe),
        "nope": tuple(nope),
    }


MODEL_BY_NAME = _build_model_lookup()
DATASET_BY_NAME = {spec.name: spec for spec in DATASETS}
_DEFAULT_MODEL_SETS = model_sets_for_profile(DEFAULT_MODEL_PROFILE)
ROPE_MODELS = _DEFAULT_MODEL_SETS["rope"]
ABSPE_MODELS = _DEFAULT_MODEL_SETS["abspe"]
NOPE_MODELS = _DEFAULT_MODEL_SETS["nope"]

SYNTHETIC_TASKS = ("local_copy_offset", "local_key_match", "delayed_copy", "long_range_retrieval")
BRIDGE_TASKS = ("copy_offset_bridge", "retrieval_bridge")
TIER1_TASK = "tier1_stratified_ppl"

INTERVENTIONS_7 = (
    "none",
    "ablate_high_medium",
    "ablate_high_strong",
    "ablate_low_medium",
    "ablate_low_strong",
    "random_medium",
    "random_strong",
)
INTERVENTIONS_STRONG = ("none", "ablate_high_strong", "ablate_low_strong", "random_strong")
TIER1_DATASETS = ("wiki40b_en_pre2019", "codesearchnet_python_snapshot")

DEFAULT_SYNTHETIC_COUNT = 200
DEFAULT_TIER1_COUNT = 100


@dataclass(frozen=True)
class Experiment2Paths:
    results_root: Path
    data_root: Path

    @staticmethod
    def default() -> "Experiment2Paths":
        root = Path(__file__).resolve().parents[1]
        return Experiment2Paths(results_root=root / "results", data_root=root / "data")
