from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

EXPECTED_KERNEL_LENGTH = 512
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ExpectedAttentionCallCounts:
    per_layer_attention_calls: Mapping[int, int]

    def __post_init__(self) -> None:
        normalized = {
            int(layer): int(count) for layer, count in self.per_layer_attention_calls.items()
        }
        if not normalized:
            raise ValueError("per_layer_attention_calls must be non-empty.")
        for layer_index, count in normalized.items():
            if layer_index < 0:
                raise ValueError(f"layer index must be non-negative, got {layer_index}.")
            if count <= 0:
                raise ValueError(
                    "per-layer attention call counts must be positive, got "
                    f"layer {layer_index}={count}."
                )
        object.__setattr__(self, "per_layer_attention_calls", normalized)

    @property
    def total_attention_calls(self) -> int:
        return int(sum(self.per_layer_attention_calls.values()))

    def require_complete(self, *, layer_count: int) -> dict[int, int]:
        required = set(range(layer_count))
        actual = set(self.per_layer_attention_calls)
        if actual != required:
            missing = sorted(required - actual)
            extra = sorted(actual - required)
            details: list[str] = []
            if missing:
                details.append(f"missing layers {missing}")
            if extra:
                details.append(f"unexpected layers {extra}")
            joined = ", ".join(details)
            raise ValueError(
                "Expected complete per-layer attention call mapping for "
                f"layers 0..{layer_count - 1}; {joined}."
            )
        return dict(self.per_layer_attention_calls)


def _as_protocol_kernel(kernel: Sequence[float] | npt.ArrayLike, *, name: str) -> FloatArray:
    values = np.asarray(kernel, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape [{EXPECTED_KERNEL_LENGTH}].")
    if values.shape[0] != EXPECTED_KERNEL_LENGTH:
        raise ValueError(
            f"{name} must have shape ({EXPECTED_KERNEL_LENGTH},), found {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")
    return values


def compact_ascii_json_seed(payload: Sequence[object]) -> int:
    encoded = json.dumps(
        list(payload),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def make_pcg64(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(seed)))


def permute_offsets(kernel: Sequence[float] | npt.ArrayLike, *, seed: int) -> FloatArray:
    source = _as_protocol_kernel(kernel, name="kernel")
    rng = make_pcg64(seed)
    return source[rng.permutation(source.shape[0])]


def zero_mean_l2_control(kernel: Sequence[float] | npt.ArrayLike, *, seed: int) -> FloatArray:
    source = _as_protocol_kernel(kernel, name="kernel")
    target_norm = float(np.linalg.norm(source))
    if target_norm == 0.0:
        raise ValueError("zero kernel norm is a hard failure.")
    rng = make_pcg64(seed)
    draw = rng.standard_normal(source.shape[0]).astype(np.float64, copy=False)
    draw -= draw.mean()
    draw_norm = float(np.linalg.norm(draw))
    if draw_norm == 0.0:
        raise ValueError("random control draw had zero norm.")
    return draw * (target_norm / draw_norm)


def causal_toeplitz_correction(
    kernel: Sequence[float] | npt.ArrayLike, *, seq_len: int | None = None
) -> FloatArray:
    values = np.asarray(kernel, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("kernel must be 1D.")
    if not np.isfinite(values).all():
        raise ValueError("kernel contains non-finite values.")
    if seq_len is None:
        seq_len = int(values.shape[0])
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if values.shape[0] != seq_len:
        raise ValueError(f"kernel has length {values.shape[0]} but seq_len={seq_len}.")
    correction = np.zeros((seq_len, seq_len), dtype=np.float64)
    for query_index in range(seq_len):
        for key_index in range(query_index + 1):
            correction[query_index, key_index] = -values[query_index - key_index]
    return correction
