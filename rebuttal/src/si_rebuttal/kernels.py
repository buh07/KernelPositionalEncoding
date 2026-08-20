from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import torch

SOURCE_OFFSET_START = 1
SOURCE_OFFSET_STOP = 511
RAW_OFFSET_START = 0
RAW_OFFSET_STOP = 511
TOP_NON_DC_COMPONENTS = 8
NUM_HEAD_BINS = 20
FloatArray = npt.NDArray[np.float64]


class _TorchFFTProtocol(Protocol):
    def rfft(
        self, input: torch.Tensor, n: int | None = None, dim: int = -1, norm: str | None = None
    ) -> torch.Tensor: ...

    def irfft(
        self, input: torch.Tensor, n: int | None = None, dim: int = -1, norm: str | None = None
    ) -> torch.Tensor: ...


_TORCH_FFT_OBJECT: object = object.__getattribute__(torch, "fft")
_TORCH_FFT = cast(_TorchFFTProtocol, _TORCH_FFT_OBJECT)


@dataclass(frozen=True, order=True)
class HeadIndex:
    layer: int
    head: int


@dataclass(frozen=True)
class SourceScoreResult:
    per_sequence_r2: FloatArray
    mean_r2: float
    smoothed_means: FloatArray


def _as_float32_tensor(values: torch.Tensor | npt.ArrayLike, *, name: str) -> torch.Tensor:
    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    if tensor.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension.")
    tensor = tensor.to(dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values.")
    return tensor


def _validate_square_matrices(matrices: torch.Tensor, *, name: str) -> torch.Tensor:
    if matrices.ndim != 3:
        raise ValueError(f"{name} must have shape [num_sequences, seq_len, seq_len].")
    if matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(f"{name} must contain square matrices.")
    return matrices


def _tensor_to_float64_array(tensor: torch.Tensor) -> FloatArray:
    flattened = tensor.detach().cpu().to(torch.float64).reshape(-1)
    values = np.fromiter(
        (float(item) for item in flattened),
        dtype=np.float64,
        count=int(flattened.numel()),
    )
    return values.reshape(tuple(int(dimension) for dimension in tensor.shape))


def _rfft(values: torch.Tensor) -> torch.Tensor:
    return _TORCH_FFT.rfft(values)


def _irfft(values: torch.Tensor, *, n: int) -> torch.Tensor:
    return _TORCH_FFT.irfft(values, n=n)


def is_rms_norm(norm_name: str | None) -> bool:
    return bool(norm_name) and norm_name.lower().startswith("rms")


def double_center_matrix(matrix: torch.Tensor) -> torch.Tensor:
    matrix = _as_float32_tensor(matrix, name="matrix")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square with shape [seq_len, seq_len].")
    row_mean = matrix.mean(dim=-1, keepdim=True)
    col_mean = matrix.mean(dim=-2, keepdim=True)
    global_mean = matrix.mean()
    centered = matrix - row_mean - col_mean + global_mean
    if not torch.isfinite(centered).all():
        raise ValueError("double-centered matrix contains non-finite values.")
    return centered


def prepare_source_matrix(matrix: torch.Tensor, *, norm_name: str | None) -> torch.Tensor:
    matrix = _as_float32_tensor(matrix, name="matrix")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square with shape [seq_len, seq_len].")
    return double_center_matrix(matrix) if is_rms_norm(norm_name) else matrix


def lower_diagonal_means(
    matrix: torch.Tensor | npt.ArrayLike,
    *,
    offset_start: int,
    offset_stop: int,
) -> torch.Tensor:
    square = _as_float32_tensor(matrix, name="matrix")
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        raise ValueError("matrix must be square with shape [seq_len, seq_len].")
    seq_len = square.shape[0]
    if offset_start < 0:
        raise ValueError("offset_start must be non-negative.")
    if offset_stop < offset_start:
        raise ValueError("offset_stop must be >= offset_start.")
    if offset_stop >= seq_len:
        raise ValueError(
            f"offset_stop={offset_stop} is out of range for seq_len={seq_len}; "
            "missing offsets are a hard failure."
        )
    means: list[torch.Tensor] = []
    for offset in range(offset_start, offset_stop + 1):
        diagonal = torch.diagonal(square, offset=-offset)
        if diagonal.numel() == 0:
            raise ValueError(f"offset {offset} has no causal pairs.")
        if not torch.isfinite(diagonal).all():
            raise ValueError(f"offset {offset} contains non-finite values.")
        means.append(diagonal.mean())
    result = torch.stack(means)
    if not torch.isfinite(result).all():
        raise ValueError("lower diagonal means contain non-finite values.")
    return result


def fft_topk_smooth(
    means: torch.Tensor | npt.ArrayLike,
    *,
    keep_non_dc: int = TOP_NON_DC_COMPONENTS,
) -> torch.Tensor:
    values = _as_float32_tensor(means, name="means")
    if values.ndim != 1:
        raise ValueError("means must have shape [num_offsets].")
    if values.numel() == 0:
        raise ValueError("means must be non-empty.")
    freq = _rfft(values)
    num_freq = int(freq.shape[0])
    if num_freq > 1 and keep_non_dc > 0:
        keep = min(int(keep_non_dc), num_freq - 1)
        keep_idx = torch.topk(torch.abs(freq[1:]), k=keep).indices + 1
        mask = torch.zeros_like(freq, dtype=torch.bool)
        mask[0] = True
        mask[keep_idx] = True
        freq = torch.where(mask, freq, torch.zeros_like(freq))
    smoothed = _irfft(freq, n=int(values.shape[0])).to(torch.float32)
    if not torch.isfinite(smoothed).all():
        raise ValueError("FFT-smoothed means contain non-finite values.")
    return smoothed


def pair_weighted_r2(
    matrix: torch.Tensor | npt.ArrayLike,
    smoothed_means: torch.Tensor | npt.ArrayLike,
    *,
    offset_start: int = SOURCE_OFFSET_START,
) -> float:
    square = _as_float32_tensor(matrix, name="matrix")
    targets = _as_float32_tensor(smoothed_means, name="smoothed_means")
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        raise ValueError("matrix must be square with shape [seq_len, seq_len].")
    if targets.ndim != 1:
        raise ValueError("smoothed_means must have shape [num_offsets].")
    diagonals: list[torch.Tensor] = []
    sse = 0.0
    for index, offset in enumerate(range(offset_start, offset_start + targets.numel())):
        diagonal = torch.diagonal(square, offset=-offset)
        if diagonal.numel() == 0:
            raise ValueError(f"offset {offset} has no causal pairs.")
        diagonals.append(diagonal)
        diff = diagonal - targets[index]
        sse += float(torch.sum(diff * diff).item())
    all_pairs = torch.cat(diagonals)
    if not torch.isfinite(all_pairs).all():
        raise ValueError("matrix contains non-finite causal pairs.")
    grand_mean = all_pairs.mean()
    centered = all_pairs - grand_mean
    sst = float(torch.sum(centered * centered).item())
    if sst == 0.0:
        return 1.0
    r2 = max(0.0, 1.0 - (sse / sst))
    if not np.isfinite(r2):
        raise ValueError("R-squared is non-finite.")
    return float(r2)


def estimate_source_r2_for_sequence(
    matrix: torch.Tensor | npt.ArrayLike,
    *,
    norm_name: str | None,
    keep_non_dc: int = TOP_NON_DC_COMPONENTS,
) -> tuple[float, FloatArray]:
    prepared = prepare_source_matrix(_as_float32_tensor(matrix, name="matrix"), norm_name=norm_name)
    means = lower_diagonal_means(
        prepared,
        offset_start=SOURCE_OFFSET_START,
        offset_stop=SOURCE_OFFSET_STOP,
    )
    smoothed = fft_topk_smooth(means, keep_non_dc=keep_non_dc)
    r2 = pair_weighted_r2(prepared, smoothed, offset_start=SOURCE_OFFSET_START)
    return r2, _tensor_to_float64_array(smoothed)


def estimate_source_scores(
    matrices: torch.Tensor | npt.ArrayLike,
    *,
    norm_name: str | None,
    keep_non_dc: int = TOP_NON_DC_COMPONENTS,
) -> SourceScoreResult:
    tensor = _validate_square_matrices(
        _as_float32_tensor(matrices, name="matrices"), name="matrices"
    )
    per_sequence: list[float] = []
    smoothed: list[FloatArray] = []
    for sequence_index, matrix in enumerate(tensor):
        r2, smoothed_means = estimate_source_r2_for_sequence(
            matrix,
            norm_name=norm_name,
            keep_non_dc=keep_non_dc,
        )
        if not np.isfinite(r2):
            raise ValueError(f"sequence {sequence_index} produced a non-finite R-squared.")
        per_sequence.append(r2)
        smoothed.append(smoothed_means)
    per_sequence_r2 = np.asarray(per_sequence, dtype=np.float64)
    if not np.isfinite(per_sequence_r2).all():
        raise ValueError("per-sequence source R-squared values contain non-finite entries.")
    return SourceScoreResult(
        per_sequence_r2=per_sequence_r2,
        mean_r2=float(per_sequence_r2.mean()),
        smoothed_means=np.stack(smoothed, axis=0),
    )


def estimate_raw_intervention_kernel(
    matrices: torch.Tensor | npt.ArrayLike,
) -> FloatArray:
    tensor = _validate_square_matrices(
        _as_float32_tensor(matrices, name="matrices"), name="matrices"
    )
    per_sequence: list[FloatArray] = []
    for sequence_index, matrix in enumerate(tensor):
        means = lower_diagonal_means(
            matrix,
            offset_start=RAW_OFFSET_START,
            offset_stop=RAW_OFFSET_STOP,
        )
        if not torch.isfinite(means).all():
            raise ValueError(f"sequence {sequence_index} produced non-finite raw kernel means.")
        per_sequence.append(_tensor_to_float64_array(means))
    stacked = np.stack(per_sequence, axis=0).astype(np.float64, copy=False)
    kernel: FloatArray = stacked.mean(axis=0)
    if not np.isfinite(kernel).all():
        raise ValueError("raw intervention kernel contains non-finite values.")
    return kernel


def estimate_source_scores_by_head(
    matrices: torch.Tensor | npt.ArrayLike,
    *,
    norm_name: str | None,
    keep_non_dc: int = TOP_NON_DC_COMPONENTS,
) -> tuple[FloatArray, FloatArray]:
    tensor = _as_float32_tensor(matrices, name="matrices")
    if tensor.ndim != 4:
        raise ValueError("matrices must have shape [num_sequences, num_heads, seq_len, seq_len].")
    if tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError("matrices must contain square head matrices.")
    num_sequences, num_heads = tensor.shape[:2]
    per_sequence = np.zeros((num_sequences, num_heads), dtype=np.float64)
    mean_scores = np.zeros(num_heads, dtype=np.float64)
    for head_index in range(num_heads):
        result = estimate_source_scores(
            tensor[:, head_index],
            norm_name=norm_name,
            keep_non_dc=keep_non_dc,
        )
        per_sequence[:, head_index] = result.per_sequence_r2
        mean_scores[head_index] = result.mean_r2
    return per_sequence, mean_scores


def estimate_raw_intervention_kernels_by_head(
    matrices: torch.Tensor | npt.ArrayLike,
) -> FloatArray:
    tensor = _as_float32_tensor(matrices, name="matrices")
    if tensor.ndim != 4:
        raise ValueError("matrices must have shape [num_sequences, num_heads, seq_len, seq_len].")
    if tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError("matrices must contain square head matrices.")
    num_heads = tensor.shape[1]
    kernels: list[FloatArray] = []
    for head_index in range(num_heads):
        kernels.append(estimate_raw_intervention_kernel(tensor[:, head_index]))
    return np.stack(kernels, axis=0)


def sort_heads_into_bins(
    mean_r2_by_head: torch.Tensor | npt.ArrayLike,
    *,
    num_bins: int = NUM_HEAD_BINS,
) -> list[list[HeadIndex]]:
    scores = np.asarray(mean_r2_by_head, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("mean_r2_by_head must have shape [num_layers, num_heads].")
    if not np.isfinite(scores).all():
        raise ValueError("mean_r2_by_head contains non-finite values.")
    num_layers, num_heads = scores.shape
    total_heads = int(num_layers * num_heads)
    if total_heads < num_bins:
        raise ValueError(f"need at least {num_bins} heads, found {total_heads}.")
    ordered = sorted(
        (
            (float(scores[layer, head]), layer, head)
            for layer in range(num_layers)
            for head in range(num_heads)
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    split_indices = np.array_split(np.arange(len(ordered), dtype=np.int64), num_bins)
    bins: list[list[HeadIndex]] = []
    for chunk_indices in split_indices:
        if chunk_indices.size == 0:
            raise ValueError("numpy.array_split produced an empty head bin.")
        bins.append(
            [
                HeadIndex(layer=ordered[int(index)][1], head=ordered[int(index)][2])
                for index in chunk_indices
            ]
        )
    return bins


def bin_mean_source_r2(
    mean_r2_by_head: torch.Tensor | npt.ArrayLike,
    bins: Sequence[Sequence[HeadIndex]],
) -> FloatArray:
    scores = np.asarray(mean_r2_by_head, dtype=np.float64)
    values: list[float] = []
    for head_group in bins:
        if not head_group:
            raise ValueError("head_group must be non-empty.")
        group_values: list[float] = []
        for head_index in head_group:
            group_values.append(float(scores[head_index.layer, head_index.head]))
        values.append(float(np.mean(group_values)))
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("bin mean source R-squared values contain non-finite entries.")
    return result


def build_causal_toeplitz_correction(
    kernel: torch.Tensor | Sequence[float],
    *,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    values = torch.as_tensor(kernel, dtype=torch.float32, device=device)
    if values.ndim != 1:
        raise ValueError(f"Kernel must be rank-1, got shape={tuple(values.shape)}.")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}.")
    if values.numel() < sequence_length:
        raise ValueError(
            "Kernel length must cover every causal offset: "
            f"len={values.numel()} seq={sequence_length}."
        )
    if not torch.isfinite(values[:sequence_length]).all():
        raise ValueError("Kernel contains non-finite values.")
    offsets = torch.arange(sequence_length, device=device)
    delta = offsets[:, None] - offsets[None, :]
    correction = torch.zeros((sequence_length, sequence_length), dtype=torch.float32, device=device)
    lower = delta >= 0
    correction[lower] = -values.index_select(0, delta[lower])
    return correction
