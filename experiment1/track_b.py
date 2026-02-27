from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch

from shared.specs import DatasetSpec, ModelSpec, SequenceLengthSpec
from shared.utils.logging import get_logger

from experiment1.paths import tokenized_path, track_a_dir, track_b_dir
from experiment1.norm_utils import normalize_logits_for_norm
from experiment1.shift_kernels import BaseEstimator, KernelFit, get_kernel_estimator

LOGGER = get_logger("experiment1.track_b")
CENTERING_LEGACY_PER_POSITION = "legacy_per_position"
CENTERING_CANONICAL_PER_POSITION = "canonical_per_position"


@dataclass
class SequenceRecord:
    seq_id: int
    tokens: list[int]
    split: str


@dataclass
class TrackBConfig:
    data_root: Path
    results_root: Path
    limit_eval_sequences: int | None = None
    limit_center_sequences: int | None = None
    device: str = "cpu"
    full_matrix_max_length: int = 256
    centering_mode: str = CENTERING_LEGACY_PER_POSITION
    output_group: str = "track_b"


class TrackBRunner:
    def __init__(
        self,
        model: ModelSpec,
        dataset: DatasetSpec,
        seq_len: SequenceLengthSpec,
        config: TrackBConfig,
    ) -> None:
        self.model_spec = model
        self.dataset_spec = dataset
        self.seq_len = seq_len
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = track_b_dir(config.results_root, model, dataset, seq_len, group=config.output_group)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.store_full = seq_len.tokens <= config.full_matrix_max_length
        self.q_mean: torch.Tensor | None = None
        self.k_mean: torch.Tensor | None = None
        self.kernel_estimator = get_kernel_estimator(model.pe_scheme)
        self.centering_frame = "none"
        self.rope_canonical_applied = False

    def run(self) -> None:
        jsonl_path = tokenized_path(
            self.config.data_root,
            self.model_spec,
            self.dataset_spec,
            self.seq_len,
        )
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Tokenized data missing for {jsonl_path}")
        from shared.models.loading import load_model
        from shared.attention import get_adapter

        loader = load_model(self.model_spec)
        model = loader.model.to(self.device)
        adapter = get_adapter(self.model_spec)
        adapter.register(model)
        if self.dataset_spec.needs_center_split:
            LOGGER.info(
                "Track B centering start model=%s dataset=%s len=%s mode=%s",
                self.model_spec.name,
                self.dataset_spec.name,
                self.seq_len.tokens,
                self.config.centering_mode,
            )
            self._compute_centering_means(jsonl_path, model, adapter)
        else:
            LOGGER.info("Track B centering skipped for %s (synthetic dataset)", self.dataset_spec.name)
        LOGGER.info(
            "Track B eval start model=%s dataset=%s len=%s",
            self.model_spec.name,
            self.dataset_spec.name,
            self.seq_len.tokens,
        )
        stats = self._process_evaluation(jsonl_path, model, adapter)
        adapter.cleanup()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._write_outputs(stats)
        LOGGER.info(
            "Track B finished model=%s dataset=%s len=%s eval_sequences=%s",
            self.model_spec.name,
            self.dataset_spec.name,
            self.seq_len.tokens,
            stats.eval_sequences,
        )

    def _compute_centering_means(self, jsonl_path: Path, model, adapter) -> None:
        sequences = _iter_sequences(jsonl_path, split="centering", limit=self.config.limit_center_sequences)
        accum_q = None
        accum_k = None
        count = 0
        for record in sequences:
            count += 1
            inputs = torch.tensor(record.tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            attention_mask = torch.ones_like(inputs, device=self.device)
            capture = adapter.capture(model, input_ids=inputs, attention_mask=attention_mask)
            q = capture.q.to(torch.float64)
            k = capture.k.to(torch.float64)
            if accum_q is None:
                accum_q = torch.zeros_like(q)
                accum_k = torch.zeros_like(k)
            if self.config.centering_mode == CENTERING_CANONICAL_PER_POSITION:
                q, k, applied = _canonicalize_capture_qk(capture, q, k)
                self.rope_canonical_applied = self.rope_canonical_applied or applied
            accum_q += q
            accum_k += k
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        if count == 0:
            raise RuntimeError("No centering sequences processed; unable to compute Track B means.")
        self.q_mean = (accum_q / count).to(torch.float32).cpu()
        self.k_mean = (accum_k / count).to(torch.float32).cpu()
        self.centering_frame = "canonical" if self.rope_canonical_applied else "captured"
        torch.save(
            {
                "q_mean": self.q_mean,
                "k_mean": self.k_mean,
                "count": count,
                "centering_mode": self.config.centering_mode,
                "frame": self.centering_frame,
                "per_position": True,
            },
            self.output_dir / "centering_means.pt",
        )

    def _process_evaluation(self, jsonl_path: Path, model, adapter) -> "TrackBStats":
        sequences = _iter_sequences(jsonl_path, split="eval", limit=self.config.limit_eval_sequences)
        accumulator: GramAccumulator | None = None
        eval_sequences = 0
        for record in sequences:
            eval_sequences += 1
            inputs = torch.tensor(record.tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            attention_mask = torch.ones_like(inputs, device=self.device)
            capture = adapter.capture(model, input_ids=inputs, attention_mask=attention_mask)
            if accumulator is None:
                accumulator = GramAccumulator(
                    num_layers=capture.q.shape[0],
                    num_heads=capture.q.shape[1],
                    seq_len=self.seq_len.tokens,
                    head_dim=capture.q.shape[-1],
                    store_full=self.store_full,
                    estimator=self.kernel_estimator,
                    norm=self.model_spec.norm,
                    centering_mode=self.config.centering_mode,
                )
            accumulator.update(capture, self.q_mean, self.k_mean)
            del capture
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        if accumulator is None:
            raise RuntimeError("No evaluation sequences processed for Track B.")
        accumulator.finalize(eval_sequences)
        return TrackBStats(eval_sequences=eval_sequences, accumulator=accumulator)

    def _write_outputs(self, stats: "TrackBStats") -> None:
        summary_rows = []
        for layer_idx in range(stats.accumulator.num_layers):
            for head_idx in range(stats.accumulator.num_heads):
                centered_fit = stats.accumulator.centered_stats[layer_idx][head_idx].fit()
                raw_fit = stats.accumulator.raw_stats[layer_idx][head_idx].fit()
                summary_rows.append(
                    {
                        "model": self.model_spec.name,
                        "dataset": self.dataset_spec.name,
                        "seq_len": self.seq_len.tokens,
                        "layer": layer_idx,
                        "head": head_idx,
                        "r2_centered": centered_fit.r2,
                        "r2_raw": raw_fit.r2,
                        "g_centered": centered_fit.g_values,
                        "g_raw": raw_fit.g_values,
                        "sequences": stats.eval_sequences,
                    }
                )
        summary_df = pd.DataFrame(summary_rows)
        summary_df = _merge_track_a(summary_df, self.model_spec, self.dataset_spec, self.seq_len, self.config.results_root)
        summary_df.to_parquet(self.output_dir / "summary.parquet", engine="pyarrow", index=False)
        with (self.output_dir / "track_b_run.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model": self.model_spec.name,
                    "dataset": self.dataset_spec.name,
                    "seq_len": self.seq_len.tokens,
                    "centering_mode": self.config.centering_mode,
                    "output_group": self.config.output_group,
                    "rope_canonical_applied": bool(self.rope_canonical_applied),
                    "notes": (
                        "Diagnostic canonical-frame per-position centering ablation"
                        if self.config.centering_mode == CENTERING_CANONICAL_PER_POSITION
                        else "Legacy Track B centering behavior"
                    ),
                },
                handle,
                indent=2,
            )
        if stats.accumulator.full_centered is not None:
            torch.save(
                {
                    "gram_centered": stats.accumulator.full_centered.to(torch.float32),
                    "gram_raw": stats.accumulator.full_raw.to(torch.float32),
                    "eval_sequences": stats.eval_sequences,
                },
                self.output_dir / "gram_matrices.pt",
            )


@dataclass
class TrackBStats:
    eval_sequences: int
    accumulator: "GramAccumulator"


class GramAccumulator:
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        store_full: bool,
        estimator: BaseEstimator,
        norm: str | None,
        centering_mode: str,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.store_full = store_full
        self.norm = norm
        self.estimator = estimator
        self.centering_mode = centering_mode
        self.centered_stats = [
            [DiagonalStats(seq_len, estimator) for _ in range(num_heads)] for _ in range(num_layers)
        ]
        self.raw_stats = [
            [DiagonalStats(seq_len, estimator) for _ in range(num_heads)] for _ in range(num_layers)
        ]
        self.full_centered = (
            torch.zeros(num_layers, num_heads, seq_len, seq_len, dtype=torch.float64) if store_full else None
        )
        self.full_raw = (
            torch.zeros(num_layers, num_heads, seq_len, seq_len, dtype=torch.float64) if store_full else None
        )

    def update(
        self,
        capture,
        q_mean: torch.Tensor | None,
        k_mean: torch.Tensor | None,
    ) -> None:
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                q = capture.q[layer_idx, head_idx].to(torch.float64)
                k = capture.k[layer_idx, head_idx].to(torch.float64)
                raw_matrix = torch.matmul(q, k.transpose(0, 1)) * self.scale
                raw_matrix = normalize_logits_for_norm(raw_matrix, self.norm)
                if q_mean is not None and k_mean is not None:
                    mean_q = q_mean[layer_idx, head_idx].to(q.device)
                    mean_k = k_mean[layer_idx, head_idx].to(k.device)
                    if (
                        self.centering_mode == CENTERING_CANONICAL_PER_POSITION
                        and getattr(capture, "rope_cos", None) is not None
                        and getattr(capture, "rope_sin", None) is not None
                    ):
                        cos = capture.rope_cos[layer_idx]
                        sin = capture.rope_sin[layer_idx]
                        q_canon = invert_rope_heads(q, cos, sin)
                        k_canon = invert_rope_heads(k, cos, sin)
                        centered_q = apply_rope_heads(q_canon - mean_q, cos, sin)
                        centered_k = apply_rope_heads(k_canon - mean_k, cos, sin)
                    else:
                        centered_q = q - mean_q
                        centered_k = k - mean_k
                else:
                    centered_q = q
                    centered_k = k
                centered_matrix = torch.matmul(centered_q, centered_k.transpose(0, 1)) * self.scale
                centered_matrix = normalize_logits_for_norm(centered_matrix, self.norm)
                self.raw_stats[layer_idx][head_idx].update(raw_matrix)
                self.centered_stats[layer_idx][head_idx].update(centered_matrix)
                if self.store_full:
                    self.full_raw[layer_idx, head_idx] += raw_matrix
                    self.full_centered[layer_idx, head_idx] += centered_matrix

    def finalize(self, eval_sequences: int) -> None:
        if self.store_full and eval_sequences > 0:
            self.full_raw /= eval_sequences
            self.full_centered /= eval_sequences


class DiagonalStats:
    def __init__(self, seq_len: int, estimator: BaseEstimator) -> None:
        self.seq_len = seq_len
        self.sum = torch.zeros(seq_len - 1, dtype=torch.float64)
        self.sq_sum = torch.zeros(seq_len - 1, dtype=torch.float64)
        self.count = torch.zeros(seq_len - 1, dtype=torch.float64)
        self.estimator = estimator

    def update(self, matrix: torch.Tensor) -> None:
        for delta in range(1, self.seq_len):
            diag = torch.diagonal(matrix, offset=-delta)
            if diag.numel() == 0:
                continue
            idx = delta - 1
            diag = diag.to(torch.float64)
            self.sum[idx] += diag.sum()
            self.sq_sum[idx] += torch.sum(diag * diag)
            self.count[idx] += diag.numel()

    def fit(self) -> KernelFit:
        return self.estimator.fit_from_stats(sums=self.sum, sq_sums=self.sq_sum, counts=self.count)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _prepare_rope_terms(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos_t = cos.to(device=x.device, dtype=x.dtype)
    sin_t = sin.to(device=x.device, dtype=x.dtype)
    while cos_t.dim() > x.dim() and cos_t.shape[0] == 1:
        cos_t = cos_t.squeeze(0)
    while sin_t.dim() > x.dim() and sin_t.shape[0] == 1:
        sin_t = sin_t.squeeze(0)
    while cos_t.dim() < x.dim():
        cos_t = cos_t.unsqueeze(0)
    while sin_t.dim() < x.dim():
        sin_t = sin_t.unsqueeze(0)
    return cos_t, sin_t


def apply_rope_heads(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos_t, sin_t = _prepare_rope_terms(x, cos, sin)
    return (x * cos_t) + (rotate_half(x) * sin_t)


def invert_rope_heads(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos_t, sin_t = _prepare_rope_terms(x, cos, sin)
    return (x * cos_t) - (rotate_half(x) * sin_t)


def _canonicalize_capture_qk(
    capture,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    rope_cos = getattr(capture, "rope_cos", None)
    rope_sin = getattr(capture, "rope_sin", None)
    if rope_cos is None or rope_sin is None:
        return q, k, False
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    for layer_idx in range(q.shape[0]):
        cos = rope_cos[layer_idx]
        sin = rope_sin[layer_idx]
        q_out[layer_idx] = invert_rope_heads(q[layer_idx], cos, sin)
        k_out[layer_idx] = invert_rope_heads(k[layer_idx], cos, sin)
    return q_out, k_out, True


def _iter_sequences(jsonl_path: Path, split: str, limit: int | None) -> Iterator[SequenceRecord]:
    seen = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != split:
                continue
            yield SequenceRecord(seq_id=row["id"], tokens=row["tokens"], split=row["split"])
            seen += 1
            if limit is not None and seen >= limit:
                break


def _merge_track_a(
    summary_df: pd.DataFrame,
    model: ModelSpec,
    dataset: DatasetSpec,
    seq_len: SequenceLengthSpec,
    results_root: Path,
) -> pd.DataFrame:
    track_a_path = track_a_dir(results_root, model, dataset, seq_len) / "summary.parquet"
    if not track_a_path.exists():
        return summary_df
    track_a_df = pd.read_parquet(track_a_path)[["layer", "head", "mean_r2"]]
    track_a_df = track_a_df.rename(columns={"mean_r2": "track_a_mean_r2"})
    merged = summary_df.merge(track_a_df, on=["layer", "head"], how="left")
    if "track_a_mean_r2" in merged.columns:
        merged["track_a_diff"] = (merged["track_a_mean_r2"] - merged["r2_centered"]).abs()
    else:
        merged["track_a_diff"] = None
    return merged
