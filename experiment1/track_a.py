from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch

from shared.specs import DatasetSpec, ModelSpec, SequenceLengthSpec
from shared.utils.logging import get_logger

from experiment1.paths import tokenized_path, track_a_dir
from experiment1.norm_utils import normalize_logits_for_norm
from experiment1.shift_kernels import KernelFit, get_kernel_estimator

LOGGER = get_logger("experiment1.track_a")


@dataclass
class SequenceRecord:
    seq_id: int
    tokens: list[int]
    split: str


@dataclass
class TrackAConfig:
    data_root: Path
    results_root: Path
    limit_sequences: int | None = None
    device: str = "cpu"
    enable_heatmaps: bool = True
    heatmap_dataset: str = "wiki40b_en_pre2019"
    heatmap_seq_len: int = 256
    heatmap_stride: int = 11


class TrackARunner:
    def __init__(
        self,
        model: ModelSpec,
        dataset: DatasetSpec,
        seq_len: SequenceLengthSpec,
        config: TrackAConfig,
    ) -> None:
        self.model_spec = model
        self.dataset_spec = dataset
        self.seq_len = seq_len
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = track_a_dir(config.results_root, model, dataset, seq_len)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_sequence_rows: list[dict] = []
        self.aggregators: dict[tuple[int, int], HeadAggregator] = {}
        self.kernel_estimator = get_kernel_estimator(model.pe_scheme)

    def run(self) -> None:
        jsonl_path = tokenized_path(
            self.config.data_root,
            self.model_spec,
            self.dataset_spec,
            self.seq_len,
        )
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Tokenized data missing for {jsonl_path}")
        manifest = _load_manifest(jsonl_path)
        LOGGER.info(
            "Track A start model=%s dataset=%s len=%s eval_sequences=%s",
            self.model_spec.name,
            self.dataset_spec.name,
            self.seq_len.tokens,
            manifest.get("eval_count"),
        )
        from shared.models.loading import load_model

        loader = load_model(self.model_spec)
        model = loader.model.to(self.device)
        from shared.attention import get_adapter

        adapter = get_adapter(self.model_spec)
        adapter.register(model)
        sequences = _iter_eval_sequences(jsonl_path, self.config.limit_sequences)
        processed = 0
        for record in sequences:
            processed += 1
            inputs = torch.tensor(record.tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            attention_mask = torch.ones_like(inputs, device=self.device)
            capture = adapter.capture(model, input_ids=inputs, attention_mask=attention_mask)
            self._maybe_store_heatmap(record.seq_id, capture)
            self._ingest_capture(record.seq_id, capture)
            if processed % 5 == 0:
                LOGGER.info(
                    "Track A progress model=%s dataset=%s len=%s processed=%s",
                    self.model_spec.name,
                    self.dataset_spec.name,
                    self.seq_len.tokens,
                    processed,
                )
            del capture
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        adapter.cleanup()
        self._write_outputs(processed)
        LOGGER.info(
            "Track A finished model=%s dataset=%s len=%s processed=%s",
            self.model_spec.name,
            self.dataset_spec.name,
            self.seq_len.tokens,
            processed,
        )

    def _maybe_store_heatmap(self, seq_id: int, capture) -> None:
        if not self.config.enable_heatmaps:
            return
        if self.dataset_spec.name != self.config.heatmap_dataset:
            return
        if self.seq_len.tokens != self.config.heatmap_seq_len:
            return
        if seq_id % self.config.heatmap_stride != 0:
            return
        heatmap_dir = self.output_dir / "heatmap"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        path = heatmap_dir / f"seq_{seq_id}.pt"
        torch.save({"logits": capture.logits.to(torch.float16)}, path)

    def _ingest_capture(self, seq_id: int, capture) -> None:
        num_layers = capture.logits.shape[0]
        num_heads = capture.logits.shape[1]
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                head_logits = capture.logits[layer_idx, head_idx]
                prepared = normalize_logits_for_norm(head_logits, self.model_spec.norm)
                fit = self.kernel_estimator.fit_logits(prepared)
                self.per_sequence_rows.append(
                    {
                        "model": self.model_spec.name,
                        "dataset": self.dataset_spec.name,
                        "seq_len": self.seq_len.tokens,
                        "sequence_id": seq_id,
                        "layer": layer_idx,
                        "head": head_idx,
                        "r2_shift": fit.r2,
                        "g_values": fit.g_values,
                    }
                )
                agg = self.aggregators.setdefault(
                    (layer_idx, head_idx),
                    HeadAggregator(self.seq_len.tokens - 1),
                )
                agg.update(fit)

    def _write_outputs(self, processed_sequences: int) -> None:
        if not self.per_sequence_rows:
            LOGGER.warning(
                "No Track A rows generated for %s/%s len=%s",
                self.model_spec.name,
                self.dataset_spec.name,
                self.seq_len.tokens,
            )
            return
        per_sequence_df = pd.DataFrame(self.per_sequence_rows)
        per_sequence_path = self.output_dir / "per_sequence.parquet"
        per_sequence_df.to_parquet(per_sequence_path, engine="pyarrow", index=False)

        summary_rows = []
        for (layer_idx, head_idx), agg in self.aggregators.items():
            summary_rows.append(
                {
                    "model": self.model_spec.name,
                    "dataset": self.dataset_spec.name,
                    "seq_len": self.seq_len.tokens,
                    "layer": layer_idx,
                    "head": head_idx,
                    "mean_r2": agg.mean_r2(),
                    "std_r2": agg.std_r2(),
                    "pooled_r2": agg.pooled_r2(),
                    "sequences": agg.count,
                    "mean_g": agg.mean_g(),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        summary_path = self.output_dir / "summary.parquet"
        summary_df.to_parquet(summary_path, engine="pyarrow", index=False)


class HeadAggregator:
    def __init__(self, g_length: int) -> None:
        self.count = 0
        self.g_sum = torch.zeros(g_length, dtype=torch.float64)
        self.r2_values: list[float] = []
        self.sse_total = 0.0
        self.sst_total = 0.0

    def update(self, fit: KernelFit) -> None:
        self.count += 1
        self.r2_values.append(fit.r2)
        self.g_sum += torch.tensor(fit.g_values, dtype=torch.float64)
        self.sse_total += fit.sse
        self.sst_total += fit.sst

    def mean_r2(self) -> float:
        return float(sum(self.r2_values) / max(self.count, 1))

    def std_r2(self) -> float:
        if self.count <= 1:
            return 0.0
        return float(statistics.pstdev(self.r2_values))

    def pooled_r2(self) -> float:
        if self.sst_total == 0:
            return 1.0
        return max(0.0, 1.0 - (self.sse_total / self.sst_total))

    def mean_g(self) -> list[float]:
        if self.count == 0:
            return []
        return (self.g_sum / self.count).tolist()


def _iter_eval_sequences(jsonl_path: Path, limit: int | None) -> Iterator[SequenceRecord]:
    processed = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "eval":
                continue
            yield SequenceRecord(seq_id=row["id"], tokens=row["tokens"], split=row["split"])
            processed += 1
            if limit is not None and processed >= limit:
                break


def _load_manifest(jsonl_path: Path) -> dict:
    manifest_path = jsonl_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _shift_metrics(logits: torch.Tensor) -> tuple[list[float], float, float, float]:
    estimator = get_kernel_estimator("learned-absolute")
    fit = estimator.fit_logits(logits)
    return fit.g_values, fit.r2, fit.sse, fit.sst
