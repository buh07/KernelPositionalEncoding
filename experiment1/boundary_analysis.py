from __future__ import annotations

"""Boundary analysis: compare R² on early-sequence vs interior positions.

Tests the causal masking × content-position entanglement hypothesis by computing
shift-invariance R² restricted to different position windows within the logit
matrix.  If causal boundary effects depress R², interior-only R² should be
substantially higher than whole-sequence R².

Windows:
  - full:      all lower-triangular entries (same as Track A)
  - boundary:  only entries where query position t < boundary_cutoff
  - interior:  only entries where both t >= interior_start AND s >= interior_start
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch

from shared.specs import DatasetSpec, ModelSpec, SequenceLengthSpec
from shared.utils.logging import get_logger

from experiment1.norm_utils import normalize_logits_for_norm
from experiment1.paths import tokenized_path
from experiment1.shift_kernels import get_kernel_estimator

LOGGER = get_logger("experiment1.boundary")

BOUNDARY_CUTOFF = 50   # positions 0..49 are "boundary"
INTERIOR_START = 50    # positions 50+ are "interior"


@dataclass
class BoundaryConfig:
    data_root: Path
    results_root: Path
    limit_sequences: int | None = None
    device: str = "cpu"
    boundary_cutoff: int = BOUNDARY_CUTOFF
    interior_start: int = INTERIOR_START


def _fit_windowed(logits: torch.Tensor, min_pos: int, max_pos: int | None, estimator) -> dict:
    """Compute R² using only entries where both t and s are within [min_pos, max_pos).

    For the lower-triangular restriction (s < t), we consider entries
    A(t, s) where min_pos <= s < t and t < max_pos.
    """
    T = logits.size(-1)
    if max_pos is None:
        max_pos = T

    diag_sums: list[float] = []
    diag_sq_sums: list[float] = []
    diag_counts: list[int] = []

    max_delta = max_pos - min_pos
    for delta in range(1, max_delta):
        # Diagonal offset=-delta gives entries (t, t-delta) for t = delta..T-1
        diag = torch.diagonal(logits, offset=-delta).to(torch.float64)
        # Restrict: t >= min_pos + delta (so s = t-delta >= min_pos) AND t < max_pos
        t_start = max(delta, min_pos + delta)
        t_end = min(T, max_pos)
        if t_start >= t_end:
            diag_sums.append(0.0)
            diag_sq_sums.append(0.0)
            diag_counts.append(0)
            continue
        # diag indices correspond to t values: diag[i] = logits[i + delta, i]
        # so diag[t - delta] = logits[t, t - delta] = A(t, s=t-delta)
        idx_start = t_start - delta
        idx_end = t_end - delta
        window = diag[idx_start:idx_end]
        diag_sums.append(window.sum().item())
        diag_sq_sums.append((window ** 2).sum().item())
        diag_counts.append(window.numel())

    if not diag_counts or sum(diag_counts) == 0:
        return {"r2": 0.0, "n_entries": 0}

    sums_t = torch.tensor(diag_sums, dtype=torch.float64)
    sq_sums_t = torch.tensor(diag_sq_sums, dtype=torch.float64)
    counts_t = torch.tensor(diag_counts, dtype=torch.float64)

    # Compute g(delta) = mean of entries at each lag
    safe_counts = torch.where(counts_t == 0, torch.ones_like(counts_t), counts_t)
    means = torch.where(counts_t == 0, torch.zeros_like(sums_t), sums_t / safe_counts)
    targets = estimator.transform(means)

    # SSE = sum over deltas of sum_t (A(t, t-delta) - g(delta))^2
    sse_terms = sq_sums_t - 2 * targets * sums_t + counts_t * (targets ** 2)
    sse = torch.sum(torch.where(counts_t == 0, torch.zeros_like(sse_terms), sse_terms)).item()

    total_sum = torch.sum(sums_t).item()
    total_sq_sum = torch.sum(sq_sums_t).item()
    total_count = torch.sum(counts_t).item()

    if total_count == 0:
        return {"r2": 0.0, "n_entries": 0}

    global_mean = total_sum / total_count
    sst = total_sq_sum - total_count * (global_mean ** 2)
    r2 = 1.0 if sst <= 0 else max(0.0, 1.0 - (sse / sst))

    return {"r2": float(r2), "n_entries": int(total_count)}


class BoundaryRunner:
    def __init__(
        self,
        model: ModelSpec,
        dataset: DatasetSpec,
        seq_len: SequenceLengthSpec,
        config: BoundaryConfig,
    ) -> None:
        self.model_spec = model
        self.dataset_spec = dataset
        self.seq_len = seq_len
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = (
            config.results_root / "boundary" / model.name / dataset.name / f"len_{seq_len.tokens}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.estimator = get_kernel_estimator(model.pe_scheme)
        self.rows: list[dict] = []

    def run(self) -> None:
        jsonl_path = tokenized_path(
            self.config.data_root, self.model_spec, self.dataset_spec, self.seq_len,
        )
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Tokenized data missing: {jsonl_path}")

        manifest = _load_manifest(jsonl_path)
        LOGGER.info(
            "Boundary analysis start model=%s dataset=%s len=%s eval_sequences=%s",
            self.model_spec.name, self.dataset_spec.name, self.seq_len.tokens,
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
            self._ingest(record.seq_id, capture)
            if processed % 5 == 0:
                LOGGER.info(
                    "Boundary progress model=%s dataset=%s len=%s processed=%s",
                    self.model_spec.name, self.dataset_spec.name,
                    self.seq_len.tokens, processed,
                )
            del capture
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        adapter.cleanup()
        self._write(processed)
        LOGGER.info(
            "Boundary analysis finished model=%s dataset=%s len=%s processed=%s",
            self.model_spec.name, self.dataset_spec.name,
            self.seq_len.tokens, processed,
        )

    def _ingest(self, seq_id: int, capture) -> None:
        T = self.seq_len.tokens
        bc = self.config.boundary_cutoff
        ist = self.config.interior_start
        n_layers = capture.logits.shape[0]
        n_heads = capture.logits.shape[1]

        for layer in range(n_layers):
            for head in range(n_heads):
                logits = capture.logits[layer, head]
                prepared = normalize_logits_for_norm(logits, self.model_spec.norm)

                full = _fit_windowed(prepared, 0, None, self.estimator)
                boundary = _fit_windowed(prepared, 0, bc, self.estimator)
                interior = _fit_windowed(prepared, ist, None, self.estimator)

                self.rows.append({
                    "model": self.model_spec.name,
                    "dataset": self.dataset_spec.name,
                    "seq_len": T,
                    "sequence_id": seq_id,
                    "layer": layer,
                    "head": head,
                    "r2_full": full["r2"],
                    "r2_boundary": boundary["r2"],
                    "r2_interior": interior["r2"],
                    "n_full": full["n_entries"],
                    "n_boundary": boundary["n_entries"],
                    "n_interior": interior["n_entries"],
                })

    def _write(self, processed: int) -> None:
        if not self.rows:
            LOGGER.warning("No boundary rows for %s/%s len=%s",
                           self.model_spec.name, self.dataset_spec.name, self.seq_len.tokens)
            return

        df = pd.DataFrame(self.rows)

        # Per-sequence file
        per_seq_path = self.output_dir / "per_sequence.parquet"
        df.to_parquet(per_seq_path, engine="pyarrow", index=False)

        # Summary: aggregate across sequences per (layer, head)
        summary = df.groupby(["model", "dataset", "seq_len", "layer", "head"]).agg(
            mean_r2_full=("r2_full", "mean"),
            mean_r2_boundary=("r2_boundary", "mean"),
            mean_r2_interior=("r2_interior", "mean"),
            std_r2_full=("r2_full", "std"),
            std_r2_boundary=("r2_boundary", "std"),
            std_r2_interior=("r2_interior", "std"),
            sequences=("sequence_id", "nunique"),
        ).reset_index()
        summary_path = self.output_dir / "summary.parquet"
        summary.to_parquet(summary_path, engine="pyarrow", index=False)

        LOGGER.info("Wrote boundary results to %s (%d rows, %d sequences)",
                     self.output_dir, len(df), processed)


# ── helpers ──────────────────────────────────────────────────────────

@dataclass
class _SeqRecord:
    seq_id: int
    tokens: list[int]
    split: str


def _iter_eval_sequences(jsonl_path: Path, limit: int | None) -> Iterator[_SeqRecord]:
    processed = 0
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("split") != "eval":
                continue
            yield _SeqRecord(seq_id=row["id"], tokens=row["tokens"], split=row["split"])
            processed += 1
            if limit is not None and processed >= limit:
                break


def _load_manifest(jsonl_path: Path) -> dict:
    mp = jsonl_path.with_suffix(".manifest.json")
    if not mp.exists():
        return {}
    with mp.open("r", encoding="utf-8") as fh:
        return json.load(fh)
