from __future__ import annotations

import json
import math
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from transformers import AutoConfig

from shared.config import default_paths
from shared.specs import DatasetSpec, ModelSpec, SequenceLengthSpec
from shared.utils.logging import get_logger

from experiment1.paths import track_a_dir, track_b_dir, spectral_dir

LOGGER = get_logger("experiment1.spectral")


@dataclass
class SpectralConfig:
    results_root: Path
    pad_multiplier: int = 4
    top_k: int = 5
    gate_threshold: float = 0.60
    track_b_group: str = "track_b"
    output_group: str = "spectral"


class SpectralRunner:
    def __init__(
        self,
        model: ModelSpec,
        dataset: DatasetSpec,
        seq_len: SequenceLengthSpec,
        config: SpectralConfig,
    ) -> None:
        self.model_spec = model
        self.dataset_spec = dataset
        self.seq_len = seq_len
        self.config = config
        self.output_dir = spectral_dir(
            config.results_root,
            model,
            dataset,
            seq_len,
            group=config.output_group,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        track_a = self._load_track_a()
        if track_a is None:
            LOGGER.warning(
                "Spectral skip (missing Track A summary) model=%s dataset=%s len=%s",
                self.model_spec.name,
                self.dataset_spec.name,
                self.seq_len.tokens,
            )
            return
        gate_value = self._gate_value(track_a)
        if gate_value is None or gate_value < self.config.gate_threshold:
            LOGGER.info(
                "Spectral gate blocked model=%s dataset=%s len=%s gate=%.3f",
                self.model_spec.name,
                self.dataset_spec.name,
                self.seq_len.tokens,
                gate_value or 0.0,
            )
            self._write_metadata(gate_value, False)
            return
        track_b = self._load_track_b()
        if track_b is None:
            LOGGER.warning(
                "Spectral skip (missing Track B summary) model=%s dataset=%s len=%s",
                self.model_spec.name,
                self.dataset_spec.name,
                self.seq_len.tokens,
            )
            return
        head_dim = _infer_head_dim(self.model_spec)
        expected_freqs = _expected_frequencies(self.model_spec, head_dim)
        rows: list[dict[str, Any]] = []
        for _, row in track_b.iterrows():
            g_values = row["g_centered"]
            analysis = _analyze_spectrum(
                g_values=g_values,
                pad_multiplier=self.config.pad_multiplier,
                top_k=self.config.top_k,
                expected_freqs=expected_freqs,
            )
            rows.append(
                {
                    "model": self.model_spec.name,
                    "dataset": self.dataset_spec.name,
                    "seq_len": self.seq_len.tokens,
                    "layer": row["layer"],
                    "head": row["head"],
                    "top_frequencies": analysis.top_frequencies,
                    "top_powers": analysis.top_powers,
                    "matched_count": analysis.matched_count,
                    "mean_relative_error": analysis.mean_relative_error,
                    "matches": analysis.matches,
                }
            )
        df = pd.DataFrame(rows)
        df.to_parquet(self.output_dir / "spectral.parquet", engine="pyarrow", index=False)
        self._write_metadata(gate_value, True)
        LOGGER.info(
            "Spectral analysis complete model=%s dataset=%s len=%s rows=%s",
            self.model_spec.name,
            self.dataset_spec.name,
            self.seq_len.tokens,
            len(rows),
        )

    def _load_track_a(self) -> pd.DataFrame | None:
        path = track_a_dir(self.config.results_root, self.model_spec, self.dataset_spec, self.seq_len) / "summary.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _load_track_b(self) -> pd.DataFrame | None:
        path = track_b_dir(
            self.config.results_root,
            self.model_spec,
            self.dataset_spec,
            self.seq_len,
            group=self.config.track_b_group,
        ) / "summary.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _gate_value(self, track_a: pd.DataFrame) -> float | None:
        early = track_a[track_a["layer"].isin([0, 1])]
        if early.empty:
            return None
        return float(early["mean_r2"].mean())

    def _write_metadata(self, gate_value: float | None, ran: bool) -> None:
        meta = {
            "gate_threshold": self.config.gate_threshold,
            "gate_value": gate_value,
            "ran": ran,
        }
        with (self.output_dir / "spectral_run.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)


@dataclass
class SpectralAnalysis:
    top_frequencies: list[float]
    top_powers: list[float]
    matched_count: int
    mean_relative_error: float
    matches: list[dict[str, float]]


def _analyze_spectrum(
    g_values: list[float],
    pad_multiplier: int,
    top_k: int,
    expected_freqs: list[float],
) -> SpectralAnalysis:
    signal_length = pad_multiplier * (len(g_values) + 1)
    signal = np.zeros(signal_length, dtype=np.float64)
    series = np.array(g_values, dtype=np.float64)
    series = series - series.mean()
    signal[: len(series)] = series
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal_length, d=1.0)
    power = np.abs(spectrum)
    power[0] = 0.0
    if len(power) <= top_k:
        indices = np.argsort(power)[::-1]
    else:
        indices = np.argpartition(power, -top_k)[-top_k:]
        indices = indices[np.argsort(power[indices])[::-1]]
    top_freqs = freqs[indices].tolist()
    top_powers = power[indices].tolist()
    matches = _match_frequencies(top_freqs, expected_freqs, len(signal))
    if matches:
        mean_error = float(sum(m["relative_error"] for m in matches) / len(matches))
    else:
        mean_error = 0.0
    return SpectralAnalysis(
        top_frequencies=top_freqs,
        top_powers=top_powers,
        matched_count=len(matches),
        mean_relative_error=mean_error,
        matches=matches,
    )


def _match_frequencies(top_freqs: list[float], expected_freqs: list[float], signal_len: int) -> list[dict[str, float]]:
    matches: list[dict[str, float]] = []
    if not expected_freqs or not top_freqs:
        return matches
    remaining = top_freqs.copy()
    bin_width = 1.0 / signal_len
    for expected in expected_freqs:
        closest_idx = None
        closest_error = None
        for idx, freq in enumerate(remaining):
            abs_diff = abs(freq - expected)
            rel_error = abs_diff / expected if expected != 0 else abs_diff
            if rel_error <= 0.10 or abs_diff <= bin_width:
                if closest_error is None or rel_error < closest_error:
                    closest_idx = idx
                    closest_error = rel_error
        if closest_idx is not None:
            freq = remaining.pop(closest_idx)
            matches.append(
                {
                    "expected": expected,
                    "observed": freq,
                    "relative_error": closest_error or 0.0,
                }
            )
    return matches


def _expected_frequencies(model: ModelSpec, head_dim: int) -> list[float]:
    scheme = model.pe_scheme.lower()
    if "rope" in scheme:
        return _rope_frequencies(head_dim)
    return []


def _rope_frequencies(head_dim: int) -> list[float]:
    dims = max(head_dim // 2, 1)
    i = np.arange(dims)
    theta = 10000 ** (-2 * i / head_dim)
    freqs = theta / (2 * math.pi)
    freqs = freqs[freqs <= 0.5]
    return freqs.tolist()


def _infer_head_dim(model: ModelSpec) -> int:
    config = _load_model_config(model)
    return int(config.hidden_size // config.num_attention_heads)


@lru_cache(maxsize=16)
def _load_model_config(model: ModelSpec):
    paths = default_paths()
    local_dir = paths.models_dir / model.name
    if (local_dir / "config.json").exists():
        return AutoConfig.from_pretrained(local_dir, local_files_only=True)
    return AutoConfig.from_pretrained(model.hf_id)
