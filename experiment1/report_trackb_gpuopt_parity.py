from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RESULTS_ROOT = Path("results")
REPORT_ROOT = RESULTS_ROOT / "reports" / "track_b_gpuopt_v1"
PARITY_THRESHOLDS = {
    "mae_raw_max": 1e-10,
    "mae_centered_max": 1e-7,
    "max_abs_any_max": 1e-5,
}


@dataclass(frozen=True)
class VariantPair:
    gpuopt_group: str
    baseline_group: str


VARIANT_PAIRS: tuple[VariantPair, ...] = (
    VariantPair("track_b_gpuopt_v1", "track_b"),
    VariantPair("track_b_canonical_perpos_gpuopt_v1", "track_b_canonical_perpos_v1"),
    VariantPair("track_b_shared_mean_gpuopt_v1", "track_b_shared_mean_v1"),
    VariantPair("track_b_bucketed_mean_b32_gpuopt_v1", "track_b_bucketed_mean_v1"),
    VariantPair("track_b_bucketed_mean_b8_gpuopt_v1", "track_b_bucketed_mean_b8_v1"),
    VariantPair("track_b_bucketed_mean_b16_gpuopt_v1", "track_b_bucketed_mean_b16_v1"),
    VariantPair("track_b_bucketed_mean_b64_gpuopt_v1", "track_b_bucketed_mean_b64_v1"),
)


def _load_group(group: str) -> pd.DataFrame:
    base = RESULTS_ROOT / group
    rows: list[pd.DataFrame] = []
    for path in base.glob("**/summary.parquet"):
        df = pd.read_parquet(path)
        needed = ["model", "dataset", "seq_len", "layer", "head", "r2_raw", "r2_centered"]
        missing = [col for col in needed if col not in df.columns]
        if missing:
            raise RuntimeError(f"{path} missing columns: {missing}")
        rows.append(df[needed].copy())
    if not rows:
        return pd.DataFrame(columns=["model", "dataset", "seq_len", "layer", "head", "r2_raw", "r2_centered"])
    out = pd.concat(rows, ignore_index=True)
    out["group"] = group
    return out


def _coverage_table() -> pd.DataFrame:
    rows = []
    for pair in VARIANT_PAIRS:
        base = RESULTS_ROOT / pair.gpuopt_group
        present = sum(1 for _ in base.glob("**/summary.parquet"))
        rows.append(
            {
                "group": pair.gpuopt_group,
                "expected_summaries": 36,
                "present_summaries": present,
                "complete": present == 36,
            }
        )
    return pd.DataFrame(rows).sort_values("group")


def _parity_table() -> pd.DataFrame:
    key_cols = ["model", "dataset", "seq_len", "layer", "head"]
    rows: list[dict[str, float | int | str | bool]] = []
    for pair in VARIANT_PAIRS:
        gpuopt_df = _load_group(pair.gpuopt_group)
        baseline_df = _load_group(pair.baseline_group)
        if gpuopt_df.empty or baseline_df.empty:
            rows.append(
                {
                    "gpuopt_group": pair.gpuopt_group,
                    "baseline_group": pair.baseline_group,
                    "overlap_rows": 0,
                    "mae_r2_raw": None,
                    "mae_r2_centered": None,
                    "max_abs_r2_raw": None,
                    "max_abs_r2_centered": None,
                    "max_abs_any_r2": None,
                    "pass_mae_raw": False,
                    "pass_mae_centered": False,
                    "pass_max_abs_any": False,
                    "pass_all": False,
                }
            )
            continue
        merged = gpuopt_df.merge(baseline_df, on=key_cols, suffixes=("_gpuopt", "_baseline"), how="inner")
        if merged.empty:
            rows.append(
                {
                    "gpuopt_group": pair.gpuopt_group,
                    "baseline_group": pair.baseline_group,
                    "overlap_rows": 0,
                    "mae_r2_raw": None,
                    "mae_r2_centered": None,
                    "max_abs_r2_raw": None,
                    "max_abs_r2_centered": None,
                    "max_abs_any_r2": None,
                    "pass_mae_raw": False,
                    "pass_mae_centered": False,
                    "pass_max_abs_any": False,
                    "pass_all": False,
                }
            )
            continue
        raw_delta = (merged["r2_raw_gpuopt"] - merged["r2_raw_baseline"]).abs()
        centered_delta = (merged["r2_centered_gpuopt"] - merged["r2_centered_baseline"]).abs()
        mae_raw = float(raw_delta.mean())
        mae_centered = float(centered_delta.mean())
        max_raw = float(raw_delta.max())
        max_centered = float(centered_delta.max())
        max_any = max(max_raw, max_centered)
        pass_mae_raw = mae_raw <= PARITY_THRESHOLDS["mae_raw_max"]
        pass_mae_centered = mae_centered <= PARITY_THRESHOLDS["mae_centered_max"]
        pass_max_any = max_any <= PARITY_THRESHOLDS["max_abs_any_max"]
        rows.append(
            {
                "gpuopt_group": pair.gpuopt_group,
                "baseline_group": pair.baseline_group,
                "overlap_rows": int(merged.shape[0]),
                "mae_r2_raw": mae_raw,
                "mae_r2_centered": mae_centered,
                "max_abs_r2_raw": max_raw,
                "max_abs_r2_centered": max_centered,
                "max_abs_any_r2": max_any,
                "pass_mae_raw": pass_mae_raw,
                "pass_mae_centered": pass_mae_centered,
                "pass_max_abs_any": pass_max_any,
                "pass_all": pass_mae_raw and pass_mae_centered and pass_max_any,
            }
        )
    return pd.DataFrame(rows).sort_values("gpuopt_group")


def _raw_invariance_table() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    key_cols = ["model", "dataset", "seq_len", "layer", "head"]
    for pair in VARIANT_PAIRS:
        df = _load_group(pair.gpuopt_group)
        if df.empty:
            continue
        frames.append(df[key_cols + ["r2_raw"]].rename(columns={"r2_raw": pair.gpuopt_group}))
    if not frames:
        return pd.DataFrame(
            [
                {
                    "rows_with_2plus_variants": 0,
                    "mean_spread_r2_raw": None,
                    "max_spread_r2_raw": None,
                }
            ]
        )
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=key_cols, how="outer")
    value_cols = [col for col in merged.columns if col not in key_cols]
    present_counts = merged[value_cols].notna().sum(axis=1)
    multi = merged[present_counts >= 2]
    if multi.empty:
        return pd.DataFrame(
            [
                {
                    "rows_with_2plus_variants": 0,
                    "mean_spread_r2_raw": None,
                    "max_spread_r2_raw": None,
                }
            ]
        )
    spread = multi[value_cols].max(axis=1, skipna=True) - multi[value_cols].min(axis=1, skipna=True)
    return pd.DataFrame(
        [
            {
                "rows_with_2plus_variants": int(multi.shape[0]),
                "mean_spread_r2_raw": float(spread.mean()),
                "max_spread_r2_raw": float(spread.max()),
            }
        ]
    )


def _synthetic_parity_table() -> pd.DataFrame:
    rows = []
    for pair in VARIANT_PAIRS:
        df = _load_group(pair.gpuopt_group)
        if df.empty:
            rows.append(
                {
                    "group": pair.gpuopt_group,
                    "rows": 0,
                    "mae_abs_centered_minus_raw": None,
                    "max_abs_centered_minus_raw": None,
                }
            )
            continue
        synthetic = df[df["dataset"] == "synthetic_random"].copy()
        if synthetic.empty:
            rows.append(
                {
                    "group": pair.gpuopt_group,
                    "rows": 0,
                    "mae_abs_centered_minus_raw": None,
                    "max_abs_centered_minus_raw": None,
                }
            )
            continue
        delta = (synthetic["r2_centered"] - synthetic["r2_raw"]).abs()
        rows.append(
            {
                "group": pair.gpuopt_group,
                "rows": int(synthetic.shape[0]),
                "mae_abs_centered_minus_raw": float(delta.mean()),
                "max_abs_centered_minus_raw": float(delta.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("group")


def _runtime_table() -> pd.DataFrame:
    rows = []
    pattern = re.compile(
        r"worker end gpu=(?P<gpu>\d+) ran=(?P<ran>\d+) skipped=(?P<skipped>\d+) failed=(?P<failed>\d+) elapsed_sec=(?P<elapsed>\d+)"
    )
    for log_path in sorted((Path("logs") / "trackb").glob("*.log")):
        last_match = None
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = pattern.search(line)
                if match:
                    last_match = match
        if last_match is None:
            continue
        filename = log_path.name
        category = "gpuopt" if "gpuopt" in filename else "legacy_or_other"
        rows.append(
            {
                "log_file": filename,
                "category": category,
                "gpu_id": int(last_match.group("gpu")),
                "ran_jobs": int(last_match.group("ran")),
                "skipped_jobs": int(last_match.group("skipped")),
                "failed_jobs": int(last_match.group("failed")),
                "elapsed_sec": int(last_match.group("elapsed")),
                "elapsed_hr": float(last_match.group("elapsed")) / 3600.0,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["log_file", "category", "gpu_id", "ran_jobs", "skipped_jobs", "failed_jobs", "elapsed_sec", "elapsed_hr"]
        )
    return pd.DataFrame(rows).sort_values(["category", "log_file"])


def _to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        formatted: list[str] = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                formatted.append("")
            elif isinstance(value, float):
                formatted.append(f"{value:.12g}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    coverage = _coverage_table()
    parity = _parity_table()
    raw_invariance = _raw_invariance_table()
    synthetic = _synthetic_parity_table()
    runtime = _runtime_table()

    all_metrics = {
        "coverage": coverage,
        "parity": parity,
        "raw_invariance": raw_invariance,
        "synthetic_centered_raw": synthetic,
        "runtime": runtime,
    }

    metrics_rows = []
    for table_name, df in all_metrics.items():
        if df.empty:
            continue
        out = df.copy()
        out["table"] = table_name
        metrics_rows.append(out)
    if metrics_rows:
        pd.concat(metrics_rows, ignore_index=True).to_parquet(REPORT_ROOT / "parity_metrics.parquet", index=False)
    else:
        pd.DataFrame({"table": []}).to_parquet(REPORT_ROOT / "parity_metrics.parquet", index=False)

    runtime.to_parquet(REPORT_ROOT / "runtime_comparison.parquet", index=False)

    pass_all = bool((parity["pass_all"] == True).all()) if not parity.empty else False
    with (REPORT_ROOT / "parity_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Track B GPU-Optimized Parity Report\n\n")
        handle.write("## Thresholds\n")
        handle.write(f"- `MAE(r2_raw) <= {PARITY_THRESHOLDS['mae_raw_max']}`\n")
        handle.write(f"- `MAE(r2_centered) <= {PARITY_THRESHOLDS['mae_centered_max']}`\n")
        handle.write(f"- `max_abs_delta(any r2 field) <= {PARITY_THRESHOLDS['max_abs_any_max']}`\n\n")
        handle.write("## Coverage\n")
        handle.write(_to_md(coverage))
        handle.write("\n\n## Variant-Matched Parity\n")
        handle.write(_to_md(parity))
        handle.write("\n\n## Raw Invariance Across GPU-Optimized Variants\n")
        handle.write(_to_md(raw_invariance))
        handle.write("\n\n## Synthetic Centered vs Raw Parity\n")
        handle.write(_to_md(synthetic))
        handle.write("\n\n## Runtime Log Summary\n")
        handle.write(_to_md(runtime))
        handle.write("\n\n## Gate Status\n")
        handle.write(f"- Overall strict parity pass: `{pass_all}`\n")

    print(f"Wrote {REPORT_ROOT / 'parity_summary.md'}")
    print(f"Wrote {REPORT_ROOT / 'parity_metrics.parquet'}")
    print(f"Wrote {REPORT_ROOT / 'runtime_comparison.parquet'}")


if __name__ == "__main__":
    main()
