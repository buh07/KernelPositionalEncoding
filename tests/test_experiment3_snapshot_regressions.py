from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sys
from scipy import stats as scipy_stats
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiment3.stats_utils import (
    dependent_corr_williams_test,
    holm_adjust,
    partial_correlation_with_intercept,
)
import experiment3.theory10_feeder_specificity as theory10


ROOT = Path("results/experiment3")


EXPECTED_T1 = {
    "llama-3.1-8b": {
        "high_mean_drop": 0.3081696428571429,
        "low_mean_drop": 0.3227678571428571,
        "gap": -0.014598214285714228,
        "paired_n": 28,
        "paired_mean_delta": -0.014598214285714284,
    },
    "olmo-2-7b": {
        "high_mean_drop": 0.0823660714285714,
        "low_mean_drop": 0.0932440476190476,
        "gap": -0.010877976190476202,
        "paired_n": 21,
        "paired_mean_delta": -0.010877976190476191,
    },
}

EXPECTED_T5 = {
    "llama-3.1-8b": {
        "high_interaction": -1.304987382650376,
        "low_interaction": 0.6554272456169128,
        "random_mean_interaction": -0.322593123515447,
    },
    "olmo-2-7b": {
        "high_interaction": -0.3851767320632935,
        "low_interaction": 1.0527612562179565,
        "random_mean_interaction": 0.22237118037541703,
    },
}

EXPECTED_T7B = {
    "llama-3.1-8b": {
        "rho": 0.06665283817807277,
        "p": 0.2880503676882328,
        "partial_r": -0.09100088503700683,
        "partial_p": 0.14732744606764256,
    },
    "olmo-2-7b": {
        "rho": 0.20636730373083081,
        "p": 0.0008952105448589476,
        "partial_r": 0.10427573593774483,
        "partial_p": 0.09661277863526703,
    },
}

EXPECTED_T10 = {
    "llama-3.1-8b": {
        "rho_induction": 0.06665283817807277,
        "rho_random_mid": -0.037911993591210794,
        "rho_low_si_late": 0.04006280041199359,
    },
    "olmo-2-7b": {
        "rho_induction": 0.20636730373083081,
        "rho_random_mid": 0.2782129777981231,
        "rho_low_si_late": -0.031518224994277866,
    },
}


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Missing artifact: {path}")


def _load_r2(model: str) -> pd.DataFrame:
    path = ROOT / "theory1_si_circuits" / model / "per_sequence_r2.parquet"
    _require(path)
    return (
        pd.read_parquet(path)
        .groupby(["layer", "head"]) ["r2"]
        .mean()
        .reset_index()
        .rename(columns={"layer": "source_layer", "head": "source_head", "r2": "mean_r2"})
    )


def test_t1_snapshot_regression() -> None:
    for model, expected in EXPECTED_T1.items():
        path = ROOT / "theory1_si_circuits" / model / "task_results.parquet"
        _require(path)
        df = pd.read_parquet(path)

        baseline = (
            df[df["condition"] == "none"]
            .groupby(["task", "span"])["accuracy"]
            .mean()
            .to_dict()
        )

        rows = []
        for condition in [c for c in df["condition"].unique() if c != "none"]:
            sub = df[df["condition"] == condition]
            for (task, span), grp in sub.groupby(["task", "span"]):
                rows.append(
                    {
                        "condition": condition,
                        "drop_raw": float(baseline[(task, span)] - grp["accuracy"].mean()),
                    }
                )
        comp = pd.DataFrame(rows)
        high = float(comp[comp["condition"] == "ablate_high_si"]["drop_raw"].mean())
        low = float(comp[comp["condition"] == "ablate_low_si"]["drop_raw"].mean())

        pivot = df.pivot_table(
            index=["task", "span", "seed"],
            columns="condition",
            values="accuracy",
            aggfunc="mean",
        )
        paired = pivot.dropna(subset=["none", "ablate_high_si", "ablate_low_si"])
        delta = (
            (paired["none"] - paired["ablate_high_si"])
            - (paired["none"] - paired["ablate_low_si"])
        )

        assert np.isclose(high, expected["high_mean_drop"], atol=1e-12)
        assert np.isclose(low, expected["low_mean_drop"], atol=1e-12)
        assert np.isclose(high - low, expected["gap"], atol=1e-12)
        assert int(len(delta)) == int(expected["paired_n"])
        assert np.isclose(float(delta.mean()), expected["paired_mean_delta"], atol=1e-12)


def test_t5_snapshot_regression() -> None:
    for model, expected in EXPECTED_T5.items():
        path = ROOT / "theory5_subword_ablation" / model / "per_position_losses.parquet"
        _require(path)
        loss_df = pd.read_parquet(path)

        condition_losses: dict[str, dict[str, np.ndarray]] = {}
        for condition in sorted(loss_df["condition"].unique()):
            condition_losses[condition] = {
                "continuation": loss_df[
                    (loss_df["condition"] == condition)
                    & (loss_df["token_type"] == "continuation")
                ]["loss"].to_numpy(),
                "word_initial": loss_df[
                    (loss_df["condition"] == condition)
                    & (loss_df["token_type"] == "word_initial")
                ]["loss"].to_numpy(),
            }

        baseline_cont = condition_losses["none"]["continuation"]
        baseline_init = condition_losses["none"]["word_initial"]

        high = condition_losses["ablate_high_si"]
        low = condition_losses["ablate_low_si"]

        high_inter = float((high["continuation"] - baseline_cont).mean() - (high["word_initial"] - baseline_init).mean())
        low_inter = float((low["continuation"] - baseline_cont).mean() - (low["word_initial"] - baseline_init).mean())

        random_inter = []
        for condition, vals in condition_losses.items():
            if condition.startswith("ablate_random"):
                random_inter.append(
                    float((vals["continuation"] - baseline_cont).mean() - (vals["word_initial"] - baseline_init).mean())
                )

        assert np.isclose(high_inter, expected["high_interaction"], atol=1e-12)
        assert np.isclose(low_inter, expected["low_interaction"], atol=1e-12)
        assert np.isclose(float(np.mean(random_inter)), expected["random_mean_interaction"], atol=1e-12)


def test_t7b_snapshot_regression() -> None:
    for model, expected in EXPECTED_T7B.items():
        patch_path = ROOT / "theory7b_activation_patching" / model / "patching_results.parquet"
        prev_path = ROOT / "theory7_induction_feeders" / model / "prev_token_scores.parquet"
        _require(patch_path)
        _require(prev_path)

        patch_df = pd.read_parquet(patch_path)
        r2_df = _load_r2(model)
        prev_df = pd.read_parquet(prev_path).rename(columns={"layer": "source_layer", "head": "source_head"})

        source = (
            patch_df
            .groupby(["source_layer", "source_head"]) ["mean_disruption"]
            .mean()
            .reset_index()
        )
        merged = pd.merge(source, r2_df, on=["source_layer", "source_head"], how="inner")

        rho, p = spearmanr(merged["mean_r2"], merged["mean_disruption"])

        merged_pt = pd.merge(
            merged,
            prev_df[["source_layer", "source_head", "prev_token_score"]],
            on=["source_layer", "source_head"],
            how="inner",
        )
        partial = partial_correlation_with_intercept(
            merged_pt["mean_r2"].values,
            merged_pt["mean_disruption"].values,
            merged_pt["prev_token_score"].values,
        )

        assert np.isclose(float(rho), expected["rho"], atol=1e-12)
        assert np.isclose(float(p), expected["p"], atol=1e-12)
        assert np.isclose(float(partial.r), expected["partial_r"], atol=1e-12)
        assert np.isclose(float(partial.p_value), expected["partial_p"], atol=1e-12)


def test_t10_snapshot_regression_and_specificity_guardrail() -> None:
    for model, expected in EXPECTED_T10.items():
        patch_path = ROOT / "theory10_feeder_specificity" / model / "patching_results.parquet"
        _require(patch_path)
        patch_df = pd.read_parquet(patch_path)
        r2_df = _load_r2(model)

        by_group: dict[str, pd.DataFrame] = {}
        rho_map: dict[str, float] = {}
        for group in ["induction", "random_mid", "low_si_late"]:
            group_df = patch_df[patch_df["target_group"] == group]
            source = (
                group_df
                .groupby(["source_layer", "source_head"]) ["mean_disruption"]
                .mean()
                .reset_index()
            )
            merged = pd.merge(source, r2_df, on=["source_layer", "source_head"], how="inner")
            by_group[group] = merged
            rho_map[group] = float(spearmanr(merged["mean_r2"], merged["mean_disruption"])[0])

        assert np.isclose(rho_map["induction"], expected["rho_induction"], atol=1e-12)
        assert np.isclose(rho_map["random_mid"], expected["rho_random_mid"], atol=1e-12)
        assert np.isclose(rho_map["low_si_late"], expected["rho_low_si_late"], atol=1e-12)

        pvals: dict[str, float] = {}
        deltas: dict[str, float] = {}
        for comp_name in ["random_mid", "low_si_late"]:
            merged = pd.merge(
                by_group["induction"][["source_layer", "source_head", "mean_disruption"]].rename(
                    columns={"mean_disruption": "di"}
                ),
                by_group[comp_name][["source_layer", "source_head", "mean_disruption"]].rename(
                    columns={"mean_disruption": "dc"}
                ),
                on=["source_layer", "source_head"],
                how="inner",
            )
            merged = pd.merge(
                merged,
                r2_df,
                on=["source_layer", "source_head"],
                how="inner",
            )
            x = scipy_stats.rankdata(merged["mean_r2"].values)
            yi = scipy_stats.rankdata(merged["di"].values)
            yc = scipy_stats.rankdata(merged["dc"].values)
            r_xy = float(np.corrcoef(x, yi)[0, 1])
            r_xz = float(np.corrcoef(x, yc)[0, 1])
            r_yz = float(np.corrcoef(yi, yc)[0, 1])
            res = dependent_corr_williams_test(r_xy=r_xy, r_xz=r_xz, r_yz=r_yz, n=len(x))
            deltas[comp_name] = r_xy - r_xz
            pvals[comp_name] = float(res.p_value)

        p_holm = holm_adjust(pvals)
        induction_specific = all(deltas[name] > 0 and p_holm[name] < 0.05 for name in p_holm)

        # Guardrail: if induction rho is below random-mid rho, it must never be induction-specific.
        if rho_map["induction"] < rho_map["random_mid"]:
            assert not induction_specific


def test_t10_guardrail_on_synthetic_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(theory10, "BOOTSTRAP_N", 200)

    rng = np.random.default_rng(0)
    n_sources = 40
    rows = []
    for source_head in range(n_sources):
        r2 = source_head / (n_sources - 1)
        for group, target_head in [("induction", 0), ("random_mid", 1), ("low_si_late", 2)]:
            if group == "induction":
                disruption = float(rng.normal(scale=1.0))
            elif group == "random_mid":
                disruption = float(2.0 * r2 + rng.normal(scale=0.05))
            else:
                disruption = float(0.5 * r2 + rng.normal(scale=0.2))
            rows.append(
                {
                    "source_layer": 0,
                    "source_head": source_head,
                    "target_layer": 1,
                    "target_head": target_head,
                    "target_group": group,
                    "mean_disruption": disruption,
                    "std_disruption": 0.0,
                    "n_pairs": 30,
                }
            )

    patch_df = pd.DataFrame(rows)
    r2_df = pd.DataFrame(
        {
            "layer": [0 for _ in range(n_sources)],
            "head": list(range(n_sources)),
            "mean_r2": [h / (n_sources - 1) for h in range(n_sources)],
        }
    )
    target_groups = {
        "induction": [theory10.HeadID(1, 0)],
        "random_mid": [theory10.HeadID(1, 1)],
        "low_si_late": [theory10.HeadID(1, 2)],
    }

    analysis = theory10.analyze_patching_results(
        patch_df=patch_df,
        r2_df=r2_df,
        prev_token_df=None,
        target_groups=target_groups,
        model_name="synthetic",
        num_pairs=30,
    )
    comp = analysis["cross_group_comparison"]
    assert comp["rho_induction"] < comp["rho_random_mid"]
    assert comp["induction_specific"] is False
