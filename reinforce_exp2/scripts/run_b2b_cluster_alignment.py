#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp2.common import RESULTS_ROOT, command_manifest, ensure_dir, timestamp_now, write_json  # noqa: E402
from reinforce_exp2.scripts._shared import emit_core_artifacts, hedges_g, holm_adjust_dict, ivw_meta, parse_models_arg, safe_float  # noqa: E402


def _depth_control_cluster(df: pd.DataFrame, cluster_id: int, metric_col: str) -> dict[str, Any]:
    strata_eff: list[float] = []
    strata_var: list[float] = []
    rows = []
    for layer, ldf in df.groupby("layer"):
        in_c = ldf[ldf["cluster_descriptor_kmeans"] == int(cluster_id)][metric_col].to_numpy(dtype=np.float64)
        out_c = ldf[ldf["cluster_descriptor_kmeans"] != int(cluster_id)][metric_col].to_numpy(dtype=np.float64)
        in_c = in_c[np.isfinite(in_c)]
        out_c = out_c[np.isfinite(out_c)]
        n1, n2 = len(in_c), len(out_c)
        if n1 < 3 or n2 < 10:
            continue
        g = hedges_g(in_c, out_c)
        var = float((n1 + n2) / max(n1 * n2, 1) + (g * g) / max(2 * (n1 + n2 - 2), 1))
        strata_eff.append(g)
        strata_var.append(var)
        rows.append({"layer": int(layer), "n_cluster": int(n1), "n_noncluster": int(n2), "hedges_g": float(g), "variance": var})

    mu, lo, hi = ivw_meta(strata_eff, strata_var)
    if np.isfinite(mu) and np.isfinite(lo) and np.isfinite(hi):
        se = float((hi - lo) / (2 * 1.96))
        z = float(mu / max(se, 1e-8))
        p_two = float(2 * (1.0 - scipy_stats.norm.cdf(abs(z))))
    else:
        p_two = float("nan")

    return {
        "ivw_hedges_g": float(mu),
        "ivw_ci95": [float(lo), float(hi)],
        "p_two": float(p_two),
        "n_valid_layers": int(len(rows)),
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="B2b Cluster-level alignment enrichment", allow_abbrev=False)
    p.add_argument("--models", default="llama-3.1-8b,olmo-2-7b,mistral-7b-v0.1")
    p.add_argument("--output-root", default=str(RESULTS_ROOT / "B2b_cluster_alignment"))
    p.add_argument("--b1-root", default=str(RESULTS_ROOT / "B1_kernel_taxonomy"))
    p.add_argument("--b2a-root", default=str(RESULTS_ROOT / "B2a_head_alignment"))
    args = p.parse_args()

    models = parse_models_arg(args.models)
    out_dir = ensure_dir(Path(args.output_root))

    b1_membership = pd.read_parquet(Path(args.b1_root) / "cluster_membership.parquet")
    b2a_scores = pd.read_parquet(Path(args.b2a_root) / "head_alignment_scores.parquet")

    work = b1_membership.merge(
        b2a_scores[["model", "layer", "head", "head_key", "signed_alignment", "absolute_alignment", "delta_signed", "delta_absolute"]],
        on=["model", "layer", "head", "head_key"],
        how="inner",
    )

    model_reports = {}
    depth_rows_all = []
    pvals_family = {}

    for model in models:
        mdf = work[work["model"] == model].copy()
        if mdf.empty:
            model_reports[model] = {"status": "missing_model_rows"}
            continue

        # confirmatory-family clusters: size>=5; cap to top 4 by size
        csz = mdf.groupby("cluster_descriptor_kmeans").size().reset_index(name="size")
        csz = csz[csz["size"] >= 5].sort_values(["size", "cluster_descriptor_kmeans"], ascending=[False, True]).reset_index(drop=True)
        conf_clusters = [int(x) for x in csz["cluster_descriptor_kmeans"].head(4).tolist()]

        cluster_rows = []
        for c in conf_clusters:
            in_c = mdf[mdf["cluster_descriptor_kmeans"] == c]
            out_c = mdf[mdf["cluster_descriptor_kmeans"] != c]

            un_s = hedges_g(in_c["signed_alignment"].to_numpy(dtype=np.float64), out_c["signed_alignment"].to_numpy(dtype=np.float64))
            un_a = hedges_g(in_c["absolute_alignment"].to_numpy(dtype=np.float64), out_c["absolute_alignment"].to_numpy(dtype=np.float64))

            dep = _depth_control_cluster(mdf, c, metric_col="absolute_alignment")
            dep_signed = _depth_control_cluster(mdf, c, metric_col="signed_alignment")

            row = {
                "cluster": int(c),
                "n_cluster": int(len(in_c)),
                "unadjusted_hedges_g_signed": safe_float(un_s),
                "unadjusted_hedges_g_absolute": safe_float(un_a),
                "depth_control_absolute": dep,
                "depth_control_signed": dep_signed,
            }
            cluster_rows.append(row)

            pvals_family[f"{model}::c{c}"] = safe_float(dep.get("p_two"))

            for dr in dep.get("rows", []):
                depth_rows_all.append(
                    {
                        "model": model,
                        "cluster": int(c),
                        "metric": "absolute_alignment",
                        **dr,
                    }
                )

        model_reports[model] = {
            "status": "ok",
            "confirmatory_clusters": conf_clusters,
            "cluster_rows": cluster_rows,
        }

    p_holm = holm_adjust_dict(pvals_family)

    # apply corrected decisions + acceptance
    support_models = 0
    depth_retention_models = 0
    practical_models = 0

    for model in models:
        rep = model_reports.get(model, {})
        if rep.get("status") != "ok":
            continue

        has_sig = False
        has_depth_retention = False
        has_practical = False

        for row in rep["cluster_rows"]:
            c = int(row["cluster"])
            key = f"{model}::c{c}"
            p_h = safe_float(p_holm.get(key)) if key in p_holm else float("nan")
            dep_abs = row["depth_control_absolute"]
            g_dep = safe_float(dep_abs.get("ivw_hedges_g"))
            g_un = safe_float(row.get("unadjusted_hedges_g_absolute"))
            atten = abs(g_dep) / max(abs(g_un), 1e-8) if np.isfinite(g_dep) and np.isfinite(g_un) else float("nan")
            sign_match = bool(np.isfinite(g_dep) and np.isfinite(g_un) and np.sign(g_dep) == np.sign(g_un))

            sig = bool(np.isfinite(p_h) and p_h < 0.05)
            depth_ok = bool(sig and sign_match and np.isfinite(atten) and atten >= 0.50)
            practical = bool(sig and np.isfinite(g_dep) and abs(g_dep) >= 0.20)

            row["depth_control_absolute"]["p_two_holm_family"] = p_h
            row["depth_control_absolute"]["attenuation_ratio_vs_unadjusted_abs_g"] = safe_float(atten)
            row["depth_control_absolute"]["sign_match_vs_unadjusted"] = sign_match
            row["depth_control_absolute"]["holm_significant"] = sig
            row["depth_control_absolute"]["depth_retention_ok"] = depth_ok
            row["depth_control_absolute"]["practical_threshold_ok"] = practical

            has_sig = has_sig or sig
            has_depth_retention = has_depth_retention or depth_ok
            has_practical = has_practical or practical

        if has_sig:
            support_models += 1
        if has_depth_retention:
            depth_retention_models += 1
        if has_practical:
            practical_models += 1

        rep["model_acceptance"] = {
            "has_holm_significant_cluster": has_sig,
            "has_depth_retention_cluster": has_depth_retention,
            "has_practical_cluster": has_practical,
        }

    c1 = bool(support_models >= 2)
    c2 = bool(depth_retention_models >= 2)
    c3 = bool(practical_models >= 2)
    b2b_supported = bool(c1 and c2 and c3)

    # outputs
    write_json(out_dir / "cluster_alignment_report.json", {"timestamp": timestamp_now(), "models": model_reports})
    write_json(out_dir / "cluster_alignment_depth_control.json", {"timestamp": timestamp_now(), "holm_family": p_holm})

    if depth_rows_all:
        pd.DataFrame(depth_rows_all).to_parquet(out_dir / "cluster_alignment_depth_control.parquet", index=False)

    prereg = {
        "experiment_id": "B2b_cluster_alignment",
        "question": "Are B1 clusters enriched for alignment after depth control?",
        "primary_hypothesis": "At least one confirmatory-family cluster remains enriched after depth control in >=2 models.",
        "primary_endpoints": ["depth-controlled cluster enrichment", "Holm-corrected p-values", "attenuation retention"],
        "secondary_endpoints": ["signed-alignment depth control"],
        "model_list": models,
        "dataset_sources": [
            str(Path(args.b1_root) / "cluster_membership.parquet"),
            str(Path(args.b2a_root) / "head_alignment_scores.parquet"),
        ],
        "inclusion_exclusion_rules": ["Confirmatory clusters must have >=5 heads; capped to top 4 by size."],
        "sample_size_plan": {"max_confirmatory_clusters_per_model": 4},
        "seed_plan": {"deterministic": True},
        "stopping_rule": "Stop after all model cluster tests complete.",
        "multiplicity_family": ["{tested_clusters_confirmatory} x {models}"],
        "acceptance_criteria": ["B2b.3 criteria per TODO."],
        "fallback_interpretation_if_null": "Keep cluster-level enrichment exploratory only.",
    }

    manifest = command_manifest(
        experiment_id="B2b_cluster_alignment",
        command="run_b2b_cluster_alignment.py",
        model="+".join(models),
        extras={"models": models, "b1_root": str(args.b1_root), "b2a_root": str(args.b2a_root)},
    )

    summary = {
        "timestamp": timestamp_now(),
        "experiment_id": "B2b_cluster_alignment",
        "analysis_tier": "confirmatory",
        "canonical_eligible": True,
        "override_used": False,
        "criteria": {
            "criterion_1_holm_sig": c1,
            "criterion_2_depth_retention": c2,
            "criterion_3_practical": c3,
        },
        "verdict": {"B2b_supported": b2b_supported},
        "models": model_reports,
    }

    claim_impact = {
        "timestamp": timestamp_now(),
        "experiment_id": "B2b_cluster_alignment",
        "claim_status": "supported" if b2b_supported else "mixed",
        "supports_main_text": bool(b2b_supported),
        "strict_only": True,
        "notes": ["If mixed, B2b should not be used as required support for B_full narrative."],
    }

    data_dictionary = {
        "experiment_id": "B2b_cluster_alignment",
        "tables": [
            {
                "path": str(out_dir / "cluster_alignment_depth_control.parquet"),
                "description": "Layer-stratified cluster depth-control rows.",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model."},
                    {"name": "cluster", "dtype": "int", "description": "Cluster id."},
                    {"name": "layer", "dtype": "int", "description": "Layer."},
                    {"name": "hedges_g", "dtype": "float", "description": "Layer-specific effect size."},
                    {"name": "variance", "dtype": "float", "description": "Effect variance proxy."},
                ],
            }
        ],
    }

    emit_core_artifacts(
        experiment_id="B2b_cluster_alignment",
        out_dir=out_dir,
        preregistration=prereg,
        manifest=manifest,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )

    write_json(out_dir / "B2b_claim_impact.json", claim_impact)
    print(f"[B2b] wrote {out_dir}")


if __name__ == "__main__":
    main()
