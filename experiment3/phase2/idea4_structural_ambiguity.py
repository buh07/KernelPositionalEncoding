#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment3.theory1_si_circuits import (  # noqa: E402
    HeadID,
    MODELS,
    head_output_ablation,
)
from shared.models.loading import load_model, load_tokenizer  # noqa: E402


TARGET_MODELS = ("llama-3.1-8b", "olmo-2-7b")


@dataclass(frozen=True)
class Stimulus:
    stimulus_id: str
    family: str
    relation_type: str
    variant_id: str
    context_label: str
    prompt: str
    option_a: str
    option_b: str
    correct_option: str | None


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_heads(entries: list[dict[str, Any]]) -> list[HeadID]:
    return [HeadID(int(e["layer"]), int(e["head"])) for e in entries]


def _stable_id(parts: list[str]) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return digest


def _build_stimuli() -> list[Stimulus]:
    stimuli: list[Stimulus] = []

    pp_variants: list[tuple[str, str]] = [
        ("man", "telescope"),
        ("woman", "camera"),
        ("guard", "binoculars"),
        ("child", "flashlight"),
        ("hiker", "map"),
        ("pilot", "radio"),
        ("chef", "knife"),
        ("scientist", "microscope"),
        ("artist", "brush"),
        ("mechanic", "wrench"),
        ("sailor", "compass"),
        ("doctor", "stethoscope"),
        ("photographer", "tripod"),
        ("student", "notebook"),
        ("runner", "watch"),
        ("farmer", "shovel"),
        ("worker", "hammer"),
        ("detective", "magnifier"),
        ("tourist", "guidebook"),
        ("hunter", "scope"),
    ]

    rc_variants: list[tuple[str, str, str]] = [
        ("daughter", "colonel", "balcony"),
        ("assistant", "manager", "office"),
        ("niece", "senator", "podium"),
        ("friend", "teacher", "hallway"),
        ("brother", "captain", "deck"),
        ("aide", "governor", "stage"),
        ("cousin", "director", "set"),
        ("student", "professor", "atrium"),
        ("guest", "host", "terrace"),
        ("intern", "engineer", "lab"),
        ("nurse", "surgeon", "ward"),
        ("reporter", "editor", "studio"),
        ("partner", "lawyer", "courthouse"),
        ("teammate", "coach", "sideline"),
        ("neighbor", "landlord", "porch"),
        ("visitor", "curator", "gallery"),
        ("author", "critic", "bookshop"),
        ("singer", "conductor", "balcony"),
        ("driver", "officer", "bridge"),
        ("chef", "owner", "kitchen"),
    ]

    coord_variants: list[tuple[str, str, str, str]] = [
        ("old", "men", "women", "station"),
        ("young", "boys", "girls", "park"),
        ("tired", "workers", "students", "terminal"),
        ("happy", "teachers", "parents", "auditorium"),
        ("angry", "drivers", "cyclists", "intersection"),
        ("calm", "doctors", "nurses", "clinic"),
        ("quiet", "tourists", "guides", "museum"),
        ("excited", "fans", "players", "arena"),
        ("wet", "runners", "walkers", "trailhead"),
        ("cold", "sailors", "pilots", "harbor"),
        ("busy", "clerks", "managers", "lobby"),
        ("nervous", "interns", "analysts", "boardroom"),
        ("hungry", "children", "adults", "cafeteria"),
        ("sleepy", "guards", "visitors", "gate"),
        ("careful", "painters", "carpenters", "workshop"),
        ("serious", "judges", "lawyers", "courthouse"),
        ("polite", "hosts", "guests", "banquet"),
        ("late", "passengers", "drivers", "platform"),
        ("confident", "debaters", "moderators", "forum"),
        ("curious", "students", "teachers", "library"),
        ("noisy", "kids", "parents", "playground"),
        ("strong", "lifters", "swimmers", "gym"),
        ("brave", "firefighters", "residents", "street"),
        ("focused", "coders", "designers", "office"),
        ("friendly", "cashiers", "customers", "market"),
        ("strict", "referees", "coaches", "field"),
        ("careless", "drivers", "pedestrians", "crosswalk"),
        ("famous", "actors", "musicians", "festival"),
        ("wealthy", "investors", "founders", "conference"),
        ("patient", "therapists", "clients", "center"),
        ("alert", "soldiers", "medics", "camp"),
        ("quiet", "monks", "novices", "temple"),
        ("eager", "recruits", "officers", "academy"),
        ("careful", "chefs", "servers", "kitchen"),
        ("quick", "sprinters", "joggers", "track"),
        ("creative", "writers", "editors", "newsroom"),
        ("steady", "captains", "sailors", "deck"),
        ("organized", "planners", "volunteers", "venue"),
        ("tall", "guards", "visitors", "entrance"),
        ("busy", "engineers", "technicians", "lab"),
    ]

    def _prompt(context: str, sentence: str, question: str) -> str:
        return f"{context} {sentence}\n{question}\nAnswer:"

    # Family 1: PP attachment (hierarchical attachment choice).
    for idx, (noun, tool) in enumerate(pp_variants):
        base = f"pp_{idx:03d}"
        sentence = f"I saw the {noun} with the {tool}."
        question = f"Who had the {tool}?"
        option_a = " I did."
        option_b = f" The {noun} did."

        items = [
            (
                "context_a",
                f"I needed a {tool} to see details clearly.",
                "A",
            ),
            (
                "context_b",
                f"The {noun} was carrying a {tool} all day.",
                "B",
            ),
            (
                "neutral",
                "Read the sentence and answer the question.",
                None,
            ),
        ]

        for context_label, context, correct in items:
            stimulus_id = _stable_id(["pp", base, context_label, noun, tool])
            stimuli.append(
                Stimulus(
                    stimulus_id=stimulus_id,
                    family="pp_attachment",
                    relation_type="hierarchical_absolute",
                    variant_id=base,
                    context_label=context_label,
                    prompt=_prompt(context, sentence, question),
                    option_a=option_a,
                    option_b=option_b,
                    correct_option=correct,
                )
            )

    # Family 2: Relative clause attachment (hierarchical choice).
    for idx, (n1, n2, loc) in enumerate(rc_variants):
        base = f"rc_{idx:03d}"
        sentence = f"The {n1} of the {n2} who was on the {loc} waved."
        question = f"Who was on the {loc}?"
        option_a = f" The {n1} was."
        option_b = f" The {n2} was."

        items = [
            (
                "context_a",
                f"Everyone noticed that the {n1} was standing on the {loc}.",
                "A",
            ),
            (
                "context_b",
                f"Everyone noticed that the {n2} was standing on the {loc}.",
                "B",
            ),
            (
                "neutral",
                "Read the sentence and answer the question.",
                None,
            ),
        ]

        for context_label, context, correct in items:
            stimulus_id = _stable_id(["rc", base, context_label, n1, n2, loc])
            stimuli.append(
                Stimulus(
                    stimulus_id=stimulus_id,
                    family="relative_clause_attachment",
                    relation_type="hierarchical_absolute",
                    variant_id=base,
                    context_label=context_label,
                    prompt=_prompt(context, sentence, question),
                    option_a=option_a,
                    option_b=option_b,
                    correct_option=correct,
                )
            )

    # Family 3: Coordination scope (local modifier scope).
    for idx, (adj, g1, g2, place) in enumerate(coord_variants):
        base = f"coord_{idx:03d}"
        sentence = f"They greeted {adj} {g1} and {g2} at the {place}."
        question = f"Were the {g2} {adj} in this sentence?"
        option_a = f" Yes, both the {g1} and the {g2} were {adj}."
        option_b = f" No, only the {g1} were {adj}."

        items = [
            (
                "context_a",
                f"In this scenario both the {g1} and the {g2} were clearly {adj}.",
                "A",
            ),
            (
                "context_b",
                f"In this scenario only the {g1} were {adj}, and the {g2} were not {adj}.",
                "B",
            ),
            (
                "neutral",
                "Read the sentence and answer the question.",
                None,
            ),
        ]

        for context_label, context, correct in items:
            stimulus_id = _stable_id(["coord", base, context_label, adj, g1, g2, place])
            stimuli.append(
                Stimulus(
                    stimulus_id=stimulus_id,
                    family="coordination_scope",
                    relation_type="local_relative",
                    variant_id=base,
                    context_label=context_label,
                    prompt=_prompt(context, sentence, question),
                    option_a=option_a,
                    option_b=option_b,
                    correct_option=correct,
                )
            )

    return stimuli


def _score_binary_options(
    *,
    model,
    tokenizer,
    device: str,
    prompt: str,
    option_a: str,
    option_b: str,
) -> dict[str, float]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_ids = tokenizer.encode(option_a, add_special_tokens=False)
    b_ids = tokenizer.encode(option_b, add_special_tokens=False)

    if len(prompt_ids) < 1:
        raise RuntimeError("Prompt tokenization produced empty sequence.")
    if len(a_ids) < 1 or len(b_ids) < 1:
        raise RuntimeError("Option tokenization produced empty sequence.")

    seq_a = prompt_ids + a_ids
    seq_b = prompt_ids + b_ids

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    max_len = max(len(seq_a), len(seq_b))
    input_ids = torch.full((2, max_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((2, max_len), dtype=torch.long, device=device)

    input_ids[0, : len(seq_a)] = torch.tensor(seq_a, dtype=torch.long, device=device)
    input_ids[1, : len(seq_b)] = torch.tensor(seq_b, dtype=torch.long, device=device)
    attention_mask[0, : len(seq_a)] = 1
    attention_mask[1, : len(seq_b)] = 1

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)

    def _sum_completion_lp(row: int, completion_ids: list[int]) -> float:
        start = len(prompt_ids)
        total = 0.0
        for j, tok in enumerate(completion_ids):
            pos = start + j - 1
            if pos < 0 or pos >= log_probs.shape[1]:
                continue
            total += float(log_probs[row, pos, int(tok)].item())
        return total

    lp_a = _sum_completion_lp(0, a_ids)
    lp_b = _sum_completion_lp(1, b_ids)

    mx = max(lp_a, lp_b)
    pa = float(np.exp(lp_a - mx) / (np.exp(lp_a - mx) + np.exp(lp_b - mx)))
    pb = 1.0 - pa
    entropy = float(-(pa * np.log(max(pa, 1e-12)) + pb * np.log(max(pb, 1e-12))))

    return {
        "logprob_a": lp_a,
        "logprob_b": lp_b,
        "margin_a_minus_b": lp_a - lp_b,
        "p_a": pa,
        "entropy": entropy,
    }


def _bootstrap_mean_ci(values: np.ndarray, *, n_boot: int, seed: int) -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": float("nan"), "ci_95": [float("nan"), float("nan")], "n": 0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = vals[idx].mean(axis=1)
    return {
        "mean": float(vals.mean()),
        "ci_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "n": int(vals.size),
    }


def _paired_bootstrap_delta(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    direction: str,
) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return {
            "mean_delta": float("nan"),
            "ci_95": [float("nan"), float("nan")],
            "p_one_sided": float("nan"),
            "n": 0,
        }

    diffs = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot = diffs[idx].mean(axis=1)

    if direction == "greater":
        p_one = float((np.sum(boot <= 0.0) + 1) / (n_boot + 1))
    elif direction == "less":
        p_one = float((np.sum(boot >= 0.0) + 1) / (n_boot + 1))
    else:
        raise ValueError(f"Unsupported direction={direction}")

    return {
        "mean_delta": float(diffs.mean()),
        "ci_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "p_one_sided": p_one,
        "n": int(diffs.size),
    }


def _load_conditions(model_name: str, include_random_control: bool) -> dict[str, list[HeadID]]:
    head_groups_path = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_groups.json"
    if not head_groups_path.exists():
        raise FileNotFoundError(f"Missing head_groups.json for {model_name}: {head_groups_path}")

    groups = _load_json(head_groups_path)
    conditions = {
        "none": [],
        "ablate_high_si": _as_heads(groups["high_si"]),
        "ablate_low_si": _as_heads(groups["low_si"]),
    }

    if include_random_control:
        random_entries = groups.get("random_draw0")
        if not isinstance(random_entries, list):
            raise RuntimeError(f"random_draw0 missing in {head_groups_path}")
        conditions["ablate_random_draw0"] = _as_heads(random_entries)

    return conditions


def _build_report(
    *,
    model_name: str,
    rows: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    labeled = rows[rows["correct_option"].notna()].copy()
    neutral = rows[rows["correct_option"].isna()].copy()

    condition_summaries: dict[str, Any] = {}
    for condition in sorted(rows["condition"].unique()):
        sub = labeled[labeled["condition"] == condition].copy()
        if sub.empty:
            continue
        condition_summaries[condition] = {
            "accuracy": _bootstrap_mean_ci(sub["is_correct"].astype(float).to_numpy(), n_boot=bootstrap_samples, seed=seed + 11),
            "correct_margin": _bootstrap_mean_ci(sub["correct_margin"].astype(float).to_numpy(), n_boot=bootstrap_samples, seed=seed + 12),
            "entropy": _bootstrap_mean_ci(sub["entropy"].astype(float).to_numpy(), n_boot=bootstrap_samples, seed=seed + 13),
        }

    paired = labeled.pivot_table(
        index=["stimulus_id", "relation_type", "family", "context_label"],
        columns="condition",
        values=["is_correct", "correct_margin", "entropy"],
        aggfunc="first",
    )

    paired_comparisons: dict[str, Any] = {}

    if ("is_correct", "ablate_high_si") in paired.columns and ("is_correct", "ablate_low_si") in paired.columns:
        paired_comparisons["high_vs_low_accuracy"] = _paired_bootstrap_delta(
            paired[("is_correct", "ablate_high_si")].to_numpy(),
            paired[("is_correct", "ablate_low_si")].to_numpy(),
            n_boot=bootstrap_samples,
            seed=seed + 21,
            direction="less",
        )

    if ("is_correct", "none") in paired.columns and ("is_correct", "ablate_high_si") in paired.columns:
        paired_comparisons["none_vs_high_accuracy"] = _paired_bootstrap_delta(
            paired[("is_correct", "none")].to_numpy(),
            paired[("is_correct", "ablate_high_si")].to_numpy(),
            n_boot=bootstrap_samples,
            seed=seed + 22,
            direction="greater",
        )

    if ("is_correct", "none") in paired.columns and ("is_correct", "ablate_low_si") in paired.columns:
        paired_comparisons["none_vs_low_accuracy"] = _paired_bootstrap_delta(
            paired[("is_correct", "none")].to_numpy(),
            paired[("is_correct", "ablate_low_si")].to_numpy(),
            n_boot=bootstrap_samples,
            seed=seed + 23,
            direction="greater",
        )

    # Relation-type interaction: does high-vs-low degradation differ for local vs hierarchical?
    interaction_payload = {
        "observed_interaction": float("nan"),
        "ci_95": [float("nan"), float("nan")],
        "p_two_sided": float("nan"),
        "n_local": 0,
        "n_hierarchical": 0,
        "definition": "(drop_high-drop_low)_local - (drop_high-drop_low)_hierarchical",
    }

    req_cols = [
        ("is_correct", "none"),
        ("is_correct", "ablate_high_si"),
        ("is_correct", "ablate_low_si"),
    ]
    if all(c in paired.columns for c in req_cols):
        tmp = paired.copy().dropna(subset=req_cols).reset_index()
        local = tmp[tmp["relation_type"] == "local_relative"].copy()
        hier = tmp[tmp["relation_type"] == "hierarchical_absolute"].copy()

        if len(local) > 0 and len(hier) > 0:
            def _delta(df: pd.DataFrame) -> np.ndarray:
                none = df[("is_correct", "none")].to_numpy(dtype=float)
                hi = df[("is_correct", "ablate_high_si")].to_numpy(dtype=float)
                lo = df[("is_correct", "ablate_low_si")].to_numpy(dtype=float)
                return (none - hi) - (none - lo)

            d_local = _delta(local)
            d_hier = _delta(hier)
            obs = float(np.mean(d_local) - np.mean(d_hier))

            rng = np.random.default_rng(seed + 31)
            bvals = np.zeros(bootstrap_samples, dtype=float)
            for i in range(bootstrap_samples):
                l_idx = rng.integers(0, len(d_local), size=len(d_local))
                h_idx = rng.integers(0, len(d_hier), size=len(d_hier))
                bvals[i] = float(np.mean(d_local[l_idx]) - np.mean(d_hier[h_idx]))

            p_two = float(2.0 * min((np.sum(bvals >= 0.0) + 1) / (bootstrap_samples + 1), (np.sum(bvals <= 0.0) + 1) / (bootstrap_samples + 1)))
            interaction_payload = {
                "observed_interaction": obs,
                "ci_95": [float(np.quantile(bvals, 0.025)), float(np.quantile(bvals, 0.975))],
                "p_two_sided": p_two,
                "n_local": int(len(d_local)),
                "n_hierarchical": int(len(d_hier)),
                "definition": "(drop_high-drop_low)_local - (drop_high-drop_low)_hierarchical",
            }

    neutral_summary: dict[str, Any] = {}
    if not neutral.empty:
        for condition in sorted(neutral["condition"].unique()):
            sub = neutral[neutral["condition"] == condition]
            neutral_summary[condition] = {
                "p_a": _bootstrap_mean_ci(sub["p_a"].astype(float).to_numpy(), n_boot=bootstrap_samples, seed=seed + 41),
                "entropy": _bootstrap_mean_ci(sub["entropy"].astype(float).to_numpy(), n_boot=bootstrap_samples, seed=seed + 42),
            }

    # Internal stimulus-quality gate (computed on intact condition only).
    quality_gate = {
        "based_on_condition": "none",
        "context_a_accuracy": float("nan"),
        "context_b_accuracy": float("nan"),
        "both_correct_rate": float("nan"),
        "flip_rate": float("nan"),
        "n_variants_total": 0,
        "n_variants_by_relation_type": {},
        "thresholds": {
            "context_a_accuracy_gte": 0.85,
            "context_b_accuracy_gte": 0.85,
            "both_correct_rate_gte": 0.75,
            "flip_rate_gte": 0.75,
            "min_variants_per_relation_type": 8,
        },
        "passes_gate": False,
    }
    none_labeled = labeled[labeled["condition"] == "none"].copy()
    if not none_labeled.empty:
        pairs = none_labeled[none_labeled["context_label"].isin(["context_a", "context_b"])].copy()
        piv = pairs.pivot_table(
            index=["variant_id", "relation_type"],
            columns="context_label",
            values=["pred_option", "is_correct"],
            aggfunc="first",
        ).reset_index()
        if ("pred_option", "context_a") in piv.columns and ("pred_option", "context_b") in piv.columns:
            flips = (
                piv[("pred_option", "context_a")].astype(str)
                != piv[("pred_option", "context_b")].astype(str)
            ).to_numpy(dtype=bool)
            flip_rate = float(np.mean(flips)) if flips.size else float("nan")
        else:
            flip_rate = float("nan")

        if ("is_correct", "context_a") in piv.columns:
            acc_a = float(np.nanmean(piv[("is_correct", "context_a")].to_numpy(dtype=float)))
        else:
            acc_a = float("nan")
        if ("is_correct", "context_b") in piv.columns:
            acc_b = float(np.nanmean(piv[("is_correct", "context_b")].to_numpy(dtype=float)))
        else:
            acc_b = float("nan")

        if ("is_correct", "context_a") in piv.columns and ("is_correct", "context_b") in piv.columns:
            both = (
                (piv[("is_correct", "context_a")].to_numpy(dtype=float) > 0.5)
                & (piv[("is_correct", "context_b")].to_numpy(dtype=float) > 0.5)
            )
            both_rate = float(np.mean(both)) if both.size else float("nan")
        else:
            both_rate = float("nan")

        relation_counts = {
            str(rt): int(n)
            for rt, n in piv.groupby("relation_type", as_index=False).size().set_index("relation_type")["size"].to_dict().items()
        }
        min_per_relation = int(min(relation_counts.values())) if relation_counts else 0
        thr = quality_gate["thresholds"]
        passes_gate = bool(
            np.isfinite(acc_a)
            and np.isfinite(acc_b)
            and np.isfinite(both_rate)
            and np.isfinite(flip_rate)
            and (acc_a >= float(thr["context_a_accuracy_gte"]))
            and (acc_b >= float(thr["context_b_accuracy_gte"]))
            and (both_rate >= float(thr["both_correct_rate_gte"]))
            and (flip_rate >= float(thr["flip_rate_gte"]))
            and (min_per_relation >= int(thr["min_variants_per_relation_type"]))
        )
        quality_gate.update(
            {
                "context_a_accuracy": acc_a,
                "context_b_accuracy": acc_b,
                "both_correct_rate": both_rate,
                "flip_rate": flip_rate,
                "n_variants_total": int(len(piv)),
                "n_variants_by_relation_type": relation_counts,
                "passes_gate": passes_gate,
            }
        )

    # High-level verdict logic (exploratory; pre-specify thresholds in report).
    none_vs_high = paired_comparisons.get("none_vs_high_accuracy", {})
    none_vs_low = paired_comparisons.get("none_vs_low_accuracy", {})
    high_vs_low = paired_comparisons.get("high_vs_low_accuracy", {})

    high_drop = _safe_float(none_vs_high.get("mean_delta"))
    low_drop = _safe_float(none_vs_low.get("mean_delta"))
    high_vs_low_delta = _safe_float(high_vs_low.get("mean_delta"))

    if np.isfinite(high_drop) and np.isfinite(low_drop):
        if high_drop > low_drop + 0.02:
            involvement = "high_si_more_causal_than_low_si"
        elif low_drop > high_drop + 0.02:
            involvement = "low_si_more_causal_than_high_si"
        else:
            involvement = "similar_impact_high_vs_low"
    else:
        involvement = "inconclusive"

    if np.isfinite(high_vs_low_delta):
        if high_vs_low_delta < -0.02:
            directional = "ablate_high_si_worse_than_ablate_low_si"
        elif high_vs_low_delta > 0.02:
            directional = "ablate_low_si_worse_than_ablate_high_si"
        else:
            directional = "minimal_directional_difference"
    else:
        directional = "inconclusive"

    return {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": "tier3_publication_readiness",
        "primary_test_id": "Idea4_structural_ambiguity",
        "multiplicity_family": "tier3_exploratory_invariance",
        "bootstrap_samples": int(bootstrap_samples),
        "n_total_rows": int(len(rows)),
        "n_labeled_rows": int(len(labeled)),
        "n_neutral_rows": int(len(neutral)),
        "condition_summaries": condition_summaries,
        "paired_comparisons": paired_comparisons,
        "relation_type_interaction": interaction_payload,
        "internal_norming_quality_gate": quality_gate,
        "neutral_parse_preferences": neutral_summary,
        "verdict": {
            "syntactic_involvement": involvement,
            "directional_difference": directional,
            "interpretation_rule": {
                "involvement_threshold": "|drop_high-drop_low| > 0.02",
                "directional_threshold": "|acc_high-acc_low| > 0.02",
            },
        },
        "limitations": [
            "Stimuli are templated and may not cover all natural ambiguity phenomena.",
            "Binary-choice scoring uses completion log-likelihood, not full generation-based QA.",
            "Neutral-context parse preference is exploratory and sensitive to lexical priors.",
            "Internal norming gate improves stimulus quality but is not a substitute for external human norming.",
        ],
    }


def run_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    include_random_control: bool,
    bootstrap_samples: int,
    seed: int,
    max_stimuli: int | None,
    variant_allowlist: set[str] | None,
    variant_allowlist_path: str | None,
) -> dict[str, Any]:
    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stimuli = _build_stimuli()
    if variant_allowlist is not None:
        stimuli = [s for s in stimuli if str(s.variant_id) in variant_allowlist]
    if max_stimuli is not None and max_stimuli > 0:
        stimuli = stimuli[: int(max_stimuli)]
    _write_json(out_dir / "stimulus_manifest.json", {
        "experiment": "Idea4_structural_ambiguity",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_stimuli": int(len(stimuli)),
        "variant_filter": {
            "active": bool(variant_allowlist is not None),
            "allowlist_path": variant_allowlist_path,
            "n_allowed_variants": int(len(variant_allowlist)) if variant_allowlist is not None else None,
        },
        "families": sorted({s.family for s in stimuli}),
        "relation_types": sorted({s.relation_type for s in stimuli}),
        "contexts": sorted({s.context_label for s in stimuli}),
        "stimuli": [asdict(s) for s in stimuli],
    })

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)

    conditions = _load_conditions(model_name, include_random_control)

    rows: list[dict[str, Any]] = []
    total = len(stimuli)

    for condition_name, heads in conditions.items():
        print(
            f"[Idea4] model={model_name} condition={condition_name} "
            f"heads={len(heads)} total_stimuli={total}",
            flush=True,
        )
        t0 = time.time()
        cm: contextlib.AbstractContextManager
        if condition_name == "none":
            cm = contextlib.nullcontext()
        else:
            cm = head_output_ablation(model, heads)

        with cm:
            for idx, stim in enumerate(stimuli, start=1):
                sc = _score_binary_options(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=stim.prompt,
                    option_a=stim.option_a,
                    option_b=stim.option_b,
                )
                pred = "A" if sc["logprob_a"] >= sc["logprob_b"] else "B"
                correct_margin = float("nan")
                is_correct = float("nan")
                if stim.correct_option == "A":
                    correct_margin = sc["logprob_a"] - sc["logprob_b"]
                    is_correct = float(pred == "A")
                elif stim.correct_option == "B":
                    correct_margin = sc["logprob_b"] - sc["logprob_a"]
                    is_correct = float(pred == "B")

                rows.append({
                    "model": model_name,
                    "condition": condition_name,
                    "stimulus_id": stim.stimulus_id,
                    "variant_id": stim.variant_id,
                    "family": stim.family,
                    "relation_type": stim.relation_type,
                    "context_label": stim.context_label,
                    "correct_option": stim.correct_option,
                    "pred_option": pred,
                    "logprob_a": sc["logprob_a"],
                    "logprob_b": sc["logprob_b"],
                    "margin_a_minus_b": sc["margin_a_minus_b"],
                    "correct_margin": correct_margin,
                    "p_a": sc["p_a"],
                    "entropy": sc["entropy"],
                    "is_correct": is_correct,
                    "prompt": stim.prompt,
                    "option_a": stim.option_a,
                    "option_b": stim.option_b,
                })

                if idx == 1 or idx % 30 == 0 or idx == total:
                    elapsed = max(1e-6, time.time() - t0)
                    per = elapsed / idx
                    eta = per * (total - idx)
                    print(
                        f"  [{condition_name}] {idx}/{total} "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )

        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "per_item_scores.parquet", index=False)

    report = _build_report(
        model_name=model_name,
        rows=df,
        bootstrap_samples=max(500, int(bootstrap_samples)),
        seed=int(seed),
    )
    _write_json(out_dir / "ambiguity_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Idea 4: Structural ambiguity and SI-head intervention")
    p.add_argument("--model", choices=["all", *TARGET_MODELS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1",
        help="Comma-separated model:device map used when --model all",
    )
    p.add_argument("--include-random-control", action="store_true")
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-stimuli", type=int, default=0, help="Optional cap for smoke/debug runs; 0 means use all stimuli")
    p.add_argument(
        "--variant-allowlist",
        default="",
        help="Optional text file (one variant_id per line) to restrict stimuli for normed reruns.",
    )
    p.add_argument(
        "--output-root",
        default="results/experiment3_phase2/idea4_structural_ambiguity",
    )
    p.add_argument(
        "--fail-on-quality-gate",
        action="store_true",
        help="Exit non-zero if model-level internal_norming_quality_gate does not pass.",
    )
    return p.parse_args()


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [t.strip() for t in raw.split(",") if t.strip()]:
        model, device = tok.split(":", 1)
        out[model.strip()] = device.strip()
    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    variant_allowlist: set[str] | None = None
    variant_allowlist_path: str | None = None
    if str(args.variant_allowlist).strip():
        allow_path = Path(args.variant_allowlist)
        if not allow_path.exists():
            raise FileNotFoundError(f"Variant allowlist path does not exist: {allow_path}")
        variant_allowlist = set()
        with allow_path.open("r", encoding="utf-8") as f:
            for line in f:
                tok = line.strip()
                if tok and not tok.startswith("#"):
                    variant_allowlist.add(tok)
        if not variant_allowlist:
            raise RuntimeError(f"Variant allowlist is empty: {allow_path}")
        variant_allowlist_path = str(allow_path)

    models = list(TARGET_MODELS) if args.model == "all" else [args.model]
    device_map = _parse_device_map(args.device_map) if args.model == "all" else {}

    summary: dict[str, Any] = {}
    for model_name in models:
        device = device_map.get(model_name, args.device)
        print(f"[Idea4] model={model_name} device={device}", flush=True)
        rep = run_model(
            model_name=model_name,
            device=device,
            output_root=output_root,
            include_random_control=bool(args.include_random_control),
            bootstrap_samples=max(500, int(args.bootstrap_samples)),
            seed=int(args.seed),
            max_stimuli=(None if int(args.max_stimuli) <= 0 else int(args.max_stimuli)),
            variant_allowlist=variant_allowlist,
            variant_allowlist_path=variant_allowlist_path,
        )
        summary[model_name] = rep
        if bool(args.fail_on_quality_gate):
            q = rep.get("internal_norming_quality_gate", {})
            if not bool(q.get("passes_gate", False)):
                raise RuntimeError(
                    f"Idea4 quality gate failed for {model_name}: "
                    f"context_a={q.get('context_a_accuracy')}, "
                    f"context_b={q.get('context_b_accuracy')}, "
                    f"flip_rate={q.get('flip_rate')}, "
                    f"relation_counts={q.get('n_variants_by_relation_type')}"
                )

    _write_json(output_root / "ambiguity_summary.json", {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variant_allowlist_path": variant_allowlist_path,
        "n_allowed_variants": int(len(variant_allowlist)) if variant_allowlist is not None else None,
        "models": summary,
    })


if __name__ == "__main__":
    main()
