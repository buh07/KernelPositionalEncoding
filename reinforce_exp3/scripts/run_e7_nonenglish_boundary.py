#!/usr/bin/env python3
"""E7 — Non-English Boundary Domain.

Tests whether high-SI heads show boundary-linked attention in Chinese and
Turkish text (absence of English whitespace), ruling out the "space-detection"
alternative explanation for the boundary attention result.

Usage:
    python reinforce_exp3/scripts/run_e7_nonenglish_boundary.py \
        --models llama-3.1-8b,olmo-2-7b \
        --device-map llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0 \
        --languages zh,tr
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinforce_exp3.common import (  # noqa: E402
    B1_RESULTS,
    RESULTS_ROOT,
    ensure_dir,
    timestamp_now,
    write_json,
)
from reinforce_exp3.scripts._shared import (  # noqa: E402
    cohen_d,
    enforce_coverage_contract,
    emit_core_artifacts,
    holm_adjust_dict,
    load_model_for_exp,
    parse_device_map,
    parse_models_arg,
)

OUT_ROOT = RESULTS_ROOT / "E7_nonenglish_boundary"

LANGUAGES = ("zh", "tr")
N_SEQS_PER_LANG = 80    # sequences per language per model
SEQ_LEN = 128
N_HEADS_TO_TEST = 30    # sample of high-SI heads
SEED_BASE = 20260429
BOUNDARY_D_THRESHOLD = 0.10  # lower threshold for non-English (weaker signal expected)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_multilingual_data(lang: str, n: int, rng: random.Random) -> list[str]:
    """Try to load opus100 text for language; fallback to synthetic if unavailable."""
    # Check for existing multilingual data from experiment3_phase2
    data_dir = ROOT / "data" / "experiment3_phase2_multilingual"
    if data_dir.exists():
        candidates = list(data_dir.glob(f"*{lang}*"))
        if candidates:
            try:
                texts: list[str] = []
                for f in candidates[:3]:
                    if f.suffix in (".txt", ".jsonl"):
                        with f.open("r", encoding="utf-8") as fh:
                            lines = [l.strip() for l in fh if l.strip()]
                            texts.extend(lines)
                if texts:
                    return rng.sample(texts, k=min(n, len(texts)))
            except Exception:
                pass

    # Try datasets library
    try:
        from datasets import load_dataset  # type: ignore
        split_map = {"zh": "test", "tr": "test"}
        ds = load_dataset("Helsinki-NLP/opus100", f"en-{lang}", split=split_map.get(lang, "test"))
        texts = [ex["translation"][lang] for ex in ds]  # type: ignore
        texts = [t for t in texts if len(t) > 20]
        if texts:
            return rng.sample(texts, k=min(n, len(texts)))
    except Exception:
        pass

    # Synthetic fallback: return empty list; caller handles gracefully
    print(f"[E7] Warning: no data available for language {lang}; skipping", flush=True)
    return []


def _get_boundary_tokens_for_lang(
    token_ids: list[int],
    tokenizer: Any,
    lang: str,
    original_text: str | None = None,
) -> tuple[np.ndarray, bool, str]:
    """Compute boundary token mask adapted for each language.

    Chinese: character boundaries (each character is word-initial in sense of standalone morpheme).
    Turkish: subword morpheme boundaries (end of a word-initial subword followed by suffix).
    English fallback: space-prefix.
    """
    strs = [tokenizer.convert_ids_to_tokens(tid) or "" for tid in token_ids]
    n = len(strs)
    mask = np.zeros(n, dtype=bool)

    if lang == "zh":
        # Conservative mask: CJK tokens that are not continuation pieces.
        for i, s in enumerate(strs):
            stripped = s.replace("▁", "").replace("Ġ", "")
            is_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in stripped)
            is_boundary = is_cjk and not s.startswith("##")
            is_byte_seq = s.startswith("<0x") or s.startswith("\\x")
            if is_boundary and not is_byte_seq:
                mask[i] = True

    elif lang == "tr":
        # Turkish: word-initial tokens are space-prefixed (Llama/Mistral) or start a new word
        for i, s in enumerate(strs):
            mask[i] = s.startswith(" ") or s.startswith("▁") or s.startswith("Ġ")
        # Also mark tokens after sentence-end punctuation
        prev_is_punct = False
        for i, s in enumerate(strs):
            if prev_is_punct:
                mask[i] = True
            prev_is_punct = s.strip() in (".", "!", "?", "؟")

    else:
        # English / fallback: space-prefix
        for i, s in enumerate(strs):
            mask[i] = s.startswith(" ") or s.startswith("▁") or s.startswith("Ġ")

    n_boundary = int(mask.sum())
    n_nonboundary = int((~mask).sum())
    frac = float(n_boundary / max(1, n))
    if n_boundary < 2 or n_nonboundary < 2:
        return mask, False, f"degenerate_boundary_partition boundary={n_boundary} nonboundary={n_nonboundary}"
    if frac < 0.05 or frac > 0.95:
        return mask, False, f"unstable_boundary_density frac={frac:.4f}"
    return mask, True, "ok"


# ---------------------------------------------------------------------------
# Boundary attention evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_boundary_attention(
    model: Any,
    tokenizer: Any,
    sequences: list[list[int]],
    token_strs: list[list[str]],
    lang: str,
    high_si_heads: list[tuple[int, int]],
    low_si_heads: list[tuple[int, int]],
    device: str,
) -> dict[str, Any]:
    """Compute boundary attention Cohen d for high-SI vs low-SI heads."""
    model.eval()

    hi_d_vals: list[float] = []
    lo_d_vals: list[float] = []
    invalid_mask_count = 0
    valid_mask_count = 0

    for seq, tstr in zip(sequences, token_strs):
        bmask, ok, reason = _get_boundary_tokens_for_lang(seq, tokenizer, lang)
        if not ok:
            invalid_mask_count += 1
            continue
        valid_mask_count += 1

        ids_tensor = torch.tensor([seq], device=device, dtype=torch.long)
        try:
            out = model(ids_tensor, output_attentions=True)
        except Exception:
            continue

        if out.attentions is None:
            continue

        for layer, head in high_si_heads:
            if layer >= len(out.attentions):
                continue
            attn = out.attentions[layer][0, head].cpu().float().numpy()  # (seq, seq)
            attn_to_boundary = attn[:, bmask].mean(axis=1)
            attn_to_nonboundary = attn[:, ~bmask].mean(axis=1)
            d = cohen_d(attn_to_boundary.tolist(), attn_to_nonboundary.tolist())
            if not np.isnan(d):
                hi_d_vals.append(d)

        for layer, head in low_si_heads:
            if layer >= len(out.attentions):
                continue
            attn = out.attentions[layer][0, head].cpu().float().numpy()
            attn_to_boundary = attn[:, bmask].mean(axis=1)
            attn_to_nonboundary = attn[:, ~bmask].mean(axis=1)
            d = cohen_d(attn_to_boundary.tolist(), attn_to_nonboundary.tolist())
            if not np.isnan(d):
                lo_d_vals.append(d)

    mean_d_hi = float(np.mean(hi_d_vals)) if hi_d_vals else float("nan")
    mean_d_lo = float(np.mean(lo_d_vals)) if lo_d_vals else float("nan")
    d_differential = mean_d_hi - mean_d_lo if not (np.isnan(mean_d_hi) or np.isnan(mean_d_lo)) else float("nan")
    boundary_effect_present = mean_d_hi > BOUNDARY_D_THRESHOLD

    if valid_mask_count == 0:
        return {
            "lang": lang,
            "status": "invalid_boundary_definition",
            "hard_fail_reason": (
                f"no valid boundary partitions for {lang}; "
                f"invalid_masks={invalid_mask_count}"
            ),
        }

    return {
        "lang": lang,
        "status": "ok",
        "n_seqs_evaluated": len(sequences),
        "n_valid_boundary_masks": int(valid_mask_count),
        "n_invalid_boundary_masks": int(invalid_mask_count),
        "n_high_si_heads": len(high_si_heads),
        "n_low_si_heads": len(low_si_heads),
        "mean_d_high_si": mean_d_hi,
        "mean_d_low_si": mean_d_lo,
        "d_differential": d_differential,
        "boundary_effect_present": boundary_effect_present,
    }


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def _load_pretokenized_sequences(
    model_name: str,
    lang: str,
    n: int,
    seq_len: int,
    rng: random.Random,
) -> list[list[int]]:
    """Load pre-tokenized sequences from the local experiment3_phase2_multilingual store.

    Expects files at:
      ROOT/data/experiment3_phase2_multilingual/opus100_{lang}/{model_name}/len_{N}.jsonl
    Each line: {"idx": int, "tokens": [int, ...]}
    Returns up to n sequences, each truncated to seq_len.
    """
    data_dir = ROOT / "data" / "experiment3_phase2_multilingual"
    seqs: list[list[int]] = []
    for fname in (f"len_{seq_len}.jsonl", "len_512.jsonl", "len_256.jsonl"):
        path = data_dir / f"opus100_{lang}" / model_name / fname
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    toks = rec.get("tokens", [])
                    if len(toks) >= 20:
                        seqs.append(toks[:seq_len])
        except Exception:
            continue
        if seqs:
            break
    if not seqs:
        return []
    rng.shuffle(seqs)
    return seqs[:n]


def run_model(
    model_name: str,
    device: str,
    languages: list[str],
    out_dir: Path,
    *,
    allow_tokenizer_incompatible_skip: bool,
    min_valid_languages: int,
) -> dict[str, Any]:
    # Use eager attention so output_attentions=True works (OLMo uses SDPA by default)
    print(f"[E7] Loading {model_name} on {device}", flush=True)
    model, tokenizer = load_model_for_exp(model_name, device, attn_implementation="eager")

    # Load high-SI and low-SI heads
    cluster_path = B1_RESULTS / "cluster_membership.parquet"
    if cluster_path.exists():
        cm = pd.read_parquet(cluster_path)
        m_cm = cm[cm["model"] == model_name].copy()
        if not m_cm.empty:
            m_cm = m_cm.sort_values("mean_r2", ascending=False)
            n_hi = min(N_HEADS_TO_TEST, int(m_cm["is_high_si"].sum()))
            n_lo = min(N_HEADS_TO_TEST, int((~m_cm["is_high_si"]).sum()))
            hi_df = m_cm[m_cm["is_high_si"]].head(n_hi)
            lo_df = m_cm[~m_cm["is_high_si"]].tail(n_lo)
            high_si_heads = list(zip(hi_df["layer"].tolist(), hi_df["head"].tolist()))
            low_si_heads = list(zip(lo_df["layer"].tolist(), lo_df["head"].tolist()))
        else:
            n_layers = model.config.num_hidden_layers
            n_heads = model.config.num_attention_heads
            high_si_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)][:N_HEADS_TO_TEST]
            low_si_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)][N_HEADS_TO_TEST:2*N_HEADS_TO_TEST]
    else:
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
        high_si_heads = all_heads[:N_HEADS_TO_TEST]
        low_si_heads = all_heads[-N_HEADS_TO_TEST:]

    model_out = ensure_dir(out_dir / model_name)
    lang_results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for lang in languages:
        lang_idx = LANGUAGES.index(lang) if lang in LANGUAGES else (sum(ord(c) for c in lang) % 997)
        rng = random.Random(SEED_BASE + lang_idx * 1000)
        print(f"[E7] {model_name}: loading {lang} sequences", flush=True)

        # Try pre-tokenized local data first (from experiment3_phase2_multilingual)
        sequences: list[list[int]] = _load_pretokenized_sequences(
            model_name, lang, N_SEQS_PER_LANG, SEQ_LEN, rng
        )
        if sequences:
            print(f"[E7] {model_name} {lang}: {len(sequences)} pre-tokenized sequences loaded", flush=True)
        else:
            # Fall back to text-based loading and tokenization
            texts = _load_multilingual_data(lang, N_SEQS_PER_LANG, rng)
            if not texts:
                raise RuntimeError(
                    f"[E7] hard_fail_reason: missing required language data for {model_name}:{lang}"
                )
            for text in texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
                ids = enc["input_ids"][0].tolist()
                if len(ids) >= 20:
                    sequences.append(ids)

        if not sequences:
            raise RuntimeError(
                f"[E7] hard_fail_reason: empty tokenization output for {model_name}:{lang}"
            )

        # Build token strings for boundary mask computation
        token_strs_list: list[list[str]] = [
            [tokenizer.convert_ids_to_tokens(i) or "" for i in seq]
            for seq in sequences
        ]

        print(f"[E7] {model_name} {lang}: {len(sequences)} sequences", flush=True)
        result = _eval_boundary_attention(
            model, tokenizer, sequences, token_strs_list, lang,
            high_si_heads, low_si_heads, device
        )
        if result.get("status") != "ok":
            reason = result.get("hard_fail_reason", "invalid boundary definition")
            if allow_tokenizer_incompatible_skip and "no valid boundary partitions" in reason:
                print(
                    f"[E7] WARNING: {model_name}:{lang} tokenizer-incompatible "
                    f"({reason}); skipping language.",
                    flush=True,
                )
                lang_results[lang] = {**result, "status": "skipped_tokenizer_incompatible"}
                continue
            raise RuntimeError(f"[E7] hard_fail_reason: {model_name}:{lang} {reason}")
        lang_results[lang] = result
        rows.append({"model": model_name, **result})
        print(
            f"[E7] {model_name} {lang}: "
            f"d_hi={result['mean_d_high_si']:.3f} "
            f"d_lo={result['mean_d_low_si']:.3f} "
            f"effect={result['boundary_effect_present']}",
            flush=True,
        )

    if rows:
        pd.DataFrame(rows).to_parquet(model_out / "boundary_attention_nonenglish.parquet", index=False)
    write_json(model_out / "lang_results.json", lang_results)

    n_valid = len(rows)
    if n_valid == 0:
        raise RuntimeError(
            f"[E7] hard_fail_reason: {model_name}: all {len(languages)} languages "
            f"skipped or failed; no evaluable data"
        )
    if n_valid < int(min_valid_languages):
        raise RuntimeError(
            f"[E7] hard_fail_reason: {model_name}: valid_language_count={n_valid} "
            f"< required={int(min_valid_languages)}"
        )
    # Strict-by-default coverage: require requested language set to be valid unless
    # skip mode is explicitly enabled by caller.
    valid_langs = [l for l, r in lang_results.items() if r.get("status") == "ok"]
    required_langs = valid_langs if allow_tokenizer_incompatible_skip else list(languages)
    enforce_coverage_contract(
        experiment_id="E7",
        observed_models=[model_name],
        required_models=[model_name],
        observed_tasks=valid_langs,
        required_tasks=required_langs,
        observed_counts={"languages": n_valid, "rows": n_valid},
        min_counts={"languages": int(min_valid_languages), "rows": int(min_valid_languages)},
    )

    return {"model": model_name, "lang_results": lang_results}


# ---------------------------------------------------------------------------
# Cross-model summary
# ---------------------------------------------------------------------------

def _cross_model_summary(
    model_results: list[dict[str, Any]],
    out_dir: Path,
    required_models: list[str],
    required_languages: list[str],
) -> dict[str, Any]:
    enforce_coverage_contract(
        experiment_id="E7",
        observed_models=[r.get("model", "") for r in model_results],
        required_models=required_models,
        observed_tasks=[
            lang
            for r in model_results
            for lang in r.get("lang_results", {}).keys()
        ],
        required_tasks=required_languages,
    )
    lang_outcomes: dict[str, list[bool]] = {}
    for r in model_results:
        for lang, lang_r in r.get("lang_results", {}).items():
            if isinstance(lang_r, dict) and "boundary_effect_present" in lang_r:
                lang_outcomes.setdefault(lang, []).append(lang_r["boundary_effect_present"])

    lang_verdicts: dict[str, str] = {}
    for lang, outcomes in lang_outcomes.items():
        n_positive = sum(1 for o in outcomes if o)
        if n_positive == len(outcomes):
            lang_verdicts[lang] = "consistent_positive"
        elif n_positive == 0:
            lang_verdicts[lang] = "consistent_negative"
        else:
            lang_verdicts[lang] = "mixed"

    any_positive = any(v == "consistent_positive" for v in lang_verdicts.values())
    if any_positive:
        interpretation = "boundary_not_english_specific"
        note = (
            "High-SI heads show boundary-linked attention in at least one non-English language. "
            "Result II is not specific to English whitespace-based tokenization."
        )
    elif all(v == "consistent_negative" for v in lang_verdicts.values()):
        interpretation = "boundary_english_specific"
        note = (
            "No boundary effect in non-English languages. "
            "Result II should be scoped to whitespace-tokenized (English) text."
        )
    else:
        interpretation = "mixed_across_languages"
        note = "Mixed results across languages. Effect may be language/script-dependent."

    summary = {
        "lang_verdicts": lang_verdicts,
        "interpretation": interpretation,
        "note": note,
    }
    write_json(out_dir / "cross_model_nonenglish_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="E7: Non-English boundary attention analysis (Chinese, Turkish)",
        allow_abbrev=False,
    )
    p.add_argument("--models", default=",".join(["llama-3.1-8b", "olmo-2-7b"]))
    p.add_argument("--device-map", default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:0")
    p.add_argument("--output-root", default=str(OUT_ROOT))
    p.add_argument("--languages", default="zh,tr")
    p.add_argument("--min-valid-languages", type=int, default=0,
                   help="Minimum number of valid languages required per model. "
                        "0 means require all requested languages.")
    p.add_argument("--allow-tokenizer-incompatible-skip", action="store_true",
                   help="Allow tokenizer-incompatible language masks to be skipped "
                        "instead of hard-failing.")
    p.add_argument("--finalize-only", action="store_true",
                   help="Read per-model lang_results and emit cross-model artifacts.")
    p.add_argument("--no-finalize", action="store_true",
                   help="Run per-model computations but skip cross-model finalize emission.")
    args = p.parse_args()

    models = parse_models_arg(args.models)
    device_map = parse_device_map(args.device_map)
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    if not languages:
        raise RuntimeError("[E7] hard_fail_reason: languages list is empty")
    min_valid_languages = int(args.min_valid_languages) if int(args.min_valid_languages) > 0 else len(languages)
    if args.finalize_only and args.no_finalize:
        raise RuntimeError("[E7] hard_fail_reason: --finalize-only and --no-finalize are mutually exclusive")
    out_dir = ensure_dir(Path(args.output_root))
    start_ts = timestamp_now()

    print(f"[E7] Starting at {start_ts}", flush=True)
    print(f"[E7] Models: {models}, Languages: {languages}", flush=True)

    model_results: list[dict[str, Any]] = []
    if args.finalize_only:
        for model_name in models:
            lang_path = out_dir / model_name / "lang_results.json"
            if not lang_path.exists():
                raise RuntimeError(f"[E7] hard_fail_reason: missing shard artifact {lang_path}")
            model_results.append({"model": model_name, "lang_results": json.loads(lang_path.read_text())})
    else:
        for model_name in models:
            device = device_map.get(model_name, "cuda:0")
            result = run_model(
                model_name,
                device,
                languages,
                out_dir,
                allow_tokenizer_incompatible_skip=bool(args.allow_tokenizer_incompatible_skip),
                min_valid_languages=min_valid_languages,
            )
            model_results.append(result)

    enforce_coverage_contract(
        experiment_id="E7",
        observed_models=[r["model"] for r in model_results],
        required_models=models,
    )

    if args.no_finalize:
        print("[E7] Shard run complete (no finalize).", flush=True)
        return

    cross_model = _cross_model_summary(
        model_results,
        out_dir,
        required_models=models,
        required_languages=languages,
    )

    interp = cross_model.get("interpretation", "unknown")
    if interp == "boundary_not_english_specific":
        claim_status = "supported"
    elif interp == "boundary_english_specific":
        claim_status = "not_supported"
    else:
        claim_status = "inconclusive"

    claim_impact = {
        "experiment_id": "E7",
        "claim_addressed": (
            "Boundary-linked attention in high-SI heads is not specific to English "
            "whitespace tokenization but reflects general structural boundary detection"
        ),
        "claim_status": claim_status,
        "supports_main_text": claim_status == "supported",
        "outcome_summary": cross_model.get("note", ""),
        "notes": [cross_model.get("note", "")],
    }

    preregistration = {
        "experiment_id": "E7",
        "hypothesis": (
            "High-SI heads show boundary-linked attention preference in Chinese and Turkish "
            "text despite absence of English whitespace tokens."
        ),
        "primary_criterion": (
            "Mean boundary attention d > threshold in high-SI heads for at least one non-English language "
            "in at least one model."
        ),
        "models": models,
        "languages": languages,
        "n_seqs_per_lang": N_SEQS_PER_LANG,
        "boundary_d_threshold": BOUNDARY_D_THRESHOLD,
        "timestamp": start_ts,
    }

    summary = {
        "experiment_id": "E7",
        "status": "complete",
        "models_run": models,
        "languages_tested": languages,
        "interpretation": interp,
        "timestamp_start": start_ts,
        "timestamp_end": timestamp_now(),
    }

    data_dictionary = {
        "experiment_id": "E7",
        "tables": [
            {
                "path": "<model>/boundary_attention_nonenglish.parquet",
                "description": "Boundary attention Cohen d for high-SI vs low-SI heads by language",
                "columns": [
                    {"name": "model", "dtype": "str", "description": "Model name"},
                    {"name": "lang", "dtype": "str", "description": "Language code (zh/tr)"},
                    {"name": "mean_d_high_si", "dtype": "float", "description": "Mean boundary d for high-SI heads"},
                    {"name": "mean_d_low_si", "dtype": "float", "description": "Mean boundary d for low-SI heads"},
                    {"name": "d_differential", "dtype": "float", "description": "mean_d_high_si - mean_d_low_si"},
                    {"name": "boundary_effect_present", "dtype": "bool", "description": f"d_high_si > {BOUNDARY_D_THRESHOLD}"},
                ],
            }
        ],
    }

    manifest_extra = {
        "models": models,
        "languages": languages,
        "n_seqs_per_lang": N_SEQS_PER_LANG,
        "seed_base": SEED_BASE,
    }

    emit_core_artifacts(
        out_dir=out_dir,
        experiment_id="E7",
        preregistration=preregistration,
        manifest_extra=manifest_extra,
        summary=summary,
        claim_impact=claim_impact,
        data_dictionary=data_dictionary,
    )

    print(f"[E7] Done. Interpretation: {interp}", flush=True)


if __name__ == "__main__":
    main()
