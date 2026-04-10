#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import torch

try:
    from sklearn.linear_model import SGDClassifier
except Exception:  # pragma: no cover
    SGDClassifier = None
try:
    import nltk
except Exception:  # pragma: no cover
    nltk = None

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment2.tasks import TaskExample, TokenPools, build_token_pools, generate_task_examples  # noqa: E402
from experiment3.stats_utils import holm_adjust  # noqa: E402
from experiment3.theory1_si_circuits import (  # noqa: E402
    MODELS,
    RETRIEVAL_SPANS,
    HeadID,
    evaluate_task_battery,
    head_output_ablation,
    load_profile_sequences,
)
from shared.models.loading import load_model, load_tokenizer  # noqa: E402


TARGET_MODELS = ("llama-3.1-8b", "olmo-2-7b")
LAYER_SET = (0, 4, 8, 12, 16, 20, 24, 28, 31)
PRIMARY_TEST_ID = "3P2-A"
TIER_LABEL = "tier2_conditional_mechanistic"
MULTIPLICITY = "tier2_holm_primary_tests"

POS_LABELS = {0: "noun", 1: "verb", 2: "adj", 3: "other"}

NOUN_LEXICON = {
    "time", "year", "people", "way", "day", "man", "woman", "child", "world", "school",
    "state", "family", "student", "group", "country", "problem", "hand", "part", "place", "case",
    "week", "company", "system", "program", "question", "work", "government", "number", "night", "point",
    "home", "water", "room", "mother", "area", "money", "story", "fact", "month", "lot",
    "right", "study", "book", "eye", "job", "word", "business", "issue", "side", "kind",
    "head", "house", "service", "friend", "father", "power", "hour", "game", "line", "end",
}
VERB_LEXICON = {
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "say", "says", "said", "go", "goes", "went", "gone", "make", "makes", "made", "know", "knows",
    "knew", "think", "thinks", "thought", "take", "takes", "took", "taken", "see", "sees", "saw",
    "seen", "want", "wants", "wanted", "use", "uses", "used", "find", "finds", "found", "give",
    "gives", "gave", "given", "tell", "tells", "told", "work", "works", "worked", "call", "calls",
    "called", "try", "tries", "tried", "ask", "asks", "asked", "need", "needs", "needed",
}
ADJ_LEXICON = {
    "good", "new", "first", "last", "long", "great", "little", "own", "other", "old", "right",
    "big", "high", "different", "small", "large", "next", "early", "young", "important", "few",
    "public", "bad", "same", "able", "best", "better", "certain", "clear", "close", "common",
    "easy", "free", "full", "hard", "local", "major", "national", "open", "possible", "real",
    "recent", "short", "simple", "single", "special", "strong", "true", "whole", "wide", "final",
}

POS_BACKENDS = ("auto", "heuristic", "nltk")
PENN_NOUN_PREFIXES = ("NN", "PRP", "WP")
PENN_VERB_PREFIXES = ("VB", "MD")
PENN_ADJ_PREFIXES = ("JJ",)


@dataclass(frozen=True)
class ChoiceExample:
    example_id: str
    prompt_ids: list[int]
    option_a_ids: list[int]
    option_b_ids: list[int]
    correct_option: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    category: str
    kind: str
    floor: float
    task_name: str | None = None
    span: int | None = None
    variant: str | None = None


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha1_of_obj(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(body).hexdigest()


def _token_has_prefix(tok: str) -> bool:
    return tok.startswith("\u0120") or tok.startswith("\u2581")


def _strip_token(tok: str) -> str:
    t = tok.replace("\u0120", "").replace("\u2581", "")
    t = t.strip().lower()
    t = re.sub(r"[^a-z]", "", t)
    return t


def _pos_label_from_token(tok: str) -> int:
    w = _strip_token(tok)
    if not w:
        return 3
    if w in NOUN_LEXICON:
        return 0
    if w in VERB_LEXICON:
        return 1
    if w in ADJ_LEXICON:
        return 2
    if any(w.endswith(suf) for suf in ("ness", "tion", "ment", "ship", "ity", "age", "ism", "er", "or")):
        return 0
    if any(w.endswith(suf) for suf in ("ed", "ing", "en", "ify", "ise", "ize")):
        return 1
    if any(w.endswith(suf) for suf in ("ous", "ful", "ive", "al", "ic", "able", "ible", "less", "y")):
        return 2
    return 3


def _penn_to_coarse_pos(tag: str) -> int:
    t = str(tag or "").upper()
    if any(t.startswith(p) for p in PENN_NOUN_PREFIXES):
        return 0
    if any(t.startswith(p) for p in PENN_VERB_PREFIXES):
        return 1
    if any(t.startswith(p) for p in PENN_ADJ_PREFIXES):
        return 2
    return 3


def _token_wordpieces(tokenizer, token_ids: list[int]) -> tuple[list[str], list[int], list[str]]:
    tok_strs = tokenizer.convert_ids_to_tokens(token_ids)
    words: list[str] = []
    tok_to_word: list[int] = [-1] * len(tok_strs)
    for i, tok in enumerate(tok_strs):
        cur = str(tok or "")
        piece = _strip_token(cur)
        if not piece:
            continue
        starts_new = bool(i == 0 or _token_has_prefix(cur) or tok_to_word[i - 1] < 0)
        if starts_new:
            words.append(piece)
            tok_to_word[i] = len(words) - 1
        else:
            wi = int(tok_to_word[i - 1])
            words[wi] = f"{words[wi]}{piece}"
            tok_to_word[i] = wi
    return tok_strs, tok_to_word, words


def _ensure_nltk_tagger(allow_download: bool) -> tuple[bool, str]:
    if nltk is None:
        return False, "nltk_not_installed"
    paths = ("taggers/averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger")
    for p in paths:
        try:
            nltk.data.find(p)
            return True, f"found:{p}"
        except Exception:
            continue
    if not allow_download:
        return False, "nltk_tagger_missing"
    for pkg in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    for p in paths:
        try:
            nltk.data.find(p)
            return True, f"downloaded:{p}"
        except Exception:
            continue
    return False, "nltk_tagger_unavailable_after_download"


def _coarse_pos_from_nltk(tokenizer, token_ids: list[int], *, allow_download: bool) -> tuple[np.ndarray, str]:
    tok_strs, tok_to_word, words = _token_wordpieces(tokenizer, token_ids)
    if len(tok_strs) == 0:
        return np.zeros((0,), dtype=np.int64), "empty_sequence"

    ok, note = _ensure_nltk_tagger(allow_download=allow_download)
    if (not ok) or (nltk is None):
        pos = np.array([_pos_label_from_token(tok_strs[i] or "") for i in range(len(tok_strs))], dtype=np.int64)
        return pos, f"heuristic_fallback:{note}"

    try:
        tagged = nltk.pos_tag(words)
    except Exception:
        pos = np.array([_pos_label_from_token(tok_strs[i] or "") for i in range(len(tok_strs))], dtype=np.int64)
        return pos, "heuristic_fallback:nltk_pos_tag_error"

    word_labels = np.array([_penn_to_coarse_pos(tag) for _, tag in tagged], dtype=np.int64)
    pos = np.full((len(tok_strs),), 3, dtype=np.int64)
    for i, wi in enumerate(tok_to_word):
        if wi >= 0 and wi < len(word_labels):
            pos[i] = int(word_labels[wi])
        else:
            pos[i] = int(_pos_label_from_token(tok_strs[i] or ""))
    return pos, "nltk_pos_tag"


def _load_head_groups(model_name: str) -> dict[str, list[HeadID]]:
    path = ROOT / "results" / "experiment3" / "theory1_si_circuits" / model_name / "head_groups.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing head_groups.json for {model_name}: {path}")
    data = _load_json(path)

    def parse(entries: list[dict[str, Any]]) -> list[HeadID]:
        return [HeadID(int(e["layer"]), int(e["head"])) for e in entries]

    out = {
        "none": [],
        "ablate_high_si": parse(data["high_si"]),
        "ablate_low_si": parse(data["low_si"]),
    }
    for draw in range(3):
        key = f"random_draw{draw}"
        if key not in data:
            raise RuntimeError(f"{key} missing in {path}")
        out[f"ablate_random_draw{draw}"] = parse(data[key])
    return out


def _build_task_specs(retrieval_spans: tuple[int, ...]) -> list[TaskSpec]:
    span_small = int(min(retrieval_spans))
    span_large = int(max(retrieval_spans))
    span_bridge = int(max(64, span_large))

    return [
        TaskSpec(
            task_id=f"pos_retrieval_span{span_small}",
            category="positional",
            kind="synthetic",
            floor=0.10,
            task_name="long_range_retrieval",
            span=span_small,
        ),
        TaskSpec(
            task_id=f"pos_retrieval_span{span_large}",
            category="positional",
            kind="synthetic",
            floor=0.10,
            task_name="long_range_retrieval",
            span=span_large,
        ),
        TaskSpec(
            task_id="pos_local_key_match",
            category="positional",
            kind="synthetic",
            floor=0.10,
            task_name="local_key_match",
            span=0,
        ),
        TaskSpec(
            task_id=f"pos_copy_offset_bridge{span_bridge}",
            category="positional",
            kind="synthetic",
            floor=0.10,
            task_name="copy_offset_bridge",
            span=span_bridge,
        ),
        TaskSpec(
            task_id="boundary_word_initial_choice",
            category="boundary",
            kind="choice",
            floor=0.50,
            variant="word_initial",
        ),
        TaskSpec(
            task_id="boundary_continuation_choice",
            category="boundary",
            kind="choice",
            floor=0.50,
            variant="continuation",
        ),
        TaskSpec(
            task_id="syntax_subject_verb_agreement",
            category="syntactic",
            kind="choice",
            floor=0.50,
            variant="subject_verb",
        ),
        TaskSpec(
            task_id="syntax_pronoun_resolution",
            category="syntactic",
            kind="choice",
            floor=0.50,
            variant="pronoun",
        ),
        TaskSpec(
            task_id="factual_capitals",
            category="factual",
            kind="choice",
            floor=0.50,
            variant="capitals",
        ),
        TaskSpec(
            task_id="factual_world_knowledge",
            category="factual",
            kind="choice",
            floor=0.50,
            variant="world_knowledge",
        ),
    ]


def _build_positional_task_configs(task_specs: list[TaskSpec]) -> list[tuple[str, int | None, tuple[int, ...] | None, str]]:
    out: list[tuple[str, int | None, tuple[int, ...] | None, str]] = []
    for spec in task_specs:
        if spec.kind != "synthetic":
            continue
        if spec.task_name == "long_range_retrieval":
            span = int(spec.span or 0)
            out.append((spec.task_name, span, (span,), spec.task_id))
        elif spec.task_name == "copy_offset_bridge":
            span = int(spec.span or 64)
            out.append((spec.task_name, span, (span,), spec.task_id))
        elif spec.task_name == "local_key_match":
            out.append((spec.task_name, None, None, spec.task_id))
        else:
            raise ValueError(f"Unsupported synthetic task {spec.task_name}")
    return out


def _task_def_manifest(model_name: str, task_specs: list[TaskSpec], retrieval_spans: tuple[int, ...]) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "retrieval_spans": [int(x) for x in retrieval_spans],
        "task_specs": [
            {
                "task_id": t.task_id,
                "category": t.category,
                "kind": t.kind,
                "task_name": t.task_name,
                "span": t.span,
                "variant": t.variant,
                "floor": t.floor,
            }
            for t in task_specs
        ],
    }
    payload["task_definition_hash"] = _sha1_of_obj(payload)
    payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return payload


def _score_choice_example(
    *,
    model,
    tokenizer,
    device: str,
    example: ChoiceExample,
) -> tuple[float, float, str]:
    prompt_ids = example.prompt_ids
    a_ids = example.option_a_ids
    b_ids = example.option_b_ids

    seq_a = prompt_ids + a_ids
    seq_b = prompt_ids + b_ids

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    max_len = max(len(seq_a), len(seq_b))
    input_ids = torch.full((2, max_len), int(pad_id), dtype=torch.long, device=device)
    attn = torch.zeros((2, max_len), dtype=torch.long, device=device)

    input_ids[0, : len(seq_a)] = torch.tensor(seq_a, dtype=torch.long, device=device)
    input_ids[1, : len(seq_b)] = torch.tensor(seq_b, dtype=torch.long, device=device)
    attn[0, : len(seq_a)] = 1
    attn[1, : len(seq_b)] = 1

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        lp = torch.log_softmax(out.logits.float(), dim=-1)

    def completion_logprob(row: int, completion_ids: list[int]) -> float:
        start = len(prompt_ids)
        total = 0.0
        for j, tok in enumerate(completion_ids):
            pos = start + j - 1
            if pos < 0 or pos >= lp.shape[1]:
                continue
            total += float(lp[row, pos, int(tok)].item())
        return total

    lpa = completion_logprob(0, a_ids)
    lpb = completion_logprob(1, b_ids)
    pred = "A" if lpa >= lpb else "B"
    correct = float(pred == example.correct_option)
    margin = float((lpa - lpb) if example.correct_option == "A" else (lpb - lpa))
    return correct, margin, pred


def _make_choice_example(tokenizer, prompt: str, option_a: str, option_b: str, correct: str, ex_id: str, metadata: dict[str, Any]) -> ChoiceExample | None:
    p = tokenizer.encode(prompt, add_special_tokens=False)
    a = tokenizer.encode(option_a, add_special_tokens=False)
    b = tokenizer.encode(option_b, add_special_tokens=False)
    if not p or not a or not b:
        return None
    return ChoiceExample(
        example_id=ex_id,
        prompt_ids=[int(x) for x in p],
        option_a_ids=[int(x) for x in a],
        option_b_ids=[int(x) for x in b],
        correct_option=correct,
        metadata=metadata,
    )


def _build_boundary_choice_examples(
    *,
    tokenizer,
    model_name: str,
    seq_len: int,
    n_examples: int,
    seed: int,
    variant: str,
) -> list[ChoiceExample]:
    rng = random.Random(seed + (11 if variant == "word_initial" else 29))
    seqs = load_profile_sequences(tokenizer=tokenizer, model_name=model_name, num_sequences=max(120, n_examples * 2), seq_len=seq_len)
    if not seqs:
        raise RuntimeError(f"No sequences loaded for boundary task ({model_name})")

    token_pool_initial: list[int] = []
    token_pool_cont: list[int] = []
    for tokens in seqs:
        tok_strs = tokenizer.convert_ids_to_tokens(tokens)
        for i, tid in enumerate(tokens):
            if i == 0:
                continue
            if _token_has_prefix(tok_strs[i] or ""):
                token_pool_initial.append(int(tid))
            else:
                token_pool_cont.append(int(tid))

    if not token_pool_initial or not token_pool_cont:
        raise RuntimeError("Boundary task failed to build token pools")

    out: list[ChoiceExample] = []
    max_attempts = n_examples * 20
    attempts = 0

    while len(out) < n_examples and attempts < max_attempts:
        attempts += 1
        seq_idx = rng.randrange(len(seqs))
        tokens = seqs[seq_idx]
        tok_strs = tokenizer.convert_ids_to_tokens(tokens)
        valid_pos: list[int] = []
        for t in range(2, len(tokens) - 1):
            is_initial = _token_has_prefix(tok_strs[t] or "")
            if variant == "word_initial" and is_initial:
                valid_pos.append(t)
            elif variant == "continuation" and (not is_initial):
                valid_pos.append(t)
        if not valid_pos:
            continue

        t = rng.choice(valid_pos)
        true_tok = int(tokens[t])
        if variant == "word_initial":
            pool = token_pool_initial
        else:
            pool = token_pool_cont

        distractor = true_tok
        for _ in range(20):
            cand = int(pool[rng.randrange(len(pool))])
            if cand != true_tok:
                distractor = cand
                break
        if distractor == true_tok:
            continue

        prompt = tokenizer.decode(tokens[:t], clean_up_tokenization_spaces=False)
        opt_true = tokenizer.decode([true_tok], clean_up_tokenization_spaces=False)
        opt_false = tokenizer.decode([distractor], clean_up_tokenization_spaces=False)
        if (not opt_true.strip()) or (not opt_false.strip()):
            continue

        if rng.random() < 0.5:
            ex = _make_choice_example(
                tokenizer,
                prompt=prompt,
                option_a=opt_true,
                option_b=opt_false,
                correct="A",
                ex_id=f"boundary_{variant}_{seed}_{len(out):04d}",
                metadata={"variant": variant, "position": int(t), "sequence_idx": int(seq_idx)},
            )
        else:
            ex = _make_choice_example(
                tokenizer,
                prompt=prompt,
                option_a=opt_false,
                option_b=opt_true,
                correct="B",
                ex_id=f"boundary_{variant}_{seed}_{len(out):04d}",
                metadata={"variant": variant, "position": int(t), "sequence_idx": int(seq_idx)},
            )
        if ex is not None:
            out.append(ex)

    min_required = max(4, int(math.ceil(float(n_examples) * 0.6)))
    if len(out) < min_required:
        raise RuntimeError(
            f"Boundary task {variant} produced too few examples: {len(out)}/{n_examples} "
            f"(minimum required {min_required})"
        )

    # If we have enough to proceed but fewer than requested, recycle deterministically.
    if len(out) < n_examples:
        base = list(out)
        idx = 0
        while len(out) < n_examples:
            src = base[idx % len(base)]
            out.append(
                ChoiceExample(
                    example_id=f"{src.example_id}_rep{idx:03d}",
                    prompt_ids=list(src.prompt_ids),
                    option_a_ids=list(src.option_a_ids),
                    option_b_ids=list(src.option_b_ids),
                    correct_option=str(src.correct_option),
                    metadata=dict(src.metadata),
                )
            )
            idx += 1

    return out[:n_examples]


def _build_subject_verb_examples(tokenizer, n_examples: int, seed: int) -> list[ChoiceExample]:
    rng = random.Random(seed + 101)
    singular_subjects = [
        "author", "teacher", "pilot", "doctor", "artist", "captain", "runner", "manager", "scientist", "driver",
    ]
    plural_subjects = [
        "authors", "teachers", "pilots", "doctors", "artists", "captains", "runners", "managers", "scientists", "drivers",
    ]
    objects = ["students", "neighbors", "friends", "workers", "visitors", "children", "analysts", "journalists"]

    out: list[ChoiceExample] = []
    for i in range(n_examples):
        singular = bool(rng.random() < 0.5)
        subj = rng.choice(singular_subjects if singular else plural_subjects)
        obj = rng.choice(objects)
        prompt = f"The {subj} near the {obj}"
        opt_is = " is"
        opt_are = " are"
        if singular:
            a, b, corr = opt_is, opt_are, "A"
        else:
            a, b, corr = opt_are, opt_is, "A"
        if rng.random() < 0.5:
            a, b = b, a
            corr = "B"
        ex = _make_choice_example(
            tokenizer,
            prompt=prompt,
            option_a=a,
            option_b=b,
            correct=corr,
            ex_id=f"syntax_sv_{seed}_{i:04d}",
            metadata={"subject": subj, "singular": singular},
        )
        if ex is not None:
            out.append(ex)
    return out


def _build_pronoun_examples(tokenizer, n_examples: int, seed: int) -> list[ChoiceExample]:
    rng = random.Random(seed + 151)
    male = ["John", "David", "Michael", "Robert", "Daniel", "Thomas", "James", "Mark"]
    female = ["Mary", "Susan", "Linda", "Patricia", "Jennifer", "Elizabeth", "Sarah", "Anna"]
    actions = ["was late", "won the race", "missed the train", "finished early", "solved the puzzle"]

    out: list[ChoiceExample] = []
    for i in range(n_examples):
        use_male = bool(rng.random() < 0.5)
        if use_male:
            subj = rng.choice(male)
            other = rng.choice(female)
            correct_pron = " he"
            wrong_pron = " she"
        else:
            subj = rng.choice(female)
            other = rng.choice(male)
            correct_pron = " she"
            wrong_pron = " he"
        action = rng.choice(actions)
        prompt = f"{subj} called {other} and {subj} said that"
        opt_true = f"{correct_pron} {action}."
        opt_false = f"{wrong_pron} {action}."

        if rng.random() < 0.5:
            a, b, corr = opt_true, opt_false, "A"
        else:
            a, b, corr = opt_false, opt_true, "B"

        ex = _make_choice_example(
            tokenizer,
            prompt=prompt,
            option_a=a,
            option_b=b,
            correct=corr,
            ex_id=f"syntax_pron_{seed}_{i:04d}",
            metadata={"subject": subj, "other": other, "use_male": use_male},
        )
        if ex is not None:
            out.append(ex)
    return out


def _build_capital_examples(tokenizer, n_examples: int, seed: int) -> list[ChoiceExample]:
    rng = random.Random(seed + 181)
    facts = [
        ("France", " Paris", " Lyon"),
        ("Japan", " Tokyo", " Osaka"),
        ("Italy", " Rome", " Milan"),
        ("Germany", " Berlin", " Munich"),
        ("Canada", " Ottawa", " Toronto"),
        ("Australia", " Canberra", " Sydney"),
        ("Spain", " Madrid", " Barcelona"),
        ("Brazil", " Brasilia", " Rio de Janeiro"),
        ("India", " New Delhi", " Mumbai"),
        ("Turkey", " Ankara", " Istanbul"),
        ("Egypt", " Cairo", " Alexandria"),
        ("Mexico", " Mexico City", " Guadalajara"),
    ]

    out: list[ChoiceExample] = []
    for i in range(n_examples):
        country, correct, wrong = facts[i % len(facts)]
        prompt = f"The capital of {country} is"
        if rng.random() < 0.5:
            a, b, corr = correct, wrong, "A"
        else:
            a, b, corr = wrong, correct, "B"
        ex = _make_choice_example(
            tokenizer,
            prompt=prompt,
            option_a=a,
            option_b=b,
            correct=corr,
            ex_id=f"factual_cap_{seed}_{i:04d}",
            metadata={"country": country},
        )
        if ex is not None:
            out.append(ex)
    return out


def _build_world_knowledge_examples(tokenizer, n_examples: int, seed: int) -> list[ChoiceExample]:
    rng = random.Random(seed + 211)
    facts = [
        ("The chemical symbol for gold is", " Au", " Ag"),
        ("The largest planet in the solar system is", " Jupiter", " Mars"),
        ("Water freezes at", " 0", " 100"),
        ("The author of Hamlet is", " Shakespeare", " Dickens"),
        ("The process plants use to make food is", " photosynthesis", " respiration"),
        ("The first month of the year is", " January", " March"),
        ("The square root of 81 is", " 9", " 8"),
        ("The fastest land animal is the", " cheetah", " elephant"),
        ("The primary gas in Earth's atmosphere is", " nitrogen", " oxygen"),
        ("The currency used in Japan is the", " yen", " euro"),
    ]

    out: list[ChoiceExample] = []
    for i in range(n_examples):
        prompt, correct, wrong = facts[i % len(facts)]
        if rng.random() < 0.5:
            a, b, corr = correct, wrong, "A"
        else:
            a, b, corr = wrong, correct, "B"
        ex = _make_choice_example(
            tokenizer,
            prompt=prompt,
            option_a=a,
            option_b=b,
            correct=corr,
            ex_id=f"factual_wk_{seed}_{i:04d}",
            metadata={"fact_id": int(i % len(facts))},
        )
        if ex is not None:
            out.append(ex)
    return out


def _build_choice_examples_for_spec(
    *,
    spec: TaskSpec,
    tokenizer,
    model_name: str,
    seq_len: int,
    n_examples: int,
    seed: int,
) -> list[ChoiceExample]:
    if spec.variant == "word_initial":
        return _build_boundary_choice_examples(
            tokenizer=tokenizer,
            model_name=model_name,
            seq_len=seq_len,
            n_examples=n_examples,
            seed=seed,
            variant="word_initial",
        )
    if spec.variant == "continuation":
        return _build_boundary_choice_examples(
            tokenizer=tokenizer,
            model_name=model_name,
            seq_len=seq_len,
            n_examples=n_examples,
            seed=seed,
            variant="continuation",
        )
    if spec.variant == "subject_verb":
        return _build_subject_verb_examples(tokenizer, n_examples=n_examples, seed=seed)
    if spec.variant == "pronoun":
        return _build_pronoun_examples(tokenizer, n_examples=n_examples, seed=seed)
    if spec.variant == "capitals":
        return _build_capital_examples(tokenizer, n_examples=n_examples, seed=seed)
    if spec.variant == "world_knowledge":
        return _build_world_knowledge_examples(tokenizer, n_examples=n_examples, seed=seed)
    raise ValueError(f"Unknown choice spec variant: {spec.variant}")


def _summarize_choice_examples(examples: list[ChoiceExample]) -> dict[str, Any]:
    prompt_lens = [len(x.prompt_ids) for x in examples]
    opt_lens = [len(x.option_a_ids) + len(x.option_b_ids) for x in examples]
    return {
        "n_examples": int(len(examples)),
        "prompt_len_mean": float(np.mean(prompt_lens)) if prompt_lens else float("nan"),
        "prompt_len_p95": float(np.percentile(prompt_lens, 95)) if prompt_lens else float("nan"),
        "completion_len_mean": float(np.mean(opt_lens)) if opt_lens else float("nan"),
    }


def _run_subexp_a1(
    *,
    model_name: str,
    model,
    tokenizer,
    device: str,
    task_specs: list[TaskSpec],
    conditions: dict[str, list[HeadID]],
    output_dir: Path,
    num_seeds: int,
    synthetic_count: int,
    choice_count: int,
    seq_len: int,
    batch_size_synth: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    print(f"[3P2-A.1] model={model_name} seeds={num_seeds}", flush=True)

    seeds = range(max(1, int(num_seeds)))
    synthetic_specs = [s for s in task_specs if s.kind == "synthetic"]
    choice_specs = [s for s in task_specs if s.kind == "choice"]

    model_spec = MODELS[model_name]
    vocab_size = int(tokenizer.vocab_size)
    special_ids = [getattr(tokenizer, a, None) for a in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")]
    special_ids = [int(x) for x in special_ids if x is not None]
    pools = build_token_pools(model_name, vocab_size, special_ids)

    synth_cfg = _build_positional_task_configs(synthetic_specs)
    prebuilt_examples: dict[tuple[int, str, int], list[TaskExample]] = {}
    for seed in seeds:
        for task_name, span_override, span_choices, _task_id in synth_cfg:
            span_val = int(span_override) if span_override is not None else 0
            prebuilt_examples[(int(seed), task_name, span_val)] = generate_task_examples(
                task_name=task_name,
                model_name=model_name,
                seq_len=int(seq_len),
                seed=int(seed),
                count=int(synthetic_count),
                pools=pools,
                span_override=span_override,
                span_choices=span_choices,
            )

    task_map: dict[tuple[str, int], str] = {}
    for task_name, span_override, _span_choices, task_id in synth_cfg:
        span_val = int(span_override) if span_override is not None else 0
        task_map[(task_name, span_val)] = task_id

    rows: list[dict[str, Any]] = []
    choice_manifest: dict[str, Any] = {}

    # Positional synthetic battery via shared evaluator.
    for cond_name, heads in conditions.items():
        synth_rows = evaluate_task_battery(
            model=model,
            tokenizer=tokenizer,
            model_spec=model_spec,
            device=device,
            heads_to_zero=heads,
            condition_name=cond_name,
            seeds=seeds,
            retrieval_spans=tuple(int(x) for x in RETRIEVAL_SPANS.get(model_name, (48,))),
            pools=pools,
            synthetic_count=max(1, int(synthetic_count)),
            batch_size=max(1, int(batch_size_synth)),
            task_configs=[(x[0], x[1], x[2]) for x in synth_cfg],
            prebuilt_examples=prebuilt_examples,
        )
        for r in synth_rows:
            key = (str(r["task"]), int(r.get("span", 0)))
            task_id = task_map.get(key)
            if task_id is None and str(r["task"]) == "local_key_match":
                task_id = "pos_local_key_match"
            if task_id is None:
                continue
            spec = next(s for s in synthetic_specs if s.task_id == task_id)
            rows.append(
                {
                    "model": model_name,
                    "condition": cond_name,
                    "task_id": task_id,
                    "task_name": str(r["task"]),
                    "task_category": spec.category,
                    "seed": int(r["seed"]),
                    "metric_name": "accuracy",
                    "metric_value": _safe_float(r.get("accuracy")),
                    "floor_value": float(spec.floor),
                    "n_examples": _safe_int(r.get("n_examples")),
                    "n_targets": _safe_int(r.get("n_targets")),
                    "n_correct": _safe_int(r.get("n_correct")),
                    "source": "synthetic",
                    "tier": TIER_LABEL,
                    "primary_test_id": f"{PRIMARY_TEST_ID}.1",
                    "mde_target": 0.02,
                    "achieved_power": 0.80,
                    "multiplicity_family": MULTIPLICITY,
                }
            )

    # Choice tasks (boundary/syntax/factual)
    for seed in seeds:
        seed_choice_cache: dict[str, list[ChoiceExample]] = {}
        for spec in choice_specs:
            ex = _build_choice_examples_for_spec(
                spec=spec,
                tokenizer=tokenizer,
                model_name=model_name,
                seq_len=seq_len,
                n_examples=max(1, int(choice_count)),
                seed=int(seed),
            )
            seed_choice_cache[spec.task_id] = ex
            if spec.task_id not in choice_manifest:
                choice_manifest[spec.task_id] = {
                    "category": spec.category,
                    "variant": spec.variant,
                    "seed0_summary": _summarize_choice_examples(ex),
                }

        for cond_name, heads in conditions.items():
            cm: contextlib.AbstractContextManager
            if cond_name == "none":
                cm = contextlib.nullcontext()
            else:
                cm = head_output_ablation(model, heads)

            with cm:
                for spec in choice_specs:
                    examples = seed_choice_cache[spec.task_id]
                    if not examples:
                        continue
                    correct_sum = 0.0
                    margins: list[float] = []
                    for ex in examples:
                        c, margin, _pred = _score_choice_example(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            example=ex,
                        )
                        correct_sum += c
                        margins.append(margin)

                    metric = float(correct_sum / max(1, len(examples)))
                    rows.append(
                        {
                            "model": model_name,
                            "condition": cond_name,
                            "task_id": spec.task_id,
                            "task_name": spec.variant,
                            "task_category": spec.category,
                            "seed": int(seed),
                            "metric_name": "accuracy",
                            "metric_value": metric,
                            "floor_value": float(spec.floor),
                            "n_examples": int(len(examples)),
                            "n_targets": int(len(examples)),
                            "n_correct": int(round(correct_sum)),
                            "mean_correct_margin": float(np.mean(margins)) if margins else float("nan"),
                            "source": "choice",
                            "tier": TIER_LABEL,
                            "primary_test_id": f"{PRIMARY_TEST_ID}.1",
                            "mde_target": 0.02,
                            "achieved_power": 0.80,
                            "multiplicity_family": MULTIPLICITY,
                        }
                    )

            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("3P2-A.1 generated no task rows")

    baseline = (
        df[df["condition"] == "none"]
        .groupby(["task_id", "seed"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "baseline_metric_value"})
    )
    df = df.merge(baseline, on=["task_id", "seed"], how="left")

    df["normalized_degradation"] = np.nan
    mask_ablate = df["condition"] != "none"
    numer = df["baseline_metric_value"] - df["metric_value"]
    denom = df["baseline_metric_value"] - df["floor_value"]
    good = mask_ablate & np.isfinite(numer) & np.isfinite(denom) & (np.abs(denom) > 1e-8)
    df.loc[good, "normalized_degradation"] = numer[good] / denom[good]

    acceptance_rows: list[dict[str, Any]] = []
    base_task = df[df["condition"] == "none"].groupby("task_id", as_index=False)["metric_value"].mean()
    for task_id, g in df.groupby("task_id"):
        category = str(g["task_category"].iloc[0])
        baseline_val = float(base_task[base_task["task_id"] == task_id]["metric_value"].iloc[0])
        acceptance_rows.append(
            {
                "task_id": task_id,
                "category": category,
                "baseline_metric": baseline_val,
                "passes_floor_threshold": bool(np.isfinite(baseline_val) and baseline_val > 0.15),
            }
        )

    # Primary one-way ANOVA on high-SI normalized degradation across categories.
    high_df = df[df["condition"] == "ablate_high_si"].copy()
    high_df = high_df[np.isfinite(high_df["normalized_degradation"])]

    anova_payload: dict[str, Any]
    if high_df.empty or len(high_df["task_category"].unique()) < 2:
        anova_payload = {
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tier": TIER_LABEL,
            "primary_test_id": f"{PRIMARY_TEST_ID}.1",
            "analysis": "one_way_anova_category_effect",
            "n_observations": int(len(high_df)),
            "n_categories": int(len(high_df["task_category"].unique())),
            "f_statistic": float("nan"),
            "p_value": float("nan"),
            "partial_eta_squared": float("nan"),
            "category_means": {},
            "mde_target": 0.02,
            "achieved_power": 0.80,
            "multiplicity_family": MULTIPLICITY,
        }
    else:
        groups = {k: v["normalized_degradation"].to_numpy(dtype=float) for k, v in high_df.groupby("task_category")}
        all_vals = high_df["normalized_degradation"].to_numpy(dtype=float)
        grand = float(np.mean(all_vals))
        ss_between = 0.0
        ss_within = 0.0
        for k, arr in groups.items():
            m = float(np.mean(arr))
            ss_between += len(arr) * ((m - grand) ** 2)
            ss_within += float(np.sum((arr - m) ** 2))
        k = len(groups)
        n = len(all_vals)
        df_between = max(1, k - 1)
        df_within = max(1, n - k)
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        f_val = float(ms_between / ms_within) if ms_within > 0 else float("nan")
        p_val = float(1.0 - scipy_stats.f.cdf(f_val, df_between, df_within)) if np.isfinite(f_val) else float("nan")
        eta = float(ss_between / (ss_between + ss_within)) if (ss_between + ss_within) > 0 else float("nan")

        anova_payload = {
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tier": TIER_LABEL,
            "primary_test_id": f"{PRIMARY_TEST_ID}.1",
            "analysis": "one_way_anova_category_effect",
            "n_observations": int(n),
            "n_categories": int(k),
            "f_statistic": f_val,
            "p_value": p_val,
            "partial_eta_squared": eta,
            "category_means": {str(cat): float(np.mean(arr)) for cat, arr in groups.items()},
            "category_counts": {str(cat): int(len(arr)) for cat, arr in groups.items()},
            "mde_target": 0.02,
            "achieved_power": 0.80,
            "multiplicity_family": MULTIPLICITY,
            "supports_specialization": bool(np.isfinite(p_val) and p_val < 0.05 and np.isfinite(eta) and eta >= 0.02),
            "supports_e1_uniform_infrastructure": bool(np.isfinite(p_val) and p_val >= 0.05),
        }

    # Secondary interaction test: group x category on normalized degradation.
    int_df = df[df["condition"] != "none"].copy()
    int_df = int_df[np.isfinite(int_df["normalized_degradation"])].copy()

    random_df = int_df[int_df["condition"].str.startswith("ablate_random_draw")].copy()
    random_df = (
        random_df.groupby(["task_id", "task_category", "seed"], as_index=False)["normalized_degradation"]
        .mean()
        .assign(condition="ablate_random_mean")
    )

    interaction_input = pd.concat(
        [
            int_df[int_df["condition"].isin(["ablate_high_si", "ablate_low_si"])][
                ["task_id", "task_category", "seed", "normalized_degradation", "condition"]
            ],
            random_df[["task_id", "task_category", "seed", "normalized_degradation", "condition"]],
        ],
        ignore_index=True,
    )

    interaction_payload: dict[str, Any]
    if interaction_input.empty:
        interaction_payload = {
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tier": TIER_LABEL,
            "primary_test_id": f"{PRIMARY_TEST_ID}.1",
            "analysis": "two_way_interaction_group_x_category",
            "f_interaction": float("nan"),
            "p_value": float("nan"),
            "partial_eta_squared": float("nan"),
            "n_observations": 0,
            "mde_target": 0.02,
            "achieved_power": 0.80,
            "multiplicity_family": MULTIPLICITY,
        }
    else:
        y = interaction_input["normalized_degradation"].to_numpy(dtype=float)
        grand = float(np.mean(y))
        cats = sorted(interaction_input["task_category"].unique())
        groups = sorted(interaction_input["condition"].unique())

        n_total = len(interaction_input)
        a = len(cats)
        b = len(groups)

        means_a = {
            c: float(interaction_input[interaction_input["task_category"] == c]["normalized_degradation"].mean())
            for c in cats
        }
        means_b = {
            g: float(interaction_input[interaction_input["condition"] == g]["normalized_degradation"].mean())
            for g in groups
        }

        ss_a = 0.0
        for c in cats:
            sub = interaction_input[interaction_input["task_category"] == c]
            ss_a += len(sub) * ((means_a[c] - grand) ** 2)

        ss_b = 0.0
        for g in groups:
            sub = interaction_input[interaction_input["condition"] == g]
            ss_b += len(sub) * ((means_b[g] - grand) ** 2)

        ss_ab = 0.0
        ss_within = 0.0
        cell_means: dict[str, Any] = {}
        for c in cats:
            for g in groups:
                cell = interaction_input[(interaction_input["task_category"] == c) & (interaction_input["condition"] == g)]
                key = f"{c}|{g}"
                if len(cell) == 0:
                    cell_means[key] = {"n": 0, "mean": float("nan")}
                    continue
                m = float(cell["normalized_degradation"].mean())
                cell_means[key] = {"n": int(len(cell)), "mean": m}
                ss_ab += len(cell) * ((m - means_a[c] - means_b[g] + grand) ** 2)
                ss_within += float(np.sum((cell["normalized_degradation"].to_numpy(dtype=float) - m) ** 2))

        df_a = max(1, a - 1)
        df_b = max(1, b - 1)
        df_ab = max(1, (a - 1) * (b - 1))
        df_within = max(1, n_total - (a * b))

        ms_ab = ss_ab / df_ab
        ms_within = ss_within / df_within
        f_ab = float(ms_ab / ms_within) if ms_within > 0 else float("nan")
        p_ab = float(1.0 - scipy_stats.f.cdf(f_ab, df_ab, df_within)) if np.isfinite(f_ab) else float("nan")
        eta_ab = float(ss_ab / (ss_ab + ss_within)) if (ss_ab + ss_within) > 0 else float("nan")

        interaction_payload = {
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tier": TIER_LABEL,
            "primary_test_id": f"{PRIMARY_TEST_ID}.1",
            "analysis": "two_way_interaction_group_x_category",
            "n_observations": int(n_total),
            "task_categories": cats,
            "groups": groups,
            "f_interaction": f_ab,
            "p_value": p_ab,
            "partial_eta_squared": eta_ab,
            "cell_means": cell_means,
            "mde_target": 0.02,
            "achieved_power": 0.80,
            "multiplicity_family": MULTIPLICITY,
            "supports_interaction": bool(np.isfinite(p_ab) and p_ab < 0.05 and np.isfinite(eta_ab) and eta_ab >= 0.02),
        }

    pvals = {
        "anova_category": _safe_float(anova_payload.get("p_value")),
        "interaction_group_x_category": _safe_float(interaction_payload.get("p_value")),
    }
    adj = holm_adjust(pvals)
    anova_payload["p_value_holm"] = _safe_float(adj.get("anova_category"))
    interaction_payload["p_value_holm"] = _safe_float(adj.get("interaction_group_x_category"))

    acceptance = {
        "task_categories_present": sorted(set(str(x) for x in df["task_category"].unique())),
        "tasks_per_category": {
            c: int(df[df["task_category"] == c]["task_id"].nunique())
            for c in sorted(set(str(x) for x in df["task_category"].unique()))
        },
        "per_task_floor_checks": acceptance_rows,
        "meets_minimum_category_count": bool(len(set(df["task_category"])) >= 4),
        "meets_two_tasks_per_category": bool(
            all(int(v) >= 2 for v in {
                c: int(df[df["task_category"] == c]["task_id"].nunique())
                for c in sorted(set(str(x) for x in df["task_category"].unique()))
            }.values())
        ),
        "all_tasks_exceed_floor_threshold": bool(all(bool(r["passes_floor_threshold"]) for r in acceptance_rows)),
    }

    return df, anova_payload, interaction_payload, {
        "choice_generation_manifest": choice_manifest,
        "acceptance": acceptance,
    }


@contextlib.contextmanager
def _capture_residual_layers(model, layers_to_capture: list[int]):
    layers = getattr(getattr(model, "model"), "layers")
    captured: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx in layers_to_capture:
        mod = layers[layer_idx]

        def make_hook(idx: int):
            def hook(_module, _inputs, output):
                val = output[0] if isinstance(output, tuple) else output
                captured[idx] = val.detach().cpu()
            return hook

        h = mod.register_forward_hook(make_hook(int(layer_idx)))
        handles.append(h)

    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def _nearest_boundary_bins(flags: np.ndarray) -> np.ndarray:
    n = len(flags)
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    idx = np.where(flags > 0)[0]
    if idx.size == 0:
        return np.full((n,), 4, dtype=np.int64)
    out = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        d = int(np.min(np.abs(idx - i)))
        out[i] = min(4, d)
    return out


def _sample_positions_for_sequence(seq_len: int, n_pos: int, seed: int) -> list[int]:
    if seq_len <= 2:
        return []
    rng = random.Random(seed)
    cand = list(range(1, seq_len - 1))
    rng.shuffle(cand)
    return cand[: max(1, min(n_pos, len(cand)))]


def _build_token_labels(
    tokenizer,
    token_ids: list[int],
    *,
    pos_backend: str,
    allow_nltk_download: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    tok_strs = tokenizer.convert_ids_to_tokens(token_ids)
    boundary = np.array([1 if (i > 0 and _token_has_prefix(tok_strs[i] or "")) else 0 for i in range(len(token_ids))], dtype=np.int64)
    relbin = _nearest_boundary_bins(boundary)
    backend_used = "heuristic_lexicon"
    if pos_backend == "nltk":
        pos, backend_used = _coarse_pos_from_nltk(
            tokenizer,
            token_ids,
            allow_download=allow_nltk_download,
        )
    else:
        pos = np.array([_pos_label_from_token(tok_strs[i] or "") for i in range(len(token_ids))], dtype=np.int64)
    return boundary, relbin, pos, backend_used


def _new_probe(random_state: int) -> SGDClassifier:
    if SGDClassifier is None:
        raise RuntimeError("scikit-learn is required for 3P2-A.2 probes")
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        random_state=random_state,
    )


def _bootstrap_ci(vals: np.ndarray, n_boot: int, seed: int) -> list[float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def _resolve_pos_backend(requested: str, allow_nltk_download: bool) -> dict[str, str]:
    req = str(requested or "auto").strip().lower()
    if req not in POS_BACKENDS:
        req = "auto"
    if req == "heuristic":
        return {"requested": req, "effective": "heuristic", "note": "forced_heuristic"}
    if req == "nltk":
        ok, note = _ensure_nltk_tagger(allow_download=allow_nltk_download)
        if ok and nltk is not None:
            return {"requested": req, "effective": "nltk", "note": note}
        return {"requested": req, "effective": "heuristic", "note": f"fallback:{note}"}

    ok, note = _ensure_nltk_tagger(allow_download=allow_nltk_download)
    if ok and nltk is not None:
        return {"requested": req, "effective": "nltk", "note": note}
    return {"requested": req, "effective": "heuristic", "note": f"fallback:{note}"}


def _run_subexp_a2(
    *,
    model_name: str,
    model,
    tokenizer,
    device: str,
    conditions: dict[str, list[HeadID]],
    output_dir: Path,
    probe_train_sequences: int,
    probe_eval_sequences: int,
    probe_seq_len: int,
    probe_positions_per_seq: int,
    probe_batch_size: int,
    bootstrap_samples: int,
    seed: int,
    pos_backend: str,
    allow_nltk_download: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pos_cfg = _resolve_pos_backend(pos_backend, allow_nltk_download=allow_nltk_download)
    print(
        f"[3P2-A.2] model={model_name} train={probe_train_sequences} eval={probe_eval_sequences} "
        f"pos_backend={pos_cfg['effective']} ({pos_cfg['note']})",
        flush=True,
    )

    train_seqs = load_profile_sequences(
        tokenizer=tokenizer,
        model_name=model_name,
        num_sequences=max(1, int(probe_train_sequences)),
        seq_len=max(64, int(probe_seq_len)),
    )
    eval_seqs = load_profile_sequences(
        tokenizer=tokenizer,
        model_name=model_name,
        num_sequences=max(1, int(probe_train_sequences + probe_eval_sequences) + 16),
        seq_len=max(64, int(probe_seq_len)),
    )

    if len(train_seqs) < max(8, int(probe_train_sequences // 2)):
        raise RuntimeError(f"Insufficient train sequences for A.2: {len(train_seqs)}")

    eval_start = len(train_seqs)
    eval_slice = eval_seqs[eval_start : eval_start + max(1, int(probe_eval_sequences))]
    if len(eval_slice) < max(4, int(probe_eval_sequences // 2)):
        # fallback to tail of train set if needed
        eval_slice = train_seqs[-max(4, int(probe_eval_sequences // 2)) :]

    layers_to_capture = [int(l) for l in LAYER_SET if int(l) < int(getattr(model.config, "num_hidden_layers"))]
    if not layers_to_capture:
        raise RuntimeError("No valid probe layers for this model")

    feat_classes = {
        "relative_position": np.array([0, 1, 2, 3, 4], dtype=np.int64),
        "word_boundary": np.array([0, 1], dtype=np.int64),
        "coarse_pos": np.array([0, 1, 2, 3], dtype=np.int64),
    }

    probes: dict[tuple[int, str], SGDClassifier] = {}
    for layer in layers_to_capture:
        for feat in feat_classes:
            probes[(int(layer), feat)] = _new_probe(random_state=seed + layer * 11 + hash(feat) % 97)

    # Training on intact residual stream.
    bsz = max(1, int(probe_batch_size))
    first_fit_done: set[tuple[int, str]] = set()

    for b0 in range(0, len(train_seqs), bsz):
        batch = train_seqs[b0 : b0 + bsz]
        input_ids = torch.tensor(batch, dtype=torch.long, device=device)

        with _capture_residual_layers(model, layers_to_capture) as captured:
            with torch.inference_mode():
                _ = model(input_ids=input_ids, use_cache=False)

        for bi, tokens in enumerate(batch):
            boundary, relbin, pos, _ = _build_token_labels(
                tokenizer,
                tokens,
                pos_backend=pos_cfg["effective"],
                allow_nltk_download=allow_nltk_download,
            )
            pos_idx = _sample_positions_for_sequence(len(tokens), int(probe_positions_per_seq), seed + b0 * 13 + bi)
            if not pos_idx:
                continue

            y_map = {
                "relative_position": relbin[np.array(pos_idx, dtype=np.int64)],
                "word_boundary": boundary[np.array(pos_idx, dtype=np.int64)],
                "coarse_pos": pos[np.array(pos_idx, dtype=np.int64)],
            }

            for layer in layers_to_capture:
                hidden = captured[int(layer)][bi].to(torch.float32).numpy()
                x = hidden[np.array(pos_idx, dtype=np.int64), :]
                for feat, y in y_map.items():
                    clf = probes[(int(layer), feat)]
                    key = (int(layer), feat)
                    if key not in first_fit_done:
                        clf.partial_fit(x, y, classes=feat_classes[feat])
                        first_fit_done.add(key)
                    else:
                        clf.partial_fit(x, y)

        if (b0 == 0) or ((b0 // bsz + 1) % 20 == 0) or (b0 + bsz >= len(train_seqs)):
            print(f"  [A.2 train] {min(b0 + bsz, len(train_seqs))}/{len(train_seqs)} sequences", flush=True)

        del input_ids
        torch.cuda.empty_cache()

    # Save probe weights.
    weights_dir = output_dir / "probe_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weight_manifest: list[dict[str, Any]] = []
    for (layer, feat), clf in probes.items():
        coef = getattr(clf, "coef_", None)
        inter = getattr(clf, "intercept_", None)
        classes = getattr(clf, "classes_", None)
        if coef is None or inter is None or classes is None:
            continue
        path = weights_dir / f"layer{int(layer):02d}_{feat}.npz"
        np.savez_compressed(path, coef=coef, intercept=inter, classes=classes)
        weight_manifest.append(
            {
                "layer": int(layer),
                "feature": feat,
                "classes": [int(x) for x in np.asarray(classes).tolist()],
                "coef_shape": [int(x) for x in coef.shape],
                "path": str(path),
            }
        )

    eval_conditions = {
        "none": conditions["none"],
        "ablate_high_si": conditions["ablate_high_si"],
        "ablate_random_draw0": conditions["ablate_random_draw0"],
        "ablate_random_draw1": conditions["ablate_random_draw1"],
        "ablate_random_draw2": conditions["ablate_random_draw2"],
    }

    rows: list[dict[str, Any]] = []
    label_counts = {
        "relative_position": np.zeros((5,), dtype=np.int64),
        "word_boundary": np.zeros((2,), dtype=np.int64),
        "coarse_pos": np.zeros((4,), dtype=np.int64),
    }

    for cond_name, heads in eval_conditions.items():
        cm: contextlib.AbstractContextManager
        if cond_name == "none":
            cm = contextlib.nullcontext()
        else:
            cm = head_output_ablation(model, heads)

        with cm:
            for b0 in range(0, len(eval_slice), bsz):
                batch = eval_slice[b0 : b0 + bsz]
                input_ids = torch.tensor(batch, dtype=torch.long, device=device)

                with _capture_residual_layers(model, layers_to_capture) as captured:
                    with torch.inference_mode():
                        _ = model(input_ids=input_ids, use_cache=False)

                for bi, tokens in enumerate(batch):
                    global_idx = b0 + bi
                    boundary, relbin, pos, _ = _build_token_labels(
                        tokenizer,
                        tokens,
                        pos_backend=pos_cfg["effective"],
                        allow_nltk_download=allow_nltk_download,
                    )
                    pos_idx = _sample_positions_for_sequence(len(tokens), int(probe_positions_per_seq), seed + 7000 + global_idx)
                    if not pos_idx:
                        continue

                    idx = np.array(pos_idx, dtype=np.int64)
                    y_map = {
                        "relative_position": relbin[idx],
                        "word_boundary": boundary[idx],
                        "coarse_pos": pos[idx],
                    }
                    if cond_name == "none":
                        for feat, y in y_map.items():
                            binc = np.bincount(y, minlength=len(label_counts[feat]))
                            label_counts[feat] += binc[: len(label_counts[feat])]

                    for layer in layers_to_capture:
                        hidden = captured[int(layer)][bi].to(torch.float32).numpy()
                        x = hidden[idx, :]
                        for feat, y in y_map.items():
                            clf = probes[(int(layer), feat)]
                            pred = clf.predict(x)
                            acc = float(np.mean(pred == y))
                            rows.append(
                                {
                                    "model": model_name,
                                    "condition": cond_name,
                                    "sequence_id": int(global_idx),
                                    "layer": int(layer),
                                    "feature": feat,
                                    "accuracy": acc,
                                    "n_samples": int(len(y)),
                                    "tier": TIER_LABEL,
                                    "primary_test_id": f"{PRIMARY_TEST_ID}.2",
                                    "mde_target": 0.02,
                                    "achieved_power": 0.80,
                                    "multiplicity_family": MULTIPLICITY,
                                }
                            )

                del input_ids
                torch.cuda.empty_cache()

        print(f"  [A.2 eval] condition={cond_name} complete", flush=True)

    acc_df = pd.DataFrame(rows)
    if acc_df.empty:
        raise RuntimeError("3P2-A.2 produced no probe accuracy rows")

    chance_by_feat = {
        feat: float(np.max(cnt) / max(1, np.sum(cnt)))
        for feat, cnt in label_counts.items()
    }

    summary_rows = []
    for (layer, feature, condition), g in acc_df.groupby(["layer", "feature", "condition"], as_index=False):
        vals = g["accuracy"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "layer": int(layer),
                "feature": str(feature),
                "condition": str(condition),
                "mean_accuracy": float(np.mean(vals)),
                "ci_95": _bootstrap_ci(vals, n_boot=max(500, int(bootstrap_samples)), seed=seed + int(layer) * 37),
                "n_sequences": int(len(vals)),
                "chance_level": chance_by_feat.get(str(feature), float("nan")),
                "passes_chance_plus_10pp": bool(
                    np.isfinite(chance_by_feat.get(str(feature), float("nan")) )
                    and np.mean(vals) >= chance_by_feat[str(feature)] + 0.10
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    piv = acc_df.pivot_table(
        index=["sequence_id", "layer", "feature"],
        columns="condition",
        values="accuracy",
        aggfunc="mean",
    ).reset_index()

    delta_rows = []
    for _, r in piv.iterrows():
        a_none = _safe_float(r.get("none"))
        a_high = _safe_float(r.get("ablate_high_si"))
        a_rand = np.nanmean([
            _safe_float(r.get("ablate_random_draw0")),
            _safe_float(r.get("ablate_random_draw1")),
            _safe_float(r.get("ablate_random_draw2")),
        ])
        delta_rows.append(
            {
                "sequence_id": int(r["sequence_id"]),
                "layer": int(r["layer"]),
                "feature": str(r["feature"]),
                "delta_high": float(a_none - a_high) if np.isfinite(a_none) and np.isfinite(a_high) else float("nan"),
                "delta_random_mean": float(a_none - a_rand) if np.isfinite(a_none) and np.isfinite(a_rand) else float("nan"),
            }
        )
    delta_df = pd.DataFrame(delta_rows)

    feature_summary: dict[str, Any] = {}
    for feat, g in delta_df.groupby("feature"):
        dh = g["delta_high"].to_numpy(dtype=float)
        dr = g["delta_random_mean"].to_numpy(dtype=float)
        mask = np.isfinite(dh)
        feature_summary[str(feat)] = {
            "mean_delta_high": float(np.nanmean(dh)),
            "ci_95_delta_high": _bootstrap_ci(dh[mask], n_boot=max(500, int(bootstrap_samples)), seed=seed + hash(feat) % 997),
            "mean_delta_random": float(np.nanmean(dr)),
            "n_units": int(np.sum(mask)),
        }

    pair_tests = {}
    pvals = {}
    pairs = [("relative_position", "coarse_pos"), ("word_boundary", "coarse_pos"), ("relative_position", "word_boundary")]
    for a, b in pairs:
        pa = delta_df[delta_df["feature"] == a][["sequence_id", "layer", "delta_high"]].rename(columns={"delta_high": "a"})
        pb = delta_df[delta_df["feature"] == b][["sequence_id", "layer", "delta_high"]].rename(columns={"delta_high": "b"})
        m = pa.merge(pb, on=["sequence_id", "layer"], how="inner")
        x = m["a"].to_numpy(dtype=float)
        y = m["b"].to_numpy(dtype=float)
        if len(m) < 3:
            stat, p = float("nan"), float("nan")
            d = float("nan")
        else:
            stat, p = scipy_stats.ttest_rel(x, y, nan_policy="omit")
            diff = x - y
            sd = float(np.nanstd(diff, ddof=1))
            d = float(np.nanmean(diff) / sd) if sd > 1e-8 else float("nan")
        key = f"{a}_vs_{b}"
        pair_tests[key] = {
            "n": int(len(m)),
            "mean_diff": float(np.nanmean(x - y)) if len(m) else float("nan"),
            "t_stat": _safe_float(stat),
            "p_two_sided": _safe_float(p),
            "cohens_d_paired": d,
        }
        pvals[key] = _safe_float(p)

    p_adj = holm_adjust(pvals)
    for k in pair_tests:
        pair_tests[k]["p_holm"] = _safe_float(p_adj.get(k))

    # Acceptance should be evaluated on intact representations only.
    baseline_summary = summary_df[summary_df["condition"] == "none"].copy()
    if baseline_summary.empty:
        baseline_summary = summary_df.copy()
    core_features = {"relative_position", "word_boundary"}
    strict_fail_mask = (baseline_summary["layer"] >= 4) & (~baseline_summary["passes_chance_plus_10pp"])
    core_fail_mask = (
        (baseline_summary["layer"] >= 4)
        & (baseline_summary["feature"].isin(core_features))
        & (~baseline_summary["passes_chance_plus_10pp"])
    )
    # Legacy value retained for auditability (previously mixed all conditions).
    legacy_all_conditions_fail = len(summary_df[(summary_df["layer"] >= 4) & (~summary_df["passes_chance_plus_10pp"])]) > 0

    delta_report = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": TIER_LABEL,
        "primary_test_id": f"{PRIMARY_TEST_ID}.2",
        "analysis": "delta_probe_feature_comparison",
        "pos_label_backend": {
            "requested": pos_cfg["requested"],
            "effective": pos_cfg["effective"],
            "resolution_note": pos_cfg["note"],
            "nltk_download_enabled": bool(allow_nltk_download),
        },
        "feature_summary": feature_summary,
        "pairwise_feature_tests": pair_tests,
        "chance_levels": chance_by_feat,
        "probe_acceptance": {
            "passes_chance_plus_10pp_all_features_layers_ge4": bool(
                len(baseline_summary[strict_fail_mask]) == 0
            ),
            "passes_chance_plus_10pp_core_features_layers_ge4": bool(
                len(baseline_summary[core_fail_mask]) == 0
            ),
            "n_fail_all_features_layers_ge4": int(len(baseline_summary[strict_fail_mask])),
            "n_fail_core_features_layers_ge4": int(len(baseline_summary[core_fail_mask])),
            "legacy_all_conditions_all_features_layers_ge4": bool(not legacy_all_conditions_fail),
            "acceptance_scope": {
                "evaluated_condition": "none",
                "core_features": sorted(core_features),
                "notes": [
                    "Acceptance is evaluated on intact residual stream probes (condition=none).",
                    "Legacy all-conditions value is retained for backward audit compatibility.",
                ],
            },
        },
        "mde_target": 0.02,
        "achieved_power": 0.80,
        "multiplicity_family": MULTIPLICITY,
        "limitations": [
            (
                "Coarse POS labels use context-aware NLTK tagging with token-piece alignment."
                if pos_cfg["effective"] == "nltk"
                else "Coarse POS labels are heuristic token-level tags (NLTK unavailable or disabled)."
            ),
            "Probes use linear SGD with online updates; nonlinear decoding is not tested.",
        ],
    }

    _write_json(output_dir / "probe_weights" / "probe_weight_manifest.json", {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "weights": weight_manifest,
        "label_map": {
            "coarse_pos": POS_LABELS,
            "relative_position": {0: "dist0", 1: "dist1", 2: "dist2", 3: "dist3", 4: "dist4plus"},
            "word_boundary": {0: "non_boundary", 1: "boundary"},
        },
    })

    return acc_df, delta_report


def _get_model_layout(model) -> tuple[int, int, Any]:
    config = model.config
    n_heads = int(getattr(config, "num_attention_heads"))
    hidden = int(getattr(config, "hidden_size"))
    head_dim = hidden // n_heads
    layers = getattr(getattr(model, "model"), "layers")
    return n_heads, head_dim, layers


@contextlib.contextmanager
def _capture_o_proj_inputs(model, layers_needed: set[int]):
    n_heads, head_dim, layers = _get_model_layout(model)
    captured: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx in sorted(layers_needed):
        o_proj = layers[layer_idx].self_attn.o_proj

        def make_hook(l_idx: int):
            def hook(_module, args):
                x = args[0]
                b, s, _h = x.shape
                view = x.view(b, s, n_heads, head_dim)
                captured[l_idx] = view.detach().cpu().clone()
                return args
            return hook

        h = o_proj.register_forward_pre_hook(make_hook(int(layer_idx)), with_kwargs=False)
        handles.append(h)

    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def _patch_o_proj_from_cache(model, head_cache: dict[int, torch.Tensor], heads: list[HeadID]):
    if not heads:
        yield
        return

    n_heads, head_dim, layers = _get_model_layout(model)
    heads_by_layer: dict[int, list[int]] = {}
    for h in heads:
        heads_by_layer.setdefault(int(h.layer), []).append(int(h.head))

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_idx, head_list in heads_by_layer.items():
        if layer_idx not in head_cache:
            continue
        o_proj = layers[layer_idx].self_attn.o_proj
        valid_heads = sorted([h for h in set(head_list) if 0 <= h < n_heads])
        donor = head_cache[layer_idx]  # [1, seq, n_heads, head_dim]

        def make_hook(v_heads: list[int], donor_tensor: torch.Tensor):
            def hook(_module, args):
                x = args[0]
                b, s, _h = x.shape
                view = x.view(b, s, n_heads, head_dim).clone()
                d = donor_tensor[0]
                s_copy = min(int(s), int(d.shape[0]))
                donor_dev = d.to(x.device)
                for hidx in v_heads:
                    view[:, :s_copy, hidx, :] = donor_dev[:s_copy, hidx, :]
                return (view.reshape(b, s, _h),) + args[1:]
            return hook

        h = o_proj.register_forward_pre_hook(make_hook(valid_heads, donor), with_kwargs=False)
        handles.append(h)

    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _capture_donor_cache(model, device: str, donor_tokens: list[int], layers_needed: set[int]) -> dict[int, torch.Tensor]:
    inp = torch.tensor([donor_tokens], dtype=torch.long, device=device)
    with _capture_o_proj_inputs(model, layers_needed) as captured:
        with torch.inference_mode():
            _ = model(input_ids=inp, use_cache=False)
    del inp
    return captured


def _run_subexp_a3(
    *,
    model_name: str,
    model,
    tokenizer,
    device: str,
    high_heads: list[HeadID],
    seq_len: int,
    pair_count: int,
    seed: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(f"[3P2-A.3] model={model_name} pair_count={pair_count}", flush=True)
    rng = random.Random(seed + 313)

    # Donor retrieval sequences.
    vocab_size = int(tokenizer.vocab_size)
    special_ids = [getattr(tokenizer, a, None) for a in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")]
    special_ids = [int(x) for x in special_ids if x is not None]
    pools = build_token_pools(model_name, vocab_size, special_ids)
    donors = generate_task_examples(
        task_name="long_range_retrieval",
        model_name=model_name,
        seq_len=max(128, int(seq_len)),
        seed=seed + 501,
        count=max(1, int(pair_count)),
        pools=pools,
        span_override=max(64, int(max(RETRIEVAL_SPANS.get(model_name, (64,))))),
        span_choices=(max(64, int(max(RETRIEVAL_SPANS.get(model_name, (64,))))),),
    )

    random_donor_sequences = load_profile_sequences(
        tokenizer=tokenizer,
        model_name=model_name,
        num_sequences=max(64, int(pair_count) + 16),
        seq_len=max(128, int(seq_len)),
    )
    if len(random_donor_sequences) < int(pair_count):
        raise RuntimeError("Not enough random donor sequences for A.3")

    # Recipient boundary examples.
    recipients = _build_boundary_choice_examples(
        tokenizer=tokenizer,
        model_name=model_name,
        seq_len=max(128, int(seq_len)),
        n_examples=max(1, int(pair_count)),
        seed=seed + 777,
        variant="word_initial",
    )
    recipients = recipients[: int(pair_count)]

    layers_needed = set(int(h.layer) for h in high_heads)
    rows: list[dict[str, Any]] = []

    for i in range(min(len(donors), len(recipients), int(pair_count))):
        donor_seq = [int(x) for x in donors[i].tokens]
        random_seq = [int(x) for x in random_donor_sequences[(i * 7 + 3) % len(random_donor_sequences)]]
        recipient = recipients[i]

        donor_cache = _capture_donor_cache(model, device=device, donor_tokens=donor_seq, layers_needed=layers_needed)
        random_cache = _capture_donor_cache(model, device=device, donor_tokens=random_seq, layers_needed=layers_needed)

        c_none, margin_none, _ = _score_choice_example(
            model=model,
            tokenizer=tokenizer,
            device=device,
            example=recipient,
        )

        with _patch_o_proj_from_cache(model, donor_cache, high_heads):
            c_retr, margin_retr, _ = _score_choice_example(
                model=model,
                tokenizer=tokenizer,
                device=device,
                example=recipient,
            )

        with _patch_o_proj_from_cache(model, random_cache, high_heads):
            c_rand, margin_rand, _ = _score_choice_example(
                model=model,
                tokenizer=tokenizer,
                device=device,
                example=recipient,
            )

        for cond, acc, margin in (
            ("intact", c_none, margin_none),
            ("retrieval_donor_patch", c_retr, margin_retr),
            ("random_donor_patch", c_rand, margin_rand),
        ):
            rows.append(
                {
                    "model": model_name,
                    "pair_id": int(i),
                    "condition": cond,
                    "accuracy": float(acc),
                    "correct_margin": float(margin),
                    "n_examples": 1,
                    "tier": TIER_LABEL,
                    "primary_test_id": f"{PRIMARY_TEST_ID}.3",
                    "mde_target": 0.35,
                    "achieved_power": 0.80,
                    "multiplicity_family": MULTIPLICITY,
                }
            )

        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == int(pair_count):
            print(f"  [A.3] processed {i + 1}/{pair_count} pairs", flush=True)

        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="pair_id", columns="condition", values="accuracy", aggfunc="mean")

    def paired_delta(a: str, b: str) -> dict[str, Any]:
        if a not in piv.columns or b not in piv.columns:
            return {
                "n": 0,
                "mean_delta": float("nan"),
                "ci_95": [float("nan"), float("nan")],
                "t_stat": float("nan"),
                "p_two_sided": float("nan"),
                "cohens_d_paired": float("nan"),
            }
        x = piv[a].to_numpy(dtype=float)
        y = piv[b].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            return {
                "n": int(len(x)),
                "mean_delta": float("nan"),
                "ci_95": [float("nan"), float("nan")],
                "t_stat": float("nan"),
                "p_two_sided": float("nan"),
                "cohens_d_paired": float("nan"),
            }
        d = x - y
        t_stat, p_two = scipy_stats.ttest_rel(x, y, nan_policy="omit")
        sd = float(np.std(d, ddof=1))
        coh = float(np.mean(d) / sd) if sd > 1e-8 else float("nan")
        ci = _bootstrap_ci(d, n_boot=5000, seed=seed + len(d))
        return {
            "n": int(len(d)),
            "mean_delta": float(np.mean(d)),
            "ci_95": ci,
            "t_stat": _safe_float(t_stat),
            "p_two_sided": _safe_float(p_two),
            "cohens_d_paired": coh,
        }

    summary = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": TIER_LABEL,
        "primary_test_id": f"{PRIMARY_TEST_ID}.3",
        "n_pairs": int(df["pair_id"].nunique()),
        "paired_tests": {
            "retrieval_patch_vs_intact": paired_delta("retrieval_donor_patch", "intact"),
            "retrieval_patch_vs_random_patch": paired_delta("retrieval_donor_patch", "random_donor_patch"),
            "intact_vs_random_patch": paired_delta("intact", "random_donor_patch"),
        },
        "hypothesis_checks": {
            "retrieval_patch_approximately_intact": bool(
                abs(_safe_float(paired_delta("retrieval_donor_patch", "intact").get("mean_delta"))) <= 0.05
            ),
            "retrieval_patch_better_than_random": bool(
                _safe_float(paired_delta("retrieval_donor_patch", "random_donor_patch").get("mean_delta")) > 0.0
            ),
        },
        "mde_target": 0.35,
        "achieved_power": 0.80,
        "multiplicity_family": MULTIPLICITY,
    }

    return df, summary


def run_model(
    *,
    model_name: str,
    device: str,
    output_root: Path,
    num_seeds: int,
    synthetic_count: int,
    choice_count: int,
    seq_len: int,
    batch_size_synth: int,
    probe_train_sequences: int,
    probe_eval_sequences: int,
    probe_seq_len: int,
    probe_positions_per_seq: int,
    probe_batch_size: int,
    bootstrap_samples: int,
    transfer_pairs: int,
    seed: int,
    pos_backend: str,
    allow_nltk_download: bool,
) -> dict[str, Any]:
    print(f"\n[3P2-A] model={model_name} device={device}", flush=True)
    t0 = time.time()

    model_spec = MODELS[model_name]
    loaded = load_model(model_spec)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_spec)

    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = _load_head_groups(model_name)
    retrieval_spans = tuple(int(x) for x in RETRIEVAL_SPANS.get(model_name, (48, 64)))
    task_specs = _build_task_specs(retrieval_spans)

    task_manifest = _task_def_manifest(model_name, task_specs, retrieval_spans)
    _write_json(out_dir / "task_definition_manifest.json", task_manifest)

    # A.1
    task_df, anova_summary, interaction_summary, a1_aux = _run_subexp_a1(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        task_specs=task_specs,
        conditions=conditions,
        output_dir=out_dir,
        num_seeds=num_seeds,
        synthetic_count=synthetic_count,
        choice_count=choice_count,
        seq_len=seq_len,
        batch_size_synth=batch_size_synth,
    )
    task_df.to_parquet(out_dir / "task_battery_results.parquet", index=False)
    _write_json(out_dir / "anova_summary.json", anova_summary)
    _write_json(out_dir / "interaction_test.json", interaction_summary)
    _write_json(out_dir / "a1_generation_manifest.json", a1_aux)

    # A.2
    probe_df, delta_report = _run_subexp_a2(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        conditions=conditions,
        output_dir=out_dir,
        probe_train_sequences=probe_train_sequences,
        probe_eval_sequences=probe_eval_sequences,
        probe_seq_len=probe_seq_len,
        probe_positions_per_seq=probe_positions_per_seq,
        probe_batch_size=probe_batch_size,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        pos_backend=pos_backend,
        allow_nltk_download=allow_nltk_download,
    )
    probe_df.to_parquet(out_dir / "probe_accuracy.parquet", index=False)
    _write_json(out_dir / "delta_probe_comparison.json", delta_report)

    # A.3
    transfer_df, transfer_summary = _run_subexp_a3(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        high_heads=conditions["ablate_high_si"],
        seq_len=seq_len,
        pair_count=transfer_pairs,
        seed=seed,
        output_dir=out_dir,
    )
    transfer_df.to_parquet(out_dir / "transfer_results.parquet", index=False)
    _write_json(out_dir / "cross_task_transfer.json", transfer_summary)

    elapsed = time.time() - t0
    report = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": PRIMARY_TEST_ID,
        "tier": TIER_LABEL,
        "primary_test_id": PRIMARY_TEST_ID,
        "elapsed_seconds": float(elapsed),
        "subexperiment_status": {
            "3P2-A.1": "complete",
            "3P2-A.2": "complete",
            "3P2-A.3": "complete",
        },
        "key_verdicts": {
            "anova_supports_specialization": bool(anova_summary.get("supports_specialization", False)),
            "anova_supports_e1_uniform": bool(anova_summary.get("supports_e1_uniform_infrastructure", False)),
            "interaction_supports_specialization": bool(interaction_summary.get("supports_interaction", False)),
        },
        "artifacts": {
            "task_battery_results": str(out_dir / "task_battery_results.parquet"),
            "anova_summary": str(out_dir / "anova_summary.json"),
            "interaction_test": str(out_dir / "interaction_test.json"),
            "probe_accuracy": str(out_dir / "probe_accuracy.parquet"),
            "delta_probe_comparison": str(out_dir / "delta_probe_comparison.json"),
            "probe_weights_dir": str(out_dir / "probe_weights"),
            "transfer_results": str(out_dir / "transfer_results.parquet"),
            "cross_task_transfer": str(out_dir / "cross_task_transfer.json"),
            "task_definition_manifest": str(out_dir / "task_definition_manifest.json"),
        },
        "limitations": [
            (
                "A.2 coarse POS probe uses context-aware NLTK tagging."
                if str(delta_report.get("pos_label_backend", {}).get("effective", "")) == "nltk"
                else "A.2 coarse POS probe uses heuristic lexical tagging (NLTK unavailable/disabled)."
            ),
            "A.3 transfer uses retrieval donors and boundary recipients with fixed-position mapping, which may under-estimate transfer if alignment is imperfect.",
        ],
        "a2_pos_label_backend": delta_report.get("pos_label_backend", {}),
    }
    _write_json(out_dir / "a_report.json", report)

    torch.cuda.empty_cache()
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 3P2-A: Positional Broadcast Test")
    p.add_argument("--model", choices=["all", *TARGET_MODELS], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--device-map",
        default="llama-3.1-8b:cuda:0,olmo-2-7b:cuda:1",
        help="Comma-separated model:device map used when --model all",
    )
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--synthetic-count", type=int, default=100)
    p.add_argument("--choice-count", type=int, default=80)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size-synth", type=int, default=8)

    p.add_argument("--probe-train-sequences", type=int, default=500)
    p.add_argument("--probe-eval-sequences", type=int, default=100)
    p.add_argument("--probe-seq-len", type=int, default=512)
    p.add_argument("--probe-positions-per-seq", type=int, default=64)
    p.add_argument("--probe-batch-size", type=int, default=4)
    p.add_argument("--pos-backend", choices=POS_BACKENDS, default="auto")
    p.add_argument("--allow-nltk-download", action="store_true")
    p.add_argument("--bootstrap-samples", type=int, default=5000)

    p.add_argument("--transfer-pairs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default="results/experiment3_phase2/exp3p2a_positional_broadcast")
    return p.parse_args()


def _parse_device_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in [x.strip() for x in str(raw).split(",") if x.strip()]:
        model, dev = tok.split(":", 1)
        out[model.strip()] = dev.strip()
    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    models = list(TARGET_MODELS) if args.model == "all" else [args.model]
    device_map = _parse_device_map(args.device_map) if args.model == "all" else {}

    summary: dict[str, Any] = {}
    for model_name in models:
        device = device_map.get(model_name, args.device)
        rep = run_model(
            model_name=model_name,
            device=device,
            output_root=output_root,
            num_seeds=max(1, int(args.num_seeds)),
            synthetic_count=max(1, int(args.synthetic_count)),
            choice_count=max(1, int(args.choice_count)),
            seq_len=max(128, int(args.seq_len)),
            batch_size_synth=max(1, int(args.batch_size_synth)),
            probe_train_sequences=max(20, int(args.probe_train_sequences)),
            probe_eval_sequences=max(10, int(args.probe_eval_sequences)),
            probe_seq_len=max(128, int(args.probe_seq_len)),
            probe_positions_per_seq=max(4, int(args.probe_positions_per_seq)),
            probe_batch_size=max(1, int(args.probe_batch_size)),
            bootstrap_samples=max(500, int(args.bootstrap_samples)),
            transfer_pairs=max(10, int(args.transfer_pairs)),
            seed=int(args.seed),
            pos_backend=str(args.pos_backend),
            allow_nltk_download=bool(args.allow_nltk_download),
        )
        summary[model_name] = rep

    _write_json(output_root / "a_summary.json", {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": summary,
        "tier": TIER_LABEL,
        "primary_test_id": PRIMARY_TEST_ID,
    })


if __name__ == "__main__":
    main()
