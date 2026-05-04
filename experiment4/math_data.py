from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from experiment3.theory1_si_circuits import load_profile_sequences


@dataclass(frozen=True)
class MCExample:
    example_id: str
    task: str
    prompt: str
    options: tuple[str, str, str, str]
    correct_index: int
    metadata: dict[str, Any]


def _rand_ndigits(rng: random.Random, digits: int) -> int:
    lo = 10 ** max(0, digits - 1)
    hi = (10 ** digits) - 1
    return int(rng.randint(lo, hi))


def _unique_choices(correct: int, distractors: list[int], rng: random.Random) -> list[int]:
    correct_i = int(correct)
    seen = {correct_i}
    pool: list[int] = []
    for d in distractors:
        d_i = int(d)
        if d_i >= 0 and d_i not in seen:
            pool.append(d_i)
            seen.add(d_i)
    while len(pool) < 3:
        jitter = int(rng.choice([-11, -7, -5, -3, -2, -1, 1, 2, 3, 5, 7, 11]))
        cand = int(correct_i + jitter)
        if cand >= 0 and cand not in seen:
            pool.append(cand)
            seen.add(cand)
    rng.shuffle(pool)
    out = [correct_i] + pool[:3]
    rng.shuffle(out)
    return out


def _addition_training_text(rng: random.Random, digits: int) -> str:
    a = _rand_ndigits(rng, digits)
    b = _rand_ndigits(rng, digits)
    ans = a + b
    return (
        f"Problem: Compute {a} + {b}.\n"
        f"Reasoning: Add digits from right to left and carry when needed.\n"
        f"Answer: {ans}"
    )


def _subtraction_training_text(rng: random.Random, digits: int) -> str:
    a = _rand_ndigits(rng, digits)
    b = _rand_ndigits(rng, digits)
    if b > a:
        a, b = b, a
    ans = a - b
    return (
        f"Problem: Compute {a} - {b}.\n"
        "Reasoning: Subtract digits from right to left and borrow when needed.\n"
        f"Answer: {ans}"
    )


def _counting_training_text(rng: random.Random) -> str:
    start = rng.randint(1, 50)
    step = rng.choice([1, 2, 3])
    n_terms = rng.randint(4, 8)
    seq = [start + i * step for i in range(n_terms)]
    next_val = start + n_terms * step
    return (
        f"Problem: Continue the counting sequence: {', '.join(str(x) for x in seq)}\n"
        f"Reasoning: The step is +{step}; continue one more term.\n"
        f"Answer: {next_val}"
    )


def _modular_training_text(rng: random.Random) -> str:
    m = rng.randint(3, 19)
    x = rng.randint(0, 250)
    y = rng.randint(0, 250)
    ans = (x + y) % m
    return (
        f"Problem: Compute ({x} + {y}) mod {m}.\n"
        f"Reasoning: Add first, then take remainder by {m}.\n"
        f"Answer: {ans}"
    )


def _sequence_training_text(rng: random.Random) -> str:
    mode = rng.choice(["arith", "geom", "fib"])
    if mode == "arith":
        start = rng.randint(1, 30)
        step = rng.randint(1, 9)
        seq = [start + i * step for i in range(5)]
        ans = seq[-1] + step
        reasoning = f"The pattern adds {step} each step."
    elif mode == "geom":
        start = rng.randint(1, 8)
        ratio = rng.choice([2, 3])
        seq = [start * (ratio ** i) for i in range(5)]
        ans = seq[-1] * ratio
        reasoning = f"The pattern multiplies by {ratio}."
    else:
        a = rng.randint(1, 5)
        b = rng.randint(2, 8)
        seq = [a, b]
        for _ in range(3):
            seq.append(seq[-1] + seq[-2])
        ans = seq[-1] + seq[-2]
        reasoning = "Each term is the sum of the two previous terms."
    return (
        f"Problem: Continue the sequence: {', '.join(str(x) for x in seq)}\n"
        f"Reasoning: {reasoning}\n"
        f"Answer: {ans}"
    )


def _build_eval_examples_add_sub(count: int, seed: int, *, task: str) -> list[MCExample]:
    rng = random.Random(seed + (11 if task == "addition" else 13))
    rows: list[MCExample] = []
    for i in range(count):
        digits = rng.randint(2, 6)
        a = _rand_ndigits(rng, digits)
        b = _rand_ndigits(rng, digits)
        if task == "subtraction" and b > a:
            a, b = b, a
        ans = a + b if task == "addition" else a - b
        prompt = f"Solve: {a} {'+' if task == 'addition' else '-'} {b} ="
        options = _unique_choices(ans, [ans - 1, ans + 1, ans - 10, ans + 10], rng)
        idx = options.index(ans)
        rows.append(
            MCExample(
                example_id=f"{task}_{seed}_{i:05d}",
                task=task,
                prompt=prompt,
                options=tuple(f" {x}" for x in options),
                correct_index=int(idx),
                metadata={"a": a, "b": b, "digits": digits},
            )
        )
    return rows


def _build_eval_examples_counting(count: int, seed: int) -> list[MCExample]:
    rng = random.Random(seed + 17)
    rows: list[MCExample] = []
    for i in range(count):
        start = rng.randint(1, 60)
        step = rng.choice([1, 2, 3, 4])
        n_terms = rng.randint(4, 7)
        seq = [start + j * step for j in range(n_terms)]
        ans = start + n_terms * step
        prompt = f"Continue: {', '.join(str(x) for x in seq)},"
        options = _unique_choices(ans, [ans - step, ans + step, ans + 2 * step, ans - 2 * step], rng)
        idx = options.index(ans)
        rows.append(
            MCExample(
                example_id=f"counting_{seed}_{i:05d}",
                task="counting",
                prompt=prompt,
                options=tuple(f" {x}" for x in options),
                correct_index=int(idx),
                metadata={"start": start, "step": step, "n_terms": n_terms},
            )
        )
    return rows


def _build_eval_examples_modular(count: int, seed: int) -> list[MCExample]:
    rng = random.Random(seed + 19)
    rows: list[MCExample] = []
    for i in range(count):
        m = rng.randint(3, 19)
        x = rng.randint(0, 250)
        y = rng.randint(0, 250)
        ans = (x + y) % m
        prompt = f"Compute ({x} + {y}) mod {m}:"
        distractors = [(x - y) % m, (x * y) % m, (x + y + 1) % m, (x + y + m - 1) % m]
        options = _unique_choices(ans, distractors, rng)
        idx = options.index(ans)
        rows.append(
            MCExample(
                example_id=f"modular_{seed}_{i:05d}",
                task="modular_arithmetic",
                prompt=prompt,
                options=tuple(f" {x}" for x in options),
                correct_index=int(idx),
                metadata={"x": x, "y": y, "m": m},
            )
        )
    return rows


def _build_eval_examples_sequence(count: int, seed: int) -> list[MCExample]:
    rng = random.Random(seed + 23)
    rows: list[MCExample] = []
    for i in range(count):
        mode = rng.choice(["arith", "geom", "fib"])
        if mode == "arith":
            start = rng.randint(1, 20)
            step = rng.randint(1, 7)
            seq = [start + j * step for j in range(5)]
            ans = seq[-1] + step
            distractors = [seq[-1], seq[-1] + 2 * step, seq[-1] - step]
        elif mode == "geom":
            start = rng.randint(1, 6)
            ratio = rng.choice([2, 3])
            seq = [start * (ratio ** j) for j in range(5)]
            ans = seq[-1] * ratio
            distractors = [seq[-1] + ratio, seq[-1] * (ratio + 1), seq[-1]]
        else:
            a = rng.randint(1, 5)
            b = rng.randint(2, 8)
            seq = [a, b]
            for _ in range(3):
                seq.append(seq[-1] + seq[-2])
            ans = seq[-1] + seq[-2]
            distractors = [seq[-1], seq[-1] + seq[-3], seq[-2] + seq[-3]]
        prompt = f"Complete the sequence: {', '.join(str(x) for x in seq)},"
        options = _unique_choices(ans, distractors, rng)
        idx = options.index(ans)
        rows.append(
            MCExample(
                example_id=f"sequence_{seed}_{i:05d}",
                task="sequence_continuation",
                prompt=prompt,
                options=tuple(f" {x}" for x in options),
                correct_index=int(idx),
                metadata={"mode": mode},
            )
        )
    return rows


def build_math_training_texts(per_task: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    rows: list[str] = []
    for _ in range(per_task):
        rows.append(_addition_training_text(rng, digits=rng.randint(2, 6)))
        rows.append(_subtraction_training_text(rng, digits=rng.randint(2, 6)))
        rows.append(_counting_training_text(rng))
        rows.append(_modular_training_text(rng))
        rows.append(_sequence_training_text(rng))
    rng.shuffle(rows)
    return rows


def load_control_texts(*, tokenizer, model_name: str, count: int, seq_len: int, seed: int) -> list[str]:
    sequences = load_profile_sequences(tokenizer=tokenizer, model_name=model_name, num_sequences=max(4, count), seq_len=seq_len)
    rows: list[str] = []
    for toks in sequences[:count]:
        txt = tokenizer.decode(toks, skip_special_tokens=True)
        txt = txt.strip()
        if txt:
            rows.append(txt)
    if len(rows) < count:
        filler = "General language control text about facts, entities, and syntax."
        rows.extend([filler for _ in range(count - len(rows))])
    rng = random.Random(seed + 101)
    rng.shuffle(rows)
    return rows[:count]


def build_training_mix(
    *,
    tokenizer,
    model_name: str,
    per_task: int,
    control_fraction: float,
    seq_len: int,
    seed: int,
) -> list[str]:
    math_rows = build_math_training_texts(per_task=per_task, seed=seed)
    n_control = int(round((control_fraction / max(1e-8, 1.0 - control_fraction)) * len(math_rows)))
    n_control = max(1, n_control)
    control_rows = load_control_texts(
        tokenizer=tokenizer,
        model_name=model_name,
        count=n_control,
        seq_len=seq_len,
        seed=seed,
    )
    all_rows = list(math_rows) + list(control_rows)
    rng = random.Random(seed + 211)
    rng.shuffle(all_rows)
    return all_rows


def build_math_eval_battery(count_per_task: int, seed: int) -> dict[str, list[MCExample]]:
    return {
        "addition": _build_eval_examples_add_sub(count_per_task, seed, task="addition"),
        "subtraction": _build_eval_examples_add_sub(count_per_task, seed, task="subtraction"),
        "counting": _build_eval_examples_counting(count_per_task, seed),
        "modular_arithmetic": _build_eval_examples_modular(count_per_task, seed),
        "sequence_continuation": _build_eval_examples_sequence(count_per_task, seed),
    }
