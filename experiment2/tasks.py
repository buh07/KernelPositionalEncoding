from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Iterable

from experiment2.long_offsets import active_long_offsets


COPY_SHORT_OFFSETS = (1, 2, 4, 8, 16)
BRIDGE_OFFSETS = (32, 64, 96)


@dataclass(frozen=True)
class TokenPools:
    filler: tuple[int, ...]
    keys: tuple[int, ...]
    values: tuple[int, ...]
    reserve: tuple[int, ...]
    query_marker: int
    retrieval_marker: int
    no_match_token: int


@dataclass(frozen=True)
class TaskExample:
    id: str
    task_name: str
    tokens: list[int]
    target_positions: list[int]
    target_tokens: list[int]
    dependency_span: int
    task_class: str
    seed: int
    model: str
    length: int
    task_params: dict
    pair_count: int | None
    query_key: int | None
    distractor_key: int | None
    match_rule: str
    has_no_match: bool

    def to_json_dict(self) -> dict:
        return asdict(self)


def build_token_pools(model_name: str, vocab_size: int, special_ids: Iterable[int]) -> TokenPools:
    excluded = {idx for idx in special_ids if idx is not None and idx >= 0}
    candidates = [idx for idx in range(vocab_size) if idx not in excluded]
    if len(candidates) < 20_000:
        raise RuntimeError(
            f"Vocabulary too small after exclusions for {model_name}: {len(candidates)} candidate tokens."
        )
    rng = random.Random(_stable_seed("pool", model_name))
    rng.shuffle(candidates)
    keys = tuple(candidates[:4096])
    values = tuple(candidates[4096:8192])
    reserve = tuple(candidates[8192:8320])
    filler = tuple(candidates[8320:])
    return TokenPools(
        filler=filler,
        keys=keys,
        values=values,
        reserve=reserve,
        query_marker=reserve[0],
        retrieval_marker=reserve[1],
        no_match_token=reserve[2],
    )


def generate_task_examples(
    *,
    task_name: str,
    model_name: str,
    seq_len: int,
    seed: int,
    count: int,
    pools: TokenPools,
    span_override: int | None = None,
    span_choices: tuple[int, ...] | None = None,
) -> list[TaskExample]:
    builders = {
        "local_copy_offset": _build_local_copy_offset,
        "delayed_copy": _build_delayed_copy,
        "local_key_match": _build_local_key_match,
        "long_range_retrieval": _build_long_range_retrieval,
        "copy_offset_bridge": _build_copy_offset_bridge,
        "retrieval_bridge": _build_retrieval_bridge,
    }
    if task_name not in builders:
        raise ValueError(f"Unsupported synthetic task: {task_name}")

    out: list[TaskExample] = []
    for seq_id in range(count):
        example: TaskExample | None = None
        for retry_idx in range(64):
            rng = random.Random(_stable_seed("example", model_name, task_name, seq_len, seed, seq_id, retry_idx))
            example = builders[task_name](
                model_name=model_name,
                seq_len=seq_len,
                seed=seed,
                seq_id=seq_id,
                rng=rng,
                pools=pools,
                span_override=span_override,
                span_choices=span_choices,
            )
            ambiguity = float(example.task_params.get("ambiguity_rate", 0.0))
            if ambiguity <= 0.05:
                break
        if example is None:
            raise RuntimeError(f"Failed to generate example for {task_name} seq_id={seq_id}")
        if float(example.task_params.get("ambiguity_rate", 0.0)) > 0.05:
            raise RuntimeError(
                f"Ambiguity audit failed for {task_name} seq_id={seq_id}: "
                f"{example.task_params.get('ambiguity_rate')} > 0.05 after deterministic retries"
            )
        _validate_example(example)
        out.append(example)
    return out


def _build_local_copy_offset(
    *,
    model_name: str,
    seq_len: int,
    seed: int,
    seq_id: int,
    rng: random.Random,
    pools: TokenPools,
    span_override: int | None = None,
    span_choices: tuple[int, ...] | None = None,
) -> TaskExample:
    valid_offsets = [k for k in COPY_SHORT_OFFSETS if k < seq_len]
    if not valid_offsets:
        raise RuntimeError(f"No valid short offsets for seq_len={seq_len}")
    offset = rng.choice(valid_offsets)
    tokens = _sample_filler_tokens(rng, pools.filler, seq_len)
    valid_targets = list(range(offset, seq_len))
    target_count = min(32, len(valid_targets))
    target_positions = _sample_nonchaining_targets(
        rng=rng,
        valid_targets=valid_targets,
        offset=offset,
        target_count=target_count,
    )
    for t in target_positions:
        tokens[t] = tokens[t - offset]
    target_tokens = [tokens[t] for t in target_positions]
    return TaskExample(
        id=f"{model_name}:{seed}:{seq_id}:local_copy_offset",
        task_name="local_copy_offset",
        tokens=tokens,
        target_positions=target_positions,
        target_tokens=target_tokens,
        dependency_span=offset,
        task_class="short",
        seed=seed,
        model=model_name,
        length=seq_len,
        task_params={
            "offset": offset,
            "target_overlap_prevented": True,
            "target_count_actual": len(target_positions),
        },
        pair_count=None,
        query_key=None,
        distractor_key=None,
        match_rule="copy_t_minus_k",
        has_no_match=False,
    )


def _build_copy_offset_bridge(
    *,
    model_name: str,
    seq_len: int,
    seed: int,
    seq_id: int,
    rng: random.Random,
    pools: TokenPools,
    span_override: int | None = None,
    span_choices: tuple[int, ...] | None = None,
) -> TaskExample:
    if span_override is not None:
        valid_offsets = [int(span_override)] if int(span_override) < seq_len else []
    else:
        valid_offsets = [k for k in BRIDGE_OFFSETS if k < seq_len]
    if not valid_offsets:
        raise RuntimeError(f"No valid bridge offsets for seq_len={seq_len}")
    offset = rng.choice(valid_offsets)
    tokens = _sample_filler_tokens(rng, pools.filler, seq_len)
    valid_targets = list(range(offset, seq_len))
    target_count = min(24, len(valid_targets))
    target_positions = _sample_nonchaining_targets(
        rng=rng,
        valid_targets=valid_targets,
        offset=offset,
        target_count=target_count,
    )
    for t in target_positions:
        tokens[t] = tokens[t - offset]
    target_tokens = [tokens[t] for t in target_positions]
    return TaskExample(
        id=f"{model_name}:{seed}:{seq_id}:copy_offset_bridge",
        task_name="copy_offset_bridge",
        tokens=tokens,
        target_positions=target_positions,
        target_tokens=target_tokens,
        dependency_span=offset,
        task_class="bridge",
        seed=seed,
        model=model_name,
        length=seq_len,
        task_params={"offset": offset, "bridge": True},
        pair_count=None,
        query_key=None,
        distractor_key=None,
        match_rule="copy_t_minus_k",
        has_no_match=False,
    )


def _build_delayed_copy(
    *,
    model_name: str,
    seq_len: int,
    seed: int,
    seq_id: int,
    rng: random.Random,
    pools: TokenPools,
    span_override: int | None = None,
    span_choices: tuple[int, ...] | None = None,
) -> TaskExample:
    if span_override is not None:
        base = (int(span_override),)
    else:
        base = span_choices if span_choices else active_long_offsets(strict=False)
    valid_offsets = [k for k in base if k < seq_len]
    if not valid_offsets:
        raise RuntimeError(f"No valid delayed-copy offsets for seq_len={seq_len}")
    offset = rng.choice(valid_offsets)
    tokens = _sample_filler_tokens(rng, pools.filler, seq_len)
    valid_targets = list(range(offset, seq_len))
    target_count = min(16, len(valid_targets))
    target_positions = _sample_nonchaining_targets(
        rng=rng,
        valid_targets=valid_targets,
        offset=offset,
        target_count=target_count,
    )
    for t in target_positions:
        tokens[t] = tokens[t - offset]
    target_tokens = [tokens[t] for t in target_positions]
    return TaskExample(
        id=f"{model_name}:{seed}:{seq_id}:delayed_copy",
        task_name="delayed_copy",
        tokens=tokens,
        target_positions=target_positions,
        target_tokens=target_tokens,
        dependency_span=offset,
        task_class="long",
        seed=seed,
        model=model_name,
        length=seq_len,
        task_params={
            "offset": offset,
            "target_overlap_prevented": True,
            "target_count_actual": len(target_positions),
        },
        pair_count=None,
        query_key=None,
        distractor_key=None,
        match_rule="copy_t_minus_k",
        has_no_match=False,
    )


def _build_local_key_match(
    *,
    model_name: str,
    seq_len: int,
    seed: int,
    seq_id: int,
    rng: random.Random,
    pools: TokenPools,
    span_override: int | None = None,
    span_choices: tuple[int, ...] | None = None,
) -> TaskExample:
    tokens = _sample_filler_tokens(rng, pools.filler, seq_len)
    pair_count = 24
    key_positions = _sample_positions_with_min_spacing(
        rng=rng,
        start=2,
        stop=seq_len - 2,
        count=pair_count,
        min_spacing=2,
    )
    keys = rng.sample(pools.keys, k=pair_count)
    values = rng.sample(pools.values, k=pair_count)
    key_to_value: dict[int, int] = {}
    pos_to_key: dict[int, int] = {}
    for pos, key, value in zip(key_positions, keys, values):
        tokens[pos] = key
        if pos + 1 < seq_len:
            tokens[pos + 1] = value
        key_to_value[key] = value
        pos_to_key[pos] = key

    occupied = set(key_positions)
    occupied.update(pos + 1 for pos in key_positions if (pos + 1) < seq_len)
    candidate_targets = [pos for pos in range(1, seq_len) if pos not in occupied]
    target_count = min(32, len(candidate_targets))
    target_positions = sorted(rng.sample(candidate_targets, k=target_count))
    target_tokens: list[int] = []
    no_match_count = 0
    max_span = 0
    for t in target_positions:
        lo = max(0, t - 16)
        hi = t - 1
        window_positions = [p for p in key_positions if lo <= p <= hi]
        if not window_positions:
            target = pools.no_match_token
            no_match_count += 1
            span = 17
        else:
            nearest = min(window_positions, key=lambda p: (t - p, p))
            key_tok = pos_to_key[nearest]
            target = key_to_value[key_tok]
            span = t - nearest
        tokens[t] = target
        target_tokens.append(target)
        max_span = max(max_span, span)

    key_overwrite_count = sum(1 for pos in key_positions if tokens[pos] != pos_to_key[pos])

    return TaskExample(
        id=f"{model_name}:{seed}:{seq_id}:local_key_match",
        task_name="local_key_match",
        tokens=tokens,
        target_positions=target_positions,
        target_tokens=target_tokens,
        dependency_span=max_span,
        task_class="short",
        seed=seed,
        model=model_name,
        length=seq_len,
        task_params={
            "window": 16,
            "pair_count": pair_count,
            "no_match_rate": no_match_count / max(target_count, 1),
            "no_match_token": pools.no_match_token,
            "key_overwrite_count": int(key_overwrite_count),
        },
        pair_count=pair_count,
        query_key=None,
        distractor_key=None,
        match_rule="nearest_preceding_key_within_16",
        has_no_match=no_match_count > 0,
    )


def _build_long_range_retrieval(
    *,
    model_name: str,
    seq_len: int,
    seed: int,
    seq_id: int,
    rng: random.Random,
    pools: TokenPools,
    span_override: int | None = None,
    span_choices: tuple[int, ...] | None = None,
) -> TaskExample:
    tokens = _sample_filler_tokens(rng, pools.filler, seq_len)
    pair_count = 16
    keys = rng.sample(pools.keys, k=pair_count * 2)
    values = rng.sample(pools.values, k=pair_count)
    true_keys = keys[:pair_count]
    distractor_keys = keys[pair_count:]
    occupied: set[int] = set()

    target_positions: list[int] = []
    target_tokens: list[int] = []
    spans: list[int] = []
    max_retries = 200
    ambiguity_count = 0
    if span_override is not None:
        base_long = (int(span_override),)
    else:
        base_long = span_choices if span_choices else active_long_offsets(strict=False)

    for idx in range(pair_count):
        true_key = true_keys[idx]
        true_value = values[idx]
        distractor_key = distractor_keys[idx]
        placed = False
        for retry in range(max_retries):
            key_pos = rng.randint(0, max(0, seq_len - 260))
            gap_options = [g for g in base_long if key_pos + g + 3 < seq_len]
            if not gap_options:
                continue
            gap = rng.choice(gap_options)
            query_pos = key_pos + gap
            positions = {key_pos, key_pos + 1, query_pos, query_pos + 1, query_pos + 2, query_pos + 3}
            if max(positions) >= seq_len:
                continue
            if positions & occupied:
                continue
            occupied |= positions
            tokens[key_pos] = true_key
            tokens[key_pos + 1] = true_value
            tokens[query_pos] = pools.retrieval_marker
            tokens[query_pos + 1] = true_key
            tokens[query_pos + 2] = distractor_key
            tokens[query_pos + 3] = true_value
            antecedent_positions = [
                p for p in key_positions_for_token(tokens=tokens, token=true_key, max_pos=query_pos)
            ]
            if len(antecedent_positions) > 1:
                ambiguity_count += 1
            target_positions.append(query_pos + 3)
            target_tokens.append(true_value)
            spans.append(query_pos - key_pos)
            placed = True
            break
        if not placed:
            raise RuntimeError(f"Failed to place retrieval pair idx={idx} for seq_len={seq_len}")

    order = sorted(range(len(target_positions)), key=lambda i: target_positions[i])
    target_positions = [target_positions[i] for i in order]
    target_tokens = [target_tokens[i] for i in order]
    spans = [spans[i] for i in order]
    return TaskExample(
        id=f"{model_name}:{seed}:{seq_id}:long_range_retrieval",
        task_name="long_range_retrieval",
        tokens=tokens,
        target_positions=target_positions,
        target_tokens=target_tokens,
        dependency_span=max(spans) if spans else 0,
        task_class="long",
        seed=seed,
        model=model_name,
        length=seq_len,
        task_params={
            "pair_count": pair_count,
            "min_gap": min([s for s in base_long if s < seq_len], default=0),
            "ambiguity_rate": ambiguity_count / max(pair_count, 1),
        },
        pair_count=pair_count,
        query_key=true_keys[0] if true_keys else None,
        distractor_key=distractor_keys[0] if distractor_keys else None,
        match_rule="nearest_valid_antecedent",
        has_no_match=False,
    )


def _build_retrieval_bridge(
    *,
    model_name: str,
    seq_len: int,
    seed: int,
    seq_id: int,
    rng: random.Random,
    pools: TokenPools,
    span_override: int | None = None,
    span_choices: tuple[int, ...] | None = None,
) -> TaskExample:
    tokens = _sample_filler_tokens(rng, pools.filler, seq_len)
    pair_count = 10
    keys = rng.sample(pools.keys, k=pair_count * 2)
    values = rng.sample(pools.values, k=pair_count)
    true_keys = keys[:pair_count]
    distractor_keys = keys[pair_count:]
    occupied: set[int] = set()
    if span_override is not None:
        spans_available = [int(span_override)] if int(span_override) + 3 < seq_len else []
    else:
        spans_available = [s for s in BRIDGE_OFFSETS if s + 3 < seq_len]
    if not spans_available:
        raise RuntimeError(f"No valid bridge retrieval spans for seq_len={seq_len}")

    target_positions: list[int] = []
    target_tokens: list[int] = []
    spans: list[int] = []
    for idx in range(pair_count):
        true_key = true_keys[idx]
        true_value = values[idx]
        distractor_key = distractor_keys[idx]
        placed = False
        for _ in range(200):
            key_pos = rng.randint(0, max(0, seq_len - 200))
            gap = rng.choice(spans_available)
            query_pos = key_pos + gap
            positions = {key_pos, key_pos + 1, query_pos, query_pos + 1, query_pos + 2, query_pos + 3}
            if max(positions) >= seq_len:
                continue
            if positions & occupied:
                continue
            occupied |= positions
            tokens[key_pos] = true_key
            tokens[key_pos + 1] = true_value
            tokens[query_pos] = pools.retrieval_marker
            tokens[query_pos + 1] = true_key
            tokens[query_pos + 2] = distractor_key
            tokens[query_pos + 3] = true_value
            target_positions.append(query_pos + 3)
            target_tokens.append(true_value)
            spans.append(gap)
            placed = True
            break
        if not placed:
            raise RuntimeError(f"Failed to place retrieval bridge pair idx={idx} for seq_len={seq_len}")

    order = sorted(range(len(target_positions)), key=lambda i: target_positions[i])
    target_positions = [target_positions[i] for i in order]
    target_tokens = [target_tokens[i] for i in order]
    spans = [spans[i] for i in order]
    return TaskExample(
        id=f"{model_name}:{seed}:{seq_id}:retrieval_bridge",
        task_name="retrieval_bridge",
        tokens=tokens,
        target_positions=target_positions,
        target_tokens=target_tokens,
        dependency_span=max(spans) if spans else 0,
        task_class="bridge",
        seed=seed,
        model=model_name,
        length=seq_len,
        task_params={"pair_count": pair_count, "bridge_spans": BRIDGE_OFFSETS},
        pair_count=pair_count,
        query_key=true_keys[0] if true_keys else None,
        distractor_key=distractor_keys[0] if distractor_keys else None,
        match_rule="nearest_valid_antecedent",
        has_no_match=False,
    )


def _sample_positions_with_min_spacing(
    *,
    rng: random.Random,
    start: int,
    stop: int,
    count: int,
    min_spacing: int,
) -> list[int]:
    positions: list[int] = []
    candidates = list(range(start, stop))
    rng.shuffle(candidates)
    for pos in candidates:
        if all(abs(pos - existing) >= min_spacing for existing in positions):
            positions.append(pos)
            if len(positions) == count:
                return sorted(positions)
    raise RuntimeError(
        f"Unable to sample {count} positions with min spacing {min_spacing} in range [{start}, {stop})."
    )


def _sample_nonchaining_targets(
    *,
    rng: random.Random,
    valid_targets: list[int],
    offset: int,
    target_count: int,
) -> list[int]:
    candidates = list(valid_targets)
    rng.shuffle(candidates)
    selected: set[int] = set()
    for t in candidates:
        # Enforce non-chaining in both directions so no chosen target depends on
        # another chosen target under the same fixed offset.
        if (t - offset) in selected or (t + offset) in selected:
            continue
        selected.add(t)
        if len(selected) >= target_count:
            break
    return sorted(selected)


def _sample_filler_tokens(rng: random.Random, filler_pool: tuple[int, ...], seq_len: int) -> list[int]:
    return [filler_pool[rng.randrange(len(filler_pool))] for _ in range(seq_len)]


def key_positions_for_token(*, tokens: list[int], token: int, max_pos: int) -> list[int]:
    out = []
    for idx in range(max(0, max_pos)):
        if tokens[idx] == token:
            out.append(idx)
    return out


def _validate_example(example: TaskExample) -> None:
    if len(example.tokens) != example.length:
        raise RuntimeError(f"Example {example.id} length mismatch: {len(example.tokens)} vs {example.length}")
    if not example.target_positions:
        raise RuntimeError(f"Example {example.id} has no target positions.")
    if len(example.target_positions) != len(example.target_tokens):
        raise RuntimeError(f"Example {example.id} target position/token length mismatch.")
    for pos in example.target_positions:
        if pos <= 0 or pos >= example.length:
            raise RuntimeError(f"Example {example.id} invalid target position {pos}.")


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)
