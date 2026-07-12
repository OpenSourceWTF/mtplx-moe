#!/usr/bin/env python3
"""Analyze Hy3 route traces for cache, global-pool, and cluster headroom.

The legacy whole-decode metrics remain in ``policies``.  The issue #9 offline
gate lives in ``held_out`` and uses a chronological train/evaluation split so
that deployable quota and cluster metadata never sees evaluation routes.
"""

from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Hashable, Iterable, Sequence, TypeVar


CacheKey = TypeVar("CacheKey", bound=Hashable)
ExpertKey = tuple[int, int]


def _sets(values: list[int], top_k: int) -> list[tuple[int, ...]]:
    if len(values) % top_k:
        raise ValueError("route entry is not divisible by top-k")
    rows = []
    for start in range(0, len(values), top_k):
        row = tuple(values[start : start + top_k])
        if len(set(row)) != len(row):
            raise ValueError("router selected a duplicate expert")
        rows.append(row)
    return rows


def _prompt_rank(prompt_rows: list[tuple[int, ...]]) -> list[int]:
    counts = Counter(expert for row in prompt_rows for expert in row)
    return [expert for expert, _count in counts.most_common()]


def _simulate_static(
    rows: list[tuple[int, ...]],
    initial_rank: list[int],
    capacity: int,
) -> tuple[int, int]:
    cache = set(initial_rank[:capacity])
    hits = sum(expert in cache for row in rows for expert in row)
    requests = sum(len(row) for row in rows)
    return int(hits), requests - int(hits)


def _simulate_lru(
    rows: list[tuple[int, ...]],
    initial_rank: list[int],
    capacity: int,
) -> tuple[int, int]:
    cache = dict.fromkeys(reversed(initial_rank[:capacity]))
    hits = misses = 0
    for row in rows:
        for expert in row:
            if expert in cache:
                hits += 1
                cache.pop(expert)
                cache[expert] = None
            else:
                misses += 1
                if capacity:
                    if len(cache) >= capacity:
                        cache.pop(next(iter(cache)))
                    cache[expert] = None
    return hits, misses


def _simulate_belady(
    rows: list[tuple[int, ...]],
    initial_rank: list[int],
    capacity: int,
) -> tuple[int, int]:
    sequence = [expert for row in rows for expert in row]
    future: dict[int, deque[int]] = defaultdict(deque)
    for position, expert in enumerate(sequence):
        future[expert].append(position)
    cache = set(initial_rank[:capacity])
    hits = misses = 0
    never = len(sequence) + 1
    for position, expert in enumerate(sequence):
        positions = future[expert]
        if not positions or positions[0] != position:
            raise AssertionError("Belady future index is inconsistent")
        positions.popleft()
        if expert in cache:
            hits += 1
            continue
        misses += 1
        if not capacity:
            continue
        if len(cache) >= capacity:
            victim = max(
                cache,
                key=lambda candidate: (
                    future[candidate][0] if future[candidate] else never,
                    candidate,
                ),
            )
            cache.remove(victim)
        cache.add(expert)
    return hits, misses


def _initial_lru_order(
    initial_rank: Sequence[CacheKey], capacity: int
) -> list[CacheKey]:
    """Return an LRU-to-MRU order from a most-desirable-first seed rank."""

    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    return list(reversed(initial_rank[:capacity]))


def _run_lru_sequence(
    sequence: Iterable[CacheKey],
    initial_lru_order: Sequence[CacheKey],
    capacity: int,
) -> tuple[int, int, list[CacheKey]]:
    """Replay one key stream and return metrics plus final LRU-to-MRU state."""

    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    cache: dict[CacheKey, None] = {}
    for key in initial_lru_order:
        if key in cache:
            cache.pop(key)
        cache[key] = None
        if len(cache) > capacity:
            cache.pop(next(iter(cache)))
    hits = misses = 0
    for key in sequence:
        if key in cache:
            hits += 1
            cache.pop(key)
            cache[key] = None
            continue
        misses += 1
        if capacity:
            if len(cache) >= capacity:
                cache.pop(next(iter(cache)))
            cache[key] = None
    return hits, misses, list(cache)


def _run_belady_sequence(
    sequence: Sequence[CacheKey],
    initial_cache: Iterable[CacheKey],
    capacity: int,
) -> tuple[int, int]:
    """Replay Belady's clairvoyant replacement bound for arbitrary keys."""

    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    future: dict[CacheKey, deque[int]] = defaultdict(deque)
    for position, key in enumerate(sequence):
        future[key].append(position)
    cache = set(initial_cache)
    if len(cache) > capacity:
        raise ValueError("initial cache exceeds capacity")
    hits = misses = 0
    never = len(sequence) + 1
    for position, key in enumerate(sequence):
        positions = future[key]
        if not positions or positions[0] != position:
            raise AssertionError("Belady future index is inconsistent")
        positions.popleft()
        if key in cache:
            hits += 1
            continue
        misses += 1
        if not capacity:
            continue
        if len(cache) >= capacity:
            victim = max(
                cache,
                key=lambda candidate: (
                    future[candidate][0] if future[candidate] else never,
                    candidate,
                ),
            )
            cache.remove(victim)
        cache.add(key)
    return hits, misses


def _global_sequence(
    layers: Sequence[int],
    routes: dict[int, list[tuple[int, ...]]],
) -> list[ExpertKey]:
    """Flatten decode in runtime order: token, layer, then router rank."""

    if not layers:
        return []
    steps = len(routes[layers[0]])
    if any(len(routes[layer]) != steps for layer in layers):
        raise ValueError("all layers must have the same route count")
    return [
        (layer, expert)
        for step in range(steps)
        for layer in layers
        for expert in routes[layer][step]
    ]


def _metric(hits: int, misses: int) -> dict[str, float | int]:
    requests = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "requests": requests,
        "hit_rate": hits / requests if requests else 0.0,
    }


def _sum_policy(
    layers: Iterable[int],
    decode: dict[int, list[tuple[int, ...]]],
    ranks: dict[int, list[int]],
    capacities: dict[int, int],
    simulator,
) -> dict[str, float | int]:
    hits = misses = 0
    for layer in layers:
        layer_hits, layer_misses = simulator(
            decode[layer], ranks[layer], capacities[layer]
        )
        hits += layer_hits
        misses += layer_misses
    return _metric(hits, misses)


def _decorate_io_metric(
    metric: dict[str, float | int | None],
    token_count: int,
    record_bytes: int,
    *,
    physical_records: int | None = None,
    measured_ssd_bytes_per_second: float | None = None,
) -> dict[str, float | int | None]:
    """Attach physical-I/O metrics without conflating misses and cluster reads."""

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    records = int(metric["misses"]) if physical_records is None else physical_records
    bytes_read = records * record_bytes
    bytes_per_token = bytes_read / token_count
    metric.update(
        {
            "misses_per_token": int(metric["misses"]) / token_count,
            "records_read": records,
            "records_read_per_token": records / token_count,
            "bytes_read": bytes_read,
            "bytes_per_token": bytes_per_token,
            "ssd_only_ceiling_tok_s_3p4GBs": (
                3.4e9 / bytes_per_token if bytes_per_token else None
            ),
            "ssd_only_ceiling_tok_s_5p1GBs": (
                5.1e9 / bytes_per_token if bytes_per_token else None
            ),
            "ssd_only_ceiling_tok_s_6p1GiBs": (
                6.1 * 2**30 / bytes_per_token if bytes_per_token else None
            ),
            "projected_tok_s_at_measured_bandwidth": (
                measured_ssd_bytes_per_second / bytes_per_token
                if bytes_per_token and measured_ssd_bytes_per_second is not None
                else None
            ),
        }
    )
    return metric


def _warm_partitioned_lru(
    layers: Sequence[int],
    train: dict[int, list[tuple[int, ...]]],
    ranks: dict[int, list[int]],
    capacities: dict[int, int],
) -> dict[int, list[int]]:
    states: dict[int, list[int]] = {}
    for layer in layers:
        _hits, _misses, states[layer] = _run_lru_sequence(
            (expert for row in train[layer] for expert in row),
            _initial_lru_order(ranks[layer], capacities[layer]),
            capacities[layer],
        )
    return states


def _evaluate_partitioned_lru(
    layers: Sequence[int],
    evaluation: dict[int, list[tuple[int, ...]]],
    warm_states: dict[int, list[int]],
    capacities: dict[int, int],
) -> dict[str, float | int]:
    hits = misses = 0
    for layer in layers:
        layer_hits, layer_misses, _state = _run_lru_sequence(
            (expert for row in evaluation[layer] for expert in row),
            warm_states[layer],
            capacities[layer],
        )
        hits += layer_hits
        misses += layer_misses
    return _metric(hits, misses)


def _evaluate_partitioned_belady(
    layers: Sequence[int],
    evaluation: dict[int, list[tuple[int, ...]]],
    warm_states: dict[int, list[int]],
    capacities: dict[int, int],
) -> dict[str, float | int]:
    hits = misses = 0
    for layer in layers:
        sequence = [expert for row in evaluation[layer] for expert in row]
        layer_hits, layer_misses = _run_belady_sequence(
            sequence,
            warm_states[layer],
            capacities[layer],
        )
        hits += layer_hits
        misses += layer_misses
    return _metric(hits, misses)


def _lru_training_hit_curves(
    layers: Sequence[int],
    train: dict[int, list[tuple[int, ...]]],
    ranks: dict[int, list[int]],
    expert_count: int,
) -> dict[int, list[int]]:
    """Measure each layer's train-prefix utility for every possible quota."""

    curves: dict[int, list[int]] = {}
    for layer in layers:
        sequence = [expert for row in train[layer] for expert in row]
        layer_curve = []
        for capacity in range(expert_count + 1):
            hits, _misses, _state = _run_lru_sequence(
                sequence,
                _initial_lru_order(ranks[layer], capacity),
                capacity,
            )
            layer_curve.append(hits)
        curves[layer] = layer_curve
    return curves


def _rebalance_trained_quotas(
    hit_curves: dict[int, list[int]],
    initial_capacities: dict[int, int],
    hysteresis_hits: int,
) -> tuple[dict[int, int], dict[str, int]]:
    """Move one slot at a time only when train-prefix LRU hits improve.

    This is intentionally deployable rather than an evaluation oracle: curves
    come exclusively from the chronological training prefix, the total slot
    count never changes, and hysteresis can reject fragile one-hit transfers.
    """

    if hysteresis_hits < 0:
        raise ValueError("hysteresis_hits must be non-negative")
    layers = sorted(hit_curves)
    if set(initial_capacities) != set(layers):
        raise ValueError("capacity and curve layers differ")
    capacities = dict(initial_capacities)
    before = sum(hit_curves[layer][capacities[layer]] for layer in layers)
    moves = 0
    while True:
        best: tuple[int, int, int] | None = None
        for donor in layers:
            donor_capacity = capacities[donor]
            if donor_capacity <= 0:
                continue
            loss = (
                hit_curves[donor][donor_capacity]
                - hit_curves[donor][donor_capacity - 1]
            )
            for recipient in layers:
                if recipient == donor:
                    continue
                recipient_capacity = capacities[recipient]
                if recipient_capacity + 1 >= len(hit_curves[recipient]):
                    continue
                gain = (
                    hit_curves[recipient][recipient_capacity + 1]
                    - hit_curves[recipient][recipient_capacity]
                )
                net = gain - loss
                candidate = (net, -donor, -recipient)
                if best is None or candidate > best:
                    best = candidate
        if best is None or best[0] <= hysteresis_hits:
            break
        _net, negative_donor, negative_recipient = best
        donor = -negative_donor
        recipient = -negative_recipient
        capacities[donor] -= 1
        capacities[recipient] += 1
        moves += 1
    after = sum(hit_curves[layer][capacities[layer]] for layer in layers)
    if sum(capacities.values()) != sum(initial_capacities.values()):
        raise AssertionError("quota rebalancing changed the slot budget")
    return capacities, {
        "reallocation_moves": moves,
        "training_hits_before": before,
        "training_hits_after": after,
        "training_hit_gain": after - before,
        "hysteresis_hits": hysteresis_hits,
    }


def _greedy_oracle_static_capacities(
    layers: list[int],
    decode: dict[int, list[tuple[int, ...]]],
    total_slots: int,
    expert_count: int,
) -> dict[int, int]:
    frequencies = {
        layer: [
            count
            for _expert, count in Counter(
                expert for row in decode[layer] for expert in row
            ).most_common(expert_count)
        ]
        for layer in layers
    }
    capacities = {layer: 0 for layer in layers}
    heap: list[tuple[int, int]] = []
    for layer in layers:
        marginal = frequencies[layer][0] if frequencies[layer] else 0
        heapq.heappush(heap, (-marginal, layer))
    for _ in range(min(total_slots, expert_count * len(layers))):
        _negative, layer = heapq.heappop(heap)
        capacities[layer] += 1
        index = capacities[layer]
        marginal = frequencies[layer][index] if index < len(frequencies[layer]) else 0
        heapq.heappush(heap, (-marginal, layer))
    return capacities


def _global_prompt_rank(
    layers: Sequence[int],
    prompt: dict[int, list[tuple[int, ...]]],
    expert_count: int,
) -> list[ExpertKey]:
    """Rank all layer/expert records by causal prompt frequency."""

    counts = Counter(_global_sequence(layers, prompt))
    keys = [(layer, expert) for layer in layers for expert in range(expert_count)]
    return sorted(keys, key=lambda key: (-counts[key], key))


def _build_coactivation_clusters(
    layers: Sequence[int],
    training: dict[int, list[tuple[int, ...]]],
    expert_count: int,
    cluster_size: int,
) -> dict[ExpertKey, tuple[ExpertKey, ...]]:
    """Greedily partition each layer using train-only top-k co-occurrence."""

    if cluster_size <= 0:
        raise ValueError("cluster_size must be positive")
    result: dict[ExpertKey, tuple[ExpertKey, ...]] = {}
    for layer in layers:
        frequency: Counter[int] = Counter()
        pair_counts: Counter[tuple[int, int]] = Counter()
        for row in training[layer]:
            frequency.update(row)
            for left_index, left in enumerate(row):
                for right in row[left_index + 1 :]:
                    pair = (left, right) if left < right else (right, left)
                    pair_counts[pair] += 1
        row_count = len(training[layer])
        pair_lift = {
            pair: count * row_count / (frequency[pair[0]] * frequency[pair[1]])
            for pair, count in pair_counts.items()
            if frequency[pair[0]] and frequency[pair[1]]
        }
        lift_degree: Counter[int] = Counter()
        for (left, right), lift in pair_lift.items():
            lift_degree[left] += max(lift - 1.0, 0.0)
            lift_degree[right] += max(lift - 1.0, 0.0)
        remaining = set(range(expert_count))
        while remaining:
            seed = max(
                remaining,
                key=lambda expert: (
                    frequency[expert],
                    lift_degree[expert],
                    -expert,
                ),
            )
            group = [seed]
            remaining.remove(seed)
            while remaining and len(group) < cluster_size:
                candidate = max(
                    remaining,
                    key=lambda expert: (
                        sum(
                            pair_lift.get(
                                (member, expert)
                                if member < expert
                                else (expert, member),
                                0.0,
                            )
                            for member in group
                        ),
                        sum(
                            pair_counts[
                                (member, expert)
                                if member < expert
                                else (expert, member)
                            ]
                            for member in group
                        ),
                        frequency[expert],
                        lift_degree[expert],
                        -expert,
                    ),
                )
                group.append(candidate)
                remaining.remove(candidate)
            cluster = tuple((layer, expert) for expert in group)
            for key in cluster:
                result[key] = cluster
    expected = len(layers) * expert_count
    if len(result) != expected:
        raise AssertionError("cluster map does not cover every expert record")
    return result


def _coactivation_cluster_training_stats(
    layers: Sequence[int],
    training: dict[int, list[tuple[int, ...]]],
    cluster_for_key: dict[ExpertKey, tuple[ExpertKey, ...]],
) -> dict[str, float | int | None]:
    """Aggregate held-out-safe within-cluster lift over independence."""

    observed = expected = 0.0
    pair_count = 0
    for layer in layers:
        rows = training[layer]
        frequency = Counter(expert for row in rows for expert in row)
        cooccurrence: Counter[tuple[int, int]] = Counter()
        for row in rows:
            for left_index, left in enumerate(row):
                for right in row[left_index + 1 :]:
                    pair = (left, right) if left < right else (right, left)
                    cooccurrence[pair] += 1
        clusters = {cluster_for_key[(layer, expert)] for expert in frequency}
        for cluster in clusters:
            experts = [expert for _cluster_layer, expert in cluster]
            for left_index, left in enumerate(experts):
                for right in experts[left_index + 1 :]:
                    pair = (left, right) if left < right else (right, left)
                    observed += cooccurrence[pair]
                    expected += (
                        frequency[left] * frequency[right] / len(rows) if rows else 0.0
                    )
                    pair_count += 1
    return {
        "training_within_cluster_pairs": pair_count,
        "training_within_cluster_cooccurrences": observed,
        "training_within_cluster_expected_independent": expected,
        "training_aggregate_lift_over_independence": (
            observed / expected if expected else None
        ),
    }


def _run_cluster_lru_sequence(
    sequence: Sequence[ExpertKey],
    initial_lru_order: Sequence[ExpertKey],
    capacity: int,
    cluster_for_key: dict[ExpertKey, tuple[ExpertKey, ...]],
) -> dict[str, object]:
    """Replay fixed-source-cluster reads with record-granularity global LRU.

    One demanded miss reads its complete source cluster.  The triggering record
    is demanded; every other physical record in that read is speculative.  A
    speculative record becomes useful on its first demand before eviction.
    Already-resident cluster mates are still charged as redundant physical
    bytes because a fixed contiguous source read cannot skip their offsets.
    """

    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    cache: dict[ExpertKey, None] = {}
    for key in initial_lru_order:
        if key in cache:
            cache.pop(key)
        cache[key] = None
        if len(cache) > capacity:
            cache.pop(next(iter(cache)))
    speculative_resident: set[ExpertKey] = set()
    hits = misses = 0
    cluster_reads = physical_records = 0
    demanded_records = speculative_records = redundant_records = 0
    speculative_admitted = speculative_used = speculative_unused_evicted = 0

    def insert(key: ExpertKey, *, speculative: bool) -> None:
        nonlocal speculative_admitted, speculative_unused_evicted
        if key in cache or not capacity:
            return
        if len(cache) >= capacity:
            victim = next(iter(cache))
            cache.pop(victim)
            if victim in speculative_resident:
                speculative_resident.remove(victim)
                speculative_unused_evicted += 1
        cache[key] = None
        if speculative:
            speculative_resident.add(key)
            speculative_admitted += 1

    for key in sequence:
        if key in cache:
            hits += 1
            if key in speculative_resident:
                speculative_resident.remove(key)
                speculative_used += 1
            cache.pop(key)
            cache[key] = None
            continue
        misses += 1
        cluster = cluster_for_key[key]
        cluster_reads += 1
        physical_records += len(cluster)
        demanded_records += 1
        speculative_records += len(cluster) - 1
        redundant_records += sum(member in cache for member in cluster)

        # Admit speculative neighbors first and the demanded record last so a
        # small test cache never evicts the record required by this access.
        for member in cluster:
            if member != key:
                insert(member, speculative=True)
        insert(key, speculative=False)
        if key in cache:
            cache.pop(key)
            cache[key] = None

    return {
        **_metric(hits, misses),
        "cluster_reads": cluster_reads,
        "physical_records_read": physical_records,
        "demanded_records_read": demanded_records,
        "speculative_records_read": speculative_records,
        "redundant_records_read": redundant_records,
        "speculative_records_admitted": speculative_admitted,
        "speculative_records_used_before_eviction": speculative_used,
        "speculative_records_unused_at_eviction": speculative_unused_evicted,
        "speculative_records_still_resident": len(speculative_resident),
        "useful_prefetch_ratio": (
            speculative_used / speculative_records if speculative_records else None
        ),
        "admitted_useful_prefetch_ratio": (
            speculative_used / speculative_admitted if speculative_admitted else None
        ),
        "read_amplification_vs_demanded": (
            physical_records / demanded_records if demanded_records else 0.0
        ),
        "final_lru_order": list(cache),
    }


def _conditional_prefetch(
    layers: list[int],
    prompt: dict[int, list[tuple[int, ...]]],
    decode: dict[int, list[tuple[int, ...]]],
    expert_count: int,
    budget: int,
) -> dict[str, float | int]:
    hits = requests = 0
    for left, right in zip(layers, layers[1:], strict=False):
        scores = [[0] * expert_count for _ in range(expert_count)]
        for current, following in zip(prompt[left], prompt[right], strict=True):
            for source in current:
                row = scores[source]
                for target in following:
                    row[target] += 1
        for current, following in zip(decode[left], decode[right], strict=True):
            aggregate = [0] * expert_count
            for source in current:
                source_scores = scores[source]
                for target, value in enumerate(source_scores):
                    aggregate[target] += value
            predicted = set(
                sorted(
                    range(expert_count),
                    key=lambda expert: (aggregate[expert], -expert),
                    reverse=True,
                )[:budget]
            )
            hits += sum(expert in predicted for expert in following)
            requests += len(following)
    return {
        "hits": hits,
        "requests": requests,
        "recall": hits / requests if requests else 0.0,
        "prefetch_experts_per_layer": budget,
        "byte_amplification_vs_top_k": budget / len(decode[layers[0]][0]),
    }


def _temporal_prefetch(
    layers: list[int],
    decode: dict[int, list[tuple[int, ...]]],
) -> dict[str, float | int]:
    hits = requests = 0
    for layer in layers:
        rows = decode[layer]
        for previous, current in zip(rows, rows[1:], strict=False):
            predicted = set(previous)
            hits += sum(expert in predicted for expert in current)
            requests += len(current)
    return {
        "hits": hits,
        "requests": requests,
        "recall": hits / requests if requests else 0.0,
    }


def _unit_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be strictly between 0 and 1")
    return parsed


def _cluster_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(sorted(set(int(part) for part in value.split(","))))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cluster sizes must be comma-separated integers"
        ) from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("cluster sizes must be positive")
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--capacity-per-layer", type=int, default=102)
    parser.add_argument("--expert-count", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--record-bytes", type=int, default=10_616_832)
    parser.add_argument(
        "--measured-ssd-gb-per-second",
        type=float,
        default=3.74,
        help="Decimal GB/s used for the issue #9 throughput projection.",
    )
    parser.add_argument("--shallow-layers", type=int, default=4)
    parser.add_argument(
        "--train-fraction",
        type=_unit_fraction,
        default=0.5,
        help="Chronological decode prefix used to train held-out policies.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        help="Exact decode-prefix length; overrides --train-fraction.",
    )
    parser.add_argument(
        "--quota-hysteresis-hits",
        type=int,
        default=1,
        help="Minimum strict train-hit gain required for each one-slot transfer.",
    )
    parser.add_argument(
        "--cluster-sizes",
        type=_cluster_sizes,
        default=(1, 2, 4),
        help="Comma-separated train-only coactivation cluster sizes.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.capacity_per_layer < 0:
        raise ValueError("capacity-per-layer must be non-negative")
    if args.expert_count <= 0:
        raise ValueError("expert-count must be positive")
    if args.capacity_per_layer > args.expert_count:
        raise ValueError("capacity-per-layer cannot exceed expert-count")
    if args.top_k <= 0 or args.top_k > args.expert_count:
        raise ValueError("top-k must be in [1, expert-count]")
    if args.record_bytes <= 0:
        raise ValueError("record-bytes must be positive")
    if args.measured_ssd_gb_per_second <= 0:
        raise ValueError("measured-ssd-gb-per-second must be positive")
    if args.quota_hysteresis_hits < 0:
        raise ValueError("quota-hysteresis-hits must be non-negative")

    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    entries = payload["entries"]
    by_phase: dict[str, dict[int, list[list[int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in entries:
        by_phase[entry["phase"]][int(entry["layer"])].append(
            [int(value) for value in entry["expert_ids"]]
        )
    layers = sorted(by_phase["decode"])
    if not layers:
        raise ValueError("trace contains no decode routes")
    missing_prefill = [layer for layer in layers if not by_phase["prefill"][layer]]
    if missing_prefill:
        raise ValueError(f"trace has no prefill routes for layers {missing_prefill}")
    prompt = {
        layer: [
            row
            for values in by_phase["prefill"][layer]
            for row in _sets(values, args.top_k)
        ]
        for layer in layers
    }
    decode = {
        layer: [_sets(values, args.top_k)[0] for values in by_phase["decode"][layer]]
        for layer in layers
    }
    decode_steps = min(len(decode[layer]) for layer in layers)
    if decode_steps < 2:
        raise ValueError("trace needs at least two complete decode steps")
    decode = {layer: rows[:decode_steps] for layer, rows in decode.items()}
    train_steps = (
        args.train_steps
        if args.train_steps is not None
        else max(1, min(decode_steps - 1, int(decode_steps * args.train_fraction)))
    )
    if not 0 < train_steps < decode_steps:
        raise ValueError("train-steps must leave at least one train and held-out step")
    evaluation_steps = decode_steps - train_steps
    train = {layer: rows[:train_steps] for layer, rows in decode.items()}
    evaluation = {layer: rows[train_steps:] for layer, rows in decode.items()}
    ranks = {layer: _prompt_rank(prompt[layer]) for layer in layers}
    uniform = {layer: args.capacity_per_layer for layer in layers}
    total_slots = args.capacity_per_layer * len(layers)
    global_rank = _global_prompt_rank(layers, prompt, args.expert_count)
    global_initial = _initial_lru_order(global_rank, total_slots)
    full_global_sequence = _global_sequence(layers, decode)

    shallow = {}
    shallow_set = set(layers[: args.shallow_layers])
    remaining_layers = [layer for layer in layers if layer not in shallow_set]
    remaining_slots = total_slots - len(shallow_set) * args.expert_count
    base, extra = divmod(max(remaining_slots, 0), max(len(remaining_layers), 1))
    for layer in layers:
        if layer in shallow_set:
            shallow[layer] = args.expert_count
        else:
            shallow[layer] = min(
                args.expert_count,
                base + (1 if remaining_layers.index(layer) < extra else 0),
            )

    oracle_capacities = _greedy_oracle_static_capacities(
        layers,
        decode,
        total_slots,
        args.expert_count,
    )
    policies = {
        "prompt_static_uniform": _sum_policy(
            layers, decode, ranks, uniform, _simulate_static
        ),
        "lru_uniform": _sum_policy(layers, decode, ranks, uniform, _simulate_lru),
        "belady_uniform": _sum_policy(layers, decode, ranks, uniform, _simulate_belady),
        "global_pool_lru": _metric(
            *_run_lru_sequence(
                full_global_sequence,
                global_initial,
                total_slots,
            )[:2]
        ),
        "global_pool_belady": _metric(
            *_run_belady_sequence(
                full_global_sequence,
                global_initial,
                total_slots,
            )
        ),
        "prompt_static_shallow_pinned": _sum_policy(
            layers, decode, ranks, shallow, _simulate_static
        ),
        "oracle_decode_frequency_allocation": _sum_policy(
            layers,
            decode,
            {
                layer: [
                    expert
                    for expert, _count in Counter(
                        expert for row in decode[layer] for expert in row
                    ).most_common()
                ]
                for layer in layers
            },
            oracle_capacities,
            _simulate_static,
        ),
    }
    for metric in policies.values():
        _decorate_io_metric(
            metric,
            decode_steps,
            args.record_bytes,
            measured_ssd_bytes_per_second=args.measured_ssd_gb_per_second * 1e9,
        )

    # Held-out uniform partitions start from the exact same causal LRU warm
    # state.  Belady sees only evaluation-future positions, never train routes.
    uniform_warm = _warm_partitioned_lru(layers, train, ranks, uniform)
    held_out_policies: dict[str, dict[str, float | int | None]] = {
        "uniform_per_layer_lru": _evaluate_partitioned_lru(
            layers, evaluation, uniform_warm, uniform
        ),
        "uniform_per_layer_belady": _evaluate_partitioned_belady(
            layers, evaluation, uniform_warm, uniform
        ),
    }

    train_global_sequence = _global_sequence(layers, train)
    evaluation_global_sequence = _global_sequence(layers, evaluation)
    _train_hits, _train_misses, global_warm = _run_lru_sequence(
        train_global_sequence,
        global_initial,
        total_slots,
    )
    global_lru_hits, global_lru_misses, _global_final = _run_lru_sequence(
        evaluation_global_sequence,
        global_warm,
        total_slots,
    )
    held_out_policies["global_pool_lru"] = _metric(global_lru_hits, global_lru_misses)
    global_belady_hits, global_belady_misses = _run_belady_sequence(
        evaluation_global_sequence,
        global_warm,
        total_slots,
    )
    held_out_policies["global_pool_belady"] = _metric(
        global_belady_hits, global_belady_misses
    )

    hit_curves = _lru_training_hit_curves(layers, train, ranks, args.expert_count)
    trained_capacities, quota_training = _rebalance_trained_quotas(
        hit_curves,
        uniform,
        args.quota_hysteresis_hits,
    )
    trained_warm = _warm_partitioned_lru(layers, train, ranks, trained_capacities)
    held_out_policies["trained_dynamic_quota_lru"] = _evaluate_partitioned_lru(
        layers,
        evaluation,
        trained_warm,
        trained_capacities,
    )
    for metric in held_out_policies.values():
        _decorate_io_metric(
            metric,
            evaluation_steps,
            args.record_bytes,
            measured_ssd_bytes_per_second=args.measured_ssd_gb_per_second * 1e9,
        )

    # Coactivation metadata may use prompt routes and the decode train prefix;
    # no held-out decode route contributes to cluster membership.
    cluster_training = {layer: prompt[layer] + train[layer] for layer in layers}
    cluster_results: dict[str, dict[str, object]] = {}
    for cluster_size in args.cluster_sizes:
        cluster_map = _build_coactivation_clusters(
            layers,
            cluster_training,
            args.expert_count,
            cluster_size,
        )
        warm_cluster = _run_cluster_lru_sequence(
            train_global_sequence,
            global_initial,
            total_slots,
            cluster_map,
        )
        evaluation_cluster = _run_cluster_lru_sequence(
            evaluation_global_sequence,
            warm_cluster["final_lru_order"],
            total_slots,
            cluster_map,
        )
        evaluation_cluster.pop("final_lru_order")
        evaluation_cluster["cluster_size"] = cluster_size
        evaluation_cluster.update(
            _coactivation_cluster_training_stats(
                layers,
                cluster_training,
                cluster_map,
            )
        )
        _decorate_io_metric(
            evaluation_cluster,
            evaluation_steps,
            args.record_bytes,
            physical_records=int(evaluation_cluster["physical_records_read"]),
            measured_ssd_bytes_per_second=args.measured_ssd_gb_per_second * 1e9,
        )
        evaluation_cluster["demanded_bytes_read"] = (
            int(evaluation_cluster["demanded_records_read"]) * args.record_bytes
        )
        evaluation_cluster["speculative_bytes_read"] = (
            int(evaluation_cluster["speculative_records_read"]) * args.record_bytes
        )
        evaluation_cluster["redundant_bytes_read"] = (
            int(evaluation_cluster["redundant_records_read"]) * args.record_bytes
        )
        evaluation_cluster["demanded_bytes_per_token"] = (
            int(evaluation_cluster["demanded_bytes_read"]) / evaluation_steps
        )
        evaluation_cluster["speculative_bytes_per_token"] = (
            int(evaluation_cluster["speculative_bytes_read"]) / evaluation_steps
        )
        cluster_results[str(cluster_size)] = evaluation_cluster

    cluster_one = cluster_results.get("1")
    if cluster_one is not None:
        if (
            cluster_one["hits"] != held_out_policies["global_pool_lru"]["hits"]
            or cluster_one["misses"] != held_out_policies["global_pool_lru"]["misses"]
        ):
            raise AssertionError(
                "cluster size 1 must equal record-granularity global LRU"
            )
        baseline_records = int(cluster_one["physical_records_read"])
        baseline_bytes = int(cluster_one["bytes_read"])
        for cluster_metric in cluster_results.values():
            records = int(cluster_metric["physical_records_read"])
            bytes_read = int(cluster_metric["bytes_read"])
            cluster_metric["physical_records_delta_vs_cluster1"] = (
                records - baseline_records
            )
            cluster_metric["bytes_delta_vs_cluster1"] = bytes_read - baseline_bytes
            cluster_metric["physical_byte_ratio_vs_cluster1"] = (
                bytes_read / baseline_bytes if baseline_bytes else None
            )

    result = {
        "schema": "mtplx-expert-route-analysis-v2",
        "source_trace": str(args.trace.resolve()),
        "layers": layers,
        "decode_steps": decode_steps,
        "top_k": args.top_k,
        "expert_count": args.expert_count,
        "record_bytes": args.record_bytes,
        "measured_ssd_gb_per_second": args.measured_ssd_gb_per_second,
        "total_persistent_slots": total_slots,
        "policies": policies,
        "capacities": {
            "uniform": uniform,
            "shallow_pinned": shallow,
            "oracle_decode_frequency": oracle_capacities,
            "trained_dynamic_quota": trained_capacities,
        },
        "held_out": {
            "split": {
                "strategy": "chronological_decode_prefix",
                "train_steps": train_steps,
                "evaluation_steps": evaluation_steps,
                "train_fraction": train_steps / decode_steps,
                "cluster_training_source": "prefill_plus_decode_train_prefix",
                "limitation": (
                    "A temporal split of one trace is not cross-prompt or "
                    "cross-domain go/no-go evidence."
                ),
            },
            "global_pool_key": "(layer_id, expert_id)",
            "policies": held_out_policies,
            "dynamic_quota_training": quota_training,
            "clusters": cluster_results,
        },
        "prefetch": {
            "temporal_previous_token": _temporal_prefetch(layers, decode),
            "prompt_trained_cross_layer_top8": _conditional_prefetch(
                layers, prompt, decode, args.expert_count, args.top_k
            ),
            "prompt_trained_cross_layer_top16": _conditional_prefetch(
                layers, prompt, decode, args.expert_count, args.top_k * 2
            ),
            "prompt_trained_cross_layer_top32": _conditional_prefetch(
                layers, prompt, decode, args.expert_count, args.top_k * 4
            ),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
