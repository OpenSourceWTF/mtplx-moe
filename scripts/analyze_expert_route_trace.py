#!/usr/bin/env python3
"""Analyze Hy3 route traces for cache, global-pool, and cluster headroom.

The legacy whole-decode metrics remain in ``policies``.  The issue #9 offline
gate lives in ``held_out`` and uses a per-reset-epoch chronological
train/evaluation split so deployable quota and cluster metadata never sees
evaluation routes and every simulated cache starts cold after a runtime reset.
"""

from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Hashable, Iterable, Sequence, TypeVar

from mtplx.expert_runtime import partition_route_waves
from mtplx.expert_streaming import (
    CacheCounters,
    GlobalExpertSlotBank,
    LayerExpertSlotBank,
)


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


def _simulate_atomic_layer_lru(
    prompt_rows: Sequence[tuple[int, ...]],
    decode_steps: Sequence[Sequence[tuple[int, ...]]],
    capacity: int,
    expert_count: int,
    *,
    evaluate_from: int = 0,
) -> tuple[int, int]:
    """Replay each B*top-k selection through the production policy atomically."""
    metric = _simulate_atomic_layer_lru_metric(
        prompt_rows,
        decode_steps,
        capacity,
        expert_count,
        evaluate_from=evaluate_from,
    )
    return int(metric["hits"]), int(metric["misses"])


def _simulate_atomic_layer_lru_metric(
    prompt_rows: Sequence[tuple[int, ...]],
    decode_steps: Sequence[Sequence[tuple[int, ...]]],
    capacity: int,
    expert_count: int,
    *,
    evaluate_from: int = 0,
    transient_slots: int | None = None,
) -> dict[str, float | int]:
    """Return assignment and unique-load evidence from production-policy replay."""

    atomic_routes = [
        tuple(expert for row in step for expert in row) for step in decode_steps
    ]
    prompt_experts = tuple(expert for row in prompt_rows for expert in row)
    maximum_unique = transient_slots or max(
        [len(set(route)) for route in atomic_routes if route]
        + ([len(set(prompt_experts))] if prompt_experts else [1])
    )
    bank = LayerExpertSlotBank(
        expert_count=expert_count,
        persistent_slots=capacity,
        transient_slots=maximum_unique,
        cache_policy="lru",
    )
    if prompt_experts:
        bank.prepare_prefill_seed(prompt_experts)
        for wave in partition_route_waves(
            prompt_experts, max_unique_experts=maximum_unique
        ):
            bank.plan(wave.experts, phase="prefill")
    counters = CacheCounters()
    for index, atomic_route in enumerate(atomic_routes):
        for wave in partition_route_waves(
            atomic_route, max_unique_experts=maximum_unique
        ):
            plan = bank.plan(wave.experts, phase="decode")
            if index >= evaluate_from:
                counters.observe(plan, expert_record_bytes=0)
    metric = _metric(counters.expert_hits, counters.expert_misses)
    metric["physical_records_read"] = (
        counters.persistent_loads + counters.transient_loads
    )
    metric["final_resident_experts"] = list(bank.resident_experts)
    return metric


def _simulate_atomic_global_lru_metric(
    layers: Sequence[int],
    prompt: dict[int, Sequence[tuple[int, ...]]],
    decode_steps: dict[int, Sequence[Sequence[tuple[int, ...]]]],
    capacity: int,
    expert_count: int,
    transient_slots: int,
    *,
    prefill_slots_per_layer: int,
    evaluate_from: int = 0,
) -> dict[str, object]:
    """Replay runtime-ordered layer routes through the production global bank."""

    bank = GlobalExpertSlotBank(
        layer_indices=layers,
        expert_count=expert_count,
        persistent_slots=capacity,
        transient_slots=transient_slots,
        prefill_slots_per_layer=prefill_slots_per_layer,
        cache_policy="lru",
    )
    for layer in layers:
        prompt_experts = tuple(expert for row in prompt[layer] for expert in row)
        if not prompt_experts:
            continue
        bank.prepare_prefill_seed(layer, prompt_experts)
        for wave in partition_route_waves(
            prompt_experts, max_unique_experts=transient_slots
        ):
            plan = bank.plan(layer, wave.experts, phase="prefill")
            bank.publish_ready(layer, plan)
    counters = CacheCounters()
    step_count = len(decode_steps[layers[0]])
    for step in range(step_count):
        for layer in layers:
            atomic_route = tuple(
                expert for row in decode_steps[layer][step] for expert in row
            )
            for wave in partition_route_waves(
                atomic_route, max_unique_experts=transient_slots
            ):
                plan = bank.plan(layer, wave.experts, phase="decode")
                bank.publish_ready(layer, plan)
                if step >= evaluate_from:
                    counters.observe(plan, expert_record_bytes=0)
    metric: dict[str, object] = _metric(counters.expert_hits, counters.expert_misses)
    metric["physical_records_read"] = (
        counters.persistent_loads + counters.transient_loads
    )
    metric["final_resident_experts_by_layer"] = {
        str(layer): list(experts)
        for layer, experts in bank.resident_experts_by_layer.items()
    }
    return metric


def _atomic_lru_training_hit_curves_epochs(
    epochs: Sequence[int],
    layers: Sequence[int],
    prompt: dict[int, dict[int, list[tuple[int, ...]]]],
    train_steps: dict[int, dict[int, list[list[tuple[int, ...]]]]],
    expert_count: int,
    transient_slots: int,
) -> dict[int, list[int]]:
    curves = {layer: [0] * (expert_count + 1) for layer in layers}
    for epoch in epochs:
        for layer in layers:
            for capacity in range(expert_count + 1):
                metric = _simulate_atomic_layer_lru_metric(
                    prompt[epoch][layer],
                    train_steps[epoch][layer],
                    capacity,
                    expert_count,
                    transient_slots=transient_slots,
                )
                curves[layer][capacity] += int(metric["hits"])
    return curves


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


def _global_batched_sequence(
    layers: Sequence[int],
    routes: dict[int, list[list[tuple[int, ...]]]],
) -> list[ExpertKey]:
    """Flatten route steps in runtime order: step, layer, row, router rank."""

    if not layers:
        return []
    steps = len(routes[layers[0]])
    if any(len(routes[layer]) != steps for layer in layers):
        raise ValueError("all layers must have the same route-step count")
    return [
        (layer, expert)
        for step in range(steps)
        for layer in layers
        for row in routes[layer][step]
        for expert in row
    ]


def _batch_union_summary(
    layers: Sequence[int],
    routes: dict[int, list[list[tuple[int, ...]]]],
) -> dict[str, object]:
    """Summarize assignment reuse within each atomic layer/step route."""

    union_size_by_layer: dict[str, list[int]] = {}
    assignment_requests = 0
    unique_records_demanded = 0
    for layer in layers:
        layer_sizes = []
        for step in routes[layer]:
            assignments = [expert for row in step for expert in row]
            union_size = len(set(assignments))
            assignment_requests += len(assignments)
            unique_records_demanded += union_size
            layer_sizes.append(union_size)
        union_size_by_layer[str(layer)] = layer_sizes
    sizes = [
        size for layer_sizes in union_size_by_layer.values() for size in layer_sizes
    ]
    shared_assignments = assignment_requests - unique_records_demanded
    return {
        "assignment_requests": assignment_requests,
        "unique_records_demanded": unique_records_demanded,
        "shared_expert_assignments": shared_assignments,
        "assignment_reuse_ratio": (
            assignment_requests / unique_records_demanded
            if unique_records_demanded
            else 0.0
        ),
        "union_size": {
            "min": min(sizes, default=0),
            "mean": sum(sizes) / len(sizes) if sizes else 0.0,
            "max": max(sizes, default=0),
            "samples": len(sizes),
        },
        "union_size_by_layer": union_size_by_layer,
    }


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


def _aggregate_metrics(
    metrics: Iterable[dict[str, float | int]],
) -> dict[str, float | int]:
    metrics = list(metrics)
    hits = sum(int(metric["hits"]) for metric in metrics)
    misses = sum(int(metric["misses"]) for metric in metrics)
    aggregate = _metric(hits, misses)
    if any("physical_records_read" in metric for metric in metrics):
        aggregate["physical_records_read"] = sum(
            int(metric.get("physical_records_read", metric["misses"]))
            for metric in metrics
        )
    if any("final_resident_experts" in metric for metric in metrics):
        aggregate["final_residency_by_run"] = [
            metric.get("final_resident_experts") for metric in metrics
        ]
    if any("final_resident_experts_by_layer" in metric for metric in metrics):
        aggregate["final_residency_by_epoch"] = [
            metric.get("final_resident_experts_by_layer") for metric in metrics
        ]
    return aggregate


def _sum_policy_epochs(
    epochs: Sequence[int],
    layers: Sequence[int],
    decode: dict[int, dict[int, list[tuple[int, ...]]]],
    ranks: dict[int, dict[int, list[int]]],
    capacities: dict[int, int],
    simulator,
) -> dict[str, float | int]:
    return _aggregate_metrics(
        _sum_policy(layers, decode[epoch], ranks[epoch], capacities, simulator)
        for epoch in epochs
    )


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
    records = (
        int(metric.get("physical_records_read", metric["misses"]))
        if physical_records is None
        else physical_records
    )
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


def _lru_training_hit_curves_epochs(
    epochs: Sequence[int],
    layers: Sequence[int],
    train: dict[int, dict[int, list[tuple[int, ...]]]],
    ranks: dict[int, dict[int, list[int]]],
    expert_count: int,
) -> dict[int, list[int]]:
    curves = {layer: [0] * (expert_count + 1) for layer in layers}
    for epoch in epochs:
        epoch_curves = _lru_training_hit_curves(
            layers, train[epoch], ranks[epoch], expert_count
        )
        for layer in layers:
            for capacity, hits in enumerate(epoch_curves[layer]):
                curves[layer][capacity] += hits
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


def _recommended_capacity_summary(
    layers: Sequence[int],
    trained_capacities: dict[int, int],
    quota_training: dict[str, int],
    held_out_policies: dict[str, dict[str, float | int | None]],
    *,
    record_bytes: int,
) -> dict[str, object]:
    """Export one train-derived allocation and its held-out evaluation.

    ``trained_capacities`` and ``quota_training`` are computed exclusively from
    the chronological training prefix.  Held-out rows contribute only to the
    reported delta against the uniform LRU baseline; they never change quotas.
    """

    ordered_layers = sorted(layers)
    if set(ordered_layers) != set(trained_capacities):
        raise ValueError("trained capacities do not match analyzed layers")
    total_slots = sum(trained_capacities[layer] for layer in ordered_layers)
    baseline = held_out_policies["uniform_per_layer_lru"]
    candidate = held_out_policies["trained_dynamic_quota_lru"]
    baseline_hits = int(baseline["hits"])
    baseline_misses = int(baseline["misses"])
    candidate_hits = int(candidate["hits"])
    candidate_misses = int(candidate["misses"])
    baseline_requests = baseline_hits + baseline_misses
    candidate_requests = candidate_hits + candidate_misses
    if baseline_requests != candidate_requests:
        raise ValueError("held-out policies evaluated different request counts")
    hit_delta = candidate_hits - baseline_hits

    return {
        "per_layer_quotas": [
            {"layer": layer, "slots": trained_capacities[layer]}
            for layer in ordered_layers
        ],
        "total_slots": total_slots,
        "record_bytes_per_slot": record_bytes,
        "total_bytes": total_slots * record_bytes,
        "training_gain": {
            "source": "prefill_rank_plus_chronological_decode_train_prefix",
            "reallocation_moves": quota_training["reallocation_moves"],
            "hits_before": quota_training["training_hits_before"],
            "hits_after": quota_training["training_hits_after"],
            "hit_delta": quota_training["training_hit_gain"],
            "hysteresis_hits": quota_training["hysteresis_hits"],
        },
        "held_out_delta": {
            "source": "chronological_decode_held_out_suffix_after_train_warmup",
            "baseline": "uniform_per_layer_lru",
            "candidate": "trained_dynamic_quota_lru",
            "direction": "candidate_minus_baseline",
            "hit_delta": hit_delta,
            "miss_delta": candidate_misses - baseline_misses,
            "hit_rate_delta": (
                hit_delta / baseline_requests if baseline_requests else 0.0
            ),
        },
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


def _aggregate_cluster_runs(metrics: Sequence[dict[str, object]]) -> dict[str, object]:
    count_keys = (
        "hits",
        "misses",
        "cluster_reads",
        "physical_records_read",
        "demanded_records_read",
        "speculative_records_read",
        "redundant_records_read",
        "speculative_records_admitted",
        "speculative_records_used_before_eviction",
        "speculative_records_unused_at_eviction",
        "speculative_records_still_resident",
    )
    result = {key: sum(int(metric[key]) for metric in metrics) for key in count_keys}
    result.update(_metric(int(result["hits"]), int(result["misses"])))
    speculative = int(result["speculative_records_read"])
    admitted = int(result["speculative_records_admitted"])
    used = int(result["speculative_records_used_before_eviction"])
    demanded = int(result["demanded_records_read"])
    result["useful_prefetch_ratio"] = used / speculative if speculative else None
    result["admitted_useful_prefetch_ratio"] = used / admitted if admitted else None
    result["read_amplification_vs_demanded"] = (
        int(result["physical_records_read"]) / demanded if demanded else 0.0
    )
    return result


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
    parser.add_argument(
        "--transient-slots",
        type=int,
        help="Runtime transient route-wave capacity (defaults to trace metadata).",
    )
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
        help=(
            "Exact decode-prefix length within every trace epoch; overrides "
            "--train-fraction."
        ),
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
    if args.record_bytes <= 0 or args.record_bytes > 2**63 - 1:
        raise ValueError("record-bytes must be a positive signed 64-bit integer")
    if args.measured_ssd_gb_per_second <= 0:
        raise ValueError("measured-ssd-gb-per-second must be positive")
    if args.quota_hysteresis_hits < 0:
        raise ValueError("quota-hysteresis-hits must be non-negative")
    if (
        args.output_json is not None
        and args.trace.expanduser().resolve() == args.output_json.expanduser().resolve()
    ):
        raise ValueError("trace input and output-json must be different")

    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    payload_transient_slots = payload.get("transient_slots")
    if (
        args.transient_slots is not None
        and payload_transient_slots is not None
        and args.transient_slots != int(payload_transient_slots)
    ):
        raise ValueError("transient-slots override differs from trace metadata")
    if args.transient_slots is not None:
        transient_slots = args.transient_slots
    elif payload_transient_slots is not None:
        transient_slots = int(payload_transient_slots)
    elif payload.get("schema") == "mtplx-expert-route-trace-v2":
        raise ValueError("v2 trace is missing configured transient_slots")
    else:
        transient_slots = args.top_k
    if transient_slots <= 0:
        raise ValueError("transient-slots must be positive")
    entries = payload["entries"]
    by_phase: dict[str, dict[int, list[list[int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    decode_entries: dict[int, list[dict[str, object]]] = defaultdict(list)
    prefill_entries: dict[int, list[dict[str, object]]] = defaultdict(list)
    reset_entries: list[dict[str, object]] = []
    current_epoch = 0
    epoch_has_routes = False
    decode_started = False
    for entry in entries:
        phase = str(entry["phase"])
        if phase == "reset":
            previous = int(entry.get("previous_trace_epoch", -1))
            current = int(entry.get("trace_epoch", -1))
            if (
                not epoch_has_routes
                or previous != current_epoch
                or current != current_epoch + 1
            ):
                raise ValueError("malformed or out-of-order trace reset marker")
            reset_entries.append(entry)
            current_epoch = current
            epoch_has_routes = False
            decode_started = False
            continue
        entry_epoch = int(entry.get("trace_epoch", 0))
        if entry_epoch != current_epoch:
            raise ValueError("trace route crosses an unmarked epoch boundary")
        if phase == "prefill" and decode_started:
            raise ValueError("prefill route appears after decode in the same epoch")
        if phase == "decode":
            decode_started = True
        epoch_has_routes = True
        layer = int(entry["layer"])
        by_phase[phase][layer].append([int(value) for value in entry["expert_ids"]])
        if phase == "decode":
            decode_entries[layer].append(entry)
        elif phase == "prefill":
            prefill_entries[layer].append(entry)
    if reset_entries and not epoch_has_routes:
        raise ValueError("trace ends with a reset marker and no following epoch routes")
    decode_layers = set(by_phase["decode"])
    if not decode_layers:
        raise ValueError("trace contains no decode routes")
    layers = sorted(decode_layers | set(by_phase["prefill"]))
    decode_counts = {layer: len(by_phase["decode"][layer]) for layer in layers}
    if len(set(decode_counts.values())) != 1:
        rendered_counts = ", ".join(
            f"{layer}: {decode_counts[layer]}" for layer in layers
        )
        raise ValueError(f"decode row counts differ by layer: {rendered_counts}")
    if any(
        entry.get("decode_step") is None or entry.get("trace_epoch") is None
        for layer in layers
        for entry in decode_entries[layer]
    ):
        raise ValueError(
            "trace_epoch and decode_step are required for every decode route; "
            "regenerate the trace"
        )
    route_coordinates = {
        layer: [
            (int(entry["trace_epoch"]), int(entry["decode_step"]))
            for entry in decode_entries[layer]
        ]
        for layer in layers
    }
    reference_coordinates = route_coordinates[layers[0]]
    if any(route_coordinates[layer] != reference_coordinates for layer in layers[1:]):
        raise ValueError(
            "trace_epoch/decode_step sequence alignment differs across layers"
        )
    epochs = sorted({epoch for epoch, _step in reference_coordinates})
    if epochs != list(range(len(epochs))):
        raise ValueError("trace_epoch sequence must start at zero and be contiguous")
    expected_resets = [
        {
            "phase": "reset",
            "previous_trace_epoch": previous,
            "trace_epoch": current,
        }
        for previous, current in zip(epochs, epochs[1:], strict=False)
    ]
    if reset_entries != expected_resets:
        raise ValueError("trace reset boundaries do not match trace_epoch transitions")
    expected_coordinates = [
        (epoch, step)
        for epoch in epochs
        for step in range(
            sum(1 for candidate, _step in reference_coordinates if candidate == epoch)
        )
    ]
    if reference_coordinates != expected_coordinates:
        raise ValueError(
            "trace_epoch/decode_step sequence must be unique, monotonic, and "
            "contiguous within every epoch"
        )
    token_counts = {
        layer: [int(entry.get("token_count", 0)) for entry in decode_entries[layer]]
        for layer in layers
    }
    reference_widths = token_counts[layers[0]]
    if any(width <= 0 for width in reference_widths):
        raise ValueError("decode token_count must be positive for every route")
    if any(token_counts[layer] != reference_widths for layer in layers[1:]):
        raise ValueError("decode token_count sequence differs across layers")
    missing_prefill = [layer for layer in layers if not by_phase["prefill"][layer]]
    if missing_prefill:
        raise ValueError(f"trace has no prefill routes for layers {missing_prefill}")
    prompt_by_epoch = {
        epoch: {
            layer: [
                row
                for entry in prefill_entries[layer]
                if int(entry.get("trace_epoch", 0)) == epoch
                for row in _sets(
                    [int(value) for value in entry["expert_ids"]], args.top_k
                )
            ]
            for layer in layers
        }
        for epoch in epochs
    }
    for epoch in epochs:
        for layer in layers:
            if not prompt_by_epoch[epoch][layer]:
                raise ValueError(
                    f"trace epoch {epoch} has no causal prefill route for layer {layer}"
                )
    prompt = {
        layer: [row for epoch in epochs for row in prompt_by_epoch[epoch][layer]]
        for layer in layers
    }
    decode_steps = decode_counts[layers[0]]
    if decode_steps < 2:
        raise ValueError("trace needs at least two complete decode steps")
    decode_by_step: dict[int, list[list[tuple[int, ...]]]] = {}
    for layer in layers:
        layer_steps = []
        for entry, token_count in zip(
            decode_entries[layer], token_counts[layer], strict=True
        ):
            values = [int(value) for value in entry["expert_ids"]]
            expected_values = token_count * args.top_k
            if len(values) != expected_values:
                raise ValueError(
                    f"decode layer {layer} step {entry['decode_step']} has "
                    f"{len(values)} expert_ids; expected token_count*top_k="
                    f"{expected_values}"
                )
            layer_steps.append(_sets(values, args.top_k))
        decode_by_step[layer] = layer_steps
    epoch_indices = {
        epoch: [
            index
            for index, coordinate in enumerate(reference_coordinates)
            if coordinate[0] == epoch
        ]
        for epoch in epochs
    }
    train_steps_by_epoch = {}
    for epoch in epochs:
        epoch_steps = len(epoch_indices[epoch])
        trained = (
            args.train_steps
            if args.train_steps is not None
            else max(1, min(epoch_steps - 1, int(epoch_steps * args.train_fraction)))
        )
        if not 0 < trained < epoch_steps:
            raise ValueError(
                "train-steps must leave at least one train and held-out step in "
                f"trace epoch {epoch}"
            )
        train_steps_by_epoch[epoch] = trained
    train_steps = sum(train_steps_by_epoch.values())
    evaluation_steps = decode_steps - train_steps
    decode_tokens = sum(reference_widths)
    train_indices = {
        index
        for epoch in epochs
        for index in epoch_indices[epoch][: train_steps_by_epoch[epoch]]
    }
    train_tokens = sum(
        width for index, width in enumerate(reference_widths) if index in train_indices
    )
    evaluation_tokens = decode_tokens - train_tokens
    decode = {
        layer: [row for step in decode_by_step[layer] for row in step]
        for layer in layers
    }
    decode_epochs = {
        epoch: {
            layer: [
                row
                for index in epoch_indices[epoch]
                for row in decode_by_step[layer][index]
            ]
            for layer in layers
        }
        for epoch in epochs
    }
    decode_step_epochs = {
        epoch: {
            layer: [decode_by_step[layer][index] for index in epoch_indices[epoch]]
            for layer in layers
        }
        for epoch in epochs
    }
    train_step_epochs = {
        epoch: {
            layer: [
                decode_by_step[layer][index]
                for index in epoch_indices[epoch][: train_steps_by_epoch[epoch]]
            ]
            for layer in layers
        }
        for epoch in epochs
    }
    evaluation_step_epochs = {
        epoch: {
            layer: [
                decode_by_step[layer][index]
                for index in epoch_indices[epoch][train_steps_by_epoch[epoch] :]
            ]
            for layer in layers
        }
        for epoch in epochs
    }
    train_epochs = {
        epoch: {
            layer: [
                row
                for index in epoch_indices[epoch][: train_steps_by_epoch[epoch]]
                for row in decode_by_step[layer][index]
            ]
            for layer in layers
        }
        for epoch in epochs
    }
    evaluation_epochs = {
        epoch: {
            layer: [
                row
                for index in epoch_indices[epoch][train_steps_by_epoch[epoch] :]
                for row in decode_by_step[layer][index]
            ]
            for layer in layers
        }
        for epoch in epochs
    }
    train = {
        layer: [row for epoch in epochs for row in train_epochs[epoch][layer]]
        for layer in layers
    }
    ranks_epochs = {
        epoch: {layer: _prompt_rank(prompt_by_epoch[epoch][layer]) for layer in layers}
        for epoch in epochs
    }
    uniform = {layer: args.capacity_per_layer for layer in layers}
    total_slots = args.capacity_per_layer * len(layers)
    if total_slots > (2**63 - 1) // args.record_bytes:
        raise ValueError("total cache byte projection exceeds signed 64-bit range")
    global_ranks_epochs = {
        epoch: _global_prompt_rank(layers, prompt_by_epoch[epoch], args.expert_count)
        for epoch in epochs
    }

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
    oracle_ranks = {
        layer: [
            expert
            for expert, _count in Counter(
                expert for row in decode[layer] for expert in row
            ).most_common()
        ]
        for layer in layers
    }
    global_epoch_sequences = {
        epoch: _global_batched_sequence(layers, decode_step_epochs[epoch])
        for epoch in epochs
    }
    policies = {
        "prompt_static_uniform": _sum_policy_epochs(
            epochs, layers, decode_epochs, ranks_epochs, uniform, _simulate_static
        ),
        "lru_uniform": _aggregate_metrics(
            _simulate_atomic_layer_lru_metric(
                prompt_by_epoch[epoch][layer],
                decode_step_epochs[epoch][layer],
                uniform[layer],
                args.expert_count,
                transient_slots=transient_slots,
            )
            for epoch in epochs
            for layer in layers
        ),
        "belady_uniform": _sum_policy_epochs(
            epochs, layers, decode_epochs, ranks_epochs, uniform, _simulate_belady
        ),
        "global_pool_lru": _aggregate_metrics(
            _simulate_atomic_global_lru_metric(
                layers,
                prompt_by_epoch[epoch],
                decode_step_epochs[epoch],
                total_slots,
                args.expert_count,
                transient_slots,
                prefill_slots_per_layer=args.capacity_per_layer,
            )
            for epoch in epochs
        ),
        "global_pool_belady": _aggregate_metrics(
            _metric(
                *_run_belady_sequence(
                    global_epoch_sequences[epoch],
                    _initial_lru_order(global_ranks_epochs[epoch], total_slots),
                    total_slots,
                )
            )
            for epoch in epochs
        ),
        "prompt_static_shallow_pinned": _sum_policy_epochs(
            epochs, layers, decode_epochs, ranks_epochs, shallow, _simulate_static
        ),
        "oracle_decode_frequency_allocation": _sum_policy_epochs(
            epochs,
            layers,
            decode_epochs,
            {epoch: oracle_ranks for epoch in epochs},
            oracle_capacities,
            _simulate_static,
        ),
    }
    for metric in policies.values():
        _decorate_io_metric(
            metric,
            decode_tokens,
            args.record_bytes,
            measured_ssd_bytes_per_second=args.measured_ssd_gb_per_second * 1e9,
        )

    # Held-out uniform partitions start from the exact same causal LRU warm
    # state.  Belady sees only evaluation-future positions, never train routes.
    held_out_policies: dict[str, dict[str, float | int | None]] = {
        "uniform_per_layer_lru": _aggregate_metrics(
            _simulate_atomic_layer_lru_metric(
                prompt_by_epoch[epoch][layer],
                decode_step_epochs[epoch][layer],
                uniform[layer],
                args.expert_count,
                evaluate_from=train_steps_by_epoch[epoch],
                transient_slots=transient_slots,
            )
            for epoch in epochs
            for layer in layers
        ),
        "uniform_per_layer_belady": _aggregate_metrics(
            _evaluate_partitioned_belady(
                layers,
                evaluation_epochs[epoch],
                _warm_partitioned_lru(
                    layers, train_epochs[epoch], ranks_epochs[epoch], uniform
                ),
                uniform,
            )
            for epoch in epochs
        ),
    }

    train_global_sequences = {
        epoch: _global_batched_sequence(layers, train_step_epochs[epoch])
        for epoch in epochs
    }
    evaluation_global_sequences = {
        epoch: _global_batched_sequence(layers, evaluation_step_epochs[epoch])
        for epoch in epochs
    }
    held_out_policies["global_pool_lru"] = _aggregate_metrics(
        _simulate_atomic_global_lru_metric(
            layers,
            prompt_by_epoch[epoch],
            decode_step_epochs[epoch],
            total_slots,
            args.expert_count,
            transient_slots,
            prefill_slots_per_layer=args.capacity_per_layer,
            evaluate_from=train_steps_by_epoch[epoch],
        )
        for epoch in epochs
    )
    # Belady remains an explicitly clairvoyant lower bound. Its initial state
    # is the production global LRU warm residency, not a sequential surrogate.
    held_out_policies["global_pool_belady"] = _aggregate_metrics(
        _metric(
            *_run_belady_sequence(
                evaluation_global_sequences[epoch],
                [
                    (int(layer), int(expert))
                    for layer, experts in _simulate_atomic_global_lru_metric(
                        layers,
                        prompt_by_epoch[epoch],
                        train_step_epochs[epoch],
                        total_slots,
                        args.expert_count,
                        transient_slots,
                        prefill_slots_per_layer=args.capacity_per_layer,
                    )["final_resident_experts_by_layer"].items()
                    for expert in experts
                ],
                total_slots,
            )
        )
        for epoch in epochs
    )

    hit_curves = _atomic_lru_training_hit_curves_epochs(
        epochs,
        layers,
        prompt_by_epoch,
        train_step_epochs,
        args.expert_count,
        transient_slots,
    )
    trained_capacities, quota_training = _rebalance_trained_quotas(
        hit_curves,
        uniform,
        args.quota_hysteresis_hits,
    )
    held_out_policies["trained_dynamic_quota_lru"] = _aggregate_metrics(
        _simulate_atomic_layer_lru_metric(
            prompt_by_epoch[epoch][layer],
            decode_step_epochs[epoch][layer],
            trained_capacities[layer],
            args.expert_count,
            evaluate_from=train_steps_by_epoch[epoch],
            transient_slots=transient_slots,
        )
        for epoch in epochs
        for layer in layers
    )
    for metric in held_out_policies.values():
        _decorate_io_metric(
            metric,
            evaluation_tokens,
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
        epoch_cluster_metrics = []
        for epoch in epochs:
            warm_cluster = _run_cluster_lru_sequence(
                train_global_sequences[epoch],
                _initial_lru_order(global_ranks_epochs[epoch], total_slots),
                total_slots,
                cluster_map,
            )
            epoch_cluster_metrics.append(
                _run_cluster_lru_sequence(
                    evaluation_global_sequences[epoch],
                    warm_cluster["final_lru_order"],
                    total_slots,
                    cluster_map,
                )
            )
        evaluation_cluster = _aggregate_cluster_runs(epoch_cluster_metrics)
        if cluster_size == 1:
            global_lru = held_out_policies["global_pool_lru"]
            records = int(global_lru["physical_records_read"])
            evaluation_cluster.update(
                {
                    **_metric(int(global_lru["hits"]), int(global_lru["misses"])),
                    "cluster_reads": records,
                    "physical_records_read": records,
                    "demanded_records_read": records,
                    "speculative_records_read": 0,
                    "redundant_records_read": 0,
                    "speculative_records_admitted": 0,
                    "speculative_records_used_before_eviction": 0,
                    "speculative_records_unused_at_eviction": 0,
                    "speculative_records_still_resident": 0,
                    "useful_prefetch_ratio": None,
                    "admitted_useful_prefetch_ratio": None,
                    "read_amplification_vs_demanded": 1.0 if records else 0.0,
                }
            )
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
            evaluation_tokens,
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
            int(evaluation_cluster["demanded_bytes_read"]) / evaluation_tokens
        )
        evaluation_cluster["speculative_bytes_per_token"] = (
            int(evaluation_cluster["speculative_bytes_read"]) / evaluation_tokens
        )
        cluster_results[str(cluster_size)] = evaluation_cluster

    cluster_one = cluster_results.get("1")
    if cluster_one is not None:
        if (
            cluster_one["hits"] != held_out_policies["global_pool_lru"]["hits"]
            or cluster_one["misses"] != held_out_policies["global_pool_lru"]["misses"]
        ):
            raise AssertionError(
                "cluster size 1 must equal record-granularity global LRU: "
                f"cluster=({cluster_one['hits']}, {cluster_one['misses']}), "
                "global=("
                f"{held_out_policies['global_pool_lru']['hits']}, "
                f"{held_out_policies['global_pool_lru']['misses']})"
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
        "trace_epochs": epochs,
        "decode_steps": decode_steps,
        "decode_tokens": decode_tokens,
        "top_k": args.top_k,
        "transient_slots": transient_slots,
        "expert_count": args.expert_count,
        "record_bytes": args.record_bytes,
        "measured_ssd_gb_per_second": args.measured_ssd_gb_per_second,
        "total_persistent_slots": total_slots,
        "batch_union": _batch_union_summary(layers, decode_by_step),
        "policies": policies,
        "capacities": {
            "uniform": uniform,
            "shallow_pinned": shallow,
            "oracle_decode_frequency": oracle_capacities,
            "trained_dynamic_quota": trained_capacities,
        },
        "recommended_capacity": _recommended_capacity_summary(
            layers,
            trained_capacities,
            quota_training,
            held_out_policies,
            record_bytes=args.record_bytes,
        ),
        "held_out": {
            "split": {
                "strategy": "chronological_decode_prefix",
                "scope": "per_epoch_chronological_prefix",
                "train_steps": train_steps,
                "evaluation_steps": evaluation_steps,
                "train_steps_by_epoch": {
                    str(epoch): train_steps_by_epoch[epoch] for epoch in epochs
                },
                "train_tokens": train_tokens,
                "evaluation_tokens": evaluation_tokens,
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
            "temporal_previous_token": (
                _temporal_prefetch(layers, decode)
                if max(reference_widths) == 1
                else {
                    "supported": False,
                    "hits": None,
                    "requests": None,
                    "recall": None,
                    "reason": ("batched traces do not carry stable request identities"),
                }
            ),
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
        from scripts.benchmark_streamed_generation import (
            reserve_json_evidence_targets,
        )

        reservation = reserve_json_evidence_targets(args.output_json, None)
        try:
            reservation.commit(args.output_json, rendered + "\n")
        finally:
            reservation.cleanup()
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
