#!/usr/bin/env python3
"""Analyze Hy3 route traces for cache headroom and exact prefetch hints."""

from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--capacity-per-layer", type=int, default=102)
    parser.add_argument("--expert-count", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--record-bytes", type=int, default=10_616_832)
    parser.add_argument("--shallow-layers", type=int, default=4)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

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
    prompt = {
        layer: _sets(by_phase["prefill"][layer][0], args.top_k)
        for layer in layers
    }
    decode = {
        layer: [
            _sets(values, args.top_k)[0]
            for values in by_phase["decode"][layer]
        ]
        for layer in layers
    }
    decode_steps = min(len(decode[layer]) for layer in layers)
    decode = {layer: rows[:decode_steps] for layer, rows in decode.items()}
    ranks = {layer: _prompt_rank(prompt[layer]) for layer in layers}
    uniform = {layer: args.capacity_per_layer for layer in layers}
    total_slots = args.capacity_per_layer * len(layers)

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
        "belady_uniform": _sum_policy(
            layers, decode, ranks, uniform, _simulate_belady
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
        misses_per_token = metric["misses"] / decode_steps
        bytes_per_token = misses_per_token * args.record_bytes
        metric["misses_per_token"] = misses_per_token
        metric["bytes_per_token"] = bytes_per_token
        metric["ssd_only_ceiling_tok_s_3p4GBs"] = (
            3.4e9 / bytes_per_token if bytes_per_token else None
        )
        metric["ssd_only_ceiling_tok_s_5p1GBs"] = (
            5.1e9 / bytes_per_token if bytes_per_token else None
        )
        metric["ssd_only_ceiling_tok_s_6p1GiBs"] = (
            6.1 * 2**30 / bytes_per_token if bytes_per_token else None
        )

    result = {
        "schema": "mtplx-expert-route-analysis-v1",
        "source_trace": str(args.trace.resolve()),
        "layers": layers,
        "decode_steps": decode_steps,
        "top_k": args.top_k,
        "expert_count": args.expert_count,
        "record_bytes": args.record_bytes,
        "total_persistent_slots": total_slots,
        "policies": policies,
        "capacities": {
            "uniform": uniform,
            "shallow_pinned": shallow,
            "oracle_decode_frequency": oracle_capacities,
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
