from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from scripts import analyze_expert_route_trace as route
from mtplx.expert_runtime import partition_route_waves
from mtplx.expert_streaming import (
    CacheCounters,
    GlobalExpertSlotBank,
    LayerExpertSlotBank,
)


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_atomic_layer_lru_matches_production_policy_for_batched_routes(
    batch: int,
) -> None:
    rng = random.Random(194 + batch)
    prompt = [(0, 1)]
    steps = [
        [tuple(rng.sample(range(12), 2)) for _request in range(batch)]
        for _step in range(7)
    ]
    flattened = [tuple(expert for row in step for expert in row) for step in steps]
    maximum_unique = max(len(set(route_ids)) for route_ids in flattened + [(0, 1)])
    bank = LayerExpertSlotBank(
        expert_count=12,
        persistent_slots=4,
        transient_slots=maximum_unique,
        cache_policy="lru",
    )
    bank.prepare_prefill_seed((0, 1))
    bank.plan((0, 1), phase="prefill")
    counters = CacheCounters()
    for route_ids in flattened:
        counters.observe(bank.plan(route_ids, phase="decode"), expert_record_bytes=0)

    assert route._simulate_atomic_layer_lru(prompt, steps, 4, 12) == (
        counters.expert_hits,
        counters.expert_misses,
    )


@pytest.mark.parametrize("transient_slots", [1, 2, 8])
def test_atomic_layer_replay_matches_production_route_waves(
    transient_slots: int,
) -> None:
    rng = random.Random(901 + transient_slots)
    prompt = [(0, 1)]
    steps = [
        [tuple(rng.sample(range(12), 2)) for _request in range(4)] for _step in range(6)
    ]
    bank = LayerExpertSlotBank(
        expert_count=12,
        persistent_slots=4,
        transient_slots=transient_slots,
        cache_policy="lru",
    )
    bank.prepare_prefill_seed((0, 1))
    for wave in partition_route_waves((0, 1), max_unique_experts=transient_slots):
        bank.plan(wave.experts, phase="prefill")
    counters = CacheCounters()
    for step in steps:
        route_ids = tuple(expert for row in step for expert in row)
        for wave in partition_route_waves(
            route_ids, max_unique_experts=transient_slots
        ):
            plan = bank.plan(wave.experts, phase="decode")
            counters.observe(plan, expert_record_bytes=0)

    metric = route._simulate_atomic_layer_lru_metric(
        prompt, steps, 4, 12, transient_slots=transient_slots
    )
    assert (metric["hits"], metric["misses"]) == (
        counters.expert_hits,
        counters.expert_misses,
    )
    assert metric["physical_records_read"] == (
        counters.persistent_loads + counters.transient_loads
    )
    assert metric["final_resident_experts"] == list(bank.resident_experts)


def test_atomic_route_counts_duplicate_assignments_but_loads_once() -> None:
    assert route._simulate_atomic_layer_lru([], [[(0,), (0,)]], 1, 2) == (0, 2)
    metric = route._simulate_atomic_layer_lru_metric([], [[(0,), (0,)]], 1, 2)
    assert metric["physical_records_read"] == 1


def test_atomic_training_curve_can_choose_a_different_quota_than_flattened_lru() -> (
    None
):
    epochs = [0]
    layers = [1, 2]
    prompt = {0: {1: [(0, 3)], 2: [(2, 0)]}}
    steps = {
        0: {
            1: [
                [(2, 1), (1, 2)],
                [(0, 2), (1, 3)],
                [(3, 2), (2, 3)],
                [(3, 2), (2, 0)],
                [(0, 1), (3, 1)],
            ],
            2: [
                [(3, 1), (1, 2)],
                [(1, 0), (1, 0)],
                [(1, 3), (1, 0)],
                [(2, 3), (1, 3)],
                [(3, 2), (2, 3)],
            ],
        }
    }
    flattened = {
        0: {
            layer: [row for step in steps[0][layer] for row in step] for layer in layers
        }
    }
    ranks = {0: {layer: route._prompt_rank(prompt[0][layer]) for layer in layers}}
    sequential = route._lru_training_hit_curves_epochs(
        epochs, layers, flattened, ranks, 4
    )
    atomic = route._atomic_lru_training_hit_curves_epochs(
        epochs, layers, prompt, steps, 4, 1
    )

    sequential_quota, _ = route._rebalance_trained_quotas(sequential, {1: 2, 2: 2}, 0)
    atomic_quota, _ = route._rebalance_trained_quotas(atomic, {1: 2, 2: 2}, 0)
    assert sequential_quota == {1: 2, 2: 2}
    assert atomic_quota == {1: 0, 2: 4}


@pytest.mark.parametrize("transient_slots", [1, 2, 8])
def test_atomic_global_replay_matches_global_bank_with_route_waves(
    transient_slots: int,
) -> None:
    layers = [1, 2]
    prompt = {1: [(0, 1)], 2: [(2, 3)]}
    steps = {
        1: [[(0, 0), (1, 2)], [(3, 4), (4, 3)], [(0, 5), (5, 0)]],
        2: [[(2, 2), (3, 1)], [(5, 0), (0, 5)], [(2, 4), (4, 2)]],
    }
    bank = GlobalExpertSlotBank(
        layer_indices=layers,
        expert_count=6,
        persistent_slots=4,
        transient_slots=transient_slots,
        prefill_slots_per_layer=2,
        cache_policy="lru",
    )
    for layer in layers:
        prompt_ids = tuple(expert for row in prompt[layer] for expert in row)
        bank.prepare_prefill_seed(layer, prompt_ids)
        for wave in partition_route_waves(
            prompt_ids, max_unique_experts=transient_slots
        ):
            plan = bank.plan(layer, wave.experts, phase="prefill")
            bank.publish_ready(layer, plan)
    counters = CacheCounters()
    for step in range(3):
        for layer in layers:
            route_ids = tuple(expert for row in steps[layer][step] for expert in row)
            for wave in partition_route_waves(
                route_ids, max_unique_experts=transient_slots
            ):
                plan = bank.plan(layer, wave.experts, phase="decode")
                bank.publish_ready(layer, plan)
                counters.observe(plan, expert_record_bytes=0)

    metric = route._simulate_atomic_global_lru_metric(
        layers,
        prompt,
        steps,
        4,
        6,
        transient_slots,
        prefill_slots_per_layer=2,
    )
    assert (metric["hits"], metric["misses"]) == (
        counters.expert_hits,
        counters.expert_misses,
    )
    assert metric["physical_records_read"] == (
        counters.persistent_loads + counters.transient_loads
    )
    assert metric["final_resident_experts_by_layer"] == {
        str(layer): list(experts)
        for layer, experts in bank.resident_experts_by_layer.items()
    }


def test_atomic_global_replay_ranks_seed_over_the_whole_prompt() -> None:
    metric = route._simulate_atomic_global_lru_metric(
        [1],
        {1: [(0,), (1,), (1,)]},
        {1: []},
        1,
        2,
        1,
        prefill_slots_per_layer=1,
    )

    assert metric["final_resident_experts_by_layer"] == {"1": [1]}


def test_global_sequence_keys_records_by_layer_in_runtime_order() -> None:
    routes = {
        1: [(0, 1), (2, 3)],
        2: [(4, 5), (6, 7)],
    }

    assert route._global_sequence([1, 2], routes) == [
        (1, 0),
        (1, 1),
        (2, 4),
        (2, 5),
        (1, 2),
        (1, 3),
        (2, 6),
        (2, 7),
    ]


def test_global_belady_is_no_worse_than_global_lru() -> None:
    sequence = [
        (1, 0),
        (2, 0),
        (1, 1),
        (1, 0),
        (2, 0),
        (1, 1),
    ]

    lru_hits, lru_misses, _state = route._run_lru_sequence(sequence, [], 2)
    belady_hits, belady_misses = route._run_belady_sequence(sequence, [], 2)

    assert belady_hits >= lru_hits
    assert belady_misses <= lru_misses


def test_global_lru_can_lend_a_cold_layers_slot_to_a_hot_layer() -> None:
    global_sequence = [(2, 0), (1, 0), (1, 1), (1, 0), (1, 1)]

    global_hits, _global_misses, _state = route._run_lru_sequence(
        global_sequence, [], 2
    )
    layer_one_hits, _misses, _state = route._run_lru_sequence([0, 1, 0, 1], [], 1)
    layer_two_hits, _misses, _state = route._run_lru_sequence([0], [], 1)

    assert global_hits == 2
    assert layer_one_hits + layer_two_hits == 0


def test_trained_quota_moves_capacity_without_changing_total_budget() -> None:
    hit_curves = {
        1: [0, 5, 9],
        2: [0, 1, 1],
    }

    capacities, training = route._rebalance_trained_quotas(
        hit_curves,
        {1: 1, 2: 1},
        hysteresis_hits=0,
    )

    assert capacities == {1: 2, 2: 0}
    assert sum(capacities.values()) == 2
    assert training == {
        "reallocation_moves": 1,
        "training_hits_before": 6,
        "training_hits_after": 9,
        "training_hit_gain": 3,
        "hysteresis_hits": 0,
    }


def test_recommended_capacity_schema_is_stable_and_accounts_for_record_bytes() -> None:
    summary = route._recommended_capacity_summary(
        [1, 2],
        {2: 0, 1: 2},
        {
            "reallocation_moves": 1,
            "training_hits_before": 6,
            "training_hits_after": 9,
            "training_hit_gain": 3,
            "hysteresis_hits": 0,
        },
        {
            "uniform_per_layer_lru": {"hits": 7, "misses": 5, "hit_rate": 7 / 12},
            "trained_dynamic_quota_lru": {
                "hits": 9,
                "misses": 3,
                "hit_rate": 0.75,
            },
        },
        record_bytes=100,
    )

    assert summary == {
        "per_layer_quotas": [
            {"layer": 1, "slots": 2},
            {"layer": 2, "slots": 0},
        ],
        "total_slots": 2,
        "record_bytes_per_slot": 100,
        "total_bytes": 200,
        "training_gain": {
            "source": "prefill_rank_plus_chronological_decode_train_prefix",
            "reallocation_moves": 1,
            "hits_before": 6,
            "hits_after": 9,
            "hit_delta": 3,
            "hysteresis_hits": 0,
        },
        "held_out_delta": {
            "source": "chronological_decode_held_out_suffix_after_train_warmup",
            "baseline": "uniform_per_layer_lru",
            "candidate": "trained_dynamic_quota_lru",
            "direction": "candidate_minus_baseline",
            "hit_delta": 2,
            "miss_delta": -2,
            "hit_rate_delta": 1 / 6,
        },
    }
    assert list(summary) == [
        "per_layer_quotas",
        "total_slots",
        "record_bytes_per_slot",
        "total_bytes",
        "training_gain",
        "held_out_delta",
    ]
    assert json.dumps(summary, separators=(",", ":")) == json.dumps(
        route._recommended_capacity_summary(
            [1, 2],
            {1: 2, 2: 0},
            {
                "reallocation_moves": 1,
                "training_hits_before": 6,
                "training_hits_after": 9,
                "training_hit_gain": 3,
                "hysteresis_hits": 0,
            },
            {
                "uniform_per_layer_lru": {
                    "hits": 7,
                    "misses": 5,
                    "hit_rate": 7 / 12,
                },
                "trained_dynamic_quota_lru": {
                    "hits": 9,
                    "misses": 3,
                    "hit_rate": 0.75,
                },
            },
            record_bytes=100,
        ),
        separators=(",", ":"),
    )


def test_recommended_capacity_keeps_training_and_held_out_provenance_separate() -> None:
    training = {
        "reallocation_moves": 1,
        "training_hits_before": 6,
        "training_hits_after": 9,
        "training_hit_gain": 3,
        "hysteresis_hits": 0,
    }
    first = route._recommended_capacity_summary(
        [1, 2],
        {1: 2, 2: 0},
        training,
        {
            "uniform_per_layer_lru": {"hits": 7, "misses": 5, "hit_rate": 7 / 12},
            "trained_dynamic_quota_lru": {
                "hits": 9,
                "misses": 3,
                "hit_rate": 0.75,
            },
        },
        record_bytes=100,
    )
    second = route._recommended_capacity_summary(
        [1, 2],
        {1: 2, 2: 0},
        training,
        {
            "uniform_per_layer_lru": {"hits": 10, "misses": 2, "hit_rate": 5 / 6},
            "trained_dynamic_quota_lru": {
                "hits": 8,
                "misses": 4,
                "hit_rate": 2 / 3,
            },
        },
        record_bytes=100,
    )

    assert first["per_layer_quotas"] == second["per_layer_quotas"]
    assert first["training_gain"] == second["training_gain"]
    assert first["held_out_delta"] != second["held_out_delta"]


def test_cli_exports_train_derived_recommendation_in_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    trace_path = tmp_path / "route-trace.json"
    entries = [
        {"phase": "prefill", "layer": layer, "expert_ids": [0]} for layer in (1, 2)
    ]
    for decode_step, step_experts in enumerate(((0, 0), (1, 1), (0, 0), (1, 1))):
        entries.extend(
            {
                "phase": "decode",
                "layer": layer,
                "trace_epoch": 0,
                "decode_step": decode_step,
                "token_count": 1,
                "expert_ids": [step_experts[layer - 1]],
            }
            for layer in (1, 2)
        )
    trace_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_expert_route_trace.py",
            str(trace_path),
            "--capacity-per-layer",
            "1",
            "--expert-count",
            "2",
            "--top-k",
            "1",
            "--record-bytes",
            "100",
            "--train-steps",
            "2",
            "--cluster-sizes",
            "1,2",
        ],
    )

    assert route.main() == 0
    payload = json.loads(capsys.readouterr().out)
    recommended = payload["recommended_capacity"]

    assert recommended["per_layer_quotas"] == [
        {"layer": 1, "slots": payload["capacities"]["trained_dynamic_quota"]["1"]},
        {"layer": 2, "slots": payload["capacities"]["trained_dynamic_quota"]["2"]},
    ]
    assert recommended["total_slots"] == payload["total_persistent_slots"]
    assert recommended["total_bytes"] == payload["total_persistent_slots"] * 100
    assert (
        recommended["training_gain"]["hit_delta"]
        == payload["held_out"]["dynamic_quota_training"]["training_hit_gain"]
    )
    assert isinstance(payload["prefetch"]["temporal_previous_token"]["hits"], int)


def _run_trace_analysis(
    tmp_path: Path,
    monkeypatch,
    capsys,
    *,
    name: str,
    suffix: tuple[tuple[int, int], ...],
) -> dict:
    trace_path = tmp_path / f"{name}.json"
    train = (
        (0, 0),
        (1, 1),
        (0, 2),
        (1, 0),
        (0, 1),
        (1, 2),
    )
    entries = [
        {"phase": "prefill", "layer": layer, "token_count": 1, "expert_ids": [0]}
        for layer in (1, 2)
    ]
    for step, step_experts in enumerate((*train, *suffix)):
        entries.extend(
            {
                "phase": "decode",
                "layer": layer,
                "trace_epoch": 0,
                "decode_step": step,
                "token_count": 1,
                "expert_ids": [step_experts[layer - 1]],
            }
            for layer in (1, 2)
        )
    trace_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_expert_route_trace.py",
            str(trace_path),
            "--capacity-per-layer",
            "1",
            "--expert-count",
            "3",
            "--top-k",
            "1",
            "--record-bytes",
            "100",
            "--train-steps",
            "6",
            "--quota-hysteresis-hits",
            "0",
            "--cluster-sizes",
            "1,2",
        ],
    )
    assert route.main() == 0
    return json.loads(capsys.readouterr().out)


def test_cli_recommendation_is_invariant_to_held_out_suffix(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    hot_layer_one = _run_trace_analysis(
        tmp_path,
        monkeypatch,
        capsys,
        name="hot-layer-one",
        suffix=((0, 0), (1, 1), (0, 0), (1, 1)),
    )["recommended_capacity"]
    hot_layer_two = _run_trace_analysis(
        tmp_path,
        monkeypatch,
        capsys,
        name="hot-layer-two",
        suffix=((2, 2), (2, 2), (2, 2), (2, 2)),
    )["recommended_capacity"]

    for field in ("per_layer_quotas", "total_slots", "total_bytes", "training_gain"):
        assert hot_layer_one[field] == hot_layer_two[field]
    assert hot_layer_one["held_out_delta"] != hot_layer_two["held_out_delta"]


def test_cli_rejects_unequal_per_layer_decode_counts_before_emitting_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    trace_path = tmp_path / "unequal.json"
    entries = [
        {"phase": "prefill", "layer": layer, "expert_ids": [0]} for layer in (1, 2)
    ] + [
        {"phase": "decode", "layer": 1, "expert_ids": [0]},
        {
            "phase": "decode",
            "layer": 2,
            "decode_step": 0,
            "token_count": 1,
            "expert_ids": [0],
        },
        {
            "phase": "decode",
            "layer": 1,
            "decode_step": 1,
            "token_count": 1,
            "expert_ids": [1],
        },
        {
            "phase": "decode",
            "layer": 2,
            "decode_step": 1,
            "token_count": 1,
            "expert_ids": [1],
        },
        {
            "phase": "decode",
            "layer": 1,
            "decode_step": 2,
            "token_count": 1,
            "expert_ids": [0],
        },
    ]
    trace_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_expert_route_trace.py",
            str(trace_path),
            "--capacity-per-layer",
            "1",
            "--expert-count",
            "2",
            "--top-k",
            "1",
        ],
    )

    with pytest.raises(ValueError, match=r"decode row counts.*1: 3.*2: 2"):
        route.main()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("bad_steps", [(0, 2, 3), (0, 1, 1), (1, 0, 2)])
def test_cli_rejects_malformed_native_decode_step_sequence(
    tmp_path: Path,
    monkeypatch,
    capsys,
    bad_steps: tuple[int, ...],
) -> None:
    trace_path = tmp_path / "misaligned.json"
    entries = [
        {"phase": "prefill", "layer": layer, "expert_ids": [0]} for layer in (1, 2)
    ]
    for layer, steps in ((1, (0, 1, 2)), (2, bad_steps)):
        entries.extend(
            {
                "phase": "decode",
                "layer": layer,
                "trace_epoch": 0,
                "decode_step": step,
                "token_count": 1,
                "expert_ids": [step % 2],
            }
            for step in steps
        )
    trace_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_expert_route_trace.py",
            str(trace_path),
            "--capacity-per-layer",
            "1",
            "--expert-count",
            "2",
            "--top-k",
            "1",
        ],
    )

    with pytest.raises(ValueError, match="decode_step sequence"):
        route.main()
    assert capsys.readouterr().out == ""


def test_cli_requires_native_decode_step_identifier(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    trace_path = tmp_path / "legacy.json"
    entries = [
        {"phase": "prefill", "layer": layer, "expert_ids": [0]} for layer in (1, 2)
    ] + [
        {
            "phase": "decode",
            "layer": layer,
            "token_count": 1,
            "expert_ids": [step % 2],
        }
        for step in range(3)
        for layer in (1, 2)
    ]
    trace_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_expert_route_trace.py",
            str(trace_path),
            "--capacity-per-layer",
            "1",
            "--expert-count",
            "2",
            "--top-k",
            "1",
        ],
    )

    with pytest.raises(ValueError, match="decode_step.*regenerate"):
        route.main()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("batch_width", [2, 8])
def test_cli_processes_every_batched_routed_row(
    tmp_path: Path,
    monkeypatch,
    capsys,
    batch_width: int,
) -> None:
    trace_path = tmp_path / f"b{batch_width}.json"
    entries = [
        {"phase": "prefill", "layer": layer, "expert_ids": [0]} for layer in (1, 2)
    ]
    for decode_step in range(4):
        for layer in (1, 2):
            entries.append(
                {
                    "phase": "decode",
                    "layer": layer,
                    "trace_epoch": 0,
                    "decode_step": decode_step,
                    "token_count": batch_width,
                    "expert_ids": [
                        (decode_step + row + layer) % 3 for row in range(batch_width)
                    ],
                }
            )
    trace_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_expert_route_trace.py",
            str(trace_path),
            "--capacity-per-layer",
            "1",
            "--expert-count",
            "3",
            "--top-k",
            "1",
            "--train-steps",
            "2",
            "--cluster-sizes",
            "1",
        ],
    )

    assert route.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decode_steps"] == 4
    assert payload["decode_tokens"] == 4 * batch_width
    assert payload["held_out"]["split"]["evaluation_tokens"] == 2 * batch_width
    assert payload["held_out"]["policies"]["uniform_per_layer_lru"]["requests"] == (
        2 * batch_width * 2
    )
    temporal = payload["prefetch"]["temporal_previous_token"]
    assert temporal == {
        "supported": False,
        "hits": None,
        "requests": None,
        "recall": None,
        "reason": "batched traces do not carry stable request identities",
    }


def test_cli_resets_cache_simulations_at_trace_epoch_boundaries(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    trace_path = tmp_path / "reset-epochs.json"
    entries = []
    for epoch in (0, 1):
        if epoch:
            entries.append(
                {
                    "phase": "reset",
                    "previous_trace_epoch": epoch - 1,
                    "trace_epoch": epoch,
                }
            )
        entries.append(
            {
                "phase": "prefill",
                "layer": 1,
                "trace_epoch": epoch,
                "token_count": 1,
                "expert_ids": [0],
            }
        )
        for decode_step in (0, 1):
            entries.append(
                {
                    "phase": "decode",
                    "layer": 1,
                    "trace_epoch": epoch,
                    "decode_step": decode_step,
                    "token_count": 1,
                    "expert_ids": [1],
                }
            )
    trace_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_expert_route_trace.py",
            str(trace_path),
            "--capacity-per-layer",
            "1",
            "--expert-count",
            "2",
            "--top-k",
            "1",
            "--cluster-sizes",
            "1",
        ],
    )

    assert route.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["policies"]["lru_uniform"]["hits"] == 2
    assert payload["policies"]["lru_uniform"]["misses"] == 2
    assert payload["held_out"]["split"]["scope"] == ("per_epoch_chronological_prefix")
    assert payload["trace_epochs"] == [0, 1]


def test_cli_rejects_reset_epoch_without_its_own_causal_prefill(
    tmp_path: Path, monkeypatch
) -> None:
    entries = [
        {
            "phase": "prefill",
            "layer": 1,
            "trace_epoch": 0,
            "token_count": 1,
            "expert_ids": [0],
        }
    ]
    for step in (0, 1):
        entries.append(
            {
                "phase": "decode",
                "layer": 1,
                "trace_epoch": 0,
                "decode_step": step,
                "token_count": 1,
                "expert_ids": [1],
            }
        )
    entries.append({"phase": "reset", "previous_trace_epoch": 0, "trace_epoch": 1})
    for step in (0, 1):
        entries.append(
            {
                "phase": "decode",
                "layer": 1,
                "trace_epoch": 1,
                "decode_step": step,
                "token_count": 1,
                "expert_ids": [0],
            }
        )
    trace_path = tmp_path / "missing-prefill.json"
    trace_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_expert_route_trace.py",
            str(trace_path),
            "--capacity-per-layer",
            "1",
            "--expert-count",
            "2",
            "--top-k",
            "1",
        ],
    )

    with pytest.raises(ValueError, match="epoch 1 has no causal prefill"):
        route.main()


def test_cli_rejects_prefill_after_decode_in_original_entry_order(
    tmp_path: Path, monkeypatch
) -> None:
    trace_path = tmp_path / "late-prefill.json"
    trace_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "phase": "decode",
                        "layer": 1,
                        "trace_epoch": 0,
                        "decode_step": 0,
                        "token_count": 1,
                        "expert_ids": [0],
                    },
                    {
                        "phase": "prefill",
                        "layer": 1,
                        "trace_epoch": 0,
                        "token_count": 1,
                        "expert_ids": [0],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["analyze", str(trace_path)])
    with pytest.raises(ValueError, match="prefill route appears after decode"):
        route.main()


def test_cli_rejects_reset_before_epoch_routes_complete(
    tmp_path: Path, monkeypatch
) -> None:
    trace_path = tmp_path / "bad-reset.json"
    trace_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "phase": "reset",
                        "previous_trace_epoch": 0,
                        "trace_epoch": 1,
                    },
                    {
                        "phase": "prefill",
                        "layer": 1,
                        "trace_epoch": 1,
                        "token_count": 1,
                        "expert_ids": [0],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["analyze", str(trace_path)])
    with pytest.raises(ValueError, match="out-of-order trace reset"):
        route.main()


def test_cli_rejects_record_bytes_outside_signed_64_bit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = tmp_path / "unused.json"
    trace_path.write_text('{"entries": []}', encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_expert_route_trace.py",
            str(trace_path),
            "--record-bytes",
            str(10**400),
        ],
    )

    with pytest.raises(ValueError, match="record-bytes.*signed 64-bit"):
        route.main()


def test_cli_rejects_output_that_aliases_input_trace(
    tmp_path: Path, monkeypatch
) -> None:
    trace_path = tmp_path / "trace.json"
    original = '{"entries": []}\n'
    trace_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_expert_route_trace.py",
            str(trace_path),
            "--output-json",
            str(trace_path),
        ],
    )

    with pytest.raises(ValueError, match="input and output-json must be different"):
        route.main()
    assert trace_path.read_text(encoding="utf-8") == original


def test_cli_requires_transient_slots_in_v2_trace(tmp_path: Path, monkeypatch) -> None:
    trace_path = tmp_path / "trace-v2.json"
    trace_path.write_text(
        json.dumps({"schema": "mtplx-expert-route-trace-v2", "entries": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["analyze", str(trace_path)])
    with pytest.raises(ValueError, match="missing configured transient_slots"):
        route.main()


def test_cli_rejects_total_cache_byte_product_overflow(
    tmp_path: Path, monkeypatch
) -> None:
    entries = [
        {
            "phase": "prefill",
            "layer": 1,
            "trace_epoch": 0,
            "token_count": 1,
            "expert_ids": [0],
        },
        *[
            {
                "phase": "decode",
                "layer": 1,
                "trace_epoch": 0,
                "decode_step": step,
                "token_count": 1,
                "expert_ids": [step],
            }
            for step in (0, 1)
        ],
    ]
    trace_path = tmp_path / "overflow.json"
    trace_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_expert_route_trace.py",
            str(trace_path),
            "--capacity-per-layer",
            "2",
            "--expert-count",
            "2",
            "--top-k",
            "1",
            "--record-bytes",
            str(2**63 - 1),
        ],
    )

    with pytest.raises(ValueError, match="total cache byte projection"):
        route.main()


def test_cluster_replay_accounts_for_useful_speculative_record() -> None:
    left = ((1, 0), (1, 1))
    right = ((1, 2), (1, 3))
    clusters = {key: cluster for cluster in (left, right) for key in cluster}

    metric = route._run_cluster_lru_sequence(
        [(1, 0), (1, 1), (1, 0)],
        [],
        2,
        clusters,
    )

    assert metric["hits"] == 2
    assert metric["misses"] == 1
    assert metric["physical_records_read"] == 2
    assert metric["demanded_records_read"] == 1
    assert metric["speculative_records_read"] == 1
    assert metric["speculative_records_used_before_eviction"] == 1
    assert metric["useful_prefetch_ratio"] == 1.0


def test_cluster_size_one_is_record_granularity_global_lru() -> None:
    sequence = [(1, 0), (2, 0), (1, 1), (1, 0), (2, 1)]
    clusters = {key: (key,) for key in set(sequence)}

    lru_hits, lru_misses, lru_state = route._run_lru_sequence(sequence, [], 3)
    clustered = route._run_cluster_lru_sequence(sequence, [], 3, clusters)

    assert (clustered["hits"], clustered["misses"]) == (lru_hits, lru_misses)
    assert clustered["physical_records_read"] == lru_misses
    assert clustered["speculative_records_read"] == 0
    assert clustered["final_lru_order"] == lru_state


def test_coactivation_clustering_groups_high_lift_pairs() -> None:
    training = {
        1: [(0, 1)] * 8 + [(2, 3)] * 6 + [(0, 2), (1, 3)],
    }

    clusters = route._build_coactivation_clusters(
        [1],
        training,
        expert_count=4,
        cluster_size=2,
    )

    assert set(clusters[(1, 0)]) == {(1, 0), (1, 1)}
    assert set(clusters[(1, 2)]) == {(1, 2), (1, 3)}
