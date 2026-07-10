from __future__ import annotations

import pytest

from mtplx.expert_streaming import (
    CacheCounters,
    ExpertCacheSimulation,
    LayerExpertSlotBank,
    RoutingPhase,
)


def test_decode_fills_persistent_slots_and_then_hits() -> None:
    bank = LayerExpertSlotBank(
        expert_count=16,
        persistent_slots=4,
        transient_slots=2,
        frequency_decay=1.0,
    )

    first = bank.plan([2, 5], phase=RoutingPhase.DECODE)
    second = bank.plan([5, 2], phase=RoutingPhase.DECODE)

    assert first.hits == ()
    assert first.misses == (2, 5)
    assert all(load.persistent for load in first.loads)
    assert second.hits == (5, 2)
    assert second.misses == ()
    assert second.loads == ()
    assert second.slots == (first.slots[1], first.slots[0])


def test_prefill_uses_transient_slots_without_polluting_decode_hotset() -> None:
    bank = LayerExpertSlotBank(
        expert_count=32,
        persistent_slots=2,
        transient_slots=2,
    )
    bank.plan([1, 2], phase="decode")
    resident_before = bank.resident_experts

    prefill = bank.plan([10, 11], phase="prefill")

    assert bank.resident_experts == resident_before
    assert prefill.slots == (2, 3)
    assert all(not load.persistent for load in prefill.loads)
    assert prefill.evictions == ()


def test_frequent_decode_expert_eventually_displaces_a_cold_resident() -> None:
    bank = LayerExpertSlotBank(
        expert_count=16,
        persistent_slots=2,
        transient_slots=1,
        frequency_decay=1.0,
    )
    bank.plan([1], phase="decode")
    bank.plan([2], phase="decode")

    first_three = bank.plan([3], phase="decode")
    second_three = bank.plan([3], phase="decode")

    assert all(not load.persistent for load in first_three.loads)
    assert any(load.persistent for load in second_three.loads)
    assert len(second_three.evictions) == 1
    assert 3 in bank.resident_experts


def test_active_hit_is_pinned_during_multi_expert_admission() -> None:
    bank = LayerExpertSlotBank(
        expert_count=16,
        persistent_slots=2,
        transient_slots=2,
        frequency_decay=1.0,
    )
    bank.plan([1, 2], phase="decode")
    for _ in range(3):
        bank.plan([3, 1], phase="decode")

    assert 1 in bank.resident_experts


def test_duplicate_router_ids_share_one_load_and_slot() -> None:
    bank = LayerExpertSlotBank(
        expert_count=8,
        persistent_slots=1,
        transient_slots=2,
    )

    plan = bank.plan([4, 4, 6], phase="prefill")

    assert plan.misses == (4, 6)
    assert len(plan.loads) == 2
    assert plan.slots[0] == plan.slots[1]


def test_route_larger_than_transient_service_tier_is_rejected() -> None:
    bank = LayerExpertSlotBank(
        expert_count=16,
        persistent_slots=2,
        transient_slots=2,
    )

    with pytest.raises(ValueError, match="transient_slots"):
        bank.plan([1, 2, 3], phase="decode")


def test_counters_charge_only_cache_misses_for_io() -> None:
    bank = LayerExpertSlotBank(
        expert_count=16,
        persistent_slots=2,
        transient_slots=2,
    )
    counters = CacheCounters()
    first = bank.plan([1, 2], phase="decode")
    second = bank.plan([1, 2], phase="decode")
    counters.observe(first, expert_record_bytes=1024)
    counters.observe(second, expert_record_bytes=1024)

    assert counters.expert_requests == 4
    assert counters.expert_hits == 2
    assert counters.expert_misses == 2
    assert counters.bytes_read == 2048
    assert counters.hit_rate == 0.5


def test_simulation_reports_io_time_and_layer_hotsets() -> None:
    simulation = ExpertCacheSimulation(
        expert_count=16,
        persistent_slots=2,
        transient_slots=2,
        expert_record_bytes=100,
    )
    simulation.observe(layer_index=0, expert_ids=[1, 2], phase="decode")
    simulation.observe(layer_index=0, expert_ids=[1, 2], phase="decode")
    simulation.observe(layer_index=1, expert_ids=[3, 4], phase="prefill")

    summary = simulation.summary(effective_ssd_bytes_per_second=1000)

    assert summary["bytes_read"] == 400
    assert summary["estimated_io_seconds"] == pytest.approx(0.4)
    assert summary["layers_observed"] == 2
    assert summary["persistent_cache_bytes"] == 400
    assert summary["transient_scratch_bytes"] == 200
    assert summary["resident_experts_by_layer"]["0"] == [1, 2]
    assert summary["resident_experts_by_layer"]["1"] == []
