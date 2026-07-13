from __future__ import annotations

import pytest

from mtplx.expert_streaming import (
    CacheCounters,
    ExpertCacheSimulation,
    GlobalExpertSlotBank,
    LayerExpertSlotBank,
    RoutingPhase,
)


def test_global_cache_lends_fixed_slots_between_layers() -> None:
    bank = GlobalExpertSlotBank(
        layer_indices=(1, 2),
        expert_count=4,
        persistent_slots=2,
        transient_slots=2,
        prefill_slots_per_layer=1,
        cache_policy="lru",
    )
    bank.prepare_prefill_seed(1, [0])
    first = bank.plan(1, [0], phase="prefill")
    bank.prepare_prefill_seed(2, [0])
    second = bank.plan(2, [0], phase="prefill")

    assert first.loads[0].persistent is True
    assert second.loads[0].persistent is True
    assert bank.occupancy_by_layer == {1: 1, 2: 1}

    # Layer 1 can take layer 2's older slot during decode. The underlying
    # physical slot count remains exactly two.
    replacement = bank.plan(1, [1], phase="decode")
    assert replacement.loads[0].persistent is True
    assert replacement.evictions[0].previous_layer == 1
    assert replacement.evictions[0].next_layer == 1
    assert bank.occupancy == 2

    layer_one_again = bank.plan(1, [0], phase="decode")
    assert layer_one_again.evictions[0].previous_layer == 2
    assert bank.occupancy_by_layer == {1: 2, 2: 0}


def test_global_prefill_seed_accepts_chunk_wider_than_route_capacity() -> None:
    bank = GlobalExpertSlotBank(
        layer_indices=(1,),
        expert_count=4,
        persistent_slots=2,
        transient_slots=1,
        prefill_slots_per_layer=2,
        cache_policy="lru",
    )

    assert bank.prepare_prefill_seed(1, [0, 1, 0]) == (0, 1)


def test_global_joining_prefill_cannot_evict_decode_hot_set() -> None:
    bank = GlobalExpertSlotBank(
        layer_indices=(1, 2),
        expert_count=4,
        persistent_slots=2,
        transient_slots=2,
        prefill_slots_per_layer=1,
        cache_policy="lru",
    )
    for layer in (1, 2):
        plan, transaction = bank.plan_transaction(layer, [0], phase="decode")
        transaction.commit()
        assert plan.loads[0].persistent is True

    resident_before = bank.resident_experts_by_layer
    snapshot_before = bank.snapshot()

    # A joining prompt wants two cold experts while the fixed global cache is
    # full.  Prefill may use the service slots, but it cannot repurpose either
    # layer's established decode-hot slot.
    assert bank.prepare_prefill_seed(1, [1, 2, 1]) == ()
    joining_prefill = bank.plan(1, [1, 2], phase="prefill")

    assert joining_prefill.slots == (2, 3)
    assert all(not load.persistent for load in joining_prefill.loads)
    assert joining_prefill.evictions == ()
    assert bank.resident_experts_by_layer == resident_before
    assert bank.occupancy == bank.persistent_slots == 2
    assert bank.snapshot() == snapshot_before


def test_global_cache_uses_total_capacity_as_transient_slot_base() -> None:
    bank = GlobalExpertSlotBank(
        layer_indices=(1, 2),
        expert_count=4,
        persistent_slots=3,
        transient_slots=2,
        prefill_slots_per_layer=0,
        cache_policy="frequency",
    )

    plan = bank.plan(2, [1, 3], phase="prefill")

    assert plan.slots == (3, 4)
    assert all(not load.persistent for load in plan.loads)


def _global_all_hit_bank(*, transient_slots: int = 1) -> GlobalExpertSlotBank:
    bank = GlobalExpertSlotBank(
        layer_indices=(1, 2),
        expert_count=4,
        persistent_slots=3,
        transient_slots=transient_slots,
        prefill_slots_per_layer=0,
        cache_policy="lru",
    )
    for expert in (0, 1, 2):
        plan, transaction = bank.plan_transaction(1, [expert], phase="decode")
        transaction.commit()
        assert plan.loads
    return bank


def _global_all_hit_state(bank: GlobalExpertSlotBank) -> tuple[object, ...]:
    return (
        bank._decode_epoch,
        tuple(bank._slot_to_key),
        dict(bank._key_to_slot),
        tuple(
            sorted(
                (key, entry.slot, entry.generation, entry.state)
                for key, entry in bank._directory.items()
            )
        ),
        tuple(bank._slot_generations),
        tuple(bank._free_slots),
        frozenset(bank._free_slot_set),
        tuple(bank._lru.items()),
        tuple(
            sorted(
                (key, history.score, history.score_epoch, history.last_used)
                for key, history in bank._history.items()
            )
        ),
        tuple(sorted(bank._layer_occupancy.items())),
        bank._evictions,
        bank._cross_layer_evictions,
    )


def test_global_all_hit_transaction_defers_state_and_rolls_back_exactly() -> None:
    bank = _global_all_hit_bank()
    before = _global_all_hit_state(bank)

    planned = bank.try_plan_all_hits_transaction(1, [2, 0, 2, 1], phase="decode")

    assert planned is not None
    route, transaction = planned
    assert route.experts == (2, 0, 2, 1)
    assert route.slots[0] == route.slots[2]
    assert route.hits == (2, 0, 1)
    assert route.misses == ()
    assert route.loads == ()
    assert all(generation is not None for generation in route.generations)
    assert _global_all_hit_state(bank) == before

    transaction.rollback_completion()
    assert _global_all_hit_state(bank) == before


def test_global_all_hit_transaction_commit_matches_normal_hit_policy() -> None:
    bank = _global_all_hit_bank()
    control = _global_all_hit_bank(transient_slots=3)
    expected = control.plan(1, [2, 0, 2, 1], phase="decode")

    planned = bank.try_plan_all_hits_transaction(1, [2, 0, 2, 1], phase="decode")

    assert planned is not None
    route, transaction = planned
    transaction.commit()
    assert route == expected
    assert _global_all_hit_state(bank) == _global_all_hit_state(control)


def test_global_all_hit_miss_is_side_effect_free() -> None:
    bank = _global_all_hit_bank()
    before = _global_all_hit_state(bank)

    assert bank.try_plan_all_hits_transaction(2, [0], phase="decode") is None

    assert _global_all_hit_state(bank) == before


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


def test_lru_admits_first_miss_and_evicts_least_recent_unpinned_expert() -> None:
    bank = LayerExpertSlotBank(
        expert_count=16,
        persistent_slots=2,
        transient_slots=1,
        cache_policy="lru",
    )
    bank.plan([1], phase="decode")
    bank.plan([2], phase="decode")
    bank.plan([1], phase="decode")

    miss = bank.plan([3], phase="decode")

    assert len(miss.evictions) == 1
    assert miss.evictions[0].previous_expert == 2
    assert any(load.persistent for load in miss.loads)
    assert set(bank.resident_experts) == {1, 3}


def test_singleton_decode_miss_does_not_win_only_because_resident_decayed() -> None:
    bank = LayerExpertSlotBank(
        expert_count=16,
        persistent_slots=2,
        transient_slots=1,
    )
    bank.plan([1], phase="decode")
    bank.plan([2], phase="decode")
    resident_before = bank.resident_experts

    singleton = bank.plan([3], phase="decode")

    assert bank.resident_experts == resident_before
    assert all(not load.persistent for load in singleton.loads)
    assert singleton.evictions == ()


def test_prefill_does_not_age_decode_admission_history() -> None:
    bank = LayerExpertSlotBank(
        expert_count=32,
        persistent_slots=2,
        transient_slots=1,
        frequency_decay=0.5,
    )
    bank.plan([1], phase="decode")
    bank.plan([2], phase="decode")
    resident_before = bank.resident_experts

    for _ in range(32):
        bank.plan([10], phase="prefill")
    singleton = bank.plan([3], phase="decode")

    assert bank.resident_experts == resident_before
    assert all(not load.persistent for load in singleton.loads)
    assert singleton.evictions == ()


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


def test_all_hit_probe_preserves_duplicate_order_beyond_transient_capacity() -> None:
    bank = LayerExpertSlotBank(
        expert_count=8,
        persistent_slots=3,
        transient_slots=1,
    )
    for expert in (0, 1, 2):
        bank.plan([expert], phase="decode")

    route = bank.try_plan_all_hits([2, 0, 2, 1], phase="decode")

    assert route is not None
    assert route.experts == (2, 0, 2, 1)
    assert route.slots[0] == route.slots[2]
    assert route.hits == (2, 0, 1)
    assert route.misses == ()
    assert route.loads == ()


def test_failed_all_hit_probe_leaves_normal_miss_planning_available() -> None:
    bank = LayerExpertSlotBank(
        expert_count=8,
        persistent_slots=1,
        transient_slots=1,
    )
    bank.plan([0], phase="decode")

    assert bank.try_plan_all_hits([0, 1], phase="decode") is None
    miss = bank.plan([1], phase="decode")

    assert miss.misses == (1,)


def test_all_hit_probe_matches_normal_wave_policy_and_next_lru_victims() -> None:
    normal = LayerExpertSlotBank(
        expert_count=8,
        persistent_slots=4,
        transient_slots=2,
        cache_policy="lru",
    )
    fast = LayerExpertSlotBank(
        expert_count=8,
        persistent_slots=4,
        transient_slots=2,
        cache_policy="lru",
    )
    for expert in (0, 1, 2, 3):
        normal.plan([expert], phase="decode")
        fast.plan([expert], phase="decode")

    for wave in ((2, 3), (0, 1)):
        expected = normal.plan(wave, phase="decode")
        actual = fast.try_plan_all_hits(wave, phase="decode")
        assert actual == expected

    assert (
        fast.plan([4], phase="decode").evictions
        == normal.plan([4], phase="decode").evictions
    )
    assert (
        fast.plan([5], phase="decode").evictions
        == normal.plan([5], phase="decode").evictions
    )


def test_failed_all_hit_probe_is_side_effect_free() -> None:
    control = LayerExpertSlotBank(
        expert_count=8,
        persistent_slots=1,
        transient_slots=1,
    )
    probe = LayerExpertSlotBank(
        expert_count=8,
        persistent_slots=1,
        transient_slots=1,
    )
    control.plan([0], phase="decode")
    probe.plan([0], phase="decode")
    before = (
        probe._decode_epoch,
        probe.resident_experts,
        tuple(
            (history.score, history.score_epoch, history.last_used)
            for history in probe._history
        ),
    )

    assert probe.try_plan_all_hits([0, 1], phase="decode") is None

    after = (
        probe._decode_epoch,
        probe.resident_experts,
        tuple(
            (history.score, history.score_epoch, history.last_used)
            for history in probe._history
        ),
    )
    assert after == before
    assert probe.plan([1], phase="decode") == control.plan([1], phase="decode")


def test_counters_preserve_router_assignment_multiplicity() -> None:
    bank = LayerExpertSlotBank(
        expert_count=8,
        persistent_slots=0,
        transient_slots=2,
    )
    counters = CacheCounters()

    plan = bank.plan([4, 4, 6], phase="decode")
    counters.observe(plan, expert_record_bytes=100)

    assert counters.expert_requests == 3
    assert counters.unique_expert_requests == 2
    assert counters.shared_expert_assignments == 1
    assert counters.expert_misses == 3
    assert counters.transient_loads == 2
    assert counters.bytes_read == 200


@pytest.mark.parametrize("invalid_id", [1.9, "1", True])
def test_router_ids_must_be_exact_integers(invalid_id: object) -> None:
    bank = LayerExpertSlotBank(
        expert_count=8,
        persistent_slots=0,
        transient_slots=2,
    )

    with pytest.raises(TypeError, match="exact integers"):
        bank.plan([invalid_id], phase="decode")


def test_route_larger_than_transient_service_tier_is_rejected() -> None:
    bank = LayerExpertSlotBank(
        expert_count=16,
        persistent_slots=2,
        transient_slots=2,
    )

    with pytest.raises(ValueError, match="transient_slots"):
        bank.plan([1, 2, 3], phase="decode")


def test_persistent_capacity_cannot_exceed_the_model_expert_count() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        LayerExpertSlotBank(
            expert_count=16,
            persistent_slots=17,
            transient_slots=8,
        )


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


def test_simulation_reports_full_allocated_bank_for_a_partial_trace() -> None:
    simulation = ExpertCacheSimulation(
        expert_count=256,
        persistent_slots=32,
        transient_slots=8,
        expert_record_bytes=21_233_664,
        allocated_layer_count=75,
    )
    simulation.observe(layer_index=3, expert_ids=range(8), phase="decode")

    summary = simulation.summary(effective_ssd_bytes_per_second=1)

    assert summary["layers_observed"] == 1
    assert summary["allocated_layer_count"] == 75
    assert summary["persistent_cache_scope"] == "configured_model"
    assert summary["persistent_cache_bytes"] == 75 * 32 * 21_233_664
    assert summary["observed_layer_cache_bytes"] == 32 * 21_233_664
