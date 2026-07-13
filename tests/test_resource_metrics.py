from __future__ import annotations

import pytest

from mtplx.resource_metrics import (
    ExpertPipelineLedger,
    PoolOccupancy,
)


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, nanoseconds: int) -> None:
        self.now_ns += int(nanoseconds)


def test_pool_occupancy_integrates_queue_workers_and_units() -> None:
    clock = FakeClock()
    metrics = PoolOccupancy(worker_capacity=2, clock_ns=clock)
    metrics.submitted(100)
    clock.advance(10)
    metrics.started(100)
    clock.advance(20)
    metrics.completed(100)

    snapshot = metrics.snapshot()

    assert snapshot["accepted_submissions"] == 1
    assert snapshot["started"] == 1
    assert snapshot["completed"] == 1
    assert snapshot["queued_work_ns"] == 10
    assert snapshot["active_work_ns"] == 20
    assert snapshot["queued_unit_ns"] == 1_000
    assert snapshot["active_unit_ns"] == 2_000
    assert snapshot["queued_work_peak"] == 1
    assert snapshot["active_work_peak"] == 1
    assert snapshot["queued_units_peak"] == 100
    assert snapshot["active_units_peak"] == 100
    assert snapshot["queued_work"] == 0
    assert snapshot["active_work"] == 0


def test_rejected_submission_restores_queue_accounting() -> None:
    metrics = PoolOccupancy(worker_capacity=1)
    metrics.submitted(64)
    metrics.rejected(64)

    snapshot = metrics.snapshot()

    assert snapshot["accepted_submissions"] == 0
    assert snapshot["rejected_submissions"] == 1
    assert snapshot["queued_work"] == 0
    assert snapshot["queued_units"] == 0


def test_pool_occupancy_rejects_impossible_transitions() -> None:
    metrics = PoolOccupancy(worker_capacity=1)

    with pytest.raises(RuntimeError, match="queue underflow"):
        metrics.started(1)

    metrics.submitted(1)
    metrics.started(1)
    with pytest.raises(RuntimeError, match="active underflow"):
        metrics.completed(2)


def test_pool_occupancy_validates_capacity_and_units() -> None:
    with pytest.raises(ValueError, match="worker_capacity"):
        PoolOccupancy(worker_capacity=0)

    metrics = PoolOccupancy(worker_capacity=1)
    with pytest.raises(ValueError, match="units"):
        metrics.submitted(-1)


def test_pipeline_ledger_integrates_record_lifecycle_and_overlap() -> None:
    clock = FakeClock()
    ledger = ExpertPipelineLedger(strict=True, clock_ns=clock)
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(5,),
        load_logical_bytes=(100,),
    )
    hit_work = ledger.begin_hit_work((2,))
    clock.advance(2)
    clock.advance(5)
    route.observe_block(5, "pin_held", elapsed_ns=5)
    clock.advance(3)
    route.submission_attempted((5,))
    route.submission_accepted((5,))
    route.reader_started((5,))
    hit_work.claim()
    hit_work.close()
    route.begin_generation_wait()
    range_token = ledger.range_started(logical_bytes=100, phase="decode")
    clock.advance(10)
    ledger.range_completed(range_token)
    route.record_verified(5)
    route.reader_completed((5,), thread_cpu_ns=4)
    route.record_runnable(5)
    route.end_generation_wait()
    clock.advance(5)
    route.claim_misses((5,))
    route.close()

    snapshot = ledger.snapshot()
    assert set(snapshot["block_counts"]) == {"pin_held", "slot_loading"}
    assert snapshot["counters"]["logical_record_jobs"] == 1
    assert snapshot["counters"]["logical_record_bytes"] == 100
    assert snapshot["counters"]["submission_attempted_record_jobs"] == 1
    assert snapshot["counters"]["submission_attempted_record_bytes"] == 100
    assert snapshot["counters"]["accepted_record_jobs"] == 1
    assert snapshot["counters"]["accepted_record_bytes"] == 100
    assert snapshot["counters"]["verified_record_jobs"] == 1
    assert snapshot["counters"]["runnable_record_jobs"] == 1
    assert snapshot["counters"]["claimed_record_jobs"] == 1
    assert snapshot["counters"]["reader_thread_cpu_ns"] == 4
    assert snapshot["integrals_ns"]["eligible_unsubmitted_record_ns"] == 10
    assert snapshot["block_ns"]["pin_held"] == 5
    assert snapshot["integrals_ns"]["generation_expert_input_wait_ns"] == 10
    assert snapshot["integrals_ns"]["generation_wait_storage_active_ns"] == 10
    assert snapshot["primary_integrals_ns"]["generation_thread_expert_input_wait"] == 10
    assert snapshot["primary_integrals_ns"]["host_runnable_work"] == 5
    assert (
        snapshot["by_phase"]["decode"]["integrals_ns"][
            "generation_wait_storage_active_ns"
        ]
        == 10
    )
    assert snapshot["integrals_ns"]["runnable_miss_unclaimed_record_ns"] == 5
    assert all(value == 0 for value in snapshot["gauges"].values())


@pytest.mark.parametrize(
    "selected",
    (
        "pin_held",
        "slot_loading",
    ),
)
def test_pipeline_ledger_preserves_block_reason_identity(selected: str) -> None:
    clock = FakeClock()
    ledger = ExpertPipelineLedger(strict=True, clock_ns=clock)
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(5,),
        load_logical_bytes=(100,),
    )

    route.observe_block(5, selected, elapsed_ns=1)
    route.close()

    snapshot = ledger.snapshot()
    for reason, count in snapshot["block_counts"].items():
        assert count == (1 if reason == selected else 0)
    for reason, duration in snapshot["block_ns"].items():
        assert duration == (1 if reason == selected else 0)
    assert snapshot["by_phase"]["decode"]["block_counts"][selected] == 1
    assert snapshot["by_phase"]["decode"]["block_ns"][selected] == 1
    assert snapshot["block_coverage"] == {
        "operation_credit": "unavailable",
        "byte_credit": "unavailable",
        "authoritative_reserve": "unavailable",
        "slot_unavailable": "unavailable",
        "pin_held": "measured",
        "slot_loading": "measured",
    }


def test_pipeline_route_close_abandons_all_nonterminal_records() -> None:
    ledger = ExpertPipelineLedger(strict=True, clock_ns=FakeClock())
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(1, 2, 3, 4, 5, 6),
        load_logical_bytes=(10, 20, 30, 40, 50, 60),
    )
    route.submission_attempted((2,))
    route.submission_attempted((3,))
    route.submission_accepted((3,))
    route.submission_attempted((4,))
    route.submission_accepted((4,))
    route.reader_started((4,))
    route.record_verified(4)
    route.reader_completed((4,), thread_cpu_ns=0)
    route.record_runnable(4)
    route.satisfied_without_submit((5,))
    route.record_runnable(5)
    route.satisfied_without_submit((6,))

    route.close()
    route.close()

    snapshot = ledger.snapshot()
    assert snapshot["counters"]["logical_record_jobs"] == 6
    assert snapshot["counters"]["satisfied_without_submit_record_jobs"] == 2
    assert snapshot["counters"]["abandoned_record_jobs"] == 6
    assert snapshot["counters"]["abandoned_record_bytes"] == 210
    assert snapshot["counters"]["claimed_record_jobs"] == 0
    assert all(value == 0 for value in snapshot["gauges"].values())


def test_pipeline_ledger_integrates_active_ranges_and_bytes_exactly() -> None:
    clock = FakeClock()
    ledger = ExpertPipelineLedger(strict=True, clock_ns=clock)

    first = ledger.range_started(logical_bytes=100, phase="decode")
    clock.advance(3)
    second = ledger.range_started(logical_bytes=50, phase="prefill")
    clock.advance(4)
    ledger.range_completed(first)
    clock.advance(2)
    ledger.range_completed(second)

    snapshot = ledger.snapshot()
    assert snapshot["counters"]["started_logical_ranges"] == 2
    assert snapshot["counters"]["started_logical_range_bytes"] == 150
    assert snapshot["counters"]["completed_logical_ranges"] == 2
    assert snapshot["integrals_ns"]["active_logical_range_ns"] == 13
    assert snapshot["integrals_ns"]["active_logical_range_byte_ns"] == 1_000
    assert snapshot["gauges"]["active_logical_ranges"] == 0
    assert snapshot["gauges"]["active_logical_range_bytes"] == 0
    assert (
        snapshot["by_phase"]["decode"]["counters"]["started_logical_range_bytes"] == 100
    )
    assert (
        snapshot["by_phase"]["prefill"]["counters"]["started_logical_range_bytes"] == 50
    )
    assert "scheduled_read_ranges" not in snapshot["counters"]


def test_pipeline_latency_histograms_are_bounded_and_complete() -> None:
    clock = FakeClock()
    ledger = ExpertPipelineLedger(strict=True, clock_ns=clock)
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(5,),
        load_logical_bytes=(100,),
    )
    route.submission_attempted((5,))
    route.submission_accepted((5,))
    route.reader_started((5,))
    span = ledger.range_started(logical_bytes=100, phase="decode")
    clock.advance(10**9 + 1)
    ledger.range_completed(span)
    route.record_verified(5)
    route.reader_completed((5,), thread_cpu_ns=0)
    route.record_runnable(5)
    route.claim_misses((5,))
    route.close()

    snapshot = ledger.snapshot()
    for histogram in snapshot["histograms"].values():
        assert sum(histogram["bucket_counts"]) == histogram["sample_count"]
        assert len(histogram["bucket_counts"]) == len(histogram["bounds_ns"]) + 1
    assert snapshot["histograms"]["logical_range_latency_ns"]["overflow_count"] == 1
    assert snapshot["histograms"]["complete_record_latency_ns"]["overflow_count"] == 1
    assert (
        snapshot["by_phase"]["decode"]["histograms"]["logical_range_latency_ns"][
            "overflow_count"
        ]
        == 1
    )
    assert (
        snapshot["by_phase"]["decode"]["histograms"]["complete_record_latency_ns"][
            "overflow_count"
        ]
        == 1
    )
    assert snapshot["coverage"]["attribution"] == "measured"


def test_pipeline_verified_record_is_not_runnable_until_explicit_publish() -> None:
    clock = FakeClock()
    ledger = ExpertPipelineLedger(strict=True, clock_ns=clock)
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(5,),
        load_logical_bytes=(100,),
    )
    route.submission_attempted((5,))
    route.submission_accepted((5,))
    route.reader_started((5,))
    clock.advance(4)
    route.record_verified(5)
    verified = ledger.snapshot()
    assert verified["gauges"]["runnable_miss_records"] == 0

    clock.advance(3)
    route.reader_completed((5,), thread_cpu_ns=0)
    route.record_runnable(5)
    clock.advance(2)
    runnable = ledger.snapshot()
    assert runnable["integrals_ns"]["runnable_miss_unclaimed_record_ns"] == 2
    histogram = runnable["histograms"]["complete_record_latency_ns"]
    assert histogram["sample_count"] == 1

    route.claim_misses((5,))
    route.close()


def test_pipeline_ledger_rejects_duplicate_and_out_of_order_transitions() -> None:
    ledger = ExpertPipelineLedger(strict=True, clock_ns=FakeClock())

    with pytest.raises(RuntimeError, match="duplicate load expert"):
        ledger.begin_route(
            layer=1,
            phase="decode",
            load_experts=(5, 5),
            load_logical_bytes=(100, 100),
        )

    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(5,),
        load_logical_bytes=(100,),
    )
    with pytest.raises(RuntimeError, match="eligible"):
        route.record_verified(5)
    with pytest.raises(RuntimeError, match="unavailable"):
        route.observe_block(5, "operation_credit", elapsed_ns=1)
    with pytest.raises(RuntimeError, match="unknown active range"):
        ledger.range_completed(999)

    route.submission_attempted((5,))
    with pytest.raises(RuntimeError, match="eligible"):
        route.submission_attempted((5,))
    route.submission_accepted((5,))
    route.reader_started((5,))
    with pytest.raises(RuntimeError, match="reader-active"):
        route.claim_misses((5,))

    route.close()
    snapshot = ledger.snapshot()
    assert snapshot["invariant_failures"] == 6


def test_pipeline_ledger_rejects_gauge_underflow() -> None:
    ledger = ExpertPipelineLedger(strict=True, clock_ns=FakeClock())
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(),
        load_logical_bytes=(),
    )
    hit_work = ledger.begin_hit_work((2,))

    hit_work.claim()
    with pytest.raises(RuntimeError, match="already claimed"):
        hit_work.claim()
    hit_work.close()
    route.close()

    snapshot = ledger.snapshot()
    assert snapshot["gauges"]["runnable_hit_work"] == 0
    assert snapshot["invariant_failures"] == 1


def test_pipeline_ledger_detects_internal_gauge_underflow_atomically() -> None:
    ledger = ExpertPipelineLedger(strict=True, clock_ns=FakeClock())

    with ledger._lock, pytest.raises(RuntimeError, match="gauge underflow"):
        ledger._change_gauges_locked(
            {
                "runnable_hit_work": -1,
                "runnable_shared_work": 1,
            },
            "unscoped",
        )

    snapshot = ledger.snapshot()
    assert snapshot["gauges"]["runnable_hit_work"] == 0
    assert snapshot["gauges"]["runnable_shared_work"] == 0
    assert snapshot["invariant_failures"] == 1


def test_pipeline_ledger_tracks_shared_work_outside_route_lifecycle() -> None:
    clock = FakeClock()
    ledger = ExpertPipelineLedger(strict=True, clock_ns=clock)
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(),
        load_logical_bytes=(),
    )
    shared_work = ledger.begin_shared_work()
    clock.advance(7)

    route.close()
    shared_work.claim()
    shared_work.close()

    snapshot = ledger.snapshot()
    assert snapshot["counters"]["claimed_shared_work"] == 1
    assert snapshot["integrals_ns"]["runnable_shared_work_ns"] == 7
    assert snapshot["gauges"]["runnable_shared_work"] == 0


def test_pipeline_work_span_close_abandons_unclaimed_work() -> None:
    ledger = ExpertPipelineLedger(strict=True, clock_ns=FakeClock())
    hit_work = ledger.begin_hit_work((2, 3))
    shared_work = ledger.begin_shared_work()

    hit_work.close()
    hit_work.close()
    shared_work.close()

    snapshot = ledger.snapshot()
    assert snapshot["counters"]["abandoned_hit_work"] == 2
    assert snapshot["counters"]["abandoned_shared_work"] == 1
    assert snapshot["gauges"]["runnable_hit_work"] == 0
    assert snapshot["gauges"]["runnable_shared_work"] == 0


def test_pipeline_submission_rejection_rolls_back_and_marks_coverage() -> None:
    ledger = ExpertPipelineLedger(strict=True, clock_ns=FakeClock())
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(5,),
        load_logical_bytes=(100,),
    )

    route.submission_attempted((5,))
    route.submission_rejected((5,))

    snapshot = ledger.snapshot()
    assert snapshot["counters"]["submission_attempts"] == 1
    assert snapshot["counters"]["submission_rejections"] == 1
    assert snapshot["counters"]["rejected_record_jobs"] == 1
    assert snapshot["counters"]["rejected_record_bytes"] == 100
    assert snapshot["gauges"]["eligible_unsubmitted_records"] == 1
    assert snapshot["gauges"]["eligible_unsubmitted_record_bytes"] == 100
    assert snapshot["gauges"]["provisional_submission_records"] == 0
    assert snapshot["coverage"]["attribution"] == "incomplete"
    assert snapshot["by_phase"]["decode"]["coverage"]["attribution"] == "incomplete"
    route.close()


def test_pipeline_worker_may_start_before_submit_returns() -> None:
    ledger = ExpertPipelineLedger(strict=True, clock_ns=FakeClock())
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(5,),
        load_logical_bytes=(100,),
    )

    route.submission_attempted((5,))
    route.reader_started((5,))
    route.submission_accepted((5,))
    route.record_verified(5)
    route.reader_completed((5,), thread_cpu_ns=0)
    route.record_runnable(5)
    route.claim_misses((5,))
    route.close()

    snapshot = ledger.snapshot()
    assert snapshot["counters"]["accepted_record_jobs"] == 1
    assert snapshot["counters"]["queued_reader_record_jobs"] == 0
    assert snapshot["counters"]["active_reader_record_jobs"] == 1
    assert all(value == 0 for value in snapshot["gauges"].values())


def test_satisfied_without_submit_is_nonterminal_until_claimed() -> None:
    clock = FakeClock()
    ledger = ExpertPipelineLedger(strict=True, clock_ns=clock)
    route = ledger.begin_route(
        layer=1,
        phase="prefill",
        load_experts=(5,),
        load_logical_bytes=(100,),
    )

    route.satisfied_without_submit((5,))
    clock.advance(3)
    route.record_runnable(5)
    clock.advance(2)
    route.claim_misses((5,))
    route.close()

    snapshot = ledger.snapshot()
    assert snapshot["integrals_ns"]["satisfied_not_runnable_record_ns"] == 3
    assert snapshot["integrals_ns"]["runnable_miss_unclaimed_record_ns"] == 2
    assert snapshot["primary_integrals_ns"]["route_publication_pending"] == 3
    assert snapshot["primary_integrals_ns"]["host_runnable_work"] == 2
    assert snapshot["counters"]["claimed_record_bytes"] == 100


def test_pipeline_primary_projection_has_honest_reader_service_state() -> None:
    clock = FakeClock()
    ledger = ExpertPipelineLedger(strict=True, clock_ns=clock)
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(5,),
        load_logical_bytes=(100,),
    )

    clock.advance(2)  # eligible and unsubmitted
    route.submission_attempted((5,))
    route.submission_accepted((5,))
    clock.advance(3)  # submitted and queued
    route.reader_started((5,))
    clock.advance(3)  # reader service before a logical range
    token = ledger.range_started(100, phase="decode")
    clock.advance(4)  # storage takes precedence over reader service
    ledger.range_completed(token)
    clock.advance(2)  # post-range hash/validation remains reader service
    route.record_verified(5)
    route.reader_completed((5,), thread_cpu_ns=0)
    route.record_runnable(5)
    route.claim_misses((5,))
    route.close()
    clock.advance(6)
    wait_route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(),
        load_logical_bytes=(),
    )
    wait_route.begin_generation_wait()
    clock.advance(5)
    wait_route.end_generation_wait()
    wait_route.close()

    primary = ledger.snapshot()["primary_integrals_ns"]
    assert primary == {
        "generation_thread_expert_input_wait": 5,
        "logical_range_active": 4,
        "reader_completion_active": 5,
        "submitted_queued": 3,
        "eligible_unsubmitted": 2,
        "host_runnable_work": 0,
        "route_publication_pending": 0,
        "no_known_useful_work": 6,
    }
    decode_primary = ledger.snapshot()["by_phase"]["decode"]
    assert decode_primary["observation_ns"] == 19
    assert sum(decode_primary["primary_integrals_ns"].values()) == 19
    assert decode_primary["primary_integrals_ns"]["no_known_useful_work"] == 0


def test_pipeline_phase_scopes_operations_bytes_and_integrals() -> None:
    clock = FakeClock()
    ledger = ExpertPipelineLedger(strict=True, clock_ns=clock)
    decode = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(1, 2),
        load_logical_bytes=(10, 20),
    )
    prefill = ledger.begin_route(
        layer=1,
        phase="prefill",
        load_experts=(3,),
        load_logical_bytes=(30,),
    )
    decode.submission_attempted((1,))
    decode.submission_accepted((1,))
    prefill.submission_attempted((3,))
    unscoped_range = ledger.range_started(7)
    clock.advance(2)

    snapshot = ledger.snapshot()
    decode_phase = snapshot["by_phase"]["decode"]
    prefill_phase = snapshot["by_phase"]["prefill"]
    unscoped_phase = snapshot["by_phase"]["unscoped"]
    assert decode_phase["counters"]["logical_record_jobs"] == 2
    assert decode_phase["counters"]["logical_record_bytes"] == 30
    assert decode_phase["counters"]["accepted_record_bytes"] == 10
    assert decode_phase["gauges"]["eligible_unsubmitted_record_bytes"] == 20
    assert prefill_phase["counters"]["submission_attempted_record_bytes"] == 30
    assert prefill_phase["gauges"]["provisional_submission_record_bytes"] == 30
    assert unscoped_phase["gauges"]["active_logical_range_bytes"] == 7
    assert unscoped_phase["integrals_ns"]["active_logical_range_byte_ns"] == 14

    ledger.range_completed(unscoped_range)
    decode.close()
    prefill.close()


def test_pipeline_phase_scopes_hit_and_shared_work() -> None:
    clock = FakeClock()
    ledger = ExpertPipelineLedger(strict=True, clock_ns=clock)
    hit_work = ledger.begin_hit_work((1, 2), phase="decode")
    shared_work = ledger.begin_shared_work(phase="prefill")
    clock.advance(4)

    hit_work.claim()
    hit_work.close()
    shared_work.close()

    snapshot = ledger.snapshot()
    decode = snapshot["by_phase"]["decode"]
    prefill = snapshot["by_phase"]["prefill"]
    assert decode["integrals_ns"]["runnable_hit_work_ns"] == 8
    assert decode["counters"]["claimed_hit_work"] == 2
    assert prefill["integrals_ns"]["runnable_shared_work_ns"] == 4
    assert prefill["counters"]["abandoned_shared_work"] == 1


def test_reader_failure_clears_records_tasks_bytes_and_keeps_cpu() -> None:
    ledger = ExpertPipelineLedger(strict=True, clock_ns=FakeClock())
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(1, 2),
        load_logical_bytes=(40, 60),
    )
    route.submission_attempted((1, 2))
    route.submission_accepted((1, 2))
    route.reader_started((1, 2))
    route.record_verified(1)

    route.reader_failed((1, 2), thread_cpu_ns=7)
    route.close()

    snapshot = ledger.snapshot()
    assert snapshot["counters"]["failed_reader_tasks"] == 1
    assert snapshot["counters"]["failed_record_jobs"] == 2
    assert snapshot["counters"]["failed_record_bytes"] == 100
    assert snapshot["counters"]["abandoned_record_jobs"] == 2
    assert snapshot["counters"]["reader_thread_cpu_ns"] == 7
    assert all(value == 0 for value in snapshot["gauges"].values())


def test_worker_may_complete_before_submission_acceptance_returns() -> None:
    ledger = ExpertPipelineLedger(strict=True, clock_ns=FakeClock())
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(5,),
        load_logical_bytes=(100,),
    )
    route.submission_attempted((5,))
    route.reader_started((5,))
    route.record_verified(5)
    route.reader_completed((5,), thread_cpu_ns=3)

    route.submission_accepted((5,))
    route.record_runnable(5)
    route.claim_misses((5,))
    route.close()

    snapshot = ledger.snapshot()
    assert snapshot["counters"]["accepted_submissions"] == 1
    assert snapshot["counters"]["completed_reader_tasks"] == 1
    assert snapshot["counters"]["claimed_record_bytes"] == 100
    assert all(value == 0 for value in snapshot["gauges"].values())


def test_generation_wait_with_runnable_work_is_raw_overlap_only() -> None:
    clock = FakeClock()
    ledger = ExpertPipelineLedger(strict=True, clock_ns=clock)
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(),
        load_logical_bytes=(),
    )
    hit_work = ledger.begin_hit_work((2,))
    route.begin_generation_wait()
    clock.advance(3)

    snapshot = ledger.snapshot()
    assert snapshot["primary_integrals_ns"]["generation_thread_expert_input_wait"] == 3
    assert snapshot["integrals_ns"]["generation_wait_runnable_ns"] == 3
    assert snapshot["invariant_failures"] == 0
    assert snapshot["coverage"]["attribution"] == "measured"
    route.end_generation_wait()
    hit_work.close()
    route.close()


def test_mixed_eligible_and_submitted_prefers_submitted_and_tracks_wait_overlap() -> (
    None
):
    clock = FakeClock()
    ledger = ExpertPipelineLedger(strict=True, clock_ns=clock)
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(1, 2),
        load_logical_bytes=(10, 20),
    )
    route.submission_attempted((1,))
    route.submission_accepted((1,))
    clock.advance(2)
    route.begin_generation_wait()
    clock.advance(3)

    snapshot = ledger.snapshot()
    assert snapshot["primary_integrals_ns"]["submitted_queued"] == 2
    assert snapshot["primary_integrals_ns"]["eligible_unsubmitted"] == 0
    assert snapshot["integrals_ns"]["generation_wait_submitted_queued_ns"] == 3
    assert snapshot["integrals_ns"]["generation_wait_eligible_unsubmitted_ns"] == 3
    assert (
        snapshot["by_phase"]["decode"]["integrals_ns"][
            "generation_wait_submitted_queued_ns"
        ]
        == 3
    )
    route.end_generation_wait()
    route.close()


def test_non_strict_violation_does_not_raise_or_mutate_valid_state() -> None:
    ledger = ExpertPipelineLedger(strict=False, clock_ns=FakeClock())
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(5,),
        load_logical_bytes=(100,),
    )
    before = ledger.snapshot()["gauges"]

    route.record_verified(5)
    ledger.range_completed(999)

    snapshot = ledger.snapshot()
    assert snapshot["gauges"] == before
    assert snapshot["invariant_failures"] == 2
    assert snapshot["coverage"]["attribution"] == "incomplete"
    assert snapshot["by_phase"]["decode"]["invariant_failures"] == 1
    assert snapshot["by_phase"]["decode"]["coverage"]["attribution"] == "incomplete"
    route.close()


def test_non_strict_transition_underflow_is_atomic() -> None:
    ledger = ExpertPipelineLedger(strict=False, clock_ns=FakeClock())
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(5,),
        load_logical_bytes=(100,),
    )
    route.submission_attempted((5,))
    ledger._gauges["provisional_reader_tasks"] = 0
    ledger._phase_gauges["decode"]["provisional_reader_tasks"] = 0

    route.submission_accepted((5,))

    snapshot = ledger.snapshot()
    assert snapshot["gauges"]["provisional_submission_records"] == 1
    assert snapshot["gauges"]["accepted_unstarted_records"] == 0
    assert snapshot["gauges"]["queued_reader_tasks"] == 0
    assert snapshot["counters"]["accepted_record_jobs"] == 0
    assert snapshot["invariant_failures"] == 1


def test_non_strict_range_completion_underflow_is_atomic() -> None:
    ledger = ExpertPipelineLedger(strict=False, clock_ns=FakeClock())
    token = ledger.range_started(100, phase="decode")
    ledger._gauges["active_logical_range_bytes"] = 0
    ledger._phase_gauges["decode"]["active_logical_range_bytes"] = 0

    ledger.range_completed(token)

    snapshot = ledger.snapshot()
    assert snapshot["gauges"]["active_logical_ranges"] == 1
    assert snapshot["counters"]["completed_logical_ranges"] == 0
    assert snapshot["invariant_failures"] == 1


def test_pipeline_can_mark_fail_open_hook_coverage_incomplete() -> None:
    ledger = ExpertPipelineLedger(strict=False, clock_ns=FakeClock())

    ledger.mark_incomplete(phase="decode")

    snapshot = ledger.snapshot()
    assert snapshot["coverage"]["attribution"] == "incomplete"
    assert snapshot["counters"]["diagnostic_hook_failures"] == 1
    assert snapshot["by_phase"]["decode"]["counters"]["diagnostic_hook_failures"] == 1
