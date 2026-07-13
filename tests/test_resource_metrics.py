from __future__ import annotations

import pytest

from mtplx.resource_metrics import PoolOccupancy


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
