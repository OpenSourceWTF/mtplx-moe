from __future__ import annotations

import threading

import subprocess

import mtplx.benchmarks.resource_telemetry as telemetry_module
from mtplx.benchmarks.resource_telemetry import (
    PowermetricsCollector,
    ResourceTelemetrySampler,
    _pipeline_interval,
    parse_powermetrics_documents,
    summarize_intervals,
)


def _synthetic_intervals(
    *,
    ssd_gib_s: float,
    queued_fraction: float = 0.0,
    active_workers: float = 0.0,
    worker_capacity: int = 4,
    expert_misses: int = 8,
    io_active_fraction: float | None = None,
    fence_active_fraction: float = 0.0,
    both_fraction: float = 0.0,
    io_cache_mode: str = "f-nocache",
) -> list[dict[str, float | int | bool | str]]:
    interval_count = 10
    queued_count = round(interval_count * queued_fraction)
    io_count = round(
        interval_count
        * (queued_fraction if io_active_fraction is None else io_active_fraction)
    )
    fence_count = round(interval_count * fence_active_fraction)
    both_count = round(interval_count * both_fraction)
    intervals: list[dict[str, float | int | bool]] = []
    for index in range(interval_count):
        io_active = index < io_count
        fence_active = index < both_count or (
            io_count <= index < io_count + max(0, fence_count - both_count)
        )
        intervals.append(
            {
                "interval_seconds": 1.0,
                "reader_read_bytes": int(ssd_gib_s * 1024**3),
                "reader_read_operations": 1024,
                "io_cache_mode": io_cache_mode,
                "ssd_gib_per_second": ssd_gib_s,
                "expert_requests": 80,
                "expert_misses": expert_misses,
                "reader_queue_nonempty": index < queued_count,
                "mean_queued_reads": 1.0 if index < queued_count else 0.0,
                "mean_active_readers": active_workers,
                "reader_worker_capacity": worker_capacity,
                "io_active": io_active,
                "completion_fence_pending": fence_active,
            }
        )
    return intervals


_PRIMARY_STATES = (
    "generation_thread_expert_input_wait",
    "logical_range_active",
    "reader_completion_active",
    "submitted_queued",
    "eligible_unsubmitted",
    "host_runnable_work",
    "route_publication_pending",
    "no_known_useful_work",
)


def _pipeline_snapshot(
    *,
    observation_ns: int = 0,
    counters: dict[str, int] | None = None,
    integrals_ns: dict[str, int] | None = None,
    primary_integrals_ns: dict[str, int] | None = None,
    block_counts: dict[str, int] | None = None,
    block_ns: dict[str, int] | None = None,
    range_histogram: dict[str, object] | None = None,
    record_histogram: dict[str, object] | None = None,
    io: dict[str, int] | None = None,
    attribution: str = "measured",
) -> dict[str, object]:
    histogram = {
        "bounds_ns": (10, 20, 30),
        "bucket_counts": (0, 0, 0, 0),
        "sample_count": 0,
        "overflow_count": 0,
    }
    return {
        "io": {
            "read_operations": 0,
            "python_preadv_invocations": 0,
            "preadv_bytes_returned": 0,
            "native_positional_calls": 0,
            "native_bytes_returned": 0,
            **dict(io or {}),
        },
        "expert_pipeline": {
            "schema": "mtplx-expert-pipeline-attribution-v1",
            "by_phase": {
                "decode": {
                    "observation_ns": observation_ns,
                    "counters": dict(counters or {}),
                    "integrals_ns": dict(integrals_ns or {}),
                    "primary_integrals_ns": {
                        **dict.fromkeys(_PRIMARY_STATES, 0),
                        "no_known_useful_work": observation_ns,
                        **dict(primary_integrals_ns or {}),
                    },
                    "block_counts": dict(block_counts or {}),
                    "block_ns": dict(block_ns or {}),
                    "block_coverage": {
                        "operation_credit": "unavailable",
                        "byte_credit": "unavailable",
                        "authoritative_reserve": "unavailable",
                        "slot_unavailable": "unavailable",
                        "pin_held": "measured",
                        "slot_loading": "measured",
                    },
                    "histograms": {
                        "logical_range_latency_ns": range_histogram or histogram,
                        "complete_record_latency_ns": record_histogram or histogram,
                    },
                    "coverage": {"attribution": attribution},
                }
            },
        },
    }


def _pipeline_row(
    *,
    seconds: int,
    wait_seconds: int = 0,
    integrals_ns: dict[str, int] | None = None,
) -> dict[str, object]:
    span_ns = seconds * 1_000_000_000
    after = _pipeline_snapshot(
        observation_ns=span_ns,
        integrals_ns={
            "generation_expert_input_wait_ns": wait_seconds * 1_000_000_000,
            **dict(integrals_ns or {}),
        },
        primary_integrals_ns={
            "generation_thread_expert_input_wait": wait_seconds * 1_000_000_000,
            "no_known_useful_work": (seconds - wait_seconds) * 1_000_000_000,
        },
    )
    return {
        "interval_seconds": float(seconds),
        "expert_pipeline": _pipeline_interval(_pipeline_snapshot(), after, span_ns),
    }


def test_pipeline_summary_weights_duration_not_sample_count() -> None:
    report = summarize_intervals(
        [
            _pipeline_row(seconds=1, wait_seconds=1),
            _pipeline_row(seconds=9, wait_seconds=0),
        ],
        ssd_ceiling_gib_s=12.47,
        powermetrics=None,
    )

    pipeline = report["expert_pipeline"]
    assert pipeline["schema"] == "mtplx-expert-pipeline-summary-v1"
    assert pipeline["source_schema"] == "mtplx-expert-pipeline-attribution-v1"
    assert pipeline["decode_observation_ns"] == 10_000_000_000
    assert pipeline["generation_expert_input_wait_fraction"] == 0.1
    assert (
        pipeline["primary_state_ns"]["generation_thread_expert_input_wait"]
        == 1_000_000_000
    )
    assert pipeline["primary_state_fraction"] == {
        "generation_thread_expert_input_wait": 0.1,
        "logical_range_active": 0.0,
        "reader_completion_active": 0.0,
        "submitted_queued": 0.0,
        "eligible_unsubmitted": 0.0,
        "host_runnable_work": 0.0,
        "route_publication_pending": 0.0,
        "no_known_useful_work": 0.9,
    }


def test_pipeline_summary_preserves_distinct_wait_overlap_durations() -> None:
    report = summarize_intervals(
        [
            _pipeline_row(
                seconds=10,
                wait_seconds=5,
                integrals_ns={
                    "generation_wait_storage_active_ns": 1_000_000_000,
                    "generation_wait_reader_task_active_ns": 2_000_000_000,
                    "generation_wait_submitted_queued_ns": 3_000_000_000,
                    "generation_wait_eligible_unsubmitted_ns": 4_000_000_000,
                    "generation_wait_runnable_ns": 500_000_000,
                },
            )
        ],
        ssd_ceiling_gib_s=None,
        powermetrics=None,
    )

    overlap = report["expert_pipeline"]["orthogonal_overlap"]
    assert overlap["denominator"] == "decode_observation_ns"
    assert overlap["duration_ns"] == {
        "generation_wait_storage_active": 1_000_000_000,
        "generation_wait_reader_task_active": 2_000_000_000,
        "generation_wait_submitted_queued": 3_000_000_000,
        "generation_wait_eligible_unsubmitted": 4_000_000_000,
        "generation_wait_runnable": 500_000_000,
    }
    assert overlap["fraction_of_decode_observation"] == {
        "generation_wait_storage_active": 0.1,
        "generation_wait_reader_task_active": 0.2,
        "generation_wait_submitted_queued": 0.3,
        "generation_wait_eligible_unsubmitted": 0.4,
        "generation_wait_runnable": 0.05,
    }
    assert (
        report["expert_pipeline"]["decode_integrals_ns"][
            "generation_wait_submitted_queued_ns"
        ]
        == 3_000_000_000
    )


def test_pipeline_summary_keeps_records_tasks_ranges_and_backend_calls_distinct() -> (
    None
):
    before = _pipeline_snapshot()
    after = _pipeline_snapshot(
        observation_ns=1_000,
        counters={
            "logical_record_jobs": 11,
            "logical_record_bytes": 1_100,
            "accepted_submissions": 7,
            "accepted_record_jobs": 9,
            "accepted_record_bytes": 900,
            "started_reader_tasks": 6,
            "completed_reader_tasks": 5,
            "failed_reader_tasks": 1,
            "started_logical_ranges": 13,
        },
        io={
            "read_operations": 17,
            "python_preadv_invocations": 23,
            "preadv_bytes_returned": 2_300,
            "native_positional_calls": 5,
            "native_bytes_returned": 500,
        },
    )
    report = summarize_intervals(
        [
            {
                "interval_seconds": 1e-6,
                "expert_pipeline": _pipeline_interval(before, after, 1_000),
            }
        ],
        ssd_ceiling_gib_s=None,
        powermetrics=None,
    )["expert_pipeline"]

    assert report["scope"] == "decode"
    assert report["logical_record_jobs"] == 11
    assert report["logical_record_bytes"] == 1_100
    assert report["accepted_executor_submissions"] == 7
    assert report["reader_tasks_started"] == 6
    assert report["reader_tasks_completed"] == 5
    assert report["reader_tasks_failed"] == 1
    assert report["submission_accepted_record_jobs"] == 9
    assert report["submission_accepted_record_bytes"] == 900
    assert report["decode_logical_ranges_started"] == 13
    assert report["sampler_window_backend"] == {
        "scope": "sampler_window_all_phases",
        "logical_range_reader_invocations": 17,
        "python_preadv_invocations": 23,
        "preadv_bytes_returned": 2_300,
        "native_positional_calls": 5,
        "native_bytes_returned": 500,
    }


def test_pipeline_summary_marks_unobservable_device_and_admission_facts_unavailable() -> (
    None
):
    report = summarize_intervals(
        [_pipeline_row(seconds=1)],
        ssd_ceiling_gib_s=None,
        powermetrics=None,
    )["expert_pipeline"]

    assert report["block_reasons"]["operation_credit"]["status"] == "unavailable"
    assert report["block_reasons"]["byte_credit"]["status"] == "unavailable"
    assert report["block_reasons"]["authoritative_reserve"]["status"] == "unavailable"
    assert report["physical_device_queue_depth"] == {"status": "unavailable"}
    assert report["gpu_expert_wait"] == {"status": "unavailable"}
    assert report["coverage"]["operation_credit"] == "unavailable"
    assert report["coverage"]["byte_credit"] == "unavailable"
    assert report["coverage"]["authoritative_reserve"] == "unavailable"
    assert report["coverage"]["physical_device_queue_depth"] == "unavailable"
    assert report["coverage"]["gpu_expert_wait"] == "unavailable"
    assert report["coverage"]["gpu_idle_time"] == "unavailable"
    assert report["coverage"]["outer_split_executor_queue"] == "unavailable"
    assert report["coverage"]["eligible_unsubmitted_cause"] == "unattributed"
    assert report["coverage"]["admitted_read_ranges"] == "unavailable"
    assert report["coverage"]["scheduled_read_ranges"] == "unavailable"


def test_pipeline_histogram_deltas_report_percentile_bucket_upper_bounds() -> None:
    before_histogram = {
        "bounds_ns": (10, 20, 30),
        "bucket_counts": (2, 3, 4, 0),
        "sample_count": 9,
        "overflow_count": 0,
    }
    after_histogram = {
        "bounds_ns": (10, 20, 30),
        "bucket_counts": (12, 11, 5, 1),
        "sample_count": 29,
        "overflow_count": 1,
    }
    interval = _pipeline_interval(
        _pipeline_snapshot(
            range_histogram=before_histogram,
            record_histogram=before_histogram,
        ),
        _pipeline_snapshot(
            observation_ns=1_000,
            range_histogram=after_histogram,
            record_histogram=after_histogram,
        ),
        1_000,
    )
    report = summarize_intervals(
        [{"interval_seconds": 1e-6, "expert_pipeline": interval}],
        ssd_ceiling_gib_s=None,
        powermetrics=None,
    )["expert_pipeline"]

    for histogram in report["latency_histograms"].values():
        assert histogram["sample_count"] == 20
        assert histogram["overflow_count"] == 1
        assert histogram["p50_upper_bound_ns"] == 10
        assert histogram["p95_upper_bound_ns"] == 30


def test_pipeline_histogram_overflow_censors_only_percentile_in_overflow() -> None:
    histogram = {
        "bounds_ns": (10, 20, 30),
        "bucket_counts": (10, 0, 0, 10),
        "sample_count": 20,
        "overflow_count": 10,
    }
    interval = _pipeline_interval(
        _pipeline_snapshot(),
        _pipeline_snapshot(
            observation_ns=1_000,
            range_histogram=histogram,
            record_histogram=histogram,
        ),
        1_000,
    )
    report = summarize_intervals(
        [{"interval_seconds": 1e-6, "expert_pipeline": interval}],
        ssd_ceiling_gib_s=None,
        powermetrics=None,
    )["expert_pipeline"]

    assert report["coverage"]["attribution"] == "measured"
    for result in report["latency_histograms"].values():
        assert result["sample_count"] == 20
        assert result["p50_upper_bound_ns"] == 10
        assert result["p50_status"] == "bounded"
        assert result["p95_upper_bound_ns"] is None
        assert result["p95_status"] == "censored_overflow"


def test_pipeline_histogram_contract_change_marks_interval_incomplete() -> None:
    before_histogram = {
        "bounds_ns": (10, 20),
        "bucket_counts": (1, 0, 0),
        "sample_count": 1,
        "overflow_count": 0,
    }
    after_histogram = {
        "bounds_ns": (10, 30),
        "bucket_counts": (2, 0, 0),
        "sample_count": 2,
        "overflow_count": 0,
    }

    interval = _pipeline_interval(
        _pipeline_snapshot(
            range_histogram=before_histogram,
            record_histogram=before_histogram,
        ),
        _pipeline_snapshot(
            observation_ns=1_000,
            range_histogram=after_histogram,
            record_histogram=after_histogram,
        ),
        1_000,
    )

    assert interval["coverage"]["attribution"] == "incomplete"


def test_pipeline_histogram_rejects_matching_but_invalid_source_totals() -> None:
    before_histogram = {
        "bounds_ns": (10, 20, 30),
        "bucket_counts": (1, 0, 0, 0),
        "sample_count": 2,
        "overflow_count": 0,
    }
    after_histogram = {
        "bounds_ns": (10, 20, 30),
        "bucket_counts": (2, 0, 0, 0),
        "sample_count": 3,
        "overflow_count": 0,
    }

    interval = _pipeline_interval(
        _pipeline_snapshot(
            range_histogram=before_histogram,
            record_histogram=before_histogram,
        ),
        _pipeline_snapshot(
            observation_ns=1_000,
            range_histogram=after_histogram,
            record_histogram=after_histogram,
        ),
        1_000,
    )

    assert interval["coverage"]["attribution"] == "incomplete"


def test_pipeline_source_phase_incomplete_propagates_to_summary() -> None:
    interval = _pipeline_interval(
        _pipeline_snapshot(),
        _pipeline_snapshot(observation_ns=1_000, attribution="incomplete"),
        1_000,
    )

    report = summarize_intervals(
        [{"interval_seconds": 1e-6, "expert_pipeline": interval}],
        ssd_ceiling_gib_s=None,
        powermetrics=None,
    )["expert_pipeline"]

    assert interval["coverage"]["attribution"] == "incomplete"
    assert report["coverage"]["attribution"] == "incomplete"
    assert report["coverage"]["decode_phase"] == "incomplete"


def test_pipeline_primary_integral_mismatch_marks_summary_incomplete() -> None:
    interval = _pipeline_interval(
        _pipeline_snapshot(),
        _pipeline_snapshot(
            observation_ns=1_000,
            primary_integrals_ns={"no_known_useful_work": 900},
        ),
        1_000,
    )

    report = summarize_intervals(
        [{"interval_seconds": 1e-6, "expert_pipeline": interval}],
        ssd_ceiling_gib_s=None,
        powermetrics=None,
    )["expert_pipeline"]

    assert report["decode_observation_ns"] == 1_000
    assert report["primary_state_ns"]["no_known_useful_work"] == 900
    assert report["coverage"]["attribution"] == "incomplete"
    assert report["coverage"]["decode_phase"] == "incomplete"


def test_pipeline_primary_integral_mismatches_cannot_cancel_between_intervals() -> None:
    intervals = []
    for primary_ns in (900, 1_100):
        intervals.append(
            {
                "interval_seconds": 1e-6,
                "expert_pipeline": _pipeline_interval(
                    _pipeline_snapshot(),
                    _pipeline_snapshot(
                        observation_ns=1_000,
                        primary_integrals_ns={"no_known_useful_work": primary_ns},
                    ),
                    1_000,
                ),
            }
        )

    report = summarize_intervals(
        intervals,
        ssd_ceiling_gib_s=None,
        powermetrics=None,
    )["expert_pipeline"]

    assert report["decode_observation_ns"] == 2_000
    assert report["primary_state_ns"]["no_known_useful_work"] == 2_000
    assert report["coverage"]["attribution"] == "incomplete"
    assert report["coverage"]["decode_phase"] == "incomplete"


def test_pipeline_missing_backend_counters_are_unavailable_not_zero_measured() -> None:
    before = _pipeline_snapshot()
    after = _pipeline_snapshot(observation_ns=1_000)
    before.pop("io")
    after.pop("io")

    interval = _pipeline_interval(before, after, 1_000)
    report = summarize_intervals(
        [{"interval_seconds": 1e-6, "expert_pipeline": interval}],
        ssd_ceiling_gib_s=None,
        powermetrics=None,
    )["expert_pipeline"]

    assert interval["coverage"]["sampler_window_backend"] == "unavailable"
    assert report["coverage"]["sampler_window_backend"] == "unavailable"
    assert report["coverage"]["attribution"] == "incomplete"
    assert report["sampler_window_backend"]["logical_range_reader_invocations"] is None


def test_pipeline_backend_counter_reset_is_not_reported_as_measured() -> None:
    interval = _pipeline_interval(
        _pipeline_snapshot(io={"read_operations": 10}),
        _pipeline_snapshot(observation_ns=1_000, io={"read_operations": 2}),
        1_000,
    )
    report = summarize_intervals(
        [{"interval_seconds": 1e-6, "expert_pipeline": interval}],
        ssd_ceiling_gib_s=None,
        powermetrics=None,
    )["expert_pipeline"]

    assert interval["coverage"]["sampler_window_backend"] == "incomplete_reset"
    assert report["coverage"]["sampler_window_backend"] == "incomplete_reset"
    assert report["coverage"]["attribution"] == "incomplete"


def test_pipeline_missing_required_histograms_marks_decode_incomplete() -> None:
    before = _pipeline_snapshot()
    after = _pipeline_snapshot(observation_ns=1_000)
    before["expert_pipeline"]["by_phase"]["decode"].pop("histograms")
    after["expert_pipeline"]["by_phase"]["decode"].pop("histograms")

    interval = _pipeline_interval(before, after, 1_000)
    report = summarize_intervals(
        [{"interval_seconds": 1e-6, "expert_pipeline": interval}],
        ssd_ceiling_gib_s=None,
        powermetrics=None,
    )["expert_pipeline"]

    assert interval["coverage"]["decode_phase"] == "incomplete"
    assert report["coverage"]["decode_phase"] == "incomplete"
    assert report["latency_histograms"] == {}


def test_unavailable_pipeline_summary_preserves_full_coverage_contract() -> None:
    report = summarize_intervals([], ssd_ceiling_gib_s=None, powermetrics=None)[
        "expert_pipeline"
    ]

    assert report["coverage"] == {
        "attribution": "unavailable",
        "decode_phase": "unavailable",
        "sampler_window_backend": "unavailable",
        "operation_credit": "unavailable",
        "byte_credit": "unavailable",
        "authoritative_reserve": "unavailable",
        "slot_capacity_admission": "unavailable",
        "outer_split_executor_queue": "unavailable",
        "eligible_unsubmitted_cause": "unattributed",
        "admitted_read_ranges": "unavailable",
        "scheduled_read_ranges": "unavailable",
        "physical_device_operations": "unavailable",
        "physical_device_bytes": "unavailable",
        "physical_device_queue_depth": "unavailable",
        "gpu_expert_wait": "unavailable",
        "gpu_idle_time": "unavailable",
        "future_layer_eligibility": "unavailable",
        "speculative_record_accounting": "unavailable",
        "python_preadv_when_native_reader": "unavailable",
    }


def test_pipeline_counter_reset_marks_interval_incomplete_without_negative_delta() -> (
    None
):
    before = _pipeline_snapshot(
        observation_ns=10_000,
        counters={"logical_record_jobs": 10},
    )
    after = _pipeline_snapshot(
        observation_ns=1_000,
        counters={"logical_record_jobs": 2},
    )

    interval = _pipeline_interval(before, after, 1_000)

    assert interval["counters"]["logical_record_jobs"] == 0
    assert interval["decode_observation_ns"] == 0
    assert interval["coverage"]["attribution"] == "incomplete"
    assert interval["coverage"]["decode_phase"] == "incomplete"
    assert interval["coverage"]["sampler_window_backend"] == ("measured_all_phases")
    assert interval["coverage"]["reset_detected"] is True
    report = summarize_intervals(
        [{"interval_seconds": 1e-6, "expert_pipeline": interval}],
        ssd_ceiling_gib_s=None,
        powermetrics=None,
    )["expert_pipeline"]
    assert report["generation_expert_input_wait_fraction"] is None
    assert report["coverage"]["attribution"] == "incomplete"


def test_backed_up_readers_below_ssd_ceiling_are_not_called_storage_bound() -> None:
    report = summarize_intervals(
        _synthetic_intervals(
            ssd_gib_s=5.0,
            queued_fraction=0.9,
            active_workers=4.0,
            worker_capacity=4,
        ),
        ssd_ceiling_gib_s=12.5,
        powermetrics=None,
    )

    assert report["evidence"]["ssd_saturation"]["status"] == "not_supported"
    assert report["evidence"]["reader_backpressure"]["status"] == "present"
    assert report["attribution"]["status"] == "incomplete"
    assert "reader_pool_or_read_shape" in report["attribution"]["candidates"]
    assert "bound_by" not in report


def test_storage_pressure_screen_routes_a_candidate_without_claiming_causality() -> (
    None
):
    report = summarize_intervals(
        _synthetic_intervals(
            ssd_gib_s=11.8,
            queued_fraction=0.8,
            active_workers=3.8,
            worker_capacity=4,
        ),
        ssd_ceiling_gib_s=12.5,
        powermetrics=None,
    )

    assert report["evidence"]["ssd_saturation"]["status"] == "supported"
    assert report["attribution"] == {
        "status": "incomplete",
        "candidates": ["storage_throughput"],
    }


def test_gpu_activity_screen_routes_a_candidate_without_claiming_causality() -> None:
    report = summarize_intervals(
        _synthetic_intervals(ssd_gib_s=2.0),
        ssd_ceiling_gib_s=12.5,
        powermetrics={
            "available": True,
            "process_gpu_busy_fraction": 0.9,
        },
    )

    assert report["evidence"]["gpu_activity"]["status"] == "supported"
    assert report["attribution"]["status"] == "incomplete"
    assert "gpu_compute" in report["attribution"]["candidates"]


def test_cached_reader_bytes_cannot_establish_ssd_saturation() -> None:
    report = summarize_intervals(
        _synthetic_intervals(
            ssd_gib_s=11.8,
            queued_fraction=0.8,
            active_workers=3.8,
            worker_capacity=4,
            io_cache_mode="buffered",
        ),
        ssd_ceiling_gib_s=12.5,
        powermetrics=None,
    )

    assert report["coverage"]["storage_reads"] == "logical_reader_bytes"
    assert report["evidence"]["ssd_saturation"]["status"] == "unavailable"
    assert report["storage"]["utilization_of_ceiling"] is None
    assert report["attribution"]["status"] == "incomplete"


def test_missing_powermetrics_is_coverage_not_zero_gpu_usage() -> None:
    report = summarize_intervals(
        _synthetic_intervals(ssd_gib_s=2.0),
        ssd_ceiling_gib_s=12.5,
        powermetrics={"available": False, "reason": "sudo requires a password"},
    )

    assert report["coverage"]["gpu"] == "unavailable"
    assert report["evidence"]["gpu_activity"]["status"] == "unavailable"
    assert report["coverage"]["dram_bandwidth"] == "unavailable"
    assert report["coverage"]["gpu_reason"] == "sudo requires a password"


def test_io_fence_coactivity_is_labeled_coarse_not_simultaneous() -> None:
    report = summarize_intervals(
        _synthetic_intervals(
            ssd_gib_s=3.0,
            io_active_fraction=0.5,
            fence_active_fraction=0.5,
            both_fraction=0.0,
        ),
        ssd_ceiling_gib_s=12.5,
        powermetrics=None,
    )

    assert report["overlap"]["measurement"] == "same_interval_coactivity"
    assert report["overlap"]["simultaneous_overlap_measured"] is False
    assert report["overlap"]["io_activity_interval_fraction"] == 0.5
    assert report["overlap"]["fence_activity_interval_fraction"] == 0.5
    assert report["overlap"]["same_interval_activity_fraction"] == 0.0
    assert "both_fraction" not in report["overlap"]
    assert "coarse_io_fence_separation" in report["attribution"]["candidates"]


def test_completion_backlog_summary_keeps_work_and_slot_occupancy() -> None:
    intervals = _synthetic_intervals(ssd_gib_s=1.0)
    for interval in intervals:
        interval["completion_tokens"] = 8
        interval["synchronous_fences"] = 10
        interval["synchronous_fence_slots"] = 80
        interval["completion_fence_registrations"] = 2
        interval["completion_fences"] = {
            "mean_queued_work": 0.25,
            "mean_active_work": 0.75,
            "mean_queued_units": 2.0,
            "mean_active_units": 6.0,
            "rejected_submissions": 0,
            "lifetime_queued_work_peak": 2,
            "lifetime_active_work_peak": 1,
            "lifetime_queued_units_peak": 16,
            "lifetime_active_units_peak": 8,
        }

    report = summarize_intervals(
        intervals,
        ssd_ceiling_gib_s=12.5,
        powermetrics=None,
    )

    assert report["completion_fences"]["mean_queued_fences"] == 0.25
    assert report["completion_fences"]["mean_active_fences"] == 0.75
    assert report["completion_fences"]["mean_queued_slots"] == 2.0
    assert report["completion_fences"]["mean_active_slots"] == 6.0
    assert report["completion_fences"]["lifetime_queued_slots_peak"] == 16
    assert report["completion_fences"]["synchronous_fences"] == 100
    assert report["completion_fences"]["synchronous_fence_slots"] == 800
    assert report["completion_fences"]["synchronous_fences_per_token"] == 1.25
    assert "synchronous_fence_or_evaluation" in report["attribution"]["candidates"]


def test_powermetrics_auth_timeout_degrades_without_prompting(
    monkeypatch,
) -> None:
    monkeypatch.setattr(telemetry_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        telemetry_module.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("sudo", 2.0)

    monkeypatch.setattr(telemetry_module.subprocess, "run", timeout)
    collector = PowermetricsCollector(enabled=True, pid=42)

    collector.start()

    assert collector.report()["available"] is False
    assert "TimeoutExpired" in collector.report()["reason"]


def test_powermetrics_plist_extracts_only_the_benchmark_pid() -> None:
    documents = [
        {
            "timestamp_ns": 1_000,
            "tasks": [
                {"pid": 41, "gpu_time_ns": 10, "cpu_ms_per_s": 20},
                {
                    "pid": 42,
                    "gpu_time_ns": 600,
                    "cpu_ms_per_s": 850,
                    "disk_read_bytes": 4096,
                    "sfi_wait_time_ns": 50,
                },
            ],
        }
    ]

    samples = parse_powermetrics_documents(documents, pid=42)

    assert samples == [
        {
            "timestamp_ns": 1_000,
            "process_gpu_time_ns": 600,
            "process_cpu_ms_per_s": 850.0,
            "process_disk_read_bytes": 4096,
            "process_wait_time_ns": 50,
        }
    ]


def test_powermetrics_parser_accepts_native_macos_plist_keys() -> None:
    samples = parse_powermetrics_documents(
        [
            {
                "timestamp": 1_750_000_000.0,
                "tasks": [
                    {
                        "pid": 42,
                        "gputime_ns": 700,
                        "gputime_ms_per_s": 625.0,
                        "cputime_sample_ms_per_s": 920.0,
                        "diskio_bytesread": 8192,
                        "sfi_ns": 75,
                    }
                ],
            }
        ],
        pid=42,
    )

    assert samples == [
        {
            "timestamp_ns": 1_750_000_000_000_000_000,
            "process_gpu_time_ns": 700,
            "process_gpu_ms_per_s": 625.0,
            "process_cpu_ms_per_s": 920.0,
            "process_disk_read_bytes": 8192,
            "process_wait_time_ns": 75,
        }
    ]


def test_sampler_differences_counters_and_occupancy_on_one_clock() -> None:
    clock_values = iter((1_000_000_000, 2_000_000_000))
    snapshots = iter(
        (
            _snapshot(read_bytes=0, read_ops=0, requests=0, queue_ns=0, active_ns=0),
            _snapshot(
                read_bytes=2 * 1024**3,
                read_ops=200,
                requests=80,
                queue_ns=500_000_000,
                active_ns=2_000_000_000,
            ),
        )
    )
    sampler = ResourceTelemetrySampler(
        lambda: next(snapshots),
        token_count=lambda: 0,
        interval_s=1.0,
        max_samples=2,
        clock_ns=lambda: next(clock_values),
    )

    sampler.capture()
    sampler.capture()
    report = sampler.report(ssd_ceiling_gib_s=12.5)

    interval = report["timeline"][0]
    assert interval["reader_gib_per_second"] == 2.0
    assert interval["reader_read_operations"] == 200
    assert interval["mean_queued_reads"] == 0.5
    assert interval["mean_active_readers"] == 2.0
    assert interval["q4_assignments_per_second"] == 80.0
    assert interval["io_and_completion_fence_seen_in_interval"] is True
    assert "io_and_completion_fence_active" not in interval


def test_bounded_sampler_preserves_run_start_for_cumulative_summary() -> None:
    clock_values = iter((1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000))
    snapshots = iter(
        _snapshot(
            read_bytes=index * 1024**3,
            read_ops=index * 100,
            requests=index * 80,
            queue_ns=0,
            active_ns=index * 1_000_000_000,
        )
        for index in range(4)
    )
    sampler = ResourceTelemetrySampler(
        lambda: next(snapshots),
        token_count=lambda: 0,
        interval_s=1.0,
        max_samples=3,
        clock_ns=lambda: next(clock_values),
    )

    for _index in range(4):
        sampler.capture()
    report = sampler.report(ssd_ceiling_gib_s=12.5)

    assert report["sample_count"] == 3
    assert report["samples_dropped"] == 1
    assert report["elapsed_seconds"] == 3.0
    assert report["storage"]["reader_read_bytes"] == 3 * 1024**3
    assert report["coverage"]["timeline"] == "retained_start_and_recent_tail"
    assert report["attribution"]["status"] == "incomplete"
    assert "increase_resource_max_samples" in report["attribution"]["candidates"]


def test_persistent_periodic_sampler_failure_marks_evidence_incomplete() -> None:
    calls = 0
    failed = threading.Event()

    def snapshot() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls >= 2:
            failed.set()
            raise RuntimeError("injected persistent snapshot failure")
        return _snapshot(
            read_bytes=0,
            read_ops=0,
            requests=0,
            queue_ns=0,
            active_ns=0,
        )

    sampler = ResourceTelemetrySampler(
        snapshot,
        token_count=lambda: 0,
        interval_s=0.001,
        max_samples=8,
    )
    with sampler:
        assert failed.wait(timeout=1.0)

    report = sampler.report(ssd_ceiling_gib_s=12.5)
    assert report["sampling_failures"] == 1
    assert report["coverage"]["timeline"] == "incomplete_sampler_failure"
    assert report["attribution"]["status"] == "incomplete"
    assert "resource_sampler_failure" in report["attribution"]["candidates"]


def _snapshot(
    *,
    read_bytes: int,
    read_ops: int,
    requests: int,
    queue_ns: int,
    active_ns: int,
) -> dict[str, object]:
    pool = {
        "worker_capacity": 4,
        "accepted_submissions": read_ops,
        "started": read_ops,
        "completed": read_ops,
        "rejected_submissions": 0,
        "queued_work": 0,
        "active_work": 0,
        "queued_units": 0,
        "active_units": 0,
        "queued_work_peak": 1,
        "active_work_peak": 4,
        "queued_units_peak": 1024,
        "active_units_peak": 4096,
        "queued_work_ns": queue_ns,
        "active_work_ns": active_ns,
        "queued_unit_ns": queue_ns * 100,
        "active_unit_ns": active_ns * 100,
    }
    idle_pool = {**pool, "worker_capacity": 1}
    return {
        "model_key": "hy3-q4",
        "quant_bits": 4,
        "expert_record_bytes": 100,
        "io_cache_mode": "f-nocache",
        "cache": {
            "route_calls": requests // 8,
            "expert_requests": requests,
            "expert_hits": max(0, requests - 8),
            "expert_misses": min(8, requests),
            "persistent_loads": 0,
            "transient_loads": 0,
            "evictions": 0,
            "bytes_read": read_bytes,
        },
        "cache_by_layer": {},
        "io": {"read_bytes": read_bytes, "read_operations": read_ops},
        "reader_pool": pool,
        "completion_fences": idle_pool,
        "mlx_memory": {},
    }
