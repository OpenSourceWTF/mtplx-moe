from __future__ import annotations

import subprocess

import mtplx.benchmarks.resource_telemetry as telemetry_module
from mtplx.benchmarks.resource_telemetry import (
    PowermetricsCollector,
    ResourceTelemetrySampler,
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


def test_storage_evidence_requires_ceiling_queue_and_worker_pressure() -> None:
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
        "status": "conclusive",
        "candidates": ["storage_throughput"],
    }


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


def test_low_overlap_is_reported_as_evidence_not_a_bound_label() -> None:
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

    assert report["overlap"]["io_active_fraction"] == 0.5
    assert report["overlap"]["completion_fence_pending_fraction"] == 0.5
    assert report["overlap"]["both_fraction"] == 0.0
    assert (
        "synchronization_or_insufficient_overlap" in report["attribution"]["candidates"]
    )


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
