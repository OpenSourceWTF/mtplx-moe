"""Pin the saturation-lane (--concurrency) harness flags and request builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mtplx.sampling import SamplerConfig

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "benchmark_streamed_generation.py"
)

_BASE_ARGS = [
    "/model",
    "/manifest",
    "--model-key",
    "hy3-q4",
    "--memory-limit",
    "112GiB",
    "--max-live-kv-tokens",
    "2048",
]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_streamed_generation", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unflagged_runs_use_the_comparable_continuous_batch_lane() -> None:
    parser = _load_module().build_parser()
    args = parser.parse_args(_BASE_ARGS)
    assert args.concurrency == 1
    assert args.max_prefills_per_step == 1
    assert args.reference_ar is False
    assert _load_module().resolve_execution_lane(args) == "continuous-batch-ar"


def test_legacy_single_stream_ar_requires_an_explicit_reference_flag() -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args([*_BASE_ARGS, "--reference-ar"])

    module.validate_reference_ar_flags(parser, args)

    assert args.reference_ar is True
    assert module.resolve_execution_lane(args) == "reference-ar"


def test_reference_ar_rejects_concurrent_saturation_lanes(capsys) -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args([*_BASE_ARGS, "--reference-ar", "--concurrency", "2"])

    with pytest.raises(SystemExit):
        module.validate_reference_ar_flags(parser, args)
    assert "--reference-ar requires --concurrency 1" in capsys.readouterr().err


def test_mtp_stays_on_a_distinct_reference_lane() -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [*_BASE_ARGS, "--enable-mtp", "--mtp-artifacts", "/artifacts"]
    )

    module.validate_mtp_flags(parser, args)
    module.validate_reference_ar_flags(parser, args)

    assert module.resolve_execution_lane(args) == "reference-mtp1"


def test_reference_ar_cannot_be_combined_with_mtp(capsys) -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            *_BASE_ARGS,
            "--reference-ar",
            "--enable-mtp",
            "--mtp-artifacts",
            "/artifacts",
        ]
    )
    module.validate_mtp_flags(parser, args)

    with pytest.raises(SystemExit):
        module.validate_reference_ar_flags(parser, args)
    assert "cannot be combined with --enable-mtp" in capsys.readouterr().err


def test_saturation_lane_flags_parse() -> None:
    parser = _load_module().build_parser()
    args = parser.parse_args(
        [*_BASE_ARGS, "--concurrency", "4", "--max-prefills-per-step", "2"]
    )
    assert args.concurrency == 4
    assert args.max_prefills_per_step == 2


def test_mixed_join_workload_flags_parse_and_validate() -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            *_BASE_ARGS,
            "--concurrency",
            "4",
            "--workload-shape",
            "mixed-join",
            "--join-after-step",
            "3",
        ]
    )

    module.validate_workload_flags(parser, args)

    assert args.workload_shape == "mixed-join"
    assert args.join_after_step == 3


def test_mixed_join_requires_room_for_a_live_decoder_and_joiner(capsys) -> None:
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [*_BASE_ARGS, "--workload-shape", "mixed-join", "--concurrency", "1"]
    )

    with pytest.raises(SystemExit):
        module.validate_workload_flags(parser, args)
    assert "mixed-join requires --concurrency at least 2" in capsys.readouterr().err


@pytest.mark.parametrize("cache_scope", ["layer", "global"])
@pytest.mark.parametrize("concurrency", [1, 2, 4, 8])
def test_run_label_distinguishes_cache_arm_and_saturation_lane(
    cache_scope: str,
    concurrency: int,
) -> None:
    module = _load_module()

    summary = module.build_configuration_summary(
        "fixture",
        cache_scope=cache_scope,
        slot_layout="component-banks",
        concurrency=concurrency,
        execution_lane="continuous-batch-ar",
        performance_settings={"cache_policy": "frequency"},
    )

    assert summary["run_label"] == "fixture"
    assert summary["configuration_label"].startswith(
        f"cache-{cache_scope}-layout-component-banks-B{concurrency}"
        "-lane-continuous-batch-ar-cfg-"
    )
    assert summary["cache_scope"] == cache_scope
    assert summary["slot_layout"] == "component-banks"
    assert summary["concurrency"] == concurrency
    assert summary["execution_lane"] == "continuous-batch-ar"


def test_reference_ar_has_a_distinct_configuration_identity() -> None:
    module = _load_module()
    common = {
        "cache_scope": "global",
        "slot_layout": "component-banks",
        "concurrency": 1,
        "performance_settings": {"cache_policy": "lru"},
    }

    continuous = module.build_configuration_summary(
        "fixture", execution_lane="continuous-batch-ar", **common
    )
    reference = module.build_configuration_summary(
        "fixture", execution_lane="reference-ar", **common
    )

    assert (
        continuous["configuration_fingerprint"]
        != (reference["configuration_fingerprint"])
    )
    assert "-lane-reference-ar-cfg-" in reference["configuration_label"]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_concurrency_is_rejected(value: str) -> None:
    parser = _load_module().build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*_BASE_ARGS, "--concurrency", value])


def test_run_concurrent_repeats_reports_aggregate_and_per_stream_rates(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from test_streamed_batch import _open_fixture_runtime

    module = _load_module()
    rt, _streaming = _open_fixture_runtime(
        tmp_path,
        max_live_kv_tokens=64,
        resource_telemetry=True,
    )
    try:
        args = SimpleNamespace(
            repeats=1,
            reset_between=False,
            concurrency=2,
            max_prefills_per_step=1,
            seed=0,
            output_dir=None,
            model_key="hy3-q4",
            cache_scope="global",
            slot_layout="component-banks",
            resource_telemetry=True,
            resource_sample_interval=0.01,
            resource_max_samples=128,
            ssd_ceiling_gib_s=12.5,
            powermetrics=False,
        )
        configuration_label = "cache-global-layout-component-banks-B2-cfg-fixture"
        rows = module._run_concurrent_repeats(
            args,
            rt,
            prompt_ids=[1, 2, 3],
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=1),
            max_tokens=4,
            run_label="fixture",
            configuration_label=configuration_label,
        )

        assert len(rows) == 1
        row = rows[0]
        assert row["run_label"] == "fixture"
        assert row["configuration_label"] == configuration_label
        assert row["concurrency"] == 2
        assert row["requested_concurrency"] == 2
        assert row["achieved_peak_concurrency"] == 2
        assert row["saturation_valid"] is True
        assert row["undersubscribed"] is False
        assert row["cache_scope"] == "global"
        assert row["slot_layout"] == "component-banks"
        assert row["execution_lane"] == "continuous-batch-ar"
        assert len(row["streams"]) == 2
        assert row["aggregate_completion_tokens"] == 8
        assert row["aggregate_completion_tokens_per_second"] > 0.0
        assert row["scheduler"]["decode_steps"] == 3
        for stream in row["streams"]:
            assert stream["completion_tokens"] == 4
            assert stream["finish_reason"] == "length"
            assert stream["completion_tokens_per_second"] > 0.0
            assert stream["decode_tokens_per_second"] > 0.0
            assert stream["ttft_seconds"] > 0.0
            assert stream["completion_latency_seconds"] >= stream["ttft_seconds"]
            assert len(stream["token_times_s"]) == 4
            assert len(stream["token_ids"]) == 4
        assert row["timing_summary"]["stream_count"] == 2
        assert row["timing_summary"]["requested_concurrency"] == 2
        assert row["timing_summary"]["achieved_peak_concurrency"] == 2
        assert row["timing_summary"]["undersubscribed"] is False
        assert row["timing_summary"]["ttft_seconds"]["p50"] > 0.0
        assert row["timing_summary"]["completion_latency_seconds"]["p99"] > 0.0
        assert row["streaming_after"]["live_kv_tokens"] == 0
        assert row["diagnostic_run"] is True
        assert row["resource_telemetry"]["schema"] == "mtplx-resource-telemetry-v1"
        assert row["resource_telemetry"]["sample_count"] >= 2
    finally:
        rt.close()


def test_mixed_join_lane_submits_prefill_while_first_stream_decodes(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from test_streamed_batch import _open_fixture_runtime

    module = _load_module()
    rt, _streaming = _open_fixture_runtime(
        tmp_path,
        max_live_kv_tokens=64,
        resource_telemetry=True,
    )
    try:
        args = SimpleNamespace(
            repeats=1,
            reset_between=False,
            concurrency=2,
            max_prefills_per_step=1,
            workload_shape="mixed-join",
            join_after_step=1,
            seed=0,
            output_dir=None,
            model_key="hy3-q4",
            cache_scope="global",
            slot_layout="component-banks",
            resource_telemetry=True,
            resource_sample_interval=0.01,
            resource_max_samples=128,
            ssd_ceiling_gib_s=None,
            powermetrics=False,
        )
        row = module._run_concurrent_repeats(
            args,
            rt,
            prompt_ids=[1, 2, 3],
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=1),
            max_tokens=5,
            run_label="fixture",
            configuration_label="mixed-B2",
        )[0]

        streams = {stream["request_id"]: stream for stream in row["streams"]}
        assert row["workload_shape"] == "mixed-join"
        assert row["join_submission_step"] == 1
        assert streams["stream-00"]["admitted_step"] == 0
        assert streams["stream-01"]["admitted_step"] == 2
        assert row["scheduler"]["live_stream_counts"][0:2] == [1, 1]
        assert 2 in row["scheduler"]["live_stream_counts"]
        assert row["achieved_peak_concurrency"] == 2
        assert row["diagnostic_run"] is True
        assert row["resource_telemetry"]["throughput"][
            "final_completion_tokens"
        ] == sum(stream["completion_tokens"] for stream in row["streams"])
    finally:
        rt.close()


@pytest.mark.parametrize("concurrency", [1, 2, 4, 8])
def test_requested_saturation_lanes_report_achieved_peak(
    tmp_path: Path,
    concurrency: int,
) -> None:
    from types import SimpleNamespace

    from test_streamed_batch import _open_fixture_runtime

    module = _load_module()
    rt, _streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=64)
    try:
        args = SimpleNamespace(
            repeats=1,
            reset_between=False,
            concurrency=concurrency,
            max_prefills_per_step=1,
            seed=0,
            output_dir=None,
            model_key="hy3-q4",
            cache_scope="global",
            slot_layout="component-banks",
            resource_telemetry=False,
            ssd_ceiling_gib_s=None,
        )
        row = module._run_concurrent_repeats(
            args,
            rt,
            prompt_ids=[1, 2, 3],
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=1),
            max_tokens=4,
            run_label="fixture",
            configuration_label=f"requested-B{concurrency}",
        )[0]

        assert row["requested_concurrency"] == concurrency
        assert row["achieved_peak_concurrency"] == concurrency
        assert row["saturation_valid"] is True
        assert row["undersubscribed"] is False
    finally:
        rt.close()


def test_kv_constrained_lane_is_marked_undersubscribed(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from test_streamed_batch import _open_fixture_runtime

    module = _load_module()
    rt, _streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=7)
    try:
        args = SimpleNamespace(
            repeats=1,
            reset_between=False,
            concurrency=2,
            max_prefills_per_step=1,
            seed=0,
            output_dir=None,
            model_key="hy3-q4",
            cache_scope="global",
            slot_layout="component-banks",
            resource_telemetry=False,
            ssd_ceiling_gib_s=None,
        )
        row = module._run_concurrent_repeats(
            args,
            rt,
            prompt_ids=[1, 2, 3],
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=1),
            max_tokens=4,
            run_label="fixture",
            configuration_label="requested-B2",
        )[0]

        assert row["requested_concurrency"] == 2
        assert row["achieved_peak_concurrency"] == 1
        assert row["saturation_valid"] is False
        assert row["undersubscribed"] is True
        assert row["evidence_label"] == "requested-B2-achieved-B1-undersubscribed"
        assert module.build_evidence_summary(
            [row],
            configuration_label="requested-B2",
            requested_concurrency=2,
        ) == {
            "configuration_label": "requested-B2",
            "evidence_label": "requested-B2-achieved-B1-undersubscribed",
            "requested_concurrency": 2,
            "achieved_peak_concurrency": 1,
            "saturation_valid": False,
            "undersubscribed": True,
            "timing_summary": row["timing_summary"],
        }
    finally:
        rt.close()


def test_concurrent_requests_are_identical_prompts_with_distinct_streams() -> None:
    module = _load_module()
    sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=1)
    requests = module.build_concurrent_requests(
        [1, 2, 3],
        concurrency=4,
        max_tokens=8,
        sampler=sampler,
        seed=5,
    )
    assert len(requests) == 4
    assert all(request.prompt_ids == (1, 2, 3) for request in requests)
    assert all(request.max_tokens == 8 for request in requests)
    assert len({request.request_id for request in requests}) == 4
    assert [request.seed for request in requests] == [5, 6, 7, 8]
    with pytest.raises(ValueError):
        module.build_concurrent_requests(
            [1], concurrency=0, max_tokens=1, sampler=sampler, seed=0
        )


def test_stream_timing_summary_uses_deterministic_r7_percentiles() -> None:
    module = _load_module()
    streams = [
        {"ttft_seconds": ttft, "completion_latency_seconds": completion}
        for ttft, completion in (
            (0.1, 1.0),
            (0.2, 2.0),
            (0.3, 3.0),
            (0.4, 4.0),
        )
    ]

    summary = module.summarize_stream_timings(
        streams,
        requested_concurrency=4,
        achieved_peak_concurrency=4,
    )

    assert summary["stream_count"] == 4
    assert summary["percentile_method"] == "linear_interpolation_r7"
    assert summary["requested_concurrency"] == 4
    assert summary["achieved_peak_concurrency"] == 4
    assert summary["saturation_valid"] is True
    assert summary["undersubscribed"] is False
    assert summary["ttft_seconds"] == pytest.approx(
        {"p50": 0.25, "p95": 0.385, "p99": 0.397}
    )
    assert summary["completion_latency_seconds"] == pytest.approx(
        {"p50": 2.5, "p95": 3.85, "p99": 3.97}
    )


def test_stream_timing_summary_marks_undersubscribed_requests() -> None:
    module = _load_module()

    summary = module.summarize_stream_timings(
        [{"ttft_seconds": 0.1, "completion_latency_seconds": 1.0}],
        requested_concurrency=2,
        achieved_peak_concurrency=1,
    )

    assert summary["saturation_valid"] is False
    assert summary["undersubscribed"] is True


def test_evidence_timing_summary_preserves_any_repeat_undersubscription() -> None:
    module = _load_module()
    stream = {"ttft_seconds": 0.1, "completion_latency_seconds": 1.0}
    rows = [
        {
            "achieved_peak_concurrency": 4,
            "undersubscribed": False,
            "streams": [stream],
        },
        {
            "achieved_peak_concurrency": 2,
            "undersubscribed": True,
            "streams": [stream],
        },
    ]

    summary = module.build_evidence_summary(
        rows,
        configuration_label="requested-B4",
        requested_concurrency=4,
    )

    assert summary["achieved_peak_concurrency"] == 4
    assert summary["undersubscribed"] is True
    assert summary["timing_summary"]["undersubscribed"] is True
    assert summary["timing_summary"]["saturation_valid"] is False


def test_concurrent_token_counter_never_drops_finished_streams() -> None:
    from types import SimpleNamespace

    counter = _load_module()._ConcurrentTokenCounter()
    counter.observe(
        SimpleNamespace(
            live=(
                SimpleNamespace(request_id="a", generated_tokens=2),
                SimpleNamespace(request_id="b", generated_tokens=3),
            )
        )
    )
    counter.observe(
        SimpleNamespace(live=(SimpleNamespace(request_id="b", generated_tokens=4),))
    )

    assert counter.count() == 6

    counter.finish(
        [
            SimpleNamespace(request_id="a", tokens=(1, 2, 3)),
            SimpleNamespace(request_id="b", tokens=(1, 2, 3, 4, 5)),
        ]
    )
    assert counter.count() == 8
