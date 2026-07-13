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


def test_unflagged_runs_stay_on_the_single_stream_reference_lane() -> None:
    parser = _load_module().build_parser()
    args = parser.parse_args(_BASE_ARGS)
    # Concurrency changes the run configuration label: an unflagged run must
    # stay comparable to every previous single-stream result.
    assert args.concurrency == 1
    assert args.max_prefills_per_step == 1


def test_saturation_lane_flags_parse() -> None:
    parser = _load_module().build_parser()
    args = parser.parse_args(
        [*_BASE_ARGS, "--concurrency", "4", "--max-prefills-per-step", "2"]
    )
    assert args.concurrency == 4
    assert args.max_prefills_per_step == 2


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
        performance_settings={"cache_policy": "frequency"},
    )

    assert summary["run_label"] == "fixture"
    assert summary["configuration_label"].startswith(
        f"cache-{cache_scope}-layout-component-banks-B{concurrency}-cfg-"
    )
    assert summary["cache_scope"] == cache_scope
    assert summary["slot_layout"] == "component-banks"
    assert summary["concurrency"] == concurrency


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
    rt, _streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=64)
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
        assert len(row["streams"]) == 2
        assert row["aggregate_completion_tokens"] == 8
        assert row["aggregate_completion_tokens_per_second"] > 0.0
        assert row["scheduler"]["decode_steps"] == 3
        for stream in row["streams"]:
            assert stream["completion_tokens"] == 4
            assert stream["finish_reason"] == "length"
            assert stream["completion_tokens_per_second"] > 0.0
            assert stream["decode_tokens_per_second"] > 0.0
            assert len(stream["token_times_s"]) == 4
            assert len(stream["token_ids"]) == 4
        assert row["streaming_after"]["live_kv_tokens"] == 0
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
