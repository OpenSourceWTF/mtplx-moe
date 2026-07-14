from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "benchmark_q2_mtp_depth_matrix.py"
)


@pytest.fixture(autouse=True)
def _fixed_matrix_environment(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    for name in (
        "MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS",
        "MTPLX_LATE_DEPTH_BEFORE",
        "MTPLX_LATE_DEPTH_AFTER",
    ):
        monkeypatch.delenv(name, raising=False)


def _load_module():
    name = "benchmark_q2_mtp_depth_matrix"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


class _FakeStreaming:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeRuntime:
    def __init__(self, model_key: str, *, mtp_enabled: bool = True) -> None:
        self.model_key = model_key
        self.mtp_enabled = mtp_enabled
        self.tokenizer = object()
        self.expert_streaming = _FakeStreaming()
        self.admissions: list[int] = []
        self.closed = False
        self.telemetry_reads = 0

    @contextlib.contextmanager
    def admit_kv_tokens(self, tokens: int):
        self.admissions.append(tokens)
        yield

    def expert_streaming_snapshot(self):
        return {
            "cache": {
                "route_calls": self.expert_streaming.reset_calls,
                "expert_hits": 3,
                "expert_misses": 3,
                "hit_rate": 0.5,
            },
            "cache_by_phase": {
                "prefill": {
                    "expert_hits": 1,
                    "expert_misses": 2,
                    "hit_rate": 1 / 3,
                },
                "decode": {
                    "expert_hits": 2,
                    "expert_misses": 1,
                    "hit_rate": 2 / 3,
                },
            },
        }

    def expert_resource_telemetry_snapshot(self):
        self.telemetry_reads += 1
        return {
            "cache": {
                "expert_bytes_read": 4096 * self.telemetry_reads,
                "read_operations": 2 * self.telemetry_reads,
            },
            "cache_by_phase": {
                "decode": {"expert_bytes_read": 4096 * self.telemetry_reads}
            },
            "reader": {"active_reads": 0, "peak_active_reads": 2},
            "expert_pipeline": {"completion_fences": 3 * self.telemetry_reads},
            "mlx_memory": {"active_memory_bytes": 1024},
        }

    def close(self) -> None:
        self.closed = True


def _stats(
    prompt_tokens: int,
    *,
    depth: int = 0,
    generated_tokens: int = 128,
    tokens: list[int] | None = None,
    graphbank: dict[str, object] | None = None,
):
    output_tokens = list(tokens or [index % 97 for index in range(generated_tokens)])
    accepted = [0 for _ in range(depth)]
    evaluated = [0 for _ in range(depth)]
    drafted = [0 for _ in range(depth)]
    events = []
    fully_accepted = 0
    bonus_tokens = 0
    cursor = 0
    event_index = 0
    while depth and cursor < len(output_tokens):
        primary = output_tokens[cursor]
        remaining = len(output_tokens) - cursor - 1
        if remaining == 0:
            events.append(
                {
                    "step": event_index,
                    "primary": primary,
                    "primary_already_emitted": bool(events),
                    "pending_primary": primary,
                    "depth": depth,
                    "requested_depth": depth,
                    "drafts": [],
                    "accepted_depths": 0,
                    "rejected_at_depth": None,
                }
            )
            break
        cycle_depth = min(depth, remaining)
        event_drafts = []
        for depth_index in range(cycle_depth):
            token = output_tokens[cursor + depth_index + 1]
            event_drafts.append(
                {
                    "depth": depth_index + 1,
                    "token": token,
                    "accepted": True,
                    "accept_probability": 1.0,
                    "correction": token,
                }
            )
            drafted[depth_index] += 1
            evaluated[depth_index] += 1
            accepted[depth_index] += 1
        event = {
            "step": event_index,
            "primary": primary,
            "primary_already_emitted": bool(events),
            "depth": depth,
            "requested_depth": depth,
            "drafts": event_drafts,
            "accepted_depths": cycle_depth,
            "rejected_at_depth": None,
        }
        cursor += cycle_depth
        if cursor + 1 < len(output_tokens):
            cursor += 1
            event["bonus_token"] = output_tokens[cursor]
            bonus_tokens += 1
        fully_accepted += 1
        events.append(event)
        event_index += 1
        if cursor >= len(output_tokens) - 1:
            break
    return SimpleNamespace(
        generated_tokens=generated_tokens,
        elapsed_s=16.0,
        decode_elapsed_s=8.0,
        decode_tok_s=16.0,
        end_to_end_tok_s=8.0,
        prompt_eval_time_s=8.0,
        new_prefill_tokens=prompt_tokens,
        prompt_target_prefill_time_s=4.0,
        prompt_target_prefill_tok_s=prompt_tokens / 4.0,
        prompt_mtp_history_time_s=4.0 if depth else 0.0,
        prompt_mtp_history_tokens=max(0, prompt_tokens - 1) if depth else 0,
        accepted_drafts=sum(accepted),
        rejected_drafts=0,
        evaluated_drafts=sum(evaluated),
        drafted_tokens=sum(drafted),
        accepted_by_depth=accepted,
        evaluated_by_depth=evaluated,
        drafted_by_depth=drafted,
        mean_accept_probability_by_depth=[1.0 for _ in range(depth)],
        fully_accepted_verify_calls=fully_accepted,
        correction_tokens=0,
        bonus_tokens=bonus_tokens,
        verify_calls=fully_accepted if depth else 127,
        requested_speculative_depth=depth,
        speculative_depth=depth,
        mtp_history_policy="committed" if depth else "none",
        mtp_history_position_base=0,
        repetition_stop_triggered=False,
        loop_guard={},
        peak_memory_bytes=3 * 1024**3,
        events=events,
        graphbank={} if graphbank is None else graphbank,
    )


def _fake_apis(
    module,
    *,
    output_tokens: int = 128,
    mismatch_depth: int | None = None,
    compiled_evidence: dict[str, object] | None = None,
):
    calls = SimpleNamespace(
        loads=[],
        configs=[],
        prompt_builds=[],
        ar=[],
        mtpk=[],
        runtimes=[],
        peak_resets=0,
        synchronizations=0,
    )

    def parse_memory(value):
        units = {"MiB": 1024**2, "GiB": 1024**3}
        for suffix, multiplier in units.items():
            if str(value).endswith(suffix):
                return int(str(value)[: -len(suffix)]) * multiplier
        return int(value)

    def config_factory(**kwargs):
        calls.configs.append(kwargs)
        return dict(kwargs)

    def load(model_root, **kwargs):
        calls.loads.append((Path(model_root), kwargs))
        runtime = _FakeRuntime(
            kwargs["expert_streaming_config"]["model_key"],
            mtp_enabled=bool(kwargs["mtp"]),
        )
        calls.runtimes.append(runtime)
        return runtime

    def prompt_builder(tokenizer, context_tokens, **kwargs):
        calls.prompt_builds.append((tokenizer, context_tokens, kwargs))
        return SimpleNamespace(
            token_ids=list(range(context_tokens)),
            metadata={
                "prompt_policy": "coding_agent_tail_v2",
                "prompt_actual_tokens": context_tokens,
                "prompt_tail_preserved": True,
            },
        )

    def sampler_factory(**kwargs):
        return ("sampler", kwargs)

    def generate_ar(runtime, prompt_ids, **kwargs):
        calls.ar.append((runtime, tuple(prompt_ids), kwargs))
        count = output_tokens if kwargs["max_tokens"] == 128 else kwargs["max_tokens"]
        tokens = [index % 97 for index in range(count)]
        return SimpleNamespace(
            tokens=tokens,
            finish_reason="length",
            stats=_stats(len(prompt_ids), generated_tokens=len(tokens), tokens=tokens),
        )

    def generate_mtpk(runtime, prompt_ids, **kwargs):
        calls.mtpk.append((runtime, tuple(prompt_ids), kwargs))
        depth = kwargs["speculative_depth"]
        count = output_tokens if kwargs["max_tokens"] == 128 else kwargs["max_tokens"]
        tokens = [index % 97 for index in range(count)]
        if mismatch_depth == depth:
            tokens[-1] += 1
        return SimpleNamespace(
            tokens=tokens,
            finish_reason="length",
            stats=_stats(
                len(prompt_ids),
                depth=depth,
                generated_tokens=len(tokens),
                tokens=tokens,
                graphbank=compiled_evidence,
            ),
            final_state=SimpleNamespace(
                safe_to_commit=True,
                generated_token_ids=tuple(tokens),
                finish_reason="length",
                final_trunk_cache=[
                    SimpleNamespace(offset=len(prompt_ids) + len(tokens))
                ],
                final_committed_mtp_cache=[
                    SimpleNamespace(offset=len(prompt_ids) + len(tokens) - 1)
                ],
                mtp_history_position_base=0,
            ),
        )

    def reset_peak_memory():
        calls.peak_resets += 1

    def synchronize():
        calls.synchronizations += 1

    def get_peak_memory():
        return 5 * 1024**3

    return (
        module.RunnerAPIs(
            load=load,
            config_factory=config_factory,
            parse_memory_bytes=parse_memory,
            prompt_builder=prompt_builder,
            sampler_factory=sampler_factory,
            generate_ar=generate_ar,
            generate_mtpk=generate_mtpk,
            reset_peak_memory=reset_peak_memory,
            get_peak_memory=get_peak_memory,
            synchronize=synchronize,
        ),
        calls,
    )


def _requests(tmp_path: Path):
    return [
        {
            "model": "hy3-q2",
            "model_root": tmp_path / "hy3-model",
            "manifest": tmp_path / "hy3-manifest.json",
            "mtp_artifacts": tmp_path / "hy3-mtp",
            "prompt_tail_text": "Hy3 fixed prompt tail.",
        },
        {
            "model": "glm52-q2",
            "model_root": tmp_path / "glm-model",
            "manifest": tmp_path / "glm-manifest.json",
            "mtp_artifacts": tmp_path / "glm-mtp",
            "prompt_tail_text": "GLM fixed prompt tail.",
        },
    ]


def test_parser_defaults_to_both_models_and_the_required_matrix() -> None:
    module = _load_module()
    args = module.build_parser().parse_args([])

    assert args.models is None
    assert args.contexts == (1024, 2048)
    assert args.hy3_depths == (1, 2, 3, 4, 5)
    assert args.glm52_depths == (1, 2, 3, 4, 5)
    assert args.memory_limit == "112GiB"
    assert args.runtime_reserve == "12GiB"
    assert args.expert_cache_limit == "64GiB"
    assert args.max_live_kv_tokens == 4096
    assert args.resource_telemetry is False
    assert args.mtp_disabled_baseline is False
    assert args.verify_strategy == "batched"
    assert args.compiled_verify_mode == "off"
    assert args.trace_routes is False


def test_compiled_verify_requires_capture_commit_before_model_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    apis, calls = _fake_apis(module)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity")

    with pytest.raises(
        module.BenchmarkConfigurationError,
        match="compiled verify requires capture_commit",
    ):
        module.run_depth_matrix(
            [{**_requests(tmp_path)[0], "depths": (1,)}],
            contexts=(1024,),
            verify_strategy="batched",
            compiled_verify_mode="parity",
            apis=apis,
        )

    assert calls.loads == []


def test_compiled_verify_requires_complete_per_row_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "on")

    with pytest.raises(
        module.BenchmarkGateError,
        match="compiled verifier emitted no evidence",
    ):
        module.run_depth_matrix(
            [{**_requests(tmp_path)[0], "depths": (1,)}],
            contexts=(1024,),
            verify_strategy="capture_commit",
            compiled_verify_mode="on",
            apis=apis,
        )


def test_compiled_verify_records_candidate_and_forwards_diagnostic_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    evidence = {
        "compiled_verify": {
            "calls": 4,
            "compiled_calls": 4,
            "fallback_calls": 0,
            "fallback_reasons": {},
            "mode": "on",
        }
    }
    apis, calls = _fake_apis(module, compiled_evidence=evidence)
    observer = object()
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "on")

    payload = module.run_depth_matrix(
        [{**_requests(tmp_path)[0], "depths": (1,)}],
        contexts=(1024,),
        verify_strategy="capture_commit",
        compiled_verify_mode="on",
        trace_routes=True,
        draft_observer=observer,
        apis=apis,
    )

    assert payload["configuration"]["candidate"] == {
        "verify_strategy": "capture_commit",
        "compiled_verify_mode": "on",
        "trace_routes": True,
    }
    assert calls.configs[0]["trace_routes"] is True
    assert calls.mtpk
    assert all(call[2]["verify_strategy"] == "capture_commit" for call in calls.mtpk)
    assert all(call[2]["draft_observer"] is observer for call in calls.mtpk)
    d1 = payload["models"][0]["observations"][1]
    assert d1["compiled_verify"] == evidence["compiled_verify"]


@pytest.mark.parametrize("value", [None, "0", "false"])
def test_matrix_requires_and_records_sustained_prefill(
    tmp_path: Path,
    monkeypatch,
    value: str | None,
) -> None:
    module = _load_module()
    apis, calls = _fake_apis(module)
    if value is None:
        monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
    else:
        monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", value)

    with pytest.raises(
        module.BenchmarkConfigurationError,
        match="MTPLX_SUSTAINED_PREFILL=1",
    ):
        module.run_depth_matrix(
            [{**_requests(tmp_path)[0], "depths": (1,)}],
            contexts=(1024,),
            apis=apis,
        )

    assert calls.loads == []

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "yes")
    payload = module.run_depth_matrix(
        [{**_requests(tmp_path)[0], "depths": (1,)}],
        contexts=(1024,),
        apis=_fake_apis(module)[0],
    )
    assert payload["configuration"]["generation_environment"] == {
        "MTPLX_SUSTAINED_PREFILL": "yes",
        "MTPLX_SUSTAINED_PREFILL_LAYOUT": "auto",
        "MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS": None,
        "MTPLX_LATE_DEPTH_BEFORE": None,
        "MTPLX_LATE_DEPTH_AFTER": None,
    }


def test_fixed_matrix_rejects_late_depth_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    apis, calls = _fake_apis(module)
    monkeypatch.setenv("MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS", "64")

    with pytest.raises(
        module.BenchmarkConfigurationError,
        match="fixed-depth matrix forbids MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS",
    ):
        module.run_depth_matrix(
            [{**_requests(tmp_path)[0], "depths": (5,)}],
            contexts=(1024,),
            apis=apis,
        )

    assert calls.loads == []


def test_matrix_loads_each_model_once_and_uses_only_canonical_generators(
    tmp_path: Path,
) -> None:
    module = _load_module()
    apis, calls = _fake_apis(module)

    payload = module.run_depth_matrix(_requests(tmp_path), apis=apis)

    assert payload["schema"] == "mtplx-q2-bf16-mtp-depth-matrix-v3"
    assert payload["passed"] is True
    assert payload["lane"] == "mtp-resident-depth-matrix"
    assert all(model["mtp_resident"] is True for model in payload["models"])
    assert [config["model_key"] for config in calls.configs] == [
        "hy3-expert-q2",
        "glm52-expert-q2",
    ]
    assert all(config["resource_telemetry"] is False for config in calls.configs)
    assert len(calls.loads) == 2
    assert all(kwargs["mtp"] is True for _root, kwargs in calls.loads)
    assert all(kwargs["mtp_precision"] == "bf16" for _root, kwargs in calls.loads)
    assert len(calls.ar) == 8
    assert len(calls.mtpk) == 40
    assert calls.peak_resets == 48
    assert calls.synchronizations >= 48
    assert [len(model["observations"]) for model in payload["models"]] == [12, 12]
    assert [model["discarded_warmup_count"] for model in payload["models"]] == [
        12,
        12,
    ]
    assert all(
        row["generated_tokens"] == 128
        for model in payload["models"]
        for row in model["observations"]
    )
    assert all(
        row["expert_resource_telemetry"] is None
        for model in payload["models"]
        for row in model["observations"]
    )
    assert all(
        row["gates"]["speculative_event_contract"]
        for model in payload["models"]
        for row in model["observations"]
        if row["requested_depth"] > 0
    )

    for runtime in calls.runtimes:
        model = next(
            row for row in payload["models"] if row["model_key"] == runtime.model_key
        )
        assert runtime.expert_streaming.reset_calls == 2 * len(model["observations"])
        cells = 6
        assert runtime.admissions == [
            tokens
            for context in (1024, 2048)
            for _cell in range(cells)
            for tokens in (context + 8, context + 128)
        ]
        assert runtime.closed is True

    for _runtime, _prompt, kwargs in calls.ar:
        assert kwargs["max_tokens"] in {8, 128}
        assert kwargs["stop_token_ids"] == set()
        assert kwargs["repetition_stop"] is False
        assert kwargs["loop_guard"] is False
    for _runtime, _prompt, kwargs in calls.mtpk:
        assert kwargs["max_tokens"] in {8, 128}
        assert kwargs["mtp_cache_policy"] == "persistent"
        assert kwargs["mtp_history_policy"] == "committed"
        assert kwargs["stop_token_ids"] == set()
        assert kwargs["repetition_stop"] is False
        assert kwargs["loop_guard"] is False
    assert sum(kwargs["max_tokens"] == 8 for *_rest, kwargs in calls.ar) == 4
    assert sum(kwargs["max_tokens"] == 8 for *_rest, kwargs in calls.mtpk) == 20

    assert len(calls.prompt_builds) == 4
    assert all(
        kwargs
        == {
            "prompt_style": "coding-agent",
            "prompt_tail": kwargs["prompt_tail"],
            "prompt_format": "raw",
            "enable_thinking": False,
        }
        for _tokenizer, _context, kwargs in calls.prompt_builds
    )


def test_ar_path_drift_is_retained_as_diagnostic_and_matrix_continues(
    tmp_path: Path,
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module, mismatch_depth=2)
    snapshots = []

    payload = module.run_depth_matrix(
        [{**_requests(tmp_path)[0], "depths": (1, 2, 3)}],
        contexts=(1024,),
        checkpoint=snapshots.append,
        apis=apis,
    )

    assert payload["status"] == "passed"
    assert payload["passed"] is True
    model = payload["models"][0]
    assert [row["cell"] for row in model["observations"]] == ["ar", "d1", "d2", "d3"]
    d2 = model["observations"][2]
    assert d2["ar_comparison"] == {
        "status": "token_divergence",
        "token_parity": False,
        "first_divergence": 127,
        "differing_token_count": 1,
        "reference_token_at_first_divergence": 30,
        "observed_token_at_first_divergence": 31,
        "reference_token_sha256": d2["ar_comparison"]["reference_token_sha256"],
        "observed_token_sha256": d2["ar_comparison"]["observed_token_sha256"],
        "divergence_attribution": "unclassified",
    }
    assert (
        d2["ar_comparison"]["reference_token_sha256"]
        != d2["ar_comparison"]["observed_token_sha256"]
    )
    assert "ar_token_parity" not in d2["gates"]
    assert d2["gates"]["speculative_event_contract"] is True
    assert [warning["phase"] for warning in model["token_divergence_observations"]] == [
        "warmup",
        "retained",
    ]
    assert all(
        warning["depth"] == 2 for warning in model["token_divergence_observations"]
    )
    assert snapshots[-1] == payload


def test_accepted_draft_without_matching_target_correction_fails_closed(
    tmp_path: Path,
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    original = apis.generate_mtpk

    def invalid_acceptance(runtime, prompt_ids, **kwargs):
        result = original(runtime, prompt_ids, **kwargs)
        if kwargs["max_tokens"] == 128 and kwargs["speculative_depth"] == 1:
            accepted = next(
                draft
                for event in result.stats.events
                for draft in event["drafts"]
                if draft.get("accepted") is True
            )
            accepted["correction"] = accepted["token"] + 1
        return result

    apis.generate_mtpk = invalid_acceptance

    with pytest.raises(module.BenchmarkGateError, match="accepted draft token"):
        module.run_depth_matrix(
            [{**_requests(tmp_path)[0], "depths": (1,)}],
            contexts=(1024,),
            apis=apis,
        )


def test_event_contract_appends_new_primary_after_eager_rejection_correction() -> None:
    module = _load_module()
    stats = SimpleNamespace(
        events=[
            {
                "primary": 1,
                "primary_already_emitted": False,
                "depth": 1,
                "requested_depth": 1,
                "drafts": [
                    {
                        "depth": 1,
                        "token": 2,
                        "accepted": False,
                        "accept_probability": 0.0,
                        "correction": 6,
                    }
                ],
                "accepted_depths": 0,
                "rejected_at_depth": 1,
            },
            {
                "primary": 7,
                "primary_already_emitted": False,
                "pending_primary": 7,
                "depth": 1,
                "requested_depth": 1,
                "drafts": [],
                "accepted_depths": 0,
                "rejected_at_depth": None,
            },
        ],
        drafted_tokens=1,
        evaluated_drafts=1,
        accepted_drafts=0,
        rejected_drafts=1,
        drafted_by_depth=[1],
        evaluated_by_depth=[1],
        accepted_by_depth=[0],
        mean_accept_probability_by_depth=[0.0],
        correction_tokens=1,
        bonus_tokens=0,
        fully_accepted_verify_calls=0,
        verify_calls=1,
    )

    summary = module._validate_speculative_event_contract(
        stats,
        model="hy3-q2",
        depth=1,
        tokens=[1, 6, 7],
    )

    assert summary == {
        "events": 2,
        "verify_events": 1,
        "drafted_records": 1,
        "evaluated_records": 1,
        "accepted_records": 0,
    }


def test_fixed_depth_event_contract_rejects_shallower_effective_depth() -> None:
    module = _load_module()
    stats = _stats(1024, depth=5, generated_tokens=6)
    stats.events[0]["depth"] = 1

    with pytest.raises(module.BenchmarkGateError, match="effective depth 1"):
        module._validate_speculative_event_contract(
            stats,
            model="hy3-q2",
            depth=5,
            tokens=[0, 1, 2, 3, 4, 5],
        )


def test_rejected_event_with_bonus_fails_contract() -> None:
    module = _load_module()
    stats = SimpleNamespace(
        events=[
            {
                "primary": 1,
                "primary_already_emitted": False,
                "depth": 1,
                "requested_depth": 1,
                "drafts": [
                    {
                        "depth": 1,
                        "token": 2,
                        "accepted": False,
                        "accept_probability": 0.0,
                        "correction": 6,
                    }
                ],
                "accepted_depths": 0,
                "rejected_at_depth": 1,
                "bonus_token": 99,
            }
        ],
        drafted_tokens=1,
        evaluated_drafts=1,
        accepted_drafts=0,
        rejected_drafts=1,
        drafted_by_depth=[1],
        evaluated_by_depth=[1],
        accepted_by_depth=[0],
        mean_accept_probability_by_depth=[0.0],
        correction_tokens=1,
        bonus_tokens=0,
        fully_accepted_verify_calls=0,
        verify_calls=1,
    )

    with pytest.raises(module.BenchmarkGateError, match="rejected event.*bonus"):
        module._validate_speculative_event_contract(
            stats,
            model="hy3-q2",
            depth=1,
            tokens=[1, 6],
        )


def test_cache_offsets_reject_missing_entries() -> None:
    module = _load_module()

    with pytest.raises(module.BenchmarkGateError, match="cache 0 is missing"):
        module._cache_offsets(
            [None, SimpleNamespace(offset=128)],
            label="target",
        )


def test_rejected_draft_cannot_equal_greedy_target_correction() -> None:
    module = _load_module()
    stats = SimpleNamespace(
        events=[
            {
                "primary": 1,
                "primary_already_emitted": False,
                "depth": 1,
                "requested_depth": 1,
                "drafts": [
                    {
                        "depth": 1,
                        "token": 2,
                        "accepted": False,
                        "accept_probability": 0.0,
                        "correction": 2,
                    }
                ],
                "accepted_depths": 0,
                "rejected_at_depth": 1,
            }
        ],
        drafted_tokens=1,
        evaluated_drafts=1,
        accepted_drafts=0,
        rejected_drafts=1,
        drafted_by_depth=[1],
        evaluated_by_depth=[1],
        accepted_by_depth=[0],
        mean_accept_probability_by_depth=[0.0],
        correction_tokens=1,
        bonus_tokens=0,
        fully_accepted_verify_calls=0,
        verify_calls=1,
    )

    with pytest.raises(module.BenchmarkGateError, match="rejected.*correction"):
        module._validate_speculative_event_contract(
            stats,
            model="hy3-q2",
            depth=1,
            tokens=[1, 2],
        )


def test_primary_already_emitted_requires_prior_pending_decision() -> None:
    module = _load_module()
    stats = SimpleNamespace(
        events=[
            {
                "primary": 1,
                "primary_already_emitted": False,
                "depth": 1,
                "requested_depth": 1,
                "drafts": [
                    {
                        "depth": 1,
                        "token": 2,
                        "accepted": False,
                        "accept_probability": 0.0,
                        "correction": 6,
                    }
                ],
                "accepted_depths": 0,
                "rejected_at_depth": 1,
            },
            {
                "primary": 6,
                "primary_already_emitted": True,
                "depth": 1,
                "requested_depth": 1,
                "drafts": [
                    {
                        "depth": 1,
                        "token": 7,
                        "accepted": True,
                        "accept_probability": 1.0,
                        "correction": 7,
                    }
                ],
                "accepted_depths": 1,
                "rejected_at_depth": None,
            },
        ],
        drafted_tokens=2,
        evaluated_drafts=2,
        accepted_drafts=1,
        rejected_drafts=1,
        drafted_by_depth=[2],
        evaluated_by_depth=[2],
        accepted_by_depth=[1],
        mean_accept_probability_by_depth=[0.5],
        correction_tokens=1,
        bonus_tokens=0,
        fully_accepted_verify_calls=1,
        verify_calls=2,
    )

    with pytest.raises(module.BenchmarkGateError, match="prior pending"):
        module._validate_speculative_event_contract(
            stats,
            model="hy3-q2",
            depth=1,
            tokens=[1, 6, 7],
        )


def test_event_contract_recomputes_fully_accepted_totals() -> None:
    module = _load_module()
    stats = SimpleNamespace(
        events=[
            {
                "primary": 1,
                "primary_already_emitted": False,
                "depth": 1,
                "requested_depth": 1,
                "drafts": [
                    {
                        "depth": 1,
                        "token": 2,
                        "accepted": False,
                        "accept_probability": 0.0,
                        "correction": 6,
                    }
                ],
                "accepted_depths": 0,
                "rejected_at_depth": 1,
            }
        ],
        drafted_tokens=1,
        evaluated_drafts=1,
        accepted_drafts=0,
        rejected_drafts=1,
        drafted_by_depth=[1],
        evaluated_by_depth=[1],
        accepted_by_depth=[0],
        mean_accept_probability_by_depth=[0.0],
        correction_tokens=1,
        bonus_tokens=0,
        fully_accepted_verify_calls=1,
        verify_calls=1,
    )

    with pytest.raises(module.BenchmarkGateError, match="derived event totals"):
        module._validate_speculative_event_contract(
            stats,
            model="hy3-q2",
            depth=1,
            tokens=[1, 6],
        )


def test_mtp_row_requires_safe_final_state_and_exact_cache_offsets(
    tmp_path: Path,
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    original = apis.generate_mtpk

    def invalid_final_state(runtime, prompt_ids, **kwargs):
        result = original(runtime, prompt_ids, **kwargs)
        result.final_state = SimpleNamespace(
            safe_to_commit=True,
            generated_token_ids=tuple(result.tokens),
            finish_reason=result.finish_reason,
            final_trunk_cache=[
                SimpleNamespace(offset=len(prompt_ids) + len(result.tokens) - 1)
            ],
            final_committed_mtp_cache=[
                SimpleNamespace(offset=len(prompt_ids) + len(result.tokens) - 1)
            ],
            mtp_history_position_base=0,
        )
        return result

    apis.generate_mtpk = invalid_final_state

    with pytest.raises(module.BenchmarkGateError, match="target cache offsets"):
        module.run_depth_matrix(
            [{**_requests(tmp_path)[0], "depths": (1,)}],
            contexts=(1024,),
            apis=apis,
        )


def test_mtp_row_requires_exact_prompt_history_length(tmp_path: Path) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    original = apis.generate_mtpk

    def invalid_history(runtime, prompt_ids, **kwargs):
        result = original(runtime, prompt_ids, **kwargs)
        result.stats.prompt_mtp_history_tokens = len(prompt_ids) - 2
        return result

    apis.generate_mtpk = invalid_history

    with pytest.raises(module.BenchmarkGateError, match="prompt MTP history"):
        module.run_depth_matrix(
            [{**_requests(tmp_path)[0], "depths": (1,)}],
            contexts=(1024,),
            apis=apis,
        )


def test_failing_event_gate_retains_generated_evidence(tmp_path: Path) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    original = apis.generate_mtpk
    snapshots = []

    def invalid_event(runtime, prompt_ids, **kwargs):
        result = original(runtime, prompt_ids, **kwargs)
        if kwargs["max_tokens"] == 128:
            accepted = next(
                draft
                for event in result.stats.events
                for draft in event["drafts"]
                if draft.get("accepted") is True
            )
            accepted["correction"] += 1
        return result

    apis.generate_mtpk = invalid_event

    with pytest.raises(module.BenchmarkGateError, match="accepted draft token"):
        module.run_depth_matrix(
            [{**_requests(tmp_path)[0], "depths": (1,)}],
            contexts=(1024,),
            checkpoint=snapshots.append,
            apis=apis,
        )

    evidence = snapshots[-1]["failure"]["evidence"]
    assert evidence["model"] == "hy3-q2"
    assert evidence["depth"] == 1
    assert len(evidence["token_ids"]) == 128
    assert evidence["generation_events"]
    assert evidence["generation_stats"]["accepted_drafts"] > 0


def test_decode_cache_gate_retains_generated_evidence(tmp_path: Path) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    original_load = apis.load
    snapshots = []

    def load_with_empty_decode_cache(*args, **kwargs):
        runtime = original_load(*args, **kwargs)

        def empty_decode_cache():
            return {
                "cache": {"expert_hits": 0, "expert_misses": 0, "hit_rate": 0.0},
                "cache_by_phase": {
                    "prefill": {
                        "expert_hits": 1,
                        "expert_misses": 1,
                        "hit_rate": 0.5,
                    },
                    "decode": {
                        "expert_hits": 0,
                        "expert_misses": 0,
                        "hit_rate": 0.0,
                    },
                },
            }

        runtime.expert_streaming_snapshot = empty_decode_cache
        return runtime

    apis.load = load_with_empty_decode_cache

    with pytest.raises(module.BenchmarkGateError, match="no routed assignments"):
        module.run_depth_matrix(
            [{**_requests(tmp_path)[0], "depths": (1,)}],
            contexts=(1024,),
            checkpoint=snapshots.append,
            apis=apis,
        )

    evidence = snapshots[-1]["failure"]["evidence"]
    assert evidence["model"] == "hy3-q2"
    assert evidence["depth"] == 0
    assert len(evidence["token_ids"]) == 8
    assert evidence["generation_stats"]["generated_tokens"] == 8
    assert evidence["expert_streaming_counters_by_phase"]["decode"] == {
        "expert_hits": 0,
        "expert_misses": 0,
        "hit_rate": 0.0,
    }


def test_checkpoint_callback_emits_independent_running_and_terminal_snapshots(
    tmp_path: Path,
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    snapshots = []

    payload = module.run_depth_matrix(
        [{**_requests(tmp_path)[0], "depths": (1,)}],
        contexts=(1024,),
        checkpoint=snapshots.append,
        apis=apis,
    )

    assert snapshots[0]["status"] == "running"
    assert snapshots[0]["passed"] is False
    assert snapshots[0]["models"] == []
    one_row = next(
        snapshot
        for snapshot in snapshots
        if snapshot["models"] and len(snapshot["models"][0]["observations"]) == 1
    )
    assert one_row["models"][0]["observations"][0]["cell"] == "ar"
    assert payload["status"] == "passed"
    assert payload["passed"] is True
    assert payload["active_cell"] is None
    assert snapshots[-1] == payload

    one_row["models"][0]["observations"][0]["cell"] = "mutated"
    assert snapshots[-1]["models"][0]["observations"][0]["cell"] == "ar"
    assert len(snapshots) == 5


def test_runner_emits_live_terminal_failure_checkpoint(
    tmp_path: Path,
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    original = apis.generate_mtpk

    def invalid_retained_d1(runtime, prompt_ids, **kwargs):
        result = original(runtime, prompt_ids, **kwargs)
        if kwargs["max_tokens"] == 128:
            accepted = next(
                draft
                for event in result.stats.events
                for draft in event["drafts"]
                if draft.get("accepted") is True
            )
            accepted["correction"] = accepted["token"] + 1
        return result

    apis.generate_mtpk = invalid_retained_d1
    snapshots = []

    with pytest.raises(module.BenchmarkGateError, match="accepted draft token"):
        module.run_depth_matrix(
            [{**_requests(tmp_path)[0], "depths": (1,)}],
            contexts=(1024,),
            checkpoint=snapshots.append,
            apis=apis,
        )

    failed = snapshots[-1]
    assert failed["status"] == "failed"
    assert failed["passed"] is False
    assert [row["cell"] for row in failed["models"][0]["observations"]] == ["ar"]
    assert {
        key: failed["failure"][key] for key in ("error", "error_type", "active_cell")
    } == {
        "error": (
            "hy3-q2 d1 accepted draft token 1 without matching target correction 2"
        ),
        "error_type": "BenchmarkGateError",
        "active_cell": {
            "model": "hy3-q2",
            "context_tokens": 1024,
            "depth": 1,
            "cell": "d1",
            "phase": "retained",
        },
    }


def test_runner_preserves_primary_gate_when_runtime_close_also_fails(
    tmp_path: Path,
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    original_generate = apis.generate_mtpk
    original_load = apis.load

    def invalid_retained_d1(runtime, prompt_ids, **kwargs):
        result = original_generate(runtime, prompt_ids, **kwargs)
        if kwargs["max_tokens"] == 128:
            accepted = next(
                draft
                for event in result.stats.events
                for draft in event["drafts"]
                if draft.get("accepted") is True
            )
            accepted["correction"] = accepted["token"] + 1
        return result

    def load_with_failed_close(*args, **kwargs):
        runtime = original_load(*args, **kwargs)

        def failed_close():
            raise RuntimeError("runtime close failed")

        runtime.close = failed_close
        return runtime

    apis.generate_mtpk = invalid_retained_d1
    apis.load = load_with_failed_close
    snapshots = []

    with pytest.raises(module.BenchmarkGateError, match="accepted draft token"):
        module.run_depth_matrix(
            [{**_requests(tmp_path)[0], "depths": (1,)}],
            contexts=(1024,),
            checkpoint=snapshots.append,
            apis=apis,
        )

    failed = snapshots[-1]
    assert failed["failure"]["error_type"] == "BenchmarkGateError"
    assert failed["cleanup_errors"] == [
        {
            "error": "runtime close failed",
            "error_type": "RuntimeError",
            "model": "hy3-q2",
        }
    ]


def test_second_model_config_failure_names_the_unstarted_model(
    tmp_path: Path,
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    original = apis.config_factory

    def fail_glm_config(**kwargs):
        if kwargs["model_key"] == "glm52-expert-q2":
            raise RuntimeError("glm config failed")
        return original(**kwargs)

    apis.config_factory = fail_glm_config
    snapshots = []

    with pytest.raises(RuntimeError, match="glm config failed"):
        module.run_depth_matrix(
            _requests(tmp_path),
            contexts=(1024,),
            checkpoint=snapshots.append,
            apis=apis,
        )

    failed = snapshots[-1]
    assert failed["status"] == "failed"
    assert failed["failure"]["active_cell"] == {
        "model": "glm52-q2",
        "context_tokens": None,
        "depth": None,
        "cell": None,
        "phase": "configuration",
    }
    assert failed["models"][0]["passed"] is True
    assert len(failed["models"]) == 1


def test_preflight_failure_uses_the_checkpoint_schema(tmp_path: Path) -> None:
    module = _load_module()
    request = _requests(tmp_path)[0]
    request.pop("prompt_tail_text")
    request["prompt_tail"] = tmp_path / "missing.txt"
    snapshots = []

    with pytest.raises(FileNotFoundError):
        module.run_depth_matrix(
            [request],
            contexts=(1024,),
            checkpoint=snapshots.append,
            apis=_fake_apis(module)[0],
        )

    failed = snapshots[-1]
    assert failed["schema"] == module.SCHEMA
    assert failed["status"] == "failed"
    assert failed["passed"] is False
    assert failed["models"] == []
    assert failed["failure"]["error_type"] == "FileNotFoundError"
    assert failed["failure"]["active_cell"] == {"phase": "configuration"}


def test_mtp_disabled_baseline_loads_once_and_runs_only_two_ar_contexts(
    tmp_path: Path,
) -> None:
    module = _load_module()
    parsed = module.build_parser().parse_args(["--mtp-disabled-baseline"])
    assert parsed.mtp_disabled_baseline is True
    apis, calls = _fake_apis(module)
    request = dict(_requests(tmp_path)[0])
    request.pop("mtp_artifacts")
    request["depths"] = ()

    payload = module.run_depth_matrix(
        [request],
        mtp_disabled_baseline=True,
        apis=apis,
    )

    assert payload["lane"] == "mtp-disabled-ar-baseline"
    assert payload["configuration"]["mtp_resident"] is False
    assert payload["configuration"]["contexts"] == [1024, 2048]
    assert len(calls.loads) == 1
    _root, load_kwargs = calls.loads[0]
    assert load_kwargs["mtp"] is False
    assert set(load_kwargs) == {
        "mtp",
        "expert_streaming_config",
        "expert_manifest",
    }
    assert calls.mtpk == []
    assert len(calls.ar) == 4
    assert calls.peak_resets == 4
    assert len(calls.prompt_builds) == 2

    model = payload["models"][0]
    assert model["lane"] == "mtp-disabled-ar-baseline"
    assert model["mtp_resident"] is False
    assert model["depths"] == []
    assert model["discarded_warmup_count"] == 2
    assert [row["context_tokens"] for row in model["observations"]] == [
        1024,
        2048,
    ]
    assert all(row["cell"] == "ar" for row in model["observations"])
    assert all(row["generated_tokens"] == 128 for row in model["observations"])
    assert all(row["peak_memory_bytes"] == 3 * 1024**3 for row in model["observations"])
    assert calls.runtimes[0].admissions == [1032, 1152, 2056, 2176]
    assert calls.runtimes[0].closed is True


def test_metrics_exclude_final_state_validation_time_from_mtp_throughput() -> None:
    module = _load_module()
    stats = _stats(1024, depth=1)
    stats.final_state_capture_time_s = 2.0

    metrics = module._metrics(stats, completion_tokens=128, depth=1)

    assert metrics["raw_elapsed_s"] == 16.0
    assert metrics["raw_decode_elapsed_s"] == 8.0
    assert metrics["final_state_capture_time_s"] == 2.0
    assert metrics["elapsed_s"] == 14.0
    assert metrics["decode_elapsed_s"] == 6.0
    assert metrics["decode_tok_s"] == pytest.approx(128 / 6)
    assert metrics["end_to_end_tok_s"] == pytest.approx(128 / 14)


def test_rows_recompute_ingestion_decode_and_acceptance_metrics(tmp_path: Path) -> None:
    module = _load_module()
    apis, calls = _fake_apis(module)

    payload = module.run_depth_matrix(
        [_requests(tmp_path)[0]],
        contexts=(1024,),
        runtime_options={"resource_telemetry": True},
        apis=apis,
    )
    depth_two = next(
        row for row in payload["models"][0]["observations"] if row["cell"] == "d2"
    )

    assert depth_two["prompt_tokens"] == 1024
    assert depth_two["new_prefill_tokens"] == 1024
    assert depth_two["ingestion_tok_s"] == 128.0
    assert depth_two["prompt_target_prefill_time_s"] == 4.0
    assert depth_two["prompt_target_prefill_tok_s"] == 256.0
    assert depth_two["decode_tok_s"] == 16.0
    assert depth_two["expert_streaming_counters_by_phase"] == {
        "prefill": {
            "expert_hits": 1,
            "expert_misses": 2,
            "hit_rate": pytest.approx(1 / 3),
        },
        "decode": {
            "expert_hits": 2,
            "expert_misses": 1,
            "hit_rate": pytest.approx(2 / 3),
        },
    }
    assert depth_two["decode_expert_cache_hit_rate"] == pytest.approx(2 / 3)
    assert depth_two["peak_memory_bytes"] == 3 * 1024**3
    assert payload["models"][0]["hard_peak_memory_bytes"] == 5 * 1024**3
    assert calls.configs[0]["resource_telemetry"] is True
    assert (
        payload["configuration"]["measurement_lane"]
        == "diagnostic-resource-instrumented"
    )
    telemetry = depth_two["expert_resource_telemetry"]
    assert telemetry["numeric_delta"] == {
        "cache": {"expert_bytes_read": 4096, "read_operations": 2},
        "cache_by_phase": {"decode": {"expert_bytes_read": 4096}},
        "reader": {"active_reads": 0, "peak_active_reads": 0},
        "expert_pipeline": {"completion_fences": 3},
        "mlx_memory": {"active_memory_bytes": 0},
    }
    assert (
        telemetry["after"]["cache"]["expert_bytes_read"]
        > telemetry["before"]["cache"]["expert_bytes_read"]
    )
    assert depth_two["accepted_drafts"] == 85
    assert depth_two["evaluated_drafts"] == 85
    assert depth_two["drafted_tokens"] == 85
    assert depth_two["conditional_hit_rate"] == 1.0
    assert depth_two["cumulative_accepted_drafted_yield"] == 1.0
    assert depth_two["accepted_per_verify"] == pytest.approx(85 / 43)
    assert depth_two["fully_accepted_verify_ratio"] == 1.0
    assert depth_two["acceptance_by_depth"] == [
        {
            "depth": 1,
            "drafted": 43,
            "evaluated": 43,
            "accepted": 43,
            "conditional_hit_rate": 1.0,
            "cumulative_accepted_drafted_yield": 1.0,
            "mean_accept_probability": 1.0,
        },
        {
            "depth": 2,
            "drafted": 42,
            "evaluated": 42,
            "accepted": 42,
            "conditional_hit_rate": 1.0,
            "cumulative_accepted_drafted_yield": 1.0,
            "mean_accept_probability": 1.0,
        },
    ]
    assert depth_two["gates"] == {
        "prompt_length_exact": True,
        "new_prefill_tokens_exact": True,
        "output_tokens_exact": True,
        "generated_count_consistent": True,
        "length_finish": True,
        "requested_depth_exact": True,
        "effective_depth_exact": True,
        "committed_history": True,
        "guards_disabled": True,
        "decode_expert_cache_metrics": True,
        "speculative_event_contract": True,
        "final_state_contract": True,
        "compiled_verify_evidence": True,
    }


def test_exact_prompt_and_output_gates_fail_closed(tmp_path: Path) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module, output_tokens=127)
    with pytest.raises(module.BenchmarkGateError, match="exactly 128"):
        module.run_depth_matrix([_requests(tmp_path)[0]], contexts=(1024,), apis=apis)

    apis, _calls = _fake_apis(module, mismatch_depth=2)
    payload = module.run_depth_matrix(
        [_requests(tmp_path)[0]], contexts=(1024,), apis=apis
    )
    depth_two = next(
        row for row in payload["models"][0]["observations"] if row["cell"] == "d2"
    )
    assert depth_two["ar_comparison"]["status"] == "token_divergence"

    bad_apis, _calls = _fake_apis(module)
    original = bad_apis.prompt_builder

    def short_prompt(tokenizer, context_tokens, **kwargs):
        built = original(tokenizer, context_tokens, **kwargs)
        built.token_ids.pop()
        return built

    bad_apis.prompt_builder = short_prompt
    with pytest.raises(module.BenchmarkGateError, match="prompt builder returned"):
        module.run_depth_matrix(
            [_requests(tmp_path)[0]], contexts=(1024,), apis=bad_apis
        )


def test_later_hard_failure_retains_completed_drift_observation(tmp_path: Path) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    original = apis.generate_mtpk

    def drift_d2_then_fail_d3(runtime, prompt_ids, **kwargs):
        result = original(runtime, prompt_ids, **kwargs)
        if kwargs["max_tokens"] == 128 and kwargs["speculative_depth"] == 2:
            result.tokens[-1] += 1
            result.stats = _stats(
                len(prompt_ids),
                depth=2,
                generated_tokens=len(result.tokens),
                tokens=result.tokens,
            )
            result.final_state.generated_token_ids = tuple(result.tokens)
        if kwargs["max_tokens"] == 128 and kwargs["speculative_depth"] == 3:
            accepted = next(
                draft
                for event in result.stats.events
                for draft in event["drafts"]
                if draft.get("accepted") is True
            )
            accepted["correction"] = accepted["token"] + 1
        return result

    apis.generate_mtpk = drift_d2_then_fail_d3
    snapshots = []

    with pytest.raises(module.BenchmarkGateError, match="accepted draft token"):
        module.run_depth_matrix(
            [{**_requests(tmp_path)[0], "depths": (1, 2, 3)}],
            contexts=(1024,),
            checkpoint=snapshots.append,
            apis=apis,
        )

    failed = snapshots[-1]
    assert failed["failure"]["active_cell"]["depth"] == 3
    assert [row["cell"] for row in failed["models"][0]["observations"]] == [
        "ar",
        "d1",
        "d2",
    ]
    depth_two = failed["models"][0]["observations"][-1]
    assert depth_two["ar_comparison"]["first_divergence"] == 127
    assert depth_two["accepted_drafts"] == 85
    assert depth_two["decode_expert_cache_hit_rate"] == pytest.approx(2 / 3)
    assert depth_two["expert_streaming_counters_by_phase"]["decode"] == {
        "expert_hits": 2,
        "expert_misses": 1,
        "hit_rate": pytest.approx(2 / 3),
    }


def test_decode_expert_cache_metrics_are_required_evidence(tmp_path: Path) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    original_load = apis.load

    def load_without_phase_counters(*args, **kwargs):
        runtime = original_load(*args, **kwargs)
        runtime.expert_streaming_snapshot = lambda: {
            "cache": {"expert_hits": 3, "expert_misses": 1, "hit_rate": 0.75}
        }
        return runtime

    apis.load = load_without_phase_counters

    with pytest.raises(module.BenchmarkGateError, match="decode expert-cache"):
        module.run_depth_matrix([_requests(tmp_path)[0]], contexts=(1024,), apis=apis)


def test_decode_expert_cache_metrics_reject_empty_or_stale_ratios() -> None:
    module = _load_module()

    with pytest.raises(module.BenchmarkGateError, match="no routed assignments"):
        module._require_decode_cache_metrics(
            {
                "decode": {
                    "expert_hits": 0,
                    "expert_misses": 0,
                    "hit_rate": 0.0,
                }
            },
            model="hy3-q2",
            depth=1,
        )

    with pytest.raises(module.BenchmarkGateError, match="disagrees"):
        module._require_decode_cache_metrics(
            {
                "decode": {
                    "expert_hits": 2,
                    "expert_misses": 1,
                    "hit_rate": 0.5,
                }
            },
            model="hy3-q2",
            depth=1,
        )


def test_warmup_drift_is_recorded_and_acceptance_counter_errors_fail_closed(
    tmp_path: Path,
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    original = apis.generate_mtpk

    def bad_warmup(runtime, prompt_ids, **kwargs):
        result = original(runtime, prompt_ids, **kwargs)
        if kwargs["max_tokens"] == 8 and kwargs["speculative_depth"] == 1:
            result.tokens[-1] += 1
            result.stats = _stats(
                len(prompt_ids),
                depth=1,
                generated_tokens=len(result.tokens),
                tokens=result.tokens,
            )
            result.final_state.generated_token_ids = tuple(result.tokens)
        return result

    apis.generate_mtpk = bad_warmup
    payload = module.run_depth_matrix(
        [{**_requests(tmp_path)[0], "depths": (1,)}],
        contexts=(1024,),
        apis=apis,
    )
    assert payload["models"][0]["token_divergence_observations"][0]["phase"] == (
        "warmup"
    )

    apis, _calls = _fake_apis(module)
    original = apis.generate_mtpk

    def inconsistent(runtime, prompt_ids, **kwargs):
        result = original(runtime, prompt_ids, **kwargs)
        if kwargs["max_tokens"] == 128:
            result.stats.evaluated_by_depth[0] = 4
            result.stats.evaluated_drafts = sum(result.stats.evaluated_by_depth)
        return result

    apis.generate_mtpk = inconsistent
    with pytest.raises(module.BenchmarkGateError, match="event counts disagree"):
        module.run_depth_matrix([_requests(tmp_path)[0]], contexts=(1024,), apis=apis)


def test_main_writes_and_prints_machine_readable_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    prompt_tail = tmp_path / "tail.txt"
    prompt_tail.write_text("fixed tail", encoding="utf-8")
    output = tmp_path / "matrix.json"

    exit_code = module.main(
        [
            "--model",
            "hy3-q2",
            "--contexts",
            "1024",
            "--hy3-q2-model-root",
            str(tmp_path / "model"),
            "--hy3-q2-manifest",
            str(tmp_path / "manifest.json"),
            "--hy3-q2-mtp-artifacts",
            str(tmp_path / "mtp"),
            "--hy3-q2-prompt-tail",
            str(prompt_tail),
            "--output-json",
            str(output),
        ],
        apis=apis,
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert printed == saved
    assert saved["passed"] is True
    assert saved["status"] == "passed"
    assert saved["active_cell"] is None
    assert [model["model"] for model in saved["models"]] == ["hy3-q2"]


def test_main_persists_completed_cells_when_later_hard_gate_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    original = apis.generate_mtpk

    def fail_retained_d2(runtime, prompt_ids, **kwargs):
        result = original(runtime, prompt_ids, **kwargs)
        if kwargs["max_tokens"] == 128 and kwargs["speculative_depth"] == 2:
            accepted = next(
                draft
                for event in result.stats.events
                for draft in event["drafts"]
                if draft.get("accepted") is True
            )
            accepted["correction"] = accepted["token"] + 1
        return result

    apis.generate_mtpk = fail_retained_d2
    prompt_tail = tmp_path / "tail.txt"
    prompt_tail.write_text("fixed tail", encoding="utf-8")
    output = tmp_path / "partial.json"

    exit_code = module.main(
        [
            "--model",
            "hy3-q2",
            "--contexts",
            "1024",
            "--hy3-q2-model-root",
            str(tmp_path / "model"),
            "--hy3-q2-manifest",
            str(tmp_path / "manifest.json"),
            "--hy3-q2-mtp-artifacts",
            str(tmp_path / "mtp"),
            "--hy3-q2-prompt-tail",
            str(prompt_tail),
            "--output-json",
            str(output),
        ],
        apis=apis,
    )

    assert exit_code == 1
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["passed"] is False
    assert [row["cell"] for row in saved["models"][0]["observations"]] == [
        "ar",
        "d1",
    ]
    assert saved["failure"]["error_type"] == "BenchmarkGateError"
    assert saved["failure"]["active_cell"] == {
        "model": "hy3-q2",
        "context_tokens": 1024,
        "depth": 2,
        "cell": "d2",
        "phase": "retained",
    }
    assert json.loads(capsys.readouterr().out) == saved
    assert list(tmp_path.glob(".partial.json.tmp-*")) == []


def test_main_persists_loader_failure_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)

    def failed_load(*_args, **_kwargs):
        raise RuntimeError("artifact provenance failed")

    apis.load = failed_load
    prompt_tail = tmp_path / "tail.txt"
    prompt_tail.write_text("fixed tail", encoding="utf-8")
    output = tmp_path / "loader.json"

    exit_code = module.main(
        [
            "--model",
            "hy3-q2",
            "--contexts",
            "1024",
            "--hy3-q2-prompt-tail",
            str(prompt_tail),
            "--output-json",
            str(output),
        ],
        apis=apis,
    )

    assert exit_code == 1
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["passed"] is False
    assert saved["models"][0]["observations"] == []
    assert saved["failure"] == {
        "error": "artifact provenance failed",
        "error_type": "RuntimeError",
        "active_cell": {
            "model": "hy3-q2",
            "context_tokens": None,
            "depth": None,
            "cell": None,
            "phase": "load",
        },
    }
    assert json.loads(capsys.readouterr().out) == saved
    assert list(tmp_path.glob(".loader.json.tmp-*")) == []


def test_main_renders_loader_failures_as_machine_readable_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)

    def failed_load(*_args, **_kwargs):
        raise RuntimeError("artifact provenance failed")

    apis.load = failed_load
    prompt_tail = tmp_path / "tail.txt"
    prompt_tail.write_text("fixed tail", encoding="utf-8")

    exit_code = module.main(
        [
            "--model",
            "hy3-q2",
            "--contexts",
            "1024",
            "--hy3-q2-prompt-tail",
            str(prompt_tail),
        ],
        apis=apis,
    )

    assert exit_code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema"] == "mtplx-q2-bf16-mtp-depth-matrix-v3"
    assert printed["status"] == "failed"
    assert printed["passed"] is False
    assert printed["models"][0]["observations"] == []
    assert printed["failure"] == {
        "error": "artifact provenance failed",
        "error_type": "RuntimeError",
        "active_cell": {
            "model": "hy3-q2",
            "context_tokens": None,
            "depth": None,
            "cell": None,
            "phase": "load",
        },
    }
