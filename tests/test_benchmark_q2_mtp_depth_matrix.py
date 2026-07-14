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
    def __init__(self, model_key: str) -> None:
        self.model_key = model_key
        self.mtp_enabled = True
        self.tokenizer = object()
        self.expert_streaming = _FakeStreaming()
        self.admissions: list[int] = []
        self.closed = False

    @contextlib.contextmanager
    def admit_kv_tokens(self, tokens: int):
        self.admissions.append(tokens)
        yield

    def expert_streaming_snapshot(self):
        return {
            "cache": {
                "route_calls": self.expert_streaming.reset_calls,
                "expert_hits": 3,
                "expert_misses": 1,
            }
        }

    def expert_resource_telemetry_snapshot(self):
        return {
            "cache": {"expert_bytes_read": 4096, "read_operations": 2},
            "cache_by_phase": {"decode": {"expert_bytes_read": 4096}},
            "reader": {"active_reads": 0, "peak_active_reads": 2},
            "expert_pipeline": {"completion_fences": 3},
            "mlx_memory": {"active_memory_bytes": 1024},
        }

    def close(self) -> None:
        self.closed = True


def _stats(
    prompt_tokens: int,
    *,
    depth: int = 0,
    generated_tokens: int = 128,
):
    accepted = [max(0, 5 - index) for index in range(depth)]
    evaluated = [6 - index for index in range(depth)]
    drafted = [6 for _ in range(depth)]
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
        evaluated_drafts=sum(evaluated),
        drafted_tokens=sum(drafted),
        accepted_by_depth=accepted,
        evaluated_by_depth=evaluated,
        drafted_by_depth=drafted,
        mean_accept_probability_by_depth=[0.75 for _ in range(depth)],
        fully_accepted_verify_calls=2 if depth else 0,
        verify_calls=4 if depth else 127,
        requested_speculative_depth=depth,
        speculative_depth=depth,
        mtp_history_policy="committed" if depth else "none",
        repetition_stop_triggered=False,
        loop_guard={},
        peak_memory_bytes=3 * 1024**3,
    )


def _fake_apis(module, *, output_tokens: int = 128, mismatch_depth: int | None = None):
    calls = SimpleNamespace(
        loads=[],
        configs=[],
        prompt_builds=[],
        ar=[],
        mtpk=[],
        runtimes=[],
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
        runtime = _FakeRuntime(kwargs["expert_streaming_config"]["model_key"])
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
            stats=_stats(len(prompt_ids), generated_tokens=len(tokens)),
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
            ),
        )

    return (
        module.RunnerAPIs(
            load=load,
            config_factory=config_factory,
            parse_memory_bytes=parse_memory,
            prompt_builder=prompt_builder,
            sampler_factory=sampler_factory,
            generate_ar=generate_ar,
            generate_mtpk=generate_mtpk,
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
    assert args.hy3_depths == (1, 2, 3, 4)
    assert args.glm52_depths == (1, 2, 3, 4, 5)
    assert args.memory_limit == "112GiB"
    assert args.runtime_reserve == "12GiB"
    assert args.expert_cache_limit == "64GiB"
    assert args.max_live_kv_tokens == 4096


def test_matrix_loads_each_model_once_and_uses_only_canonical_generators(
    tmp_path: Path,
) -> None:
    module = _load_module()
    apis, calls = _fake_apis(module)

    payload = module.run_depth_matrix(_requests(tmp_path), apis=apis)

    assert payload["schema"] == "mtplx-q2-bf16-mtp-depth-matrix-v1"
    assert payload["passed"] is True
    assert [config["model_key"] for config in calls.configs] == [
        "hy3-expert-q2",
        "glm52-expert-q2",
    ]
    assert len(calls.loads) == 2
    assert all(kwargs["mtp"] is True for _root, kwargs in calls.loads)
    assert all(kwargs["mtp_precision"] == "bf16" for _root, kwargs in calls.loads)
    assert len(calls.ar) == 8
    assert len(calls.mtpk) == 36
    assert [len(model["observations"]) for model in payload["models"]] == [10, 12]
    assert [model["discarded_warmup_count"] for model in payload["models"]] == [
        10,
        12,
    ]
    assert all(
        row["generated_tokens"] == 128
        for model in payload["models"]
        for row in model["observations"]
    )

    for runtime in calls.runtimes:
        model = next(
            row for row in payload["models"] if row["model_key"] == runtime.model_key
        )
        assert runtime.expert_streaming.reset_calls == 2 * len(model["observations"])
        cells = 5 if runtime.model_key == "hy3-expert-q2" else 6
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
    assert sum(kwargs["max_tokens"] == 8 for *_rest, kwargs in calls.mtpk) == 18

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


def test_rows_recompute_ingestion_decode_and_acceptance_metrics(tmp_path: Path) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)

    payload = module.run_depth_matrix(
        [_requests(tmp_path)[0]], contexts=(1024,), apis=apis
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
    assert depth_two["peak_memory_bytes"] == 3 * 1024**3
    assert depth_two["expert_resource_telemetry"] == {
        "cache": {"expert_bytes_read": 4096, "read_operations": 2},
        "cache_by_phase": {"decode": {"expert_bytes_read": 4096}},
        "reader": {"active_reads": 0, "peak_active_reads": 2},
        "expert_pipeline": {"completion_fences": 3},
        "mlx_memory": {"active_memory_bytes": 1024},
    }
    assert depth_two["accepted_drafts"] == 9
    assert depth_two["evaluated_drafts"] == 11
    assert depth_two["drafted_tokens"] == 12
    assert depth_two["conditional_hit_rate"] == pytest.approx(9 / 11)
    assert depth_two["cumulative_accepted_drafted_yield"] == 0.75
    assert depth_two["accepted_per_verify"] == 2.25
    assert depth_two["fully_accepted_verify_ratio"] == 0.5
    assert depth_two["acceptance_by_depth"] == [
        {
            "depth": 1,
            "drafted": 6,
            "evaluated": 6,
            "accepted": 5,
            "conditional_hit_rate": pytest.approx(5 / 6),
            "cumulative_accepted_drafted_yield": pytest.approx(5 / 6),
            "mean_accept_probability": 0.75,
        },
        {
            "depth": 2,
            "drafted": 6,
            "evaluated": 5,
            "accepted": 4,
            "conditional_hit_rate": 0.8,
            "cumulative_accepted_drafted_yield": pytest.approx(4 / 6),
            "mean_accept_probability": 0.75,
        },
    ]
    assert depth_two["gates"] == {
        "prompt_length_exact": True,
        "new_prefill_tokens_exact": True,
        "output_tokens_exact": True,
        "generated_count_consistent": True,
        "length_finish": True,
        "ar_token_parity": True,
        "ar_finish_reason_parity": True,
        "requested_depth_exact": True,
        "effective_depth_exact": True,
        "committed_history": True,
        "guards_disabled": True,
    }


def test_exact_prompt_and_output_gates_fail_closed(tmp_path: Path) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module, output_tokens=127)
    with pytest.raises(module.BenchmarkGateError, match="exactly 128"):
        module.run_depth_matrix([_requests(tmp_path)[0]], contexts=(1024,), apis=apis)

    apis, _calls = _fake_apis(module, mismatch_depth=2)
    with pytest.raises(module.BenchmarkGateError, match="diverged from AR"):
        module.run_depth_matrix([_requests(tmp_path)[0]], contexts=(1024,), apis=apis)

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


def test_warmup_parity_and_acceptance_counter_errors_fail_closed(
    tmp_path: Path,
) -> None:
    module = _load_module()
    apis, _calls = _fake_apis(module)
    original = apis.generate_mtpk

    def bad_warmup(runtime, prompt_ids, **kwargs):
        result = original(runtime, prompt_ids, **kwargs)
        if kwargs["max_tokens"] == 8 and kwargs["speculative_depth"] == 1:
            result.tokens[-1] += 1
        return result

    apis.generate_mtpk = bad_warmup
    with pytest.raises(module.BenchmarkGateError, match="diverged from AR"):
        module.run_depth_matrix([_requests(tmp_path)[0]], contexts=(1024,), apis=apis)

    apis, _calls = _fake_apis(module)
    original = apis.generate_mtpk

    def inconsistent(runtime, prompt_ids, **kwargs):
        result = original(runtime, prompt_ids, **kwargs)
        if kwargs["max_tokens"] == 128:
            result.stats.evaluated_by_depth[0] = 4
            result.stats.evaluated_drafts = sum(result.stats.evaluated_by_depth)
        return result

    apis.generate_mtpk = inconsistent
    with pytest.raises(module.BenchmarkGateError, match="accepted <= evaluated"):
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
    assert [model["model"] for model in saved["models"]] == ["hy3-q2"]


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
    assert json.loads(capsys.readouterr().out) == {
        "schema": "mtplx-q2-bf16-mtp-depth-matrix-v1",
        "passed": False,
        "error": "artifact provenance failed",
        "error_type": "RuntimeError",
    }
