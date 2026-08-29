from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def _harness():
    path = Path(__file__).parents[1] / "scripts/qwen38_flash_next_oq4_harness.py"
    spec = importlib.util.spec_from_file_location("qwen38_resident_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harness_loads_resident_model_and_only_configures_ngram_cache() -> None:
    harness = _harness()
    assert sys.path[0] == str(harness.ROOT)
    args = SimpleNamespace(
        mode="mtp",
        prompt_tokens=32,
        max_tokens=8,
        ngram_cache_gib=1,
        runtime_target_gib=75,
    )

    kwargs = harness._resident_load_kwargs(args)

    assert kwargs == {
        "mtp": True,
        "ngram_cache_limit_bytes": 1024**3,
        "ngram_context_tokens": 40,
        "ngram_target_residency_bytes": 75 * 1024**3,
    }
    assert not any("expert" in key for key in kwargs)


def test_harness_defaults_to_smallest_viable_resident_target() -> None:
    harness = _harness()

    args = harness._parse_args(["--smoke", "--preflight-only"])

    assert args.runtime_target_gib == 82
    assert args.profile == "sustained"


def test_production_workload_is_16k_1k_thinking_sampler_contract() -> None:
    harness = _harness()

    args = harness._parse_args([])

    assert (args.prompt_tokens, args.max_tokens) == (16_384, 1_024)
    assert args.ngram_cache_gib == 1
    assert harness._benchmark_lane_environment(args) == {
        "MTPLX_COMPILED_VERIFY": "1",
        "MTPLX_FUSE_PROJ": "gdn,attn,hyper,ple",
        "MTPLX_QWEN4_WHOLE_MOE_M2": "1",
        "MTPLX_QWEN4_HYPER_M2": "1",
    }
    assert harness._sampler_contract(headline=False) == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    assert harness._sampler_contract(headline=True) == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }


def test_guarded_child_waits_for_stopped_service_memory_to_become_reclaimable() -> None:
    harness = _harness()
    readings = iter((79 * 1024**3, 81 * 1024**3, 82 * 1024**3))
    sleeps = []

    available = harness._wait_for_resident_target_memory(
        82 * 1024**3,
        timeout_s=10,
        read_available=lambda: next(readings),
        sleep=lambda seconds: sleeps.append(seconds),
        now=iter((0.0, 1.0, 2.0)).__next__,
    )

    assert available == 82 * 1024**3
    assert sleeps == [1.0, 1.0]


def test_smoke_uses_short_instruction_without_changing_exact_fixture(tmp_path) -> None:
    harness = _harness()
    prompt_file = tmp_path / "prompt.jsonl"
    prompt_file.write_text('{"prompt":"an intentionally long exact workload"}\n')

    assert harness._run_instruction(prompt_file, smoke=True) == (
        "Write a Python function that adds two integers."
    )
    assert harness._run_instruction(prompt_file, smoke=False) == (
        "an intentionally long exact workload"
    )


def test_exact_prompt_keeps_constructed_ids_when_decode_is_not_canonical() -> None:
    harness = _harness()

    class BoundaryTokenizer:
        def __init__(self) -> None:
            self.encoded: list[str] = []

        def apply_chat_template(self, messages, **_kwargs):
            assert messages[0]["content"] == harness._CONTENT_SENTINEL
            return f"P{harness._CONTENT_SENTINEL}S"

        def encode(self, text):
            self.encoded.append(text)
            return [ord(char) for char in text]

        def decode(self, _tokens):
            # BPE decode is not generally a canonical inverse of encode.
            return "decoded noncanonical prompt"

    tokenizer = BoundaryTokenizer()
    prompt, tokens = harness.build_exact_python_prompt(
        tokenizer,
        context="context",
        instruction="do it",
        target_tokens=16,
        reasoning_effort="low",
    )

    assert prompt == "decoded noncanonical prompt"
    assert len(tokens) == 16
    assert prompt not in tokenizer.encoded


def test_outer_harness_retries_only_pre_child_canonical_lock_race() -> None:
    harness = _harness()
    args = harness._parse_args(["--smoke"])
    calls = []
    waits = []

    def call(command, *, cwd):
        calls.append((command, cwd))
        return 1 if len(calls) == 1 else 0

    result = harness._call_guard_with_race_retry(
        args,
        call=call,
        lock_owned=lambda _path: True,
        wait_for_service=lambda _timeout: waits.append(_timeout),
    )

    assert result == 0
    assert len(calls) == 2
    assert waits == [args.lock_timeout_seconds]


def test_headline_forwards_requested_warmup_runs() -> None:
    harness = _harness()
    args = harness._parse_args(["--headline", "--warmup-runs", "1"])

    command = harness._outer_command(args)

    assert args.warmup_runs == 1
    warmup_index = command.index("--warmup-runs")
    assert command[warmup_index + 1] == "1"


def test_outer_harness_forwards_explicit_stock_lane_control() -> None:
    harness = _harness()
    args = harness._parse_args(
        [
            "--smoke",
            "--fuse-proj",
            "none",
            "--no-whole-moe-m2",
            "--no-hyper-m2",
            "--no-compiled-verify",
        ]
    )

    command = harness._outer_command(args)

    assert command[command.index("--fuse-proj") + 1] == "none"
    assert "--no-whole-moe-m2" in command
    assert "--no-hyper-m2" in command
    assert "--no-compiled-verify" in command
    assert harness._benchmark_lane_environment(args) == {
        "MTPLX_COMPILED_VERIFY": "0",
    }


def test_receipt_includes_existing_generation_timing_breakdown() -> None:
    harness = _harness()
    stats = SimpleNamespace(
        verify_target_distribution_time_s=1.25,
        accept_time_s=2.5,
        repair_time_s=3.75,
        pre_first_token_setup_s=4.0,
    )

    receipt = harness._diagnostic_timing_receipt(stats)

    assert receipt["verify_target_distribution_time_s"] == 1.25
    assert receipt["accept_time_s"] == 2.5
    assert receipt["repair_time_s"] == 3.75
    assert receipt["pre_first_token_setup_s"] == 4.0
    assert receipt["verify_joint_eval_time_s"] == 0.0


def test_receipt_includes_existing_speculation_and_prompt_copy_counts() -> None:
    harness = _harness()
    stats = SimpleNamespace(
        speculative_depth=3,
        requested_speculative_depth=3,
        context_copy_active=True,
        context_copy_rounds=4,
        context_copy_drafted_tokens=70,
        context_copy_accepted_tokens=61,
    )

    receipt = harness._speculation_receipt(stats)

    assert receipt == {
        "speculative_depth": 3,
        "requested_speculative_depth": 3,
        "context_copy_active": True,
        "context_copy_rounds": 4,
        "context_copy_drafted_tokens": 70,
        "context_copy_accepted_tokens": 61,
    }
