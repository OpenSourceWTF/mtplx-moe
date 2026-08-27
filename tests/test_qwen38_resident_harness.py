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
