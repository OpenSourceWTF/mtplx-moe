from __future__ import annotations

import importlib.util
from pathlib import Path
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
