from __future__ import annotations

import os
from argparse import Namespace
from types import SimpleNamespace

import pytest

import mtplx.generation as generation
import mtplx.runtime as runtime
from mtplx.benchmarks.programming_prompts import (
    PROGRAMMING_ARTIFACT_KINDS,
    build_programming_context,
    programming_context_stats,
)
from mtplx.prefill_bench import (
    DEFAULT_FINAL_REQUEST,
    _prompt_build_for_context,
    _token_ids_for_context,
    run_prefill_ladder,
)


class _CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token)) for token in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = ""
        for message in messages:
            text += f"<{message['role']}>\n{message['content']}\n</{message['role']}>\n"
        if add_generation_prompt:
            text += "<assistant>\n"
        return self.encode(text) if tokenize else text


def test_programming_context_is_deterministic_and_structurally_varied() -> None:
    first = build_programming_context(minimum_characters=80_000)
    second = build_programming_context(minimum_characters=80_000)

    assert first == second
    assert len(first) >= 80_000
    stats = programming_context_stats(first)
    assert set(stats["artifact_kinds"]) == set(PROGRAMMING_ARTIFACT_KINDS)
    assert stats["artifact_count"] >= len(PROGRAMMING_ARTIFACT_KINDS) * 2
    assert stats["largest_duplicate_count"] <= 2
    for phrase in ("def ", "class ", "pytest", "README", "pyproject.toml"):
        assert phrase in first


def test_programming_context_rejects_non_positive_target() -> None:
    with pytest.raises(ValueError, match="minimum_characters must be positive"):
        build_programming_context(minimum_characters=0)


def test_prefill_prompt_preserves_coherent_tail() -> None:
    tokenizer = _CharTokenizer()

    prompt = _prompt_build_for_context(tokenizer, 4096)
    text = tokenizer.decode(prompt.token_ids)

    assert len(prompt.token_ids) == 4096
    assert DEFAULT_FINAL_REQUEST in text
    assert text.endswith("<assistant>\n")
    assert prompt.metadata["prompt_policy"] == "coding_agent_tail_v2"
    assert prompt.metadata["prompt_format"] == "chat"
    assert prompt.metadata["prompt_enable_thinking"] is False
    assert prompt.metadata["prompt_tail_preserved"] is True
    assert prompt.metadata["prompt_release_valid"] is True
    assert prompt.metadata["prompt_filler_tokens"] > 0
def test_prefill_prompt_legacy_mode_keeps_diagnostic_hard_truncate() -> None:
    tokenizer = _CharTokenizer()

    prompt = _prompt_build_for_context(tokenizer, 256, prompt_style="legacy-repeat")

    assert len(prompt.token_ids) == 256
    assert prompt.metadata["prompt_policy"] == "legacy_repeat_hard_truncate"
    assert prompt.metadata["prompt_tail_preserved"] is False
    assert prompt.metadata["prompt_release_valid"] is False


def test_token_ids_for_context_accepts_custom_tail() -> None:
    tokenizer = _CharTokenizer()
    tail = "\n\n# Final user request\nPatch the benchmark harness.\n"

    ids = _token_ids_for_context(tokenizer, 1024, prompt_tail=tail)

    text = tokenizer.decode(ids)
    assert tail in text
    assert text.endswith("<assistant>\n")


def test_run_prefill_ladder_fake_runtime_records_release_valid_prompt(
    monkeypatch,
) -> None:
    import mtplx.prefill_bench as prefill_bench

    tokenizer = _CharTokenizer()
    tail = "\n\n# Final user request\nPatch the benchmark harness.\n"
    captured: dict[str, object] = {}
    cleanup_calls = {"count": 0}

    def fake_load(model: str, *, mtp: bool):
        assert model == "fake-model"
        assert mtp is True
        return SimpleNamespace(tokenizer=tokenizer)

    def fake_generate_mtpk(rt, prompt_ids, **kwargs):
        captured["prompt_text"] = rt.tokenizer.decode(prompt_ids)
        captured["prefill_layout_env"] = os.environ.get("MTPLX_SUSTAINED_PREFILL_LAYOUT")
        callback = kwargs.get("token_callback")
        if callback is not None:
            callback([101])
        return SimpleNamespace(
            tokens=[101, 102],
            stats={
                "generated_tokens": 2,
                "prompt_eval_time_s": 0.5,
                "elapsed_s": 0.7,
                "prompt_tps": 512.0,
                "accepted_drafts": 3,
                "drafted_tokens": 4,
                "speculative_depth": 2,
                "requested_speculative_depth": 3,
                "long_context_mtp_depth_policy": {
                    "policy": "auto",
                    "active": True,
                    "reason": "long_context_depth_cap",
                },
                "verify_calls": 1,
                "verify_time_s": 0.01,
                "draft_time_s": 0.02,
                "peak_memory_bytes": 1024**3,
            },
        )

    monkeypatch.setattr(runtime, "load", fake_load)
    monkeypatch.setattr(generation, "generate_mtpk", fake_generate_mtpk)
    monkeypatch.setattr(
        prefill_bench,
        "_sync_and_clear_cache_between_contexts",
        lambda: cleanup_calls.__setitem__("count", cleanup_calls["count"] + 1) or 0.123,
    )
    before_env = dict(os.environ)
    try:
        payload = run_prefill_ladder(
            Namespace(
                contexts="512,1k",
                full=False,
                profile="sustained",
                model="fake-model",
                generation_mode="mtp",
                max_tokens=2,
                dry_run=False,
                prompt_style="coding-agent",
                prompt_format="chat",
                prefill_layout="contiguous-dense-decode",
                prompt_tail=tail,
                prompt_tail_file=None,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                draft_temperature=None,
                draft_top_p=None,
                draft_top_k=None,
                speculative_depth=3,
                seed=0,
                fanmax=False,
                disable_thinking=True,
                enable_thinking=False,
            )
        )
    finally:
        os.environ.clear()
        os.environ.update(before_env)

    assert tail in str(captured["prompt_text"])
    assert str(captured["prompt_text"]).endswith("<assistant>\n")
    assert captured["prefill_layout_env"] == "contiguous_dense_decode"
    assert payload["prefill_layout"]["requested"] == "contiguous-dense-decode"
    assert payload["prefill_layout"]["env_value"] == "contiguous_dense_decode"
    assert payload["seed"] == 0
    assert payload["vary_seed_by_context"] is False
    assert payload["inter_context_cache_cleanup"]["enabled"] is True
    assert payload["inter_context_cache_cleanup"]["events"] == 1
    assert payload["inter_context_cache_cleanup"]["time_s"] == 0.123
    assert cleanup_calls["count"] == 1
    assert payload["prompt"]["release_valid"] is True
    assert payload["prompt"]["format"] == "chat"
    assert payload["prompt"]["enable_thinking"] is False
    assert payload["recommended_plugged_in_commands"]
    assert "--prefill-layout contiguous-dense-decode" in payload[
        "recommended_plugged_in_commands"
    ][0]
    row = payload["rows"][0]
    assert row["post_row_inter_context_cache_cleanup_time_s"] == 0.123
    assert row["requested_prefill_layout"] == "contiguous-dense-decode"
    assert row["prompt_release_valid"] is True
    assert row["prompt_tail_preserved"] is True
    assert row["prompt_tail_sha256"] == payload["prompt"]["tail_sha256"]
    assert row["generated_tokens"] == 2
    assert row["seed"] == 0
    assert row["speculative_depth"] == 2
    assert row["requested_speculative_depth"] == 3
    assert row["long_context_mtp_depth_policy"]["reason"] == "long_context_depth_cap"
