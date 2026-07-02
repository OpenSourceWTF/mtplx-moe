from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.generation import (
    _clear_cache_every,
    _defer_verify_hidden_eval_enabled,
    _make_target_prefill_cache,
    _maybe_repage_target_prefill_cache,
    _prefill,
    _prefill_chunk_cache_cleanup_every,
    _prefill_chunk_size,
    _prefill_committed_mtp_history_streaming,
    _sustained_prefill_layout,
    generate_ar,
    generate_mtpk,
    restore_or_prefill_prompt_state,
)
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig


class TinyTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(token)) for token in tokens)


class TinyModel:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        position_offset=None,
    ):
        return hidden_states

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        self.calls.append(
            {
                "tokens": int(input_ids.shape[1]),
                "return_hidden": bool(return_hidden),
                "emit_logits": bool(emit_logits),
                "logits_keep": logits_keep,
            }
        )
        length = int(input_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if not emit_logits:
            if return_hidden:
                return None, hidden
            return None
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = mx.zeros((1, keep, 4), dtype=mx.float32)
        logits = logits + mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


class KwargsOnlyTinyModel(TinyModel):
    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        **kwargs,
    ):
        return super().__call__(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
            **kwargs,
        )


class AcceptingTinyMTPModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        position_offset=None,
    ):
        length = int(next_token_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        logits = mx.zeros((1, length, 4), dtype=mx.float32)
        logits = logits + mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


class OffsetCache:
    def __init__(self):
        self.offset = 0
        self.trimmed: list[int] = []

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(int(self.offset), int(n))
        self.offset -= n
        self.trimmed.append(n)
        return n


class RejectingTinyMTPModel(AcceptingTinyMTPModel):
    def __init__(self):
        super().__init__()
        self.target_cache = [OffsetCache()]

    def make_cache(self):
        return self.target_cache

    def __call__(self, input_ids, *, cache=None, **kwargs):
        if cache:
            for entry in cache:
                entry.offset += int(input_ids.shape[1])
        return super().__call__(input_ids, cache=cache, **kwargs)

    def mtp_forward(self, *args, **kwargs):
        result = super().mtp_forward(*args, **kwargs)
        if isinstance(result, tuple):
            logits, hidden = result
            logits = mx.zeros_like(logits) + mx.array(
                [0.0, 0.0, 1.0, 0.0],
                dtype=mx.float32,
            )
            return logits, hidden
        return mx.zeros_like(result) + mx.array(
            [0.0, 0.0, 1.0, 0.0],
            dtype=mx.float32,
        )


def _runtime(model: TinyModel, *, mtp_enabled: bool = True) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=TinyTokenizer(),
        model_path=Path("tiny"),
        mtp_enabled=mtp_enabled,
        contract=MTPContract(),
    )


def test_contiguous_then_repage_cache_layout_restores_paged_env(monkeypatch):
    cache: list[object] = []
    events: list[tuple[str, str | None]] = []

    class Runtime:
        def make_cache(self):
            events.append(("make_cache", os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN")))
            return cache

    def configure(received_cache):
        events.append(("repage", os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN")))
        assert received_cache is cache
        return {"enabled": 1, "entries": 0, "skipped": 0}

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_then_repage")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_OWNED_ATTN_KV", "1")
    monkeypatch.setenv("MTPLX_BLOCK_OWNED_ATTN_KV", "1")
    monkeypatch.setattr(
        "mtplx.cache_state.configure_tail_owned_attention_kv_cache",
        configure,
    )

    made_cache = _make_target_prefill_cache(Runtime())
    elapsed = _maybe_repage_target_prefill_cache(made_cache)

    assert elapsed >= 0.0
    assert events == [("make_cache", "0"), ("repage", "1")]
    assert os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert os.environ["MTPLX_OWNED_ATTN_KV"] == "1"
    assert os.environ["MTPLX_BLOCK_OWNED_ATTN_KV"] == "1"


def test_contiguous_dense_decode_cache_layout_does_not_repage(monkeypatch):
    cache: list[object] = []
    events: list[tuple[str, str | None]] = []

    class Runtime:
        def make_cache(self):
            events.append(("make_cache", os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN")))
            return cache

    def configure(_received_cache):
        raise AssertionError("dense decode layout must not repage after prefill")

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_dense_decode")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_OWNED_ATTN_KV", "1")
    monkeypatch.setenv("MTPLX_BLOCK_OWNED_ATTN_KV", "1")
    monkeypatch.setattr(
        "mtplx.cache_state.configure_tail_owned_attention_kv_cache",
        configure,
    )

    made_cache = _make_target_prefill_cache(Runtime())
    elapsed = _maybe_repage_target_prefill_cache(made_cache)

    assert elapsed == 0.0
    assert events == [("make_cache", "0")]
    assert os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert os.environ["MTPLX_OWNED_ATTN_KV"] == "1"
    assert os.environ["MTPLX_BLOCK_OWNED_ATTN_KV"] == "1"


def test_session_restore_uses_prefill_layout_cache_factory(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_dense_decode")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    captured: dict[str, object] = {}

    class Bank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return SimpleNamespace(prefix_len=3)

        def restore(self, _rt, _prompt_ids, **kwargs):
            captured.update(kwargs)
            cache_factory = kwargs["cache_factory"]
            assert callable(cache_factory)
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=cache_factory(),
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4],
        mtp_history_policy="committed",
        session_bank=Bank(),
        restore_mode="reference_lease",
    )

    assert captured["mode"] == "clone"
    assert captured["cache_factory"] is not None
    assert prompt_state.cache_hit is True
    assert prompt_state.restore_mode == "clone"


def test_live_frontier_reference_restore_survives_prefill_layout_factory(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_dense_decode")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    captured: dict[str, object] = {}

    class Bank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return SimpleNamespace(prefix_len=3)

        def restore(self, _rt, _prompt_ids, **kwargs):
            captured.update(kwargs)
            assert callable(kwargs["cache_factory"])
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=["live-frontier-cache"],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="reference_lease",
            )

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4],
        mtp_history_policy="committed",
        session_bank=Bank(),
        restore_mode="reference_lease",
    )

    assert captured["mode"] == "reference_lease"
    assert captured["cache_factory"] is not None
    assert prompt_state.cache_hit is True
    assert prompt_state.restore_mode == "reference_lease"


def test_auto_sustained_prefill_policy_keeps_dense_decode_through_128k(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "2048")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "2048")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY", "auto")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "auto")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY", "auto")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD", "16384")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT", "256")

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "65536")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"
    assert _prefill_chunk_size() == 2048
    assert _prefill_chunk_cache_cleanup_every() == 1
    assert _defer_verify_hidden_eval_enabled() is True
    assert _clear_cache_every() == 256

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "131072")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"
    assert _prefill_chunk_size() == 2048
    assert _prefill_chunk_cache_cleanup_every() == 1
    assert _defer_verify_hidden_eval_enabled() is True
    assert _clear_cache_every() == 256

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "196608")
    assert _sustained_prefill_layout() == "contiguous_then_repage"
    assert _prefill_chunk_cache_cleanup_every() == 2
    assert _clear_cache_every() == 0


def test_auto_sustained_prefill_policy_repages_when_paged_kv_quant_is_enabled(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "131072")
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "65536")

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "q8")
    assert _sustained_prefill_layout() == "contiguous_then_repage"

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "q4")
    assert _sustained_prefill_layout() == "contiguous_then_repage"

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "off")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"


def test_non_sustained_long_context_prefill_is_blocked_before_full_hidden_eval(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
    monkeypatch.delenv("MTPLX_ALLOW_UNSAFE_LONG_CONTEXT_PREFILL", raising=False)
    monkeypatch.setenv("MTPLX_UNSAFE_LONG_CONTEXT_PREFILL_GUARD_TOKENS", "8")
    model = TinyModel()

    with pytest.raises(
        RuntimeError, match="Blocked unsafe long-context MTP prefill path"
    ):
        restore_or_prefill_prompt_state(
            _runtime(model, mtp_enabled=True),
            list(range(8)),
            mtp_history_policy="committed",
        )

    assert model.calls == []


def test_non_sustained_long_context_prefill_guard_has_explicit_escape_hatch(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
    monkeypatch.setenv("MTPLX_ALLOW_UNSAFE_LONG_CONTEXT_PREFILL", "1")
    monkeypatch.setenv("MTPLX_UNSAFE_LONG_CONTEXT_PREFILL_GUARD_TOKENS", "8")
    model = TinyModel()

    restore_or_prefill_prompt_state(
        _runtime(model, mtp_enabled=True),
        list(range(8)),
        mtp_history_policy="committed",
    )

    assert model.calls


def test_generate_ar_does_not_request_hidden_by_default(monkeypatch):
    monkeypatch.delenv("MTPLX_AR_RETURN_HIDDEN", raising=False)
    monkeypatch.delenv("MTPLX_DIAGNOSTIC_AR_RETURN_HIDDEN", raising=False)
    model = TinyModel()

    out = generate_ar(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        stop_token_ids=set(),
    )

    assert out.stats.ar_return_hidden is False
    assert out.stats.forward_ar_hidden_calls == 0
    assert out.stats.forward_ar_plain_calls >= 1
    assert out.stats.prompt_target_prefill_time_s == out.stats.prompt_eval_time_s
    assert out.stats.prompt_mtp_history_time_s == 0.0
    assert out.stats.prompt_target_prefill_tok_s > 0.0
    assert out.stats.tok_s == out.stats.decode_tok_s
    assert out.stats.decode_elapsed_s == pytest.approx(
        out.stats.elapsed_s - out.stats.prompt_eval_time_s
    )
    assert out.stats.end_to_end_tok_s <= out.stats.decode_tok_s
    assert all(call["return_hidden"] is False for call in model.calls)


def test_lazy_bonus_verify_shortens_full_accept_verify_input(monkeypatch):
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    model = AcceptingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=5,
        sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=20),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens[:4] == [1, 1, 1, 1]
    assert len(out.tokens) == 5
    assert [call["tokens"] for call in model.calls] == [1, 3, 1]
    assert out.stats.verify_calls == 1
    assert out.stats.commit_time_s > 0.0
    assert out.stats.events[0]["lazy_bonus_verify"]["enabled"] is True
    assert out.stats.events[0]["lazy_bonus_verify"]["verify_input_tokens"] == 3
    assert out.stats.events[0]["defer_verify_hidden_eval"]["rows"] == 3


def test_lazy_target_distributions_inline_bonus_avoids_bonus_reforward(monkeypatch):
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    monkeypatch.setenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", "1")
    model = AcceptingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=5,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens[:4] == [1, 1, 1, 1]
    assert len(out.tokens) == 5
    assert [call["tokens"] for call in model.calls] == [1, 4]
    assert out.stats.verify_calls == 1
    assert out.stats.lazy_bonus_commit_time_s == 0.0
    assert out.stats.events[0]["lazy_bonus_verify"]["enabled"] is False
    assert (
        out.stats.events[0]["lazy_bonus_verify"]["disabled_by"]
        == "lazy_target_distributions"
    )
    assert out.stats.events[0]["lazy_bonus_verify"]["verify_input_tokens"] == 4
    assert "lazy_bonus_commit_forward" not in out.stats.events[0].get("timing_s", {})
    assert out.stats.events[0]["target_distribution_materialized"]["mode"] == (
        "lazy_accept_bonus_path"
    )


def test_lazy_target_distributions_stop_after_first_rejection(monkeypatch):
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    monkeypatch.setenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", "1")
    model = RejectingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=3,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens[:1] == [1]
    assert out.stats.events[0]["rejected_at_depth"] == 1
    assert out.stats.target_distribution_materialized_rows == 1
    assert out.stats.target_distribution_materialized_windows == 1
    assert out.stats.events[0]["target_distribution_materialized"]["rows"] == 1


@pytest.mark.parametrize(
    ("model_cls", "sampler"),
    [
        (AcceptingTinyMTPModel, SamplerConfig(temperature=0.0, top_p=1.0, top_k=20)),
        (RejectingTinyMTPModel, SamplerConfig(temperature=0.6, top_p=1.0, top_k=1)),
        (AcceptingTinyMTPModel, SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)),
    ],
)
def test_lazy_target_distributions_match_dense_reference(
    monkeypatch,
    model_cls,
    sampler,
):
    def run_once(*, lazy: bool):
        monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
        monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
        monkeypatch.delenv("MTPLX_LAZY_BONUS_VERIFY", raising=False)
        if lazy:
            monkeypatch.setenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", "1")
        else:
            monkeypatch.delenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", raising=False)
        return generate_mtpk(
            _runtime(model_cls(), mtp_enabled=True),
            [0],
            max_tokens=5,
            sampler=sampler,
            speculative_depth=3,
            mtp_history_policy="committed",
            verify_strategy="batched",
            stop_token_ids=set(),
            seed=123,
        )

    dense = run_once(lazy=False)
    lazy = run_once(lazy=True)

    assert lazy.tokens == dense.tokens
    assert lazy.stats.accepted_by_depth == dense.stats.accepted_by_depth
    assert lazy.stats.drafted_by_depth == dense.stats.drafted_by_depth
    assert lazy.stats.rejected_drafts == dense.stats.rejected_drafts
    assert lazy.stats.bonus_tokens == dense.stats.bonus_tokens
    assert lazy.finish_reason == dense.finish_reason


def test_lazy_bonus_verify_skips_d1_by_default(monkeypatch):
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    model = AcceptingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=1,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens == [1, 1, 1]
    lazy = out.stats.events[0]["lazy_bonus_verify"]
    assert lazy["enabled"] is False
    assert lazy["min_depth"] == 2
    assert lazy["verify_input_tokens"] == 2
    assert "lazy_bonus_commit_forward" not in out.stats.events[0].get("timing_s", {})


def test_omit_speculative_bonus_skips_bonus_distribution_row(monkeypatch):
    monkeypatch.setenv("MTPLX_OMIT_SPECULATIVE_BONUS", "1")
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "1")
    model = AcceptingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
        speculative_depth=1,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )

    assert out.tokens == [1, 1]
    assert out.stats.target_distribution_materialized_rows == 1
    assert out.stats.events[0]["speculative_bonus"] == {
        "omitted": True,
        "distribution_row_needed": False,
    }
    assert out.stats.events[0]["defer_verify_hidden_eval"]["rows"] == 1
    assert "bonus_token" not in out.stats.events[0]
    assert out.stats.bonus_tokens == 0


def test_trim_commit_keeps_rejected_verify_prefix_without_reforward(monkeypatch):
    monkeypatch.delenv("MTPLX_LAZY_BONUS_VERIFY", raising=False)
    model = RejectingTinyMTPModel()

    out = generate_mtpk(
        _runtime(model, mtp_enabled=True),
        [0],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        speculative_depth=1,
        mtp_history_policy="cycle",
        verify_strategy="trim_commit",
        stop_token_ids=set(),
    )

    assert out.tokens == [1, 1]
    assert [call["tokens"] for call in model.calls] == [1, 2]
    assert model.target_cache[0].trimmed == [1]
    assert out.stats.events[0]["capture_repair"] == "trimmed_prefix_commit"
    assert "repair_forward" not in out.stats.events[0].get("timing_s", {})


def test_sustained_prefill_chunks_without_full_prompt_logits(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert model.calls[-1]["logits_keep"] == 1
    assert rt.diagnostic_counters["prefill_chunks"] == 2
    assert rt.diagnostic_counters.get("full_logits_tokens_emitted", 0) == 0
    assert rt.diagnostic_counters["final_logits_tokens_emitted"] == 1


def test_warm_restored_suffix_prefill_is_chunked_and_typed_for_abort(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[list[int]] = []
    prefill_events: list[dict[str, object]] = []

    class Bank:
        last_miss_reason = None

        def restore(self, *_args, **_kwargs):
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=[],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert hidden_states.shape[1] == len(token_ids)
        assert force_eval is True
        appended.append(list(token_ids))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4, 5, 6],
        mtp_history_policy="committed",
        session_bank=Bank(),
        prefill_callback=prefill_events.append,
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == 3
    assert prompt_state.suffix_tokens == 4
    assert [call["tokens"] for call in model.calls] == [2, 1, 1]
    assert [call["return_hidden"] for call in model.calls] == [True, True, True]
    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert appended == [[3], [4, 5], [6]]
    assert rt.diagnostic_counters["restored_suffix_prefill_chunks"] == 2
    chunk_events = [event for event in prefill_events if event["phase"] == "chunk"]
    assert [event["tokens_done"] for event in chunk_events] == [3, 5, 6, 7]
    assert [event["tokens_total"] for event in chunk_events] == [7, 7, 7, 7]
    assert [event["cached_tokens"] for event in chunk_events] == [3, 3, 3, 3]
    assert [event["new_prefill_tokens"] for event in chunk_events] == [4, 4, 4, 4]
    assert chunk_events[-1]["live_prefill_tok_s"] is not None


def test_restore_prefers_larger_near_gap_over_shorter_exact_prefix(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[list[int]] = []

    exact_entry = SimpleNamespace(prefix_len=3)
    near_entry = SimpleNamespace(
        prefix_len=8,
        token_ids=tuple(range(8)),
        session_id="session-1",
        model_path=str(rt.model_path),
        hidden_variant="post_norm",
        template_hash=None,
        mtp_history_policy="committed",
        draft_head_identity=None,
        policy_fingerprint=None,
        snapshot_epoch=8,
        mtp_snapshot_epoch=8,
        mtp_history_snapshot=object(),
        mtp_history_cache_ref=None,
        hits=0,
        last_access_s=0.0,
    )

    class Bank:
        last_miss_reason = None

        def __init__(self):
            self.restore_calls = 0
            self.prefix_restore_calls: list[tuple[int, str]] = []

        def longest_prefix(self, _prompt_ids):
            return exact_entry

        def near_prefix_candidates(self, _prompt_ids, **kwargs):
            assert kwargs["allow_block_prefix"] is False
            return [(near_entry, 7)]

        def restore_entry_prefix_cache(
            self,
            _rt,
            _entry,
            prefix_len,
            *,
            mode,
            cache_factory=None,
        ):
            assert cache_factory is None or callable(cache_factory)
            self.prefix_restore_calls.append((int(prefix_len), str(mode)))
            return [], [], "clone"

        def restore(self, *_args, **_kwargs):
            self.restore_calls += 1
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=exact_entry.prefix_len),
                cache=[],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert hidden_states.shape[1] == len(token_ids)
        assert force_eval is True
        appended.append(list(token_ids))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)
    bank = Bank()
    prefill_events: list[dict[str, object]] = []

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        mtp_history_policy="committed",
        session_bank=bank,
        prefill_callback=prefill_events.append,
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == 7
    assert prompt_state.suffix_tokens == 2
    assert prompt_state.restore_mode == "near_prefix_clone"
    assert bank.restore_calls == 0
    assert bank.prefix_restore_calls == [(7, "clone")]
    assert near_entry.hits == 1
    chunk_events = [event for event in prefill_events if event["phase"] == "chunk"]
    assert [event["tokens_done"] for event in chunk_events] == [7, 8, 9]
    assert [event["cached_tokens"] for event in chunk_events] == [7, 7, 7]
    assert [event["new_prefill_tokens"] for event in chunk_events] == [2, 2, 2]


def test_opencode_compact_restore_prefers_block_prefix_over_short_exact(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[list[int]] = []
    opencode_compact_policy = (
        "tool_prompt_mode=compact;"
        "tool_contract=compact_tool_contract:schema_free:v1;"
        "opencode_prompt_contract=opencode_agent"
    )

    exact_entry = SimpleNamespace(prefix_len=3)
    block_entry = SimpleNamespace(
        prefix_len=20,
        token_ids=tuple(range(20)),
        session_id="opencode-session",
        model_path=str(rt.model_path),
        hidden_variant="post_norm",
        template_hash=None,
        mtp_history_policy="committed",
        draft_head_identity=None,
        policy_fingerprint=opencode_compact_policy,
        snapshot_epoch=20,
        mtp_snapshot_epoch=20,
        mtp_history_snapshot=object(),
        mtp_history_cache_ref=None,
        hits=0,
        last_access_s=0.0,
    )

    class Bank:
        last_miss_reason = None

        def __init__(self):
            self.restore_calls = 0
            self.prefix_restore_calls: list[tuple[int, str]] = []
            self.near_allow_block: list[bool] = []

        def longest_prefix(self, _prompt_ids):
            return exact_entry

        def near_prefix_candidates(self, _prompt_ids, **kwargs):
            self.near_allow_block.append(bool(kwargs["allow_block_prefix"]))
            if not kwargs["allow_block_prefix"]:
                return []
            return [(block_entry, 8)]

        def restore_entry_prefix_cache(
            self,
            _rt,
            _entry,
            prefix_len,
            *,
            mode,
            cache_factory=None,
        ):
            assert cache_factory is None or callable(cache_factory)
            self.prefix_restore_calls.append((int(prefix_len), str(mode)))
            return [], [], "clone"

        def restore(self, *_args, **_kwargs):
            self.restore_calls += 1
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=exact_entry.prefix_len),
                cache=[],
                logits=mx.zeros((1, 4), dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
            )

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert hidden_states.shape[1] == len(token_ids)
        assert force_eval is True
        appended.append(list(token_ids))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)
    bank = Bank()

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        list(range(12)),
        mtp_history_policy="committed",
        session_bank=bank,
        policy_fingerprint=opencode_compact_policy,
    )

    assert prompt_state.cache_hit is True
    assert prompt_state.cached_tokens == 8
    assert prompt_state.suffix_tokens == 4
    assert prompt_state.restore_mode == "block_prefix_clone"
    assert bank.restore_calls == 0
    assert bank.near_allow_block == [True]
    assert bank.prefix_restore_calls == [(8, "clone")]
    assert block_entry.hits == 1
    assert appended == [[8], [9, 10], [11]]


def test_ssd_near_prefix_restore_time_is_cache_time_not_decode_time(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    ssd_entry = SimpleNamespace(
        prefix_len=8,
        token_ids=tuple(range(8)),
        session_id="session-ssd",
        model_path=str(rt.model_path),
        hidden_variant="post_norm",
        template_hash=None,
        mtp_history_policy="committed",
        draft_head_identity=None,
        policy_fingerprint=None,
        snapshot_epoch=8,
        mtp_snapshot_epoch=8,
        mtp_history_snapshot=object(),
        mtp_history_cache_ref=None,
        cache_source="ssd",
        ssd_cache_hit=True,
        ssd_restore_s=1.25,
        hits=0,
        last_access_s=0.0,
    )

    class Bank:
        last_miss_reason = "prefix_divergence_at_token"

        def longest_prefix(self, _prompt_ids):
            return None

        def restore(self, *_args, **_kwargs):
            return None

        def near_prefix_candidates(self, _prompt_ids, **_kwargs):
            return [(ssd_entry, 7)]

        def restore_entry_prefix_cache(
            self,
            _rt,
            _entry,
            prefix_len,
            *,
            mode,
            cache_factory=None,
        ):
            assert int(prefix_len) == 7
            assert mode == "clone"
            assert cache_factory is None or callable(cache_factory)
            return [], [], "clone"

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        assert hidden_states.shape[1] == len(token_ids)
        assert force_eval is True
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)

    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        mtp_history_policy="committed",
        session_bank=Bank(),
    )

    assert prompt_state.cache_source == "ssd"
    assert prompt_state.ssd_cache_hit is True
    assert prompt_state.ssd_restore_s == 1.25
    assert prompt_state.cache_restore_time_s >= 1.25


def test_block_prefix_restore_matches_target_default(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", raising=False)
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    class Bank:
        last_miss_reason = "prefix_divergence_at_token"

        def __init__(self):
            self.near_kwargs: list[dict[str, object]] = []

        def longest_prefix(self, _prompt_ids):
            return None

        def restore(self, *_args, **_kwargs):
            return None

        def near_prefix_candidates(self, _prompt_ids, **kwargs):
            self.near_kwargs.append(kwargs)
            return []

    bank = Bank()
    prompt_state = restore_or_prefill_prompt_state(
        rt,
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        mtp_history_policy="committed",
        session_bank=bank,
    )

    assert prompt_state.cache_hit is False
    assert prompt_state.cached_tokens == 0
    assert bank.near_kwargs
    assert bank.near_kwargs[-1]["allow_block_prefix"] is True


def test_sustained_prefill_chunk_cache_cleanup_is_explicit(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP", "1")
    calls: list[str] = []
    monkeypatch.setattr("mtplx.generation.mx.synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(
        "mtplx.generation.mx.clear_cache", lambda: calls.append("clear")
    )
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert calls == ["sync", "clear", "sync", "clear"]
    assert rt.diagnostic_counters["prefill_chunk_cache_cleanup_events"] == 2


def test_sustained_prefill_stock_cache_only_requires_unsafe_allow(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_STOCK_CACHE_ONLY", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [False, False, True]
    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert rt.diagnostic_counters.get("prefill_stock_cache_only_calls", 0) == 0


def test_sustained_prefill_stock_cache_only_is_explicit_unsafe(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_STOCK_CACHE_ONLY", "1")
    monkeypatch.setenv("MTPLX_ALLOW_UNSAFE_PREFILL_STOCK_CACHE_ONLY", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [False, False, True]
    assert [call["emit_logits"] for call in model.calls] == [True, True, True]
    assert rt.diagnostic_counters["prefill_external_cache_only_calls"] == 2
    assert rt.diagnostic_counters["prefill_stock_cache_only_calls"] == 2


def test_sustained_prefill_omlx_external_is_safe_profile_path(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_OMLX_EXTERNAL", "1")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["tokens"] for call in model.calls] == [2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [False, False, True]
    assert [call["emit_logits"] for call in model.calls] == [True, True, True]
    assert rt.diagnostic_counters["prefill_external_cache_only_calls"] == 2
    assert rt.diagnostic_counters["prefill_omlx_external_calls"] == 2
    assert rt.diagnostic_counters.get("prefill_stock_cache_only_calls", 0) == 0


def test_sustained_prefill_forwards_logits_controls_through_patched_kwargs_wrapper(
    monkeypatch,
):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = KwargsOnlyTinyModel()
    rt = _runtime(model, mtp_enabled=True)

    _prefill(rt, [10, 11, 12, 13, 14], return_hidden=True)

    assert [call["emit_logits"] for call in model.calls] == [False, False, True]
    assert rt.diagnostic_counters.get("full_logits_tokens_emitted", 0) == 0


def test_last_window_mtp_history_skips_discarded_chunk_hidden(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = TinyModel()
    rt = _runtime(model, mtp_enabled=True)
    appended: list[tuple[list[int], int | None]] = []

    def append_history(
        _rt,
        _mtp_cache,
        hidden_states,
        token_ids,
        *,
        mtp_hidden_variant,
        position_offset=None,
        force_eval=False,
        input_embeddings=None,
    ):
        appended.append((list(token_ids), position_offset))
        return 0.0

    monkeypatch.setattr("mtplx.generation._append_mtp_history", append_history)

    _prefill_committed_mtp_history_streaming(
        rt,
        list(range(9)),
        mtp_hidden_variant="post_norm",
        history_window_tokens=3,
    )

    assert [call["tokens"] for call in model.calls] == [2, 2, 2, 2, 1]
    assert [call["return_hidden"] for call in model.calls] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert appended == [([6], 5), ([7, 8], 6)]


def test_32k_prefill_peak_memory_bounded():
    """
    Regression guard for the Ivan/Benchand 32K memory balloon.
    Run only on the Apple Silicon long-context QA machine.
    """
    if os.environ.get("MTPLX_RUN_32K_MEMORY_QA") != "1":
        pytest.skip("set MTPLX_RUN_32K_MEMORY_QA=1 on the long-context QA Mac")
    model_path = os.environ.get("MTPLX_32K_QA_MODEL")
    if not model_path:
        pytest.skip("set MTPLX_32K_QA_MODEL to a local runnable MTPLX model")

    from mtplx.runtime import load

    rt = load(model_path, mtp=True)
    text = "def f(x): return x + 1\n" * 4096
    prompt_ids = rt.tokenizer.encode(text)[:32768]
    if len(prompt_ids) < 32000:
        pytest.skip("QA prompt did not tokenize to 32K tokens")

    mx.reset_peak_memory()
    os.environ["MTPLX_SUSTAINED_PREFILL"] = "1"
    os.environ["MTPLX_PREFILL_CHUNK_SIZE"] = "2048"
    os.environ["MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS"] = "0"
    _prefill(rt, prompt_ids, return_hidden=True)
    peak_gb = mx.get_peak_memory() / (1024**3)

    assert peak_gb < 35.0, f"32K Sustained prefill peak was {peak_gb:.1f} GB"
