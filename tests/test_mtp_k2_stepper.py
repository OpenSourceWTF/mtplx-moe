from __future__ import annotations

import inspect
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import mlx.core as mx
import pytest

from mtplx import generation as generation_module
from mtplx.cache_state import CacheSnapshot
from mtplx.generation import (
    _clear_cache_every,
    _defer_verify_hidden_eval_enabled,
    _prefill_chunk_size,
    _sustained_prefill_layout,
    execute_solo_mtpk2_prefill_ticket,
    execute_solo_mtpk2_verify_ticket,
    generate_mtpk,
    make_mtpk2_request_state,
    make_mtpk2_request_state_from_environment,
    prefill_chunk_size_override,
)
from mtplx.mtp_k2_stepper import (
    MTPK2AcceptanceContext,
    MTPK2ContextCopyResult,
    MTPK2ContextCopyTicket,
    MTPK2PrefillResult,
    MTPK2PrefillTicket,
    MTPK2RequestCancelled,
    MTPK2RequestConfig,
    MTPK2RequestState,
    MTPK2VerifyResult,
    MTPK2VerifyTicket,
    current_mtpk2_request_config,
    drive_solo_mtpk2,
)
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig
from mtplx.session_bank import SessionBank


def _fixed_request_config(
    *,
    prompt_tokens: int = 3,
    layout: str = "contiguous_dense_decode",
) -> MTPK2RequestConfig:
    return MTPK2RequestConfig(
        prompt_tokens=prompt_tokens,
        prefill_chunk_tokens=1024,
        environment=MappingProxyType({}),
        compiled_verify_mode="off",
        compiled_verify_max_len=6,
        loop_guard_config=SimpleNamespace(enabled=False),
        target_prefill_cache_factory=lambda: [],
        target_prefill_repage_route=lambda _cache: None,
        mtp_cache_factory=lambda: [],
        sustained_prefill_enabled=True,
        state_rebase_every=0,
        sustained_prefill_layout=layout,
        defer_verify_hidden_eval=True,
        clear_cache_every=1024,
        skip_verify_snapshot=True,
        omit_speculative_bonus=False,
        batch_target_arrays=True,
        batch_target_distributions=True,
        compiled_verify_stats=True,
    )


def test_k2_tickets_are_immutable_and_keep_request_local_context() -> None:
    cache = [object()]
    acceptance = MTPK2AcceptanceContext(
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        hidden_variant="post_norm",
    )
    ticket = MTPK2VerifyTicket(
        request_id="request-a",
        input_ids=[[7, 8, 9]],
        request_cache=cache,
        draft_distributions=("q1", "q2"),
        acceptance_context=acceptance,
    )

    assert ticket.request_cache is cache
    assert ticket.draft_distributions == ("q1", "q2")
    with pytest.raises(FrozenInstanceError):
        ticket.request_id = "request-b"  # type: ignore[misc]


def test_prefill_ticket_contract_locks_1024_token_chunks() -> None:
    cache = [object()]
    chunks = [
        MTPK2PrefillTicket(
            request_id="request-a",
            input_ids=list(range(start, min(start + 1024, 2500))),
            request_cache=cache,
            prompt_start=start,
            prompt_stop=min(start + 1024, 2500),
        )
        for start in range(0, 2500, 1024)
    ]

    assert [(ticket.prompt_start, ticket.prompt_stop) for ticket in chunks] == [
        (0, 1024),
        (1024, 2048),
        (2048, 2500),
    ]
    assert [len(ticket.input_ids) for ticket in chunks] == [1024, 1024, 452]
    assert all(len(ticket.input_ids) <= 1024 for ticket in chunks)


def test_solo_driver_consumes_prefill_then_verify_tickets_in_order() -> None:
    cache = ["request-cache"]
    acceptance = MTPK2AcceptanceContext(
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        hidden_variant="post_norm",
    )
    tickets_seen: list[str] = []

    def commit_prefix(_steps):
        return cache

    def machine():
        prefill = yield MTPK2PrefillTicket(
            request_id="request-a",
            input_ids=[1, 2],
            request_cache=cache,
            prompt_start=0,
            prompt_stop=2,
        )
        assert prefill == MTPK2PrefillResult(
            logits="prefill-logits",
            hidden="prefill-hidden",
            request_cache=cache,
        )
        verify = yield MTPK2VerifyTicket(
            request_id="request-a",
            input_ids=[[3, 4, 5]],
            request_cache=cache,
            draft_distributions=("q1", "q2"),
            acceptance_context=acceptance,
        )
        assert verify == MTPK2VerifyResult(
            logits="verify-logits",
            hidden="verify-hidden",
            captures={"layer": {"state": "capture"}},
            request_cache=cache,
            commit_prefix=commit_prefix,
        )
        return "finished"

    config = _fixed_request_config()
    state = MTPK2RequestState(
        request_id="request-a",
        _machine=machine(),
        config=config,
        lane=SimpleNamespace(),
        width1_commit_route=commit_prefix,
        target_cache=cache,
        mtp_cache=["mtp-cache"],
        tokens=[],
        rng=SimpleNamespace(),
        sampler=SimpleNamespace(),
        draft_sampler=SimpleNamespace(),
        constraint=None,
        stop_token_ids=set(),
        token_callback=None,
        prefill_callback=None,
        cancel_event=None,
        session_id=None,
        stats=SimpleNamespace(),
    )

    def execute_prefill(ticket):
        tickets_seen.append(f"prefill:{ticket.prompt_start}:{ticket.prompt_stop}")
        return MTPK2PrefillResult(
            logits="prefill-logits",
            hidden="prefill-hidden",
            request_cache=ticket.request_cache,
        )

    def execute_verify(ticket):
        tickets_seen.append(f"verify:{ticket.input_ids}")
        return MTPK2VerifyResult(
            logits="verify-logits",
            hidden="verify-hidden",
            captures={"layer": {"state": "capture"}},
            request_cache=ticket.request_cache,
            commit_prefix=commit_prefix,
        )

    result = drive_solo_mtpk2(
        state,
        execute_prefill=execute_prefill,
        execute_verify=execute_verify,
    )

    assert result == "finished"
    assert tickets_seen == ["prefill:0:2", "verify:[[3, 4, 5]]"]
    assert state.pending_ticket is None


def test_solo_driver_executes_context_copy_as_an_explicit_phase_ticket() -> None:
    cache = ["request-cache"]
    tickets_seen: list[str] = []

    def machine():
        result = yield MTPK2ContextCopyTicket(
            request_id="request-a",
            input_ids=[[7, 8, 9, 10]],
            request_cache=cache,
            hidden_variant="post_norm",
            capture_backend="linear_gdn_from_conv_tape",
        )
        assert result == MTPK2ContextCopyResult(
            logits="copy-logits",
            hidden="copy-hidden",
            captures={"layer": {"state": "copy-capture"}},
            request_cache=cache,
        )
        return "finished"

    state = _synthetic_state(machine())

    def execute_context_copy(ticket):
        tickets_seen.append(f"context-copy:{ticket.input_ids}")
        return MTPK2ContextCopyResult(
            logits="copy-logits",
            hidden="copy-hidden",
            captures={"layer": {"state": "copy-capture"}},
            request_cache=ticket.request_cache,
        )

    result = drive_solo_mtpk2(
        state,
        execute_prefill=lambda _ticket: pytest.fail("unexpected prefill"),
        execute_context_copy=execute_context_copy,
        execute_verify=lambda _ticket: pytest.fail("unexpected verify"),
    )

    assert result == "finished"
    assert tickets_seen == ["context-copy:[[7, 8, 9, 10]]"]


def _synthetic_state(
    machine,
    *,
    config: MTPK2RequestConfig | None = None,
) -> MTPK2RequestState:
    resolved_config = config or _fixed_request_config()
    return MTPK2RequestState(
        request_id="synthetic",
        _machine=machine,
        config=resolved_config,
        lane=SimpleNamespace(),
        width1_commit_route=_return_request_cache,
        target_cache=[],
        mtp_cache=None,
        tokens=[],
        rng=SimpleNamespace(),
        sampler=SimpleNamespace(),
        draft_sampler=SimpleNamespace(),
        constraint=None,
        stop_token_ids=set(),
        token_callback=None,
        prefill_callback=None,
        cancel_event=None,
        session_id=None,
        stats=SimpleNamespace(),
    )


def test_request_state_exclusively_advances_and_scopes_request_config() -> None:
    config = _fixed_request_config()
    cache: list[object] = []

    def machine():
        assert current_mtpk2_request_config() is config
        prefill_result = yield MTPK2PrefillTicket(
            request_id="synthetic",
            input_ids=[1],
            request_cache=cache,
            prompt_start=0,
            prompt_stop=1,
        )
        assert current_mtpk2_request_config() is config
        verify_result = yield MTPK2VerifyTicket(
            request_id="synthetic",
            input_ids=[[2, 3, 4]],
            request_cache=prefill_result.request_cache,
            draft_distributions=("q1", "q2"),
            acceptance_context=MTPK2AcceptanceContext(
                verify_strategy="capture_commit",
                verify_core="linear-gdn-from-conv-tape",
                hidden_variant="post_norm",
            ),
        )
        assert current_mtpk2_request_config() is config
        return verify_result.logits

    state = _synthetic_state(machine(), config=config)
    assert current_mtpk2_request_config() is None
    assert state.start() is state.pending_ticket
    assert state.status == "ready"
    with pytest.raises(AttributeError):
        getattr(state, "machine")

    def execute_prefill(ticket):
        assert current_mtpk2_request_config() is config
        assert _prefill_chunk_size() == 1024
        assert _sustained_prefill_layout() == config.sustained_prefill_layout
        assert _defer_verify_hidden_eval_enabled() is True
        assert _clear_cache_every() == config.clear_cache_every
        return MTPK2PrefillResult(
            logits="prefill",
            hidden="hidden",
            request_cache=ticket.request_cache,
        )

    prefill_result = state.execute_pending(execute_prefill)
    assert current_mtpk2_request_config() is None
    assert state.resume(prefill_result) is state.pending_ticket

    def commit_prefix(_steps):
        return cache

    def execute_verify(ticket):
        assert current_mtpk2_request_config() is config
        return MTPK2VerifyResult(
            logits="done",
            hidden="hidden",
            captures={},
            request_cache=ticket.request_cache,
            commit_prefix=commit_prefix,
        )

    verify_result = state.execute_pending(execute_verify)
    assert state.resume(verify_result) is None
    assert state.status == "finished"
    assert state.output == "done"
    state.close()
    assert state.status == "finished"
    assert current_mtpk2_request_config() is None


def test_request_state_failure_and_close_clear_pending_and_context() -> None:
    error = ValueError("executor failed")

    def machine():
        yield MTPK2PrefillTicket(
            request_id="synthetic",
            input_ids=[1],
            request_cache=[],
            prompt_start=0,
            prompt_stop=1,
        )

    state = _synthetic_state(machine())
    state.start()
    with pytest.raises(ValueError, match="executor failed"):
        state.fail(error)

    assert state.status == "failed"
    assert state.error is error
    assert state.pending_ticket is None
    state.close()
    assert state.status == "failed"
    assert current_mtpk2_request_config() is None

    other = _synthetic_state(machine())
    other.start()
    other.close()
    assert other.status == "closed"
    assert other.pending_ticket is None
    assert current_mtpk2_request_config() is None


class _CancelEvent:
    def __init__(self, *, set_: bool = False) -> None:
        self.set_ = set_

    def is_set(self) -> bool:
        return self.set_


def test_request_state_cancel_before_start_is_terminal_and_context_clean() -> None:
    def machine():
        yield MTPK2PrefillTicket(
            request_id="synthetic",
            input_ids=[1],
            request_cache=[],
            prompt_start=0,
            prompt_stop=1,
        )

    state = _synthetic_state(machine())
    state.cancel_event = _CancelEvent(set_=True)

    assert state.start() is None
    assert state.status == "cancelled"
    assert state.pending_ticket is None
    state.close()
    assert state.status == "cancelled"
    assert current_mtpk2_request_config() is None


def test_request_state_cancel_after_execute_discards_result_cache() -> None:
    original_cache = [object()]
    result_cache = [object()]
    cancel = _CancelEvent()

    def machine():
        yield MTPK2PrefillTicket(
            request_id="synthetic",
            input_ids=[1],
            request_cache=original_cache,
            prompt_start=0,
            prompt_stop=1,
        )

    state = _synthetic_state(machine())
    state.target_cache = original_cache
    state.cancel_event = cancel
    state.start()
    result = state.execute_pending(
        lambda _ticket: MTPK2PrefillResult(
            logits="discarded",
            hidden=None,
            request_cache=result_cache,
        )
    )
    cancel.set_ = True

    assert state.resume(result) is None
    assert state.status == "cancelled"
    assert state.target_cache is original_cache
    assert state.pending_ticket is None
    state.close()
    assert state.status == "cancelled"
    assert current_mtpk2_request_config() is None


def test_request_state_cancel_after_ready_skips_ticket_dispatch() -> None:
    cancel = _CancelEvent()
    dispatched: list[MTPK2PrefillTicket] = []

    def machine():
        yield MTPK2PrefillTicket(
            request_id="synthetic",
            input_ids=[1],
            request_cache=[],
            prompt_start=0,
            prompt_stop=1,
        )

    state = _synthetic_state(machine())
    state.cancel_event = cancel
    assert isinstance(state.start(), MTPK2PrefillTicket)
    cancel.set_ = True

    with pytest.raises(MTPK2RequestCancelled):
        state.execute_pending(lambda ticket: dispatched.append(ticket))

    assert dispatched == []
    assert state.status == "cancelled"
    assert state.pending_ticket is None
    assert current_mtpk2_request_config() is None


class _OffsetCache:
    def __init__(self) -> None:
        self.offset = 0
        self.trimmed: list[int] = []

    @property
    def state(self):
        return None

    @state.setter
    def state(self, _value) -> None:
        pass

    @property
    def meta_state(self):
        return (self.offset,)

    @meta_state.setter
    def meta_state(self, value) -> None:
        self.offset = int(value[0])

    def is_trimmable(self) -> bool:
        return True

    def trim(self, count: int) -> int:
        trimmed = min(self.offset, int(count))
        self.offset -= trimmed
        self.trimmed.append(trimmed)
        return trimmed


class _Tokenizer:
    eos_token_id = None
    pad_token_id = None

    def decode(self, tokens, **_kwargs):
        return "".join(str(int(token)) for token in tokens)


class _TrackingK2Model:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def make_cache(self):
        return [_OffsetCache()]

    def make_mtp_cache(self):
        return []

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant=None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        del hidden_variant
        length = int(input_ids.shape[1])
        self.calls.append(length)
        for entry in cache or []:
            entry.offset += length
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        keep = length if logits_keep is None else min(length, max(1, logits_keep))
        logits = mx.zeros((1, keep, 4), dtype=mx.float32)
        logits = logits + mx.array([0.0, 10.0, 0.0, 0.0], dtype=mx.float32)
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        return_hidden: bool = False,
        mtp_hidden_variant=None,
        position_offset=None,
    ):
        del hidden_states, mtp_cache, concat_order, mtp_hidden_variant, position_offset
        length = int(next_token_ids.shape[1])
        logits = mx.zeros((1, length, 4), dtype=mx.float32)
        logits = logits + mx.array([0.0, 10.0, 0.0, 0.0], dtype=mx.float32)
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        return (logits, hidden) if return_hidden else logits

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        del next_token_ids
        return hidden_states


class _FactoryTrackingK2Model(_TrackingK2Model):
    def __init__(self) -> None:
        super().__init__()
        self.target_cache_builds = 0
        self.mtp_cache_builds = 0

    def make_cache(self):
        self.target_cache_builds += 1
        return [_OffsetCache()]

    def make_mtp_cache(self):
        self.mtp_cache_builds += 1
        return [_OffsetCache()]


class _RowSensitiveK2Model(_TrackingK2Model):
    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant=None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        del hidden_variant
        length = int(input_ids.shape[1])
        self.calls.append(length)
        for entry in cache or []:
            entry.offset += length
        token_values = input_ids.astype(mx.float32)
        positions = mx.arange(length, dtype=mx.float32)[None, :]
        hidden = mx.stack(
            [token_values, token_values + positions],
            axis=-1,
        )
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        logits = mx.stack(
            [
                token_values + positions,
                token_values * 0.0 + 10.0,
                -token_values - positions,
                token_values * 0.25 + positions,
            ],
            axis=-1,
        )
        if logits_keep is not None:
            logits = logits[:, -min(length, max(1, logits_keep)) :, :]
        return (logits, hidden) if return_hidden else logits


class _RejectingK2Model(_TrackingK2Model):
    def mtp_forward(self, *args, **kwargs):
        result = super().mtp_forward(*args, **kwargs)
        if isinstance(result, tuple):
            _logits, hidden = result
            logits = mx.zeros_like(_logits)
            logits = logits + mx.array(
                [0.0, 0.0, 10.0, 0.0],
                dtype=mx.float32,
            )
            return logits, hidden
        logits = mx.zeros_like(result)
        return logits + mx.array([0.0, 0.0, 10.0, 0.0], dtype=mx.float32)


class _RejectingSecondK2Model(_TrackingK2Model):
    def __init__(self) -> None:
        super().__init__()
        self.draft_calls = 0

    def mtp_forward(self, *args, **kwargs):
        self.draft_calls += 1
        result = super().mtp_forward(*args, **kwargs)
        if self.draft_calls == 1:
            return result
        if isinstance(result, tuple):
            _logits, hidden = result
            logits = mx.zeros_like(_logits)
            logits = logits + mx.array(
                [0.0, 0.0, 10.0, 0.0],
                dtype=mx.float32,
            )
            return logits, hidden
        logits = mx.zeros_like(result)
        return logits + mx.array([0.0, 0.0, 10.0, 0.0], dtype=mx.float32)


def _return_request_cache(cache, _captures, *, steps):
    del steps
    return cache


def _tracking_runtime(
    model: _TrackingK2Model,
    *,
    commit_route=None,
) -> MTPLXRuntime:
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        model_path=Path("tracking-k2"),
        mtp_enabled=True,
        contract=MTPContract(),
    )
    runtime.qwen27b_k2_dual_lane = SimpleNamespace(
        backend_id="qwen3_next",
        depth=2,
        hidden_variant="post_norm",
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        max_width=2,
        capture_commit_for=(
            lambda _width, _row: commit_route
            if commit_route is not None
            else _return_request_cache
        ),
    )
    return runtime


def _request_kwargs(**overrides):
    kwargs = {
        "max_tokens": 4,
        "sampler": SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        "speculative_depth": 2,
        "seed": 7,
        "stop_token_ids": set(),
        "mtp_history_policy": "committed",
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
    }
    kwargs.update(overrides)
    return kwargs


def test_interleaved_prefill_executors_keep_frozen_request_local_context(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MTPLX_SKIP_VERIFY_SNAPSHOT", raising=False)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "12")
    monkeypatch.setenv("MTPLX_DEFER_VERIFY_HIDDEN_EVAL", "auto")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY", "auto")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD", "8")
    monkeypatch.setenv("MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT", "7")
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "sentinel")
    runtime = _tracking_runtime(_TrackingK2Model())

    short = make_mtpk2_request_state(
        runtime,
        [1] * 10,
        request_id="short",
        **_request_kwargs(),
    )
    monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "0")
    monkeypatch.setenv("MTPLX_SKIP_VERIFY_SNAPSHOT", "1")
    long = make_mtpk2_request_state(
        runtime,
        [2] * 20,
        request_id="long",
        **_request_kwargs(),
    )

    assert short.config.sustained_prefill_layout == "contiguous_dense_decode"
    assert short.config.defer_verify_hidden_eval is True
    assert short.config.clear_cache_every == 7
    assert short.config.batch_target_arrays is True
    assert short.config.skip_verify_snapshot is False
    assert long.config.sustained_prefill_layout == "contiguous_then_repage"
    assert long.config.defer_verify_hidden_eval is False
    assert long.config.clear_cache_every == 0
    assert long.config.batch_target_arrays is False
    assert long.config.skip_verify_snapshot is True

    assert isinstance(short.start(), MTPK2PrefillTicket)
    assert isinstance(long.start(), MTPK2PrefillTicket)

    def observe_and_execute(state):
        def execute(ticket):
            assert current_mtpk2_request_config() is state.config
            assert _prefill_chunk_size() == 1024
            assert (
                _sustained_prefill_layout()
                == state.config.sustained_prefill_layout
            )
            assert (
                _defer_verify_hidden_eval_enabled()
                is state.config.defer_verify_hidden_eval
            )
            assert _clear_cache_every() == state.config.clear_cache_every
            return execute_solo_mtpk2_prefill_ticket(runtime, ticket)

        return state.execute_pending(execute)

    short_result = observe_and_execute(short)
    assert current_mtpk2_request_config() is None
    long_result = observe_and_execute(long)
    assert current_mtpk2_request_config() is None
    assert os.environ["MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS"] == "sentinel"

    short.resume(short_result)
    long.resume(long_result)
    short.close()
    long.close()
    assert current_mtpk2_request_config() is None


def test_explicit_request_environment_is_frozen_before_state_construction(
    monkeypatch,
) -> None:
    environment = {"MTPLX_SUSTAINED_PREFILL": "1"}
    marker = object()
    captured = {}

    def make_from_snapshot(
        _runtime,
        prompt_ids,
        *,
        request_id,
        _resolved_depth,
        _context_copy_requested,
        _environment,
        **kwargs,
    ):
        captured["prompt_ids"] = prompt_ids
        captured["request_id"] = request_id
        captured["environment"] = _environment
        captured["kwargs"] = kwargs
        assert generation_module._generation_env_get(
            "MTPLX_SUSTAINED_PREFILL"
        ) == "1"
        return marker

    monkeypatch.setattr(
        generation_module,
        "_make_mtpk2_request_state_from_snapshot",
        make_from_snapshot,
    )

    result = make_mtpk2_request_state_from_environment(
        object(),
        [1, 2, 3],
        request_id="frozen-boundary",
        environment=environment,
        seed=17,
    )
    environment["MTPLX_SUSTAINED_PREFILL"] = "0"

    assert result is marker
    assert captured["prompt_ids"] == [1, 2, 3]
    assert captured["request_id"] == "frozen-boundary"
    assert captured["environment"]["MTPLX_SUSTAINED_PREFILL"] == "1"
    assert captured["kwargs"]["seed"] == 17
    with pytest.raises(TypeError):
        captured["environment"]["MTPLX_SUSTAINED_PREFILL"] = "0"
    source = inspect.getsource(make_mtpk2_request_state_from_environment)
    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert "_temporary_env" not in source


def test_explicit_request_environment_controls_loop_guard_without_ambient_read(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_LOOP_GUARD", "0")
    environment = dict(os.environ)
    environment.update(
        {
            "MTPLX_CONTEXT_COPY": "0",
            "MTPLX_LOOP_GUARD": "1",
            "MTPLX_STATE_REBASE_EVERY": "0",
            "MTPLX_SUSTAINED_PREFILL": "1",
        }
    )

    state = make_mtpk2_request_state_from_environment(
        _tracking_runtime(_TrackingK2Model()),
        [1, 2, 3],
        request_id="frozen-loop-guard",
        environment=environment,
        **_request_kwargs(loop_guard=True),
    )

    assert state.config.loop_guard_config.enabled is True


def test_fixed_request_copies_prompt_and_freezes_all_generation_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "0")
    monkeypatch.setenv("MTPLX_STATE_REBASE_EVERY", "0")
    monkeypatch.setenv("MTPLX_LOOP_GUARD", "1")
    prompt = list(range(2051))
    runtime = _tracking_runtime(_TrackingK2Model())
    state = make_mtpk2_request_state(
        runtime,
        prompt,
        request_id="frozen-inputs",
        **_request_kwargs(loop_guard=True),
    )

    prompt.clear()
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "0")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity")
    monkeypatch.setenv("MTPLX_STATE_REBASE_EVERY", "8")
    monkeypatch.setenv("MTPLX_LOOP_GUARD", "0")
    original_environment_get = os.environ.get

    def forbid_loop_guard_environment(name, *args):
        if str(name).startswith("MTPLX_LOOP_GUARD"):
            pytest.fail(
                "fixed request re-read Loop Guard environment after construction"
            )
        return original_environment_get(name, *args)

    monkeypatch.setattr(
        "mtplx.generation.CompiledVerifyBank",
        lambda *_args, **_kwargs: pytest.fail(
            "fixed request rebuilt compiled route from mutated environment"
        ),
    )
    monkeypatch.setattr(
        "mtplx.loop_guard.os.environ.get",
        forbid_loop_guard_environment,
    )
    for helper_name in (
        "context_copy_block_k",
        "context_copy_min_ext",
        "context_copy_ng_min",
        "context_copy_ng_max",
    ):
        monkeypatch.setattr(
            f"mtplx.context_copy.{helper_name}",
            lambda: pytest.fail("inactive fixed context-copy read environment"),
        )
    monkeypatch.setattr(
        "mtplx.generation._target_prefill_cache_layout_scope",
        lambda: pytest.fail(
            "fixed request mutated process environment for prefill cache"
        ),
    )

    assert state.config.sustained_prefill_enabled is True
    assert state.config.compiled_verify_mode == "off"
    assert state.config.loop_guard_config.enabled is True
    assert state.config.state_rebase_every == 0
    ticket = state.start()
    spans: list[tuple[int, int]] = []
    while isinstance(ticket, MTPK2PrefillTicket):
        spans.append((ticket.prompt_start, ticket.prompt_stop))
        assert ticket.prompt_stop - ticket.prompt_start <= 1024
        result = state.execute_pending(
            lambda pending: execute_solo_mtpk2_prefill_ticket(runtime, pending)
        )
        ticket = state.resume(result)

    assert spans == [(0, 1024), (1024, 2048), (2048, 2050), (2050, 2051)]
    assert isinstance(ticket, MTPK2VerifyTicket)
    assert tuple(ticket.input_ids.shape) == (1, 3)
    state.close()


def test_fixed_k2_rejects_paged_target_repage_before_generator_publication(
    monkeypatch,
) -> None:
    cache_environment = {
        "MTPLX_CONTEXT_COPY": "0",
        "MTPLX_SUSTAINED_PREFILL": "1",
        "MTPLX_SUSTAINED_PREFILL_LAYOUT": "contiguous_then_repage",
        "MTPLX_VLLM_METAL_PAGED_ATTN": "1",
        "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE": "32",
        "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS": "77",
        "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": "fast_sdpa_gather",
        "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN": "0",
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT": "0",
        "MTPLX_VLLM_METAL_PAGED_KV_QUANT": "q8",
        "MTPLX_VLLM_METAL_PAGED_MTP_ATTN": "0",
        "MTPLX_DYNAMIC_PAGED_KV": "0",
    }
    for name, value in cache_environment.items():
        monkeypatch.setenv(name, value)
    model = _FactoryTrackingK2Model()
    runtime = _tracking_runtime(model)
    generator_calls: list[bool] = []

    def published_generator(*_args, **_kwargs):
        generator_calls.append(True)
        raise AssertionError("paged target repage reached generator publication")

    monkeypatch.setattr(
        "mtplx.generation._generate_mtpk_machine",
        published_generator,
    )
    monkeypatch.setattr(
        "mtplx.cache_state.bind_tail_owned_attention_kv_cache_route",
        lambda _environment: pytest.fail(
            "paged target repage route bound before construction rejection"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="contiguous_then_repage.*MTPLX_VLLM_METAL_PAGED_ATTN=0",
    ):
        make_mtpk2_request_state(
            runtime,
            [0, 1, 2],
            request_id="paged-target-repage",
            **_request_kwargs(),
        )

    assert generator_calls == []
    assert model.target_cache_builds == 0
    assert model.mtp_cache_builds == 0
    assert model.calls == []


def test_fixed_auto_dense_paged_target_stays_nonpaged_after_environment_mutation(
    monkeypatch,
) -> None:
    cache_environment = {
        "MTPLX_CONTEXT_COPY": "0",
        "MTPLX_SUSTAINED_PREFILL": "1",
        "MTPLX_SUSTAINED_PREFILL_LAYOUT": "auto",
        "MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT": "131072",
        "MTPLX_MTP_HISTORY_POLICY": "committed",
        "MTPLX_VLLM_METAL_PAGED_ATTN": "1",
        "MTPLX_VLLM_METAL_PAGED_KV_QUANT": "off",
        "MTPLX_VLLM_METAL_PAGED_MTP_ATTN": "0",
    }
    for name, value in cache_environment.items():
        monkeypatch.setenv(name, value)

    model = _TrackingK2Model()
    runtime = _tracking_runtime(model)
    prompt = list(range(8192))
    state = make_mtpk2_request_state(
        runtime,
        prompt,
        request_id="live-dense-paged-setting",
        **_request_kwargs(),
    )
    assert state.config.prompt_tokens == 8192
    assert state.config.sustained_prefill_layout == "contiguous_dense_decode"

    monkeypatch.setenv(
        "MTPLX_SUSTAINED_PREFILL_LAYOUT",
        "contiguous_then_repage",
    )
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "0")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "q8")
    monkeypatch.setattr(
        "mtplx.cache_state.install_vllm_metal_paged_attention_kv_cache",
        lambda *_args, **_kwargs: pytest.fail(
            "fixed auto/dense request installed paged target cache"
        ),
    )

    ticket = state.start()
    spans: list[tuple[int, int]] = []
    while isinstance(ticket, MTPK2PrefillTicket):
        spans.append((ticket.prompt_start, ticket.prompt_stop))
        assert all(isinstance(entry, _OffsetCache) for entry in ticket.request_cache)
        prefill_result = state.execute_pending(
            lambda pending: execute_solo_mtpk2_prefill_ticket(runtime, pending)
        )
        ticket = state.resume(prefill_result)

    verify_ticket = ticket
    assert isinstance(verify_ticket, MTPK2VerifyTicket)
    assert all(isinstance(entry, _OffsetCache) for entry in verify_ticket.request_cache)
    assert sum(stop - start for start, stop in spans) == len(prompt)
    assert max(stop - start for start, stop in spans) <= 1024

    verify_result = state.execute_pending(
        lambda pending: execute_solo_mtpk2_verify_ticket(runtime, pending)
    )
    assert all(isinstance(entry, _OffsetCache) for entry in verify_result.request_cache)
    assert model.calls[:-1] == [stop - start for start, stop in spans]
    assert model.calls[-1] == 3
    state.close()


def test_fixed_k2_rejects_paged_mtp_before_generator_publication(
    monkeypatch,
) -> None:
    cache_environment = {
        "MTPLX_CONTEXT_COPY": "0",
        "MTPLX_SUSTAINED_PREFILL": "1",
        "MTPLX_SUSTAINED_PREFILL_LAYOUT": "auto",
        "MTPLX_MTP_HISTORY_POLICY": "committed",
        "MTPLX_VLLM_METAL_PAGED_MTP_ATTN": "1",
    }
    for name, value in cache_environment.items():
        monkeypatch.setenv(name, value)

    model = _TrackingK2Model()
    runtime = _tracking_runtime(model)
    generator_calls: list[bool] = []

    def published_generator(*_args, **_kwargs):
        generator_calls.append(True)
        raise AssertionError("paged MTP request reached generator publication")

    monkeypatch.setattr(
        "mtplx.generation._generate_mtpk_machine",
        published_generator,
    )

    with pytest.raises(
        RuntimeError,
        match="MTPLX_VLLM_METAL_PAGED_MTP_ATTN=0",
    ):
        make_mtpk2_request_state(
            runtime,
            [0, 1, 2],
            request_id="paged-mtp",
            **_request_kwargs(),
        )

    assert generator_calls == []
    assert model.calls == []
    assert runtime.diagnostic_counters.get("make_mtp_cache_calls", 0) == 0


def test_fixed_committed_history_uses_frozen_nonpaged_mtp_cache_factory(
    monkeypatch,
) -> None:
    cache_environment = {
        "MTPLX_CONTEXT_COPY": "0",
        "MTPLX_SUSTAINED_PREFILL": "1",
        "MTPLX_SUSTAINED_PREFILL_LAYOUT": "auto",
        "MTPLX_MTP_HISTORY_POLICY": "committed",
        "MTPLX_VLLM_METAL_PAGED_MTP_ATTN": "0",
    }
    for name, value in cache_environment.items():
        monkeypatch.setenv(name, value)

    class _MtpCacheModel(_TrackingK2Model):
        def make_mtp_cache(self):
            return [_OffsetCache()]

    model = _MtpCacheModel()
    runtime = _tracking_runtime(model)
    state = make_mtpk2_request_state(
        runtime,
        [0, 1, 2],
        request_id="frozen-mtp-cache",
        **_request_kwargs(),
    )
    made_mtp_caches: list[list[object]] = []
    bound_mtp_cache_factory = state.config.mtp_cache_factory

    def track_mtp_cache_factory():
        cache = bound_mtp_cache_factory()
        made_mtp_caches.append(cache)
        return cache

    state.config = replace(
        state.config,
        mtp_cache_factory=track_mtp_cache_factory,
    )
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_MTP_ATTN", "1")
    monkeypatch.setattr(
        model,
        "make_mtp_cache",
        lambda: pytest.fail("fixed request used mutated model MTP cache factory"),
    )

    ticket = state.start()
    assert isinstance(ticket, MTPK2PrefillTicket)
    assert runtime.diagnostic_counters["make_mtp_cache_calls"] == 1
    assert len(made_mtp_caches) == 1
    assert isinstance(made_mtp_caches[0][0], _OffsetCache)
    state.close()


def test_fixed_committed_ram_restore_uses_bound_cache_factories_after_admission(
    monkeypatch,
) -> None:
    cache_environment = {
        "MTPLX_CONTEXT_COPY": "0",
        "MTPLX_SUSTAINED_PREFILL": "1",
        "MTPLX_SUSTAINED_PREFILL_LAYOUT": "auto",
        "MTPLX_MTP_HISTORY_POLICY": "committed",
        "MTPLX_VLLM_METAL_PAGED_MTP_ATTN": "0",
        "MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE": "1",
    }
    for name, value in cache_environment.items():
        monkeypatch.setenv(name, value)

    model = _FactoryTrackingK2Model()
    runtime = _tracking_runtime(model)
    restore_factories: list[tuple[object, object]] = []
    restore_modes: list[str] = []
    live_target_ref = [object()]
    live_mtp_ref = [object()]
    entry = SimpleNamespace(
        prefix_len=3,
        gdn_boundaries=[],
        cache_ref=live_target_ref,
        mtp_history_cache_ref=live_mtp_ref,
        live_ref_only=False,
    )

    class _RamBank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return entry

        def restore(
            self,
            _runtime,
            _prompt_ids,
            *,
            mode,
            cache_factory,
            mtp_cache_factory,
            **_kwargs,
        ):
            restore_modes.append(str(mode))
            restore_factories.append((cache_factory, mtp_cache_factory))
            cache = cache_factory()
            mtp_cache = mtp_cache_factory()
            cache[0].offset = 3
            mtp_cache[0].offset = 3
            return SimpleNamespace(
                entry=entry,
                cache=cache,
                logits=mx.array([[0.0, 10.0, 0.0, 0.0]], dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=mtp_cache,
                restore_mode="clone",
                cache_source="ram",
            )

    state = make_mtpk2_request_state(
        runtime,
        [0, 1, 2],
        request_id="frozen-ram-restore",
        **_request_kwargs(
            session_bank=_RamBank(),
            session_restore_mode="reference",
        ),
    )
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_MTP_ATTN", "1")
    monkeypatch.setattr(
        model,
        "make_cache",
        lambda: pytest.fail("RAM restore used mutated target cache factory"),
    )
    monkeypatch.setattr(
        model,
        "make_mtp_cache",
        lambda: pytest.fail("RAM restore used mutated MTP cache factory"),
    )
    monkeypatch.setattr(
        runtime,
        "make_cache",
        lambda: pytest.fail("RAM restore used ambient runtime target factory"),
    )
    monkeypatch.setattr(
        runtime,
        "make_mtp_cache",
        lambda: pytest.fail("RAM restore used ambient runtime MTP factory"),
    )
    monkeypatch.setattr(
        "mtplx.generation._session_live_frontier_reference_restore_enabled",
        lambda: pytest.fail("fixed RAM restore read live-frontier environment"),
    )

    ticket = state.start()

    assert isinstance(ticket, MTPK2VerifyTicket)
    assert restore_modes == ["clone"]
    assert entry.cache_ref is live_target_ref
    assert entry.mtp_history_cache_ref is live_mtp_ref
    assert restore_factories == [
        (
            state.config.target_prefill_cache_factory,
            state.config.mtp_cache_factory,
        )
    ]
    assert model.target_cache_builds == 1
    assert model.mtp_cache_builds == 1
    assert runtime.diagnostic_counters["make_mtp_cache_calls"] == 1
    assert model.calls == []
    state.close()


def test_fixed_committed_near_prefix_restore_uses_bound_mtp_factory(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_MTP_HISTORY_POLICY", "committed")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_MTP_ATTN", "0")
    monkeypatch.setenv("MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE", "1")
    model = _FactoryTrackingK2Model()
    runtime = _tracking_runtime(model)
    live_target_ref = [object()]
    live_mtp_ref = [object()]
    entry = SimpleNamespace(
        prefix_len=72,
        model_path=str(runtime.model_path),
        hidden_variant="post_norm",
        mtp_history_policy="committed",
        mtp_history_snapshot=object(),
        mtp_snapshot_epoch=72,
        snapshot_epoch=72,
        has_recurrent=False,
        live_ref_only=False,
        cache_ref=live_target_ref,
        mtp_history_cache_ref=live_mtp_ref,
        gdn_boundaries=[],
        hits=0,
        last_access_s=0.0,
        cache_source="ram",
    )
    restore_factories: list[tuple[object, object]] = []
    restore_modes: list[str] = []

    class _NearPrefixBank:
        last_miss_reason = None
        last_prefix_diagnostic = None

        def longest_prefix(self, _prompt_ids):
            return None

        def near_prefix_candidates(self, *_args, **_kwargs):
            return [(entry, 68)]

        def restore_entry_prefix_cache(
            self,
            _runtime,
            _entry,
            matched,
            *,
            mode,
            cache_factory,
            mtp_cache_factory,
            **_kwargs,
        ):
            restore_modes.append(str(mode))
            restore_factories.append((cache_factory, mtp_cache_factory))
            cache = cache_factory()
            mtp_cache = mtp_cache_factory()
            cache[0].offset = int(matched) - 1
            mtp_cache[0].offset = int(matched) - 1
            return cache, mtp_cache, "clone", int(matched), None

    state = make_mtpk2_request_state(
        runtime,
        list(range(70)),
        request_id="frozen-near-prefix-restore",
        **_request_kwargs(
            session_bank=_NearPrefixBank(),
            session_restore_mode="reference",
        ),
    )
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_MTP_ATTN", "1")
    monkeypatch.setattr(
        runtime,
        "make_mtp_cache",
        lambda: pytest.fail("near-prefix restore used ambient MTP factory"),
    )
    monkeypatch.setattr(
        "mtplx.generation._session_live_frontier_reference_restore_enabled",
        lambda: pytest.fail(
            "fixed near-prefix restore read live-frontier environment"
        ),
    )

    ticket = state.start()

    assert isinstance(ticket, MTPK2PrefillTicket)
    assert (ticket.prompt_start, ticket.prompt_stop) == (67, 68)
    assert restore_modes == ["clone"]
    assert entry.cache_ref is live_target_ref
    assert entry.mtp_history_cache_ref is live_mtp_ref
    assert restore_factories == [
        (
            state.config.target_prefill_cache_factory,
            state.config.mtp_cache_factory,
        )
    ]
    assert model.target_cache_builds == 1
    assert model.mtp_cache_builds == 1
    assert model.calls == []
    state.close()


def test_fixed_committed_ssd_restore_uses_bound_cache_factories_after_admission(
    monkeypatch,
) -> None:
    cache_environment = {
        "MTPLX_CONTEXT_COPY": "0",
        "MTPLX_SUSTAINED_PREFILL": "1",
        "MTPLX_SUSTAINED_PREFILL_LAYOUT": "auto",
        "MTPLX_MTP_HISTORY_POLICY": "committed",
        "MTPLX_VLLM_METAL_PAGED_MTP_ATTN": "0",
        "MTPLX_SESSION_NEAR_PREFIX_RESTORE": "0",
    }
    for name, value in cache_environment.items():
        monkeypatch.setenv(name, value)

    model = _FactoryTrackingK2Model()
    runtime = _tracking_runtime(model)
    snapshot = CacheSnapshot(states=(None,), meta_states=((3,),))
    record = SimpleNamespace(
        token_ids=(0, 1, 2),
        cache_snapshot=snapshot,
        logits=mx.array([[0.0, 10.0, 0.0, 0.0]], dtype=mx.float32),
        hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
        mtp_history_snapshot=snapshot,
        nbytes=0,
        restore_s=0.0,
        metadata={
            "model_path": str(runtime.model_path),
            "mtp_enabled": True,
            "hidden_variant": "post_norm",
            "mtp_history_policy": "committed",
            "snapshot_epoch": 3,
            "mtp_snapshot_epoch": 3,
        },
    )

    class _ColdTier:
        def lookup(self, *_args, **_kwargs):
            return record

    bank = SessionBank(cold_tier=_ColdTier())
    state = make_mtpk2_request_state(
        runtime,
        [0, 1, 2],
        request_id="frozen-ssd-restore",
        **_request_kwargs(session_bank=bank),
    )
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_MTP_ATTN", "1")
    monkeypatch.setattr(
        model,
        "make_cache",
        lambda: pytest.fail("SSD restore used mutated target cache factory"),
    )
    monkeypatch.setattr(
        model,
        "make_mtp_cache",
        lambda: pytest.fail("SSD restore used mutated MTP cache factory"),
    )
    monkeypatch.setattr(
        runtime,
        "make_cache",
        lambda: pytest.fail("SSD restore used ambient runtime target factory"),
    )
    monkeypatch.setattr(
        runtime,
        "make_mtp_cache",
        lambda: pytest.fail("SSD restore used ambient runtime MTP factory"),
    )

    ticket = state.start()

    assert isinstance(ticket, MTPK2VerifyTicket)
    assert bank.last_restore_source == "ssd"
    assert model.target_cache_builds == 1
    assert model.mtp_cache_builds == 1
    assert runtime.diagnostic_counters["make_mtp_cache_calls"] == 1
    assert model.calls == []
    state.close()


def test_actual_k2_machine_yields_before_each_real_1024_prefill_forward(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "256")
    model = _TrackingK2Model()
    runtime = _tracking_runtime(model)

    with prefill_chunk_size_override(1024):
        state = make_mtpk2_request_state(
            runtime,
            list(range(2051)),
            request_id="chunked",
            **_request_kwargs(),
        )

    first = state.start()
    assert state.prefill_chunk_tokens == 1024
    assert isinstance(first, MTPK2PrefillTicket)
    assert (first.prompt_start, first.prompt_stop) == (0, 1024)
    assert tuple(first.input_ids.shape) == (1, 1024)
    assert model.calls == []
    assert first.request_cache[0].offset == 0
    assert runtime.diagnostic_counters.get("prefill_chunks", 0) == 0
    assert _prefill_chunk_size() == 256

    first_result = state.execute_pending(
        lambda ticket: execute_solo_mtpk2_prefill_ticket(runtime, ticket)
    )
    assert model.calls == [1024]
    assert first.request_cache[0].offset == 1024
    assert runtime.diagnostic_counters.get("prefill_chunks", 0) == 0

    second = state.resume(first_result)
    assert isinstance(second, MTPK2PrefillTicket)
    assert (second.prompt_start, second.prompt_stop) == (1024, 2048)
    assert tuple(second.input_ids.shape) == (1, 1024)
    assert model.calls == [1024]
    assert second.request_cache[0].offset == 1024
    assert runtime.diagnostic_counters["prefill_chunks"] == 1


def test_actual_k2_machine_reaches_b1_verify_ticket_after_exact_prefill_chunks(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS", "0")
    model = _TrackingK2Model()
    runtime = _tracking_runtime(model)

    with prefill_chunk_size_override(1024):
        state = make_mtpk2_request_state(
            runtime,
            list(range(2051)),
            request_id="verify",
            **_request_kwargs(),
        )

    ticket = state.start()
    spans: list[tuple[int, int]] = []
    while isinstance(ticket, MTPK2PrefillTicket):
        spans.append((ticket.prompt_start, ticket.prompt_stop))
        result = state.execute_pending(
            lambda pending: execute_solo_mtpk2_prefill_ticket(runtime, pending)
        )
        ticket = state.resume(result)

    assert spans == [(0, 1024), (1024, 2048), (2048, 2050), (2050, 2051)]
    assert isinstance(ticket, MTPK2VerifyTicket)
    assert tuple(ticket.input_ids.shape) == (1, 3)
    assert len(ticket.draft_distributions) == 2
    assert ticket.request_cache[0].offset == 2051
    assert model.calls == [1024, 1024, 2, 1]
    assert state.tokens == [1]


def test_exact_session_restore_without_suffix_yields_no_fake_prefill_ticket(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    model = _TrackingK2Model()
    runtime = _tracking_runtime(model)
    restored_cache = [_OffsetCache()]
    restored_cache[0].offset = 3

    class _Bank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return SimpleNamespace(prefix_len=3)

        def restore(self, *_args, **_kwargs):
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=restored_cache,
                logits=mx.array([[0.0, 10.0, 0.0, 0.0]], dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                # The fixed lane admits only committed-history requests, so
                # an exact bank hit must carry its committed MTP cache too.
                mtp_history_cache=[],
                restore_mode="clone",
                cache_source="ram",
            )

    with prefill_chunk_size_override(1024):
        state = make_mtpk2_request_state(
            runtime,
            [10, 11, 12],
            request_id="restored",
            **_request_kwargs(session_bank=_Bank()),
        )

    ticket = state.start()
    assert isinstance(ticket, MTPK2VerifyTicket)
    assert model.calls == []
    assert ticket.request_cache is restored_cache
    assert ticket.request_cache[0].offset == 3


def test_restored_suffix_prefill_ticket_uses_absolute_1024_token_span(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_SMALL_SUFFIX_FUSED_MAX", "512")
    model = _TrackingK2Model()
    runtime = _tracking_runtime(model)
    restored_cache = [_OffsetCache()]
    restored_cache[0].offset = 1500

    class _Bank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return SimpleNamespace(prefix_len=1500)

        def restore(self, *_args, **_kwargs):
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=1500),
                cache=restored_cache,
                logits=mx.array([[0.0, 10.0, 0.0, 0.0]], dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                mtp_history_cache=[],
                restore_mode="clone",
                cache_source="ram",
            )

    with prefill_chunk_size_override(1024):
        state = make_mtpk2_request_state(
            runtime,
            list(range(2601)),
            request_id="restored-suffix",
            **_request_kwargs(
                mtp_history_policy="committed",
                session_bank=_Bank(),
            ),
        )

    ticket = state.start()
    assert isinstance(ticket, MTPK2PrefillTicket)
    assert (ticket.prompt_start, ticket.prompt_stop) == (1500, 2524)
    assert tuple(ticket.input_ids.shape) == (1, 1024)
    assert model.calls == []
    assert restored_cache[0].offset == 1500


def test_exact_prefix_plus_one_restored_suffix_preserves_final_chunk_timing(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_SMALL_SUFFIX_FUSED_MAX", "0")
    model = _TrackingK2Model()
    runtime = _tracking_runtime(model)
    restored_cache = [_OffsetCache()]
    restored_cache[0].offset = 3
    prefill_events: list[dict] = []

    class _Bank:
        last_miss_reason = None

        def longest_prefix(self, _prompt_ids):
            return SimpleNamespace(prefix_len=3)

        def restore(self, *_args, **_kwargs):
            return SimpleNamespace(
                entry=SimpleNamespace(prefix_len=3),
                cache=restored_cache,
                logits=mx.array([[0.0, 10.0, 0.0, 0.0]], dtype=mx.float32),
                hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
                # The fixed lane admits only committed-history requests, so
                # the restored prefix must include its committed MTP cache.
                mtp_history_cache=[],
                restore_mode="clone",
                cache_source="ram",
            )

    state = make_mtpk2_request_state(
        runtime,
        [10, 11, 12, 13],
        request_id="restored-plus-one",
        **_request_kwargs(
            session_bank=_Bank(),
            prefill_callback=prefill_events.append,
        ),
    )

    ticket = state.start()
    assert isinstance(ticket, MTPK2PrefillTicket)
    assert (ticket.prompt_start, ticket.prompt_stop) == (3, 4)
    prefill_result = state.execute_pending(
        lambda pending: execute_solo_mtpk2_prefill_ticket(runtime, pending)
    )
    verify = state.resume(prefill_result)
    assert isinstance(verify, MTPK2VerifyTicket)
    chunk = next(event for event in prefill_events if event["phase"] == "chunk")
    assert chunk["tokens_done"] == 3
    assert chunk["elapsed_s"] >= 0.0


def test_k2_admission_rejects_nested_decode_state_rebase(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_STATE_REBASE_EVERY", "8")
    model = _TrackingK2Model()

    with pytest.raises(RuntimeError, match="STATE_REBASE_EVERY=0"):
        make_mtpk2_request_state(
            _tracking_runtime(model),
            [0, 1, 2],
            request_id="rebase",
            **_request_kwargs(),
        )

    assert model.calls == []


def test_k2_request_freezes_compiled_verify_bank_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    compiled_environment = {
        "MTPLX_COMPILED_VERIFY": "1",
        "MTPLX_COMPILED_VERIFY_MAX_LEN": "3",
        "MTPLX_COMPILED_VERIFY_FORCE": "1",
        "MTPLX_COMPILED_VERIFY_PREWARM": "0",
        "MTPLX_COMPILED_VERIFY_MAX_CONTEXT": "4096",
        "MTPLX_COMPILED_VERIFY_BOUNDARY": "both",
        "MTPLX_COMPILED_VERIFY_DONATION": "1",
        "MTPLX_COMPILED_VERIFY_GROWTH_RESERVE": "512",
        "MTPLX_COMPILED_VERIFY_SHARED_TRACES": "1",
        "MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q": "16",
        "MTPLX_OWNED_ATTN_KV": "0",
        "MTPLX_OWNED_RECURRENT_STATE": "0",
    }
    for name, value in compiled_environment.items():
        monkeypatch.setenv(name, value)
    model = _TrackingK2Model()
    runtime = _tracking_runtime(model)
    state = make_mtpk2_request_state(
        runtime,
        [0, 1, 2],
        request_id="compiled-verify",
        **_request_kwargs(),
    )
    assert state.config.compiled_verify_mode == "on"
    assert state.config.compiled_verify_max_len == 3

    original_environment_get = os.environ.get

    def forbid_compiled_bank_environment(name, *args):
        if name in compiled_environment:
            pytest.fail(f"fixed request re-read compiled bank option {name}")
        return original_environment_get(name, *args)

    for name in compiled_environment:
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr(
        "mtplx.graphbank.os.environ.get",
        forbid_compiled_bank_environment,
    )
    ticket = state.start()
    while isinstance(ticket, MTPK2PrefillTicket):
        result = state.execute_pending(
            lambda pending: execute_solo_mtpk2_prefill_ticket(runtime, pending)
        )
        ticket = state.resume(result)
    assert isinstance(ticket, MTPK2VerifyTicket)
    assert isinstance(
        state.execute_pending(
            lambda pending: execute_solo_mtpk2_verify_ticket(runtime, pending)
        ),
        MTPK2VerifyResult,
    )
    state.close()


def test_k2_admission_binds_context_copy_ticket_under_live_default(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "1")
    model = _TrackingK2Model()
    runtime = _tracking_runtime(model)

    class _PromptMatch:
        def __init__(self, *_args, **_kwargs):
            pass

        def sync(self, _prompt):
            pass

        def find(self, _history, *, max_pos=None):
            assert max_pos == 12
            return 0, 4

    monkeypatch.setattr("mtplx.context_copy.NgramIndex", _PromptMatch)
    state = make_mtpk2_request_state(
        runtime,
        [1] * 12,
        request_id="context-copy-default",
        **_request_kwargs(),
    )
    assert state.context_copy_enabled is True

    ticket = state.start()
    while isinstance(ticket, MTPK2PrefillTicket):
        result = state.execute_pending(
            lambda pending: execute_solo_mtpk2_prefill_ticket(runtime, pending)
        )
        after_explicit_execution = list(model.calls)
        ticket = state.resume(result)
        if isinstance(ticket, MTPK2ContextCopyTicket | MTPK2VerifyTicket):
            assert model.calls == after_explicit_execution

    assert isinstance(ticket, MTPK2ContextCopyTicket)
    assert int(ticket.input_ids.shape[1]) > 3


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"adaptive_policy": SimpleNamespace(current_depth=1)}, "adaptive_policy=None"),
        ({"draft_margin_threshold": 0.5}, "draft_margin_threshold=None"),
    ],
)
def test_k2_admission_rejects_variable_depth_policies(
    monkeypatch,
    override,
    message,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    model = _TrackingK2Model()

    with pytest.raises(RuntimeError, match=message):
        make_mtpk2_request_state(
            _tracking_runtime(model),
            [0, 1, 2],
            request_id="variable-depth",
            **_request_kwargs(**override),
        )

    assert model.calls == []


@pytest.mark.parametrize(
    ("runtime_field", "lane_field", "request_override"),
    [
        ("mtp_enabled", None, None),
        (None, "missing", None),
        (None, ("backend_id", "other"), None),
        (None, ("depth", 1), None),
        (None, ("hidden_variant", "pre_norm"), None),
        (None, ("verify_strategy", "batched"), None),
        (None, ("verify_core", "stock"), None),
        (None, ("max_width", 1), None),
        (None, ("capture_commit_for", None), None),
        (None, None, {"speculative_depth": 1}),
        (None, None, {"base_hidden_variant": "pre_norm"}),
        (None, None, {"mtp_hidden_variant": "pre_norm"}),
        (None, None, {"verify_strategy": "batched"}),
        (None, None, {"verify_core": "stock"}),
        (None, None, {"draft_core": "device-d2"}),
        (None, None, {"mtp_cache_policy": "fresh"}),
        (None, None, {"mtp_history_policy": "cycle"}),
    ],
)
def test_k2_exact_admission_rejects_before_generator_publication(
    monkeypatch,
    runtime_field,
    lane_field,
    request_override,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    model = _TrackingK2Model()
    runtime = _tracking_runtime(model)
    if runtime_field == "mtp_enabled":
        runtime.mtp_enabled = False
    if lane_field == "missing":
        runtime.qwen27b_k2_dual_lane = None
    elif lane_field is not None:
        name, value = lane_field
        setattr(runtime.qwen27b_k2_dual_lane, name, value)
    generator_calls: list[bool] = []

    def published_generator(*_args, **_kwargs):
        generator_calls.append(True)
        raise AssertionError("invalid request reached generator publication")

    monkeypatch.setattr(
        "mtplx.generation._generate_mtpk_machine",
        published_generator,
    )
    with pytest.raises(RuntimeError, match="resumable K2 construction requires"):
        make_mtpk2_request_state(
            runtime,
            [0, 1, 2],
            request_id="bad-contract",
            **_request_kwargs(**(request_override or {})),
        )

    assert generator_calls == []
    assert model.calls == []


def test_k2_context_copy_and_lazy_bonus_remain_disabled_after_admission(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
    model = _TrackingK2Model()
    runtime = _tracking_runtime(model)
    state = make_mtpk2_request_state(
        runtime,
        [1] * 12,
        request_id="fixed-construction",
        **_request_kwargs(max_tokens=4),
    )
    assert state.context_copy_enabled is False
    assert state.lazy_bonus_verify_requested is False

    class _PromptMatchMustNotRun:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("fixed K2 request rebuilt a context-copy route")

    monkeypatch.setattr("mtplx.context_copy.NgramIndex", _PromptMatchMustNotRun)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "1")
    ticket = state.start()
    while isinstance(ticket, MTPK2PrefillTicket):
        result = state.execute_pending(
            lambda pending: execute_solo_mtpk2_prefill_ticket(runtime, pending)
        )
        after_explicit_execution = list(model.calls)
        ticket = state.resume(result)
        if isinstance(ticket, MTPK2VerifyTicket):
            assert model.calls == after_explicit_execution

    assert isinstance(ticket, MTPK2VerifyTicket)
    assert tuple(ticket.input_ids.shape) == (1, 3)
    assert ticket.acceptance_context.event["lazy_bonus_verify"] == {
        "enabled": False,
        "requested": False,
        "disabled_by": None,
        "min_depth": 2,
        "verify_input_tokens": 3,
        "draft_tokens": 2,
    }


def test_public_context_copy_route_stays_legacy_and_can_issue_variable_t(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "1")
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "0")
    model = _TrackingK2Model()

    class _PromptMatch:
        def __init__(self, *_args, **_kwargs):
            pass

        def sync(self, _prompt):
            pass

        def find(self, _history, *, max_pos=None):
            assert max_pos == 12
            return 0, 4

    def fail_fixed_state(*_args, **_kwargs):
        raise AssertionError("ordinary context-copy request entered fixed K2")

    monkeypatch.setattr("mtplx.context_copy.NgramIndex", _PromptMatch)
    monkeypatch.setattr(
        "mtplx.generation.make_mtpk2_request_state",
        fail_fixed_state,
    )
    output = generate_mtpk(
        _tracking_runtime(model),
        [1] * 12,
        **_request_kwargs(max_tokens=6),
    )

    assert output.stats.context_copy_active is True
    assert output.stats.context_copy_rounds == 1
    assert 6 in model.calls


def test_public_legacy_lazy_bonus_is_resolved_before_callback_env_change(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_DROP_EVENTS", "0")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "1")
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
    monkeypatch.setenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", "0")
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY_MIN_DEPTH", "2")
    model = _TrackingK2Model()

    def callback(_batch):
        monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "0")

    monkeypatch.setattr(
        "mtplx.generation.make_mtpk2_request_state",
        lambda *_args, **_kwargs: pytest.fail(
            "ordinary lazy-bonus request entered fixed K2"
        ),
    )
    output = generate_mtpk(
        _tracking_runtime(model),
        [0, 1, 2],
        token_callback=callback,
        **_request_kwargs(max_tokens=4),
    )

    lazy = output.stats.events[0]["lazy_bonus_verify"]
    assert lazy["requested"] is True
    assert lazy["enabled"] is True
    assert lazy["verify_input_tokens"] == 2


def test_k2_admission_installs_1024_chunks_independent_of_ambient(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "2048")
    model = _TrackingK2Model()

    with prefill_chunk_size_override(2048):
        state = make_mtpk2_request_state(
            _tracking_runtime(model),
            list(range(2051)),
            request_id="fixed-1024",
            **_request_kwargs(),
        )

    assert state.prefill_chunk_tokens == 1024
    first = state.start()
    assert isinstance(first, MTPK2PrefillTicket)
    assert (first.prompt_start, first.prompt_stop) == (0, 1024)
    assert tuple(first.input_ids.shape) == (1, 1024)
    assert model.calls == []


def test_k2_admission_freezes_enabled_context_copy_from_request_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    model = _TrackingK2Model()
    environment = {
        **os.environ,
        "MTPLX_CONTEXT_COPY": "1",
    }

    state = make_mtpk2_request_state_from_environment(
        _tracking_runtime(model),
        [0, 1, 2],
        request_id="context-copy",
        environment=environment,
        **_request_kwargs(),
    )

    assert state.context_copy_enabled is True
    state.close()


def test_k2_admission_rejects_long_context_depth_cap_before_publication(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY", "auto")
    monkeypatch.setenv("MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD", "0")
    monkeypatch.setenv("MTPLX_LONG_CONTEXT_MTP_DEPTH", "1")
    model = _TrackingK2Model()

    with pytest.raises(RuntimeError, match="effective speculative depth 2"):
        make_mtpk2_request_state(
            _tracking_runtime(model),
            [0, 1, 2],
            request_id="depth-cap",
            **_request_kwargs(),
        )

    assert model.calls == []


def test_public_requested_depth2_cap_routes_to_legacy_serial_depth1(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY", "auto")
    monkeypatch.setenv("MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD", "0")
    monkeypatch.setenv("MTPLX_LONG_CONTEXT_MTP_DEPTH", "1")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    model = _TrackingK2Model()

    def fail_fixed_state(*_args, **_kwargs):
        raise AssertionError("depth-capped serial request entered fixed K2")

    monkeypatch.setattr(
        "mtplx.generation.make_mtpk2_request_state",
        fail_fixed_state,
    )
    output = generate_mtpk(
        _tracking_runtime(model),
        [0, 1, 2],
        **_request_kwargs(max_tokens=3),
    )

    assert output.tokens == [1, 1, 1]
    assert output.stats.requested_speculative_depth == 2
    assert output.stats.speculative_depth == 1


def test_fixed_k2_state_only_publishes_exact_t3_verify_tickets(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    model = _RejectingK2Model()
    runtime = _tracking_runtime(model)
    state = make_mtpk2_request_state(
        runtime,
        [0, 1, 2],
        request_id="fixed-t3",
        **_request_kwargs(max_tokens=3),
    )

    ticket = state.start()
    shapes: list[tuple[int, ...]] = []
    while ticket is not None:
        if isinstance(ticket, MTPK2PrefillTicket):
            result = state.execute_pending(
                lambda pending: execute_solo_mtpk2_prefill_ticket(runtime, pending)
            )
        else:
            shapes.append(tuple(ticket.input_ids.shape))
            result = state.execute_pending(
                lambda pending: execute_solo_mtpk2_verify_ticket(runtime, pending)
            )
        ticket = state.resume(result)

    assert shapes
    assert shapes == [(1, 3)] * len(shapes)


@pytest.mark.parametrize(
    ("model_type", "max_tokens", "expected_steps"),
    [
        (_RejectingK2Model, 2, 1),
        (_RejectingSecondK2Model, 3, 2),
    ],
)
def test_fixed_k2_installs_exact_commit_hook_return_without_generic_repair(
    monkeypatch,
    model_type,
    max_tokens,
    expected_steps,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    runtime = _tracking_runtime(model_type())
    state = make_mtpk2_request_state(
        runtime,
        [0, 1, 2],
        request_id=f"commit-{expected_steps}",
        **_request_kwargs(max_tokens=max_tokens),
    )

    ticket = state.start()
    while isinstance(ticket, MTPK2PrefillTicket):
        prefill_result = state.execute_pending(
            lambda pending: execute_solo_mtpk2_prefill_ticket(runtime, pending)
        )
        ticket = state.resume(prefill_result)
    assert isinstance(ticket, MTPK2VerifyTicket)
    base_result = state.execute_pending(
        lambda pending: execute_solo_mtpk2_verify_ticket(runtime, pending)
    )
    replacement_cache = [_OffsetCache()]
    commit_calls: list[int] = []

    def commit_prefix(steps):
        commit_calls.append(int(steps))
        return replacement_cache

    def forbidden(*_args, **_kwargs):
        raise AssertionError("fixed K2 used a generic commit or target repair")

    monkeypatch.setattr("mtplx.gdn_capture.commit_captured_prefix", forbidden)
    monkeypatch.setattr(runtime, "forward_ar", forbidden)
    result = replace(base_result, commit_prefix=commit_prefix)
    assert state.resume(result) is None

    assert commit_calls == [expected_steps]
    assert state.target_cache is replacement_cache
    assert state.status == "finished"


def test_fixed_k2_binds_width1_commit_before_runtime_lane_can_change(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    runtime = _tracking_runtime(_RejectingK2Model())
    bind_calls: list[tuple[int, int]] = []
    committed_cache = [_OffsetCache()]

    def commit_route(_cache, _captures, *, steps):
        assert steps == 1
        return committed_cache

    def bind_route(width, row):
        bind_calls.append((int(width), int(row)))
        return commit_route

    runtime.qwen27b_k2_dual_lane.capture_commit_for = bind_route
    state = make_mtpk2_request_state(
        runtime,
        [0, 1, 2],
        request_id="bound-lane",
        **_request_kwargs(max_tokens=2),
    )
    assert bind_calls == [(1, 0)]

    runtime.qwen27b_k2_dual_lane = SimpleNamespace(
        capture_commit_for=lambda *_args, **_kwargs: pytest.fail(
            "executor re-read runtime lane after admission"
        )
    )
    ticket = state.start()
    while isinstance(ticket, MTPK2PrefillTicket):
        result = state.execute_pending(
            lambda pending: execute_solo_mtpk2_prefill_ticket(runtime, pending)
        )
        ticket = state.resume(result)
    result = state.execute_pending(
        lambda pending: execute_solo_mtpk2_verify_ticket(runtime, pending)
    )

    assert result.commit_prefix(1) is committed_cache
    assert bind_calls == [(1, 0)]
    state.close()


def test_fixed_k2_commit_hook_exception_fails_request_without_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    runtime = _tracking_runtime(_RejectingK2Model())
    state = make_mtpk2_request_state(
        runtime,
        [0, 1, 2],
        request_id="commit-error",
        **_request_kwargs(max_tokens=2),
    )

    ticket = state.start()
    while isinstance(ticket, MTPK2PrefillTicket):
        result = state.execute_pending(
            lambda pending: execute_solo_mtpk2_prefill_ticket(runtime, pending)
        )
        ticket = state.resume(result)
    base_result = state.execute_pending(
        lambda pending: execute_solo_mtpk2_verify_ticket(runtime, pending)
    )
    failure = RuntimeError("installed commit failed")

    def fail_commit(_steps):
        raise failure

    with pytest.raises(RuntimeError, match="installed commit failed"):
        state.resume(replace(base_result, commit_prefix=fail_commit))

    assert state.status == "failed"
    assert state.error is failure
    assert state.pending_ticket is None


@pytest.mark.parametrize(
    ("max_tokens", "expected_tokens", "expected_callbacks", "expected_stats"),
    [
        (1, [1], [[1]], (0, 0, 0)),
        (2, [1, 1], [[1], [1]], (1, 2, 1)),
        (3, [1, 1, 1], [[1], [1, 1]], (1, 2, 2)),
    ],
)
def test_fixed_k2_short_tails_keep_t3_and_preserve_visible_cap(
    monkeypatch,
    max_tokens,
    expected_tokens,
    expected_callbacks,
    expected_stats,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    model = _TrackingK2Model()
    runtime = _tracking_runtime(model)
    callbacks: list[list[int]] = []
    commit_calls: list[int] = []

    def commit_prefix(cache, _captures, *, steps):
        commit_calls.append(int(steps))
        trim = 3 - int(steps)
        for entry in cache:
            entry.trim(trim)
        return cache

    runtime.qwen27b_k2_dual_lane.capture_commit_for = (
        lambda _width, _row: commit_prefix
    )
    state = make_mtpk2_request_state(
        runtime,
        [0, 1, 2],
        request_id=f"tail-{max_tokens}",
        **_request_kwargs(
            max_tokens=max_tokens,
            token_callback=callbacks.append,
            capture_final_state=True,
        ),
    )

    verify_shapes: list[tuple[int, ...]] = []
    ticket = state.start()
    while ticket is not None:
        if isinstance(ticket, MTPK2PrefillTicket):
            result = state.execute_pending(
                lambda pending: execute_solo_mtpk2_prefill_ticket(runtime, pending)
            )
        else:
            verify_shapes.append(tuple(ticket.input_ids.shape))
            result = state.execute_pending(
                lambda pending: execute_solo_mtpk2_verify_ticket(runtime, pending)
            )
        ticket = state.resume(result)
    output = state.output

    assert verify_shapes == [(1, 3)] * len(verify_shapes)
    assert output.tokens == expected_tokens
    assert callbacks == expected_callbacks
    verify_calls, drafted_tokens, accepted_drafts = expected_stats
    assert output.stats.verify_calls == verify_calls
    assert output.stats.drafted_tokens == drafted_tokens
    assert output.stats.accepted_drafts == accepted_drafts
    assert output.final_state.generated_token_ids == tuple(expected_tokens)
    if max_tokens == 2:
        assert commit_calls == [2]
        # With sustained prefill frozen off at construction, this short prompt
        # is one causal T3 prefill followed by one fixed T3 verify. Committing
        # two verified positions trims the unused third row, leaving exactly
        # prompt(3) + generated(2) cache positions.
        assert model.calls == [3, 3]
        assert output.final_state.final_trunk_cache[0].offset == 5


def test_fixed_final_pending_bonus_uses_ticketed_t3_commit(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    control_runtime = _tracking_runtime(_RowSensitiveK2Model())
    control = generate_mtpk(
        control_runtime,
        [0, 1, 2],
        **_request_kwargs(max_tokens=4, capture_final_state=True),
    )
    model = _RowSensitiveK2Model()
    runtime = _tracking_runtime(model)
    commit_calls: list[int] = []

    def commit_prefix(cache, _captures, *, steps):
        commit_calls.append(int(steps))
        trim = 3 - int(steps)
        for entry in cache:
            entry.trim(trim)
        return cache

    runtime.qwen27b_k2_dual_lane.capture_commit_for = (
        lambda _width, _row: commit_prefix
    )
    state = make_mtpk2_request_state(
        runtime,
        [0, 1, 2],
        request_id="final-pending",
        **_request_kwargs(max_tokens=4, capture_final_state=True),
    )

    ticket = state.start()
    while isinstance(ticket, MTPK2PrefillTicket):
        result = state.execute_pending(
            lambda pending: execute_solo_mtpk2_prefill_ticket(runtime, pending)
        )
        ticket = state.resume(result)
    assert isinstance(ticket, MTPK2VerifyTicket)
    first_verify = state.execute_pending(
        lambda pending: execute_solo_mtpk2_verify_ticket(runtime, pending)
    )

    final_ticket = state.resume(first_verify)
    assert isinstance(final_ticket, MTPK2VerifyTicket)
    assert tuple(final_ticket.input_ids.shape) == (1, 3)
    assert final_ticket.acceptance_context.purpose == "final_commit"
    assert int(final_ticket.input_ids[0, 0].item()) == state.tokens[-1]

    before_final_dispatch = final_ticket.request_cache[0].offset
    final_result = state.execute_pending(
        lambda pending: execute_solo_mtpk2_verify_ticket(runtime, pending)
    )
    assert not mx.array_equal(
        final_result.logits[:, 0, :],
        final_result.logits[:, 1, :],
    )
    assert not mx.array_equal(
        final_result.hidden[:, 0:1, :],
        final_result.hidden[:, 1:2, :],
    )
    assert mx.array_equal(
        final_result.logits[:, 0, :],
        control.final_state.final_logits,
    )
    assert mx.array_equal(
        final_result.hidden[:, 0:1, :],
        control.final_state.final_hidden,
    )
    assert final_result.request_cache[0].offset == before_final_dispatch + 3
    assert state.resume(final_result) is None
    output = state.output

    assert commit_calls == [1]
    assert output.final_state.final_trunk_cache[0].offset == before_final_dispatch + 1
    assert state.status == "finished"
    assert output.tokens == control.tokens == [1, 1, 1, 1]
    assert output.final_state.safe_to_commit is True
    assert (
        output.final_state.final_trunk_cache[0].offset
        == control.final_state.final_trunk_cache[0].offset
        == 7
    )
    assert mx.array_equal(
        output.final_state.final_logits,
        control.final_state.final_logits,
    )
    assert mx.array_equal(
        output.final_state.final_hidden,
        control.final_state.final_hidden,
    )


def test_generate_mtpk_restores_explicit_keyword_only_public_signature() -> None:
    parameters = inspect.signature(generate_mtpk).parameters
    assert list(parameters) == [
        "rt",
        "prompt_ids",
        "abort_check",
        "max_tokens",
        "sampler",
        "speculative_depth",
        "seed",
        "stop_token_ids",
        "base_hidden_variant",
        "mtp_hidden_variant",
        "mtp_cache_policy",
        "mtp_history_policy",
        "draft_sampler",
        "draft_margin_threshold",
        "min_speculative_depth",
        "verify_strategy",
        "verify_core",
        "draft_core",
        "mtp_corrector",
        "adaptive_policy",
        "online_hidden_corrector_alpha",
        "online_hidden_corrector_decay",
        "online_hidden_corrector_warmup",
        "online_hidden_corrector_max_feed_depth",
        "online_hidden_corrector_key",
        "online_correction_cache",
        "online_correction_cache_min_depth",
        "online_correction_cache_key",
        "prompt_correction_cache",
        "prompt_correction_cache_min_depth",
        "adapter_ensemble_q",
        "adapter_ensemble_epsilon",
        "adapter_ensemble_min_depth",
        "mtp_topk_reranker",
        "token_callback",
        "session_bank",
        "session_id",
        "session_restore_mode",
        "session_template_hash",
        "session_draft_head_identity",
        "session_policy_fingerprint",
        "capture_final_state",
        "commit_prompt_state_to_bank",
        "commit_prompt_state_keep_live_ref",
        "trace_label",
        "trace_metadata",
        "prefill_callback",
        "repetition_stop",
        "loop_guard",
        "thinking_guard",
        "vision_splice",
        "constraint",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in parameters.items()
        if name not in {"rt", "prompt_ids"}
    )
    assert "kwargs" not in parameters


def test_public_depth2_stays_ordinary_serial_with_ambient_prefill_chunk(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setattr(
        generation_module,
        "_GENERATION_FEATURE_POLICY",
        generation_module.bind_generation_feature_policy(os.environ),
    )
    model = _TrackingK2Model()
    runtime = _tracking_runtime(model)
    callback_batches: list[list[int]] = []
    monkeypatch.setattr(
        "mtplx.generation.make_mtpk2_request_state",
        lambda *_args, **_kwargs: pytest.fail(
            "ordinary depth-two request entered fixed K2"
        ),
    )

    with prefill_chunk_size_override(1024):
        output = generate_mtpk(
            runtime,
            [0, 1, 2, 3],
            token_callback=callback_batches.append,
            **_request_kwargs(),
        )

    assert output.tokens == [1, 1, 1, 1]
    assert callback_batches == [[1], [1, 1], [1]]
    assert output.stats.generated_tokens == 4
    assert output.stats.verify_calls == 1
    assert output.stats.accepted_drafts == 2
    assert output.stats.rejected_drafts == 0
    assert output.stats.bonus_tokens == 1
    assert output.stats.prefill_chunk_size == 1024
    assert model.calls == [3, 1, 3]


def test_public_depth2_rejection_preserves_capture_commit_and_callback_order(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_DROP_EVENTS", "0")
    monkeypatch.setattr(
        generation_module,
        "_GENERATION_FEATURE_POLICY",
        generation_module.bind_generation_feature_policy(os.environ),
    )
    model = _RejectingK2Model()
    runtime = _tracking_runtime(model)
    callback_batches: list[list[int]] = []
    commit_calls: list[tuple[int, int]] = []

    def commit_prefix(cache, _captures, *, keep_tokens, verified_tokens, **_kwargs):
        commit_calls.append((int(keep_tokens), int(verified_tokens)))
        trim = int(verified_tokens) - int(keep_tokens)
        for entry in cache:
            entry.trim(trim)
        return True

    monkeypatch.setattr("mtplx.gdn_capture.commit_captured_prefix", commit_prefix)
    monkeypatch.setattr(
        "mtplx.generation.make_mtpk2_request_state",
        lambda *_args, **_kwargs: pytest.fail(
            "ordinary rejecting request entered fixed K2"
        ),
    )

    with prefill_chunk_size_override(1024):
        output = generate_mtpk(
            runtime,
            [0, 1, 2, 3],
            token_callback=callback_batches.append,
            **_request_kwargs(max_tokens=3),
        )

    assert output.tokens == [1, 1, 1]
    assert callback_batches == [[1], [1], [1]]
    assert output.stats.accepted_drafts == 0
    assert output.stats.rejected_drafts == 2
    assert output.stats.correction_tokens == 0
    assert output.stats.verify_calls == 2
    assert output.stats.events[0]["rejected_at_depth"] == 1
    assert output.stats.events[0]["capture_repair"] == "captured_prefix_commit"
    assert commit_calls == [(1, 3), (1, 2)]
    assert model.calls == [3, 1, 3, 2]
