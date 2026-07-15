from __future__ import annotations

import inspect

import mlx.core as mx
import numpy as np
import pytest

import mtplx.generation as generation
from mtplx.generation import (
    COMPILED_MTP_DRAFT_DEPTHS,
    COMPILED_MTP_DRAFT_PRIMARY_DEPTH,
    COMPILED_MTP_DRAFT_PRIMARY_WIDTH,
    CompiledMTPDraftBank,
    CompiledMTPDraftNotPrewarmed,
    _device_argmax_token,
    compiled_mtp_draft_fallback_reason,
    compiled_mtp_draft_report,
)


class _DenseToyCache:
    """Stock-KV-shaped cache used to prove live committed-state parity."""

    step = 32

    def __init__(self, *, capacity: int = 32) -> None:
        self.keys = mx.zeros((1, 1, capacity, 1), dtype=mx.float32)
        self.values = mx.zeros((1, 1, capacity, 1), dtype=mx.float32)
        self.offset = 0

    def update_and_fetch(self, keys, values):
        steps = int(keys.shape[2])
        self.keys = mx.slice_update(
            self.keys,
            keys,
            mx.array(self.offset, dtype=mx.int32),
            axes=(2,),
        )
        self.values = mx.slice_update(
            self.values,
            values,
            mx.array(self.offset, dtype=mx.int32),
            axes=(2,),
        )
        self.offset += steps
        return self.keys, self.values

    def trim(self, count: int) -> int:
        self.offset = max(0, self.offset - int(count))
        return int(count)


class _ToyDraftRuntime:
    vocab_size = 64

    def __init__(self, *, fail_on_depth: int | None = None) -> None:
        self.fail_on_depth = fail_on_depth
        self.finished_caches: list[list[object]] = []

    def draft_mtp(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache,
        return_hidden,
        mtp_hidden_variant,
        mtp_depth,
    ):
        del mtp_hidden_variant
        assert return_hidden is True
        depth = int(mtp_depth)
        if self.fail_on_depth == depth:
            raise RuntimeError("synthetic prewarm failure")
        token = (next_token_ids.astype(mx.int32) + depth) % self.vocab_size
        kv = token.astype(mx.float32).reshape(1, 1, 1, 1)
        mtp_cache[0].update_and_fetch(kv, kv + 100.0)
        vocab = mx.arange(self.vocab_size).reshape(1, 1, -1)
        logits = -mx.abs(vocab - token.reshape(1, 1, 1)).astype(mx.float32)
        hidden = hidden_states + token.astype(mx.float32).reshape(1, 1, 1)
        return logits, hidden

    def finish_mtp_cycle(self, mtp_cache) -> None:
        self.finished_caches.append(mtp_cache)


def _seeded_cache(*, capacity: int = 32) -> list[_DenseToyCache]:
    cache = _DenseToyCache(capacity=capacity)
    for token in (7.0, 11.0):
        value = mx.array([[[[token]]]], dtype=mx.float32)
        cache.update_and_fetch(value, value + 100.0)
    mx.eval(cache.keys, cache.values)
    return [cache]


def _install_fake_compile(monkeypatch) -> list[object]:
    compile_calls: list[object] = []

    def fake_compile(fn):
        compile_calls.append(fn)
        return fn

    monkeypatch.setattr(generation.mx, "compile", fake_compile)
    return compile_calls


def _serial_draft(
    runtime: _ToyDraftRuntime,
    *,
    depth: int,
    hidden,
    token_ids,
    cache,
) -> list[int]:
    tokens = []
    current_hidden = hidden
    current_token = token_ids
    for draft_depth in range(1, depth + 1):
        logits, current_hidden = runtime.draft_mtp(
            current_hidden,
            current_token,
            mtp_cache=cache,
            return_hidden=True,
            mtp_hidden_variant="post_norm",
            mtp_depth=draft_depth,
        )
        current_token = _device_argmax_token(logits)
        tokens.append(current_token)
    matrix = mx.concatenate(tokens, axis=1)
    mx.eval(matrix, cache[0].keys, cache[0].values)
    return [int(value) for value in np.asarray(matrix).reshape(-1)]


def test_compiled_bank_caches_one_callable_per_depth_and_shape(monkeypatch) -> None:
    compile_calls = _install_fake_compile(monkeypatch)
    runtime = _ToyDraftRuntime()
    bank = CompiledMTPDraftBank(runtime, mtp_hidden_variant="post_norm")
    hidden = mx.zeros((1, 1, 1), dtype=mx.float32)
    token_ids = mx.array([[3]], dtype=mx.int32)

    for depth in COMPILED_MTP_DRAFT_DEPTHS:
        cache = _seeded_cache()
        first = bank.prewarm(
            depth,
            hidden,
            token_ids,
            mtp_cache=cache,
            reserve_tokens=16,
        )
        second = bank.prewarm(
            depth,
            hidden,
            token_ids,
            mtp_cache=cache,
            reserve_tokens=16,
        )
        assert first["compiled_callable"] is True
        assert first["cache_hit"] is False
        assert second["compiled_callable"] is False
        assert second["cache_hit"] is True
        assert first["cache_leaf_count"] == 3

    report = bank.to_dict()
    assert COMPILED_MTP_DRAFT_DEPTHS == tuple(range(1, 8))
    assert COMPILED_MTP_DRAFT_PRIMARY_DEPTH == 3
    assert COMPILED_MTP_DRAFT_PRIMARY_WIDTH == 4
    assert report["cached_depths"] == list(COMPILED_MTP_DRAFT_DEPTHS)
    assert len(compile_calls) == len(COMPILED_MTP_DRAFT_DEPTHS)
    assert all(
        report["per_depth"][str(depth)]["compile_count"] == 1
        for depth in COMPILED_MTP_DRAFT_DEPTHS
    )


def test_compiled_depth_uses_device_dependencies_and_explicit_live_state(
    monkeypatch,
) -> None:
    _install_fake_compile(monkeypatch)
    bank = CompiledMTPDraftBank(_ToyDraftRuntime(), mtp_hidden_variant="post_norm")
    hidden = mx.zeros((1, 1, 1), dtype=mx.float32)
    cache = _seeded_cache()

    report = bank.prewarm(
        7,
        hidden,
        mx.array([[2]], dtype=mx.int32),
        mtp_cache=cache,
        reserve_tokens=16,
    )

    assert report["cache_leaf_count"] == 3
    assert report["cache_slots"] == 1
    assert report["device_dependency_edges"] == 6
    assert report["prewarm_host_syncs"] == 1
    assert ".item(" not in inspect.getsource(bank._trace_depth)
    assert ".item(" not in inspect.getsource(bank.run)


@pytest.mark.parametrize("depth", COMPILED_MTP_DRAFT_DEPTHS)
def test_compiled_tokens_and_live_committed_cache_match_serial(
    monkeypatch,
    depth: int,
) -> None:
    _install_fake_compile(monkeypatch)
    hidden = mx.zeros((1, 1, 1), dtype=mx.float32)
    token_ids = mx.array([[5]], dtype=mx.int32)
    serial_cache = _seeded_cache()
    compiled_cache = _seeded_cache()
    expected_tokens = _serial_draft(
        _ToyDraftRuntime(),
        depth=depth,
        hidden=hidden,
        token_ids=token_ids,
        cache=serial_cache,
    )
    before_offset = compiled_cache[0].offset
    before_keys = np.asarray(compiled_cache[0].keys).copy()
    before_values = np.asarray(compiled_cache[0].values).copy()
    bank = CompiledMTPDraftBank(
        _ToyDraftRuntime(),
        mtp_hidden_variant="post_norm",
    )

    bank.prewarm(
        depth,
        hidden,
        token_ids,
        mtp_cache=compiled_cache,
        reserve_tokens=16,
    )
    assert compiled_cache[0].offset == before_offset
    assert np.array_equal(np.asarray(compiled_cache[0].keys), before_keys)
    assert np.array_equal(np.asarray(compiled_cache[0].values), before_values)

    tokens, dispatch = bank.run(
        depth,
        hidden,
        primary=5,
        mtp_cache=compiled_cache,
        reserve_tokens=16,
    )
    mx.eval(compiled_cache[0].keys, compiled_cache[0].values)

    assert tokens == expected_tokens
    assert dispatch["committed_history"] is True
    assert dispatch["host_syncs"] == 1
    assert dispatch["host_token_transfers"] == 1
    assert dispatch["live_cache_commits"] == 1
    assert compiled_cache[0].offset == serial_cache[0].offset
    assert np.array_equal(
        np.asarray(compiled_cache[0].keys),
        np.asarray(serial_cache[0].keys),
    )
    assert np.array_equal(
        np.asarray(compiled_cache[0].values),
        np.asarray(serial_cache[0].values),
    )


@pytest.mark.parametrize("depth", COMPILED_MTP_DRAFT_DEPTHS)
def test_real_mx_compile_commits_live_cache_state(depth: int) -> None:
    hidden = mx.zeros((1, 1, 1), dtype=mx.float32)
    serial_cache = _seeded_cache()
    compiled_cache = _seeded_cache()
    expected_tokens = _serial_draft(
        _ToyDraftRuntime(),
        depth=depth,
        hidden=hidden,
        token_ids=mx.array([[5]], dtype=mx.int32),
        cache=serial_cache,
    )
    bank = CompiledMTPDraftBank(
        _ToyDraftRuntime(),
        mtp_hidden_variant="post_norm",
    )
    bank.prewarm(
        depth,
        hidden,
        mx.array([[5]], dtype=mx.int32),
        mtp_cache=compiled_cache,
        reserve_tokens=16,
    )

    tokens, _dispatch = bank.run(
        depth,
        hidden,
        primary=5,
        mtp_cache=compiled_cache,
        reserve_tokens=16,
    )
    mx.eval(compiled_cache[0].keys, compiled_cache[0].values)

    assert tokens == expected_tokens
    assert compiled_cache[0].offset == serial_cache[0].offset
    assert np.array_equal(
        np.asarray(compiled_cache[0].keys),
        np.asarray(serial_cache[0].keys),
    )
    assert np.array_equal(
        np.asarray(compiled_cache[0].values),
        np.asarray(serial_cache[0].values),
    )


def test_compiled_dispatch_never_compiles_or_retraces_organically(monkeypatch) -> None:
    compile_calls = _install_fake_compile(monkeypatch)
    runtime = _ToyDraftRuntime()
    bank = CompiledMTPDraftBank(runtime, mtp_hidden_variant="post_norm")
    hidden = mx.zeros((1, 1, 1), dtype=mx.float32)

    with pytest.raises(CompiledMTPDraftNotPrewarmed):
        bank.run(
            3,
            hidden,
            primary=1,
            mtp_cache=_seeded_cache(),
            reserve_tokens=16,
        )

    bank.prewarm(
        3,
        hidden,
        mx.array([[1]], dtype=mx.int32),
        mtp_cache=_seeded_cache(),
        reserve_tokens=16,
    )
    compile_count = len(compile_calls)
    with pytest.raises(CompiledMTPDraftNotPrewarmed, match="shape was not prewarmed"):
        bank.run(
            3,
            hidden,
            primary=1,
            mtp_cache=_seeded_cache(capacity=64),
            reserve_tokens=16,
        )

    assert len(compile_calls) == compile_count == 1
    assert bank.to_dict()["organic_compile_calls"] == 0


def test_failed_prewarm_finishes_scratch_backend_cycle(monkeypatch) -> None:
    _install_fake_compile(monkeypatch)
    runtime = _ToyDraftRuntime(fail_on_depth=2)
    bank = CompiledMTPDraftBank(runtime, mtp_hidden_variant="post_norm")

    with pytest.raises(RuntimeError, match="synthetic prewarm failure"):
        bank.prewarm(
            3,
            mx.zeros((1, 1, 1), dtype=mx.float32),
            mx.array([[1]], dtype=mx.int32),
            mtp_cache=_seeded_cache(),
            reserve_tokens=16,
        )

    assert len(runtime.finished_caches) == 1


def test_failed_dispatch_finishes_scratch_cycle_without_committing_live_state(
    monkeypatch,
) -> None:
    _install_fake_compile(monkeypatch)
    runtime = _ToyDraftRuntime()
    bank = CompiledMTPDraftBank(runtime, mtp_hidden_variant="post_norm")
    cache = _seeded_cache()
    hidden = mx.zeros((1, 1, 1), dtype=mx.float32)
    token_ids = mx.array([[1]], dtype=mx.int32)
    bank.prewarm(
        3,
        hidden,
        token_ids,
        mtp_cache=cache,
        reserve_tokens=16,
    )
    before_offset = cache[0].offset
    before_keys = np.asarray(cache[0].keys).copy()
    before_values = np.asarray(cache[0].values).copy()
    runtime.fail_on_depth = 2

    with pytest.raises(RuntimeError, match="synthetic prewarm failure"):
        bank.run(
            3,
            hidden,
            primary=1,
            mtp_cache=cache,
            reserve_tokens=16,
        )

    assert len(runtime.finished_caches) == 2
    assert cache[0].offset == before_offset
    assert np.array_equal(np.asarray(cache[0].keys), before_keys)
    assert np.array_equal(np.asarray(cache[0].values), before_values)


@pytest.mark.parametrize(
    ("cache", "message"),
    [
        ([], "exactly one live cache slot"),
        ([_DenseToyCache(), _DenseToyCache()], "exactly one live cache slot"),
    ],
)
def test_prewarm_fails_closed_for_unsupported_live_cache(
    monkeypatch,
    cache,
    message: str,
) -> None:
    _install_fake_compile(monkeypatch)
    bank = CompiledMTPDraftBank(_ToyDraftRuntime(), mtp_hidden_variant="post_norm")

    with pytest.raises(RuntimeError, match=message):
        bank.prewarm(
            3,
            mx.zeros((1, 1, 1), dtype=mx.float32),
            mx.array([[1]], dtype=mx.int32),
            mtp_cache=cache,
            reserve_tokens=16,
        )


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"depth": 8, "requested_depth": 8}, "unsupported_depth"),
        ({"requested": "device-d2", "depth": 3}, "legacy_device_d2_depth"),
        ({"draft_temperature": 0.5}, "non_greedy_draft_sampler"),
        ({"mtp_cache_policy": "fresh"}, "mtp_cache_policy:fresh"),
        ({"mtp_history_policy": "cycle"}, "mtp_history_policy:cycle"),
        ({"draft_margin_threshold": 0.1}, "draft_margin_threshold"),
        ({"adaptive": True}, "adaptive_depth"),
        ({"mtp_corrector": True}, "mtp_corrector"),
        ({"online_hidden_corrector": True}, "online_hidden_corrector"),
        ({"correction_cache": True}, "correction_cache"),
        ({"adapter_ensemble": True}, "adapter_ensemble"),
        ({"topk_reranker": True}, "topk_reranker"),
        ({"position_mode": "absolute"}, "mtp_position_policy:absolute"),
        ({"dynamic_depth": True}, "dynamic_depth"),
        ({"depth": 4, "requested_depth": 3}, "depth_above_requested"),
    ],
)
def test_compiled_contract_fails_closed_with_reason(override, reason: str) -> None:
    values = {
        "requested": "device-k",
        "depth": 3,
        "requested_depth": 3,
        "draft_temperature": 0.0,
        "mtp_cache_policy": "persistent",
        "mtp_history_policy": "committed",
        "draft_margin_threshold": None,
        "adaptive": False,
        "mtp_corrector": False,
        "online_hidden_corrector": False,
        "correction_cache": False,
        "adapter_ensemble": False,
        "topk_reranker": False,
        "position_mode": "cache",
        "dynamic_depth": False,
    }
    values.update(override)

    assert compiled_mtp_draft_fallback_reason(**values) == reason


def test_compiled_contract_accepts_fixed_greedy_committed_depths() -> None:
    for depth in COMPILED_MTP_DRAFT_DEPTHS:
        assert (
            compiled_mtp_draft_fallback_reason(
                requested="device-k",
                depth=depth,
                requested_depth=depth,
                draft_temperature=0.0,
                mtp_cache_policy="persistent",
                mtp_history_policy="committed",
                draft_margin_threshold=None,
                adaptive=False,
                mtp_corrector=False,
                online_hidden_corrector=False,
                correction_cache=False,
                adapter_ensemble=False,
                topk_reranker=False,
                position_mode="cache",
                dynamic_depth=False,
            )
            is None
        )


def test_public_report_exposes_per_depth_committed_cache_metrics(monkeypatch) -> None:
    _install_fake_compile(monkeypatch)
    runtime = _ToyDraftRuntime()
    bank = CompiledMTPDraftBank(runtime, mtp_hidden_variant="post_norm")
    runtime._compiled_mtp_draft_banks = {"post_norm": bank}
    bank.prewarm(
        3,
        mx.zeros((1, 1, 1), dtype=mx.float32),
        mx.array([[1]], dtype=mx.int32),
        mtp_cache=_seeded_cache(),
        reserve_tokens=16,
    )

    report = compiled_mtp_draft_report(runtime)

    assert report["primary_depth"] == 3
    assert report["primary_width"] == 4
    depth = report["variants"]["post_norm"]["per_depth"]["3"]
    assert depth["compile_count"] == 1
    assert depth["shape_prewarm_count"] == 1
