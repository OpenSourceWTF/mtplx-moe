from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.models.qwen4_omlx import QSAKVCache


def _config() -> dict:
    return {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "hc_count": 4,
            "hc_lowrank": 320,
            "num_hidden_layers": 48,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "output_gate_type": "sigmoid",
            "rms_norm_eps": 1e-6,
        },
    }


def _cache() -> list[QSAKVCache]:
    cache = [QSAKVCache(4)]
    for entry in cache:
        entry.keys = mx.zeros((1, 2, 1, 4), dtype=mx.bfloat16)
        entry.values = mx.zeros((1, 2, 1, 4), dtype=mx.bfloat16)
        entry.offset = 1
        entry.indexer.keys = mx.zeros((1, 1, 4), dtype=mx.bfloat16)
        entry.indexer.offset = 1
        entry.indexer._pooled_keys = mx.zeros((1, 1, 4), dtype=mx.bfloat16)
        entry.indexer.pooled_offset = 1
    return cache


def _runtime(cache: list[QSAKVCache] | None = None):
    constructed = _cache() if cache is None else cache
    runtime = SimpleNamespace(
        model=SimpleNamespace(make_mtp_cache=lambda: [QSAKVCache(4)]),
        qwen4_depth1_batched_target_arrays=True,
    )
    runtime.make_mtp_cache = runtime.model.make_mtp_cache

    def draft_mtp(
        hidden,
        token,
        *,
        mtp_cache,
        return_hidden,
        mtp_hidden_variant,
        mtp_depth,
        position_offset,
    ):
        del mtp_hidden_variant, mtp_depth, position_offset
        assert return_hidden
        for entry in mtp_cache:
            entry.keys = entry.keys + mx.ones_like(entry.keys)
            entry.values = entry.values + mx.ones_like(entry.values)
            entry.indexer.keys = entry.indexer.keys + mx.ones_like(entry.indexer.keys)
            entry.indexer._pooled_keys = (
                entry.indexer._pooled_keys + mx.ones_like(entry.indexer._pooled_keys)
            )
        logits = mx.zeros((1, 1, 32), dtype=mx.float32) + token.reshape(1, 1, 1)
        return logits, hidden + mx.ones_like(hidden)

    runtime.draft_mtp = draft_mtp
    runtime.live_cache = constructed
    return runtime


def test_installer_binds_only_exact_qwen4_qsa_topology():
    from mtplx.qwen4_cycle_fold import install_qwen4_cycle_fold

    runtime = _runtime()

    report = install_qwen4_cycle_fold(runtime, config=_config())

    assert report == {"installed": True, "ticket_rows": 1, "qsa_layers": 1}
    assert callable(runtime.qwen4_cycle_fold_issue)


def test_installer_rejects_non_qsa_mtp_cache():
    from mtplx.qwen4_cycle_fold import (
        Qwen4CycleFoldConfigError,
        install_qwen4_cycle_fold,
    )

    runtime = _runtime()
    runtime.make_mtp_cache = lambda: [SimpleNamespace()]

    with pytest.raises(Qwen4CycleFoldConfigError, match="one QSA cache entry"):
        install_qwen4_cycle_fold(runtime, config=_config())


def test_installer_rejects_before_exact_depth1_runtime_is_bound():
    from mtplx.qwen4_cycle_fold import (
        Qwen4CycleFoldConfigError,
        install_qwen4_cycle_fold,
    )

    runtime = _runtime()
    runtime.qwen4_depth1_batched_target_arrays = False

    with pytest.raises(Qwen4CycleFoldConfigError, match="depth-one runtime"):
        install_qwen4_cycle_fold(runtime, config=_config())


def test_ticket_submits_logits_hidden_and_all_qsa_arrays(monkeypatch):
    from mtplx.qwen4_cycle_fold import install_qwen4_cycle_fold

    submitted = []
    monkeypatch.setattr(mx, "async_eval", lambda *roots: submitted.extend(roots))
    cache = _cache()
    runtime = _runtime(cache)
    install_qwen4_cycle_fold(runtime, config=_config())
    hidden = mx.zeros((1, 1, 8), dtype=mx.bfloat16)

    ticket = runtime.qwen4_cycle_fold_issue(
        hidden=hidden,
        primary=17,
        mtp_cache=cache,
        mtp_hidden_variant="post_norm",
        compiled_aux_prefetch="owned-prefetch",
    )

    assert ticket.primary == 17
    assert ticket.compiled_aux_prefetch == "owned-prefetch"
    assert submitted[:2] == [ticket.logits, ticket.hidden]
    expected = tuple(
        leaf
        for entry in cache
        for leaf in (
            entry.keys,
            entry.values,
            entry.indexer.keys,
            entry.indexer._pooled_keys,
        )
    )
    assert tuple(submitted[2:]) == expected
