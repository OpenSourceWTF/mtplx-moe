"""Focused tests for the selected-row QSA attention candidate."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.base import scaled_dot_product_attention  # noqa: E402

from mtplx.models.qwen4_omlx import (  # noqa: E402
    Attention,
    QSAIndexer,
    QSAKVCache,
    TextArgs,
    _qsa_compact_runtime_enabled,
    _qsa_compact_attention,
)
from mtplx.attention_context import attention_phase  # noqa: E402
from mtplx.qwen4_projection_fusion import FusedProjectionAttention  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_stream():
    with mx.stream(mx.cpu):
        yield


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=2,
        indexer_budget=4,
        indexer_compress_ratio=2,
        rms_norm_eps=1e-6,
    )


def _identity_rope(positions):
    shape = (*positions.shape, 2)
    return mx.ones(shape, dtype=mx.float32), mx.zeros(shape, dtype=mx.float32)


def _history_cache(
    indexer: QSAIndexer,
    length: int,
    attention_head_dim: int | None = None,
    batch: int = 1,
) -> QSAKVCache:
    cache = QSAKVCache(indexer.compress_ratio)
    cache.indexer.state = mx.zeros(
        (batch, length, indexer.head_dim), dtype=mx.float32
    )
    head_dim = indexer.head_dim if attention_head_dim is None else attention_head_dim
    cache.update_and_fetch(
        mx.zeros((batch, 1, length, head_dim), dtype=mx.float32),
        mx.zeros((batch, 1, length, head_dim), dtype=mx.float32),
    )
    return cache


def _materialize(*arrays):
    mx.eval(*arrays)
    return [np.asarray(array) for array in arrays]


def test_compact_selection_preserves_incremental_selected_set_and_cache_ownership():
    indexer = QSAIndexer(_tiny_args())
    qk = mx.arange(3 * 4, dtype=mx.float32).reshape(1, 3, 4) / 7
    dense_cache = _history_cache(indexer, 8)
    compact_cache = _history_cache(indexer, 8)

    dense = indexer.select_projected(
        qk, _identity_rope, dense_cache.indexer, offset=8
    )
    compact = indexer.select_projected_compact(
        qk, _identity_rope, compact_cache.indexer, offset=8
    )
    dense_np, indices_np, valid_np = _materialize(
        dense, compact.indices, compact.valid
    )

    assert compact.indices.shape == (1, 3, 2 * 2 + 1)
    assert dense_cache.indexer_offset == compact_cache.indexer_offset == 11
    for row in range(qk.shape[1]):
        expected = set(np.flatnonzero(dense_np[0, 0, row]))
        observed = set(indices_np[0, row][valid_np[0, row]])
        assert observed == expected


def test_compact_attention_matches_dense_selected_rows():
    indexer = QSAIndexer(_tiny_args())
    qk = mx.arange(7 * 4, dtype=mx.float32).reshape(1, 7, 4) / 11
    selection = indexer.select_projected_compact(
        qk, _identity_rope, None, offset=7
    )
    dense_keep = indexer.select_projected(qk, _identity_rope, None, offset=7)

    q = mx.arange(2 * 7 * 4, dtype=mx.float32).reshape(1, 2, 7, 4) / 13
    k = mx.arange(1 * 2 * 7 * 4, dtype=mx.float32).reshape(1, 2, 7, 4) / 17
    v = mx.sin(mx.arange(1 * 2 * 7 * 4, dtype=mx.float32)).reshape(1, 2, 7, 4)
    neg = mx.finfo(q.dtype).min
    additive = mx.where(
        mx.arange(7)[None, None, None, :] == 2,
        mx.array(-0.75, mx.float32),
        mx.array(0, mx.float32),
    )
    dense_mask = mx.where(
        dense_keep, mx.array(0, q.dtype), mx.array(neg, q.dtype)
    ) + additive
    dense = scaled_dot_product_attention(
        q, k, v, cache=None, scale=0.5, mask=dense_mask
    )
    compact = _qsa_compact_attention(
        q, k, v, selection, scale=0.5, mask=additive
    )
    dense_np, compact_np = _materialize(dense, compact)
    np.testing.assert_allclose(compact_np, dense_np, rtol=1e-2, atol=2e-3)


def test_boolean_mask_uses_true_permitted_false_negative_semantics():
    indexer = QSAIndexer(_tiny_args())
    qk = mx.arange(7 * 4, dtype=mx.float32).reshape(1, 7, 4) / 11
    selection = indexer.select_projected_compact(
        qk, _identity_rope, None, offset=7
    )
    dense_keep = indexer.select_projected(qk, _identity_rope, None, offset=7)

    q = mx.arange(2 * 7 * 4, dtype=mx.float32).reshape(1, 2, 7, 4) / 13
    k = mx.arange(1 * 2 * 7 * 4, dtype=mx.float32).reshape(1, 2, 7, 4) / 17
    v = mx.sin(mx.arange(1 * 2 * 7 * 4, dtype=mx.float32)).reshape(1, 2, 7, 4)
    permitted = mx.arange(7)[None, None, None, :] != 2
    neg = mx.finfo(q.dtype).min
    dense_mask = mx.where(
        dense_keep & permitted, mx.array(0, q.dtype), mx.array(neg, q.dtype)
    )
    dense = scaled_dot_product_attention(
        q, k, v, cache=None, scale=0.5, mask=dense_mask
    )
    compact = _qsa_compact_attention(
        q, k, v, selection, scale=0.5, mask=permitted
    )
    dense_np, compact_np = _materialize(dense, compact)
    np.testing.assert_allclose(compact_np, dense_np, rtol=1e-2, atol=2e-3)


def test_ratio_four_boundaries_invisible_picks_and_partial_tails():
    args = TextArgs(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=2,
        indexer_budget=8,
        indexer_compress_ratio=4,
        rms_norm_eps=1e-6,
    )
    indexer = QSAIndexer(args)
    qk = mx.arange(12 * 4, dtype=mx.float32).reshape(1, 12, 4) / 7
    dense = indexer.select_projected(qk, _identity_rope, None, offset=0)
    compact = indexer.select_projected_compact(qk, _identity_rope, None, offset=0)
    dense_np, indices_np, valid_np = _materialize(
        dense, compact.indices, compact.valid
    )
    assert compact.indices.shape == (1, 12, 2 * 4 + 3)
    for row in range(12):
        expected = set(np.flatnonzero(dense_np[0, 0, row]))
        observed = set(indices_np[0, row][valid_np[0, row]])
        assert observed == expected
    assert [int(valid_np[0, row].sum()) for row in range(4)] == [1, 2, 3, 4]

    for new_rows, tail in ((1, 1), (2, 2), (3, 3), (4, 0)):
        dense_cache = _history_cache(indexer, 8)
        compact_cache = _history_cache(indexer, 8)
        current = qk[:, :new_rows]
        dense = indexer.select_projected(
            current, _identity_rope, dense_cache.indexer, offset=8
        )
        compact = indexer.select_projected_compact(
            current, _identity_rope, compact_cache.indexer, offset=8
        )
        dense_np, indices_np, valid_np = _materialize(
            dense, compact.indices, compact.valid
        )
        expected_tail = set(range(8, 8 + tail))
        observed_last = set(indices_np[0, -1][valid_np[0, -1]])
        assert expected_tail <= observed_last
        assert observed_last == set(np.flatnonzero(dense_np[0, 0, -1]))


def test_prefill_rows_stay_dense_even_when_tiny(monkeypatch):
    layer = Attention(_tiny_args())
    rope = _identity_rope

    def compact_must_not_run(*_args, **_kwargs):
        raise AssertionError("compact QSA must not run during prefill")

    monkeypatch.setattr(layer.indexer, "select_projected_compact", compact_must_not_run)
    for rows in (2, 3, 4):
        x = mx.arange(rows * 8, dtype=mx.float32).reshape(1, rows, 8) / 9
        with attention_phase("prefill"):
            out = layer(x, rope, None, None, None)
        mx.eval(out)


def test_logical_m_is_sequence_rows_for_batched_incremental_calls(monkeypatch):
    layer = Attention(_tiny_args())
    cache = _history_cache(layer.indexer, 8, attention_head_dim=4, batch=2)
    x = mx.arange(2 * 3 * 8, dtype=mx.float32).reshape(2, 3, 8) / 9
    calls = 0
    compact = layer.indexer.select_projected_compact

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return compact(*args, **kwargs)

    monkeypatch.setattr(layer.indexer, "select_projected_compact", counted)
    with attention_phase("decode_verify"):
        out = layer(x, _identity_rope, None, cache, cache.indexer)
    mx.eval(out)
    assert calls == 1


def test_compact_route_is_installed_for_stock_and_fused_attention():
    stock_source = inspect.getsource(Attention.__call__)
    fused_source = inspect.getsource(FusedProjectionAttention.__call__)
    for source in (stock_source, fused_source):
        assert "select_projected_compact" in source
        assert "_qsa_compact_attention" in source
        assert "_qsa_compact_runtime_enabled" in source
    assert "QSA_COMPACT_MAX_ROWS" in inspect.getsource(_qsa_compact_runtime_enabled)


def test_fused_attention_matches_stock_compact_and_wide_routes():
    args = _tiny_args()
    stock = Attention(args)
    fused = FusedProjectionAttention(args)
    fused.update(stock.parameters())
    fused._mtplx_fused_max_rows = 4
    fused._mtplx_fused_indexer_qkv_splits = [4, 20, 24]
    fused._mtplx_fused_indexer_qkv_proj = lambda x: mx.concatenate(
        [
            fused.indexer.index_qk_proj(x),
            fused.q_proj(x),
            fused.k_proj(x),
            fused.v_proj(x),
        ],
        axis=-1,
    )
    rope = _identity_rope
    x_small = mx.arange(3 * 8, dtype=mx.float32).reshape(1, 3, 8) / 9
    stock_small_cache = _history_cache(stock.indexer, 8, args.head_dim)
    fused_small_cache = _history_cache(fused.indexer, 8, args.head_dim)
    with attention_phase("decode_verify"):
        stock_small = stock(
            x_small, rope, None, stock_small_cache, stock_small_cache.indexer
        )
        fused_small = fused(
            x_small, rope, None, fused_small_cache, fused_small_cache.indexer
        )

    x_wide = mx.arange(5 * 8, dtype=mx.float32).reshape(1, 5, 8) / 9
    stock_wide_cache = _history_cache(stock.indexer, 8, args.head_dim)
    fused_wide_cache = _history_cache(fused.indexer, 8, args.head_dim)
    stock_wide = stock(
        x_wide, rope, None, stock_wide_cache, stock_wide_cache.indexer
    )
    fused_wide = fused(
        x_wide, rope, None, fused_wide_cache, fused_wide_cache.indexer
    )
    stock_small, fused_small, stock_wide, fused_wide = _materialize(
        stock_small, fused_small, stock_wide, fused_wide
    )
    np.testing.assert_allclose(fused_small, stock_small, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fused_wide, stock_wide, rtol=1e-5, atol=1e-5)
