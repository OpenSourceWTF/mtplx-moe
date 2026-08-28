from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.cache_state import rollback_after_verify, snapshot_cache
from mtplx.graphbank import promote_kv_cache_offsets
from mtplx.models.qwen4_omlx import (
    QSAIndexer,
    QSAKVCache,
    RotaryEmbedding,
    TextArgs,
    _IndexerCache,
)


@pytest.fixture(autouse=True)
def _cpu_stream():
    with mx.stream(mx.cpu):
        yield


def _indexer() -> QSAIndexer:
    return QSAIndexer(
        TextArgs(
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=8,
            indexer_budget=8,
            indexer_compress_ratio=4,
        )
    )


def _qk(rows: int) -> mx.array:
    values = mx.arange(rows * 24, dtype=mx.float32).reshape(1, rows, 24)
    return mx.sin(values * mx.array(0.017, dtype=mx.float32)).astype(mx.float16)


def test_incremental_qsa_mask_equals_full_history_oracle() -> None:
    indexer = _indexer()
    rope = RotaryEmbedding(4, 10_000.0)
    qk = _qk(20)

    expected = indexer.select_projected(qk, rope, None, 0)
    cache = _IndexerCache()
    indexer.select_projected(qk[:, :12], rope, cache, 0)
    pooled_prefix = cache.pooled_keys
    actual = indexer.select_projected(qk[:, 12:], rope, cache, 12)
    mx.eval(expected, actual, pooled_prefix, cache.pooled_keys)

    assert cache.offset == 20
    assert cache.pooled_offset == 5
    assert mx.array_equal(cache.pooled_keys[:, :3], pooled_prefix)
    assert mx.array_equal(actual, expected[:, :, 12:, :])


def test_qsa_trim_preserves_only_wholly_accepted_pooled_blocks() -> None:
    indexer = _indexer()
    rope = RotaryEmbedding(4, 10_000.0)
    qk = _qk(20)
    cache = _IndexerCache()
    indexer.select_projected(qk[:, :14], rope, cache, 0)

    assert cache.pooled_offset == 3
    assert cache.trim(3) == 3
    assert cache.offset == 11
    assert cache.pooled_offset == 2

    actual = indexer.select_projected(qk[:, 11:], rope, cache, 11)
    expected = indexer.select_projected(qk, rope, None, 0)
    mx.eval(actual, expected)

    assert cache.offset == 20
    assert cache.pooled_offset == 5
    assert mx.array_equal(actual, expected[:, :, 11:, :])


def test_qsa_state_restore_keeps_raw_authority_and_drops_derived_pool() -> None:
    indexer = _indexer()
    rope = RotaryEmbedding(4, 10_000.0)
    cache = QSAKVCache()
    keys = mx.zeros((1, 1, 12, 8), dtype=mx.float16)
    values = mx.zeros((1, 1, 12, 8), dtype=mx.float16)
    cache.update_and_fetch(keys, values)
    indexer.select_projected(_qk(12), rope, cache.indexer, 0)
    state = cache.state
    meta_state = cache.meta_state

    restored = QSAKVCache.from_state(state, meta_state)

    assert restored.offset == 12
    assert restored.indexer.offset == 12
    assert restored.indexer.pooled_offset == 0
    assert restored.indexer.pooled_keys is None
    assert mx.array_equal(restored.indexer.keys[:, :12], cache.indexer.keys[:, :12])


def test_qsa_prompt_cache_serializes_raw_keys_as_tensor_state(tmp_path) -> None:
    from mlx_lm.models import cache as mlx_cache

    indexer = _indexer()
    cache = QSAKVCache()
    cache.update_and_fetch(
        mx.zeros((1, 1, 12, 8), dtype=mx.float16),
        mx.zeros((1, 1, 12, 8), dtype=mx.float16),
    )
    indexer.select_projected(_qk(12), RotaryEmbedding(4, 10_000.0), cache.indexer, 0)
    path = tmp_path / "qsa.safetensors"

    mlx_cache.save_prompt_cache(str(path), [cache])
    restored = mlx_cache.load_prompt_cache(str(path))[0]

    assert restored.meta_state == "4"
    assert restored.offset == 12
    assert restored.indexer.offset == 12
    assert restored.indexer.pooled_offset == 0
    assert mx.array_equal(restored.indexer.state, cache.indexer.state)


def test_generic_qsa_rollback_restores_raw_prefix_correctly() -> None:
    indexer = _indexer()
    rope = RotaryEmbedding(4, 10_000.0)
    cache = QSAKVCache()
    cache.update_and_fetch(
        mx.zeros((1, 1, 12, 8), dtype=mx.float16),
        mx.zeros((1, 1, 12, 8), dtype=mx.float16),
    )
    indexer.select_projected(_qk(12), rope, cache.indexer, 0)
    snapshot = snapshot_cache([cache])

    cache.update_and_fetch(
        mx.zeros((1, 1, 2, 8), dtype=mx.float16),
        mx.zeros((1, 1, 2, 8), dtype=mx.float16),
    )
    indexer.select_projected(_qk(14)[:, 12:], rope, cache.indexer, 12)
    rollback_after_verify([cache], snapshot, verified_tokens=2)

    assert cache.offset == 12
    assert cache.indexer.offset == 12
    assert cache.indexer.pooled_offset == 0
    actual = indexer.select_projected(_qk(20)[:, 12:], rope, cache.indexer, 12)
    expected = indexer.select_projected(_qk(20), rope, None, 0)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected[:, :, 12:, :])


def test_qsa_raw_cache_reserves_the_full_16k_1k_decode_window() -> None:
    cache = _IndexerCache()
    visible = cache.update(mx.zeros((1, 16_384, 8), dtype=mx.float16))

    assert visible.shape == (1, 16_384, 8)
    assert cache.keys.shape[1] >= 17_410


def test_graphbank_does_not_drop_qsa_auxiliary_state_during_promotion() -> None:
    entry = QSAKVCache()
    entry.update_and_fetch(
        mx.zeros((1, 1, 12, 8), dtype=mx.float16),
        mx.zeros((1, 1, 12, 8), dtype=mx.float16),
    )
    cache = [entry]

    promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=2)

    assert promoted == 0
    assert failures == {"auxiliary_qsa_state": 1}
    assert cache[0] is entry
