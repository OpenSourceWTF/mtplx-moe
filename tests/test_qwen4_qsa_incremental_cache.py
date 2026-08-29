from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.cache_state import rollback_after_verify, snapshot_cache
from mtplx.graphbank import (
    CompiledVerifyBank,
    TensorOffsetQSAKVCache,
    build_verify_state_spec,
    promote_kv_cache_offsets,
)
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


def _populated_qsa_cache(rows: int = 12) -> QSAKVCache:
    entry = QSAKVCache()
    entry.update_and_fetch(
        mx.zeros((1, 1, rows, 8), dtype=mx.float16),
        mx.zeros((1, 1, rows, 8), dtype=mx.float16),
    )
    _indexer().select_projected(
        _qk(rows), RotaryEmbedding(4, 10_000.0), entry.indexer, 0
    )
    return entry


def test_graphbank_promotes_qsa_with_all_auxiliary_state() -> None:
    entry = _populated_qsa_cache()
    cache = [entry]

    promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=2)

    assert promoted == 1
    assert failures == {}
    assert isinstance(cache[0], TensorOffsetQSAKVCache)
    leaves = cache[0].state_leaves
    assert len(leaves) == 7
    assert all(isinstance(leaf, mx.array) for leaf in leaves)
    assert int(cache[0].size()) == 12
    assert int(cache[0].indexer.size()) == 12
    assert int(cache[0].indexer.pooled_size()) == 3

    spec, reason = build_verify_state_spec(cache)
    assert reason is None
    assert spec == [(0, "qsa", 7)]


def test_qsa_promotion_right_shapes_fixed_buffers_to_request_envelope() -> None:
    cache = [_populated_qsa_cache(rows=12)]

    promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=8)

    assert promoted == 1 and failures == {}
    entry = cache[0]
    assert entry.keys.shape[2] == 20
    assert entry.values.shape[2] == 20
    assert entry.indexer.raw_keys.shape[1] == 20
    assert entry.indexer.pooled_keys.shape[1] == 5


def test_tensor_qsa_trim_keeps_kv_raw_and_pool_offsets_consistent() -> None:
    cache = [_populated_qsa_cache()]
    promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=2)
    assert promoted == 1 and failures == {}
    entry = cache[0]
    entry.update_and_fetch(
        mx.zeros((1, 1, 2, 8), dtype=mx.float16),
        mx.zeros((1, 1, 2, 8), dtype=mx.float16),
    )
    entry.indexer.update(_qk(14)[:, 12:, -8:])
    entry.indexer.set_pooled_offset(mx.array(3, dtype=mx.int32))

    assert entry.trim(1) == 1
    mx.eval(entry.offset, entry.indexer.offset, entry.indexer.pooled_offset)

    assert int(entry.size()) == 13
    assert int(entry.indexer.size()) == 13
    assert int(entry.indexer.pooled_size()) == 3


def test_compiled_bank_demotes_tensor_qsa_to_stock_cache() -> None:
    cache = [_populated_qsa_cache()]
    promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=2)
    assert promoted == 1 and failures == {}
    adapter = cache[0]
    expected_raw = mx.array(adapter.indexer.raw_keys)
    expected_pool = mx.array(adapter.indexer.pooled_keys)
    bank = CompiledVerifyBank(object())

    assert bank.demote(cache) == 1
    restored = cache[0]
    assert type(restored) is QSAKVCache
    assert restored.offset == 12
    assert restored.indexer.offset == 12
    assert restored.indexer.pooled_offset == 3
    assert mx.array_equal(restored.indexer.keys[:, :12], expected_raw[:, :12])
    assert mx.array_equal(restored.indexer.pooled_keys, expected_pool[:, :3])


@pytest.mark.parametrize("prefix_rows", [12, 13, 14, 15])
def test_tensor_qsa_compact_selection_matches_eager_across_pool_phases(
    prefix_rows: int,
) -> None:
    indexer = _indexer()
    rope = RotaryEmbedding(4, 10_000.0)
    qk = _qk(prefix_rows + 2)
    eager = QSAKVCache()
    fixed = QSAKVCache()
    for entry in (eager, fixed):
        entry.update_and_fetch(
            mx.zeros((1, 1, prefix_rows, 8), dtype=mx.float16),
            mx.zeros((1, 1, prefix_rows, 8), dtype=mx.float16),
        )
        indexer.select_projected_compact(qk[:, :prefix_rows], rope, entry.indexer, 0)
    fixed_cache = [fixed]
    promoted, failures = promote_kv_cache_offsets(fixed_cache, reserve_tokens=7)
    assert promoted == 1 and failures == {}
    fixed = fixed_cache[0]

    expected = indexer.select_projected_compact(
        qk[:, prefix_rows:], rope, eager.indexer, prefix_rows
    )
    actual = indexer.select_projected_compact(
        qk[:, prefix_rows:], rope, fixed.indexer, fixed.offset
    )
    mx.eval(expected.indices, expected.valid, actual.indices, actual.valid)

    assert mx.array_equal(actual.indices, expected.indices)
    assert mx.array_equal(actual.valid, expected.valid)
    assert int(fixed.indexer.size()) == prefix_rows + 2
    assert int(fixed.indexer.pooled_size()) == (prefix_rows + 2) // 4
    assert mx.array_equal(
        fixed.indexer.raw_keys[:, : prefix_rows + 2],
        eager.indexer.keys[:, : prefix_rows + 2],
    )
    assert mx.array_equal(
        fixed.indexer.pooled_keys[:, : (prefix_rows + 2) // 4],
        eager.indexer.pooled_keys,
    )


def test_tensor_qsa_dense_selection_masks_fixed_capacity_for_context_copy_rows() -> None:
    indexer = _indexer()
    rope = RotaryEmbedding(4, 10_000.0)
    prefix_rows = 12
    verify_rows = 5
    qk = _qk(prefix_rows + verify_rows)
    eager = QSAKVCache()
    fixed = QSAKVCache()
    for entry in (eager, fixed):
        entry.update_and_fetch(
            mx.zeros((1, 1, prefix_rows, 8), dtype=mx.float16),
            mx.zeros((1, 1, prefix_rows, 8), dtype=mx.float16),
        )
        indexer.select_projected(qk[:, :prefix_rows], rope, entry.indexer, 0)

    fixed_cache = [fixed]
    promoted, failures = promote_kv_cache_offsets(fixed_cache, reserve_tokens=8)
    assert promoted == 1 and failures == {}
    fixed = fixed_cache[0]

    expected = indexer.select_projected(
        qk[:, prefix_rows:], rope, eager.indexer, prefix_rows
    )
    actual = indexer.select_projected(
        qk[:, prefix_rows:], rope, fixed.indexer, fixed.offset
    )
    mx.eval(expected, actual)

    logical_length = prefix_rows + verify_rows
    assert actual.shape == (1, 1, verify_rows, fixed.keys.shape[2])
    assert mx.array_equal(actual[..., :logical_length], expected)
    assert not mx.any(actual[..., logical_length:]).item()


def test_tensor_qsa_rejected_row_is_overwritten_before_pool_visibility() -> None:
    indexer = _indexer()
    rope = RotaryEmbedding(4, 10_000.0)
    prefix = _qk(14)
    rejected = _qk(16)[:, 14:]
    replacement = mx.cos(mx.arange(24, dtype=mx.float32) * 0.031).reshape(1, 1, 24)
    accepted_history = mx.concatenate([prefix, rejected[:, :1], replacement], axis=1)

    fixed = QSAKVCache()
    fixed.update_and_fetch(
        mx.zeros((1, 1, 14, 8), dtype=mx.float16),
        mx.zeros((1, 1, 14, 8), dtype=mx.float16),
    )
    indexer.select_projected_compact(prefix, rope, fixed.indexer, 0)
    fixed_cache = [fixed]
    promote_kv_cache_offsets(fixed_cache, reserve_tokens=8)
    fixed = fixed_cache[0]
    indexer.select_projected_compact(rejected, rope, fixed.indexer, fixed.offset)
    fixed.update_and_fetch(
        mx.zeros((1, 1, 2, 8), dtype=mx.float16),
        mx.zeros((1, 1, 2, 8), dtype=mx.float16),
    )
    fixed.trim(1)
    actual = indexer.select_projected_compact(
        replacement, rope, fixed.indexer, fixed.offset
    )

    oracle = _IndexerCache()
    expected = indexer.select_projected_compact(accepted_history, rope, oracle, 0)
    mx.eval(actual.indices, actual.valid, expected.indices, expected.valid)
    assert mx.array_equal(actual.indices, expected.indices[:, -1:])
    assert mx.array_equal(actual.valid, expected.valid[:, -1:])
    assert int(fixed.indexer.size()) == 16
    assert int(fixed.indexer.pooled_size()) == 4
