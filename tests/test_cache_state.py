from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.attention_context import attention_phase
from mtplx.cache_state import (
    BlockOwnedKVCache,
    OwnedRecurrentStateCache,
    TailOwnedKVCache,
    TensorOffsetVllmMetalPagedKVCache,
    VllmMetalPagedKVCache,
    _paged_gqa_sdpa_route_decision_from_env,
    _paged_gqa_sdpa_route_from_env,
    configure_owned_recurrent_state_cache,
    configure_tail_owned_attention_kv_cache,
    detach_array_leaf,
    detach_attention_cache_state,
    detach_cache_state,
    detach_recurrent_cache_state,
    install_block_owned_attention_kv_cache,
    install_owned_recurrent_state_cache,
    install_tail_owned_attention_kv_cache,
    install_vllm_metal_paged_attention_kv_cache,
    owned_recurrent_state_stats,
    rollback_after_verify,
    restore_cache,
    snapshot_cache,
    snapshot_untrimmable_cache,
    tail_owned_attention_kv_stats,
    trim_verified_window_to_prefix,
)
from mtplx.kv_quant import PagedKVQuantConfig


class DummyCache:
    def __init__(self):
        self.state = [mx.array([1, 2, 3])]
        self.meta_state = ("meta", "3")


def test_restore_cache_rewinds_mutated_array_state():
    cache = [DummyCache()]
    snap = snapshot_cache(cache)
    cache[0].state = [mx.array([9])]
    cache[0].meta_state = ("meta", "1")

    restore_cache(cache, snap)

    assert cache[0].meta_state == ("meta", "3")
    assert cache[0].state[0].tolist() == [1, 2, 3]


def test_restore_cache_can_skip_layout_specific_meta_state():
    cache = [DummyCache()]
    snap = snapshot_cache(cache)
    cache[0].state = [mx.array([9])]
    cache[0].meta_state = ("dense-layout", "1")

    restore_cache(cache, snap, restore_meta_state=False)

    assert cache[0].meta_state == ("dense-layout", "1")
    assert cache[0].state[0].tolist() == [1, 2, 3]


def test_restore_cache_preserves_list_state_identity():
    cache = [DummyCache()]
    original_state = cache[0].state
    snap = snapshot_cache(cache)
    cache[0].state[0] = mx.array([9])

    restore_cache(cache, snap)

    assert cache[0].state is original_state
    assert cache[0].state[0].tolist() == [1, 2, 3]


def test_snapshot_cache_does_not_alias_later_mlx_array_mutation():
    from mlx_lm.models.cache import KVCache

    kv = KVCache()
    keys = mx.array([[[[1.0], [2.0]]]])
    values = mx.array([[[[3.0], [4.0]]]])
    kv.update_and_fetch(keys, values)
    snap = snapshot_cache([kv])

    kv.update_and_fetch(mx.array([[[[9.0]]]]), mx.array([[[[10.0]]]]))
    restore_cache([kv], snap)

    assert kv.offset == 2
    assert kv.keys.tolist() == [[[[1.0], [2.0]]]]
    assert kv.values.tolist() == [[[[3.0], [4.0]]]]


class TrimmableDummyCache:
    def __init__(self):
        self.trimmed = 0
        self.offset = 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.trimmed += n
        self.offset -= n
        return n

    @property
    def state(self):
        return [mx.array([5])]

    @state.setter
    def state(self, value):
        raise AssertionError("trimmable cache should not be restored by state assignment")

    @property
    def meta_state(self):
        return ""


def test_rollback_after_verify_trims_kv_and_restores_recurrent_state():
    recurrent = DummyCache()
    kv = TrimmableDummyCache()
    cache = [recurrent, kv]
    snap = snapshot_untrimmable_cache(cache)

    recurrent.state = [mx.array([9])]
    recurrent.meta_state = ("meta", "advanced")
    rollback_after_verify(cache, snap, verified_tokens=3)

    assert recurrent.meta_state == ("meta", "3")
    assert recurrent.state[0].tolist() == [1, 2, 3]
    assert kv.trimmed == 3


def test_trim_verified_window_to_prefix_requires_all_trimmable_snapshot():
    kv = TrimmableDummyCache()
    kv.offset = 8
    snap = snapshot_untrimmable_cache([kv])

    assert trim_verified_window_to_prefix(
        [kv],
        snap,
        verified_tokens=5,
        keep_tokens=2,
    )
    assert kv.offset == 5
    assert kv.trimmed == 3

    recurrent = DummyCache()
    kv = TrimmableDummyCache()
    kv.offset = 8
    snap = snapshot_untrimmable_cache([recurrent, kv])

    assert not trim_verified_window_to_prefix(
        [recurrent, kv],
        snap,
        verified_tokens=5,
        keep_tokens=2,
    )
    assert kv.offset == 8


def test_detach_recurrent_cache_state_replaces_requested_list_leaves():
    recurrent = DummyCache()
    recurrent.state = [
        mx.array([1, 2, 3]) + mx.zeros((), dtype=mx.int32),
        mx.array([4, 5, 6]) + mx.zeros((), dtype=mx.int32),
    ]
    original_state = recurrent.state
    original_conv = original_state[0]
    original_gdn = original_state[1]

    stats = detach_recurrent_cache_state(
        [recurrent],
        components={"gdn"},
        mode="contiguous_eval",
    )

    assert recurrent.state is original_state
    assert recurrent.state[0] is original_conv
    assert recurrent.state[1] is not original_gdn
    assert recurrent.state[1].tolist() == [4, 5, 6]
    assert stats["entries"] == 1
    assert stats["arrays"] == 1
    assert stats["bytes"] == recurrent.state[1].nbytes


def test_detach_recurrent_cache_state_skips_trimmable_entries():
    kv = TrimmableDummyCache()

    stats = detach_recurrent_cache_state(
        [kv],
        components={"gdn", "conv"},
        mode="contiguous_eval",
    )

    assert stats == {"entries": 0, "arrays": 0, "bytes": 0}


def test_owned_recurrent_state_cache_reuses_fixed_buffers():
    owned = OwnedRecurrentStateCache(size=2)
    first = mx.array([[1.0, 2.0]])
    second = mx.array([[3.0, 4.0]]) + mx.zeros((), dtype=mx.float32)

    owned.replace_state([None, first])
    first_buffer = owned[1]
    owned.replace_state([None, second])

    assert owned[1] is first_buffer
    assert owned[1].tolist() == [[3.0, 4.0]]
    assert owned.owner_allocations == 1
    assert owned.owner_inplace_updates == 1
    assert owned.owner_updates == 2


def test_owned_recurrent_state_cache_keeps_speculative_writes_out_of_owner_buffer():
    owned = OwnedRecurrentStateCache(size=2)
    owned.replace_state([None, mx.array([[1.0]])])
    owner_buffer = owned[1]

    owned[1] = mx.array([[9.0]])
    assert owned[1].tolist() == [[9.0]]
    assert owned.owner_updates == 1

    owned.replace_state([None, mx.array([[2.0]])])
    assert owned[1] is owner_buffer
    assert owned[1].tolist() == [[2.0]]
    assert owned.owner_updates == 2


def test_owned_recurrent_state_restore_uses_owner_buffers():
    owned = OwnedRecurrentStateCache(size=2)
    owned.replace_state([mx.array([[1.0]]), mx.array([[2.0]])])
    first_conv = owned[0]
    first_gdn = owned[1]
    snap = snapshot_cache([owned])

    owned.replace_state([mx.array([[9.0]]), mx.array([[10.0]])])
    restore_cache([owned], snap)

    assert owned[0] is first_conv
    assert owned[1] is first_gdn
    assert owned[0].tolist() == [[1.0]]
    assert owned[1].tolist() == [[2.0]]


def test_install_owned_recurrent_state_cache_replaces_arrays_cache_only():
    from mlx_lm.models.cache import ArraysCache, KVCache

    recurrent = ArraysCache(size=2)
    kv = KVCache()
    cache = [recurrent, kv]

    stats = install_owned_recurrent_state_cache(cache)

    assert stats["entries"] == 1
    assert stats["skipped"] == 1
    assert isinstance(cache[0], OwnedRecurrentStateCache)
    assert cache[1] is kv


def test_configure_owned_recurrent_state_cache_uses_environment(monkeypatch):
    from mlx_lm.models.cache import ArraysCache

    cache = [ArraysCache(size=2)]
    monkeypatch.setenv("MTPLX_OWNED_RECURRENT_STATE", "1")

    stats = configure_owned_recurrent_state_cache(cache)

    assert stats["enabled"] == 1
    assert stats["entries"] == 1
    assert isinstance(cache[0], OwnedRecurrentStateCache)


def test_owned_recurrent_state_stats_aggregates_entries():
    cache = [OwnedRecurrentStateCache(size=2)]
    cache[0].replace_state([mx.array([[1.0]]), mx.array([[2.0]])])

    stats = owned_recurrent_state_stats(cache)

    assert stats["enabled"] == 1
    assert stats["entries"] == 1
    assert stats["updates"] == 2
    assert stats["arrays"] == 2
    assert stats["allocations"] == 2


class AttentionDummyCache:
    def __init__(self):
        self.keys = mx.array([[[[1.0], [2.0]]]])
        self.values = mx.array([[[[3.0], [4.0]]]])

    def is_trimmable(self):
        return True


def test_detach_attention_cache_state_eval_only_accounts_kv_arrays():
    kv = AttentionDummyCache()

    stats = detach_attention_cache_state([kv], mode="eval_only")

    assert stats["entries"] == 1
    assert stats["arrays"] == 2
    assert stats["bytes"] == kv.keys.nbytes + kv.values.nbytes
    assert kv.keys.tolist() == [[[[1.0], [2.0]]]]


def test_detach_array_leaf_supports_metal_copy_leaf_mode():
    value = mx.array([1.0, 2.0, 3.0]) + mx.zeros((), dtype=mx.float32)

    detached = detach_array_leaf(value, mode="metal_copy_leaf")

    assert detached.tolist() == [1.0, 2.0, 3.0]


def test_detach_cache_state_combines_recurrent_and_attention_groups():
    recurrent = DummyCache()
    recurrent.state = [mx.array([1]), mx.array([2])]
    kv = AttentionDummyCache()

    stats = detach_cache_state(
        [recurrent, kv],
        components={"gdn", "attn"},
        mode="eval_only",
    )

    assert stats["entries"] == 2
    assert stats["arrays"] == 3


def test_tail_owned_kv_cache_matches_stock_kv_cache_updates():
    from mlx_lm.models.cache import KVCache

    stock = KVCache()
    owned = TailOwnedKVCache(mode="contiguous_eval")
    first_k = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)
    first_v = 10 + first_k
    next_k = 100 + mx.arange(2, dtype=mx.float32).reshape(1, 1, 2, 1)
    next_v = 200 + mx.arange(2, dtype=mx.float32).reshape(1, 1, 2, 1)

    stock_k, stock_v = stock.update_and_fetch(first_k, first_v)
    owned_k, owned_v = owned.update_and_fetch(first_k, first_v)
    stock_k, stock_v = stock.update_and_fetch(next_k, next_v)
    owned_k, owned_v = owned.update_and_fetch(next_k, next_v)
    mx.eval(stock_k, stock_v, owned_k, owned_v)

    assert owned.size() == stock.size() == 6
    assert owned_k.tolist() == stock_k.tolist()
    assert owned_v.tolist() == stock_v.tolist()
    assert owned.tail_owner_updates == 2
    assert owned.tail_owner_arrays == 4
    assert owned.tail_owner_bytes == (
        first_k.nbytes + first_v.nbytes + next_k.nbytes + next_v.nbytes
    )


def test_install_tail_owned_attention_kv_cache_replaces_stock_kv_only():
    from mlx_lm.models.cache import KVCache

    recurrent = DummyCache()
    kv = KVCache()
    cache = [recurrent, kv]

    stats = install_tail_owned_attention_kv_cache(cache, mode="contiguous_eval")

    assert stats["entries"] == 1
    assert stats["skipped"] == 1
    assert cache[0] is recurrent
    assert isinstance(cache[1], TailOwnedKVCache)


def test_configure_tail_owned_attention_kv_cache_uses_environment(monkeypatch):
    from mlx_lm.models.cache import KVCache

    cache = [KVCache()]
    monkeypatch.setenv("MTPLX_OWNED_ATTN_KV", "tail")
    monkeypatch.setenv("MTPLX_OWNED_ATTN_KV_MODE", "eval_only")

    stats = configure_tail_owned_attention_kv_cache(cache)

    assert stats["enabled"] == 1
    assert stats["entries"] == 1
    assert isinstance(cache[0], TailOwnedKVCache)
    assert cache[0].mode == "eval_only"


def test_tail_owned_attention_kv_stats_aggregates_entries():
    cache = [TailOwnedKVCache(mode="eval_only")]
    cache[0].update_and_fetch(
        mx.ones((1, 1, 1, 1)),
        2 * mx.ones((1, 1, 1, 1)),
    )

    stats = tail_owned_attention_kv_stats(cache)

    assert stats["enabled"] == 1
    assert stats["entries"] == 1
    assert stats["mode"] == "eval_only"
    assert stats["updates"] == 1
    assert stats["arrays"] == 2


def test_block_owned_kv_cache_matches_stock_across_block_boundary_and_trim():
    from mlx_lm.models.cache import KVCache

    stock = KVCache()
    block = BlockOwnedKVCache(mode="contiguous_eval", block_size=3)
    chunks = [
        (
            mx.arange(2, dtype=mx.float32).reshape(1, 1, 2, 1),
            10 + mx.arange(2, dtype=mx.float32).reshape(1, 1, 2, 1),
        ),
        (
            100 + mx.arange(3, dtype=mx.float32).reshape(1, 1, 3, 1),
            200 + mx.arange(3, dtype=mx.float32).reshape(1, 1, 3, 1),
        ),
    ]

    for keys, values in chunks:
        stock_k, stock_v = stock.update_and_fetch(keys, values)
        block_k, block_v = block.update_and_fetch(keys, values)
    mx.eval(stock_k, stock_v, block_k, block_v)

    assert block.size() == stock.size() == 5
    assert block_k.tolist() == stock_k.tolist()
    assert block_v.tolist() == stock_v.tolist()
    assert len(block.key_blocks) == 2

    stock.trim(2)
    block.trim(2)
    keys = 300 + mx.ones((1, 1, 1, 1))
    values = 400 + mx.ones((1, 1, 1, 1))
    stock_k, stock_v = stock.update_and_fetch(keys, values)
    block_k, block_v = block.update_and_fetch(keys, values)
    mx.eval(stock_k, stock_v, block_k, block_v)

    assert block.size() == stock.size() == 4
    assert block_k.tolist() == stock_k.tolist()
    assert block_v.tolist() == stock_v.tolist()


def test_install_block_owned_attention_kv_cache_replaces_stock_kv_only():
    from mlx_lm.models.cache import KVCache

    recurrent = DummyCache()
    kv = KVCache()
    cache = [recurrent, kv]

    stats = install_block_owned_attention_kv_cache(
        cache,
        mode="contiguous_eval",
        block_size=512,
    )

    assert stats["entries"] == 1
    assert stats["skipped"] == 1
    assert stats["block_size"] == 512
    assert cache[0] is recurrent
    assert isinstance(cache[1], BlockOwnedKVCache)
    assert cache[1].block_size == 512


def test_vllm_metal_paged_kv_cache_matches_stock_kv_cache_updates_and_trim():
    from mlx_lm.models.cache import KVCache

    stock = KVCache()
    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=4)
    chunks = [
        (
            mx.arange(3, dtype=mx.float32).reshape(1, 1, 3, 1),
            10 + mx.arange(3, dtype=mx.float32).reshape(1, 1, 3, 1),
        ),
        (
            100 + mx.arange(5, dtype=mx.float32).reshape(1, 1, 5, 1),
            200 + mx.arange(5, dtype=mx.float32).reshape(1, 1, 5, 1),
        ),
    ]

    for keys, values in chunks:
        stock_k, stock_v = stock.update_and_fetch(keys, values)
        paged_k, paged_v = paged.update_and_fetch(keys, values)
    mx.eval(stock_k, stock_v, paged_k, paged_v)

    assert paged.size() == stock.size() == 8
    assert paged_k.tolist() == stock_k.tolist()
    assert paged_v.tolist() == stock_v.tolist()
    assert paged.paged_stats()["updates"] == 2
    assert paged.paged_stats()["capacity"] == 16

    stock.trim(3)
    paged.trim(3)
    keys = 300 + mx.ones((1, 1, 2, 1))
    values = 400 + mx.ones((1, 1, 2, 1))
    stock_k, stock_v = stock.update_and_fetch(keys, values)
    paged_k, paged_v = paged.update_and_fetch(keys, values)
    mx.eval(stock_k, stock_v, paged_k, paged_v)

    assert paged.size() == stock.size() == 7
    assert paged_k.tolist() == stock_k.tolist()
    assert paged_v.tolist() == stock_v.tolist()


def test_install_vllm_metal_paged_attention_kv_cache_replaces_stock_kv_only(monkeypatch):
    from mlx_lm.models.cache import KVCache

    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", lambda: object())
    recurrent = DummyCache()
    kv = KVCache()
    cache = [recurrent, kv]

    stats = install_vllm_metal_paged_attention_kv_cache(
        cache,
        block_size=16,
        num_blocks=64,
    )

    assert stats["entries"] == 1
    assert stats["skipped"] == 1
    assert stats["block_size"] == 16
    assert stats["num_blocks"] == 64
    assert cache[0] is recurrent
    assert isinstance(cache[1], VllmMetalPagedKVCache)


def test_configure_tail_owned_attention_kv_cache_uses_vllm_metal_env(monkeypatch):
    from mlx_lm.models.cache import KVCache

    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", lambda: object())
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE", "16")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS", "32")
    cache = [KVCache()]

    stats = configure_tail_owned_attention_kv_cache(cache)

    assert stats["mode"] == "vllm_metal_paged"
    assert stats["external_ops_required"] == 1
    assert stats["entries"] == 1
    assert isinstance(cache[0], VllmMetalPagedKVCache)
    assert cache[0].block_size == 16
    assert cache[0].num_blocks == 32


def test_dynamic_paged_kv_sizes_capacity_from_request(monkeypatch):
    from mlx_lm.models.cache import KVCache

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV_TOKENS", str(32768 + 128 + 3))
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV_MARGIN", "128")
    cache = [KVCache()]

    stats = configure_tail_owned_attention_kv_cache(cache)

    expected_blocks = ((32768 + 128 + 3 + 128) + 16 - 1) // 16
    assert stats["num_blocks"] >= expected_blocks
    assert isinstance(cache[0], VllmMetalPagedKVCache)
    assert cache[0].capacity >= 32768 + 128 + 3 + 128


def test_paged_kv_grows_on_dynamic_overflow(monkeypatch):
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=1)
    keys = mx.zeros((1, 2, 6, 3), dtype=mx.float32)
    values = mx.zeros((1, 2, 6, 3), dtype=mx.float32)

    paged.update_without_fetch(keys, values)

    assert paged.capacity >= 6
    assert paged.paged_stats()["grow_events"] == 1


def test_paged_active_array_assertion_guards_dense_fallback(monkeypatch):
    monkeypatch.setenv("MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "4")
    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=4)
    keys = mx.zeros((1, 2, 4, 3), dtype=mx.float32)
    values = mx.zeros((1, 2, 4, 3), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    with pytest.raises(RuntimeError, match="materialize active K/V arrays"):
        _ = paged.state


def test_paged_attention_records_phase_aware_large_q_bailout(monkeypatch):
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "1")

    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=16)
    keys = mx.zeros((1, 2, 32, 8), dtype=mx.float32)
    values = mx.zeros((1, 2, 32, 8), dtype=mx.float32)
    queries = mx.zeros((1, 8, 4, 8), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    with attention_phase("prefill"):
        assert paged.paged_attention(queries, scale=8**-0.5, mask="causal") is None
        paged.record_dense_fallback()

    stats = paged.paged_stats()
    assert stats["prefill_dense_fallback_calls"] == 1
    assert stats["paged_attention_bailouts_by_phase_reason"] == {
        "prefill:q_len_gt_max": 1
    }


def test_mlx_vector_large_q_routes_to_partitioned_paged(monkeypatch):
    class FakeOps:
        def __init__(self):
            self.calls = 0

        def paged_attention_v2_online_partitioned(self, *args, **kwargs):
            self.calls += 1

    fake_ops = FakeOps()
    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", lambda: fake_ops)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE", "8")

    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=16)
    keys = mx.zeros((1, 2, 32, 8), dtype=mx.float32)
    values = mx.zeros((1, 2, 32, 8), dtype=mx.float32)
    queries = mx.zeros((1, 8, 4, 8), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    with attention_phase("prefill"):
        actual = paged.paged_attention(queries, scale=8**-0.5, mask="causal")

    assert actual is not None
    assert fake_ops.calls == 1
    stats = paged.paged_stats()
    assert stats["partitioned_paged_calls"] == 1
    assert stats["paged_attention_large_q_path"] == "partitioned_paged"
    assert stats["dense_fallback_calls"] == 0


def test_mlx_vector_mid_q_respects_gqa_threadgroup_limit(monkeypatch):
    import mtplx.cache_state as cache_state
    import mtplx.kernels.sdpa_2pass_paged as paged_kernel

    class FakeOps:
        def __init__(self):
            self.calls = 0

        def paged_attention_v2_online_partitioned(self, *args, **kwargs):
            self.calls += 1

    def fail_tail(**_kwargs):
        raise AssertionError("illegal GQA q_len must not reach sdpa_2pass_paged_tail")

    fake_ops = FakeOps()
    monkeypatch.setattr(cache_state, "_load_vllm_metal_ops", lambda: fake_ops)
    monkeypatch.setattr(paged_kernel, "sdpa_2pass_paged_tail", fail_tail)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "16")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE", "8")

    paged = VllmMetalPagedKVCache(block_size=16, num_blocks=128)
    keys = mx.zeros((1, 4, 2048, 8), dtype=mx.float32)
    values = mx.zeros((1, 4, 2048, 8), dtype=mx.float32)
    queries = mx.zeros((1, 24, 14, 8), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    actual = paged.paged_attention(queries, scale=8**-0.5, mask="causal")

    assert actual is not None
    assert fake_ops.calls == 1
    stats = paged.paged_stats()
    assert stats["partitioned_paged_calls"] == 1
    assert stats["paged_attention_large_q_path"] == "partitioned_paged"
    assert stats["dense_fallback_calls"] == 0


def test_packaged_paged_tail_declines_oversized_threadgroup():
    from mtplx.kernels.sdpa_2pass_paged import sdpa_2pass_paged_tail

    queries = mx.zeros((1, 24, 14, 8), dtype=mx.float32)
    key_cache = mx.zeros((128, 16, 4, 8), dtype=mx.float32)
    value_cache = mx.zeros((128, 16, 4, 8), dtype=mx.float32)

    actual = sdpa_2pass_paged_tail(
        queries=queries,
        key_cache=key_cache,
        value_cache=value_cache,
        offset=2048,
        block_size=16,
        scale=8**-0.5,
        mask="causal",
        max_q_len=16,
        sliding_window=-1,
    )

    assert actual is None


def test_large_q_split_fallback_stays_in_paged_storage(monkeypatch):
    from mlx_lm.models.base import scaled_dot_product_attention

    def missing_ops():
        raise RuntimeError("no external ops")

    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", missing_ops)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_LARGE_Q_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_LARGE_Q_KV_CHUNK_SIZE", "7")

    mx.random.seed(1357)
    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=16)
    queries = mx.random.normal((1, 8, 4, 8), dtype=mx.float32)
    keys = mx.random.normal((1, 2, 32, 8), dtype=mx.float32)
    values = mx.random.normal((1, 2, 32, 8), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=8**-0.5,
        mask="causal",
    )
    actual = paged.paged_attention(queries, scale=8**-0.5, mask="causal")
    mx.eval(expected, actual)

    assert actual is not None
    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) <= 1e-3
    stats = paged.paged_stats()
    assert stats["large_q_split_sdpa_fallback_calls"] == 1
    assert stats["active_array_calls"] == 0
    assert stats["dense_fallback_calls"] == 0


def test_large_q_split_fallback_assertion_fails_release_qa(monkeypatch):
    def missing_ops():
        raise RuntimeError("no external ops")

    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", missing_ops)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "1")
    monkeypatch.setenv("MTPLX_ASSERT_NO_LARGE_Q_SPLIT_FALLBACK", "1")

    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=16)
    queries = mx.zeros((1, 8, 4, 8), dtype=mx.float32)
    keys = mx.zeros((1, 2, 32, 8), dtype=mx.float32)
    values = mx.zeros((1, 2, 32, 8), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    with pytest.raises(RuntimeError, match="large-q split SDPA fallback"):
        paged.paged_attention(queries, scale=8**-0.5, mask="causal")


def test_long_context_dense_fallback_guard_accepts_new_override(monkeypatch):
    monkeypatch.setenv("MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "4")
    monkeypatch.setenv("MTPLX_ALLOW_LONG_CONTEXT_DENSE_FALLBACK", "1")
    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=4)
    keys = mx.zeros((1, 2, 4, 3), dtype=mx.float32)
    values = mx.zeros((1, 2, 4, 3), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    assert paged.state[0] is not None


def test_sustained_product_flag_does_not_forbid_dense_fallback(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "4")
    monkeypatch.delenv("MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS", raising=False)
    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=4)
    keys = mx.zeros((1, 2, 4, 3), dtype=mx.float32)
    values = mx.zeros((1, 2, 4, 3), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    assert paged.long_context_dense_fallback_forbidden() is False


def test_paged_active_array_assertion_still_forbids_dense_fallback(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "4")
    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=4)
    keys = mx.zeros((1, 2, 4, 3), dtype=mx.float32)
    values = mx.zeros((1, 2, 4, 3), dtype=mx.float32)
    paged.update_without_fetch(keys, values)

    assert paged.long_context_dense_fallback_forbidden() is True


def test_configure_vllm_metal_paged_cache_mlx_vector_is_packaged(monkeypatch):
    from mlx_lm.models.cache import KVCache

    def fail_if_external_ops_loads():
        raise AssertionError("mlx_vector_paged should not require vllm-metal checkout")

    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", fail_if_external_ops_loads)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    cache = [KVCache()]

    stats = configure_tail_owned_attention_kv_cache(cache)

    assert stats["mode"] == "vllm_metal_paged"
    assert stats["attention_impl"] == "mlx_vector_paged"
    assert stats["external_ops_required"] == 0
    assert stats["entries"] == 1
    assert isinstance(cache[0], VllmMetalPagedKVCache)


def test_configure_vllm_metal_paged_cache_can_enable_turboquant(monkeypatch):
    from mlx_lm.models.cache import KVCache

    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", lambda: object())
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_TURBOQUANT", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT", "q8_0")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT", "q3_0")
    cache = [KVCache()]

    stats = configure_tail_owned_attention_kv_cache(cache)

    assert stats["mode"] == "vllm_metal_paged_turboquant"
    assert stats["external_ops_required"] == 1
    assert stats["turboquant"] == 1
    assert stats["turboquant_k_quant"] == "q8_0"
    assert stats["turboquant_v_quant"] == "q3_0"
    assert isinstance(cache[0], VllmMetalPagedKVCache)
    assert cache[0].turboquant is True
    assert cache[0].turboquant_config.key_quant == "q8_0"
    assert cache[0].turboquant_config.value_quant == "q3_0"


def test_configure_vllm_metal_paged_cache_can_enable_plain_q8_kv_quant(monkeypatch):
    from mlx_lm.models.cache import KVCache

    def fail_if_external_ops_loads():
        raise AssertionError("plain q8 paged KV must not require TurboQuant ops")

    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", fail_if_external_ops_loads)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "q8")
    cache = [KVCache()]

    stats = configure_tail_owned_attention_kv_cache(cache)

    assert stats["mode"] == "vllm_metal_paged_kv_q8"
    assert stats["external_ops_required"] == 0
    assert stats["kv_quant"] == 1
    assert stats["kv_quant_mode"] == "q8"
    assert isinstance(cache[0], VllmMetalPagedKVCache)
    assert cache[0].kv_quant is True
    assert cache[0].kv_quant_config.normalized_mode == "q8"


def test_vllm_metal_paged_q8_kv_quant_roundtrips_active_state():
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    mx.random.seed(1357)
    keys = mx.random.normal((1, 2, 7, 16), dtype=mx.float16)
    values = mx.random.normal((1, 2, 7, 16), dtype=mx.float16)
    cache = VllmMetalPagedKVCache(
        block_size=4,
        num_blocks=4,
        kv_quant_config=PagedKVQuantConfig("q8"),
    )

    cache.update_without_fetch(keys, values)
    restored_keys, restored_values = cache.state
    mx.eval(restored_keys, restored_values)

    assert cache.key_cache.dtype == mx.int8
    assert cache.value_cache.dtype == mx.int8
    plain_capacity_bytes = 2 * cache.capacity * 2 * 16 * 2
    quant_capacity_bytes = (
        cache.key_cache.nbytes
        + cache.value_cache.nbytes
        + cache.key_scale_cache.nbytes
        + cache.value_scale_cache.nbytes
    )
    assert quant_capacity_bytes < plain_capacity_bytes
    assert restored_keys.shape == keys.shape
    assert restored_values.shape == values.shape
    key_diff = mx.max(mx.abs(restored_keys.astype(mx.float32) - keys.astype(mx.float32)))
    value_diff = mx.max(mx.abs(restored_values.astype(mx.float32) - values.astype(mx.float32)))
    mx.eval(key_diff, value_diff)
    assert float(key_diff.item()) <= 2e-2
    assert float(value_diff.item()) <= 2e-2
    stats = cache.paged_stats()
    assert stats["mode"] == "vllm_metal_paged_kv_q8"
    assert stats["kv_quant"] == 1
    assert stats["kv_quant_mode"] == "q8"


def test_vllm_metal_paged_q4_kv_quant_roundtrips_active_state():
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    mx.random.seed(2469)
    keys = mx.random.normal((1, 2, 7, 16), dtype=mx.float16)
    values = mx.random.normal((1, 2, 7, 16), dtype=mx.float16)
    cache = VllmMetalPagedKVCache(
        block_size=4,
        num_blocks=4,
        kv_quant_config=PagedKVQuantConfig("q4"),
    )

    cache.update_without_fetch(keys, values)
    restored_keys, restored_values = cache.state
    mx.eval(restored_keys, restored_values)

    assert cache.key_cache.dtype == mx.uint8
    assert cache.value_cache.dtype == mx.uint8
    assert cache.key_cache.shape[-1] == 8
    plain_capacity_bytes = 2 * cache.capacity * 2 * 16 * 2
    quant_capacity_bytes = (
        cache.key_cache.nbytes
        + cache.value_cache.nbytes
        + cache.key_scale_cache.nbytes
        + cache.value_scale_cache.nbytes
    )
    assert quant_capacity_bytes < plain_capacity_bytes
    key_diff = mx.max(mx.abs(restored_keys.astype(mx.float32) - keys.astype(mx.float32)))
    value_diff = mx.max(mx.abs(restored_values.astype(mx.float32) - values.astype(mx.float32)))
    mx.eval(key_diff, value_diff)
    assert float(key_diff.item()) <= 0.25
    assert float(value_diff.item()) <= 0.25


def test_vllm_metal_paged_q8_kv_quant_attention_matches_stock_with_tolerance(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    from mlx_lm.models.base import scaled_dot_product_attention

    def fail_if_external_ops_loads():
        raise AssertionError("plain q8 attention must dequant through in-tree MLX SDPA")

    monkeypatch.setattr("mtplx.cache_state._load_vllm_metal_ops", fail_if_external_ops_loads)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    mx.random.seed(97531)
    q_len = 4
    kv_len = 128
    dim = 64
    queries = 0.25 * mx.random.normal((1, 4, q_len, dim), dtype=mx.float16)
    keys = 0.25 * mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    values = 0.25 * mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    scale = dim**-0.5
    cache = VllmMetalPagedKVCache(
        block_size=16,
        num_blocks=16,
        kv_quant_config=PagedKVQuantConfig("q8"),
    )
    cache.update_without_fetch(keys, values)

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask="causal",
    )
    actual = cache.paged_attention(queries, scale=scale, mask="causal")
    assert actual is not None
    mx.eval(expected, actual)

    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) <= 3e-2
    stats = cache.paged_stats()
    assert stats["mode"] == "vllm_metal_paged_kv_q8"
    assert stats["kv_quant_attention_calls"] == 1
    assert stats["kv_quant_dequant_calls"] >= 1
    assert stats["kv_quant_dequant_tokens"] >= kv_len
    assert stats["kv_quant_dequant_time_s"] >= 0.0


def test_paged_gqa_sdpa_route_env_is_explicit_and_long_context_only(monkeypatch):
    assert (
        _paged_gqa_sdpa_route_from_env(
            q_len=4,
            offset=100_000,
            query_heads=48,
            kv_heads=8,
        )
        == ""
    )

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE", "auto")
    assert (
        _paged_gqa_sdpa_route_from_env(
            q_len=4,
            offset=100_000,
            query_heads=48,
            kv_heads=8,
        )
        == "async_per_head"
    )
    assert (
        _paged_gqa_sdpa_route_from_env(
            q_len=1,
            offset=100_000,
            query_heads=48,
            kv_heads=8,
        )
        == ""
    )
    assert (
        _paged_gqa_sdpa_route_from_env(
            q_len=4,
            offset=16_384,
            query_heads=48,
            kv_heads=8,
        )
        == ""
    )

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_CONTEXT", "0")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE", "per-head")
    assert (
        _paged_gqa_sdpa_route_from_env(
            q_len=5,
            offset=16_384,
            query_heads=48,
            kv_heads=8,
        )
        == "per_head"
    )


def test_paged_gqa_sdpa_route_miss_records_shape_reason(monkeypatch):
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE", "async-per-head")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_CONTEXT", "0")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_Q", "4")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MAX_Q", "5")

    decision = _paged_gqa_sdpa_route_decision_from_env(
        q_len=17,
        offset=100_000,
        query_heads=48,
        kv_heads=8,
    )
    assert decision.route == ""
    assert decision.reason == "q_len_gt_max"

    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=64)
    with attention_phase("decode_verify"):
        cache._record_gqa_route_miss(
            decision,
            offset=100_000,
            q_len=17,
            query_heads=48,
            kv_heads=8,
        )

    stats = cache.paged_stats()
    assert stats["gqa_sdpa_route_misses_by_phase_reason"] == {
        "decode_verify:q_len_gt_max": 1
    }
    assert stats["gqa_sdpa_route_misses_by_q_len"] == {
        "decode_verify:q17:q_len_gt_max": 1
    }
    assert stats["gqa_sdpa_last_route_miss"] == {
        "phase": "decode_verify",
        "reason": "q_len_gt_max",
        "requested_route": "async_per_head",
        "offset": 100_000,
        "q_len": 17,
        "query_heads": 48,
        "kv_heads": 8,
        "min_context": 0,
        "min_q": 4,
        "max_q": 5,
    }


def test_vllm_metal_paged_gqa_sdpa_route_matches_stock_attention(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    from mlx_lm.models.base import scaled_dot_product_attention

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE", "per_head")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_CONTEXT", "0")

    mx.random.seed(8642)
    q_len = 4
    kv_len = 512
    dim = 128
    queries = mx.random.normal((1, 8, q_len, dim), dtype=mx.float16)
    keys = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    values = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    scale = dim**-0.5
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=64)
    cache.update_without_fetch(keys, values)

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask="causal",
    )
    actual = cache.paged_attention(queries, scale=scale, mask="causal")
    assert actual is not None
    mx.eval(expected, actual)

    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) <= 2e-2
    stats = cache.paged_stats()
    assert stats["gqa_sdpa_calls"] == 1
    assert stats["gqa_sdpa_calls_by_route"] == {"per_head": 1}


def test_vllm_metal_paged_attention_matches_stock_attention_with_tolerance():
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    from mlx_lm.models.base import scaled_dot_product_attention

    mx.random.seed(1234)
    q_len = 4
    kv_len = 21
    dim = 128
    queries = mx.random.normal((1, 8, q_len, dim), dtype=mx.float16)
    keys = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    values = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    scale = dim**-0.5
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=4)
    cache.update_without_fetch(keys, values)

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask="causal",
    )
    actual = cache.paged_attention(queries, scale=scale)
    assert actual is not None
    mx.eval(expected, actual)

    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) <= 2e-2


def test_vllm_metal_partitioned_paged_attention_matches_stock_attention(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    from mlx_lm.models.base import scaled_dot_product_attention

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "0")

    mx.random.seed(4321)
    q_len = 4
    kv_len = 640
    dim = 128
    queries = mx.random.normal((1, 8, q_len, dim), dtype=mx.float16)
    keys = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    values = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    scale = dim**-0.5
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=64)
    cache.update_without_fetch(keys, values)

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask="causal",
    )
    actual = cache.paged_attention(queries, scale=scale)
    assert actual is not None
    mx.eval(expected, actual)

    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) <= 2e-2
    assert cache.paged_stats()["partitioned_attention_calls"] == 1


def test_vllm_metal_paged_attention_exact_gather_matches_stock_attention(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    from mlx_lm.models.base import scaled_dot_product_attention

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "fast_sdpa_gather")

    mx.random.seed(2468)
    q_len = 4
    kv_len = 77
    dim = 128
    queries = mx.random.normal((1, 8, q_len, dim), dtype=mx.float16)
    keys = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    values = mx.random.normal((1, 2, kv_len, dim), dtype=mx.float16)
    scale = dim**-0.5
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=8)
    cache.update_without_fetch(keys, values)

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask="causal",
    )
    actual = cache.paged_attention(queries, scale=scale, mask="causal")
    assert actual is not None
    mx.eval(expected, actual)

    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) == 0.0


def test_vllm_metal_paged_attention_mlx_vector_paged_matches_stock_attention(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    from mlx_lm.models.base import scaled_dot_product_attention

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")

    mx.random.seed(9753)
    q_len = 4
    kv_len = 2048
    dim = 128
    queries = mx.random.normal((1, 8, q_len, dim), dtype=mx.bfloat16)
    keys = mx.random.normal((1, 2, kv_len, dim), dtype=mx.bfloat16)
    values = mx.random.normal((1, 2, kv_len, dim), dtype=mx.bfloat16)
    scale = dim**-0.5
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=128)
    cache.update_without_fetch(keys, values)

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask="causal",
    )
    actual = cache.paged_attention(queries, scale=scale, mask="causal")
    assert actual is not None
    mx.eval(expected, actual)

    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) <= 3e-2


def test_vllm_metal_paged_packaged_impl_decline_does_not_load_external_ops(monkeypatch):
    import mtplx.cache_state as cache_state
    import mtplx.kernels.sdpa_2pass_paged as paged_kernel

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "1")
    monkeypatch.setattr(paged_kernel, "sdpa_2pass_paged_tail", lambda **_kwargs: None)

    def fail_external_ops():
        raise AssertionError("packaged paged attention must not load external ops")

    monkeypatch.setattr(cache_state, "_load_vllm_metal_ops", fail_external_ops)

    q_len = 4
    kv_len = 32
    dim = 16
    queries = mx.zeros((1, 8, q_len, dim), dtype=mx.float32)
    keys = mx.zeros((1, 2, kv_len, dim), dtype=mx.float32)
    values = mx.zeros((1, 2, kv_len, dim), dtype=mx.float32)
    cache = VllmMetalPagedKVCache(block_size=16, num_blocks=4)
    cache.update_without_fetch(keys, values)

    assert cache.paged_attention(queries, scale=dim**-0.5, mask="causal") is None


def test_tensor_offset_vllm_metal_paged_attention_matches_stock_attention(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    from mlx_lm.models.base import scaled_dot_product_attention

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "8")
    monkeypatch.setenv("MTPLX_GRAPHBANK_PAGED_STATIC_MAX_OFFSET", "32")

    mx.random.seed(8642)
    q_len = 4
    kv_len = 77
    dim = 128
    queries = mx.random.normal((1, 8, q_len, dim), dtype=mx.bfloat16)
    keys = mx.random.normal((1, 2, kv_len, dim), dtype=mx.bfloat16)
    values = mx.random.normal((1, 2, kv_len, dim), dtype=mx.bfloat16)
    scale = dim**-0.5
    paged = VllmMetalPagedKVCache(block_size=16, num_blocks=8)
    paged.update_without_fetch(keys, values)
    cache = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(paged)
    assert cache.paged_stats()["static_max_offset"] == 32

    expected = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask="causal",
    )
    actual = cache.paged_attention(queries, scale=scale, mask="causal")
    assert actual is not None
    mx.eval(expected, actual)

    diff = mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) <= 3e-2


def test_tensor_offset_vllm_metal_paged_cache_updates_offset_inside_compile():
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")

    paged = VllmMetalPagedKVCache(block_size=4, num_blocks=4)
    paged.update_without_fetch(
        mx.ones((1, 1, 2, 1), dtype=mx.float32),
        2 * mx.ones((1, 1, 2, 1), dtype=mx.float32),
    )
    cache = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(paged)

    def update(keys, values):
        cache.update_without_fetch(keys, values)
        return cache.compile_state

    compiled = mx.compile(update, inputs=cache.compile_state, outputs=cache.compile_state)
    compiled(
        3 * mx.ones((1, 1, 2, 1), dtype=mx.float32),
        4 * mx.ones((1, 1, 2, 1), dtype=mx.float32),
    )
    mx.eval(cache.compile_state)

    assert cache.size() == 4
    keys, values = cache.state
    mx.eval(keys, values)
    assert keys[0, 0, :4, 0].tolist() == [1.0, 1.0, 3.0, 3.0]
    assert values[0, 0, :4, 0].tolist() == [2.0, 2.0, 4.0, 4.0]


def _paged_cache_with_data(*, block_size: int = 4, num_blocks: int = 4):
    paged = VllmMetalPagedKVCache(block_size=block_size, num_blocks=num_blocks)
    keys = mx.arange(6, dtype=mx.float32).reshape(1, 1, 6, 1)
    values = 10 + mx.arange(6, dtype=mx.float32).reshape(1, 1, 6, 1)
    paged.update_without_fetch(keys, values)
    return paged


def test_promote_preserve_paged_param_keeps_paged_storage(monkeypatch):
    from mtplx.graphbank import promote_kv_cache_offsets

    monkeypatch.delenv("MTPLX_GRAPHBANK_PRESERVE_PAGED_KV", raising=False)
    cache = [_paged_cache_with_data()]

    promoted, failures = promote_kv_cache_offsets(
        cache, reserve_tokens=4, preserve_paged=True
    )

    assert promoted == 1
    assert failures == {}
    assert isinstance(cache[0], TensorOffsetVllmMetalPagedKVCache)
    assert cache[0].size() == 6
    # Physical pages carried over by reference — no densify, no copy.
    assert cache[0].cache[0].shape == (4, 4, 1, 1)


def test_promote_default_still_follows_env_for_paged_entries(monkeypatch):
    from mtplx.graphbank import TensorOffsetKVCache, promote_kv_cache_offsets

    monkeypatch.delenv("MTPLX_GRAPHBANK_PRESERVE_PAGED_KV", raising=False)
    cache = [_paged_cache_with_data()]
    promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=4)
    # Historical trap: without preserve_paged the paged entry falls through the
    # dense path and its `.keys` property densifies the paged storage.
    assert promoted == 1
    assert failures == {}
    assert isinstance(cache[0], TensorOffsetKVCache)

    monkeypatch.setenv("MTPLX_GRAPHBANK_PRESERVE_PAGED_KV", "1")
    cache = [_paged_cache_with_data()]
    promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=4)
    assert promoted == 1
    assert failures == {}
    assert isinstance(cache[0], TensorOffsetVllmMetalPagedKVCache)


def test_promote_preserve_paged_refuses_quantized_paged_entries(monkeypatch):
    from mtplx.graphbank import promote_kv_cache_offsets

    monkeypatch.delenv("MTPLX_GRAPHBANK_PRESERVE_PAGED_KV", raising=False)
    quantized = VllmMetalPagedKVCache(
        block_size=4,
        num_blocks=4,
        kv_quant_config=PagedKVQuantConfig("q8"),
    )
    quantized.update_without_fetch(
        mx.random.normal((1, 2, 5, 16), dtype=mx.float16),
        mx.random.normal((1, 2, 5, 16), dtype=mx.float16),
    )
    cache = [quantized]

    promoted, failures = promote_kv_cache_offsets(
        cache, reserve_tokens=4, preserve_paged=True
    )

    assert promoted == 0
    assert failures == {"quantized_paged_kv_cache": 1}
    assert cache[0] is quantized


def test_tensor_offset_paged_static_max_offset_attr_beats_env(monkeypatch):
    monkeypatch.setenv("MTPLX_GRAPHBANK_PAGED_STATIC_MAX_OFFSET", "32")
    adapter = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(
        _paged_cache_with_data()
    )

    assert adapter._static_attention_max_offset() == 32
    assert adapter.paged_stats()["static_max_offset"] == 32

    adapter.static_max_offset = 64
    assert adapter._static_attention_max_offset() == 64
    assert adapter.paged_stats()["static_max_offset"] == 64

    monkeypatch.delenv("MTPLX_GRAPHBANK_PAGED_STATIC_MAX_OFFSET", raising=False)
    assert adapter._static_attention_max_offset() == 64
    adapter.static_max_offset = None
    assert adapter._static_attention_max_offset() is None


def test_tensor_offset_paged_demote_round_trips_offset_and_buffers():
    paged = _paged_cache_with_data()
    adapter = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(paged)
    adapter.update_without_fetch(
        100 + mx.arange(2, dtype=mx.float32).reshape(1, 1, 2, 1),
        200 + mx.arange(2, dtype=mx.float32).reshape(1, 1, 2, 1),
    )

    restored = adapter.to_paged_cache()

    assert isinstance(restored, VllmMetalPagedKVCache)
    assert type(restored) is VllmMetalPagedKVCache
    assert isinstance(restored.offset, int)
    assert restored.offset == 8
    # Original buffers by reference — bit-exact, no copy.
    assert restored.key_cache is adapter.cache[0]
    assert restored.value_cache is adapter.cache[1]
    assert restored.block_size == 4
    assert restored.num_blocks == 4

    keys, values = restored.state
    mx.eval(keys, values)
    assert keys[0, 0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 101.0]
    assert values[0, 0, :, 0].tolist() == [
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 200.0, 201.0,
    ]

    # Shape metadata restored: the next write appends without re-allocating.
    restored.update_without_fetch(
        mx.array([[[[300.0]]]]), mx.array([[[[400.0]]]])
    )
    assert restored.size() == 9
    keys, _ = restored.state
    mx.eval(keys)
    assert keys[0, 0, 8, 0].item() == 300.0

    # demote() is the bank-facing alias.
    assert isinstance(adapter.demote(), VllmMetalPagedKVCache)


def test_tensor_offset_paged_meta_state_round_trip():
    adapter = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(
        _paged_cache_with_data()
    )

    assert adapter.meta_state == ("4", "4", "6")

    adapter.meta_state = ("4", "8", "3")
    assert adapter.num_blocks == 8
    assert adapter.size() == 3
    assert isinstance(adapter.cache[2], mx.array)


def test_tensor_offset_kv_cache_demote_restores_stock_container():
    from mlx_lm.models.cache import KVCache

    from mtplx.graphbank import TensorOffsetKVCache, promote_kv_cache_offsets

    stock = KVCache()
    stock.update_and_fetch(
        mx.arange(3, dtype=mx.float32).reshape(1, 1, 3, 1),
        10 + mx.arange(3, dtype=mx.float32).reshape(1, 1, 3, 1),
    )
    cache = [stock]
    promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=4)
    assert promoted == 1 and failures == {}
    adapter = cache[0]
    assert isinstance(adapter, TensorOffsetKVCache)
    adapter.update_and_fetch(
        mx.array([[[[7.0], [8.0]]]]), mx.array([[[[9.0], [11.0]]]])
    )

    restored = adapter.demote()

    assert type(restored) is KVCache
    assert isinstance(restored.offset, int)
    assert restored.offset == 5
    assert restored.keys is adapter.cache[0]
    assert restored.values is adapter.cache[1]
    keys, values = restored.state
    mx.eval(keys, values)
    assert keys[0, 0, :, 0].tolist() == [0.0, 1.0, 2.0, 7.0, 8.0]
    assert values[0, 0, :, 0].tolist() == [10.0, 11.0, 12.0, 9.0, 11.0]

    # Stock trim/update behavior intact after demotion.
    restored.trim(2)
    assert restored.offset == 3
    restored.update_and_fetch(mx.array([[[[42.0]]]]), mx.array([[[[43.0]]]]))
    assert restored.offset == 4
