"""KV-quant memory honesty and per-request numerics routing (F29).

The kv_quant dequant mirror must never invert the feature's memory promise:
it is offset-sized (not capacity-sized), q8-only (q4 never builds a bf16
mirror — its kernel route reads the head-major QUANTIZED bank instead, and
its dequant fallbacks materialize transiently), released when a request
latches the q8-kernel route, and it survives quantized-store growth without
a full rebuild. Numerics are
routed once per request: a request must not hop between kernel math and
dequant math because its offset crossed the two-pass threshold
mid-generation (temp-0 exactness). trim() deliberately keeps the latched
route — speculative-verify rejections retract rows mid-request, and
re-latching there would reintroduce the switch at the threshold boundary.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.attention_context import attention_phase
from mtplx.cache_state import (
    VllmMetalPagedKVCache,
    install_vllm_metal_paged_attention_kv_cache,
)
from mtplx.kv_quant import PagedKVQuantConfig

DIM = 128
KV_HEADS = 2
Q_HEADS = 8  # gqa 4 -> safe kernel q_len 8


def _skip_without_metal() -> None:
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")


def _rows(count: int, seed: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    keys = 0.5 * mx.random.normal((1, KV_HEADS, count, DIM), dtype=mx.float16)
    values = 0.5 * mx.random.normal((1, KV_HEADS, count, DIM), dtype=mx.float16)
    return keys, values


def _queries(q_len: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return 0.3 * mx.random.normal((1, Q_HEADS, q_len, DIM), dtype=mx.float16)


def _build_cache(
    mode: str,
    *,
    block_size: int = 4,
    num_blocks: int = 64,
) -> VllmMetalPagedKVCache:
    return VllmMetalPagedKVCache(
        block_size=block_size,
        num_blocks=num_blocks,
        kv_quant_config=PagedKVQuantConfig(mode),
    )


def test_q8_mirror_is_offset_sized_not_capacity_sized(monkeypatch):
    """A 256-token-capacity cache at offset 24 must not mirror 256 rows:
    the capacity-sized mirror was the 1.5-2.0x kv-quant memory inversion."""

    _skip_without_metal()
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "0")

    cache = _build_cache("q8", block_size=4, num_blocks=64)
    keys, values = _rows(24, seed=101)
    with attention_phase("prefill"):
        cache.update_without_fetch(keys, values)
    got_k, got_v = cache._active_arrays()
    mx.eval(got_k, got_v)

    capacity_rows = cache.capacity
    assert capacity_rows == 256
    memo = cache._dequant_memo
    assert memo is not None
    mirror_rows = int(memo["mirror_k"].shape[0])
    assert int(memo["mirror_v"].shape[0]) == mirror_rows
    assert mirror_rows >= int(cache.offset)
    assert mirror_rows <= 2 * int(cache.offset)
    assert mirror_rows < capacity_rows

    # Decode appends grow the mirror geometrically, still tracking offset.
    for step in range(3):
        tail_k, tail_v = _rows(1, seed=200 + step)
        with attention_phase("ar_decode"):
            cache.update_without_fetch(tail_k, tail_v)
        got_k, got_v = cache._active_arrays()
        mx.eval(got_k, got_v)
        mirror_rows = int(cache._dequant_memo["mirror_k"].shape[0])
        assert mirror_rows >= int(cache.offset)
        assert mirror_rows <= 2 * int(cache.offset)
        assert mirror_rows < capacity_rows


def test_q4_allocates_no_mirror(monkeypatch):
    """q4 never builds the bf16 mirror (its kernel route reads the quantized
    bank instead): a persistent bf16 mirror would just stack bf16 on top of
    the quantized store forever, so it must not exist."""

    _skip_without_metal()
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "1")

    cache = _build_cache("q4", block_size=4, num_blocks=64)
    keys, values = _rows(40, seed=303)
    with attention_phase("prefill"):
        cache.update_without_fetch(keys, values)

    first_k, first_v = cache._active_arrays()
    mx.eval(first_k, first_v)
    assert cache._dequant_memo is None
    second_k, second_v = cache._active_arrays()
    mx.eval(second_k, second_v)
    assert cache._dequant_memo is None

    # Same quantized bytes -> same dequant math on every materialization.
    assert float(mx.abs(first_k - second_k).max().item()) == 0.0
    assert float(mx.abs(first_v - second_v).max().item()) == 0.0
    assert first_k.shape == keys.shape
    key_diff = mx.max(mx.abs(first_k.astype(mx.float32) - keys.astype(mx.float32)))
    mx.eval(key_diff)
    assert float(key_diff.item()) <= 0.25


def test_q4_decode_avoids_full_bf16_materialization(monkeypatch):
    """q4 decode attention must serve through the chunked online-softmax
    path: no mirror, no full-width bf16 K/V per step, answers matching
    SDPA over the dequantized state."""

    _skip_without_metal()
    from mlx_lm.models.base import scaled_dot_product_attention

    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "1")

    cache = _build_cache("q4", block_size=16, num_blocks=16)
    keys, values = _rows(200, seed=404)
    with attention_phase("prefill"):
        cache.update_without_fetch(keys, values)
    queries = _queries(1, seed=405)

    with attention_phase("ar_decode"):
        out = cache.paged_attention(queries, scale=DIM**-0.5, mask="causal")

    assert out is not None
    mx.eval(out)
    assert cache.kv_quant_kernel_calls == 0
    assert cache.large_q_split_sdpa_fallback_calls == 1
    assert cache._dequant_memo is None

    reference = _build_cache("q4", block_size=16, num_blocks=16)
    reference.update_without_fetch(keys, values)
    ref_k, ref_v = reference.state
    expected = scaled_dot_product_attention(
        queries,
        ref_k,
        ref_v,
        cache=None,
        scale=DIM**-0.5,
        mask="causal",
    )
    diff = mx.max(mx.abs(out.astype(mx.float32) - expected.astype(mx.float32)))
    mx.eval(diff)
    assert float(diff.item()) <= 3e-2


def test_q8_mirror_released_after_kernel_engagement(monkeypatch):
    """Once a request latches the kernel route, the prefill-era mirror is
    dead weight and must be freed — released once per request: a later
    shape-driven dequant call may rebuild it, and the next kernel call must
    not re-release it (that would thrash full-prefix rebuild stalls)."""

    _skip_without_metal()
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "64")

    cache = _build_cache("q8", block_size=16, num_blocks=16)
    keys, values = _rows(96, seed=505)
    with attention_phase("prefill"):
        cache.update_without_fetch(keys, values)
    warm_k, warm_v = cache._active_arrays()
    mx.eval(warm_k, warm_v)
    assert cache._dequant_memo is not None

    with attention_phase("ar_decode"):
        out = cache.paged_attention(_queries(1, seed=506), scale=DIM**-0.5, mask="causal")
    assert out is not None
    mx.eval(out)
    assert cache.kv_quant_kernel_calls == 1
    assert cache._dequant_memo is None

    # A verify burst past the kernel's q budget falls back and rebuilds the
    # mirror (offset-sized); the next kernel call must NOT re-release it.
    burst_k, burst_v = _rows(12, seed=507)
    with attention_phase("decode_verify"):
        cache.update_without_fetch(burst_k, burst_v)
        burst_out = cache.paged_attention(
            _queries(12, seed=508), scale=DIM**-0.5, mask="causal"
        )
    assert burst_out is not None
    mx.eval(burst_out)
    assert cache._dequant_memo is not None
    with attention_phase("ar_decode"):
        tail_k, tail_v = _rows(1, seed=509)
        cache.update_without_fetch(tail_k, tail_v)
        tail_out = cache.paged_attention(_queries(1, seed=510), scale=DIM**-0.5, mask="causal")
    assert tail_out is not None
    mx.eval(tail_out)
    assert cache.kv_quant_kernel_calls == 2
    assert cache._dequant_memo is not None


def test_kv_quant_route_latches_once_per_request(monkeypatch):
    """A request that starts attending below the two-pass threshold keeps
    dequant math for its whole generation — crossing the threshold
    mid-generation must not switch numerics (temp-0 exactness). The next
    prompt write re-latches from its own starting offset."""

    _skip_without_metal()
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "64")

    cache = _build_cache("q8", block_size=4, num_blocks=64)
    keys, values = _rows(40, seed=606)
    with attention_phase("prefill"):
        cache.update_without_fetch(keys, values)
    assert cache.paged_stats()["kv_quant_route"] == ""

    with attention_phase("ar_decode"):
        out = cache.paged_attention(_queries(1, seed=607), scale=DIM**-0.5, mask="causal")
    assert out is not None
    stats = cache.paged_stats()
    assert stats["kv_quant_route"] == "dequant"
    assert stats["kv_quant_route_offset"] == 40
    assert cache.kv_quant_kernel_calls == 0

    # Generate across the threshold: still ONE numerics path.
    for step in range(30):
        tail_k, tail_v = _rows(1, seed=700 + step)
        with attention_phase("ar_decode"):
            cache.update_without_fetch(tail_k, tail_v)
    assert int(cache.offset) == 70
    with attention_phase("ar_decode"):
        out = cache.paged_attention(_queries(1, seed=608), scale=DIM**-0.5, mask="causal")
    assert out is not None
    assert cache.kv_quant_kernel_calls == 0
    assert cache.paged_stats()["kv_quant_route"] == "dequant"

    # A new prompt write starts a new request: re-latch from its offset.
    more_k, more_v = _rows(10, seed=609)
    with attention_phase("prefill"):
        cache.update_without_fetch(more_k, more_v)
    assert cache.paged_stats()["kv_quant_route"] == ""
    with attention_phase("ar_decode"):
        out = cache.paged_attention(_queries(1, seed=610), scale=DIM**-0.5, mask="causal")
    assert out is not None
    stats = cache.paged_stats()
    assert stats["kv_quant_route"] == "kernel"
    assert stats["kv_quant_route_offset"] == 80
    assert cache.kv_quant_kernel_calls == 1


def test_kv_quant_route_survives_verify_reject_trim(monkeypatch):
    """trim() retracts rejected speculative rows MID-request: the latched
    route must survive it. Re-latching on trim would switch numerics when a
    rejection lands the offset back across the threshold — the exact bug
    per-request routing exists to kill. The next prompt write is the real
    request boundary and re-latches from its own starting offset."""

    _skip_without_metal()
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "64")

    cache = _build_cache("q8", block_size=4, num_blocks=64)
    keys, values = _rows(66, seed=616)
    with attention_phase("prefill"):
        cache.update_without_fetch(keys, values)
    with attention_phase("ar_decode"):
        out = cache.paged_attention(_queries(1, seed=617), scale=DIM**-0.5, mask="causal")
    assert out is not None
    assert cache.paged_stats()["kv_quant_route"] == "kernel"
    assert cache.kv_quant_kernel_calls == 1

    # Speculative rejection retracts below the threshold: same request,
    # same math.
    cache.trim(10)
    assert int(cache.offset) == 56
    assert cache.paged_stats()["kv_quant_route"] == "kernel"
    with attention_phase("ar_decode"):
        out = cache.paged_attention(_queries(1, seed=618), scale=DIM**-0.5, mask="causal")
    assert out is not None
    assert cache.kv_quant_kernel_calls == 2
    assert cache.paged_stats()["kv_quant_route"] == "kernel"

    # The next prompt write is a request boundary: re-latch from the new
    # starting offset (56 + 4 = 60, below the threshold -> dequant).
    more_k, more_v = _rows(4, seed=619)
    with attention_phase("prefill"):
        cache.update_without_fetch(more_k, more_v)
    assert cache.paged_stats()["kv_quant_route"] == ""
    with attention_phase("ar_decode"):
        out = cache.paged_attention(_queries(1, seed=620), scale=DIM**-0.5, mask="causal")
    assert out is not None
    assert cache.paged_stats()["kv_quant_route"] == "dequant"
    assert cache.kv_quant_kernel_calls == 2


def test_kv_quant_route_is_structurally_dequant_when_kernel_cannot_engage(monkeypatch):
    """Kill-switched q4, geometry the packed-quant kernel refuses, and
    sliding-window layers cannot use a kv_quant kernel: their route latches
    dequant regardless of offset."""

    _skip_without_metal()
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "64")

    # q4 with the dedicated kill-switch off: structurally dequant.
    monkeypatch.setenv("MTPLX_KV_QUANT_Q4_KERNEL", "0")
    q4_cache = _build_cache("q4", block_size=16, num_blocks=16)
    keys, values = _rows(200, seed=808)
    with attention_phase("prefill"):
        q4_cache.update_without_fetch(keys, values)
    with attention_phase("ar_decode"):
        out = q4_cache.paged_attention(_queries(1, seed=809), scale=DIM**-0.5, mask="causal")
    assert out is not None
    assert q4_cache.paged_stats()["kv_quant_route"] == "dequant"
    assert q4_cache.kv_quant_kernel_calls == 0
    monkeypatch.delenv("MTPLX_KV_QUANT_Q4_KERNEL")

    # q4 head_dim outside the packed-quant kernel's {64, 128, 256} set:
    # structurally dequant even with every switch on.
    narrow = _build_cache("q4", block_size=16, num_blocks=16)
    mx.random.seed(818)
    narrow_keys = 0.5 * mx.random.normal((1, KV_HEADS, 200, 96), dtype=mx.float16)
    narrow_values = 0.5 * mx.random.normal((1, KV_HEADS, 200, 96), dtype=mx.float16)
    with attention_phase("prefill"):
        narrow.update_without_fetch(narrow_keys, narrow_values)
    mx.random.seed(819)
    narrow_q = 0.3 * mx.random.normal((1, Q_HEADS, 1, 96), dtype=mx.float16)
    with attention_phase("ar_decode"):
        out = narrow.paged_attention(narrow_q, scale=96**-0.5, mask="causal")
    assert out is not None
    assert narrow.paged_stats()["kv_quant_route"] == "dequant"
    assert narrow.kv_quant_kernel_calls == 0

    windowed = _build_cache("q8", block_size=16, num_blocks=16)
    with attention_phase("prefill"):
        windowed.update_without_fetch(keys, values)
    with attention_phase("ar_decode"):
        out = windowed.paged_attention(
            _queries(1, seed=810),
            scale=DIM**-0.5,
            mask="causal",
            sliding_window=64,
        )
    assert out is not None
    assert windowed.paged_stats()["kv_quant_route"] == "dequant"
    assert windowed.kv_quant_kernel_calls == 0


def test_q8_grow_preserves_mirror_contents_byte_exactly(monkeypatch):
    """Growing the quantized store must extend the mirror's world without a
    full rebuild: flat row indices are append-stable, so the valid prefix
    stays byte-exact and only the new tail is dequantized."""

    _skip_without_metal()
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "0")
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    monkeypatch.delenv("MTPLX_CONTEXT_WINDOW_TOKENS", raising=False)

    base_k, base_v = _rows(12, seed=909)
    tail_k, tail_v = _rows(8, seed=910)

    cache = _build_cache("q8", block_size=4, num_blocks=4)
    with attention_phase("prefill"):
        cache.update_without_fetch(base_k, base_v)
    warm_k, warm_v = cache._active_arrays()
    mx.eval(warm_k, warm_v)
    assert cache.kv_quant_dequant_memo_rebuilds == 1
    assert cache.kv_quant_dequant_tokens == 12

    with attention_phase("decode_verify"):
        cache.update_without_fetch(tail_k, tail_v)
    assert cache.grow_events >= 1
    got_k, got_v = cache._active_arrays()
    mx.eval(got_k, got_v)
    assert cache.kv_quant_dequant_memo_rebuilds == 1
    assert cache.kv_quant_dequant_tokens == 20

    fresh = _build_cache("q8", block_size=4, num_blocks=4)
    fresh.update_without_fetch(base_k, base_v)
    fresh.update_without_fetch(tail_k, tail_v)
    want_k, want_v = fresh._active_arrays()
    mx.eval(want_k, want_v)
    assert got_k.shape == want_k.shape
    assert float(mx.abs(got_k - want_k).max().item()) == 0.0
    assert float(mx.abs(got_v - want_v).max().item()) == 0.0


def test_kv_quant_kernel_mask_gate_is_strict(monkeypatch):
    """Array masks must decline the kernel (and the chunked fallback)
    through the strict isinstance idiom, not through MLX's operator
    fallback for `array != str` — and the request must still be served."""

    _skip_without_metal()
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "64")

    cache = _build_cache("q8", block_size=16, num_blocks=16)
    keys, values = _rows(96, seed=111)
    with attention_phase("prefill"):
        cache.update_without_fetch(keys, values)
    queries = _queries(1, seed=112)
    array_mask = mx.ones((1, 1, 1, 96), dtype=mx.bool_)

    direct = cache._kv_quant_2pass_attention(
        queries,
        scale=DIM**-0.5,
        mask=array_mask,
        sliding_window=-1,
        q_len=1,
    )
    assert direct is None

    split = cache._large_q_split_sdpa_fallback(
        queries,
        scale=DIM**-0.5,
        sliding_window=-1,
        mask=array_mask,
    )
    assert split is None

    with attention_phase("ar_decode"):
        out = cache.paged_attention(queries, scale=DIM**-0.5, mask=array_mask)
    assert out is not None
    mx.eval(out)
    assert cache.kv_quant_kernel_calls == 0


def test_nbytes_counts_live_mirror_and_only_live_mirror(monkeypatch):
    """The bytes stat must not hide the mirror: memory arithmetic that
    omits a live bf16 working copy is how the inversion went unnoticed."""

    _skip_without_metal()
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "0")

    cache = _build_cache("q8", block_size=4, num_blocks=64)
    keys, values = _rows(32, seed=222)
    with attention_phase("prefill"):
        cache.update_without_fetch(keys, values)
    quant_bytes = (
        int(cache.key_cache.nbytes)
        + int(cache.value_cache.nbytes)
        + int(cache.key_scale_cache.nbytes)
        + int(cache.value_scale_cache.nbytes)
    )
    assert cache.nbytes == quant_bytes

    warm_k, warm_v = cache._active_arrays()
    mx.eval(warm_k, warm_v)
    memo = cache._dequant_memo
    assert memo is not None
    mirror_bytes = int(memo["mirror_k"].nbytes) + int(memo["mirror_v"].nbytes)
    assert cache.nbytes == quant_bytes + mirror_bytes

    cache._invalidate_dequant_memo()
    assert cache.nbytes == quant_bytes


def test_capacity_tracks_pages_not_claim_across_stomps(monkeypatch):
    """#310: re-configs and snapshot restores stomp num_blocks on live
    buffers without reallocating. Capacity must stay a fact about the
    allocated pages — a lying claim skipped the growth guard, fancy-index
    scatter silently dropped out-of-range rows while the offset advanced,
    and a later real grow crashed _dequant_active_arrays broadcasting the
    short mirror into the full-offset one. CPU-only shape test, no Metal."""

    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    monkeypatch.delenv("MTPLX_CONTEXT_WINDOW_TOKENS", raising=False)

    cache = _build_cache("q8", block_size=16, num_blocks=8)
    keys, values = _rows(100, seed=310)
    cache.update_without_fetch(keys, values)
    warm_k, warm_v = cache._active_arrays()
    mx.eval(warm_k, warm_v)
    assert cache.capacity == 128

    # Writer stand-in (install re-config / meta_state restore): raise the
    # claim without reallocating a single page.
    cache.num_blocks = 64
    assert cache.capacity == 128

    tail_k, tail_v = _rows(51, seed=311)
    cache.update_without_fetch(tail_k, tail_v)  # crosses the physical boundary
    assert int(cache.offset) == 151
    assert cache.grow_events == 1
    got_k, got_v = cache._active_arrays()
    mx.eval(got_k, got_v)
    assert int(got_k.shape[2]) == int(cache.offset)  # no silent truncation
    assert int(got_v.shape[2]) == int(cache.offset)
    assert cache.capacity == int(cache.key_cache.shape[0]) * int(
        cache.key_cache.shape[1]
    )

    # Lower the claim below the grown pages: the next write must neither
    # re-grow from a stale base nor break the dequant mirror.
    cache.num_blocks = 8
    one_k, one_v = _rows(1, seed=312)
    cache.update_without_fetch(one_k, one_v)
    got_k, got_v = cache._active_arrays()
    mx.eval(got_k, got_v)
    assert int(got_k.shape[2]) == int(cache.offset) == 152


def test_q8_mirror_invariant_survives_reconfig(monkeypatch):
    """memo["tokens"] <= mirror rows and mirror rows >= offset must hold
    across an install-time re-config of a live cache (#310): the re-config
    requests room via an explicit grow, it never redefines the pages under
    the mirror."""

    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    monkeypatch.delenv("MTPLX_CONTEXT_WINDOW_TOKENS", raising=False)

    cache = _build_cache("q8", block_size=16, num_blocks=8)
    keys, values = _rows(100, seed=313)
    cache.update_without_fetch(keys, values)
    warm_k, warm_v = cache._active_arrays()
    mx.eval(warm_k, warm_v)
    memo = cache._dequant_memo
    assert memo is not None
    assert int(memo["tokens"]) <= int(memo["mirror_k"].shape[0])
    assert int(memo["mirror_k"].shape[0]) >= int(cache.offset)

    stats = install_vllm_metal_paged_attention_kv_cache(
        [cache],
        block_size=16,
        num_blocks=64,
        kv_quant_config=PagedKVQuantConfig("q8"),
    )
    assert stats["entries"] == 1
    assert cache.capacity >= 16 * 64  # room request honored by growing
    assert cache.capacity == int(cache.key_cache.shape[0]) * int(
        cache.key_cache.shape[1]
    )

    tail_k, tail_v = _rows(51, seed=314)
    cache.update_without_fetch(tail_k, tail_v)
    got_k, got_v = cache._active_arrays()
    mx.eval(got_k, got_v)
    memo = cache._dequant_memo
    assert memo is not None
    mirror_rows = int(memo["mirror_k"].shape[0])
    assert int(memo["tokens"]) <= mirror_rows
    assert mirror_rows >= int(cache.offset)
    assert int(got_k.shape[2]) == int(cache.offset) == 151


def test_q8_boundary_smoke_pins_16384_to_19295_crossing(monkeypatch):
    """The reporter's literal crossing (#310): 1024 16-row blocks (16384
    rows), a stomped claim, then writes landing the offset at 19295.
    Pre-fix this crashed _dequant_active_arrays broadcasting the 16384-row
    mirror into the 19295-row one; now the pages grow at the write and the
    mirror follows. head_dim=8 keeps the int8 store ~0.5MB."""

    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    monkeypatch.delenv("MTPLX_CONTEXT_WINDOW_TOKENS", raising=False)

    head_dim = 8

    def rows(count: int, seed: int) -> tuple[mx.array, mx.array]:
        mx.random.seed(seed)
        keys = 0.5 * mx.random.normal(
            (1, KV_HEADS, count, head_dim), dtype=mx.float16
        )
        values = 0.5 * mx.random.normal(
            (1, KV_HEADS, count, head_dim), dtype=mx.float16
        )
        return keys, values

    cache = _build_cache("q8", block_size=16, num_blocks=1024)
    keys, values = rows(16384, seed=315)
    cache.update_without_fetch(keys, values)
    warm_k, warm_v = cache._active_arrays()
    mx.eval(warm_k, warm_v)
    assert cache.capacity == 16384

    cache.num_blocks = 4096  # stomped claim: 65536 rows that do not exist
    assert cache.capacity == 16384

    tail_k, tail_v = rows(19295 - 16384, seed=316)
    cache.update_without_fetch(tail_k, tail_v)
    assert int(cache.offset) == 19295
    assert cache.grow_events == 1
    got_k, got_v = cache._active_arrays()
    mx.eval(got_k, got_v)
    assert int(got_k.shape[2]) == 19295
    assert int(got_v.shape[2]) == 19295
