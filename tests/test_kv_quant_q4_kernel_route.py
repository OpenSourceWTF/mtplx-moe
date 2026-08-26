"""q4 packed-quant kernel route: decision logic + head-major bank contract.

Phase 0.3 of the kv-quant zero-speed-loss package: q4 no longer runs the
full-prefix-dequant-every-round path — requests whose geometry the
packed-quant kernel (kernels/sdpa_gqa_packed_quant) supports latch the
"kernel" route and read a persistent head-major QUANTIZED bank (payloads +
fp32 row scales), extended tail-only per step under the dequant memo's
lifecycle. These tests pin the route decision and the bank's exactness /
lifecycle with pure mx ops; the Metal kernel itself is exercised by
tests/test_sdpa_gqa_packed_quant.py (GPU lane) and is monkeypatched here
wherever a dispatch would otherwise occur.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.attention_context import attention_phase
from mtplx.cache_state import VllmMetalPagedKVCache
from mtplx.kv_quant import PagedKVQuantConfig, quantize_symmetric

DIM = 128
KV_HEADS = 2
Q_HEADS = 8  # gqa 4


def _build_cache(
    mode: str = "q4",
    *,
    block_size: int = 16,
    num_blocks: int = 16,
) -> VllmMetalPagedKVCache:
    return VllmMetalPagedKVCache(
        block_size=block_size,
        num_blocks=num_blocks,
        kv_quant_config=PagedKVQuantConfig(mode),
    )


def _rows(count: int, seed: int, dim: int = DIM) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    keys = 0.5 * mx.random.normal((1, KV_HEADS, count, dim), dtype=mx.float16)
    values = 0.5 * mx.random.normal((1, KV_HEADS, count, dim), dtype=mx.float16)
    return keys, values


def _queries(q_len: int, seed: int, dim: int = DIM) -> mx.array:
    mx.random.seed(seed)
    return 0.3 * mx.random.normal((1, Q_HEADS, q_len, dim), dtype=mx.float16)


def _prefill(cache: VllmMetalPagedKVCache, count: int, seed: int) -> None:
    keys, values = _rows(count, seed)
    with attention_phase("prefill"):
        cache.update_without_fetch(keys, values)


# ---------------------------------------------------------------------------
# Route decision (pure logic, no kernel dispatch)
# ---------------------------------------------------------------------------


def test_q4_route_latches_kernel_above_threshold(monkeypatch):
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "1")
    monkeypatch.delenv("MTPLX_KV_QUANT_Q4_KERNEL", raising=False)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "64")

    cache = _build_cache("q4")
    _prefill(cache, 200, seed=11)
    decision = cache._kv_quant_route_decision(
        _queries(1, seed=12), sliding_window=-1
    )
    assert decision == "kernel"


def test_q4_route_latches_dequant_below_threshold(monkeypatch):
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "1024")

    cache = _build_cache("q4")
    _prefill(cache, 40, seed=21)
    decision = cache._kv_quant_route_decision(
        _queries(1, seed=22), sliding_window=-1
    )
    assert decision == "dequant"


@pytest.mark.parametrize(
    "env_name",
    ["MTPLX_KV_QUANT_2PASS_KERNEL", "MTPLX_KV_QUANT_Q4_KERNEL"],
)
def test_q4_route_respects_kill_switches(monkeypatch, env_name):
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "64")
    monkeypatch.setenv(env_name, "0")

    cache = _build_cache("q4")
    _prefill(cache, 200, seed=31)
    decision = cache._kv_quant_route_decision(
        _queries(1, seed=32), sliding_window=-1
    )
    assert decision == "dequant"


def test_q4_route_refuses_sliding_window_and_bad_geometry(monkeypatch):
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "64")

    cache = _build_cache("q4")
    _prefill(cache, 200, seed=41)
    assert (
        cache._kv_quant_route_decision(_queries(1, seed=42), sliding_window=64)
        == "dequant"
    )
    # Ragged GQA: query heads not a multiple of kv heads.
    mx.random.seed(43)
    ragged = 0.3 * mx.random.normal((1, 7, 1, DIM), dtype=mx.float16)
    assert (
        cache._kv_quant_route_decision(ragged, sliding_window=-1) == "dequant"
    )

    # Head dim outside the packed-quant kernel's set.
    narrow = _build_cache("q4")
    keys, values = _rows(200, seed=44, dim=96)
    with attention_phase("prefill"):
        narrow.update_without_fetch(keys, values)
    assert (
        narrow._kv_quant_route_decision(
            _queries(1, seed=45, dim=96), sliding_window=-1
        )
        == "dequant"
    )


def test_packed_quant_static_blocks_from_ceiling():
    """Blocks derive from the STATIC ceiling (compiled bucket), clamped to
    capacity — the shape-stability contract of the compiled adapter."""
    from mtplx.kernels.sdpa_gqa_packed import _blocks_for_capacity
    from mtplx.kernels.sdpa_gqa_packed_quant import _static_blocks

    assert _static_blocks(131072, None) == _blocks_for_capacity(131072)
    assert _static_blocks(131072, 4096) == _blocks_for_capacity(4096)
    assert _static_blocks(2048, 131072) == _blocks_for_capacity(2048)  # clamp
    assert _static_blocks(2048, 0) == _blocks_for_capacity(1)  # floor
    for capacity, max_offset in ((64, 64), (32768, 8192), (262144, 32768)):
        blocks = _static_blocks(capacity, max_offset)
        assert blocks > 0 and blocks % 32 == 0


def test_packed_quant_safe_q_len_geometry_table():
    cache = _build_cache("q4")
    _prefill(cache, 32, seed=51)
    # Shipped-class geometry: supported, two-bank ceiling 8.
    assert cache._packed_quant_safe_q_len(query_heads=Q_HEADS) == 8
    # 24/4 (Qwen3.8-27B full-attn): 32 * 6 threads per group is legal.
    assert cache._packed_quant_safe_q_len(query_heads=KV_HEADS * 6) == 8
    # Ragged GQA refuses.
    assert cache._packed_quant_safe_q_len(query_heads=7) == 0
    # Unallocated cache refuses.
    fresh = _build_cache("q4")
    assert fresh._packed_quant_safe_q_len(query_heads=Q_HEADS) == 0


# ---------------------------------------------------------------------------
# Head-major quant bank: exactness + lifecycle (pure mx ops)
# ---------------------------------------------------------------------------


def _assert_bank_matches_pages(cache: VllmMetalPagedKVCache) -> None:
    """The bank prefix must hold byte-identical payloads/scales to the pages."""
    offset = int(cache.offset)
    bank = cache._quant_bank
    assert bank is not None
    assert int(bank["tokens"]) == offset
    heads = int(cache.key_cache.shape[2])
    for name, source in (
        ("k", cache.key_cache),
        ("v", cache.value_cache),
        ("ks", cache.key_scale_cache),
        ("vs", cache.value_scale_cache),
    ):
        expect = (
            source.reshape(-1, heads, int(source.shape[3]))[:offset]
            .transpose(1, 0, 2)[None, ...]
        )
        got = bank[name][:, :, :offset, :]
        mx.eval(expect, got)
        if name in ("ks", "vs"):
            assert (
                float(mx.abs(got - expect.astype(mx.float32)).max().item())
                == 0.0
            )
        else:
            assert int((got != expect).sum().item()) == 0


def test_quant_bank_matches_pages_and_direct_quantization():
    cache = _build_cache("q4")
    keys, values = _rows(75, seed=61)
    with attention_phase("prefill"):
        cache.update_without_fetch(keys, values)

    bank_k, bank_v, bank_ks, bank_vs = cache._quant_bank_arrays()
    _assert_bank_matches_pages(cache)

    # Transpose commutes with the rowwise quantizer: the bank must hold the
    # SAME integers/scales as quantizing the head-major inputs directly —
    # the layout the packed-quant kernel's own exactness tests feed it.
    direct_k, direct_ks = quantize_symmetric(keys, bits=4)
    direct_v, direct_vs = quantize_symmetric(values, bits=4)
    offset = int(cache.offset)
    mx.eval(bank_k, bank_ks, bank_v, bank_vs, direct_k, direct_ks, direct_v, direct_vs)
    assert int((bank_k[:, :, :offset, :] != direct_k).sum().item()) == 0
    assert int((bank_v[:, :, :offset, :] != direct_v).sum().item()) == 0
    assert (
        float(mx.abs(bank_ks[:, :, :offset, :] - direct_ks).max().item()) == 0.0
    )
    assert (
        float(mx.abs(bank_vs[:, :, :offset, :] - direct_vs).max().item()) == 0.0
    )
    assert bank_ks.dtype == mx.float32 and bank_vs.dtype == mx.float32


def test_quant_bank_extends_tail_only():
    cache = _build_cache("q4")
    _prefill(cache, 60, seed=71)
    cache._quant_bank_arrays()
    assert cache.kv_quant_bank_rebuilds == 1
    extended_after_build = int(cache.kv_quant_bank_extended_tokens)
    assert extended_after_build == 60

    tail_k, tail_v = _rows(3, seed=72)
    with attention_phase("decode_verify"):
        cache.update_without_fetch(tail_k, tail_v)
    cache._quant_bank_arrays()
    assert cache.kv_quant_bank_rebuilds == 1  # no rebuild
    assert int(cache.kv_quant_bank_extended_tokens) == extended_after_build + 3
    _assert_bank_matches_pages(cache)

    # A second call at the same offset extends nothing.
    cache._quant_bank_arrays()
    assert int(cache.kv_quant_bank_extended_tokens) == extended_after_build + 3


def test_quant_bank_is_offset_sized_not_capacity_sized():
    cache = _build_cache("q4", block_size=4, num_blocks=64)  # capacity 256
    _prefill(cache, 24, seed=81)
    bank_k, _bank_v, _ks, _vs = cache._quant_bank_arrays()
    assert int(bank_k.shape[2]) < cache.capacity
    assert int(bank_k.shape[2]) >= 24


def test_quant_bank_survives_trim_and_rewrite():
    cache = _build_cache("q4")
    _prefill(cache, 50, seed=91)
    cache._quant_bank_arrays()

    cache.trim(10)
    assert int(cache._quant_bank["tokens"]) == 40

    replacement_k, replacement_v = _rows(10, seed=92)
    with attention_phase("decode_verify"):
        cache.update_without_fetch(replacement_k, replacement_v)
    cache._quant_bank_arrays()
    _assert_bank_matches_pages(cache)


def test_quant_bank_survives_page_growth(monkeypatch):
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    cache = _build_cache("q4", block_size=4, num_blocks=8)  # capacity 32
    _prefill(cache, 30, seed=93)
    cache._quant_bank_arrays()

    tail_k, tail_v = _rows(20, seed=94)
    with attention_phase("decode_verify"):
        cache.update_without_fetch(tail_k, tail_v)  # grows the pages
    assert cache.grow_events >= 1
    cache._quant_bank_arrays()
    _assert_bank_matches_pages(cache)


def test_quant_bank_dropped_on_buffer_reload():
    cache = _build_cache("q4")
    _prefill(cache, 40, seed=95)
    cache._quant_bank_arrays()
    assert cache._quant_bank is not None

    keys, values = _rows(16, seed=96)
    cache.state = (keys, values)
    assert cache._quant_bank is None


def test_nbytes_counts_live_quant_bank():
    cache = _build_cache("q4")
    _prefill(cache, 64, seed=97)
    before = int(cache.nbytes)
    cache._quant_bank_arrays()
    after = int(cache.nbytes)
    bank = cache._quant_bank
    bank_bytes = sum(int(bank[name].nbytes) for name in ("k", "v", "ks", "vs"))
    assert after == before + bank_bytes


# ---------------------------------------------------------------------------
# Dispatch plumbing (kernel monkeypatched: no Metal dispatch in this file)
# ---------------------------------------------------------------------------


def test_q4_kernel_dispatch_passes_bank_and_bits(monkeypatch):
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "64")

    cache = _build_cache("q4")
    _prefill(cache, 100, seed=98)

    calls: list[dict] = []
    sentinel = mx.zeros((1, Q_HEADS, 1, DIM), dtype=mx.float16)

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        return sentinel

    import mtplx.kernels.sdpa_gqa_packed_quant as quant_mod

    monkeypatch.setattr(
        quant_mod, "sdpa_gqa_packed_tail_quant", fake_kernel
    )

    out = cache._kv_quant_2pass_attention(
        _queries(1, seed=99),
        scale=DIM**-0.5,
        mask="causal",
        sliding_window=-1,
        q_len=1,
    )
    assert out is sentinel
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["bits"] == 4
    assert kwargs["offset"] == 100
    assert kwargs["max_q_len"] == 8
    bank = cache._quant_bank
    assert kwargs["k_q"] is bank["k"]
    assert kwargs["v_q"] is bank["v"]
    assert kwargs["k_scale"] is bank["ks"]
    assert kwargs["v_scale"] is bank["vs"]


def test_q4_kernel_dispatch_bails_before_bank_on_wide_q(monkeypatch):
    """Prefill-width bursts must not build the bank they cannot consume."""
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "1")

    cache = _build_cache("q4")
    _prefill(cache, 100, seed=100)
    out = cache._kv_quant_2pass_attention(
        _queries(16, seed=101),
        scale=DIM**-0.5,
        mask="causal",
        sliding_window=-1,
        q_len=16,
    )
    assert out is None
    assert cache._quant_bank is None


def test_q4_kernel_route_bail_lands_on_bounded_split_lane(monkeypatch):
    """A kernel-routed q4 call the packed kernel declines must use the
    chunked online-softmax lane, never the full-width dequant fallback."""
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    monkeypatch.setenv("MTPLX_KV_QUANT_2PASS_KERNEL", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD", "64")

    cache = _build_cache("q4")
    _prefill(cache, 100, seed=102)

    import mtplx.kernels.sdpa_gqa_packed_quant as quant_mod

    monkeypatch.setattr(
        quant_mod, "sdpa_gqa_packed_tail_quant", lambda **kwargs: None
    )
    with attention_phase("ar_decode"):
        out = cache.paged_attention(
            _queries(1, seed=103), scale=DIM**-0.5, mask="causal"
        )
    assert out is not None
    mx.eval(out)
    assert cache.paged_stats()["kv_quant_route"] == "kernel"
    assert cache.kv_quant_kernel_calls == 0
    assert cache.large_q_split_sdpa_fallback_calls == 1
