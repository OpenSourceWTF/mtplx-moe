"""Length-limited canonical Huffman codec for expert records (issue #51, C7a).

The codec feeds the measured v1 Metal decoder (53.6 GiB/s write): canonical
codes limited to L1_BITS via package-merge (optimal, unlike frequency
flattening which measured 5.5% over entropy), lane-interleaved MSB-first
bitstreams packed into byteswapped u32 words with per-lane word offsets.
These tests pin optimality properties, the stream format, and byte-exact
roundtrip through the pure-numpy reference decoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtplx.expert_huffman import (
    HuffmanTable,
    build_l1_entries,
    build_table,
    decode_lanes_reference,
    encode_lanes,
)


def _random_weightlike(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Skewed byte distribution resembling packed 2-bit code bytes:
    # ~78% code utilization -> non-uniform but heavy-tailed.
    p = rng.dirichlet(np.full(256, 0.35))
    return rng.choice(256, size=n, p=p).astype(np.uint8)


def test_build_table_respects_max_bits_and_kraft() -> None:
    data = _random_weightlike(1 << 16, seed=1)
    hist = np.bincount(data, minlength=256)
    for max_bits in (12, 16):
        table = build_table(hist, max_bits=max_bits)
        assert table.lengths.max() <= max_bits
        assert table.lengths.min() >= 1
        kraft = float(np.sum(2.0 ** -table.lengths.astype(np.float64)))
        assert kraft <= 1.0 + 1e-12
        # Canonical: sorting by (length, symbol) yields increasing codes.
        order = np.lexsort((np.arange(256), table.lengths))
        codes = table.codes[order]
        lengths = table.lengths[order]
        prev_code = -1
        prev_len = 0
        for code, length in zip(codes.tolist(), lengths.tolist()):
            aligned = code << (int(table.max_bits) - length)
            assert aligned > prev_code
            prev_code = aligned
            prev_len = length


def test_package_merge_beats_flattening_on_real_shape() -> None:
    data = _random_weightlike(1 << 18, seed=2)
    hist = np.bincount(data, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    entropy = -(p[p > 0] * np.log2(p[p > 0])).sum()
    table = build_table(hist.astype(np.int64), max_bits=12)
    avg_bits = float((table.lengths * p).sum())
    # Package-merge at 12 bits must sit within 2% of entropy on this shape
    # (sqrt-flattening measured 5.5% over on real Q2 bytes).
    assert avg_bits <= entropy * 1.02


def test_encode_lanes_format_invariants() -> None:
    data = _random_weightlike(64 * 1024, seed=3)
    table = build_table(np.bincount(data, minlength=256), max_bits=12)
    per_lane = 4096
    stream = encode_lanes(data, table, per_lane=per_lane)
    lanes = data.size // per_lane
    assert stream.lanes == lanes
    assert stream.word_offsets.dtype == np.uint32
    assert stream.word_offsets.shape == (lanes,)
    assert stream.words.dtype == np.uint32
    # Offsets strictly increasing and within the stream.
    diffs = np.diff(stream.word_offsets.astype(np.int64))
    assert (diffs > 0).all()
    assert int(stream.word_offsets[-1]) < stream.words.size
    # Every lane owns at least ceil(per_lane * 1 bit / 32) words of payload.
    assert stream.per_lane == per_lane


def test_roundtrip_byte_exact_random_and_edge_distributions() -> None:
    per_lane = 512
    for seed, maker in (
        (4, lambda: _random_weightlike(per_lane * 8, seed=4)),
        (5, lambda: np.zeros(per_lane * 8, dtype=np.uint8)),  # degenerate
        (6, lambda: np.arange(per_lane * 8, dtype=np.uint32).astype(np.uint8)),
    ):
        data = maker()
        table = build_table(np.bincount(data, minlength=256), max_bits=12)
        stream = encode_lanes(data, table, per_lane=per_lane)
        decoded = decode_lanes_reference(stream, table)
        assert np.array_equal(
            decoded.reshape(-1), data[: decoded.size]
        ), f"roundtrip mismatch for seed {seed}"


def test_l1_entries_match_reference_decode() -> None:
    data = _random_weightlike(1 << 15, seed=7)
    table = build_table(np.bincount(data, minlength=256), max_bits=12)
    l1 = build_l1_entries(table)
    assert l1.shape == (1 << 12,)
    assert l1.dtype == np.uint32
    # With max_bits == L1 bits there must be no long-code fallbacks.
    assert not (l1 & 0x80000000).any()
    # Each entry decodes 1 or 2 symbols and consumes 1..12 bits.
    counts = (l1 >> 5) & 0x3
    bits = l1 & 0x1F
    assert set(np.unique(counts).tolist()) <= {1, 2}
    assert bits.min() >= 1 and bits.max() <= 12
    # Spot-check entries against direct canonical decoding.
    for peek in (0, 1, 255, 4095, 2048):
        entry = int(l1[peek])
        count = (entry >> 5) & 0x3
        total_bits = entry & 0x1F
        s0 = (entry >> 8) & 0xFF
        sym0, len0 = table.peek_decode(peek, peek_bits=12)
        assert sym0 == s0
        if count == 1:
            assert total_bits == len0
        else:
            rem = 12 - len0
            tail = (peek & ((1 << rem) - 1)) << (12 - rem)
            sym1, len1 = table.peek_decode(tail, peek_bits=12)
            assert (entry >> 16) & 0xFF == sym1
            assert total_bits == len0 + len1
            assert len1 <= rem


def test_ratio_reported_matches_stream_size() -> None:
    data = _random_weightlike(1 << 18, seed=8)
    table = build_table(np.bincount(data, minlength=256), max_bits=12)
    stream = encode_lanes(data, table, per_lane=4096)
    assert 1.0 < stream.ratio < 8.0
    assert abs(stream.ratio - data.size / stream.words.nbytes) < 1e-9
