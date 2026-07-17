"""Length-limited canonical Huffman codec for expert records (issue #51, C7a).

Produces exactly the stream format the measured C7b v1 Metal decoder
consumes (53.6 GiB/s write / 128.6 consume, byte-exact on real Q2 bytes):

- code lengths from **package-merge** (optimal under a max-length bound;
  the sqrt-flattening shortcut measured 5.5% over entropy — don't),
- canonical codes assigned in (length, symbol) order,
- lane-interleaved MSB-first bitstreams, each lane padded to a u32
  boundary, packed into **byteswapped** u32 words so the GPU's
  little-endian aligned loads see MSB-first bit order directly,
- per-lane word offsets, and a 12-bit first-level decode table whose
  entries pack {consumed bits, symbol count, sym0, sym1}.

Everything here is offline/open-path tooling: no locks, no host work in
any decode hot path (the GPU kernel owns that).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

L1_BITS = 12


@dataclass(frozen=True)
class HuffmanTable:
    lengths: np.ndarray  # uint8[256]
    codes: np.ndarray  # uint32[256], right-aligned canonical codes
    max_bits: int

    def peek_decode(self, peek: int, *, peek_bits: int) -> tuple[int, int]:
        """Reference decode of the first symbol from a peek window."""

        for length in range(1, self.max_bits + 1):
            prefix = peek >> (peek_bits - length)
            matches = np.nonzero(
                (self.lengths == length) & (self.codes == prefix)
            )[0]
            if matches.size:
                return int(matches[0]), length
        raise ValueError(f"peek {peek:#x} decodes no symbol")


@dataclass(frozen=True)
class LaneStream:
    words: np.ndarray  # uint32, byteswapped MSB-first payload
    word_offsets: np.ndarray  # uint32[lanes]
    lanes: int
    per_lane: int
    ratio: float


def _package_merge_lengths(freqs: np.ndarray, max_bits: int) -> np.ndarray:
    """Optimal length-limited code lengths (boundary package-merge)."""

    freqs = np.maximum(np.asarray(freqs, dtype=np.int64), 1)
    n = freqs.size
    if n < 2:
        raise ValueError("package-merge needs at least two symbols")
    if (1 << max_bits) < n:
        raise ValueError(f"max_bits {max_bits} cannot code {n} symbols")
    symbols = sorted((int(f), (i,)) for i, f in enumerate(freqs))
    merged = list(symbols)
    for _ in range(max_bits - 1):
        packages = [
            (
                merged[i][0] + merged[i + 1][0],
                merged[i][1] + merged[i + 1][1],
            )
            for i in range(0, len(merged) - 1, 2)
        ]
        merged = sorted(symbols + packages)
    lengths = np.zeros(n, dtype=np.uint8)
    for _, members in merged[: 2 * (n - 1)]:
        for sym in members:
            lengths[sym] += 1
    return lengths


def _canonical_codes(lengths: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(lengths.size), lengths))
    codes = np.zeros(lengths.size, dtype=np.uint32)
    code = 0
    prev = 0
    for sym in order:
        length = int(lengths[sym])
        code <<= length - prev
        codes[sym] = code
        code += 1
        prev = length
    return codes


def build_table(hist: np.ndarray, *, max_bits: int = L1_BITS) -> HuffmanTable:
    hist = np.asarray(hist)
    if hist.shape != (256,):
        raise ValueError("histogram must cover exactly 256 byte symbols")
    lengths = _package_merge_lengths(hist, max_bits)
    if lengths.max() > max_bits or lengths.min() < 1:
        raise AssertionError("package-merge produced out-of-bound lengths")
    return HuffmanTable(
        lengths=lengths,
        codes=_canonical_codes(lengths),
        max_bits=int(max_bits),
    )


def encode_lanes(
    data: np.ndarray,
    table: HuffmanTable,
    *,
    per_lane: int,
) -> LaneStream:
    data = np.asarray(data, dtype=np.uint8)
    if per_lane <= 0:
        raise ValueError("per_lane must be positive")
    lanes = data.size // per_lane
    if lanes == 0:
        raise ValueError("input smaller than one lane")
    rows = data[: lanes * per_lane].reshape(lanes, per_lane)
    sym_len = table.lengths[rows].astype(np.int64)
    lane_bits = sym_len.sum(axis=1)
    # Pad each lane to a u32 boundary with one slack word so the decoder's
    # 32-bit refill never reads past the payload.
    lane_bytes = ((lane_bits + 7) // 8 + 4 + 3) // 4 * 4
    offsets = np.zeros(lanes + 1, dtype=np.int64)
    np.cumsum(lane_bytes, out=offsets[1:])
    stream = np.zeros(int(offsets[-1]), dtype=np.uint8)
    codes = table.codes
    for lane in range(lanes):
        row = rows[lane]
        lens = sym_len[lane]
        offs = np.concatenate(([0], np.cumsum(lens)[:-1]))
        bits = np.zeros(int(lane_bits[lane]), dtype=np.uint8)
        cs = codes[row]
        for bit in range(int(lens.max())):
            mask = lens > bit
            bits[offs[mask] + bit] = (cs[mask] >> (lens[mask] - 1 - bit)) & 1
        packed = np.packbits(bits)
        stream[offsets[lane] : offsets[lane] + len(packed)] = packed
    words = stream.view(np.uint32).byteswap()
    return LaneStream(
        words=words,
        word_offsets=(offsets[:-1] // 4).astype(np.uint32),
        lanes=lanes,
        per_lane=int(per_lane),
        ratio=float(rows.size / words.nbytes),
    )


def build_l1_entries(table: HuffmanTable) -> np.ndarray:
    """First-level decode table in the v1 kernel layout.

    entry = consumed_bits | count << 5 | sym0 << 8 | sym1 << 16; the high
    bit flags prefixes whose first code exceeds L1_BITS (impossible when
    the table is built with max_bits == L1_BITS).
    """

    size = 1 << L1_BITS
    first_sym = np.zeros(size, dtype=np.int64)
    first_len = np.zeros(size, dtype=np.int64)
    ok = np.zeros(size, dtype=bool)
    for sym in range(256):
        length = int(table.lengths[sym])
        if length > L1_BITS:
            continue
        base = int(table.codes[sym]) << (L1_BITS - length)
        span = 1 << (L1_BITS - length)
        first_sym[base : base + span] = sym
        first_len[base : base + span] = length
        ok[base : base + span] = True
    entries = np.zeros(size, dtype=np.uint32)
    for peek in range(size):
        if not ok[peek]:
            entries[peek] = 0x80000000
            continue
        s0 = int(first_sym[peek])
        n0 = int(first_len[peek])
        rem = L1_BITS - n0
        entry = n0 | (1 << 5) | (s0 << 8)
        if rem > 0:
            tail = (peek & ((1 << rem) - 1)) << (L1_BITS - rem)
            if ok[tail] and int(first_len[tail]) <= rem:
                s1 = int(first_sym[tail])
                n1 = int(first_len[tail])
                entry = (n0 + n1) | (2 << 5) | (s0 << 8) | (s1 << 16)
        entries[peek] = entry
    return entries


def decode_lanes_reference(stream: LaneStream, table: HuffmanTable) -> np.ndarray:
    """Pure-numpy reference decoder over the kernel-facing word stream."""

    raw = stream.words.byteswap().view(np.uint8)
    out = np.zeros((stream.lanes, stream.per_lane), dtype=np.uint8)
    lut_sym = np.zeros(1 << table.max_bits, dtype=np.uint8)
    lut_len = np.zeros(1 << table.max_bits, dtype=np.uint8)
    for sym in range(256):
        length = int(table.lengths[sym])
        base = int(table.codes[sym]) << (table.max_bits - length)
        span = 1 << (table.max_bits - length)
        lut_sym[base : base + span] = sym
        lut_len[base : base + span] = length
    for lane in range(stream.lanes):
        byte_base = int(stream.word_offsets[lane]) * 4
        bitpos = 0
        for i in range(stream.per_lane):
            b0 = byte_base + (bitpos >> 3)
            shift = bitpos & 7
            window = (
                (int(raw[b0]) << 24)
                | (int(raw[b0 + 1]) << 16)
                | (int(raw[b0 + 2]) << 8)
                | int(raw[b0 + 3])
            )
            peek = (window >> (32 - shift - table.max_bits)) & (
                (1 << table.max_bits) - 1
            )
            out[lane, i] = lut_sym[peek]
            bitpos += int(lut_len[peek])
    return out
