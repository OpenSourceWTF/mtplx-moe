"""Static order-0 byte-rANS codec for expert record payloads (issue #51, C7).

The measured verdict on Hy3-Q2 / GLM-Q2 expert bytes is **order-0
saturated**: zstd-19 sits within 0.1% of the byte-entropy bound and higher
order models buy nothing (see the C7 archaeology). So the codec here is the
textbook static order-0 range-Asymmetric-Numeral-System coder — one global
256-symbol frequency table per *component class*, no context modelling.

Shape, chosen for the Metal decode kernel's SIMD width (Apple GPU SIMD-group
= 32 threads):

- ``rans32x`` = a **32-bit** rANS state, byte renormalization
  (``RANS_L = 1 << 23`` so the state lives in ``[2**23, 2**31)``), scale
  ``M = 1 << 12``.
- ``x`` = **32-way interleave**: each raw segment is split into 32 lanes and
  every lane is an independent rANS stream. One assignment's 32 lanes map to
  exactly one SIMD-group, so a group shares the threadgroup-resident tables
  and its lane streams stay within one bank region. Per-lane independence
  (each lane its own byte cursor) is what makes the decode embarrassingly
  parallel — the same structure the C7b Huffman kernel used to clear its
  decode-rate bar.

Container (per component class of one layer):

- a normalized ``freq[256]`` table (sum == ``M``); ``cum`` and the
  ``cum2sym[M]`` slot table are derived on host,
- a payload = every expert's compressed blob concatenated, each blob =
  its 32 lane streams concatenated,
- a directory ``[expert, lane]`` of **payload-relative** byte offsets so a
  device thread finds its lane stream from the router's expert id.

Each lane stream is ``[4-byte LE initial state][renorm bytes]`` read forward
by the decoder: the encoder runs the symbols in reverse (ryg ``rans_byte``
convention) so the decoder emits them in order.

Everything here is offline / open-path tooling: no locks, no host work in
any decode hot path (the GPU kernel owns that).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCALE_BITS = 12
M = 1 << SCALE_BITS  # 4096
RANS_L = 1 << 23  # lower bound of the normalized state interval
LANES = 32  # interleave width == Apple GPU SIMD-group size
_UINT32_MAX = (1 << 32) - 1  # widest lane-directory offset the container holds

# On-disk container for one compressed component bank (issue #51, C7). The
# blob is fully self-describing: a fixed header, the normalized frequency
# table (u16 each; every value <= M == 4096 fits), the payload-relative lane
# directory (u32 each), the concatenated lane-stream payload, and a zero
# guard so the device decoder's final refill never reads past the buffer.
RANS_CONTAINER_MAGIC = b"RNSX"
RANS_CONTAINER_VERSION = 1
_HEADER_DTYPE = np.dtype(
    [
        ("magic", "S4"),
        ("version", "<u2"),
        ("lanes", "<u2"),
        ("expert_count", "<u4"),
        ("seg_len", "<u4"),
        ("payload_len", "<u4"),
    ]
)
RANS_GUARD_BYTES = 8


class RansError(ValueError):
    """Raised when a rANS stream or table is malformed."""


@dataclass(frozen=True)
class RansTable:
    """Static order-0 byte table (sum of frequencies == ``M``)."""

    freq: np.ndarray  # uint32[256]
    cum: np.ndarray  # uint32[257], cumulative, cum[256] == M
    cum2sym: np.ndarray  # uint8[M], slot -> symbol

    @property
    def scale_bits(self) -> int:
        return SCALE_BITS


@dataclass(frozen=True)
class LaneStreams:
    """One component bank encoded as ``expert_count`` interleaved blobs."""

    payload: np.ndarray  # uint8, concatenation of every expert's lane streams
    directory: np.ndarray  # uint32[expert_count, LANES], payload-relative offset
    seg_len: int  # raw bytes per expert segment
    per_lane: int  # raw bytes per lane (seg_len // LANES)
    expert_count: int
    lanes: int
    ratio: float  # raw bytes / compressed payload bytes


def table_from_freq(freq: np.ndarray) -> RansTable:
    """Build the ``cum``/``cum2sym`` decode tables from normalized frequencies.

    ``freq`` must be a 256-entry non-negative array summing to exactly ``M``;
    every symbol that a stream can emit needs ``freq >= 1``. This is the exact
    reconstruction the decoder performs from an on-disk container header, so
    the encoder and every decode path share one table definition.
    """

    freq = np.asarray(freq, dtype=np.int64)
    if freq.shape != (256,):
        raise RansError("frequency table must cover exactly 256 byte symbols")
    if np.any(freq < 0):
        raise RansError("frequencies must be non-negative")
    if int(freq.sum()) != M:
        raise RansError(f"frequencies must sum to M={M}, got {int(freq.sum())}")
    cum = np.zeros(257, dtype=np.uint32)
    np.cumsum(freq, out=cum[1:])
    cum2sym = np.zeros(M, dtype=np.uint8)
    for sym in range(256):
        lo = int(cum[sym])
        hi = int(cum[sym + 1])
        if hi > lo:
            cum2sym[lo:hi] = sym
    return RansTable(freq=freq.astype(np.uint32), cum=cum, cum2sym=cum2sym)


def build_table(hist: np.ndarray) -> RansTable:
    """Normalize a 256-bin byte histogram to a static rANS table."""

    hist = np.asarray(hist, dtype=np.int64)
    if hist.shape != (256,):
        raise RansError("histogram must cover exactly 256 byte symbols")
    if hist.sum() <= 0:
        raise RansError("histogram is empty")
    return table_from_freq(_normalize(hist, M))


def _normalize(hist: np.ndarray, target: int) -> np.ndarray:
    """Scale a histogram so used symbols keep freq >= 1 and the sum == target."""

    total = int(hist.sum())
    used = hist > 0
    if int(used.sum()) > target:
        raise RansError(
            f"cannot fit {int(used.sum())} used symbols into {target} slots"
        )
    freq = np.floor(hist.astype(np.float64) * target / total).astype(np.int64)
    freq[used & (freq == 0)] = 1
    # Correct the rounding drift by nudging the largest bins (they absorb a
    # +/-1 change with the least relative distortion, and stay >= 1).
    while True:
        diff = target - int(freq.sum())
        if diff == 0:
            break
        if diff > 0:
            i = int(np.argmax(freq))
            freq[i] += 1
        else:
            # Only shrink bins that stay >= 1 afterwards.
            candidates = np.where(freq > 1)[0]
            i = int(candidates[np.argmax(freq[candidates])])
            freq[i] -= 1
    return freq


def histogram(data: np.ndarray) -> np.ndarray:
    """256-bin byte histogram."""

    data = np.asarray(data, dtype=np.uint8)
    return np.bincount(data, minlength=256).astype(np.int64)


def _require_encodable(data: np.ndarray, table: RansTable) -> None:
    """Reject data carrying a symbol the table assigns ``freq == 0``.

    A zero frequency is legal in a table for symbols the stream never emits
    (a real table is mostly zeros), so the table alone cannot be validated.
    Encoding such a symbol is what is unrepresentable: ``x_max`` collapses to
    0, the renormalization loop's ``x >= x_max`` is then always true, and the
    encoder spins forever (the vectorized path also divides by zero). Fail
    closed here rather than hang.
    """

    counts = np.bincount(np.asarray(data, dtype=np.uint8).ravel(), minlength=256)
    # `counts > 0` first: a bitwise AND against raw counts would silently miss
    # any symbol whose count is even (32 & 1 == 0).
    missing = np.nonzero((counts > 0) & (np.asarray(table.freq, dtype=np.int64) == 0))[0]
    if missing.size:
        raise RansError(
            "data contains symbols with zero frequency in the table and "
            f"cannot be encoded: {missing[:8].tolist()}"
            f"{' ...' if missing.size > 8 else ''}"
        )


def _require_directory_fits(max_offset: int) -> None:
    """Reject a payload whose lane offsets overflow the uint32 directory.

    The scalar reference raises naturally here (assigning past 2**32-1 into a
    uint32 array overflows); the vectorized path's ``astype(np.uint32)`` would
    wrap silently and decode plausible garbage, so it must raise too -- the
    two encoders are documented and tested as bit-identical.
    """

    if max_offset > _UINT32_MAX:
        raise RansError(
            f"lane directory offset {max_offset} exceeds the uint32 range; "
            f"encode this bank in chunks under {_UINT32_MAX + 1} payload bytes"
        )


def _encode_lane(symbols: np.ndarray, table: RansTable) -> bytes:
    """rANS-encode one lane (ryg rans_byte convention, forward-readable)."""

    freq = table.freq
    cum = table.cum
    x = RANS_L
    out = bytearray()  # renorm bytes, appended in reverse-emission order
    x_max_base = (RANS_L >> SCALE_BITS) << 8
    for sym in symbols[::-1]:
        f = int(freq[sym])
        x_max = x_max_base * f
        while x >= x_max:
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // f) << SCALE_BITS) + (x % f) + int(cum[sym])
    # Flush the 4-byte final state. Append high byte first: after the whole
    # buffer is reversed below, the state lands little-endian at the front,
    # which is exactly what the decoder's RansDecInit reads.
    out.append((x >> 24) & 0xFF)
    out.append((x >> 16) & 0xFF)
    out.append((x >> 8) & 0xFF)
    out.append(x & 0xFF)
    out.reverse()
    return bytes(out)


def encode_segment(
    data: np.ndarray, table: RansTable, *, lanes: int = LANES
) -> list[bytes]:
    """Encode one expert segment into ``lanes`` independent lane streams."""

    data = np.asarray(data, dtype=np.uint8)
    if data.size % lanes:
        raise RansError(
            f"segment of {data.size} bytes is not divisible by {lanes} lanes"
        )
    _require_encodable(data, table)
    per_lane = data.size // lanes
    rows = data.reshape(lanes, per_lane)
    return [_encode_lane(rows[lane], table) for lane in range(lanes)]


def encode_bank_scalar(
    segments: np.ndarray, table: RansTable, *, lanes: int = LANES
) -> LaneStreams:
    """Reference bank encoder (per-lane scalar loop). Slow; used to pin down
    :func:`encode_bank`'s vectorized output byte-for-byte in tests."""

    segments = np.asarray(segments, dtype=np.uint8)
    if segments.ndim != 2:
        raise RansError("segments must be a 2-D [expert, byte] array")
    expert_count, seg_len = segments.shape
    if seg_len % lanes:
        raise RansError(
            f"segment length {seg_len} is not divisible by {lanes} lanes"
        )
    per_lane = seg_len // lanes
    payload = bytearray()
    directory = np.zeros((expert_count, lanes), dtype=np.uint32)
    for expert in range(expert_count):
        lane_blobs = encode_segment(segments[expert], table, lanes=lanes)
        for lane, blob in enumerate(lane_blobs):
            directory[expert, lane] = len(payload)
            payload.extend(blob)
    payload_arr = np.frombuffer(bytes(payload), dtype=np.uint8)
    raw = expert_count * seg_len
    return LaneStreams(
        payload=payload_arr,
        directory=directory,
        seg_len=int(seg_len),
        per_lane=int(per_lane),
        expert_count=int(expert_count),
        lanes=int(lanes),
        ratio=float(raw / max(payload_arr.size, 1)),
    )


def encode_bank(
    segments: np.ndarray, table: RansTable, *, lanes: int = LANES
) -> LaneStreams:
    """Encode a component bank: ``segments`` is ``[expert_count, seg_len]``.

    Vectorized across every lane of every expert at once (a single reverse
    pass over the ``per_lane`` symbol positions), then the emitted bytes are
    assembled columnar — no per-byte Python. Bit-identical to
    :func:`encode_bank_scalar`. Returns the concatenated payload plus a
    payload-relative ``[expert, lane]`` byte-offset directory.

    Memory during encode scales with the compressed size of ``segments``;
    a whole-corpus writer chunks experts to bound it.
    """

    segments = np.asarray(segments, dtype=np.uint8)
    if segments.ndim != 2:
        raise RansError("segments must be a 2-D [expert, byte] array")
    expert_count, seg_len = segments.shape
    if seg_len % lanes:
        raise RansError(
            f"segment length {seg_len} is not divisible by {lanes} lanes"
        )
    _require_encodable(segments, table)
    per_lane = seg_len // lanes
    n_lanes = expert_count * lanes
    rows = segments.reshape(expert_count, lanes, per_lane).reshape(n_lanes, per_lane)
    freq = table.freq.astype(np.uint64)
    cum = table.cum.astype(np.uint64)
    x = np.full(n_lanes, RANS_L, dtype=np.uint64)
    x_max_base = np.uint64((RANS_L >> SCALE_BITS) << 8)
    all_lanes = np.arange(n_lanes, dtype=np.int64)

    emit_lane: list[np.ndarray] = []
    emit_byte: list[np.ndarray] = []
    emit_seq: list[np.ndarray] = []
    seq = 0
    for j in range(per_lane - 1, -1, -1):
        s = rows[:, j]
        f = freq[s]
        c = cum[s]
        x_max = x_max_base * f
        while True:
            active = x >= x_max
            if not active.any():
                break
            idx = np.nonzero(active)[0]
            emit_lane.append(idx)
            emit_byte.append((x[idx] & np.uint64(0xFF)).astype(np.uint8))
            emit_seq.append(np.full(idx.size, seq, dtype=np.int64))
            seq += 1
            x[idx] >>= np.uint64(8)
        x = ((x // f) << np.uint64(SCALE_BITS)) + (x % f) + c
    # Flush each lane's 4-byte final state, high byte first (see _encode_lane).
    for shift in (24, 16, 8, 0):
        emit_lane.append(all_lanes)
        emit_byte.append(((x >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.uint8))
        emit_seq.append(np.full(n_lanes, seq, dtype=np.int64))
        seq += 1

    lane_col = np.concatenate(emit_lane)
    byte_col = np.concatenate(emit_byte)
    seq_col = np.concatenate(emit_seq)
    # Stable order = each lane's bytes in append order (ascending seq).
    order = np.lexsort((seq_col, lane_col))
    lane_sorted = lane_col[order]
    byte_sorted = byte_col[order]
    counts = np.bincount(lane_sorted, minlength=n_lanes)
    offsets = np.zeros(n_lanes + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    # Reverse within each lane: the decoder reads the reverse of append order.
    pos_in_lane = np.arange(lane_sorted.size) - offsets[lane_sorted]
    rev_index = offsets[lane_sorted] + (counts[lane_sorted] - 1 - pos_in_lane)
    payload_arr = np.empty(lane_sorted.size, dtype=np.uint8)
    payload_arr[rev_index] = byte_sorted
    _require_directory_fits(int(offsets[-2]) if n_lanes else 0)
    directory = offsets[:-1].reshape(expert_count, lanes).astype(np.uint32)
    raw = expert_count * seg_len
    return LaneStreams(
        payload=payload_arr,
        directory=directory,
        seg_len=int(seg_len),
        per_lane=int(per_lane),
        expert_count=int(expert_count),
        lanes=int(lanes),
        ratio=float(raw / max(payload_arr.size, 1)),
    )


def decode_lane_reference(
    payload: np.ndarray, offset: int, per_lane: int, table: RansTable
) -> np.ndarray:
    """Pure-numpy reference decode of one lane (mirrors the Metal kernel)."""

    raw = np.asarray(payload, dtype=np.uint8)
    cum2sym = table.cum2sym
    freq = table.freq
    cum = table.cum
    pos = offset
    x = (
        int(raw[pos])
        | (int(raw[pos + 1]) << 8)
        | (int(raw[pos + 2]) << 16)
        | (int(raw[pos + 3]) << 24)
    )
    pos += 4
    out = np.zeros(per_lane, dtype=np.uint8)
    mask = M - 1
    for i in range(per_lane):
        slot = x & mask
        sym = int(cum2sym[slot])
        x = int(freq[sym]) * (x >> SCALE_BITS) + slot - int(cum[sym])
        while x < RANS_L:
            x = (x << 8) | int(raw[pos])
            pos += 1
        out[i] = sym
    return out


def decode_bank_reference(streams: LaneStreams, table: RansTable) -> np.ndarray:
    """Reference decode of a whole component bank back to ``[expert, seg_len]``."""

    out = np.zeros((streams.expert_count, streams.seg_len), dtype=np.uint8)
    for expert in range(streams.expert_count):
        for lane in range(streams.lanes):
            offset = int(streams.directory[expert, lane])
            decoded = decode_lane_reference(
                streams.payload, offset, streams.per_lane, table
            )
            lo = lane * streams.per_lane
            out[expert, lo : lo + streams.per_lane] = decoded
    return out


# ---------------------------------------------------------------- container


@dataclass(frozen=True)
class RansContainer:
    """Parsed view of a serialized component container (no payload copy)."""

    table: RansTable
    directory: np.ndarray  # uint32[expert_count * lanes], payload-relative
    payload: np.ndarray  # uint8 view into the blob
    expert_count: int
    lanes: int
    seg_len: int
    per_lane: int


def serialize_component(streams: LaneStreams, table: RansTable) -> bytes:
    """Pack one encoded component bank + its table into a self-describing blob.

    Layout: ``[header | freq u16[256] | directory u32[E*L] | payload | guard]``.
    The directory is flattened row-major ``[expert, lane]`` exactly as the
    Metal kernel indexes it (``directory[expert * lanes + lane]``).
    """

    if int(table.freq.sum()) != M:
        raise RansError("table frequencies do not sum to M")
    header = np.zeros((), dtype=_HEADER_DTYPE)
    header["magic"] = RANS_CONTAINER_MAGIC
    header["version"] = RANS_CONTAINER_VERSION
    header["lanes"] = streams.lanes
    header["expert_count"] = streams.expert_count
    header["seg_len"] = streams.seg_len
    header["payload_len"] = streams.payload.size
    parts = [
        header.tobytes(),
        table.freq.astype("<u2").tobytes(),
        streams.directory.reshape(-1).astype("<u4").tobytes(),
        np.ascontiguousarray(streams.payload, dtype=np.uint8).tobytes(),
        b"\x00" * RANS_GUARD_BYTES,
    ]
    return b"".join(parts)


def container_raw_bytes(expert_count: int, seg_len: int) -> int:
    """Uncompressed byte length a container decodes back to."""

    return int(expert_count) * int(seg_len)


def deserialize_component(blob) -> RansContainer:
    """Parse a serialized container into decode-ready arrays (no payload copy)."""

    buf = np.frombuffer(blob, dtype=np.uint8)
    header_size = _HEADER_DTYPE.itemsize
    if buf.size < header_size:
        raise RansError("container is shorter than its header")
    header = buf[:header_size].view(_HEADER_DTYPE)[0]
    if bytes(header["magic"]) != RANS_CONTAINER_MAGIC:
        raise RansError("container magic mismatch")
    if int(header["version"]) != RANS_CONTAINER_VERSION:
        raise RansError(f"unsupported container version {int(header['version'])}")
    lanes = int(header["lanes"])
    expert_count = int(header["expert_count"])
    seg_len = int(header["seg_len"])
    payload_len = int(header["payload_len"])
    if lanes <= 0 or expert_count <= 0 or seg_len <= 0:
        raise RansError("container geometry is degenerate")
    if seg_len % lanes:
        raise RansError("container seg_len is not divisible by lanes")
    per_lane = seg_len // lanes
    cursor = header_size
    freq_bytes = 256 * 2
    dir_bytes = expert_count * lanes * 4
    end = cursor + freq_bytes + dir_bytes + payload_len
    if buf.size < end + RANS_GUARD_BYTES:
        raise RansError("container is truncated")
    freq = buf[cursor : cursor + freq_bytes].view("<u2").astype(np.uint32)
    cursor += freq_bytes
    directory = np.ascontiguousarray(
        buf[cursor : cursor + dir_bytes].view("<u4")
    )
    cursor += dir_bytes
    payload = buf[cursor : cursor + payload_len]
    table = table_from_freq(freq)
    if directory.size and int(directory.max()) + 4 > payload_len + RANS_GUARD_BYTES:
        raise RansError("container directory points past the payload")
    return RansContainer(
        table=table,
        directory=directory,
        payload=payload,
        expert_count=expert_count,
        lanes=lanes,
        seg_len=seg_len,
        per_lane=per_lane,
    )


def decode_container_reference(blob) -> np.ndarray:
    """Pure-numpy decode of a container back to ``[expert_count, seg_len]``."""

    c = deserialize_component(blob)
    streams = LaneStreams(
        payload=np.ascontiguousarray(c.payload),
        directory=c.directory.reshape(c.expert_count, c.lanes),
        seg_len=c.seg_len,
        per_lane=c.per_lane,
        expert_count=c.expert_count,
        lanes=c.lanes,
        ratio=float(container_raw_bytes(c.expert_count, c.seg_len))
        / max(c.payload.size, 1),
    )
    return decode_bank_reference(streams, c.table)
