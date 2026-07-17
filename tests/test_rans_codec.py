"""Static order-0 byte-rANS codec (issue #51, C7): CPU encode/decode + container.

Pure-numpy coverage — no MLX/Metal. Pins table normalization, the vectorized
encoder's byte-for-byte parity with the scalar reference, lane round-trips
across byte distributions, and the self-describing on-disk container.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtplx.expert_rans as R


def _table(seg: np.ndarray) -> R.RansTable:
    return R.build_table(R.histogram(seg.reshape(-1)))


# ------------------------------------------------------------------ tables


def test_histogram_counts_every_byte() -> None:
    data = np.array([0, 0, 255, 7, 7, 7], dtype=np.uint8)
    hist = R.histogram(data)
    assert hist.shape == (256,)
    assert hist[0] == 2 and hist[7] == 3 and hist[255] == 1
    assert hist.sum() == data.size


def test_build_table_normalizes_to_M() -> None:
    rng = np.random.default_rng(0)
    seg = rng.integers(0, 256, size=4096).astype(np.uint8)
    table = _table(seg)
    assert int(table.freq.sum()) == R.M
    assert int(table.cum[0]) == 0 and int(table.cum[256]) == R.M
    # Every symbol that appears keeps at least one slot.
    used = np.bincount(seg, minlength=256) > 0
    assert np.all(table.freq[used] >= 1)
    # cum2sym inverts cum: each slot maps back inside its symbol's range.
    for sym in range(256):
        lo, hi = int(table.cum[sym]), int(table.cum[sym + 1])
        if hi > lo:
            assert np.all(table.cum2sym[lo:hi] == sym)


def test_table_from_freq_rejects_bad_tables() -> None:
    with pytest.raises(R.RansError):
        R.table_from_freq(np.zeros(255, dtype=np.uint32))
    bad = np.zeros(256, dtype=np.int64)
    bad[0] = R.M - 1  # sums to M-1
    with pytest.raises(R.RansError, match="sum to M"):
        R.table_from_freq(bad)


def test_build_table_rejects_empty_histogram() -> None:
    with pytest.raises(R.RansError, match="empty"):
        R.build_table(np.zeros(256, dtype=np.int64))


def test_build_table_handles_full_alphabet() -> None:
    # 256 distinct used symbols fit exactly into M slots.
    hist = np.ones(256, dtype=np.int64)
    table = R.build_table(hist)
    assert int(table.freq.sum()) == R.M
    assert np.all(table.freq >= 1)


# ---------------------------------------------------------------- encode/decode


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "shape,dist",
    [
        ((4, 32 * 12), "uniform"),
        ((3, 32 * 9), "skewed"),
        ((6, 32 * 5), "tiny_alphabet"),
        ((1, 32), "uniform"),  # single expert, single lane-row
    ],
)
def test_vectorized_matches_scalar_and_roundtrips(seed, shape, dist) -> None:
    rng = np.random.default_rng(seed)
    if dist == "uniform":
        seg = rng.integers(0, 256, size=shape).astype(np.uint8)
    elif dist == "skewed":
        seg = rng.normal(128, 10, size=shape).clip(0, 255).astype(np.uint8)
    else:
        seg = rng.integers(0, 5, size=shape).astype(np.uint8)
    table = _table(seg)
    scalar = R.encode_bank_scalar(seg, table)
    vector = R.encode_bank(seg, table)
    assert np.array_equal(scalar.payload, vector.payload)
    assert np.array_equal(scalar.directory, vector.directory)
    decoded = R.decode_bank_reference(vector, table)
    assert np.array_equal(decoded, seg)


def test_single_symbol_bank_roundtrips() -> None:
    # Degenerate: one symbol owns every slot (freq == M). Decode must not
    # underflow the state or over-read the stream.
    seg = np.full((2, 32 * 4), 7, dtype=np.uint8)
    table = _table(seg)
    assert int(table.freq[7]) == R.M
    streams = R.encode_bank(seg, table)
    assert np.array_equal(R.decode_bank_reference(streams, table), seg)


def test_encode_segment_requires_lane_multiple() -> None:
    table = _table(np.arange(64, dtype=np.uint8))
    with pytest.raises(R.RansError, match="divisible"):
        R.encode_segment(np.zeros(31, dtype=np.uint8), table)


def test_directory_is_payload_relative_and_ascending() -> None:
    rng = np.random.default_rng(3)
    seg = rng.integers(0, 256, size=(5, 32 * 6)).astype(np.uint8)
    streams = R.encode_bank(seg, _table(seg))
    flat = streams.directory.reshape(-1)
    assert flat[0] == 0
    assert np.all(np.diff(flat.astype(np.int64)) > 0)
    assert int(flat.max()) < streams.payload.size


# ------------------------------------------------------------------ container


def test_container_roundtrips_bitwise() -> None:
    rng = np.random.default_rng(7)
    seg = rng.normal(120, 25, size=(6, 32 * 20)).clip(0, 255).astype(np.uint8)
    table = _table(seg)
    streams = R.encode_bank(seg, table)
    blob = R.serialize_component(streams, table)

    container = R.deserialize_component(blob)
    assert container.expert_count == 6
    assert container.lanes == R.LANES
    assert container.seg_len == 32 * 20
    assert np.array_equal(container.table.freq, table.freq)
    assert np.array_equal(container.directory, streams.directory.reshape(-1))
    assert np.array_equal(R.decode_container_reference(blob), seg)


def test_container_has_trailing_guard() -> None:
    seg = np.arange(32 * 3, dtype=np.uint8).reshape(1, -1)
    blob = R.serialize_component(R.encode_bank(seg, _table(seg)), _table(seg))
    assert blob[-R.RANS_GUARD_BYTES:] == b"\x00" * R.RANS_GUARD_BYTES


def test_container_rejects_corruption() -> None:
    seg = np.arange(32 * 3, dtype=np.uint8).reshape(1, -1)
    table = _table(seg)
    blob = bytearray(R.serialize_component(R.encode_bank(seg, table), table))
    bad_magic = bytes(b"XXXX" + blob[4:])
    with pytest.raises(R.RansError, match="magic"):
        R.deserialize_component(bad_magic)
    with pytest.raises(R.RansError, match="truncated|shorter"):
        R.deserialize_component(bytes(blob[: R._HEADER_DTYPE.itemsize + 8]))
