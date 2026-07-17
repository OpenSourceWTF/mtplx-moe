"""In-kernel rANS decoder (issue #51, C7): Metal decode parity + container.

Skipped where MLX/Metal is unavailable. The decode kernel must reproduce the
CPU reference (and thus the raw bytes) bitwise for arbitrary routed
assignments, honor the 2**31 output-element limit, and round-trip the
self-describing container.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtplx.expert_rans as R

mx = pytest.importorskip("mlx.core")

import mtplx.expert_rans_metal as RM  # noqa: E402  (after importorskip)


def _encode(seg: np.ndarray):
    table = R.build_table(R.histogram(seg.reshape(-1)))
    streams = R.encode_bank(seg, table)
    payload = mx.array(streams.payload)
    directory = mx.array(streams.directory.reshape(-1).astype(np.uint32))
    cum2sym, freq, cum = RM.table_device_arrays(table)
    return table, streams, payload, directory, cum2sym, freq, cum


def _decode(streams, payload, directory, indices, cum2sym, freq, cum, assignments):
    out = RM.decode_component(
        payload,
        directory,
        indices,
        cum2sym,
        freq,
        cum,
        lanes=streams.lanes,
        per_lane=streams.per_lane,
        seg_len=streams.seg_len,
        assignments=assignments,
    )
    mx.eval(out)
    return np.array(out).reshape(assignments, streams.seg_len)


@pytest.mark.parametrize(
    "shape,dist",
    [
        ((8, 32 * 16), "uniform"),
        ((4, 32 * 10), "skewed"),
        ((3, 32 * 4), "tiny_alphabet"),
    ],
)
def test_metal_decode_matches_input(shape, dist) -> None:
    rng = np.random.default_rng(11)
    if dist == "uniform":
        seg = rng.integers(0, 256, size=shape).astype(np.uint8)
    elif dist == "skewed":
        seg = rng.normal(128, 9, size=shape).clip(0, 255).astype(np.uint8)
    else:
        seg = rng.integers(0, 4, size=shape).astype(np.uint8)
    experts = shape[0]
    table, streams, payload, directory, cum2sym, freq, cum = _encode(seg)
    indices = mx.array(np.arange(experts, dtype=np.int32))
    got = _decode(streams, payload, directory, indices, cum2sym, freq, cum, experts)
    assert np.array_equal(got, seg)
    # And bitwise-identical to the pure-numpy reference decoder.
    assert np.array_equal(got, R.decode_bank_reference(streams, table))


def test_metal_decode_arbitrary_routing() -> None:
    rng = np.random.default_rng(5)
    seg = rng.integers(0, 256, size=(6, 32 * 8)).astype(np.uint8)
    _table, streams, payload, directory, cum2sym, freq, cum = _encode(seg)
    order = np.array([5, 5, 0, 3, 1, 0, 4, 2], dtype=np.int32)
    indices = mx.array(order)
    got = _decode(
        streams, payload, directory, indices, cum2sym, freq, cum, order.size
    )
    assert np.array_equal(got, seg[order])


def test_metal_decode_single_symbol() -> None:
    seg = np.full((2, 32 * 3), 42, dtype=np.uint8)
    _table, streams, payload, directory, cum2sym, freq, cum = _encode(seg)
    indices = mx.array(np.arange(2, dtype=np.int32))
    got = _decode(streams, payload, directory, indices, cum2sym, freq, cum, 2)
    assert np.array_equal(got, seg)


def test_decode_component_output_limit() -> None:
    rng = np.random.default_rng(1)
    seg = rng.integers(0, 256, size=(2, 32)).astype(np.uint8)
    _table, _streams, payload, directory, cum2sym, freq, cum = _encode(seg)
    indices = mx.array(np.zeros(2, dtype=np.int32))
    # Geometry is self-consistent (lanes * per_lane == seg_len) but the total
    # output exceeds the 2**31 element shape limit — must raise before dispatch.
    with pytest.raises(ValueError, match="2\\*\\*31|exceeds"):
        RM.decode_component(
            payload,
            directory,
            indices,
            cum2sym,
            freq,
            cum,
            lanes=32,
            per_lane=(1 << 26),
            seg_len=(1 << 31),
            assignments=2,
        )


def test_decode_container_full_and_subset() -> None:
    rng = np.random.default_rng(9)
    seg = rng.normal(100, 30, size=(5, 32 * 12)).clip(0, 255).astype(np.uint8)
    table = R.build_table(R.histogram(seg.reshape(-1)))
    blob = R.serialize_component(R.encode_bank(seg, table), table)

    full = RM.decode_container(blob)
    mx.eval(full)
    assert np.array_equal(np.array(full).reshape(5, 32 * 12), seg)

    idx = mx.array(np.array([4, 0, 2], dtype=np.int32))
    subset = RM.decode_container(blob, idx)
    mx.eval(subset)
    assert np.array_equal(np.array(subset).reshape(3, 32 * 12), seg[[4, 0, 2]])
