"""Metal decode of static order-0 byte-rANS expert banks (issue #51, C7).

One dispatch decodes one component class for every routed assignment of a
layer. Thread ``t`` owns lane ``t % LANES`` of assignment ``t / LANES``,
resolves the expert id from the router's indices ON DEVICE, follows the
component's payload-relative lane directory to its stream, and emits raw
segment bytes into the assignment's slot.

The decoder is the textbook ryg ``rans_byte`` step (32-bit state, byte
renormalization, ``RANS_L = 1 << 23``, ``M = 1 << 12``): peek the low
``SCALE_BITS`` of the state as a slot, look the symbol up in a
threadgroup-resident ``cum2sym`` table, advance the state, and refill from
the payload byte cursor. Everything lives inside the lazy MLX graph — no
host contact, no locks, no misses by construction.

The kernel must NOT be captured by ``mx.compile`` (metal_kernel bodies are
not traceable) and its output element count must stay under 2**31.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

from mtplx.expert_rans import (
    M,
    RANS_L,
    SCALE_BITS,
    RansTable,
    deserialize_component,
)

_HEADER = "#include <metal_stdlib>\nusing namespace metal;"

# Compile-time constants baked per (lanes, per_lane, seg_len). The state math
# is the reference decoder verbatim; the only device-specific work is the
# cooperative load of the tables into threadgroup memory.
_BODY_TEMPLATE = """
    threadgroup uchar cum2sym[{m}];
    threadgroup uint  freq_tg[256];
    threadgroup uint  cum_tg[256];
    for (uint i = thread_position_in_threadgroup.x;
         i < {m}u; i += threads_per_threadgroup.x) {{
        cum2sym[i] = cum2sym_dev[i];
    }}
    for (uint i = thread_position_in_threadgroup.x;
         i < 256u; i += threads_per_threadgroup.x) {{
        freq_tg[i] = freq_dev[i];
        cum_tg[i]  = cum_dev[i];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint t = thread_position_in_grid.x;
    uint assignment = t / {lanes}u;
    uint lane = t % {lanes}u;
    uint expert = uint(indices[assignment]);
    uint pos = directory[expert * {lanes}u + lane];

    uint x = uint(payload[pos])
           | (uint(payload[pos + 1u]) << 8)
           | (uint(payload[pos + 2u]) << 16)
           | (uint(payload[pos + 3u]) << 24);
    pos += 4u;

    ulong out_base = ulong(assignment) * {seg_len}u + ulong(lane) * {per_lane}u;
    for (uint i = 0u; i < {per_lane}u; ++i) {{
        uint slot = x & {mask}u;
        uint sym = uint(cum2sym[slot]);
        x = freq_tg[sym] * (x >> {scale_bits}u) + slot - cum_tg[sym];
        while (x < {rans_l}u) {{
            x = (x << 8) | uint(payload[pos]);
            pos += 1u;
        }}
        out[out_base + i] = uchar(sym);
    }}
"""


@lru_cache(maxsize=None)
def _kernel(lanes: int, per_lane: int, seg_len: int):
    source = _BODY_TEMPLATE.format(
        m=M,
        mask=M - 1,
        lanes=lanes,
        per_lane=per_lane,
        seg_len=seg_len,
        scale_bits=SCALE_BITS,
        rans_l=RANS_L,
    )
    return mx.fast.metal_kernel(
        name=f"rans32x_seg_{lanes}_{per_lane}_{seg_len}",
        input_names=["payload", "directory", "indices", "cum2sym_dev",
                     "freq_dev", "cum_dev"],
        output_names=["out"],
        header=_HEADER,
        source=source,
    )


def table_device_arrays(table: RansTable) -> tuple[mx.array, mx.array, mx.array]:
    """Materialize a table's ``(cum2sym, freq, cum)`` as device arrays."""

    cum2sym = mx.array(np.asarray(table.cum2sym, dtype=np.uint8))
    freq = mx.array(np.asarray(table.freq, dtype=np.uint32))
    cum = mx.array(np.asarray(table.cum[:256], dtype=np.uint32))
    return cum2sym, freq, cum


def decode_component(
    payload: mx.array,
    directory: mx.array,
    indices: mx.array,
    cum2sym: mx.array,
    freq: mx.array,
    cum: mx.array,
    *,
    lanes: int,
    per_lane: int,
    seg_len: int,
    assignments: int,
    threadgroup: int = 256,
) -> mx.array:
    """Decode one component's raw bytes for every routed assignment.

    ``payload`` is a ``uint8`` device array; ``directory`` is a flat
    ``uint32[expert_count * lanes]`` of payload-relative byte offsets;
    ``indices`` is ``int32[assignments]`` of expert ids. Returns a
    ``uint8[assignments * seg_len]`` array in raw segment byte order.
    """

    if assignments <= 0:
        raise ValueError("assignments must be positive")
    if lanes * per_lane != seg_len:
        raise ValueError("lanes * per_lane must equal seg_len")
    if assignments * seg_len >= (1 << 31):
        raise ValueError(
            f"decode output of {assignments} x {seg_len} bytes exceeds the "
            "2**31 element shape limit; chunk or deduplicate assignments"
        )
    kernel = _kernel(lanes, per_lane, seg_len)
    total_threads = assignments * lanes
    tg = min(threadgroup, total_threads)
    # Keep whole assignments (LANES threads) inside one threadgroup so a
    # SIMD-group shares the tables and lane streams stay bank-local.
    tg = max(tg - (tg % lanes), lanes) if tg >= lanes else tg
    (out,) = kernel(
        inputs=[payload, directory, indices, cum2sym, freq, cum],
        grid=(total_threads, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(assignments * seg_len,)],
        output_dtypes=[mx.uint8],
    )
    return out


def decode_container(blob, indices: mx.array | None = None) -> mx.array:
    """Decode one serialized component container into raw segment bytes.

    Parses the self-describing blob on the host (offline path), then runs the
    Metal decoder. ``indices`` selects which expert id each output assignment
    decodes; the default (``None``) rebuilds the whole bank in expert order,
    i.e. ``[expert_count * seg_len]`` raw bytes with row == expert id.
    """

    c = deserialize_component(blob)
    payload = mx.array(np.ascontiguousarray(c.payload))
    directory = mx.array(np.ascontiguousarray(c.directory, dtype=np.uint32))
    cum2sym = mx.array(np.asarray(c.table.cum2sym, dtype=np.uint8))
    freq = mx.array(np.asarray(c.table.freq, dtype=np.uint32))
    cum = mx.array(np.asarray(c.table.cum[:256], dtype=np.uint32))
    if indices is None:
        indices = mx.arange(c.expert_count, dtype=mx.int32)
        assignments = c.expert_count
    else:
        assignments = int(indices.shape[0])
    return decode_component(
        payload,
        directory,
        indices,
        cum2sym,
        freq,
        cum,
        lanes=c.lanes,
        per_lane=c.per_lane,
        seg_len=c.seg_len,
        assignments=assignments,
    )
