"""Metal decode of compressed banked expert segments (issue #51, C7c).

One dispatch decodes one component class for every routed assignment of a
layer: thread t owns lane ``t % LANES`` of assignment ``t / LANES``, looks
up the expert id from the router's indices ON DEVICE, follows the layer's
lane directory to its stream, and emits raw segment bytes (BF16 segments
are group-plane-unsplit during the write). The measured kernel family ran
53.6 GiB/s byte-exact (C7b v1); everything here lives inside the lazy
graph — no host contact, no locks, no misses by construction.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

from mtplx.expert_huffman import HuffmanTable, _canonical_codes

L1_BITS = 12

_HEADER = "#include <metal_stdlib>\nusing namespace metal;"

_BODY_TEMPLATE = """
    threadgroup uint l1[{l1_size}];
    for (uint i = thread_position_in_threadgroup.x;
         i < {l1_size}u;
         i += threads_per_threadgroup.x) {{
        l1[i] = l1_dev[i];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint t = thread_position_in_grid.x;
    uint assignment = t / {lanes}u;
    uint lane = t % {lanes}u;
    uint expert = uint(indices[assignment]);
    uint widx = directory[expert * {dir_stride}u + {dir_base}u + lane];
    ulong buf = 0;
    uint have = 0;
    uint produced = 0;
    ulong out_base = ulong(assignment) * {seg_len}u;
    uint lane_base = lane * {lane_bytes}u;
    while (produced < {lane_bytes}u) {{
        if (have < 32u) {{
            buf |= ulong(payload[widx]) << (32u - have);
            widx += 1u;
            have += 32u;
        }}
        uint peek = uint(buf >> {peek_shift}u);
        uint entry = l1[peek];
        uint count = (entry >> 5) & 0x3u;
        uint p0 = lane_base + produced;
        {emit0}
        if (count == 2u && produced + 1u < {lane_bytes}u) {{
            uint p1 = p0 + 1u;
            {emit1}
            produced += 2u;
            uint n = entry & 0x1Fu;
            buf <<= n;
            have -= n;
        }} else {{
            uint n0 = (count == 2u)
                ? sym_len_dev[(entry >> 8) & 0xFFu]
                : (entry & 0x1Fu);
            buf <<= n0;
            have -= n0;
            produced += 1u;
        }}
    }}
"""

_EMIT_IDENTITY = "out[out_base + {pos}] = uchar((entry >> {shift}) & 0xFFu);"
_EMIT_PLANE = (
    "{{ uint g = {pos} >> 7; uint r = {pos} & 127u; "
    "uint raw = (g << 7) + ((r < 64u) ? (r << 1) : (((r - 64u) << 1) + 1u)); "
    "out[out_base + raw] = uchar((entry >> {shift}) & 0xFFu); }}"
)


@lru_cache(maxsize=None)
def _kernel(lanes: int, lane_bytes: int, seg_len: int, dir_stride: int,
            dir_base: int, plane: bool):
    emit = _EMIT_PLANE if plane else _EMIT_IDENTITY
    source = _BODY_TEMPLATE.format(
        l1_size=1 << L1_BITS,
        lanes=lanes,
        lane_bytes=lane_bytes,
        seg_len=seg_len,
        dir_stride=dir_stride,
        dir_base=dir_base,
        peek_shift=64 - L1_BITS,
        emit0=emit.format(pos="p0", shift="8"),
        emit1=emit.format(pos="p1", shift="16"),
    )
    return mx.fast.metal_kernel(
        name=(
            f"huff_seg_{lanes}_{lane_bytes}_{dir_stride}_{dir_base}"
            f"_{'p' if plane else 'i'}"
        ),
        input_names=["payload", "directory", "indices", "l1_dev", "sym_len_dev"],
        output_names=["out"],
        header=_HEADER,
        source=source,
    )


def build_class_tables(lengths_by_class: dict) -> dict:
    """Manifest code lengths -> {class: (l1 mx.array, sym_len mx.array)}."""

    out = {}
    for kind, lengths_list in lengths_by_class.items():
        lengths = np.asarray(lengths_list, dtype=np.uint8)
        table = HuffmanTable(
            lengths=lengths, codes=_canonical_codes(lengths), max_bits=L1_BITS
        )
        size = 1 << L1_BITS
        first_sym = np.zeros(size, dtype=np.int64)
        first_len = np.zeros(size, dtype=np.int64)
        for sym in range(256):
            length = int(lengths[sym])
            base = int(table.codes[sym]) << (L1_BITS - length)
            span = 1 << (L1_BITS - length)
            first_sym[base : base + span] = sym
            first_len[base : base + span] = length
        entries = np.zeros(size, dtype=np.uint32)
        for peek in range(size):
            s0 = int(first_sym[peek])
            n0 = int(first_len[peek])
            rem = L1_BITS - n0
            entry = n0 | (1 << 5) | (s0 << 8)
            if rem > 0:
                tail = (peek & ((1 << rem) - 1)) << (L1_BITS - rem)
                if int(first_len[tail]) <= rem:
                    entry = (
                        (n0 + int(first_len[tail]))
                        | (2 << 5)
                        | (s0 << 8)
                        | (int(first_sym[tail]) << 16)
                    )
            entries[peek] = entry
        out[kind] = (
            mx.array(entries),
            mx.array(lengths.astype(np.uint32)),
        )
    return out


def decode_component(
    payload: mx.array,
    directory: mx.array,
    indices: mx.array,
    l1: mx.array,
    sym_len: mx.array,
    *,
    lanes: int,
    lane_bytes: int,
    seg_len: int,
    dir_stride: int,
    dir_base: int,
    plane: bool,
    assignments: int,
) -> mx.array:
    """Decode one component's raw bytes for every routed assignment."""

    kernel = _kernel(lanes, lane_bytes, seg_len, dir_stride, dir_base, plane)
    total_threads = assignments * lanes
    (out,) = kernel(
        inputs=[payload, directory, indices, l1, sym_len],
        grid=(total_threads, 1, 1),
        threadgroup=(min(256, total_threads), 1, 1),
        output_shapes=[(assignments * seg_len,)],
        output_dtypes=[mx.uint8],
    )
    return out
