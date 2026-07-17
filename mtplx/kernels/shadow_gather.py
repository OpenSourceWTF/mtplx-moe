"""Shadow-bank gather matmul (issue #51 miss fallback).

``shadow_gather_mm`` runs one projection of the expert MLP against a
dense low-precision shadow bank: for each routed assignment (hidden row,
expert id) it computes ``W_e @ x`` where ``W_e`` is the expert's shadow
row block. Row index into the bank is the expert id (island-bank
convention). Decode row counts are tiny (1-8), so the kernel gives each
output element one thread that walks the g64 groups, unpacking sign bits
(``b1``) or base-3 trits (``t158``) and accumulating in fp32.

Scales are stored as bf16 bit patterns in u16 and widened in-kernel via
``as_type`` so the CPU (numpy) and GPU decode paths share one
representation.

Eager-only: ``mx.fast.metal_kernel`` is not traceable under
``mx.compile`` — callers must keep the shadow path out of compiled
graphs (the helper materializes its output with ``mx.eval``).
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from mtplx.expert_shadow import (
    SHADOW_GROUP,
    _B1_WORDS_PER_GROUP,
    _T158_BYTES_PER_GROUP,
    ShadowCodecError,
)

_DTYPE_TAG = {mx.bfloat16: "bf16", mx.float16: "fp16", mx.float32: "fp32"}

_HEADER = """
using namespace metal;
constant constexpr uint SHADOW_GROUP = 64;

inline float shadow_scale(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}
"""

_B1_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint rows = uint(row_count);
    uint out_dim = uint(out_size);
    uint in_dim = uint(in_size);
    uint total = rows * out_dim;
    if (tid >= total) {
        return;
    }
    uint row = tid / out_dim;
    uint out_index = tid % out_dim;
    uint expert = uint(ids[row]);
    uint groups = in_dim / SHADOW_GROUP;
    const device uint* w = packed + (size_t(expert) * out_dim + out_index) * (groups * 2);
    const device ushort* s = scales + (size_t(expert) * out_dim + out_index) * groups;
    const device T* xr = x + size_t(row) * in_dim;
    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float dot = 0.0f;
        for (uint word = 0; word < 2; ++word) {
            uint sign_bits = w[g * 2 + word];
            uint base = g * SHADOW_GROUP + word * 32;
            for (uint bit = 0; bit < 32; ++bit) {
                float value = float(xr[base + bit]);
                dot += ((sign_bits >> bit) & 1u) ? value : -value;
            }
        }
        acc += shadow_scale(s[g]) * dot;
    }
    out[size_t(row) * out_dim + out_index] = static_cast<T>(acc);
"""

_T158_SOURCE = """
    uint tid = thread_position_in_grid.x;
    uint rows = uint(row_count);
    uint out_dim = uint(out_size);
    uint in_dim = uint(in_size);
    uint total = rows * out_dim;
    if (tid >= total) {
        return;
    }
    uint row = tid / out_dim;
    uint out_index = tid % out_dim;
    uint expert = uint(ids[row]);
    uint groups = in_dim / SHADOW_GROUP;
    const device uchar* w = packed + (size_t(expert) * out_dim + out_index) * (groups * 13);
    const device ushort* s = scales + (size_t(expert) * out_dim + out_index) * groups;
    const device T* xr = x + size_t(row) * in_dim;
    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float dot = 0.0f;
        for (uint byte_index = 0; byte_index < 13; ++byte_index) {
            uint trits = uint(w[g * 13 + byte_index]);
            uint slot = byte_index * 5;
            for (uint lane = 0; lane < 5; ++lane) {
                uint trit = trits % 3u;
                trits /= 3u;
                uint element = slot + lane;
                if (element < SHADOW_GROUP) {
                    dot += (float(trit) - 1.0f) * float(xr[g * SHADOW_GROUP + element]);
                }
            }
        }
        acc += shadow_scale(s[g]) * dot;
    }
    out[size_t(row) * out_dim + out_index] = static_cast<T>(acc);
"""


@lru_cache(maxsize=None)
def _shadow_gather_kernel(codec: str, dtype: mx.Dtype):
    if codec == "b1":
        source = _B1_SOURCE
    elif codec == "t158":
        source = _T158_SOURCE
    else:  # pragma: no cover - guarded by callers
        raise ShadowCodecError(f"unknown shadow codec {codec!r}")
    dtype_tag = _DTYPE_TAG.get(dtype, "generic")
    return mx.fast.metal_kernel(
        name=f"mtplx_shadow_gather_{codec}_{dtype_tag}",
        input_names=["x", "ids", "packed", "scales", "row_count", "out_size", "in_size"],
        output_names=["out"],
        header=_HEADER,
        source=source,
    )


def shadow_gather_mm(
    x: mx.array,
    expert_ids: mx.array,
    packed: mx.array,
    scales: mx.array,
    *,
    codec: str,
    threadgroup_size: int = 64,
) -> mx.array:
    """Gather-matmul one projection from a shadow bank.

    ``x``: (rows, in) hidden rows, one per assignment. ``expert_ids``:
    (rows,) int32 bank rows. ``packed``: (experts, out, words) u32 for
    ``b1`` / (experts, out, bytes) u8 for ``t158``. ``scales``:
    (experts, out, in/64) u16 bf16 bits. Returns (rows, out) in
    ``x.dtype`` (fp32 accumulate).
    """

    if x.ndim != 2:
        raise ShadowCodecError(f"shadow_gather_mm expects 2-D x, got {x.shape}")
    if packed.ndim != 3 or scales.ndim != 3:
        raise ShadowCodecError("shadow bank arrays must be (experts, out, packed)")
    rows, in_dim = int(x.shape[0]), int(x.shape[1])
    out_dim = int(packed.shape[1])
    groups = in_dim // SHADOW_GROUP
    if in_dim % SHADOW_GROUP:
        raise ShadowCodecError(
            f"shadow input dim {in_dim} is not a multiple of {SHADOW_GROUP}"
        )
    if int(scales.shape[1]) != out_dim or int(scales.shape[2]) != groups:
        raise ShadowCodecError(
            f"shadow scales shape {tuple(scales.shape)} does not match "
            f"out={out_dim} groups={groups}"
        )
    expected_words = groups * (
        _B1_WORDS_PER_GROUP if codec == "b1" else _T158_BYTES_PER_GROUP
    )
    if int(packed.shape[2]) != expected_words:
        raise ShadowCodecError(
            f"shadow packed shape {tuple(packed.shape)} does not match "
            f"codec {codec!r} groups={groups}"
        )
    total = rows * out_dim
    if total <= 0:
        raise ShadowCodecError("shadow_gather_mm requires at least one assignment")
    if total >= 2**31:  # metal_kernel output element counts must stay < 2^31
        raise ShadowCodecError(f"shadow gather grid too large: {total}")
    kernel = _shadow_gather_kernel(codec, x.dtype)
    grid_x = -(-total // threadgroup_size) * threadgroup_size
    (out,) = kernel(
        inputs=[
            mx.contiguous(x),
            expert_ids.astype(mx.int32),
            packed,
            scales,
            int(rows),
            int(out_dim),
            int(in_dim),
        ],
        template=[("T", x.dtype)],
        grid=(grid_x, 1, 1),
        threadgroup=(int(threadgroup_size), 1, 1),
        output_shapes=[(rows, out_dim)],
        output_dtypes=[x.dtype],
    )
    return out
