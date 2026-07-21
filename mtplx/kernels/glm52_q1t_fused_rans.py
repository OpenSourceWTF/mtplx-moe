"""Direct rANS32x -> t158 gather matmul for the GLM-5.2 Q1T lane."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx

from mtplx.expert_rans import LANES, M, RANS_L, SCALE_BITS
from mtplx.expert_shadow import SHADOW_GROUP
from mtplx.kernels.shadow_gather import _T158_LUT


_DTYPE_TAG = {mx.bfloat16: "bf16", mx.float16: "fp16", mx.float32: "fp32"}


class Glm52Q1TFusedRansError(ValueError):
    """Raised when the fixed fused-rANS projection geometry is invalid."""


@dataclass(frozen=True)
class BoundGlm52Q1TFusedRansProjection:
    """Construction-qualified direct projection dispatch."""

    kernel: Callable[..., tuple[mx.array]]
    packed_payload: mx.array
    packed_directory: mx.array
    packed_cum2sym: mx.array
    packed_freq: mx.array
    packed_cum: mx.array
    packed_payload_offsets: mx.array
    scales_payload: mx.array
    scales_directory: mx.array
    scales_cum2sym: mx.array
    scales_freq: mx.array
    scales_cum: mx.array
    scales_payload_offsets: mx.array
    expert_base: mx.array
    in_dim: int
    out_dim: int
    threads_per_tg: int
    input_repeat: int
    dtype: mx.Dtype

    def __call__(self, x: mx.array, expert_ids: mx.array) -> mx.array:
        assignments = int(expert_ids.shape[0])
        (out,) = self.kernel(
            inputs=[
                x,
                expert_ids,
                self.packed_payload,
                self.packed_directory,
                self.packed_cum2sym,
                self.packed_freq,
                self.packed_cum,
                self.packed_payload_offsets,
                self.scales_payload,
                self.scales_directory,
                self.scales_cum2sym,
                self.scales_freq,
                self.scales_cum,
                self.scales_payload_offsets,
                self.expert_base,
            ],
            template=[("T", self.dtype)],
            grid=(assignments * self.out_dim, 1, 1),
            threadgroup=(self.threads_per_tg, 1, 1),
            output_shapes=[(assignments, self.out_dim)],
            output_dtypes=[self.dtype],
        )
        return out


def _header(uniform_packed_rans: bool = False) -> str:
    generic = f"""
#include <metal_stdlib>
using namespace metal;

constant float T158_LUT[1215] = {{ {_T158_LUT} }};

inline float shadow_scale(ushort bits) {{
    return as_type<float>(uint(bits) << 16);
}}

inline uchar rans_next(
    thread uint& state,
    thread uint& position,
    const device uchar* payload,
    const device uchar* cum2sym,
    const device uint* freq,
    const device uint* cum
) {{
    uint slot = state & {M - 1}u;
    uint symbol = uint(cum2sym[slot]);
    state = freq[symbol] * (state >> {SCALE_BITS}u) + slot - cum[symbol];
    while (state < {RANS_L}u) {{
        state = (state << 8) | uint(payload[position]);
        position += 1u;
    }}
    return uchar(symbol);
}}

inline uchar rans_next_transition(
    thread uint& state,
    thread uint& position,
    const device uchar* payload,
    const device uint* transition
) {{
    uint entry = transition[state & {M - 1}u];
    uint symbol = entry & 255u;
    uint frequency = ((entry >> 8u) & {M - 1}u) + 1u;
    uint residue = entry >> 20u;
    state = frequency * (state >> {SCALE_BITS}u) + residue;
    while (state < {RANS_L}u) {{
        state = (state << 8) | uint(payload[position]);
        position += 1u;
    }}
    return uchar(symbol);
}}
"""
    if not uniform_packed_rans:
        return generic
    return (
        generic
        + f"""

inline uchar rans_next_uniform_packed(
    thread uint& state,
    thread uint& position,
    const device uchar* payload
) {{
    uint symbol = (state & {M - 1}u) >> 4u;
    uint reduced = ((state >> 8u) & ~15u) | (state & 15u);
    state = (reduced << 8u) | uint(payload[position]);
    position += 1u;
    return uchar(symbol);
}}
"""
    )


def _source(
    in_dim: int,
    out_dim: int,
    threads_per_tg: int,
    uniform_packed_rans: bool = False,
    input_repeat: int = 1,
) -> str:
    groups = in_dim // SHADOW_GROUP
    tiles = out_dim // LANES
    body: list[str] = []
    packed_next = (
        "rans_next_uniform_packed(packed_state, packed_pos, packed_payload)"
        if uniform_packed_rans
        else (
            "rans_next(packed_state, packed_pos, packed_payload, "
            "packed_cum2sym, packed_freq, packed_cum)"
        )
    )
    for byte_index in range(12):
        body.append(
            f"        {{ uint bv = uint({packed_next}) * 5u; "
            f"uint k = base + {byte_index * 5}u;"
        )
        for slot in range(5):
            body.append(
                f"          gd = fma(T158_LUT[bv + {slot}u], "
                f"float(x[size_t(input_row) * {in_dim}u + k + {slot}u]), gd);"
            )
        body.append("        }")
    body.append(f"        {{ uint bv = uint({packed_next}) * 5u; uint k = base + 60u;")
    for slot in range(4):
        body.append(
            f"          gd = fma(T158_LUT[bv + {slot}u], "
            f"float(x[size_t(input_row) * {in_dim}u + k + {slot}u]), gd);"
        )
    body.append("        }")
    inner = "\n".join(body)
    return f"""
    uint linear = thread_position_in_grid.x;
    uint assignment = linear / {out_dim}u;
    uint input_row = assignment / {input_repeat}u;
    uint output = linear - assignment * {out_dim}u;
    uint lane = output % {LANES}u;
    uint tile = output / {LANES}u;
    uint expert = uint(expert_ids[assignment]) - uint(expert_base[0]);
    uint directory_index = (expert * {tiles}u + tile) * {LANES}u + lane;
    uint packed_pos = packed_directory[directory_index]
        - packed_payload_offsets[expert];
    uint scales_pos = scales_directory[directory_index]
        - scales_payload_offsets[expert];
    uint packed_state = uint(packed_payload[packed_pos])
        | (uint(packed_payload[packed_pos + 1u]) << 8)
        | (uint(packed_payload[packed_pos + 2u]) << 16)
        | (uint(packed_payload[packed_pos + 3u]) << 24);
    uint scales_state = uint(scales_payload[scales_pos])
        | (uint(scales_payload[scales_pos + 1u]) << 8)
        | (uint(scales_payload[scales_pos + 2u]) << 16)
        | (uint(scales_payload[scales_pos + 3u]) << 24);
    packed_pos += 4u;
    scales_pos += 4u;

    float acc = 0.0f;
    for (uint group = 0u; group < {groups}u; ++group) {{
        uint scale_lo = uint(rans_next(
            scales_state, scales_pos, scales_payload, scales_cum2sym,
            scales_freq, scales_cum));
        uint scale_hi = uint(rans_next(
            scales_state, scales_pos, scales_payload, scales_cum2sym,
            scales_freq, scales_cum));
        float scale = shadow_scale(ushort(scale_lo | (scale_hi << 8)));
        float gd = 0.0f;
        uint base = group * {SHADOW_GROUP}u;
{inner}
        acc += scale * gd;
    }}
    out[size_t(assignment) * {out_dim}u + output] = static_cast<T>(acc);
"""


@lru_cache(maxsize=None)
def _kernel(
    dtype: mx.Dtype,
    in_dim: int,
    out_dim: int,
    threads_per_tg: int,
    uniform_packed_rans: bool,
    input_repeat: int,
):
    model_tag = "uniform_packed" if uniform_packed_rans else "component_table"
    return mx.fast.metal_kernel(
        name=(
            "mtplx_glm52_q1t_fused_rans_t158_"
            f"{model_tag}_{_DTYPE_TAG[dtype]}_k{in_dim}_n{out_dim}_"
            f"t{threads_per_tg}_r{input_repeat}"
        ),
        input_names=[
            "x",
            "expert_ids",
            "packed_payload",
            "packed_directory",
            "packed_cum2sym",
            "packed_freq",
            "packed_cum",
            "packed_payload_offsets",
            "scales_payload",
            "scales_directory",
            "scales_cum2sym",
            "scales_freq",
            "scales_cum",
            "scales_payload_offsets",
            "expert_base",
        ],
        output_names=["out"],
        header=_header(uniform_packed_rans),
        source=_source(
            in_dim,
            out_dim,
            threads_per_tg,
            uniform_packed_rans=uniform_packed_rans,
            input_repeat=input_repeat,
        ),
    )


def _expert_mlp_input_names() -> list[str]:
    names = ["x", "expert_ids"]
    for projection in ("gate", "up", "down"):
        names.extend(
            (
                f"{projection}_packed_payload",
                f"{projection}_packed_directory",
                f"{projection}_packed_payload_offsets",
                f"{projection}_scales_payload",
                f"{projection}_scales_directory",
                f"{projection}_scales_payload_offsets",
                f"{projection}_scales_cum2sym",
                f"{projection}_scales_freq",
                f"{projection}_scales_cum",
            )
        )
    names.append("expert_base")
    return names


def _expert_mlp_source(
    hidden_size: int,
    expert_hidden_size: int,
    threads_per_tg: int,
) -> str:
    gate_groups = hidden_size // SHADOW_GROUP
    down_groups = expert_hidden_size // SHADOW_GROUP
    return f"""
    uint assignment = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint expert = uint(expert_ids[assignment]) - uint(expert_base[0]);
    threadgroup T activated[{expert_hidden_size}];

    for (uint output = tid; output < {expert_hidden_size}u;
         output += {threads_per_tg}u) {{
        uint directory_index = expert * {expert_hidden_size}u + output;
        uint gate_packed_pos = gate_packed_directory[directory_index]
            - gate_packed_payload_offsets[expert];
        uint gate_scales_pos = gate_scales_directory[directory_index]
            - gate_scales_payload_offsets[expert];
        uint up_packed_pos = up_packed_directory[directory_index]
            - up_packed_payload_offsets[expert];
        uint up_scales_pos = up_scales_directory[directory_index]
            - up_scales_payload_offsets[expert];
        uint gate_packed_state = uint(gate_packed_payload[gate_packed_pos])
            | (uint(gate_packed_payload[gate_packed_pos + 1u]) << 8)
            | (uint(gate_packed_payload[gate_packed_pos + 2u]) << 16)
            | (uint(gate_packed_payload[gate_packed_pos + 3u]) << 24);
        uint gate_scales_state = uint(gate_scales_payload[gate_scales_pos])
            | (uint(gate_scales_payload[gate_scales_pos + 1u]) << 8)
            | (uint(gate_scales_payload[gate_scales_pos + 2u]) << 16)
            | (uint(gate_scales_payload[gate_scales_pos + 3u]) << 24);
        uint up_packed_state = uint(up_packed_payload[up_packed_pos])
            | (uint(up_packed_payload[up_packed_pos + 1u]) << 8)
            | (uint(up_packed_payload[up_packed_pos + 2u]) << 16)
            | (uint(up_packed_payload[up_packed_pos + 3u]) << 24);
        uint up_scales_state = uint(up_scales_payload[up_scales_pos])
            | (uint(up_scales_payload[up_scales_pos + 1u]) << 8)
            | (uint(up_scales_payload[up_scales_pos + 2u]) << 16)
            | (uint(up_scales_payload[up_scales_pos + 3u]) << 24);
        gate_packed_pos += 4u;
        gate_scales_pos += 4u;
        up_packed_pos += 4u;
        up_scales_pos += 4u;
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (uint group = 0u; group < {gate_groups}u; ++group) {{
            uint gate_scale_lo = uint(rans_next(
                gate_scales_state, gate_scales_pos, gate_scales_payload,
                gate_scales_cum2sym, gate_scales_freq, gate_scales_cum));
            uint gate_scale_hi = uint(rans_next(
                gate_scales_state, gate_scales_pos, gate_scales_payload,
                gate_scales_cum2sym, gate_scales_freq, gate_scales_cum));
            uint up_scale_lo = uint(rans_next(
                up_scales_state, up_scales_pos, up_scales_payload,
                up_scales_cum2sym, up_scales_freq, up_scales_cum));
            uint up_scale_hi = uint(rans_next(
                up_scales_state, up_scales_pos, up_scales_payload,
                up_scales_cum2sym, up_scales_freq, up_scales_cum));
            float gate_scale = shadow_scale(
                ushort(gate_scale_lo | (gate_scale_hi << 8)));
            float up_scale = shadow_scale(
                ushort(up_scale_lo | (up_scale_hi << 8)));
            float gate_gd = 0.0f;
            float up_gd = 0.0f;
            uint base = group * {SHADOW_GROUP}u;
            for (uint byte_index = 0u; byte_index < 13u; ++byte_index) {{
                uint gate_bv = uint(rans_next_uniform_packed(
                    gate_packed_state, gate_packed_pos,
                    gate_packed_payload)) * 5u;
                uint up_bv = uint(rans_next_uniform_packed(
                    up_packed_state, up_packed_pos,
                    up_packed_payload)) * 5u;
                uint count = byte_index == 12u ? 4u : 5u;
                uint k = base + byte_index * 5u;
                for (uint slot = 0u; slot < count; ++slot) {{
                    float value = float(x[size_t(assignment) * {hidden_size}u + k + slot]);
                    gate_gd = fma(T158_LUT[gate_bv + slot], value, gate_gd);
                    up_gd = fma(T158_LUT[up_bv + slot], value, up_gd);
                }}
            }}
            gate_acc += gate_scale * gate_gd;
            up_acc += up_scale * up_gd;
        }}
        T gate_value = static_cast<T>(gate_acc);
        T up_value = static_cast<T>(up_acc);
        auto sigmoid = 1 / (1 + metal::exp(metal::abs(gate_value)));
        sigmoid = (gate_value < T(0)) ? sigmoid : 1 - sigmoid;
        T silu = gate_value * sigmoid;
        activated[output] = T(silu * up_value);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint output = tid; output < {hidden_size}u;
         output += {threads_per_tg}u) {{
        uint directory_index = expert * {hidden_size}u + output;
        uint packed_pos = down_packed_directory[directory_index]
            - down_packed_payload_offsets[expert];
        uint scales_pos = down_scales_directory[directory_index]
            - down_scales_payload_offsets[expert];
        uint packed_state = uint(down_packed_payload[packed_pos])
            | (uint(down_packed_payload[packed_pos + 1u]) << 8)
            | (uint(down_packed_payload[packed_pos + 2u]) << 16)
            | (uint(down_packed_payload[packed_pos + 3u]) << 24);
        uint scales_state = uint(down_scales_payload[scales_pos])
            | (uint(down_scales_payload[scales_pos + 1u]) << 8)
            | (uint(down_scales_payload[scales_pos + 2u]) << 16)
            | (uint(down_scales_payload[scales_pos + 3u]) << 24);
        packed_pos += 4u;
        scales_pos += 4u;
        float acc = 0.0f;
        for (uint group = 0u; group < {down_groups}u; ++group) {{
            uint scale_lo = uint(rans_next(
                scales_state, scales_pos, down_scales_payload,
                down_scales_cum2sym, down_scales_freq, down_scales_cum));
            uint scale_hi = uint(rans_next(
                scales_state, scales_pos, down_scales_payload,
                down_scales_cum2sym, down_scales_freq, down_scales_cum));
            float scale = shadow_scale(ushort(scale_lo | (scale_hi << 8)));
            float gd = 0.0f;
            uint base = group * {SHADOW_GROUP}u;
            for (uint byte_index = 0u; byte_index < 13u; ++byte_index) {{
                uint bv = uint(rans_next_uniform_packed(
                    packed_state, packed_pos, down_packed_payload)) * 5u;
                uint count = byte_index == 12u ? 4u : 5u;
                uint k = base + byte_index * 5u;
                for (uint slot = 0u; slot < count; ++slot) {{
                    gd = fma(
                        T158_LUT[bv + slot],
                        float(activated[k + slot]),
                        gd);
                }}
            }}
            acc += scale * gd;
        }}
        out[size_t(assignment) * {hidden_size}u + output]
            = static_cast<T>(acc);
    }}
"""


@lru_cache(maxsize=None)
def _expert_mlp_kernel(
    dtype: mx.Dtype,
    hidden_size: int,
    expert_hidden_size: int,
    threads_per_tg: int,
):
    return mx.fast.metal_kernel(
        name=(
            "mtplx_glm52_q1t_fused_rans_expert_mlp_"
            f"{_DTYPE_TAG[dtype]}_h{hidden_size}_e{expert_hidden_size}_"
            f"t{threads_per_tg}"
        ),
        input_names=_expert_mlp_input_names(),
        output_names=["out"],
        header=_header(True),
        source=_expert_mlp_source(
            hidden_size,
            expert_hidden_size,
            threads_per_tg,
        ),
    )


@dataclass(frozen=True)
class BoundGlm52Q1TFusedRansExpertMlp:
    """One construction-bound direct rANS gate/up/SwiGLU/down dispatch."""

    kernel: Callable[..., tuple[mx.array]]
    component_inputs: tuple[mx.array, ...]
    expert_base: mx.array
    hidden_size: int
    threads_per_tg: int
    dtype: mx.Dtype
    output_count: int = 1

    def __call__(self, x: mx.array, expert_ids: mx.array) -> mx.array:
        assignments = int(expert_ids.shape[0])
        (out,) = self.kernel(
            inputs=[x, expert_ids, *self.component_inputs, self.expert_base],
            template=[("T", self.dtype)],
            grid=(assignments * self.threads_per_tg, 1, 1),
            threadgroup=(self.threads_per_tg, 1, 1),
            output_shapes=[(assignments, self.hidden_size)],
            output_dtypes=[self.dtype],
        )
        return out


def _expert_mlp_projection_inputs(
    projection: BoundGlm52Q1TFusedRansProjection,
) -> tuple[mx.array, ...]:
    return (
        projection.packed_payload,
        projection.packed_directory,
        projection.packed_payload_offsets,
        projection.scales_payload,
        projection.scales_directory,
        projection.scales_payload_offsets,
        projection.scales_cum2sym,
        projection.scales_freq,
        projection.scales_cum,
    )


def bind_glm52_q1t_fused_rans_expert_mlp(
    *,
    gate: BoundGlm52Q1TFusedRansProjection,
    up: BoundGlm52Q1TFusedRansProjection,
    down: BoundGlm52Q1TFusedRansProjection,
    threads_per_tg: int,
) -> BoundGlm52Q1TFusedRansExpertMlp:
    """Validate one expert's complete direct route once at construction."""

    threads_per_tg = int(threads_per_tg)
    if (
        gate.dtype != up.dtype
        or gate.dtype != down.dtype
        or gate.in_dim != up.in_dim
        or gate.out_dim != up.out_dim
        or down.in_dim != gate.out_dim
        or down.out_dim != gate.in_dim
        or gate.input_repeat != 1
        or up.input_repeat != 1
        or down.input_repeat != 1
        or threads_per_tg not in (32, 64, 128, 256, 512, 1024)
    ):
        raise Glm52Q1TFusedRansError("expert MLP projection geometry is incompatible")
    return BoundGlm52Q1TFusedRansExpertMlp(
        kernel=_expert_mlp_kernel(
            gate.dtype,
            gate.in_dim,
            gate.out_dim,
            threads_per_tg,
        ),
        component_inputs=(
            *_expert_mlp_projection_inputs(gate),
            *_expert_mlp_projection_inputs(up),
            *_expert_mlp_projection_inputs(down),
        ),
        expert_base=gate.expert_base,
        hidden_size=gate.in_dim,
        threads_per_tg=threads_per_tg,
        dtype=gate.dtype,
    )


def _expert_gate_up_source(
    hidden_size: int,
    expert_hidden_size: int,
    threads_per_tg: int,
) -> str:
    source = _expert_mlp_source(
        hidden_size,
        expert_hidden_size,
        threads_per_tg,
    ).split("    threadgroup_barrier", 1)[0]
    old_preamble = f"""
    uint assignment = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint expert = uint(expert_ids[assignment]) - uint(expert_base[0]);
    threadgroup T activated[{expert_hidden_size}];

    for (uint output = tid; output < {expert_hidden_size}u;
         output += {threads_per_tg}u) {{
"""
    new_preamble = f"""
    uint linear = thread_position_in_grid.x;
    uint assignment = linear / {expert_hidden_size}u;
    uint output = linear - assignment * {expert_hidden_size}u;
    uint expert = uint(expert_ids[assignment]) - uint(expert_base[0]);
"""
    source = source.replace(old_preamble, new_preamble, 1)
    source = source.replace(
        "        activated[output] = T(silu * up_value);\n    }\n",
        f"        out[size_t(assignment) * {expert_hidden_size}u + output] "
        "= T(silu * up_value);\n",
        1,
    )
    return source


@lru_cache(maxsize=None)
def _expert_gate_up_kernel(
    dtype: mx.Dtype,
    hidden_size: int,
    expert_hidden_size: int,
    threads_per_tg: int,
):
    return mx.fast.metal_kernel(
        name=(
            "mtplx_glm52_q1t_fused_rans_expert_gate_up_"
            f"{_DTYPE_TAG[dtype]}_h{hidden_size}_e{expert_hidden_size}_"
            f"t{threads_per_tg}"
        ),
        input_names=[*_expert_mlp_input_names()[: 2 + 2 * 9], "expert_base"],
        output_names=["out"],
        header=_header(True),
        source=_expert_gate_up_source(
            hidden_size,
            expert_hidden_size,
            threads_per_tg,
        ),
    )


@dataclass(frozen=True)
class BoundGlm52Q1TFusedRansExpertGateUp:
    """One output-parallel direct rANS gate/up/exact-SwiGLU dispatch."""

    kernel: Callable[..., tuple[mx.array]]
    component_inputs: tuple[mx.array, ...]
    expert_base: mx.array
    out_dim: int
    threads_per_tg: int
    dtype: mx.Dtype

    def __call__(self, x: mx.array, expert_ids: mx.array) -> mx.array:
        assignments = int(expert_ids.shape[0])
        (out,) = self.kernel(
            inputs=[x, expert_ids, *self.component_inputs, self.expert_base],
            template=[("T", self.dtype)],
            grid=(assignments * self.out_dim, 1, 1),
            threadgroup=(self.threads_per_tg, 1, 1),
            output_shapes=[(assignments, self.out_dim)],
            output_dtypes=[self.dtype],
        )
        return out


def bind_glm52_q1t_fused_rans_expert_gate_up(
    *,
    gate: BoundGlm52Q1TFusedRansProjection,
    up: BoundGlm52Q1TFusedRansProjection,
    threads_per_tg: int,
) -> BoundGlm52Q1TFusedRansExpertGateUp:
    """Validate one expert's fused gate/up route once at construction."""

    threads_per_tg = int(threads_per_tg)
    if (
        gate.dtype != up.dtype
        or gate.in_dim != up.in_dim
        or gate.out_dim != up.out_dim
        or gate.input_repeat != 1
        or up.input_repeat != 1
        or threads_per_tg not in (32, 64, 128, 256, 512, 1024)
        or gate.out_dim % threads_per_tg
    ):
        raise Glm52Q1TFusedRansError(
            "expert gate/up projection geometry is incompatible"
        )
    return BoundGlm52Q1TFusedRansExpertGateUp(
        kernel=_expert_gate_up_kernel(
            gate.dtype,
            gate.in_dim,
            gate.out_dim,
            threads_per_tg,
        ),
        component_inputs=(
            *_expert_mlp_projection_inputs(gate),
            *_expert_mlp_projection_inputs(up),
        ),
        expert_base=gate.expert_base,
        out_dim=gate.out_dim,
        threads_per_tg=threads_per_tg,
        dtype=gate.dtype,
    )


def _require_array(
    value: mx.array,
    *,
    name: str,
    dtype: mx.Dtype,
    size: int | None = None,
) -> None:
    if not isinstance(value, mx.array) or value.ndim != 1 or value.dtype != dtype:
        raise Glm52Q1TFusedRansError(
            f"{name} must be a one-dimensional {dtype} MLX array"
        )
    if size is not None and int(value.size) != size:
        raise Glm52Q1TFusedRansError(
            f"{name} has {int(value.size)} elements; expected {size}"
        )


def bind_glm52_q1t_fused_rans_projection(
    *,
    packed_payload: mx.array,
    packed_directory: mx.array,
    packed_cum2sym: mx.array,
    packed_freq: mx.array,
    packed_cum: mx.array,
    packed_payload_offsets: mx.array | None = None,
    scales_payload: mx.array,
    scales_directory: mx.array,
    scales_cum2sym: mx.array,
    scales_freq: mx.array,
    scales_cum: mx.array,
    scales_payload_offsets: mx.array | None = None,
    expert_count: int,
    expert_base: int = 0,
    in_dim: int,
    out_dim: int,
    output_tile: int,
    threads_per_tg: int,
    dtype: mx.Dtype,
    uniform_packed_rans: bool = False,
    input_repeat: int = 1,
) -> BoundGlm52Q1TFusedRansProjection:
    """Validate a lane-owned component pair once and bind its direct launch."""

    expert_count = int(expert_count)
    expert_base = int(expert_base)
    in_dim = int(in_dim)
    out_dim = int(out_dim)
    output_tile = int(output_tile)
    threads_per_tg = int(threads_per_tg)
    input_repeat = int(input_repeat)
    if not isinstance(uniform_packed_rans, bool):
        raise Glm52Q1TFusedRansError("uniform_packed_rans must be a bool")
    if expert_count < 1:
        raise Glm52Q1TFusedRansError("expert_count must be positive")
    if expert_base < 0:
        raise Glm52Q1TFusedRansError("expert_base must be non-negative")
    if packed_payload_offsets is None:
        packed_payload_offsets = mx.array(
            [0] * expert_count,
            dtype=mx.uint32,
        )
    if scales_payload_offsets is None:
        scales_payload_offsets = mx.array(
            [0] * expert_count,
            dtype=mx.uint32,
        )
    if input_repeat < 1:
        raise Glm52Q1TFusedRansError("input_repeat must be positive")
    if output_tile != LANES:
        raise Glm52Q1TFusedRansError(
            f"output_tile must equal the rANS lane count {LANES}"
        )
    if in_dim < 1 or in_dim % SHADOW_GROUP:
        raise Glm52Q1TFusedRansError(
            f"in_dim must be a positive multiple of {SHADOW_GROUP}"
        )
    if out_dim < output_tile or out_dim % output_tile:
        raise Glm52Q1TFusedRansError(
            f"out_dim must be a positive multiple of {output_tile}"
        )
    if threads_per_tg not in (32, 64, 128, 256, 512, 1024) or out_dim % threads_per_tg:
        raise Glm52Q1TFusedRansError(
            "threads_per_tg must be 32, 64, 128, 256, 512, or 1024 and divide out_dim"
        )
    if dtype not in _DTYPE_TAG:
        raise Glm52Q1TFusedRansError(f"unsupported output dtype {dtype}")
    tiles = out_dim // output_tile
    directory_size = expert_count * tiles * LANES
    for name, value in (
        ("packed_payload", packed_payload),
        ("scales_payload", scales_payload),
    ):
        _require_array(value, name=name, dtype=mx.uint8)
        if int(value.size) < 4:
            raise Glm52Q1TFusedRansError(f"{name} is shorter than one rANS state")
    for name, value in (
        ("packed_directory", packed_directory),
        ("scales_directory", scales_directory),
    ):
        _require_array(value, name=name, dtype=mx.uint32, size=directory_size)
    for name, value in (
        ("packed_payload_offsets", packed_payload_offsets),
        ("scales_payload_offsets", scales_payload_offsets),
    ):
        _require_array(value, name=name, dtype=mx.uint32, size=expert_count)
    for prefix, cum2sym, freq, cum in (
        ("packed", packed_cum2sym, packed_freq, packed_cum),
        ("scales", scales_cum2sym, scales_freq, scales_cum),
    ):
        _require_array(
            cum2sym,
            name=f"{prefix}_cum2sym",
            dtype=mx.uint8,
            size=M,
        )
        _require_array(
            freq,
            name=f"{prefix}_freq",
            dtype=mx.uint32,
            size=256,
        )
        _require_array(
            cum,
            name=f"{prefix}_cum",
            dtype=mx.uint32,
            size=256,
        )
    return BoundGlm52Q1TFusedRansProjection(
        kernel=_kernel(
            dtype,
            in_dim,
            out_dim,
            threads_per_tg,
            uniform_packed_rans,
            input_repeat,
        ),
        packed_payload=packed_payload,
        packed_directory=packed_directory,
        packed_cum2sym=packed_cum2sym,
        packed_freq=packed_freq,
        packed_cum=packed_cum,
        packed_payload_offsets=packed_payload_offsets,
        scales_payload=scales_payload,
        scales_directory=scales_directory,
        scales_cum2sym=scales_cum2sym,
        scales_freq=scales_freq,
        scales_cum=scales_cum,
        scales_payload_offsets=scales_payload_offsets,
        expert_base=mx.array([expert_base], dtype=mx.uint32),
        in_dim=in_dim,
        out_dim=out_dim,
        threads_per_tg=threads_per_tg,
        input_repeat=input_repeat,
        dtype=dtype,
    )


_CACHED_EXPERT_HEADER_WORDS = 18
_CACHED_EXPERT_ROUTE_TABLE_BYTES = 256 * 4


def _cached_component_pointers(prefix: str, component_index: int) -> str:
    header = component_index * 3
    return f"""
    const device uint* {prefix}_directory =
        (const device uint*)(slot_bytes + slot_header[{header}u]);
    const device uchar* {prefix}_payload =
        slot_bytes + slot_header[{header + 1}u];
    uint {prefix}_source_payload_offset = slot_header[{header + 2}u];
"""


def _cached_gate_up_uniform_group_source(hidden_size: int) -> str:
    lines: list[str] = []
    for byte_index in range(13):
        lines.append("        {")
        if byte_index == 0:
            lines.extend(
                (
                    "            uint gate_bv = gate_packed_symbol * 5u;",
                    "            uint up_bv = up_packed_symbol * 5u;",
                )
            )
        else:
            payload_index = byte_index - 1
            lines.extend(
                (
                    "            uint gate_packed_rans_byte_"
                    f"{payload_index} = uint(gate_packed_payload["
                    f"gate_packed_pos + {payload_index}u]);",
                    "            uint up_packed_rans_byte_"
                    f"{payload_index} = uint(up_packed_payload["
                    f"up_packed_pos + {payload_index}u]);",
                    "            uint gate_bv = ((gate_packed_carry << 4u) | "
                    f"(gate_packed_rans_byte_{payload_index} >> 4u)) * 5u;",
                    "            uint up_bv = ((up_packed_carry << 4u) | "
                    f"(up_packed_rans_byte_{payload_index} >> 4u)) * 5u;",
                    "            gate_packed_carry = "
                    f"gate_packed_rans_byte_{payload_index} & 15u;",
                    "            up_packed_carry = "
                    f"up_packed_rans_byte_{payload_index} & 15u;",
                )
            )
        count = 4 if byte_index == 12 else 5
        for slot in range(count):
            offset = byte_index * 5 + slot
            lines.extend(
                (
                    f"            float value_{slot} = float("
                    "x[size_t(assignment) * "
                    f"{hidden_size}u + base + {offset}u]);",
                    "            gate_gd = fma("
                    f"T158_LUT[gate_bv + {slot}u], value_{slot}, gate_gd);",
                    "            up_gd = fma("
                    f"T158_LUT[up_bv + {slot}u], value_{slot}, up_gd);",
                )
            )
        lines.append("        }")
    lines.extend(
        (
            "        uint gate_packed_rans_byte_12 = uint("
            "gate_packed_payload[gate_packed_pos + 12u]);",
            "        uint up_packed_rans_byte_12 = uint("
            "up_packed_payload[up_packed_pos + 12u]);",
            "        gate_packed_symbol = (gate_packed_carry << 4u) | "
            "(gate_packed_rans_byte_12 >> 4u);",
            "        up_packed_symbol = (up_packed_carry << 4u) | "
            "(up_packed_rans_byte_12 >> 4u);",
            "        gate_packed_carry = gate_packed_rans_byte_12 & 15u;",
            "        up_packed_carry = up_packed_rans_byte_12 & 15u;",
            "        gate_packed_pos += 13u;",
            "        up_packed_pos += 13u;",
        )
    )
    return "\n".join(lines)


def _cached_down_uniform_group_source(expert_hidden_size: int) -> str:
    lines: list[str] = []
    for byte_index in range(13):
        lines.append("        {")
        if byte_index == 0:
            lines.append("            uint bv = packed_symbol * 5u;")
        else:
            payload_index = byte_index - 1
            lines.extend(
                (
                    "            uint packed_rans_byte_"
                    f"{payload_index} = uint(down_packed_payload["
                    f"packed_pos + {payload_index}u]);",
                    "            uint bv = ((packed_carry << 4u) | "
                    f"(packed_rans_byte_{payload_index} >> 4u)) * 5u;",
                    "            packed_carry = "
                    f"packed_rans_byte_{payload_index} & 15u;",
                )
            )
        count = 4 if byte_index == 12 else 5
        for slot in range(count):
            offset = byte_index * 5 + slot
            lines.extend(
                (
                    f"            gd = fma(T158_LUT[bv + {slot}u],",
                    "                float(x[size_t(assignment) * "
                    f"{expert_hidden_size}u + base + {offset}u]), gd);",
                )
            )
        lines.append("        }")
    lines.extend(
        (
            "        uint packed_rans_byte_12 = uint("
            "down_packed_payload[packed_pos + 12u]);",
            "        packed_symbol = (packed_carry << 4u) | "
            "(packed_rans_byte_12 >> 4u);",
            "        packed_carry = packed_rans_byte_12 & 15u;",
            "        packed_pos += 13u;",
        )
    )
    return "\n".join(lines)


def _cached_gate_up_source(
    hidden_size: int,
    expert_hidden_size: int,
    *,
    bank_slot_bytes: int | None = None,
    persistent_slots: int | None = None,
) -> str:
    groups = hidden_size // SHADOW_GROUP
    uniform_group = _cached_gate_up_uniform_group_source(hidden_size)
    pointers = "".join(
        (
            _cached_component_pointers("gate_packed", 0),
            _cached_component_pointers("gate_scales", 1),
            _cached_component_pointers("up_packed", 2),
            _cached_component_pointers("up_scales", 3),
        )
    )
    if bank_slot_bytes is None:
        slot_binding = """
    const device uint* slot_header = (const device uint*)slot_bytes;
"""
    else:
        slot_binding = f"""
    uint expert = uint(expert_ids[assignment]);
    const device uint* expert_slots = (const device uint*)(
        persistent_bytes
        + size_t({persistent_slots}u) * {bank_slot_bytes}u
        - {_CACHED_EXPERT_ROUTE_TABLE_BYTES}u);
    uint logical_slot = uint(expert_slots[expert]);
    const device uchar* slot_bytes = logical_slot < {persistent_slots}u
        ? persistent_bytes + size_t(logical_slot) * {bank_slot_bytes}u
        : transient_bytes
            + size_t(logical_slot - {persistent_slots}u) * {bank_slot_bytes}u;
    const device uint* slot_header = (const device uint*)slot_bytes;
"""
    return f"""
    uint linear = thread_position_in_grid.x;
    uint assignment = linear / {expert_hidden_size}u;
    uint output = linear - assignment * {expert_hidden_size}u;
{slot_binding}
{pointers}
    uint gate_packed_pos = gate_packed_directory[output]
        - gate_packed_source_payload_offset;
    uint gate_scales_pos = gate_scales_directory[output]
        - gate_scales_source_payload_offset;
    uint up_packed_pos = up_packed_directory[output]
        - up_packed_source_payload_offset;
    uint up_scales_pos = up_scales_directory[output]
        - up_scales_source_payload_offset;
    uint gate_packed_initial = uint(gate_packed_payload[gate_packed_pos])
        | (uint(gate_packed_payload[gate_packed_pos + 1u]) << 8)
        | (uint(gate_packed_payload[gate_packed_pos + 2u]) << 16)
        | (uint(gate_packed_payload[gate_packed_pos + 3u]) << 24);
    uint gate_scales_state = uint(gate_scales_payload[gate_scales_pos])
        | (uint(gate_scales_payload[gate_scales_pos + 1u]) << 8)
        | (uint(gate_scales_payload[gate_scales_pos + 2u]) << 16)
        | (uint(gate_scales_payload[gate_scales_pos + 3u]) << 24);
    uint up_packed_initial = uint(up_packed_payload[up_packed_pos])
        | (uint(up_packed_payload[up_packed_pos + 1u]) << 8)
        | (uint(up_packed_payload[up_packed_pos + 2u]) << 16)
        | (uint(up_packed_payload[up_packed_pos + 3u]) << 24);
    uint up_scales_state = uint(up_scales_payload[up_scales_pos])
        | (uint(up_scales_payload[up_scales_pos + 1u]) << 8)
        | (uint(up_scales_payload[up_scales_pos + 2u]) << 16)
        | (uint(up_scales_payload[up_scales_pos + 3u]) << 24);
    uint gate_packed_symbol = (gate_packed_initial & {M - 1}u) >> 4u;
    uint gate_packed_carry = gate_packed_initial & 15u;
    uint up_packed_symbol = (up_packed_initial & {M - 1}u) >> 4u;
    uint up_packed_carry = up_packed_initial & 15u;
    gate_packed_pos += 4u;
    gate_scales_pos += 4u;
    up_packed_pos += 4u;
    up_scales_pos += 4u;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint group = 0u; group < {groups}u; ++group) {{
        uint gate_scale_lo = uint(rans_next_transition(
            gate_scales_state, gate_scales_pos, gate_scales_payload,
            gate_scales_transition));
        uint gate_scale_hi = uint(rans_next_transition(
            gate_scales_state, gate_scales_pos, gate_scales_payload,
            gate_scales_transition));
        uint up_scale_lo = uint(rans_next_transition(
            up_scales_state, up_scales_pos, up_scales_payload,
            up_scales_transition));
        uint up_scale_hi = uint(rans_next_transition(
            up_scales_state, up_scales_pos, up_scales_payload,
            up_scales_transition));
        float gate_scale = shadow_scale(
            ushort(gate_scale_lo | (gate_scale_hi << 8)));
        float up_scale = shadow_scale(
            ushort(up_scale_lo | (up_scale_hi << 8)));
        float gate_gd = 0.0f;
        float up_gd = 0.0f;
        uint base = group * {SHADOW_GROUP}u;
{uniform_group}
        gate_acc += gate_scale * gate_gd;
        up_acc += up_scale * up_gd;
    }}
    T gate_value = static_cast<T>(gate_acc);
    T up_value = static_cast<T>(up_acc);
    auto sigmoid = 1 / (1 + metal::exp(metal::abs(gate_value)));
    sigmoid = (gate_value < T(0)) ? sigmoid : 1 - sigmoid;
    T silu = gate_value * sigmoid;
    out[size_t(assignment) * {expert_hidden_size}u + output]
        = T(silu * up_value);
"""


def _cached_down_source(
    hidden_size: int,
    expert_hidden_size: int,
    *,
    bank_slot_bytes: int | None = None,
    persistent_slots: int | None = None,
) -> str:
    groups = expert_hidden_size // SHADOW_GROUP
    uniform_group = _cached_down_uniform_group_source(expert_hidden_size)
    pointers = "".join(
        (
            _cached_component_pointers("down_packed", 4),
            _cached_component_pointers("down_scales", 5),
        )
    )
    if bank_slot_bytes is None:
        slot_binding = """
    const device uint* slot_header = (const device uint*)slot_bytes;
"""
    else:
        slot_binding = f"""
    uint expert = uint(expert_ids[assignment]);
    const device uint* expert_slots = (const device uint*)(
        persistent_bytes
        + size_t({persistent_slots}u) * {bank_slot_bytes}u
        - {_CACHED_EXPERT_ROUTE_TABLE_BYTES}u);
    uint logical_slot = uint(expert_slots[expert]);
    const device uchar* slot_bytes = logical_slot < {persistent_slots}u
        ? persistent_bytes + size_t(logical_slot) * {bank_slot_bytes}u
        : transient_bytes
            + size_t(logical_slot - {persistent_slots}u) * {bank_slot_bytes}u;
    const device uint* slot_header = (const device uint*)slot_bytes;
"""
    return f"""
    uint linear = thread_position_in_grid.x;
    uint assignment = linear / {hidden_size}u;
    uint output = linear - assignment * {hidden_size}u;
{slot_binding}
{pointers}
    uint packed_pos = down_packed_directory[output]
        - down_packed_source_payload_offset;
    uint scales_pos = down_scales_directory[output]
        - down_scales_source_payload_offset;
    uint packed_initial = uint(down_packed_payload[packed_pos])
        | (uint(down_packed_payload[packed_pos + 1u]) << 8)
        | (uint(down_packed_payload[packed_pos + 2u]) << 16)
        | (uint(down_packed_payload[packed_pos + 3u]) << 24);
    uint scales_state = uint(down_scales_payload[scales_pos])
        | (uint(down_scales_payload[scales_pos + 1u]) << 8)
        | (uint(down_scales_payload[scales_pos + 2u]) << 16)
        | (uint(down_scales_payload[scales_pos + 3u]) << 24);
    uint packed_symbol = (packed_initial & {M - 1}u) >> 4u;
    uint packed_carry = packed_initial & 15u;
    packed_pos += 4u;
    scales_pos += 4u;
    float acc = 0.0f;
    for (uint group = 0u; group < {groups}u; ++group) {{
        uint scale_lo = uint(rans_next_transition(
            scales_state, scales_pos, down_scales_payload,
            down_scales_transition));
        uint scale_hi = uint(rans_next_transition(
            scales_state, scales_pos, down_scales_payload,
            down_scales_transition));
        float scale = shadow_scale(ushort(scale_lo | (scale_hi << 8)));
        float gd = 0.0f;
        uint base = group * {SHADOW_GROUP}u;
{uniform_group}
        acc += scale * gd;
    }}
    out[size_t(assignment) * {hidden_size}u + output]
        = static_cast<T>(acc);
"""


@lru_cache(maxsize=None)
def _cached_gate_up_kernel(
    dtype: mx.Dtype,
    hidden_size: int,
    expert_hidden_size: int,
    threads_per_tg: int,
):
    return mx.fast.metal_kernel(
        name=(
            "mtplx_glm52_q1t_cached_rans_gate_up_"
            f"{_DTYPE_TAG[dtype]}_h{hidden_size}_e{expert_hidden_size}_"
            f"t{threads_per_tg}"
        ),
        input_names=[
            "x",
            "slot_bytes",
            "gate_scales_transition",
            "up_scales_transition",
        ],
        output_names=["out"],
        header=_header(True),
        source=_cached_gate_up_source(hidden_size, expert_hidden_size),
    )


@lru_cache(maxsize=None)
def _cached_down_kernel(
    dtype: mx.Dtype,
    hidden_size: int,
    expert_hidden_size: int,
    threads_per_tg: int,
):
    return mx.fast.metal_kernel(
        name=(
            "mtplx_glm52_q1t_cached_rans_down_"
            f"{_DTYPE_TAG[dtype]}_h{hidden_size}_e{expert_hidden_size}_"
            f"t{threads_per_tg}"
        ),
        input_names=[
            "x",
            "slot_bytes",
            "down_scales_transition",
        ],
        output_names=["out"],
        header=_header(True),
        source=_cached_down_source(hidden_size, expert_hidden_size),
    )


@lru_cache(maxsize=None)
def _cached_bank_gate_up_kernel(
    dtype: mx.Dtype,
    hidden_size: int,
    expert_hidden_size: int,
    threads_per_tg: int,
    slot_bytes: int,
    persistent_slots: int,
):
    return mx.fast.metal_kernel(
        name=(
            "mtplx_glm52_q1t_cached_rans_bank_gate_up_"
            f"{_DTYPE_TAG[dtype]}_h{hidden_size}_e{expert_hidden_size}_"
            f"t{threads_per_tg}_s{slot_bytes}_p{persistent_slots}"
        ),
        input_names=[
            "x",
            "expert_ids",
            "persistent_bytes",
            "transient_bytes",
            "gate_scales_transition",
            "up_scales_transition",
        ],
        output_names=["out"],
        header=_header(True),
        source=_cached_gate_up_source(
            hidden_size,
            expert_hidden_size,
            bank_slot_bytes=slot_bytes,
            persistent_slots=persistent_slots,
        ),
    )


@lru_cache(maxsize=None)
def _cached_bank_down_kernel(
    dtype: mx.Dtype,
    hidden_size: int,
    expert_hidden_size: int,
    threads_per_tg: int,
    slot_bytes: int,
    persistent_slots: int,
):
    return mx.fast.metal_kernel(
        name=(
            "mtplx_glm52_q1t_cached_rans_bank_down_"
            f"{_DTYPE_TAG[dtype]}_h{hidden_size}_e{expert_hidden_size}_"
            f"t{threads_per_tg}_s{slot_bytes}_p{persistent_slots}"
        ),
        input_names=[
            "x",
            "expert_ids",
            "persistent_bytes",
            "transient_bytes",
            "down_scales_transition",
        ],
        output_names=["out"],
        header=_header(True),
        source=_cached_down_source(
            hidden_size,
            expert_hidden_size,
            bank_slot_bytes=slot_bytes,
            persistent_slots=persistent_slots,
        ),
    )


@dataclass(frozen=True)
class BoundGlm52Q1TFusedRansCachedGateUp:
    kernel: Callable[..., tuple[mx.array]]
    inputs: tuple[mx.array, ...]
    out_dim: int
    threads_per_tg: int
    dtype: mx.Dtype

    def __call__(self, x: mx.array) -> mx.array:
        assignments = int(x.shape[0])
        (out,) = self.kernel(
            inputs=[x, *self.inputs],
            template=[("T", self.dtype)],
            grid=(assignments * self.out_dim, 1, 1),
            threadgroup=(self.threads_per_tg, 1, 1),
            output_shapes=[(assignments, self.out_dim)],
            output_dtypes=[self.dtype],
        )
        return out


@dataclass(frozen=True)
class BoundGlm52Q1TFusedRansCachedDown:
    kernel: Callable[..., tuple[mx.array]]
    inputs: tuple[mx.array, ...]
    out_dim: int
    threads_per_tg: int
    dtype: mx.Dtype

    def __call__(self, x: mx.array) -> mx.array:
        assignments = int(x.shape[0])
        (out,) = self.kernel(
            inputs=[x, *self.inputs],
            template=[("T", self.dtype)],
            grid=(assignments * self.out_dim, 1, 1),
            threadgroup=(self.threads_per_tg, 1, 1),
            output_shapes=[(assignments, self.out_dim)],
            output_dtypes=[self.dtype],
        )
        return out


@dataclass(frozen=True)
class BoundGlm52Q1TFusedRansCachedExpert:
    gate_up: BoundGlm52Q1TFusedRansCachedGateUp
    down: BoundGlm52Q1TFusedRansCachedDown
    output_count: int = 1

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(self.gate_up(x))


@dataclass(frozen=True)
class BoundGlm52Q1TFusedRansCachedBankGateUp:
    kernel: Callable[..., tuple[mx.array]]
    inputs: tuple[mx.array, ...]
    out_dim: int
    threads_per_tg: int
    dtype: mx.Dtype

    def __call__(
        self,
        x: mx.array,
        expert_ids: mx.array,
    ) -> mx.array:
        assignments = int(x.shape[0])
        (out,) = self.kernel(
            inputs=[x, expert_ids, *self.inputs],
            template=[("T", self.dtype)],
            grid=(assignments * self.out_dim, 1, 1),
            threadgroup=(self.threads_per_tg, 1, 1),
            output_shapes=[(assignments, self.out_dim)],
            output_dtypes=[self.dtype],
        )
        return out


@dataclass(frozen=True)
class BoundGlm52Q1TFusedRansCachedBankDown:
    kernel: Callable[..., tuple[mx.array]]
    inputs: tuple[mx.array, ...]
    out_dim: int
    threads_per_tg: int
    dtype: mx.Dtype

    def __call__(
        self,
        x: mx.array,
        expert_ids: mx.array,
    ) -> mx.array:
        assignments = int(x.shape[0])
        (out,) = self.kernel(
            inputs=[x, expert_ids, *self.inputs],
            template=[("T", self.dtype)],
            grid=(assignments * self.out_dim, 1, 1),
            threadgroup=(self.threads_per_tg, 1, 1),
            output_shapes=[(assignments, self.out_dim)],
            output_dtypes=[self.dtype],
        )
        return out


@dataclass(frozen=True)
class BoundGlm52Q1TFusedRansCachedBank:
    gate_up: BoundGlm52Q1TFusedRansCachedBankGateUp
    down: BoundGlm52Q1TFusedRansCachedBankDown
    output_count: int = 1

    def __call__(
        self,
        x: mx.array,
        expert_ids: mx.array,
    ) -> mx.array:
        activated = self.gate_up(x, expert_ids)
        return self.down(activated, expert_ids)


def _validate_cached_scale_transition(name: str, transition: mx.array) -> None:
    _require_array(transition, name=name, dtype=mx.uint32, size=M)


def bind_glm52_q1t_fused_rans_cached_expert(
    *,
    slot_bytes: mx.array,
    gate_scales_transition: mx.array,
    up_scales_transition: mx.array,
    down_scales_transition: mx.array,
    hidden_size: int,
    expert_hidden_size: int,
    gate_up_threads_per_tg: int,
    down_threads_per_tg: int,
    dtype: mx.Dtype,
) -> BoundGlm52Q1TFusedRansCachedExpert:
    """Bind one fixed compressed-cache slot to direct final-output kernels."""

    hidden_size = int(hidden_size)
    expert_hidden_size = int(expert_hidden_size)
    gate_up_threads_per_tg = int(gate_up_threads_per_tg)
    down_threads_per_tg = int(down_threads_per_tg)
    _require_array(slot_bytes, name="slot_bytes", dtype=mx.uint8)
    if int(slot_bytes.size) < _CACHED_EXPERT_HEADER_WORDS * 4:
        raise Glm52Q1TFusedRansError("compressed-cache slot header is truncated")
    if (
        hidden_size < 1
        or expert_hidden_size < 1
        or hidden_size % SHADOW_GROUP
        or expert_hidden_size % SHADOW_GROUP
        or hidden_size % LANES
        or expert_hidden_size % LANES
    ):
        raise Glm52Q1TFusedRansError("cached GLM projection geometry is incompatible")
    valid_threads = (32, 64, 128, 256, 512, 1024)
    if (
        gate_up_threads_per_tg not in valid_threads
        or down_threads_per_tg not in valid_threads
        or expert_hidden_size % gate_up_threads_per_tg
        or hidden_size % down_threads_per_tg
        or dtype not in _DTYPE_TAG
    ):
        raise Glm52Q1TFusedRansError("cached GLM launch geometry is incompatible")
    _validate_cached_scale_transition("gate_scales_transition", gate_scales_transition)
    _validate_cached_scale_transition("up_scales_transition", up_scales_transition)
    _validate_cached_scale_transition("down_scales_transition", down_scales_transition)
    return BoundGlm52Q1TFusedRansCachedExpert(
        gate_up=BoundGlm52Q1TFusedRansCachedGateUp(
            kernel=_cached_gate_up_kernel(
                dtype,
                hidden_size,
                expert_hidden_size,
                gate_up_threads_per_tg,
            ),
            inputs=(slot_bytes, gate_scales_transition, up_scales_transition),
            out_dim=expert_hidden_size,
            threads_per_tg=gate_up_threads_per_tg,
            dtype=dtype,
        ),
        down=BoundGlm52Q1TFusedRansCachedDown(
            kernel=_cached_down_kernel(
                dtype,
                hidden_size,
                expert_hidden_size,
                down_threads_per_tg,
            ),
            inputs=(slot_bytes, down_scales_transition),
            out_dim=hidden_size,
            threads_per_tg=down_threads_per_tg,
            dtype=dtype,
        ),
    )


def bind_glm52_q1t_fused_rans_cached_bank(
    *,
    persistent_bytes: mx.array,
    transient_bytes: mx.array,
    slot_bytes: int,
    persistent_slots: int,
    gate_scales_transition: mx.array,
    up_scales_transition: mx.array,
    down_scales_transition: mx.array,
    hidden_size: int,
    expert_hidden_size: int,
    gate_up_threads_per_tg: int,
    down_threads_per_tg: int,
    dtype: mx.Dtype,
) -> BoundGlm52Q1TFusedRansCachedBank:
    """Bind one layer's fixed compressed cache banks to routed-ID kernels."""

    slot_bytes = int(slot_bytes)
    persistent_slots = int(persistent_slots)
    hidden_size = int(hidden_size)
    expert_hidden_size = int(expert_hidden_size)
    gate_up_threads_per_tg = int(gate_up_threads_per_tg)
    down_threads_per_tg = int(down_threads_per_tg)
    _require_array(persistent_bytes, name="persistent_bytes", dtype=mx.uint8)
    _require_array(transient_bytes, name="transient_bytes", dtype=mx.uint8)
    if (
        slot_bytes < _CACHED_EXPERT_HEADER_WORDS * 4
        or slot_bytes % 4
        or persistent_slots < 1
        or int(persistent_bytes.size) != persistent_slots * slot_bytes
        or int(transient_bytes.size) < slot_bytes
    ):
        raise Glm52Q1TFusedRansError("compressed cache bank geometry is incompatible")
    if (
        hidden_size < 1
        or expert_hidden_size < 1
        or hidden_size % SHADOW_GROUP
        or expert_hidden_size % SHADOW_GROUP
        or hidden_size % LANES
        or expert_hidden_size % LANES
    ):
        raise Glm52Q1TFusedRansError("cached GLM projection geometry is incompatible")
    valid_threads = (32, 64, 128, 256, 512, 1024)
    if (
        gate_up_threads_per_tg not in valid_threads
        or down_threads_per_tg not in valid_threads
        or expert_hidden_size % gate_up_threads_per_tg
        or hidden_size % down_threads_per_tg
        or dtype not in _DTYPE_TAG
    ):
        raise Glm52Q1TFusedRansError("cached GLM launch geometry is incompatible")
    _validate_cached_scale_transition("gate_scales_transition", gate_scales_transition)
    _validate_cached_scale_transition("up_scales_transition", up_scales_transition)
    _validate_cached_scale_transition("down_scales_transition", down_scales_transition)
    bank_inputs = (persistent_bytes, transient_bytes)
    return BoundGlm52Q1TFusedRansCachedBank(
        gate_up=BoundGlm52Q1TFusedRansCachedBankGateUp(
            kernel=_cached_bank_gate_up_kernel(
                dtype,
                hidden_size,
                expert_hidden_size,
                gate_up_threads_per_tg,
                slot_bytes,
                persistent_slots,
            ),
            inputs=(
                *bank_inputs,
                gate_scales_transition,
                up_scales_transition,
            ),
            out_dim=expert_hidden_size,
            threads_per_tg=gate_up_threads_per_tg,
            dtype=dtype,
        ),
        down=BoundGlm52Q1TFusedRansCachedBankDown(
            kernel=_cached_bank_down_kernel(
                dtype,
                hidden_size,
                expert_hidden_size,
                down_threads_per_tg,
                slot_bytes,
                persistent_slots,
            ),
            inputs=(*bank_inputs, down_scales_transition),
            out_dim=hidden_size,
            threads_per_tg=down_threads_per_tg,
            dtype=dtype,
        ),
    )
