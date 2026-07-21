"""Shadow-bank gather matmul (issue #51 miss fallback, #114 optimized).

``shadow_gather_mm`` runs one projection of the expert MLP against a
dense low-precision shadow bank: for each routed assignment (hidden row,
expert id) it computes ``W_e @ x`` where ``W_e`` is the expert's shadow
row block. Row index into the bank is the expert id (island-bank
convention). Decode row counts are tiny (1-8).

Kernel design (issue #114 — the shipped PR #102 kernel was a naive
one-thread-per-output baseline flagged "perf unmeasured"; the GPU window
measured it slow and a t158-hybrid regression). The measured lessons that
shaped this rewrite:

- **Keep one thread per output.** At decode sizes the matmul is limited
  by how many independent weight streams are in flight (memory-level
  parallelism). A SIMD-group-per-output variant that had 32 lanes
  cooperate on a single output was measured *slower* than the naive
  kernel here (it collapses MLP 32x and adds a reduction). One thread per
  output maximizes MLP and won.
- **Read the packed weights vectorized.** Each thread walks its own
  output row (contiguous in the bank), reading b1 sign words as a single
  ``uint2`` load per group. This, plus the large system cache absorbing
  the strided cross-thread pattern, keeps b1 near ``mx.gather_qmm``.
- **Decode t158 trits with a compile-time base-3 LUT.** The naive kernel
  spent most of its t158 time on per-slot ``%3``/``/3`` chains; a 1215-
  entry ``constant`` lookup (byte, slot -> trit-1) replaced that and cut
  t158 ~2.5x. t158 remains ALU-heavy.
- **Stage x per threadgroup for b1, read it broadcast-from-device for
  t158.** b1 benefits from the staged reuse; the ALU-bound t158 prefers
  the occupancy freed by dropping the staging barrier and threadgroup
  buffer (measured).

Honest standing versus stock ``mx.gather_qmm`` at hy3 decode shapes
(queued-lane microbench, see research notes / the PR): **b1 matches or
beats gather_qmm** (viable for the #111 q1-primary lane on zero-free
sources), but **t158 still trails gather_qmm ~1.6-2.1x** — it is only
~25% lighter than Q2 yet much heavier to unpack, so the Q2 shadow/hybrid
lane stays disadvantaged (the #26/#65 "custom loses to stock" precedent).
This kernel is nonetheless ~2-3x faster than the shipped naive baseline.

The bank layout is unchanged from PR #102, so callers
(``mtplx.models.expert_mlx``) and the numpy-reference parity tests are
untouched.

Scales are stored as bf16 bit patterns in u16 and widened in-kernel via
``as_type`` so the CPU (numpy) and GPU decode paths share one
representation.

Eager-only: ``mx.fast.metal_kernel`` is not traceable under
``mx.compile`` — callers must keep the shadow path out of compiled
graphs (the helper materializes its output with ``mx.eval``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx

from mtplx.expert_shadow import (
    SHADOW_GROUP,
    _B1_WORDS_PER_GROUP,
    _T158_BYTES_PER_GROUP,
    ShadowCodecError,
)

_DTYPE_TAG = {mx.bfloat16: "bf16", mx.float16: "fp16", mx.float32: "fp32"}

# Per-codec defaults tuned on the hy3 decode shapes (gate/up out=1536
# in=4096, down out=4096 in=1536, rows 1-8) via the queued-lane microbench.
_DEFAULTS = {
    "b1": {"threads_per_tg": 256, "stage": True},
    "t158": {"threads_per_tg": 128, "stage": False},
}
# Above this input width, x staging would exceed the threadgroup memory
# budget (4 bytes/element); fall back to broadcast-from-device reads.
_STAGE_MAX_IN = 8192

# base-3 trit LUT for t158: LUT[byte * 5 + slot] = decoded (trit - 1),
# with trit = (byte // 3**slot) % 3. Values are in {-1, 0, 1}.
_POW3 = (1, 3, 9, 27, 81)
_T158_LUT = ",".join(
    f"{((bv // _POW3[j]) % 3) - 1}.0f" for bv in range(243) for j in range(5)
)


@dataclass(frozen=True)
class BoundShadowGather:
    """One construction-qualified shadow gather launch.

    Array ownership and shapes are invariants of the installed component-bank
    route.  The call therefore submits the fixed launch directly; it does not
    repeat codec, dtype, bank-shape, or grid validation.
    """

    kernel: Callable[..., tuple[mx.array]]
    dtype: mx.Dtype
    rows: int
    in_dim: int
    out_dim: int
    threads_per_tg: int
    grid_x: int

    def __call__(
        self,
        x: mx.array,
        expert_ids: mx.array,
        packed: mx.array,
        scales: mx.array,
    ) -> mx.array:
        (out,) = self.kernel(
            inputs=[
                mx.contiguous(x),
                expert_ids,
                packed,
                scales,
                self.rows,
                self.out_dim,
                self.in_dim,
            ],
            template=[("T", self.dtype)],
            grid=(self.grid_x, 1, 1),
            threadgroup=(self.threads_per_tg, 1, 1),
            output_shapes=[(self.rows, self.out_dim)],
            output_dtypes=[self.dtype],
        )
        return out


def _validate_bound_geometry(
    codec: str,
    dtype: mx.Dtype,
    rows: int,
    in_dim: int,
    out_dim: int,
    packed_shape: Sequence[int],
    scales_shape: Sequence[int],
    threads_per_tg: int,
    stage: bool,
) -> int:
    if codec not in _DEFAULTS:
        raise ShadowCodecError(f"unknown shadow codec {codec!r}")
    if dtype not in _DTYPE_TAG:
        raise ShadowCodecError(f"unsupported shadow gather dtype {dtype}")
    rows = int(rows)
    in_dim = int(in_dim)
    out_dim = int(out_dim)
    threads_per_tg = int(threads_per_tg)
    if rows < 1 or in_dim < 1 or out_dim < 1:
        raise ShadowCodecError("bound shadow geometry must be positive")
    if in_dim % SHADOW_GROUP:
        raise ShadowCodecError(
            f"shadow input dim {in_dim} is not a multiple of {SHADOW_GROUP}"
        )
    if threads_per_tg < 1:
        raise ShadowCodecError("threads_per_tg must be >= 1")
    if stage and in_dim > _STAGE_MAX_IN:
        raise ShadowCodecError(
            f"staged shadow input dim {in_dim} exceeds {_STAGE_MAX_IN}"
        )
    if len(packed_shape) != 3 or len(scales_shape) != 3:
        raise ShadowCodecError("shadow bank arrays must be 3-D")
    groups = in_dim // SHADOW_GROUP
    expected_words = groups * (
        _B1_WORDS_PER_GROUP if codec == "b1" else _T158_BYTES_PER_GROUP
    )
    if (
        int(packed_shape[0]) < 1
        or int(packed_shape[1]) != out_dim
        or int(packed_shape[2]) != expected_words
    ):
        raise ShadowCodecError(
            f"shadow packed shape {tuple(packed_shape)} does not match "
            f"codec {codec!r}, out={out_dim}, groups={groups}"
        )
    if (
        int(scales_shape[0]) != int(packed_shape[0])
        or int(scales_shape[1]) != out_dim
        or int(scales_shape[2]) != groups
    ):
        raise ShadowCodecError(
            f"shadow scales shape {tuple(scales_shape)} does not match "
            f"packed capacity={int(packed_shape[0])}, out={out_dim}, groups={groups}"
        )
    total = rows * out_dim
    if total >= 2**31:
        raise ShadowCodecError(f"shadow gather grid too large: {total}")
    blocks_per_row = -(-out_dim // threads_per_tg)
    return rows * blocks_per_row * threads_per_tg


def bind_shadow_gather_mm(
    *,
    codec: str,
    dtype: mx.Dtype,
    rows: int,
    in_dim: int,
    out_dim: int,
    packed_shape: Sequence[int],
    scales_shape: Sequence[int],
    threads_per_tg: int,
    stage: bool,
) -> BoundShadowGather:
    """Validate fixed bank geometry once and return its direct launch."""

    grid_x = _validate_bound_geometry(
        codec,
        dtype,
        rows,
        in_dim,
        out_dim,
        packed_shape,
        scales_shape,
        threads_per_tg,
        stage,
    )
    kernel = _shadow_gather_kernel(
        codec,
        dtype,
        int(threads_per_tg),
        int(in_dim),
        bool(stage),
    )
    return BoundShadowGather(
        kernel=kernel,
        dtype=dtype,
        rows=int(rows),
        in_dim=int(in_dim),
        out_dim=int(out_dim),
        threads_per_tg=int(threads_per_tg),
        grid_x=grid_x,
    )


def _header() -> str:
    return f"""
using namespace metal;
constant constexpr uint SHADOW_GROUP = 64;
constant float T158_LUT[1215] = {{ {_T158_LUT} }};

inline float shadow_scale(ushort bits) {{
    return as_type<float>(uint(bits) << 16);
}}
"""


def _x_access(stage: bool) -> tuple[str, str, str]:
    """Return (staging preamble, x-index prefix, x-index suffix)."""

    if stage:
        preamble = (
            "    threadgroup float xs[IN_STAGE];\n"
            "    for (uint i = tid; i < in_dim; i += TPB) {\n"
            "        xs[i] = float(x[size_t(row) * in_dim + i]);\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        )
        return preamble, "xs[", "]"
    return "", "float(x[size_t(row) * in_dim + ", "])"


_COMMON_HEAD = """
    uint tg = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    uint rows = uint(row_count);
    uint out_dim = uint(out_size);
    uint in_dim = uint(in_size);
    uint blocks_per_row = (out_dim + TPB - 1) / TPB;
    uint row = tg / blocks_per_row;
    if (row >= rows) {
        return;
    }
    uint block = tg % blocks_per_row;
"""


def _b1_source(threads_per_tg: int, in_dim: int, stage: bool) -> str:
    pre, xa, xt = _x_access(stage)
    return f"""
    constexpr uint TPB = {int(threads_per_tg)}u;
    constexpr uint IN_STAGE = {int(in_dim)}u;
{_COMMON_HEAD}
{pre}
    uint o = block * TPB + tid;
    if (o >= out_dim) {{
        return;
    }}
    uint expert = uint(ids[row]);
    uint groups = in_dim / SHADOW_GROUP;
    // b1 packs 2 u32 sign words per group; read them as one uint2 load.
    const device uint2* pw = (const device uint2*)packed;
    size_t rbase = (size_t(expert) * out_dim + o) * size_t(groups);
    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {{
        uint2 w = pw[rbase + g];
        float sc = shadow_scale(scales[rbase + g]);
        float gd = 0.0f;
        uint base = g * SHADOW_GROUP;
        _Pragma("unroll")
        for (uint b = 0; b < 32u; ++b) {{
            float xv = {xa}base + b{xt};
            gd += ((w.x >> b) & 1u) ? xv : -xv;
        }}
        _Pragma("unroll")
        for (uint b = 0; b < 32u; ++b) {{
            float xv = {xa}base + 32u + b{xt};
            gd += ((w.y >> b) & 1u) ? xv : -xv;
        }}
        acc += sc * gd;
    }}
    out[size_t(row) * out_dim + o] = static_cast<T>(acc);
"""


def _t158_source(threads_per_tg: int, in_dim: int, stage: bool) -> str:
    pre, xa, xt = _x_access(stage)
    # bytes 0..11 hold 5 trits each (slots 0..59); byte 12 holds slots
    # 60..63 (its 5th slot, element 64, is unused padding).
    body = []
    for bi in range(12):
        body.append(f"        {{ uint bv = uint(packed[gb + {bi}u]) * 5u; uint k = base + {bi * 5}u;")
        for j in range(5):
            body.append(
                f"          gd = fma(T158_LUT[bv + {j}u], {xa}k + {j}u{xt}, gd);"
            )
        body.append("        }")
    body.append("        { uint bv = uint(packed[gb + 12u]) * 5u; uint k = base + 60u;")
    for j in range(4):
        body.append(
            f"          gd = fma(T158_LUT[bv + {j}u], {xa}k + {j}u{xt}, gd);"
        )
    body.append("        }")
    inner = "\n".join(body)
    return f"""
    constexpr uint TPB = {int(threads_per_tg)}u;
    constexpr uint IN_STAGE = {int(in_dim)}u;
{_COMMON_HEAD}
{pre}
    uint o = block * TPB + tid;
    if (o >= out_dim) {{
        return;
    }}
    uint expert = uint(ids[row]);
    uint groups = in_dim / SHADOW_GROUP;
    size_t rbytes = (size_t(expert) * out_dim + o) * size_t(groups * 13u);
    size_t rscale = (size_t(expert) * out_dim + o) * size_t(groups);
    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {{
        float sc = shadow_scale(scales[rscale + g]);
        float gd = 0.0f;
        uint base = g * SHADOW_GROUP;
        size_t gb = rbytes + size_t(g) * 13u;
{inner}
        acc += sc * gd;
    }}
    out[size_t(row) * out_dim + o] = static_cast<T>(acc);
"""


@lru_cache(maxsize=None)
def _shadow_gather_kernel(
    codec: str, dtype: mx.Dtype, threads_per_tg: int, in_dim: int, stage: bool
):
    if codec == "b1":
        source = _b1_source(threads_per_tg, in_dim, stage)
    elif codec == "t158":
        source = _t158_source(threads_per_tg, in_dim, stage)
    else:  # pragma: no cover - guarded by callers
        raise ShadowCodecError(f"unknown shadow codec {codec!r}")
    dtype_tag = _DTYPE_TAG.get(dtype, "generic")
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_shadow_gather_{codec}_{dtype_tag}_t{threads_per_tg}"
            f"_k{in_dim}_s{int(stage)}"
        ),
        input_names=["x", "ids", "packed", "scales", "row_count", "out_size", "in_size"],
        output_names=["out"],
        header=_header(),
        source=source,
    )


def shadow_gather_mm(
    x: mx.array,
    expert_ids: mx.array,
    packed: mx.array,
    scales: mx.array,
    *,
    codec: str,
    threads_per_tg: int | None = None,
    stage: bool | None = None,
) -> mx.array:
    """Gather-matmul one projection from a shadow bank.

    ``x``: (rows, in) hidden rows, one per assignment. ``expert_ids``:
    (rows,) int32 bank rows. ``packed``: (experts, out, words) u32 for
    ``b1`` / (experts, out, bytes) u8 for ``t158``. ``scales``:
    (experts, out, in/64) u16 bf16 bits. Returns (rows, out) in
    ``x.dtype`` (fp32 accumulate).

    ``threads_per_tg`` / ``stage`` override the per-codec tuned defaults
    (one output element per thread; optionally stage ``x[row]`` in
    threadgroup memory). Callers normally leave these at the defaults.
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

    defaults = _DEFAULTS.get(codec)
    if defaults is None:  # pragma: no cover - guarded above via expected_words
        raise ShadowCodecError(f"unknown shadow codec {codec!r}")
    tpt = int(defaults["threads_per_tg"] if threads_per_tg is None else threads_per_tg)
    if tpt < 1:
        raise ShadowCodecError("threads_per_tg must be >= 1")
    use_stage = bool(defaults["stage"] if stage is None else stage)
    if use_stage and in_dim > _STAGE_MAX_IN:
        use_stage = False  # keep threadgroup memory within budget

    total = rows * out_dim
    if total <= 0:
        raise ShadowCodecError("shadow_gather_mm requires at least one assignment")
    if total >= 2**31:  # metal_kernel output element counts must stay < 2^31
        raise ShadowCodecError(f"shadow gather grid too large: {total}")

    kernel = _shadow_gather_kernel(codec, x.dtype, tpt, in_dim, use_stage)
    blocks_per_row = -(-out_dim // tpt)
    threadgroups = rows * blocks_per_row
    grid_x = threadgroups * tpt
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
        threadgroup=(tpt, 1, 1),
        output_shapes=[(rows, out_dim)],
        output_dtypes=[x.dtype],
    )
    return out
