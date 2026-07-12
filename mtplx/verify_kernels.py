"""MTPLX verify-shape quantized-matmul kernels (original implementation).

Speculative verify multiplies a skinny row batch (M = 1 + draft depth) against
the model's quantized weight matrices. Stock MLX qmm is tuned for M=1 (decode)
and large-M (prefill); the 4..6-row verify shapes underuse memory bandwidth,
and at huge N (lm_head) the tiny-tile grid thrashes the scheduler.

Design (MTPLX, 2026-07-02):
  One threadgroup carries NSG independent simdgroups. Each simdgroup owns its
  own BN=4 column tile and performs the FULL K reduction lane-strided over the
  pack-interleaved quantized words - no K split, no partial buffers, no
  threadgroup memory, no barriers. simd_sum reduces within the simdgroup and
  lanes 0..M*BN-1 write the tile. The simdgroups share nothing; they ride in
  one threadgroup purely to give the scheduler fewer, heavier units, which
  (a) fixes the huge-N tiny-tile thrash (lm_head: 3.5 -> 1.38 ms/call chained,
  1.18x the weight-stream floor, measured 2026-07-02) and (b) keeps the
  scheduling footprint of the co-residency-proven K-split shape without its
  barrier cost.

Empirical constraints inherited from the 2026-06-12/07-02 measurement ledger
(outputs/m4-rewrite-20260612, outputs/sprint-kernels-20260702, research repo):
  - pack-interleaved 32-bit weight loads are the load-bearing coalescing
    pattern; wider per-lane loads and group-aligned lanes both regress.
  - scales/biases are L1-served; per-pack scalar loads are free.
  - explicit accumulator arrays with forced unrolls are load-bearing
    (array-indexed rows without unrolls stack-spill, 10x).
  - 24 accumulators/thread is the proven register ceiling (32 kills occupancy).

Numerics: accumulation order differs from stock qmm (fp32 accumulate,
lane-strided K) -> bf16 tail-ULP class differences, same as every custom
verify kernel. Gated by the distribution/exactness corpus before product use
(scripts/r1_chisquare_verifier_correctness.py and the promotion gates).

Supported: 4-bit, 6-bit and 8-bit affine layouts, group_size in {32, 64,
128}, bf16/fp16 activations, M in 4..6 (D3-D5 verify). Everything else falls
back to stock. Plain SIMD - no NAX/G17/macOS gate; runs on all Apple Silicon.

6-bit (2026-07-07, the 9B-tier lane): MLX packs 6-bit affine values
bit-contiguously little-endian - value k occupies bits [6k, 6k+6) of the
byte stream (mlx/backend/metal/kernels/quantized.h, get_pack_factor 4 values
per 3 bytes). The split-K kernels process one 16-value "hexpack" (3 uint32
words, 12 bytes) per lane iteration so weight loads stay coalesced 32-bit
reads and activation loads stay Vec8-wide (two per row). A hexpack never
straddles a scale group because 16 divides every supported group_size.
Only the split-K morphology is implemented for 6-bit (the in-context
winner); the msg wide tile stays 4/8-bit.
"""

from __future__ import annotations

import os

import mlx.core as mx

_KERNEL_CACHE: dict[tuple, object] = {}

# Simdgroups per threadgroup. 8 (=256 threads, 32 columns) won the 2026-07-02
# sweep vs 4 and 16; env knob kept for serve-path co-residency sweeps.
_M4_NSG = max(1, min(24, int(os.environ.get("MTPLX_VK_M4_NSG", "8") or 8)))
_M6_NSG = max(1, min(24, int(os.environ.get("MTPLX_VK_M6_NSG", "4") or 4)))


def _fma_block(m: int, bits: int) -> str:
    """Emit the dequant + FMA inner block for one 32-bit pack column set.

    bits=4: 8 weights per pack word. bits=8: 4 weights per pack word.
    Named scalar weight temps + explicit acc indices; the enclosing loop is
    wrapped in _Pragma("unroll") by the caller.
    """
    per = 8 if bits == 4 else 4
    mask = "0xFu" if bits == 4 else "0xFFu"
    shift = 4 if bits == 4 else 8
    lines = [f"for (int ki = 0; ki < {per}; ++ki) {{"]
    for j in range(4):
        lines.append(
            f"    float w{j} = float((p{j} >> (ki * {shift})) & {mask}) * s{j} + b{j};"
        )
    for j in range(4):
        for r in range(m):
            lines.append(f"    acc[{j} * {m} + {r}] += float(v{r}[ki{'' if bits == 4 else ' + koff'}]) * w{j};")
    lines.append("}")
    return "\n            ".join(lines)


def _build_msg_kernel(m: int, bits: int, group_size: int, dtype: mx.Dtype, nsg: int):
    """Multi-simdgroup column-parallel kernel for M=m rows, 4- or 8-bit affine."""
    key = ("msg", m, bits, group_size, dtype, nsg)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    xloads = "\n            ".join(
        f"Vec8 v{r} = xv[({r} * K + k_base) / 8];" for r in range(m)
    )
    if bits == 4:
        # One 32-bit pack = 8 weights; lane-strided packs of K/8.
        pack_setup = """
            int k_base = pack * 8;
            int gi = k_base / GS;
            uint32_t p0 = w_q[(n0 + 0) * K_by_p + pack];
            uint32_t p1 = w_q[(n0 + 1) * K_by_p + pack];
            uint32_t p2 = w_q[(n0 + 2) * K_by_p + pack];
            uint32_t p3 = w_q[(n0 + 3) * K_by_p + pack];
        """
        body = f"""
        for (int pack = int(lane); pack < K_by_p; pack += 32) {{
            {pack_setup}
            {xloads}
            float s0 = float(scales[(n0 + 0) * K_by_gs + gi]);
            float s1 = float(scales[(n0 + 1) * K_by_gs + gi]);
            float s2 = float(scales[(n0 + 2) * K_by_gs + gi]);
            float s3 = float(scales[(n0 + 3) * K_by_gs + gi]);
            float b0 = float(biases[(n0 + 0) * K_by_gs + gi]);
            float b1 = float(biases[(n0 + 1) * K_by_gs + gi]);
            float b2 = float(biases[(n0 + 2) * K_by_gs + gi]);
            float b3 = float(biases[(n0 + 3) * K_by_gs + gi]);
            _Pragma("unroll")
            {_fma_block(m, 4)}
        }}
        """
        k_by_p_expr = "K / 8"
    else:
        # 8-bit: one 32-bit word = 4 weights. Process word PAIRS (8 weights)
        # per iteration so activation loads stay Vec8-wide; the pair spans one
        # 8-weight span, so gi is shared (GS >= 32 guarantees no group split
        # inside a pair for gs in {32,64,128}).
        body = f"""
        for (int pair = int(lane); pair < K_by_p; pair += 32) {{
            int k_base = pair * 8;
            int gi = k_base / GS;
            {xloads}
            _Pragma("unroll")
            for (int wsel = 0; wsel < 2; ++wsel) {{
                int koff = wsel * 4;
                uint32_t p0 = w_q[(n0 + 0) * (K / 4) + pair * 2 + wsel];
                uint32_t p1 = w_q[(n0 + 1) * (K / 4) + pair * 2 + wsel];
                uint32_t p2 = w_q[(n0 + 2) * (K / 4) + pair * 2 + wsel];
                uint32_t p3 = w_q[(n0 + 3) * (K / 4) + pair * 2 + wsel];
                float s0 = float(scales[(n0 + 0) * K_by_gs + gi]);
                float s1 = float(scales[(n0 + 1) * K_by_gs + gi]);
                float s2 = float(scales[(n0 + 2) * K_by_gs + gi]);
                float s3 = float(scales[(n0 + 3) * K_by_gs + gi]);
                float b0 = float(biases[(n0 + 0) * K_by_gs + gi]);
                float b1 = float(biases[(n0 + 1) * K_by_gs + gi]);
                float b2 = float(biases[(n0 + 2) * K_by_gs + gi]);
                float b3 = float(biases[(n0 + 3) * K_by_gs + gi]);
                _Pragma("unroll")
                {_fma_block(m, 8)}
            }}
        }}
        """
        k_by_p_expr = "K / 8"

    n_acc = 4 * m
    source = f"""
        using namespace metal;
        constexpr int GS = {group_size};
        constexpr int NSG = {nsg};
        constexpr int MROWS = {m};

        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_p = {k_by_p_expr};
        int K_by_gs = K / GS;
        int n0 = (int(tg_n) * NSG + int(sg)) * 4;
        if (n0 + 3 >= N) {{ return; }}

        float acc[{n_acc}];
        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        {body}

        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        if (lane < {n_acc}) {{
            int j = int(lane) / MROWS;
            int row = int(lane) - j * MROWS;
            y[row * N + n0 + j] = T(acc[int(lane)]);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_vk_m{m}_q{bits}_nsg{nsg}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def _eligible(m: int, K: int, N: int, bits: int, group_size: int, dtype, nsg: int) -> bool:
    return (
        int(bits) in (4, 8)
        and int(group_size) in (32, 64, 128)
        and dtype in (mx.bfloat16, mx.float16)
        and 4 <= int(m) <= 6
        and int(K) % 64 == 0
        and int(N) % (4 * nsg) == 0
    )


def vk_eligible_m4(m: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    return int(m) == 4 and _eligible(4, K, N, bits, group_size, dtype, _M4_NSG)


def vk_eligible_m6(m: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    return 5 <= int(m) <= 6 and _eligible(6, K, N, bits, group_size, dtype, _M6_NSG)


def _run(m: int, x2: mx.array, w_q: mx.array, scales: mx.array, biases: mx.array,
         *, bits: int, group_size: int, nsg: int) -> mx.array:
    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(w_q.shape[0])
    if M < m:
        pad = mx.zeros((m - M, K), dtype=x2.dtype)
        xm = mx.contiguous(mx.concatenate([x2, pad], axis=0))
    else:
        xm = mx.contiguous(x2)
    kernel = _build_msg_kernel(m, bits, group_size, x2.dtype, nsg)
    cols = 4 * nsg
    (y,) = kernel(
        inputs=[xm, w_q, scales, biases, K, N],
        template=[("T", x2.dtype)],
        grid=(32 * nsg, N // cols, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(m, N)],
        output_dtypes=[x2.dtype],
    )
    return y[:M, :] if M < m else y


def vk_qmm_m4(x2, w_q, scales, biases, *, bits: int = 4, group_size: int = 64):
    """4-row verify matmul (D3 shape), msg geometry."""
    return _run(4, x2, w_q, scales, biases, bits=bits, group_size=group_size, nsg=_M4_NSG)


def vk_qmm_m6(x2, w_q, scales, biases, *, bits: int = 4, group_size: int = 64):
    """5..6-row verify matmul (D4/D5 shapes), msg geometry; pads M=5 to 6."""
    return _run(6, x2, w_q, scales, biases, bits=bits, group_size=group_size, nsg=_M6_NSG)


def vk_qmm_m4_ksplit(x2, w_q, scales, biases, *, bits: int = 4, group_size: int = 64):
    """4-row verify matmul, split-K morphology (the in-context winner)."""
    return _run_ksplit(4, x2, w_q, scales, biases, bits=bits, group_size=group_size, dual=False)


def vk_qmm_m6_ksplit(x2, w_q, scales, biases, *, bits: int = 4, group_size: int = 64):
    """5..6-row verify matmul, split-K morphology; pads M=5 to 6."""
    return _run_ksplit(6, x2, w_q, scales, biases, bits=bits, group_size=group_size, dual=False)


# ---------------------------------------------------------------------------
# Split-K family. The 2026-07-02 serve-path pair falsified the "bigger
# threadgroups co-reside better" hypothesis (msg/oct: -20..-28% in-context
# despite iso wins): under mixed scheduling with attention/GDN kernels the
# WINNING shape is many small column tiles (grid y = N/4) with the K
# reduction split across simdgroups — deep occupancy queues + intra-tile
# latency hiding. This is the textbook split-K reduction (same morphology as
# MLX steel's splitk); implementation below is MTPLX-original, parameterized
# over M rows and 4/8-bit affine layouts, with optional dual-pack software
# pipelining (two coalesced 32-bit weight words in flight per lane).
# ---------------------------------------------------------------------------


# The 16 6-bit values of one hexpack, extracted from three little-endian
# uint32 words (wa, wb, wc): value k = bits [6k, 6k+6) of the 96-bit stream.
_HEXPACK6_EXTRACT = (
    "(wa & 0x3Fu)",
    "((wa >> 6) & 0x3Fu)",
    "((wa >> 12) & 0x3Fu)",
    "((wa >> 18) & 0x3Fu)",
    "((wa >> 24) & 0x3Fu)",
    "(((wa >> 30) & 0x3u) | ((wb & 0xFu) << 2))",
    "((wb >> 4) & 0x3Fu)",
    "((wb >> 10) & 0x3Fu)",
    "((wb >> 16) & 0x3Fu)",
    "((wb >> 22) & 0x3Fu)",
    "(((wb >> 28) & 0xFu) | ((wc & 0x3u) << 4))",
    "((wc >> 2) & 0x3Fu)",
    "((wc >> 8) & 0x3Fu)",
    "((wc >> 14) & 0x3Fu)",
    "((wc >> 20) & 0x3Fu)",
    "((wc >> 26) & 0x3Fu)",
)


def _pack_block(m: int, bits: int, sfx: str) -> str:
    """Emit loads + dequant + FMA for one pack index variable pack{sfx}.

    Column-major construction: all loads up front, then one sequential
    dequant+FMA chain per output column. The 2026-07-02 midform race showed
    the interleaved (all-columns-per-k-step) construction costs ~2.4 ms per
    verify call in-context vs column-major chains — the compiler pipelines
    independent per-column blocks better than one wide interleaved block.
    """
    p = f"pack{sfx}"
    if bits == 6:
        # One hexpack = 16 values = 3 words; two Vec8 activation loads/row.
        lines = [f"int k_base{sfx} = {p} * 16;", f"int gi{sfx} = k_base{sfx} / GS;"]
        for r in range(m):
            lines.append(f"Vec8 v{sfx}_{r} = xv[({r} * K + k_base{sfx}) / 8];")
            lines.append(f"Vec8 u{sfx}_{r} = xv[({r} * K + k_base{sfx}) / 8 + 1];")
        for j in range(4):
            lines.append(
                f"uint32_t wa{sfx}_{j} = w_q[(n0 + {j}) * K_by_w + {p} * 3];"
                f" uint32_t wb{sfx}_{j} = w_q[(n0 + {j}) * K_by_w + {p} * 3 + 1];"
                f" uint32_t wc{sfx}_{j} = w_q[(n0 + {j}) * K_by_w + {p} * 3 + 2];"
            )
        for j in range(4):
            lines.append(
                f"float s{sfx}_{j} = float(scales[(n0 + {j}) * K_by_gs + gi{sfx}]);"
                f" float b{sfx}_{j} = float(biases[(n0 + {j}) * K_by_gs + gi{sfx}]);"
            )
        for j in range(4):
            block = [
                "{",
                f"    uint32_t wa = wa{sfx}_{j};",
                f"    uint32_t wb = wb{sfx}_{j};",
                f"    uint32_t wc = wc{sfx}_{j};",
                f"    float s = s{sfx}_{j};",
                f"    float b = b{sfx}_{j};",
            ]
            for ki, expr in enumerate(_HEXPACK6_EXTRACT):
                block.append(f"    float w6_{ki} = float{expr} * s + b;")
                xv = f"v{sfx}_{{r}}[{ki}]" if ki < 8 else f"u{sfx}_{{r}}[{ki - 8}]"
                for r in range(m):
                    block.append(
                        f"    acc[{j} * {m} + {r}] += float({xv.format(r=r)}) * w6_{ki};"
                    )
            block.append("}")
            lines.extend(block)
        return "\n            ".join(lines)
    lines = [f"int k_base{sfx} = {p} * 8;", f"int gi{sfx} = k_base{sfx} / GS;"]
    for r in range(m):
        lines.append(f"Vec8 v{sfx}_{r} = xv[({r} * K + k_base{sfx}) / 8];")
    if bits == 4:
        for j in range(4):
            lines.append(f"uint32_t p{sfx}_{j} = w_q[(n0 + {j}) * K_by_p + {p}];")
        for j in range(4):
            lines.append(
                f"float s{sfx}_{j} = float(scales[(n0 + {j}) * K_by_gs + gi{sfx}]);"
                f" float b{sfx}_{j} = float(biases[(n0 + {j}) * K_by_gs + gi{sfx}]);"
            )
        for j in range(4):
            block = [
                "{",
                f"    uint32_t packed = p{sfx}_{j};",
                f"    float s = s{sfx}_{j};",
                f"    float b = b{sfx}_{j};",
                "    for (int ki = 0; ki < 8; ++ki) {",
                "        float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;",
            ]
            for r in range(m):
                block.append(f"        acc[{j} * {m} + {r}] += float(v{sfx}_{r}[ki]) * wv;")
            block.extend(["    }", "}"])
            lines.extend(block)
    else:
        # 8-bit: two words per 8-weight span, same activation Vec8 width;
        # per-column chain covers both words to keep the scalar chain shape.
        for j in range(4):
            lines.append(
                f"uint32_t pa{sfx}_{j} = w_q[(n0 + {j}) * K_by_w + {p} * 2];"
                f" uint32_t pb{sfx}_{j} = w_q[(n0 + {j}) * K_by_w + {p} * 2 + 1];"
            )
        for j in range(4):
            lines.append(
                f"float s{sfx}_{j} = float(scales[(n0 + {j}) * K_by_gs + gi{sfx}]);"
                f" float b{sfx}_{j} = float(biases[(n0 + {j}) * K_by_gs + gi{sfx}]);"
            )
        for j in range(4):
            block = [
                "{",
                f"    uint32_t pa = pa{sfx}_{j};",
                f"    uint32_t pb = pb{sfx}_{j};",
                f"    float s = s{sfx}_{j};",
                f"    float b = b{sfx}_{j};",
                "    for (int ki = 0; ki < 4; ++ki) {",
                "        float wa = float((pa >> (ki * 8)) & 0xFFu) * s + b;",
                "        float wb = float((pb >> (ki * 8)) & 0xFFu) * s + b;",
            ]
            for r in range(m):
                block.append(f"        acc[{j} * {m} + {r}] += float(v{sfx}_{r}[ki]) * wa;")
                block.append(f"        acc[{j} * {m} + {r}] += float(v{sfx}_{r}[ki + 4]) * wb;")
            block.extend(["    }", "}"])
            lines.extend(block)
    return "\n            ".join(lines)


def _build_ksplit_kernel(
    m: int,
    bits: int,
    group_size: int,
    dtype: mx.Dtype,
    *,
    k_parts: int,
    dual: bool,
    kconst: int = 0,
):
    key = ("ksplit", m, bits, group_size, dtype, k_parts, dual, int(kconst))
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    n_acc = 4 * m
    if dual:
        loop = f"""
        for (int packA = p_start + int(lane); packA < p_end; packA += 64) {{
            {_pack_block(m, bits, "A")}
            int packB = packA + 32;
            if (packB < p_end) {{
                {_pack_block(m, bits, "B")}
            }}
        }}
        """
    else:
        loop = f"""
        for (int packA = p_start + int(lane); packA < p_end; packA += 32) {{
            {_pack_block(m, bits, "A")}
        }}
        """

    # Pack unit: values consumed per lane iteration. 4/8-bit process one
    # 8-value span; 6-bit processes one 16-value hexpack (3 words).
    # K_by_w = 32-bit weight words per output row for the multi-word
    # layouts (8-bit: K/4; 6-bit: 3*K/16).
    pack_unit = 16 if bits == 6 else 8
    words_per_row = "3 * (K / 16)" if bits == 6 else "K / 4"
    if kconst:
        k_decl = f"""constexpr int K = {int(kconst)};
        constexpr int K_by_p = K / {pack_unit};
        constexpr int K_by_w = {words_per_row};
        constexpr int K_by_gs = K / GS;
        constexpr int per_part = K_by_p / K_PARTS;"""
    else:
        k_decl = f"""int K = int(K_size);
        int K_by_p = K / {pack_unit};
        int K_by_w = {words_per_row};
        int K_by_gs = K / GS;
        int per_part = K_by_p / K_PARTS;"""

    source = f"""
        using namespace metal;
        constexpr int GS = {group_size};
        constexpr int K_PARTS = {k_parts};

        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        {k_decl}
        int N = int(N_size);
        int n0 = int(tg_n) * 4;
        int p_start = int(part) * per_part;
        int p_end = (int(part) == K_PARTS - 1) ? K_by_p : p_start + per_part;

        float acc[{n_acc}];
        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        {loop}

        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        threadgroup float partials[K_PARTS * {n_acc}];
        if (lane == 0) {{
            _Pragma("unroll")
            for (int i = 0; i < {n_acc}; ++i) {{
                partials[int(part) * {n_acc} + i] = acc[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (part == 0 && lane < {n_acc}) {{
            float total = 0.0f;
            _Pragma("unroll")
            for (int p = 0; p < K_PARTS; ++p) {{
                total += partials[p * {n_acc} + int(lane)];
            }}
            int j = int(lane) / {m};
            int row = int(lane) - j * {m};
            y[row * N + n0 + j] = T(total);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    # kconst MUST be in the kernel name: Metal host_names are global, and two
    # different-K specializations sharing a name bind the wrong binary (GPU
    # page fault, caught by the midform gate 2026-07-02).
    kname = (
        f"mtplx_vk_ks_m{m}_q{bits}_kp{k_parts}{'d' if dual else ''}"
        f"{f'_k{int(kconst)}' if kconst else ''}_gs{group_size}_{dtype_tag}"
    )
    kernel = mx.fast.metal_kernel(
        name=kname,
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def _run_ksplit(
    m: int,
    x2,
    w_q,
    scales,
    biases,
    *,
    bits: int,
    group_size: int,
    dual: bool,
    kconst: bool = False,
):
    M = int(x2.shape[0])
    K = int(x2.shape[1])
    N = int(w_q.shape[0])
    if M < m:
        pad = mx.zeros((m - M, K), dtype=x2.dtype)
        xm = mx.contiguous(mx.concatenate([x2, pad], axis=0))
    else:
        xm = mx.contiguous(x2)
    k_parts = 2 if N >= 4096 else 4
    kernel = _build_ksplit_kernel(
        m, bits, group_size, x2.dtype, k_parts=k_parts, dual=dual,
        kconst=K if kconst else 0,
    )
    (y,) = kernel(
        inputs=[xm, w_q, scales, biases, K, N],
        template=[("T", x2.dtype)],
        grid=(32 * k_parts, N // 4, 1),
        threadgroup=(32 * k_parts, 1, 1),
        output_shapes=[(m, N)],
        output_dtypes=[x2.dtype],
    )
    return y[:M, :] if M < m else y


def vk_eligible_ksplit(m: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    # K % 64 also guarantees the 6-bit hexpack count (K/16) splits evenly
    # across k_parts in {2, 4}.
    return (
        int(bits) in (4, 6, 8)
        and int(group_size) in (32, 64, 128)
        and dtype in (mx.bfloat16, mx.float16)
        and 4 <= int(m) <= 6
        and int(K) % 64 == 0
        and int(N) % 4 == 0
    )


def vk_qmm_m4_impl(impl: str, x2, w_q, scales, biases, *, bits: int = 4, group_size: int = 64):
    """Dispatch a named m4 implementation (kernel-race plumbing).

    vk        clean-room split-K (the port's proven geometry, our code)
    vk_u2     split-K + dual-pack pipelining
    vk_hybrid split-K everywhere, msg/oct tile for huge-N (lm_head)
    oct       msg geometry everywhere (falsified in-context 2026-07-02;
              kept for diagnostics)
    """
    N = int(w_q.shape[0])
    if impl == "oct":
        return _run(4, x2, w_q, scales, biases, bits=bits, group_size=group_size, nsg=_M4_NSG)
    if impl == "twin":
        # H1 twin-tile: 2 independent barrier-free simdgroups (64 threads,
        # grid N/8) — the port's kp2 scheduling footprint without its barrier.
        return _run(4, x2, w_q, scales, biases, bits=bits, group_size=group_size, nsg=2)
    if impl == "vk_hybrid" and N >= 100000 and N % (4 * _M4_NSG) == 0:
        return _run(4, x2, w_q, scales, biases, bits=bits, group_size=group_size, nsg=_M4_NSG)
    dual = impl == "vk_u2"
    kconst = impl in ("vk_k", "vk_hybrid")
    return _run_ksplit(
        4, x2, w_q, scales, biases,
        bits=bits, group_size=group_size, dual=dual, kconst=kconst,
    )
