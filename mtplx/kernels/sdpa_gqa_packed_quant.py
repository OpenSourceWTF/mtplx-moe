"""Dequant-in-flight packed GQA attention over q8/q4 KV (2026-08-25).

The product's KV-quant lane stores rowwise-symmetric quantized KV
(kv_quant.quantize_symmetric): int8 (q8) or nibble-packed uint8 (q4) with ONE
fp32 scale per row. The existing consumption path dequantizes through generic
ops and runs the walk at ~28GB/s — 2.5x SLOWER than bf16 (kv_walk_bench
receipt). This kernel walks the PACKED buffers directly and dequantizes in
registers, so the memory traffic drops to 0.51x (q8) / 0.26x (q4) of bf16.

Topology is byte-for-byte the proven sdpa_gqa_packed_tail one-bank+two-bank
design (one simdgroup per KV head, float4 query banks, blocks-strided walk,
fp32 LSE partials; the reduce kernel is REUSED from the bf16 module). Only the
K/V load path differs:

- q8: per-thread 8x int8 loads; the row dot runs on raw integer values in
  float and the fp32 row scale multiplies the SCORE once per row (1 mult
  instead of 8).
- q4: per-thread 4x uint8 loads, nibbles unpacked as (b & 0xF) - 8 and
  (b >> 4) - 8 (matching kv_quant's even|odd<<4 packing), same fold.
- V: the row scale folds into the exp-weight before the FMA loop.

Exactness contract: identical math to dequantize_symmetric + the bf16 kernel
in fp32 accumulation; the test suite pins it against that reference.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .sdpa_gqa_packed import (  # shared, proven pieces
    _bail,
    _blocks_for_capacity,
    _paged_reduce_kernel,
)


@lru_cache(maxsize=None)
def _quant_partials_kernel():
    if not mx.metal.is_available():
        return None

    # KBITS in {8, 4}. QL in [1, 8]: rows 0..3 on bank 1, 4..7 on bank 2
    # (banks compile out below their row counts exactly like the bf16 kernel).
    source = """
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;
        constexpr int QH = (QL > 4) ? (QL - 4) : 0;
        constexpr int k_bytes_per_thread =
            (KBITS == 8) ? qk_per_thread : (qk_per_thread / 2);
        constexpr int v_bytes_per_thread =
            (KBITS == 8) ? v_per_thread : (v_per_thread / 2);
        constexpr int KROW = (KBITS == 8) ? D : (D / 2);

        typedef float U;

        const int kv_head_idx = threadgroup_position_in_grid.x;
        const int block_idx = threadgroup_position_in_grid.z;
        const int gqa_idx = thread_position_in_threadgroup.y;
        const int simd_lid = thread_index_in_simdgroup;
        const int n_kv = static_cast<int>(offset[0]);

        const int q_head_idx = kv_head_idx * GQA_F + gqa_idx;

        thread U q[QL][qk_per_thread];
        thread U o[QL][v_per_thread];
        float4 max_score = Limits<float>::finite_min;
        float4 sum_exp = 0.0f;
        float4 max_score2 = Limits<float>::finite_min;
        float4 sum_exp2 = 0.0f;

        for (int j = 0; j < QL; ++j) {
            const device InT* q_ptr = queries
                + (q_head_idx * QL + j) * D + simd_lid * qk_per_thread;
            for (int i = 0; i < qk_per_thread; ++i) {
                q[j][i] = static_cast<U>(scale) * static_cast<U>(q_ptr[i]);
            }
            for (int i = 0; i < v_per_thread; ++i) {
                o[j][i] = 0.0f;
            }
        }

        const device uint8_t* k_ptr = k_q
            + (size_t)kv_head_idx * k_head_seq * KROW
            + (size_t)block_idx * KROW
            + simd_lid * k_bytes_per_thread;
        const device uint8_t* v_ptr = v_q
            + (size_t)kv_head_idx * v_head_seq * KROW
            + (size_t)block_idx * KROW
            + simd_lid * v_bytes_per_thread;
        const device float* ks_ptr = k_scale
            + (size_t)kv_head_idx * k_head_seq + block_idx;
        const device float* vs_ptr = v_scale
            + (size_t)kv_head_idx * v_head_seq + block_idx;

        const int n_full = n_kv - QL;

        #define MTPLX_LOAD_KVEC(dst, src) \\
            if (KBITS == 8) { \\
                const device uchar4* s4 = \\
                    reinterpret_cast<const device uchar4*>(src); \\
                for (int i = 0; i < qk_per_thread / 4; ++i) { \\
                    const char4 c = as_type<char4>(s4[i]); \\
                    dst[4 * i + 0] = static_cast<U>(c.x); \\
                    dst[4 * i + 1] = static_cast<U>(c.y); \\
                    dst[4 * i + 2] = static_cast<U>(c.z); \\
                    dst[4 * i + 3] = static_cast<U>(c.w); \\
                } \\
            } else { \\
                const device uchar4* s4 = \\
                    reinterpret_cast<const device uchar4*>(src); \\
                const uchar4 b = s4[0]; \\
                dst[0] = static_cast<U>(static_cast<int>(b.x & 0x0F) - 8); \\
                dst[1] = static_cast<U>(static_cast<int>(b.x >> 4) - 8); \\
                dst[2] = static_cast<U>(static_cast<int>(b.y & 0x0F) - 8); \\
                dst[3] = static_cast<U>(static_cast<int>(b.y >> 4) - 8); \\
                dst[4] = static_cast<U>(static_cast<int>(b.z & 0x0F) - 8); \\
                dst[5] = static_cast<U>(static_cast<int>(b.z >> 4) - 8); \\
                dst[6] = static_cast<U>(static_cast<int>(b.w & 0x0F) - 8); \\
                dst[7] = static_cast<U>(static_cast<int>(b.w >> 4) - 8); \\
            }

        for (int n = block_idx; n <= n_full; n += blocks) {
            U k_vec[qk_per_thread];
            MTPLX_LOAD_KVEC(k_vec, k_ptr)
            const U ks = static_cast<U>(*ks_ptr);
            float4 score = 0.0f;
            float4 score2 = 0.0f;
            for (int i = 0; i < qk_per_thread; ++i) {
                score.x += q[0][i] * k_vec[i];
                if (QL > 1) score.y += q[1][i] * k_vec[i];
                if (QL > 2) score.z += q[2][i] * k_vec[i];
                if (QL > 3) score.w += q[3][i] * k_vec[i];
                if (QH > 0) score2.x += q[4][i] * k_vec[i];
                if (QH > 1) score2.y += q[5][i] * k_vec[i];
                if (QH > 2) score2.z += q[6][i] * k_vec[i];
                if (QH > 3) score2.w += q[7][i] * k_vec[i];
            }
            score *= ks;
            if (QH > 0) score2 *= ks;
            for (int off = 16; off > 0; off >>= 1) {
                score += simd_shuffle_xor(score, off);
                if (QH > 0) score2 += simd_shuffle_xor(score2, off);
            }
            float4 new_max = metal::max(max_score, score);
            float4 factor = fast::exp(max_score - new_max);
            float4 exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp = sum_exp * factor + exp_score;
            float4 factor2 = 1.0f;
            float4 exp_score2 = 0.0f;
            if (QH > 0) {
                float4 new_max2 = metal::max(max_score2, score2);
                factor2 = fast::exp(max_score2 - new_max2);
                exp_score2 = fast::exp(score2 - new_max2);
                max_score2 = new_max2;
                sum_exp2 = sum_exp2 * factor2 + exp_score2;
            }
            const U vs = static_cast<U>(*vs_ptr);
            {
                const float4 es = exp_score * vs;
                const float4 es2 = exp_score2 * vs;
                U v_vec[v_per_thread];
                MTPLX_LOAD_KVEC(v_vec, v_ptr)
                for (int i = 0; i < v_per_thread; ++i) {
                    const U v = v_vec[i];
                    o[0][i] = o[0][i] * factor.x + es.x * v;
                    if (QL > 1) o[1][i] = o[1][i] * factor.y + es.y * v;
                    if (QL > 2) o[2][i] = o[2][i] * factor.z + es.z * v;
                    if (QL > 3) o[3][i] = o[3][i] * factor.w + es.w * v;
                    if (QH > 0) o[4][i] = o[4][i] * factor2.x + es2.x * v;
                    if (QH > 1) o[5][i] = o[5][i] * factor2.y + es2.y * v;
                    if (QH > 2) o[6][i] = o[6][i] * factor2.z + es2.z * v;
                    if (QH > 3) o[7][i] = o[7][i] * factor2.w + es2.w * v;
                }
            }
            k_ptr += (size_t)blocks * KROW;
            v_ptr += (size_t)blocks * KROW;
            ks_ptr += blocks;
            vs_ptr += blocks;
        }

        {
            const int first = n_full + 1;
            int n0 = block_idx;
            if (n0 < first) {
                const int steps = (first - n0 + blocks - 1) / blocks;
                n0 += steps * blocks;
            }
            const device uint8_t* k_tail = k_q
                + (size_t)kv_head_idx * k_head_seq * KROW
                + (size_t)n0 * KROW + simd_lid * k_bytes_per_thread;
            const device uint8_t* v_tail = v_q
                + (size_t)kv_head_idx * v_head_seq * KROW
                + (size_t)n0 * KROW + simd_lid * v_bytes_per_thread;
            const device float* ks_tail = k_scale
                + (size_t)kv_head_idx * k_head_seq + n0;
            const device float* vs_tail = v_scale
                + (size_t)kv_head_idx * v_head_seq + n0;
            for (int n = n0; n < n_kv; n += blocks) {
                U k_vec[qk_per_thread];
                MTPLX_LOAD_KVEC(k_vec, k_tail)
                const U ks = static_cast<U>(*ks_tail);
                float4 score = 0.0f;
                float4 score2 = 0.0f;
                for (int i = 0; i < qk_per_thread; ++i) {
                    score.x += q[0][i] * k_vec[i];
                    if (QL > 1) score.y += q[1][i] * k_vec[i];
                    if (QL > 2) score.z += q[2][i] * k_vec[i];
                    if (QL > 3) score.w += q[3][i] * k_vec[i];
                    if (QH > 0) score2.x += q[4][i] * k_vec[i];
                    if (QH > 1) score2.y += q[5][i] * k_vec[i];
                    if (QH > 2) score2.z += q[6][i] * k_vec[i];
                    if (QH > 3) score2.w += q[7][i] * k_vec[i];
                }
                score *= ks;
                if (QH > 0) score2 *= ks;
                for (int off = 16; off > 0; off >>= 1) {
                    score += simd_shuffle_xor(score, off);
                    if (QH > 0) score2 += simd_shuffle_xor(score2, off);
                }
                float4 vis;
                vis.x = (n <= n_kv - QL + 0) ? 1.0f : 0.0f;
                vis.y = (QL > 1 && n <= n_kv - QL + 1) ? 1.0f : 0.0f;
                vis.z = (QL > 2 && n <= n_kv - QL + 2) ? 1.0f : 0.0f;
                vis.w = (QL > 3 && n <= n_kv - QL + 3) ? 1.0f : 0.0f;
                score = score * vis + (1.0f - vis) * Limits<float>::finite_min;
                float4 new_max = metal::max(max_score, score);
                float4 factor = fast::exp(max_score - new_max);
                float4 exp_score = fast::exp(score - new_max) * vis;
                max_score = new_max;
                sum_exp = sum_exp * factor + exp_score;
                float4 factor2 = 1.0f;
                float4 exp_score2 = 0.0f;
                if (QH > 0) {
                    float4 vis2;
                    vis2.x = (n <= n_kv - QL + 4) ? 1.0f : 0.0f;
                    vis2.y = (QH > 1 && n <= n_kv - QL + 5) ? 1.0f : 0.0f;
                    vis2.z = (QH > 2 && n <= n_kv - QL + 6) ? 1.0f : 0.0f;
                    vis2.w = (QH > 3 && n <= n_kv - QL + 7) ? 1.0f : 0.0f;
                    score2 = score2 * vis2
                        + (1.0f - vis2) * Limits<float>::finite_min;
                    float4 new_max2 = metal::max(max_score2, score2);
                    factor2 = fast::exp(max_score2 - new_max2);
                    exp_score2 = fast::exp(score2 - new_max2) * vis2;
                    max_score2 = new_max2;
                    sum_exp2 = sum_exp2 * factor2 + exp_score2;
                }
                const U vs = static_cast<U>(*vs_tail);
                {
                    const float4 es = exp_score * vs;
                    const float4 es2 = exp_score2 * vs;
                    U v_vec[v_per_thread];
                    MTPLX_LOAD_KVEC(v_vec, v_tail)
                    for (int i = 0; i < v_per_thread; ++i) {
                        const U v = v_vec[i];
                        o[0][i] = o[0][i] * factor.x + es.x * v;
                        if (QL > 1) o[1][i] = o[1][i] * factor.y + es.y * v;
                        if (QL > 2) o[2][i] = o[2][i] * factor.z + es.z * v;
                        if (QL > 3) o[3][i] = o[3][i] * factor.w + es.w * v;
                        if (QH > 0) o[4][i] = o[4][i] * factor2.x + es2.x * v;
                        if (QH > 1) o[5][i] = o[5][i] * factor2.y + es2.y * v;
                        if (QH > 2) o[6][i] = o[6][i] * factor2.z + es2.z * v;
                        if (QH > 3) o[7][i] = o[7][i] * factor2.w + es2.w * v;
                    }
                }
                k_tail += (size_t)blocks * KROW;
                v_tail += (size_t)blocks * KROW;
                ks_tail += blocks;
                vs_tail += blocks;
            }
        }

        for (int j = 0; j < QL; ++j) {
            const int o_offset = q_head_idx * QL + j;
            device InT* p = partials
                + ((size_t)o_offset * blocks + block_idx) * V
                + simd_lid * v_per_thread;
            for (int i = 0; i < v_per_thread; ++i) {
                p[i] = static_cast<InT>(o[j][i]);
            }
            if (simd_lid == 0) {
                const float se = (j < 4) ? sum_exp[j] : sum_exp2[j - 4];
                const float ms = (j < 4) ? max_score[j] : max_score2[j - 4];
                sums[o_offset * blocks + block_idx] = se;
                maxs[o_offset * blocks + block_idx] = ms;
            }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_sdpa_gqa_packed_quant_partials",
        input_names=[
            "queries",
            "k_q",
            "k_scale",
            "v_q",
            "v_scale",
            "offset",
            "k_head_seq",
            "v_head_seq",
            "scale",
            "blocks",
        ],
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


def sdpa_gqa_packed_tail_quant(
    *,
    queries,
    k_q,
    k_scale,
    v_q,
    v_scale,
    offset,
    scale,
    bits: int,
    max_q_len: int = 8,
):
    """Packed-quant GQA tail attention. Returns None (bail) on contract miss.

    queries: (1, HQ, QL, D) bf16/fp16. k_q/v_q: (1, HKV, capacity, D) int8 for
    q8 or (1, HKV, capacity, D//2) uint8 for q4 (kv_quant even|odd<<4 packing).
    k_scale/v_scale: (1, HKV, capacity, 1) fp32 rowwise. Same whole-buffer
    contract as the bf16 kernel: pass allocated buffers, never sliced views.
    """
    bits = int(bits)
    if bits not in (8, 4):
        return _bail("quant_bits")
    bsz, hq, q_len, d = (int(x) for x in queries.shape)
    if bsz != 1:
        return _bail("batch_size")
    if q_len < 1 or q_len > min(8, int(max_q_len)):
        return _bail("q_len")
    hk = int(k_q.shape[1])
    capacity = int(k_q.shape[2])
    if int(v_q.shape[1]) != hk or int(v_q.shape[2]) != capacity:
        return _bail("kv_layout_mismatch")
    packed = d if bits == 8 else d // 2
    if int(k_q.shape[3]) != packed or int(v_q.shape[3]) != packed:
        return _bail("packed_dim_mismatch")
    for s in (k_scale, v_scale):
        if int(s.shape[1]) != hk or int(s.shape[2]) != capacity:
            return _bail("scale_layout_mismatch")
        if s.dtype != mx.float32:
            return _bail("scale_dtype")
    if d not in (64, 128, 256):
        return _bail("head_dim_unsupported")
    if hk <= 0 or hq % hk:
        return _bail("gqa_heads")
    gqa_factor = hq // hk
    if 32 * gqa_factor > 1024:
        return _bail("threadgroup_width")
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return _bail("query_dtype")
    expect_kv = mx.int8 if bits == 8 else mx.uint8
    if k_q.dtype != expect_kv or v_q.dtype != expect_kv:
        return _bail("kv_dtype")

    if isinstance(offset, mx.array):
        if offset.size != 1:
            return _bail("offset_shape")
        offset_arr = offset.astype(mx.int32).reshape(1)
    else:
        offset_int = int(offset)
        if offset_int <= 0 or offset_int > capacity:
            return _bail("offset_range")
        offset_arr = mx.array([offset_int], dtype=mx.int32)

    blocks = _blocks_for_capacity(capacity)
    if blocks <= 0 or blocks % 32:
        return _bail("blocks_geometry")

    kernel = _quant_partials_kernel()
    reduce_kernel = _paged_reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return _bail("kernel_unavailable")

    # int8 buffers reach Metal as uint8 pointers; the kernel casts per KBITS.
    k_q_view = k_q.view(mx.uint8) if bits == 8 else k_q
    v_q_view = v_q.view(mx.uint8) if bits == 8 else v_q

    partial_shape = (bsz, hq, q_len, blocks, d)
    stats_shape = (bsz, hq, q_len, blocks)
    partials, sums, maxs = kernel(
        inputs=[
            queries,
            k_q_view,
            k_scale,
            v_q_view,
            v_scale,
            offset_arr,
            capacity,
            capacity,
            float(scale),
            int(blocks),
        ],
        template=[
            ("InT", queries.dtype),
            ("D", d),
            ("V", d),
            ("GQA_F", gqa_factor),
            ("QL", q_len),
            ("KBITS", bits),
        ],
        grid=(hk * 32, gqa_factor, blocks),
        threadgroup=(32, gqa_factor, 1),
        output_shapes=[partial_shape, stats_shape, stats_shape],
        output_dtypes=[queries.dtype, mx.float32, mx.float32],
    )

    (out,) = reduce_kernel(
        inputs=[partials, sums, maxs, int(blocks)],
        template=[
            ("InT", queries.dtype),
            ("V", d),
        ],
        grid=(bsz * hq * 1024, q_len, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )
    return out
