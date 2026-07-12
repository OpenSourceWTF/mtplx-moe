"""Packed-row GQA verify attention for tiny query windows over long KV.

Why this kernel exists (2026-07-05 speed war, Lane A):

MLX's fused SDPA routes q_len 2..8 to ``sdpa_vector_2pass`` with a
``(32, gqa, q_len)`` threadgroup — every (gqa x q_len) simdgroup issues its
own device loads for the *same* KV rows.  At the Qwen3.6-27B verify shape
(Hq=24, Hk=4, D=256, q_len=4) that measured ~160 GB/s useful KV bandwidth at
128k context vs ~387 GB/s for the q=1 vector kernel — the single largest
term of the long-context decode wall (49 ms of a 152 ms verify call).

This kernel keeps the q=1 thread topology — threadgroup ``(32, gqa, 1)``,
one KV block-lane streamed exactly once per simdgroup — and carries all
``q_len`` query rows in registers.  The per-row score reductions are packed
into a single ``float4`` simd_shuffle_xor butterfly so the shuffle-chain
latency is paid once per KV row instead of ``q_len`` times.  Measured
2026-07-05 (M5 Max, bf16, per-iteration eval): 207 GB/s useful at 128k
(+27% vs fused q=4), 185 GB/s at 65k (+22%), ties-or-wins at 16k.  Max
|diff| vs an fp32 reference: 0.0011 (stock fused: 0.0006) — same numeric
class; acceptance-decision parity is gated separately before any default.

Contract:
- ``queries``: ``[1, Hq, q_len, D]`` bf16/fp16, ``2 <= q_len <= 4``.
- ``keys``/``values``: the FULL capacity-padded contiguous buffers
  (``KVCache.keys`` / ``TensorOffsetKVCache.cache[0]``) — never the
  ``[..., :offset, :]`` views, which would force a whole-buffer copy.
- ``offset``: rows in use (python int or int32 scalar ``mx.array`` for
  compiled graphs).  Rows ``>= offset`` are never read.
- Semantics: tail-causal — query row ``j`` attends to rows
  ``n <= offset - q_len + j`` — identical to ``make_mask``'s tail window
  and to fused SDPA's ``mask="causal"`` with a KV cache.
"""

from __future__ import annotations

from functools import lru_cache
import os

import mlx.core as mx

from .sdpa_2pass_paged import _paged_reduce_kernel


def _env_blocks_override() -> int:
    raw = (os.environ.get("MTPLX_GQA_PACKED_SDPA_BLOCKS") or "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _blocks_for_capacity(capacity: int) -> int:
    """Block-lane count tuned on M5 Max (2026-07-05 race, proto5)."""

    override = _env_blocks_override()
    if override:
        return override
    if capacity >= 65536:
        return 1024
    if capacity >= 16384:
        return 512
    return 256


@lru_cache(maxsize=None)
def _packed_partials_kernel():
    if not mx.metal.is_available():
        return None

    # QL is a template constant in [2, 4]; the float4 lanes above QL are
    # dead weight (zero query, outputs never written back).
    source = """
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;

        typedef float U;

        const int kv_head_idx = threadgroup_position_in_grid.x;
        const int block_idx = threadgroup_position_in_grid.z;
        const int gqa_idx = thread_position_in_threadgroup.y;
        const int simd_lid = thread_index_in_simdgroup;
        const int n_kv = static_cast<int>(offset[0]);

        const int q_head_idx = kv_head_idx * GQA_F + gqa_idx;

        thread U q[QL][qk_per_thread];
        thread U o[QL][v_per_thread];
        float4 max_score = Limits<U>::finite_min;
        float4 sum_exp = 0.0f;

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

        const device InT* k_ptr = keys
            + (size_t)kv_head_idx * k_head_seq * D
            + (size_t)block_idx * D
            + simd_lid * qk_per_thread;
        const device InT* v_ptr = values
            + (size_t)kv_head_idx * v_head_seq * D
            + (size_t)block_idx * D
            + simd_lid * v_per_thread;

        // Rows visible to every query row run branch-free; the last QL-1
        // rows take the per-row causal predicate in the tail loop below.
        const int n_full = n_kv - QL;

        for (int n = block_idx; n <= n_full; n += blocks) {
            U k_vec[qk_per_thread];
            for (int i = 0; i < qk_per_thread; ++i) {
                k_vec[i] = static_cast<U>(k_ptr[i]);
            }
            float4 score = 0.0f;
            for (int i = 0; i < qk_per_thread; ++i) {
                score.x += q[0][i] * k_vec[i];
                if (QL > 1) score.y += q[1][i] * k_vec[i];
                if (QL > 2) score.z += q[2][i] * k_vec[i];
                if (QL > 3) score.w += q[3][i] * k_vec[i];
            }
            for (int off = 16; off > 0; off >>= 1) {
                score += simd_shuffle_xor(score, off);
            }
            float4 new_max = metal::max(max_score, score);
            float4 factor = fast::exp(max_score - new_max);
            float4 exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp = sum_exp * factor + exp_score;
            for (int i = 0; i < v_per_thread; ++i) {
                const U v = static_cast<U>(v_ptr[i]);
                o[0][i] = o[0][i] * factor.x + exp_score.x * v;
                if (QL > 1) o[1][i] = o[1][i] * factor.y + exp_score.y * v;
                if (QL > 2) o[2][i] = o[2][i] * factor.z + exp_score.z * v;
                if (QL > 3) o[3][i] = o[3][i] * factor.w + exp_score.w * v;
            }
            k_ptr += (size_t)blocks * D;
            v_ptr += (size_t)blocks * D;
        }

        {
            const int first = n_full + 1;
            int n0 = block_idx;
            if (n0 < first) {
                const int steps = (first - n0 + blocks - 1) / blocks;
                n0 += steps * blocks;
            }
            const device InT* k_tail = keys
                + (size_t)kv_head_idx * k_head_seq * D
                + (size_t)n0 * D + simd_lid * qk_per_thread;
            const device InT* v_tail = values
                + (size_t)kv_head_idx * v_head_seq * D
                + (size_t)n0 * D + simd_lid * v_per_thread;
            for (int n = n0; n < n_kv; n += blocks) {
                U k_vec[qk_per_thread];
                for (int i = 0; i < qk_per_thread; ++i) {
                    k_vec[i] = static_cast<U>(k_tail[i]);
                }
                float4 score = 0.0f;
                for (int i = 0; i < qk_per_thread; ++i) {
                    score.x += q[0][i] * k_vec[i];
                    if (QL > 1) score.y += q[1][i] * k_vec[i];
                    if (QL > 2) score.z += q[2][i] * k_vec[i];
                    if (QL > 3) score.w += q[3][i] * k_vec[i];
                }
                for (int off = 16; off > 0; off >>= 1) {
                    score += simd_shuffle_xor(score, off);
                }
                // Query row j attends to n iff n <= n_kv - QL + j.
                float4 vis;
                vis.x = (n <= n_kv - QL + 0) ? 1.0f : 0.0f;
                vis.y = (QL > 1 && n <= n_kv - QL + 1) ? 1.0f : 0.0f;
                vis.z = (QL > 2 && n <= n_kv - QL + 2) ? 1.0f : 0.0f;
                vis.w = (QL > 3 && n <= n_kv - QL + 3) ? 1.0f : 0.0f;
                score = score * vis + (1.0f - vis) * Limits<U>::finite_min;
                float4 new_max = metal::max(max_score, score);
                float4 factor = fast::exp(max_score - new_max);
                float4 exp_score = fast::exp(score - new_max) * vis;
                max_score = new_max;
                sum_exp = sum_exp * factor + exp_score;
                for (int i = 0; i < v_per_thread; ++i) {
                    const U v = static_cast<U>(v_tail[i]);
                    o[0][i] = o[0][i] * factor.x + exp_score.x * v;
                    if (QL > 1) o[1][i] = o[1][i] * factor.y + exp_score.y * v;
                    if (QL > 2) o[2][i] = o[2][i] * factor.z + exp_score.z * v;
                    if (QL > 3) o[3][i] = o[3][i] * factor.w + exp_score.w * v;
                }
                k_tail += (size_t)blocks * D;
                v_tail += (size_t)blocks * D;
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
                sums[o_offset * blocks + block_idx] = sum_exp[j];
                maxs[o_offset * blocks + block_idx] = max_score[j];
            }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_sdpa_gqa_packed_partials",
        input_names=[
            "queries",
            "keys",
            "values",
            "offset",
            "k_head_seq",
            "v_head_seq",
            "scale",
            "blocks",
        ],
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


def sdpa_gqa_packed_tail(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    offset: int | mx.array,
    scale: float,
    max_q_len: int = 4,
) -> mx.array | None:
    """Tail-causal SDPA over the first ``offset`` rows of full KV buffers.

    Returns ``None`` when the shape/dtype contract is not met so callers can
    fall back to the stock fused path.
    """

    if not mx.metal.is_available():
        return None
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return None
    bsz, hq, q_len, d = (int(x) for x in queries.shape)
    if bsz != 1:
        return None
    if q_len < 2 or q_len > min(4, int(max_q_len)):
        return None
    hk = int(keys.shape[1])
    capacity = int(keys.shape[2])
    if int(values.shape[1]) != hk or int(values.shape[2]) != capacity:
        return None
    kd = int(keys.shape[3])
    vdim = int(values.shape[3])
    if kd != d or vdim != d:
        return None
    if d not in (64, 96, 128, 256):
        return None
    if hk <= 0 or hq % hk:
        return None
    gqa_factor = hq // hk
    if 32 * gqa_factor > 1024:
        return None
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return None
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return None
    # NOTE: callers must pass the whole allocated buffers (contiguous by
    # construction), never `[..., :offset, :]` views. MLX python exposes no
    # contiguity flag to assert on; a sliced view would still be CORRECT
    # (metal_kernel's ensure_row_contiguous copies it) but would silently
    # reintroduce the whole-buffer copy this kernel exists to avoid.

    if isinstance(offset, mx.array):
        if offset.size != 1:
            return None
        offset_arr = offset.astype(mx.int32).reshape(1)
    else:
        offset_int = int(offset)
        if offset_int <= 0 or offset_int > capacity:
            return None
        offset_arr = mx.array([offset_int], dtype=mx.int32)

    blocks = _blocks_for_capacity(capacity)
    if blocks <= 0 or blocks % 32:
        return None

    kernel = _packed_partials_kernel()
    reduce_kernel = _paged_reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return None

    partial_shape = (bsz, hq, q_len, blocks, vdim)
    stats_shape = (bsz, hq, q_len, blocks)
    partials, sums, maxs = kernel(
        inputs=[
            queries,
            keys,
            values,
            offset_arr,
            capacity,
            capacity,
            float(scale),
            int(blocks),
        ],
        template=[
            ("InT", queries.dtype),
            ("D", d),
            ("V", vdim),
            ("GQA_F", gqa_factor),
            ("QL", q_len),
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
            ("V", vdim),
        ],
        grid=(bsz * hq * 1024, q_len, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )
    return out
