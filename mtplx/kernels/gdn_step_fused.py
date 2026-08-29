"""One-dispatch GDN decode step: conv+silu+l2norm + g/beta + delta + gated norm.

The 2026-08-27 dispatch ladder's slice 3. At qL=1 the GDN layer today runs
in_proj GEMV -> mtplx_gdn_conv_norm -> (sigmoid/compute_g elementwise) ->
gated_delta_step -> SigmoidRMSNormGated elementwise -> out_proj GEMV: four
kernel/elementwise dispatches BETWEEN the two library GEMVs, all on one
dependent chain. This kernel replaces those four with ONE dispatch, keeping
both GEMVs as library calls per the kernel-shape law (receipts: in_proj +3%,
gate_up +2.7% merges won; hand-rolled GEMV replacements lost 3-for-3).

Geometry (family): conv_dim 10240 = q 16x128 | k 16x128 | v 48x128,
Hk=16, Hv=48, Dk=Dv=128, conv kernel 4, sigmoid output gate, fp32 state.

One threadgroup per value head (48 TGs x 256 threads):
  phase 1  conv over the head's 384 channels (its q/k head shared by 3 TGs,
           recomputed — 4-tap dot, trivial; conv-state roll written by the
           hv%3==0 owner for q/k, uniquely for v) + silu
  phase 1b per-head l2norm on q/k (q pre-scaled by Dk^-0.5), g/beta from
           A_log/dt_bias/a/b (each thread redundantly, register-cheap)
  phase 2  delta recurrence: 8 simdgroups x 16 rows; per row the 32 lanes
           hold state[dv, :] (4 floats each), mirroring mlx_lm's
           gated_delta_step exactly (decay -> kv_mem -> delta -> update ->
           out), state written back fp32
  phase 3  SigmoidRMSNormGated epilogue with the module's exact rounding:
           out rounded to T (the stock kernel emits InT), rms_norm in fp32,
           normed rounded to T, then sigmoid(z)*x in fp32, rounded to T

Verify rows (S>1) and ragged/masked shapes never reach this kernel — the
module gate mirrors mtplx_gdn_conv_norm's, so capture-commit's stash path
(which needs the stock chain's materialized rows) is untouched.
"""

from functools import lru_cache

import mlx.core as mx

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_SRC = """
    constexpr int C = 10240;                 // conv channels
    constexpr int QK = 2048;                 // q width == k width
    constexpr int HD = 128;                  // head dim (Dk == Dv)
    constexpr int HV_PER_HK = 3;             // 48 v heads / 16 k heads
    constexpr float INV_SCALE = 0.08838834764831845f;  // 128^-0.5

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint hv = threadgroup_position_in_grid.z;
    const uint hk = hv / HV_PER_HK;

    threadgroup float qn[HD];
    threadgroup float kn[HD];
    threadgroup float vraw[HD];
    threadgroup float outv[HD];
    threadgroup float part[8];

    // ---- phase 1: conv + silu over this head's 384 channels ----
    for (uint i = tid; i < 384; i += 256) {
      uint g;
      if (i < HD) {
        g = hk * HD + i;                       // q section
      } else if (i < 2 * HD) {
        g = QK + hk * HD + (i - HD);           // k section
      } else {
        g = 2 * QK + hv * HD + (i - 2 * HD);   // v section
      }
      const float acc =
          (float)cw[g * 4 + 0] * (float)state[0 * C + g] +
          (float)cw[g * 4 + 1] * (float)state[1 * C + g] +
          (float)cw[g * 4 + 2] * (float)state[2 * C + g] +
          (float)cw[g * 4 + 3] * (float)xnew[g];
      const float sv = acc / (1.0f + metal::exp(-acc));

      // rolled conv state: v channels owned uniquely; q/k written by the
      // hv%3==0 sibling so each channel has exactly one writer.
      if (i >= 2 * HD || (hv % HV_PER_HK) == 0) {
        state_out[0 * C + g] = state[1 * C + g];
        state_out[1 * C + g] = state[2 * C + g];
        state_out[2 * C + g] = xnew[g];
      }

      if (i < HD) {
        qn[i] = sv;
      } else if (i < 2 * HD) {
        kn[i - HD] = sv;
      } else {
        // stock path rounds v to T at the conv-kernel boundary
        vraw[i - 2 * HD] = (float)(T)sv;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- phase 1b: per-head l2norm (fp32, eps on sum of squares) ----
    {
      float p = 0.0f;
      if (tid < HD) {
        p = qn[tid] * qn[tid];
      } else if (tid < 2 * HD) {
        p = kn[tid - HD] * kn[tid - HD];
      }
      p = simd_sum(p);
      if (lane == 0) part[sg] = p;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float qsum = part[0] + part[1] + part[2] + part[3];
      const float ksum = part[4] + part[5] + part[6] + part[7];
      const float qinv = metal::rsqrt(qsum + 1e-6f) * INV_SCALE;
      const float kinv = metal::rsqrt(ksum + 1e-6f);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < HD) {
        qn[tid] = (float)(T)(qn[tid] * qinv);
      } else if (tid < 2 * HD) {
        kn[tid - HD] = (float)(T)(kn[tid - HD] * kinv);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- g / beta (redundant per thread; matches compute_g + sigmoid) ----
    const float a_v = (float)a_row[hv] + (float)dt_bias[hv];
    // softplus = logaddexp(x, 0) = max(x,0) + log1p(exp(-|x|))
    const float sp = metal::max(a_v, 0.0f)
                   + metal::log(1.0f + metal::exp(-metal::abs(a_v)));
    const float g_dec = metal::exp(-metal::exp((float)A_log[hv]) * sp);
    const float b_v = (float)b_row[hv];
    const float beta = 1.0f / (1.0f + metal::exp(-b_v));

    // ---- phase 2: delta recurrence, one simdgroup per row, stride 8 ----
    for (uint dv = sg; dv < HD; dv += 8) {
      const uint srow = (hv * HD + dv) * HD;
      float s0 = (float)dstate[srow + 4 * lane + 0] * g_dec;
      float s1 = (float)dstate[srow + 4 * lane + 1] * g_dec;
      float s2 = (float)dstate[srow + 4 * lane + 2] * g_dec;
      float s3 = (float)dstate[srow + 4 * lane + 3] * g_dec;
      const float k0 = kn[4 * lane + 0];
      const float k1 = kn[4 * lane + 1];
      const float k2 = kn[4 * lane + 2];
      const float k3 = kn[4 * lane + 3];
      float kv_mem = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
      kv_mem = simd_sum(kv_mem);
      const float delta = (vraw[dv] - kv_mem) * beta;
      s0 += k0 * delta;
      s1 += k1 * delta;
      s2 += k2 * delta;
      s3 += k3 * delta;
      float out = s0 * qn[4 * lane + 0] + s1 * qn[4 * lane + 1]
                + s2 * qn[4 * lane + 2] + s3 * qn[4 * lane + 3];
      out = simd_sum(out);
      dstate_out[srow + 4 * lane + 0] = (StT)s0;
      dstate_out[srow + 4 * lane + 1] = (StT)s1;
      dstate_out[srow + 4 * lane + 2] = (StT)s2;
      dstate_out[srow + 4 * lane + 3] = (StT)s3;
      if (lane == 0) {
        // the stock kernel emits InT here; keep that rounding
        outv[dv] = (float)(T)out;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- phase 3: SigmoidRMSNormGated epilogue ----
    {
      float e = 0.0f;
      if (tid < HD) e = outv[tid] * outv[tid];
      e = simd_sum(e);
      if (lane == 0 && sg < 4) part[sg] = e;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float mean = (part[0] + part[1] + part[2] + part[3]) / (float)HD;
      const float rinv = metal::rsqrt(mean + 1e-6f);
      if (tid < HD) {
        const float normed = (float)(T)(outv[tid] * rinv * (float)norm_w[tid]);
        const float zf = (float)z_row[hv * HD + tid];
        const float gate = 1.0f / (1.0f + metal::exp(-zf));
        y[hv * HD + tid] = (T)(gate * normed);
      }
    }
"""


@lru_cache(maxsize=None)
def _kernel(t_dtype: mx.Dtype, s_dtype: mx.Dtype):
    t_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(t_dtype, "unk")
    s_tag = {mx.float32: "f32", mx.bfloat16: "bf16"}.get(s_dtype, "unk")
    return mx.fast.metal_kernel(
        name=f"mtplx_gdn_step_fused_{t_tag}_{s_tag}",
        input_names=[
            "xnew",
            "z_row",
            "a_row",
            "b_row",
            "state",
            "cw",
            "A_log",
            "dt_bias",
            "dstate",
            "norm_w",
        ],
        output_names=["y", "state_out", "dstate_out"],
        header=_HEADER,
        source=_SRC,
    )


def fused_gdn_step(
    qkv_row: mx.array,
    z_row: mx.array,
    a_row: mx.array,
    b_row: mx.array,
    conv_state: mx.array,
    conv_w: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    delta_state: mx.array,
    norm_w: mx.array,
):
    """qkv_row [10240] (post in_proj, pre conv); z_row [6144]; a_row/b_row
    [48]; conv_state [3, 10240]; conv_w [10240, 4(, 1)]; delta_state
    [48, 128, 128] (fp32); norm_w [128]. Returns (y [6144] gated+normed,
    new conv_state [3, 10240], new delta_state [48, 128, 128])."""
    cw = conv_w.reshape(10240, 4)
    kern = _kernel(qkv_row.dtype, delta_state.dtype)
    y, ns, nds = kern(
        inputs=[
            qkv_row,
            z_row,
            a_row,
            b_row,
            conv_state.reshape(-1),
            cw.reshape(-1),
            A_log,
            dt_bias,
            delta_state.reshape(-1),
            norm_w,
        ],
        template=[("T", qkv_row.dtype), ("StT", delta_state.dtype)],
        grid=(256, 1, 48),
        threadgroup=(256, 1, 1),
        output_shapes=[(6144,), (3, 10240), (48, 128, 128)],
        output_dtypes=[qkv_row.dtype, qkv_row.dtype, delta_state.dtype],
    )
    return y, ns, nds
