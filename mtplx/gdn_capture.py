"""GDN state-capture verify helpers for Qwen3.5/Qwen3.6."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def _env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _cache_context_len(cache: Any) -> int:
    if cache is None:
        return 0
    best = 0
    for entry in cache:
        if entry is None:
            continue
        offset = getattr(entry, "offset", None)
        if isinstance(offset, mx.array):
            continue
        if offset is not None:
            best = max(best, int(offset or 0))
            continue
        size = getattr(entry, "size", None)
        if callable(size):
            try:
                best = max(best, int(size() or 0))
            except Exception:
                pass
    return best


def _target_layer_eval_every(context_len: int) -> int:
    schedule = os.environ.get("MTPLX_TARGET_LAYER_EVAL_SCHEDULE", "").strip()
    selected = 0
    if schedule:
        for part in schedule.replace(";", ",").split(","):
            item = part.strip()
            if not item:
                continue
            try:
                threshold_text, every_text = item.split(":", 1)
                threshold = int(threshold_text)
                every = int(every_text)
            except ValueError:
                continue
            if int(context_len) >= threshold:
                selected = max(0, every)
        return selected
    return int(os.environ.get("MTPLX_TARGET_LAYER_EVAL_EVERY", "0") or "0")


def _make_linear_conv1d_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto c_idx = thread_position_in_grid.x;
        auto b_idx = thread_position_in_grid.y;

        if (c_idx >= ConvDim) {
          return;
        }

        for (int t = 0; t < T; ++t) {
          auto parent_idx = t - 1;

          float acc = 0.0f;
          for (int k = 0; k < Keep; ++k) {
            float x;
            if (parent_idx < 0) {
              x = static_cast<float>(
                base_conv_state[(b_idx * Keep + k) * ConvDim + c_idx]
              );
            } else {
              x = static_cast<float>(
                conv_states[
                  (((b_idx * T + parent_idx) * Keep + k) * ConvDim) + c_idx
                ]
              );
            }
            auto w = static_cast<float>(conv_weight[c_idx * (Keep + 1) + k]);
            acc += x * w;
          }

          auto qkv_t = qkv + (b_idx * T + t) * ConvDim;
          acc += static_cast<float>(qkv_t[c_idx])
            * static_cast<float>(conv_weight[c_idx * (Keep + 1) + Keep]);

          conv_out[(b_idx * T + t) * ConvDim + c_idx] =
            static_cast<InT>(acc);

          for (int k = 0; k < Keep; ++k) {
            InT value;
            if (k + 1 < Keep) {
              if (parent_idx < 0) {
                value = base_conv_state[(b_idx * Keep + k + 1) * ConvDim + c_idx];
              } else {
                value = conv_states[
                  (((b_idx * T + parent_idx) * Keep + k + 1) * ConvDim) + c_idx
                ];
              }
            } else {
              value = qkv_t[c_idx];
            }
            conv_states[
              (((b_idx * T + t) * Keep + k) * ConvDim) + c_idx
            ] = value;
          }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_conv1d_capture",
        input_names=["qkv", "base_conv_state", "conv_weight", "T"],
        output_names=["conv_out", "conv_states"],
        source=source,
    )


def _make_linear_gated_delta_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        for (int t = 0; t < T; ++t) {
          auto parent_idx = t - 1;

          const device StT* parent_state;
          if (parent_idx < 0) {
            parent_state = state_in + (n * Dv + dv_idx) * Dk;
          } else {
            parent_state = states
              + (((b_idx * T + parent_idx) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          }

          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(parent_state[s_idx]);
          }

          auto q_t = q + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto k_t = k + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto v_t = v + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_t[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_t[dv_idx] - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_t[s_idx] * delta;
            out += state[i] * q_t[s_idx];
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          auto state_t = states
            + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state_t[s_idx] = static_cast<StT>(state[i]);
          }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_capture_v2",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


def _make_linear_gated_delta_final_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(state_ptr[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          auto q_t = q + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto k_t = k + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto v_t = v + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_t[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_t[dv_idx] - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_t[s_idx] * delta;
            out += state[i] * q_t[s_idx];
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }
        }

        auto state_out_ptr = state_out + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state_out_ptr[s_idx] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_final_v1",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float q_scale = inv_scale * inv_scale;
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float q_shared[Dk];
        threadgroup float k_shared[Dk];

        for (int t = 0; t < T; ++t) {
          auto parent_idx = t - 1;

          const device StT* parent_state;
          if (parent_idx < 0) {
            parent_state = state_in + (n * Dv + dv_idx) * Dk;
          } else {
            parent_state = states
              + (((b_idx * T + parent_idx) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          }

          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(parent_state[s_idx]);
          }

          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto q_t = conv_t + hk_idx * Dk;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          if (local_dv_idx == 0) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            float q_raw[n_per_t];
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              q_raw[i] = static_cast<float>(q_t[s_idx]);
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              q_sum += q_raw[i] * q_raw[i];
              k_sum += k_raw[i] * k_raw[i];
            }
            q_sum = simd_sum(q_sum);
            k_sum = simd_sum(k_sum);
            float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              q_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_val;
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            auto q_val = q_shared[s_idx];
            state[i] = state[i] + k_val * delta;
            out += state[i] * q_val;
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          auto state_t = states
            + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state_t[s_idx] = static_cast<StT>(state[i]);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_v1",
        input_names=["conv_out", "g", "beta", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_stream_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float q_scale = inv_scale * inv_scale;
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float q_shared[Dk];
        threadgroup float k_shared[Dk];

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(state_ptr[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto q_t = conv_t + hk_idx * Dk;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          if (local_dv_idx == 0) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            float q_raw[n_per_t];
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              q_raw[i] = static_cast<float>(q_t[s_idx]);
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              q_sum += q_raw[i] * q_raw[i];
              k_sum += k_raw[i] * k_raw[i];
            }
            q_sum = simd_sum(q_sum);
            k_sum = simd_sum(k_sum);
            float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              q_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_val;
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            auto q_val = q_shared[s_idx];
            state[i] = state[i] + k_val * delta;
            out += state[i] * q_val;
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          int capture_t = t - CaptureStart;
          if (capture_t >= 0) {
            auto state_t = states
              + (((b_idx * CaptureT + capture_t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto rounded = static_cast<StT>(state[i]);
              state_t[s_idx] = rounded;
              state[i] = static_cast<float>(rounded);
            }
          } else {
            for (int i = 0; i < n_per_t; ++i) {
              state[i] = static_cast<float>(static_cast<StT>(state[i]));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_stream_v1",
        input_names=["conv_out", "g", "beta", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_tape_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float q_scale = inv_scale * inv_scale;
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float q_shared[Dk];
        threadgroup float k_shared[Dk];

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(state_ptr[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto q_t = conv_t + hk_idx * Dk;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          if (local_dv_idx == 0) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            float q_raw[n_per_t];
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              q_raw[i] = static_cast<float>(q_t[s_idx]);
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              q_sum += q_raw[i] * q_raw[i];
              k_sum += k_raw[i] * k_raw[i];
            }
            q_sum = simd_sum(q_sum);
            k_sum = simd_sum(k_sum);
            float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              q_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_val;
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            auto q_val = q_shared[s_idx];
            state[i] = state[i] + k_val * delta;
            out += state[i] * q_val;
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
            tape[((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx] = delta;
          }

          for (int i = 0; i < n_per_t; ++i) {
            state[i] = static_cast<float>(static_cast<StT>(state[i]));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        auto state_t = final_state + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state_t[s_idx] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_tape_v1",
        input_names=["conv_out", "g", "beta", "state_in", "T"],
        output_names=["y", "final_state", "tape"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_tape_replay_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float k_shared[Dk];

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(state_ptr[s_idx]);
        }

        for (int t = 0; t < Steps; ++t) {
          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto g_t = g + (b_idx * T + t) * Hv;

          if (local_dv_idx == 0) {
            float k_sum = 0.0f;
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              k_sum += k_raw[i] * k_raw[i];
            }
            k_sum = simd_sum(k_sum);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          auto delta = tape[((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_t[hv_idx];
            state[i] = state[i] + k_shared[s_idx] * delta;
            state[i] = static_cast<float>(static_cast<StT>(state[i]));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        auto state_t = state_out + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state_t[s_idx] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_tape_replay_v1",
        input_names=["tape", "conv_out", "g", "state_in", "T"],
        output_names=["state_out"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_inline_g_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float q_scale = inv_scale * inv_scale;
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float q_shared[Dk];
        threadgroup float k_shared[Dk];
        threadgroup float g_shared;
        threadgroup float beta_shared;

        for (int t = 0; t < T; ++t) {
          auto parent_idx = t - 1;

          const device StT* parent_state;
          if (parent_idx < 0) {
            parent_state = state_in + (n * Dv + dv_idx) * Dk;
          } else {
            parent_state = states
              + (((b_idx * T + parent_idx) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          }

          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(parent_state[s_idx]);
          }

          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto q_t = conv_t + hk_idx * Dk;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
          auto a_t = a + (b_idx * T + t) * Hv;
          auto b_t = b + (b_idx * T + t) * Hv;

          if (dk_idx == 0 && local_dv_idx == 0) {
            InT b_val = b_t[hv_idx];
            auto beta_y = 1 / (1 + metal::exp(metal::abs(b_val)));
            InT beta_val = (b_val < InT(0)) ? beta_y : 1 - beta_y;

            InT a_val = a_t[hv_idx] + dt_bias[hv_idx];
            constexpr InT inf = metal::numeric_limits<InT>::infinity();
            InT maxval = metal::max(a_val, InT(0));
            InT minval = metal::min(a_val, InT(0));
            InT softplus_val = (minval == -inf || maxval == inf)
              ? maxval
              : (maxval + log1p(metal::exp(minval - maxval)));
            float decay_a = metal::exp(float(A_log[hv_idx]));
            beta_shared = static_cast<float>(beta_val);
            g_shared = metal::exp(-decay_a * float(softplus_val));
          }

          if (local_dv_idx == 0) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            float q_raw[n_per_t];
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              q_raw[i] = static_cast<float>(q_t[s_idx]);
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              q_sum += q_raw[i] * q_raw[i];
              k_sum += k_raw[i] * k_raw[i];
            }
            q_sum = simd_sum(q_sum);
            k_sum = simd_sum(k_sum);
            float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              q_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            state[i] = state[i] * g_shared;
            kv_mem += state[i] * k_val;
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem)
            * beta_shared;

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            auto q_val = q_shared[s_idx];
            state[i] = state[i] + k_val * delta;
            out += state[i] * q_val;
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          auto state_t = states
            + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state_t[s_idx] = static_cast<StT>(state[i]);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_inline_g_v1",
        input_names=["conv_out", "a", "b", "A_log", "dt_bias", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


_linear_conv1d_kernel = _make_linear_conv1d_kernel()
_linear_gated_delta_kernel = _make_linear_gated_delta_kernel()
_linear_gated_delta_final_kernel = _make_linear_gated_delta_final_kernel()
_linear_gated_delta_from_conv_kernel = _make_linear_gated_delta_from_conv_kernel()
_linear_gated_delta_from_conv_stream_kernel = (
    _make_linear_gated_delta_from_conv_stream_kernel()
)
_linear_gated_delta_from_conv_tape_kernel = (
    _make_linear_gated_delta_from_conv_tape_kernel()
)
_linear_gated_delta_from_conv_tape_replay_kernel = (
    _make_linear_gated_delta_from_conv_tape_replay_kernel()
)
_linear_gated_delta_from_conv_inline_g_kernel = (
    _make_linear_gated_delta_from_conv_inline_g_kernel()
)

_LINEAR_GDN_ALIASES = {"linear_gdn", "linear_gdn_len5"}
_LINEAR_GDN_FROM_CONV_ALIASES = {
    "linear_gdn_from_conv",
    "linear_gdn_from_conv_len5",
}
_LINEAR_GDN_FROM_CONV_STREAM_ALIASES = {
    "linear_gdn_from_conv_stream",
    "linear_gdn_from_conv_stream_len5",
}
_LINEAR_GDN_FROM_CONV_STREAM_SKIP0_ALIASES = {
    "linear_gdn_from_conv_stream_skip0",
    "linear_gdn_from_conv_stream_skip0_len5",
}
_LINEAR_GDN_FROM_CONV_TAPE_ALIASES = {
    "linear_gdn_from_conv_tape",
    "linear_gdn_from_conv_tape_len5",
}
_LINEAR_GDN_FROM_CONV_INLINE_G_ALIASES = {
    "linear_gdn_from_conv_inline_g",
    "linear_gdn_from_conv_inline_g_len5",
}
_LINEAR_GDN_FINAL_ALIASES = {"linear_gdn_final", "linear_gdn_final_len5"}
_DEMOTED_GDN_ALIASES = {
    "linear_gdn_conv",
    "linear_gdn_len6",
    "linear_gdn_mlp_gateup",
}


def _contiguous_recurrent_leaf(value: mx.array) -> mx.array:
    # Mirrors mlx-lm #1077's cache ownership fix: the authoritative recurrent
    # leaf must not retain the larger per-position capture buffer.
    return mx.contiguous(value)


def _maybe_contiguous_authoritative_gdn_leaf(value: mx.array) -> mx.array:
    if not _env_enabled("MTPLX_CAPTURE_CONTIGUOUS_GDN_STATE"):
        return value
    return _contiguous_recurrent_leaf(value)


def _gdn_tape_meta(gdn: Any) -> dict[str, int]:
    return {
        "conv_dim": int(gdn.conv_dim),
        "head_k_dim": int(gdn.head_k_dim),
        "head_v_dim": int(gdn.head_v_dim),
        "num_k_heads": int(gdn.num_k_heads),
        "num_v_heads": int(gdn.num_v_heads),
        "key_dim": int(gdn.key_dim),
    }


def _gdn_meta_int(meta: Any, name: str) -> int:
    if isinstance(meta, dict):
        return int(meta[name])
    return int(getattr(meta, name))


def resolve_gdn_capture_backend(backend: str | None = None) -> str:
    """Resolve the GDN capture backend with backwards-compatible env support."""
    if backend is None:
        env_value = os.environ.get("MTPLX_CAPTURE_CUSTOM_KERNEL")
        if env_value is None:
            return "stock"
        normalized_env = env_value.lower().replace("-", "_")
        if normalized_env in {"1", "true", "yes", "on"} | _LINEAR_GDN_ALIASES:
            return "linear_gdn"
        if normalized_env in _LINEAR_GDN_FROM_CONV_ALIASES:
            return "linear_gdn_from_conv"
        if normalized_env in _LINEAR_GDN_FROM_CONV_STREAM_ALIASES:
            return "linear_gdn_from_conv_stream"
        if normalized_env in _LINEAR_GDN_FROM_CONV_STREAM_SKIP0_ALIASES:
            return "linear_gdn_from_conv_stream_skip0"
        if normalized_env in _LINEAR_GDN_FROM_CONV_TAPE_ALIASES:
            return "linear_gdn_from_conv_tape"
        if normalized_env in _LINEAR_GDN_FROM_CONV_INLINE_G_ALIASES:
            return "linear_gdn_from_conv_inline_g"
        if normalized_env in _LINEAR_GDN_FINAL_ALIASES:
            return "linear_gdn_final"
        if normalized_env in {"0", "false", "no", "off", "stock"}:
            return "stock"
        if normalized_env in _DEMOTED_GDN_ALIASES:
            raise ValueError(
                f"MTPLX_CAPTURE_CUSTOM_KERNEL backend {env_value!r} is not promoted; "
                "use 'stock', 'linear-gdn', 'linear-gdn-len5', or "
                "'linear-gdn-from-conv'"
            )
        raise ValueError(
            "MTPLX_CAPTURE_CUSTOM_KERNEL must be one of 1/0, true/false, "
            "'linear-gdn', 'linear-gdn-len5', 'linear-gdn-from-conv', or 'stock'"
        )
    normalized = backend.replace("-", "_")
    if normalized == "stock":
        return "stock"
    if normalized in _LINEAR_GDN_ALIASES:
        return "linear_gdn"
    if normalized in _LINEAR_GDN_FROM_CONV_ALIASES:
        return "linear_gdn_from_conv"
    if normalized in _LINEAR_GDN_FROM_CONV_STREAM_ALIASES:
        return "linear_gdn_from_conv_stream"
    if normalized in _LINEAR_GDN_FROM_CONV_STREAM_SKIP0_ALIASES:
        return "linear_gdn_from_conv_stream_skip0"
    if normalized in _LINEAR_GDN_FROM_CONV_TAPE_ALIASES:
        return "linear_gdn_from_conv_tape"
    if normalized in _LINEAR_GDN_FROM_CONV_INLINE_G_ALIASES:
        return "linear_gdn_from_conv_inline_g"
    if normalized in _LINEAR_GDN_FINAL_ALIASES:
        return "linear_gdn_final"
    if normalized in _DEMOTED_GDN_ALIASES:
        raise ValueError(
            f"GDN capture backend {backend!r} is not promoted; use 'stock' or "
            "'linear-gdn-len5'"
        )
    raise ValueError(
        "GDN capture backend must be 'stock', 'linear-gdn', 'linear-gdn-len5', "
        "'linear-gdn-from-conv', or diagnostic 'linear-gdn-final'"
    )


def _linear_conv1d_capture(qkv: mx.array, base_conv_state: mx.array, conv_weight: mx.array):
    if _linear_conv1d_kernel is None:
        return None
    B, T, conv_dim = qkv.shape
    keep = int(base_conv_state.shape[1])
    if (
        len(conv_weight.shape) != 3
        or int(conv_weight.shape[0]) != conv_dim
        or int(conv_weight.shape[1]) != keep + 1
        or int(conv_weight.shape[2]) != 1
    ):
        return None
    input_type = qkv.dtype
    raw_conv, conv_states = _linear_conv1d_kernel(
        inputs=[qkv, base_conv_state, conv_weight, T],
        template=[("InT", input_type), ("Keep", keep), ("ConvDim", conv_dim)],
        grid=(conv_dim, B, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, T, conv_dim), (B, T, keep, conv_dim)],
        output_dtypes=[input_type, input_type],
    )
    return nn.silu(raw_conv), conv_states


def _matching_quantized_linears(left: Any, right: Any) -> bool:
    if not isinstance(left, nn.QuantizedLinear) or not isinstance(right, nn.QuantizedLinear):
        return False
    if "bias" in left or "bias" in right:
        return False
    return (
        int(left.bits) == int(right.bits)
        and int(left.group_size) == int(right.group_size)
        and str(left.mode) == str(right.mode)
        and tuple(left.weight.shape[1:]) == tuple(right.weight.shape[1:])
        and tuple(left.scales.shape[1:]) == tuple(right.scales.shape[1:])
        and tuple(left.biases.shape[1:]) == tuple(right.biases.shape[1:])
    )


def _fused_quantized_pair(
    owner: Any,
    cache_name: str,
    inputs: mx.array,
    left: nn.QuantizedLinear,
    right: nn.QuantizedLinear,
) -> tuple[mx.array, mx.array] | None:
    if not _matching_quantized_linears(left, right):
        return None
    cached = getattr(owner, cache_name, None)
    if cached is None:
        weight = mx.concatenate([left.weight, right.weight], axis=0)
        scales = mx.concatenate([left.scales, right.scales], axis=0)
        biases = mx.concatenate([left.biases, right.biases], axis=0)
        mx.eval(weight, scales, biases)
        cached = (weight, scales, biases, int(left.weight.shape[0]))
        setattr(owner, cache_name, cached)
    weight, scales, biases, split_at = cached
    out = mx.quantized_matmul(
        inputs,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=int(left.group_size),
        bits=int(left.bits),
        mode=str(left.mode),
    )
    left_out, right_out = mx.split(out, [int(split_at)], axis=-1)
    return left_out, right_out


def _fused_quantized_many(
    owner: Any,
    cache_name: str,
    inputs: mx.array,
    modules: tuple[nn.QuantizedLinear, ...],
) -> tuple[mx.array, ...] | None:
    if not modules:
        return None
    first = modules[0]
    if any(not _matching_quantized_linears(first, module) for module in modules[1:]):
        return None
    if "bias" in first:
        return None
    cached = getattr(owner, cache_name, None)
    if cached is None:
        weight = mx.concatenate([module.weight for module in modules], axis=0)
        scales = mx.concatenate([module.scales for module in modules], axis=0)
        biases = mx.concatenate([module.biases for module in modules], axis=0)
        mx.eval(weight, scales, biases)
        sizes = [int(module.weight.shape[0]) for module in modules]
        split_points = []
        running = 0
        for size in sizes[:-1]:
            running += size
            split_points.append(running)
        cached = (weight, scales, biases, tuple(split_points))
        setattr(owner, cache_name, cached)
    weight, scales, biases, split_points = cached
    out = mx.quantized_matmul(
        inputs,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=int(first.group_size),
        bits=int(first.bits),
        mode=str(first.mode),
    )
    return tuple(mx.split(out, list(split_points), axis=-1))


def _gdn_input_projections(gdn: Any, inputs: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    fuse_mode = os.environ.get("MTPLX_FUSE_GDN_PROJECTIONS", "").lower()
    if fuse_mode in {"all", "4to1", "one"}:
        fused = _fused_quantized_many(
            gdn,
            "_mtplx_fused_qkvzba",
            inputs,
            (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a),
        )
        if fused is not None:
            qkv, z, b, a = fused
            return qkv, z, b, a
    if fuse_mode in {"1", "true", "yes", "on"}:
        qkvz = _fused_quantized_pair(
            gdn,
            "_mtplx_fused_qkvz",
            inputs,
            gdn.in_proj_qkv,
            gdn.in_proj_z,
        )
        ba = _fused_quantized_pair(
            gdn,
            "_mtplx_fused_ba",
            inputs,
            gdn.in_proj_b,
            gdn.in_proj_a,
        )
        if qkvz is not None and ba is not None:
            qkv, z = qkvz
            b, a = ba
            return qkv, z, b, a
    return (
        gdn.in_proj_qkv(inputs),
        gdn.in_proj_z(inputs),
        gdn.in_proj_b(inputs),
        gdn.in_proj_a(inputs),
    )


def _stock_conv1d_capture(qkv: mx.array, base_conv_state: mx.array, gdn: Any):
    """Run the exact MLX Conv1d path and capture each linear-prefix state."""
    B, T, _ = qkv.shape
    keep = int(base_conv_state.shape[1])
    conv_input = mx.concatenate([base_conv_state, qkv], axis=1)
    conv_out = nn.silu(gdn.conv1d(conv_input))
    conv_states = mx.stack(
        [conv_input[:, i + 1 : i + 1 + keep, :] for i in range(T)],
        axis=1,
    )
    return conv_out, conv_states


def _linear_gated_delta_capture(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
):
    if _linear_gated_delta_kernel is None:
        return None
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk % 32 != 0:
        return None
    input_type = q.dtype
    state_type = state.dtype
    return _linear_gated_delta_kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _linear_gated_delta_final(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
):
    if _linear_gated_delta_final_kernel is None:
        return None
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk % 32 != 0:
        return None
    input_type = q.dtype
    state_type = state.dtype
    return _linear_gated_delta_final_kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _linear_gated_delta_from_conv_capture(
    conv_out: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    gdn: Any,
):
    if _linear_gated_delta_from_conv_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != int(gdn.conv_dim):
        return None
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    if Dk % 32 != 0:
        return None
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", "32"))
    except ValueError:
        tgy = 32
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    return _linear_gated_delta_from_conv_kernel(
        inputs=[conv_out, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _linear_gated_delta_from_conv_stream_capture(
    conv_out: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    gdn: Any,
    *,
    capture_start: int = 0,
):
    if _linear_gated_delta_from_conv_stream_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != int(gdn.conv_dim):
        return None
    capture_start = int(capture_start)
    if capture_start < 0 or capture_start >= int(T):
        return None
    capture_t = int(T) - capture_start
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    if Dk % 32 != 0:
        return None
    default_tgy = "8" if capture_start else "32"
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", default_tgy))
    except ValueError:
        tgy = 32
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    return _linear_gated_delta_from_conv_stream_kernel(
        inputs=[conv_out, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
            ("CaptureStart", capture_start),
            ("CaptureT", capture_t),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), (B, capture_t, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _linear_gated_delta_from_conv_tape_capture(
    conv_out: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    gdn: Any,
):
    if _linear_gated_delta_from_conv_tape_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != int(gdn.conv_dim):
        return None
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    if Dk % 32 != 0:
        return None
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", "8"))
    except ValueError:
        tgy = 8
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    return _linear_gated_delta_from_conv_tape_kernel(
        inputs=[conv_out, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk), (B, T, Hv, Dv)],
        output_dtypes=[input_type, state_type, mx.float32],
    )


def _linear_gated_delta_from_conv_tape_replay(
    tape: mx.array,
    conv_out: mx.array,
    g: mx.array,
    state: mx.array,
    gdn_meta: Any,
    *,
    steps: int,
):
    if _linear_gated_delta_from_conv_tape_replay_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != _gdn_meta_int(gdn_meta, "conv_dim"):
        return None
    steps = int(steps)
    if steps <= 0 or steps > int(T):
        return None
    Dk = _gdn_meta_int(gdn_meta, "head_k_dim")
    Dv = _gdn_meta_int(gdn_meta, "head_v_dim")
    Hk = _gdn_meta_int(gdn_meta, "num_k_heads")
    Hv = _gdn_meta_int(gdn_meta, "num_v_heads")
    if Dk % 32 != 0:
        return None
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", "8"))
    except ValueError:
        tgy = 8
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    (state_out,) = _linear_gated_delta_from_conv_tape_replay_kernel(
        inputs=[tape, conv_out, g, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", _gdn_meta_int(gdn_meta, "key_dim")),
            ("ConvDim", _gdn_meta_int(gdn_meta, "conv_dim")),
            ("Steps", steps),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[state.shape],
        output_dtypes=[state_type],
    )
    return state_out


def _linear_gated_delta_from_conv_inline_g_capture(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    gdn: Any,
):
    if _linear_gated_delta_from_conv_inline_g_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != int(gdn.conv_dim):
        return None
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    if Dk % 32 != 0:
        return None
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", "32"))
    except ValueError:
        tgy = 32
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    return _linear_gated_delta_from_conv_inline_g_kernel(
        inputs=[conv_out, a, b, gdn.A_log, gdn.dt_bias, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _stock_gated_delta_capture(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    mask: Any,
    gdn: Any,
):
    """Capture per-position recurrent state through stock MLX single-token steps."""
    from mlx_lm.models.gated_delta import gated_delta_update

    T = int(q.shape[1])
    outs = []
    states = []
    current = state
    for idx in range(T):
        step_mask = None
        if mask is not None and not isinstance(mask, str):
            step_mask = mask[:, idx : idx + 1]
        out, current = gated_delta_update(
            q[:, idx : idx + 1, :, :],
            k[:, idx : idx + 1, :, :],
            v[:, idx : idx + 1, :, :],
            a[:, idx : idx + 1, :],
            b[:, idx : idx + 1, :],
            gdn.A_log,
            gdn.dt_bias,
            current,
            step_mask,
            use_kernel=not gdn.training,
        )
        outs.append(out)
        states.append(current)
    return mx.concatenate(outs, axis=1), mx.stack(states, axis=1)


def gdn_forward_with_capture(
    gdn: Any,
    inputs: mx.array,
    mask: Any = None,
    cache: Any = None,
    *,
    capture_backend: str | None = None,
):
    if getattr(gdn, "sharding_group", None) is not None:
        return gdn(inputs, mask=mask, cache=cache), None

    from mlx_lm.models.gated_delta import compute_g

    B, S, _ = inputs.shape
    qkv, z, b, a = _gdn_input_projections(gdn, inputs)
    z = z.reshape(B, S, gdn.num_v_heads, gdn.head_v_dim)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (B, gdn.conv_kernel_size - 1, gdn.conv_dim),
            dtype=inputs.dtype,
        )

    conv_capture = None
    if _env_enabled("MTPLX_LINEAR_CONV1D_CAPTURE"):
        conv_capture = _linear_conv1d_capture(qkv, conv_state, gdn.conv1d.weight)
    if conv_capture is None:
        conv_capture = _stock_conv1d_capture(qkv, conv_state, gdn)
    conv_out, conv_states = conv_capture
    backend = resolve_gdn_capture_backend(capture_backend)

    state = cache[1] if cache and cache[1] is not None else None
    if state is None:
        state = mx.zeros((B, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), dtype=mx.float32)

    final_only_capture = False
    capture_start = 0
    if backend == "linear_gdn_from_conv_inline_g":
        delta_result = _linear_gated_delta_from_conv_inline_g_capture(
            conv_out,
            a,
            b,
            state,
            gdn,
        )
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, states = delta_result
    elif backend == "linear_gdn_from_conv_tape":
        beta = mx.sigmoid(b)
        g = compute_g(gdn.A_log, a, gdn.dt_bias)
        delta_result = _linear_gated_delta_from_conv_tape_capture(
            conv_out,
            g,
            beta,
            state,
            gdn,
        )
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, final_state, tape = delta_result
        states = final_state[:, None, :, :, :]
    elif backend in {"linear_gdn_from_conv_stream", "linear_gdn_from_conv_stream_skip0"}:
        beta = mx.sigmoid(b)
        g = compute_g(gdn.A_log, a, gdn.dt_bias)
        capture_start = 1 if backend == "linear_gdn_from_conv_stream_skip0" else 0
        delta_result = _linear_gated_delta_from_conv_stream_capture(
            conv_out,
            g,
            beta,
            state,
            gdn,
            capture_start=capture_start,
        )
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, states = delta_result
    elif backend in {"linear_gdn", "linear_gdn_from_conv"}:
        use_from_conv = backend == "linear_gdn_from_conv" or os.environ.get(
            "MTPLX_LINEAR_GDN_FROM_CONV", ""
        ).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        beta = mx.sigmoid(b)
        g = compute_g(gdn.A_log, a, gdn.dt_bias)
        if use_from_conv:
            delta_result = _linear_gated_delta_from_conv_capture(conv_out, g, beta, state, gdn)
        else:
            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
                    [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                    [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
                )
            ]
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
            delta_result = _linear_gated_delta_capture(q, k, v, g, beta, state)
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, states = delta_result
    elif backend == "linear_gdn_final":
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
                [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        beta = mx.sigmoid(b)
        g = compute_g(gdn.A_log, a, gdn.dt_bias)
        delta_result = _linear_gated_delta_final(q, k, v, g, beta, state)
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, final_state = delta_result
        states = final_state[:, None, :, :, :]
        final_only_capture = True
    else:
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
                [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, states = _stock_gated_delta_capture(q, k, v, a, b, state, mask, gdn)

    if cache is not None:
        cache[0] = mx.contiguous(conv_states[:, -1, :, :])
        cache[1] = _maybe_contiguous_authoritative_gdn_leaf(states[:, -1, :, :, :])
        cache.advance(S)

    tail_projected = False
    if os.environ.get("MTPLX_NATIVE_GDN_TAIL", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        from .kernels.native_gdn_tail import native_gdn_norm_gate_out_qmv8

        out = native_gdn_norm_gate_out_qmv8(
            out,
            z,
            gdn.norm.weight,
            gdn.norm.eps,
            gdn.out_proj,
            num_simdgroups=int(os.environ.get("MTPLX_NATIVE_GDN_TAIL_SIMDGROUPS") or 2),
        )
        tail_projected = True
    elif os.environ.get("MTPLX_FUSE_GDN_NORM_GATE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        from .kernel_selfcheck import lane_disabled
        from .kernels.fused_norm import fused_gdn_norm_gate

        if lane_disabled("fused_gdn_norm_gate"):
            out = gdn.norm(out, z)
        else:
            out = fused_gdn_norm_gate(out, z, gdn.norm.weight, gdn.norm.eps)
    else:
        out = gdn.norm(out, z)
    if not tail_projected:
        out = out.reshape(B, S, -1)
        if os.environ.get("MTPLX_GDN_OUT_QMV8", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            from .verify_qmv import stocklike_qmv8_matmul

            out = stocklike_qmv8_matmul(out, gdn.out_proj)
        else:
            out = gdn.out_proj(out)
    if final_only_capture:
        return out, {"final_only": True}
    if backend == "linear_gdn_from_conv_tape":
        return out, {
            "conv_states": conv_states,
            "conv_out": conv_out,
            "g": g,
            "state_in": state,
            "tape": tape,
            "gdn_meta": _gdn_tape_meta(gdn),
        }
    if capture_start:
        return out, {
            "conv_states": conv_states[:, capture_start:, :, :],
            "states": states,
            "capture_start": capture_start,
        }
    return out, {"conv_states": conv_states, "states": states}


def forward_with_gdn_capture(
    model: Any,
    inputs: mx.array,
    cache=None,
    return_hidden: bool = False,
    *,
    hidden_variant: str | None = None,
    capture_backend: str | None = None,
):
    text_model = getattr(model, "language_model", model)
    inner = text_model.model
    hidden_states = inner.embed_tokens(inputs)
    if cache is None:
        cache = [None] * len(inner.layers)

    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
    captures: dict[int, dict[str, mx.array]] = {}
    backend = resolve_gdn_capture_backend(capture_backend)
    context_len = _cache_context_len(cache)
    layer_eval_every = _target_layer_eval_every(context_len)
    layer_eval_threshold = int(
        os.environ.get("MTPLX_TARGET_LAYER_EVAL_CONTEXT_THRESHOLD", "0") or "0"
    )
    layer_eval_max_q = int(os.environ.get("MTPLX_TARGET_LAYER_EVAL_MAX_Q", "8") or "8")
    layer_eval_enabled = (
        layer_eval_every > 0
        and int(inputs.shape[1]) <= max(1, layer_eval_max_q)
        and context_len >= max(0, layer_eval_threshold)
    )

    for layer_idx, (layer, layer_cache) in enumerate(zip(inner.layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        normed = layer.input_layernorm(hidden_states)
        if layer.is_linear:
            if os.environ.get("MTPLX_ABLATE_LINEAR_ATTN", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                r = mx.zeros_like(normed)
            else:
                r, capture = gdn_forward_with_capture(
                    layer.linear_attn,
                    normed,
                    mask=mask,
                    cache=layer_cache,
                    capture_backend=backend,
                )
                if capture is not None:
                    if capture.get("final_only"):
                        captures["__final_only__"] = True
                    else:
                        captures[layer_idx] = capture
        else:
            r = layer.self_attn(normed, mask=mask, cache=layer_cache)
        if os.environ.get("MTPLX_FUSE_POST_NORM_RESIDUAL", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            from .kernel_selfcheck import lane_disabled
            from .kernels.fused_norm import fused_add_rmsnorm

            if lane_disabled("fused_add_rmsnorm"):
                h = hidden_states + r
                mlp_input = layer.post_attention_layernorm(h)
            else:
                h, mlp_input = fused_add_rmsnorm(
                    hidden_states,
                    r,
                    layer.post_attention_layernorm.weight,
                    layer.post_attention_layernorm.eps,
                    threadgroup_size=512,
                )
        else:
            h = hidden_states + r
            mlp_input = layer.post_attention_layernorm(h)
        hidden_states = h + layer.mlp(mlp_input)
        if layer_eval_enabled and (layer_idx + 1) % layer_eval_every == 0:
            mx.eval(hidden_states)

    pre_norm = hidden_states
    post_norm = inner.norm(hidden_states)
    logits = inner.embed_tokens.as_linear(post_norm) if text_model.args.tie_word_embeddings else text_model.lm_head(post_norm)
    if return_hidden:
        hidden = pre_norm if hidden_variant == "pre_norm" else post_norm
        return logits, hidden, captures
    return logits, captures


def commit_captured_prefix(
    cache: list[Any],
    captures: dict[int, dict[str, mx.array]],
    keep_tokens: int,
    verified_tokens: int,
    *,
    detach_components: set[str] | None = None,
    detach_mode: str = "selected_slice_contiguous_eval",
    detach_stats: dict[str, int] | None = None,
) -> bool:
    if keep_tokens <= 0 or keep_tokens > verified_tokens:
        return False
    if captures.get("__final_only__"):
        return False
    detach_requested = {
        item.strip().lower().replace("-", "_")
        for item in (detach_components or set())
        if item
    }
    trim_tokens = verified_tokens - keep_tokens
    capture_index = keep_tokens - 1
    for capture in captures.values():
        if isinstance(capture, dict):
            capture_start = int(capture.get("capture_start", 0))
            if capture_index - capture_start < 0:
                return False
    for layer_idx, entry in enumerate(cache):
        capture = captures.get(layer_idx)
        if capture is not None and hasattr(entry, "state"):
            capture_start = int(capture.get("capture_start", 0))
            adjusted_index = capture_index - capture_start
            conv_state = mx.contiguous(capture["conv_states"][:, adjusted_index, :, :])
            if "conv" in detach_requested:
                from .cache_state import detach_array_leaf

                conv_state = detach_array_leaf(conv_state, mode=detach_mode)
                if detach_stats is not None:
                    detach_stats["arrays"] = int(detach_stats.get("arrays", 0)) + 1
                    detach_stats["bytes"] = int(detach_stats.get("bytes", 0)) + int(conv_state.nbytes)
            if "tape" in capture:
                replayed_state = _linear_gated_delta_from_conv_tape_replay(
                    capture["tape"],
                    capture["conv_out"],
                    capture["g"],
                    capture["state_in"],
                    capture.get("gdn_meta", capture.get("gdn")),
                    steps=capture_index + 1,
                )
                if replayed_state is None:
                    return False
                gdn_state = _maybe_contiguous_authoritative_gdn_leaf(replayed_state)
            else:
                gdn_state = _contiguous_recurrent_leaf(
                    capture["states"][:, adjusted_index, :, :, :]
                )
            if "gdn" in detach_requested:
                from .cache_state import detach_array_leaf

                gdn_state = detach_array_leaf(gdn_state, mode=detach_mode)
                if detach_stats is not None:
                    detach_stats["arrays"] = int(detach_stats.get("arrays", 0)) + 1
                    detach_stats["bytes"] = int(detach_stats.get("bytes", 0)) + int(gdn_state.nbytes)
            from .cache_state import replace_recurrent_cache_state

            replace_recurrent_cache_state(entry, [conv_state, gdn_state])
        elif trim_tokens and hasattr(entry, "is_trimmable") and entry.is_trimmable():
            entry.trim(trim_tokens)
    return True
