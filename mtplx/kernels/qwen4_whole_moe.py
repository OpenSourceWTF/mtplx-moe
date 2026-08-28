"""Fixed Metal stages for the exact two-row Qwen4 target MoE."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


HIDDEN = 2560
EXPERTS = 512
TOP_K = 10
INTERMEDIATE = 640
ACTIVATION_SLOTS = 11
SUPPORTED_ROWS = (2, 3)
THREADS = 128
STAGE1_THREADS = 32
STAGE1_EXPERTS_PER_GROUP = 4
STAGE1_GROUPS = EXPERTS // STAGE1_EXPERTS_PER_GROUP
STAGE2_GROUPS = ACTIVATION_SLOTS * (INTERMEDIATE // 16)
ROUTE_THREADS = 256
ROUTE_SIMD_GROUPS = ROUTE_THREADS // 32

_KERNELS: dict[tuple[int, str], Any] = {}


def _require_rows(rows: int) -> int:
    rows = int(rows)
    if rows not in SUPPORTED_ROWS:
        raise ValueError(
            f"Qwen4 whole-MoE rows must be one of {SUPPORTED_ROWS}: {rows}"
        )
    return rows


def _preamble(rows: int) -> str:
    return f"""
        using namespace metal;
        constexpr uint HIDDEN = {HIDDEN};
        constexpr uint EXPERTS = {EXPERTS};
        constexpr uint TOP_K = {TOP_K};
        constexpr uint INTERMEDIATE = {INTERMEDIATE};
        constexpr uint ACTIVATION_SLOTS = {ACTIVATION_SLOTS};
        constexpr uint ROWS = {rows};
    """


def _stage1_source(rows: int) -> str:
    return _preamble(rows) + r"""
        constexpr uint EXPERTS_PER_GROUP = 4;
        constexpr uint VALUES_PER_LANE = 16;
        constexpr uint K_BLOCK = VALUES_PER_LANE * 32;
        constexpr uint SHARED_GROUP = 64;

        uint expert_tile = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;

        uint expert_base = expert_tile * EXPERTS_PER_GROUP;
        float result[ROWS][4] = {};
        const device uchar* shared_gate_weights =
            reinterpret_cast<const device uchar*>(shared_gate_weight);
        float shared_gate_result[ROWS] = {};
        for (uint k_block = 0; k_block < HIDDEN; k_block += K_BLOCK) {
            uint k_lane = k_block + lane * VALUES_PER_LANE;
            float input_values[ROWS][VALUES_PER_LANE];
            for (uint row = 0; row < ROWS; ++row) {
                for (uint item = 0; item < VALUES_PER_LANE; ++item) {
                    input_values[row][item] = float(
                        value[row * HIDDEN + k_lane + item]);
                }
            }
            for (uint result_index = 0; result_index < 4; ++result_index) {
                uint expert = expert_base + result_index;
                uint weight_base = expert * HIDDEN + k_lane;
                for (uint item = 0; item < VALUES_PER_LANE; ++item) {
                    float weight_value = float(router_weight[weight_base + item]);
                    for (uint row = 0; row < ROWS; ++row) {
                        result[row][result_index] +=
                            input_values[row][item] * weight_value;
                    }
                }
            }
            if (expert_tile == 0) {
                uint metadata_index = k_lane / SHARED_GROUP;
                float scale = float(shared_gate_scales[metadata_index]);
                float bias = float(shared_gate_biases[metadata_index]);
                float input_sum[ROWS] = {};
                float quantized_dot[ROWS] = {};
                for (uint item = 0; item < VALUES_PER_LANE; ++item) {
                    float weight_value = float(shared_gate_weights[k_lane + item]);
                    for (uint row = 0; row < ROWS; ++row) {
                        float input_value = input_values[row][item];
                        input_sum[row] += input_value;
                        quantized_dot[row] += input_value * weight_value;
                    }
                }
                for (uint row = 0; row < ROWS; ++row) {
                    shared_gate_result[row] +=
                        scale * quantized_dot[row] + input_sum[row] * bias;
                }
            }
        }
        for (uint row = 0; row < ROWS; ++row) {
            for (uint result_index = 0; result_index < 4; ++result_index) {
                float reduced = simd_sum(result[row][result_index]);
                if (lane == 0) {
                    uint expert = expert_base + result_index;
                    router_logits[row * EXPERTS + expert] = reduced;
                }
            }
        }

        if (expert_tile == 0) {
            for (uint row = 0; row < ROWS; ++row) {
                float reduced = simd_sum(shared_gate_result[row]);
                if (lane == 0) {
                    shared_gate[row] = bfloat(reduced);
                }
            }
        }
    """


def _route_source(rows: int) -> str:
    return _preamble(rows) + r"""
        constexpr uint SIMD_GROUPS = 8;
        constexpr uint LOCAL_CANDIDATES = SIMD_GROUPS * TOP_K;

        uint row = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        threadgroup float local_logits[LOCAL_CANDIDATES];
        threadgroup uint local_indices[LOCAL_CANDIDATES];

        uint expert0 = simd_gid * 64 + lane;
        uint expert1 = expert0 + 32;
        float candidate0 = router_logits[row * EXPERTS + expert0];
        float candidate1 = router_logits[row * EXPERTS + expert1];

        _Pragma("unroll")
        for (uint rank = 0; rank < TOP_K; ++rank) {
            bool take1 = candidate1 > candidate0
                || (candidate1 == candidate0 && expert1 > expert0);
            float lane_logit = take1 ? candidate1 : candidate0;
            uint lane_expert = take1 ? expert1 : expert0;
            float winner_logit = simd_max(lane_logit);
            float winner_expert_value = simd_max(
                lane_logit == winner_logit ? float(lane_expert) : -1.0f);
            uint winner_expert = uint(winner_expert_value);
            if (lane == 0) {
                uint destination = simd_gid * TOP_K + rank;
                local_logits[destination] = winner_logit;
                local_indices[destination] = winner_expert;
            }
            if (lane_expert == winner_expert) {
                if (take1) {
                    candidate1 = -INFINITY;
                } else {
                    candidate0 = -INFINITY;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_gid == 0) {
            uint slot0 = lane;
            uint slot1 = lane + 32;
            uint slot2 = lane + 64;
            float merge0 = local_logits[slot0];
            float merge1 = local_logits[slot1];
            float merge2 = slot2 < LOCAL_CANDIDATES
                ? local_logits[slot2] : -INFINITY;
            uint index0 = local_indices[slot0];
            uint index1 = local_indices[slot1];
            uint index2 = slot2 < LOCAL_CANDIDATES
                ? local_indices[slot2] : 0;

            _Pragma("unroll")
            for (uint rank = 0; rank < TOP_K; ++rank) {
                uint selected = 0;
                float lane_logit = merge0;
                uint lane_expert = index0;
                if (merge1 > lane_logit
                    || (merge1 == lane_logit && index1 > lane_expert)) {
                    selected = 1;
                    lane_logit = merge1;
                    lane_expert = index1;
                }
                if (merge2 > lane_logit
                    || (merge2 == lane_logit && index2 > lane_expert)) {
                    selected = 2;
                    lane_logit = merge2;
                    lane_expert = index2;
                }
                float winner_logit = simd_max(lane_logit);
                float winner_expert_value = simd_max(
                    lane_logit == winner_logit ? float(lane_expert) : -1.0f);
                uint winner_expert = uint(winner_expert_value);
                if (lane == 0) {
                    uint destination = row * TOP_K + rank;
                    expert_ids[destination] = winner_expert;
                    selected_logits[destination] = winner_logit;
                }
                if (lane_expert == winner_expert) {
                    if (selected == 0) {
                        merge0 = -INFINITY;
                    } else if (selected == 1) {
                        merge1 = -INFINITY;
                    } else {
                        merge2 = -INFINITY;
                    }
                }
            }
        }
    """


def _stage2_source(rows: int) -> str:
    return _preamble(rows) + r"""
        constexpr uint OUTPUT_TILES = INTERMEDIATE / 16;
        constexpr uint Q4_GROUP = 32;
        constexpr uint Q4_VALUES_PER_LANE = 16;
        constexpr uint Q4_BLOCK = Q4_VALUES_PER_LANE * 32;
        constexpr uint Q8_GROUP = 128;
        constexpr uint Q8_VALUES_PER_LANE = 16;
        constexpr uint Q8_BLOCK = Q8_VALUES_PER_LANE * 32;

        uint group = threadgroup_position_in_grid.x;
        uint slot = group / OUTPUT_TILES;
        uint tile = group - slot * OUTPUT_TILES;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        if (slot < TOP_K) {
            for (uint row = 0; row < ROWS; ++row) {
                uint expert = expert_ids[row * TOP_K + slot];
                const device uchar* gate_bytes =
                    reinterpret_cast<const device uchar*>(routed_gate_weight)
                    + expert * INTERMEDIATE * (HIDDEN / 2);
                const device uchar* up_bytes =
                    reinterpret_cast<const device uchar*>(routed_up_weight)
                    + expert * INTERMEDIATE * (HIDDEN / 2);
                const device bfloat* gate_scale_values = routed_gate_scales
                    + expert * INTERMEDIATE * (HIDDEN / Q4_GROUP);
                const device bfloat* gate_bias_values = routed_gate_biases
                    + expert * INTERMEDIATE * (HIDDEN / Q4_GROUP);
                const device bfloat* up_scale_values = routed_up_scales
                    + expert * INTERMEDIATE * (HIDDEN / Q4_GROUP);
                const device bfloat* up_bias_values = routed_up_biases
                    + expert * INTERMEDIATE * (HIDDEN / Q4_GROUP);
                float gate_result[4] = {};
                float up_result[4] = {};

                for (uint k_block = 0; k_block < HIDDEN; k_block += Q4_BLOCK) {
                    uint k_lane = k_block + lane * Q4_VALUES_PER_LANE;
                    float input_values[Q4_VALUES_PER_LANE];
                    float input_sum = 0.0f;
                    for (uint item = 0; item < Q4_VALUES_PER_LANE; item += 4) {
                        float x0 = float(value[row * HIDDEN + k_lane + item]);
                        float x1 = float(value[row * HIDDEN + k_lane + item + 1]);
                        float x2 = float(value[row * HIDDEN + k_lane + item + 2]);
                        float x3 = float(value[row * HIDDEN + k_lane + item + 3]);
                        input_sum += x0 + x1 + x2 + x3;
                        input_values[item] = x0;
                        input_values[item + 1] = x1 / 16.0f;
                        input_values[item + 2] = x2 / 256.0f;
                        input_values[item + 3] = x3 / 4096.0f;
                    }
                    for (uint result_index = 0; result_index < 4; ++result_index) {
                        uint output_column = output_base + result_index;
                        uint weight_offset = output_column * (HIDDEN / 2) + k_lane / 2;
                        const device ushort* gate_packed =
                            reinterpret_cast<const device ushort*>(gate_bytes + weight_offset);
                        const device ushort* up_packed =
                            reinterpret_cast<const device ushort*>(up_bytes + weight_offset);
                        float gate_dot = 0.0f;
                        float up_dot = 0.0f;
                        for (uint piece = 0; piece < Q4_VALUES_PER_LANE / 4; ++piece) {
                            ushort gate_bits = gate_packed[piece];
                            ushort up_bits = up_packed[piece];
                            uint item = piece * 4;
                            gate_dot += input_values[item] * float(gate_bits & 0x000f)
                                + input_values[item + 1] * float(gate_bits & 0x00f0)
                                + input_values[item + 2] * float(gate_bits & 0x0f00)
                                + input_values[item + 3] * float(gate_bits & 0xf000);
                            up_dot += input_values[item] * float(up_bits & 0x000f)
                                + input_values[item + 1] * float(up_bits & 0x00f0)
                                + input_values[item + 2] * float(up_bits & 0x0f00)
                                + input_values[item + 3] * float(up_bits & 0xf000);
                        }
                        uint metadata_index = output_column * (HIDDEN / Q4_GROUP)
                            + k_lane / Q4_GROUP;
                        gate_result[result_index] +=
                            float(gate_scale_values[metadata_index]) * gate_dot
                            + input_sum * float(gate_bias_values[metadata_index]);
                        up_result[result_index] +=
                            float(up_scale_values[metadata_index]) * up_dot
                            + input_sum * float(up_bias_values[metadata_index]);
                    }
                }
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    float gate_sum = simd_sum(gate_result[result_index]);
                    float up_sum = simd_sum(up_result[result_index]);
                    if (lane == 0) {
                        bfloat gate_value = bfloat(gate_sum);
                        bfloat up_value = bfloat(up_sum);
                        auto sigmoid_y = 1 / (1 + metal::exp(metal::abs(gate_value)));
                        bfloat sigmoid_value = gate_value < bfloat(0.0f)
                            ? bfloat(sigmoid_y) : bfloat(1 - sigmoid_y);
                        bfloat silu = bfloat(gate_value * sigmoid_value);
                        uint output_column = output_base + result_index;
                        activations[(row * ACTIVATION_SLOTS + slot) * INTERMEDIATE
                            + output_column] = bfloat(silu * up_value);
                    }
                }
            }
        } else {
            const device uchar* shared_gate_bytes =
                reinterpret_cast<const device uchar*>(shared_gate_weight);
            const device uchar* shared_up_bytes =
                reinterpret_cast<const device uchar*>(shared_up_weight);
            float gate_result[ROWS][4] = {};
            float up_result[ROWS][4] = {};
            for (uint k_block = 0; k_block < HIDDEN; k_block += Q8_BLOCK) {
                uint k_lane = k_block + lane * Q8_VALUES_PER_LANE;
                float input_values[ROWS][Q8_VALUES_PER_LANE];
                float input_sum[ROWS] = {};
                for (uint row = 0; row < ROWS; ++row) {
                    for (uint item = 0; item < Q8_VALUES_PER_LANE; ++item) {
                        float x = float(value[row * HIDDEN + k_lane + item]);
                        input_values[row][item] = x;
                        input_sum[row] += x;
                    }
                }
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint output_column = output_base + result_index;
                    uint weight_offset = output_column * HIDDEN + k_lane;
                    uint metadata_index = output_column * (HIDDEN / Q8_GROUP)
                        + k_lane / Q8_GROUP;
                    float gate_scale = float(shared_gate_scales[metadata_index]);
                    float gate_bias = float(shared_gate_biases[metadata_index]);
                    float up_scale = float(shared_up_scales[metadata_index]);
                    float up_bias = float(shared_up_biases[metadata_index]);
                    for (uint row = 0; row < ROWS; ++row) {
                        float gate_dot = 0.0f;
                        float up_dot = 0.0f;
                        for (uint item = 0; item < Q8_VALUES_PER_LANE; ++item) {
                            gate_dot += input_values[row][item]
                                * float(shared_gate_bytes[weight_offset + item]);
                            up_dot += input_values[row][item]
                                * float(shared_up_bytes[weight_offset + item]);
                        }
                        gate_result[row][result_index] += gate_scale * gate_dot
                            + input_sum[row] * gate_bias;
                        up_result[row][result_index] += up_scale * up_dot
                            + input_sum[row] * up_bias;
                    }
                }
            }
            for (uint row = 0; row < ROWS; ++row) {
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    float gate_sum = simd_sum(gate_result[row][result_index]);
                    float up_sum = simd_sum(up_result[row][result_index]);
                    if (lane == 0) {
                        bfloat gate_value = bfloat(gate_sum);
                        bfloat up_value = bfloat(up_sum);
                        auto sigmoid_y = 1 / (1 + metal::exp(metal::abs(gate_value)));
                        bfloat sigmoid_value = gate_value < bfloat(0.0f)
                            ? bfloat(sigmoid_y) : bfloat(1 - sigmoid_y);
                        bfloat silu = bfloat(gate_value * sigmoid_value);
                        uint output_column = output_base + result_index;
                        activations[(row * ACTIVATION_SLOTS + TOP_K) * INTERMEDIATE
                            + output_column] = bfloat(silu * up_value);
                    }
                }
            }
        }
    """


def _stage3_source(rows: int) -> str:
    return _preamble(rows) + r"""
        constexpr uint OUTPUT_TILES = HIDDEN / 16;
        constexpr uint Q4_GROUP = 32;
        constexpr uint Q4_VALUES_PER_LANE = 16;
        constexpr uint Q4_BLOCK = Q4_VALUES_PER_LANE * 32;
        constexpr uint Q8_GROUP = 128;
        constexpr uint Q8_VALUES_PER_LANE = 16;
        constexpr uint Q8_BLOCK = Q8_VALUES_PER_LANE * 32;

        uint group = threadgroup_position_in_grid.x;
        uint row = group / OUTPUT_TILES;
        uint tile = group - row * OUTPUT_TILES;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        bfloat routed_accumulator[4] = {
            bfloat(0.0f), bfloat(0.0f), bfloat(0.0f), bfloat(0.0f)};
        for (uint slot = 0; slot < TOP_K; ++slot) {
            uint expert = expert_ids[row * TOP_K + slot];
            const device uchar* down_bytes =
                reinterpret_cast<const device uchar*>(routed_down_weight)
                + expert * HIDDEN * (INTERMEDIATE / 2);
            const device bfloat* down_scale_values = routed_down_scales
                + expert * HIDDEN * (INTERMEDIATE / Q4_GROUP);
            const device bfloat* down_bias_values = routed_down_biases
                + expert * HIDDEN * (INTERMEDIATE / Q4_GROUP);
            float down_result[4] = {};
            for (uint k_block = 0; k_block < INTERMEDIATE; k_block += Q4_BLOCK) {
                uint k_lane = k_block + lane * Q4_VALUES_PER_LANE;
                if (k_lane < INTERMEDIATE) {
                    float input_values[Q4_VALUES_PER_LANE];
                    float input_sum = 0.0f;
                    for (uint item = 0; item < Q4_VALUES_PER_LANE; item += 4) {
                        uint activation_base = (row * ACTIVATION_SLOTS + slot)
                            * INTERMEDIATE + k_lane + item;
                        float x0 = float(activations[activation_base]);
                        float x1 = float(activations[activation_base + 1]);
                        float x2 = float(activations[activation_base + 2]);
                        float x3 = float(activations[activation_base + 3]);
                        input_sum += x0 + x1 + x2 + x3;
                        input_values[item] = x0;
                        input_values[item + 1] = x1 / 16.0f;
                        input_values[item + 2] = x2 / 256.0f;
                        input_values[item + 3] = x3 / 4096.0f;
                    }
                    for (uint result_index = 0; result_index < 4; ++result_index) {
                        uint output_column = output_base + result_index;
                        uint weight_offset = output_column * (INTERMEDIATE / 2)
                            + k_lane / 2;
                        const device ushort* packed =
                            reinterpret_cast<const device ushort*>(down_bytes + weight_offset);
                        float quantized_dot = 0.0f;
                        for (uint piece = 0; piece < Q4_VALUES_PER_LANE / 4; ++piece) {
                            ushort bits = packed[piece];
                            uint item = piece * 4;
                            quantized_dot += input_values[item] * float(bits & 0x000f)
                                + input_values[item + 1] * float(bits & 0x00f0)
                                + input_values[item + 2] * float(bits & 0x0f00)
                                + input_values[item + 3] * float(bits & 0xf000);
                        }
                        uint metadata_index = output_column * (INTERMEDIATE / Q4_GROUP)
                            + k_lane / Q4_GROUP;
                        down_result[result_index] +=
                            float(down_scale_values[metadata_index]) * quantized_dot
                            + input_sum * float(down_bias_values[metadata_index]);
                    }
                }
            }
            for (uint result_index = 0; result_index < 4; ++result_index) {
                float down_sum = simd_sum(down_result[result_index]);
                if (lane == 0) {
                    bfloat down_value = bfloat(down_sum);
                    bfloat route_product = bfloat(float(down_value)
                        * float(route_scores[row * TOP_K + slot]));
                    routed_accumulator[result_index] = bfloat(
                        float(routed_accumulator[result_index])
                        + float(route_product));
                }
            }
        }

        const device uchar* shared_down_bytes =
            reinterpret_cast<const device uchar*>(shared_down_weight);
        float shared_result[4] = {};
        for (uint k_block = 0; k_block < INTERMEDIATE; k_block += Q8_BLOCK) {
            uint k_lane = k_block + lane * Q8_VALUES_PER_LANE;
            if (k_lane < INTERMEDIATE) {
                float input_values[Q8_VALUES_PER_LANE];
                float input_sum = 0.0f;
                for (uint item = 0; item < Q8_VALUES_PER_LANE; ++item) {
                    float x = float(activations[(row * ACTIVATION_SLOTS + TOP_K)
                        * INTERMEDIATE + k_lane + item]);
                    input_values[item] = x;
                    input_sum += x;
                }
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint output_column = output_base + result_index;
                    uint weight_offset = output_column * INTERMEDIATE + k_lane;
                    uint metadata_index = output_column * (INTERMEDIATE / Q8_GROUP)
                        + k_lane / Q8_GROUP;
                    float scale = float(shared_down_scales[metadata_index]);
                    float bias = float(shared_down_biases[metadata_index]);
                    float quantized_dot = 0.0f;
                    for (uint item = 0; item < Q8_VALUES_PER_LANE; ++item) {
                        quantized_dot += input_values[item]
                            * float(shared_down_bytes[weight_offset + item]);
                    }
                    shared_result[result_index] += scale * quantized_dot
                        + input_sum * bias;
                }
            }
        }
        for (uint result_index = 0; result_index < 4; ++result_index) {
            float shared_sum = simd_sum(shared_result[result_index]);
            if (lane == 0) {
                bfloat gate_value = shared_gate[row];
                auto sigmoid_y = 1 / (1 + metal::exp(metal::abs(gate_value)));
                bfloat sigmoid_value = gate_value < bfloat(0.0f)
                    ? bfloat(sigmoid_y) : bfloat(1 - sigmoid_y);
                bfloat shared_value = bfloat(shared_sum);
                bfloat gated_shared = bfloat(sigmoid_value * shared_value);
                uint output_column = output_base + result_index;
                output[row * HIDDEN + output_column] = bfloat(
                    float(routed_accumulator[result_index]) + float(gated_shared));
            }
        }
    """


def _stage3_residual_source(rows: int) -> str:
    stock_store = """output[row * HIDDEN + output_column] = bfloat(
                    float(routed_accumulator[result_index]) + float(gated_shared));"""
    fused_store = """bfloat mlp_value = bfloat(
                    float(routed_accumulator[result_index]) + float(gated_shared));
                for (uint stream = 0; stream < 4; ++stream) {
                    uint hidden_index = (row * 4 + stream) * HIDDEN + output_column;
                    bfloat inject_value = inject[row * 4 + stream];
                    bfloat product = bfloat(mlp_value * inject_value);
                    hidden[hidden_index] = bfloat(residual[hidden_index] + product);
                }"""
    source = _stage3_source(rows).replace(stock_store, fused_store)
    if source == _stage3_source(rows):
        raise RuntimeError("Qwen4 stage3 residual store substitution failed")
    return source


def sources(rows: int = 2) -> dict[str, str]:
    rows = _require_rows(rows)
    return {
        "stage1": _stage1_source(rows),
        "route": _route_source(rows),
        "stage2": _stage2_source(rows),
        "stage3": _stage3_source(rows),
        "stage3_residual": _stage3_residual_source(rows),
    }


def launch_geometry(
    rows: int = 2,
) -> dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]]:
    rows = _require_rows(rows)
    return {
        "stage1": (
            (STAGE1_GROUPS * STAGE1_THREADS, 1, 1),
            (STAGE1_THREADS, 1, 1),
        ),
        "stage2": ((STAGE2_GROUPS * THREADS, 1, 1), (THREADS, 1, 1)),
        "stage3": (
            (rows * (HIDDEN // 16) * THREADS, 1, 1),
            (THREADS, 1, 1),
        ),
    }


def _kernel(rows: int, key: str, input_names: list[str], output_name: str):
    cache_key = (rows, key)
    kernel = _KERNELS.get(cache_key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mtplx_qwen4_whole_moe_{key}_m{rows}",
            input_names=input_names,
            output_names=[output_name],
            source=sources(rows)[key],
            ensure_row_contiguous=True,
        )
        _KERNELS[cache_key] = kernel
    return kernel


def _stage1_kernel(rows: int):
    cache_key = (rows, "stage1")
    kernel = _KERNELS.get(cache_key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mtplx_qwen4_whole_moe_stage1_m{rows}",
            input_names=[
                "value",
                "router_weight",
                "shared_gate_weight",
                "shared_gate_scales",
                "shared_gate_biases",
            ],
            output_names=["router_logits", "shared_gate"],
            source=sources(rows)["stage1"],
            ensure_row_contiguous=True,
        )
        _KERNELS[cache_key] = kernel
    return kernel


def _route_kernel(rows: int):
    cache_key = (rows, "route")
    kernel = _KERNELS.get(cache_key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mtplx_qwen4_whole_moe_route_top10_m{rows}",
            input_names=["router_logits"],
            output_names=["expert_ids", "selected_logits"],
            source=sources(rows)["route"],
            ensure_row_contiguous=True,
        )
        _KERNELS[cache_key] = kernel
    return kernel


def bind_stages(
    *,
    rows: int,
    router: Any,
    routed: Any,
    shared: Any,
    shared_gate: Any,
):
    """Bind invariant kernels and weight arrays for one exact row count."""
    rows = _require_rows(rows)
    stage1_kernel = _stage1_kernel(rows)
    route_kernel = _route_kernel(rows)
    stage1_static = (
        router.weight,
        shared_gate.weight,
        shared_gate.scales,
        shared_gate.biases,
    )
    stage2_kernel = _kernel(
        rows,
        "stage2",
        [
            "value", "expert_ids",
            "routed_gate_weight", "routed_gate_scales", "routed_gate_biases",
            "routed_up_weight", "routed_up_scales", "routed_up_biases",
            "shared_gate_weight", "shared_gate_scales", "shared_gate_biases",
            "shared_up_weight", "shared_up_scales", "shared_up_biases",
        ],
        "activations",
    )
    stage2_static = tuple(
        array
        for projection in (
            routed.gate_proj,
            routed.up_proj,
            shared.gate_proj,
            shared.up_proj,
        )
        for array in (projection.weight, projection.scales, projection.biases)
    )
    stage3_kernel = _kernel(
        rows,
        "stage3",
        [
            "activations", "expert_ids", "route_scores", "shared_gate",
            "routed_down_weight", "routed_down_scales", "routed_down_biases",
            "shared_down_weight", "shared_down_scales", "shared_down_biases",
        ],
        "output",
    )
    stage3_static = (
        routed.down_proj.weight,
        routed.down_proj.scales,
        routed.down_proj.biases,
        shared.down_proj.weight,
        shared.down_proj.scales,
        shared.down_proj.biases,
    )
    stage3_residual_kernel = _kernel(
        rows,
        "stage3_residual",
        [
            "activations", "expert_ids", "route_scores", "shared_gate",
            "residual", "inject",
            "routed_down_weight", "routed_down_scales", "routed_down_biases",
            "shared_down_weight", "shared_down_scales", "shared_down_biases",
        ],
        "hidden",
    )

    def stage1(value: Any):
        return stage1_kernel(
            inputs=[value, *stage1_static],
            grid=(STAGE1_GROUPS * STAGE1_THREADS, 1, 1),
            threadgroup=(STAGE1_THREADS, 1, 1),
            output_shapes=[(rows, EXPERTS), (rows,)],
            output_dtypes=[mx.float32, mx.bfloat16],
        )

    def route_top10(router_logits: Any):
        expert_ids, selected_logits = route_kernel(
            inputs=[router_logits],
            grid=(rows * ROUTE_THREADS, 1, 1),
            threadgroup=(ROUTE_THREADS, 1, 1),
            output_shapes=[(rows, TOP_K), (rows, TOP_K)],
            output_dtypes=[mx.uint32, mx.float32],
        )
        return expert_ids, selected_logits

    def stage2(value: Any, expert_ids: Any):
        (activations,) = stage2_kernel(
            inputs=[value, expert_ids, *stage2_static],
            grid=(STAGE2_GROUPS * THREADS, 1, 1),
            threadgroup=(THREADS, 1, 1),
            output_shapes=[(rows, ACTIVATION_SLOTS, INTERMEDIATE)],
            output_dtypes=[mx.bfloat16],
        )
        return activations

    def stage3(
        activations: Any,
        expert_ids: Any,
        route_scores: Any,
        shared_gate_values: Any,
    ):
        (output,) = stage3_kernel(
            inputs=[
                activations,
                expert_ids,
                route_scores,
                shared_gate_values,
                *stage3_static,
            ],
            grid=(rows * (HIDDEN // 16) * THREADS, 1, 1),
            threadgroup=(THREADS, 1, 1),
            output_shapes=[(rows, HIDDEN)],
            output_dtypes=[mx.bfloat16],
        )
        return output

    def stage3_residual(
        activations: Any,
        expert_ids: Any,
        route_scores: Any,
        shared_gate_values: Any,
        residual: Any,
        inject: Any,
    ):
        (hidden,) = stage3_residual_kernel(
            inputs=[
                activations,
                expert_ids,
                route_scores,
                shared_gate_values,
                residual,
                inject,
                *stage3_static,
            ],
            grid=(rows * (HIDDEN // 16) * THREADS, 1, 1),
            threadgroup=(THREADS, 1, 1),
            output_shapes=[(rows, 4 * HIDDEN)],
            output_dtypes=[mx.bfloat16],
        )
        return hidden

    return stage1, route_top10, stage2, stage3, stage3_residual


def stage1(value: Any, binding: Any, *, rows: int = 2):
    rows = _require_rows(rows)
    shared_gate = binding.shared_gate
    kernel = _stage1_kernel(rows)
    router = binding.router
    router_weight = router.weight
    (router_logits, shared_gate_values) = kernel(
        inputs=[
            value,
            router_weight,
            shared_gate.weight,
            shared_gate.scales,
            shared_gate.biases,
        ],
        grid=(STAGE1_GROUPS * STAGE1_THREADS, 1, 1),
        threadgroup=(STAGE1_THREADS, 1, 1),
        output_shapes=[(rows, EXPERTS), (rows,)],
        output_dtypes=[mx.float32, mx.bfloat16],
    )
    return router_logits, shared_gate_values


def route_top10(router_logits: Any, *, rows: int = 2):
    rows = _require_rows(rows)
    kernel = _route_kernel(rows)
    expert_ids, selected_logits = kernel(
        inputs=[router_logits],
        grid=(rows * ROUTE_THREADS, 1, 1),
        threadgroup=(ROUTE_THREADS, 1, 1),
        output_shapes=[(rows, TOP_K), (rows, TOP_K)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return expert_ids, selected_logits


def stage2(value: Any, expert_ids: Any, binding: Any, *, rows: int = 2):
    rows = _require_rows(rows)
    routed = binding.routed
    shared = binding.shared
    kernel = _kernel(
        rows,
        "stage2",
        [
            "value", "expert_ids",
            "routed_gate_weight", "routed_gate_scales", "routed_gate_biases",
            "routed_up_weight", "routed_up_scales", "routed_up_biases",
            "shared_gate_weight", "shared_gate_scales", "shared_gate_biases",
            "shared_up_weight", "shared_up_scales", "shared_up_biases",
        ],
        "activations",
    )
    inputs = [value, expert_ids]
    for projection in (routed.gate_proj, routed.up_proj, shared.gate_proj, shared.up_proj):
        inputs.extend([projection.weight, projection.scales, projection.biases])
    (activations,) = kernel(
        inputs=inputs,
        grid=(STAGE2_GROUPS * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, ACTIVATION_SLOTS, INTERMEDIATE)],
        output_dtypes=[mx.bfloat16],
    )
    return activations


def stage3(
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
    *,
    rows: int = 2,
):
    rows = _require_rows(rows)
    routed_down = binding.routed.down_proj
    shared_down = binding.shared.down_proj
    kernel = _kernel(
        rows,
        "stage3",
        [
            "activations", "expert_ids", "route_scores", "shared_gate",
            "routed_down_weight", "routed_down_scales", "routed_down_biases",
            "shared_down_weight", "shared_down_scales", "shared_down_biases",
        ],
        "output",
    )
    (output,) = kernel(
        inputs=[
            activations, expert_ids, route_scores, shared_gate,
            routed_down.weight, routed_down.scales, routed_down.biases,
            shared_down.weight, shared_down.scales, shared_down.biases,
        ],
        grid=(rows * (HIDDEN // 16) * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, HIDDEN)],
        output_dtypes=[mx.bfloat16],
    )
    return output


__all__ = [
    "bind_stages",
    "launch_geometry",
    "route_top10",
    "sources",
    "stage1",
    "stage2",
    "stage3",
]
