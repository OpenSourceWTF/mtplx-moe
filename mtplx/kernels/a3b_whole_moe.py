"""Fixed Metal entrypoints for exact A3B whole-MoE small-row stages."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


HIDDEN = 2048
EXPERTS = 256
TOP_K = 8
INTERMEDIATE = 512
ACTIVATION_SLOTS = 9
STAGE1_THREADS = 256
TILED_THREADS = 128
STAGE2_THREADGROUPS = ACTIVATION_SLOTS * (INTERMEDIATE // 16)
STAGE3_THREADGROUPS = HIDDEN // 16

_KERNELS: dict[str, Any] = {}


def _fixed_source(*, stage: int, rows: int, variant: str) -> str:
    common = f"""
        using namespace metal;

        constexpr uint HIDDEN = {HIDDEN};
        constexpr uint EXPERTS = {EXPERTS};
        constexpr uint TOP_K = {TOP_K};
        constexpr uint INTERMEDIATE = {INTERMEDIATE};
        constexpr uint ACTIVATION_SLOTS = {ACTIVATION_SLOTS};
        constexpr uint ROWS = {rows};
    """
    if stage == 1:
        return common + _stage1_source(target=variant.startswith("target"))
    if stage == 2:
        return common + _stage2_source(target=variant.startswith("target"))
    return common + _stage3_source(target=variant.startswith("target"))


def _stage1_source(*, target: bool) -> str:
    projection = _target_stage1_projection() if target else _mtp_stage1_projection()
    prologue = """
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint row = threadgroup_position_in_grid.x;

        threadgroup bfloat router_logits[ROWS * EXPERTS];
        threadgroup bfloat probabilities[ROWS * EXPERTS];
        threadgroup float simd_values[8];
        threadgroup float local_probabilities[64];
        threadgroup int local_indices[64];
        threadgroup float merged_probabilities[TOP_K];
        threadgroup int merged_indices[TOP_K];
    """
    finalize = """
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_logit = float(router_logits[row * EXPERTS + tid]);
        float local_maximum = simd_max(local_logit);
        if (lane == 0) {
            simd_values[simd_gid] = local_maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float maximum_candidate = lane < 8 ? simd_values[lane] : -INFINITY;
        float row_maximum = simd_max(maximum_candidate);
        if (simd_gid == 0 && lane == 0) {
            simd_values[0] = row_maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float probability = metal::exp(local_logit - simd_values[0]);
        float local_sum = simd_sum(probability);
        if (lane == 0) {
            simd_values[simd_gid] = local_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum_candidate = lane < 8 ? simd_values[lane] : 0.0f;
        float row_sum = simd_sum(sum_candidate);
        if (simd_gid == 0 && lane == 0) {
            simd_values[0] = row_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        probabilities[row * EXPERTS + tid] = bfloat(
            probability / simd_values[0]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float candidate_probability = float(
            probabilities[row * EXPERTS + tid]);
        int candidate_index = int(tid);
        for (int rank = 0; rank < int(TOP_K); ++rank) {
            float winner_probability = simd_max(candidate_probability);
            float winner_index_value = simd_max(
                candidate_probability == winner_probability
                    ? float(candidate_index)
                    : -1.0f);
            int winner_index = int(winner_index_value);
            if (lane == 0) {
                int destination = int(simd_gid * TOP_K) + rank;
                local_probabilities[destination] = winner_probability;
                local_indices[destination] = winner_index;
            }
            if (candidate_index == winner_index) {
                candidate_probability = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_gid == 0) {
            int slot0 = int(lane);
            int slot1 = int(lane) + 32;
            float probability0 = local_probabilities[slot0];
            float probability1 = local_probabilities[slot1];
            int index0 = local_indices[slot0];
            int index1 = local_indices[slot1];

            for (int rank = 0; rank < int(TOP_K); ++rank) {
                bool take1 = probability1 > probability0
                    || (probability1 == probability0 && index1 > index0);
                float lane_probability = take1 ? probability1 : probability0;
                int lane_index = take1 ? index1 : index0;
                float winner_probability = simd_max(lane_probability);
                float winner_index_value = simd_max(
                    lane_probability == winner_probability
                        ? float(lane_index)
                        : -1.0f);
                int winner_index = int(winner_index_value);
                if (lane == 0) {
                    merged_probabilities[rank] = winner_probability;
                    merged_indices[rank] = winner_index;
                }
                if (lane_index == winner_index) {
                    if (take1) {
                        probability1 = -INFINITY;
                    } else {
                        probability0 = -INFINITY;
                    }
                }
            }

            if (lane == 0) {
                bfloat rounded_denominator = bfloat(0.0f);
                for (int output_rank = 0; output_rank < int(TOP_K); ++output_rank) {
                    rounded_denominator = bfloat(
                        float(rounded_denominator)
                        + merged_probabilities[TOP_K - 1 - output_rank]);
                }
                for (int output_rank = 0; output_rank < int(TOP_K); ++output_rank) {
                    int source_rank = int(TOP_K) - 1 - output_rank;
                    int destination = int(row * TOP_K) + output_rank;
                    expert_ids[destination] = uint(merged_indices[source_rank]);
                    route_scores[destination] = bfloat(
                        merged_probabilities[source_rank]
                        / float(rounded_denominator));
                }
            }
        }
    """
    return prologue + projection + finalize


def _target_stage1_projection() -> str:
    return """
        constexpr uint ROUTER_GROUP = 64;
        constexpr uint Q8_VALUES_PER_LANE = 8;
        constexpr uint Q8_BLOCK = Q8_VALUES_PER_LANE * 32;

        // qdot8_affine: preserve the accepted q8 QMV lane decomposition.
        float router_result[4];
        for (uint subtile = 0; subtile < 8; ++subtile) {
            for (uint result_index = 0; result_index < 4; ++result_index) {
                router_result[result_index] = 0.0f;
            }
            uint expert_base = subtile * 32 + simd_gid * 4;
            for (uint k_block = 0; k_block < HIDDEN; k_block += Q8_BLOCK) {
                uint k_lane = k_block + lane * Q8_VALUES_PER_LANE;
                float input_values[Q8_VALUES_PER_LANE];
                float input_sum = 0.0f;
                for (uint item = 0; item < Q8_VALUES_PER_LANE; ++item) {
                    float input_value = float(value[row * HIDDEN + k_lane + item]);
                    input_values[item] = input_value;
                    input_sum += input_value;
                }
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint expert = expert_base + result_index;
                    uint weight_base = expert * HIDDEN + k_lane;
                    uint metadata_index = expert * (HIDDEN / ROUTER_GROUP)
                        + k_lane / ROUTER_GROUP;
                    float scale = float(router_scales[metadata_index]);
                    float bias = float(router_biases[metadata_index]);
                    float quantized_dot = 0.0f;
                    const device uchar* weights =
                        reinterpret_cast<const device uchar*>(router_weight);
                    for (uint item = 0; item < Q8_VALUES_PER_LANE; ++item) {
                        quantized_dot += input_values[item]
                            * float(weights[weight_base + item]);
                    }
                    router_result[result_index] +=
                        scale * quantized_dot + input_sum * bias;
                }
            }
            for (uint result_index = 0; result_index < 4; ++result_index) {
                float reduced = simd_sum(router_result[result_index]);
                if (lane == 0) {
                    uint expert = expert_base + result_index;
                    router_logits[row * EXPERTS + expert] = bfloat(reduced);
                }
            }
        }

        float shared_partial = 0.0f;
        if (simd_gid == 0) {
            for (uint k_block = 0; k_block < HIDDEN; k_block += Q8_BLOCK) {
                uint k_lane = k_block + lane * Q8_VALUES_PER_LANE;
                float input_sum = 0.0f;
                float quantized_dot = 0.0f;
                uint metadata_index = k_lane / ROUTER_GROUP;
                float scale = float(shared_gate_scales[metadata_index]);
                float bias = float(shared_gate_biases[metadata_index]);
                const device uchar* weights =
                    reinterpret_cast<const device uchar*>(shared_gate_weight);
                for (uint item = 0; item < Q8_VALUES_PER_LANE; ++item) {
                    float input_value = float(value[row * HIDDEN + k_lane + item]);
                    input_sum += input_value;
                    quantized_dot += input_value * float(weights[k_lane + item]);
                }
                shared_partial += scale * quantized_dot + input_sum * bias;
            }
            float shared_reduced = simd_sum(shared_partial);
            if (lane == 0) {
                shared_gate[row] = bfloat(shared_reduced);
            }
        }

    """


def _mtp_stage1_projection() -> str:
    return """
        // dense_bf16_dot: one SIMDgroup owns four router outputs at a time.
        float router_result[4];
        for (uint subtile = 0; subtile < 8; ++subtile) {
            for (uint result_index = 0; result_index < 4; ++result_index) {
                router_result[result_index] = 0.0f;
            }
            uint expert_base = subtile * 32 + simd_gid * 4;
            for (uint k_lane = lane; k_lane < HIDDEN; k_lane += 32) {
                float input_value = float(value[row * HIDDEN + k_lane]);
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint expert = expert_base + result_index;
                    router_result[result_index] += input_value
                        * float(router_weight[expert * HIDDEN + k_lane]);
                }
            }
            for (uint result_index = 0; result_index < 4; ++result_index) {
                float reduced = simd_sum(router_result[result_index]);
                if (lane == 0) {
                    uint expert = expert_base + result_index;
                    router_logits[row * EXPERTS + expert] = bfloat(reduced);
                }
            }
        }

        float shared_partial = 0.0f;
        if (simd_gid == 0) {
            for (uint k_lane = lane; k_lane < HIDDEN; k_lane += 32) {
                shared_partial += float(value[row * HIDDEN + k_lane])
                    * float(shared_gate_weight[k_lane]);
            }
            float shared_reduced = simd_sum(shared_partial);
            if (lane == 0) {
                shared_gate[row] = bfloat(shared_reduced);
            }
        }

    """


def _stage2_source(*, target: bool) -> str:
    if target:
        return _target_stage2_source()
    return _mtp_stage2_source()


def _target_stage2_source() -> str:
    return """
        constexpr uint ROUTED_GROUP = 64;
        constexpr uint VALUES_PER_LANE = 16;
        constexpr uint K_BLOCK = VALUES_PER_LANE * 32;

        uint group = threadgroup_position_in_grid.x;
        uint slot = group / 32;
        uint tile = group - slot * 32;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        for (uint row = 0; row < ROWS; ++row) {
            uint expert = slot < TOP_K
                ? expert_ids[row * TOP_K + slot]
                : uint(0);
            const device uint* gate_words = slot < TOP_K
                ? routed_gate_up_weight
                    + expert * 2 * INTERMEDIATE * (HIDDEN / 8)
                : shared_gate_up_weight;
            const device uint* up_words = gate_words
                + INTERMEDIATE * (HIDDEN / 8);
            const device bfloat* gate_scale_values = slot < TOP_K
                ? routed_gate_up_scales
                    + expert * 2 * INTERMEDIATE * (HIDDEN / ROUTED_GROUP)
                : shared_gate_up_scales;
            const device bfloat* gate_bias_values = slot < TOP_K
                ? routed_gate_up_biases
                    + expert * 2 * INTERMEDIATE * (HIDDEN / ROUTED_GROUP)
                : shared_gate_up_biases;
            const device bfloat* up_scale_values = gate_scale_values
                + INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
            const device bfloat* up_bias_values = gate_bias_values
                + INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
            const device uchar* gate_bytes =
                reinterpret_cast<const device uchar*>(gate_words);
            const device uchar* up_bytes =
                reinterpret_cast<const device uchar*>(up_words);

            float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint k_block = 0; k_block < HIDDEN; k_block += K_BLOCK) {
                uint k_lane = k_block + lane * VALUES_PER_LANE;
                float input_values[VALUES_PER_LANE];
                float input_sum = 0.0f;
                for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
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
                        reinterpret_cast<const device ushort*>(
                            gate_bytes + weight_offset);
                    const device ushort* up_packed =
                        reinterpret_cast<const device ushort*>(
                            up_bytes + weight_offset);
                    float gate_quantized_dot = 0.0f;
                    float up_quantized_dot = 0.0f;
                    // qdot4_affine
                    for (uint piece = 0; piece < VALUES_PER_LANE / 4; ++piece) {
                        ushort gate_bits = gate_packed[piece];
                        ushort up_bits = up_packed[piece];
                        uint item = piece * 4;
                        gate_quantized_dot +=
                            input_values[item] * float(gate_bits & 0x000f)
                            + input_values[item + 1] * float(gate_bits & 0x00f0)
                            + input_values[item + 2] * float(gate_bits & 0x0f00)
                            + input_values[item + 3] * float(gate_bits & 0xf000);
                        up_quantized_dot +=
                            input_values[item] * float(up_bits & 0x000f)
                            + input_values[item + 1] * float(up_bits & 0x00f0)
                            + input_values[item + 2] * float(up_bits & 0x0f00)
                            + input_values[item + 3] * float(up_bits & 0xf000);
                    }
                    uint metadata_index = output_column * (HIDDEN / ROUTED_GROUP)
                        + k_lane / ROUTED_GROUP;
                    gate_result[result_index] +=
                        float(gate_scale_values[metadata_index]) * gate_quantized_dot
                        + input_sum * float(gate_bias_values[metadata_index]);
                    up_result[result_index] +=
                        float(up_scale_values[metadata_index]) * up_quantized_dot
                        + input_sum * float(up_bias_values[metadata_index]);
                }
            }

            for (uint result_index = 0; result_index < 4; ++result_index) {
                float gate_sum = simd_sum(gate_result[result_index]);
                float up_sum = simd_sum(up_result[result_index]);
                if (lane == 0) {
                    bfloat gate_value = bfloat(gate_sum);
                    bfloat up_value = bfloat(up_sum);
                    auto sigmoid_y = 1 / (
                        1 + metal::exp(metal::abs(gate_value)));
                    bfloat sigmoid_mlx_exact = gate_value < bfloat(0.0f)
                        ? bfloat(sigmoid_y)
                        : bfloat(1 - sigmoid_y);
                    bfloat silu = bfloat(gate_value * sigmoid_mlx_exact);
                    uint output_column = output_base + result_index;
                    uint output_index =
                        (row * ACTIVATION_SLOTS + slot) * INTERMEDIATE
                        + output_column;
                    activations[output_index] = bfloat(silu * up_value);
                }
            }
        }
    """


def _mtp_stage2_source() -> str:
    return """
        constexpr uint ROUTED_GROUP = 32;
        constexpr uint VALUES_PER_LANE = 16;
        constexpr uint K_BLOCK = VALUES_PER_LANE * 32;

        uint group = threadgroup_position_in_grid.x;
        uint slot = group / 32;
        uint tile = group - slot * 32;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        for (uint row = 0; row < ROWS; ++row) {
            float gate_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float up_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (slot < TOP_K) {
                uint expert = expert_ids[row * TOP_K + slot];
                const device uint* gate_words = routed_gate_up_weight
                    + expert * 2 * INTERMEDIATE * (HIDDEN / 8);
                const device uchar* gate_bytes =
                    reinterpret_cast<const device uchar*>(gate_words);
                const device uchar* up_bytes =
                    reinterpret_cast<const device uchar*>(
                        gate_words + INTERMEDIATE * (HIDDEN / 8));
                const device bfloat* gate_scale_values = routed_gate_up_scales
                    + expert * 2 * INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
                const device bfloat* gate_bias_values = routed_gate_up_biases
                    + expert * 2 * INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
                const device bfloat* up_scale_values = gate_scale_values
                    + INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
                const device bfloat* up_bias_values = gate_bias_values
                    + INTERMEDIATE * (HIDDEN / ROUTED_GROUP);
                for (uint k_block = 0; k_block < HIDDEN; k_block += K_BLOCK) {
                    uint k_lane = k_block + lane * VALUES_PER_LANE;
                    float input_values[VALUES_PER_LANE];
                    float input_sum = 0.0f;
                    for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
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
                        uint weight_offset =
                            output_column * (HIDDEN / 2) + k_lane / 2;
                        const device ushort* gate_packed =
                            reinterpret_cast<const device ushort*>(
                                gate_bytes + weight_offset);
                        const device ushort* up_packed =
                            reinterpret_cast<const device ushort*>(
                                up_bytes + weight_offset);
                        float gate_quantized_dot = 0.0f;
                        float up_quantized_dot = 0.0f;
                        // qdot4_affine
                        for (uint piece = 0;
                             piece < VALUES_PER_LANE / 4;
                             ++piece) {
                            ushort gate_bits = gate_packed[piece];
                            ushort up_bits = up_packed[piece];
                            uint item = piece * 4;
                            gate_quantized_dot +=
                                input_values[item] * float(gate_bits & 0x000f)
                                + input_values[item + 1] * float(gate_bits & 0x00f0)
                                + input_values[item + 2] * float(gate_bits & 0x0f00)
                                + input_values[item + 3] * float(gate_bits & 0xf000);
                            up_quantized_dot +=
                                input_values[item] * float(up_bits & 0x000f)
                                + input_values[item + 1] * float(up_bits & 0x00f0)
                                + input_values[item + 2] * float(up_bits & 0x0f00)
                                + input_values[item + 3] * float(up_bits & 0xf000);
                        }
                        uint metadata_index =
                            output_column * (HIDDEN / ROUTED_GROUP)
                            + k_lane / ROUTED_GROUP;
                        gate_result[result_index] +=
                            float(gate_scale_values[metadata_index])
                                * gate_quantized_dot
                            + input_sum * float(gate_bias_values[metadata_index]);
                        up_result[result_index] +=
                            float(up_scale_values[metadata_index])
                                * up_quantized_dot
                            + input_sum * float(up_bias_values[metadata_index]);
                    }
                }
            } else {
                // dense_shared_dot
                for (uint k_lane = lane; k_lane < HIDDEN; k_lane += 32) {
                    float input_value = float(value[row * HIDDEN + k_lane]);
                    for (uint result_index = 0; result_index < 4; ++result_index) {
                        uint output_column = output_base + result_index;
                        gate_result[result_index] += input_value * float(
                            shared_gate_up_weight[
                                output_column * HIDDEN + k_lane]);
                        up_result[result_index] += input_value * float(
                            shared_gate_up_weight[
                                (INTERMEDIATE + output_column) * HIDDEN + k_lane]);
                    }
                }
            }

            for (uint result_index = 0; result_index < 4; ++result_index) {
                float gate_sum = simd_sum(gate_result[result_index]);
                float up_sum = simd_sum(up_result[result_index]);
                if (lane == 0) {
                    bfloat gate_value = bfloat(gate_sum);
                    bfloat up_value = bfloat(up_sum);
                    auto sigmoid_y = 1 / (
                        1 + metal::exp(metal::abs(gate_value)));
                    bfloat sigmoid_mlx_exact = gate_value < bfloat(0.0f)
                        ? bfloat(sigmoid_y)
                        : bfloat(1 - sigmoid_y);
                    bfloat silu = bfloat(gate_value * sigmoid_mlx_exact);
                    uint output_column = output_base + result_index;
                    uint output_index =
                        (row * ACTIVATION_SLOTS + slot) * INTERMEDIATE
                        + output_column;
                    activations[output_index] = bfloat(silu * up_value);
                }
            }
        }
    """


def _stage3_source(*, target: bool) -> str:
    if target:
        return _target_stage3_source()
    return _mtp_stage3_source()


def _target_stage3_source() -> str:
    return """
        constexpr uint ROUTED_GROUP = 64;
        constexpr uint VALUES_PER_LANE = 16;

        uint tile = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        for (uint row = 0; row < ROWS; ++row) {
            bfloat routed_accumulator[4] = {
                bfloat(0.0f), bfloat(0.0f), bfloat(0.0f), bfloat(0.0f)};
            for (uint slot = 0; slot < TOP_K; ++slot) {
                uint expert = expert_ids[row * TOP_K + slot];
                const device uchar* down_bytes =
                    reinterpret_cast<const device uchar*>(
                        routed_down_weight
                        + expert * HIDDEN * (INTERMEDIATE / 8));
                const device bfloat* down_scale_values = routed_down_scales
                    + expert * HIDDEN * (INTERMEDIATE / ROUTED_GROUP);
                const device bfloat* down_bias_values = routed_down_biases
                    + expert * HIDDEN * (INTERMEDIATE / ROUTED_GROUP);
                uint k_lane = lane * VALUES_PER_LANE;
                float input_values[VALUES_PER_LANE];
                float input_sum = 0.0f;
                for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
                    uint activation_base =
                        (row * ACTIVATION_SLOTS + slot) * INTERMEDIATE
                        + k_lane + item;
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
                float down_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint output_column = output_base + result_index;
                    uint weight_offset =
                        output_column * (INTERMEDIATE / 2) + k_lane / 2;
                    const device ushort* down_packed =
                        reinterpret_cast<const device ushort*>(
                            down_bytes + weight_offset);
                    float quantized_dot = 0.0f;
                    // qdot4_affine
                    for (uint piece = 0; piece < VALUES_PER_LANE / 4; ++piece) {
                        ushort packed = down_packed[piece];
                        uint item = piece * 4;
                        quantized_dot +=
                            input_values[item] * float(packed & 0x000f)
                            + input_values[item + 1] * float(packed & 0x00f0)
                            + input_values[item + 2] * float(packed & 0x0f00)
                            + input_values[item + 3] * float(packed & 0xf000);
                    }
                    uint metadata_index =
                        output_column * (INTERMEDIATE / ROUTED_GROUP)
                        + k_lane / ROUTED_GROUP;
                    down_result[result_index] =
                        float(down_scale_values[metadata_index]) * quantized_dot
                        + input_sum * float(down_bias_values[metadata_index]);
                }
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    float down_sum = simd_sum(down_result[result_index]);
                    if (lane == 0) {
                        bfloat down_value = bfloat(down_sum);
                        bfloat route_product = bfloat(
                            float(down_value)
                            * float(route_scores[row * TOP_K + slot]));
                        routed_accumulator[result_index] = bfloat(
                            float(routed_accumulator[result_index])
                            + float(route_product));
                    }
                }
            }

            uint shared_k_lane = lane * VALUES_PER_LANE;
            float shared_inputs[VALUES_PER_LANE];
            float shared_input_sum = 0.0f;
            for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
                uint activation_base =
                    (row * ACTIVATION_SLOTS + TOP_K) * INTERMEDIATE
                    + shared_k_lane + item;
                float x0 = float(activations[activation_base]);
                float x1 = float(activations[activation_base + 1]);
                float x2 = float(activations[activation_base + 2]);
                float x3 = float(activations[activation_base + 3]);
                shared_input_sum += x0 + x1 + x2 + x3;
                shared_inputs[item] = x0;
                shared_inputs[item + 1] = x1 / 16.0f;
                shared_inputs[item + 2] = x2 / 256.0f;
                shared_inputs[item + 3] = x3 / 4096.0f;
            }
            const device uchar* shared_down_bytes =
                reinterpret_cast<const device uchar*>(shared_down_weight);
            float shared_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint result_index = 0; result_index < 4; ++result_index) {
                uint output_column = output_base + result_index;
                uint weight_offset =
                    output_column * (INTERMEDIATE / 2) + shared_k_lane / 2;
                const device ushort* down_packed =
                    reinterpret_cast<const device ushort*>(
                        shared_down_bytes + weight_offset);
                float quantized_dot = 0.0f;
                // qdot4_affine shared
                for (uint piece = 0; piece < VALUES_PER_LANE / 4; ++piece) {
                    ushort packed = down_packed[piece];
                    uint item = piece * 4;
                    quantized_dot +=
                        shared_inputs[item] * float(packed & 0x000f)
                        + shared_inputs[item + 1] * float(packed & 0x00f0)
                        + shared_inputs[item + 2] * float(packed & 0x0f00)
                        + shared_inputs[item + 3] * float(packed & 0xf000);
                }
                uint metadata_index =
                    output_column * (INTERMEDIATE / ROUTED_GROUP)
                    + shared_k_lane / ROUTED_GROUP;
                shared_result[result_index] =
                    float(shared_down_scales[metadata_index]) * quantized_dot
                    + shared_input_sum * float(shared_down_biases[metadata_index]);
            }

            for (uint result_index = 0; result_index < 4; ++result_index) {
                float shared_sum = simd_sum(shared_result[result_index]);
                if (lane == 0) {
                    bfloat shared_value = bfloat(shared_sum);
                    bfloat gate_value = shared_gate[row];
                    auto sigmoid_y = 1 / (
                        1 + metal::exp(metal::abs(gate_value)));
                    bfloat sigmoid_mlx_exact = gate_value < bfloat(0.0f)
                        ? bfloat(sigmoid_y)
                        : bfloat(1 - sigmoid_y);
                    bfloat gated_shared = bfloat(
                        sigmoid_mlx_exact * shared_value);
                    uint output_column = output_base + result_index;
                    uint output_index = row * HIDDEN + output_column;
                    output[output_index] = bfloat(
                        float(routed_accumulator[result_index])
                        + float(gated_shared));
                }
            }
        }
    """


def _mtp_stage3_source() -> str:
    return """
        constexpr uint ROUTED_GROUP = 32;
        constexpr uint VALUES_PER_LANE = 16;

        uint tile = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint output_base = tile * 16 + simd_gid * 4;

        for (uint row = 0; row < ROWS; ++row) {
            bfloat routed_accumulator[4] = {
                bfloat(0.0f), bfloat(0.0f), bfloat(0.0f), bfloat(0.0f)};
            for (uint slot = 0; slot < TOP_K; ++slot) {
                uint expert = expert_ids[row * TOP_K + slot];
                const device uchar* down_bytes =
                    reinterpret_cast<const device uchar*>(
                        routed_down_weight
                        + expert * HIDDEN * (INTERMEDIATE / 8));
                const device bfloat* down_scale_values = routed_down_scales
                    + expert * HIDDEN * (INTERMEDIATE / ROUTED_GROUP);
                const device bfloat* down_bias_values = routed_down_biases
                    + expert * HIDDEN * (INTERMEDIATE / ROUTED_GROUP);
                uint k_lane = lane * VALUES_PER_LANE;
                float input_values[VALUES_PER_LANE];
                float input_sum = 0.0f;
                for (uint item = 0; item < VALUES_PER_LANE; item += 4) {
                    uint activation_base =
                        (row * ACTIVATION_SLOTS + slot) * INTERMEDIATE
                        + k_lane + item;
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
                float down_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint output_column = output_base + result_index;
                    uint weight_offset =
                        output_column * (INTERMEDIATE / 2) + k_lane / 2;
                    const device ushort* down_packed =
                        reinterpret_cast<const device ushort*>(
                            down_bytes + weight_offset);
                    float quantized_dot = 0.0f;
                    // qdot4_affine
                    for (uint piece = 0; piece < VALUES_PER_LANE / 4; ++piece) {
                        ushort packed = down_packed[piece];
                        uint item = piece * 4;
                        quantized_dot +=
                            input_values[item] * float(packed & 0x000f)
                            + input_values[item + 1] * float(packed & 0x00f0)
                            + input_values[item + 2] * float(packed & 0x0f00)
                            + input_values[item + 3] * float(packed & 0xf000);
                    }
                    uint metadata_index =
                        output_column * (INTERMEDIATE / ROUTED_GROUP)
                        + k_lane / ROUTED_GROUP;
                    down_result[result_index] =
                        float(down_scale_values[metadata_index]) * quantized_dot
                        + input_sum * float(down_bias_values[metadata_index]);
                }
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    float down_sum = simd_sum(down_result[result_index]);
                    if (lane == 0) {
                        bfloat down_value = bfloat(down_sum);
                        bfloat route_product = bfloat(
                            float(down_value)
                            * float(route_scores[row * TOP_K + slot]));
                        routed_accumulator[result_index] = bfloat(
                            float(routed_accumulator[result_index])
                            + float(route_product));
                    }
                }
            }

            // dense_shared_down
            float shared_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint k_lane = lane; k_lane < INTERMEDIATE; k_lane += 32) {
                float input_value = float(
                    activations[(row * ACTIVATION_SLOTS + TOP_K) * INTERMEDIATE
                        + k_lane]);
                for (uint result_index = 0; result_index < 4; ++result_index) {
                    uint output_column = output_base + result_index;
                    shared_result[result_index] += input_value * float(
                        shared_down_weight[output_column * INTERMEDIATE + k_lane]);
                }
            }
            for (uint result_index = 0; result_index < 4; ++result_index) {
                float shared_sum = simd_sum(shared_result[result_index]);
                if (lane == 0) {
                    bfloat shared_value = bfloat(shared_sum);
                    bfloat gate_value = shared_gate[row];
                    auto sigmoid_y = 1 / (
                        1 + metal::exp(metal::abs(gate_value)));
                    bfloat sigmoid_mlx_exact = gate_value < bfloat(0.0f)
                        ? bfloat(sigmoid_y)
                        : bfloat(1 - sigmoid_y);
                    bfloat gated_shared = bfloat(
                        sigmoid_mlx_exact * shared_value);
                    uint output_column = output_base + result_index;
                    uint output_index = row * HIDDEN + output_column;
                    output[output_index] = bfloat(
                        float(routed_accumulator[result_index])
                        + float(gated_shared));
                }
            }
        }
    """


def all_whole_moe_sources() -> dict[str, str]:
    """Return all fixed sources for construction tests and self-checks."""

    return {
        "target_m1_stage1": _fixed_source(stage=1, rows=1, variant="target_q8g64"),
        "target_m2_stage1": _fixed_source(stage=1, rows=2, variant="target_q8g64"),
        "mtp_m1_stage1": _fixed_source(stage=1, rows=1, variant="mtp_dense"),
        "target_m1_stage2": _fixed_source(stage=2, rows=1, variant="target_q4g64"),
        "target_m2_stage2": _fixed_source(stage=2, rows=2, variant="target_q4g64"),
        "mtp_m1_stage2": _fixed_source(stage=2, rows=1, variant="mtp_q4g32_dense"),
        "target_m1_stage3": _fixed_source(stage=3, rows=1, variant="target_q4g64"),
        "target_m2_stage3": _fixed_source(stage=3, rows=2, variant="target_q4g64"),
        "mtp_m1_stage3": _fixed_source(stage=3, rows=1, variant="mtp_q4g32_dense"),
    }


def whole_moe_launch_table() -> dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Describe the fixed grids without probing a runtime tensor."""

    return {
        "target_m1_stage1": ((256, 1, 1), (256, 1, 1)),
        "target_m2_stage1": ((512, 1, 1), (256, 1, 1)),
        "mtp_m1_stage1": ((256, 1, 1), (256, 1, 1)),
        "target_m1_stage2": ((STAGE2_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "target_m2_stage2": ((STAGE2_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "mtp_m1_stage2": ((STAGE2_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "target_m1_stage3": ((STAGE3_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "target_m2_stage3": ((STAGE3_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
        "mtp_m1_stage3": ((STAGE3_THREADGROUPS * 128, 1, 1), (128, 1, 1)),
    }


def _build_kernel(
    key: str,
    *,
    input_names: list[str],
    output_names: list[str],
):
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mtplx_a3b_whole_moe_{key}",
            input_names=input_names,
            output_names=output_names,
            source=all_whole_moe_sources()[key],
            ensure_row_contiguous=True,
        )
        _KERNELS[key] = kernel
    return kernel


_TARGET_STAGE1_INPUT_NAMES = [
    "value",
    "router_weight",
    "router_scales",
    "router_biases",
    "shared_gate_weight",
    "shared_gate_scales",
    "shared_gate_biases",
]
_MTP_STAGE1_INPUT_NAMES = ["value", "router_weight", "shared_gate_weight"]
_STAGE1_OUTPUT_NAMES = ["expert_ids", "route_scores", "shared_gate"]
_TARGET_STAGE2_INPUT_NAMES = [
    "value",
    "expert_ids",
    "routed_gate_up_weight",
    "routed_gate_up_scales",
    "routed_gate_up_biases",
    "shared_gate_up_weight",
    "shared_gate_up_scales",
    "shared_gate_up_biases",
]
_MTP_STAGE2_INPUT_NAMES = [
    "value",
    "expert_ids",
    "routed_gate_up_weight",
    "routed_gate_up_scales",
    "routed_gate_up_biases",
    "shared_gate_up_weight",
]
_STAGE2_OUTPUT_NAMES = ["activations"]
_TARGET_STAGE3_INPUT_NAMES = [
    "activations",
    "expert_ids",
    "route_scores",
    "shared_gate",
    "routed_down_weight",
    "routed_down_scales",
    "routed_down_biases",
    "shared_down_weight",
    "shared_down_scales",
    "shared_down_biases",
]
_MTP_STAGE3_INPUT_NAMES = [
    "activations",
    "expert_ids",
    "route_scores",
    "shared_gate",
    "routed_down_weight",
    "routed_down_scales",
    "routed_down_biases",
    "shared_down_weight",
]
_STAGE3_OUTPUT_NAMES = ["output"]


def _build_target_m1_stage1_kernel():
    return _build_kernel(
        "target_m1_stage1",
        input_names=_TARGET_STAGE1_INPUT_NAMES,
        output_names=_STAGE1_OUTPUT_NAMES,
    )


def _build_target_m2_stage1_kernel():
    return _build_kernel(
        "target_m2_stage1",
        input_names=_TARGET_STAGE1_INPUT_NAMES,
        output_names=_STAGE1_OUTPUT_NAMES,
    )


def _build_mtp_m1_stage1_kernel():
    return _build_kernel(
        "mtp_m1_stage1",
        input_names=_MTP_STAGE1_INPUT_NAMES,
        output_names=_STAGE1_OUTPUT_NAMES,
    )


def _build_target_m1_stage2_kernel():
    return _build_kernel(
        "target_m1_stage2",
        input_names=_TARGET_STAGE2_INPUT_NAMES,
        output_names=_STAGE2_OUTPUT_NAMES,
    )


def _build_target_m2_stage2_kernel():
    return _build_kernel(
        "target_m2_stage2",
        input_names=_TARGET_STAGE2_INPUT_NAMES,
        output_names=_STAGE2_OUTPUT_NAMES,
    )


def _build_mtp_m1_stage2_kernel():
    return _build_kernel(
        "mtp_m1_stage2",
        input_names=_MTP_STAGE2_INPUT_NAMES,
        output_names=_STAGE2_OUTPUT_NAMES,
    )


def _build_target_m1_stage3_kernel():
    return _build_kernel(
        "target_m1_stage3",
        input_names=_TARGET_STAGE3_INPUT_NAMES,
        output_names=_STAGE3_OUTPUT_NAMES,
    )


def _build_target_m2_stage3_kernel():
    return _build_kernel(
        "target_m2_stage3",
        input_names=_TARGET_STAGE3_INPUT_NAMES,
        output_names=_STAGE3_OUTPUT_NAMES,
    )


def _build_mtp_m1_stage3_kernel():
    return _build_kernel(
        "mtp_m1_stage3",
        input_names=_MTP_STAGE3_INPUT_NAMES,
        output_names=_STAGE3_OUTPUT_NAMES,
    )


def _launch_target_stage1(kernel: Any, value: Any, binding: Any, *, rows: int):
    router = binding.router
    shared_gate = binding.shared_scalar_gate
    return kernel(
        inputs=[
            value,
            router.weight,
            router.scales,
            router.biases,
            shared_gate.weight,
            shared_gate.scales,
            shared_gate.biases,
        ],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, 8), (rows, 8), (rows, 1)],
        output_dtypes=[mx.uint32, mx.bfloat16, mx.bfloat16],
    )


def _launch_mtp_stage1(kernel: Any, value: Any, binding: Any):
    return kernel(
        inputs=[
            value,
            binding.router.weight,
            binding.shared_scalar_gate.weight,
        ],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 8), (1, 8), (1, 1)],
        output_dtypes=[mx.uint32, mx.bfloat16, mx.bfloat16],
    )


def target_m1_stage1(value: Any, binding: Any):
    """Launch fixed target M1 route and shared-gate ownership."""

    return _launch_target_stage1(
        _build_target_m1_stage1_kernel(), value, binding, rows=1
    )


def target_m2_stage1(value: Any, binding: Any):
    """Launch fixed target M2 route and shared-gate ownership."""

    return _launch_target_stage1(
        _build_target_m2_stage1_kernel(), value, binding, rows=2
    )


def mtp_m1_stage1(value: Any, binding: Any):
    """Launch fixed MTP M1 route and shared-gate ownership."""

    return _launch_mtp_stage1(_build_mtp_m1_stage1_kernel(), value, binding)


def _target_stage2_inputs(value: Any, expert_ids: Any, binding: Any) -> list[Any]:
    routed_gate_up = binding.routed_gate_up
    shared_gate_up = binding.shared_gate_up
    return [
        value,
        expert_ids,
        routed_gate_up.weight,
        routed_gate_up.scales,
        routed_gate_up.biases,
        shared_gate_up.weight,
        shared_gate_up.scales,
        shared_gate_up.biases,
    ]


def _launch_target_stage2(
    kernel: Any,
    value: Any,
    expert_ids: Any,
    binding: Any,
    *,
    rows: int,
):
    (activations,) = kernel(
        inputs=_target_stage2_inputs(value, expert_ids, binding),
        grid=(STAGE2_THREADGROUPS * TILED_THREADS, 1, 1),
        threadgroup=(TILED_THREADS, 1, 1),
        output_shapes=[(rows, ACTIVATION_SLOTS, INTERMEDIATE)],
        output_dtypes=[mx.bfloat16],
    )
    return activations


def _launch_mtp_stage2(
    kernel: Any,
    value: Any,
    expert_ids: Any,
    binding: Any,
):
    routed_gate_up = binding.routed_gate_up
    (activations,) = kernel(
        inputs=[
            value,
            expert_ids,
            routed_gate_up.weight,
            routed_gate_up.scales,
            routed_gate_up.biases,
            binding.shared_gate_up.weight,
        ],
        grid=(STAGE2_THREADGROUPS * TILED_THREADS, 1, 1),
        threadgroup=(TILED_THREADS, 1, 1),
        output_shapes=[(1, ACTIVATION_SLOTS, INTERMEDIATE)],
        output_dtypes=[mx.bfloat16],
    )
    return activations


def target_m1_stage2(value: Any, expert_ids: Any, binding: Any):
    """Launch fixed BF16 target M1 `[1,9,512]` activation ownership."""

    return _launch_target_stage2(
        _build_target_m1_stage2_kernel(), value, expert_ids, binding, rows=1
    )


def target_m2_stage2(value: Any, expert_ids: Any, binding: Any):
    """Launch fixed BF16 target M2 `[2,9,512]` activation ownership."""

    return _launch_target_stage2(
        _build_target_m2_stage2_kernel(), value, expert_ids, binding, rows=2
    )


def mtp_m1_stage2(value: Any, expert_ids: Any, binding: Any):
    """Launch fixed BF16 MTP M1 `[1,9,512]` activation ownership."""

    return _launch_mtp_stage2(
        _build_mtp_m1_stage2_kernel(), value, expert_ids, binding
    )


def _target_stage3_inputs(
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
) -> list[Any]:
    routed_down = binding.routed_down
    shared_down = binding.shared_down
    return [
        activations,
        expert_ids,
        route_scores,
        shared_gate,
        routed_down.weight,
        routed_down.scales,
        routed_down.biases,
        shared_down.weight,
        shared_down.scales,
        shared_down.biases,
    ]


def _launch_target_stage3(
    kernel: Any,
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
    *,
    rows: int,
):
    (output,) = kernel(
        inputs=_target_stage3_inputs(
            activations, expert_ids, route_scores, shared_gate, binding
        ),
        grid=(STAGE3_THREADGROUPS * TILED_THREADS, 1, 1),
        threadgroup=(TILED_THREADS, 1, 1),
        output_shapes=[(rows, HIDDEN)],
        output_dtypes=[mx.bfloat16],
    )
    return output


def _launch_mtp_stage3(
    kernel: Any,
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
):
    routed_down = binding.routed_down
    (output,) = kernel(
        inputs=[
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            routed_down.weight,
            routed_down.scales,
            routed_down.biases,
            binding.shared_down.weight,
        ],
        grid=(STAGE3_THREADGROUPS * TILED_THREADS, 1, 1),
        threadgroup=(TILED_THREADS, 1, 1),
        output_shapes=[(1, HIDDEN)],
        output_dtypes=[mx.bfloat16],
    )
    return output


def target_m1_stage3(
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
):
    """Launch fixed target M1 output ownership."""

    return _launch_target_stage3(
        _build_target_m1_stage3_kernel(),
        activations,
        expert_ids,
        route_scores,
        shared_gate,
        binding,
        rows=1,
    )


def target_m2_stage3(
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
):
    """Launch fixed row-paired target M2 output ownership."""

    return _launch_target_stage3(
        _build_target_m2_stage3_kernel(),
        activations,
        expert_ids,
        route_scores,
        shared_gate,
        binding,
        rows=2,
    )


def mtp_m1_stage3(
    activations: Any,
    expert_ids: Any,
    route_scores: Any,
    shared_gate: Any,
    binding: Any,
):
    """Launch fixed MTP M1 output ownership."""

    return _launch_mtp_stage3(
        _build_mtp_m1_stage3_kernel(),
        activations,
        expert_ids,
        route_scores,
        shared_gate,
        binding,
    )


def bind_target_m1(binding: Any):
    """Bind the three fixed target M1 kernels once at installation."""

    stage1_kernel = _build_target_m1_stage1_kernel()
    stage2_kernel = _build_target_m1_stage2_kernel()
    stage3_kernel = _build_target_m1_stage3_kernel()

    def call(value: Any):
        expert_ids, route_scores, shared_gate = _launch_target_stage1(
            stage1_kernel, value, binding, rows=1
        )
        activations = _launch_target_stage2(
            stage2_kernel, value, expert_ids, binding, rows=1
        )
        output = _launch_target_stage3(
            stage3_kernel,
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            binding,
            rows=1,
        )
        return output.reshape(*value.shape)

    return call


def bind_target_m2(binding: Any):
    """Bind the three fixed row-paired target M2 kernels once at installation."""

    stage1_kernel = _build_target_m2_stage1_kernel()
    stage2_kernel = _build_target_m2_stage2_kernel()
    stage3_kernel = _build_target_m2_stage3_kernel()

    def call(value: Any):
        expert_ids, route_scores, shared_gate = _launch_target_stage1(
            stage1_kernel, value, binding, rows=2
        )
        activations = _launch_target_stage2(
            stage2_kernel, value, expert_ids, binding, rows=2
        )
        output = _launch_target_stage3(
            stage3_kernel,
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            binding,
            rows=2,
        )
        return output.reshape(*value.shape)

    return call


def bind_mtp_m1(binding: Any):
    """Bind the three fixed MTP M1 kernels once at installation."""

    stage1_kernel = _build_mtp_m1_stage1_kernel()
    stage2_kernel = _build_mtp_m1_stage2_kernel()
    stage3_kernel = _build_mtp_m1_stage3_kernel()

    def call(value: Any):
        expert_ids, route_scores, shared_gate = _launch_mtp_stage1(
            stage1_kernel, value, binding
        )
        activations = _launch_mtp_stage2(
            stage2_kernel, value, expert_ids, binding
        )
        output = _launch_mtp_stage3(
            stage3_kernel,
            activations,
            expert_ids,
            route_scores,
            shared_gate,
            binding,
        )
        return output.reshape(*value.shape)

    return call
