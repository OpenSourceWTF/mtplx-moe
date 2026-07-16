"""Tagged last-arrival primitives and the Hy3 fixed-M4 one-dispatch router."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import mlx.core as mx

from mtplx.hy3_router_fp32 import (
    Hy3RouterFP32Ineligible,
    _balanced_splitk_reduction_source,
    hy3_router_fp32_available,
)


_UINT32_MASK = 0xFFFFFFFF
_TAG_MULTIPLIER = 0x9E3779B9
_TAG_OFFSET = 0x51F15EED
_PAYLOAD_MULTIPLIER = 0x85EBCA6B
_LCG_MULTIPLIER = 1_664_525
_LCG_INCREMENT = 1_013_904_223
_SUPPORTED_THREADGROUPS = frozenset((16, 24, 32, 48))

_ROUTER_ROWS = 4
_ROUTER_PADDED_ROWS = 8
_ROUTER_EXPERTS = 192
_ROUTER_TOP_K = 8
_ROUTER_K_PARTS = 16
_ROUTER_SIMD_GROUPS = 4
_ROUTER_THREADGROUPS = 48
_ROUTER_THREADS = _ROUTER_SIMD_GROUPS * 32
_ROUTER_PARTIAL_WORDS = _ROUTER_K_PARTS * _ROUTER_PADDED_ROWS * _ROUTER_EXPERTS
_ROUTER_FLAG_WORDS = _ROUTER_THREADGROUPS * 2
_ROUTER_SCRATCH_WORDS = _ROUTER_PARTIAL_WORDS + _ROUTER_FLAG_WORDS

_ROUTER_EPOCH_LOCK = threading.Lock()
_ROUTER_EPOCH = 0


def _lcg_coefficients(limit: int = 287) -> tuple[tuple[int, ...], tuple[int, ...]]:
    multipliers = [1]
    increments = [0]
    for _ in range(limit):
        multipliers.append((multipliers[-1] * _LCG_MULTIPLIER) & _UINT32_MASK)
        increments.append(
            (increments[-1] * _LCG_MULTIPLIER + _LCG_INCREMENT) & _UINT32_MASK
        )
    return tuple(multipliers), tuple(increments)


_LCG_MULTIPLIERS, _LCG_INCREMENTS = _lcg_coefficients()


@dataclass(frozen=True, slots=True)
class TaggedArrivalLayout:
    """Scratch geometry for independent no-initialization elections."""

    threadgroups: int = 16
    elections: int = 1024

    def __post_init__(self) -> None:
        if int(self.threadgroups) not in _SUPPORTED_THREADGROUPS:
            raise ValueError(
                "tagged Hy3 arrival requires 16, 24, 32, or 48 threadgroups"
            )
        if int(self.elections) <= 0:
            raise ValueError("tagged arrival elections must be positive")

    @property
    def ready_words(self) -> int:
        return int(self.threadgroups)

    @property
    def check_words(self) -> int:
        return int(self.threadgroups)

    @property
    def flag_words(self) -> int:
        return self.ready_words + self.check_words

    @property
    def payload_words(self) -> int:
        return int(self.threadgroups)

    @property
    def metadata_words(self) -> int:
        return 3

    @property
    def words_per_election(self) -> int:
        return self.flag_words + self.payload_words + self.metadata_words

    @property
    def total_words(self) -> int:
        return int(self.elections) * self.words_per_election

    @property
    def total_bytes(self) -> int:
        return self.total_words * 4


def tagged_arrival_tag(event: int) -> int:
    """Return the nonrepeating 32-bit tag for one litmus event."""

    return ((int(event) & _UINT32_MASK) * _TAG_MULTIPLIER + _TAG_OFFSET) & _UINT32_MASK


def tagged_arrival_payload(*, event: int, group: int, seed: int) -> int:
    """Mirror the device payload and producer-delay calculation exactly."""

    event_u32 = int(event) & _UINT32_MASK
    group_u32 = int(group) & _UINT32_MASK
    seed_u32 = int(seed) & _UINT32_MASK
    state = (
        tagged_arrival_tag(event_u32)
        ^ seed_u32
        ^ (((group_u32 + 1) * _PAYLOAD_MULTIPLIER) & _UINT32_MASK)
    )
    delay_rounds = 32 + ((seed_u32 + event_u32 * 17 + group_u32 * 29) & 255)
    return (
        state * _LCG_MULTIPLIERS[delay_rounds] + _LCG_INCREMENTS[delay_rounds]
    ) & _UINT32_MASK


def tagged_arrival_checksums(
    *,
    event: int,
    seed: int,
    threadgroups: int = 16,
) -> tuple[int, int]:
    """Return the sum and rotated-XOR payload checksums for one election."""

    payloads = [
        tagged_arrival_payload(event=event, group=group, seed=seed)
        for group in range(int(threadgroups))
    ]
    payload_sum = sum(payloads) & _UINT32_MASK
    payload_xor = 0
    for group, payload in enumerate(payloads):
        shift = group & 31
        rotated = (
            payload
            if shift == 0
            else (((payload << shift) | (payload >> (32 - shift))) & _UINT32_MASK)
        )
        payload_xor ^= rotated
    return payload_sum, payload_xor & _UINT32_MASK


def tagged_arrival_litmus_source(layout: TaggedArrivalLayout) -> str:
    """Emit the device-scope tagged-election litmus kernel body."""

    return f"""
        using namespace metal;

        constexpr uint THREADGROUPS = {int(layout.threadgroups)};
        constexpr uint ELECTIONS = {int(layout.elections)};
        constexpr uint READY_WORDS = THREADGROUPS;
        constexpr uint CHECK_WORDS = THREADGROUPS;
        constexpr uint FLAG_WORDS = READY_WORDS + CHECK_WORDS;
        constexpr uint PAYLOAD_WORDS = THREADGROUPS;
        constexpr uint METADATA_WORDS = 3;
        constexpr uint WORDS_PER_ELECTION =
            FLAG_WORDS + PAYLOAD_WORDS + METADATA_WORDS;
        constexpr uint TAG_MULTIPLIER = {_TAG_MULTIPLIER}u;
        constexpr uint TAG_OFFSET = {_TAG_OFFSET}u;
        constexpr uint PAYLOAD_MULTIPLIER = {_PAYLOAD_MULTIPLIER}u;
        constexpr uint LCG_MULTIPLIER = {_LCG_MULTIPLIER}u;
        constexpr uint LCG_INCREMENT = {_LCG_INCREMENT}u;

        uint global_group = threadgroup_position_in_grid.x;
        uint group_round = global_group / ELECTIONS;
        uint election = global_group - group_round * ELECTIONS;
        uint event_id = base_event + election;
        uint local_group = (group_round + event_id) % THREADGROUPS;
        uint local_thread = thread_index_in_threadgroup;
        uint tag = event_id * TAG_MULTIPLIER + TAG_OFFSET;

        device uint* event_scratch =
            scratch + election * WORDS_PER_ELECTION;
        device atomic_uint* ready =
            reinterpret_cast<device atomic_uint*>(event_scratch);
        device atomic_uint* checks =
            reinterpret_cast<device atomic_uint*>(
                event_scratch + READY_WORDS);
        device atomic_uint* payloads =
            reinterpret_cast<device atomic_uint*>(event_scratch + FLAG_WORDS);
        device uint* metadata =
            event_scratch + FLAG_WORDS + PAYLOAD_WORDS;

        if (local_thread == 0) {{
            uint state = tag ^ seed
                ^ ((local_group + 1) * PAYLOAD_MULTIPLIER);
            uint delay_rounds = 32
                + ((seed + event_id * 17 + local_group * 29) & 255);
            for (uint delay = 0; delay < delay_rounds; ++delay) {{
                state = state * LCG_MULTIPLIER + LCG_INCREMENT;
            }}
            atomic_store_explicit(
                &payloads[local_group], state, memory_order_relaxed);
        }}
        threadgroup_barrier(mem_flags::mem_device);

        if (local_thread == 0) {{
            atomic_thread_fence(
                mem_flags::mem_device,
                memory_order_seq_cst,
                thread_scope_device);
            atomic_store_explicit(&ready[local_group], tag, memory_order_relaxed);
            atomic_store_explicit(&checks[local_group], ~tag, memory_order_relaxed);
            atomic_thread_fence(
                mem_flags::mem_device,
                memory_order_seq_cst,
                thread_scope_device);

            bool all_ready = true;
            for (uint producer = 0; producer < THREADGROUPS; ++producer) {{
                all_ready = all_ready
                    && atomic_load_explicit(&ready[producer], memory_order_relaxed)
                        == tag
                    && atomic_load_explicit(&checks[producer], memory_order_relaxed)
                        == ~tag;
            }}
            if (all_ready) {{
                atomic_thread_fence(
                    mem_flags::mem_device,
                    memory_order_seq_cst,
                    thread_scope_device);
                uint expected = tag;
                bool won = false;
                do {{
                    won = atomic_compare_exchange_weak_explicit(
                        &ready[0],
                        &expected,
                        ~tag,
                        memory_order_relaxed,
                        memory_order_relaxed);
                }} while (!won && expected == tag);

                if (won) {{
                    uint payload_sum = 0;
                    uint payload_xor = 0;
                    for (
                        uint producer = 0;
                        producer < THREADGROUPS;
                        ++producer
                    ) {{
                        uint payload = atomic_load_explicit(
                            &payloads[producer], memory_order_relaxed);
                        payload_sum += payload;
                        uint shift = producer & 31;
                        uint rotated = shift == 0
                            ? payload
                            : (payload << shift) | (payload >> (32 - shift));
                        payload_xor ^= rotated;
                    }}
                    metadata[0] = local_group;
                    metadata[1] = payload_sum;
                    metadata[2] = payload_xor;
                    atomic_thread_fence(
                        mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);
                }}
            }}
        }}
    """


def _next_router_epoch() -> int:
    """Return one process-lifetime-unique epoch for reusable output storage."""

    global _ROUTER_EPOCH
    with _ROUTER_EPOCH_LOCK:
        if _ROUTER_EPOCH >= _UINT32_MASK:
            raise RuntimeError("Hy3 last-arrival router epoch space is exhausted")
        _ROUTER_EPOCH += 1
        return _ROUTER_EPOCH


def _router_scaling_literal(scaling_factor: float) -> str:
    if not math.isfinite(float(scaling_factor)) or float(scaling_factor) <= 0.0:
        raise Hy3RouterFP32Ineligible(
            "Hy3 router scaling factor must be finite and positive"
        )
    literal = format(float(scaling_factor), ".9g")
    if "." not in literal and "e" not in literal.lower():
        literal += ".0"
    return literal


def _router_exp_call(sigmoid_mode: str, operand: str) -> str:
    if sigmoid_mode == "precise":
        return f"exp({operand})"
    raise Hy3RouterFP32Ineligible(
        "Hy3 last-arrival router sigmoid mode must be precise"
    )


def hy3_router_last_arrival_source(
    *,
    scaling_factor: float = 2.826,
    sigmoid_mode: Literal["precise"] = "precise",
) -> str:
    """Emit the fixed-M4 one-dispatch MPP R1, election, and precise R2 body."""

    scaling_literal = _router_scaling_literal(scaling_factor)
    exp_call = _router_exp_call(sigmoid_mode, "-total")
    reduction = _balanced_splitk_reduction_source(_ROUTER_K_PARTS)
    return f"""
        using namespace metal;
        using namespace mpp::tensor_ops;

        constexpr int ROWS = 4;
        constexpr int PADDED_ROWS = 8;
        constexpr int BN = 16;
        constexpr int K = 4096;
        constexpr int N = 192;
        constexpr int TOPK = 8;
        constexpr int P = 16;
        constexpr int NT = 12;
        constexpr int KS = 256;
        constexpr int SGPTG = 4;
        constexpr int GROUPS_PER_PART = 3;
        constexpr int THREADGROUPS = 48;
        constexpr int STRIDE = PADDED_ROWS * N;
        constexpr int PARTIAL_WORDS = P * STRIDE;
        constexpr int READY_OFFSET = PARTIAL_WORDS;
        constexpr int CHECK_OFFSET = READY_OFFSET + THREADGROUPS;
        constexpr uint TAG_MULTIPLIER = {_TAG_MULTIPLIER}u;
        constexpr uint TAG_OFFSET = {_TAG_OFFSET}u;
        constexpr int CANDIDATES_PER_LANE = 6;
        constexpr float ROUTING_SCALE = {scaling_literal}f;

        uint tg = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tid = thread_index_in_threadgroup;
        int part = int(tg) / GROUPS_PER_PART;
        int group_in_part = int(tg) - part * GROUPS_PER_PART;
        int n_tile_index = group_in_part * SGPTG + int(simd_gid);
        int n0 = n_tile_index * BN;
        int k0 = part * KS;

        device float* partials = scratch;
        device atomic_uint* ready =
            reinterpret_cast<device atomic_uint*>(scratch + READY_OFFSET);
        device atomic_uint* checks =
            reinterpret_cast<device atomic_uint*>(scratch + CHECK_OFFSET);

        threadgroup float A_tile[PADDED_ROWS * KS];
        threadgroup uint elected;
        if (tid == 0) {{
            elected = 0u;
        }}
        for (
            int offset = int(tid);
            offset < PADDED_ROWS * KS;
            offset += SGPTG * 32
        ) {{
            int row = offset / KS;
            int column = offset - row * KS;
            A_tile[offset] = row < ROWS
                ? x[row * K + k0 + column]
                : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        tensor<threadgroup float, dextents<int, 2>, tensor_inline> A(
            A_tile,
            dextents<int, 2>{{KS, PADDED_ROWS}},
            array<int, 2>{{1, KS}});
        tensor<device bfloat, dextents<int, 2>, tensor_inline> B(
            (device bfloat*)weight + k0 * N + n0,
            dextents<int, 2>{{BN, KS}},
            array<int, 2>{{1, N}});
        tensor<device float, dextents<int, 2>, tensor_inline> C(
            partials + part * PADDED_ROWS * N + n0,
            dextents<int, 2>{{BN, PADDED_ROWS}},
            array<int, 2>{{1, N}});

        constexpr auto desc = matmul2d_descriptor(
            PADDED_ROWS,
            BN,
            KS,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply);
        matmul2d<desc, metal::execution_simdgroup> op;
        op.run(A, B, C);
        threadgroup_barrier(mem_flags::mem_device);

        uint tag = epoch * TAG_MULTIPLIER + TAG_OFFSET;
        if (tid == 0) {{
            atomic_thread_fence(
                mem_flags::mem_device,
                memory_order_seq_cst,
                thread_scope_device);
            atomic_store_explicit(&ready[tg], tag, memory_order_relaxed);
            atomic_store_explicit(&checks[tg], ~tag, memory_order_relaxed);
            atomic_thread_fence(
                mem_flags::mem_device,
                memory_order_seq_cst,
                thread_scope_device);

            bool all_ready = true;
            for (uint producer = 0; producer < THREADGROUPS; ++producer) {{
                all_ready = all_ready
                    && atomic_load_explicit(
                        &ready[producer], memory_order_relaxed) == tag
                    && atomic_load_explicit(
                        &checks[producer], memory_order_relaxed) == ~tag;
            }}
            if (all_ready) {{
                atomic_thread_fence(
                    mem_flags::mem_device,
                    memory_order_seq_cst,
                    thread_scope_device);
                uint expected = tag;
                bool won = false;
                do {{
                    won = atomic_compare_exchange_weak_explicit(
                        &ready[0],
                        &expected,
                        ~tag,
                        memory_order_relaxed,
                        memory_order_relaxed);
                }} while (!won && expected == tag);
                if (won) {{
                    elected = 1u;
                    atomic_thread_fence(
                        mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (elected == 0u) {{
            return;
        }}
        atomic_thread_fence(
            mem_flags::mem_device,
            memory_order_seq_cst,
            thread_scope_device);

        uint row = simd_gid;
        float candidate_selection[CANDIDATES_PER_LANE];
        float candidate_unbiased[CANDIDATES_PER_LANE];
        int candidate_indices[CANDIDATES_PER_LANE];
        float merged_unbiased[TOPK];
        int merged_indices[TOPK];

        _Pragma("unroll")
        for (int slot = 0; slot < CANDIDATES_PER_LANE; ++slot) {{
            uint expert = lane + uint(slot) * 32;
            uint index = row * N + expert;
            {reduction}
            float score = 1.0f / (1.0f + {exp_call});
            candidate_selection[slot] = score + expert_bias[expert];
            candidate_unbiased[slot] = score;
            candidate_indices[slot] = int(expert);
        }}

        _Pragma("unroll")
        for (int rank = 0; rank < TOPK; ++rank) {{
            float lane_selection = -INFINITY;
            float lane_unbiased = 0.0f;
            int lane_index = -1;
            _Pragma("unroll")
            for (int slot = 0; slot < CANDIDATES_PER_LANE; ++slot) {{
                float selection = candidate_selection[slot];
                int index = candidate_indices[slot];
                bool higher = selection > lane_selection;
                bool later_equal = selection == lane_selection
                    && index > lane_index;
                if (higher || later_equal) {{
                    lane_selection = selection;
                    lane_unbiased = candidate_unbiased[slot];
                    lane_index = index;
                }}
            }}

            float winner_selection = simd_max(lane_selection);
            float winner_index_value = simd_max(
                lane_selection == winner_selection
                    ? float(lane_index)
                    : -1.0f);
            int winner_index = int(winner_index_value);
            float winner_unbiased = simd_sum(
                lane_index == winner_index ? lane_unbiased : 0.0f);
            if (lane == 0) {{
                merged_unbiased[rank] = winner_unbiased;
                merged_indices[rank] = winner_index;
            }}
            _Pragma("unroll")
            for (int slot = 0; slot < CANDIDATES_PER_LANE; ++slot) {{
                if (candidate_indices[slot] == winner_index) {{
                    candidate_selection[slot] = -INFINITY;
                }}
            }}
        }}

        if (lane == 0) {{
            float score_sum = 0.0f;
            for (int output = 0; output < TOPK; ++output) {{
                score_sum += merged_unbiased[TOPK - 1 - output];
            }}
            float scale = ROUTING_SCALE / (score_sum + 1e-20f);
            for (int output = 0; output < TOPK; ++output) {{
                int source = TOPK - 1 - output;
                uint output_index = row * TOPK + output;
                expert_ids[output_index] = merged_indices[source];
                router_scores[output_index] = merged_unbiased[source] * scale;
            }}
        }}
    """


@lru_cache(maxsize=8)
def _build_hy3_router_last_arrival_kernel(
    scaling_factor: float,
    sigmoid_mode: str,
):
    return mx.fast.metal_kernel(
        name=(
            "mtplx_hy3_router_last_arrival_m4_n16_p16_sg4_"
            f"{sigmoid_mode.replace('-', '_')}"
        ),
        input_names=["x", "weight", "expert_bias", "epoch"],
        output_names=["expert_ids", "router_scores", "scratch"],
        header="#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
        source=hy3_router_last_arrival_source(
            scaling_factor=scaling_factor,
            sigmoid_mode=sigmoid_mode,
        ),
        ensure_row_contiguous=True,
    )


def _validate_router_residents(weight: mx.array, expert_bias: mx.array) -> None:
    if (
        weight.ndim != 2
        or tuple(int(dimension) for dimension in weight.shape) != (4096, 192)
        or weight.dtype != mx.bfloat16
    ):
        raise Hy3RouterFP32Ineligible(
            "Hy3 last-arrival router requires one K-major BF16 [4096, 192] "
            "resident weight"
        )
    if (
        expert_bias.ndim != 1
        or tuple(int(dimension) for dimension in expert_bias.shape) != (192,)
        or expert_bias.dtype != mx.float32
    ):
        raise Hy3RouterFP32Ineligible(
            "Hy3 last-arrival router expert bias must be FP32 with shape (192,)"
        )


@dataclass(frozen=True, slots=True)
class Hy3RouterLastArrivalOutput:
    """Rows4 result matching the fixed-K3 router contract without importing it."""

    expert_ids: mx.array
    route_weights: mx.array
    dispatch_count: int = 1
    _scratch: mx.array | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if tuple(int(dimension) for dimension in self.expert_ids.shape) != (1, 4, 8):
            raise Hy3RouterFP32Ineligible(
                "Hy3 last-arrival expert IDs must have shape [1, 4, 8]"
            )
        if self.expert_ids.dtype != mx.int32:
            raise Hy3RouterFP32Ineligible(
                "Hy3 last-arrival expert IDs must have int32 dtype"
            )
        if tuple(int(dimension) for dimension in self.route_weights.shape) != (1, 4, 8):
            raise Hy3RouterFP32Ineligible(
                "Hy3 last-arrival route weights must have shape [1, 4, 8]"
            )
        if self.route_weights.dtype != mx.float32:
            raise Hy3RouterFP32Ineligible(
                "Hy3 last-arrival route weights must have FP32 dtype"
            )
        if int(self.dispatch_count) != 1:
            raise Hy3RouterFP32Ineligible(
                "Hy3 last-arrival Rows4 router requires exactly one dispatch"
            )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def rows(self) -> int:
        return _ROUTER_ROWS

    @property
    def top_k(self) -> int:
        return _ROUTER_TOP_K

    @property
    def assignment_count(self) -> int:
        return _ROUTER_ROWS * _ROUTER_TOP_K


def hy3_router_last_arrival_route(
    value: mx.array,
    weight: mx.array,
    expert_bias: mx.array,
    *,
    available: bool | None = None,
    top_k: int = 8,
    route_norm: bool = True,
    scaling_factor: float = 2.826,
    sigmoid_mode: Literal["precise"] = "precise",
) -> Hy3RouterLastArrivalOutput:
    """Route fixed `[1, 4, 4096]` hidden rows through one Metal dispatch."""

    supported = hy3_router_fp32_available() if available is None else bool(available)
    if (
        not supported
        or value.ndim != 3
        or tuple(int(dimension) for dimension in value.shape) != (1, 4, 4096)
        or value.dtype != mx.float32
    ):
        raise Hy3RouterFP32Ineligible(
            "Hy3 last-arrival router requires FP32 hidden rows shaped [1, 4, 4096] "
            "on the qualified Metal device"
        )
    _validate_router_residents(weight, expert_bias)
    if int(top_k) != 8:
        raise Hy3RouterFP32Ineligible("Hy3 last-arrival router requires top-8")
    if not route_norm:
        raise Hy3RouterFP32Ineligible(
            "Hy3 last-arrival router requires normalized route weights"
        )
    _router_scaling_literal(scaling_factor)
    _router_exp_call(sigmoid_mode, "-total")

    kernel = _build_hy3_router_last_arrival_kernel(
        float(scaling_factor),
        sigmoid_mode,
    )
    epoch = mx.array(_next_router_epoch(), dtype=mx.uint32)
    expert_ids, route_weights, scratch = kernel(
        inputs=[value.reshape(4, 4096), weight, expert_bias, epoch],
        grid=(_ROUTER_THREADGROUPS * _ROUTER_THREADS, 1, 1),
        threadgroup=(_ROUTER_THREADS, 1, 1),
        output_shapes=[(4, 8), (4, 8), (_ROUTER_SCRATCH_WORDS,)],
        output_dtypes=[mx.int32, mx.float32, mx.float32],
    )
    return Hy3RouterLastArrivalOutput(
        expert_ids=expert_ids.reshape(1, 4, 8),
        route_weights=route_weights.reshape(1, 4, 8),
        dispatch_count=1,
        _scratch=scratch,
    )


@dataclass(frozen=True, slots=True)
class Hy3RouterLastArrival:
    """Resident-parameter callable implementing the fixed-K3 Rows4 protocol."""

    weight: mx.array
    expert_bias: mx.array
    available: bool | None = None
    top_k: int = 8
    route_norm: bool = True
    scaling_factor: float = 2.826
    sigmoid_mode: Literal["precise"] = "precise"

    def __post_init__(self) -> None:
        _validate_router_residents(self.weight, self.expert_bias)
        if int(self.top_k) != 8:
            raise Hy3RouterFP32Ineligible("Hy3 last-arrival router requires top-8")
        if not self.route_norm:
            raise Hy3RouterFP32Ineligible(
                "Hy3 last-arrival router requires normalized route weights"
            )
        _router_scaling_literal(self.scaling_factor)
        _router_exp_call(self.sigmoid_mode, "-total")

    def __call__(self, hidden_rows: mx.array) -> Hy3RouterLastArrivalOutput:
        return hy3_router_last_arrival_route(
            hidden_rows,
            self.weight,
            self.expert_bias,
            available=self.available,
            top_k=self.top_k,
            route_norm=self.route_norm,
            scaling_factor=self.scaling_factor,
            sigmoid_mode=self.sigmoid_mode,
        )
