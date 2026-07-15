"""Tagged last-arrival primitives for the Hy3 one-dispatch router probe."""

from __future__ import annotations

from dataclasses import dataclass


_UINT32_MASK = 0xFFFFFFFF
_TAG_MULTIPLIER = 0x9E3779B9
_TAG_OFFSET = 0x51F15EED
_PAYLOAD_MULTIPLIER = 0x85EBCA6B
_LCG_MULTIPLIER = 1_664_525
_LCG_INCREMENT = 1_013_904_223


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
        if int(self.threadgroups) != 16:
            raise ValueError("tagged Hy3 arrival requires exactly 16 threadgroups")
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


def tagged_arrival_checksums(*, event: int, seed: int) -> tuple[int, int]:
    """Return the sum and rotated-XOR payload checksums for one election."""

    payloads = [
        tagged_arrival_payload(event=event, group=group, seed=seed)
        for group in range(16)
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
        uint local_group = (group_round + event_id) & (THREADGROUPS - 1);
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
