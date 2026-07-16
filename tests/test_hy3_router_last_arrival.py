from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")

from mtplx.hy3_router_last_arrival import (  # noqa: E402
    TaggedArrivalLayout,
    tagged_arrival_litmus_source,
    tagged_arrival_payload,
    tagged_arrival_tag,
)


def test_tagged_arrival_layout_is_uninitialized_and_independently_checkable() -> None:
    layout = TaggedArrivalLayout(threadgroups=16, elections=1024)

    assert layout.ready_words == 16
    assert layout.check_words == 16
    assert layout.flag_words == 32
    assert layout.payload_words == 16
    assert layout.metadata_words == 3
    assert layout.words_per_election == 51
    assert layout.total_words == 51 * 1024
    assert layout.total_bytes == 51 * 1024 * 4


@pytest.mark.parametrize("threadgroups", (16, 24, 32, 48))
def test_tagged_arrival_layout_supports_retained_router_group_counts(
    threadgroups: int,
) -> None:
    layout = TaggedArrivalLayout(threadgroups=threadgroups, elections=7)

    assert layout.ready_words == threadgroups
    assert layout.check_words == threadgroups
    assert layout.payload_words == threadgroups
    assert layout.words_per_election == 3 * threadgroups + 3


@pytest.mark.parametrize("threadgroups", (0, 1, 3, 12, 17, 40, 64))
def test_tagged_arrival_layout_rejects_unretained_router_group_counts(
    threadgroups: int,
) -> None:
    with pytest.raises(ValueError, match="16, 24, 32, or 48 threadgroups"):
        TaggedArrivalLayout(threadgroups=threadgroups, elections=1)


@pytest.mark.parametrize("elections", (0, -1))
def test_tagged_arrival_layout_requires_positive_elections(elections: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        TaggedArrivalLayout(threadgroups=16, elections=elections)


def test_tagged_arrival_tags_do_not_repeat_across_the_litmus_window() -> None:
    tags = {tagged_arrival_tag(event) for event in range(1_000_000)}

    assert len(tags) == 1_000_000
    assert all(tag != ((~tag) & 0xFFFFFFFF) for tag in tags)


def test_tagged_arrival_payload_changes_with_event_group_and_seed() -> None:
    baseline = tagged_arrival_payload(event=9, group=3, seed=51)

    assert baseline != tagged_arrival_payload(event=10, group=3, seed=51)
    assert baseline != tagged_arrival_payload(event=9, group=4, seed=51)
    assert baseline != tagged_arrival_payload(event=9, group=3, seed=52)


def test_tagged_arrival_source_has_no_initialized_counter_or_readiness_spin() -> None:
    source = tagged_arrival_litmus_source(
        TaggedArrivalLayout(threadgroups=16, elections=1024)
    )

    assert "atomic_thread_fence(" in source
    assert "memory_order_seq_cst" in source
    assert "thread_scope_device" in source
    assert "atomic_store_explicit(&ready[local_group]" in source
    assert "atomic_store_explicit(&checks[local_group]" in source
    assert "atomic_load_explicit(&ready[producer]" in source
    assert "atomic_load_explicit(&checks[producer]" in source
    assert "atomic_compare_exchange_weak_explicit(" in source
    assert "&ready[0]" in source
    assert "metadata[0] = local_group" in source
    assert "threadgroup_position_in_grid.x" in source
    assert "threadgroup_barrier(mem_flags::mem_device)" in source
    assert "atomic_fetch_add" not in source
    assert "init_value" not in source
    assert "while (atomic_load" not in source


def test_tagged_arrival_source_maps_one_group_of_each_election() -> None:
    source = tagged_arrival_litmus_source(
        TaggedArrivalLayout(threadgroups=16, elections=256)
    )

    assert "global_group / ELECTIONS" in source
    assert "global_group - group_round * ELECTIONS" in source
    assert "(group_round + event_id) % THREADGROUPS" in source
    assert "& (THREADGROUPS - 1)" not in source
    assert "event_id * TAG_MULTIPLIER + TAG_OFFSET" in source
