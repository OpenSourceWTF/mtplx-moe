from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.qwen4_ngram import (
    NGramCacheConfig,
    NGramManifest,
    NGramRuntimeBudget,
    NGramShard,
    plan_ngram_cache,
    plan_production_ngram_cache,
    qwen4_ngram_transient_bytes,
    qwen4_kv_mtp_reserve_bytes,
)


def _manifest(rows: int = 1_000_000) -> NGramManifest:
    data_bytes = rows * 100
    return NGramManifest(
        source_repo="repo",
        source_revision="revision",
        storage="affine-q4-g32",
        row_width=160,
        row_bytes=100,
        padded_rows=rows,
        shards=(
            NGramShard(
                name="ngram.bin",
                tensor="ngram",
                start_row=0,
                row_count=rows,
                data_offset=0,
                data_bytes=data_bytes,
                file_size=data_bytes,
                sha256="0" * 64,
            ),
        ),
    )


def test_capacity_accounts_slots_metadata_routes_alignment_and_transients() -> None:
    plan = plan_ngram_cache(
        _manifest(),
        NGramCacheConfig(
            cache_limit_bytes=1_050,
            transient_limit_bytes=350,
            max_inflight_io_bytes=300,
            max_open_files=2,
            bypass_page_cache=True,
            eviction="lru",
            allocation_alignment_bytes=64,
        ),
    )

    assert plan.slot_count == 10
    assert plan.payload_bytes == 1_000
    assert plan.slot_metadata_bytes == 10 * (8 + 4 + 4 + 1 + 4 + 8 + 4 + 4)
    assert plan.route_capacity == 32
    assert plan.route_table_bytes == 32 * (4 + 4)
    assert plan.transient_bytes == 300
    assert plan.transient_metadata_bytes == 3
    assert plan.total_reserved_bytes % 64 == 0
    assert plan.total_reserved_bytes == (
        plan.payload_bytes
        + plan.slot_metadata_bytes
        + plan.route_table_bytes
        + plan.transient_bytes
        + plan.transient_metadata_bytes
        + plan.alignment_bytes
    )


def test_checked_kv_capacity_arithmetic_rejects_signed_64_bit_overflow() -> None:
    with pytest.raises(OverflowError, match="signed 64-bit"):
        qwen4_kv_mtp_reserve_bytes((1 << 63) - 1)


def test_qwen4_transient_bytes_follow_configured_prefill_geometry() -> None:
    assert qwen4_ngram_transient_bytes(4_096) == 4_096 * 16 * 100


def test_qwen4_transient_bytes_reject_overflow() -> None:
    with pytest.raises(OverflowError, match="signed 64-bit"):
        qwen4_ngram_transient_bytes((1 << 63) - 1)


def test_transient_pool_tracks_capacity_without_a_full_slot_census() -> None:
    from mtplx.qwen4_ngram import _TransientPool

    class IndexedSlots:
        def __init__(self, values):
            self.values = list(values)

        def __getitem__(self, index):
            return self.values[index]

        def __setitem__(self, index, value):
            self.values[index] = value

        def __iter__(self):
            raise AssertionError("reservation must not census every transient slot")

    pool = _TransientPool(
        SimpleNamespace(transient_bytes=800, transient_metadata_bytes=8),
        row_bytes=100,
    )
    pool.used = IndexedSlots(pool.used)

    reservations = pool.reserve_fragmented((3, 2))
    assert pool.free_slots == 3
    for _group, _offset, start, count in reservations:
        pool.release(start, count)
    assert pool.free_slots == 8


def test_packed_lru_uses_free_chain_then_oldest_unprotected_slots() -> None:
    from mtplx.qwen4_ngram import _PackedCacheIndex

    plan = plan_ngram_cache(
        _manifest(rows=3),
        NGramCacheConfig(
            cache_limit_bytes=300,
            transient_limit_bytes=100,
            max_inflight_io_bytes=100,
            max_open_files=2,
            bypass_page_cache=True,
            eviction="lru",
        ),
    )
    packed = _PackedCacheIndex(plan)
    try:
        assert [packed.pop_free_slot() for _ in range(4)] == [0, 1, 2, None]
        for slot, row in enumerate((10, 11, 12)):
            packed.rows[slot] = row
            packed.touch_lru(slot)
        packed.touch_lru(0)
        packed.pins[2] = 1

        assert packed.oldest_unpinned_slots(2, protected={1}) == (0,)
    finally:
        packed.release()


def test_production_budget_rejects_runtime_without_minimum_cache() -> None:
    manifest = _manifest()
    budget = NGramRuntimeBudget(
        measured_base_residency_bytes=900_000,
        kv_mtp_reserve_bytes=50_000,
        metal_working_reserve_bytes=25_000,
        safety_margin_bytes=24_950,
        minimum_payload_bytes=100,
        allocation_alignment_bytes=64,
        target_residency_bytes=1_000_000,
        payload_ceiling_bytes=1_000,
    )

    with pytest.raises(ValueError, match="minimum viable"):
        plan_production_ngram_cache(
            manifest,
            budget,
            transient_limit_bytes=100,
            max_inflight_io_bytes=100,
            max_open_files=2,
            bypass_page_cache=True,
            eviction="lru",
        )


def test_production_budget_accepts_unbounded_user_ceiling_and_caps_to_fit() -> None:
    manifest = _manifest()
    budget = NGramRuntimeBudget(
        measured_base_residency_bytes=100_000,
        kv_mtp_reserve_bytes=50_000,
        metal_working_reserve_bytes=25_000,
        safety_margin_bytes=25_000,
        minimum_payload_bytes=manifest.row_bytes,
        allocation_alignment_bytes=64,
        target_residency_bytes=1_000_000,
        payload_ceiling_bytes=1024**4,
    )

    planned = plan_production_ngram_cache(
        manifest,
        budget,
        transient_limit_bytes=100,
        max_inflight_io_bytes=100,
        max_open_files=2,
        bypass_page_cache=True,
        eviction="lru",
    )

    assert planned.payload_formula_ceiling_bytes == 800_000
    assert planned.cache.payload_bytes <= 800_000
    assert planned.projected_residency_bytes <= budget.target_residency_bytes
