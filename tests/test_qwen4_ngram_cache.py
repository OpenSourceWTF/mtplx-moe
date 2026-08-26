from __future__ import annotations

import hashlib
import os
import random
import threading
from concurrent.futures import CancelledError
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from mtplx import qwen4_ngram
from mtplx.qwen4_ngram import (
    NGramAcquireFuture,
    NGramCacheClosed,
    NGramCacheConfig,
    NGramCacheError,
    NGramCacheFull,
    NGramCacheIOError,
    NGramFileIdentity,
    NGramLease,
    NGramManifest,
    NGramRowCache,
    NGramShard,
    SlotTicket,
    VerifiedNGramArtifact,
    VerifiedNGramShard,
    plan_ngram_cache,
)

ROW_BYTES = 8
ROW_COUNT = 12


def fixture_row(row: int) -> bytes:
    return bytes((row,)) * ROW_BYTES


class RecordingReader:
    def __init__(self) -> None:
        self.reads: list[tuple[int, int, int]] = []
        self.nocache_ranges: list[tuple[int, int, int]] = []

    def read_into(
        self,
        shard: VerifiedNGramShard,
        offset: int,
        target: memoryview,
    ) -> int:
        length = target.nbytes
        item = (shard.fileno(), offset, length)
        self.reads.append(item)
        return os.preadv(shard.fileno(), [target], offset)


class ShortReader(RecordingReader):
    def read_into(self, *args: object, **kwargs: object) -> int:
        return super().read_into(*args, **kwargs) - 1  # type: ignore[arg-type]


class BrokenReader(RecordingReader):
    def read_into(self, *args: object, **kwargs: object) -> int:
        del args, kwargs
        raise OSError("synthetic read failure")


class ControlledReader(RecordingReader):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def read_into(self, *args: object, **kwargs: object) -> int:
        self.started.set()
        assert self.release.wait(timeout=5)
        return super().read_into(*args, **kwargs)  # type: ignore[arg-type]


class ControlledFatalReader(ControlledReader):
    def read_into(self, *args: object, **kwargs: object) -> int:
        self.started.set()
        assert self.release.wait(timeout=5)
        raise KeyboardInterrupt("synthetic fatal reader")


@dataclass
class CacheFixture:
    cache: NGramRowCache
    artifact: VerifiedNGramArtifact
    reader: RecordingReader

    def close(self) -> None:
        self.cache.close()
        self.artifact.close()


def fixture_cache(
    tmp_path: Path,
    *,
    slots: int = 4,
    transient_rows: int = 12,
    inflight_rows: int = 12,
    max_open_files: int = 3,
    eviction: str = "lru",
    bypass_page_cache: bool = False,
    reader: RecordingReader | None = None,
) -> CacheFixture:
    tmp_path.mkdir(parents=True, exist_ok=True)
    split = 6
    shards: list[NGramShard] = []
    verified: list[VerifiedNGramShard] = []
    for index, (start, count) in enumerate(((0, split), (split, ROW_COUNT - split))):
        payload = b"".join(fixture_row(row) for row in range(start, start + count))
        path = tmp_path / f"shard-{index}.bin"
        path.write_bytes(payload)
        descriptor = os.open(path, os.O_RDONLY)
        metadata = os.fstat(descriptor)
        shard = NGramShard(
            name=path.name,
            tensor=f"ngram.weight.{index}",
            start_row=start,
            row_count=count,
            data_offset=0,
            data_bytes=len(payload),
            file_size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        shards.append(shard)
        verified.append(
            VerifiedNGramShard(
                shard,
                descriptor,
                NGramFileIdentity(
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                    size=metadata.st_size,
                    mtime_ns=metadata.st_mtime_ns,
                    ctime_ns=metadata.st_ctime_ns,
                ),
            )
        )
    manifest = NGramManifest(
        source_repo="fixture/repo",
        source_revision="fixture-revision",
        storage="bf16",
        row_width=ROW_BYTES // 2,
        row_bytes=ROW_BYTES,
        padded_rows=ROW_COUNT,
        shards=tuple(shards),
    ).with_digest()
    artifact = VerifiedNGramArtifact(
        manifest, os.open(tmp_path, os.O_RDONLY), tuple(verified)
    )
    selected_reader = reader or RecordingReader()
    cache = NGramRowCache(
        artifact,
        NGramCacheConfig(
            cache_limit_bytes=slots * ROW_BYTES,
            transient_limit_bytes=transient_rows * ROW_BYTES,
            max_inflight_io_bytes=inflight_rows * ROW_BYTES,
            max_open_files=max_open_files,
            bypass_page_cache=bypass_page_cache,
            eviction=eviction,  # type: ignore[arg-type]
        ),
        reader=selected_reader,
        allocator=bytearray,
    )
    return CacheFixture(cache, artifact, selected_reader)


def test_cache_deduplicates_pins_and_never_grows(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path, slots=4)
    try:
        lease = fixture.cache.acquire_rows((7, 7, 8, 9))
        assert lease.slot_ids[0] == lease.slot_ids[1]
        assert lease.row_bytes(0) == fixture_row(7)
        assert fixture.cache.arena_bytes == 4 * ROW_BYTES
        with pytest.raises(NGramCacheFull):
            fixture.cache.acquire_rows((10, 11))
        lease.release()
        replacement = fixture.cache.acquire_rows((10,))
        assert replacement.row_bytes(0) == fixture_row(10)
        replacement.release()
        assert fixture.cache.arena_bytes == 4 * ROW_BYTES
    finally:
        fixture.close()


def test_adjacent_rows_coalesce_only_within_one_shard(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path, slots=5)
    try:
        lease = fixture.cache.acquire_rows((4, 5, 6, 7, 9))
        lease.release()
        first_fd = fixture.artifact.shards[0].fileno()
        second_fd = fixture.artifact.shards[1].fileno()
        assert sorted(fixture.reader.reads) == sorted(
            [
                (first_fd, 4 * ROW_BYTES, 2 * ROW_BYTES),
                (second_fd, 0, 2 * ROW_BYTES),
                (second_fd, 3 * ROW_BYTES, ROW_BYTES),
            ]
        )
        assert fixture.reader.nocache_ranges == []
    finally:
        fixture.close()


def test_default_reader_applies_selected_f_nocache_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_cache(tmp_path, slots=1)
    fixture.cache.close()
    calls: list[tuple[int, int, int]] = []
    assert qwen4_ngram.fcntl is not None
    monkeypatch.setattr(qwen4_ngram.fcntl, "F_NOCACHE", 48, raising=False)
    monkeypatch.setattr(
        qwen4_ngram.fcntl,
        "fcntl",
        lambda descriptor, command, value: calls.append(
            (descriptor, command, value)
        ),
    )
    cache = NGramRowCache(
        fixture.artifact,
        NGramCacheConfig(
            cache_limit_bytes=ROW_BYTES,
            transient_limit_bytes=ROW_BYTES,
            max_inflight_io_bytes=ROW_BYTES,
            max_open_files=3,
            bypass_page_cache=True,
            eviction="lru",
        ),
        allocator=bytearray,
    )
    try:
        expected = sorted(
            (shard.fileno(), 48, 1) for shard in fixture.artifact.shards
        )
        assert sorted(calls) == expected
        lease = cache.acquire_rows((1,))
        assert lease.row_bytes(0) == fixture_row(1)
        lease.release()
        cache.acquire_rows((7,)).release()
        assert sorted(calls) == expected
    finally:
        cache.close()
        fixture.artifact.close()


def test_f_nocache_failure_rejects_construction_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_cache(tmp_path)
    fixture.cache.close()
    allocations: list[int] = []
    assert qwen4_ngram.fcntl is not None
    monkeypatch.setattr(qwen4_ngram.fcntl, "F_NOCACHE", 48, raising=False)

    def fail_f_nocache(descriptor: int, command: int, value: int) -> None:
        del descriptor, command, value
        raise OSError("synthetic F_NOCACHE failure")

    monkeypatch.setattr(qwen4_ngram.fcntl, "fcntl", fail_f_nocache)
    try:
        with pytest.raises(NGramCacheIOError, match="F_NOCACHE"):
            NGramRowCache(
                fixture.artifact,
                NGramCacheConfig(
                    cache_limit_bytes=ROW_BYTES,
                    transient_limit_bytes=ROW_BYTES,
                    max_inflight_io_bytes=ROW_BYTES,
                    max_open_files=3,
                    bypass_page_cache=True,
                    eviction="lru",
                ),
                allocator=lambda size: (allocations.append(size), bytearray(size))[1],
            )
        assert allocations == []
    finally:
        fixture.artifact.close()


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "extra",
        "reordered",
        "name",
        "range",
        "offset",
        "bytes",
        "identity",
        "closed",
    ],
)
def test_malformed_retained_routes_fail_before_arena_or_reader_work(
    tmp_path: Path, corruption: str
) -> None:
    fixture = fixture_cache(tmp_path)
    fixture.cache.close()
    original_shards = fixture.artifact.shards
    first = original_shards[0]
    original_metadata = first.shard
    original_identity = first.identity
    if corruption == "missing":
        fixture.artifact.shards = original_shards[:1]
    elif corruption == "extra":
        fixture.artifact.shards = original_shards + (original_shards[0],)
    elif corruption == "reordered":
        fixture.artifact.shards = tuple(reversed(original_shards))
    elif corruption == "name":
        first.shard = replace(first.shard, name="wrong.bin")
    elif corruption == "range":
        first.shard = replace(first.shard, row_count=5, data_bytes=5 * ROW_BYTES)
    elif corruption == "offset":
        first.shard = replace(first.shard, data_offset=1, file_size=6 * ROW_BYTES + 1)
    elif corruption == "bytes":
        first.shard = replace(first.shard, data_bytes=5 * ROW_BYTES)
    elif corruption == "identity":
        first.identity = replace(first.identity, inode=first.identity.inode + 1)
    else:
        first.close()

    allocations: list[int] = []
    reader = RecordingReader()
    try:
        with pytest.raises((NGramCacheClosed, NGramCacheIOError, ValueError)):
            NGramRowCache(
                fixture.artifact,
                NGramCacheConfig(
                    cache_limit_bytes=ROW_BYTES,
                    transient_limit_bytes=ROW_BYTES,
                    max_inflight_io_bytes=ROW_BYTES,
                    max_open_files=4,
                    bypass_page_cache=False,
                    eviction="lru",
                ),
                reader=reader,
                allocator=lambda size: (allocations.append(size), bytearray(size))[1],
            )
        assert allocations == []
        assert reader.reads == []
    finally:
        fixture.artifact.shards = original_shards
        first.shard = original_metadata
        first.identity = original_identity
        fixture.artifact.close()


def test_hits_do_not_read_and_lru_evicts_oldest_unpinned(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path, slots=2)
    try:
        fixture.cache.acquire_rows((1,)).release()
        fixture.cache.acquire_rows((2,)).release()
        reads = len(fixture.reader.reads)
        fixture.cache.acquire_rows((1,)).release()
        assert len(fixture.reader.reads) == reads
        fixture.cache.acquire_rows((3,)).release()
        fixture.cache.acquire_rows((2,)).release()
        assert len(fixture.reader.reads) == reads + 2
    finally:
        fixture.close()


def test_frequency_policy_is_construction_selected(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path, slots=2, eviction="frequency")
    try:
        fixture.cache.acquire_rows((1,)).release()
        fixture.cache.acquire_rows((2,)).release()
        fixture.cache.acquire_rows((1,)).release()
        fixture.cache.acquire_rows((3,)).release()
        reads = len(fixture.reader.reads)
        fixture.cache.acquire_rows((1,)).release()
        assert len(fixture.reader.reads) == reads
        fixture.cache.acquire_rows((2,)).release()
        assert len(fixture.reader.reads) == reads + 1
    finally:
        fixture.close()


def test_stale_completion_cannot_publish_reassigned_slot(tmp_path: Path) -> None:
    reader = ControlledReader()
    fixture = fixture_cache(tmp_path, slots=1, reader=reader)
    try:
        first = fixture.cache.acquire_rows_async((1,))
        assert reader.started.wait(timeout=5)
        assert first.cancel()
        second = fixture.cache.acquire_rows_async((2,))
        reader.release.set()
        with pytest.raises(CancelledError):
            first.result(timeout=5)
        lease = second.result(timeout=5)
        assert lease.row_bytes(0) == fixture_row(2)
        lease.release()
    finally:
        reader.release.set()
        fixture.close()


def test_pending_duplicate_shares_slot_and_survives_one_waiter_cancel(
    tmp_path: Path,
) -> None:
    reader = ControlledReader()
    fixture = fixture_cache(tmp_path, slots=1, reader=reader)
    try:
        first = fixture.cache.acquire_rows_async((1, 1))
        assert reader.started.wait(timeout=5)
        second = fixture.cache.acquire_rows_async((1,))
        assert first.cancel()
        reader.release.set()
        lease = second.result(timeout=5)
        assert lease.row_bytes(0) == fixture_row(1)
        lease.release()
        assert len(reader.reads) == 1
    finally:
        reader.release.set()
        fixture.close()


@pytest.mark.parametrize(
    ("transient_rows", "inflight_rows"),
    [(1, 12), (12, 1)],
)
def test_global_io_reservation_blocks_a_concurrent_miss(
    tmp_path: Path, transient_rows: int, inflight_rows: int
) -> None:
    reader = ControlledReader()
    fixture = fixture_cache(
        tmp_path,
        slots=2,
        transient_rows=transient_rows,
        inflight_rows=inflight_rows,
        reader=reader,
    )
    try:
        first = fixture.cache.acquire_rows_async((1,))
        assert reader.started.wait(timeout=5)
        with pytest.raises(NGramCacheFull):
            fixture.cache.acquire_rows_async((2,))
        reader.release.set()
        first.result(timeout=5).release()
        assert fixture.cache.inflight_bytes == 0
        assert fixture.cache.transient_bytes == 0
    finally:
        reader.release.set()
        fixture.close()


@pytest.mark.parametrize("reader_type", [ShortReader, BrokenReader])
def test_read_failure_fails_closed(
    tmp_path: Path, reader_type: type[RecordingReader]
) -> None:
    reader = reader_type()
    fixture = fixture_cache(tmp_path, slots=1, reader=reader)
    try:
        with pytest.raises(NGramCacheIOError):
            fixture.cache.acquire_rows((1,))
        fixture.cache.close()
        fixture.artifact.close()
    finally:
        if not fixture.cache.closed:
            fixture.close()


@pytest.mark.parametrize(
    ("transient_rows", "inflight_rows"),
    [(1, 12), (12, 1)],
)
def test_complete_miss_set_budgets_are_rejected_before_submission(
    tmp_path: Path, transient_rows: int, inflight_rows: int
) -> None:
    fixture = fixture_cache(
        tmp_path,
        slots=4,
        transient_rows=transient_rows,
        inflight_rows=inflight_rows,
    )
    try:
        with pytest.raises(NGramCacheFull):
            fixture.cache.acquire_rows_async((1, 2))
        assert fixture.reader.reads == []
        assert fixture.cache.inflight_bytes == 0
        lease = fixture.cache.acquire_rows((3,))
        lease.release()
    finally:
        fixture.close()


def test_pinned_victim_refusal_is_atomic(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path, slots=2)
    try:
        pinned = fixture.cache.acquire_rows((1, 2))
        with pytest.raises(NGramCacheFull):
            fixture.cache.acquire_rows((3,))
        assert pinned.row_bytes(0) == fixture_row(1)
        assert pinned.row_bytes(1) == fixture_row(2)
        pinned.release()
    finally:
        fixture.close()


def test_release_is_idempotent_and_released_lease_is_unreadable(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path, slots=1)
    try:
        lease = fixture.cache.acquire_rows((1, 1))
        lease.release()
        lease.release()
        with pytest.raises(NGramCacheClosed):
            lease.row_bytes(0)
        fixture.cache.acquire_rows((2,)).release()
    finally:
        fixture.close()


def test_reset_invalidates_leases_and_drains_worker_before_reuse(tmp_path: Path) -> None:
    reader = ControlledReader()
    fixture = fixture_cache(tmp_path, slots=1, reader=reader)
    reset_done = threading.Event()
    try:
        pending = fixture.cache.acquire_rows_async((1,))
        assert reader.started.wait(timeout=5)
        reset_thread = threading.Thread(
            target=lambda: (fixture.cache.reset(), reset_done.set())
        )
        reset_thread.start()
        assert not reset_done.wait(timeout=0.05)
        reader.release.set()
        reset_thread.join(timeout=5)
        assert reset_done.is_set()
        with pytest.raises(CancelledError):
            pending.result(timeout=5)
        fixture.cache.acquire_rows((2,)).release()
    finally:
        reader.release.set()
        fixture.close()


def test_reset_invalidates_existing_lease_without_closing_artifact(
    tmp_path: Path,
) -> None:
    fixture = fixture_cache(tmp_path, slots=1)
    try:
        lease = fixture.cache.acquire_rows((1,))
        fixture.cache.reset()
        with pytest.raises(NGramCacheClosed):
            lease.row_bytes(0)
        assert not fixture.artifact.closed
        replacement = fixture.cache.acquire_rows((2,))
        assert replacement.row_bytes(0) == fixture_row(2)
        replacement.release()
    finally:
        fixture.close()


def test_close_drains_workers_and_prevents_new_acquires(tmp_path: Path) -> None:
    reader = ControlledReader()
    fixture = fixture_cache(tmp_path, slots=1, reader=reader)
    closed = threading.Event()
    try:
        pending = fixture.cache.acquire_rows_async((1,))
        assert reader.started.wait(timeout=5)
        closer = threading.Thread(target=lambda: (fixture.cache.close(), closed.set()))
        closer.start()
        assert not closed.wait(timeout=0.05)
        reader.release.set()
        closer.join(timeout=5)
        assert closed.is_set()
        with pytest.raises(CancelledError):
            pending.result(timeout=5)
        with pytest.raises(NGramCacheClosed):
            fixture.cache.acquire_rows((2,))
    finally:
        reader.release.set()
        fixture.cache.close()
        fixture.artifact.close()


def test_cache_uses_retained_descriptor_after_path_replacement(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path, slots=1)
    try:
        path = tmp_path / fixture.artifact.shards[0].shard.name
        path.unlink()
        path.write_bytes(b"x" * (6 * ROW_BYTES))
        lease = fixture.cache.acquire_rows((1,))
        assert lease.row_bytes(0) == fixture_row(1)
        lease.release()
    finally:
        fixture.close()


def test_open_descriptor_bound_is_enforced_at_construction(tmp_path: Path) -> None:
    artifact_fixture = fixture_cache(tmp_path, max_open_files=3)
    artifact_fixture.cache.close()
    try:
        with pytest.raises(ValueError, match="max_open_files"):
            NGramRowCache(
                artifact_fixture.artifact,
                NGramCacheConfig(
                    cache_limit_bytes=ROW_BYTES,
                    transient_limit_bytes=ROW_BYTES,
                    max_inflight_io_bytes=ROW_BYTES,
                    max_open_files=2,
                    bypass_page_cache=False,
                    eviction="lru",
                ),
                reader=RecordingReader(),
                allocator=bytearray,
            )
    finally:
        artifact_fixture.artifact.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"cache_limit_bytes": 0},
        {"transient_limit_bytes": 0},
        {"max_inflight_io_bytes": 0},
        {"max_open_files": 0},
        {"eviction": "random"},
        {"bypass_page_cache": 1},
        {"cache_limit_bytes": True},
    ],
)
def test_invalid_config_fails_at_construction(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    fixture = fixture_cache(tmp_path)
    fixture.cache.close()
    values: dict[str, object] = {
        "cache_limit_bytes": ROW_BYTES,
        "transient_limit_bytes": ROW_BYTES,
        "max_inflight_io_bytes": ROW_BYTES,
        "max_open_files": 3,
        "bypass_page_cache": False,
        "eviction": "lru",
    }
    values.update(overrides)
    try:
        with pytest.raises((TypeError, ValueError)):
            NGramRowCache(
                fixture.artifact,
                NGramCacheConfig(**values),  # type: ignore[arg-type]
                reader=RecordingReader(),
                allocator=bytearray,
            )
    finally:
        fixture.artifact.close()


def test_only_exact_verified_artifact_is_accepted() -> None:
    config = NGramCacheConfig(
        cache_limit_bytes=ROW_BYTES,
        transient_limit_bytes=ROW_BYTES,
        max_inflight_io_bytes=ROW_BYTES,
        max_open_files=1,
        bypass_page_cache=False,
        eviction="lru",
    )
    with pytest.raises(TypeError, match="VerifiedNGramArtifact"):
        NGramRowCache(object(), config, reader=RecordingReader(), allocator=bytearray)


def planning_manifest(storage: str) -> NGramManifest:
    row_width = 160
    row_bytes = 320 if storage == "bf16" else 100
    return NGramManifest(
        source_repo="fixture/repo",
        source_revision="fixture-revision",
        storage=storage,  # type: ignore[arg-type]
        row_width=row_width,
        row_bytes=row_bytes,
        padded_rows=1,
        shards=(
            NGramShard(
                name="plan.bin",
                tensor="plan.weight",
                start_row=0,
                row_count=1,
                data_offset=0,
                data_bytes=row_bytes,
                file_size=row_bytes,
                sha256="0" * 64,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("storage", "expected"),
    [
        (
            "bf16",
            (6_710_886, 2_147_483_520, 194_615_694, 16_777_216, 134_217_728, 2_477_368_538),
        ),
        (
            "affine-q4-g32",
            (21_474_836, 2_147_483_600, 622_770_244, 67_108_864, 536_870_912, 3_308_183_741),
        ),
    ],
)
def test_two_gib_plans_use_complete_rows_without_large_allocations(
    storage: str, expected: tuple[int, ...]
) -> None:
    config = NGramCacheConfig(
        cache_limit_bytes=2 * 1024**3,
        transient_limit_bytes=1024**2,
        max_inflight_io_bytes=1024**2,
        max_open_files=2,
        bypass_page_cache=False,
        eviction="lru",
    )
    plan = plan_ngram_cache(planning_manifest(storage), config)
    assert (
        plan.slot_count,
        plan.payload_bytes,
        plan.slot_metadata_bytes,
        plan.route_capacity,
        plan.route_table_bytes,
        plan.total_reserved_bytes,
    ) == expected
    assert plan.requested_payload_bytes == 2 * 1024**3


def test_planner_rejects_packed_index_overflow() -> None:
    manifest = planning_manifest("bf16")
    config = NGramCacheConfig(
        cache_limit_bytes=((1 << 32) - 1) * manifest.row_bytes,
        transient_limit_bytes=manifest.row_bytes,
        max_inflight_io_bytes=manifest.row_bytes,
        max_open_files=2,
        bypass_page_cache=False,
        eviction="lru",
    )
    with pytest.raises(ValueError, match="slot-index domain"):
        plan_ngram_cache(manifest, config)


def test_non_multiple_payload_budget_allocates_only_complete_rows(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path, slots=1)
    fixture.cache.close()
    sizes: list[int] = []
    cache = NGramRowCache(
        fixture.artifact,
        NGramCacheConfig(
            cache_limit_bytes=ROW_BYTES + 1,
            transient_limit_bytes=ROW_BYTES,
            max_inflight_io_bytes=ROW_BYTES,
            max_open_files=3,
            bypass_page_cache=False,
            eviction="lru",
        ),
        reader=RecordingReader(),
        allocator=lambda size: (sizes.append(size), bytearray(size))[1],
    )
    try:
        assert sizes == [ROW_BYTES]
        assert cache.arena_bytes == ROW_BYTES
    finally:
        cache.close()
        fixture.artifact.close()


@pytest.mark.parametrize("eviction", ["lru", "frequency"])
def test_full_cache_mixed_hit_miss_never_evicts_requested_hit(
    tmp_path: Path, eviction: str
) -> None:
    fixture = fixture_cache(tmp_path, slots=2, eviction=eviction)
    try:
        fixture.cache.acquire_rows((1, 2)).release()
        lease = fixture.cache.acquire_rows((1, 3))
        assert lease.row_bytes(0) == fixture_row(1)
        assert lease.row_bytes(1) == fixture_row(3)
        lease.release()
    finally:
        fixture.close()


def test_copy_forgery_and_cross_cache_tickets_cannot_mutate_pins(
    tmp_path: Path,
) -> None:
    left = fixture_cache(tmp_path / "left", slots=1)
    right = fixture_cache(tmp_path / "right", slots=1)
    try:
        future = left.cache.acquire_rows_async((1,))
        lease = future.result(timeout=5)
        with pytest.raises(TypeError):
            copy(left.cache)
        with pytest.raises(TypeError):
            deepcopy(left.cache)
        with pytest.raises(TypeError):
            copy(future)
        with pytest.raises(TypeError):
            deepcopy(future)
        with pytest.raises(TypeError):
            copy(lease)
        with pytest.raises(TypeError):
            deepcopy(lease)
        with pytest.raises(TypeError, match="private"):
            NGramLease(left.cache, (), object(), object())
        with pytest.raises(TypeError, match="private"):
            NGramAcquireFuture(left.cache, (), (), object(), object())
        right_lease = right.cache.acquire_rows((2,))
        cross_ticket = right_lease._tickets[0]
        left.cache._release_ticket_pin(cross_ticket, cancel_loading=False)
        with pytest.raises(NGramCacheFull):
            left.cache.acquire_rows((3,))
        forged = SlotTicket(lease.slot_ids[0], lease._tickets[0].generation)
        left.cache._release_ticket_pin(forged, cancel_loading=False)
        with pytest.raises(NGramCacheFull):
            left.cache.acquire_rows((3,))
        right_lease.release()
        lease.release()
    finally:
        left.close()
        right.close()


def test_worker_failure_poisons_lane_and_drains_storage(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path, slots=6, reader=BrokenReader())
    try:
        with pytest.raises(NGramCacheIOError, match="synthetic read failure") as caught:
            fixture.cache.acquire_rows((0, 2, 4, 6, 8, 10))
        assert caught.value.__context__ is None
        assert fixture.cache.inflight_bytes == 0
        message = "synthetic read failure"
        for row in (2, 3):
            with pytest.raises(NGramCacheIOError, match=message):
                fixture.cache.acquire_rows((row,))
        assert fixture.cache.transient_storage_bytes == fixture.cache.plan.transient_bytes
    finally:
        fixture.close()


def test_failed_future_retains_only_a_released_transient_view(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path, reader=BrokenReader())
    acquisition = fixture.cache.acquire_rows_async((1,))
    try:
        with pytest.raises(NGramCacheIOError):
            acquisition.result(timeout=5)
        worker_error = acquisition._futures[0].exception(timeout=5)
        assert worker_error is not None
        traceback = worker_error.__traceback__
        retained_target: memoryview | None = None
        while traceback is not None:
            if traceback.tb_frame.f_code.co_name == "_read_publish":
                retained_target = traceback.tb_frame.f_locals.get("target")
            traceback = traceback.tb_next
        assert retained_target is not None
        with pytest.raises(ValueError, match="released memoryview"):
            _ = retained_target.nbytes
        fixture.cache.close()
        assert fixture.cache.transient_storage_bytes == 0
    finally:
        fixture.cache.close()
        fixture.artifact.close()


def test_default_allocator_is_imported_lazily_without_mlx_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_cache(tmp_path)
    fixture.cache.close()
    calls: list[int] = []
    from mtplx import mmap_mlx

    monkeypatch.setattr(
        mmap_mlx,
        "allocate_metal_u8",
        lambda size: (calls.append(size), bytearray(size))[1],
    )
    cache = NGramRowCache(
        fixture.artifact,
        NGramCacheConfig(
            cache_limit_bytes=ROW_BYTES,
            transient_limit_bytes=ROW_BYTES,
            max_inflight_io_bytes=ROW_BYTES,
            max_open_files=3,
            bypass_page_cache=False,
            eviction="lru",
        ),
        reader=RecordingReader(),
    )
    try:
        assert calls == [ROW_BYTES]
    finally:
        cache.close()
        fixture.artifact.close()


def test_route_table_collisions_and_tombstones_do_not_grow(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path, slots=2)
    try:
        route_bytes = fixture.cache.route_table_bytes
        fixture.cache.acquire_rows((1, 5)).release()  # same low bits in capacity four
        for row in (9, 1, 5, 9, 1):
            lease = fixture.cache.acquire_rows((row,))
            assert lease.row_bytes(0) == fixture_row(row)
            lease.release()
        assert fixture.cache.route_table_bytes == route_bytes
        assert len(fixture.cache._packed.route_keys) == fixture.cache.plan.route_capacity
    finally:
        fixture.close()


def test_route_deletion_churn_has_bounded_absent_probe(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path, slots=2)
    try:
        for row in range(ROW_COUNT):
            fixture.cache.acquire_rows((row,)).release()
        assert ROW_COUNT > fixture.cache.plan.route_capacity
        missing, probes = fixture.cache._packed.lookup_with_probes(100)
        assert missing is None
        assert probes <= fixture.cache.plan.slot_count + 1
        for row in (ROW_COUNT - 2, ROW_COUNT - 1):
            assert fixture.cache._packed.lookup(row) is not None
    finally:
        fixture.close()


def test_backward_shift_route_table_matches_random_oracle(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path, slots=2)
    rng = random.Random(38)
    oracle: dict[int, int] = {}
    try:
        for _ in range(500):
            if oracle and (len(oracle) == 2 or rng.randrange(2) == 0):
                row = rng.choice(tuple(oracle))
                fixture.cache._packed.remove(row)
                del oracle[row]
            else:
                row = rng.randrange(200)
                if row not in oracle:
                    slot = len(oracle)
                    fixture.cache._packed.insert(row, slot)
                    oracle[row] = slot
            for row, slot in oracle.items():
                assert fixture.cache._packed.lookup(row) == slot
            absent = next(row for row in range(200, 400) if row not in oracle)
            value, probes = fixture.cache._packed.lookup_with_probes(absent)
            assert value is None
            assert probes <= len(oracle) + 1
    finally:
        fixture.close()


def test_fragmented_transient_pool_splits_physical_group_without_rejection(
    tmp_path: Path,
) -> None:
    fixture = fixture_cache(tmp_path, slots=4, transient_rows=8)
    try:
        for slot in (1, 3, 5, 7):
            fixture.cache._transient_pool.used[slot] = 1
        lease = fixture.cache.acquire_rows((0, 1, 2, 3))
        assert [lease.row_bytes(index) for index in range(4)] == [
            fixture_row(row) for row in range(4)
        ]
        lease.release()
        assert sorted(length for _fd, _offset, length in fixture.reader.reads) == [
            ROW_BYTES
        ] * 4
        for slot in (1, 3, 5, 7):
            fixture.cache._transient_pool.used[slot] = 0
    finally:
        fixture.close()


def test_two_close_callers_wait_and_release_all_fixed_storage(tmp_path: Path) -> None:
    reader = ControlledReader()
    fixture = fixture_cache(tmp_path, slots=1, reader=reader)
    first_done = threading.Event()
    second_done = threading.Event()
    pending = fixture.cache.acquire_rows_async((1,))
    assert reader.started.wait(timeout=5)
    first = threading.Thread(target=lambda: (fixture.cache.close(), first_done.set()))
    second = threading.Thread(target=lambda: (fixture.cache.close(), second_done.set()))
    first.start()
    second.start()
    assert not first_done.wait(timeout=0.05)
    assert not second_done.wait(timeout=0.05)
    reader.release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert first_done.is_set() and second_done.is_set()
    assert fixture.cache.arena_bytes == 0
    assert fixture.cache.transient_storage_bytes == 0
    assert fixture.cache.metadata_bytes == 0
    assert fixture.cache.route_table_bytes == 0
    assert fixture.cache.total_reserved_bytes == 0
    assert fixture.cache._packed.slot_count == 0
    with pytest.raises(CancelledError):
        pending.result(timeout=5)
    fixture.artifact.close()


def test_f_nocache_second_shard_failure_rolls_back_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_cache(tmp_path)
    fixture.cache.close()
    calls: list[tuple[int, int]] = []
    assert qwen4_ngram.fcntl is not None
    monkeypatch.setattr(qwen4_ngram.fcntl, "F_NOCACHE", 48, raising=False)
    descriptors = [shard.fileno() for shard in fixture.artifact.shards]

    def transactional(descriptor: int, command: int, value: int) -> None:
        assert command == 48
        calls.append((descriptor, value))
        if descriptor == descriptors[1] and value == 1:
            raise OSError("second shard failure")

    monkeypatch.setattr(qwen4_ngram.fcntl, "fcntl", transactional)
    try:
        with pytest.raises(NGramCacheIOError, match="second shard failure"):
            NGramRowCache(
                fixture.artifact,
                NGramCacheConfig(
                    cache_limit_bytes=ROW_BYTES,
                    transient_limit_bytes=ROW_BYTES,
                    max_inflight_io_bytes=ROW_BYTES,
                    max_open_files=3,
                    bypass_page_cache=True,
                    eviction="lru",
                ),
                reader=RecordingReader(),
                allocator=bytearray,
            )
        assert calls == [(descriptors[0], 1), (descriptors[1], 1), (descriptors[0], 0)]
        assert fixture.artifact._cache_owner is None
    finally:
        fixture.artifact.close()


def test_page_cache_policy_is_restored_before_false_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_cache(tmp_path)
    fixture.cache.close()
    calls: list[tuple[int, int]] = []
    assert qwen4_ngram.fcntl is not None
    monkeypatch.setattr(qwen4_ngram.fcntl, "F_NOCACHE", 48, raising=False)
    monkeypatch.setattr(
        qwen4_ngram.fcntl,
        "fcntl",
        lambda descriptor, command, value: calls.append((descriptor, value)),
    )
    config = NGramCacheConfig(
        cache_limit_bytes=ROW_BYTES,
        transient_limit_bytes=ROW_BYTES,
        max_inflight_io_bytes=ROW_BYTES,
        max_open_files=3,
        bypass_page_cache=True,
        eviction="lru",
    )
    cache = NGramRowCache(
        fixture.artifact, config, reader=RecordingReader(), allocator=bytearray
    )
    descriptors = [shard.fileno() for shard in fixture.artifact.shards]
    cache.close()
    assert calls == [(descriptor, 1) for descriptor in descriptors] + [
        (descriptor, 0) for descriptor in descriptors
    ]
    false_cache = NGramRowCache(
        fixture.artifact,
        replace(config, bypass_page_cache=False),
        reader=RecordingReader(),
        allocator=bytearray,
    )
    false_cache.close()
    assert len(calls) == 2 * len(descriptors)
    fixture.artifact.close()


def test_page_cache_policy_rolls_back_when_later_allocation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_cache(tmp_path)
    fixture.cache.close()
    calls: list[tuple[int, int]] = []
    assert qwen4_ngram.fcntl is not None
    monkeypatch.setattr(qwen4_ngram.fcntl, "F_NOCACHE", 48, raising=False)
    monkeypatch.setattr(
        qwen4_ngram.fcntl,
        "fcntl",
        lambda descriptor, command, value: calls.append((descriptor, value)),
    )
    descriptors = [shard.fileno() for shard in fixture.artifact.shards]
    try:
        with pytest.raises(MemoryError, match="synthetic allocation"):
            NGramRowCache(
                fixture.artifact,
                NGramCacheConfig(
                    cache_limit_bytes=ROW_BYTES,
                    transient_limit_bytes=ROW_BYTES,
                    max_inflight_io_bytes=ROW_BYTES,
                    max_open_files=3,
                    bypass_page_cache=True,
                    eviction="lru",
                ),
                reader=RecordingReader(),
                allocator=lambda _size: (_ for _ in ()).throw(
                    MemoryError("synthetic allocation")
                ),
            )
        assert calls == [(descriptor, 1) for descriptor in descriptors] + [
            (descriptor, 0) for descriptor in descriptors
        ]
        assert fixture.artifact._cache_owner is None
    finally:
        fixture.artifact.close()


def test_restore_failure_is_persistent_but_artifact_can_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_cache(tmp_path)
    fixture.cache.close()
    assert qwen4_ngram.fcntl is not None
    monkeypatch.setattr(qwen4_ngram.fcntl, "F_NOCACHE", 48, raising=False)

    def fail_restore(_descriptor: int, _command: int, value: int) -> None:
        if value == 0:
            raise OSError("persistent restore failure")

    monkeypatch.setattr(qwen4_ngram.fcntl, "fcntl", fail_restore)
    cache = NGramRowCache(
        fixture.artifact,
        NGramCacheConfig(
            cache_limit_bytes=ROW_BYTES,
            transient_limit_bytes=ROW_BYTES,
            max_inflight_io_bytes=ROW_BYTES,
            max_open_files=3,
            bypass_page_cache=True,
            eviction="lru",
        ),
        reader=RecordingReader(),
        allocator=bytearray,
    )
    with pytest.raises(NGramCacheIOError, match="persistent restore failure"):
        cache.close()
    with pytest.raises(NGramCacheIOError, match="persistent restore failure"):
        cache.close()
    assert fixture.artifact._cache_owner is None
    with pytest.raises(NGramCacheIOError, match="poisoned for cache reuse"):
        NGramRowCache(
            fixture.artifact,
            cache.config,
            reader=RecordingReader(),
            allocator=bytearray,
        )
    fixture.artifact.close()


def test_poison_revokes_preexisting_materialized_lease_and_future(
    tmp_path: Path,
) -> None:
    fixture = fixture_cache(tmp_path, slots=2)
    try:
        materialized = fixture.cache.acquire_rows_async((1,))
        old_lease = materialized.result(timeout=5)
        fixture.cache._reader = BrokenReader()
        with pytest.raises(NGramCacheIOError, match="synthetic read failure"):
            fixture.cache.acquire_rows((2,))
        with pytest.raises(NGramCacheIOError, match="synthetic read failure"):
            old_lease.row_bytes(0)
        with pytest.raises(NGramCacheIOError, match="synthetic read failure"):
            materialized.result(timeout=5)
    finally:
        fixture.close()


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_fatal_reader_never_leaves_reset_or_close_transition_stuck(
    tmp_path: Path, operation: str
) -> None:
    reader = ControlledFatalReader()
    fixture = fixture_cache(tmp_path, reader=reader)
    fixture.cache.acquire_rows_async((1,))
    assert reader.started.wait(timeout=5)
    errors: list[BaseException] = []
    finished = threading.Event()

    def operate() -> None:
        try:
            getattr(fixture.cache, operation)()
        except BaseException as exc:  # noqa: BLE001 - capture fatal thread result
            errors.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=operate)
    worker.start()
    reader.release.set()
    worker.join(timeout=5)
    assert finished.is_set()
    assert len(errors) == 1
    assert isinstance(errors[0], KeyboardInterrupt)
    assert not fixture.cache._resetting
    if operation == "reset":
        with pytest.raises(KeyboardInterrupt, match="synthetic fatal reader"):
            fixture.cache.close()
    else:
        assert fixture.cache.closed
        with pytest.raises(KeyboardInterrupt, match="synthetic fatal reader"):
            fixture.cache.close()
    fixture.artifact.close()


def test_empty_and_overflowing_acquisitions_are_rejected(tmp_path: Path) -> None:
    fixture = fixture_cache(tmp_path)
    try:
        with pytest.raises(ValueError, match="empty"):
            fixture.cache.acquire_rows(())
        with pytest.raises(TypeError, match="exact integers"):
            fixture.cache.acquire_rows((True,))
        with pytest.raises(IndexError, match="out of range"):
            fixture.cache.acquire_rows((1 << 200,))
    finally:
        fixture.close()


def test_default_reader_capability_and_exclusive_owner_fail_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_cache(tmp_path)
    sizes: list[int] = []
    config = NGramCacheConfig(
        cache_limit_bytes=ROW_BYTES,
        transient_limit_bytes=ROW_BYTES,
        max_inflight_io_bytes=ROW_BYTES,
        max_open_files=3,
        bypass_page_cache=False,
        eviction="lru",
    )
    try:
        with pytest.raises(NGramCacheError, match="already has a cache owner"):
            NGramRowCache(
                fixture.artifact,
                config,
                reader=RecordingReader(),
                allocator=lambda size: (sizes.append(size), bytearray(size))[1],
            )
        assert sizes == []
        fixture.cache.close()
        monkeypatch.setattr(qwen4_ngram.os, "preadv", None)
        with pytest.raises(TypeError, match="preadv"):
            NGramRowCache(
                fixture.artifact,
                config,
                allocator=lambda size: (sizes.append(size), bytearray(size))[1],
            )
        assert sizes == []
    finally:
        fixture.cache.close()
        fixture.artifact.close()
