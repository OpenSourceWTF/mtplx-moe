from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from mtplx import qwen4_ngram
from mtplx.qwen4_ngram import (
    NGramCacheClosed,
    NGramCacheConfig,
    NGramCacheFull,
    NGramCacheIOError,
    NGramFileIdentity,
    NGramManifest,
    NGramRowCache,
    NGramShard,
    VerifiedNGramArtifact,
    VerifiedNGramShard,
)

ROW_BYTES = 8
ROW_COUNT = 12


def fixture_row(row: int) -> bytes:
    return bytes((row,)) * ROW_BYTES


class RecordingReader:
    def __init__(self) -> None:
        self.reads: list[tuple[int, int, int]] = []
        self.nocache_ranges: list[tuple[int, int, int]] = []

    def read(
        self,
        shard: VerifiedNGramShard,
        offset: int,
        length: int,
        *,
        bypass_page_cache: bool,
    ) -> bytes:
        item = (shard.fileno(), offset, length)
        self.reads.append(item)
        payload = shard.pread(offset, length)
        if bypass_page_cache:
            self.nocache_ranges.append(item)
        return payload


class ShortReader(RecordingReader):
    def read(self, *args: object, **kwargs: object) -> bytes:
        return super().read(*args, **kwargs)[:-1]  # type: ignore[arg-type]


class BrokenReader(RecordingReader):
    def read(self, *args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise OSError("synthetic read failure")


class ControlledReader(RecordingReader):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def read(self, *args: object, **kwargs: object) -> bytes:
        self.started.set()
        assert self.release.wait(timeout=5)
        return super().read(*args, **kwargs)  # type: ignore[arg-type]


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
    max_open_files: int = 2,
    eviction: str = "lru",
    bypass_page_cache: bool = False,
    reader: RecordingReader | None = None,
) -> CacheFixture:
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
            max_open_files=2,
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
                    max_open_files=2,
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
                    max_open_files=3,
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
    artifact_fixture = fixture_cache(tmp_path, max_open_files=2)
    artifact_fixture.cache.close()
    try:
        with pytest.raises(ValueError, match="max_open_files"):
            NGramRowCache(
                artifact_fixture.artifact,
                NGramCacheConfig(
                    cache_limit_bytes=ROW_BYTES,
                    transient_limit_bytes=ROW_BYTES,
                    max_inflight_io_bytes=ROW_BYTES,
                    max_open_files=1,
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
        {"cache_limit_bytes": ROW_BYTES + 1},
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
        "max_open_files": 2,
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
