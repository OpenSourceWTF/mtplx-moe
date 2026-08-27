from __future__ import annotations

from concurrent.futures import Future
import hashlib
import os

from mtplx.qwen4_ngram import (
    NGramCacheConfig,
    NGramFileIdentity,
    NGramManifest,
    NGramRowCache,
    NGramShard,
    VerifiedNGramArtifact,
    VerifiedNGramShard,
)


class _Reader:
    def read_into(self, shard, offset, target):
        target[:] = os.pread(shard.fileno(), target.nbytes, offset)
        return target.nbytes


class _InlineExecutor:
    def __init__(self) -> None:
        self.submissions = 0

    def submit(self, function, *args):
        self.submissions += 1
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **_kwargs) -> None:
        return None


def _cache(tmp_path, *, slots=2, rows=8):
    source = tmp_path / "ngram.bin"
    payload = b"".join(bytes([row]) * 100 for row in range(rows))
    source.write_bytes(payload)
    shard = NGramShard(
        name=source.name,
        tensor="ngram",
        start_row=0,
        row_count=rows,
        data_offset=0,
        data_bytes=len(payload),
        file_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    manifest = NGramManifest(
        source_repo="repo",
        source_revision="revision",
        storage="affine-q4-g32",
        row_width=160,
        row_bytes=100,
        padded_rows=rows,
        shards=(shard,),
    )
    descriptor = os.open(source, os.O_RDONLY)
    metadata = os.fstat(descriptor)
    identity = NGramFileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )
    artifact = VerifiedNGramArtifact(
        manifest,
        os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)),
        (VerifiedNGramShard(shard, descriptor, identity),),
    )
    cache = NGramRowCache(
        artifact,
        NGramCacheConfig(
            cache_limit_bytes=slots * 100,
            transient_limit_bytes=rows * 100,
            max_inflight_io_bytes=rows * 100,
            max_open_files=2,
            bypass_page_cache=False,
            eviction="lru",
        ),
        reader=_Reader(),
        allocator=bytearray,
    )
    cache._executor.shutdown(wait=True)
    executor = _InlineExecutor()
    cache._executor = executor
    return cache, artifact, executor


def test_lru_eviction_preserves_pinned_rows_and_exact_bytes(tmp_path) -> None:
    cache, artifact, _executor = _cache(tmp_path)
    try:
        pinned = cache.acquire_rows([0])
        with cache.acquire_rows([1]) as lease:
            assert lease.row_bytes(0) == bytes([1]) * 100
        with cache.acquire_rows([2]) as lease:
            assert lease.row_bytes(0) == bytes([2]) * 100
        assert cache._packed.lookup(0) is not None
        assert cache._packed.lookup(1) is None
        pinned.release()
    finally:
        cache.close()
        artifact.close()


def test_one_acquisition_submits_one_bounded_io_task(tmp_path) -> None:
    cache, artifact, executor = _cache(tmp_path, slots=4)
    try:
        with cache.acquire_rows([0, 2, 4]) as lease:
            assert lease.row_bytes(1) == bytes([2]) * 100
        assert executor.submissions == 1
    finally:
        cache.close()
        artifact.close()
