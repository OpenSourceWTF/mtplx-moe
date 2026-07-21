"""Bounded positional I/O for manifest-described expert records."""

from __future__ import annotations

import hashlib
import fcntl
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .expert_manifest import (
    ExpertManifest,
    ExpertManifestError,
    ExpertRecord,
    resolve_artifact_member,
)

try:
    _IOV_MAX = os.sysconf("SC_IOV_MAX")
except (ValueError, OSError, AttributeError):
    _IOV_MAX = 512
if _IOV_MAX <= 0:
    _IOV_MAX = 512


class ExpertIOError(RuntimeError):
    """Base error for a record that did not reach a complete verified state."""


class ExpertIOCancelled(ExpertIOError):
    pass


class ExpertIODeadlineExceeded(ExpertIOError):
    pass


class ExpertIOShortRead(ExpertIOError):
    pass


class ExpertIOIntegrityError(ExpertIOError):
    pass


@dataclass
class ExpertIOMetrics:
    record_requests: int = 0
    source_record_requests: int = 0
    sidecar_record_requests: int = 0
    # Backward-compatible name for logical range-reader invocations.
    read_operations: int = 0
    python_preadv_invocations: int = 0
    preadv_bytes_returned: int = 0
    native_positional_calls: int = 0
    native_bytes_returned: int = 0
    requested_bytes: int = 0
    read_bytes: int = 0
    read_ns: int = 0
    # Streamed rANS decode-on-miss (issue #113): compressed records read
    # fewer bytes off SSD than they decode to. ``read_bytes`` already counts
    # only the (smaller) bytes actually pulled; ``bytes_read_saved`` makes the
    # win explicit as ``raw - stored`` and never changes the memory plan.
    decoded_records: int = 0
    decoded_raw_bytes: int = 0
    bytes_read_saved: int = 0
    decode_ns: int = 0
    open_files_peak: int = 0
    short_reads: int = 0
    integrity_errors: int = 0
    cancellations: int = 0
    deadline_errors: int = 0
    io_errors: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **values: int) -> None:
        with self._lock:
            for name, value in values.items():
                setattr(self, name, int(getattr(self, name)) + int(value))

    def observe_open_count(self, count: int) -> None:
        with self._lock:
            self.open_files_peak = max(self.open_files_peak, int(count))

    def as_dict(self) -> dict[str, int | float]:
        with self._lock:
            result = {
                name: int(getattr(self, name))
                for name in (
                    "record_requests",
                    "source_record_requests",
                    "sidecar_record_requests",
                    "read_operations",
                    "python_preadv_invocations",
                    "preadv_bytes_returned",
                    "native_positional_calls",
                    "native_bytes_returned",
                    "requested_bytes",
                    "read_bytes",
                    "read_ns",
                    "decoded_records",
                    "decoded_raw_bytes",
                    "bytes_read_saved",
                    "decode_ns",
                    "open_files_peak",
                    "short_reads",
                    "integrity_errors",
                    "cancellations",
                    "deadline_errors",
                    "io_errors",
                )
            }
        result["read_mib_per_second"] = (
            result["read_bytes"] / 1024**2 / (result["read_ns"] / 1e9)
            if result["read_ns"]
            else 0.0
        )
        return result

    def reads_per_record(self) -> float:
        """Positional syscalls issued per record request (I/O efficiency).

        The numerator is the per-syscall counters
        (``python_preadv_invocations`` + ``native_positional_calls``) -- the
        real ``preadv``/native ``pread`` calls -- not ``read_operations``,
        which counts logical range-reader invocations (one per
        ``_read_range_into``/``_readv_range_into`` call) rather than syscalls.
        The denominator is ``record_requests``. Returns ``0.0`` before any
        record has been requested. Pure derived accessor; no hot-path cost.
        """

        with self._lock:
            syscalls = int(self.python_preadv_invocations) + int(
                self.native_positional_calls
            )
            records = int(self.record_requests)
        return syscalls / records if records else 0.0

    def assert_read_efficiency(self, max_reads_per_record: float) -> float:
        """Fail closed when reads-per-record exceeds an I/O budget.

        Lets a benchmark gate assert the coalesced source path actually
        collapses a record's contiguous segments into a small number of
        positional syscalls. Returns the measured ratio on success; raises
        :class:`ExpertIOError` naming the actual ratio and its terms on
        failure.
        """

        with self._lock:
            syscalls = int(self.python_preadv_invocations) + int(
                self.native_positional_calls
            )
            records = int(self.record_requests)
        ratio = syscalls / records if records else 0.0
        if ratio > float(max_reads_per_record):
            raise ExpertIOError(
                f"expert reads-per-record {ratio:.4f} exceeds the "
                f"{float(max_reads_per_record):.4f} budget "
                f"({syscalls} positional syscalls over {records} records)"
            )
        return ratio


@dataclass
class _FDEntry:
    fd: int
    users: int = 0


class PositionalExpertReader:
    """Thread-safe bounded descriptor cache with exact positional reads.

    Reads go directly into a caller-owned fixed slot buffer.  No record-sized
    temporary allocation is made by this class.  The optional native backend
    has the same contract and is loaded lazily when available.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        max_open_files: int = 16,
        max_read_chunk_bytes: int = 8 * 1024 * 1024,
        use_native: bool = True,
        bypass_page_cache: bool = False,
        pipeline_ledger: Any | None = None,
        codec_sidecar: Any | None = None,
        codec_verify: bool = True,
    ) -> None:
        if isinstance(max_open_files, bool) or not isinstance(max_open_files, int):
            raise TypeError("max_open_files must be an integer")
        if max_open_files <= 0:
            raise ValueError("max_open_files must be positive")
        if isinstance(max_read_chunk_bytes, bool) or not isinstance(
            max_read_chunk_bytes, int
        ):
            raise TypeError("max_read_chunk_bytes must be an integer")
        if max_read_chunk_bytes <= 0:
            raise ValueError("max_read_chunk_bytes must be positive")
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"expert artifact root is not a directory: {self.root}")
        self.max_open_files = max_open_files
        self.max_read_chunk_bytes = max_read_chunk_bytes
        if not isinstance(bypass_page_cache, bool):
            raise TypeError("bypass_page_cache must be bool")
        self.bypass_page_cache = bypass_page_cache
        self.pipeline_ledger = pipeline_ledger
        # Streamed compressed sidecar (issue #113). When present with a
        # ``rans32x-v1`` codec, records with a codec entry are read from the
        # (smaller) compressed sidecar and decoded through the in-kernel rANS
        # decoder before landing in the slot -- bitwise-identical to reading
        # the uncompressed record. ``None`` leaves every read byte-unchanged.
        self.codec_sidecar = codec_sidecar
        # Decoded-record hash verification rides on top of the caller's
        # verify_hash: the rANS container carries its own structural guards
        # (lane directory + guard bytes), so the post-decode sha256 is a
        # separately priced belt the config can drop without touching the
        # uncompressed path's integrity mode.
        self.codec_verify = bool(codec_verify)
        self._codec_record_map = (
            codec_sidecar.record_map() if codec_sidecar is not None else None
        )
        self._decode_container = None  # lazily bound Metal decoder (needs MLX)
        self.metrics = ExpertIOMetrics()
        self._condition = threading.Condition()
        self._entries: OrderedDict[str, _FDEntry] = OrderedDict()
        self._closed = False
        self._native_read_into = self._load_native_reader() if use_native else None

    @staticmethod
    def _load_native_reader() -> Any | None:
        try:
            from mtplx_native_expert_io import pread_exact_into

            return pread_exact_into
        except Exception:
            return None

    @property
    def backend(self) -> str:
        return "native" if self._native_read_into is not None else "python-preadv"

    @property
    def cache_mode(self) -> str:
        return "f-nocache" if self.bypass_page_cache else "buffered"

    def _evict_idle_locked(self) -> bool:
        for key, entry in list(self._entries.items()):
            if entry.users:
                continue
            del self._entries[key]
            try:
                os.close(entry.fd)
            except OSError:
                pass
            return True
        return False

    @contextmanager
    def _lease(self, relative_name: str) -> Iterator[int]:
        resolved = resolve_artifact_member(self.root, relative_name)
        key = str(resolved)
        with self._condition:
            while True:
                if self._closed:
                    raise ExpertIOError("expert reader is closed")
                entry = self._entries.get(key)
                if entry is not None:
                    entry.users += 1
                    self._entries.move_to_end(key)
                    break
                if (
                    len(self._entries) < self.max_open_files
                    or self._evict_idle_locked()
                ):
                    flags = os.O_RDONLY
                    flags |= getattr(os, "O_CLOEXEC", 0)
                    flags |= getattr(os, "O_NOFOLLOW", 0)
                    try:
                        fd = os.open(resolved, flags)
                        if self.bypass_page_cache:
                            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                    except OSError as exc:
                        try:
                            os.close(fd)
                        except (OSError, UnboundLocalError):
                            pass
                        raise ExpertIOError(
                            f"could not open {relative_name}: {exc}"
                        ) from exc
                    entry = _FDEntry(fd=fd, users=1)
                    self._entries[key] = entry
                    self.metrics.observe_open_count(len(self._entries))
                    break
                self._condition.wait()
        try:
            yield entry.fd
        finally:
            with self._condition:
                entry.users -= 1
                self._entries.move_to_end(key)
                self._condition.notify_all()

    @staticmethod
    def _check_cancelled(
        cancel_event: threading.Event | None,
        deadline_ns: int | None,
    ) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise ExpertIOCancelled("expert read was cancelled")
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise ExpertIODeadlineExceeded("expert read deadline exceeded")

    @staticmethod
    def _writable_bytes(destination: Any) -> memoryview:
        try:
            view = memoryview(destination)
        except TypeError as exc:
            raise TypeError(
                "destination must support the writable buffer protocol"
            ) from exc
        if view.readonly:
            raise TypeError("destination must be writable")
        if not view.c_contiguous:
            raise TypeError("destination must be C-contiguous")
        try:
            return view.cast("B")
        except TypeError as exc:
            raise TypeError("destination must be byte-addressable") from exc

    def _start_pipeline_range(
        self,
        pipeline_ledger: Any,
        logical_bytes: int,
        pipeline_phase: str | None,
    ) -> tuple[Any | None, Any]:
        """Begin optional diagnostics without changing the read outcome."""

        try:
            if pipeline_phase is None:
                token = pipeline_ledger.range_started(logical_bytes)
            else:
                token = pipeline_ledger.range_started(
                    logical_bytes,
                    phase=pipeline_phase,
                )
            return pipeline_ledger, token
        except Exception:
            self._mark_pipeline_incomplete(pipeline_ledger, pipeline_phase)
            return None, None

    @staticmethod
    def _mark_pipeline_incomplete(
        pipeline_ledger: Any,
        pipeline_phase: str | None,
    ) -> None:
        try:
            if pipeline_phase is None:
                pipeline_ledger.mark_incomplete()
            else:
                pipeline_ledger.mark_incomplete(phase=pipeline_phase)
        except Exception:
            pass

    @classmethod
    def _finish_pipeline_range(
        cls,
        pipeline_ledger: Any | None,
        token: Any,
        pipeline_phase: str | None,
    ) -> None:
        """Finish optional diagnostics without masking data-path outcomes."""

        if pipeline_ledger is None:
            return
        try:
            pipeline_ledger.range_completed(token)
        except Exception:
            cls._mark_pipeline_incomplete(pipeline_ledger, pipeline_phase)

    def _read_range_into(
        self,
        relative_name: str,
        source_offset: int,
        destination: memoryview,
        *,
        cancel_event: threading.Event | None,
        deadline_ns: int | None,
        pipeline_phase: str | None = None,
    ) -> None:
        self._check_cancelled(cancel_event, deadline_ns)
        requested = len(destination)
        started = time.monotonic_ns()
        read_total = 0
        python_preadv_invocations = 0
        preadv_bytes_returned = 0
        native_positional_calls = 0
        native_bytes_returned = 0
        pipeline_ledger = self.pipeline_ledger
        range_token = None
        if pipeline_ledger is not None:
            pipeline_ledger, range_token = self._start_pipeline_range(
                pipeline_ledger,
                requested,
                pipeline_phase,
            )
        try:
            with self._lease(relative_name) as fd:
                while read_total < requested:
                    self._check_cancelled(cancel_event, deadline_ns)
                    count = min(self.max_read_chunk_bytes, requested - read_total)
                    target = destination[read_total : read_total + count]
                    try:
                        native_read_into = self._native_read_into
                        if native_read_into is not None:
                            try:
                                native_positional_calls += 1
                                read_now = int(
                                    native_read_into(
                                        fd,
                                        source_offset + read_total,
                                        target,
                                    )
                                )
                            except Exception as exc:
                                # nanobind maps ``std::system_error`` to a
                                # RuntimeError rather than OSError.  Preserve
                                # the reader's fail-closed public contract and
                                # metrics regardless of backend exception type.
                                self.metrics.update(io_errors=1)
                                raise ExpertIOError(
                                    f"native positional read failed: {exc}"
                                ) from exc
                        else:
                            python_preadv_invocations += 1
                            read_now = int(
                                os.preadv(fd, [target], source_offset + read_total)
                            )
                    except InterruptedError:
                        continue
                    if read_now <= 0:
                        self.metrics.update(short_reads=1)
                        raise ExpertIOShortRead(
                            f"short read from {relative_name} at "
                            f"{source_offset + read_total}; wanted {requested - read_total} bytes"
                        )
                    if native_read_into is not None:
                        native_bytes_returned += read_now
                    else:
                        preadv_bytes_returned += read_now
                    read_total += read_now
        except ExpertIOCancelled:
            self.metrics.update(cancellations=1)
            raise
        except ExpertIODeadlineExceeded:
            self.metrics.update(deadline_errors=1)
            raise
        except ExpertIOError:
            raise
        except OSError as exc:
            self.metrics.update(io_errors=1)
            raise ExpertIOError(f"positional read failed: {exc}") from exc
        finally:
            read_elapsed_ns = time.monotonic_ns() - started
            if pipeline_ledger is not None:
                self._finish_pipeline_range(
                    pipeline_ledger,
                    range_token,
                    pipeline_phase,
                )
            self.metrics.update(
                read_operations=1,
                python_preadv_invocations=python_preadv_invocations,
                preadv_bytes_returned=preadv_bytes_returned,
                native_positional_calls=native_positional_calls,
                native_bytes_returned=native_bytes_returned,
                requested_bytes=requested,
                read_bytes=read_total,
                read_ns=read_elapsed_ns,
            )

    def _readv_range_into(
        self,
        relative_name: str,
        source_offset: int,
        destinations: tuple[memoryview, ...],
        *,
        cancel_event: threading.Event | None,
        deadline_ns: int | None,
        pipeline_phase: str | None = None,
    ) -> None:
        """Scatter one contiguous file range into component-bank rows."""

        self._check_cancelled(cancel_event, deadline_ns)
        requested = sum(len(destination) for destination in destinations)
        started = time.monotonic_ns()
        read_total = 0
        python_preadv_invocations = 0
        preadv_bytes_returned = 0
        pipeline_ledger = self.pipeline_ledger
        range_token = None
        if pipeline_ledger is not None:
            pipeline_ledger, range_token = self._start_pipeline_range(
                pipeline_ledger,
                requested,
                pipeline_phase,
            )
        pending = [destination for destination in destinations if len(destination)]
        try:
            with self._lease(relative_name) as fd:
                while pending:
                    self._check_cancelled(cancel_event, deadline_ns)
                    try:
                        python_preadv_invocations += 1
                        # preadv rejects vectors above IOV_MAX with EINVAL;
                        # the partial-read loop below resumes the remainder.
                        read_now = int(
                            os.preadv(
                                fd,
                                pending[:_IOV_MAX],
                                source_offset + read_total,
                            )
                        )
                    except InterruptedError:
                        continue
                    if read_now <= 0:
                        self.metrics.update(short_reads=1)
                        raise ExpertIOShortRead(
                            f"short scatter read from {relative_name} at "
                            f"{source_offset + read_total}; wanted "
                            f"{requested - read_total} bytes"
                        )
                    preadv_bytes_returned += read_now
                    read_total += read_now
                    consumed = read_now
                    next_pending: list[memoryview] = []
                    for destination in pending:
                        if consumed >= len(destination):
                            consumed -= len(destination)
                            continue
                        if consumed:
                            destination = destination[consumed:]
                            consumed = 0
                        next_pending.append(destination)
                    pending = next_pending
        except ExpertIOCancelled:
            self.metrics.update(cancellations=1)
            raise
        except ExpertIODeadlineExceeded:
            self.metrics.update(deadline_errors=1)
            raise
        except ExpertIOError:
            raise
        except OSError as exc:
            self.metrics.update(io_errors=1)
            raise ExpertIOError(f"positional scatter read failed: {exc}") from exc
        finally:
            read_elapsed_ns = time.monotonic_ns() - started
            if pipeline_ledger is not None:
                self._finish_pipeline_range(
                    pipeline_ledger,
                    range_token,
                    pipeline_phase,
                )
            self.metrics.update(
                read_operations=1,
                python_preadv_invocations=python_preadv_invocations,
                preadv_bytes_returned=preadv_bytes_returned,
                requested_bytes=requested,
                read_bytes=read_total,
                read_ns=read_elapsed_ns,
            )

    @staticmethod
    def _contiguous_source_runs(
        segments: tuple[Any, ...],
    ) -> list[tuple[int, int]]:
        """Group segment indices into maximal contiguous same-shard runs.

        Consecutive segments join a run when they share a shard file and the
        next begins exactly where the previous one ends
        (``next.offset == prev.offset + prev.length``) -- i.e. they form one
        physical extent a single positional read can service. Each run is
        returned as a half-open ``(start_index, count)`` span over ``segments``
        in order; a shard change or an offset gap starts a new run so
        genuinely non-contiguous or multi-shard records keep one read per
        segment (the exact prior fallback).
        """

        runs: list[tuple[int, int]] = []
        start = 0
        for index in range(1, len(segments)):
            previous = segments[index - 1]
            current = segments[index]
            if (
                current.shard == previous.shard
                and current.offset == previous.offset + previous.length
            ):
                continue
            runs.append((start, index - start))
            start = index
        if segments:
            runs.append((start, len(segments) - start))
        return runs

    def _codec_entry(self, record: ExpertRecord) -> Any | None:
        """The compressed-sidecar entry for a record, or None (codec inactive)."""

        if self._codec_record_map is None:
            return None
        return self._codec_record_map.get((record.layer, record.expert))

    def _decode_container_fn(self) -> Any:
        """Lazily bind the Metal rANS decoder (importing MLX only on demand)."""

        fn = self._decode_container
        if fn is None:
            from mtplx.expert_rans_metal import decode_container

            fn = decode_container
            self._decode_container = fn
        return fn

    def _decode_targets(
        self, destination: Any, record: ExpertRecord
    ) -> tuple[memoryview | None, tuple[memoryview, ...] | None]:
        record_views = getattr(destination, "record_views", None)
        if callable(record_views):
            component_views = tuple(record_views(record))
            if len(component_views) != len(record.segments):
                raise ValueError("component slot does not cover every record segment")
            if sum(len(view) for view in component_views) != record.logical_bytes:
                raise ValueError("component slot byte count differs from expert record")
            return None, component_views
        view = self._writable_bytes(destination)
        if len(view) != record.logical_bytes:
            raise ValueError(
                f"slot buffer has {len(view)} bytes; record needs {record.logical_bytes}"
            )
        return view, None

    def _read_record_decoded(
        self,
        codec_entry: Any,
        manifest: ExpertManifest,
        record: ExpertRecord,
        destination: Any,
        *,
        verify_hash: bool,
        cancel_event: threading.Event | None,
        deadline_ns: int | None,
        pipeline_phase: str | None,
    ) -> str:
        """Read a compressed record, decode it, and land raw bytes in the slot.

        The SSD read pulls only the compressed container (accounted in
        ``read_bytes``); the in-kernel rANS decoder rebuilds the raw record;
        the raw bytes are copied into the slot's component/flat views exactly
        as the uncompressed path would have written them. Bitwise-identical to
        reading the uncompressed record (the #112 decode-store parity).
        """

        import numpy as np

        assert self.codec_sidecar is not None
        verify_hash = verify_hash and self.codec_verify
        self.metrics.update(record_requests=1, sidecar_record_requests=1)
        view, component_views = self._decode_targets(destination, record)
        try:
            # 1) Pull the (smaller) compressed container off SSD. Reusing the
            #    range reader keeps native/preadv, cancel/deadline, and the
            #    read-bytes accounting identical -- read_bytes counts only the
            #    compressed bytes actually moved.
            staging = bytearray(int(codec_entry.length))
            self._read_range_into(
                self.codec_sidecar.file,
                int(codec_entry.offset),
                memoryview(staging),
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
                pipeline_phase=pipeline_phase,
            )
            # 2) Decode through the Metal kernel (host round-trip is a fast
            #    unified-memory copy, far below the SSD read it replaces).
            decode_started = time.monotonic_ns()
            decoded = self._decode_container_fn()(bytes(staging))
            raw = np.array(decoded, dtype=np.uint8).reshape(-1)
            if raw.size < record.logical_bytes:
                raise ExpertIOShortRead(
                    f"decoded record ({record.layer}, {record.expert}) is "
                    f"{raw.size} bytes; record needs {record.logical_bytes}"
                )
            raw_view = memoryview(raw)[: record.logical_bytes]
            # 3) Land raw bytes exactly where the uncompressed path would.
            if component_views is None:
                assert view is not None
                view[:] = raw_view
            else:
                cursor = 0
                for target in component_views:
                    end = cursor + len(target)
                    target[:] = raw_view[cursor:end]
                    cursor = end
            decode_ns = time.monotonic_ns() - decode_started
            self.metrics.update(
                decoded_records=1,
                decoded_raw_bytes=record.logical_bytes,
                bytes_read_saved=max(
                    record.logical_bytes - int(codec_entry.length), 0
                ),
                decode_ns=decode_ns,
            )
            if verify_hash:
                digest = hashlib.sha256(raw_view).hexdigest()
            else:
                digest = "unverified"
        finally:
            if component_views is not None:
                for component_view in component_views:
                    try:
                        component_view.release()
                    except Exception:
                        pass
        if verify_hash:
            if record.sha256 is None:
                self.metrics.update(integrity_errors=1)
                raise ExpertIOIntegrityError("expert record has no trusted hash")
            if digest != record.sha256:
                self.metrics.update(integrity_errors=1)
                raise ExpertIOIntegrityError(
                    f"expert record hash mismatch: ({record.layer}, {record.expert})"
                )
        return digest

    def read_record_into(
        self,
        manifest: ExpertManifest,
        record: ExpertRecord,
        destination: Any,
        *,
        prefer_sidecar: bool = True,
        verify_hash: bool = True,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
        pipeline_phase: str | None = None,
    ) -> str:
        """Fill a fixed record buffer and return its SHA-256 digest."""

        codec_entry = self._codec_entry(record)
        if codec_entry is not None:
            return self._read_record_decoded(
                codec_entry,
                manifest,
                record,
                destination,
                verify_hash=verify_hash,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
                pipeline_phase=pipeline_phase,
            )

        record_views = getattr(destination, "record_views", None)
        component_views: tuple[memoryview, ...] | None = None
        if callable(record_views):
            component_views = tuple(record_views(record))
            if len(component_views) != len(record.segments):
                raise ValueError("component slot does not cover every record segment")
            if sum(len(view) for view in component_views) != record.logical_bytes:
                raise ValueError("component slot byte count differs from expert record")
            view = None
        else:
            view = self._writable_bytes(destination)
            if len(view) != record.logical_bytes:
                raise ValueError(
                    f"slot buffer has {len(view)} bytes; record needs {record.logical_bytes}"
                )
        self.metrics.update(record_requests=1)
        try:
            if prefer_sidecar and manifest.sidecar is not None:
                if record.sidecar_offset is None or record.sidecar_length is None:
                    raise ExpertIOError("manifest sidecar record is incomplete")
                self.metrics.update(sidecar_record_requests=1)
                if component_views is None:
                    assert view is not None
                    self._read_range_into(
                        manifest.sidecar.file,
                        record.sidecar_offset,
                        view,
                        cancel_event=cancel_event,
                        deadline_ns=deadline_ns,
                        pipeline_phase=pipeline_phase,
                    )
                else:
                    self._readv_range_into(
                        manifest.sidecar.file,
                        record.sidecar_offset,
                        component_views,
                        cancel_event=cancel_event,
                        deadline_ns=deadline_ns,
                        pipeline_phase=pipeline_phase,
                    )
            else:
                self.metrics.update(source_record_requests=1)
                segments = record.segments
                segment_cursors: list[int] = []
                cursor = 0
                for segment in segments:
                    segment_cursors.append(cursor)
                    cursor += segment.length
                if cursor != record.logical_bytes:
                    raise ExpertIOShortRead(
                        "expert source segments did not fill the slot"
                    )
                # Contiguous same-shard segments are a single physical extent
                # on disk, so coalesce each run into ONE positional read (flat
                # view) or ONE scatter (component views) instead of paying a
                # syscall per segment. Non-contiguous or multi-shard runs keep
                # the per-segment fallback, exact prior behavior.
                for start, count in self._contiguous_source_runs(segments):
                    run = segments[start : start + count]
                    run_shard = run[0].shard
                    run_offset = run[0].offset
                    if component_views is None:
                        assert view is not None
                        run_start = segment_cursors[start]
                        run_stop = run_start + sum(
                            segment.length for segment in run
                        )
                        self._read_range_into(
                            run_shard,
                            run_offset,
                            view[run_start:run_stop],
                            cancel_event=cancel_event,
                            deadline_ns=deadline_ns,
                            pipeline_phase=pipeline_phase,
                        )
                    elif count == 1:
                        # Genuinely isolated segment: keep the single-range
                        # reader (native-backend eligible), exact prior path.
                        self._read_range_into(
                            run_shard,
                            run_offset,
                            component_views[start],
                            cancel_event=cancel_event,
                            deadline_ns=deadline_ns,
                            pipeline_phase=pipeline_phase,
                        )
                    else:
                        self._readv_range_into(
                            run_shard,
                            run_offset,
                            component_views[start : start + count],
                            cancel_event=cancel_event,
                            deadline_ns=deadline_ns,
                            pipeline_phase=pipeline_phase,
                        )
            if verify_hash:
                hasher = hashlib.sha256()
                if component_views is None:
                    assert view is not None
                    hasher.update(view)
                else:
                    for component_view in component_views:
                        hasher.update(component_view)
                digest = hasher.hexdigest()
            else:
                # Do not report the manifest hash as if these bytes were
                # verified; trust modes must be visible in telemetry.
                digest = "unverified"
        finally:
            if component_views is not None:
                for component_view in component_views:
                    try:
                        component_view.release()
                    except Exception:
                        pass
        if verify_hash:
            if record.sha256 is None:
                self.metrics.update(integrity_errors=1)
                raise ExpertIOIntegrityError("expert record has no trusted hash")
            if digest != record.sha256:
                self.metrics.update(integrity_errors=1)
                raise ExpertIOIntegrityError(
                    f"expert record hash mismatch: ({record.layer}, {record.expert})"
                )
        return digest

    def read_component_records_into(
        self,
        manifest: ExpertManifest,
        items: tuple[tuple[ExpertRecord, Any], ...],
        *,
        verify_hash: bool = True,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
        pipeline_phase: str | None = None,
    ) -> tuple[str, ...]:
        """Read offset-ordered adjacent sidecar records with scatter preadv."""

        if not items:
            return ()
        if self._codec_record_map is not None and all(
            self._codec_entry(record) is not None for record, _destination in items
        ):
            # Compressed records are entropy-coded, not scatter-contiguous:
            # decode each one individually (the batch scatter fast path has no
            # meaning under a codec). Per-record decode still lands bytes in
            # slot order, bitwise-identical to the uncompressed batch.
            digests: list[str] = []
            for record, destination in items:
                codec_entry = self._codec_entry(record)
                assert codec_entry is not None
                digests.append(
                    self._read_record_decoded(
                        codec_entry,
                        manifest,
                        record,
                        destination,
                        verify_hash=verify_hash,
                        cancel_event=cancel_event,
                        deadline_ns=deadline_ns,
                        pipeline_phase=pipeline_phase,
                    )
                )
            return tuple(digests)
        if manifest.sidecar is None:
            raise ExpertIOError("component record batch requires a sidecar")
        prepared: list[tuple[int, ExpertRecord, tuple[memoryview, ...]]] = []
        for index, (record, destination) in enumerate(items):
            record_views = getattr(destination, "record_views", None)
            if not callable(record_views):
                raise TypeError("component record batch requires component slots")
            views = tuple(record_views(record))
            if sum(len(view) for view in views) != record.logical_bytes:
                raise ValueError("component slot byte count differs from expert record")
            if record.sidecar_offset is None or record.sidecar_length is None:
                raise ExpertIOError("manifest sidecar record is incomplete")
            prepared.append((index, record, views))
        prepared.sort(key=lambda item: int(item[1].sidecar_offset or 0))
        self.metrics.update(
            record_requests=len(prepared),
            sidecar_record_requests=len(prepared),
        )
        digests = [""] * len(prepared)
        try:
            groups: list[list[tuple[int, ExpertRecord, tuple[memoryview, ...]]]] = []
            for item in prepared:
                if not groups:
                    groups.append([item])
                    continue
                previous = groups[-1][-1][1]
                expected = int(previous.sidecar_offset or 0) + int(
                    previous.sidecar_length or 0
                )
                if int(item[1].sidecar_offset or 0) == expected:
                    groups[-1].append(item)
                else:
                    groups.append([item])
            for group in groups:
                flat_views = tuple(
                    view for _index, _record, views in group for view in views
                )
                self._readv_range_into(
                    manifest.sidecar.file,
                    int(group[0][1].sidecar_offset or 0),
                    flat_views,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                    pipeline_phase=pipeline_phase,
                )
                for original_index, record, views in group:
                    if verify_hash:
                        hasher = hashlib.sha256()
                        for view in views:
                            hasher.update(view)
                        digest = hasher.hexdigest()
                        if record.sha256 is None:
                            self.metrics.update(integrity_errors=1)
                            raise ExpertIOIntegrityError(
                                "expert record has no trusted hash"
                            )
                        if digest != record.sha256:
                            self.metrics.update(integrity_errors=1)
                            raise ExpertIOIntegrityError(
                                "expert record hash mismatch: "
                                f"({record.layer}, {record.expert})"
                            )
                    else:
                        # Same telemetry honesty as the single-record path.
                        digest = "unverified"
                    digests[original_index] = digest
        finally:
            for _index, _record, views in prepared:
                for view in views:
                    try:
                        view.release()
                    except Exception:
                        pass
        return tuple(digests)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            while any(entry.users for entry in self._entries.values()):
                self._condition.wait()
            entries = tuple(self._entries.values())
            self._entries.clear()
            self._condition.notify_all()
        for entry in entries:
            try:
                os.close(entry.fd)
            except OSError:
                pass

    def __enter__(self) -> PositionalExpertReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def manifest_error_as_io_error(exc: ExpertManifestError) -> ExpertIOError:
    return ExpertIOError(str(exc))
