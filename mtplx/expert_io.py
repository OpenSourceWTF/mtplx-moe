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
                cursor = 0
                for index, segment in enumerate(record.segments):
                    end = cursor + segment.length
                    target = (
                        component_views[index]
                        if component_views is not None
                        else view[cursor:end]
                    )
                    self._read_range_into(
                        segment.shard,
                        segment.offset,
                        target,
                        cancel_event=cancel_event,
                        deadline_ns=deadline_ns,
                        pipeline_phase=pipeline_phase,
                    )
                    cursor = end
                if cursor != record.logical_bytes:
                    raise ExpertIOShortRead(
                        "expert source segments did not fill the slot"
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
