"""Bounded positional I/O for manifest-described expert records."""

from __future__ import annotations

import hashlib
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
    read_operations: int = 0
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
                    except OSError as exc:
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

    def _read_range_into(
        self,
        relative_name: str,
        source_offset: int,
        destination: memoryview,
        *,
        cancel_event: threading.Event | None,
        deadline_ns: int | None,
    ) -> None:
        self._check_cancelled(cancel_event, deadline_ns)
        requested = len(destination)
        started = time.monotonic_ns()
        read_total = 0
        try:
            with self._lease(relative_name) as fd:
                while read_total < requested:
                    self._check_cancelled(cancel_event, deadline_ns)
                    count = min(self.max_read_chunk_bytes, requested - read_total)
                    target = destination[read_total : read_total + count]
                    try:
                        if self._native_read_into is not None:
                            try:
                                read_now = int(
                                    self._native_read_into(
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
            self.metrics.update(
                read_operations=1,
                requested_bytes=requested,
                read_bytes=read_total,
                read_ns=time.monotonic_ns() - started,
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
    ) -> str:
        """Fill a fixed record buffer and return its SHA-256 digest."""

        view = self._writable_bytes(destination)
        if len(view) != record.logical_bytes:
            raise ValueError(
                f"slot buffer has {len(view)} bytes; record needs {record.logical_bytes}"
            )
        self.metrics.update(record_requests=1)
        if prefer_sidecar and manifest.sidecar is not None:
            if record.sidecar_offset is None or record.sidecar_length is None:
                raise ExpertIOError("manifest sidecar record is incomplete")
            self.metrics.update(sidecar_record_requests=1)
            self._read_range_into(
                manifest.sidecar.file,
                record.sidecar_offset,
                view,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
            )
        else:
            self.metrics.update(source_record_requests=1)
            cursor = 0
            for segment in record.segments:
                end = cursor + segment.length
                self._read_range_into(
                    segment.shard,
                    segment.offset,
                    view[cursor:end],
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                )
                cursor = end
            if cursor != record.logical_bytes:
                raise ExpertIOShortRead("expert source segments did not fill the slot")
        digest = hashlib.sha256(view).hexdigest()
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
