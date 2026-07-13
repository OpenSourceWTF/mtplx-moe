from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx import expert_io
from mtplx.expert_io import ExpertIOShortRead, PositionalExpertReader


def _write_source(tmp_path: Path, payload: bytes = b"abcdefgh") -> tuple[str, bytes]:
    relative_name = "source.bin"
    (tmp_path / relative_name).write_bytes(payload)
    return relative_name, payload


def test_partial_preadv_separates_logical_range_reader_invocations_and_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    relative_name, payload = _write_source(tmp_path)
    returns = iter((3, 5))

    def partial_preadv(_fd: int, buffers: list[memoryview], offset: int) -> int:
        returned = next(returns)
        buffers[0][:returned] = payload[offset : offset + returned]
        return returned

    monkeypatch.setattr(expert_io.os, "preadv", partial_preadv)
    destination = bytearray(len(payload))

    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        reader._read_range_into(
            relative_name,
            0,
            memoryview(destination),
            cancel_event=None,
            deadline_ns=None,
        )
        metrics = reader.metrics.as_dict()

    assert destination == payload
    assert metrics["read_operations"] == 1
    assert metrics["python_preadv_invocations"] == 2
    assert metrics["preadv_bytes_returned"] == len(payload)
    assert metrics["native_positional_calls"] == 0
    assert metrics["native_bytes_returned"] == 0


def test_interrupted_preadv_counts_attempt_but_not_returned_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    relative_name, payload = _write_source(tmp_path)
    attempts = 0

    def interrupted_once(_fd: int, buffers: list[memoryview], offset: int) -> int:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InterruptedError
        buffers[0][:] = payload[offset : offset + len(buffers[0])]
        return len(buffers[0])

    monkeypatch.setattr(expert_io.os, "preadv", interrupted_once)
    destination = bytearray(len(payload))

    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        reader._read_range_into(
            relative_name,
            0,
            memoryview(destination),
            cancel_event=None,
            deadline_ns=None,
        )
        metrics = reader.metrics.as_dict()

    assert destination == payload
    assert metrics["read_operations"] == 1
    assert metrics["python_preadv_invocations"] == 2
    assert metrics["preadv_bytes_returned"] == len(payload)


def test_native_read_counts_only_native_invocations_and_bytes(tmp_path: Path) -> None:
    relative_name, payload = _write_source(tmp_path)
    destination = bytearray(len(payload))

    with PositionalExpertReader(tmp_path, use_native=False) as reader:

        def native_read(_fd: int, offset: int, target: memoryview) -> int:
            target[:] = payload[offset : offset + len(target)]
            return len(target)

        reader._native_read_into = native_read
        reader._read_range_into(
            relative_name,
            0,
            memoryview(destination),
            cancel_event=None,
            deadline_ns=None,
        )
        metrics = reader.metrics.as_dict()

    assert destination == payload
    assert metrics["read_operations"] == 1
    assert metrics["native_positional_calls"] == 1
    assert metrics["native_bytes_returned"] == len(payload)
    assert metrics["python_preadv_invocations"] == 0
    assert metrics["preadv_bytes_returned"] == 0


def test_source_record_counts_each_segment_as_one_logical_range_reader_invocation(
    tmp_path: Path,
) -> None:
    relative_name, payload = _write_source(tmp_path)
    lengths = (2, 3, 3)
    offset = 0
    segments = []
    for length in lengths:
        segments.append(
            SimpleNamespace(shard=relative_name, offset=offset, length=length)
        )
        offset += length
    manifest = SimpleNamespace(sidecar=None)
    record = SimpleNamespace(
        logical_bytes=len(payload),
        segments=tuple(segments),
        sha256=None,
    )
    destination = bytearray(len(payload))

    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        digest = reader.read_record_into(
            manifest,
            record,
            destination,
            prefer_sidecar=False,
            verify_hash=False,
        )
        metrics = reader.metrics.as_dict()

    assert destination == payload
    assert digest == "unverified"
    assert metrics["record_requests"] == 1
    assert metrics["read_operations"] == len(lengths)
    assert metrics["python_preadv_invocations"] == len(lengths)
    assert metrics["preadv_bytes_returned"] == len(payload)


def test_partial_scatter_preadv_counts_each_python_invocation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    relative_name, payload = _write_source(tmp_path)
    returns = iter((4, 4))

    def partial_preadv(_fd: int, _buffers: list[memoryview], _offset: int) -> int:
        return next(returns)

    monkeypatch.setattr(expert_io.os, "preadv", partial_preadv)

    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        reader._readv_range_into(
            relative_name,
            0,
            (memoryview(bytearray(3)), memoryview(bytearray(5))),
            cancel_event=None,
            deadline_ns=None,
        )
        metrics = reader.metrics.as_dict()

    assert metrics["read_operations"] == 1
    assert metrics["python_preadv_invocations"] == 2
    assert metrics["preadv_bytes_returned"] == len(payload)
    assert metrics["native_positional_calls"] == 0
    assert metrics["native_bytes_returned"] == 0


def test_optional_pipeline_ledger_brackets_a_logical_range_reader_invocation(
    tmp_path: Path,
) -> None:
    relative_name, payload = _write_source(tmp_path)

    class RecordingLedger:
        def __init__(self) -> None:
            self.events: list[tuple[str, int]] = []
            self.token = object()

        def range_started(self, logical_bytes: int) -> object:
            self.events.append(("started", logical_bytes))
            return self.token

        def range_completed(self, token: object) -> None:
            assert token is self.token
            self.events.append(("completed", len(payload)))

    ledger = RecordingLedger()
    destination = bytearray(len(payload))

    with PositionalExpertReader(
        tmp_path,
        use_native=False,
        pipeline_ledger=ledger,
    ) as reader:
        reader._read_range_into(
            relative_name,
            0,
            memoryview(destination),
            cancel_event=None,
            deadline_ns=None,
        )

    assert destination == payload
    assert ledger.events == [
        ("started", len(payload)),
        ("completed", len(payload)),
    ]


def test_pipeline_ledger_completion_error_does_not_mask_read_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    relative_name, payload = _write_source(tmp_path)

    class FailingLedger:
        def __init__(self) -> None:
            self.started = 0
            self.completed = 0
            self.token = object()

        def range_started(self, _logical_bytes: int) -> object:
            self.started += 1
            return self.token

        def range_completed(self, token: object) -> None:
            assert token is self.token
            self.completed += 1
            raise RuntimeError("injected ledger failure")

    monkeypatch.setattr(expert_io.os, "preadv", lambda *_args: 0)
    ledger = FailingLedger()
    reader = PositionalExpertReader(
        tmp_path,
        use_native=False,
        pipeline_ledger=ledger,
    )
    try:
        with pytest.raises(ExpertIOShortRead):
            reader._read_range_into(
                relative_name,
                0,
                memoryview(bytearray(len(payload))),
                cancel_event=None,
                deadline_ns=None,
            )
        metrics = reader.metrics.as_dict()
    finally:
        reader.close()

    assert metrics["read_operations"] == 1
    assert metrics["python_preadv_invocations"] == 1
    assert metrics["preadv_bytes_returned"] == 0
    assert ledger.started == 1
    assert ledger.completed == 1


def test_pipeline_ledger_completion_error_does_not_change_successful_read(
    tmp_path: Path,
) -> None:
    relative_name, payload = _write_source(tmp_path)

    class FailingLedger:
        def __init__(self) -> None:
            self.completed = 0

        def range_started(self, _logical_bytes: int) -> object:
            return object()

        def range_completed(self, _token: object) -> None:
            self.completed += 1
            raise RuntimeError("injected ledger failure")

    ledger = FailingLedger()
    destination = bytearray(len(payload))
    with PositionalExpertReader(
        tmp_path,
        use_native=False,
        pipeline_ledger=ledger,
    ) as reader:
        reader._read_range_into(
            relative_name,
            0,
            memoryview(destination),
            cancel_event=None,
            deadline_ns=None,
        )

    assert destination == payload
    assert ledger.completed == 1


def test_pipeline_ledger_start_error_does_not_change_successful_read(
    tmp_path: Path,
) -> None:
    relative_name, payload = _write_source(tmp_path)

    class FailingLedger:
        def __init__(self) -> None:
            self.completed = 0

        def range_started(self, _logical_bytes: int) -> object:
            raise RuntimeError("injected ledger start failure")

        def range_completed(self, _token: object) -> None:
            self.completed += 1

    ledger = FailingLedger()
    destination = bytearray(len(payload))
    with PositionalExpertReader(
        tmp_path,
        use_native=False,
        pipeline_ledger=ledger,
    ) as reader:
        reader._read_range_into(
            relative_name,
            0,
            memoryview(destination),
            cancel_event=None,
            deadline_ns=None,
        )
        metrics = reader.metrics.as_dict()

    assert destination == payload
    assert metrics["read_operations"] == 1
    assert metrics["python_preadv_invocations"] == 1
    assert metrics["preadv_bytes_returned"] == len(payload)
    assert ledger.completed == 0
