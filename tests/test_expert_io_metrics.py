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


class _RecordingRangeLedger:
    def __init__(self) -> None:
        self.started: list[tuple[int, str | None]] = []
        self.completed: list[int] = []

    def range_started(
        self,
        logical_bytes: int,
        *,
        phase: str | None = None,
    ) -> int:
        self.started.append((logical_bytes, phase))
        return len(self.started)

    def range_completed(self, token: int) -> None:
        self.completed.append(token)


class _ComponentDestination:
    def __init__(self, lengths: tuple[int, ...]) -> None:
        self.buffers = tuple(bytearray(length) for length in lengths)

    def record_views(self, _record: object) -> tuple[memoryview, ...]:
        return tuple(memoryview(buffer) for buffer in self.buffers)

    def payload(self) -> bytes:
        return b"".join(self.buffers)


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

    ledger = _RecordingRangeLedger()
    with PositionalExpertReader(
        tmp_path,
        use_native=False,
        pipeline_ledger=ledger,
    ) as reader:
        digest = reader.read_record_into(
            manifest,
            record,
            destination,
            prefer_sidecar=False,
            verify_hash=False,
            pipeline_phase="decode",
        )
        metrics = reader.metrics.as_dict()

    assert destination == payload
    assert digest == "unverified"
    assert metrics["record_requests"] == 1
    assert metrics["read_operations"] == len(lengths)
    assert metrics["python_preadv_invocations"] == len(lengths)
    assert metrics["preadv_bytes_returned"] == len(payload)
    assert ledger.started == [(length, "decode") for length in lengths]
    assert ledger.completed == [1, 2, 3]


def test_public_sidecar_single_and_component_paths_propagate_phase(
    tmp_path: Path,
) -> None:
    relative_name, payload = _write_source(tmp_path)
    record = SimpleNamespace(
        layer=1,
        expert=2,
        logical_bytes=len(payload),
        segments=(SimpleNamespace(length=3), SimpleNamespace(length=5)),
        sidecar_offset=0,
        sidecar_length=len(payload),
        sha256=None,
    )
    manifest = SimpleNamespace(sidecar=SimpleNamespace(file=relative_name))
    ledger = _RecordingRangeLedger()
    contiguous = bytearray(len(payload))
    component = _ComponentDestination((3, 5))

    with PositionalExpertReader(
        tmp_path,
        use_native=False,
        pipeline_ledger=ledger,
    ) as reader:
        reader.read_record_into(
            manifest,
            record,
            contiguous,
            verify_hash=False,
            pipeline_phase="decode",
        )
        reader.read_record_into(
            manifest,
            record,
            component,
            verify_hash=False,
            pipeline_phase="prefill",
        )

    assert contiguous == payload
    assert component.payload() == payload
    assert ledger.started == [
        (len(payload), "decode"),
        (len(payload), "prefill"),
    ]
    assert ledger.completed == [1, 2]


def test_public_grouped_component_scatter_propagates_phase(tmp_path: Path) -> None:
    relative_name, payload = _write_source(tmp_path)
    first = SimpleNamespace(
        layer=1,
        expert=1,
        logical_bytes=4,
        segments=(SimpleNamespace(length=2), SimpleNamespace(length=2)),
        sidecar_offset=0,
        sidecar_length=4,
        sha256=None,
    )
    second = SimpleNamespace(
        layer=1,
        expert=2,
        logical_bytes=4,
        segments=(SimpleNamespace(length=1), SimpleNamespace(length=3)),
        sidecar_offset=4,
        sidecar_length=4,
        sha256=None,
    )
    first_destination = _ComponentDestination((2, 2))
    second_destination = _ComponentDestination((1, 3))
    manifest = SimpleNamespace(sidecar=SimpleNamespace(file=relative_name))
    ledger = _RecordingRangeLedger()

    with PositionalExpertReader(
        tmp_path,
        use_native=False,
        pipeline_ledger=ledger,
    ) as reader:
        digests = reader.read_component_records_into(
            manifest,
            ((first, first_destination), (second, second_destination)),
            verify_hash=False,
            pipeline_phase="decode",
        )

    assert digests == ("unverified", "unverified")
    assert first_destination.payload() == payload[:4]
    assert second_destination.payload() == payload[4:]
    assert ledger.started == [(len(payload), "decode")]
    assert ledger.completed == [1]


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
            self.incomplete_phases: list[str] = []

        def range_started(self, _logical_bytes: int, *, phase: str) -> object:
            assert phase == "decode"
            self.started += 1
            return self.token

        def range_completed(self, token: object) -> None:
            assert token is self.token
            self.completed += 1
            raise RuntimeError("injected ledger failure")

        def mark_incomplete(self, *, phase: str) -> None:
            self.incomplete_phases.append(phase)

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
                pipeline_phase="decode",
            )
        metrics = reader.metrics.as_dict()
    finally:
        reader.close()

    assert metrics["read_operations"] == 1
    assert metrics["python_preadv_invocations"] == 1
    assert metrics["preadv_bytes_returned"] == 0
    assert ledger.started == 1
    assert ledger.completed == 1
    assert ledger.incomplete_phases == ["decode"]


def test_pipeline_ledger_completion_error_does_not_change_successful_read(
    tmp_path: Path,
) -> None:
    relative_name, payload = _write_source(tmp_path)

    class FailingLedger:
        def __init__(self) -> None:
            self.completed = 0
            self.incomplete_phases: list[str] = []

        def range_started(self, _logical_bytes: int, *, phase: str) -> object:
            assert phase == "prefill"
            return object()

        def range_completed(self, _token: object) -> None:
            self.completed += 1
            raise RuntimeError("injected ledger failure")

        def mark_incomplete(self, *, phase: str) -> None:
            self.incomplete_phases.append(phase)

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
            pipeline_phase="prefill",
        )

    assert destination == payload
    assert ledger.completed == 1
    assert ledger.incomplete_phases == ["prefill"]


def test_pipeline_ledger_start_error_does_not_change_successful_read(
    tmp_path: Path,
) -> None:
    relative_name, payload = _write_source(tmp_path)

    class FailingLedger:
        def __init__(self) -> None:
            self.completed = 0
            self.incomplete_phases: list[str] = []

        def range_started(self, _logical_bytes: int, *, phase: str) -> object:
            assert phase == "decode"
            raise RuntimeError("injected ledger start failure")

        def range_completed(self, _token: object) -> None:
            self.completed += 1

        def mark_incomplete(self, *, phase: str) -> None:
            self.incomplete_phases.append(phase)

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
            pipeline_phase="decode",
        )
        metrics = reader.metrics.as_dict()

    assert destination == payload
    assert metrics["read_operations"] == 1
    assert metrics["python_preadv_invocations"] == 1
    assert metrics["preadv_bytes_returned"] == len(payload)
    assert ledger.completed == 0
    assert ledger.incomplete_phases == ["decode"]


def test_telemetry_off_does_not_enter_pipeline_range_helpers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    relative_name, payload = _write_source(tmp_path)
    destination = bytearray(len(payload))

    with PositionalExpertReader(tmp_path, use_native=False) as reader:
        monkeypatch.setattr(
            reader,
            "_start_pipeline_range",
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("telemetry-off start helper entered")
            ),
        )
        monkeypatch.setattr(
            reader,
            "_finish_pipeline_range",
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("telemetry-off finish helper entered")
            ),
        )
        reader._read_range_into(
            relative_name,
            0,
            memoryview(destination),
            cancel_event=None,
            deadline_ns=None,
        )
        reader._readv_range_into(
            relative_name,
            0,
            (memoryview(bytearray(3)), memoryview(bytearray(5))),
            cancel_event=None,
            deadline_ns=None,
        )

    assert destination == payload


def test_per_call_phase_scopes_single_and_scatter_ranges(
    tmp_path: Path,
) -> None:
    relative_name, payload = _write_source(tmp_path)

    ledger = _RecordingRangeLedger()
    with PositionalExpertReader(
        tmp_path,
        use_native=False,
        pipeline_ledger=ledger,
    ) as reader:
        reader._read_range_into(
            relative_name,
            0,
            memoryview(bytearray(len(payload))),
            cancel_event=None,
            deadline_ns=None,
            pipeline_phase="decode",
        )
        reader._readv_range_into(
            relative_name,
            0,
            (memoryview(bytearray(3)), memoryview(bytearray(5))),
            cancel_event=None,
            deadline_ns=None,
            pipeline_phase="prefill",
        )

    assert ledger.started == [
        (len(payload), "decode"),
        (len(payload), "prefill"),
    ]
    assert ledger.completed == [1, 2]
