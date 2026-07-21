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


def test_source_record_coalesces_contiguous_segments_into_one_read(
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
        reads_per_record = reader.metrics.reads_per_record()

    assert destination == payload
    assert digest == "unverified"
    assert metrics["record_requests"] == 1
    # Three contiguous same-shard segments collapse into ONE positional read
    # covering the whole record (one range-reader invocation, one preadv,
    # one bracketed pipeline range).
    assert metrics["read_operations"] == 1
    assert metrics["python_preadv_invocations"] == 1
    assert metrics["preadv_bytes_returned"] == len(payload)
    assert metrics["read_bytes"] == len(payload)
    assert ledger.started == [(len(payload), "decode")]
    assert ledger.completed == [1]
    assert reads_per_record == 1.0


def test_source_record_keeps_per_segment_reads_across_offset_gaps(
    tmp_path: Path,
) -> None:
    relative_name = "gapped.bin"
    source = bytes((index % 251) + 1 for index in range(16))
    (tmp_path / relative_name).write_bytes(source)
    # Offsets 0, 6, 12 leave gaps between the three 2-byte segments, so each
    # segment is its own run and stays one positional read (the exact prior
    # per-segment fallback for genuinely non-contiguous records).
    offsets = (0, 6, 12)
    length = 2
    segments = tuple(
        SimpleNamespace(shard=relative_name, offset=offset, length=length)
        for offset in offsets
    )
    manifest = SimpleNamespace(sidecar=None)
    record = SimpleNamespace(
        logical_bytes=length * len(offsets),
        segments=segments,
        sha256=None,
    )
    destination = bytearray(length * len(offsets))

    ledger = _RecordingRangeLedger()
    with PositionalExpertReader(
        tmp_path,
        use_native=False,
        pipeline_ledger=ledger,
    ) as reader:
        reader.read_record_into(
            manifest,
            record,
            destination,
            prefer_sidecar=False,
            verify_hash=False,
            pipeline_phase="decode",
        )
        metrics = reader.metrics.as_dict()
        reads_per_record = reader.metrics.reads_per_record()

    expected = b"".join(source[o : o + length] for o in offsets)
    assert bytes(destination) == expected
    assert metrics["read_operations"] == len(offsets)
    assert metrics["python_preadv_invocations"] == len(offsets)
    assert metrics["read_bytes"] == length * len(offsets)
    assert ledger.started == [(length, "decode")] * len(offsets)
    assert reads_per_record == float(len(offsets))


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


# --------------------------------------------------------------------------
# Task A: contiguous-run coalescing of the source-path fallback.
# Task B: reads-per-record efficiency accounting.
# --------------------------------------------------------------------------


def _contiguous_source(
    directory: Path,
    *,
    length: int,
    count: int,
    stride: int,
    relative_name: str = "src.bin",
) -> tuple[SimpleNamespace, SimpleNamespace, bytes, bytearray]:
    """Build a same-shard source record with a fixed per-segment stride.

    ``stride == length`` packs the segments contiguously (one coalesced read);
    ``stride > length`` leaves gaps so each segment stays its own read.
    """

    directory.mkdir(parents=True, exist_ok=True)
    file_size = max(stride * count, length * count)
    source = bytes((index % 251) + 1 for index in range(file_size))
    (directory / relative_name).write_bytes(source)
    segments = tuple(
        SimpleNamespace(shard=relative_name, offset=index * stride, length=length)
        for index in range(count)
    )
    record = SimpleNamespace(
        logical_bytes=length * count,
        segments=segments,
        sha256=None,
    )
    manifest = SimpleNamespace(sidecar=None)
    expected = b"".join(
        source[index * stride : index * stride + length] for index in range(count)
    )
    return manifest, record, expected, bytearray(length * count)


def _syscalls(metrics: dict) -> int:
    return metrics["python_preadv_invocations"] + metrics["native_positional_calls"]


def test_contiguous_source_runs_groups_and_splits() -> None:
    runs = PositionalExpertReader._contiguous_source_runs

    def seg(shard: str, offset: int, length: int) -> SimpleNamespace:
        return SimpleNamespace(shard=shard, offset=offset, length=length)

    # Fully contiguous single shard -> one run.
    assert runs((seg("a", 0, 2), seg("a", 2, 3), seg("a", 5, 5))) == [(0, 3)]
    # An offset gap splits the run.
    assert runs((seg("a", 0, 2), seg("a", 4, 2))) == [(0, 1), (1, 1)]
    # A shard change splits even when the offsets would otherwise chain.
    assert runs((seg("a", 0, 2), seg("b", 2, 2))) == [(0, 1), (1, 1)]
    # Two contiguous, a gap, then two contiguous -> two runs.
    assert runs(
        (seg("a", 0, 2), seg("a", 2, 2), seg("a", 10, 2), seg("a", 12, 2))
    ) == [(0, 2), (2, 2)]
    # Degenerate inputs.
    assert runs((seg("a", 0, 4),)) == [(0, 1)]
    assert runs(()) == []


def test_reads_per_record_uses_syscall_counters_over_record_requests() -> None:
    metrics = expert_io.ExpertIOMetrics()
    # No records requested yet -> trivially zero (no divide-by-zero).
    assert metrics.reads_per_record() == 0.0
    metrics.update(
        record_requests=4,
        python_preadv_invocations=5,
        native_positional_calls=3,
        # read_operations is a logical range-reader counter, NOT a syscall
        # counter, so it must not enter the numerator.
        read_operations=99,
    )
    assert metrics.reads_per_record() == (5 + 3) / 4


def test_assert_read_efficiency_passes_under_budget_and_fails_over() -> None:
    metrics = expert_io.ExpertIOMetrics()
    metrics.update(
        record_requests=2,
        python_preadv_invocations=2,
        native_positional_calls=0,
    )
    # ratio == 1.0
    assert metrics.assert_read_efficiency(1.0) == 1.0
    assert metrics.assert_read_efficiency(1.5) == 1.0
    with pytest.raises(expert_io.ExpertIOError, match="reads-per-record 1.0000 exceeds"):
        metrics.assert_read_efficiency(0.5)
    # Empty metrics are trivially efficient.
    assert expert_io.ExpertIOMetrics().assert_read_efficiency(0.0) == 0.0


def test_flat_source_record_collapses_six_contiguous_segments_to_one_syscall(
    tmp_path: Path,
) -> None:
    length, count = 4, 6

    # "before" the fix: six non-contiguous segments stay one read each.
    manifest, record, expected, destination = _contiguous_source(
        tmp_path / "gapped", length=length, count=count, stride=length * 2
    )
    with PositionalExpertReader(tmp_path / "gapped", use_native=False) as reader:
        reader.read_record_into(
            manifest, record, destination,
            prefer_sidecar=False, verify_hash=False,
        )
        before = reader.metrics.as_dict()
        before_ratio = reader.metrics.reads_per_record()
    assert bytes(destination) == expected
    assert _syscalls(before) >= 6
    assert _syscalls(before) == count
    assert before["read_operations"] == count
    assert before_ratio == float(count)

    # "after" the fix: six contiguous segments coalesce to a single read.
    manifest, record, expected, destination = _contiguous_source(
        tmp_path / "packed", length=length, count=count, stride=length
    )
    with PositionalExpertReader(tmp_path / "packed", use_native=False) as reader:
        reader.read_record_into(
            manifest, record, destination,
            prefer_sidecar=False, verify_hash=False,
        )
        after = reader.metrics.as_dict()
        after_ratio = reader.metrics.reads_per_record()
    assert bytes(destination) == expected
    assert _syscalls(after) == 1
    assert after["read_operations"] == 1
    assert after["read_bytes"] == length * count
    assert after_ratio == 1.0


def test_component_source_record_collapses_six_contiguous_segments_to_one_scatter(
    tmp_path: Path,
) -> None:
    length, count = 4, 6

    # Contiguous component views -> one preadv scatter for the whole record.
    manifest, record, expected, _flat = _contiguous_source(
        tmp_path / "packed", length=length, count=count, stride=length
    )
    packed_dest = _ComponentDestination((length,) * count)
    with PositionalExpertReader(tmp_path / "packed", use_native=False) as reader:
        reader.read_record_into(
            manifest, record, packed_dest,
            prefer_sidecar=False, verify_hash=False,
        )
        packed = reader.metrics.as_dict()
        packed_ratio = reader.metrics.reads_per_record()
    assert packed_dest.payload() == expected
    assert packed["read_operations"] == 1
    assert _syscalls(packed) == 1
    assert packed_ratio == 1.0

    # Non-contiguous component views -> per-segment fallback preserved.
    manifest, record, expected, _flat = _contiguous_source(
        tmp_path / "gapped", length=length, count=count, stride=length * 2
    )
    gapped_dest = _ComponentDestination((length,) * count)
    with PositionalExpertReader(tmp_path / "gapped", use_native=False) as reader:
        reader.read_record_into(
            manifest, record, gapped_dest,
            prefer_sidecar=False, verify_hash=False,
        )
        gapped = reader.metrics.as_dict()
        gapped_ratio = reader.metrics.reads_per_record()
    assert gapped_dest.payload() == expected
    assert gapped["read_operations"] == count
    assert _syscalls(gapped) == count
    assert gapped_ratio == float(count)


def test_flat_source_record_coalesces_native_positional_calls(
    tmp_path: Path,
) -> None:
    length, count = 4, 6
    manifest, record, expected, destination = _contiguous_source(
        tmp_path / "packed", length=length, count=count, stride=length
    )
    source = bytes((tmp_path / "packed" / "src.bin").read_bytes())

    with PositionalExpertReader(tmp_path / "packed", use_native=False) as reader:

        def native_read(_fd: int, offset: int, target: memoryview) -> int:
            target[:] = source[offset : offset + len(target)]
            return len(target)

        reader._native_read_into = native_read
        reader.read_record_into(
            manifest, record, destination,
            prefer_sidecar=False, verify_hash=False,
        )
        metrics = reader.metrics.as_dict()
        ratio = reader.metrics.reads_per_record()

    assert bytes(destination) == expected
    # The coalesced contiguous flat run uses the native single-range reader
    # exactly once, not once per segment.
    assert metrics["native_positional_calls"] == 1
    assert metrics["python_preadv_invocations"] == 0
    assert ratio == 1.0
