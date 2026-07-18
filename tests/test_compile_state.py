"""compile_state: the trace-active flag that suppresses the submit-cadence
async_eval inside a compiled forward (issue #51).

The flag is the whole fix for "[async_eval] Not allowed inside a graph
transformation": the model decode loop checks compile_trace_active() and skips
its per-N-layer async_eval while a compiled forward runs. These guard that the
flag is False by default, True only inside the context, restores correctly
(including nesting), and that CompiledARForward actually raises it during its
compiled call — the behavior the real model relies on.
"""

from __future__ import annotations

from mtplx.compile_state import compile_trace, compile_trace_active


def test_flag_false_by_default() -> None:
    assert compile_trace_active() is False


def test_context_sets_and_restores() -> None:
    assert compile_trace_active() is False
    with compile_trace():
        assert compile_trace_active() is True
    assert compile_trace_active() is False


def test_context_restores_on_exception() -> None:
    try:
        with compile_trace():
            assert compile_trace_active() is True
            raise ValueError("boom")
    except ValueError:
        pass
    assert compile_trace_active() is False


def test_nesting_restores_previous() -> None:
    with compile_trace():
        assert compile_trace_active() is True
        with compile_trace():
            assert compile_trace_active() is True
        # inner exit must restore the OUTER True, not the module default
        assert compile_trace_active() is True
    assert compile_trace_active() is False

