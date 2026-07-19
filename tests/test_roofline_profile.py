"""Pure-logic tests for the roofline profiler's new probe handling (issue #51 T0).

The GPU-bound bench paths need the real model, but the bench-on-fill bookkeeping
and the dtype recorder are pure — and they encode two behaviors that must not
regress: a wave thunk that DECLINES its shape (returns None) is recorded as
ineligible rather than crashing the whole profile dump, and note_dtype records
each key exactly once (first-seen wins, so it captures the load-time dtype).
"""

from __future__ import annotations

import mlx.core as mx

import mtplx.roofline_profile as rp


def _reset() -> None:
    rp._SAMPLES.clear()
    rp._RESULTS.clear()
    rp._BYTES.clear()
    rp._ORDER.clear()
    rp._DTYPES.clear()


def test_ineligible_thunk_recorded_not_raised(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_ROOFLINE_PROFILE", "1")
    _reset()
    # A wave that declines its shape returns None every call.
    for _ in range(rp._MAX):
        rp._reg("moe_wave(32)", lambda: None, 1000)
    assert "moe_wave(32)" in rp._RESULTS
    secs, payload = rp._RESULTS["moe_wave(32)"]
    assert secs is None
    assert "ineligible" in payload


def test_real_thunk_benched_on_fill(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_ROOFLINE_PROFILE", "1")
    _reset()
    weight = mx.random.normal((256, 256))
    mx.eval(weight)
    for _ in range(rp._MAX):
        rp._reg("dense", lambda w=weight: w @ w, weight.nbytes)
    secs, nbytes = rp._RESULTS["dense"]
    assert isinstance(secs, float) and secs > 0
    assert nbytes == weight.nbytes


def test_note_dtype_first_seen_wins(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_ROOFLINE_PROFILE", "1")
    _reset()
    rp.note_dtype("k_norm", mx.zeros((4,), dtype=mx.bfloat16))
    rp.note_dtype("k_norm", mx.zeros((4,), dtype=mx.float32))  # must be ignored
    # dtype stringifies as "mlx.core.bfloat16"; first-seen (bf16) must win
    assert "bfloat16" in rp._DTYPES["k_norm"]
    assert "float32" not in rp._DTYPES["k_norm"]
    rp.note_dtype("missing", None)  # None is a no-op, must not raise
    assert "missing" not in rp._DTYPES


def test_disabled_is_noop(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_ROOFLINE_PROFILE", raising=False)
    _reset()
    rp.note_dtype("k", mx.zeros((2,), dtype=mx.bfloat16))
    assert rp._DTYPES == {}
