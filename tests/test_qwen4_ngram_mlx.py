from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mtplx.models.qwen4_ngram_mlx import AffineQ4NGramRows


class _Lease:
    slot_ids = (0, 1)

    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class _Cache:
    manifest = SimpleNamespace(
        storage="affine-q4-g32",
        row_width=32,
        row_bytes=20,
    )

    def __init__(self) -> None:
        self.arena_object = mx.array(np.arange(40, dtype=np.uint8))
        self.lease = _Lease()

    def acquire_prevalidated_rows(self, _requested):
        return self.lease

    def acquire_prevalidated_rows_async(self, _requested):
        return SimpleNamespace(cancel=lambda: True)


def test_m2_rows_do_not_force_an_internal_mlx_eval(monkeypatch):
    cache = _Cache()
    rows = AffineQ4NGramRows(cache, row_width=32)
    packed = cache.arena_object.reshape((2, 20))
    expected = mx.dequantize(
        packed[:, :16].view(mx.uint32),
        packed[:, 16:18].view(mx.bfloat16),
        packed[:, 18:].view(mx.bfloat16),
        group_size=32,
        bits=4,
        mode="affine",
    ).reshape((1, 2, 1, 32))
    mx.eval(expected)
    expected_host = np.asarray(expected.astype(mx.float32)).copy()

    def reject_eval(*_values):
        raise AssertionError("M=2 row ownership must not drain the verifier graph")

    monkeypatch.setattr(mx, "eval", reject_eval)
    output = rows(mx.array([[[0], [1]]], dtype=mx.int64))

    assert output.shape == (1, 2, 1, 32)
    assert cache.lease.released is True
    rows._arena_host_rows[:] = 0
    monkeypatch.undo()
    mx.eval(output)
    np.testing.assert_array_equal(
        np.asarray(output.astype(mx.float32)), expected_host
    )


def test_prefetched_rows_overlap_io_then_release_the_prefetch_pin(monkeypatch):
    calls: list[tuple] = []

    class Future:
        def cancel(self):
            calls.append(("cancel",))
            return True

    cache = _Cache()
    future = Future()

    def acquire_async(requested):
        calls.append(("async", requested))
        return future

    def acquire(requested):
        calls.append(("sync", requested))
        return cache.lease

    cache.acquire_prevalidated_rows_async = acquire_async
    cache.acquire_prevalidated_rows = acquire
    rows = AffineQ4NGramRows(cache, row_width=32)

    prefetched = rows.prefetch_prevalidated_rows((0,))
    output = rows.materialize_prevalidated_rows(
        (0, 1), logical_shape=(1, 2, 1), prefetched=prefetched
    )

    assert output.shape == (1, 2, 1, 32)
    assert calls == [("async", (0,)), ("sync", (0, 1)), ("cancel",)]
    assert cache.lease.released is True
