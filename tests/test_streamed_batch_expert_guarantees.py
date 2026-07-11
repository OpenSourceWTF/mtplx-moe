"""Stage 5 guarantees: expert-union loading and prefill/decode isolation.

These tests pin the two batching-critical properties of the existing route
machinery rather than reimplementing them:

- ``LayerExpertSlotBank.plan`` deduplicates unique experts and
  ``partition_route_waves`` partitions unique experts, so one record selected
  by every live stream in a decode step is planned, read, and hashed once
  (``mtplx/expert_streaming.py`` and ``mtplx/expert_runtime.py``).
- Prefill-phase misses are serviced through transient slots and can claim
  only empty persistent slots via the prefill seed, so a joining stream's
  prefill can never evict a decode-hot persistent expert even while other
  streams keep decoding (``LayerExpertSlotBank.plan`` prefill branch).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from test_streamed_batch import _open_fixture_runtime, _request

from mtplx.streamed_batch import StreamedBatchRunner


def _force_expert(rt, expert: int) -> None:
    """Route every token of the fixture's sparse layer to one expert."""

    router = rt.model.model.layers[1].mlp.router
    bias = [0.0, 0.0]
    bias[expert] = 100.0
    router.expert_bias = mx.array(bias, dtype=mx.float32)


def test_record_selected_by_both_streams_loads_once_per_step(
    tmp_path: Path,
) -> None:
    rt, streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=64)
    try:
        # Prefill routes to expert 0 so that expert 1 is cold when the first
        # batched decode step selects it from both streams at once.
        _force_expert(rt, 0)
        boundaries: dict[int, dict] = {}

        def on_step(state) -> None:
            if state.step == 0:
                _force_expert(rt, 1)
            boundaries[state.step] = {
                "cache": dict(streaming.counters.as_dict()),
                "io": dict(streaming.snapshot()["slots"]["io"]),
            }

        runner = StreamedBatchRunner(rt, max_concurrency=2, on_step=on_step)
        runner.submit(_request("a", [1, 2, 3], max_tokens=3))
        runner.submit(_request("b", [7, 8, 9], max_tokens=3))
        runner.run()

        before = boundaries[0]
        after = boundaries[1]
        cache_delta = {
            key: after["cache"][key] - before["cache"][key]
            for key in ("route_calls", "expert_requests", "expert_misses",
                        "persistent_loads", "transient_loads", "bytes_read")
        }
        # One decode step, two streams, one shared cold expert: two routed
        # assignments collapse into a single wave, a single plan, and a
        # single record load.  (The decode admission policy may promote the
        # now doubly-hot expert into the persistent slot; which tier served
        # the load is policy, the single load is the union guarantee.)
        assert cache_delta["route_calls"] == 1
        assert cache_delta["expert_requests"] == 2
        assert cache_delta["expert_misses"] == 2
        assert cache_delta["persistent_loads"] + cache_delta["transient_loads"] == 1
        assert cache_delta["bytes_read"] == streaming.spec.expert_record_bytes
        # The physical read path agrees with the plan: one record request.
        assert (
            after["io"]["record_requests"] - before["io"]["record_requests"] == 1
        )
        assert (
            after["io"]["read_bytes"] - before["io"]["read_bytes"]
            >= streaming.spec.expert_record_bytes
        )
    finally:
        rt.close()


def test_joining_prefill_never_evicts_the_decode_hot_expert(
    tmp_path: Path,
) -> None:
    rt, streaming = _open_fixture_runtime(tmp_path, max_live_kv_tokens=64)
    try:
        # Stream "hot" decodes expert 0 long enough to make it decisively
        # decode-hot in the single persistent slot of the sparse layer.
        _force_expert(rt, 0)
        runner_box: list[StreamedBatchRunner] = []
        residency: dict[int, tuple[int, ...]] = {}
        counters: dict[int, dict] = {}
        joined: list[int] = []

        def on_step(state) -> None:
            residency[state.step] = tuple(streaming._banks[1].resident_experts)
            counters[state.step] = dict(streaming.counters.as_dict())
            if state.step == 8 and not joined:
                joined.append(state.step)
                # The joiner routes every prompt token to the cold expert 1.
                _force_expert(rt, 1)
                runner_box[0].submit(
                    _request("joiner", [7, 8, 9, 10], max_tokens=2)
                )

        runner = StreamedBatchRunner(rt, max_concurrency=2, on_step=on_step)
        runner_box.append(runner)
        runner.submit(_request("hot", [1, 2, 3], max_tokens=12))
        results = {result.request_id: result for result in runner.run()}

        # The joiner prefilled at the very next step boundary while "hot"
        # kept decoding, and both streams completed.
        assert results["joiner"].admitted_step == 9
        assert len(results["joiner"].tokens) == 2
        assert len(results["hot"].tokens) == 12

        # Expert 0 stayed resident through the joiner's prefill (and for the
        # whole run): the prefill was serviced through transient slots.
        assert residency[9] == (0,)
        assert all(resident == (0,) for resident in residency.values())
        final = streaming.counters.as_dict()
        assert final["evictions"] == 0
        # Exactly one persistent load ever happened: expert 0's prefill seed.
        assert final["persistent_loads"] == 1
        # The joiner's prefill produced transient-slot service, not
        # persistent admission.
        assert (
            counters[9]["transient_loads"] > counters[8]["transient_loads"]
        )
        assert (
            counters[9]["persistent_loads"] == counters[8]["persistent_loads"]
        )
    finally:
        rt.close()
