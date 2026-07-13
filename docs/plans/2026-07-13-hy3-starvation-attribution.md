# Hy3 Starvation Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add behavior-neutral, telemetry-only Phase 1 evidence that distinguishes dependency underfill, unissued authoritative work, reader service, and generation-thread expert-input wait.

**Architecture:** A diagnostic-only `ExpertPipelineLedger` integrates record, reader-task, logical-range, runnable-work, and generation-wait state changes on one monotonic clock. Split routes own bounded record lifecycle handles so cancellation cannot leak occupancy. The existing resource sampler differences cumulative integrals and publishes a versioned, honestly labeled report; scheduling, cache, MLX ownership, and telemetry-off execution remain unchanged.

**Tech Stack:** Python 3.11+, `threading`, `time.monotonic_ns`, `time.thread_time_ns`, `concurrent.futures`, pytest, Ruff, existing MTPLX resource telemetry.

**Assumptions:**
- Assumes Phase 1 runs on frozen #29 and observes the existing whole-record decode path — it will NOT schedule stripes or future-layer work.
- Assumes `RoutePlan.loads` is the authoritative unique-record eligibility set — assignment duplicates are NOT separate record jobs.
- Assumes diagnostic telemetry may synchronize its own counters — it will NOT add a lock or clock read to the telemetry-off path.
- Assumes host dispatch is the observable consumption boundary — it will NOT claim physical GPU consumption or GPU idle time.
- Assumes Python `os.preadv` is visible only in the Python reader — native reads will NOT be relabeled as Python `preadv` calls.

---

## File Structure

- `mtplx/resource_metrics.py` — pipeline ledger, route handle, state integrals, bounded latency histograms, and snapshots.
- `mtplx/expert_io.py` — logical-range lifecycle plus exact Python/native call and returned-byte metrics.
- `mtplx/expert_slots.py` — record blocking, accepted submission, reader-task, ready/failure lifecycle hooks.
- `mtplx/expert_runtime.py` — create the ledger, begin/close split-route attribution, and measure blocking next-miss waits.
- `mtplx/models/expert_mlx.py` — host-dispatch markers for hit, shared, and completed-miss work.
- `mtplx/benchmarks/resource_telemetry.py` — same-clock differencing, duration-weighted summaries, coverage, and schema v2.
- `tests/test_resource_metrics.py` — deterministic fake-clock ledger tests.
- `tests/test_expert_io_metrics.py` — range/syscall/native-reader identity tests.
- `tests/test_expert_slots_runtime.py` — end-to-end record lifecycle and telemetry-off regressions.
- `tests/test_streamed_models.py` — ordered hit/shared/miss dispatch and wait-boundary tests.
- `tests/test_resource_telemetry.py` — report identities, overlaps, percentiles, and unavailable coverage.
- `tests/test_benchmark_streamed_generation_concurrency_cli.py` — schema contract update.
- `docs/RESOURCE_TELEMETRY.md` — interpretation and claim boundaries.
- `project-map.md` — Phase 1 attribution entrypoint and no-device-claim constraint.

### Task 1: Implement deterministic pipeline-ledger primitives

**Files:**
- Modify: `mtplx/resource_metrics.py`
- Modify: `tests/test_resource_metrics.py`

**Security flag:** `none`

**Does NOT cover:** This task has no runtime hooks and cannot observe I/O, MLX, credits, or device state.

- [x] **Step 1: Write failing lifecycle and orthogonal-state tests**

Add tests using the existing `FakeClock` pattern:

```python
def test_pipeline_ledger_integrates_record_lifecycle_and_overlap() -> None:
    clock = FakeClock()
    ledger = ExpertPipelineLedger(strict=True, clock_ns=clock)
    route = ledger.begin_route(
        layer=1,
        phase="decode",
        load_experts=(5,),
        load_logical_bytes=(100,),
    )
    hit_work = ledger.begin_hit_work((2,), phase="decode")
    clock.advance(2)
    route.observe_block(5, "pin_held", elapsed_ns=5)
    clock.advance(8)
    route.submission_attempted((5,))
    route.submission_accepted((5,))
    route.reader_started((5,))
    hit_work.claim()
    route.begin_generation_wait()
    range_token = ledger.range_started(logical_bytes=100, phase="decode")
    clock.advance(10)
    ledger.range_completed(range_token)
    route.record_verified(5)
    route.reader_completed((5,), thread_cpu_ns=4)
    route.record_runnable(5)
    route.end_generation_wait()
    clock.advance(5)
    route.claim_misses((5,))
    route.close()

    snapshot = ledger.snapshot()
    assert snapshot["counters"]["logical_record_jobs"] == 1
    assert snapshot["counters"]["logical_record_bytes"] == 100
    assert snapshot["counters"]["accepted_record_jobs"] == 1
    assert snapshot["counters"]["accepted_record_bytes"] == 100
    assert snapshot["counters"]["verified_record_jobs"] == 1
    assert snapshot["counters"]["runnable_record_jobs"] == 1
    assert snapshot["counters"]["claimed_record_jobs"] == 1
    assert snapshot["integrals_ns"]["eligible_unsubmitted_record_ns"] == 10
    assert snapshot["block_ns"]["pin_held"] == 5
    assert snapshot["integrals_ns"]["generation_expert_input_wait_ns"] == 10
    assert snapshot["integrals_ns"]["generation_wait_storage_active_ns"] == 10
    assert snapshot["integrals_ns"]["runnable_miss_unclaimed_record_ns"] == 5
```

Add focused tests for the two actually observable block reasons, `pin_held` and
`slot_loading`; record one fake-clock nanosecond under each and assert that only
the selected reason's count and duration change. Operation credit, byte credit,
authoritative reserve, and a distinct slot-capacity admission stage do not exist
in #29 and remain coverage=`unavailable`, not zero-valued measured counters.
At runtime these observations are published as elapsed durations after the
corresponding slot condition is released. Add separate tests for exact
record/byte counters, worker-start-before-submit-return, rejection rollback and
incomplete coverage, satisfied-without-new-submit continuing through runnable
and claimed, route close abandoning every nonterminal record, active range
bytes and phase isolation, six-state primary precedence, bounded histograms,
decode/prefill/unscoped snapshots, and fail-open non-strict transitions. Strict
unit-test mode raises on duplicate/out-of-order transitions; runtime mode never
changes a data-path outcome.

- [x] **Step 2: Run tests and verify RED**

Run: `uv run --frozen --extra dev python -m pytest -q tests/test_resource_metrics.py`

Expected: FAIL because `ExpertPipelineLedger` and its route lifecycle do not exist.

- [x] **Step 3: Implement the ledger API**

Add two public classes. `ExpertPipelineLedger` accepts strict/non-strict mode and
an injectable monotonic nanosecond clock; it exposes load-only `begin_route`,
phase-scoped `range_started(logical_bytes)`, `range_completed(span_token)`,
`begin_hit_work`, `begin_shared_work`, and `snapshot`.
The start method returns an opaque integer span token so concurrent equal-sized
ranges cannot be confused. `ExpertPipelineRoute`
exposes elapsed block observation, satisfied-without-new-submit, provisional
submission attempt plus acceptance/rejection, reader start/complete/failure,
generation-wait begin/end, separate verified and route-part-runnable record
transitions, miss claim, and idempotent close. Exact bytes are registered with
each load expert and derived for every later transition. Hit/shared spans are
ledger-level and never owned by the route. Every valid state mutation calls
`_accrue(now_ns)` under one ledger-local diagnostic lock.

Use fixed nanosecond histogram buckets ending in an overflow bucket. Export
only cumulative bucket counts, operation/byte counters, gauges, integrals,
six-state primary integrals, block counts/time, invariant failures, coverage,
and global plus phase-scoped snapshots; do not retain a raw event list or a
fixed target-depth rule.

- [x] **Step 4: Run focused tests and verify GREEN**

Run: `uv run --frozen --extra dev python -m pytest -q tests/test_resource_metrics.py`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add mtplx/resource_metrics.py tests/test_resource_metrics.py
git commit -m "feat(bench): add expert pipeline attribution ledger"
```

### Task 2: Separate logical ranges from Python and native reader calls

**Files:**
- Modify: `mtplx/expert_io.py`
- Create: `tests/test_expert_io_metrics.py`

**Security flag:** `none`

**Does NOT cover:** Native calls do not establish Python `preadv` coverage, and neither backend establishes physical device operations or queue depth.

- [x] **Step 1: Write failing partial-read and native-coverage tests**

```python
def test_partial_preadv_counts_range_calls_and_returned_bytes(tmp_path, monkeypatch):
    reader, relative_name = open_reader_fixture(tmp_path, payload=b"abcdefgh")
    destination = memoryview(bytearray(8))
    returns = iter((3, 5))
    monkeypatch.setattr(os, "preadv", lambda *_args: next(returns))
    reader._read_range_into(relative_name, 0, destination,
                            cancel_event=None, deadline_ns=None)
    metrics = reader.metrics.as_dict()
    assert metrics["read_operations"] == 1
    assert metrics["python_preadv_invocations"] == 2
    assert metrics["preadv_bytes_returned"] == 8

```

Add three more tests with exact assertions: an interrupted call followed by an
8-byte success yields two `python_preadv_invocations` and eight returned bytes; one
native 8-byte call yields one `native_positional_calls`, eight native bytes,
and zero Python `preadv` calls; and a source record with N manifest segments
yields one record request, N logical range jobs, and N Python calls when each
range completes in one invocation. Add exact-once range start/completion tests
for successful and failing reads, plus start- and completion-hook failures that
prove diagnostics cannot alter the original read result or metrics.

- [x] **Step 2: Run tests and verify RED**

Run: `uv run --frozen --extra dev python -m pytest -q tests/test_expert_io_metrics.py`

Expected: FAIL because the explicit syscall/native fields and range-latency histogram do not exist.

- [x] **Step 3: Add exact local call accounting**

Extend `ExpertIOMetrics` with:

```python
python_preadv_invocations: int = 0
preadv_bytes_returned: int = 0
native_positional_calls: int = 0
native_bytes_returned: int = 0
```

In `_read_range_into` and `_readv_range_into`, increment local attempt counters
immediately before each backend invocation and local returned bytes only after a
positive return. Publish all values in the existing final metrics update so no
new metrics lock is taken per partial read. If a ledger is present, bracket the
logical range with `range_started`/`range_completed`; isolate hook exceptions
from the data path and leave the default reader path unchanged when it is
`None`. Task 3 will supply the route's phase-scoped range tracker per record.

- [x] **Step 4: Run focused and existing reader tests**

Run: `uv run --frozen --extra dev python -m pytest -q tests/test_expert_io_metrics.py tests/test_expert_slots_runtime.py -k 'positional_reader or source_record or scatter'`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add mtplx/expert_io.py tests/test_expert_io_metrics.py
git commit -m "feat(bench): distinguish read ranges from syscalls"
```

### Task 3: Wire authoritative record lifecycle through split routes

**Files:**
- Modify: `mtplx/expert_io.py`
- Modify: `mtplx/expert_slots.py`
- Modify: `mtplx/expert_runtime.py`
- Modify: `tests/test_expert_io_metrics.py`
- Modify: `tests/test_expert_slots_runtime.py`

**Security flag:** `none`

**Does NOT cover:** Eligibility is current-route authoritative demand only; no future layer, speculation, credit admission, reserve, or scheduling policy is introduced.

- [x] **Step 1: Write failing controlled-runtime tests**

Add this controlled-runtime test, then add separate pin/loading, failure, and
telemetry-off cases using the same `_open_tiny_runtime` fixture:

```python
def test_split_route_counts_unique_loads_not_assignment_duplicates(tmp_path) -> None:
    runtime = _open_tiny_runtime(tmp_path, resource_telemetry=True)
    try:
        with runtime.begin_split_route(1, [0, 0], phase="decode") as pending:
            ready = pending.finish_misses()
            assert ready is not None
            pending.release_misses(ready)
        pipeline = runtime.resource_telemetry_snapshot(
            mx_module=object()
        )["expert_pipeline"]
        assert pipeline["counters"]["logical_record_jobs"] == 1
        assert pipeline["counters"]["accepted_record_jobs"] == 1
    finally:
        runtime.close()
```

Add a controlled-submit test whose worker fully finishes before `submit`
returns; acceptance must remain legal after verified/completed publication.
The pin/loading test must hold a pinned ready victim and a separately loading
generation long enough for each waiter to enter its condition, then assert both
durations are positive and neither count aliases the other. The failure test
injects a reader exception and asserts ready/claimed remain zero while
abandoned increments. Add a close-before-future-settles regression,
successful-path and original-error-not-masked instrumentation regressions, and
unclaimed hit/shared exception cleanup. The telemetry-off case asserts the
ledger attribute is `None`, reader range helpers are never entered, and
`expert_pipeline` is absent from an ordinary snapshot. Add prefill/decode
single-record and component-scatter reads to prove range operations and bytes
remain in their per-call phase.

- [x] **Step 2: Run tests and verify RED**

Run: `uv run --frozen --extra dev python -m pytest -q tests/test_expert_slots_runtime.py -k 'pipeline or split_route_counts or pin_and_loading'`

Expected: FAIL because the runtime has no route handle or `expert_pipeline` snapshot.

- [x] **Step 3: Wire the ledger without changing scheduling**

Construct one non-strict ledger in `ExpertStreamingRuntime.open` only for
`resource_telemetry=True`, pass it to the reader and slot pool, and expose the
same object from the runtime. Start a load-only route handle after miss parts
are formed, using unique `miss_plan.loads` and exact manifest bytes. Pass it as
an optional keyword through `ensure_route_part`, `_ensure_route_locked`, and
`_ensure_route_owned`. Thread the route phase per call through
`read_record_into`, `read_component_records_into`, `_read_range_into`, and
`_readv_range_into`; direct reader calls remain `unscoped`. Guard the
telemetry-off reader branch before any diagnostic helper call.

Measure elapsed `slot_loading` and `pin_held` time only around actual
`Condition.wait` calls, but publish the observation after releasing the slot
condition so slot and ledger locks are never nested. Publish a provisional
submission attempt before `executor.submit`, record acceptance only after a
`Future` is returned, allow `_run_tracked` to start from provisional state, and
roll the provisional state back on rejection. Preserve the existing
`RouteIOAdmission` rollback meaning unchanged. A reader-specific wrapper calls
reader-start before `_fill`/`_fill_batch`, captures thread CPU, calls
reader-complete on success or reader-failed before bare-raising the original
exception, and supports completion before late submit acceptance. Mark verified
complete-record readiness only after digest validation and READY publication,
outside the slot condition. Mark a miss host-runnable separately after waits,
pins, bindings, task completion, and `ReadyRoute` construction. Mark a
deduplicated non-owner satisfied only after `_wait_ready` confirms READY.
Hit/shared runnable spans remain ledger-level and
phase-scoped so the all-hit fast path is covered without changing `ReadyRoute`
ownership. Close a locally owned handle in every `begin_split_route` failure
before ownership transfer. After transfer, do not treat
`PendingSplitRoute.close()` as terminal while callbacks can still settle;
finalize the handle from `PendingSplitRoute._finalize_if_ready`, after releasing
its state lock and only when all callbacks have drained. Every inline hook must
preserve the original successful or
failing data-path result and mark attribution incomplete on instrumentation
failure.

- [x] **Step 4: Run focused and slot/runtime suites**

Run: `uv run --frozen --extra dev python -m pytest -q tests/test_resource_metrics.py tests/test_expert_io_metrics.py tests/test_expert_slots_runtime.py`

Expected: PASS with identical slot generations, pins, policy rollback, and output buffers.

- [x] **Step 5: Commit**

```bash
git add mtplx/expert_io.py mtplx/expert_slots.py mtplx/expert_runtime.py tests/test_expert_io_metrics.py tests/test_expert_slots_runtime.py
git commit -m "feat(bench): trace authoritative expert record lifecycle"
```

### Task 4: Measure runnable work and only the blocking next-miss wait

**Files:**
- Modify: `mtplx/expert_runtime.py`
- Modify: `mtplx/models/expert_mlx.py`
- Modify: `tests/test_streamed_models.py`
- Modify: `tests/test_expert_slots_runtime.py`

**Security flag:** `none`

**Does NOT cover:** `mx.eval`, Q4 compute, policy publication, and release time are not expert-input wait; host-known runnable work is not a GPU-runnable claim.

- [x] **Step 1: Write failing ordered-work tests**

Use controlled futures and the existing fake-MLX ordered-overlap fixture. The
blocking case releases a miss future only after the iterator has entered its
wait and asserts positive expert-input wait. The already-completed case resolves
the future before iteration and asserts zero wait. A two-future case completes
futures after two distinct blocking spans and asserts both waits are measured;
a buffered-first/late-second case asserts only the second span is measured. The
ordered model case
records callbacks and asserts `claim_hits`, `claim_shared`, and `claim_misses`
occur immediately before their corresponding evaluation callbacks. Open one
ledger-level shared span after decode phase is known so the all-hit fast path
and split path use the same shared-work lifecycle. The all-hit case asserts its
hit is claimed, its deferred shared callback is claimed, and wait events and
nanoseconds remain zero. Advance the fake clock
during hit Q4, shared evaluation, miss Q4, policy commit, and release, and
assert none of those advances appears in the expert-input-wait counter.

- [x] **Step 2: Run tests and verify RED**

Run: `uv run --frozen --extra dev python -m pytest -q tests/test_streamed_models.py tests/test_expert_slots_runtime.py -k 'expert_input_wait or runnable or already_completed'`

Expected: FAIL because runnable claims and the blocking iterator boundary are not instrumented.

- [x] **Step 3: Add host-dispatch and wait boundaries**

Replace the implicit `for future in as_completed(snapshot)` wait with an
explicit iterator step and an explicit `remaining` future set. Remove every
yielded future before the next readiness check. Call `begin_generation_wait()`
only when no future in `remaining` is already done; bracket only
`next(completions)` and always end the span in `finally`. Mark hits, shared
work, and a completed miss claimed immediately before their existing
evaluation dispatch calls. Do not move or add any MLX operation, future,
condition, or release.

- [x] **Step 4: Run focused suites**

Run: `uv run --frozen --extra dev python -m pytest -q tests/test_streamed_models.py tests/test_expert_slots_runtime.py`

Expected: PASS with existing ordered overlap behavior unchanged.

- [x] **Step 5: Commit**

```bash
git add mtplx/expert_runtime.py mtplx/models/expert_mlx.py tests/test_streamed_models.py tests/test_expert_slots_runtime.py
git commit -m "feat(bench): measure expert input wait boundaries"
```

### Task 5: Publish honest same-clock attribution in resource telemetry v2

**Files:**
- Modify: `mtplx/benchmarks/resource_telemetry.py`
- Modify: `tests/test_resource_telemetry.py`
- Modify: `tests/test_benchmark_streamed_generation_concurrency_cli.py`
- Modify: `docs/RESOURCE_TELEMETRY.md`
- Modify: `docs/specs/2026-07-13-hy3-starvation-attribution-design.md`
- Modify: `docs/plans/2026-07-13-hy3-starvation-attribution.md`
- Modify: `project-map.md`

**Security flag:** `none`

**Does NOT cover:** The report cannot promote an optimization, infer physical device QD, or call host expert-input wait GPU wait.

- [x] **Step 1: Write failing interval and report-contract tests**

```python
def test_pipeline_summary_weights_duration_not_sample_count():
    intervals = [pipeline_interval(seconds=1, wait_seconds=1),
                 pipeline_interval(seconds=9, wait_seconds=0)]
    report = summarize_intervals(intervals, ssd_ceiling_gib_s=12.47,
                                 powermetrics=None)
    assert report["expert_pipeline"]["generation_expert_input_wait_fraction"] == 0.1

```

Add exact report tests that preserve separate wait/storage, wait/reader-task,
wait/submitted, wait/eligible, and wait/runnable durations and fractions against
the named decode-observation denominator. Assert unequal values survive for
accepted executor calls, accepted record jobs/bytes, started/completed/failed
reader tasks, decode ranges, and all-phase sampler-window Python/native backend
calls and returned bytes. Assert credit, reserve, slot-admission, device-QD, and
GPU-wait/idle coverage all equal `"unavailable"`. Prove missing backend counters
become `null` with unavailable backend coverage, backend counter reset has its
own incomplete status, and decode coverage remains independent. Feed histogram
bucket deltas whose 50th and 95th samples land in known buckets, then assert the
reported percentile upper bounds equal those bucket boundaries; separately
prove an overflow censors only the percentile that lands there and a missing
required histogram makes decode incomplete. Prove primary nanoseconds are
preserved but decode coverage becomes incomplete when their sum differs from
`decode_observation_ns`.

Update the CLI schema assertion to expect `mtplx-resource-telemetry-v2`.

- [x] **Step 2: Run tests and verify RED**

Run: `uv run --frozen --extra dev python -m pytest -q tests/test_resource_telemetry.py tests/test_benchmark_streamed_generation_concurrency_cli.py`

Expected: FAIL because the sampler does not difference or summarize the pipeline block and still emits schema v1.

- [x] **Step 3: Implement v2 differencing and documentation**

Add `_pipeline_interval(before, after, span_ns)` for cumulative counter,
integral, block, and histogram deltas. Duration-weight all fractions. Emit:

```python
{
    "schema": "mtplx-resource-telemetry-v2",
    "expert_pipeline": {
        "schema": "mtplx-expert-pipeline-summary-v1",
        "source_schema": "mtplx-expert-pipeline-attribution-v1",
        "scope": "decode",
        "logical_record_jobs": int,
        "logical_record_bytes": int,
        "accepted_executor_submissions": int,
        "submission_accepted_record_jobs": int,
        "submission_accepted_record_bytes": int,
        "reader_tasks_started": int,
        "reader_tasks_completed": int,
        "reader_tasks_failed": int,
        "decode_logical_ranges_started": int,
        "decode_observation_ns": int,
        "decode_counters": dict[str, int],
        "decode_integrals_ns": dict[str, int],
        "primary_state_ns": dict[str, int],
        "generation_expert_input_wait_fraction": float | None,
        "orthogonal_overlap": {
            "denominator": "decode_observation_ns",
            "duration_ns": dict[str, int],
            "fraction_of_decode_observation": dict[str, float | None],
        },
        "primary_state_fraction": {
            "generation_thread_expert_input_wait": float | None,
            "logical_range_active": float | None,
            "reader_completion_active": float | None,
            "submitted_queued": float | None,
            "eligible_unsubmitted": float | None,
            "host_runnable_work": float | None,
            "route_publication_pending": float | None,
            "no_known_useful_work": float | None,
        },
        "block_reasons": dict[str, object],
        "latency_histograms": {
            "logical_range_latency_ns": {
                "p50_upper_bound_ns": int | None,
                "p50_status": "bounded | censored_overflow | unavailable",
                "p95_upper_bound_ns": int | None,
                "p95_status": "bounded | censored_overflow | unavailable",
            },
            "complete_record_latency_ns": dict[str, object],
        },
        "sampler_window_backend": {
            "scope": "sampler_window_all_phases",
            "logical_range_reader_invocations": int | None,
            "python_preadv_invocations": int | None,
            "preadv_bytes_returned": int | None,
            "native_positional_calls": int | None,
            "native_bytes_returned": int | None,
        },
        "physical_device_queue_depth": {"status": "unavailable"},
        "gpu_expert_wait": {"status": "unavailable"},
        "coverage": {
            "attribution": "measured | incomplete",
            "decode_phase": "measured | incomplete",
            "sampler_window_backend": (
                "measured_all_phases | unavailable | incomplete_reset | incomplete"
            ),
            "operation_credit": "unavailable",
            "byte_credit": "unavailable",
            "authoritative_reserve": "unavailable",
            "slot_capacity_admission": "unavailable",
            "outer_split_executor_queue": "unavailable",
            "eligible_unsubmitted_cause": "unattributed",
            "admitted_read_ranges": "unavailable",
            "scheduled_read_ranges": "unavailable",
            "physical_device_operations": "unavailable",
            "physical_device_bytes": "unavailable",
            "physical_device_queue_depth": "unavailable",
            "gpu_expert_wait": "unavailable",
            "gpu_idle_time": "unavailable",
            "future_layer_eligibility": "unavailable",
            "speculative_record_accounting": "unavailable",
            "python_preadv_when_native_reader": "unavailable",
        },
    },
}
```

Keep v1 fields unchanged. The summary is decode-scoped, but its nested backend
deltas are explicitly all-phase over the sampler window. Preserve exact
nanosecond durations alongside fractions, emit `null` rather than zero when the
decode denominator is absent, and document identities, the telemetry-only lock,
histogram bounds, and unavailable claims in `docs/RESOURCE_TELEMETRY.md`. Point
`project-map.md` to the new Phase 1 evidence.

- [x] **Step 4: Run full verification**

Run:

```bash
uv run --frozen --extra dev --extra server python -m pytest -q
uv run --frozen --extra dev --extra server ruff check mtplx/resource_metrics.py mtplx/expert_io.py mtplx/expert_slots.py mtplx/expert_runtime.py mtplx/models/expert_mlx.py mtplx/benchmarks/resource_telemetry.py tests/test_resource_metrics.py tests/test_expert_io_metrics.py tests/test_expert_slots_runtime.py tests/test_streamed_models.py tests/test_resource_telemetry.py tests/test_benchmark_streamed_generation_concurrency_cli.py
uv run --frozen --extra dev --extra server ruff format --check mtplx/resource_metrics.py mtplx/expert_io.py mtplx/expert_slots.py mtplx/expert_runtime.py mtplx/models/expert_mlx.py mtplx/benchmarks/resource_telemetry.py tests/test_resource_metrics.py tests/test_expert_io_metrics.py tests/test_expert_slots_runtime.py tests/test_streamed_models.py tests/test_resource_telemetry.py tests/test_benchmark_streamed_generation_concurrency_cli.py
```

Expected: all tests pass; Ruff check and format pass.

- [x] **Step 5: Commit**

```bash
git add mtplx/benchmarks/resource_telemetry.py tests/test_resource_telemetry.py tests/test_benchmark_streamed_generation_concurrency_cli.py docs/RESOURCE_TELEMETRY.md project-map.md docs/specs/2026-07-13-hy3-starvation-attribution-design.md docs/plans/2026-07-13-hy3-starvation-attribution.md
git commit -m "docs(bench): publish starvation attribution contract"
```

## Hardware evidence after code verification

The hardware lane is diagnostic only. Start from a clean source SHA and use the
exact frozen #29 model, prompt, cache, reader, token-count, and thermal
conditions. Create a run ID using the `CONTRIBUTING.md` grammar and store raw
machine output under `benchmarks/raw/<benchmark>/<run-id>/`; write only the
curated reproducibility summary under `benchmarks/results/`.

Run at least four balanced telemetry-off/telemetry-on pairs in alternating
AB/BA order. Record every exact command and interval, and require parity for
generated tokens, router decisions, model/config hashes, logical record bytes,
cache counters, and failures. Estimate instrumentation overhead from the paired
telemetry-off lane; never use telemetry-on TPS as a promotion headline.

Each repeat uses this frozen command shape; `TELEMETRY_ARGS` is exactly
`(--no-resource-telemetry)` for control and
`(--resource-telemetry --resource-sample-interval 0.25
--resource-max-samples 4096 --ssd-ceiling-gib-s 12.47 --no-powermetrics)` for
attribution:

```bash
MODEL="$HOME/.cache/huggingface/hub/models--pipenetwork--Hy3-4bit/snapshots/160619d3f96c8470350b6dac0ef033a8381551e3"
LABEL="issue30-${VARIANT}-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short=12 HEAD)"
OUT="benchmarks/raw/moe-runtime/$LABEL"

uv run --frozen --extra dev --extra server python \
  scripts/benchmark_streamed_generation.py \
  "$MODEL" "$MODEL/expert-manifest-sidecar.json" \
  --model-key hy3-q4 \
  --memory-limit 120259084288 \
  --runtime-reserve 8589934592 \
  --expert-cache-limit 83034243072 \
  --max-live-kv-tokens 18888 \
  --cache-policy lru \
  --cache-scope global \
  --slot-layout component-banks \
  --transient-slots 32 \
  --read-chunk 67108864 \
  --f-nocache \
  --trust-sidecar \
  --no-enable-mtp \
  --chat \
  --prompt-file benchmarks/prompts/moe_streaming_realistic.md \
  --generation-profile deterministic \
  --max-tokens 256 \
  --concurrency 1 \
  --max-prefills-per-step 1 \
  --workload-shape static \
  --no-window-telemetry \
  "${TELEMETRY_ARGS[@]}" \
  --run-label "$LABEL" \
  --output-dir "$OUT" \
  --output-json "$OUT/result.json"
```

Before the exclusive window, capture whether Qwen is running. Stop it only for
the measurements, and install a shell cleanup trap that restores the captured
state on success, failure, or interruption. After restoration, verify the Qwen
API and loaded-model response before publishing the Phase 1 report to issue
#30. No striping or scheduler implementation begins until this evidence is
reviewed.
