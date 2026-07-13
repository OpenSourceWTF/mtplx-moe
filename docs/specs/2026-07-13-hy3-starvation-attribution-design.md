# Hy3 Starvation Attribution Design

## Status and predecessor

This design implements only Phase 1 of issue #30. It is based on finalized
issue #29 commit `b7ba5ab526a44c4abbeb11a1462794f3ea79d1a0` and supersedes the
causal-attribution portions of the older record-native and striping plans.

The measured problem is real: the B1 diagnostic averaged 3.732 active readers
out of 32 and 6.065 GiB/s of reader-returned data while the checked-in host
concurrency-8 control reached 12.469 GiB/s. Those measurements prove underfill,
but they do not distinguish dependency-limited demand, unissued eligible work,
reader service, or generation-thread expert-input wait. Phase 1 exists to make
that distinction before any scheduling behavior changes.

## Scope

Phase 1 adds opt-in, event-time attribution for:

- authoritative logical record eligibility plus attempted, accepted, and
  rejected submission;
- queued and active reader tasks;
- active logical read ranges;
- logical range jobs, actual Python `preadv` calls, and returned bytes;
- slot-loading and pin-held wait duration;
- verified record readiness and host dispatch for execution;
- host-known runnable hit, shared, and completed-miss work;
- generation-thread time blocked waiting for the next miss future;
- orthogonal overlap between storage activity, runnable work, dependency
  underfill, and generation-thread expert-input wait;
- bounded latency histograms for logical ranges and complete records.

The default telemetry-off path retains the exact #29 submission and execution
path. Diagnostic results cannot be used as headline TPS.

## Non-goals

This phase does not add striping, prefetch, speculation, operation or byte
credits, an authoritative reserve, a scheduler, a cache-policy change, a new
runtime barrier, or a new production-path lock. It does not claim physical
device operations, physical device queue depth, device bytes, exact GPU idle
time, or GPU expert-wait. It does not treat `mx.eval` duration as fence
overhead.

## Metric ontology

The report keeps these identities separate:

| Field | Meaning |
| --- | --- |
| `logical_record_jobs` / `logical_record_bytes` | Unique authoritative records exposed by `RoutePlan.loads`, and their exact manifest logical bytes. |
| `submission_attempted_*` | Records, bytes, and executor calls provisionally exposed before `submit`; this closes the worker-start-before-return race. |
| `accepted_submissions` | Executor calls for which `executor.submit` returned a `Future`. |
| `submission_accepted_*` | Records and bytes covered by calls for which `executor.submit` returned a `Future`. Acceptance is never inferred from worker start. |
| `submission_rejected_*` | Provisional work rolled back because `executor.submit` raised. Any rejection marks the attribution interval incomplete. |
| `started_reader_tasks` / `completed_reader_tasks` / `failed_reader_tasks` | Explicit executor-job lifecycle counts. A task may contain one record today and may contain a batch elsewhere. |
| `logical_range_reader_invocations` | Calls into one logical contiguous range reader. This is the backward-compatible `ExpertIOMetrics.read_operations`; it is not a queued or physical operation count. |
| `python_preadv_invocations` | Python `os.preadv` invocation attempts, including interrupted attempts. It does not prove kernel entry. |
| `preadv_bytes_returned` | Positive bytes returned by Python `os.preadv`. |
| `native_positional_calls` | Native-reader invocations; these do not claim Python `preadv` coverage. |
| `native_bytes_returned` | Positive bytes returned by the native positional reader. |
| `host_active_ranges` | Logical range readers currently executing, integrated at range start/finish events. |
| `physical_device_queue_depth` | Always `unavailable` in this phase. |

`read_bytes` means reader-returned bytes. With `F_NOCACHE` it is defensible as
uncached reader traffic, not as physical NAND traffic.

The v2 summary preserves those identities under explicit public names:
`accepted_executor_submissions`, `submission_accepted_record_jobs` and bytes,
`reader_tasks_started`/`completed`/`failed`, and
`decode_logical_ranges_started`. Backend counters are nested under
`sampler_window_backend`; they cover all phases in the sampler window and are
not presented as decode-only.

## Architecture

### Diagnostic event-time ledger

`ExpertPipelineLedger` lives in `mtplx/resource_metrics.py`. It is constructed
only when `resource_telemetry=True`; otherwise every new runtime reference is
`None` and no new clock, lock, histogram, or callback is entered.

The ledger stores cumulative counters, state-time integrals, and bounded
histogram buckets rather than raw events. A short internal lock orders state
changes from the generation, split-route, and reader threads. The lock exists
only in the diagnostic lane and is never held across I/O, hashing, MLX work,
future waits, slot conditions, policy transactions, or executor submission.
No lock is added to the production timing lane.

Each split route receives a ledger-owned, load-only route handle after
`_plan_route_transaction` produces its miss plan. Each record is registered
with its exact logical byte count. Hits and shared work use separate opaque
ledger spans and are never owned by the load route. The route tracks unique
load experts through these transitions:

```text
eligible -> submission-provisional -> accepted/reader-active -> verified
                                                        -> route-part-runnable
                                                        -> claimed-for-execution
        \-> satisfied-without-new-submit -> route-part-runnable -> claimed
        \-> abandoned-on-failure-or-async-terminal-close
```

Submitting is a three-state handshake: provisional state is published before
`executor.submit`, success is counted only after `submit` returns, and
rejection rolls provisional state back. A worker is allowed to move from
provisional to active before the submitter records acceptance. Closing the
route at its true asynchronous success/failure terminal point resolves any
remaining records as abandoned, preventing failed or cancelled work from
leaking nonzero occupancy into later samples. `PendingSplitRoute.close()` alone
is not terminal while callbacks or futures can still settle.
“Claimed” means host dispatch into `evaluate_bindings`; it does not claim that
the GPU has physically consumed the record.

### Exact hooks

- `ExpertStreamingRuntime.begin_split_route` starts the load-only route handle
  after forming `miss_plan` and `miss_parts`. Eligibility is the unique
  `miss_plan.loads` plus exact manifest bytes, never router assignment count or
  future count.
- `ExpertSlotPool._wait_ready` measures only elapsed time around actual
  `Condition.wait` calls for an existing loading generation or held pins. It
  publishes the elapsed observation after releasing the slot condition, so the
  ledger lock is never nested under a slot lock. A plan load satisfied by an
  existing generation is marked `satisfied_without_new_submit` only after
  `_wait_ready` confirms READY; it remains nonterminal until route-part
  publication and host claim.
- `_submit_tracked` publishes a provisional attempt before `executor.submit`,
  records acceptance only after a `Future` returns, and rolls provisional state
  back on rejection. `_run_tracked` may legally start between those events.
- `_run_tracked`, `_fill`, and `_fill_batch` record reader-task activity,
  verified complete-record readiness, failures, latency, and reader-thread CPU.
  Verified slot readiness ends record service time but does not yet make a miss
  runnable. The route part becomes host-runnable only after `_ensure_route_owned`
  has completed waits, pins, bindings, and `ReadyRoute` construction.
- `PositionalExpertReader` records range start/finish against the route's phase
  scope and keeps local backend-call
  counters, publishing one metrics update at range completion. Interrupted
  `preadv` attempts count as calls but return no bytes.
- The streamed MLX decode path opens ledger-level, phase-scoped hit and
  shared-work spans.
  Hit spans begin after either an all-hit `ReadyRoute` or split `hit_ready` is
  known. The shared span begins once decode phase and the callback are known.
  These spans cover both split-route overlap and the all-hit fast path without
  changing `ReadyRoute` ownership, then claim work exactly where the existing
  Q4 or `shared_work()` callback is dispatched.
- The streamed MLX decode path marks hit/shared/completed-miss work claimed
  immediately before host dispatch to its existing evaluation function.
- `PendingSplitRoute.iter_ready_misses` measures only a blocking `next()` on
  the completion iterator. It maintains an explicit remaining-future set and
  removes every yielded future before deciding whether the next step can block;
  otherwise the first completed future would suppress timing of later waits.
  It does not include hit work, shared work, Q4 evaluation, policy publication,
  release, or cleanup.
- Every runtime hook is fail-open for data-path behavior: instrumentation
  invariant failures mark coverage incomplete but cannot change a successful
  route result or mask its original exception. Strict transition failures are
  confined to deterministic ledger unit tests.

### Orthogonal state-time accounting

Every ledger counter, gauge, integral, and histogram is retained both globally
and by phase (`decode`, `prefill`, or `unscoped`). The B1 report uses only `decode`;
prefill cannot contaminate decode fractions. State changes accrue cumulative
record and byte nanoseconds for independent facts:

- eligible-but-unsubmitted records;
- queued and active reader tasks;
- active logical ranges;
- verified/satisfied records awaiting route publication and runnable misses
  awaiting host claim;
- runnable hit, shared, and completed-miss work;
- generation-thread expert-input wait;
- pairwise overlap between expert-input wait and logical-range activity,
  reader completion outside a range, runnable work, eligible unsubmitted work,
  or queued submitted work.

No fixed target such as eight is embedded in the ledger. Eight was a host
concurrency label, not a causal threshold. The report exposes exact available
useful work as eligible, provisional/queued, reader-active, verified, and
runnable states. A later optimization gate may compare service depth with
`min(configured_reader_capacity, currently_available_useful_records)`, but it
must not call low depth starvation when fewer useful records exist.

The orthogonal durations are authoritative. Primary time accrues only while the
phase has an open route, active logical range, runnable hit/shared span, or
generation wait; inactive prefill/setup/teardown time cannot inflate decode's
fallback. The report publishes this covered observation duration explicitly.
Within that window it derives one deterministic decode-only primary state with
this precedence at every event boundary:

1. `generation_thread_expert_input_wait` when the generation thread is inside
   the blocking next-miss span;
2. `logical_range_active` when a logical range reader is executing;
3. `reader_completion_active` when a reader task is active outside a logical
   range (for example hashing, validation, or READY publication);
4. `submitted_queued` when accepted/provisional work is waiting for reader
   service;
5. `eligible_unsubmitted` when authoritative work remains unsubmitted;
6. `host_runnable_work` when a hit, shared callback, or completed miss is known
   runnable but has not yet been host-dispatched;
7. `route_publication_pending` when a verified or existing READY record is not
   yet part of a constructed route;
8. `no_known_useful_work` otherwise within the covered phase window.

If generation wait overlaps host-known runnable work, the time remains in the
first primary state and is also retained in the orthogonal overlap counter. It
is evidence to interpret, not automatically an instrumentation failure. The
report never relabels this host state as GPU wait.

## Coverage contract

The Phase 1 report must say `unavailable`, not zero, for:

- operation-credit blocking (there is no operation-credit mechanism in #29);
- byte-credit blocking (there is no byte-credit mechanism in #29);
- authoritative-reserve blocking (there is no reserve mechanism in #29);
- admitted work (there is no distinct admission stage in #29);
- outer split-executor queue occupancy; eligibility begins before that submit,
  so eligible-unsubmitted time includes the delay but its cause remains
  explicitly `unattributed` rather than being mislabeled as reader-queue time;
- slot-capacity admission blocking (pin-held and existing-generation loading
  waits are measured separately, but #29 has no distinct slot admission stage);
- independently queued logical ranges in the frozen whole-record executor;
- physical device operations, bytes, and queue depth;
- exact `gpu_idle_time` and `gpu_expert_wait`;
- future-layer or speculative eligibility;
- useful, late, cancelled, or unused speculative records;
- Python `preadv` coverage when the native reader serves the range.

Pin-held and slot-loading waits are measured. There is no distinct slot-capacity
admission mechanism in #29, so it is not invented as a counter.

## Report and schema

The runtime's cheap snapshot gains an `expert_pipeline` source object with
schema `mtplx-expert-pipeline-attribution-v1`. The existing same-clock sampler
differences its cumulative counters and integrals alongside reader-pool
telemetry. The resource report schema becomes `mtplx-resource-telemetry-v2`;
v1 field meanings remain stable. Its `expert_pipeline` report block is a
derived summary with schema `mtplx-expert-pipeline-summary-v1`, identifies the
runtime schema in `source_schema`, and declares `scope="decode"`.

The summary exposes exact `primary_state_ns`, `decode_integrals_ns`, block
durations, and orthogonal-overlap durations. Fractions use the summed
`decode_observation_ns` denominator and are `null` when that denominator is
zero. `orthogonal_overlap.denominator` names that denominator explicitly, and
its independent durations and fractions must not be summed. The eight primary
durations must sum to `decode_observation_ns`; a mismatch retains the raw values
but marks attribution and decode-phase coverage incomplete. Backend-call
deltas live under `sampler_window_backend` with
`scope="sampler_window_all_phases"`; `logical_range_reader_invocations` there
means the backward-compatible `ExpertIOMetrics.read_operations`, not the
decode ledger's started-range count. Backend coverage is independent of decode
coverage and is `measured_all_phases`, `unavailable`, `incomplete_reset`, or
`incomplete`; absent backend counters are `null`, not measured zeroes. Overall
attribution becomes incomplete if either scope is incomplete.

Histogram percentiles are reported as bucket upper-bound estimates with sample
and overflow counts. The finite bounds are 1 us, 10 us, 100 us, 1 ms, 10 ms,
100 ms, and 1 s. Each percentile status is `bounded`, `censored_overflow`, or
`unavailable`. Overflow censors only the affected percentile above the largest
bound; it does not invalidate exact counters or state integrals. A route
instrumentation invariant violation, invalid histogram contract, or missing
required logical-range/complete-record histogram does mark decode coverage
incomplete.

## Testing strategy

- Fake-clock ledger tests prove phase isolation, record/byte transitions,
  submission-start races, orthogonal overlap, cleanup, histogram bounds,
  fail-open runtime hooks, and strict-test underflow rejection.
- Reader tests force partial reads and `InterruptedError` to prove that ranges,
  syscalls, and bytes remain distinct; native calls cannot claim `preadv`.
- Runtime tests prove eligibility uses unique loads, accepted records differ
  from reader tasks, pin/loading waits remain distinct, failed work never
  becomes ready, and telemetry-off avoids the ledger entirely.
- Controlled futures prove completed records become runnable before host claim,
  two distinct future waits produce two distinct blocking spans, buffered
  futures do not add wait, and only blocking completion waits add expert-input
  wait.
- Sampler tests use unequal interval widths so state fractions are duration
  weighted, preserve orthogonal overlaps, and retain explicit unavailable
  coverage.
- Existing slot, streaming, telemetry, and streamed-model suites lock #29
  behavior.

## Evidence and promotion

Phase 1 is diagnostic and does not itself authorize a runtime optimization.
Hardware evidence uses a clean commit, the frozen #29 model/configuration, and
balanced repeated telemetry-off/telemetry-on pairs. Qwen is unloaded only for
the exclusive measurement window and restored and API-verified in a cleanup
trap. Raw artifacts use the repository run-id grammar under `benchmarks/raw/`;
the checked-in result under `benchmarks/results/` records exact commands,
commit, model hashes, repeat order, thermal/power metadata, and parity fields.
At least four balanced pairs are required before estimating instrumentation
overhead. Every run must preserve exact token output, routes, cache counters,
logical record bytes, and model/config hashes, then publish:

- accepted executor-call, logical record, reader-task, decode-range, Python and
  native backend-call, and returned-byte counts with their declared scopes;
- time-weighted eligible, queued, active, ready, runnable, and wait states;
- block reasons and unavailable coverage;
- range and complete-record latency distributions;
- reader-thread and generation-thread CPU time.

Phase 2 or Phase 3 may proceed only if the report finds measurable unissued
eligible work or material generation-thread expert-input wait that a separately
measured feed can cover.

## Failure-mode check

1. **Instrumentation perturbs the diagnostic.** Severity: material but
   contained. The ledger is telemetry-only, never used for headline TPS, holds
   no lock across work, and will be compared with a matched telemetry-off lane.
2. **A failed route leaks occupancy and fabricates starvation.** Severity:
   critical. Route handles own every record state and close/abort resolves all
   nonterminal records; invariant failures mark the report incomplete.
3. **Host events are mistaken for device facts.** Severity: critical. The
   ontology and coverage block keep host tasks/ranges/syscalls separate and
   explicitly leave device queue depth, GPU wait, and native `preadv` coverage
   unavailable.
