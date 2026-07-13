# Reading Streamed-Generation Resource Telemetry

Resource telemetry answers a different question from the headline benchmark:

- Tokens per second says whether a candidate helped.
- Resource telemetry says which resource or operation has evidence of limiting
  further improvement.

Telemetry-enabled runs are diagnostic. Do not use their token rate as a
promotion headline until the result is reproduced with telemetry disabled
under the same model, prompt, cache, seed, concurrency, and generation flags.
The enabled runtime allocates a phase-scoped `ExpertPipelineLedger`. Its short
local lock orders diagnostic state changes, but is never held across I/O,
hashing, MLX work, future waits, slot conditions, policy transactions, or
executor submission. The disabled runtime keeps the original executor
submission path: it allocates no pipeline ledger and enters no ledger clock,
lock, histogram, or callback.

## Run a diagnostic lane

```bash
python3 scripts/benchmark_streamed_generation.py MODEL_ROOT MANIFEST \
  --model-key hy3-q4 \
  --memory-limit 112GiB \
  --max-live-kv-tokens 2048 \
  --max-tokens 256 \
  --resource-telemetry \
  --resource-sample-interval 0.25 \
  --resource-max-samples 4096 \
  --f-nocache \
  --ssd-ceiling-gib-s 12.47 \
  --powermetrics \
  --no-window-telemetry \
  --output-json benchmarks/raw/resource-diagnostic/run.json
```

`powermetrics` is optional. The harness uses `sudo -n` and never prompts. If
authorization is unavailable, `coverage.gpu` is `unavailable` with the reason.
Authorize separately with `sudo -v` before the run when process GPU, CPU, wait,
and disk-I/O samples are required.

For the matched headline lane, remove `--resource-telemetry`,
`--powermetrics`, `--resource-sample-interval`, `--resource-max-samples`, and
`--ssd-ceiling-gib-s`. Keep every model and generation condition unchanged.

## Read the report in this order

1. Read `coverage`. An unavailable counter is unknown, never zero.
   If `coverage.timeline` is `retained_start_and_recent_tail`, cumulative
   throughput still spans the run, but interval attribution is deliberately
   incomplete; increase `--resource-max-samples` and repeat.
2. Read `expert_pipeline.coverage`, then confirm its summary `schema`,
   `source_schema`, and `scope`. Decode and backend coverage are independent;
   interpret decode fractions only against `decode_observation_ns`, and read
   backend-call deltas against their own all-phase status.
3. Read `expert_pipeline.orthogonal_overlap` before
   `primary_state_fraction`. The overlaps retain independent facts; the primary
   states provide one precedence partition for orientation.
4. Compare logical records, accepted executor calls, accepted records, reader
   tasks, decode ranges, and backend calls without treating them as one kind of
   operation.
5. Read `storage` and `reader_pool` together. SSD throughput alone cannot
   distinguish device saturation from an underfed device.
6. Read `completion_fences` and `overlap` to see whether reads and outstanding
   Metal-consumer work appear in the same sampling intervals. This is coarse
   coactivity evidence, not a simultaneous-overlap measurement.
7. Read `host.generation_thread_core_fraction` and measured GPU evidence.
8. Use `cache_by_layer` to find whether aggregate demand is concentrated in a
   subset of routed layers.
9. Treat `attribution.candidates` as the supported next investigations. This
   sampler does not perform a causal intervention, so its attribution remains
   `incomplete` even when a pressure screen is positive.

## Evidence matrix

| Observed evidence | Defensible conclusion | Operations to investigate |
| --- | --- | --- |
| Read queue nonempty, readers near capacity, uncached reader throughput near the supplied SSD ceiling | The storage-pressure screen is positive; storage throughput is a candidate | Reader bytes per token, cache hit rate, eviction policy, record layout; vary read shape or demand and measure the paired throughput response |
| Read queue nonempty and readers near capacity, but SSD below its ceiling | Backpressure exists before the device ceiling | Reader count, request size, batching, positional-read shape, completion processing |
| Cache misses occur while the read queue is empty and SSD use is low | The device is being starved | Route submission, dependency ordering, prefetch distance, producer serialization |
| Generation thread is near one full core while measured SSD and GPU activity are low | Host orchestration is a candidate | Python routing, wave construction, dispatch preparation, per-token bookkeeping |
| Measured process GPU activity is high while the read queue is empty | GPU compute is a candidate | Kernel shape, quantized matmul, gather/scatter, batch geometry |
| I/O-active and completion-fence-active intervals are common individually but rarely occur in the same sampling interval | Coarse I/O/fence separation is a candidate | Fence placement, work submission order, miss/compute pipelining; add a narrower probe before claiming serialization |
| Synchronous fences are frequent while asynchronous completion-fence occupancy is absent | Explicit evaluation or fence placement is a candidate | `mx.eval` sites, forced-sync route waves, graph boundaries, slot-release ordering |
| All measured resources remain below their ceilings, or GPU/DRAM coverage is absent | Attribution is incomplete | Obtain missing counters or add a narrower probe; do not name a bottleneck |

The current evidence thresholds are screening heuristics, not promotion gates
or causal cutoffs. Reports retain the continuous measurements, and a candidate
must be tested with a matched intervention and repeated uncertainty estimate
before it is called a bottleneck:

- SSD saturation requires at least 75% of the supplied SSD ceiling, at least
  75% mean reader-capacity use, and a nonempty read queue in at least 50% of
  intervals.
- Reader backpressure requires the reader-capacity and queue conditions, but
  does not require SSD saturation.
- Submission/dependency starvation requires cache misses, less than 10% queued
  intervals, and less than 40% of the supplied SSD ceiling.

No expert-pipeline state fraction is itself a promotion gate or causal cutoff,
and the ledger embeds no target reader depth such as eight. A candidate still
requires a matched intervention and repeated measurement.

## Expert-pipeline attribution in schema v2

The resource report has schema `mtplx-resource-telemetry-v2`. Its
`expert_pipeline` block is a decode summary with schema
`mtplx-expert-pipeline-summary-v1`; `source_schema` identifies the raw runtime
ledger as `mtplx-expert-pipeline-attribution-v1`. These names are deliberately
different: the runtime object is cumulative source telemetry, while the report
is a sampler-window summary.

### Keep operation identities separate

| Field | Exact meaning |
| --- | --- |
| `logical_record_jobs` / `logical_record_bytes` | Unique authoritative miss records from `RoutePlan.loads` and their manifest logical bytes; not router assignments or futures. |
| `accepted_executor_submissions` | Calls for which `executor.submit` returned a `Future`; worker start does not imply acceptance. |
| `submission_accepted_record_jobs` / `submission_accepted_record_bytes` | Records and logical bytes covered by accepted executor submissions. One task may cover a different number of records. |
| `reader_tasks_started`, `reader_tasks_completed`, `reader_tasks_failed` | Executor-task lifecycle counts. They are not record or range counts. |
| `decode_logical_ranges_started` | Logical contiguous range readers started in the decode ledger phase. |
| `sampler_window_backend.logical_range_reader_invocations` | Backward-compatible `ExpertIOMetrics.read_operations`: entries into a logical contiguous range reader, including failing calls; not completed decode ranges, kernel calls, or device operations. |
| `sampler_window_backend.python_preadv_invocations` | Python `os.preadv` attempts, including interrupted attempts. This does not prove kernel entry. |
| `sampler_window_backend.preadv_bytes_returned` | Positive bytes returned by Python `os.preadv`. |
| `sampler_window_backend.native_positional_calls` / `native_bytes_returned` | Native positional-reader attempts and returned bytes. Native calls provide no Python `preadv` coverage. |

The summary-level record, task, and decode-range fields have `scope="decode"`.
`sampler_window_backend.scope` is `sampler_window_all_phases`: those reader
deltas cover the full sampler window and must not be presented as decode-only.
Its coverage is `measured_all_phases`, `unavailable`, `incomplete_reset`, or
`incomplete`. Missing backend counters are `null`, never measured zeroes.

### Decode observation and primary precedence

`decode_observation_ns` is the measured decode time during which the ledger had
an open route, active logical range, open hit/shared-work span, or generation
wait. It excludes inactive prefill, setup, and teardown. Exact durations appear
in `primary_state_ns`; `primary_state_fraction` divides each duration by the
summed `decode_observation_ns`. When that denominator is zero, fractions and
`generation_expert_input_wait_fraction` are `null`, not zero.

The primary state is selected at every event boundary in this precedence:

1. `generation_thread_expert_input_wait` — the generation thread is blocked on
   the next miss completion.
2. `logical_range_active` — a logical range reader is executing.
3. `reader_completion_active` — a reader task is active outside a logical
   range, such as hashing, validation, or READY publication.
4. `submitted_queued` — provisional or accepted work awaits reader service.
5. `eligible_unsubmitted` — authoritative current-route work is not submitted.
6. `host_runnable_work` — hit, shared, or completed-miss work is known runnable
   but has not been host-dispatched.
7. `route_publication_pending` — a verified or existing READY record is not yet
   part of a constructed route.
8. `no_known_useful_work` — none of the preceding states applies inside the
   covered decode window.

This precedence produces one orientation state at a time. It does not erase the
independent state integrals in `decode_integrals_ns`. The eight exact
`primary_state_ns` values must sum to `decode_observation_ns`; a mismatch keeps
the raw values but marks `coverage.attribution` and `coverage.decode_phase`
incomplete.

### Orthogonal overlap is independent evidence

`orthogonal_overlap.duration_ns` retains five independent intersections with
generation-thread expert-input wait: logical-range activity, any reader-task
activity, submitted-queued work, eligible-unsubmitted work, and host-runnable
work. They may overlap each other and must not be added together.
`orthogonal_overlap.denominator` is `decode_observation_ns`, and
`fraction_of_decode_observation` uses that denominator. A wait/runnable overlap
is host evidence to investigate; it is not automatically an invariant failure
and is never relabeled GPU wait.

### Histograms are bounded, not interpolated

`latency_histograms` reports logical-range and complete-record latency using
fixed upper bounds of 1 us, 10 us, 100 us, 1 ms, 10 ms, 100 ms, and 1 s, plus an
overflow bucket. `p50_upper_bound_ns` and `p95_upper_bound_ns` are bucket upper
bounds, not interpolated percentiles. Their status is `bounded`,
`censored_overflow`, or `unavailable`. If a percentile lands in overflow, only
that percentile is `null` and censored above the largest bound; exact counters,
state integrals, and other bounded percentiles remain valid. A missing required
logical-range or complete-record histogram marks decode coverage incomplete.

### Coverage and unavailable claims

`block_reasons.pin_held` and `block_reasons.slot_loading` contain measured count
and duration. They cover elapsed `Condition.wait` time, published only after the
slot condition is released. Operation credit, byte credit, authoritative
reserve, and distinct slot-capacity admission do not exist in the frozen #29
runtime and remain `unavailable`, not measured zeroes.

The report also leaves physical device operations, physical device bytes,
physical device queue depth, `gpu_expert_wait`, and `gpu_idle_time` unavailable.
Its coverage also marks the outer split-executor queue, independently
admitted/scheduled ranges, `future_layer_eligibility`,
`speculative_record_accounting`, and `python_preadv_when_native_reader`
unavailable; eligible-unsubmitted cause remains unattributed.
`F_NOCACHE` supports the term
"uncached reader traffic," never physical NAND traffic. Submission rejection,
counter reset, histogram-contract change, hook failure, or ledger invariant
failure makes attribution incomplete; a valid histogram overflow alone does
not.

`coverage.decode_phase` and `coverage.sampler_window_backend` are independent.
`coverage.attribution` is the aggregate: it becomes incomplete when either
scope is incomplete, while retaining the exact scope-specific status and raw
values.

## What each field actually means

`storage.mean_gib_per_second`
: Bytes returned by the positional reader divided by benchmark wall time. The
  harness compares this with an SSD ceiling only when `--f-nocache` is active;
  otherwise `coverage.storage_reads` is `logical_reader_bytes` and storage
  saturation is unavailable. `slots.io.read_mib_per_second` instead uses summed
  reader service time and can overstate sustained device throughput.

`storage.iops` and `storage.reader_read_operations`
: Compatibility names derived from logical range-reader invocations. They are
  neither Python/native backend-call rates nor physical device IOPS; use
  `expert_pipeline.sampler_window_backend` for the separated backend counts.

`reader_pool.mean_queued_reads`
: Occupancy integrated at executor state changes. A sustained value above zero
  means accepted read work waited for a reader.

`reader_pool.mean_active_readers`
: Mean occupied reader workers. Compare it with `worker_capacity`; aggregate
  process CPU percentage is not a substitute.

`reader_pool.mean_queued_bytes` and `mean_active_bytes`
: Record bytes waiting for or executing in the reader pool. They are internal
  in-flight demand, not SSD cache residency.

`completion_fences.mean_active_slots`
: Slot generations whose last Metal consumer has not completed. This is
  outstanding Metal-dependent work, not GPU utilization.

`completion_fences.synchronous_fences` and `synchronous_fences_per_token`
: Calls that force an MLX evaluation before a routed slot may be released,
  counted only in diagnostic runs. A high rate identifies an operation worth
  narrowing; it does not prove that the evaluation itself dominates wall time.

`completion_fences.registrations` and `registered_slots`
: Routed completion barriers and the slot bindings they cover. Compare these
  operation counts with both synchronous fences and measured occupancy; a
  registration count alone says nothing about how long a barrier waited.

`overlap.same_interval_activity_fraction`
: Fraction of sampling intervals in which both reader work and an active
  completion fence were observed. Independent state-change integrals feed this
  sampled signal, so work may have occurred sequentially inside one interval.
  `simultaneous_overlap_measured` is therefore always false. Use this field to
  choose a narrower probe, not to claim simultaneous overlap or serialization.

`throughput.q4_assignments_per_second`
: Routed Q4 expert assignments observed by the cache policy per wall second.
  This is model demand, not DRAM bandwidth.

`host.generation_thread_core_fraction`
: Caller-thread CPU time divided by the generation window. A value near 1.0
  means roughly one core was occupied; it does not describe the reader threads.

`powermetrics.process_gpu_busy_fraction`
: Per-process GPU milliseconds per second reported by `powermetrics`, normalized
  to a fraction. If hardware or privilege does not expose it, the field remains
  unavailable rather than becoming zero.

`cache_by_layer`
: Before/after demand deltas by routed layer. Sort by `expert_misses`,
  `bytes_read`, or eviction count before applying a model-wide optimization.

## Claims the report does not support

- Pending completion fences are not measured GPU utilization.
- Routed expert bytes are not measured DRAM traffic.
- Low SSD utilization does not prove serialization.
- Elapsed time alone does not identify a resource bottleneck.
- System-wide GPU activity is not per-process GPU activity.
- A peak without occupancy or throughput context does not establish sustained
  pressure.
- Host expert-input wait is not GPU wait.
- Python or native positional calls are not physical device operations.

The schema therefore does not emit `bound_by`. It reports evidence, coverage,
and attribution status so an optimizing agent can choose the next operation to
measure without turning missing data into a performance claim.
