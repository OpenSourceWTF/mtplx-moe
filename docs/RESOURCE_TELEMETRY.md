# Reading Streamed-Generation Resource Telemetry

Resource telemetry answers a different question from the headline benchmark:

- Tokens per second says whether a candidate helped.
- Resource telemetry says which resource or operation has evidence of limiting
  further improvement.

Telemetry-enabled runs are diagnostic. Do not use their token rate as a
promotion headline until the result is reproduced with telemetry disabled
under the same model, prompt, cache, seed, concurrency, and generation flags.
The disabled runtime keeps the original executor submission path: it does not
allocate telemetry occupancy trackers or add telemetry locks and wrappers.

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
2. Read `storage` and `reader_pool` together. SSD throughput alone cannot
   distinguish device saturation from an underfed device.
3. Read `completion_fences` and `overlap` to see whether reads and outstanding
   Metal-consumer work appear in the same sampling intervals. This is coarse
   coactivity evidence, not a simultaneous-overlap measurement.
4. Read `host.generation_thread_core_fraction` and measured GPU evidence.
5. Use `cache_by_layer` to find whether aggregate demand is concentrated in a
   subset of routed layers.
6. Treat `attribution.candidates` as the supported next investigations. This
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

## What each field actually means

`storage.mean_gib_per_second`
: Bytes returned by the positional reader divided by benchmark wall time. The
  harness compares this with an SSD ceiling only when `--f-nocache` is active;
  otherwise `coverage.storage_reads` is `logical_reader_bytes` and storage
  saturation is unavailable. `slots.io.read_mib_per_second` instead uses summed
  reader service time and can overstate sustained device throughput.

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

The schema therefore does not emit `bound_by`. It reports evidence, coverage,
and attribution status so an optimizing agent can choose the next operation to
measure without turning missing data into a performance claim.
