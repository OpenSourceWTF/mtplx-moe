> **Repository scope:** This is a repository-wide `davidtai/MTPLX` issue.
> **Target branch:** `codex/moe-ssd-hy3-glm52`.
> **Implementation status:** connected to MTPLX runtime, CLI, one-shot, and
> server execution. Decode admission, transient prefill, bounded waves,
> rollback, KV limits, reset/close, and health telemetry are implemented.

## Objective

Integrate the expert slot policy into prefill, decode, batching, CLI/server
configuration, and telemetry without changing model routing semantics or
exceeding the resolved memory plan.

The policy acts only after the current layer's resident router produces top-k
IDs. It cannot pre-route future layers. Correct execution remains:

```text
router(layer n) -> cache plan -> load/await misses -> expert compute(layer n)
```

Prediction/prefetch may later use history as an optimization, but prefetched
data cannot determine route IDs and cannot be required for correctness.

## Proposed files and API

- `mtplx/expert_streaming.py`
  - retain decode-frequency admission, per-layer persistent banks, reusable
    transient slots, and aggregate counters.
- `mtplx/expert_runtime.py`
  - connect `RoutePlan` to slot loader and model adapter; own request phase,
    batching, completion, and cache lifetime.
- `mtplx/runtime.py`, `mtplx/cli.py`, and server profile/config models
  - expose explicit streaming and memory-limit fields.
- `mtplx/metrics.py` or existing metrics surface
  - export per-model/per-layer cache and I/O measurements with bounded label
    cardinality.
- `scripts/simulate_expert_cache.py`
  - consume both built-in model descriptors and recorded route traces.

Proposed configuration:

```python
ExpertStreamingConfig(
    enabled=True,
    memory_limit_bytes=96 * 2**30,
    max_live_kv_tokens=32768,
    expert_cache_limit_bytes=None,
    runtime_reserve_bytes=16 * 2**30,
    io_staging_bytes=512 * 2**20,
    execution_workspace_bytes=0,
    policy="decode-tinylfu",
    frequency_decay=0.995,
    prefill_admission=False,
    max_inflight_io_bytes=512 * 2**20,
)
```

## Policy semantics

- Maintain a persistent slot bank per sparse layer and one transient scratch
  bank reused across sequential layers.
- Count decode routes in decayed-frequency admission. Decode misses enter an
  empty persistent slot or replace an unpinned colder resident; otherwise use a
  transient slot.
- Prefill misses use transient service and do not evict a useful decode hot set.
- Pin every expert used by the current dispatch until its Metal completion.
- Preserve duplicate top-k IDs/order if the model emits them, while deduplicating
  physical reads for a route.
- For a batch, plan the union of selected experts for that layer. If its unique
  count exceeds planned transient capacity, split into bounded microbatches or
  use already-persistent slots; never allocate an overflow bank.
- Keep policy state runtime-local and model-specific. Multiple requests may
  share ready experts, but cancellation must release only that request's pins.
- Report cache warmness explicitly; never compare warm throughput to a cold
  baseline without labeling it.

## Runtime integration

- Resolve and print the memory plan before loading resident weights.
- Reconcile `memory_limit_bytes` with `MTPLX_MEMORY_LIMIT_BYTES`, reject
  conflicting caps, and invoke the existing `_apply_metal_memory_caps` path
  before resident or slot allocation.
- Validate requested context against planned KV/DSA bytes at request admission.
- Pass `prefill` or `decode` explicitly; do not infer the phase from tensor shape
  inside the policy.
- Await each layer's `ReadyRoute` before slot-backed expert dispatch, and retain
  the ready-generation token until compute completes.
- Provide an administrative cache reset that waits for active pins and clears
  policy metadata without rebuilding fixed buffers.
- On graceful shutdown, stop admissions, cancel queued reads, wait for active
  device work, then close files and buffers in that order.

## Failure handling

- Reject unsupported policy names, invalid decay/caps, or transient capacity
  below the maximum route/microbatch requirement before serving.
- Propagate native read/device errors to affected requests and increment failure
  counters; do not mark misses as hits after a failed load.
- Detect observed model/cache/I/O allocations beyond plan and stop admissions
  with planned-versus-observed diagnostics.
- A request exceeding planned context or batch capacity gets a clear admission
  error or documented microbatching, never an implicit allocation.
- Keep a hard maximum for trace/metric buffers so observability cannot defeat
  the memory limit.

## Required telemetry

- Routes, expert requests, hits, misses, hit rate, persistent/transient loads,
  evictions, resident occupancy, and cache reset count.
- Requested/read bytes, bytes/token, SSD read/queue/wait latency, concurrent
  reads, and short-read/checksum/device errors.
- Router time, load-wait time, expert compute time, time-to-first-token, decode
  tokens/s, and prefill tokens/s.
- Planned fixed/KV/transient/persistent/reserve bytes and observed peak unified
  memory from `mx.get_active_memory()`/peak counters, tagged by model and phase
  (not expert ID).

## Acceptance criteria

- [ ] Deterministic trace tests reproduce policy decisions across runs for Hy3
      and GLM-5.2 descriptors.
- [ ] Prefill-only traces do not change the persistent decode hot set; decode
      hot experts are admitted and cold one-offs use transient slots.
- [ ] Concurrent same-expert requests share one load and hold independent pins;
      cancelling one request does not corrupt the other.
- [ ] Batch-union overflow follows bounded microbatching/rejection and peak
      allocation stays within the plan.
- [ ] Zero persistent slots, full persistent residency, reset, shutdown, native
      error, and request cancellation paths are covered end-to-end.
- [ ] CLI and server APIs return the resolved plan and expose all required
      telemetry without unbounded expert-ID labels.
- [ ] Integration assertions prove each layer's router precedes its reads and
      expert dispatch, with no all-layer upfront routing.

## Dependencies

- Depends on router/memory plan, model adapter hooks, and native loader.
- Uses manifest identity in telemetry and diagnostics.
- Provides the measurements consumed by parity/benchmark release gates.
