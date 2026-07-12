# Engineering Review: Serving a Very Large MoE LLM on Apple Silicon with NVMe-Backed Routed Experts

## 1. System Overview and Execution Sequence

The proposed design serves a very large mixture-of-experts (MoE) language model on an Apple Silicon workstation. The router, attention layers, embeddings, normalization layers, and shared experts are resident in unified memory (CPU/GPU shared DRAM). Routed expert weights are stored as affine 4-bit tensors on a local NVMe SSD. At each sparse layer, the router selects eight experts; on a cache miss, selected experts are loaded into a fixed, user-bounded memory bank. Decode requests populate a frequency-decayed hot cache, while prompt prefill uses transient slots so a single large prompt cannot evict the long-lived decode working set.

The steady-state execution sequence for a decode step is:

1. Token embedding lookup in unified memory.
2. For each transformer layer:
   a. Normalization and attention in unified memory.
   b. Router computes expert assignments (top-8 experts per token or per batch).
   c. For each selected expert: check hot cache; on miss, issue async NVMe read into a pinned slot in the bounded memory bank.
   d. Execute expert MLP (matmul in 4-bit affine format, dequantized on the fly or via AMX/NEON paths).
   e. Combine expert outputs with router weights.
3. Final normalization and LM head in unified memory.

Prefill follows the same path but uses transient slots that are reclaimed immediately after the prefill pass completes, isolating prefill pressure from decode residency.

## 2. Concurrency and Slot Pinning

The bounded memory bank must be partitioned into fixed-size slots, each sized to the largest routed expert (or to a quantized chunk boundary). Slot pinning should be implemented via a lock-free slot table:

- Decode hot cache: N pinned slots, owned by the decode scheduler. Admission is frequency-decayed (exponential moving average of token hits per expert ID).
- Prefill transient pool: M slots, allocated per prefill request and freed on completion. No cross-request retention.

Concurrency:
- Router and attention run on the Apple Silicon GPU/ANE or CPU matrix engine while NVMe reads are issued from a dedicated I/O actor (libdispatch or a custom SPDK-style poller).
- Expert execution must wait on a per-slot completion semaphore; do not block the router thread.
- Prefill and decode should use separate queues; prefill transient slots must never alias decode hot slots.

## 3. Memory Accounting

The user-bounded memory bank should be sized explicitly:

- Total bank = decode_slots × expert_size + prefill_slots × expert_size + alignment headroom.
- Unified memory resident components (router, attention, embeddings, norms, shared experts) are accounted separately and locked at process start.
- Use `malloc_zone` or `vm_allocate` with `VM_FLAGS_PURGABLE` off for the bank to prevent compaction surprises.
- Expose live slot count, free bytes, and peak decode residency as process metrics.

## 4. Positional I/O and Integrity Checks

Experts are stored as affine 4-bit tensors: a scale/zero-point (fp16 or fp32) plus packed 4-bit weights. Each expert file or shard region should include:

- A 64-byte header: expert ID, layer ID, tensor shape, affine parameters, checksum algorithm.
- A trailing BLAKE3 or xxHash64 over the payload.
- Offset tables if stored in component-major safetensor shards.

On load:
- Read header + payload + checksum.
- Verify checksum before mapping into the slot.
- If mismatched, mark expert corrupt, fall back to shared-expert-only path or reject the request.

NVMe reads should use aligned, large-block (>=64 KB) transfers to amortize I/O latency. Prefer `pread` with `POSIX_FADV_DONTNEED` after prefill to avoid polluting the page cache.

## 5. Cache Admission and Eviction

Decode hot cache:
- Admission requires a hit-rate threshold over a decay window (e.g., EMA half-life of 200 decode steps).
- Eviction uses least-frequently-used with decay; never evict a slot mid-execution.
- Prefill transient slots are not admission-eligible.

This separation ensures a 32K-token prefill cannot evict the decode working set that serves hundreds of concurrent small requests.

## 6. Failure Handling

The system must tolerate:
- Corrupt expert blobs (checksum fail): isolate and alert; route around if shared experts suffice.
- NVMe read timeout: bound at 50 ms; on timeout, abort the step and return a degraded response.
- Slot exhaustion under prefill storms: reject new prefill with 429-style backpressure.
- Unified memory pressure from non-model allocations: reserve bank via `task_policy` and monitor `phys_footprint`.
- Router producing out-of-range expert IDs: clamp and log; never dereference invalid slot.

## 7. Observability

Instrument:
- Per-layer expert miss rate, NVMe read latency histogram, slot utilization.
- Decode vs prefill cache hit ratio.
- Checksum failure count, timeout count, fallback activations.
- End-to-end token latency by request class.

Use structured logs and a local Prometheus node exporter; ship to a central collector if allowed.

## 8. Correctness Testing

Tests must include:
- Deterministic replay of router outputs with forced misses to validate load+compute path.
- Bit-exact comparison of 4-bit affine dequant vs reference fp16 expert on a fixed input.
- Prefill-then-decode isolation test: large prefill must not change decode cache contents.
- Corruption injection: flip a byte in an expert blob; verify rejection and fallback.
- Slot accounting unit tests: over-admission, double-free, and concurrent prefill/decode.

## 9. Safetensor Shards vs Expert-Major Sidecar

Two storage layouts are possible:

**Component-major safetensor shards** store all layers’ weights in one or few large files, with experts interleaved by layer. Reading one expert requires seeking into a large shard and reading a small region. This is simple for training checkpoints but causes many small random reads on NVMe, poor read amplification, and higher latency under concurrent misses.

**Expert-major sidecar** stores each expert (or grouped experts) as its own contiguous object, optionally generated by a post-training repack step. This gives large sequential reads, better NVMe throughput, and simpler integrity scoping. Tradeoff: extra disk space and a repack pipeline; must be kept in sync with the canonical checkpoint.

Recommendation: ship component-major as source of truth, but generate an optional expert-major sidecar at deploy time and verify checksums against the primary. Production should read from the sidecar for latency; fall back to primary shards only for recovery.

## 10. Five Concrete Failure Modes and Mitigations

1. **NVMe wear or controller stall** – Mitigation: per-read timeout + circuit breaker that disables expert loading and runs shared-expert-only degraded mode.
2. **4-bit dequant drift causing output divergence** – Mitigation: nightly differential test against fp16 reference on golden prompts; alert on perplexity delta.
3. **Prefill slot leak from canceled requests** – Mitigation: RAII-style slot handles with deterministic reclaim in the scheduler; leak detector in CI.
4. **Router selects expert not present on disk (version mismatch)** – Mitigation: manifest validation at boot; reject startup if expert count mismatches model config.
5. **Unified memory oversubscription from Apple OS background tasks** – Mitigation: reserve bank at launch, monitor footprint, and cap concurrent requests.

## 11. Staged Rollout Plan

- **Stage 0 (Lab):** Load model in unified memory; experts from sidecar; synthetic traffic. Validate correctness and latency.
- **Stage 1 (Canary):** Internal users, decode-only, small batch. Monitor miss rate and latency.
- **Stage 2 (Prefill):** Enable prefill transient pool; watch isolation guarantees.
- **Stage 3 (Load):** Increase concurrency to target; tune slot counts.
- **Stage 4 (Full):** Production traffic with fallbacks armed.

## 12. Go / No-Go Criteria

Go if:
- Decode p99 expert-load latency < 30 ms on NVMe.
- Prefill does not evict decode hot slots in 100% of tests.
- Checksum failure rate = 0 in canary.
- Differential perplexity delta vs reference < 1%.
- No unified-memory OOM over 72h soak.

No-go if:
- Any data-corruption path reaches users.
- Slot leak detected in canary.
- Fallback rate exceeds 1% under normal load.

This design is sound for a single-workstation MoE deployment if the isolation, integrity, and observability controls above are implemented before Stage 1.
