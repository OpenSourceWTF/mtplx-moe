# Engineering Review: Serving a Very Large MoE LLM on Apple Silicon with NVMe-Backed Routed Experts

## 1. System Overview and Execution Sequence

The proposed design serves a very large mixture-of-experts (MoE) language model on an Apple Silicon workstation. The router, attention layers, embeddings, normalization layers, and shared experts are resident in unified memory (DRAM mapped into the Apple Silicon memory controller). Routed expert weights are too large to fit entirely in unified memory and are stored as affine 4-bit tensors on a local NVMe SSD. At each sparse layer, the router selects eight experts. On a cache miss, selected experts are loaded from NVMe into a fixed, user-bounded memory bank (a reserved region of unified memory). Decode requests populate a frequency-decayed hot cache; prompt prefill uses transient slots that are isolated from the long-lived decode working set.

A single forward pass executes as follows:

1. Token embeddings and positional encodings are read from unified memory.
2. For each transformer block:
   a. Attention, normalization, and shared expert compute execute from unified memory.
   b. The router scores tokens and selects the top-eight experts for that layer.
   c. The scheduler checks the expert cache. Hits are computed in place. Misses are queued for NVMe load into the bounded memory bank.
   d. Once all eight experts are resident, expert compute executes; outputs are combined via the router weights.
3. The final norm and language-head projection run from unified memory.

Because Apple Silicon uses a unified memory architecture, there is no explicit host-to-device copy; "loading into the memory bank" means mapping or copying the affine 4-bit tensor into a reserved virtual region and ensuring it is not paged out (via `mlock` or equivalent advisory locking where permitted).

## 2. Concurrency and Slot Pinning

The bounded memory bank should be partitioned into fixed-size slots, each sized to the largest routed expert (or to a quantized multiple). Slot assignment must be deterministic per layer and expert ID to avoid duplicate loads. We recommend:

- A per-layer slot table mapping `expert_id -> slot_index`.
- A read-write lock per slot: shared for compute, exclusive for load/evict.
- A load coalescing queue: if multiple tokens in a batch miss the same expert, only one NVMe read is issued; subsequent waiters block on a slot-ready futex.
- Prefill transient slots are drawn from a separate arena so decode hot-cache entries are never evicted by a large prompt. Transient slots are freed at prefill completion.

Pinning should be soft: Apple Silicon does not expose full `mlock` semantics identically to Linux, so we should use a combination of memory prioritization and a watchdog that re-faults critical tensors if the OS reclaims them.

## 3. Memory Accounting

The user-bounded memory bank must be accounted precisely:

- Total bank size = `slots * slot_size + alignment padding`.
- Unified memory headroom must reserve space for router, attention, embeddings, shared experts, KV cache, and OS.
- Each slot records: `layer_id`, `expert_id`, `state (empty/loading/ready/evicting)`, `last_used_ns`, `hit_count`.
- A global accounting thread samples resident size and rejects admission if the bank would overflow.

Because 4-bit affine tensors require dequantization scales, each slot must also store scale/zero-point metadata; this should be included in `slot_size`.

## 4. Positional I/O and Integrity Checks

NVMe reads should be issued via aligned, large-block reads (e.g., 64–256 KB) to amortize syscall and controller overhead. For each expert file or sidecar region:

- Store a SHA-256 or BLAKE3 checksum per expert tensor in a manifest.
- On load, verify the checksum before marking the slot ready.
- Use `pread` with explicit offsets to avoid seek overhead.
- For component-major safetensor shards, an expert may span many disjoint byte ranges; prefer `preadv` or a sidecar.

Integrity failures must not silently corrupt inference; a bad load should evict and retry once, then fail the request if unrecoverable.

## 5. Cache Admission and Eviction

Decode hot cache uses frequency-decayed admission:

- Each expert maintains an exponential decay counter: `score = score * 0.99 + 1` on hit.
- Admission requires either a miss with high projected reuse or a decay score below a threshold for an existing resident cold expert.
- Eviction selects the lowest decay score among non-pinned, non-transient slots.
- Prefill transient slots bypass decay and are freed immediately after prefill.

This prevents a single large prompt from evicting the decode working set, satisfying the isolation requirement.

## 6. Failure Handling

The system must handle:

- NVMe read errors: retry once, then mark expert unavailable and route around it if the model allows fallback to shared experts.
- Corrupt tensor: checksum mismatch triggers reload; repeated failure disables that expert and alerts ops.
- Slot exhaustion under bursty prefill: reject new prefill or queue it; never steal decode slots.
- Unified memory pressure: watchdog detects reclaim and re-pins; if impossible, shed load.
- Router selecting an expert not present on disk (version mismatch): fail fast with a clear error.

## 7. Observability

Instrument:

- Per-layer cache hit/miss ratios.
- NVMe read latency histogram.
- Slot residency and eviction events.
- Decode token throughput and expert-load stall time.
- Checksum failure counts.

Export via Prometheus or a local stats socket; include per-request expert-load traces for debugging.

## 8. Correctness Testing

Tests should cover:

- Numerical equivalence between full-memory and NVMe-backed expert execution (within 4-bit tolerance).
- Router determinism with and without cache warming.
- Eviction correctness: no token computes against a partially loaded expert.
- Prefill/decode isolation: a huge prefill does not evict decode experts.
- Corruption injection: flipped bytes on disk cause checksum failure, not silent error.

## 9. Safetensor Shards vs Expert-Major Sidecar

**Component-major safetensor shards** store the model in its original save layout, often with experts interleaved or laid out by layer in one large file. Reading an expert requires seeking to multiple offset ranges. This is simple for provenance but causes many small or scattered reads, hurting NVMe throughput and increasing latency variance.

**Expert-major sidecar** reorganizes experts into contiguous, self-describing files (or one file with a directory). This makes loads sequential and checksum-friendly, but requires a one-time conversion step and extra disk space. It also decouples serving layout from training checkpoint format.

Tradeoff: sidecar improves performance and integrity handling at the cost of conversion tooling and duplication. For production, we recommend an optional sidecar generated at deploy time, with fallback to direct shard reads.

## 10. Five Concrete Failure Modes and Mitigations

1. **NVMe controller saturation during bursty decode**
   Mitigation: rate-limit expert loads; prioritize in-flight decode over prefill loads.

2. **Unified memory reclaim of pinned bank under pressure**
   Mitigation: periodic touch pages; monitor resident set; shed load if re-pin fails.

3. **Checksum mismatch due to silent disk corruption**
   Mitigation: per-expert checksums; reload and disable expert on repeated failure.

4. **Router references expert ID beyond stored set (version drift)**
   Mitigation: validate manifest at startup; fail request with clear version error.

5. **Prefill transient slot leak after early termination**
   Mitigation: RAII/scoped slot handles; background reaper for orphaned slots.

## 11. Staged Rollout Plan

- **Stage 0 (Shadow):** Load experts from NVMe but keep full model in memory; compare outputs.
- **Stage 1 (Decode-only NVMe):** Enable NVMe experts for decode; keep prefill in memory.
- **Stage 2 (Prefill transient):** Enable isolated prefill slots.
- **Stage 3 (Full sidecar):** Switch to expert-major sidecar if metrics justify.
- **Stage 4 (Full production):** Enable eviction and decay tuning.

## 12. Go/No-Go Criteria

Go if:

- Decode throughput within 5% of all-in-memory baseline.
- p99 expert-load stall < 20 ms.
- Zero silent corruption events over 24h.
- Prefill isolation verified under adversarial sizing.

No-go if:

- Unified memory instability under load.
- Checksum failures exceed 0.1% of loads.
- Router/expert mismatch occurs in testing.

This design is sound provided slot isolation, integrity checks, and observability are implemented before Stage 2.
