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

Because 4-bit affine tensors require dequantization scales, each slot must also store scale/zero-point metadata, either inline or via a sidecar index.

## 4. Positional I/O and Integrity Checks

NVMe reads should be issued at expert granularity using aligned, large-block reads (e.g., 64–256 KB) to amortize syscall overhead. Each expert file (or sidecar segment) should include:

- A header with `model_version`, `layer_id`, `expert_id`, `dtype`, `shape`, `checksum`.
- A trailing BLAKE3 or SHA-256 integrity tag over the raw bytes.
- On load, the checksum is verified before the slot is marked ready. Corrupt loads are discarded and retried from a replica path if available.

Affine 4-bit layouts must be parsed exactly as produced by the export tool; any mismatch in block size or packing order is a silent-correctness hazard.

## 5. Cache Admission and Eviction

Decode hot cache uses frequency-decayed admission:

- Each expert hit increments a counter with exponential decay (`score *= 0.999 per ms`).
- Admission requires either a miss with active batch demand or a decayed score above a threshold.
- Eviction selects the lowest decayed-score ready slot not currently in use.
- Prefill transient slots bypass decay and are never promoted to hot cache.

This protects the decode working set from a single large prompt and keeps long-lived experts resident.

## 6. Failure Handling and Concrete Failure Modes

**Failure Mode 1: NVMe read latency spike during decode.**
Mitigation: Bound per-step load wait with a timeout; if exceeded, defer token to a retry queue and continue with available experts using a fallback shared-expert approximation, logging a degraded output.

**Failure Mode 2: Corrupted expert tensor on disk.**
Mitigation: Checksum verification on load; quarantine the file and fall back to a replicated copy or to a CPU-dequantized recompute path.

**Failure Mode 3: Unified memory pressure causes OS to reclaim bank pages.**
Mitigation: Memory prioritization plus a reclaim watchdog that re-loads evicted hot experts and alerts if reclaim rate exceeds threshold.

**Failure Mode 4: Router selects an expert ID outside the valid range (export bug).**
Mitigation: Validate expert indices against the layer's exported expert count before I/O; treat out-of-range as a hard error with panic-in-dev, fallback-in-prod.

**Failure Mode 5: Concurrent prefill and decode contend on the same NVMe queue.**
Mitigation: Separate NVMe submission queues or strict prioritization of decode loads; prefill uses idle bandwidth only.

**Failure Mode 6: Sidecar index drift from main safetensors.**
Mitigation: Store a shared manifest hash linking both; refuse to boot if mismatch.

## 7. Observability

Expose:

- Per-layer cache hit/miss ratios.
- NVMe read latency histograms.
- Bank occupancy and eviction counts.
- Decode fallback occurrences.
- Checksum failure counts.

Use a lightweight in-process metrics ring buffer exported via HTTP or Unix socket.

## 8. Correctness Testing

- Unit tests for 4-bit affine pack/unpack against reference FP16.
- Golden-output tests on a fixed prompt set with full-memory baseline vs NVMe-backed.
- Chaos tests: inject disk corruption, latency, and memory pressure.
- Differential tests comparing top-8 expert selection with and without caching.

## 9. Safetensor Shards vs Expert-Major Sidecar

Reading directly from component-major safetensor shards keeps a single source of truth and simplifies export. However, it requires seeking and extracting small expert blocks from large files, increasing NVMe random reads and parse cost.

An optional expert-major sidecar stores each expert as a contiguous file (or segment), enabling large sequential reads and simpler integrity tagging. Tradeoff: duplicate storage and a manifest-coupling risk. We recommend a sidecar for production if NVMe capacity allows, with the safetensor shard as the authoritative source for rebuild.

## 10. Staged Rollout Plan

- **Stage 0:** All experts in unified memory (baseline).
- **Stage 1:** NVMe backing with sidecar, decode only, small batch.
- **Stage 2:** Prefill transient slots enabled.
- **Stage 3:** Full concurrency and eviction tuned.
- **Stage 4:** Chaos testing in production hours.

**Go/No-Go Criteria:**
- Decode p99 latency within 1.2x of Stage 0.
- Zero silent checksum passes on corrupt data.
- Bank occupancy stable below limit for 24h.
- No unexplained fallback rate above 0.1%.

If all criteria hold, proceed; otherwise halt and review.
