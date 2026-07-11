# Engineering Review: Serving a Very Large MoE LLM on Apple Silicon with NVMe-Backed Routed Experts

## 1. Context and Design Summary

We are reviewing a production design for serving a very large mixture-of-experts (MoE) language model on a single Apple Silicon workstation. The machine uses unified memory (CPU/GPU/shared RAM on the same bus) and a local NVMe SSD. The router, attention layers, embeddings, normalization layers, and any shared experts are resident in unified memory at all times. Routed expert weights are too large to fit in RAM and are stored as affine 4-bit tensors on NVMe. At each sparse layer, the router selects eight experts; if an expert is not already present, its weights are loaded from disk into a fixed, user-bounded memory bank. Decode requests use a frequency-decayed hot cache; prompt prefill uses transient slots so a large prefill cannot evict the decode working set.

This review covers execution flow, concurrency, memory accounting, I/O integrity, cache policy, failure handling, observability, correctness, the safetensor vs. sidecar tradeoff, failure modes, and rollout.

## 2. Execution Sequence

For a decode step:
1. Token embedding lookup in unified memory.
2. For each transformer block:
   a. Attention and normalization in RAM.
   b. Router computes expert indices (8 per token group/layer).
   c. For each selected expert: check hot cache; on miss, issue async NVMe read into a pinned slot in the bounded memory bank.
   d. While experts load, compute shared expert and attention projections that do not depend on missing routed experts.
   e. Once all 8 experts for the layer are resident, execute expert MLP.
3. Final norm and LM head in RAM.

For prefill:
1. Same as above, but expert slots are taken from transient pool, not hot cache.
2. Transient slots are freed immediately after the prefill pass.

## 3. Concurrency and Slot Pinning

The memory bank is divided into:
- Hot cache: N pinned slots, owned by decode workers.
- Transient pool: M slots, owned by prefill workers.

Slot pinning must be reference-counted. A decode batch may hold a hot slot for several milliseconds; the loader must not evict it. Use a lock-free slot table with atomic state (FREE, LOADING, READY, PINNED). Prefill workers never touch hot slots. Decode workers never use transient slots. This isolates latency spikes from prefill from degrading decode tail latency.

## 4. Memory Accounting

The bounded memory bank size must be computed from `sizeof(quantized_expert) * (hot_slots + transient_slots)` plus alignment padding. On Apple Silicon, use `mlock`/`vm_allocate` with `PURGABLE` off for hot slots to avoid compression. Track peak RSS and fail fast if allocation exceeds the user bound. Expose `used_bytes`, `hot_bytes`, `transient_bytes` as live metrics.

## 5. Positional I/O and Integrity Checks

Each expert tensor is stored as an affine 4-bit blob: scale/zero per block plus packed nibbles. On load:
- Read header (offset, length, sha256).
- Read body; verify checksum before marking READY.
- If checksum fails, mark slot FAILED and retry from replica or fall back to a shared expert approximation (if safe).

Use direct I/O (`O_DIRECT` via a wrapper) to avoid page-cache pollution, since experts are large and rarely reused within a prefill. Align buffers to 4 KiB.

## 6. Cache Admission and Eviction

Hot cache uses frequency-decayed admission:
- On decode miss, increment counter; admit only if counter exceeds threshold within window.
- Eviction uses LRU-with-decay; never evict pinned slots.
- Transient slots are not admitted to hot cache.

This prevents a one-time prefill from polluting the decode working set.

## 7. Failure Handling

- Disk read error: retry once; on second failure, disable expert and route to shared expert.
- Corrupt tensor: checksum fails; quarantine file path; alert.
- Slot exhaustion: prefill waits or rejects; decode never waits for prefill slots.
- Crash mid-load: slot left LOADING; watchdog resets to FREE after timeout.
- Unified memory pressure: admit fewer hot experts; degrade gracefully.

## 8. Observability

Emit:
- `expert_load_latency_ms` (p50/p99).
- `cache_hit_ratio` per layer.
- `nvme_queue_depth`.
- `slot_state_transitions`.
- `checksum_failures`.

Use Instruments or `os_signpost` for timeline debugging on Apple Silicon.

## 9. Correctness Testing

- Unit test router→slot mapping determinism.
- Fuzz test loader with truncated files.
- Differential test: compare 4-bit NVMe expert output to in-RAM fp16 reference on fixed prompts.
- Integration test: kill NVMe sim and verify fallback.

## 10. Safetensor Shards vs. Expert-Major Sidecar

Component-major safetensor shards store experts interleaved with other weights. Reading one expert requires seeking across the file. An optional expert-major sidecar stores each expert contiguously. Tradeoff: sidecar improves read locality and reduces I/O amplification, but duplicates data and must be kept consistent. For 4-bit affine blobs, sidecar is recommended if prefill latency matters.

## 11. Five Concrete Failure Modes

1. NVMe saturation from parallel prefill: mitigate with transient slot cap.
2. Silent tensor corruption: mitigate with per-expert checksums.
3. Hot-cache thrash from bursty decode: mitigate with decay admission.
4. Slot leak from exception: mitigate with RAII/scoped guards.
5. Unified memory oversubscription: mitigate with hard RSS limit and reject.

## 12. Staged Rollout and Go/No-Go

Stage 1: Shadow mode, NVMe disabled, all experts in RAM.
Stage 2: Enable sidecar reads for prefill only.
Stage 3: Enable decode hot cache.
Stage 4: Full production.

Go: p99 expert load < 5 ms; checksum fail rate < 1e-6; no decode stalls from prefill.
No-go: RSS exceeds bound; repeated corruption; prefill evicts decode slots.

This design is sound if isolation and integrity are enforced as specified.
