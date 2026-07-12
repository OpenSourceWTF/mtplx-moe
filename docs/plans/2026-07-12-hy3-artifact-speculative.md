# Hy3 Artifact and Speculative Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide isolated, reproducible experiments for GPU-oriented expert packing, KV/cache budget exchange, hint-only route prefetch, and a separately labeled lower-bit cold tier.
**Architecture:** Build on PR 2 but keep each arm behind a separate selector and benchmark label. No arm may depend on another arm being enabled. Full artifact migration occurs only after a synthetic layout win.
**Tech Stack:** Python binary codecs, manifest validation, MLX KV quantization, expert runtime hint API, pytest, quality/performance harnesses.
**Assumptions:** Assumes the current sidecar remains authoritative v1 input; this plan will NOT rewrite 161 GB before a layer-local proof. Assumes lower-bit MLX affine kernels are available for the chosen bits; unsupported hardware will fail closed rather than dequantize silently.

---

### Task 1: Add a reversible synthetic v2 codec

- [ ] Create failing property tests for packing/unpacking every v1 component, 144-byte four-group alignment, gate/up pairing, independently addressable down rows, corruption, truncation, and layout-ID mismatch.
- [ ] Create `mtplx/expert_record_v2.py` with `pack_v1_record`, `unpack_v2_record`, a versioned header, and canonical reconstruction.
- [ ] Add a layer-local benchmark that reports pack cost, stream count, bytes, and kernel time without building a full sidecar.
- [ ] Promote a full exporter task only if the synthetic arm clears 5%; otherwise document no-go.

### Task 2: Add a KV-to-expert budget experiment

- [ ] Write failing memory-plan tests for BF16 and Q8 KV byte accounting at 4K/16K/64K/128K while holding the total limit fixed.
- [ ] Add an explicit KV storage descriptor to the expert memory plan; never infer savings from a label.
- [ ] Reallocate only proven KV savings to persistent expert slots and expose both byte counts in telemetry.
- [ ] Run attention parity/quality plus expert-hit/bytes benchmarks as an isolated `kv-q8` arm.
- [ ] Do not evaluate Q4 KV until Q8 has an accepted quality method.

### Task 3: Add an authoritative-router-safe prefetch API

- [ ] Write failing tests that predicted IDs may warm transient/prefetch state but cannot produce a hit unless the authoritative route later selects them, cannot evict pinned or decode-hot authoritative records, and yield identical tokens when wrong.
- [ ] Add `ExpertStreamingRuntime.prefetch_hint(layer, expert_ids, source, deadline)` with separate counters for requested, loaded, useful, wasted, cancelled, and amplified bytes.
- [ ] Give authoritative misses priority and cancellation over speculative reads.
- [ ] Add offline prior-token/transition predictors first; make MTP guidance a separate adapter charged for head time and resident bytes.
- [ ] Evaluate held-out recall and net latency; disable any predictor that misses the 5% gate or amplifies bytes.

### Task 4: Add a separately branded lower-bit cold tier

- [ ] Write failing artifact-identity tests proving Q2/Q3 records cannot claim the Q4 model key or digest.
- [ ] Extend the manifest/model descriptor with an explicit cold-tier quantization identity and supported-kernel check.
- [ ] Keep hot records Q4 and apply lower bits only to the opt-in cold tier.
- [ ] Add perplexity, deterministic continuation, reasoning/tool-use fixture, and long-generation quality reports.
- [ ] Reject silent dequantization, unbounded promotion copies, or quality results without the matching performance artifact.

### Task 5: Verify and publish PR 3

- [ ] Run each experiment alone against PR 2 with identical machine and memory settings.
- [ ] Run full pytest and changed-file Ruff.
- [ ] Save separate result files for layout, KV, prefetch, and precision.
- [ ] Document go/no-go independently; do not combine sub-5% arms into one claimed win.
- [ ] Push `experiment/hy3-artifact-speculative` and open a draft PR against `experiment/hy3-record-native-exec`, linking #31.
