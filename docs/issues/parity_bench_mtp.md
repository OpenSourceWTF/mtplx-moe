> **Repository scope:** This is a repository-wide `davidtai/MTPLX` issue.
> **Target branch:** `codex/moe-ssd-hy3-glm52`.
> **Implementation status:** AR streaming and reproducible I/O, generation, and
> golden-logit harnesses now exist. Tiny parity/failure tests pass. Full artifact
> results remain deliberately unpublished, and MTP remains disabled because the
> pinned Q4 artifacts omit its weights.

## Objective

**Status: NOT IMPLEMENTED (verified 2026-07-18).** Every deliverable is absent -- the `tests/parity/`, `tests/integration/`, and `docs/benchmarks/` directories do not exist, and no parser accepts the `--mode ar|mtp` or `--mtp-manifest` flags described below. Kept as a design record.

Build a reproducible correctness and performance gate for Hy3 Q4 and GLM-5.2
Q4 expert streaming. Ship autoregressive (AR) streaming first. Treat MTP as a
separate artifact and correctness milestone, because both community Q4 targets
omit the official next-token prediction layer.

Performance is meaningful only after proving the layer-local order and routing
math:

```text
resident router[n] -> top-k -> bounded load of misses -> expert compute[n]
```

It is invalid to route all layers up front, because later routers depend on
earlier hidden states.

## Proposed files and fixtures

- `tests/parity/test_hy3_streaming_parity.py`
- `tests/parity/test_glm52_streaming_parity.py`
- `tests/parity/test_router_oracles.py`
- `tests/integration/test_streaming_failures.py`
- `benchmarks/benchmark_expert_streaming.py`
- `benchmarks/prompts/moe-streaming.jsonl`
- `benchmarks/results/moe-streaming/<hardware>/<revision>.json`
- `docs/benchmarks/moe-ssd-streaming.md`

Every result JSON should include code commit, immutable model revision, manifest
digest, sidecar digest if used, macOS/MLX/`mlx-lm` versions, machine/SSD, memory
plan, context length, batch size, prompt hash, cache start state, and raw sample
measurements. Report medians and distributions, not one best run.

## Correctness ladder

1. **Manifest parity:** selected source tensor slices equal manifest/sidecar
   records byte-for-byte.
2. **Router parity:** official/reference top-k IDs and routing weights equal the
   MTPLX resident-router path for fixed hidden states and near-tie fixtures.
3. **Expert parity:** one expert loaded into a slot matches the existing
   full-resident Q4 expert output within a documented tolerance.
4. **Layer parity:** full sparse layer output matches with hits, transient misses,
   persistent admission, and eviction.
5. **Model parity:** fixed-prompt logits and generated tokens match an equivalent
   full-resident/reference path for a tractable model slice or sufficiently
   provisioned host.
6. **Failure parity:** corruption, missing records, cancellation, and low-memory
   plans fail deterministically without changing to a fallback execution path.

For GLM-5.2, compare FP32 router behavior against the official Transformers
implementation, including correction-bias selection and unbiased selected
weights. Test IndexShare at sequences longer than 2,048 and 4,096 tokens and
assert 21 full indexer layers plus 57 reuses. DSA token-index selection is not
an MoE cache hit and must be evaluated separately.

## Benchmark matrix

- Cold cache, warmed cache, and hot-repeat prompt sets.
- Prefill then single-stream decode; concurrent requests; bounded microbatches.
- At least three persistent-cache sizes, including zero slots and a realistic
  user threshold, plus a full-resident/simulator reference where feasible.
- Context lengths that expose KV/DSA tradeoffs. GLM-5.2 cache is exactly 95,232
  bytes/token (about 2.906 GiB at 32K and 11.625 GiB at 128K); Hy3 is 327,680
  bytes/token.
- Direct safetensors segments versus aligned sidecar.
- Effective SSD bandwidth reported from actual expert bytes and wait time.

Required measurements: time-to-first-token, prefill tokens/s, decode tokens/s,
p50/p95 token latency, route/cache hit rate, bytes read/token, I/O queue/read/wait
time, router/expert compute time, evictions, errors, and observed peak unified
memory. Include cold theoretical context, not as a measured claim: Hy3 needs
6.249 GiB and GLM-5.2 needs 11.865234 GiB of expert data for a completely cold
token.

## MTP scope and gates

- Hy3's community final Q4 omits MTP layer 80.
- GLM-5.2's community Q4 omits MTP layer 78 despite
  `num_nextn_predict_layers=1` metadata.
- AR model loading and generation must not depend on either MTP artifact.
- Generate any supported MTP Q4 artifact from pinned official weights with the
  same quantization/manifest validation pipeline; do not accept ambiguous
  community conversion metadata as the default.
- MTP has its own router/expert bank and shares only architecture-defined
  embeddings/output head. Never create a randomly initialized replacement head.
- Add MTP only after AR route/logit/token parity and memory-bound tests pass.

Proposed mode contract:

```text
--mode ar              # works with target Q4 artifact alone
--mode mtp             # requires explicit validated MTP manifest/artifact
--mtp-manifest PATH
```

## Failure handling

- Mark a run invalid if the artifact/manifest/code revision, cache start state,
  memory plan, or raw samples are absent.
- Stop benchmark collection on route/logit parity failure, checksum failure,
  unplanned allocation, thermal throttling outside a declared band, or runtime
  fallback. Preserve diagnostics and label the result failed.
- Reject MTP mode before target model allocation if required explicit MTP keys,
  shared-weight aliases, shapes, dtypes, or manifest provenance are missing.
- Do not publish performance claims extrapolated from other quantization levels,
  hardware, or third-party implementations as MTPLX measurements.

## Acceptance criteria

- [ ] The six-step correctness ladder passes for both Hy3 and GLM-5.2 AR paths.
- [ ] GLM FP32 router near-tie and IndexShare long-context fixtures pass against
      the official oracle.
- [ ] Fault injection proves no corrupt/stale/partial slot reaches expert
      compute and no missing MTP key triggers broad shard loading.
- [ ] Peak memory remains within the user plan for every benchmark cell,
      including cancellation and concurrent-request cases.
- [ ] Result files contain full provenance and raw measurements and can be
      regenerated from a documented command.
- [ ] Cold/warm/cache-size comparisons report all required latency, throughput,
      I/O, cache, and memory metrics.
- [ ] AR is documented as supported independently; MTP remains disabled unless
      its separate parity suite passes with an explicit validated artifact.

## Dependencies

- Depends on both model adapters, manifest, native loader, memory planner, and
  runtime cache integration.
- AR validation blocks any MTP implementation or performance claim.
- Benchmark evidence is the final epic/release gate, not an implementation
  shortcut.
