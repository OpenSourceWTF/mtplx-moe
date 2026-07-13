# Hy3 cache scheduling experiment (#29)

The current immediate-predecessor comparison measures default `e7669d43a8033e4c0cf4d4aea93563967481be75` against #29 head `bf34a0ad917495db386edb90e27fd1de42da8929` on an Apple M5 Max with 128 GB unified memory. Earlier same-commit cache-scope measurements at `aef62bc0e4ff0c716933dd16a2acd0154f112a93` remain useful isolation evidence. A conservative serialized control at `058b40b28de97ddce23184e2c62ed1f41cb6ba2d` was rejected because its resource-wide exclusion premise was not established. All comparable headline runs use the same Hy3 Q4 artifact, deterministic AR generation, component-bank layout, LRU policy, 7,821 persistent slots, 32 transient slots, and an exact 83,034,243,072-byte expert-cache ceiling.

## Decision

**Retain #29.** The fixed 5% promotion gate is retired: it had no empirical noise model or causal basis. The replacement rule requires identical output, positive paired direction, repeatability small relative to the observed effect, and no material correctness or concurrency regression. If those checks disagree, collect more pairs rather than deciding from a fixed percentage.

The current telemetry-off pairs are +4.265% and +4.135%, with a +4.200% mean. Base run spread is 0.498% and #29 spread is 0.373%, while all four runs stop at 1,905 tokens with the same SHA-256 `484e182a68604821f69d56d0b15488d26723e6123f6a57f8158f8b20a4c6ed1c`. The earlier same-commit cache-scope pairs were also positive (+5.043% and +4.674%). This is repeatable positive evidence, not a universal throughput guarantee.

The bounded transaction-journal refactor remains useful independently. It removes an accidental O(cache capacity) Python snapshot from every global route while preserving exact rollback behavior. The global allocation also cuts physical expert reads; the end-to-end effect is modest but repeatable under the matched protocol.

## Current immediate-predecessor headline

Resource telemetry, rolling slot snapshots, and Qwen were disabled during the timed lanes. The pair order was base/global/global/base to expose drift.

| Pair | Default layer tok/s | #29 global tok/s | Gain | Default elapsed | #29 elapsed |
|---|---:|---:|---:|---:|---:|
| 1 | 6.1534 | 6.4159 | +4.265% | 309.583 s | 296.921 s |
| 2, reverse order | 6.1840 | 6.4398 | +4.135% | 308.051 s | 295.818 s |
| Mean | **6.1687** | **6.4278** | **+4.200%** | - | - |

## New resource diagnostic

A separate 256-token diagnostic at `bf34a0a` used the same #29 global-cache configuration with `--resource-telemetry --ssd-ceiling-gib-s 12.47`. Its 4.817 tok/s is not headline timing evidence.

| Signal | Measurement | Interpretation boundary |
|---|---:|---|
| Uncached reader throughput | 6.065 GiB/s (48.64% of supplied ceiling) | Device ceiling was not reached |
| Mean active readers | 3.732 / 32 (11.66%) | Reader pool had substantial unused capacity |
| Mean queued reads | 0.377 | Queue was shallow |
| Queue-nonempty intervals | 99.52% | Work was usually present despite shallow depth |
| Logical reader operations | 31,684 (123.77/token) | Reader jobs, not kernel syscall count |
| Synchronous fences | 39,910 (155.90/token) | Operation count; it does not prove fence wall-time dominance |
| Asynchronous fence occupancy | 0.0 mean active fences | The diagnostic could not measure simultaneous I/O/Metal overlap |

GPU and DRAM-bandwidth coverage were unavailable. Attribution therefore remains `incomplete`; the report routes follow-up toward fence/evaluation placement and increasing useful read concurrency without asserting either as the bottleneck.

## Rejected serialized-control premise

Commit `058b40b` tested a conservative schedule that drained every component-bank miss before any Q4 dispatch. It was motivated by a hypothesis that CPU writes to one logical row required exclusion from Metal reads of disjoint sibling rows in the same MLX allocation.

The measured overlapped runs retained exact token parity and reported no output-integrity failure; no attributable crash was documented. The serialized control itself excluded the disputed overlap and therefore could not validate the hypothesis. The existing per-slot path already prevents same-slot replacement until its Metal consumer completes, so the resource-wide exclusion premise remains unproven. The barrier is removed rather than promoted, and the original exact-parity benchmark payload remains the promotion evidence for this cache-policy experiment.

The corrected layer-scope B1 control retained exact output parity: both repeats stopped naturally at 1,905 tokens and produced SHA-256 `484e182a68604821f69d56d0b15488d26723e6123f6a57f8158f8b20a4c6ed1c`.

| Repeat | Corrected layer tok/s | Elapsed | TTFT |
|---|---:|---:|---:|
| 1 | 2.9332 | 649.464 s | 16.338 s |
| 2 | 2.9757 | 640.181 s | 10.239 s |
| Mean | **2.9545** | - | - |

The serialized-control mean is 51.587% below the original layer mean of 6.1027 tok/s while reading the same 1,768,689,893,376 expert bytes in repeat 1. It documents the cost of draining all misses before compute, but it does not establish that the original schedule was unsafe. The remaining serialized global and paired arms were stopped because the control itself was not viable. A prior 458-token launch omitted `--chat`; it is retained locally as harness-debugging evidence and excluded from every comparison.

## Historical same-commit cache-scope headline

Each run stopped naturally after the same 1,905 generated tokens. All four outputs have SHA-256 `484e182a68604821f69d56d0b15488d26723e6123f6a57f8158f8b20a4c6ed1c`.

| Pair | Layer tok/s | Global tok/s | Gain | Layer TTFT | Global TTFT |
|---|---:|---:|---:|---:|---:|
| 1 | 6.0932 | 6.4005 | 5.043% | 14.690 s | 14.756 s |
| 2, reverse order | 6.1121 | 6.3978 | 4.674% | 14.786 s | 14.854 s |
| Mean | 6.1027 | 6.3991 | **4.858%** | - | - |

Run spread is 0.311% for layer scope and 0.042% for global scope. Both paired directions are positive; the former fixed 5% cutoff is no longer used to discard that evidence.

Pair 1 cache economics:

| Metric | Layer | Global | Change |
|---|---:|---:|---:|
| Expert bytes read | 1,768,689,893,376 | 1,630,883,414,016 | -7.79% |
| Persistent loads | 160,979 | 147,999 | -8.06% |
| Evictions | 153,158 | 140,178 | -8.47% |
| Physical read operations | 165,678 | 152,698 | -7.83% |
| Decode hit rate | 87.272% | 88.351% | +1.079 pp |
| Peak MLX memory | 89,145,312,888 B | 89,145,312,888 B | unchanged |

## Saturation and mixed traffic

The short saturation lanes use 64 generated tokens per stream. Every lane achieved its requested peak concurrency and reports `saturation_valid=true`, `undersubscribed=false`.

| Workload | Layer tok/s | Global tok/s | Gain | Layer TTFT p50 | Global TTFT p50 | Layer completion p95 | Global completion p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static B2 | 3.9478 | 4.0274 | 2.017% | 10.587 s | 10.446 s | 31.692 s | 31.054 s |
| Static B4 | 5.4090 | 5.5747 | 3.063% | 6.675 s | 6.346 s | 45.112 s | 43.737 s |
| Static B8 | 6.7095 | 6.9017 | 2.864% | 6.592 s | 6.351 s | 71.162 s | 69.094 s |
| Mixed-join B4 | 5.3758 | 5.4436 | 1.261% | 6.607 s | 6.450 s | 44.870 s | 44.303 s |

The mixed lane starts one decoder, submits the other streams after decode step 2, and permits one joining prefill per step. Its mean live-stream count is 3.706 and peak is 4. Tests separately pin the invariant that a joining global prefill cannot evict the live decoder's hot set.

## Held-out capacity analysis

The diagnostic trace contains 254 decode steps and 160,528 assignment requests. At B1 every top-8 union contains eight unique experts, so there is no within-step assignment sharing to exploit. A train-only dynamic quota uses the same 7,821 slots and 83,034,243,072 bytes as the uniform baseline. It gains 286 training hits, but only 22 held-out hits: +0.000271954 hit rate, or +0.0272 percentage points. This same-prompt chronological suffix is diagnostic evidence only and does not justify changing production quotas.

## Transaction hot path

The original global transaction copied the full directory, generation map, free-slot deque, occupancy counter, and LRU state for every routed layer call. At 7,821 persistent slots, a local Python microbenchmark measured:

| Route | Before | Bounded journal |
|---|---:|---:|
| All hit | 297.828 us | 8.192 us |
| Forced miss | 2,453.428 us | 6.128 us |

This is a host-bookkeeping microbenchmark, not an end-to-end throughput claim. The refactor uses monotonic LRU ranks and journals only touched entries; rare rollback rebuilds the exact LRU ordering. Review fuzzed 144,000 randomized transitions across LRU/frequency policies, capacities 0/1/2/4/8/12, prefill/decode, commit/rollback, partial load admission, invalidation, generation reconciliation, and reset. It also ran 5,000 exact rollback cases and 3,000 partial-admission cases against detached reference state.

## Reproduction

The common command is:

```bash
MODEL="$HOME/.cache/huggingface/hub/models--pipenetwork--Hy3-4bit/snapshots/160619d3f96c8470350b6dac0ef033a8381551e3"
SCOPE=layer       # repeat with global
BATCH=1           # repeat with 2, 4, and 8
TOKENS=2048       # use 64 for B2/B4/B8 and mixed-join
LABEL="issue29-${SCOPE}-B${BATCH}-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short=12 HEAD)"
OUT="benchmarks/raw/moe-runtime/$LABEL"

uv run --frozen --extra dev --extra server python \
  scripts/benchmark_streamed_generation.py \
  "$MODEL" "$MODEL/expert-manifest-sidecar.json" \
  --model-key hy3-q4 \
  --memory-limit 120259084288 \
  --runtime-reserve 8589934592 \
  --expert-cache-limit 83034243072 \
  --max-live-kv-tokens 18888 \
  --cache-policy lru \
  --cache-scope "$SCOPE" \
  --slot-layout component-banks \
  --transient-slots 32 \
  --read-chunk 67108864 \
  --f-nocache \
  --trust-sidecar \
  --no-enable-mtp \
  --chat \
  --prompt-file benchmarks/prompts/moe_streaming_realistic.md \
  --generation-profile deterministic \
  --max-tokens "$TOKENS" \
  --concurrency "$BATCH" \
  --max-prefills-per-step 1 \
  --workload-shape static \
  --no-window-telemetry \
  --run-label "$LABEL" \
  --output-dir "$OUT" \
  --output-json "$OUT/result.json"
```

For mixed traffic, change `--workload-shape static` to `--workload-shape mixed-join --join-after-step 2`. The paired long runs were launched in alternating order. The complete local payloads are retained under `benchmarks/raw/moe-runtime/` with the run labels recorded in the adjacent JSON summary.

## Verification and limits

- Full `pytest -q` reached 100% at current head `bf34a0a`; only pre-existing skips and deprecation warnings remained.
- Two independent post-fix reviews passed. They specifically verified that telemetry-disabled static, mixed-join, AR-reference, and MTP-reference timed regions retain their predecessor operations.
- Ruff passed on every changed Python file.
- Qwen was stopped only for the exclusive MLX window, then restored with `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tea.qwen.plist`. `launchctl print gui/$(id -u)/com.tea.qwen` reported the service running and `/v1/models` returned `mtplx-qwen36-27b-optimized-speed`.
- The new diagnostic retained 208 samples with zero drops and zero sampler failures. GPU and DRAM bandwidth remained unavailable; neither is treated as zero.
- Historical process samples averaged roughly 55-57% CPU and 82.7 GiB RSS; instantaneous `ioreg` samples showed 0% GPU during sampled I/O-heavy intervals. These are not full-run GPU-utilization claims.
- This issue is AR-only. MTP, kernels, serialization changes, KV precision, sidecars beyond the existing verified artifact, and lower-bit tiers are out of scope.
