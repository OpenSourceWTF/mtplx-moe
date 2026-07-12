# MoE Runtime PR Benchmark Report

Date: 2026-07-11
Remote base: `origin/codex/moe-ssd-hy3-glm52@4146f72`
Current sequential tip: `fb4c1d5f6bcdf7a96f7685d372e522d5f25f0846`

## Result

The original independent audit retained only PR #13. The approved sequential
salvage is rebuilding and repairing each rejected candidate on the latest
retained tip. Current count: **2 retained (#13 and repaired #15), 1 repaired but
rejected on measurement (#12), and 5 pending repair/re-gate**. This report will
become the requested consolidated seven-PR report as each remaining gate
completes; prior failure findings remain below so the repair delta stays
auditable.

| PR | Clean candidate | Correctness result | Performance result | Decision |
| --- | --- | --- | --- | --- |
| #11 | `3d7c158` | 50 focused passed; 1,980 passed / 4 skipped full suite; semantic hotset and request-lifecycle failures | Not run: correctness stopped the gate | Skip |
| #12 | `a8ef882` -> repaired `124f4ce` | Exact 128K Q8 accounting and 80-cache attestation repaired; concurrency/env/close/server bypasses fail closed; 2,033 passed / 4 skipped; both reviews approved | Token-identical short lane: 4.7052 -> 0.9896 tok/s, **-78.97%**; reads -27.03% but peak MLX +20.97 GB. Runner hooks classify the candidate as serialized, not bandwidth-bound. | **Skip; do not promote the 100-slot plan** |
| #13 | `939fe57` | 43 focused passed; 1,978 passed / 4 skipped full suite; independent reviews approved | 6.0523 -> 6.5033 decode tok/s mean, **+7.45%** over two matched pairs; token-identical | **Retain** |
| #14 | `c424381` | 49 focused passed; 1,979 passed / 4 skipped full suite; miss failure can hang rollback and leak pins | Not run: correctness stopped the gate | Skip |
| #15 | `a5be248` + repair `e0e93b0` | RED reproduced tuple/shared-work, route-wave, and pin-cleanup failures; GREEN 40 focused passed; 1,985 passed / 4 skipped full suite; both reviews approved | Six balanced pairs: decode mean 6.5446 -> 6.5549 tok/s, **+0.16%**; median +0.23%; 4/6 positive; token/counters identical | **Retain at `fb4c1d5`**; effect is small and order-sensitive |
| #16 | `4106348` | Exact focused gate: 68 passed / 2 failed; additional device-fence and policy-accounting failures | Not run: correctness stopped the gate | Skip |
| #17 | `faf1527` | 51 focused passed; 1,981 passed / 4 skipped full suite; asynchronous fence errors are not fail-closed | Not run: correctness stopped the gate | Skip |
| #18 | `8743a93` | 36 focused passed; 1,985 passed / 4 skipped full suite; storage can close while partial Metal work remains pinned | Not run: correctness stopped the gate | Skip |

"Not run" is a gate result, not an estimated zero. Hardware performance was
intentionally not measured after a candidate failed correctness, because a fast
unsafe implementation is not promotable.

## Gate contract and fixed lane

For this salvage series, the approved design in
`docs/specs/2026-07-11-moe-runtime-pr-salvage-design.md` supersedes the old 5%
minimum. A throughput candidate is retainable when its repeated matched mean
and median are positive, direction is not systematically reversed, deterministic
token parity and the candidate contract pass, and resource counters do not
materially regress.

The realistic Hy3 lane used:

- Apple M5 Max, 40-core GPU, 128 GiB unified memory
- pinned `pipenetwork/Hy3-4bit` revision
  `160619d3f96c8470350b6dac0ef033a8381551e3`
- 112 GiB total cap, 78 GiB expert-cache cap, 8 GiB runtime reserve
- 4,096 live KV tokens, 99 persistent records per layer, 32 transient records
- layer-local LRU, component-bank slots, 64 MiB reads, `F_NOCACHE`
- checked-in 313-token engineering prompt, chat template, thinking disabled
- deterministic greedy AR, seed 0, concurrency 1, natural stop within a
  2,048-token ceiling
- window telemetry disabled for the headline timing lane

PR #12 used a contract-specific 128K lane instead: the expert-cache cap was
omitted, both arms physically used the same 131,200-token Q8 cache, and only the
planner changed from conservative BF16 accounting/75 slots to attested Q8
accounting/100 slots. Its fail-fast short lane used a fixed 256-token decode
ceiling after the 2,048-token candidate remained active for 19 minutes without
finishing.

Before every salvage hardware run, the fixed streamed/probe process check was
empty. The Qwen model launch agent was disabled and unloaded for the complete
hardware lane; its gateway remained loaded.

## PR #13: shared-MLP / miss-I/O overlap

PR #13 was rebuilt without PR #15 and measured as two matched, process-isolated
A/B pairs.

| Pair | Base decode tok/s | PR #13 decode tok/s | Gain | Base end-to-end tok/s | PR #13 end-to-end tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.0770 | 6.4924 | +6.84% | 5.7873 | 6.1638 |
| 2 | 6.0276 | 6.5142 | +8.07% | 5.7411 | 6.1759 |
| Mean | **6.0523** | **6.5033** | **+7.45%** | **5.7642** | **6.1699** |

The workload remained identical across all four runs:

- 1,905 completion tokens, natural `stop`
- token SHA-256 `484e182a68604821f69d56d0b15488d26723e6123f6a57f8158f8b20a4c6ed1c`
- decode hit rate 87.2721%
- 1,768,689,893,376 total expert bytes and 165,678 read operations
- zero short reads, I/O errors, or integrity errors
- peak MLX memory: 89,145,824,176 bytes base and 89,145,807,988 bytes candidate

Raw timing artifacts:

- [`hy3-q4-gated-pr13-base-cbank99-r1.json`](../benchmarks/results/hy3-q4-gated-pr13-base-cbank99-r1.json)
- [`hy3-q4-gated-pr13-candidate-cbank99-r1.json`](../benchmarks/results/hy3-q4-gated-pr13-candidate-cbank99-r1.json)
- [`hy3-q4-gated-pr13-base-cbank99-r2.json`](../benchmarks/results/hy3-q4-gated-pr13-base-cbank99-r2.json)
- [`hy3-q4-gated-pr13-candidate-cbank99-r2.json`](../benchmarks/results/hy3-q4-gated-pr13-candidate-cbank99-r2.json)

The earlier PR #13 measurement on the PR #15-contaminated local history was
about +3.1%. It was not used for promotion. The clean standalone rerun above is
the retained evidence.

## PR #15: all-hit fast path

The reviewed source `7652fa2` was transplanted as `a5be248`; contaminated merge
`bbe0b0d` and follow-up `05f2f13` were not inherited. RED tests reproduced that
the raw source returned a bare MLX array from PR #13's tuple-returning `_run`,
collapsed policy route waves, and could skip the injected second-wave Q4 error.
Repair `e0e93b0` keeps the probe inside authoritative route waves, preserves
epochs/counters/shared-work ordering, releases each wave's pins on error, and
skips reorder operations only for one complete ordered output.

Validation:

- focused PR #15 suites: **40 passed**;
- full suite: **1,985 passed, 4 skipped**;
- Ruff check, Ruff format check, and `git diff --check`: pass;
- independent spec and code-quality reviews: approve.

The historical `+0.78%` effect was too small for a one-pair decision. The
repaired stack therefore ran six process-isolated pairs with balanced arm order:

| Pair | Order | Base decode tok/s | Repaired #15 decode tok/s | Gain |
| --- | --- | ---: | ---: | ---: |
| 1 | base -> candidate | 6.5486 | 6.4844 | -0.98% |
| 2 | base -> candidate | 6.5629 | 6.5867 | +0.36% |
| 3 | base -> candidate | 6.5828 | 6.5467 | -0.55% |
| 4 | candidate -> base | 6.5251 | 6.5490 | +0.37% |
| 5 | candidate -> base | 6.5349 | 6.5651 | +0.46% |
| 6 | candidate -> base | 6.5131 | 6.5976 | +1.30% |
| Pooled mean | balanced | **6.5446** | **6.5549** | **+0.16%** |
| Pooled median | balanced | **6.5417** | **6.5570** | **+0.23%** |

The median pair gain is `+0.36%`; four of six pairs are positive. The arm-order
strata are visibly different (`-0.39%` base-first versus `+0.71%`
candidate-first), so the result must be described as small and order-sensitive,
not as a precise universal speedup. Balancing the two strata yields a positive
`+0.16%` treatment estimate, consistent in direction with the historical
measurement and sufficient under the explicitly approved salvage override.

All 12 headline arms produced 1,905 tokens, natural `stop`, and token SHA-256
`484e182a68604821f69d56d0b15488d26723e6123f6a57f8158f8b20a4c6ed1c`.
Cache counters, slot metrics, expert bytes, read operations, and MLX memory were
machine-compared and identical. Every run ended with zero pins and zero short
reads, I/O errors, or integrity errors.

The separate runner-hook replay (excluded from headline timing) measured:

- SSD: 5.369 GiB/s mean, 5.239 p50, 6.998 p95, 9.112 max;
- 928,933,768 physical expert bytes per decode token and 43.06% of the measured
  12.469 GiB/s SSD ceiling;
- routed-weight memory-traffic floor: 48.49 GB/s mean, 59.78 GB/s p95;
- 80.44 evictions per decode token; classification: `mixed: no single resource dominates`;
- peak MLX memory: 89,145,807,988 bytes; final pins: 0.

Raw headline JSONs:

- [`hy3-q4-gated-pr15-base-cbank99-r1.json`](../benchmarks/results/hy3-q4-gated-pr15-base-cbank99-r1.json) and [`candidate r1`](../benchmarks/results/hy3-q4-gated-pr15-candidate-cbank99-r1.json)
- [`hy3-q4-gated-pr15-base-cbank99-r2.json`](../benchmarks/results/hy3-q4-gated-pr15-base-cbank99-r2.json) and [`candidate r2`](../benchmarks/results/hy3-q4-gated-pr15-candidate-cbank99-r2.json)
- [`hy3-q4-gated-pr15-base-cbank99-r3.json`](../benchmarks/results/hy3-q4-gated-pr15-base-cbank99-r3.json) and [`candidate r3`](../benchmarks/results/hy3-q4-gated-pr15-candidate-cbank99-r3.json)
- [`hy3-q4-gated-pr15-base-cbank99-r4.json`](../benchmarks/results/hy3-q4-gated-pr15-base-cbank99-r4.json) and [`candidate r4`](../benchmarks/results/hy3-q4-gated-pr15-candidate-cbank99-r4.json)
- [`hy3-q4-gated-pr15-base-cbank99-r5.json`](../benchmarks/results/hy3-q4-gated-pr15-base-cbank99-r5.json) and [`candidate r5`](../benchmarks/results/hy3-q4-gated-pr15-candidate-cbank99-r5.json)
- [`hy3-q4-gated-pr15-base-cbank99-r6.json`](../benchmarks/results/hy3-q4-gated-pr15-base-cbank99-r6.json) and [`candidate r6`](../benchmarks/results/hy3-q4-gated-pr15-candidate-cbank99-r6.json)
- [`hy3-q4-gated-pr15-instrumented-cbank99-r1.json`](../benchmarks/results/hy3-q4-gated-pr15-instrumented-cbank99-r1.json)

The companion response Markdown for every JSON is committed beside it. Decision:
**retain repaired PR #15 at `fb4c1d5`**.

## PR #12: Q8 KV accounting

The original source `a8ef882` was transplanted as `e65e8f8`. The repair series
ended at `124f4ce` and made the memory contract fail closed: it charges the
128-token margin and 16-token block rounding, rejects TurboQuant precedence,
attests all 80 Q8 caches, serializes full-capacity Q8 admissions, scopes the
process-wide KV environment through load/compute/close, protects temporary env
overrides, and keeps Q8 expert streaming out of the raw-model `ar_batch` lane.

Validation of the repaired range:

- final full suite: **2,033 passed, 4 skipped**;
- focused Q8, lifecycle, concurrency, server, and runner suites: pass;
- Ruff, range-format, and `git diff --check`: pass;
- independent spec and code-quality reviews: approve.

The special 128K lane deliberately used the same physical cache in both arms so
Q8 numerics could not confound the planning change:

| Metric | Conservative base `1d3d52f` | Repaired candidate `124f4ce` |
| --- | ---: | ---: |
| Planner KV charge | 42,949,672,960 B (BF16) | 21,831,680,000 B (Q8) |
| Persistent slots/layer | 75 | 100 |
| Measured physical KV | 80 Q8 entries, 8,200 x 16-token blocks, 21,831,680,000 B | same; attestation planned=realized |
| Peak MLX | 90,324,091,556 B | 111,292,219,996 B |
| Headroom below runtime cap | 21.35 GB | **376,929,700 B** |

The natural-stop 2,048-token base completed at 4.5960 decode tok/s. The matching
candidate was still active after 19 minutes and was terminated without a result.
The fail-fast gate therefore used the same prompt/config/cache with a fixed
256-token decode ceiling:

| Metric | Base | Repaired #12 | Delta |
| --- | ---: | ---: | ---: |
| Decode throughput | 4.7052 tok/s | 0.9896 tok/s | **-78.97%** |
| Decode elapsed | 54.41 s | 258.69 s | +375.5% |
| Rolling p95 latency | 225.5 ms/token | 1,018.2 ms/token | **+351.5%** |
| Expert misses | 230,681 | 218,025 | -5.49% |
| Expert/read bytes | 497,080,074,240 | 362,713,448,448 | -27.03% |
| Read operations | 45,750 | 33,094 | -27.66% |
| Evictions | 33,302 | 20,725 | -37.77% |

Both arms produced exactly 256 tokens, finish `length`, and token SHA-256
`7aec884257673ffb673a09b8982470dc5cf79c2341e553ea1dc485c97594c6f8`.
Expert requests and route calls were identical; pins, short reads, I/O errors,
integrity errors, deadline errors, and cancellations were zero.

The matched runner-hook replay excluded from headline timing confirmed both the
physical layout and the resource diagnosis:

| Hook | Base | Repaired #12 |
| --- | ---: | ---: |
| Decode throughput | 4.7846 tok/s | 0.7272 tok/s |
| SSD mean / p95 | 6.670 / 8.751 GiB/s | 0.913 / 1.593 GiB/s |
| SSD bytes/decode token | 1,941,719,040 | 1,416,849,408 |
| Routed-memory floor mean | 54.85 GB/s | 10.33 GB/s |
| Evictions/decode token | 130.09 | 80.96 |
| Classification | mixed | **serialization: route/read/compute critical path** |

The candidate reduces storage traffic, but neither SSD nor memory bandwidth is
saturated during its slowdown. Reinvesting every Q8 byte into 25 additional
resident expert slots consumes another 20.97 GB and leaves effectively no MLX
headroom. The premise for this optimization is therefore disproven on the
measured host. A host-specific slot-tuning policy is out of scope because the
existing conservative plan is already safe and substantially faster.

Raw artifacts:

- [`hy3-q4-gated-pr12-base-q8physical-bf16plan-128k-r1.json`](../benchmarks/results/hy3-q4-gated-pr12-base-q8physical-bf16plan-128k-r1.json)
- [`hy3-q4-gated-pr12-base-q8physical-bf16plan-128k-short256-r1.json`](../benchmarks/results/hy3-q4-gated-pr12-base-q8physical-bf16plan-128k-short256-r1.json) and [`candidate`](../benchmarks/results/hy3-q4-gated-pr12-candidate-q8plan-128k-short256-r1.json)
- [`hy3-q4-gated-pr12-base-instrumented-q8physical-bf16plan-128k-short256-r1.json`](../benchmarks/results/hy3-q4-gated-pr12-base-instrumented-q8physical-bf16plan-128k-short256-r1.json) and [`candidate`](../benchmarks/results/hy3-q4-gated-pr12-candidate-instrumented-q8plan-128k-short256-r1.json)

The companion response Markdown is committed beside each JSON. Decision:
**do not promote PR #12; keep the passed-only integration tip at `fb4c1d5`**.

## Correctness-gated candidates

### PR #11: prompt-wide hotset

The final chunk can contain both the old resident and a newly dominant expert.
The old resident is then pinned as a hit, so the new heavy hitter is served
transiently and discarded; the first-decode hotset remains stale. Aborted
prefills also leak counts into the next request, and global scope rejects
realistic chunk cardinality before wave partitioning.

### PR #14: ready-miss streaming

Each miss part owns a separate cancellation signal. If one part fails while a
sibling read is already running, `Future.cancel()` cannot stop the sibling and
cleanup blocks before rollback. Mixed hit/miss submission failure also leaks a
pinned hit, and streamed versus legacy completion APIs disagree on pin ownership.

### PR #16: Metal-resident routing

The exact clean candidate fails two PR #13 focused tests even with the feature
disabled. On the enabled path, an exception during the speculative probe can
release the mapping lease before candidate Q4 work is fenced. The shortcut also
bypasses `route_waves`: a four-expert/two-transient probe changed two host route
calls into one and changed the next LRU evictions from `(2->4, 3->5)` to
`(0->4, 1->5)`. Invalid router IDs also reach `mx.take` before host validation.

### PR #17: completion fences

A replacement can pass the asynchronous-error check, block on the old pin, and
then proceed after that fence fails and releases the pin. Only a later route
observes the failure. Terminal fence failures can also disappear through
`snapshot()` or `close()` when no later route runs.

### PR #18: projection/read pipeline

After a down-suffix failure, slot-pool `active_routes` can reach zero while the
projection generation remains pinned. `close()` then tears down component-bank
storage still referenced by staged Metal work. If projection-release device
synchronization raises, rollback is skipped and the original suffix failure is
masked.

## Memory and SSD bandwidth

The bandwidth tools ran separately from the promotion timing pairs.

### Unified memory

The matched 4 GiB MLX STREAM lane measured:

- GPU read: **558.0 GB/s** (91% of the published 614 GB/s peak)
- GPU triad: **322.3 GB/s** (52% of peak)

Raw artifact:
[`m5max-memory-bandwidth-matched-4gib-20260711.json`](../benchmarks/results/m5max-memory-bandwidth-matched-4gib-20260711.json).

### Cold-cache SSD ceiling

The storage sweep read 512 real Hy3 expert records per sample, three repeats,
at queue depths 1, 8, 32, and 64. All 48 samples began with 0 resident pages;
none was classified as cached or non-storage.

| Path | Recommended repeated median | Peak repeated median |
| --- | ---: | ---: |
| `pread` + `F_NOCACHE` | 12.469 GiB/s at QD8 | 12.676 GiB/s at QD64 |
| buffered `pread` | 12.162 GiB/s at QD8 | 12.649 GiB/s at QD64 |
| `mmap` + `MADV_WILLNEED` | 10.619 GiB/s at QD32 | 10.619 GiB/s at QD32 |
| demand-fault `mmap` | 3.729 GiB/s at QD8 | 3.729 GiB/s at QD8 |

Raw artifact:
[`m5max-hy3-ssd-bandwidth-20260711.json`](../benchmarks/results/m5max-hy3-ssd-bandwidth-20260711.json).

### Runner-hook workload replay

The instrumentation-only replay was excluded from the promotion timing pairs.
Its runner hooks measured the actual PR #13 workload at 0.5-second intervals:

- SSD: 5.313 GiB/s mean, 5.188 p50, 6.910 p95, 9.216 max
- 928,933,768 SSD bytes per decode token and no idle sample intervals
- routed-weight unified-memory traffic floor: 47.99 GB/s mean, 59.34 GB/s p95
- classification: `mixed: no single resource dominates`

The memory number is an analytic floor from routed expert requests times record
bytes, not a physical DRAM counter. The MLX STREAM result above is the measured
device-memory ceiling.

Raw artifact:
[`hy3-q4-gated-pr13-instrumented-cbank99-r1.json`](../benchmarks/results/hy3-q4-gated-pr13-instrumented-cbank99-r1.json).

## Current branch verification

The retained PR #13 tip `939fe57` collected 1,982 tests and completed with
1,978 passed / 4 expected skips. After repaired PR #15, the current sequential
tip collected 1,989 tests and completed with **1,985 passed / 4 expected
skips**; focused, Ruff check, Ruff format check, and `git diff --check` all pass.
The contaminated PR #15 merge/fix commits remain absent from ancestry.

## PR #19 dry merge

The earlier dry merge combined runtime tip `939fe57` with
`codex/hy3-q4-native-serialized@28af6c5` in the isolated
`eval/gated-pr19-drymerge` worktree.

Git merged the three overlapping runtime/test files without textual conflicts,
but the raw auto-merge reproduced the known duplicate
`HotExpertSwitchGLU` import in `tests/test_streamed_models.py` (Ruff F811).
PR #19 also brings 13 Python files that need current Ruff formatting plus
pre-existing whitespace in two benchmark Markdown artifacts.

After temporary dry-merge-only cleanup:

- duplicate import removed
- 13 files mechanically Ruff-formatted
- all 31 changed Python files pass Ruff check and format check
- Python compilation and `git diff --check` pass
- full synthesized suite: **2,032 passed, 4 skipped** in 108.20 seconds

Neither source branch was modified by this synthesis. The dry merge now also
predates repaired PR #15. When PR #19 is rebased, apply the
duplicate-import/format cleanup there and refresh its throughput matrix against
the final sequential runtime tip.

The compact machine-readable decision companion is
[`moe-runtime-gate-matrix.md`](../benchmarks/results/moe-runtime-gate-matrix.md).
