# MoE Runtime PR Benchmark Report

Date: 2026-07-11
Remote base: `origin/codex/moe-ssd-hy3-glm52@4146f72`
Passed-only tip: `939fe57206eaceb6b6f2cf6a84ecc393a3a08cde`

## Result

PRs #11 through #18 were rebuilt independently from the latest passed-only tip.
The final count is **1 retained, 7 skipped, 0 untested**. PR #13 is the only
candidate that cleared correctness and the repository's measured performance
gate. PR #15 missed the performance bar. The other six candidates failed a
correctness or fail-closed contract before GPU benchmarking.

| PR | Clean candidate | Correctness result | Performance result | Decision |
| --- | --- | --- | --- | --- |
| #11 | `3d7c158` | 50 focused passed; 1,980 passed / 4 skipped full suite; semantic hotset and request-lifecycle failures | Not run: correctness stopped the gate | Skip |
| #12 | `d06b483` | 141 focused passed / 1 skipped; 1,984 passed / 4 skipped full suite; physical Q8 allocation is not attested or fully budgeted | Not run: contract stopped the gate; 36.7% remains an estimate | Skip |
| #13 | `939fe57` | 43 focused passed; 1,978 passed / 4 skipped full suite; independent reviews approved | 6.0523 -> 6.5033 decode tok/s mean, **+7.45%** over two matched pairs; token-identical | **Retain** |
| #14 | `c424381` | 49 focused passed; 1,979 passed / 4 skipped full suite; miss failure can hang rollback and leak pins | Not run: correctness stopped the gate | Skip |
| #15 | `7652fa2` | Historical focused/full validation passed | 6.1224 -> 6.1701 decode tok/s, **+0.78%**; token-identical | Skip: below 5% |
| #16 | `4106348` | Exact focused gate: 68 passed / 2 failed; additional device-fence and policy-accounting failures | Not run: correctness stopped the gate | Skip |
| #17 | `faf1527` | 51 focused passed; 1,981 passed / 4 skipped full suite; asynchronous fence errors are not fail-closed | Not run: correctness stopped the gate | Skip |
| #18 | `8743a93` | 36 focused passed; 1,985 passed / 4 skipped full suite; storage can close while partial Metal work remains pinned | Not run: correctness stopped the gate | Skip |

"Not run" is a gate result, not an estimated zero. Hardware performance was
intentionally not measured after a candidate failed correctness, because a fast
unsafe implementation is not promotable.

## Gate contract and fixed lane

The authoritative retention rule in
`docs/MOE_SSD_STREAMING_OPTIMIZATION_ROADMAP.md` requires at least 5% on the
candidate's target lane in repeated runs, no material regression, deterministic
token parity, and compliance with the memory plan.

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

Before every hardware run, the fixed streamed/probe process check was empty.
Qwen 3.6 was unloaded while tests and benchmarks ran; its gateway remained
loaded.

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

The historical realistic-lane result was 6.1224 -> 6.1701 decode tok/s,
approximately **+0.78%**, with exact token parity and effectively unchanged
memory/I/O counters. This is below the repository's 5% bar, so PR #15 and its
local merge commit `bbe0b0d` are intentionally absent. The raw artifact is
`benchmarks/results/hy3-q4-pr15-ab-cbank99.json` in benchmark commit
`ca5691c`; the commit message records the matched baseline and candidate values.

## Correctness-gated candidates

### PR #11: prompt-wide hotset

The final chunk can contain both the old resident and a newly dominant expert.
The old resident is then pinned as a hit, so the new heavy hitter is served
transiently and discarded; the first-decode hotset remains stale. Aborted
prefills also leak counts into the next request, and global scope rejects
realistic chunk cardinality before wave partitioning.

### PR #12: Q8 KV accounting

The planner's per-token arithmetic is correct, but runtime cache construction
does not attest that all target caches actually became Q8 and does not fail on
skipped conversions or inherited TurboQuant. The planner also charges live
tokens while allocation rounds capacity and adds a 128-token margin. At 131,072
tokens that leaves 21,299,200 Q8 bytes outside the plan. The claimed 36.7%
expert-read reduction is replay-estimated and was not promoted as measured.

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

## Final branch verification

On `939fe57`, the final suite collected 1,982 tests and completed with exit 0
(1,978 passed, 4 expected skips). Ruff check, Ruff format check, and
`git diff --check` pass on all six changed Python files. PR #15 and every other
skipped candidate are absent from ancestry.

## PR #19 dry merge

The dry merge combined runtime tip `939fe57` with
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

Neither source branch was modified by this synthesis. When PR #19 is rebased,
apply the duplicate-import/format cleanup there and refresh its throughput
matrix: its checked-in measurements predate PR #13 and do not represent the
final runtime base.

The compact machine-readable decision companion is
[`moe-runtime-gate-matrix.md`](../benchmarks/results/moe-runtime-gate-matrix.md).
