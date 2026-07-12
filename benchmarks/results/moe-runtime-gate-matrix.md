# MoE Runtime Candidate Gate Matrix

Date: 2026-07-11
Clean integration base: `origin/codex/moe-ssd-hy3-glm52@4146f72`
Passed-only integration branch: `codex/moe-runtime-gated-integration`

## Gate contract

The sequential salvage contract in
`docs/specs/2026-07-11-moe-runtime-pr-salvage-design.md` supersedes the old 5%
minimum. A throughput candidate may be retained when repeated matched evidence
is positive in mean and median, the direction is not systematically reversed,
and correctness, deterministic token parity, memory planning, and resource
counters pass. Raw results, the exact command/profile, and the immediate
predecessor are required. Correctness-only changes use their contract-specific
gate and must remain materially non-regressed.

## Status

Sequential salvage in progress: **2 retained (#13 and repaired #15), 1 repaired
but rejected on measurement (#12), and 5 pending repair/re-gate**. Pending rows
preserve the prior clean-candidate failure evidence until their sequential
repair task is complete.

| PR | Source | Purpose | Correctness evidence | Measured result | Gate status | Integration status |
| --- | --- | --- | --- | --- | --- | --- |
| #11 | `fb6f4a4` -> `3d7c158` | Prompt-wide expert hotset seeding | Fresh: 50 focused passed; full suite 1,980 passed, 4 skipped; lint/format/diff checks pass. Semantic probes show a newly dominant expert in the final chunk stays transient when the stale seed is a hit/pinned, leaving the wrong decode hotset; aborted prefills also leak seed counts into the next request. Global scope additionally rejects a realistic chunk with more unique experts than transient slots before wave partitioning. | No GPU run because the correctness contract failed first | **Failed: does not reliably realize the prompt-wide hotset** | Not integrated; candidate preserved on `eval/gated-pr11` |
| #12 | `a8ef882` -> `e65e8f8` + repairs through `124f4ce` | Q8 KV physical-memory accounting | Exact rounding/margin, 80-cache Q8 attestation, single-live-cache admission, process env lifetime, close races, and raw `ar_batch` bypass repaired; full suite 2,033 passed / 4 skipped; spec and quality reviews approved. | Token-identical short lane: 4.7052 -> 0.9896 tok/s (**-78.97%**); read bytes -27.03%, but peak MLX +20.97 GB with only 376.9 MB cap headroom. Instrumented: 4.7846 -> 0.7272 tok/s, SSD 6.670 -> 0.913 GiB/s; classified as serialized, not bandwidth-bound. | **Failed performance/resource gate** | **Not integrated; passed-only tip remains `fb4c1d5`** |
| #13 | `1efb308` -> `939fe57` | Overlap resident shared MLP with pending decode miss I/O | Fresh: 43 focused passed; full suite 1,978 passed, 4 skipped; Ruff check/format and diff check pass; spec and quality reviews approved | Repeated fixed lane: 6.0523 -> 6.5033 decode tok/s mean, **+7.45%**; pair gains +6.84% and +8.07%; all tokens identical | **Passed** | **Retained at `939fe57`** |
| #14 | `87ea0b7` -> `c424381` | Stream ready misses in physical completion order | Fresh: 49 focused passed; full suite 1,979 passed, 4 skipped; lint/format/diff checks pass. If one miss part fails, a running sibling has no shared cancellation and cleanup can block forever before rollback. A mixed hit/miss submit failure also leaks the pinned hit, and the streamed/legacy APIs disagree on pin ownership. | No GPU run because cancellation and pin-lifetime correctness failed first | **Failed: miss failure can hang rollback and leak pins** | Not integrated; candidate preserved on `eval/gated-pr14` |
| #15 | `7652fa2` -> `a5be248` + repair `e0e93b0` | All-hit decode fast path | RED reproduced the PR #13 tuple/shared-work break, flattened route-wave accounting, and second-wave pin-cleanup gap. GREEN: 40 focused passed; full suite 1,985 passed / 4 skipped; spec and quality reviews approved; Ruff/format/diff checks pass. | Six order-balanced process-isolated pairs: pooled decode mean 6.5446 -> 6.5549 tok/s, **+0.16%**; pooled median **+0.23%**; median pair gain **+0.36%**; 4/6 pairs positive; token/counter identical. Runner hooks: 5.369 GiB/s mean SSD, 6.998 p95, 48.49 GB/s routed-memory floor. | **Passed under salvage override; small and order-sensitive** | **Retained at `fb4c1d5`; `bbe0b0d` and `05f2f13` not inherited** |
| #16 | `99f0c2b` -> `4106348` | Metal-resident expert routing | Exact clean candidate focused gate: 68 passed, 2 failed; env-off construction reads `runtime.spec` that PR13 test doubles do not expose. Semantic probes also show unfenced speculative Q4 work on probe exceptions, route-wave bypass changing LRU epochs/next evictions, and invalid router IDs reaching `mx.take` before host validation. | No GPU run because focused and semantic correctness failed first | **Failed: default path regresses and Metal shortcut changes policy semantics** | Not integrated; candidate preserved on `eval/gated-pr16-clean` |
| #17 | `d484a94` -> `faf1527` | Metal completion fences for slot reuse | Fresh: 51 focused passed; full suite 1,981 passed, 4 skipped; lint/format/diff checks pass. A deterministic race shows the next replacement checks for a fence error before waiting on the pin, then proceeds after that fence fails/releases it; only a later route observes the error. Terminal fence errors can also escape `snapshot()`/`close()`. | No GPU run because the fail-closed correctness contract failed first | **Failed: asynchronous fence errors are not fail-closed** | Not integrated; candidate preserved on `eval/gated-pr17` |
| #18 | `f37be96` -> `8743a93` | Gate/up compute overlapped with down-projection reads | Fresh: 36 focused passed; full suite 1,985 passed, 4 skipped; lint/format/diff checks pass. After an injected suffix failure, `active_routes=0` while a projection pin remains; `close()` tears down the component-bank storage with that pin live. If projection-release synchronization raises, rollback is skipped and the suffix error is masked. | No GPU run because partial-route lifetime correctness failed first | **Failed: storage can close while staged Metal work remains pinned** | Not integrated; candidate preserved on `eval/gated-pr18-clean` |

## PR #13 standalone evidence

The earlier `+3.1%` PR #13 result was measured after PR #15 had entered the local
history and did not clear the repository's documented 5% threshold. PR #13 was
therefore rebuilt from `4146f72` without PR #15 and rerun as two matched A/B pairs.

| Metric | Base mean | PR #13 mean | Result |
| --- | ---: | ---: | ---: |
| Decode throughput | 6.0523 tok/s | 6.5033 tok/s | +7.45% |
| Reader-active SSD throughput | 3,577.2 MiB/s | 3,560.9 MiB/s | unchanged workload |
| Wall-effective expert-read rate during decode | 5.233 GiB/s | 5.623 GiB/s | overlap exposes more I/O per wall second |
| Decode hit rate | 87.2721% | 87.2721% | identical |
| Expert bytes read | 1,768,689,893,376 | 1,768,689,893,376 | identical |
| Read operations | 165,678 | 165,678 | identical |
| Peak MLX memory | 89,145,824,176 bytes | 89,145,807,988 bytes | effectively identical |
| Token SHA-256 | `484e182a...4c6ed1c` | `484e182a...4c6ed1c` | identical |
| Short reads / I/O errors / integrity errors | 0 / 0 / 0 | 0 / 0 / 0 | pass |

Raw results:

- `hy3-q4-gated-pr13-base-cbank99-r1.json`
- `hy3-q4-gated-pr13-candidate-cbank99-r1.json`
- `hy3-q4-gated-pr13-base-cbank99-r2.json`
- `hy3-q4-gated-pr13-candidate-cbank99-r2.json`

Matched unified-memory ceiling (separate, non-overlapping MLX STREAM lane):

- GPU read: 558.0 GB/s (91% of the published 614 GB/s peak)
- GPU triad: 322.3 GB/s (52% of peak)
- Raw result: `m5max-memory-bandwidth-matched-4gib-20260711.json`

Verified-cold Hy3 sidecar SSD sweep (48 runs; zero resident pages before every
lane, zero cached/non-storage samples, zero warnings):

- `pread` + `F_NOCACHE`: 12.469 GiB/s repeated-median at QD8;
  12.676 GiB/s peak repeated-median at QD64
- buffered `pread`: 12.162 GiB/s repeated-median at QD8;
  12.649 GiB/s peak repeated-median at QD64
- `mmap` + `MADV_WILLNEED`: 10.619 GiB/s repeated-median at QD32
- demand-fault `mmap`: 3.729 GiB/s repeated-median at QD8
- Raw result: `m5max-hy3-ssd-bandwidth-20260711.json`

Runner-hook replay (separate, instrumentation-only branch; excluded from the
promotion timing pairs):

- SSD: 5.313 GiB/s mean, 5.188 GiB/s p50, 6.910 GiB/s p95, 9.216 GiB/s max
- SSD utilization: 42.61% of the configured 12.47 GiB/s ceiling; no idle sample intervals
- Routed-weight unified-memory traffic floor: 47.99 GB/s mean, 59.34 GB/s p95
- The memory figure is an analytic lower bound (`expert_requests × record_bytes`),
  not a physical DRAM counter; the standalone MLX STREAM result above is the
  measured device-memory bandwidth.
- Raw result: `hy3-q4-gated-pr13-instrumented-cbank99-r1.json`

## Final validation and PR #19 synthesis

- Final runtime tip `939fe57`: 1,982 collected; 1,978 passed and 4 expected
  skips; changed-file Ruff check/format and `git diff --check` pass.
- Raw auto-merge with `codex/hy3-q4-native-serialized@28af6c5`: no textual
  conflicts, but reproduces duplicate `HotExpertSwitchGLU` import (Ruff F811),
  13 changed Python files needing current formatting, and whitespace in two
  PR #19 benchmark Markdown files.
- Temporary dry-merge resolution: duplicate removed, 13 files formatted, all
  31 changed Python files pass Ruff/format/compile/diff checks; full suite
  **2,032 passed, 4 skipped**.
- PR #19's throughput matrix predates PR #13 and must be refreshed after its
  rebase onto the final runtime tip.
