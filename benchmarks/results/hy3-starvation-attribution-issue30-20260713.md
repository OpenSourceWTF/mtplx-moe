# Hy3 expert-pipeline starvation attribution (#30)

Date: 2026-07-13

Frozen predecessor: `experiment/hy3-cache-scheduling@b7ba5ab526a44c4abbeb11a1462794f3ea79d1a0`

Measured source: `59bc40c37efe7cd3d25149fcb87f3f33db29a7f3`

## Outcome

Retain Phase 1 as opt-in attribution infrastructure. The measurement narrows
the B1 wall, but it does not promote a scheduling or kernel change.

Across four balanced telemetry-off/on pairs, the potentially blocking
next-miss step occupied **40.515% of pooled covered decode time**. This is an
upper bound, not exact generation-thread wait. Within that upper-bound span,
**95.224%** overlapped host-known storage activity and **96.879%** overlapped
an active reader task. Those overlaps are independent and cannot be added.
Only **0.116%** overlapped submitted-queued work and **0.060%** overlapped
eligible-unsubmitted work.

In this concurrency-1 covered-decode lane, the upper-bound step was
predominantly co-located with host-known current-route read activity and rarely
with ledger submitted-queued or eligible-unsubmitted states. This does not
identify the awaited future, and outer split-executor occupancy remains
unavailable. Sampler telemetry averaged **3.291 of 32 active readers**, a
**0.463** submitted queue, and **5.371 GiB/s** of reader-returned traffic, or
**43.072%** of the supplied 12.47 GiB/s host ceiling. Mean exact-demand depth
did not approach 32, so raising the cap alone is unsupported for this workload;
the sampler's lifetime peak of 32 still shows that short bursts can fill it.

The next narrow performance experiment should test 2/4/8-way exact-demand
range striping within a required record while preserving authoritative routing,
deterministic parity, and the existing no-extra-barrier rule. Phase 1 itself
does not authorize that implementation, speculation, cache-policy changes,
MLX scheduling changes, or a kernel change.

## Campaign validity

The issue-specific runner executed this physical order:

`off-p01`, `on-p01`, `on-p02`, `off-p02`, `off-p03`, `on-p03`, `on-p04`,
`off-p04`.

All eight arms exited successfully and passed the selected stream, route,
cache, manifest, normalized-configuration, and final-health parity checks.
Every arm used the same normalized full configuration fingerprint
`e581971b10015721a8bf437a578262ce7e933b39679774316276e0855c9e7e54`.
The raw fingerprints were stable within each arm and differed only as expected
for the resource-telemetry toggle: `427569328848c72c` off and
`b1ecf851fecab0f8` on.

Qwen was unloaded only for the exclusive hardware interval. The runner captured
and restored the exact initial tuple; both initial and final state were loaded
with only `mtplx-qwen36-27b-optimized-speed`. The manifest reaches `complete`
only after the exclusive lane is released. A separate live post-run check found
no remaining campaign process; that observation is not a manifest field.

## Telemetry cost

Telemetry-on is a diagnostic lane, not a performance candidate or headline
runtime result.

| Pair | Off decode tok/s | On decode tok/s | On/off delta | Off aggregate tok/s | On aggregate tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.709807 | 5.914750 | -11.849% | 4.777718 | 4.285853 |
| 2, reverse order | 6.748687 | 5.936428 | -12.036% | 4.816146 | 4.297337 |
| 3 | 6.747068 | 5.950994 | -11.799% | 4.817084 | 4.306218 |
| 4, reverse order | 6.592186 | 5.717530 | -13.268% | 4.740770 | 4.173744 |
| Arm mean | **6.699437** | **5.879926** | -12.233% ratio of means | **4.787930** | **4.265788** |

The median paired decode delta was **-11.942%**, with a full range of
**-13.268% to -11.799%**. Aggregate completion throughput had a median paired
delta of **-10.689%**, range **-11.961% to -10.295%**. Attribution fractions
come only from the diagnostic arms; throughput conclusions come from the
telemetry-off controls.

## Decode-state attribution

The table pools nanosecond durations over the four telemetry-on arms and divides
by the summed `decode_observation_ns`. These are mutually exclusive orientation
states over the ledger's covered decode window, not fractions of total request
wall time.

| Primary state | Pooled fraction | Four-run range |
| --- | ---: | ---: |
| Potentially blocking next-miss step | **40.515%** | 39.373%-40.993% |
| Host-runnable work | 23.705% | 23.312%-24.366% |
| Logical range active | 20.529% | 20.352%-20.796% |
| No known useful work | 8.676% | 8.591%-8.883% |
| Eligible, unsubmitted | 3.593% | 3.562%-3.606% |
| Submitted, queued | 1.983% | 1.974%-2.000% |
| Reader completion active | 0.802% | 0.797%-0.809% |
| Route publication pending | 0.197% | 0.193%-0.201% |

The eight primary durations sum exactly to `decode_observation_ns` in every
run. The next-miss state has the highest precedence, so independent facts that
coincide with it appear in the overlap ledger below.

| Next-miss upper-bound overlap | Fraction of covered decode | Fraction of upper-bound span |
| --- | ---: | ---: |
| Reader task active | 39.251% | **96.879%** |
| Storage range active | 38.581% | **95.224%** |
| Host-runnable work | 1.153% | 2.847% |
| Submitted, queued | 0.047% | 0.116% |
| Eligible, unsubmitted | 0.024% | 0.060% |

These rows are orthogonal intersections and may overlap one another. In
particular, storage-active and reader-task-active are not two additive causes.

## Pipeline depth and service

Each telemetry-on arm recorded exactly:

- 19,164 authoritative decode miss records and 203,460,968,448 logical bytes;
- 19,164 accepted submissions, reader tasks started/completed, ready records,
  runnable records, claims, and completed decode logical ranges;
- zero submission rejections, reader failures, abandoned records, pin-held
  waits, slot-loading waits, or diagnostic-hook failures;
- 31,684 all-phase logical reader invocations and Python `preadv` attempts,
  returning 346,098,106,368 bytes; and
- 15,002-15,305 potentially blocking next-miss events.

Each arm also recorded 39,910 synchronous fence calls, or 155.898/token, and
zero mean active asynchronous fences. These are operation/occupancy facts, not
fence duration, GPU utilization, or proof of avoidable synchronization.

Across the four diagnostic arms, the decode ledger averaged about **1.048
active current-route logical ranges** and **1.072 active reader tasks**. Mean
per-record state durations were 0.137 ms eligible-to-submit, 0.060 ms accepted
but unstarted, 1.850 ms reader-active, 1.819 ms inside the logical range,
0.036 ms ready but not runnable, and 0.118 ms runnable but unclaimed. Complete
record and logical-range p50/p95 histogram estimates were bounded above by
10 ms with no overflow; those are bucket bounds, not interpolated percentiles.

The sampler-window backend counters include prefill and decode, while lifecycle
and primary-state counters above are decode-scoped. Record jobs, executor tasks,
logical ranges, Python `preadv` attempts, and physical device operations are
different identities.

## Claim boundary

- `potentially_blocking_next_miss_step` is `measured_upper_bound`. The prior
  readiness scan and `next(as_completed(...))` are not atomic, and the end hook
  may wait for the telemetry ledger lock.
- Exact generation-thread expert-input wait, GPU expert-wait, GPU idle time,
  physical device bytes/operations/queue depth, future-layer eligibility, and
  speculative accounting remain `unavailable`, not zero.
- Eligible-unsubmitted cause remains `unattributed`; Phase 1 cannot assign it
  to the outer split executor.
- The 346,098,106,368 returned bytes are uncached host reader traffic under
  `F_NOCACHE`, not physical NAND traffic.
- The supplied 12.47 GiB/s ceiling is a host benchmark reference. The observed
  43.072% fraction does not prove instantaneous device utilization or physical
  queue depth.
- Telemetry is materially perturbing in this lane. Its local lock exists only
  when explicitly enabled; telemetry-off creates no ledger and adds no runtime
  synchronization barrier.
- The cooperative exclusive lane controls launchers that honor the repository
  convention. It cannot exclude a deliberately unrelated process that bypasses
  that convention.

## Verification

- The full repository `pytest -q` gate reached 100% with exit code 0 after the
  curated evidence was added; four expected skips and existing dependency
  deprecation warnings remained.
- The issue-specific campaign-runner suite passed all 16 cases.
- Ruff check passed, and all 14 changed Python files were already formatted.
- The curated JSON parsed successfully and an automated cross-check matched its
  campaign status, physical order, Qwen tuple, parity, paired decode TPS, and
  pooled next-miss duration/fraction back to the retained raw payload.
- Independent source/runner reviews found no Critical or Important findings.
  A separate raw-to-curated review found no numeric or schema discrepancy and
  required the causal wording limits now used in this report.
- The only stub-scan matches are intentional non-stubs: a pre-existing
  `UnboundExpertSwitch` description and the campaign runner's plan-only label
  template.

## Environment and reproduction

- MacBook Pro `Mac17,6`, Apple M5 Max, 128 GiB unified memory
- macOS 26.5.2 build 25F84, arm64
- Python 3.12.13, MLX 0.31.2, mlx-lm 0.31.3
- pinned `pipenetwork/Hy3-4bit` revision
  `160619d3f96c8470350b6dac0ef033a8381551e3`
- global LRU component-bank cache, 83,034,243,072-byte expert ceiling,
  32 transient slots, 64 MiB read chunks, `F_NOCACHE`
- deterministic chat AR, seed 0, thinking disabled, 313 prompt tokens,
  256 generated tokens, concurrency 1, no MTP, no window telemetry
- no thermal or performance warning was reported before or after any arm; fan
  coverage was unavailable

The raw campaign is intentionally ignored at:

`benchmarks/raw/moe-runtime/moe-runtime-issue30-campaign-p14517-20260713T190009Z-59bc40c37efe/`

Its atomic manifest retains every exact argv, start/end time, signature, health
gate, host fact, thermal sample, and Qwen transition. Reproduce the fixed
campaign from a clean source SHA with:

```bash
uv run --frozen --extra dev --extra server python \
  scripts/run_issue30_starvation_attribution.py
```

The machine-readable curated companion is
`benchmarks/results/hy3-starvation-attribution-issue30-20260713.json`.
