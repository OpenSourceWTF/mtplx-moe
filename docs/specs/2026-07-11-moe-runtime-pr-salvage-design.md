# MoE Runtime PR Salvage Design

**Date:** 2026-07-11
**Branch:** `codex/moe-runtime-gated-integration`
**Starting tip:** `f2d9408314e036b8d1f554d03ecc82cf50d53b6e`
**Draft PR:** #20

## Goal

Integrate the useful work from PRs #11, #12, #14, #15, #16, #17, and #18
without inheriting contaminated local history, repair every reproduced
correctness defect, and retain each candidate whose intended benefit survives a
matched gate against the latest passing integration tip.

PR #13 remains the first retained optimization and the behavioral baseline for
all subsequent work.

## Scope

The work will be performed one candidate at a time in this order:

1. PR #15, all-hit expert decode (`7652fa2`)
2. PR #12, Q8 KV accounting (`a8ef882`)
3. PR #17, Metal completion fences (`d484a94`)
4. PR #14, ready-miss streaming (`87ea0b7`)
5. PR #18, projection/read pipelining (`f37be96`)
6. PR #16, Metal-resident routing (`99f0c2b`)
7. PR #11, prompt-wide hotsets (`fb6f4a4`)

Each source commit will be applied directly to the latest retained integration
tip. Any conflict resolution and repair will be explicit, reviewable commits;
the local PR #15 merge `bbe0b0d`, its contaminated interaction history, and
other precomposed local stacks will not be inherited.

## Non-goals

- Merging any candidate into the upstream target branch or closing its source
  pull request.
- Replacing the expert runtime with a new cache, scheduler, or I/O architecture.
- Relaxing token parity, lifecycle safety, pin ownership, memory planning, or
  cleanup guarantees to obtain a throughput gain.
- Folding unrelated PR #19 cleanup into this salvage series.
- Claiming an unmeasured candidate is neutral or beneficial.

## Integration architecture

The integration branch is a monotonic sequence of passing tips:

```text
base 4146f72
  -> PR #13 939fe57
  -> evidence f2d9408
  -> repaired #15
  -> repaired #12
  -> repaired #17
  -> repaired #14
  -> repaired #18
  -> repaired #16
  -> repaired #11
```

For every candidate, the comparison base is the immediately preceding retained
tip. A candidate may advance the branch only after its focused regressions,
full suite, static checks, and matched benchmark gate pass. This keeps both
performance attribution and cross-PR interaction coverage intact.

The source commit and its repair commits remain separate whenever practical.
This preserves provenance while making the safety changes auditable.

## Candidate contracts

### PR #15: all-hit expert decode

- Preserve PR #13's `shared_work` overlap when every routed expert is resident.
- Preserve route-wave boundaries, policy epochs, cache counters, and future
  victim selection; the fast path may eliminate I/O, not scheduler semantics.
- Fall back to the existing miss path without changing pin ownership or error
  propagation.

### PR #12: Q8 KV accounting

- Count only a Q8 implementation that the active runtime actually selects.
- TurboQuant precedence, explicit skips, and unsupported configurations must
  not be credited as Q8 savings.
- Plan the physical rounded allocation, including the 128-token capacity
  increment, rather than only the live logical token count.
- Admission must fail before model allocation when the attested plan exceeds the
  memory budget.

### PR #17: Metal completion fences

- A replacement may not reuse a slot until both pin drainage and the prior
  Metal completion fence have succeeded.
- Fence failure is fail-closed and becomes visible to the current operation,
  runtime snapshots, and `close()`.
- A timed-out close remains retryable and must not silently convert the runtime
  into a partially closed state.

### PR #14: ready-miss streaming

- A failure in any sibling miss cancels or drains the rest without waiting
  forever on a running future.
- Cleanup is bounded, preserves the primary exception, and releases every hit
  and miss pin acquired before a submit or decode failure.
- Streamed and legacy paths have one unambiguous owner for every pin.

### PR #18: projection/read pipeline

- Projection pins stay live until all suffix Metal work that reads the
  projection has completed.
- Storage cannot close while a projection pin or dependent Metal operation is
  live.
- Cleanup errors do not mask the primary suffix failure, but remain observable.
- Cancellation and deadlines propagate through projection scheduling, reads,
  suffix execution, and cleanup.

### PR #16: Metal-resident routing

- With the experiment disabled, construction and execution require no new
  runtime-spec fields and remain behaviorally identical to the prior tip.
- Expert IDs are range-validated on the host before `mx.take` or speculative Q4
  work can observe them.
- The Metal directory fast path preserves route-wave partitioning, policy
  epochs, cache counters, and victim selection.
- Every exceptional exit fences launched speculative work before releasing
  route or slot lifetime.

### PR #11: prompt-wide hotsets

- The final prefill wave may replace a stale resident with the prompt's actual
  heavy hitter; temporary final-wave pins cannot defeat the selected hotset.
- Prompt preparation state has an explicit request boundary and is cleared on
  success, cancellation, and failure before a later request begins.
- Global routes with more unique experts than transient capacity are wave
  partitioned before capacity validation rather than rejected wholesale.
- Layer and global banks, plus LRU and frequency policies, must satisfy the same
  hotset and cleanup contracts.

## Error handling and invariants

The repaired stack follows these rules across all candidates:

1. Validate host-visible IDs, capacities, and runtime support before launching
   device work or changing cache state.
2. After device work is launched, fence it before releasing any slot, route,
   projection, or storage lifetime it can still access.
3. Cleanup is idempotent and bounded. A cleanup failure is recorded without
   replacing the first operational failure.
4. Cancellation and deadlines are forwarded through all asynchronous layers.
5. Runtime state after any failed request is safe for an immediately following
   request or an explicit retry of `close()`.

## Test strategy

Every behavior repair uses a strict RED-GREEN cycle:

1. Add the smallest regression that reproduces one documented defect.
2. Run it and confirm the expected failure on the unmodified candidate.
3. Apply one focused repair.
4. Re-run the regression and the candidate's focused suite.
5. Run the full project suite, Ruff check/format verification, and
   `git diff --check`.

Concurrency and lifecycle tests use controlled futures, events, and injected
failures instead of timing-only assertions. Policy tests compare route calls,
epochs, resident sets, and the next victim, not merely output tensors. Memory
planning tests assert physical byte counts at rounding boundaries and every
runtime-selection branch.

## Performance and resource gate

The previous roadmap's 5% minimum is overridden for this salvage series.
A candidate is retainable when its intended benefit is repeatable and positive,
even when the gain is below 5%, provided all mandatory safety checks pass.

The matched gate for each performance candidate uses:

- the immediately preceding retained tip as base;
- identical model, prompt, cache-bank setting, sampling parameters, and runner;
- Qwen 3.6 unloaded before GPU measurements while its gateway remains available;
- process-isolated, alternating base/candidate pairs with at least three pairs
  for sub-5% effects;
- token-ID and stop-condition parity;
- runner hooks for decode throughput, expert bytes/operations, memory pressure,
  and SSD traffic;
- the established unified-memory and cold-cache SSD bandwidth measurements as
  hardware context.

Acceptance is contract-specific:

- Throughput candidates need a positive matched median and mean with no
  systematic pair reversal.
- Memory-planning candidates may pass through a demonstrated reduction in
  admitted/allocated bytes or a corrected fail-fast decision while throughput
  remains non-regressed.
- Correctness/lifecycle candidates may pass on the repaired contract when their
  optimization path remains enabled and throughput, memory, and SSD traffic do
  not regress materially.

Results that overlap measurement noise trigger more matched pairs. A candidate
with no demonstrable benefit after the expanded run is documented but not
retained merely because its tests pass.

## Documentation and rollout

After each candidate gate:

- append its exact tip, test commands, benchmark files, runner-hook results, and
  decision to `docs/MOE_RUNTIME_PR_BENCHMARKS.md`;
- update `benchmarks/results/moe-runtime-gate-matrix.md`;
- write new machine-generated JSON/Markdown artifacts under
  `benchmarks/raw/moe-runtime/<run-id>/`; this raw tree is ignored, so attach
  artifacts externally when durable retention is required while retaining the
  consolidated reports in the repository;
- push only a passing retained tip to draft PR #20.

Qwen 3.6 will be restored after the final hardware run. If a candidate cannot
be repaired within three evidence-backed fix attempts, work stops for an
architecture review rather than stacking speculative changes.

## Failure-mode check

1. **Cross-PR semantic drift (critical):** a later fast path can bypass epochs,
   counters, or PR #13 overlap. Sequential latest-tip tests and policy-state
   comparisons are mandatory before each benchmark.
2. **Small gains confused with noise (critical):** PR #15's historical +0.78%
   is too small for a single pair. Alternating process-isolated pairs are
   expanded until direction and resource counters are stable.
3. **Cleanup tests pass while device work remains live (critical):** host state
   alone cannot prove safety. Injected fence failures and controlled outstanding
   Metal work must demonstrate that reuse and close are blocked or fail closed.
4. **Provenance is lost during conflict resolution (minor):** each source SHA is
   recorded in commit messages and benchmark docs; repairs remain separate when
practical.
