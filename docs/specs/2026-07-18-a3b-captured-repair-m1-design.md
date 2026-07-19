# A3B Captured-Primary M1 Repair Design

**Status:** Approved on 2026-07-18

**Base:** `perf/a3b-174-beneficial-stack` at `960849b`

## Objective

Reduce the measured A3B K1 rejection-repair cost without changing the existing
target-prefix sampling schedule. The current exact route re-forwards
`[primary, correction]` through the fixed M2 compiled target graph after every
rejection. The replacement commits the already-computed state after `primary`
and forwards only `correction` through a fixed M1 continuation.

The native repeat-2 trace recorded 282 rejections over 1,595 cycles. Repair
consumed 3.016895 seconds, or 10.698209 ms per rejection and 1.891470 ms per
cycle. This is the largest remaining independently measured avoidable region.

## Governing constraints

- Correct by construction: validate external model/config/request facts,
  dtype, topology, and self-check results once. Trust the factories, promoted
  cache ownership, and trace containers constructed from those facts; do not
  rescan or revalidate internal artifacts.
- The installed path contains no environment reads, shape or dtype checks,
  eligibility predicates, status lookups, fallback accounting, exception
  demotion, stock fallback, or engagement counters.
- Preserve the existing K1 target-prefix sampling order exactly. Rejection must
  not use `pending_primary` and must not defer `correction` into the next cycle.
- Preserve the current row-major M1-M8 QMM routes, row-owned router, combine
  tail, M2/TGY4 GDN post-conv arithmetic, request-aware KV reserve, and zero
  compiled-verifier fallback/growth contract.
- The experiment remains default-off and independently attributable. It does
  not include target-logit summarization or packed full-attention Q/K/V.

## Current execution

For each K1 cycle, target-prefix verification runs the fixed compiled M2 input
`[primary, draft]`. The compiled graph already returns 30 pairs of per-position
GDN captures plus the final state of all 40 layers. On rejection, the route
currently discards the captures, restores a pre-verify snapshot, and calls the
same M2 graph with `[primary, correction]`.

That second M2 pass is redundant. Row 0 of the first pass is the exact state
after `primary`, which is the only state required to continue with
`correction`.

## Constructed model contract

When `MTPLX_FUSE_GDN_POST_CONV=1`, model construction validates the existing
exact A3B contract and returns one immutable post-conv factory containing
separate M1 and M2 callable tuples in proven 30-GDN/40-layer order:

| Route | Input geometry | State geometry | Output geometry | Kernel |
|---|---|---|---|---|
| M1 correction | BF16 `conv=[1,1,8192]`, `a/b=[1,1,32]` | FP32 `[1,32,128,128]` | BF16 `[1,1,32,128]`, FP32 captures `[1,1,32,128,128]` | TGY4 |
| M2 verification | BF16 `conv=[1,2,8192]`, `a/b=[1,2,32]` | FP32 `[1,32,128,128]` | BF16 `[1,2,32,128]`, FP32 captures `[1,2,32,128,128]` | TGY4 |

The compiled-target factory owns that exact post-conv factory directly, and
model loading retains the compiled-target factory on the runtime rather than
writing a discoverable model attribute. Generation routes on that direct
runtime ownership and passes the factory into request construction. M1 and M2
trace construction each consume their corresponding tuple; there is no dynamic
logical-M lookup or per-module installation state. Both entrypoints encode
batch, sequence, head ownership, axis, dtype, epsilon-dependent arithmetic,
grid, and TGY4 constants directly.

The post-conv experiment therefore requires the compiled-target-prefix flag at
model load. Enabling post-conv without its only owning route is a configuration
error; the system never reports an orphaned factory as installed.

The model self-check must validate both M1 and M2 against the current stock
capture arithmetic with deterministic nonzero state before either route is
installed. A failure prevents installation.

## Constructed request contract

The A3B request installer validates only external request facts: K1
target-prefix ownership, the stochastic top-k sampler used by the natural
contract, stock surroundings, compiled boundary/donation, and the maximum
context. Greedy and non-top-k requests cannot construct this exact route, so a
rejection always owns a correction token without a hot-path sentinel or
fallback. Generation proves both exact and generic target-prefix sampler
compatibility once before prompt construction, after which every valid request
calls the raw sampler with no optional-result check or wrapper. The installer
directly transforms the ten known full-attention
positions in the prompt cache produced by the exact runtime into the fixed
tensor-offset construction with request-aware capacity, then trusts the exact
shadow topology and state layout it constructs without type, shape, dtype,
capacity, eager-mode, or bucket rescans. Successful construction
prebinds:

- one fixed compiled M2 verifier;
- one fixed compiled M1 correction continuation;
- the 30 GDN capture output positions;
- the 30 GDN real-state destinations;
- the ten full-attention K/V/offset inputs and destinations.

No generic `CompiledVerifyBank` dispatch, capture reconstruction, trim probe,
or rollback operation is used after installation.

The rejection commit is likewise route-owned: the exact block appends its
construction-guaranteed correction and enters M1 directly. The generic
optional-correction branch is unreachable because the exact block continues
before it.

## State ownership

After M2 verifies `[primary, draft]`, the route owns both final M2 state and the
following exact primary state:

| Layer ownership | State after `primary` | Rejection continuation |
|---|---|---|
| 30 GDN convolution caches | `conv_states[:, 0, :, :]`, shape `[1,3,8192]` | Passed directly to M1 |
| 30 GDN recurrent caches | `states[:, 0, :, :, :]`, shape `[1,32,128,128]` | Passed directly to M1 |
| 10 full-attention K buffers | M2 buffer containing both appended rows | Reused without copying |
| 10 full-attention V buffers | M2 buffer containing both appended rows | Reused without copying |
| 10 tensor offsets | M2 final offset minus one | M1 overwrites the rejected draft slot |

Values beyond a tensor cache's offset are invisible. Passing the M2 K/V
buffers with `offset - 1` is therefore the exact prefix state; M1's
`slice_update` overwrites the discarded draft position with `correction`.

The route does not install partial primary state into the production cache.
On acceptance it installs M2 final state. On rejection it supplies the primary
state directly as inputs to M1 and installs only the M1 final state.

## Generation schedule

### Acceptance

1. Run M2 `[primary, draft]`.
2. Accept the draft using the existing pre-sampled target-prefix token.
3. Keep M2 final cache state, logits, and hidden state.

### Rejection

1. Run M2 `[primary, draft]`.
2. Select `correction` using the existing target-prefix sample.
3. Run fixed M1 `[correction]` from the captured-primary state.
4. Install M1 final cache state.
5. Use M1 logits and hidden state as the live target outputs for the next
   normal primary sample.

The next primary is still sampled at the start of the next cycle. This is why
the prior deferred-correction experiment is not reused: promoting the already
emitted correction to `pending_primary` skipped that normal primary sample,
changed RNG ordering, and changed output SHA and acceptance.

## MTP history

The committed MTP history entry for token `correction` must use
`verify_hidden[:, 0:1, :]`, the target hidden state after `primary`. It must not
use the M1 hidden output after `correction`. The M1 hidden output becomes the
live hidden state used to draft after the next normally sampled primary.

## Snapshot removal

The exact installed route no longer needs `snapshot_untrimmable_cache` or
`rollback_after_verify`. Generation selects the exact no-snapshot path from
the request-owned route object; it does not reread
`MTPLX_SKIP_VERIFY_SNAPSHOT` per cycle. Other strategies retain their existing
snapshot behavior unchanged.

## Reporting

Engagement is derived from existing request statistics:

- `m2_verify_calls = verify_calls`
- `m1_repair_calls = correction_tokens`
- `compiled_calls = verify_calls + correction_tokens`

The exact report names the two graph keys, reports zero fallback and zero
growth demotion, and does not add per-call counters.

## Correctness gates

- Fixed M1 and M2 GDN entrypoints match stock capture arithmetic for
  deterministic nonzero BF16 inputs and FP32 recurrent state.
- M2 `[A,D]` primary capture followed by M1 `[C]` matches reference M2 `[A,C]`
  for logits, hidden state, all 30 GDN cache pairs, and all ten attention cache
  prefixes.
- The exact target forward feeds the unchanged single common target sampler
  call before acceptance logic, and output tokens match for accept-only,
  reject-only, and mixed paths without an RNG experiment.
- MTP history pairs `C` with hidden-after-`A`.
- Acceptance performs one M2 call. Rejection performs one M2 plus one M1 call.
- The exact route does not call `pending_primary`, generic capture commit,
  snapshot, rollback, trim, or M2 repair.
- Flag-off construction and every non-exact strategy remain unchanged.

## Benchmark contract

After tests pass, benchmark the candidate worktree against unchanged
`960849b` using only:

- upstream `long_code_uncapped` natural stop;
- K1;
- TGY4;
- process-isolated C/X/C/X;
- repeat-2 verdict cells;
- exclusive GPU lane;
- identical control settings and sampler;
- exact M1/M2 graph-key and derived-call gates;
- zero compiled fallback and growth demotion.

Do not run fixed 1024/1024, TGY8, RNG experiments, fake component timers,
barriers, or direct-call microbenchmark verdicts. Record the commit, artifact,
hash, throughput, cycle time, tokens/cycle, acceptance, output SHA, and verdict
in `bench/a3b/OPTIMIZATION_LEDGER.md`. Do not stack a wash or regression.

## Expected performance boundary

At the traced rejection rate, every 1 ms removed per rejection saves about
0.176803 ms per cycle. M1 must be below 7.870 ms to save 0.5 ms/cycle and below
5.042 ms to save 1.0 ms/cycle. The zero-cost ceiling is about 191.37 tok/s at
unchanged tokens/cycle, so this experiment is necessary but not sufficient for
the 200 tok/s goal.

The next independent candidates, explicitly outside this design, are compiled
M2 target-logit summarization and packed full-attention Q/K/V projection.
