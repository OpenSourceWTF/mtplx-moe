# Qwen4 row-serial cycle fold

## Goal

Reduce the GPU starvation between a committed depth-one MTP window and the
next draft without changing Qwen4 arithmetic.  This is the first scheduling
stage toward 90 decode tokens/second on the production 16,384-input / 1,024-
output Python workload.

The retained production control is commit `39d0697b`, with temperature 1.0,
top-p 0.95, top-k 20, low thinking, native MTP depth one, physical target
verify M=2, capture-prefix commit, and a 1 GiB n-gram payload ceiling.

## Evidence and rejected physical-M2 fold

The profiler attributes 2.3191 seconds of the 14.9725-second decode window to
gaps between nonempty Metal command buffers.  A fold that replaced the
committed-history M1 and next-draft M1 with one physical M2 call looked like a
way to reduce this gap, but the exact 16K real-model gate rejected it.

Production-equivalent sequential M1+M1 versus physical M2 had these maximum
absolute differences:

- final logits: 0.375;
- final hidden: 0.21875;
- QSA K/V/indexer state: 0.5 / 0.046875 / 0.0625.

Running both sequential rows under the same compact-QSA phase still differed,
so the problem is not phase selection alone.  The first differences appear in
M-dependent input fusion, hyper projections, and Q/K/V/indexer projections.
The MTP MoE was exact.  At the inspected boundary, argmax and the filtered
temperature-1/top-p-0.95/top-k-20 distribution were identical, but the
divergent cache would feed every later cycle.  That is not a tie-only change
and is not an acceptable construction invariant.

The physical-M2 history/draft fold is therefore rejected.

## Architecture

Install one exact Qwen4 depth-one cycle-fold route during runtime
construction.  The route owns a compiled, row-serial body:

1. Run the committed-history update as the existing M1 operation under
   `attention_phase("ar_decode")`.
2. Run the following draft as the existing M1 operation under the ordinary
   `unknown` attention phase.
3. Return the final draft logits and hidden state plus every mutated MTP-cache
   leaf.
4. Submit the complete returned state with one `mx.async_eval` before event
   construction, token streaming, telemetry, or trace emission.

The body is logically folded into one submission but retains two physical M1
graphs.  It does not concatenate rows, change projection geometry, or select
M=2 kernels.

```text
accepted-prefix commit
        |
        v
history M1 (ar_decode)
        |
        v
draft M1 (unknown)
        |
        v
async submit logits + hidden + complete MTP cache
        |
        +---- CPU telemetry / streaming
        |
        v
next target verification
```

## Construction contract

The route is installed only when runtime construction proves all of the
following:

- exact `qwen4_exp` artifact and native one-layer MTP head;
- depth one, persistent committed-history MTP cache;
- QSA cache topology and fixed 1K-output capacity are installed;
- capture-prefix target verification and its compiled M=2 route are installed;
- the row-serial callable and full cache-root extractor are bound;
- exact production M1 shapes are warmed before the measured decode.

If any invariant fails, construction does not install this route.  The enabled
route has no eligibility recheck, environment read, exception fallback, or
engagement counter in the decode loop.

Request-varying control such as stop/max-token completion and context-copy
ownership remains outside the route.  A cycle ticket is issued only after
those decisions are settled.  An issued ticket is never cancelled or rolled
back.

## State ownership

The compiled callable follows the existing `CompiledVerifyBank` firewall:

- promote the one MTP QSA cache to fixed-capacity tensor offsets once;
- pass K, V, raw indexer keys, pooled keys, and all logical offsets as explicit
  graph inputs and outputs;
- reseed a shadow cache before every trace;
- mirror-commit returned leaves to the request-owned cache;
- demote only at the final-state boundary.

The async root set includes final logits, final hidden, K, V, raw indexer keys,
pooled keys, KV offset, indexer offset, and pooled offset.  Partial token-only
evaluation is prohibited because it leaves cache work to be rediscovered on
the next cycle.

## Decode integration

Depth-one production cycles always end with a known pending primary:

- accepted draft: the sampled bonus is the pending primary;
- rejected draft: the residual correction is the pending primary.

After target-cache commit and MTP rollback/history selection, construct the
row-serial ticket using the retained target hidden row and pending primary.
Issue it before `append_event`, `emit_new_tokens`, or `emit_trace`.  The next
loop iteration consumes the ticket instead of rebuilding the first draft.

Context-copy planning must run before ticket issue.  When the next iteration
belongs to context copy, no MTP ticket is submitted.  This is a genuine
runtime decision, not an invariant eligibility fallback.

Sampling and acceptance remain unchanged in this first stage.  Device-side
stochastic decision fusion is a separate follow-up after this scheduler is
measured, so RNG order cannot be confounded with the scheduling result.

## Failure handling

- A compile or parity failure aborts route installation before generation.
- An enabled route never catches an exception and silently runs stock code.
- Stop/max-token and context-copy decisions occur before submission, preventing
  speculative mutation of a cache that will not be consumed.
- The route never uses a dedicated Metal stream or Python thread pool; one
  submission owner retains all mutable MLX state.

## Verification

Focused gates:

1. Real 16K state: compiled row-serial output versus ordinary sequential M1+M1
   for final logits, both hidden rows, every active QSA cache element, and all
   offsets.
2. Repeat across QSA compression residues 0, 1, 2, and 3 and a cache-capacity
   boundary.
3. Full generation: identical seed, token digest, RNG state, acceptance and
   rejection counts, bonus/correction counts, and zero unexpected repair.
4. Stop, max-token, rejection-correction, all-accept bonus, context-copy, and
   final-state capture ownership.

Performance gate:

- alternate unchanged control and candidate on the exact production 16K/1K
  workload after one warmup;
- retain only a repeated decode-TPS improvement with no utilization regression;
- re-profile the named transition families and report GPU busy, idle, and
  utilization;
- commit and update PR #368 only after the parity and production gates pass.

## Follow-up stages toward 90 TPS

This fold can hide host work but cannot by itself reach 90 TPS.  After it is
retained or rejected by measurement:

1. fuse stochastic p/q acceptance, residual correction, and next-primary
   selection on device while preserving reserved NumPy RNG order;
2. submit the next PLE row dependency and draft state in the same cycle ticket;
3. re-profile active GPU time and reduce at least 1.3 seconds of kernel work,
   because eliminating all currently measured idle time only reaches about
   80.9 TPS.

Large monolithic kernels, cross-thread stream ownership, and physical-M2 MTP
arithmetic remain non-goals.
