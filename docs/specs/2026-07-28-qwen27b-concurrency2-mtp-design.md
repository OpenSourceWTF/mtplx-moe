# Qwen 3.6 27B Concurrency-2 MTP Design

Date: 2026-07-28

Status: design approved; prefill-chunking requirement added during execution

Target branch: `perf/qwen27b-concurrency2-mtp`

Target model: `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`

## Goal

Run one or two concurrent Qwen 3.6 27B requests through native depth-2 MTP
without dropping the two-request cohort to autoregressive generation.

The same owner must keep prompt prefill interruptible at the configured
1024-token boundary. A decode-ready K2 request runs before the owner starts the
next prefill chunk, so a long prompt cannot monopolize the model worker and
hide the concurrency-2 decode gain behind admission latency.

The enabled lane must multiplex at MTP cycle boundaries:

- one decode-ready request uses the existing depth-2 target-verify shape
  `B=1, T=3` (logical QMM `M=3`);
- two decode-ready requests share one target forward with shape `B=2, T=3`
  (logical QMM `M=6`);
- when a request joins, finishes, or is cancelled, the next cycle selects the
  fixed width-1 or width-2 route;
- request-local sampling, acceptance, commit state, output ordering, session
  identity, and cancellation remain independent.

The performance target is at least 1.35x the unchanged serialized K2 control's
aggregate output throughput at concurrency 2, with no more than a 1% regression
at concurrency 1.

## Existing Behavior

The production Qwen service currently uses the serial scheduler with
`decode_batch_max=1`. MTPLX's existing `ar_batch` mode already makes a
runtime concurrency decision, but two active requests are sent to mlx-lm's
batched AR generator and report `mtp_disabled_reason=batch_size_gt_1`.

That path does not meet this design's goal because it gives up the native MTP
speedup precisely when the second request arrives.

The Qwen q4 verify-kernel family already contains an `M=6` entrypoint and the
active runtime self-checks that kernel. This design does not assume that the
existing geometry is suitable merely because the name and row count match.
The real Qwen layer shapes, q4 affine layout, group size, activation dtype,
flattening order, and full-model scheduling behavior must pass construction
self-checks and paired benchmarks before the lane can be installed.

## Scope

In scope:

- Qwen 3.6 27B Optimized-Speed only;
- q4 affine trunk weights with group size 64 and the checkpoint's installed
  activation dtype;
- fixed native MTP depth 2;
- one or two active decode requests;
- `capture_commit` verification using the current Qwen committed-history
  contract;
- independent prompt prefill and session-bank restore before cohorting;
- configured 1024-token prefill chunks with a scheduler yield between chunks;
- request-local samplers, constraints, stop conditions, seeds, callbacks,
  metrics, cancellation, and final session-bank commits;
- construction-time topology validation, exact cache-ownership self-checks,
  profile-compatible numeric self-checks, and prebound width-1/width-2
  execution routes;
- local CPU, Metal, and end-to-end benchmark evidence.

Out of scope:

- Hy3, GLM, Laguna, A3B, or any other model family;
- concurrency greater than 2;
- MTP depths other than 2;
- changing the checkpoint, quantization, top-k, sampler defaults, or
  speculative acceptance algorithm;
- padding an absent request to keep the width-2 kernel active;
- a generic batching framework for arbitrary MTP backends;
- batching prompt prefill;
- globally shrinking prefill chunks below 1024 tokens;
- new per-token, per-layer, per-cycle, or per-dispatch engagement counters;
- silently falling back from the enabled MTP-pair lane to AR, serial MTP, or
  stock kernels after construction.

## Construction Contract

The concurrency-2 lane is installed once, after model load and before warmup.
Installation must prove:

1. the backend is Qwen 3.6 27B native MTP;
2. the requested speculative depth is exactly 2;
3. the target trunk is the expected q4 affine/group-size-64 layout;
4. every routed quantized linear has a supported `(K, N, dtype, layout)`
   geometry for both its width-1 and width-2 entrypoint;
5. Qwen attention caches support batched offsets and lossless extraction;
6. Qwen GDN/causal-convolution recurrent caches support merge and extraction
   without cross-row ownership;
7. actual-model `M=6` kernel output and one-cycle cache state pass the
   construction self-check against the unchanged `B=2` stock reference.

Installation produces an immutable route object containing:

- the fixed width-1 target callable;
- the fixed width-2 target callable;
- prebound cache merge and extraction functions for each layer type;
- the fixed depth, dtype, quantization, group size, hidden variant, and commit
  strategy;
- the scheduler's maximum cohort width of 2.

The enabled hot path may select only on the number of decode-ready requests,
which genuinely varies at runtime. It must not re-read environment variables,
revalidate model metadata, test eligibility, catch a custom-kernel failure and
retry stock, or count dispatches.

If the operator explicitly requests this lane and construction fails, server
startup fails once with the violated invariant. When the lane is not requested,
the existing service construction is unchanged.

## Runtime Architecture

### Request state

Each request owns an `MTPK2RequestState` containing:

- prompt and generated token history;
- target logits and post-norm hidden state;
- trunk attention and recurrent caches;
- committed MTP-history cache;
- draft state for the current cycle;
- sampler, constraint, stop, callback, and cancellation state;
- session-bank identity and final-commit metadata;
- cumulative statistics already present in `GenerationStats`.

No mutable cache object is shared between request states outside one cohort
forward.

### Cycle preparation

Each decode-ready request independently:

1. samples its primary target token from its current target distribution;
2. drafts exactly two MTP tokens with its own committed MTP history;
3. creates an immutable `MTPK2VerifyTicket` containing the three target input
   tokens, draft distributions, request-local acceptance context, and cache
   handles.

Preparing a ticket does not mutate authoritative target cache state.

### Chunked prefill admission

Each request owns its prompt cache and advances prefill by at most the installed
`prefill_chunk_tokens=1024` budget. After each chunk, the request yields to the
single owner scheduler.

The owner selects work in this order:

1. run one or two already decode-ready K2 verify tickets;
2. admit a pending request if an active slot is free;
3. run one prefill chunk for an admitted request;
4. re-evaluate decode readiness before starting another prefill chunk.

This retains the Laguna investigation's measured 1024-token throughput
boundary. It does not transplant Laguna's AR batch topology or globally reduce
the chunk size: Qwen prefill uses its own target cache, arithmetic, and profile,
and must pass Qwen-specific parity and throughput gates.

### Cohort formation

The single model-owner thread drains at most two verify tickets:

- one ticket selects the fixed `B=1, T=3` route immediately;
- two tickets select the fixed `B=2, T=3` route;
- a request is never delayed solely to wait for a partner;
- a second request may join only between target cycles;
- cancelled or finished requests are removed before route selection.

All admitted requests have already passed the fixed construction contract.
There is no per-cycle eligibility decision.

### Width-2 target execution

For two tickets, the coordinator:

1. stacks token rows in stable request order;
2. merges each request's target caches using the prebound per-layer operation;
3. invokes one target `forward_ar_capture` on `[2, 3]`;
4. evaluates the complete output and captured state once;
5. extracts logits, hidden state, attention caches, and recurrent captures
   back into request-local results using the same stable row order.

Flattened q4 linear work has logical `M=6`, so one fixed-weight read serves
both requests' three verify rows. The implementation must preserve the target
model's existing tensor layout; it must not transpose or repack activations
merely to imitate another model's kernel.

### Independent acceptance and commit

After extraction, each request independently runs the unchanged exact
speculative-acceptance algorithm with its own RNG, draft distributions,
constraint, and stop tokens.

The request's accepted prefix selects only its row's captured recurrent states
and advances only its row's attention offsets. Rejection repair and
`capture_commit` operate on the extracted request-local state. One request's
acceptance depth cannot select, trim, or commit another request's cache.

After commit, surviving requests return to the decode-ready queue. Cohort width
is selected again at the next cycle boundary, allowing `1 -> 2 -> 1` without
padding or reinitializing the surviving request.

### Prefill, session bank, and postcommit

Prompt prefill and session-bank restore remain request-local and use existing
code. Restored caches are converted to the installed request-local cache types
before decode admission. Cohort merge exists only for a target cycle.

Final session-bank commits receive extracted request-local caches and the
request's own token history. Cohort containers never enter the session bank.

## Interfaces

The implementation adds four internal contracts:

| Contract | Required contents and behavior |
|---|---|
| `Qwen27BK2DualLane` | Immutable width-1 and width-2 target callables, cache merge/extract callables, fixed depth 2, and maximum width 2. |
| `MTPK2RequestState` | Request-owned tokens, logits, hidden state, trunk and MTP caches, RNG and sampler state, constraints, stops, callbacks, cancellation, session identity, and existing generation statistics. |
| `MTPK2VerifyTicket` | Immutable request ID, three target input tokens, draft distributions, acceptance context, and request-local cache handles for one cycle. |
| `MTPK2PrefillTicket` | Immutable request ID, prompt slice of at most 1024 tokens, request-local target cache, and exact prompt-position metadata. |
| `MTPK2CohortRunner.step(requests)` | Accepts exactly one or two decode-ready request states and returns one cycle result per input in stable order. |

The public activation surface reuses the existing experimental cohort mode:

```text
--scheduler-mode mtp_cohort_experimental
--experimental-mtp-cohorts
--max-active-requests 2
--decode-batch-max 2
--batch-wait-ms 0
--prefill-chunk-tokens 1024
```

For the target Qwen contract, this mode installs `Qwen27BK2DualLane`. It must
not retain the existing `batch_size_gt_1` AR fallback.

## Error Handling

- Construction mismatch: fail server startup with the exact model, layer, and
  invariant that prevented installation.
- Construction self-check mismatch: fail startup; never install only part of
  the model's width-2 route.
- Cohort forward failure: fail the affected cohort clearly. Do not retry
  either request through a different execution lane.
- Cancellation before a cycle: remove the request before route selection.
- Cancellation during a shared forward: finish the forward, discard only the
  cancelled row, extract the survivor's authoritative state, and continue the
  survivor at width 1.
- Per-request rejection or stop: commit or finish that row independently,
  then select the next cycle width from the remaining decode-ready requests.
- Memory admission failure: leave the second request queued before cohort
  formation. Do not admit it and then downgrade the shared cycle.

## Testing Strategy

Implementation follows test-driven development.

### CPU state-machine tests

Write and observe failures for:

- immediate width-1 execution with no cohort wait;
- stable width-2 request ordering;
- `1 -> 2 -> 1` joins and departures;
- independent accept counts, RNGs, stop conditions, and callbacks;
- one-row rejection while the other fully accepts;
- cancellation before and during a shared cycle;
- no `ar_batch` fallback, and no retry through the width-1 MTP route after a
  width-2 execution failure;
- startup refusal for the wrong backend, depth, quantization, group size,
  cache type, or incomplete route table;
- no environment reads or eligibility checks from `MTPK2CohortRunner.step`.
- prefill slices never exceed 1024 tokens;
- a ready decode cycle runs before the next long-prompt prefill chunk;
- chunked and unchanged whole-prompt prefill produce the same final logits,
  hidden state, cache state, and generated tokens.

### Metal construction and parity tests

On the actual Metal-capable lane:

- validate every installed Qwen q4 shape at `M=3` and `M=6`;
- compare the candidate custom `M=6` output to unchanged stock QMM at the same
  `B=2, T=3` shape using the existing turbo-profile numeric contract;
- compare one full candidate target cycle to an unchanged `B=2` stock target
  cycle, including logits, hidden state, attention offsets, recurrent state,
  and extracted cache ownership;
- verify that mutating or committing one extracted row cannot change the other;
- run deterministic two-request acceptance and token tests against the
  unchanged `B=2` stock cohort;
- verify width-1 output remains identical to the current production K2 path.

`B=2` results are compared with an unchanged `B=2` reference, not with a
single-row execution whose floating-point scheduling may differ.

### End-to-end tests

- two simultaneous OpenAI streaming requests;
- a decode-ready short request arriving during a multi-chunk long prefill,
  proving admission between 1024-token chunks;
- one short and one long completion, proving mid-cohort departure;
- independent session IDs with restore and final session-bank commit;
- one tool/constraint request paired with an unconstrained request;
- client cancellation while both requests share a target forward;
- existing scheduler telemetry shows width 1 and width 2 through current
  fields without adding hot-path counters.

## Performance Gate

Use guarded, exclusive Metal windows and the unchanged production Qwen K2
runtime as the control. Run at least three paired repeats per cell.

Cells:

- concurrency 1, short prompt, 256 generated tokens;
- concurrency 2, two distinct short prompts, 256 generated tokens each;
- concurrency 2, two distinct approximately 4K prompts, 256 generated tokens
  each;
- concurrency 2 with uneven completion lengths to exercise `2 -> 1`;
- a production-shaped long-prompt pair to expose TTFT or prefill regressions.
- a long-prefill/short-request overlap cell to expose head-of-line admission
  delay.

Record:

- aggregate and per-request decode throughput;
- end-to-end throughput and TTFT;
- accepted tokens per target cycle;
- target forward and verify time from existing generation statistics;
- peak resident memory using `vmmap -summary` or `footprint`;
- generated-token and parity receipts;
- the scheduler's existing lane and active-batch fields.

Promotion requires all of:

- concurrency-1 throughput no worse than 1% below unchanged control;
- concurrency-2 aggregate output throughput at least 1.35x unchanged
  serialized K2;
- no material TTFT regression on the 4K or production-shaped cells;
- the overlap cell admits ready decode work between prefill chunks and does not
  regress long-prompt prefill throughput by more than 5%;
- all correctness and isolation gates green;
- no enabled-path fallback, invariant validation, environment read, or new
  engagement counter.

A candidate that misses any gate remains local and is not enabled in the
service script.

## Rollout

1. Implement and verify only on `perf/qwen27b-concurrency2-mtp`.
2. Keep the public mode experimental and off by default.
3. Do not change, restart, or benchmark through the live Qwen service outside
   a guarded exclusive window.
4. After local gates pass, present the receipts and branch state for approval.
5. Only after explicit approval, update the Qwen service launcher to request
   the fixed cohort lane and cap both active requests and decode width at 2.
6. Verify live health, one solo request, two simultaneous requests, session
   restore, tool use, and cancellation; restore the prior service immediately
   if any acceptance check fails.
7. Do not push, open a PR, or modify another model lane without separate
   authorization.

## Failure-Mode Review

### Cross-request cache contamination

Severity: critical.

The hybrid Qwen trunk contains both attention KV and recurrent GDN state.
Generic tensor concatenation is insufficient unless row ownership and
per-request offsets survive extraction. The design addresses this with
prebound per-layer merge/extract operations, same-shape stock references, and
mutation-isolation tests. The lane is not installed if any cache type lacks a
lossless operation.

### Independent rejection or completion corrupts cohort state

Severity: critical.

The two rows can accept different draft depths, stop independently, or be
cancelled. The design never commits from a shared cache container. It extracts
row state first, then applies the unchanged request-local acceptance and commit
logic, and reforms the cohort only at the next cycle boundary.

### `M=6` loses to two `M=3` cycles

Severity: critical to the performance premise.

The existing kernel's row count alone does not prove a Qwen win. Real Qwen
shapes may become dequant-ALU, register, or scheduler limited. The paired
unchanged-control gate rejects the feature unless aggregate throughput clears
1.35x while width 1 remains flat.

### Dynamic joining harms latency

Severity: minor if bounded as designed.

A request never waits for a partner, and a partner joins only after its own
prefill is complete. The maximum scheduling delay is the current target cycle.
The 4K and uneven-length benchmark cells make any remaining latency cost
visible before promotion.
