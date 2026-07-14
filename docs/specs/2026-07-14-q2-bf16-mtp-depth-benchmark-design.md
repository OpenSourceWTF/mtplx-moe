# Q2 trunks with BF16 MTP depth benchmark

## Status

Approved in conversation on 2026-07-14, with the following additions:

- report ingestion throughput, prefill time, decode throughput, and MTP hit rates;
- use a 512-token qualification test and an exact 1,024-token standard
  benchmark, each with 128 generated tokens;
- sweep Hy3 recurrent draft lengths 1 through 4 and GLM-5.2 lengths 1 through
  5.

This document is the implementation contract. The feature remains experimental
and off by default.

## Premise

The benchmark should exist. Neither current expert-Q2 model can enable MTP in
the streamed harness, and the retained Hy3 BF16-MTP result is not a valid
answer: it diverges from deterministic AR at generated token 10. Without a
correctness-gated depth sweep, MTPLX cannot determine whether a BF16 MTP head
helps the faster Q2 trunks or which recurrent draft length is useful.

The work is proportional only if it stays isolated to artifact packaging,
streamed MTP injection, correctness instrumentation, and the benchmark lane.
It must not change default serving behavior or promote either Q2 trunk, both
of which already failed their independent quality gates.

## Precision contract

### Target trunks

- `hy3-expert-q2`: routed expert projections are affine Q2, group size 64,
  with BF16 scales and biases. The router weight is BF16; the projection runs
  in BF16 and routing logits/selection are promoted to FP32.
- `glm52-expert-q2`: routed expert projections are affine Q2, group size 64,
  with BF16 scales and biases. The router weight is BF16; router matmul,
  correction, and selection run in FP32.

### MTP heads

- Hy3 layer 80 and GLM-5.2 layer 78 remain BF16 for all layer-local weights,
  except their FP32 router correction biases.
- MTP routed experts are resident BF16 weights. They are not converted to Q2.
- Embeddings and output heads are shared with the target model, as defined by
  each architecture. Therefore a "BF16 MTP head" does not introduce a second
  BF16 embedding or LM head beside a quantized target.
- GLM MTP router arithmetic must match the target's FP32 routing path.

## Scope

### 1. Isolated integration branch

Use the flat worktree
`/Users/davidtai/projects/OpenSourceWTF/.worktrees/q2-bf16-mtp-bench` on branch
`codex/q2-bf16-mtp-bench`. Start from the completed Hy3-Q2 head and integrate
only the GLM-Q2 changes required for this benchmark. Do not modify either
completed artifact branch or the repository's default branch.

### 2. GLM BF16 MTP artifact

The pinned source is `zai-org/GLM-5.2` revision
`b4734de4facf877f85769a911abafc5283eab3d9`. The source staging directory is
external to the repository:

`/Users/davidtai/.cache/huggingface/glm52-mtp-layer78-source`

Extract exactly the 791 `model.layers.78.*` tensors from official shards
`00270` through `00274` into an exact-bit consolidated artifact under:

`/Users/davidtai/.cache/huggingface/glm52-mtp-layer78`

The selected payload is 19,905,841,664 bytes. Its manifest must bind the source
repository, revision, index digest, five source shard digests, selected tensor
inventory, shapes, dtypes, byte count, output digest, and producer commit.
Extraction must be atomic and fail on missing, extra, duplicate, mis-typed, or
mis-shaped tensors. No model weights belong in Git.

The existing Hy3 artifact remains:

`/Users/davidtai/.cache/huggingface/hy3-mtp-layer80/layer80-bf16.safetensors`

### 3. Streamed Hy3 MTP support

Permit the existing validated Hy3 BF16 injector for `hy3-expert-q2`. Keep the
current trunk manifest, slot geometry, and Q2 executor unchanged. Route every
benchmark draft length, including one, through `generate_mtpk` so all depths
share one verification and accounting implementation.

Hy3 has one trained NextN layer, and the vendor deployment recipe uses two
speculative tokens. Recurrent depths above that recipe are experiments, not
additional trained heads. Benchmark depths 1 through 4, but label depths 3 and
4 exploratory.

### 4. Streamed GLM-5.2 MTP support

Add a GLM-5.2-specific streamed MTP adapter instead of treating the checkpoint
as GLM-4. The adapter must:

- construct layer 78 with the local `glm52_mlx` DSA/MoE geometry;
- reuse the target embedding and target LM head;
- apply `enorm`, `hnorm`, `eh_proj`, the MTP decoder block, and shared-head
  norm in the checkpoint-defined order;
- keep the 256 routed MTP experts resident in BF16;
- use FP32 MoE router matmul and selection;
- expose the normal `mtp_forward`, `mtp_update_cache`, and `make_mtp_cache`
  runtime contract;
- support recurrent draft lengths 1 through 5 using the one trained layer.

GLM-5.2 declares `index_share_for_mtp_iteration=true`. At the first recurrent
step of a speculative cycle, layer 78 computes its DSA top-k indices. Steps
two through five reuse that selection. GLM's trained KVShare semantics reuse
the first MTP step's layer-78 cache view for the later recurrent steps; this is
not direct reuse of the target layer-77 KV tensors.

Headline runs use the production `committed` MTP-history policy. The committed
cache contains layer-78 MLA KV and indexer-key history derived only from
target-grounded states across speculative cycles. At committed offset `O`, D1
consumes the target-selected primary token and target hidden state, advances
both caches to `O+1`, computes top-k, and establishes the fixed cycle boundary
`C1`. D2 through D5 reuse that top-k and must not advance the indexer-key cache.
They may write discardable main-KV scratch, but the fixed indices must exclude
those speculative positions so attention semantically reads only through `C1`.

After verification, roll the main MTP KV back to `O+1`, retain the target-
grounded D1 row, and reconstruct deeper committed history only from target
verification states. Then invalidate cycle top-k. Rejection, stop, exception,
or adaptive early exit follows the same cleanup. Session snapshots include the
committed MLA and indexer-key caches but never cycle top-k or D2+ scratch. Cycle
scratch is request/cache scoped, not a model-global field.

The `cycle` history policy may be used only as a separately labeled diagnostic;
it is not a headline substitute. The adapter must fail closed if committed
history, IndexShare, or KVShare cannot be implemented and rolled back exactly.

For every cold MTP row with prompt IDs `p[0:N]`, target prefill produces the
required hidden variant `h[0:N]`. Start an empty committed MTP cache, then append
the aligned pairs `(h[i], p[i+1])` for `i=0..N-2`, in order. Thus the prompt
history has `N-1` rows before cycle one; GLM's MLA and indexer-key caches must
both report logical offset `N-1`. The campaign pins `mtp_position_mode=cache`,
so `mtp_history_position_base` is exactly zero. No SessionBank or prefix restore
is allowed in this campaign.
Rebuild this history after every MTP-row reset. Charge the full append cost to
`prompt_mtp_history_time_s` and total `prompt_eval_time_s`, and therefore to
reported MTP ingestion TPS. AR does not build MTP history.

### 5. Canonical generation and exactness

Use `generate_mtpk` for Hy3 draft lengths 1 through 4 and GLM draft lengths 1
through 5. The deterministic sampler is temperature 0, top-p 1, and top-k 1.
Pass an empty stop-token set and `max_tokens=128`, so stopping is disabled until
exactly 128 completion tokens have been emitted.
Each MTP row must produce exactly the same generated token IDs and length finish
reason as its paired AR control.

The retained Hy3 `generate_mtp1` evidence diverges at token 10 after an
accepted-bonus/rejected-draft sequence. Add a regression test for that event
shape. The depth sweep must not use the legacy path. If `generate_mtpk` also
diverges, stop the campaign and debug target-cache commit/rollback before
collecting performance evidence.

Exact verification guarantees output correctness even when recurrent MTP
quality declines; declining acceptance is a benchmark result, not permission
to alter target output.

## Benchmark design

### 512-token qualification and 1,024-token standard sweep

For each model, load the Q2 target and BF16 MTP head once. The Hy3 cells are
`AR,D1,D2,D3,D4`; the GLM cells are `AR,D1,D2,D3,D4,D5`. Reset target KV,
committed MTP history, cycle scratch, expert-cache state, generation counters,
and row-local resource counters between observations.

The qualification test uses exactly 512 input tokens and 128 generated tokens,
with one observation per cell. It establishes end-to-end exactness and records
all metrics, but it is not used for headline performance claims. Any depth that
fails exact AR equivalence at 512 is debugged before the standard sweep.
Qualification order is the listed cell order: AR followed by ascending depth.

The reported standard benchmark uses exactly 1,024 input tokens and 128
generated tokens, with three observations per cell in a fixed mirrored/rotated
schedule. The fixed schedules are:

- Hy3 replicate 1: `AR,D1,D2,D3,D4`;
- Hy3 replicate 2: `D4,D3,D2,D1,AR`;
- Hy3 replicate 3: `D2,D4,AR,D3,D1`;
- GLM replicate 1: `AR,D1,D2,D3,D4,D5`;
- GLM replicate 2: `D5,D4,D3,D2,D1,AR`;
- GLM replicate 3: `D2,D4,D1,AR,D5,D3`.

Each MTP observation is paired with the single AR observation in the same
replicate; that AR row is deliberately reused for every depth in its replicate.
The output records the schedule, replicate, cell position, and pair identity.

Before the standard rows, run and discard one 1,024-input/8-output warm-up for
AR and every depth so lazy allocation and depth-specific compilation are not in
headline timing. Reset all model and expert-cache state after every warm-up,
without clearing the compiled-program cache.

Build prompts with `_prompt_build_for_context` using `prompt_style=coding-agent`,
raw format, and thinking disabled. Use these model-specific prompt tails:

- Hy3: `benchmarks/fixtures/hy3-q2-benchmark-prompt.txt`, SHA-256
  `4825f939cd4c7228f1c2cd164515cc4541f71d4f3a6a6a180b31587991a582a4`;
- GLM: `benchmarks/fixtures/glm52-q2-benchmark-prompt.txt`, SHA-256
  `14530bcb61a1c7c0b56ec41104ed968977ae9f92c73bb10efc8884170f83df90`.

Input length is the final tokenizer-ID sequence returned by the builder,
including any tokenizer-added special token. There is no chat template, system
prompt, padding, or reusable prefix KV. Fail if the coherent tail is not
preserved or the builder does not return exactly 512 or 1,024 IDs. Record the
prompt-builder policy/version and metadata, tokenizer revision/digest, tail
digest, and final token-ID digest; every cell at a given model/context must use
the same IDs.

Common configuration:

- memory limit: 112 GiB;
- runtime reserve: 12 GiB;
- expert cache: 64 GiB;
- maximum live KV tokens: 4,096;
- cache policy: frequency, per layer, component banks;
- transient slots: 8;
- read chunk: 8 MiB with `F_NOCACHE`;
- single stream, deterministic seed 0;
- production MTP history policy: committed;
- MTP position mode: cache, position base 0;
- repetition stop and loop guard disabled;
- no resource/window telemetry in headline timing rows.

The AR cell runs with the MTP artifact resident so all cells have the same
memory reservation and expert-cache capacity. It calls target-only AR and does
not execute the MTP head.

Report the 512-token qualification rows separately from the 1,024-token
standard rows. Comparative performance conclusions come from the repeated
1,024-token rows; the 512-token rows remain visible as correctness and
short-context evidence.

### Diagnostic resource pair

After headline timing, run `AR,best-depth,best-depth,AR` with resource telemetry.
Do not mix those rows into headline TPS. Record expert bytes, read operations,
cache hits/misses/evictions, SSD throughput, reader occupancy, queue depth,
completion-fence activity, and peak memory.

An exact depth's primary score is the median of its three within-replicate
paired decode-TPS percentage changes. The secondary score is the equivalent
end-to-end-TPS change. The diagnostic depth is the exact depth with the highest
primary score; ties within one percentage point use the secondary score and
then the shallower depth. If every exact depth loses, select the least-negative
depth and report explicitly that none won.

Report every paired delta, median, range, and median absolute deviation. Define
the observed AR noise floor as the largest absolute deviation of an AR row from
the three-row AR median, divided by that median. A depth is only a performance
candidate if both its decode and end-to-end median paired gains exceed their AR
noise floors and at least two of three pairs improve. Three replicates may
identify a candidate but cannot promote production behavior; promotion requires
a separate confirmation campaign.

Every MLX window must atomically acquire the shared hardware lock, stop the
exact Qwen launch agent only after acquiring it, restore the exact initial
Qwen state in `finally`, and release the lock. The guard must match only MTPLX
benchmark processes; macOS `runningboardd`, `lsd`, and `containermanagerd` are
not benchmark blockers.

## Metric contract

Each row records raw counts and derived metrics. Derived values must be
recomputable from raw fields.

### Required user-facing metrics

- Reuse the existing `GenerationStats` fields `prompt_eval_time_s`,
  `prompt_target_prefill_time_s`, `prompt_target_prefill_tok_s`,
  `prompt_mtp_history_time_s`, and `decode_tok_s`; add fields only where the
  current schema lacks the required raw count or attribution.
- **Ingestion TPS:** newly ingested prompt tokens divided by total prompt
  evaluation time (`new_prefill_tokens / prompt_eval_time_s`). Also report
  target-only prefill TPS separately.
- **Prefill time:** total prompt evaluation seconds, target prefill seconds,
  and MTP-history/preparation seconds.
- **Decode TPS:** generated completion tokens divided by decode elapsed time.
  The decode interval includes all MTP draft, target verification, correction,
  and commit work but excludes prompt prefill. Report end-to-end completion TPS
  separately, never under the decode label.
- **MTP hit rate by configured draft length:** accepted drafts divided by
  evaluated drafts, plus accepted drafts per verification cycle.
- **Decode expert-cache hit rate:** read
  `expert_streaming_counters_by_phase.decode`, using `expert_hits /
  (expert_hits + expert_misses)`. The row also exposes the same derived value
  as `decode_expert_cache_hit_rate`. This measures routed-expert assignments
  served from the expert cache; it is not MTP acceptance. Do not use the
  aggregate `expert_streaming_counters.hit_rate` as the decode headline,
  because that aggregate also includes prompt prefill. Missing decode hits,
  misses, or hit rate is a failed evidence gate rather than a nullable metric.

### Per-depth acceptance metrics

Add `evaluated_by_depth`. Existing `drafted_by_depth` counts proposals made
before verification and is not a conditional accuracy denominator after an
earlier rejection. For each depth report:

- proposals drafted;
- proposals actually evaluated;
- proposals accepted;
- conditional hit rate = accepted / evaluated;
- cumulative prefix yield = accepted / drafted;
- mean acceptance probability;
- first-rejection depth distribution;
- accepted drafts per verification call;
- whole-block acceptance rate.

### Integrity and resource fields

Record model/trunk/MTP artifact identities, source commit, prompt identity,
sampler, requested and effective depth, execution order, token IDs, first
divergence, finish reason, repetition status, peak memory, complete memory
plan, target/draft/verify/repair seconds, bonus/correction counts, and the
expert-resource counters from the diagnostic pair. Also record machine/OS/MLX
identity, applied Metal memory limit, hardware-lock/Qwen transitions, and run
start/end timestamps; timestamps are provenance, not performance metrics.

Apply the 112 GiB limit before allocation. Reset the MLX peak counter before
model load/setup and before every qualification, warm-up, standard, and
diagnostic row; after each stage, record its stage-local peak and allocation
delta. Maintain a software `hard_peak_bytes` equal to the maximum of every
recorded stage peak, including compilation/warm-up, and never erase it from the
evidence when the MLX counter resets. Abort as soon as any stage exceeds the
cap. If stage-local reset is unavailable, label the value process-wide and fail
per-depth memory comparisons rather than presenting inherited high-water marks
as row-local peaks.

## Interfaces

The streamed benchmark CLI gains an explicit, model-aware draft-length option:
Hy3 accepts 1 through 4 and GLM accepts 1 through 5. A depth-sweep mode selects
the fixed schedules above while loading one runtime. Single-depth mode remains
available for diagnosis.

MTP artifact identity becomes model-family aware:

- Hy3 validates its existing layer-80 BF16 artifact and source revision.
- GLM validates the consolidated layer-78 artifact and provenance manifest.

Unsupported model keys, quantized MTP heads, missing provenance, depths outside
the model-specific range, non-committed headline history, missing GLM
IndexShare/KVShare support, or incompatible memory plans are CLI errors before
target allocation.

## Gates

A row is valid only when:

1. artifact inventory, dtype, revision, and digest validation passes;
2. measured input context is exactly the requested 512 or 1,024 tokens;
3. both AR and MTP emit exactly 128 completion tokens and finish by length;
4. MTP token IDs and finish reason exactly match paired deterministic AR;
5. every AR row at a model/context produces the same deterministic token IDs;
6. every setup/qualification/warm-up/standard/diagnostic stage peak and the
   retained `hard_peak_bytes` remain below the applied 112 GiB limit;
7. no repetition guard, load fallback, missing metric, or counter inconsistency
   occurs;
8. Qwen is restored to its exact pre-window state.

Candidate labeling follows the paired/noise-floor rule above. This campaign
does not promote production behavior and reports negative results without
weakening correctness, quality, or memory gates.

## Testing strategy

Use test-driven development. Required failing tests precede production edits:

- Hy3 expert-Q2 accepts only the validated BF16 layer-80 artifact;
- GLM exact 791-tensor inventory and provenance validation;
- GLM missing/extra/wrong-dtype/wrong-shape rejection;
- target embedding and LM-head object/weight sharing;
- GLM FP32 router parity on fixed hidden states, including near-tie routes;
- GLM IndexShare computes once at recurrent step one and reuses through five;
- GLM D1 advances MLA and indexer-key offsets, while D2 through D5 leave the
  indexer-key offset fixed and cannot attend beyond the D1 cache boundary;
- cold prompt history appends `(h[i], p[i+1])`, reaches `N-1` MLA/indexer-key
  rows, records its position base, and is rebuilt and timed for every MTP row;
- IndexShare/KVShare scratch resets on new cycles and rejection/rollback while
  target-confirmed committed history survives;
- stop, exception, and early-exit paths invalidate top-k and D2+ scratch;
- session snapshots retain committed MLA/indexer-key history but exclude cycle
  top-k and speculative scratch;
- headline rows require production `committed` MTP history;
- Hy3 depths 1 through 4 and GLM depths 1 through 5 all use `generate_mtpk`
  and record requested/effective depth;
- accepted-bonus followed by rejection preserves AR token output;
- `evaluated_by_depth` and conditional/cumulative rates use correct
  denominators;
- exact 512/1,024 prompt construction, prompt-policy/digest identity,
  128-token output, and output schema;
- exact fixed schedules, within-replicate AR pairing, and three standard
  observations per cell;
- warm-up rows are discarded and cannot enter headline aggregates;
- qualification/warm-up/compile peaks remain in `hard_peak_bytes` after reset;
- peak-memory reset produces stage-local rather than inherited peaks;
- 112 GiB memory-plan rejection before allocation;
- benchmark guard ignores unrelated macOS daemons and restores Qwen on success,
  failure, interruption, and timeout.

Before headline measurement, run tiny-model tests, fixed-hidden router/indexer
parity, real-artifact structural verification, one-token smoke, 32-token AR
equivalence at every depth, and then the 512-input/128-output qualification at
every depth.

## Failure-mode check

### Critical: benchmark measures a broken speculative loop

Evidence already shows legacy Hy3 MTP divergence. Mitigation: one canonical
`generate_mtpk` path, a regression for pending bonus plus rejection, and exact
AR equality before any performance row is valid.

### Critical: GLM recurrent IndexShare is implemented as repeated index search

That would benchmark a different architecture and distort both accuracy and
TPS. Mitigation: explicit cycle-scoped shared top-k state, call-count/state
tests through depth five, and fail-closed cache-policy constraints.

### Critical: GLM KVShare commits or attends to speculative D2+ cache rows

That changes the distribution the MTP layer was trained against and can leak
rejected state into later cycles. Mitigation: explicit D1 boundary and
indexer-key offsets, fixed-top-k range tests, rollback to `O+1`, reconstruction
only from target verification states, and unconditional scratch invalidation.

### Critical: GLM's 18.54 GiB BF16 head causes duplicate load workspace or
memory pressure

Mitigation: exact-bit consolidated artifact, one runtime load per sweep,
64 GiB expert cache, explicit 12 GiB reserve, pre-allocation budget validation,
and retained load-time/steady-state peak memory. Abort rather than swap.

### Minor: thermal/order bias favors a depth

Mitigation: three observations per cell in the fixed mirrored/rotated schedule and
headline rows without sampling telemetry.

### Minor: "hit rate" hides unevaluated deeper proposals

Mitigation: add `evaluated_by_depth` and report conditional hit and cumulative
prefix yield separately.

### Minor: deeper Hy3 recurrence is outside its common deployment recipe

Mitigation: label depths 3 and 4 exploratory and make no production claim
from them.

## Non-goals

- Quantizing either MTP head to Q2 or Q4.
- Changing Q2 trunk weights, page serialization, cache policy, or default model
  selection.
- Repairing either Q2 trunk's failed perplexity/quality gate.
- Enabling MTP by default in serving.
- Claiming results represent vLLM/SGLang kernels or distributed GPU serving.
- Replacing production committed history with the cycle-policy diagnostic.
- Promoting a benchmark candidate directly into production.

## Rollout

Land the artifact extractor and strict loader first, then Hy3 and GLM runtime
support, then the canonical depth path and metrics, and finally the benchmark
campaign. Keep every change behind explicit model key, MTP artifact, and draft
length flags. A valid negative result is the expected acceptable outcome.

## Evidence

- Tencent's Hy3 model card identifies one MTP layer and deploys it with two
  speculative tokens:
  <https://huggingface.co/tencent/Hy3#deployment>.
- The pinned GLM-5.2 configuration declares
  `index_share_for_mtp_iteration=true`:
  <https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/config.json>.
- GLM's architecture notes describe seven-step recurrent training/inference
  and reuse of both first-step top-k and KV cache:
  <https://z.ai/blog/glm-5.2>.
- vLLM's recurrent proposer independently shows first-step top-k computation
  followed by skip-top-k recurrent steps:
  <https://github.com/vllm-project/vllm/blob/50ac1c7bab47f14d56d86967532574824d02260e/vllm/v1/spec_decode/llm_base_proposer.py#L568-L767>.
