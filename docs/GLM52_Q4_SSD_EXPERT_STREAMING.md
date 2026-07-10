# GLM-5.2 Q4 SSD expert streaming

Status: AR implementation complete on the experimental branch, including FP32
routing, IndexShare, resident-only loading, native positional I/O, and direct
MLX/Metal slot-bank execution. Full-checkpoint parity and hardware benchmarks
remain release gates.

This document specializes the shared design in
[`MOE_SSD_STREAMING_PLAN.md`](MOE_SSD_STREAMING_PLAN.md) for
`mlx-community/GLM-5.2-4bit`.

## Target and artifact contract

Pinned sources used for the design:

- Official model: `zai-org/GLM-5.2`, revision
  `b4734de4facf877f85769a911abafc5283eab3d9`.
- Official GLM implementation/documentation: `zai-org/GLM-5`, revision
  `431634c155dab6cc2e30535c6eaa26c01caac675`.
- Q4 target: `mlx-community/GLM-5.2-4bit`, revision
  `6b347a6472d46bf55de65ee34032136a3929d778`.

The Q4 artifact contains 91 safetensors shards, 3,481 tensors, and
418,320,895,488 tensor bytes (389.5917 GiB). Its index ends at transformer
layer 77. Although the config declares one next-token-prediction layer, the Q4
artifact contains no layer-78 MTP tensors; it is an autoregressive target.

Do not let an absent layer 78 trigger a generic all-shard fallback. The loader
must identify this artifact as AR-only and either continue in explicit AR mode
or fail an explicit MTP request before loading routed tensors.

## Exact sparse layout

GLM-5.2 has 78 target layers:

- Layers 0-2 use dense MLPs.
- Layers 3-77 are 75 sparse MoE layers.
- Hidden size is 6,144.
- Expert intermediate size is 2,048.
- Every sparse layer has 256 routed experts and selects exactly eight.
- Every sparse layer also has one shared expert, which remains resident.

One affine Q4/group-64 routed expert consists of gate, up, and down
projections. The inspected MLX tensor layouts are:

```text
gate/up packed weights: [256, 2048, 768] uint32
gate/up scales+biases:  [256, 2048,  96] bfloat16
down packed weights:    [256, 6144, 256] uint32
down scales+biases:     [256, 6144,  32] bfloat16
```

For one expert, each projection represents `6144 * 2048` source parameters.
Packed weights plus BF16 scale and affine-bias metadata give:

```text
one projection =  7,077,888 bytes =  6.75 MiB
one expert     = 21,233,664 bytes = 20.25 MiB
```

The record is exactly 1,296 blocks of 16 KiB, which is convenient for an
aligned expert-major sidecar. Alignment does not remove the requirement to
validate each component's offset, length, shape, dtype, and hash.

Across `75 * 256` experts, routed records occupy:

```text
407,686,348,800 bytes = 379.6875 GiB
```

All non-routed Q4 artifact tensors total:

```text
10,634,546,688 bytes = 9.904193 GiB
```

That resident figure includes attention, shared experts, embeddings, dense
layers, routers, norms, and output tensors. It does not include growing KV
state, runtime/OS headroom, or expert slot buffers.

## Router-first execution

At sparse layer `n`, the resident attention path produces the hidden state,
then the resident MoE gate selects experts. Only after those ids are known does
the runtime resolve or load the selected records and execute their MLPs.

```text
hidden_n
  -> resident attention/DSA
  -> resident MoE router -> ids[8], weights[8]
  -> lookup (n, id) in fixed expert bank
  -> pread missing records into bounded slots
  -> gate/up -> activation -> down
  -> weighted sum + resident shared expert + residual
  -> hidden_(n+1)
```

It is not possible to run all 75 MoE routers at token start. Router `n + 1`
needs `hidden_(n+1)`, which includes the result of the experts selected at
layer `n`. Safe overlap begins only after the current layer's ids are known:
reads for multiple misses, execution of hot hits, and shared-expert compute can
overlap if their final weighted combine preserves reference semantics.

## Routing must be FP32-exact

The official GLM-5.2 configuration requires FP32 router math. The behavioral
oracle is the official/merged `modeling_glm_moe_dsa.py` implementation. For
each sparse layer, the adapter must preserve this ordering:

1. Project the hidden state with the router in FP32.
2. Apply sigmoid to produce unbiased expert scores.
3. Add the FP32 expert correction bias for selection only.
4. Select top eight using the corrected scores.
5. Gather the original, unbiased sigmoid scores at those ids.
6. Normalize the eight gathered weights.
7. Multiply by the routed scaling factor (`2.5`).

The Q4 conversion stores each gate weight as BF16 `[256, 6144]` and its
correction bias as FP32 `[256]`. The 75 gates and correction biases total
236,006,400 bytes (225.073 MiB), so keeping them resident is inexpensive
relative to the 379.6875 GiB expert corpus.

The inspected Q4 config predates the explicit `moe_router_dtype` field, and a
BF16 matmul followed by an FP32 sigmoid is not equivalent to an FP32 router
projection. The GLM adapter must force the official behavior and prove exact
top-8 parity before streamed expert execution. A one-id difference is a model
execution difference, not a tolerable numeric approximation.

## DSA IndexShare is a separate selector

GLM-5.2 also uses a DSA attention indexer to select historical token positions.
It must not be confused with the MoE expert router:

| Selector | Selects | Cache effect |
| --- | --- | --- |
| DSA indexer | Historical KV/token positions | Controls sparse attention; remains resident |
| MoE router | Eight feed-forward experts | Drives SSD expert record lookup |

The correct IndexShare schedule has 21 full indexer layers:

```text
0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34,
38, 42, 46, 50, 54, 58, 62, 66, 70, 74
```

The remaining 57 layers reuse indices from the preceding full layer. The
streaming branch must first vendor or land an IndexShare-correct GLM-5.2 model
adapter; stock model construction that creates a new indexer at every layer
will request nonexistent parameters and cannot serve as the reference.

Exact BF16 MLA+DSA cache storage is about 95,232 bytes per token across the
target model. Planning examples are:

| Context | KV/DSA cache |
| ---: | ---: |
| 32K | about 2.906 GiB |
| 64K | about 5.813 GiB |
| 128K | about 11.625 GiB |
| 256K | about 23.25 GiB |
| 1M | about 93 GiB |

These bytes must be reserved before allocating persistent experts. Correctness
tests must cross the 2K and 4K regions; short prompts do not exercise the
failure modes reported around incomplete DSA/IndexShare implementations.

## Memory-budget examples

One fully cold token selects `75 * 8 = 600` expert records:

```text
600 * 20.25 MiB = 12,150 MiB = 11.865234 GiB/token
```

At an effective 5.5 GiB/s SSD-to-slot rate, the cold I/O ceiling is only about
0.46 token/s before compute. A 90% record hit rate reduces reads to about
1.1865 GiB/token and gives a theoretical I/O ceiling near 4.6 tokens/s. These
are ceilings, not throughput claims.

Equal persistent capacity across all 75 sparse layers costs:

| Slots/layer | Persistent expert bytes | Capacity fraction/layer |
| ---: | ---: | ---: |
| 8 | 11.8652 GiB | 3.125% |
| 16 | 23.7305 GiB | 6.25% |
| 32 | 47.4609 GiB | 12.5% |
| 48 | 71.1914 GiB | 18.75% |
| 64 | 94.9219 GiB | 25% |
| 96 | 142.3828 GiB | 37.5% |
| 128 | 189.8438 GiB | 50% |

Top-8 single-token transient scratch needs only eight records globally:

```text
8 * 20.25 MiB = 162 MiB
```

It is global because layers execute sequentially. The persistent allocation is
per layer because expert 17 in layer 12 is unrelated to expert 17 in layer 13.

The user-facing planner accepts:

- A total memory limit.
- Context-token budget.
- Runtime/macOS reserve.
- Optional expert-cache limit.
- Transient slot count, defaulting to top-8 for single-token decode.
- Explicit I/O staging and execution-workspace bytes when they are not aliased
  to fixed expert slots.

It subtracts the 9.9042 GiB resident model, KV/DSA bytes, reserve, and transient
scratch before deriving an integer slot count per sparse layer. Startup fails
if fixed costs do not fit. Native allocation must use that fixed result and
must not create a full expert tensor or grow the bank on a miss.

The context value is the aggregate maximum live KV-token count across admitted
sequences; zero is load-only. The runtime must apply MTPLX's MLX memory cap
before loading, reject a conflicting `MTPLX_MEMORY_LIMIT_BYTES`, and prevent
context or batch admission from growing beyond the resolved plan.

## GLM-specific implementation seams

### Resident-only loader

The load path must intercept before generic `mlx_lm.load`, which otherwise
materializes the full 389.6 GiB artifact. It should:

1. Validate the pinned model and quantization fingerprints.
2. Instantiate the IndexShare-correct target model.
3. Load only the resident allowlist.
4. Bind each routed `(layer, expert)` to a manifest record without loading it.
5. Reject unexpected or missing resident tensors.
6. Treat omitted layer 78 as explicit AR-only metadata.

### Routed MLP seam

Keep the gate and shared expert resident. Replace the stacked resident
`switch_mlp(x, ids)` operation with a slot-backed equivalent that consumes the
same selected ids and weights. The first native kernel should preserve the
existing quantized gate/up activation/down sequence exactly; cache policy must
not be visible to model math.

For prefill or continuous batches, selected ids may have a union much larger
than eight. Group rows by expert, run the union in bounded waves, and accumulate
each row with its original router weight. Never size scratch to an unbounded
batch union.

### Slot lifetime

A slot is not reusable when an MLX call returns; it is reusable only after the
last Metal command reading it has completed. Associate every fill and dispatch
with a generation, validate the expected `(layer, expert, generation)` at
dispatch, and fence before overwrite. Use a full synchronization for the first
correctness baseline if necessary, then replace it with narrower command-buffer
or shared-event fencing.

## Build order and gates

1. **Land the exact layout descriptor and planner.** Reproduce all byte totals
   above and prove every plan remains under its user limit.
2. **Implement the checked manifest.** Sampled original-shard reads and
   optional sidecar reads must be byte-identical.
3. **Land IndexShare-correct resident loading.** Load no routed payload and
   match the fully resident model before 2K and beyond 4K context.
4. **Prove FP32 router parity.** Exact ids and tolerance-matching weights at all
   75 sparse layers are a hard gate.
5. **Add fixed native slots and checked `pread`.** Corruption, short-read, and
   stale-generation tests must fail closed.
6. **Bind slot-backed Q4 kernels.** Projection, layer, logits, and greedy-token
   comparisons must pass for forced hits and forced misses.
7. **Add bounded batching and telemetry.** Measure prompt and decode
   independently; report bytes/read latency/hit rate/kernel time/tokens per
   second.
8. **Tune cache policy.** Compare OS page cache alone against explicit hot
   banks at multiple slot counts before selecting a default.
9. **Evaluate MTP separately.** Only after AR correctness and sustained
   throughput are established.

## Benchmark matrix

| Dimension | Required points | What it answers |
| --- | --- | --- |
| Context | <2K, 4K, 32K, 64K, hardware-permitting longer | IndexShare correctness and KV pressure |
| Persistent slots/layer | 0, 8, 16, 32, 48, 64, 96, hardware-permitting 128 | Hit curve and memory/throughput tradeoff |
| Cache state | Cold filesystem cache, warm, steady decode | Separates SSD, OS cache, and expert-bank effects |
| Phase | Prefill and decode | Prevents prompt throughput from hiding decode stalls |
| Workload | Single stream, continuous batch, forced high expert union | Tests scratch-wave scheduling |
| Admission | Transient-only prefill, shared admission, decode-only hot admission | Measures cache pollution |
| Router | Official FP32 vs candidate adapter | Proves exact ids before performance work |
| Slot path | Resident reference, streamed hit, streamed miss | Proves cache-invariant math and isolates I/O |
| Lifetime stress | Forced read delay, rapid eviction, repeated generations | Detects stale Metal reads |
| Quality | Fixed greedy suite and long generations | Detects drift, gibberish, or repetition |

Every result must state the immutable artifact revision, hardware and SSD,
macOS, memory and reserve limits, context, batch/prompt shape, record alignment,
slot count, admission mode, and warm/cold state. Report peak memory, bytes read
and written, router/read/kernel time, hit rate, time to first token, prefill
tokens/s, and decode tokens/s.

## MTP boundary

The official BF16 model has an additional layer 78, but the selected community
Q4 artifact omits it. MTP also shares top-level embeddings and output head and
has its own router/expert bank; constructing an independent random output head
is incorrect. A separate community MTP conversion is not assumed trustworthy
or layout-compatible with this target. Generate a Q4 MTP artifact from the
pinned official source, validate router dtype/correction bias, and add it as a
separate manifest and memory reservation.

MTP verification selects experts for several speculative positions, so its
record union and I/O behavior differ from one-token AR decode. It is enabled
only if it matches the resident reference, passes repetition tests, stays under
the same user memory limit, and improves end-to-end throughput.

## References

- <https://huggingface.co/zai-org/GLM-5.2/tree/b4734de4facf877f85769a911abafc5283eab3d9>
- <https://github.com/zai-org/GLM-5/tree/431634c155dab6cc2e30535c6eaa26c01caac675>
- <https://huggingface.co/mlx-community/GLM-5.2-4bit/tree/6b347a6472d46bf55de65ee34032136a3929d778>
- <https://github.com/huggingface/transformers/blob/bca7eee6650f22fbfcb124c4642f343a3afd21f1/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py>
- <https://github.com/ml-explore/mlx-lm/pull/1410>
- <https://github.com/ml-explore/mlx-lm/pull/1463>
