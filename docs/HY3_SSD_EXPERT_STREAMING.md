# Hy3 SSD expert streaming — fork design

Status: Phase 0 cache policy and trace simulator implemented; native I/O and
Metal execution are not implemented yet.

## Goal

Run final Tencent Hy3 4-bit on an Apple Silicon machine whose unified memory is
smaller than the full checkpoint. Keep the MoE routers (the "deciders"),
attention, norms, embeddings, output head, shared experts, and KV state
resident. Keep a learned hot set of routed experts resident per layer and load
cold experts from an expert-major SSD sidecar only after the router selects
them.

This belongs in MTPLX's MLX/Metal execution layer. It is not a patch to Apple's
kernel Metal driver.

## Why the cache is per layer

Expert 17 in layer 12 is a different tensor from expert 17 in layer 13. A
whole-model "path" is also too combinatorial to cache as a unit: Hy3 selects
multiple experts at every routed layer. The stable cache key is therefore:

```text
(model fingerprint, quantization fingerprint, layer, expert)
```

The full router path remains useful telemetry, but admission and eviction are
per-layer decisions.

## Proposed execution flow

```text
resident attention/shared path
        |
resident router -> top-k expert ids -> slot-bank lookup
                                      | hit
                                      +------> fixed Metal expert slot
                                      | miss
                                      +------> aligned pread from sidecar
                                               into persistent or transient slot
        |
quantized gate/up -> SwiGLU -> down -> weighted combine -> residual
```

The slot bank has two tiers:

1. Persistent decode-hot slots. Admission uses decayed frequency plus recency.
2. Transient service slots, at least as large as router top-k. Prefill misses
   and decode misses that are not hot enough use these slots without evicting
   the hot set.

The persistent allocation is per layer, but the physical transient Metal
scratch bank should be global and reused as layers execute sequentially. Eight
top-k scratch records are needed in memory once, not once per layer.

Eight scratch records describe the first single-stream, one-token decode path.
A prefill chunk, continuous decode batch, or MTP verify batch can have a much
larger union of expert ids. It must group tokens by expert and execute misses in
bounded waves, accumulating each token's weighted result; it must not require
the whole union to be resident at once. Phase 0 trace events are deliberately
one token at one layer, so a wider event is rejected instead of silently
under-sizing the native scratch bank.

Prefill and decode must not share one blind LRU policy. Prompt tokens touch a
wide one-off expert set and can erase the decode hot set immediately before it
becomes valuable.

The explicit hot tier is a hypothesis, not a foregone conclusion. Flash-MoE's
48 GB measurements found that a second Metal LRU duplicated the macOS page
cache, increased memory pressure, and lost to plain `pread`. Phase 1 must keep
two first-class modes: `persistent_slots=0` (transient buffers plus the OS page
cache) and the learned persistent bank proposed here. If the explicit bank
wins, cold reads should be tested with `F_NOCACHE` so the same expert record is
not retained twice. The winning policy is selected from route traces and
hardware measurements, not assumed from cache capacity alone.

## Phase 0 in this branch

- `mtplx/expert_streaming.py` implements the per-layer admission/eviction plan
  without importing MLX.
- `scripts/simulate_expert_cache.py` replays JSONL route traces and reports hit
  rate, bytes read, and an I/O-time floor for a cache size.
- `tests/test_expert_streaming.py` covers hot-set persistence, prefill
  isolation, transient service, admission, and I/O accounting.

Trace event format:

```json
{"phase":"decode","layer":12,"experts":[7,19,31,44,58,90,121,180]}
```

Example simulation:

```bash
python scripts/simulate_expert_cache.py routes.jsonl \
  --persistent-slots-per-layer 64 \
  --transient-slots 8 \
  --expert-record-bytes 10000000 \
  --ssd-gib-per-second 5.5
```

`expert-record-bytes` must come from the validated layout manifest. Do not infer
it from the repository's total byte size once dense/shared tensors have been
split out.

## Hy3 Q4 sizing hypothesis

The final Hy3 config and the pinned `pipenetwork/Hy3-4bit` community conversion
agree on 80 target layers,
layer 0 dense, 192 routed experts, top-8 routing, hidden size 4096, expert hidden
size 1536, affine 4-bit groups of 64, and 8-bit routers. For one routed expert,
gate/up/down each contain `4096 * 1536` values. Including packed Q4 weights and
fp16 scales+biases gives this provisional record size:

```text
3 * (4096 * 1536 * 4/8 bytes
     + (4096 * 1536 / 64) * 2 bytes scales
     + (4096 * 1536 / 64) * 2 bytes biases)
= 10,616,832 bytes
= 10.125 MiB per (layer, expert)
```

The pinned community conversion contains 149.98 GiB of routed experts and 4.612
GiB of non-routed base tensors. Its 79 resident 8-bit routers, including scales,
affine biases, and fp32 correction biases, total only about 63 MiB. A cold
top-8 token needs about 6.25 GiB of expert records before cache hits. At an
effective 5.5 GiB/s random-read rate, cold I/O alone caps decode below 0.9
tokens/s.

On the 128 GB reference Mac, 96 persistent slots per routed layer consume about
74.99 GiB. Under uniform routing that is a 50% capacity hit rate and a 1.76
tokens/s I/O-only ceiling; useful expert skew can improve it, while compute and
cache-management overhead reduce it. These are planning numbers, not benchmark
claims. The sidecar exporter must replace them with sizes measured from its
manifest.

The examined final-model community Q4 indexes omit checkpoint layer 80 even
though their config still declares one NextN predictor. They are AR-only in
practice. Initial streaming work must prove AR first; MTP requires separately
packaging and validating the official BF16 layer-80 weights (or a new Q4
conversion) and adding its own expert bank.

Routing is a discrete, precision-sensitive boundary. The parity baseline must
retain the Q4 artifact's resident 8-bit gate and run router math/selection in
FP32. A future conversion should A/B resident BF16 gate weights as well: all 79
gates are only about 124 MiB in BF16, so improving routing fidelity costs about
61 MiB over the inspected Q8 layout, negligible beside the expert bank.

Context length materially changes the cache budget: BF16 KV for the 80 target
layers is about 0.3125 MiB per token, or 10/20/40/80 GiB at roughly
32K/64K/128K/256K. Cache sizing must reserve KV and macOS/runtime headroom
before assigning persistent expert slots.

Primary configuration references:

- <https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/config.json>
- <https://huggingface.co/XavierLocalAI/Hy3-4bit/blob/c4f3e2d53d0de330dd3750cc352fbe3a62fef956/config.json>
- <https://huggingface.co/pipenetwork/Hy3-4bit/blob/160619d3f96c8470350b6dac0ef033a8381551e3/config.json>

## Phase 1: route telemetry and expert layout

1. Land or vendor a pinned Hy3 MLX model implementation; upstream mlx-lm Hy3
   work is not yet a released dependency.
2. Wrap each Hy3 MoE router to record selected expert ids by phase and layer.
   Decode traces must include batch/session identity so cross-request locality
   is not mistaken for within-sequence locality.
3. Generate and hash a manifest of the original safetensors expert-slab file
   offsets first. The expert dimension is contiguous, so this proves selective
   loading without writing a second 150 GiB artifact.
4. Benchmark those component reads against an optional expert-major sidecar.
   The derived format combines gate/up/down packed Q4 weights plus
   scales/biases into aligned records and is kept only if fewer, larger reads
   justify its disk cost.
5. Export a dense checkpoint without routed expert payloads.
6. Validate that dense + one loaded expert record reproduces the resident
   module's output on deterministic vectors before optimizing I/O.

The manifest must pin the source model revision, tensor names/shapes/dtypes,
quantization mode and group size, byte offsets/lengths, alignment, and hashes.

The closest public MLX implementation seam is SharpAI's MIT-licensed
`mlx-swift`/`mlx-swift-lm` fork: it writes an expert slab from safetensors
directly into an already-evaluated MLX array at a slot byte offset, then uses a
stacked `gatherQuantizedMM`. This demonstrates that the buffer path is possible,
but its current code relies on caller-side global GPU synchronization and lacks
the manifest, expert-index, epoch, and source-integrity checks required here.
Port the narrow mechanism with attribution; do not copy its lifetime assumptions
unchanged.

- <https://github.com/SharpAI/mlx-swift/blob/133864c733c8d4178547f8fe92897da6a788368f/Source/Cmlx/mlx-c/mlx/c/fast.cpp#L865-L1188>
- <https://github.com/SharpAI/mlx-swift-lm/blob/b9bf50bdafef02fffd5b83598a61bbf7d47434f9/Libraries/MLXLMCommon/SwitchLayers.swift#L48-L421>

## Phase 2: native slot bank

Add a nanobind MLX extension beside `native_extensions/verify_mlp`:

- Allocate fixed shared Metal buffers once; never allocate a new model-sized
  array on a cache miss.
- Read with positional, aligned `pread` into the selected slot. Start with the
  normal OS page cache baseline; measure it against explicit hot slots, with
  and without `F_NOCACHE` on misses.
- Expose slot generations so compiled/indirect command buffers cannot replay a
  stale expert after eviction.
- Run Q4 gate/up/SwiGLU/down from slot buffers and return the weighted expert
  contribution as an MLX array.
- For multi-token calls, group rows by expert and dispatch bounded expert waves;
  preserve the original token/router-weight mapping during accumulation.
- Export counters: hits, misses, bytes, read latency, per-layer occupancy,
  evictions, and time in router/I/O/Metal.

Cold-record `mmap` page faults are not the baseline. Existing Apple Silicon
Flash-MoE measurements report a large cold-`mmap` regression versus explicit
reads. The fork should reproduce that comparison on this machine rather than
assuming virtual memory is an expert cache.

## MTP sequencing

Start with AR correctness and throughput. Hy3's MTP head has its own routed MoE
layer and therefore needs a separate slot bank. A verify batch can request the
union of experts for several speculative positions, increasing SSD traffic.
MTP is promoted only if it beats AR while passing exactness and repetition
gates under the same cache budget.

## Validation gates

1. Sidecar record hashes and bounds validate before any inference.
2. Router ids and scores match the resident implementation.
3. Each streamed expert projection matches resident Q4 output within the
   quantized kernel tolerance.
4. Layer output, logits, and greedy tokens match on fixed vectors.
5. Cache hit/miss behavior cannot affect numerical output.
6. Corrupt/short reads fail closed; stale slot generations cannot execute.
7. Benchmarks report prefill and decode separately and include cold, warm, and
   steady-state runs.

## Go/no-go measurements

- Real expert record bytes and dense resident footprint.
- Decode expert-frequency distribution and hit-rate curves at 8/16/32/64/96
  persistent slots per layer.
- Effective random-read bandwidth into the chosen Metal-buffer path.
- OS-page-cache-only versus explicit-hot-bank throughput and memory pressure.
- I/O bytes/token after cache hits.
- AR tokens/s and MTP tokens/s with identical quality gates.
- Memory pressure, thermals, and SSD writes. Expert streaming should be
  read-only; large write counts indicate a packaging/cache bug.
