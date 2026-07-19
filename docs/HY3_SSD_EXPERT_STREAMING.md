# Hy3 SSD expert streaming — fork design

**Scope note (2026-07): this document describes the original Q4, depth-1
design.** The shipping line is Q2 streamed experts (`hy3-expert-q2`,
`~/.cache/huggingface/hy3-expert-only-mlx-q2`) at MTP depth 3, with dense
islands (`--island-layer-count`) and load-time projection quantization
(`--proj-quant`). Sizing figures below are Q4-era planning numbers, not
measurements of the current path. See "Current levers" at the end for the
flags that actually drive today's results.

Status: AR implementation complete on the experimental branch, including the
resident Hy3 overlay, native positional I/O, and direct MLX/Metal slot-bank
execution. MTP through the layer-80 NextN head is implemented behind an
opt-in flag (see "MTP: the layer-80 NextN head"). Full-checkpoint parity and
hardware benchmarks remain release gates for both paths.

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

For the M1 correctness baseline, each layer must complete this lifetime:
fill scratch under a fresh epoch, build and execute the routed MLP, materialize
the complete layer output, then wait for the last scratch-reading Metal command
before the next `pread`. A command-buffer/shared-event fence is the production
mechanism; a full MLX/GPU synchronization is the conservative first proof. This
hard-serializes layers, so it is a low-memory baseline rather than the final
throughput path.

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
python3 scripts/simulate_expert_cache.py routes.jsonl \
  --model hy3-q4 \
  --persistent-slots-per-layer 64 \
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
practice. The layer-80 head is therefore packaged separately from the
official BF16 checkpoint and loaded as resident weights when MTP is
requested; see "MTP: the layer-80 NextN head" below.

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

## MTP: the layer-80 NextN head

Status: implemented behind an opt-in flag. AR stays the default; an
unflagged run is byte-identical to the AR-only branch. The real-checkpoint
parity run and the AR-vs-MTP benchmark are still release gates.

### Packaging

The pinned Q4 conversion omits layer 80, so the head is packaged from the
official `tencent/Hy3` BF16 checkpoint (revision `716aa724...`) into three
sibling artifacts:

1. `scripts/extract_mtp_layer80.py` copies all 593 `model.layers.80.*`
   tensors bit-exactly into `layer80-bf16.safetensors`.
2. `scripts/quantize_mtp_layer80.py` emits `layer80-q4.safetensors`: the
   576 routed expert projections in the pinned affine-Q4/gs64 expert
   segment format (U32 packed + BF16 scales/biases, 10,616,832 bytes per
   expert).
3. `scripts/quantize_mtp_layer80_residents.py` emits
   `layer80-residents-q.safetensors`: attention q/k/v/o and the shared
   expert in affine Q4/gs64, the router gate in affine Q8/gs64, norms and
   `eh_proj` in BF16, `expert_bias` in F32 — the same resident conventions
   the pinned artifact uses for trunk layers 1-79.

Loading is fail-closed: revision mismatches, missing/extra tensors, shape
disagreements between experts, and wrong leaf dtypes abort MTP enablement.

### Resident experts, not a second slot bank

Earlier planning assumed the head would need its own streamed expert bank.
It does not get one. Layer 80's 192 experts total about 1.94 GiB in Q4 and
every draft routes through the head, so streaming them would put SSD reads
directly on the speculative path it is supposed to shorten. At MTP-enable
time the experts load as ordinary resident weights — one stacked quantized
`SwitchGLU`, the same shape resident MLX MoE layers use — while trunk
layers 1-79 keep the existing slot bank unchanged.

The head is standard NextN: the shifted token's embedding and the trunk's
hidden state pass through `enorm`/`hnorm`, are concatenated and projected
by `eh_proj` [4096, 8192] into one transformer block (attention plus the
head's own 192-expert top-8 MoE), then the head's checkpoint
`final_layernorm` feeds the shared trunk `lm_head`. Speculation reuses the
existing exact rejection-sampling draft/verify loop (`generate_mtp1`,
depth 1): acceptance rate changes speed, never tokens.

Verify batches reach the streamed trunk as short multi-token forwards. The
wave scheduler already groups tokens by expert; with MTP enabled the
runtime classifies forwards up to primary+draft width as decode routing so
verify traffic keeps training the persistent decode hot set instead of
seeding prefill-transient slots. With MTP off the classification is exactly
the historical single-token rule.

Budget note: the head adds one attention layer of KV (about 1.25% per
token over the 80 trunk layers) and ~2.1 GiB of resident weights; both must
come out of the same memory plan when sizing persistent slots for MTP runs.

### Enabling it

Python: `mtplx.runtime.load(root, mtp=True, mtp_artifacts=<dir>,
expert_streaming_config=..., expert_manifest=...)` where `<dir>` holds
`layer80-residents-q.safetensors` and `layer80-q4.safetensors`
(`~/.cache/huggingface/hy3-mtp-layer80` on the reference machine).

Benchmark harness (release gate; run only on an otherwise idle machine —
never beside the production qwen server):

```bash
python3 scripts/benchmark_streamed_generation.py \
  <model_root> <manifest.json> \
  --model-key hy3-q4 \
  --memory-limit 112GiB \
  --max-live-kv-tokens 8192 \
  --enable-mtp \
  --mtp-artifacts ~/.cache/huggingface/hy3-mtp-layer80 \
  --output-json outputs/hy3-mtp-run.json
```

Run the same command without `--enable-mtp`/`--mtp-artifacts` for the AR
baseline; `generation_stats` in the MTP payload carries drafted, accepted,
and rejected counts. MTP is promoted only if it beats AR while passing
exactness and repetition gates under the same cache budget.

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

## Concurrent streams (Stage 5 batching)

`mtplx/streamed_batch.py` decodes several AR streams together: each decode
step evaluates every live stream as one `[B, 1, H]` forward, so a sparse
layer routes the union of the experts selected across streams and a record
selected by several streams in the same step is planned, read, and hashed
once. Streams keep structurally separate per-sequence KV caches (their own
offsets, no left-padding); a joining stream prefills at a decode step
boundary through the single-stream prefill path, which keeps its misses in
transient slots and cannot evict a decode-hot persistent expert. Each stream
reserves its full `prompt + max_tokens` KV budget through `admit_kv_tokens`
before any forward runs and releases it the moment it finishes.

Batch size is part of the run configuration label. With one live stream the
runner is byte-identical to `generate_ar`; with two or more, the batched
kernels see different shapes, so `B > 1` outputs legitimately differ from
`B = 1` runs of the same prompt and results are only comparable at equal
batch sizes. The benchmark harness exposes this lane as
`scripts/benchmark_streamed_generation.py --concurrency N`, reporting both
aggregate and per-stream tok/s.

## Current levers (2026-07)

The flags below drive the shipping Q2 results. They are the levers the
Q4-era sections above do not mention. All are on
`scripts/benchmark_q2_mtp_depth_matrix.py`.

| Flag | Choices / default | What it does |
| --- | --- | --- |
| `--island-layer-count N` | int, default `None` | Pins the N worst-streaming layers fully resident as dense "islands" using the model's measured pin order. `79` means zero streamed layers (full residency). Independent of `--expert-cache-limit`. |
| `--proj-quant {q8,q4}` | default `None` | Quantizes the trunk `*_proj` Linears at load (group-64 affine): attention `q/k/v/o_proj` and `gate/up/down_proj` in `mlp` and `shared_mlp`. Routers, embeddings, the LM head, and norms keep loaded precision. No artifact rebuild. `--resident-quant` is a deprecated alias. |
| `--kv-quant {q8,q4}` | default `None` | Quantizes the trunk KV cache. **Requires `--verify-strategy batched`.** |
| `--hy3-router-kernel` | 6 choices, default `mpp-r1-fused-r2` | Router arithmetic mode. See the measured A/B below. |
| `--q2-expert-kernel` | `stock`/`nax`/`fused`/`fused-nax`, default `stock` | Q2 expert execution arm. |
| `--slot-layout` | `direct-slots`/`component-banks`/`metal-mmap` | Expert slot backing. Note the default differs by script: `component-banks` here, `direct-slots` in `benchmark_streamed_generation.py`, `probe_mtp_draft_rank.py`, and `compare_streamed_quality.py`. |

**Router precision differs by checkpoint — this trips people up.** On the
shipping Q2 line (`--model hy3-q2` → spec `hy3-expert-q2`,
`~/.cache/huggingface/hy3-expert-only-mlx-q2`) the router gates are **BF16**,
all 79 of them, and `--proj-quant` does not touch them: `proj_quant_covers`
matches only `.self_attn.` and `*_proj` suffixes, and `router.gate` has
neither. So the championship configuration runs **bf16 routers**.

Q8 routers belong to the *other* artifact — `hy3-q4-mlx-mtp` and the pinned
`pipenetwork/Hy3-4bit` snapshot (spec `hy3-q4`, whose `router_storage` is
`"affine-q8 with fp32 correction bias"`), which is the older streamed regime.
The design note earlier in this document recommended A/B-ing BF16 gates
because all 79 are only ~124 MiB in BF16; the Q2 conversion took that
recommendation, which is why the two artifacts differ.

Routing is a discrete, precision-sensitive boundary; router math and selection
run in FP32 in both cases.

### Router kernel A/B (2026-07-18, 3 reps, arms interleaved per rep)

Measured on the config below; the router kernel was the only variable.

| Kernel | mean tok/s |
| --- | --- |
| `mpp-fp32-splitk-r1-fused-r2` | **40.59** |
| `mpp-r1-fused-r2` (current default) | 40.01 |
| `mpp-row-owned-fused` | 38.27 |
| `steel-r1-fused-r2` | 38.25 |
| `mpp-r1-last-arrival-fused-r2` | 37.90 |
| `stock` | 37.77 |

`mpp-fp32-splitk-r1-fused-r2` applies split-K to the **MTP** routers only
(`splitk_m1 = selector == "mpp-fp32-splitk-r1-fused-r2" and is_mtp_router`,
`configure_hy3_router_kernels` in `mtplx/models/hy3_mlx.py`), which is why it
separates from plain `mpp-r1-fused-r2`.

**The default is still `mpp-r1-fused-r2`** — the fastest kernel is opt-in.

### Reference configuration

```bash
python3 scripts/benchmark_q2_mtp_depth_matrix.py \
  --model hy3-q2 --contexts 2048 --output-tokens 1024 --hy3-depths 2 \
  --hy3-q2-model-root    ~/.cache/huggingface/hy3-expert-only-mlx-q2 \
  --hy3-q2-manifest      ~/.cache/huggingface/hy3-expert-only-mlx-q2/expert-manifest.json \
  --hy3-q2-mtp-artifacts ~/.cache/huggingface/hy3-mtp-layer80 \
  --memory-limit 108GiB --runtime-reserve 7GiB \
  --expert-cache-limit 2GiB --max-live-kv-tokens 3072 \
  --cache-policy frequency --cache-scope layer \
  --slot-layout component-banks --transient-slots 8 \
  --q2-expert-kernel stock \
  --hy3-router-kernel mpp-fp32-splitk-r1-fused-r2 \
  --read-chunk 8MiB --f-nocache \
  --verify-strategy capture_commit --compiled-verify-mode off \
  --draft-core device-k --no-trace-routes \
  --island-layer-count 79 \
  --proj-quant q4 \
  --expert-integrity headers-only \
  --split-route-release deferred
```

Environment: `MTPLX_SUSTAINED_PREFILL=1`, `MTPLX_DEFERRED_PIN_RELEASE=1`,
`MTPLX_HY3_SUBMIT_CADENCE=8`, with `MTPLX_MEMORY_LIMIT_BYTES` unset.

At `--island-layer-count 79` nothing streams, so `--expert-cache-limit` is
vestigial in this configuration.

### Running a guarded window

Never run two benchmarks at once — two concurrent ~95 GB Hy3 runs
kernel-panicked this box on 2026-07-17. Use the shared harness rather than
hand-rolling the guards:

```python
from mtplx.benchmarks.window import (
    exclusive_mlx_lock, gpu_run_lock_clear, guarded_qwen,
)
from mtplx.qwen_guard import qwen_stopped_for_mlx

assert gpu_run_lock_clear(), "another benchmark holds the GPU"
with exclusive_mlx_lock(timeout_seconds=28800.0):
    with guarded_qwen(lambda: qwen_stopped_for_mlx(plist=plist)):
        ...  # arms interleaved within each rep; burn a warm-up cell first
```

Aborts cluster at window start, so discard a warm-up cell before measuring.

### Presets — don't memorize the flags

The reference configuration above is a named preset. These are equivalent:

```bash
# the 25-flag command
python3 scripts/benchmark_q2_mtp_depth_matrix.py --model hy3-q2 --contexts 2048 ... 

# the same thing
python3 scripts/benchmark_q2_mtp_depth_matrix.py --preset championship
```

```bash
--list-presets              # what's defined, one line each
--show-preset championship  # the fully expanded flags + env, without running
--preset championship       # run it
--preset championship --proj-quant q8   # run it with one flag substituted
```

Presets are applied as argparse **defaults**, so anything you type explicitly
beats the preset — there is no precedence surprise. Order is:

    runner defaults  <  preset (through its `extends` chain)  <  typed flags

Shipped presets live in `benchmarks/presets.toml` (in-repo, so a change to an
operating point is a reviewable diff). Machine-local bundles that shouldn't be
committed go in `~/.mtplx/benchmark-presets.toml`, which layers on top.

| Preset | What it is |
| --- | --- |
| `hy3-q2` | Base: Q2 streamed experts + BF16 layer-80 MTP head |
| `full-residency` | All 79 layers resident, nothing streams |
| `championship` | The 40.59 tok/s router A/B winner |
| `shipped-default` | Same envelope, shipped router kernel (the 40.01 baseline) |
| `streamed` | The ~6 tok/s SSD-streamed regime |
| `q8-projections` | Championship with `--proj-quant q8` |
| `k3` | Championship at depth 3 (a measured net loss; kept for reproduction) |
| `kv-q8` | Championship with a q8 KV cache |
| `telemetry` / `roofline` | Diagnostic variants |

A preset naming a flag the runner doesn't have is a **hard error**, not a
silent no-op — a preset that quietly drops `--proj-quant` would be a silently
different experiment. Values are checked against each flag's `choices` too.
