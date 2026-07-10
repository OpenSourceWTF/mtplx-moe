# SSD-streamed MoE plan: Hy3 Q4 and GLM-5.2 Q4

Status: implementation complete on the experimental branch; tiny end-to-end
Hy3/GLM fixtures, failure injection, and full pinned artifact metadata audits
pass. Full 166/418 GB checkpoint parity and performance measurements remain a
release gate, so this document makes no full-model speed claim.

Target branch: `codex/moe-ssd-hy3-glm52`

## Decision

Yes: expert selection can be separated from expert execution at each MoE
layer. The router is small enough to remain resident. It runs first, returns
the selected expert ids and weights, and the runtime resolves those ids against
a bounded in-memory expert bank. Missing records are read from SSD into either
a persistent hot slot or a global transient slot before the routed MLP runs.

The separation is layer-local, not whole-model lookahead. Router input at layer
`n` depends on the hidden state produced by layer `n - 1`, so the runtime cannot
select every layer's expert path at the start of a token. The legal ordering is:

```text
layer n hidden state
    -> resident attention (and resident shared expert where applicable)
    -> resident MoE router
    -> selected (layer, expert) ids and weights
    -> slot lookup
         hit  -> existing persistent slot
         miss -> checked SSD read into persistent or transient slot
    -> quantized expert gate/up/down
    -> weighted combine + residual
    -> layer n + 1 hidden state
```

Once the router has produced ids, the implementation may overlap independent
record reads, run already-resident experts, and compute a shared expert. It
must not advance past the layer's combine until every selected contribution is
complete.

This is an MLX/Metal runtime change. It does not require modifying Apple's
Metal driver or macOS virtual-memory implementation.

## Model scope

The first implementation targets autoregressive, single-token decode for two
affine Q4/group-64 checkpoints:

| Property | Hy3 Q4 | GLM-5.2 Q4 |
| --- | ---: | ---: |
| Target transformer layers | 80 | 78 |
| Sparse MoE layers | 79 (layers 1-79) | 75 (layers 3-77) |
| Routed experts per sparse layer | 192 | 256 |
| Experts selected per token/layer | 8 | 8 |
| One routed expert record | 10.125 MiB | 20.25 MiB |
| Routed expert corpus | about 149.984 GiB | 379.6875 GiB |
| Fully cold expert reads/token | about 6.249 GiB | 11.8652 GiB |
| Non-routed resident checkpoint data | about 4.612 GiB | 9.9042 GiB |
| Resident router data | about 63 MiB | about 225 MiB |
| Q4 community artifact includes MTP | No (layer 80 omitted) | No (layer 78 omitted) |

The resident footprint includes embeddings, norms, attention, routers, shared
experts, dense MLPs, and the output head. KV state, MLX allocator headroom,
command buffers, and application memory are separate reservations.

These are pinned-artifact planning figures. The exporter must derive byte
offsets, lengths, shapes, dtypes, and hashes from each exact safetensors
revision; the native reader must trust the validated manifest rather than the
table above.

## Resident selector boundary

The runtime keeps all routing dependencies resident:

- The current hidden state and attention path.
- Every sparse-layer MoE router, including selection correction bias.
- Dense and shared experts.
- GLM-5.2's DSA attention indexer and IndexShare state.
- KV state for the configured context budget.

GLM-5.2 contains two selectors with different jobs. The DSA indexer selects
historical tokens for attention; the MoE router selects feed-forward experts.
Both remain resident, and the expert cache keys only on MoE selections. The
IndexShare schedule has 21 full indexer layers (`0`, `1`, `2`, then `6` through
`74` in steps of four); the other 57 layers reuse the preceding full layer's
indices.

Router behavior is a correctness boundary, not an approximate cache hint. For
GLM-5.2, the adapter must reproduce the official FP32 routing sequence: FP32
router projection, sigmoid scores, FP32 correction bias used only for
selection, gather the corresponding unbiased sigmoid scores, normalize the
selected top-8 weights, and apply the routing scale. A different top-8 set
executes different parameters and invalidates every downstream comparison.

## Memory threshold contract

The user supplies a total memory ceiling and may optionally supply a tighter
expert-cache ceiling. The planner subtracts fixed and context-dependent costs
before assigning persistent expert slots:

```text
fixed = resident_model
      + kv_bytes_per_token * context_tokens
      + global_transient_slots * expert_record_bytes
      + explicit_io_staging
      + explicit_execution_workspace
      + runtime_and_os_reserve

available_for_persistent = max(0, total_memory_limit - fixed)
persistent_budget = min(available_for_persistent, optional_expert_cache_limit)
slots_per_layer = floor(
    persistent_budget / (sparse_layer_count * expert_record_bytes)
)
```

`slots_per_layer` is capped at the model's experts per layer. The initial
planner uses an equal per-layer allocation because every token visits every
sparse layer and a slot can only hold an expert from its own layer. Telemetry
may later justify an explicitly bounded non-uniform allocation.

`context_tokens` is the maximum aggregate live KV-token count admitted by the
runtime, not merely one request's advertised window. Zero is an explicit
load-only plan. Known staging and workspace allocations are charged separately;
the reserve covers remaining MLX, Metal, MTPLX, and OS headroom.

The limit is strict at the slot-bank boundary:

- Persistent slot buffers are allocated once and never grow after startup.
- A global transient bank is reused across sequential layers. Top-8
  single-token decode needs eight records globally, not eight per layer.
- A miss cannot allocate an untracked temporary expert array.
- If fixed reservations do not fit, startup fails with a memory-plan report.
- Multi-token unions larger than the transient bank run in bounded waves.
- Cache admission may replace a persistent slot, but cannot increase capacity.

The process can still be terminated by macOS if the user reserves too little
headroom for unrelated applications or opaque framework allocations. Tests
therefore track both the configured slot-bank bytes and observed process/unified
memory high-water marks.

This planner is fixed-buffer accounting, not an independent macOS memory
controller. Runtime integration must reconcile it with MTPLX's existing
`MTPLX_MEMORY_LIMIT_BYTES`/`mx.set_memory_limit` path, reject conflicting caps,
set the MLX limit before allocation, reject KV growth beyond the plan, and
compare `mx.get_active_memory()` peak telemetry with the accounted budget.

## Cache policy

Each persistent cache key is:

```text
(model revision, quantization revision, layer, expert)
```

The proposed bank has two tiers:

1. **Persistent hot slots.** Decode admission uses a frequency-and-recency
   estimate so repeated experts displace colder residents.
2. **Global transient scratch.** Cold, one-off, or prefill records execute
   without polluting the decode hot set.

Prefill and decode are accounted separately. A prompt can touch a broad expert
set immediately before decode, so blindly admitting every prefill miss would
destroy the useful decode working set. `persistent_slots=0` remains a supported
baseline: transient reads plus the macOS page cache may beat a second explicit
cache on some machines.

Every slot carries a generation/epoch. The native implementation must prove
that the final Metal command reading generation `g` has completed before an SSD
read overwrites that slot with generation `g + 1`. The first correctness build
may fully synchronize at each streamed layer; later builds can replace that
barrier with command-buffer or shared-event fences.

## Artifact layout

Phase 1 should first prove selective reads from original safetensors offsets.
An optional expert-major sidecar is accepted only if it materially improves
read amplification or alignment. Each record contains one expert's Q4
gate/up/down weights plus scales and affine biases.

The manifest must pin:

- Source repository and immutable revision.
- Every source shard name, size, and digest.
- Model, quantization mode, group size, tensor dtype, and tensor shape.
- `(layer, expert)` to file/offset/length mappings.
- Record alignment, component offsets, and record digest.
- Resident tensor allowlist and proof that routed payloads were not loaded.

Malformed bounds, unexpected shapes, short reads, checksum failures, and
revision mismatches fail closed before a buffer is exposed to a kernel.

## Staged implementation plan

### Stage 0 — model-independent policy and planning

- Encode pinned Hy3 Q4 and GLM-5.2 Q4 layout descriptors.
- Calculate persistent capacity from the user's memory and context limits.
- Keep cache simulation free of MLX imports for fast deterministic tests.
- Report planned resident, KV, transient, persistent, reserve, and unallocated
  bytes before model load.

Gate: sizing formulas reproduce the artifact inventories and never exceed the
configured limit.

### Stage 1 — manifests and resident-only loading

- Build a safetensors inventory tool with immutable revision and digest checks.
- Export a validated offset manifest and optional aligned sidecar.
- Add an MTPLX load path that instantiates resident tensors without generic
  `mlx_lm.load` materializing the entire checkpoint.
- Fail early when a config advertises an MTP layer absent from the artifact;
  never fall back to scanning or loading every shard.

Gate: process high-water memory stays near resident plan, and arbitrary sampled
records read through the manifest match the original tensor bytes.

### Stage 2 — exact resident routers and model adapters

- Preserve Hy3's artifact-specific router quantization and FP32 selection.
- Port GLM-5.2 IndexShare before attempting streamed execution.
- Match the official GLM-5.2 FP32 top-8 ids, weights, and correction-bias
  semantics at every sparse layer.
- Keep attention-token selection and MoE-expert selection as separate APIs.

Gate: router ids are exact and weights match tolerance on fixed vectors,
prefill, decode, and long-context cases beyond 2K/4K tokens.

### Stage 3 — native checked record fill

- Allocate fixed shared Metal slot buffers from the approved plan.
- Add positional, aligned `pread` into slot component ranges.
- Associate each read, cache entry, and dispatch with a slot generation.
- Add completion fences before slot reuse and failure injection for short or
  corrupt reads.
- Export per-layer hit/miss, bytes, latency, occupancy, eviction, and stall
  counters.

Gate: stress tests cannot execute a stale generation, leak slot capacity, or
produce a different result for hit versus miss.

### Stage 4 — routed Q4 execution

- Bind slot-backed gate/up/down data to MLX/Metal quantized matmul kernels.
- Group batch rows by selected expert and accumulate original router weights.
- Execute expert unions larger than scratch capacity in bounded waves.
- Materialize the layer result before reusing global scratch.

Gate: expert projections, layer outputs, logits, and greedy tokens match a
fully resident Q4 reference within declared kernel tolerances.

### Stage 5 — scheduling and cache-policy tuning

- Overlap independent reads, resident hits, and shared-expert compute after
  routing.
- Compare OS-page-cache-only, explicit persistent cache, and explicit cache
  with cold-read cache bypass where supported.
- Tune admission independently for prompt, single-stream decode, and
  continuous batching.
- Consider non-uniform layer capacity only from trace-backed evidence.

Gate: a selected mode improves end-to-end tokens/s without exceeding memory,
changing output, or increasing SSD writes beyond packaging activity.

### Stage 6 — optional MTP

Prove autoregressive inference first. The inspected Hy3 and GLM-5.2 community
Q4 artifacts omit their declared MTP layers (80 and 78 respectively). A later
MTP release needs a separately generated, validated Q4 artifact, correct
embedding/output-head sharing, and an independent expert bank. Verification
batches also need bounded-wave execution for the union of proposed positions.

Gate: MTP passes AR-equivalent correctness and repetition tests and improves
end-to-end throughput under the same total memory ceiling.

## Acceptance gates

The feature is not considered implemented until all of these hold:

1. **Artifact:** every offset is bounds-checked and source/record hashes pass.
2. **Routing:** selected ids are exact; selected weights match the resident
   oracle within the declared precision tolerance.
3. **Kernel:** streamed and resident expert projections agree on deterministic
   vectors for every component shape.
4. **Model:** layer outputs, logits, and greedy tokens agree on fixed prompts.
5. **Cache invariance:** hit, persistent miss, and transient miss produce the
   same numerical result.
6. **Lifetime:** slot-generation stress and forced I/O delay never expose stale
   or partially filled data.
7. **Memory:** configured slot bytes never grow; observed high-water memory
   stays within the total ceiling plus a documented measurement tolerance.
8. **Failure:** short reads, corruption, missing MTP tensors, and incompatible
   manifests fail closed with actionable errors.
9. **Quality:** long decode does not introduce repetition, gibberish, or
   router drift versus the resident reference.
10. **Performance:** cold, warm, and steady-state results report I/O bytes,
    router time, read time, kernel time, hit rate, and tokens/s separately.

## Risk and benchmark matrix

| Risk or question | Required comparison | Pass condition |
| --- | --- | --- |
| Router precision changes expert ids | Streamed adapter vs official/resident router on fixed hidden states | Exact top-8 ids; weights within tolerance |
| GLM IndexShare is implemented incorrectly | Contexts below 2K and beyond 2K/4K; inspect all 78 layers | Correct 21 full-layer schedule and matching logits/tokens |
| Generic loading materializes routed tensors | Startup/high-water trace with resident-only loader vs ordinary load | No routed corpus allocation; high-water near declared plan |
| SSD reads cannot feed decode fast enough | Cold/warm random aligned reads at 1/8/batched records | Report effective GiB/s and I/O-only tokens/s ceiling |
| Explicit cache duplicates macOS page cache | `persistent_slots=0` vs 8/16/32/64/96+ slots per layer | Choose by end-to-end tokens/s and memory pressure |
| Prefill destroys decode locality | Unified admission vs prefill-bypass/transient-only admission | Decode hit rate and tokens/s do not regress after prompt |
| Slot overwritten while GPU still reads it | Delayed reads/dispatches with rapid eviction and generation checks | Zero stale-generation executions across stress run |
| Batch expert union exceeds scratch | Batch sizes and prompt chunks with unions above eight | Bounded memory and resident-equivalent output |
| Memory threshold is not truly bounded | Sweep context and cache ceilings under pressure | Planned buffers fixed; high-water stays within tolerance |
| Streaming harms quality | Fixed prompt suite, long generation, AR reference diff | Matching greedy tokens or documented sampling parity |
| MTP increases I/O more than speed | AR vs MTP under identical memory/context/quality settings | MTP promoted only with net tokens/s gain |
| Thermal or SSD behavior hides regressions | Cold/warm/steady runs with thermals, pressure, bytes read/written | Stable sustained result; streaming path remains read-only |

Minimum benchmark reports must identify model and artifact revisions, hardware,
macOS version, memory limit, runtime reserve, context length, batch/prefill
shape, cache policy, slot count, record alignment, SSD, and whether filesystem
cache was warm.

## What this branch proves today

The branch now implements the complete AR data path:

- Strict revision-pinned manifests and optional resumable 16 KiB-aligned
  expert-major sidecars, with shape/dtype/range/hash validation.
- Resident-only startup loading that never constructs routed parameter trees.
- Hy3 and GLM-5.2 MLX overlays, including GLM FP32 MoE routing and the exact
  21-full-layer IndexShare schedule.
- Decode-aware hot admission, transient-only prefill misses, fixed slot
  generations/pins, bounded descriptors and in-flight I/O, cancellation, and
  corruption/short-read failure handling.
- Optional native GIL-free `pread`, directly into stable MLX/Metal slot-bank
  buffers. Q4 component arrays are shared-buffer views; evaluated expert work
  is synchronized before a slot generation can be replaced.
- Runtime/CLI/server integration, aggregate live-KV admission, MLX allocator
  cap reconciliation, and `/health` cache/I/O/memory telemetry.

The full pinned checkpoint indexes were audited without downloading payloads:
Hy3 has 2,323 resident keys and 711 routed leaves; GLM-5.2 has 2,806 resident
keys and 675 routed leaves. Tiny artifacts execute complete model forwards
through the streamed path. What is not yet proven is full-checkpoint numerical
parity, high-water memory, thermals, and sustained SSD throughput on target
hardware. Use these gates before publishing a result:

```bash
python scripts/audit_streamed_model_layout.py --model hy3-q4
python scripts/benchmark_expert_io.py MODEL MANIFEST \
  --operations 256 --queue-depth 8 --cache-state steady --ssd-label "internal NVMe"
python scripts/verify_streamed_parity.py MODEL MANIFEST probes.jsonl \
  --model-key hy3-q4 --memory-limit 96GiB --max-live-kv-tokens 32768
python scripts/benchmark_streamed_generation.py MODEL MANIFEST \
  --model-key hy3-q4 --memory-limit 96GiB --max-live-kv-tokens 32768
```

`--cache-state` is provenance, not cache manipulation: the I/O benchmark never
claims to purge macOS's filesystem cache. Every output includes immutable model
and manifest identity plus raw cache/I/O/MLX measurements.
