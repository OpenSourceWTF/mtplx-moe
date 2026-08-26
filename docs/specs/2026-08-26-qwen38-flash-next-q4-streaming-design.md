# Qwen3.8-Flash-Next Q4 Streaming MTPLX Design

Status: approved for implementation
Date: 2026-08-26
Code target: `OpenSourceWTF/mtplx-moe`
Branch: `port/qwen38-flash-next-q4`
Source artifact: `Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c`
Planned derivative: `OpensourceWTF/Qwen3.8-Flash-Next-MTPLX-Q4`

## Objective

Run the official Qwen3.8-Flash-Next text model on the 128 GiB Apple Silicon
machine with:

- every routed MoE expert in MLX affine 4-bit, group size 64;
- fixed-budget SSD streaming for routed experts and n-gram embeddings;
- native autoregressive generation;
- the checkpoint's one-layer MTP module driving MTPLX speculative decoding;
- construction-time artifact and route validation, with no enabled-path silent
  fallback;
- a new `OpenSourceWTF/mtplx-moe` pull request; and
- a verified derivative upload to the authenticated `OpensourceWTF` Hugging
  Face account.

## Evidence and geometry

The source checkpoint contains approximately 180B parameters in 131
safetensors shards (179,999,981,459 weight elements plus 35 integer metadata
elements in the pinned index). The text configuration has 48 target layers,
hidden size 2560, 512 routed experts per layer, top-10 routing, one shared
expert, expert intermediate size 640, and one MTP layer.

The routed tensors comprise 123,312,537,600 parameters across the 48 target
layers plus the MTP layer:

- `gate_up_proj`: `[512, 1280, 2560]` per layer;
- `down_proj`: `[512, 2560, 640]` per layer.

MLX affine Q4/group-64 stores packed weights plus BF16 scales and biases. One
expert record is exactly 2,764,800 bytes, and all 49 x 512 records occupy
69,363,302,400 bytes, or 64.60 GiB. This is the fixed expert-Q4 artifact
contract.

The n-gram memory consists of 128 tensors of shape `[2,500,012, 160]`, totaling
51,200,245,760 BF16 parameters (95.37 GiB). With `ngram_size=3` and
`heads_per_ngram=8`, each token deterministically selects eight bigram and
eight trigram rows. The useful BF16 payload is 16 x 160 x 2 = 5,120 bytes per
token.

The 64.60 GiB figure therefore describes the affine-Q4 routed-expert bank, not
the complete derivative. Keeping the n-gram table in BF16 produces an
approximately 170.2 GiB on-disk derivative: 64.60 GiB of Q4 routed experts,
95.37 GiB of immutable BF16 n-gram backing data, and approximately 10.2 GiB of
the remaining model tensors. Exact file lengths and header overhead are
recorded by the converter manifest rather than inferred from these rounded
totals. The n-gram table is never quantized or approximated.

## Scope

### Included

1. A native MLX text implementation of `qwen4_exp_text`:
   - Gated DeltaNet for three of every four layers;
   - Qwen Sparse Attention for every fourth layer;
   - four-branch Gated Residual reads and writes;
   - the layer-2 PLE/n-gram injection path;
   - top-10-of-512 routed MoE plus the shared expert;
   - the model's RMSNorm, rotary embedding, and output-head semantics.
2. Q4/group-64 routed expert conversion and SSD streaming for all target and
   MTP experts.
3. Exact BF16 n-gram streaming through an independently sized fixed cache.
4. Native AR and MTP speculative generation, including proposal, target
   verification, rejection repair, cache rollback, and commit.
6. Real-model correctness, quality, performance, memory, and artifact receipts.
7. Pull-request delivery and a verified Hugging Face derivative upload.

### Not included

- Image/video inference in the first PR. The official tokenizer and vision
  files are retained in the derivative, but the PR makes no multimodal support
  claim.
- Custom performance kernels before a correct stock-MLX baseline identifies a
  measured bottleneck.
- Quantizing routers, shared experts, GDN/QSA projections, residual mixers,
  token embeddings, or the output head.
- Quantizing or approximating the n-gram table.
- Supporting arbitrary Qwen4-family geometries in this first port.

## Architecture

### Text data flow

```text
tokens + token history
        |
        +--> deterministic 16-row n-gram address plan
        |          |
        |          +--> async fixed-cache acquisition
        |
token embedding -> layer 0 -> wait/materialize n-gram rows -> layer 1 PLE
        -> layers 2..47 -> final Gated Residual combine -> norm -> LM head

Every decoder layer:
  Gated Residual read -> GDN or QSA -> Gated Residual write
  Gated Residual read -> top-10 routed MoE + shared expert -> write
```

The checkpoint stores PLE weights under zero-indexed layer 1, corresponding to
human-numbered layer 2. Address planning begins before layer 0, so the official
host-prefetch overlap covers the first layer's compute.

### Qwen4 model implementation

The Hugging Face Transformers `qwen4_exp` implementation is the arithmetic
reference. The MTPLX implementation must preserve:

- the exact GDN recurrence orientation, convolution state, FP32 recurrence
  state where specified, sigmoid output gate, and zero-centered RMSNorm;
- QSA micro-block compression, index scores, block budget, tail inclusion,
  RoPE layout, and selected-token ordering;
- four-branch residual ownership and elementwise read gating;
- n-gram hashing, EOS-aware shifts, head-specific prime moduli and offsets,
  PLE convolution, projections, and injection point;
- router softmax/top-k order, top-10 weights, routed SwiGLU arithmetic, shared
  expert gate, and accumulation dtype;
- target-cache and MTP-cache ownership.

No existing Qwen3-Next kernel or module is accepted by name or topology. Each
candidate reuse requires shape, arithmetic, ownership, layout, and numerical
parity evidence for this checkpoint.

### Routed-expert storage

The converter reads one pinned source shard at a time. Qwen stores gate and up
rows fused as `[512, 1280, 2560]`; the converter splits the first and second
640-row halves before quantization. Affine groups run along the unchanged
2560-element input dimension, so splitting does not alter any group's values
or quantization geometry. For each expert it then applies MLX affine
Q4/group-64 independently and writes a record-major `experts.bin` bank:

```text
record(layer, expert):
  gate.weight     | gate.scales    | gate.biases    |
  up.weight       | up.scales      | up.biases      |
  down.weight    | down.scales    | down.biases
```

Streamed layer IDs 0..47 identify target decoder layers and synthetic streamed
layer ID 48 identifies `mtp.layers.0`. Model construction exposes exactly
these 49 sparse blocks to the binder without adding the MTP block to the
target model's 48-layer forward loop.

`expert-manifest.json` records exact shapes, offsets, lengths, source tensor
names, per-record hashes, source revision, and whole-bank hash. Resident BF16
tensors are re-sharded without numerical conversion. Existing expert-streaming
memory planning, slot generations, I/O admission, pinning, deferred release,
and component execution are reused only after a Qwen4-specific construction
self-check proves the Q4 record layout and top-10 execution path.

The experimental lane is selected at construction. An invalid bank, missing
record, wrong quantization descriptor, incompatible MLX version, or failed
self-check aborts model construction. There is no custom-then-stock branch in
generation.

### Fixed-budget n-gram cache

N-gram storage and cache ownership are separate from routed-expert storage.
The complete 95.37 GiB BF16 table remains immutable on SSD and is never mapped
or copied into unified memory as a whole. Only exact rows required by active
tokens occupy the fixed cache. Its ownership resembles a second KV cache, but
unlike KV state its values are static, addressable from token history, and
safely reloadable after eviction; it does not grow monotonically with context.
The cache configuration is fixed at construction:

```text
NGramCacheConfig(
    storage = bf16,
    calculated_cache_payload_bytes,
    transient_limit_bytes,
    max_inflight_io_bytes,
    max_open_files,
    bypass_page_cache,
)
```

The implementation uses:

- one contiguous preallocated byte arena;
- row-sized slots: 320 bytes for BF16, 100 bytes for affine Q4/group-32;
- a route map from the complete source-table identity and row index to
  `(slot, generation)`;
- fixed transient reservations sized for the configured prefill chunk and
  maximum physical MTP verification width;
- asynchronous positional reads, with adjacent source rows coalesced into one
  read and scattered into slots;
- pin claims that survive until the layer-2 PLE MLX evaluation has consumed
  the compact gathered rows;
- generation tickets so a stale completion cannot publish into a reassigned
  slot;
- frequency or LRU eviction selected at construction;
- `F_NOCACHE`/equivalent bypass after explicit reads so the operating-system
  page cache cannot silently defeat the byte limit;
- reset and close barriers that drain reads and releases deterministically.

Construction opens the artifact root by walking every path component with
no-follow directory descriptors, then opens all n-gram shards relative to that
root descriptor. Full payload verification binds the cache to retained shard
descriptors and recorded device/inode/size identities; generation never
re-traverses verified pathnames. Symlinks and multi-link shard files are
rejected, and every failure path closes all descriptors already acquired.

The pinned production configuration is calculated once at model/request
construction. Its row payload is at most 20 GiB and the complete cache
reservation includes every arena, slot-metadata array, open-addressed route
table, per-allocation alignment pad, and transient buffer. Given measured base
residency and explicit KV/MTP, Metal-working, and safety reserves, construction
first computes the 95 GiB formula ceiling and then reduces to the largest whole
row count whose complete reservation fits. It fails before generation if the
configured minimum viable row cache and runtime do not fit. Small explicit
payload overrides remain test-only.

The hot path receives a prebound `acquire_rows` callable for exact BF16. It
does not read environment variables, revalidate the manifest, choose a storage
representation, instrument each lookup, or fall back to uncached reads.

For a prefill chunk, all 16 x T addresses are planned together and duplicate
rows are removed before I/O. For an MTP target verification window, all
physical candidate rows are planned together. The MTP draft module itself has
no PLE layer; the authoritative target verification owns the n-gram lookups
for proposed positions.

### MTP and speculative decoding

The target model is always authoritative. The checkpoint's one MTP layer is
reused iteratively to propose future tokens. The integration must preserve:

- the primary hidden-state alignment expected by `mtp.fc_hidden`;
- the token embedding path expected by `mtp.fc_embedding`;
- physical verification width `1 + draft_depth`;
- QSA-index reuse only where the checkpoint and official report permit it;
- target probability evaluation, rejection sampling/residual repair, and exact
  rollback/commit of GDN, QSA KV, convolution, token-history, and n-gram state;
- independent target and draft caches;
- standard AR as an explicit separately constructed route.

Correctness starts at fixed greedy depth 1, then depth 2, before any adaptive
or larger-depth sweep. A larger depth is promoted only from measured
acceptance and end-to-end wall time on the exact artifact.

## Interfaces and configuration

Planned public controls:

- `--ngram-cache-limit SIZE`: test/diagnostic override; pinned production uses
  the measured construction-time calculation with a 20 GiB payload ceiling.
- `--expert-cache-limit SIZE`: existing fixed routed-expert cache budget.
- existing MTPLX `--depth`, profile, context, sampling, and receipt controls.

`mtplx inspect --json` will expose source revision, artifact hashes, expert and
n-gram BF16 storage descriptors, planned payload/metadata/route/alignment/
transient bytes, supported
MTP depths, and `can_run`. It must fail before allocation when the requested
cache budgets cannot cover the configured prefill chunk or verification width.

## Error handling and integrity

Construction validates once:

- exact source repository and revision;
- Qwen Community License inclusion;
- model type, architecture, layer schedule, topology, shapes, dtypes, and all
  tensor counts;
- expert and n-gram manifests, hashes, offsets, and file sizes;
- expert affine quantization mode and group divisibility;
- memory-plan arithmetic against the 128 GiB process limit and runtime reserve;
- exact small-tensor self-checks for expert and n-gram dequantization;
- native extension and MLX compatibility.

Runtime short reads, digest failures, stale cache publications, or I/O worker
failures terminate the request and poison the lane. They never switch to a
different numerical path.

## Test strategy

Production changes follow test-driven development.

### CPU and small-model tests

1. Parse and pin the real config and 1,658-tensor index without weight data.
2. Match official n-gram multipliers, prime moduli, EOS-aware shifts, head
   offsets, and selected row IDs.
3. Prove exact fixed-byte cache planning and fail-before-allocation behavior.
4. Cover hit, miss, duplicate, eviction, pinned-victim refusal, stale ticket,
   short read, reset, and close races.
5. Prove adjacent-run coalescing and fixed maximum in-flight bytes.
6. Prove exact BF16 row byte identity through eviction and reload.
7. Cross-check Gated Residual, GDN, QSA, PLE, MoE, and full decoder-layer
   outputs against official Transformers on tiny deterministic configurations.
8. Cover AR and MTP cache growth, proposal, rejection, rollback, and commit.
9. Verify manifests, source provenance, derivative license files, and receipt
   schemas.

### Real-artifact gates

All real MLX work acquires `/tmp/mtplx-gpu-exclusive.lock` and restores the
pre-existing Qwen service afterward.

1. Convert sampled real expert rows and extract exact BF16 n-gram rows; record
   reconstruction error and hash provenance respectively.
2. Load the complete derivative under the 128 GiB memory plan.
3. Run deterministic AR prompts at short, 1K, and 16K contexts.
4. Run fixed-depth MTP and prove nonzero accepted drafts, correct physical row
   counts, cache rollback, and successful natural stopping.
5. Compare AR and MTP with matched prompts and seeds; report prefill, decode,
   wall time, peak memory, acceptance, output-count correctness, measured base
   residency, cache payload and overhead, and KV/MTP allocation.

Raw token identity is diagnostic, not the quality criterion. Final quality
uses paired task pass sets and McNemar statistics. Wall-time deltas are labeled
"faster" and kept separate from prefill/decode throughput.

## Acceptance criteria

The task is complete only when all of the following are current and evidenced:

1. The expert bank is 69,363,302,400 logical bytes (64.60 GiB), affine Q4,
   group size 64, with all 25,088 target-plus-MTP expert records present.
2. Fixed-size expert and n-gram caches stay within their configured byte plans;
   no hidden page-cache or dynamic-slot growth invalidates the process budget.
3. `mtplx inspect` reports the derivative runnable on this 128 GiB machine.
4. Real AR generation completes coherently and naturally at short, 1K, and
   16K contexts.
5. Real MTP generation accepts drafted tokens, survives rejection and rollback,
   and is faster in end-to-end wall time than matched AR on at least one
   realistic 1K-input coding workload.
6. The shipped n-gram representation is the exact BF16 table streamed from SSD
   through the construction-sized eviction cache.
7. Focused tests, the documented repository baseline, build, hygiene, and fresh
   environment smoke checks pass, with unrelated pre-existing failures clearly
   separated.
8. A focused PR is open against `OpenSourceWTF/mtplx-moe`, with raw receipts and
   artifact provenance linked.
9. `OpensourceWTF/Qwen3.8-Flash-Next-MTPLX-Q4` contains the verified derivative,
   Qwen Community License and attribution, manifests, checksums, model card,
   and exact MTPLX usage; a clean remote download reproduces inspection and a
   generation smoke test.

## Failure-mode review

### Critical: n-gram I/O does not overlap enough

Random rows may cause page amplification or first-layer compute may be too
short to hide storage latency. Guarded lookup traces outside the production hot
path select LRU or CLOCK eviction from measured reuse behavior.
The BF16 lane measures cache hit rate, coalesced bytes, I/O wait at the layer-2
boundary, and end-to-end wall time outside the measured hot path. Eviction can
only cause an exact SSD reload; it cannot change numerical results. The port is
not called performant from theoretical payload size alone.

### Critical: Qwen4 arithmetic or MTP alignment differs from prior Qwen

Qwen4 changes token mixing, sparse attention, residual ownership, n-gram
injection, and MTP index reuse. The implementation is derived from the official
source and tiny-model parity, not transplanted from Qwen3-Next. MTP is not
enabled on the real model until fixed-depth cache and logit gates pass.

### Critical: artifact peak disk or memory exceeds the machine

Conversion is shard-streamed and writes directly into final banks. Before
download, preflight reserves source, output, temporary, and safety bytes. If
space is still insufficient, only the verified re-downloadable 91 GiB Hy3
artifact may be removed under the user's authorization; unique receipts and
generated banks are never deleted.

### Minor: first PR is text-only

The official checkpoint is multimodal, but adding a new vision path does not
help the requested MTP/speculative text objective and would broaden the first
PR substantially. Vision files remain available for a separately gated follow-
up; the model card and CLI must state the limitation plainly.

## Rollout

1. Land tiny-model arithmetic and cache tests.
2. Land converter and real-header manifest tests.
3. Produce sampled conversion receipts.
4. Convert the pinned full artifact.
5. Verify AR, then fixed-depth MTP, then measured BF16 n-gram streaming.
6. Run matched quality/performance gates under exclusive GPU ownership.
7. Final review, push, and open the code PR.
8. Upload only the verified shipping representation, verify remote hashes and a
   clean-download smoke test, then publish the model card and PR links.
