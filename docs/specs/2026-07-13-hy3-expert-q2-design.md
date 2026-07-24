# Hy3 streamed expert-only MLX Q2 artifact

Status: approved experimental design; production implementation has not started

## Context and premise

The conversion source is the pinned local artifact at
`/Users/davidtai/.cache/huggingface/hy3-expert-only-mlx-q4`. It combines
byte-exact resident tensors derived from
`tencent/Hy3@716aa7241bd6d95896be4ebfc761162a9c4d49ef` with the routed affine-Q4
expert sidecar derived from
`pipenetwork/Hy3-4bit@160619d3f96c8470350b6dac0ef033a8381551e3`.
Its local manifest is pinned by all of these identities:

- model key `hy3-expert-only-q4`;
- manifest file SHA-256
  `e7fcfd6c69486456af4261d908d95f8a84a391d6a273ff1cff02a15f73fac92d`;
- canonical manifest SHA-256
  `507ca09cebb9ef5180c46401db7b61d8a9759ffd04ffbc97c5dbba0e9ef89f43`;
- conversion-provenance SHA-256
  `e832743f84f09f5a548a8734b2ab6d75043e32723223e90b1c05da074b42e7f2`;
- Q4 sidecar size `161,036,107,776` and SHA-256
  `5ba698b9b2c51bca66254e5d8d35101325e37dfe40744294d4aa233c980472ae`;
- resident index SHA-256
  `b901cc98a86131b519d69294d65a20023b5ac4d5706c96bd4bf128ef7e41ef5e`.

Hy3 has 79 routed layers, 192 experts per routed layer, and top-8 routing.
The source therefore contains 15,168 routed records. Requantizing only those
records from MLX affine Q4/group-64 to affine Q2/group-64 reduces each record
from `10,616,832` to `5,898,240` bytes, a 44.44% reduction. The same
`83,034,243,072`-byte global cache used by the issue #29 benchmark can hold
7,821 Q4 records (51.562% of the routed bank) or 14,077 Q2 records (92.807%).
That is a strong enough cache-capacity premise to test.

The source is already Q4. This design is deliberately a Q4-to-Q2
double-quantization experiment, not a claim that Q2 preserves model quality.
The experiment should not exist as a production lane unless its measured
quality and resource behavior pass the gates below. If it fails, the next
credible experiment is BF16-derived or calibration-aware expert quantization,
not a weaker acceptance threshold.

## Considered approaches

### 1. Uniformly quantize the complete checkpoint to Q2

This maximizes nominal size reduction but changes routers, attention, shared
experts, embeddings, norms, and the LM head. It confounds routing changes with
expert-storage changes and violates the requirement that the known-good
production behavior remain intact. Rejected.

### 2. Derive expert Q2 directly from the official BF16 checkpoint

This should have a better quality ceiling than Q4-to-Q2, but it requires
reading and reconstructing the full official expert bank and creates a
different experiment from the requested pinned local Q4 derivation. It remains
the fallback only if double quantization fails quality. Deferred.

### 3. Requantize only the authoritative Q4 sidecar and reuse residents exactly

This isolates the variable under test. The converter reads one verified Q4
record at a time, requantizes its three projections, writes one Q2 record, and
copies only the already-built resident shards without reserializing their
payloads. Recommended.

## Scope

Build an explicit local artifact with this precision policy:

| Component | Output representation | Runtime placement |
| --- | --- | --- |
| Routed experts, layers 1-79 | MLX affine Q2, group size 64 | `experts.bin` plus bounded cache |
| Routers and correction biases | Source BF16 gates plus FP32 correction biases, bytes unchanged | Resident |
| Attention and shared experts | Source tensor bytes unchanged | Resident |
| Embedding, norms, and LM head | Source tensor bytes unchanged | Resident |
| Layer-80 MTP head | Not included | Separate existing artifact only |

The converter consumes exactly `79 * 192 = 15,168` source records. Every output
record contains the canonical nine components for `gate_proj`, `up_proj`, and
`down_proj`: packed U32 weight, BF16 scales, and BF16 biases.

The per-record Q2 geometry is exact:

| Component | Shape | Dtype | Bytes |
| --- | --- | --- | ---: |
| gate/up packed weight, each | `(1536, 256)` | U32 | 1,572,864 |
| down packed weight | `(4096, 96)` | U32 | 1,572,864 |
| gate/up scale or bias, each | `(1536, 64)` | BF16 | 196,608 |
| down scale or bias, each | `(4096, 24)` | BF16 | 196,608 |

Thus:

- packed weights per record: `4,718,592` bytes;
- scales and biases per record: `1,179,648` bytes;
- record: `5,898,240` bytes, exactly 360 16-KiB alignment units;
- routed sidecar: `89,464,504,320` bytes;
- unchanged router payload: `124,316,928` bytes, stored as source BF16 gate
  weights plus FP32 correction biases;
- unchanged resident tensor payload: `17,494,289,664` bytes;
- total logical tensor bytes: `106,958,793,984` bytes.

## Non-goals

- Do not change the production `hy3-q4` descriptor, auto-detection, or default
  behavior.
- Do not quantize any resident tensor or reinterpret the whole checkpoint as
  Q2 in `config.json`.
- Do not download, splice, or imply use of a third-party Hy3 Q2 checkpoint.
- Do not copy the source artifact's unreferenced 34 community Q4 shards into
  the target.
- Do not copy or quantize the source artifact's `mtp/` directory. The exact
  tensor total in this design is AR-only.
- Do not enable the existing layer-80 MTP package for the Q2 trunk without a
  separate parity, acceptance, memory, and TPS experiment.
- Do not change cache admission, eviction, prediction, or scheduling policy.
- Do not produce a conventional self-contained `mlx_lm` checkpoint. The
  authoritative routed weights remain in MTPLX's sidecar format.

## Architecture and data flow

The output is a sibling of the existing model, outside Git:

```text
/Users/davidtai/.cache/huggingface/hy3-expert-only-mlx-q2/
  config.json
  model.safetensors.index.json
  model-00001-of-00018.safetensors
  ...
  model-00018-of-00018.safetensors
  experts.bin
  expert-manifest.json
  conversion-manifest.json
  tokenizer and chat-template files
```

The builder uses a sibling work directory named
`.hy3-expert-only-mlx-q2.incomplete`. Its CPU-only staging phase validates the
source manifest, conversion provenance, resident index, resident shard hashes,
ancillary-file hashes, and available space. It copies the 18 resident shard
files and their index byte-for-byte; it never repacks resident tensors. A
copy-on-write filesystem clone may be used, but hard links are forbidden
because a target write must not alias the source inode.

The MLX conversion phase processes records in sorted `(layer, expert)` order.
For each record it:

1. reads the exact Q4 sidecar range and checks the source record SHA-256;
2. slices the nine canonical Q4 components and validates their metadata;
3. dequantizes one projection with
   `mx.dequantize(..., bits=4, group_size=64, mode="affine")`;
4. requantizes it with
   `mx.quantize(..., bits=2, group_size=64, mode="affine")`;
5. evaluates, validates, serializes, and releases that projection before the
   next projection;
6. writes and fsyncs the complete Q2 record before fsyncing its journal entry.

The work directory is published by atomic rename only after deep verification.
A rerun accepts only the contiguous journal prefix whose source and output
hashes still match current bytes and the pinned build fingerprint.

## Format and runtime contracts

Two explicit descriptors are needed:

- `hy3-expert-only-q4` describes the pinned local Q4 source and provides the
  fair resident-identical control;
- `hy3-expert-q2` describes the experimental output.

Both retain Hy3's 80-layer layout, routed layers 1-79, 192 experts, top-k 8,
resident router semantics, `327,680` KV bytes per token, and no embedded MTP.
The Q2 descriptor changes only the routed quantization width and the exact
routed/total byte accounting. Model-type auto-detection continues to resolve
`hy_v3` to `hy3-q4`; the two new keys are explicit choices only.

The manifest format is extended without weakening Q4:

- affine expert manifests may declare bits 2 or 4;
- packed-weight shapes derive from the selected descriptor's bit width;
- authoritative sidecar records have exactly nine ordered, contiguous
  segments covering their declared record range;
- resident entries may refer only to hash-pinned safetensors shards;
- the authoritative output contains the 18 referenced resident shards plus
  one sidecar shard, not the source manifest's unused Q4 shards;
- model key, source identity, bit width, group size, record Cartesian product,
  component shapes, resident inventory, and all byte totals fail closed.

Every direct, gathered, component-bank, mapped, all-hit, and split-route MLX
expert call receives `runtime.spec.quant_bits`. No default changes from 4, and
existing Q4 tests continue to prove 4-bit execution.

## Provenance, resume, and publication

`conversion-manifest.json` records at least:

- the local source path and model key;
- official resident source repo/revision;
- community Q4 oracle repo/revision;
- source manifest file and canonical hashes;
- source conversion-provenance, index, config, and sidecar hashes;
- `q4_to_q2`, source bits 4, target bits 2, group size 64, affine mode;
- producer Git commit/dirty state and MLX version;
- every target resident shard, ancillary file, record, and sidecar hash;
- journal digest/count and pilot-report digest;
- exact routed, resident, and total bytes.

Final builds require a clean producer commit. The source is never mutated or
deleted. An existing final output is refused. Neither a partial sidecar nor a
work directory is loadable as the final model.

## Operational isolation

CPU-only phases (`preflight`, resident/ancillary `stage`, `finalize`, and
`verify`) leave Qwen in its current state and do not import MLX. Only
real-record `pilot`, expert `convert`, quality evaluation, and hardware
benchmarks enter an exclusive MLX window.

The exclusive wrapper captures Qwen's launchd, process, API, and exact model
list state; fails closed if those disagree; stops Qwen; runs one child command;
and restores the exact captured state in `finally`, including child failure or
signal. It may never become a blanket wrapper around hashing or file copying.

## Testing and gates

Implementation is test-first.

### Structural and numerical gates

1. Descriptor tests prove all Q2 shapes and exact record/routed/resident/total
   bytes, while leaving `hy3-q4` unchanged.
2. Manifest tests accept only affine Q2/Q4, distinguish the two explicit Hy3
   keys, and reject malformed sidecar or resident inventories.
3. MLX tests prove the selected bit width reaches every direct and gathered
   execution path and preserve existing Q4 behavior.
4. Synthetic conversion tests prove canonical order, bounded memory, exact
   resident file identity, journal recovery, provenance, and atomic publish.
5. A real-record pilot samples early, middle, and late routed layers and edge
   expert IDs. It records Q4-versus-Q2 cosine and normalized error per
   projection and requires finite Q2 plus direct-versus-streamed Q2 parity.
6. Deep verification checks all 15,168 record hashes, the complete sidecar,
   all 18 resident shard hashes, exact resident index/header inventory, copied
   config/tokenizer hashes, and absence of routed or MTP tensors in residents.

### Quality gate

The Q4 and Q2 lanes use the same resident tensors, tokenizer, prompts, corpus,
cache budget, and deterministic settings. The evaluator records greedy token
agreement and first divergence as diagnostics. The hard gate is finite
execution and teacher-forced perplexity no more than 5% above the local Q4
control on the identical tokenized corpus. Failure keeps the artifact rejected;
the threshold is not relaxed after observing the result.

### Resource and throughput gate

The paired campaign uses the issue #29 byte budget and runtime conditions:
112-GiB total limit, 8-GiB runtime reserve, `83,034,243,072` expert-cache bytes,
18,888 live KV tokens, global LRU, component banks, 32 transient slots,
64-MiB reads, and `F_NOCACHE`.

One order-balanced telemetry campaign reports cache capacity, hit/miss rates,
physical and logical reads, bytes per generated token, reader throughput and
occupancy, miss-wait upper bounds, fences, evictions, memory pressure, and TPS.
At least four telemetry-off pairs use balanced Q4-first/Q2-first order.
Promotion requires every paired TPS delta to be positive, both lanes to reach
the fixed output-token cap, the Q2 lane not to increase expert bytes per token,
no memory-plan/swap/integrity failure, and the quality gate to pass. The full
paired range and median are reported; there is no invented universal percent
threshold.

Serialization success, a smaller file, or a modeled cache hit rate is not a
promotion result.

## Rollout

Tracked implementation belongs on the isolated `codex/hy3-expert-q2` branch.
All model bytes, journals, pilots, quality output, and raw benchmarks remain
outside Git. The artifact and model key remain explicit and experimental even
after gates pass; changing the production default requires a separate decision
and PR.

## Failure-mode review

1. **Double quantization destroys useful expert accuracy.** Critical. The
   real-record diagnostics and fixed perplexity gate expose it. Failure rejects
   the lane and routes follow-up to BF16-derived or calibration-aware Q2.
2. **Q2 bytes are interpreted through Q4 geometry.** Critical. Distinct model
   keys, descriptor-derived shapes, exact manifest/spec validation, bit-width
   spies, and streamed/direct parity make this a load failure rather than
   silent corruption.
3. **The output quietly changes residents or includes the MTP head.** Critical.
   Whole-file resident hashes, exact index/header inventory, the fixed
   `17,494,289,664`-byte resident total, and an explicit `mtp/` exclusion prevent
   this.
4. **Interruption publishes a partial artifact or mixes source versions.**
   Critical. Hash-checked contiguous resume, fsync ordering, a stable work
   directory, deep verification, and atomic rename prevent partial publication.
5. **Larger cache capacity does not improve end-to-end throughput.** Minor for
   artifact correctness but decisive for retention. Same-budget resource
   evidence must show whether storage wait was removed or compute became the
   wall; no speed claim follows from capacity math alone.
