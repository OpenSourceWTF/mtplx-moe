# Kimi K3 Q2_K to t158 streaming runtime

Date: 2026-07-27

## Goal

Build and run a resumable MTPLX expert-streaming artifact from the immutable
`GrEarl/Kimi-K3-GGUF` revision
`0169245d3ea1473a3f9f03bca821d855df5fb2a3`.

Only the 92 routed-MoE layers are requantized. Every BF16 and F32 resident
tensor, including shared experts, latent-MoE projections, attention, the
dense layer, LM head, and vision tower, is copied without numerical
conversion. The first runnable target is K3 text generation through the same
MTPLX service and benchmark boundary used by the endorsed GLM streaming
package; the byte-preserved vision and multimodal tensors remain in the
artifact but are not loaded by the text-only route.

## Measured source contract

All 96 GGUF headers were inspected at the pinned revision:

- 2,904 tensors total: 2,122 BF16, 506 F32, and 276 Q2_K.
- Resident tensor payload: 114,404,258,816 bytes.
- Text resident payload: 113,509,540,864 bytes across 2,460 tensors.
- Byte-preserved non-text payload: 802,428,928 vision bytes plus 92,289,024
  multimodal-projector bytes.
- Routed Q2_K payload: 893,399,334,912 bytes.
- Routed layers: exactly layers 1 through 92.
- Every routed layer owns exactly `ffn_gate_exps`, `ffn_up_exps`, and
  `ffn_down_exps`.
- Each layer owns 896 experts and routes top 16.
- Gate/up expert shape is `(3072, 3584)`; down is `(3584, 3072)`.

The GGUF expert tensors are merged with the expert ID as the outermost
dimension. One expert is therefore one contiguous Q2_K slice in each of the
three tensors.

## Conversion arithmetic

Q2_K is decoded exactly according to its 256-value, 84-byte block:

1. sixteen packed scale/min nibbles;
2. sixty-four 2-bit value bytes;
3. FP16 `d`;
4. FP16 `dmin`;
5. `value = d * scale * q - dmin * min`.

The decoded matrix is then encoded with MTPLX's existing `t158` definition
along its input axis in groups of 64:

- threshold: `0.7 * mean(abs(group))`;
- trits: `{-1, 0, +1}`;
- scale: mean absolute value of nonzero trits;
- packing: five base-3 trits per byte, 13 bytes per group;
- scale storage: one BF16 value per group.

This is 15 bytes per 64 source values, or 1.875 physical bits per weight.
One expert record is 7,741,440 bytes. One routed-layer part is
6,936,330,240 bytes. The complete routed bank is 638,142,382,080 bytes and
the tensor payload of the assembled artifact is 752,546,640,896 bytes.
The text runtime prices 751,651,922,944 bytes: the routed bank plus only the
113,509,540,864 text residents.

This is explicitly a Q2_K-to-t158 requantization. A pilot must report error
against the decoded Q2_K source; it must not describe that result as error
against the original MXFP4 checkpoint.

## Output layout

The artifact is assembled in a new directory and never mutates the source.

- One raw expert part per routed layer:
  `experts-t158-layer-001-of-092.bin` through
  `experts-t158-layer-092-of-092.bin`.
- Records are ordered by expert ID within each layer.
- Each record contains, in order, gate packed/scales, up packed/scales, and
  down packed/scales.
- `expert-manifest.json` uses MTPLX's multi-part sidecar schema. Record
  offsets are relative to their part and no record crosses a part boundary.
- Resident GGUF tensors are copied into sharded safetensors with GGUF
  dimensions reversed into row-major safetensors shapes. BF16/F32 payload
  bytes are otherwise unchanged.
- The safetensors index excludes the three routed expert tensors because
  the manifest is their authoritative storage.
- Official Kimi K3 configuration, tokenizer, processor, model code, and
  generation metadata are pinned and copied from
  `moonshotai/Kimi-K3@9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.
- A conversion receipt records both source revisions, inventory totals,
  codec parameters, and output hashes.
- The text runtime receives a flattened, pinned `kimi_linear` configuration
  and a text-only resident index. The original top-level `kimi_k3`
  configuration and every byte-preserved multimodal tensor remain available
  alongside it, but are outside the MTPLX text resident allowlist.

## Resume and construction boundary

Each layer is written to a `.partial` file. Completion requires the exact
part size, record count, per-record hashes, and whole-part hash; only then is
the file renamed to its final name. A completed part is immutable and can be
adopted on resume only when its journal matches the pinned source revision,
source tensor identities, codec, geometry, and file hash.

The final manifest and safetensors index are written only after all 92 expert
parts and every resident shard pass validation. Source and output paths must
be different. Existing unrelated model directories are never touched.

## Quality gate

Before the full burn:

1. Decode selected real Q2_K blocks and compare them to an independent
   `gguf` reference decoder.
2. Convert at least one real expert from a routed layer.
3. Decode its t158 record and report RMSE, relative RMSE, and cosine against
   decoded Q2_K for gate, up, down, and the concatenated record.
4. Reject non-finite values, wrong shapes, wrong source types, or unexpected
   tensor names before installing the full conversion route.

No quality threshold is invented in advance. The measured pilot is retained
as a receipt so the additional loss is visible.

## Exact text-runtime contract

The installed `mlx-lm` 0.31.3 `kimi_linear` module is not K3-compatible. The
MTPLX overlay therefore owns an exact K3 text-model port instead of routing
the checkpoint into that older same-named topology.

The port preserves these measured K3 semantics:

- model hidden size 7,168, latent routed-expert width 3,584, MoE intermediate
  width 3,072, 896 routed experts, top 16, and two resident shared experts;
- routing on the original 7,168-wide hidden state, sigmoid scores plus
  correction bias for selection only, unbiased selected scores for weights,
  and top-16 renormalization;
- latent-MoE down projection before the streamed experts, K3 SITU rather than
  SwiGLU inside every routed expert, post-expert RMS normalization, latent-MoE
  up projection, and the shared-expert result from the original hidden state;
- SITU in FP32:
  `4*tanh(gate/4)*sigmoid(gate) * 25*tanh(up/25)`, cast back to the input
  dtype;
- q-LoRA MLA, compressed KV, the checkpoint's no-position-embedding behavior,
  sigmoid MLA output gate, one latent cache head of width 576, and the
  published head dimensions. The q-a and kv-a low-rank RMS norms use the
  production runtime's configuration epsilon of `1e-5`;
- KDA short convolutions, L2-normalized Q/K arithmetic, vector forget gate,
  beta gate, full-rank output gate, and the safe forget activation
  `-5*sigmoid(exp(A_log)*(g+dt_bias))`;
- attention-residual blocks every 12 layers and the final output-residual
  projection/norm.

The pinned checkpoint stores each KDA `A_log` as 128 F32 values even though
the published configuration and runtime use 96 heads. The day-zero vLLM K3
runtime at `658f2f56e557e8141f8726b4626c51496d78d914` resolves this at its
weight-loading boundary by taking the first 96 values. MTPLX preserves that
explicit K3 mapping and rejects every other short or surplus shape; it does
not expose a general crop rule.

This vLLM revision is the production arithmetic oracle where the pinned
Hugging Face remote module and the day-zero validated serving path disagree.
Other official model semantics are retained directly. In particular, vLLM's
final-norm relocation exists to feed its speculative draft and is not copied
into this no-MTP K3 route.

The batch-one fixed KDA cache is priced explicitly: 434,110,464 bytes for 69
FP32 recurrent states and 15,261,696 bytes for their BF16 convolution states,
449,372,160 bytes total. The 24 latent MLA layers add 27,648 bytes per cached
token.

The streamed switch is installed only after model construction has validated
the K3 geometry, codec, layer ownership, resident allowlist, and exact
activation route. The enabled hot path directly calls the prebound SITU t158
component-bank implementation; it contains no invariant revalidation,
try-custom fallback, or per-token engagement counter.

K3 has no MTP weights in this checkpoint, so the first runtime and benchmarks
run autoregressively with MTP disabled. That limitation is reported rather
than silently substituting another model's speculative head.

## Runtime correctness and benchmark gates

Before a full-model benchmark:

1. Compare SITU, router selection/weights, latent-MoE assembly, MLA, KDA
   recurrence, and attention-residual helpers against independent fixed-weight
   reference formulas on small deterministic shapes.
2. Load a tiny K3 fixture through the same sanitize and construction route and
   require exact tensor ownership with no missing or unexpected text weights.
3. Install one real converted expert layer, execute it through the streamed
   SITU route, and compare its decoded-t158 output to the same t158 weights
   evaluated densely.
4. Construct the full resident model before generation so any topology,
   manifest, dtype, or cache mismatch fails once outside the measured path.
5. Run a deterministic generation smoke test, then collect cold construction
   time, prefill throughput, autoregressive token throughput, resident memory
   with `vmmap -summary` or `footprint`, and MTPLX streaming/cache statistics.

All performance results are labeled as Q2_K-to-t158, text-only, and no-MTP.
