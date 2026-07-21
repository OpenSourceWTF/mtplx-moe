# Task O7 — IQ-transcode pilot

**Question.** Does transcoding published imatrix-calibrated GGUF i-quants
(IQ1_M / IQ2_XXS / IQ3_XXS) into our MLX-servable formats (t158 ternary,
affine q2 gs64, affine q4 gs64) inherit their calibration quality, or does the
transcode loss eat it?

**Answer (weight-space).** The transcode loss eats it. Every source x target
delta is negative: encoding our format from the i-quant dequant is always
worse than encoding the same format directly from bf16. The calibration prior
does **not** survive the transcode in weight space.

## Source substitution (read this first)

`LordNeel/Hy3-GGUF` — the repo named in the task/notes — **exists and is
imatrix-calibrated, but carries none of IQ1_M / IQ2_XXS / IQ3_XXS.** Its only
IQ-family quant is `IQ2_M`; the rest are K-quants (`Q3_K_L`, `Q4_K_M`,
`Q6_K`). The cited recipe ("40 layers IQ1_M / 39 IQ2_XXS gate-up / IQ3_XXS
down") does not match that repo.

Substituted **`bartowski/Hy3-GGUF`**, the canonical i-quant producer, which
carries exactly IQ1_M, IQ2_XXS and IQ3_XXS and is genuinely imatrix
(HF tag `imatrix`; embedded `quantize.imatrix.chunks_count=812`,
`entries_count=876`, `dataset=/training_dir/calibration_datav5.txt`,
`file=.../Hy3-imatrix.gguf`). This satisfies the experiment's premise (imatrix
i-quants exist for Hy3). An i-quant Hy3 GGUF with imatrix does exist, so no
STOP was warranted.

## Decoder validation

- **gguf 0.19.0** `gguf.quants.dequantize` provides pure-numpy decoders for
  all three formats with the correct ggml block sizes (IQ1_M 56 B/256,
  IQ2_XXS 66 B/256, IQ3_XXS 98 B/256).
- **Hand-validated IQ2_XXS on a real fetched block** with an independent
  scalar reimplementation — sign derived from the even-parity rule (not gguf's
  ksigns constant), per-subgroup scale from the top-4 bits, 256x8 codebook
  decoded independently from `grid_hex`. Result: **bitwise-equal to
  gguf.quants** (`max_abs_diff = 0.0`, all 256 elements nonzero, scale +/-0.065).
- **Correctness gate (cos vs bf16) passes for all three**: source cosines land
  0.872-0.984, finite, plausible scale, monotone with bitrate. Orientation /
  layer / expert mapping confirmed (GGUF `blk.L.ffn_{gate,up}_exps` expert-slice
  `(1536,4096)` aligns directly to `model.layers.L.mlp.experts.E.{proj}_proj`,
  no transpose).

## Verdict table (means over n=8 samples/format; `[min]` in brackets)

Samples: gate/up_exps only, spread across layers {1,12,23,34,45,56,67,78},
mixed gate/up, 8 distinct experts. Each sampled tensor's actual ggml type was
verified `== source format` before use (gate/up are uniformly the source
format across 79 layers; 1 layer is Q4_0 and is excluded by the filter).

| source | target | cos(bf16, src) | cos(bf16, tgt<-bf16) | cos(bf16, tgt<-IQx) | cos(src, tgt<-IQx) transcode-loss | **delta = iqx - bf16** |
|---|---|---|---|---|---|---|
| IQ1_M (1.75b) | t158 | 0.9006 [0.8723] | 0.8994 [0.8836] | 0.8609 [0.8184] | 0.9568 [0.9264] | **-0.0386 [-0.0652]** |
| IQ1_M | q2 gs64 | 0.9006 | 0.9223 [0.9096] | 0.8600 [0.8290] | 0.9514 [0.9364] | **-0.0623 [-0.0806]** |
| IQ2_XXS (2.06b) | t158 | 0.9406 [0.9224] | 0.8994 [0.8836] | 0.8480 [0.8053] | 0.9112 [0.8879] | **-0.0514 [-0.0783]** |
| IQ2_XXS | q2 gs64 | 0.9406 | 0.9223 [0.9096] | 0.8583 [0.8254] | 0.9186 [0.8959] | **-0.0640 [-0.0842]** |
| IQ3_XXS (3.06b) | t158 | 0.9830 [0.9782] | 0.8994 [0.8836] | 0.8875 [0.8651] | 0.9006 [0.8848] | **-0.0119 [-0.0185]** |
| IQ3_XXS | q2 gs64 | 0.9830 | 0.9223 [0.9096] | 0.9098 [0.8926] | 0.9233 [0.9099] | **-0.0126 [-0.0170]** |
| IQ3_XXS | q4 gs64 | 0.9830 | 0.9958 [0.9950] | 0.9772 [0.9715] | 0.9942 [0.9934] | **-0.0185 [-0.0235]** |

Download: **76.3 MB total** (31.4 MB split headers + 43.3 MB IQ expert slices +
1.6 MB hand-validation block) — far under the 2 GB budget. bf16 ground truth
was read from local shards via `pread` at safetensors offsets (0 download).

## Reading

Every delta is negative, so **the imatrix calibration prior does not transfer
through a transcode into our weight-space formats** — a two-hop
bf16->i-quant->target path lands 4-6 cosine points below encoding the target
directly from bf16 for IQ1_M/IQ2_XXS, and ~1-2 points below for IQ3_XXS. The
delta magnitude tracks source fidelity: IQ3_XXS is already a near-bf16
reconstruction (src cos 0.983), so re-encoding it barely differs from encoding
bf16, whereas IQ1_M/IQ2_XXS inject enough error that our coarse affine/ternary
grids re-quantize a corrupted signal. The `cos(src, tgt<-IQx)` column shows our
codecs reproduce the i-quant dequant itself faithfully (0.90-0.99), so the
damage is not a codec failure — it is the unavoidable compounding of two lossy
grids. Notably the published i-quants beat our own targets on the *source*
side: IQ2_XXS weight-cosine 0.9406 exceeds our q2-gs64-from-bf16 0.9223, and
IQ1_M 0.9006 ties t158-from-bf16 0.8994 — the better reconstruction lives in
the i-quant's own non-linear grid and is destroyed the moment we flatten it
into an affine/ternary grid.

**Caveat / anomaly to flag.** This pilot measures *unweighted weight-space*
cosine, but imatrix calibration optimizes *activation-weighted* output error —
it deliberately trades weight-space fidelity for importance-weighted output
fidelity. So a negative weight-space delta is consistent with, and does not
falsify, the possibility that the calibration still helps in activation space;
it only shows the benefit is not visible as, and cannot be inherited as,
plain weight-space fidelity by a re-quantizing transcode. Two secondary
anomalies: (1) **layer 1** is a consistent outlier across all formats — lowest
source cosine (0.872 IQ1_M) and most negative delta — the first MoE block is
genuinely more quantization-sensitive; layers 12-78 are near-identical to four
decimals. (2) In the IQ2_XXS and IQ1_M files the `down_exps` are bumped one
tier (IQ2_XS / IQ2_XXS respectively) and in IQ3_XXS half the `down_exps` are
Q4_K — the "IQx everywhere" mental model is wrong; only gate/up carry the
nominal type uniformly, which is why sampling was restricted to gate/up with a
per-tensor type check.
