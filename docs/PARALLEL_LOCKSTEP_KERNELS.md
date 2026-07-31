# Parallel lockstep decoding — the width-2 MTP cohort and its kernels

How MTPLX decodes two requests in lockstep on one model, what the custom
kernels contribute, when the lane pays, and why the MoE (A3B) family uses a
different concurrency mechanism. Numbers are from receipted windows on an
M5 Max, 2026-07-26 → 07-31.

## Architecture: one shared forward for two streams

The cohort lane (`mtplx/qwen27b_mtp_cohort.py`, served via `--scheduler-mode
mtp_cohort_experimental`) pairs two decode-ready requests into one target
forward of shape `B=2, T=3` — each stream contributes its committed token plus
two MTP draft positions, so a cycle verifies six rows in a single pass instead
of two three-row passes. Per-request state is carried in immutable
`MTPK2VerifyTicket`s; construction installs a fixed route table over the
model's 497 dense qlinears after validating the exact model contract
(identity, quantization, layer topology, structural digests, and a numeric
self-check of every route at `dmax == 0.0` against stock).

Solo requests keep the ordinary depth-2 MTP path (`solo_mtp_protected`); the
lane only changes what happens when two requests are simultaneously
decode-ready.

## The kernels

Two hand Metal kernels compute the 6-row verify shapes bit-exactly:

- **M6 K-split** (`nax_verify._build_kernel_m6_ksplit_np`): the layer
  projections. Two SIMD-group K-partitions, scalar `v0..v5` row registers,
  explicit unrolls (load-bearing — an un-unrolled probe measured 0.08–0.16×),
  `simd_sum` reduction. Beats stock qmm 1.14–1.8× on the live shapes.
- **Wide LM-head QMV** (`_build_kernel_m6_qmv_wide_vec6`): the unique
  `(5120, 248320)` head. Reproduces MLX 0.32's `qmv_wide` arithmetic exactly —
  stock affine decode, eight-lane K ownership, per-vector accumulation order,
  shuffle reduction — which is what makes it bit-identical to stock.

**Bit widths.** Both kernels support 4-bit and 8-bit (2026-07-30): bit width
enters only as values-per-uint32 (8 nibbles vs 4 bytes) in the K-split, and as
the nibble-vs-byte decode block in the qmv_wide (the q4 path masks the high
nibble in place and folds the ×16 into the scale because stock does; at q8
each byte is the value). Both verified `dmax == 0.0` against stock at both
widths on the real shapes. **6-bit is supported on the K-split kernel**
(2026-07-31): values straddle word boundaries at 5.33/uint32, so the decode
works on 3-word/16-value blocks of the continuous little-endian bitstream
(LCM(6,32) = 96 bits, straddles at fields 5 and 10, all shifts compile-time).
Dequant verified bit-identical to `mx.dequantize` via one-hot probe (dmax 0.0
at gs 32/64/128, both callers); full matmul sits in the family's
reduction-order tolerance band (worst rel 1.3e-3) with argmax parity. The
rendered 4-bit source is test-asserted byte-identical to the pre-change
original. Consumer: the 35B-A3B Balance build's 6-bit dense projections
(routing them onto the lane is a separate dispatcher decision, deliberately
not changed here).
Cache keys and kernel names carry the bit width so variants cannot collide.

## Per-build contracts: `CohortProfile`

Everything model-specific — identity, publish-manifest filename, bit width,
stock-qmm reference label, qlinear count, structural digests — lives in a
`CohortProfile` selected by model directory at install time. Shared facts
(64 layers, depth 2, gs64, affine, verify strategy) stay module constants. A
profile with an unpinned qlinear digest reports the observed digest instead of
comparing, so a new build is pinned from its own construction receipt, never a
guessed constant. Candidate-tier upstream contracts (the 8-bit Quality build
ships `exactness_baseline.status: candidate-promoted-by-user-decision`)
require an explicit operator opt-in (`MTPLX_COHORT_ALLOW_CANDIDATE_CONTRACT`),
recorded in the receipt.

## Measured results (promotion gate: pair ≥1.35×, solo ≥0.99, token parity)

| build | c1 control→cand | c2 control→cand | pair ratio |
|---|---|---|---|
| 27B 4-bit Speed | 51.1 → 51.8 | 49.9 → **70.5** | 1.414 |
| 27B 8-bit Quality | 36.1 → 36.6 | 35.7 → **55.3** | **1.548** |

Live serving spot-checks: Speed 52.6 solo → 73.1 aggregate at two streams;
Quality 36.8 → 58.2. Token and acceptance parity hold in every gate.

## When the lane pays — and when it cannot

Lockstep needs both streams *decode-ready at the same cycle boundaries*.
Short/chat-shaped traffic pairs constantly and collects the full ratio.
Prefill-heavy traffic (e.g. 33k-token scan prompts) starves the pair: one
stream decodes while the other prefills, and measured aggregate *collapsed* to
~11 tok/s vs 32 solo. MTPLX's default `serial` scheduler exists for exactly
that regime.

## Why the A3B (MoE) family does not use this lane

The A3B's own receipts closed the question: width-2 MTP measured **dead**
(ΔC 0.882 ms/forward vs ~0.4 break-even, 2026-07-21), the architecture is
row-bound, and plain batched AR beats speculative decode at batch ("16 AR
streams beat 8 spec"; 696 tok/s aggregate at B=64). Its concurrency mechanism
is the fold-in lane (`mtplx/batched_decode.py`): fixed-shape cohorts, REFILL
continuous batching, `decode_mode=ar` row-packing, with a per-row byte-identity
contract (a stream batched among others commits the identical sequence to the
same stream run alone). A serving bridge for that lane — admission, one-run
REFILL, live per-row streaming via an `on_commit` observer at the driver's
commit site — is in progress on `serve/a3b-foldin-serving`.

The cohort's dense route table also has no MoE analogue (expert weights run
through `gather_qmm`, not the 497 dense qlinears), which is the structural
reason the M6 kernels have no 6-bit consumer: the only 6-bit build in the
lineup is the A3B Balance.

## Quality assurance summary

- Kernel exactness: `dmax == 0.0` vs stock, both widths, real shapes,
  MLX-0.32-pinned tests (the kernels reproduce that version's stock
  arithmetic; a different MLX changes the reference itself).
- Lane parity: committed tokens and acceptance identical with the lane on/off.
- End-to-end: full EvalPlus (HumanEval+/MBPP+) through the lane-enabled
  serving stacks matches published references — see
  `QWEN36_QUANT_EVAL_MATRIX_2026-07-31.md`.
