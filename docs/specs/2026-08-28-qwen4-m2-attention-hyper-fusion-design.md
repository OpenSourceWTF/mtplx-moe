# Qwen4 M=2 Attention Hyper D/U Fusion

## Scope

Extend the accepted fixed-M=2 hyper D/U route from the MLP-side hyper
connection to the attention-side hyper connection. Promotion remains gated by
the exact 16,384-input/1,024-output temperature-1 production workload.

This does not change the grouped RMS norm, attention/GDN arithmetic, sampling,
MTP acceptance, cache ownership, rollback, prefill, M=3, or batched serving.

## Evidence

The post-fusion MLX 0.32.2 trace records one remaining stock attention hyper
chain per active layer: about 48 grouped norms, down-plus-inject q4/g32
projections, and up q4/g32 projections per verifier call. The accepted MLP D/U
route reduced linear-layer command buffers from 314.4 to 293.4 microseconds and
full-attention buffers from 440.8 to 434.0 microseconds. Explicit GPU completion
waits remain negligible.

## Design

Keep `attn_hyper_connection.hc_norm` unchanged. For exact
`(1, 2, 10240)` BF16 capture traffic, feed its lazy normalized output into the
existing construction-bound D/U callable. The callable owns fresh MLX output
arrays and captures immutable affine-q4/group-32 weights; it has no global
scratch, request key, counter, environment read, or exception fallback.

Construction validates and binds all 48 attention modules alongside the 48 MLP
modules, runs finite per-output parity checks, and installs the capture route
only if the complete set succeeds. Partial installation fails before measured
generation. Logical shapes other than exact batch-1/M=2 take the explicit stock
attention hyper route.

The capture loop prebinds the selected attention callable once before entering
the 48-layer loop. The enabled fixed-shape branch performs only the stock norm
and the two existing D/U dispatches. This preserves concurrent dispatching and
avoids a monolithic norm/D/U kernel with a global synchronization problem.

## Alternatives

- **Selected:** stock grouped norm plus existing D/U kernels. It removes the
  measured duplicate projections with the smallest arithmetic and scheduling
  change.
- Fuse norm, D, and U into a new kernel. Rejected because D needs all normalized
  streams after four independent reductions; a monolithic launch would require
  repeated work or unsafe cross-threadgroup synchronization and would increase
  head-of-line risk.
- Leave attention hyper stock. Retained as the explicit non-M2 and control
  route, but it leaves the measured duplicate D/U work untouched.

## Failure modes and gates

- A batch greater than one with sequence length two could silently collapse
  rows. The optimized route therefore requires the complete exact shape for
  both hidden and normalized arrays; B>1 remains stock.
- A partial 48-layer install could mix arithmetic within one forward. The
  installer rejects any partial attention or MLP binding set.
- A numerically valid BF16 difference can change stochastic tie breaks. The
  construction self-check rejects non-finite output and deltas over the existing
  per-output tolerance; production receipts record digest and acceptance rather
  than requiring the old stochastic trajectory.
- The extra custom boundary could lose to MLX scheduling. Promotion requires a
  repeatable exact-workload improvement over the 56.66 tok/s mean and no repair.

## Verification

Focused tests cover construction binding, partial-install rejection, exact M=2
routing, B>1 stock routing, finite parity, and harness lane recording. The
candidate is then measured with the unchanged guarded production harness. A
regression is reverted; an improvement is committed and delivered to PR #368.

## Measured outcome

The exact production candidate produced 59.16, 58.68, and 57.67 tok/s, a
three-run mean of 58.50 tok/s versus the preceding 56.66 tok/s mean. All three
runs generated exactly 1,024 tokens with the production sampler, stable digest
`d4870b96...`, 393/567 accepted drafts, 582 verifier calls, 187 corrections,
and zero repair.

The bracketed unchanged control produced 56.74 tok/s with 13.284 seconds of
verifier evaluation. The adjacent candidate produced 57.67 tok/s with 12.785
seconds of verifier evaluation, reducing evaluation by 0.50 seconds despite
five additional verifier calls. Total wall was 111.26 seconds for control and
111.60 seconds for candidate: the 0.29-second decode saving was offset by a
0.30-second increase outside prefill and decode. The change is therefore a
repeatable decode-throughput improvement, not yet a demonstrated total-wall
improvement.
