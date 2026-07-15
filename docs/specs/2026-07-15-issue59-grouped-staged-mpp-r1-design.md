# Issue 59 grouped staged-MPP R1 design

## Scope

Optimize the authoritative Hy3 MPP router projection for the real Apple M5 Max
and Hy3 shape: padded `M=8`, `K=4096`, `N=192`, FP32 activations, K-major BF16
weights, and FP32 partials. Preserve the selected precise SIMD R2 and its
deterministic MPP arithmetic contract.

This work is independently attributable to R1. Fast sigmoid belongs to #60;
split-K FP32 arithmetic belongs to #58; expert and MTP-model kernels are out of
scope.

## Measured gap

The retained N16/P8 implementation launches 96 one-SIMDgroup threadgroups. Each
task reads an `8 x 512` FP32 activation slice (16 KiB) even though twelve N tiles
consume the same slice for each K part. Across eight K parts, activation reads
therefore total 1.5 MiB, equal to the complete 1.5 MiB BF16 router weight. The
current frontier changes N and P but never groups N tiles or stages their common
activation operand.

The existing search also omitted the already-supported P2 and P32 schedules.
Those remain a separate direct-operand screen so grouping gains are not confused
with a different K partition.

## Alternatives

### A. One SIMDgroup per threadgroup, direct operands

This is the control. It maximizes independently schedulable threadgroups but
rereads the activation slice for every N tile.

### B. Multiple SIMDgroups per threadgroup, direct operands

Group N-tile tasks that share a K part while leaving both operands in device
memory. This isolates threadgroup scheduling effects but relies on the cache to
reuse activation data.

### C. Multiple SIMDgroups per threadgroup with staged activation

Load the common `M x K-span` activation slice into threadgroup memory once, then
let each SIMDgroup run one N tile against its own device-resident weight slice.
This is the recommended `grouped-staged` arm because it removes redundant activation traffic
without changing the MPP dot product, split-K reduction tree, partial layout, or
R2.

## Selected architecture

Tasks are ordered by K part and then N tile. One threadgroup owns exactly one K
part and a contiguous group of N tiles; groups never cross a K-part boundary.
The SIMDgroup count must divide `192 / N-tile`:

- N16: 2, 3, 4, or 6 SIMDgroups;
- N32: 2, 3, or 6 SIMDgroups;
- N64: 3 SIMDgroups.

The first screen covers P8, P16, and P32. Their staged activation footprints are
16 KiB, 8 KiB, and 4 KiB respectively. P4's 32 KiB operand is retained only as
a bounded follow-up if device limits and occupancy reporting show it is legal;
P1/P2 cannot fit the activation operand in the intended threadgroup-memory
budget.

Each threadgroup:

1. cooperatively loads the padded eight-row activation slice into FP32
   threadgroup memory;
2. executes one MPP `matmul2d` per SIMDgroup with the shared threadgroup left
   operand and a distinct BF16 device-weight N tile;
3. stores to the unchanged `[P, 8, 192]` FP32 partial layout;
4. returns to the existing precise G6 SIMD R2 kernel.

The `grouped-direct` and `grouped-staged` arms share the same task mapping. This
separates reduced scheduling overhead from actual activation reuse.

## Contracts and telemetry

The runtime remains fail-closed outside `M1...M8 x K4096 x N192` on the
qualified G17 MPP device. Construction-time K-major weights and memory
accounting do not change.

Every result records:

- N tile, K parts, SIMDgroups/threadgroup, total threadgroups and SIMDgroups;
- staged threadgroup bytes and modeled activation/weight/partial traffic;
- compile, resource, dispatch, and correctness failures separately;
- projection and complete precise-router timing;
- exact logits relative to the one-SIMDgroup MPP control;
- top-8 correctness relative to candidate logits, finite normalized weights,
  and repeated-run determinism;
- direct paired intervals against N16/P8 at every M1...M8, with K3/M4 primary.

The screen ranks legal arms. A winner receives a high-repeat K3 refinement and
then the mandatory isolated 1,024/1,024 K0-K7 end-to-end qualification. A
compiler rejection is a compiler result, not performance evidence.

## Failure-mode check

1. **Threadgroup MPP operands compile but reduce occupancy enough to regress.**
   Direct-grouped controls attribute scheduling separately, and P8/P16/P32 plus
   legal SIMDgroup counts expose the occupancy frontier before rejection.
2. **A group crosses a K-part boundary and silently reads the wrong activation
   slice.** Group-count divisibility is validated before dispatch; source and
   correctness tests assert one part per threadgroup and bit-exact MPP logits.
3. **Threadgroup storage or register pressure exceeds a device limit.** Resource
   failures are captured per arm. Smaller P spans and SIMDgroup counts remain
   independently testable, so one illegal topology cannot abort the frontier.

## Rollout

Add the benchmark and kernel behind an internal tiling selector. Do not change
the #59 default from N16/P8 until a refined positive interval and correctness
gate pass. Promote only the winning schedule; retain the complete screen as
Issue 59 evidence.
