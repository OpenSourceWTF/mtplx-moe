# Hy3 Q2 fused K-row expert execution

## Status and ordering

This design extends Issue 51 after the sustained Next-K matrix. Execution is
autonomous and ordered so every mechanism remains attributable:

1. finish the stock Hy3-Q2 K=0...6 matrix at 1,024/2,048/4,096 input tokens
   and exactly 1,028 output tokens;
2. resolve the existing batched-versus-serial parity drift;
3. test Q2 NAX by itself, including K=0;
4. test fused K-row Q2 execution by itself;
5. test fused K-row Q2 plus NAX only if both individual operator arms are
   correct and measurable.

The 128-output Issue 51 qualification lane remains unchanged. The 1,028-output
lane is sustained experimental evidence, not a replacement qualification
contract.

## Scope

The target is the streamed Hy3 expert-only affine-Q2 artifact:

- component-bank expert slots;
- 4,096-wide BF16 inputs and outputs;
- 1,536-wide expert hidden state;
- affine Q2 weights, group size 64, BF16 scales and biases;
- router top-k 8;
- target verification rows `R = K + 1`, for K=0...6;
- assignment rows `M = R * 8`, split only when the bounded route-wave and slot
  ownership contracts require it.

The work preserves the router as authoritative but may replace its arithmetic
with an exact fused FP32 implementation before optimizing expert arithmetic.
It does not predict routes, admit speculative slots, alter the artifact, weaken
cache ownership, change verifier acceptance, or make NAX a hardware requirement.

## Existing boundary

`HotExpertSwitchGLU` currently expands each routed target row into top-k expert
assignments and executes a component-bank wave with:

```text
gather_qmm(gate) + gather_qmm(up) + SwiGLU + gather_qmm(down)
```

Rows within one `gather_qmm` may select different component-bank slots. Slot
bindings remain pinned until the lazy graph has consumed their generations.
Wave outputs are concatenated and restored to router assignment order before
Hy3 applies routing scores and the top-k reduction.

The existing NAX patch does not cover this path: it dispatches 4/6/8-bit
`nn.QuantizedLinear`, while Hy3 streamed experts call 2-bit `mx.gather_qmm`.

Immediately before that boundary, `Router` already evaluates all `K+1` target
rows together, but leaves FP32 projection, sigmoid, expert-bias addition,
top-k, score gather, normalization, and scaling as separate graph operations.
The selected IDs must be known before the CPU can resolve and pin streamed
expert slots, so router and expert work are necessarily two stages.

## Considered approaches

### A. One full-MLP kernel per down-output tile

Compute gate, up, SwiGLU, and down in one launch. This removes the most
dispatches but either stores all 1,536 intermediate values per assignment in
threadgroup memory or recomputes gate/up for every 4,096-output down tile.
The latter repeats most expert weight reads; the former does not scale to a
large mixed-assignment tile. This approach is rejected.

### B. Two fused stages separated by authoritative routing

Stage R fuses the batched FP32 router projection and deterministic top-k score
pipeline for all `K+1` rows. It emits `(K+1,8)` expert IDs and FP32 routing
weights. The existing runtime then resolves slots and starts required reads.
Stage E performs the parallel Q2 expert work for `8*(K+1)` assignments.

This is the recommended system boundary. There can be no router-to-expert
single kernel because streamed weight residency is decided between the stages.

### C. Two-stage assignment-indexed expert fusion

Use one gather-aware kernel for gate+up+exact-SwiGLU and one gather-aware down
kernel. Both accept all assignment rows in the current component-bank wave and
one slot index per row. This removes a projection dispatch and the separate
SwiGLU graph while retaining a single reusable `[M, 1536]` intermediate.
It supports singleton and mixed-expert rows without Python grouping.

This is the recommended fused K-row arm.

### D. Group by expert and use a shared-RHS NAX tile

Group assignment rows by `(component bank, slot generation)`, pad each group to
the NAX tile, and reuse one expert matrix across its rows. This matches
MetalPerformancePrimitives tensor-operator geometry, but repeated-expert group
sizes are often small and grouping/scatter overhead is real. It is therefore
the NAX-only arm first, then the implementation substrate for the combined arm
only if it wins.

## Architecture

### Common eligibility contract

Every custom operator is fail-closed. It is eligible only for:

- Metal available on an Apple GPU;
- BF16 or FP16 activation matching BF16/FP16 affine metadata;
- bits=2, group size=64, affine mode;
- exact Hy3 dimensions 4096 -> 1536 -> 4096;
- component-bank buffers with one valid slot index per assignment;
- assignment count in `1..56`;
- complete pinned bindings for the lifetime of the lazy graph;
- decode phase only.

Ineligible calls use the unchanged `mx.gather_qmm` path. Prefill remains stock.
Kernel selection is fixed before an expert wave starts and never changes while
that wave is in flight.

### Arm R: fused FP32 router independently

`hy3_router_fp32.py` accepts `[K+1,4096]` activation rows plus the resident
router weight and expert bias. It performs the FP32 projection, sigmoid,
selection-bias addition, deterministic top-8 selection, original-score gather,
optional route normalization, and scaling. Its public result is exactly the
same pair used today: `(indices, routing_weights)` with shapes `(K+1,8)`.

The implementation must preserve the existing top-k membership and order,
including ties, and the FP32 normalization reduction order. It is benchmarked
alone with stock experts before it participates in any Stage-E combination.

### Arm N: Q2 NAX independently

`q2_nax.py` provides a Q2 MPP/NAX matmul accepting one shared RHS and `1..16`
rows, padding inactive rows to the tensor tile. Q2 unpacking consumes one
32-bit word per 16 weights and applies the existing affine BF16 scale/bias
contract.

`HotExpertSwitchGLU` groups a wave by pinned `(bank, bank_index, generation)`
and invokes three NAX matmuls with the existing standalone SwiGLU between them.
It restores original assignment order with the existing deterministic gather.
K=0 is intentionally eligible so its singleton cost is measured, not assumed.

NAX availability gates only the NAX implementation. A plain-SIMD diagnostic
must not be reported as a NAX result.

### Arm F: fused K-row execution independently

`q2_fused_expert.py` exposes:

```python
q2_gate_up_swiglu_gather(
    x, bank_arrays, slot_indices, *, group_size=64
) -> hidden

q2_down_gather(
    hidden, bank_arrays, slot_indices, *, group_size=64
) -> assignment_outputs
```

The first kernel reads each activation tile once, dequantizes the selected
gate/up rows, accumulates in the stock-compatible order, rounds gate and up to
the activation dtype, and applies the MLX-compatible SwiGLU formula. The second
kernel applies the selected down matrix. Both launch over all assignment rows
in the current bank wave, so K parallelism is an explicit grid dimension.

This arm keeps the current output-position restoration and Hy3 weighted top-k
combine unchanged. Weighted reduction fusion is a later optimization only
after Arm F passes; it is not needed to satisfy the K-row kernel contract.

### Arm FN: fused K-row plus NAX

The combined arm groups rows by pinned expert as Arm N does. Its first NAX
kernel computes gate and up from a shared activation tile and writes exact
SwiGLU output; its second NAX kernel computes down. Grouping, scatter, slot
ownership, and fallback match Arm N exactly, isolating the value of fusion.

The combination is compared against stock, Arm N, and Arm F. It is retained
only if it adds value over the faster individual arm.

### Two-stage combinations

After Arm R and the Stage-E arms complete independently, measure R+F and R+FN.
R+F is the complete non-NAX two-stage fused pipeline. R+FN is the final
router-fused, expert-fused, NAX-powered pipeline. Each combination must beat
its faster immediate constituent, not only the all-stock control.

## Exactness and ownership contracts

Operator tests compare every arm with stock Q2 at M=1...7 per shared expert and
mixed assignment counts 8,16,...,56. They cover repeated and unique expert
IDs, multiple banks, nontrivial output permutations, and every route-wave
split.

Required invariants:

- finite output and bounded elementwise error against stock;
- deterministic repeated output for the same shape and slot generation;
- exact downstream router IDs and order at every layer;
- exact verifier accepts, corrections, bonuses, and emitted tokens;
- exact target and committed-MTP cache offsets after accept and rejection;
- identical expert requests, slot generations, cache accounting, and final
  slot health;
- no pin release before the final custom-kernel consumer completes;
- no unsupported hardware or shape silently enters a custom lane.

Router tests additionally cover K=0...6, exact expert-ID membership and order,
exact gathered/scaled FP32 weights, tie behavior, route normalization on/off,
and downstream route identity at every layer.

The known 2,048-token stock serial-versus-batched divergence must be classified
and fixed before a custom arm can be promoted. A custom arm may not use that
pre-existing drift to relax its comparison.

## Measurement plan

### Operator gate

Use real component-bank records and retained route geometries. For each arm and
K=0...6 report:

- gate/up/SwiGLU, down, grouping, scatter, and total elapsed time;
- FP32 router projection, selection, normalization, and total Stage-R time;
- assignment count, unique experts, repeated-expert fraction, and group-size
  histogram;
- effective packed-weight bandwidth;
- median and paired distribution versus immediate stock control;
- maximum and percentile elementwise error.

Arm N must be tested first. Arm R and Arm F follow independently. Arm FN is
forbidden until both Stage-E individual arms complete their gates; R+F and
R+FN are forbidden until Arm R also passes.

### End-to-end gate

Run the winning operator configurations at 1,024/2,048/4,096 input tokens with
1,028 output tokens and the matrix-selected K values. Report ingest, prefill,
decode, cache hit rate, MTP acceptance, verify calls, mean/peak readers, reader
saturation, memory, and token parity.

An arm advances only with a positive paired decode interval against its
immediate control and no material regression in prefill, memory, cache hit
rate, correctness, or final health. Arm FN must beat the faster of Arm N and
Arm F, R+F must beat both R and F, and R+FN must beat both R and FN. A
combination does not pass merely by beating stock.

## Error handling and rollout

Kernel build or dispatch failure marks the candidate row failed; it does not
fall back after partial custom execution. Eligibility misses use the stock path
before dispatch and increment explicit eligibility/fallback counters.

All arms are experimental, environment-gated, and off by default. The issue
artifact records hardware family, OS, MLX version, kernel source hash, selected
arm, eligible calls, fallback calls, and failure details. Promotion requires
the 128-output qualification lane after sustained evidence passes.

## Failure-mode review

1. **Critical: padding makes singleton NAX slower and masks any K benefit.**
   K=0 and every observed group-size bucket are measured separately. NAX is
   rejected or restricted to profitable group sizes; Arm F remains independent.
2. **Critical: custom accumulation changes a downstream router choice.**
   Layer-by-layer router parity is a hard gate, and the 2,048 stock shape drift
   is resolved first. No tolerance is applied to route or token identity.
3. **Critical: a lazy kernel reads a component-bank slot after reuse.**
   Custom outputs remain roots of the existing completion fence, with tests for
   delayed evaluation and generation replacement. Any fence fallback or stale
   generation fails the candidate.
4. **Minor: mixed banks require multiple launches and dilute fusion.**
   The existing wave/bank partition is retained and charged in real-geometry
   benchmarks. Cross-bank fusion is a non-goal because it would broaden slot
   ownership and memory-layout risk.
5. **Critical: fused top-k changes membership or ordering on a close/tied
   router score.** The stock router output is compared exactly for every row
   and layer; any membership, order, or FP32 weight difference rejects Arm R.
