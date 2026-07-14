# Issue #31 C2 shared gate/up full-decode result

Decision: advance only C2, the construction-time packed shared-MLP gate/up
layout. The retained comparison is the control and C2 subset of the 32-run
packed-projection campaign; no C1 implementation is part of this change.

## Contract

- Model: `pipenetwork/Hy3-4bit@160619d3f96c8470350b6dac0ef033a8381551e3`
- Prompt: 313 tokens
- Completion: 256 tokens, 255 timed decode steps
- Batch: B1 autoregressive decode; MTP and thinking disabled
- Runtime: global LRU component banks
- Telemetry: window and resource telemetry disabled
- Repeats: 8 order-balanced control/C2 pairs
- Candidate layout: 79 packed sparse-layer shared MLPs

## Decode result

| Layout | Mean TPS | Median TPS | Mean paired ratio | 95% CI |
|---|---:|---:|---:|---:|
| Control | 6.087938 | 6.133327 | 1.000000x | - |
| C2 shared gate/up | 6.253842 | 6.286900 | 1.028066x (+2.807%) | [1.007289, 1.047961] |

C2 was faster in 6 of 8 pairs, and the paired bootstrap interval excludes
parity. All 16 retained control/C2 runs had identical generated tokens and
text, route traces, deterministic cache and I/O counters, configuration and
manifest identity, and clean final route/pin/slot health. Each candidate run
verified that the loaded model contained 79 packed shared MLP modules.

The optimization is selected once at model construction with
`MTPLX_FUSE_HY3_SHARED_GATE_UP_PROJECTIONS=1`; there is no per-token feature
conditional. Model sanitization replaces the original gate/up arrays with the
evaluated packed arrays rather than retaining a second full-model copy.

Machine-readable evidence is in
`benchmarks/results/hy3-shared-gate-up-decode-issue31-20260713.json`.
