# Issue #31 C2: shared gate/up 2-to-1 packing

Decision: **retain C2**. The operator result advanced to an isolated production
implementation, and the subsequent full-decode campaign confirmed the gain.

## Exact target

- Runtime predecessor: `1faa0f960fc8b73a2eb828a563f352f101d2cf6f`
- Artifact: `pipenetwork/Hy3-4bit@160619d3f96c8470350b6dac0ef033a8381551e3`
- Layer: `model.layers.1.mlp.shared_mlp`
- Input: B1 `[1, 1, 4096]` BF16
- Gate/up: two affine-Q4 4096-to-1536 projections, group size 64

The control evaluates both existing `QuantizedLinear` calls through SwiGLU. The candidate concatenates packed weights/scales/biases, runs one 4096-to-3072 QMM, splits gate/up, and evaluates the same SwiGLU. Timing includes Python graph construction, split, dispatch, activation, and synchronization.

## Correctness, layout, and result

The evaluated SwiGLU output was bit-exact. The original and packed layouts are each 7,077,888 bytes. Packing took 3,729.292 us and temporarily required 14,155,776 bytes; production must release the two originals after each layer is packed.

The run used 32 warmups, 16 alternating pairs of 256 calls, and 20,000 bootstrap resamples.

| Metric | Two-QMM control | Packed one-QMM |
|---|---:|---:|
| Mean | 236.881592 us | 230.084463 us |
| Median | 233.343262 us | 231.172119 us |
| Range | 223.330730-283.612957 us | 216.360027-238.541344 us |

Mean paired speedup was **1.030061x** (+3.006%); median was **1.015024x**; bootstrap mean 95% CI was **[1.005981, 1.066515]**. One late control sample (283.613 us) increases the mean, so the full paired distribution and median remain part of the evidence.

At the mean difference of 6.797130 us across 79 sparse shared MLPs, the optimistic whole-model saving is about 0.537 ms/token, roughly +0.34% at 6.244 tok/s. This estimate was context, not a gate. The later full-decode result measured a +2.807% paired gain with 95% CI [0.729%, 4.796%]; see `hy3-shared-gate-up-decode-issue31-20260713.md`.

## Commands and raw evidence

```text
python -m pytest tests/test_hy3_shared_gate_up_probe.py -q
python benchmarks/hy3_shared_gate_up.py --warmup 32 --iterations 256 --rounds 16 --bootstrap-resamples 20000 --output-json /tmp/mtplx-issue31-c2-shared-gate-up.json
```

The focused contract passed 2 tests; Ruff passed. Qwen restoration and exclusive-lane release were verified.

Raw result: `/tmp/mtplx-issue31-c2-shared-gate-up.json`.
