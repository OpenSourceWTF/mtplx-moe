# Turbo verify kernels (experimental, opt-in)

Status: experimental, off by default, pending exactness-policy review.

```bash
MTPLX_NAX_VERIFY=1 mtplx serve ...
```

When enabled at model load, 4-bit affine projections route through
verify-specialized Metal kernels (ported from bstnxbt/dflash-mlx, Apache-2.0)
for batches of 4..16 rows — the shape of native-MTP speculative verification.
Single-token decode, drafting, and prompt prefill are untouched and remain
bit-identical to stock MLX.

- 4-row K-split kernel: any Apple Silicon.
- 16-row tile via Metal 4 tensor ops: Apple M5-class GPUs (G17) on
  macOS 26.2+, used for depths above 3.

Measured on M5 Max / Qwen3.6-27B Optimized-Speed, reasoning on, 2026-06-12:
1k-token decode 48.3 -> 65.5 tok/s mean over four matched seeds; official
flappy envelope 55.7 -> 64.5; live server completion 55.0 -> 66.7; 10k-token
generation 59.5 tok/s sustained. 6-bit models (9B Optimized-Speed) are not
eligible. MoE (35B-A3B) routes only dense projections: ~neutral.

Numerics: not bit-exact versus stock kernels (different accumulation order).
Argmax-identical on all probed positions; at the product sampler
(temp 0.6 / top_p 0.95 / top_k 20) the live D3 verify path measured total
variation 0.0 and sample agreement 1.0 on every probed cell
(`scripts/nax_distribution_gate_expanded` in the research workspace is the
gate). Speculative acceptance remains mathematically exact with respect to
the verify-computed target distribution. Do not use for bit-exactness QA
(`mtplx qa exactness` reference runs, batch-equivalence gates).
