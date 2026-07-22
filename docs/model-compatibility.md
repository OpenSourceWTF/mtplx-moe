# Model Compatibility

MTPLX separates detection from support.

| Tier | Meaning | Default behavior |
|---|---|---|
| Verified | `mtplx_runtime.json` exists and matches the expected contract | Run |
| Architecture-compatible, unverified | Qwen3-Next MTP markers exist, but no MTPLX contract | Refuse unless explicitly forced |
| AR-only | An exact architecture-specific AR loader is installed, but the checkpoint has no MTP head | Run only with target-only AR selected |
| Incompatible architecture | MTP markers exist for an unsupported architecture | Exit with roadmap pointer |
| No MTP | No MTP head detected | Exit with a clear message |

The AR-only tier is narrow by design. It currently recognizes the exact 4-bit
geometry and storage map of `pipenetwork/Laguna-S-2.1-MLX-4bit` at revision
`5544297f819d50330bc3616dd15cbc7edb598b2f`. Local cache admission also
requires the pinned source marker, all 13 shards at their reviewed sizes, the
index, tokenizer, generation config, and Poolside chat template. Other Laguna
variants remain blocked until they have their own construction-time validation
and runtime evidence.
