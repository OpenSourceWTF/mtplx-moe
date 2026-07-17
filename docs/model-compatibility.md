# Model Compatibility

MTPLX separates detection from support.

| Tier | Meaning | Default behavior |
|---|---|---|
| Verified | `mtplx_runtime.json` exists and matches the expected contract | Run |
| Architecture-compatible, unverified | Qwen3-Next MTP markers exist, but no MTPLX contract | Refuse unless explicitly forced |
| Incompatible architecture | MTP markers exist for an unsupported architecture | Exit with roadmap pointer |
| No MTP | No MTP head detected | Exit with a clear message |

Product runs require a verified artifact contract for a backend the installed architecture registry
marks runnable. Use `mtplx model architectures --json` for the current family matrix and `mtplx
inspect MODEL --json` for the artifact-level decision.
