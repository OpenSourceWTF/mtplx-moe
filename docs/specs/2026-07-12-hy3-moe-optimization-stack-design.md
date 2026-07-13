# Hy3 MoE Optimization Stack Design

**Status:** Approved through issues #29, #30, and #31 and the instruction to build the work as stacked PRs.

## Goal

Improve Hy3 affine-Q4 streamed-MoE inference without assuming that a monolithic fused kernel is the dominant lever. Preserve the resident router as the authoritative source of expert IDs, preserve the current total memory ceiling, and attribute every retained performance change to one measured mechanism.

## Stack

1. `experiment/hy3-cache-scheduling` on `experiment/moe-pr13-pr14-stack`
   - Enable the existing global cache policy with component-bank execution.
   - Expose held-out capacity evidence and measure B=1/2/4/8.
2. `experiment/hy3-record-native-exec` on `experiment/hy3-cache-scheduling`
   - Consume the existing v1 record layout directly.
   - Evaluate a two-stage sparse primitive and router-weighted down reduction.
3. `experiment/hy3-artifact-speculative` on `experiment/hy3-record-native-exec`
   - Evaluate GPU-oriented packing, KV/cache budget exchange, hint-only prefetch, and a separately labeled lower-bit cold tier.

Each branch remains independently testable against its immediate base. A negative experiment stays opt-in and documented; it is not silently promoted.

## Invariants

- Hy3 layers 1-79 route top-8 of 192 experts and add one resident shared expert.
- Router logits are promoted to FP32, selection uses sigmoid plus correction bias, and mixture weights use the uncorrected normalized sigmoid scores scaled by 2.826.
- With `enable_moe_fp32_combine=false`, routed weighting and reduction remain in activation dtype and preserve the accepted reduction order.
- The v1 record remains 10,616,832 bytes and contains the nine ordered affine-Q4 components.
- Persistent expert bytes, transient bytes, KV bytes, runtime reserve, and total memory are reported separately.
- Prediction may initiate prefetch but never changes, skips, or overrides an authoritative route.

## Performance contract

Every mechanism is measured against its immediate predecessor with the same artifact, prompt, sampler, memory limit, and machine state. Long-running runs require at least two measurements within 5%; shorter runs require at least three. Promotion requires a repeated 5% improvement on the declared lane, no material regression on required lanes, deterministic parity where exactness is expected, and bounded memory.

## Failure modes

1. **Global component banks serialize unrelated layers.** The global policy already uses one transaction lock; the implementation must show that fewer physical reads outweigh lock and larger-bank gather costs.
2. **Weighted reduction changes logits.** Kernel timing is invalid until layer output, logits, and deterministic tokens match the deployed activation-dtype combine contract.
3. **A bundled experimental result is uninterpretable.** Layout, KV, prefetch, and precision arms must remain separate and may not be promoted from a combined benchmark.

## Rollback

All new paths are opt-in until they pass their promotion gate. Existing `cache_scope="layer"`, component-bank execution, v1 manifests, authoritative routing, and Q4 artifacts remain available without migration.
