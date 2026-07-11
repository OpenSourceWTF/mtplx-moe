# Engineering Review: Serving a Very Large MoE LLM on Apple Silicon with NVMe-Backed Routed Experts

## 1. Context and Design Summary

We are reviewing a production design for serving a very large mixture-of-experts (MoE) language model on a single Apple Silicon workstation. The machine uses unified memory (CPU/GPU/shared RAM on the same bus) and a local NVMe SSD. The router, attention layers, embeddings, normalization layers, and any shared experts are resident in unified memory at all times. Routed expert weights are too large to fit in RAM and are stored as affine 4-bit tensors on NVMe. At each sparse layer, the router selects eight experts; if an expert is not already present, its weights are loaded from disk into a fixed, user-bounded memory bank. Decode requests use a frequency-decayed hot cache; prompt prefill uses transient slots so a large prefill cannot evict the decode working set.

This review covers execution flow, concurrency, memory accounting, I/O integrity, cache policy, failure handling, observability, correctness, the safetensor vs. sidecar tradeoff, failure modes, and rollout.

## 2. Execution Sequence

For a decode step:
