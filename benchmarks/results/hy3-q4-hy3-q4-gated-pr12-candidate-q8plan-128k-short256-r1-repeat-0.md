# Engineering Review: Serving a Very Large MoE LLM on Apple Silicon with NVMe-Backed Routed Experts

## 1. System Overview and Execution Sequence

The proposed design serves a very large mixture-of-experts (MoE) language model on an Apple Silicon workstation. The router, attention layers, embeddings, normalization layers, and shared experts are resident in unified memory (CPU/GPU shared DRAM). Routed expert weights are stored as affine 4-bit tensors on a local NVMe SSD. At each sparse layer, the router selects eight experts; on a cache miss, selected experts are loaded into a fixed, user-bounded memory bank. Decode requests populate a frequency-decayed hot cache, while prompt prefill uses transient slots so a single large prompt cannot evict the long-lived decode working set.

The steady-state execution sequence for a decode step is:

1. Token embedding lookup in unified memory.
2. For each transformer layer:
   a. Normalization and attention in unified memory.
   b. Router computes expert assignments (top-8 experts per token or per batch).
   c. For each selected expert: check hot cache; on miss, issue async NVMe read into a pinned slot in the bounded memory bank.
  
