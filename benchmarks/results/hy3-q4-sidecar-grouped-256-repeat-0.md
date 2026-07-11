# Engineering Review: Serving a Very Large MoE LLM on Apple Silicon with NVMe-Backed Routed Experts

## 1. System Overview and Execution Sequence

The proposed design serves a very large mixture-of-experts (MoE) language model on an Apple Silicon workstation. The router, attention layers, embeddings, normalization layers, and shared experts are resident in unified memory (DRAM mapped into the Apple Silicon memory controller). Routed expert weights are too large to fit entirely in unified memory and are stored as affine 4-bit tensors on a local NVMe SSD. At each sparse layer, the router selects eight experts. On a cache miss, selected experts are loaded from NVMe into a fixed, user-bounded memory bank (a reserved region of unified memory). Decode requests populate a frequency-decayed hot cache; prompt prefill uses transient slots that are isolated from the long-lived decode working set.

A single forward pass executes as follows:

1. Token embeddings and positional encodings are read from unified memory.
2. For each transformer block:
   a. Attention, normalization, and shared expert compute execute from unified memory.
   b. The router scores tokens and selects the top-eight experts for that layer.
   c. The scheduler checks the
