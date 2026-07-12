You are reviewing a production design for serving a very large mixture-of-experts
language model on an Apple Silicon workstation. The router, attention layers,
embeddings, normalization layers, and shared experts remain in unified memory.
Routed expert weights are stored as affine 4-bit tensors on a local NVMe SSD. At
each sparse layer, the router selects eight experts; selected experts are loaded
into a fixed, user-bounded memory bank on a miss. Decode requests populate a
frequency-decayed hot cache, while prompt prefill uses transient slots so a single
large prompt cannot evict the long-lived decode working set.

Write a detailed engineering review of this design for a team preparing a real
deployment. Cover the execution sequence, concurrency and slot pinning, memory
accounting, positional I/O and integrity checks, cache admission and eviction,
failure handling, observability, and correctness testing. Explain the important
tradeoffs between reading directly from component-major safetensor shards and
using an optional expert-major sidecar. Identify at least five concrete failure
modes and give a practical mitigation for each. End with a staged rollout plan
and explicit go/no-go criteria. Use clear sections and enough technical detail
that an engineer could turn the review into implementation tasks. Aim for at
least 1,000 words; do not stop after a short summary.
