# Native MTP On MLX

MTPLX explores the built-in MTP heads in Qwen3-Next models on Apple Silicon.

The core idea is straightforward: use the model's own MTP head to propose tokens, then use exact speculative sampling to accept or reject them against the target distribution. This is different from greedy prefix-match systems and different from external-draft-model systems.

This note records the original v0.1 research boundary, when the cold path was stronger than sustained
no-fan long-context throughput. It is historical context, not the current product capability matrix;
see the repository README, [Profiles](../profiles.md), and [Architecture support](../architectures.md).
