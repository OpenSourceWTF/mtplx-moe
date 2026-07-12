"""Runtime sampler helpers for sparse top-k speculative sampling."""

from __future__ import annotations

from collections.abc import Mapping

import mlx.core as mx
import numpy as np

from .sampling import PENALTY_MAX, PENALTY_MIN, SamplerConfig, SparseDistribution


def apply_penalties_mlx(
    logits: mx.array,
    token_counts: Mapping[int, int] | None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    penalty_overlay: Mapping[int, float] | None = None,
) -> mx.array:
    """On-device MLX twin of ``sampling.apply_penalties`` for one logit row.

    Subtracts the additive OpenAI/vLLM penalty on the RAW logits — before any
    temperature/top-k/top-p — via a sparse scatter over only the seen tokens
    (``logits.at[ids].add(-deltas)``): O(unique_seen), no dense vocab-sized
    allocation and no host round-trip beyond the small (ids, deltas) transfer.
    Penalties clamp to [-2, 2]. ``penalty_overlay`` is the Loop Guard's sparse
    token->subtraction map (not clamped; the guard caps it). Returns the same
    array unchanged (no-op) when nothing is active, preserving exactness.

    Expects a 1-D logit row (``logits.shape == (vocab,)``); batched callers apply
    it per row (the MTP draft block is a handful of positions, not a hot loop).
    """
    presence = min(max(float(presence_penalty), PENALTY_MIN), PENALTY_MAX)
    frequency = min(max(float(frequency_penalty), PENALTY_MIN), PENALTY_MAX)
    counts_active = bool(token_counts) and (presence != 0.0 or frequency != 0.0)
    overlay_active = bool(penalty_overlay)
    if not counts_active and not overlay_active:
        return logits
    if counts_active:
        ids = np.fromiter(token_counts.keys(), dtype=np.int64, count=len(token_counts))
        counts = np.fromiter(token_counts.values(), dtype=np.float64, count=len(token_counts))
        deltas = (frequency * counts + presence * (counts > 0.0)).astype(np.float32)
        logits = logits.at[mx.array(ids)].add(mx.array(-deltas))
    if overlay_active:
        overlay_ids = np.fromiter(
            penalty_overlay.keys(), dtype=np.int64, count=len(penalty_overlay)
        )
        overlay_vals = np.fromiter(
            penalty_overlay.values(), dtype=np.float64, count=len(penalty_overlay)
        ).astype(np.float32)
        logits = logits.at[mx.array(overlay_ids)].add(mx.array(-overlay_vals))
    return logits


class BatchedSparseDistributions:
    def __init__(
        self,
        token_ids: np.ndarray,
        probs: np.ndarray,
        *,
        vocab_size: int,
    ) -> None:
        token_ids = np.asarray(token_ids, dtype=np.int64)
        probs = np.asarray(probs, dtype=np.float64)
        if token_ids.ndim != 2 or probs.ndim != 2:
            raise ValueError("BatchedSparseDistributions expects 2D arrays")
        if token_ids.shape != probs.shape:
            raise ValueError("token_ids/probs shape mismatch")
        row_sums = probs.sum(axis=1)
        if np.any(row_sums <= 0) or not np.all(np.isfinite(row_sums)):
            raise ValueError("each sparse distribution row needs positive mass")
        self.token_ids = token_ids
        self.probs = probs / row_sums[:, None]
        self.vocab_size = int(vocab_size)

    def probability(self, row: int, token_id: int) -> float:
        hits = np.nonzero(self.token_ids[int(row)] == int(token_id))[0]
        if hits.size == 0:
            return 0.0
        return float(self.probs[int(row), int(hits[0])])

    def to_distribution(self, row: int) -> SparseDistribution:
        row = int(row)
        keep = self.probs[row] > 0
        return SparseDistribution(
            self.token_ids[row, keep],
            self.probs[row, keep],
            self.vocab_size,
        )

    def sample(self, row: int, rng: np.random.Generator) -> int:
        row = int(row)
        keep = self.probs[row] > 0
        return int(rng.choice(self.token_ids[row, keep], p=self.probs[row, keep]))


def sparse_distribution_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
    *,
    token_counts: Mapping[int, int] | None = None,
    penalty_overlay: Mapping[int, float] | None = None,
) -> SparseDistribution | None:
    """Return an exact sparse distribution for top-p then top-k sampling.

    The Qwen coding sampler uses `top_k=20`, so the final support can never be
    larger than 20 tokens. We still compute the full-vocab logsumexp on MLX so
    top-p decisions use true full-distribution probability mass, then move only
    the small support to NumPy for deterministic speculative correction.

    ``token_counts`` (completion tokens seen so far, scoped by the caller) applies
    the additive presence/frequency penalty to the raw logits BEFORE the
    temperature divide — a no-op (same array) when penalties are 0.
    ``penalty_overlay`` is the Loop Guard's sparse steering map, applied at the
    same raw-logit stage.
    """

    if config.temperature <= 0 or config.top_k <= 0:
        return None

    row = apply_penalties_mlx(
        logits.reshape(-1),
        token_counts,
        config.presence_penalty,
        config.frequency_penalty,
        penalty_overlay=penalty_overlay,
    )
    flat = row.astype(mx.float32) / float(config.temperature)
    vocab_size = int(flat.shape[-1])
    k = min(int(config.top_k), vocab_size)
    if k <= 0:
        return None

    top_idx = mx.argpartition(-flat, kth=k - 1, axis=-1)[:k]
    top_vals = flat[top_idx]
    order = mx.argsort(-top_vals, axis=-1)
    top_idx = top_idx[order]
    top_vals = top_vals[order]

    if config.top_p >= 1.0:
        top_probs_full = mx.softmax(top_vals, axis=-1)
    else:
        log_total = mx.logsumexp(flat, axis=-1)
        top_probs_full = mx.exp(top_vals - log_total)
    mx.eval(top_idx, top_probs_full)

    token_ids = np.asarray(top_idx, dtype=np.int64).reshape(-1)
    probs_full = np.asarray(top_probs_full, dtype=np.float64).reshape(-1)

    if 0 < config.top_p < 1.0:
        cumulative_before = np.concatenate(([0.0], np.cumsum(probs_full[:-1])))
        keep = cumulative_before < float(config.top_p)
        if keep.size:
            keep[0] = True
    else:
        keep = np.ones_like(probs_full, dtype=bool)

    token_ids = token_ids[keep]
    probs = probs_full[keep]
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        token_ids = token_ids[:1]
        probs = np.array([1.0], dtype=np.float64)

    return SparseDistribution(token_ids=token_ids, probs=probs, vocab_size=vocab_size)


def sparse_distributions_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
) -> list[SparseDistribution] | None:
    """Return exact sparse distributions for a batch of logit rows.

    This is the batched equivalent of ``sparse_distribution_from_mlx_logits``.
    It keeps the same top-k/top-p semantics but shares the MLX materialization
    boundary across rows.
    """

    if config.temperature <= 0 or config.top_k <= 0:
        return None

    rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32) / float(config.temperature)
    vocab_size = int(rows.shape[-1])
    k = min(int(config.top_k), vocab_size)
    if k <= 0:
        return None

    top_idx = mx.argpartition(-rows, kth=k - 1, axis=-1)[:, :k]
    top_vals = mx.take_along_axis(rows, top_idx, axis=-1)
    order = mx.argsort(-top_vals, axis=-1)
    top_idx = mx.take_along_axis(top_idx, order, axis=-1)
    top_vals = mx.take_along_axis(top_vals, order, axis=-1)

    if config.top_p >= 1.0:
        top_probs_full = mx.softmax(top_vals, axis=-1)
    else:
        log_total = mx.logsumexp(rows, axis=-1)
        top_probs_full = mx.exp(top_vals - log_total[:, None])
    mx.eval(top_idx, top_probs_full)

    token_rows = np.asarray(top_idx, dtype=np.int64)
    prob_rows = np.asarray(top_probs_full, dtype=np.float64)
    distributions: list[SparseDistribution] = []

    for token_ids, probs_full in zip(token_rows, prob_rows, strict=True):
        if 0 < config.top_p < 1.0:
            cumulative_before = np.concatenate(([0.0], np.cumsum(probs_full[:-1])))
            keep = cumulative_before < float(config.top_p)
            if keep.size:
                keep[0] = True
        else:
            keep = np.ones_like(probs_full, dtype=bool)

        kept_ids = token_ids[keep]
        probs = probs_full[keep]
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            kept_ids = kept_ids[:1]
            probs = np.array([1.0], dtype=np.float64)

        distributions.append(
            SparseDistribution(
                token_ids=kept_ids,
                probs=probs,
                vocab_size=vocab_size,
            )
        )

    return distributions


def batched_sparse_distributions_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
) -> BatchedSparseDistributions | None:
    """Return batched sparse distributions without per-row Python objects."""

    if config.temperature <= 0 or config.top_k <= 0:
        return None

    rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32) / float(config.temperature)
    vocab_size = int(rows.shape[-1])
    k = min(int(config.top_k), vocab_size)
    if k <= 0:
        return None

    top_idx = mx.argpartition(-rows, kth=k - 1, axis=-1)[:, :k]
    top_vals = mx.take_along_axis(rows, top_idx, axis=-1)
    order = mx.argsort(-top_vals, axis=-1)
    top_idx = mx.take_along_axis(top_idx, order, axis=-1)
    top_vals = mx.take_along_axis(top_vals, order, axis=-1)

    if config.top_p >= 1.0:
        top_probs_full = mx.softmax(top_vals, axis=-1)
    else:
        log_total = mx.logsumexp(rows, axis=-1)
        top_probs_full = mx.exp(top_vals - log_total[:, None])
    mx.eval(top_idx, top_probs_full)

    token_rows = np.asarray(top_idx, dtype=np.int64)
    prob_rows = np.asarray(top_probs_full, dtype=np.float64)

    if 0 < config.top_p < 1.0:
        cumulative_before = np.concatenate(
            (
                np.zeros((prob_rows.shape[0], 1), dtype=np.float64),
                np.cumsum(prob_rows[:, :-1], axis=1),
            ),
            axis=1,
        )
        keep = cumulative_before < float(config.top_p)
        if keep.size:
            keep[:, 0] = True
        prob_rows = np.where(keep, prob_rows, 0.0)

    row_sums = prob_rows.sum(axis=1)
    bad = (~np.isfinite(row_sums)) | (row_sums <= 0)
    if np.any(bad):
        prob_rows[bad, :] = 0.0
        prob_rows[bad, 0] = 1.0

    return BatchedSparseDistributions(token_rows, prob_rows, vocab_size=vocab_size)


def sample_token_ids_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
) -> mx.array | None:
    """Sample token ids on-device with the same top-k/top-p semantics.

    This is for exact target-prefix verification paths that need sampled target
    ids, not p/q residual distributions. It keeps the small-support sampling on
    MLX and returns one token id per input row.
    """

    if config.temperature <= 0:
        return mx.argmax(logits, axis=-1)

    rows = logits.astype(mx.float32) / float(config.temperature)
    vocab_size = int(rows.shape[-1])
    if config.top_k <= 0:
        if 0 < config.top_p < 1.0:
            return None
        return mx.random.categorical(rows)

    k = min(int(config.top_k), vocab_size)
    if k <= 0:
        return None

    top_idx = mx.argpartition(-rows, kth=k - 1, axis=-1)[..., :k]
    top_vals = mx.take_along_axis(rows, top_idx, axis=-1)
    order = mx.argsort(-top_vals, axis=-1)
    top_idx = mx.take_along_axis(top_idx, order, axis=-1)
    top_vals = mx.take_along_axis(top_vals, order, axis=-1)

    if 0 < config.top_p < 1.0:
        log_total = mx.logsumexp(rows, axis=-1, keepdims=True)
        top_probs = mx.exp(top_vals - log_total)
        higher_mass = mx.cumsum(top_probs, axis=-1) - top_probs
        first = mx.arange(k) == 0
        keep = (higher_mass < float(config.top_p)) | first
        top_vals = mx.where(keep, top_vals, -float("inf"))

    sampled_offsets = mx.random.categorical(top_vals)
    return mx.take_along_axis(top_idx, sampled_offsets[..., None], axis=-1)[..., 0]
