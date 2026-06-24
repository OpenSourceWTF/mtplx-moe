"""Production-parity presence/frequency penalty tests.

Encodes the OpenAI/vLLM-faithful semantics (see the implementation spec):
additive on raw logits before temperature, completion-token counts only,
clamped to [-2, 2], default 0.0 = exact no-op (preserves MTP exactness).

RED until penalties are implemented: `SamplerConfig` does not yet accept
`presence_penalty`/`frequency_penalty`, and `distribution_from_logits` does not
yet accept `token_counts`.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mtplx.fast_sampling import apply_penalties_mlx
from mtplx.sampling import SamplerConfig, apply_penalties, distribution_from_logits


def test_presence_penalty_demotes_a_seen_token():
    # Token 0 leads token 1 by exactly 1.0. It has appeared in the completion,
    # so a presence_penalty of 2.0 (one-off) must drop its logit 5.0 -> 3.0,
    # below token 1's 4.0, flipping the argmax. No temperature/top-p/top-k
    # distortion (temperature=1.0, top_p=1.0, top_k=0).
    logits = np.array([5.0, 4.0, 0.0, 0.0])
    counts = {0: 3}  # token 0 appeared 3x; presence is a flat one-off regardless of count
    cfg = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, presence_penalty=2.0)
    probs = distribution_from_logits(logits, cfg, token_counts=counts)
    assert int(np.argmax(probs)) == 1


def test_frequency_penalty_scales_with_count():
    # frequency_penalty is linear in the count: 0.5 * 4 = 2.0 -> token 0: 5.0 -> 3.0
    # < token 1's 4.0. (A count of 1 at 0.5 would be only -0.5 and NOT flip it,
    # which is what distinguishes frequency from presence.)
    logits = np.array([5.0, 4.0, 0.0, 0.0])
    cfg = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, frequency_penalty=0.5)
    assert int(np.argmax(distribution_from_logits(logits, cfg, token_counts={0: 4}))) == 1
    # count too small to flip -> token 0 still wins (proves it's count-scaled, not flat)
    assert int(np.argmax(distribution_from_logits(logits, cfg, token_counts={0: 1}))) == 0


def test_zero_penalties_are_byte_identical_to_baseline():
    # Default 0.0 penalties must be an EXACT no-op vs the current sampler, even
    # when token_counts is supplied. This is the exactness-preservation guarantee
    # that keeps MTP byte-identical to today when the knobs are unset.
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    base = distribution_from_logits(logits, SamplerConfig(temperature=0.7, top_p=0.9, top_k=3))
    withp = distribution_from_logits(
        logits,
        SamplerConfig(temperature=0.7, top_p=0.9, top_k=3, presence_penalty=0.0, frequency_penalty=0.0),
        token_counts={0: 3, 1: 2},
    )
    assert np.array_equal(base, withp)


def test_penalties_clamped_to_openai_range():
    # Out-of-range penalties clamp to [-2, 2] (OpenAI/vLLM range). With clamping,
    # presence_penalty=100 becomes 2.0: token 0 (5.0) -> 3.0, still > token 1 (2.0),
    # so token 0 keeps winning. Without clamping it would crater to -95 and flip.
    logits = np.array([5.0, 2.0])
    cfg = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, presence_penalty=100.0)
    assert int(np.argmax(distribution_from_logits(logits, cfg, token_counts={0: 1}))) == 0


def test_apply_penalties_mlx_matches_numpy_reference():
    # On-device MLX penalty must produce the same penalized logits as the NumPy
    # reference, so the runtime hot path and the reference path stay in lockstep.
    raw = [5.0, 4.0, 3.0, 1.0, 0.0]
    counts = {0: 3, 2: 1}
    ref = apply_penalties(np.array(raw), counts, presence_penalty=1.5, frequency_penalty=0.5)
    mlx_out = apply_penalties_mlx(mx.array(raw, dtype=mx.float32), counts,
                                  presence_penalty=1.5, frequency_penalty=0.5)
    mx.eval(mlx_out)
    assert np.allclose(np.asarray(mlx_out), ref, atol=1e-5)


def test_apply_penalties_mlx_zero_is_exact_noop_identity():
    logits = mx.array([5.0, 4.0, 0.0], dtype=mx.float32)
    # both penalties 0 -> same object back (no scatter, no copy) => exactness
    assert apply_penalties_mlx(logits, {0: 9}, 0.0, 0.0) is logits
    # no counts -> also a no-op
    assert apply_penalties_mlx(logits, None, 2.0, 2.0) is logits
