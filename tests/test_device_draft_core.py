"""Device draft core: q-construction parity, sampling law, state signature."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.fast_sampling import sparse_distribution_from_mlx_logits  # noqa: E402
from mtplx.generation import (  # noqa: E402
    _device_core_state_signature,
    _device_draft_q_arrays,
)
from mtplx.sampling import SamplerConfig  # noqa: E402


def _host_q(logits: mx.array, config: SamplerConfig) -> dict[int, float]:
    sparse = sparse_distribution_from_mlx_logits(logits, config)
    assert sparse is not None
    return {int(t): float(p) for t, p in zip(sparse.token_ids, sparse.probs)}


def _device_q(logits: mx.array, config: SamplerConfig) -> dict[int, float]:
    ids, probs = _device_draft_q_arrays(
        logits.reshape(-1),
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
    )
    ids_np = np.asarray(ids, dtype=np.int64).reshape(-1)
    probs_np = np.asarray(probs, dtype=np.float64).reshape(-1)
    keep = probs_np > 0
    kept = probs_np[keep]
    return {int(t): float(p) for t, p in zip(ids_np[keep], kept / kept.sum())}


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("scale", [1.0, 6.0])
def test_device_q_matches_host_sparse_distribution(seed: int, scale: float) -> None:
    rng = np.random.default_rng(seed)
    logits = mx.array((rng.standard_normal(512) * scale).astype(np.float32))
    config = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)

    host = _host_q(logits, config)
    device = _device_q(logits, config)

    assert set(device) == set(host)
    for token, prob in host.items():
        assert device[token] == pytest.approx(prob, abs=2e-5), token


def test_device_q_top_p_disabled_branch() -> None:
    rng = np.random.default_rng(3)
    logits = mx.array(rng.standard_normal(256).astype(np.float32))
    config = SamplerConfig(temperature=0.8, top_p=1.0, top_k=16)

    host = _host_q(logits, config)
    device = _device_q(logits, config)

    assert set(device) == set(host)
    for token, prob in host.items():
        assert device[token] == pytest.approx(prob, abs=2e-5), token


def test_device_inverse_cdf_sampling_matches_q() -> None:
    # The compiled chain samples via inverse-CDF over the normalized kept
    # support; the empirical law over many keys must match q itself.
    rng = np.random.default_rng(11)
    logits = mx.array((rng.standard_normal(128) * 4.0).astype(np.float32))
    ids, q_norm = _device_draft_q_arrays(logits, temperature=0.6, top_k=20, top_p=0.95)
    cdf = mx.cumsum(q_norm, axis=-1)
    k = int(ids.shape[0])

    draws = 20_000
    keys = mx.random.split(mx.random.key(1234), draws)
    counts: dict[int, int] = {}
    picks = []
    for i in range(draws):
        u = mx.random.uniform(key=keys[i])
        picks.append(mx.minimum((cdf <= u).sum(), k - 1).astype(mx.int32))
    mx.eval(picks)
    ids_np = np.asarray(ids, dtype=np.int64)
    for pick in picks:
        token = int(ids_np[int(pick.item())])
        counts[token] = counts.get(token, 0) + 1

    probs_np = np.asarray(q_norm, dtype=np.float64)
    for i, token in enumerate(ids_np):
        expected = probs_np[i]
        if expected == 0.0:
            assert counts.get(int(token), 0) == 0
            continue
        observed = counts.get(int(token), 0) / draws
        sigma = (expected * (1 - expected) / draws) ** 0.5
        assert abs(observed - expected) < max(5 * sigma, 5e-4), (token, observed, expected)


class _FakeTensorOffsetEntry:
    def __init__(self, keys: mx.array, values: mx.array, offset: int) -> None:
        self.compile_state = [[keys, values, mx.array(offset, dtype=mx.int32)], [None, None, None]]


def test_state_signature_survives_shape_stable_swaps() -> None:
    keys_a = mx.zeros((1, 4, 32, 64), dtype=mx.float16)
    values_a = mx.zeros((1, 4, 32, 64), dtype=mx.float16)
    cache = [_FakeTensorOffsetEntry(keys_a, values_a, 3)]
    first = _device_core_state_signature(cache)

    # Same shapes, brand-new arrays (the routine eager-append swap).
    cache[0].compile_state[0][0] = mx.ones((1, 4, 32, 64), dtype=mx.float16)
    cache[0].compile_state[0][2] = mx.array(9, dtype=mx.int32)
    assert _device_core_state_signature(cache) == first

    # Capacity growth changes the traced shapes and must invalidate.
    cache[0].compile_state[0][0] = mx.zeros((1, 4, 64, 64), dtype=mx.float16)
    assert _device_core_state_signature(cache) != first
