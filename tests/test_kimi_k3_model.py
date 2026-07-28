from __future__ import annotations

import math
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mtplx.cache_state import restore_cache, snapshot_cache
from mtplx.models.kimi_k3_mlx import (
    KDACache,
    KimiDeltaAttention,
    KimiLatentMoE,
    KimiMLAAttention,
    KimiMoERouter,
    MLALatentCache,
    Model,
    ModelArgs,
    _normalize_kda_qk,
    _weighted_routed_sum,
    apply_attn_res,
    map_kda_a_log,
    situ,
)


def _np(value: mx.array) -> np.ndarray:
    mx.eval(value)
    if value.dtype == mx.bfloat16:
        value = value.astype(mx.float32)
        mx.eval(value)
    return np.asarray(value)


def _set_linear(module: nn.Linear, values: np.ndarray) -> None:
    module.weight = mx.array(values, dtype=mx.float32)


def _rms_np(value: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    work = value.astype(np.float32)
    return work / np.sqrt(np.mean(work * work, axis=-1, keepdims=True) + eps) * weight


def test_situ_uses_fp32_formula_and_casts_back() -> None:
    gate = mx.array([[6.0, -3.0]], dtype=mx.bfloat16)
    up = mx.array([[30.0, -12.0]], dtype=mx.bfloat16)

    actual = situ(gate, up)
    gate_np = _np(gate).astype(np.float32)
    up_np = _np(up).astype(np.float32)
    expected = (
        4.0
        * np.tanh(gate_np / 4.0)
        * (1.0 / (1.0 + np.exp(-gate_np)))
        * (25.0 * np.tanh(up_np / 25.0))
    )

    assert actual.dtype == mx.bfloat16
    np.testing.assert_allclose(_np(actual).astype(np.float32), expected, rtol=8e-3)


def test_router_bias_only_selects_and_unbiased_scores_are_renormalized() -> None:
    router = KimiMoERouter(
        hidden_size=2,
        num_experts=3,
        top_k=2,
        routed_scaling_factor=1.5,
    )
    router.weight = mx.array([[2.0, 0.0], [1.0, 0.0], [-3.0, 0.0]], dtype=mx.float32)
    router.e_score_correction_bias = mx.array([0.0, 0.0, 10.0])
    hidden = mx.array([[[1.0, 0.0]]], dtype=mx.bfloat16)

    indices, weights = router(hidden)
    chosen = _np(indices)[0, 0]
    actual = _np(weights)[0, 0]
    logits = np.array([2.0, 1.0, -3.0], dtype=np.float32)
    unbiased = 1.0 / (1.0 + np.exp(-logits))
    expected = unbiased[chosen]
    expected = expected / expected.sum() * 1.5

    assert set(chosen.tolist()) == {0, 2}
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


class _FakeSwitch(nn.Module):
    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        factors = indices.astype(x.dtype)[..., None] + 1
        return x[..., None, :] * factors


def test_latent_moe_routes_original_but_switches_projected_hidden() -> None:
    moe = KimiLatentMoE(
        hidden_size=4,
        routed_hidden_size=2,
        intermediate_size=3,
        num_experts=3,
        top_k=2,
        num_shared_experts=2,
        routed_scaling_factor=1.0,
        rms_norm_eps=1e-5,
    )
    moe.switch_mlp = _FakeSwitch()
    moe.gate.weight = mx.array(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]
    )
    moe.gate.e_score_correction_bias = mx.array([0.0, 0.0, 0.0])
    _set_linear(
        moe.routed_expert_down_proj,
        np.array([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, -0.5]]),
    )
    _set_linear(
        moe.routed_expert_up_proj,
        np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 1.0]]),
    )
    moe.routed_expert_norm.weight = mx.array([1.25, 0.75])
    _set_linear(moe.shared_experts.gate_proj, np.zeros((6, 4), dtype=np.float32))
    _set_linear(moe.shared_experts.up_proj, np.zeros((6, 4), dtype=np.float32))
    _set_linear(moe.shared_experts.down_proj, np.zeros((4, 6), dtype=np.float32))
    hidden = mx.array([[[1.0, 0.5, -0.25, 0.75]]], dtype=mx.float32)

    actual = _np(moe(hidden))
    original = _np(hidden)
    logits = original @ _np(moe.gate.weight).T
    scores = 1.0 / (1.0 + np.exp(-logits))
    indices = np.argpartition(-scores, kth=1, axis=-1)[..., :2]
    weights = np.take_along_axis(scores, indices, axis=-1)
    weights /= weights.sum(axis=-1, keepdims=True)
    latent = original @ _np(moe.routed_expert_down_proj.weight).T
    selected = latent[..., None, :] * (indices[..., None] + 1)
    routed = (selected * weights[..., None]).sum(axis=-2)
    routed = _rms_np(routed, _np(moe.routed_expert_norm.weight), 1e-5)
    expected = routed @ _np(moe.routed_expert_up_proj.weight).T

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_weighted_routed_sum_rounds_back_to_bf16_before_norm_and_up() -> None:
    selected = mx.array([[[[1.0, 0.5], [3.0, -1.0], [-0.25, 2.0]]]], dtype=mx.bfloat16)
    weights = mx.array([[[0.2, 0.3, 0.5]]], dtype=mx.float32)

    actual = _weighted_routed_sum(selected, weights)
    expected_fp32 = (_np(selected).astype(np.float32) * _np(weights)[..., None]).sum(
        axis=-2
    )
    expected = mx.array(expected_fp32, dtype=mx.bfloat16)

    assert actual.dtype == mx.bfloat16
    np.testing.assert_array_equal(_np(actual), _np(expected))


def _tiny_mla() -> KimiMLAAttention:
    module = KimiMLAAttention(
        hidden_size=4,
        num_heads=2,
        q_lora_rank=3,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=2,
        rms_norm_eps=1e-5,
    )
    rng = np.random.default_rng(7)
    for linear in (
        module.q_a_proj,
        module.q_b_proj,
        module.kv_a_proj_with_mqa,
        module.kv_b_proj,
        module.g_proj,
        module.o_proj,
    ):
        _set_linear(linear, rng.normal(0, 0.2, size=linear.weight.shape))
    module.q_a_layernorm.weight = mx.array([0.8, 1.1, 1.2])
    module.kv_a_layernorm.weight = mx.array([1.2, 0.7])
    return module


def _mla_numpy(module: KimiMLAAttention, hidden: np.ndarray) -> np.ndarray:
    b, length, _ = hidden.shape
    h = module.num_heads
    q = hidden @ _np(module.q_a_proj.weight).T
    q = _rms_np(q, _np(module.q_a_layernorm.weight), 1e-5)
    q = q @ _np(module.q_b_proj.weight).T
    q = q.reshape(b, length, h, module.q_head_dim).transpose(0, 2, 1, 3)
    q_nope = q[..., : module.qk_nope_head_dim]
    q_rot = q[..., module.qk_nope_head_dim :]
    kv = hidden @ _np(module.kv_a_proj_with_mqa.weight).T
    latent = kv[..., : module.kv_lora_rank]
    k_rot = kv[..., module.kv_lora_rank :]
    latent = _rms_np(latent, _np(module.kv_a_layernorm.weight), 1e-5)
    kv_weight = _np(module.kv_b_proj.weight).reshape(
        h, module.qk_nope_head_dim + module.v_head_dim, module.kv_lora_rank
    )
    q_latent = np.einsum(
        "bhln,hnr->bhlr", q_nope, kv_weight[:, : module.qk_nope_head_dim]
    )
    scores = np.einsum("bhlr,btr->bhlt", q_latent, latent)
    scores += np.einsum("bhld,btd->bhlt", q_rot, k_rot)
    scores *= module.scale
    causal = np.arange(length)[:, None] >= np.arange(length)[None, :]
    scores = np.where(causal[None, None], scores, -np.inf)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    latent_out = np.einsum("bhlt,btr->bhlr", probs, latent)
    value = np.einsum(
        "bhlr,hvr->bhlv", latent_out, kv_weight[:, module.qk_nope_head_dim :]
    )
    value = value.transpose(0, 2, 1, 3).reshape(b, length, -1)
    gate = hidden @ _np(module.g_proj.weight).T
    value *= 1.0 / (1.0 + np.exp(-gate))
    return value @ _np(module.o_proj.weight).T


def test_mla_matches_numpy_and_uses_one_width_sum_latent_cache_head() -> None:
    module = _tiny_mla()
    hidden = np.random.default_rng(4).normal(0, 0.3, size=(1, 3, 4)).astype(np.float32)
    cache = MLALatentCache(module.kv_lora_rank, module.qk_rope_head_dim)

    actual = _np(module(mx.array(hidden), cache=cache))
    expected = _mla_numpy(module, hidden)

    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=1e-4)
    assert cache.state is not None
    assert cache.state.shape == (1, 1, 3, 3)


def test_mla_prefill_equals_tokenwise_cached_decode() -> None:
    module = _tiny_mla()
    hidden = mx.array(
        np.random.default_rng(9).normal(0, 0.2, size=(1, 4, 4)),
        dtype=mx.float32,
    )
    prefill = module(hidden, cache=MLALatentCache(2, 1))
    cache = MLALatentCache(2, 1)
    tokenwise = mx.concatenate(
        [module(hidden[:, i : i + 1], cache=cache) for i in range(4)], axis=1
    )

    np.testing.assert_allclose(_np(prefill), _np(tokenwise), rtol=5e-3, atol=1e-4)


def test_mla_cache_grows_by_capacity_and_trims_logically() -> None:
    cache = MLALatentCache(2, 1, step=4)
    first_latent = mx.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=mx.float32)
    first_rotary = mx.array([[[[5.0], [6.0]]]], dtype=mx.float32)

    latent, rotary = cache.update_and_fetch(first_latent, first_rotary)

    assert cache.offset == cache.size() == 2
    assert cache.capacity == 4
    assert cache.nbytes == 4 * 3 * 4
    assert not cache.empty()
    assert cache.is_trimmable()
    np.testing.assert_array_equal(_np(latent), _np(first_latent))
    np.testing.assert_array_equal(_np(rotary), _np(first_rotary))

    second_latent = mx.array(
        [[[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]], dtype=mx.float32
    )
    second_rotary = mx.array([[[[13.0], [14.0], [15.0]]]], dtype=mx.float32)
    cache.update_and_fetch(second_latent[..., :2, :], second_rotary[..., :2, :])

    assert cache.offset == 4
    assert cache.capacity == 4
    cache.update_and_fetch(second_latent[..., 2:, :], second_rotary[..., 2:, :])
    assert cache.offset == 5
    assert cache.capacity == 8
    assert cache.trim(2) == 2
    assert cache.offset == cache.size() == 3
    assert cache.capacity == 8

    replacement_latent = mx.array([[[[21.0, 22.0]]]], dtype=mx.float32)
    replacement_rotary = mx.array([[[[23.0]]]], dtype=mx.float32)
    latent, rotary = cache.update_and_fetch(replacement_latent, replacement_rotary)

    expected_latent = np.array([[[[1.0, 2.0], [3.0, 4.0], [7.0, 8.0], [21.0, 22.0]]]])
    expected_rotary = np.array([[[[5.0], [6.0], [13.0], [23.0]]]])
    assert cache.offset == 4
    assert cache.capacity == 8
    np.testing.assert_array_equal(_np(latent), expected_latent)
    np.testing.assert_array_equal(_np(rotary), expected_rotary)


def test_mla_cache_single_token_growth_is_geometric() -> None:
    cache = MLALatentCache(2, 1, step=4)
    capacities = []
    capacity_by_size = []
    expected = []
    previous_capacity = 0

    for index in range(33):
        latent = mx.array([[[[float(index), float(index) + 0.25]]]], dtype=mx.float32)
        rotary = mx.array([[[[-float(index)]]]], dtype=mx.float32)
        cache.update_and_fetch(latent, rotary)
        expected.append([float(index), float(index) + 0.25, -float(index)])
        capacity_by_size.append(cache.capacity)
        if cache.capacity != previous_capacity:
            capacities.append(cache.capacity)
            previous_capacity = cache.capacity

    assert capacities == [4, 8, 16, 32, 64]
    assert len(capacities) == math.ceil(math.log2(cache.size() / cache.step)) + 1
    for logical_size, capacity in enumerate(capacity_by_size, start=1):
        assert logical_size <= capacity < 2 * max(logical_size, cache.step)
    assert cache.state is not None
    np.testing.assert_array_equal(
        _np(cache.state), np.array([[expected]], dtype=np.float32)
    )


def test_mla_populated_snapshot_restore_preserves_continuation_parity() -> None:
    module = _tiny_mla()
    hidden = mx.array(
        np.random.default_rng(43).normal(0, 0.2, size=(1, 5, 4)),
        dtype=mx.float32,
    )
    live = MLALatentCache(2, 1, step=4)
    module(hidden[:, :3], cache=live)
    snapshot = snapshot_cache([live])
    expected = module(hidden[:, 3:], cache=live)

    restored = MLALatentCache(2, 1, step=4)
    module(hidden[:, :1], cache=restored)
    restore_cache([restored], snapshot)
    actual = module(hidden[:, 3:], cache=restored)

    assert restored.offset == live.offset == 5
    assert restored.size() == live.size() == 5
    assert restored.state is not None
    assert live.state is not None
    np.testing.assert_allclose(_np(actual), _np(expected), rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(
        _np(restored.state), _np(live.state), rtol=3e-5, atol=3e-5
    )


def _mla_numpy_cached(
    module: KimiMLAAttention, hidden: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    b, length, _ = hidden.shape
    h = module.num_heads
    kv_weight = _np(module.kv_b_proj.weight).reshape(
        h, module.qk_nope_head_dim + module.v_head_dim, module.kv_lora_rank
    )
    key_weight = kv_weight[:, : module.qk_nope_head_dim]
    value_weight = kv_weight[:, module.qk_nope_head_dim :]
    cache = np.zeros(
        (b, 1, 0, module.kv_lora_rank + module.qk_rope_head_dim),
        dtype=np.float32,
    )
    outputs = []
    for index in range(length):
        token = hidden[:, index : index + 1]
        q = token @ _np(module.q_a_proj.weight).T
        q = _rms_np(q, _np(module.q_a_layernorm.weight), 1e-5)
        q = q @ _np(module.q_b_proj.weight).T
        q = q.reshape(b, 1, h, module.q_head_dim).transpose(0, 2, 1, 3)
        q_nope = q[..., : module.qk_nope_head_dim]
        q_rot = q[..., module.qk_nope_head_dim :]
        compressed = token @ _np(module.kv_a_proj_with_mqa.weight).T
        latent = _rms_np(
            compressed[..., : module.kv_lora_rank],
            _np(module.kv_a_layernorm.weight),
            1e-5,
        )
        k_rot = compressed[..., module.kv_lora_rank :]
        combined = np.concatenate([latent, k_rot], axis=-1)[:, None]
        cache = np.concatenate([cache, combined], axis=-2)
        all_latent = cache[..., : module.kv_lora_rank]
        all_k_rot = cache[..., module.kv_lora_rank :]
        q_latent = np.einsum("bhln,hnr->bhlr", q_nope, key_weight)
        scores = q_latent @ all_latent.swapaxes(-1, -2)
        scores += q_rot @ all_k_rot.swapaxes(-1, -2)
        scores *= module.scale
        scores -= scores.max(axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        latent_output = probabilities @ all_latent
        output = np.einsum("bhlr,hvr->bhlv", latent_output, value_weight)
        output = output.transpose(0, 2, 1, 3).reshape(b, 1, h * module.v_head_dim)
        gate = token @ _np(module.g_proj.weight).T
        output *= 1.0 / (1.0 + np.exp(-gate))
        outputs.append(output @ _np(module.o_proj.weight).T)
    return np.concatenate(outputs, axis=1), cache


def test_mla_cached_decode_matches_independent_outputs_and_final_cache() -> None:
    module = _tiny_mla()
    hidden = np.random.default_rng(29).normal(0, 0.2, size=(1, 4, 4)).astype(np.float32)
    cache = MLALatentCache(2, 1)
    actual = mx.concatenate(
        [
            module(mx.array(hidden[:, index : index + 1]), cache=cache)
            for index in range(hidden.shape[1])
        ],
        axis=1,
    )
    expected_output, expected_cache = _mla_numpy_cached(module, hidden)

    np.testing.assert_allclose(_np(actual), expected_output, rtol=5e-3, atol=1e-4)
    assert cache.state is not None
    np.testing.assert_allclose(_np(cache.state), expected_cache, rtol=5e-3, atol=1e-4)


def _tiny_kda() -> KimiDeltaAttention:
    module = KimiDeltaAttention(
        hidden_size=4,
        num_heads=2,
        head_dim=2,
        conv_kernel_size=3,
        rms_norm_eps=1e-5,
        gate_lower_bound=-5.0,
    )
    rng = np.random.default_rng(11)
    for linear in (
        module.q_proj,
        module.k_proj,
        module.v_proj,
        module.f_a_proj,
        module.f_b_proj,
        module.b_proj,
        module.g_proj,
        module.o_proj,
    ):
        _set_linear(linear, rng.normal(0, 0.18, size=linear.weight.shape))
    for conv in (module.q_conv1d, module.k_conv1d, module.v_conv1d):
        conv.weight = mx.array(
            rng.normal(0, 0.2, size=conv.weight.shape), dtype=mx.float32
        )
    module.A_log = mx.array(np.log([1.5, 2.0]), dtype=mx.float32)
    module.dt_bias = mx.array([0.1, -0.2, 0.05, 0.15], dtype=mx.float32)
    module.o_norm.weight = mx.array([0.9, 1.1], dtype=mx.float32)
    return module


def _conv_np(projected: np.ndarray, weight: np.ndarray, kernel: int) -> np.ndarray:
    padded = np.concatenate(
        [
            np.zeros((projected.shape[0], kernel - 1, projected.shape[-1])),
            projected,
        ],
        axis=1,
    )
    out = []
    channel_weight = weight[..., 0]
    for i in range(projected.shape[1]):
        out.append((padded[:, i : i + kernel] * channel_weight.T).sum(axis=1))
    out = np.stack(out, axis=1)
    return out / (1.0 + np.exp(-out))


def _kda_numpy(module: KimiDeltaAttention, hidden: np.ndarray) -> np.ndarray:
    b, length, _ = hidden.shape
    h, d = module.num_heads, module.head_dim
    qs = _conv_np(
        hidden @ _np(module.q_proj.weight).T, _np(module.q_conv1d.weight), 3
    ).reshape(b, length, h, d)
    ks = _conv_np(
        hidden @ _np(module.k_proj.weight).T, _np(module.k_conv1d.weight), 3
    ).reshape(b, length, h, d)
    vs = _conv_np(
        hidden @ _np(module.v_proj.weight).T, _np(module.v_conv1d.weight), 3
    ).reshape(b, length, h, d)
    qs = qs / np.sqrt((qs * qs).sum(-1, keepdims=True) + 1e-6) / math.sqrt(d)
    ks = ks / np.sqrt((ks * ks).sum(-1, keepdims=True) + 1e-6)
    raw_g = hidden @ _np(module.f_a_proj.weight).T
    raw_g = raw_g @ _np(module.f_b_proj.weight).T
    raw_g = raw_g.reshape(b, length, h, d)
    beta = hidden @ _np(module.b_proj.weight).T
    beta = 1.0 / (1.0 + np.exp(-beta))
    a = np.exp(_np(module.A_log))[None, None, :, None]
    dt = _np(module.dt_bias).reshape(h, d)
    log_decay = -5.0 / (1.0 + np.exp(-(a * (raw_g + dt))))
    decay = np.exp(log_decay)
    state = np.zeros((b, h, d, d), dtype=np.float32)
    outputs = []
    for i in range(length):
        state *= decay[:, i, :, None, :]
        memory = (state * ks[:, i, :, None, :]).sum(-1)
        delta = (vs[:, i] - memory) * beta[:, i, :, None]
        state += ks[:, i, :, None, :] * delta[..., None]
        outputs.append((state * qs[:, i, :, None, :]).sum(-1))
    out = np.stack(outputs, axis=1)
    out = _rms_np(out, _np(module.o_norm.weight), 1e-5)
    output_gate = hidden @ _np(module.g_proj.weight).T
    output_gate = output_gate.reshape(b, length, h, d)
    out *= 1.0 / (1.0 + np.exp(-output_gate))
    out = out.reshape(b, length, h * d)
    return out @ _np(module.o_proj.weight).T


def test_kda_matches_numpy_safe_gate_recurrence() -> None:
    module = _tiny_kda()
    hidden = (
        np.random.default_rng(15).normal(0, 0.25, size=(1, 4, 4)).astype(np.float32)
    )

    actual = _np(module(mx.array(hidden), cache=KDACache()))
    expected = _kda_numpy(module, hidden)

    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=3e-4)


def test_kda_qk_l2_normalization_is_fp32_scalar_eps_then_bf16() -> None:
    q = mx.array(
        [[[[0.125, -0.25, 0.5], [0.75, -0.5, 0.25]]]],
        dtype=mx.bfloat16,
    )
    k = mx.array(
        [[[[0.375, 0.25, -0.125], [-0.625, 0.5, 0.125]]]],
        dtype=mx.bfloat16,
    )

    actual_q, actual_k = _normalize_kda_qk(q, k)
    q_fp32 = _np(q).astype(np.float32)
    k_fp32 = _np(k).astype(np.float32)
    expected_q_fp32 = q_fp32 / np.sqrt(
        np.sum(q_fp32 * q_fp32, axis=-1, keepdims=True) + 1e-6
    )
    expected_k_fp32 = k_fp32 / np.sqrt(
        np.sum(k_fp32 * k_fp32, axis=-1, keepdims=True) + 1e-6
    )
    expected_q = mx.array(expected_q_fp32, dtype=mx.bfloat16)
    expected_k = mx.array(expected_k_fp32, dtype=mx.bfloat16)

    assert actual_q.dtype == mx.bfloat16
    assert actual_k.dtype == mx.bfloat16
    np.testing.assert_array_equal(_np(actual_q), _np(expected_q))
    np.testing.assert_array_equal(_np(actual_k), _np(expected_k))
    recurrent_q = actual_q.astype(mx.float32) * (3**-0.5)
    expected_recurrent_q = _np(expected_q).astype(np.float32) * (3**-0.5)
    np.testing.assert_array_equal(_np(recurrent_q), expected_recurrent_q)


def test_kda_prefill_equals_tokenwise_cached_decode() -> None:
    module = _tiny_kda()
    hidden = mx.array(
        np.random.default_rng(17).normal(0, 0.2, size=(1, 4, 4)),
        dtype=mx.float32,
    )
    prefill = module(hidden, cache=KDACache())
    cache = KDACache()
    tokenwise = mx.concatenate(
        [module(hidden[:, i : i + 1], cache=cache) for i in range(4)], axis=1
    )

    np.testing.assert_allclose(_np(prefill), _np(tokenwise), rtol=3e-2, atol=3e-4)


def _cast_kda_compute_dtype(module: KimiDeltaAttention, dtype: mx.Dtype) -> None:
    for linear in (
        module.q_proj,
        module.k_proj,
        module.v_proj,
        module.f_a_proj,
        module.f_b_proj,
        module.b_proj,
        module.g_proj,
        module.o_proj,
    ):
        linear.weight = linear.weight.astype(dtype)
    for conv in (module.q_conv1d, module.k_conv1d, module.v_conv1d):
        conv.weight = conv.weight.astype(dtype)
    module.o_norm.weight = module.o_norm.weight.astype(dtype)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (mx.float32, 3e-5, 3e-5),
        (mx.bfloat16, 1e-2, 1e-2),
    ],
    ids=["float32", "bfloat16"],
)
@pytest.mark.parametrize("cache_present", [False, True], ids=["local", "provided"])
def test_kda_chunked_prefill_matches_independent_direct_oracle(
    dtype: mx.Dtype,
    rtol: float,
    atol: float,
    cache_present: bool,
) -> None:
    module = _tiny_kda()
    _cast_kda_compute_dtype(module, dtype)
    rng = np.random.default_rng(47)
    prefix = mx.array(rng.normal(0, 0.2, size=(1, 3, 4)), dtype=dtype)
    hidden = mx.array(rng.normal(0, 0.2, size=(1, 129, 4)), dtype=dtype)
    mask_values = np.ones((1, 129), dtype=np.bool_)
    mask_values[:, [0, 63, 127, 128]] = False
    mask = mx.array(mask_values)
    oracle_cache = KDACache() if cache_present else None
    actual_cache = KDACache() if cache_present else None
    if cache_present:
        module._call_direct(prefix, cache=oracle_cache)
        module._call_direct(prefix, cache=actual_cache)

    expected = module._call_direct(hidden, mask=mask, cache=oracle_cache)
    actual = module(hidden, mask=mask, cache=actual_cache)
    mx.eval(expected, actual)

    assert expected.dtype == actual.dtype == dtype
    np.testing.assert_allclose(_np(actual), _np(expected), rtol=rtol, atol=atol)
    if cache_present:
        assert oracle_cache is not None
        assert actual_cache is not None
        assert oracle_cache.offset == actual_cache.offset == 132
        for expected_state, actual_state in zip(oracle_cache.state, actual_cache.state):
            assert expected_state is not None
            assert actual_state is not None
            assert expected_state.dtype == actual_state.dtype
            np.testing.assert_allclose(
                _np(actual_state), _np(expected_state), rtol=rtol, atol=atol
            )
        assert actual_cache.q_conv is not None
        assert actual_cache.q_conv.dtype == dtype
        assert actual_cache.recurrent is not None
        assert actual_cache.recurrent.dtype == mx.float32


def test_kda_length_route_keeps_decode_direct_and_slices_prefill_masks() -> None:
    module = _tiny_kda()
    direct = module._direct_executor
    calls: list[tuple[int, np.ndarray | None]] = []

    def record_direct(
        hidden_states: mx.array,
        mask: mx.array | None = None,
        cache: KDACache | None = None,
    ) -> mx.array:
        calls.append(
            (
                int(hidden_states.shape[1]),
                None if mask is None else _np(mask).copy(),
            )
        )
        return direct(hidden_states, mask, cache)

    module._direct_executor = record_direct
    decode_cache = KDACache()
    module(mx.zeros((1, 1, 4), dtype=mx.float32), cache=decode_cache)
    direct_mask = mx.ones((1, 128), dtype=mx.bool_)
    module(mx.zeros((1, 128, 4), dtype=mx.float32), mask=direct_mask)
    long_mask_values = (np.arange(129)[None, :] % 3) != 0
    module(
        mx.zeros((1, 129, 4), dtype=mx.float32),
        mask=mx.array(long_mask_values),
    )

    assert [length for length, _ in calls] == [1, 128, 128, 1]
    assert calls[0][1] is None
    np.testing.assert_array_equal(calls[1][1], np.ones((1, 128), dtype=np.bool_))
    np.testing.assert_array_equal(calls[2][1], long_mask_values[:, :128])
    np.testing.assert_array_equal(calls[3][1], long_mask_values[:, 128:])
    assert decode_cache.offset == 1


def _kda_numpy_cached(
    module: KimiDeltaAttention, hidden: np.ndarray
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    b, length, _ = hidden.shape
    h, d = module.num_heads, module.head_dim
    kernel = module.conv_kernel_size
    projection = h * d
    q_state = np.zeros((b, kernel - 1, projection), dtype=np.float32)
    k_state = np.zeros_like(q_state)
    v_state = np.zeros_like(q_state)
    recurrent = np.zeros((b, h, d, d), dtype=np.float32)
    outputs = []
    for index in range(length):
        token = hidden[:, index : index + 1]
        projected = [
            token @ _np(linear.weight).T
            for linear in (module.q_proj, module.k_proj, module.v_proj)
        ]
        conv_outputs = []
        new_states = []
        for value, state, conv in zip(
            projected,
            (q_state, k_state, v_state),
            (module.q_conv1d, module.k_conv1d, module.v_conv1d),
        ):
            padded = np.concatenate([state, value], axis=1)
            raw = (padded * _np(conv.weight)[..., 0].T[None, :, :]).sum(
                axis=1, keepdims=True
            )
            conv_outputs.append(raw / (1.0 + np.exp(-raw)))
            new_states.append(padded[:, -(kernel - 1) :])
        q_state, k_state, v_state = new_states
        q, k, v = [value.reshape(b, 1, h, d) for value in conv_outputs]
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        q /= math.sqrt(d)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        raw_gate = token @ _np(module.f_a_proj.weight).T
        raw_gate = raw_gate @ _np(module.f_b_proj.weight).T
        raw_gate = raw_gate.reshape(b, 1, h, d)
        beta = token @ _np(module.b_proj.weight).T
        beta = 1.0 / (1.0 + np.exp(-beta))
        a = np.exp(_np(module.A_log))[None, None, :, None]
        dt = _np(module.dt_bias).reshape(h, d)
        decay = np.exp(-5.0 / (1.0 + np.exp(-(a * (raw_gate + dt)))))
        recurrent *= decay[:, 0, :, None, :]
        memory = (recurrent * k[:, 0, :, None, :]).sum(-1)
        delta = (v[:, 0] - memory) * beta[:, 0, :, None]
        recurrent += k[:, 0, :, None, :] * delta[..., None]
        output = (recurrent * q[:, 0, :, None, :]).sum(-1)[:, None]
        output = _rms_np(output, _np(module.o_norm.weight), 1e-5)
        output_gate = token @ _np(module.g_proj.weight).T
        output_gate = output_gate.reshape(b, 1, h, d)
        output *= 1.0 / (1.0 + np.exp(-output_gate))
        output = output.reshape(b, 1, projection)
        outputs.append(output @ _np(module.o_proj.weight).T)
    return np.concatenate(outputs, axis=1), (
        q_state,
        k_state,
        v_state,
        recurrent,
    )


def test_kda_cached_decode_matches_independent_outputs_and_final_caches() -> None:
    module = _tiny_kda()
    hidden = np.random.default_rng(31).normal(0, 0.2, size=(1, 4, 4)).astype(np.float32)
    cache = KDACache()
    actual = mx.concatenate(
        [
            module(mx.array(hidden[:, index : index + 1]), cache=cache)
            for index in range(hidden.shape[1])
        ],
        axis=1,
    )
    expected_output, expected_cache = _kda_numpy_cached(module, hidden)

    np.testing.assert_allclose(_np(actual), expected_output, rtol=5e-3, atol=3e-4)
    for actual_state, expected_state in zip(
        (cache.q_conv, cache.k_conv, cache.v_conv, cache.recurrent),
        expected_cache,
    ):
        assert actual_state is not None
        np.testing.assert_allclose(
            _np(actual_state), expected_state, rtol=5e-3, atol=3e-4
        )


def test_kda_populated_snapshot_restore_preserves_continuation_parity() -> None:
    module = _tiny_kda()
    hidden = mx.array(
        np.random.default_rng(37).normal(0, 0.2, size=(1, 5, 4)),
        dtype=mx.float32,
    )
    live = KDACache()
    module(hidden[:, :3], cache=live)
    snapshot = snapshot_cache([live])
    expected = module(hidden[:, 3:], cache=live)

    restored = KDACache()
    module(hidden[:, :1], cache=restored)
    assert restored.offset == 1
    restore_cache([restored], snapshot)
    actual = module(hidden[:, 3:], cache=restored)

    assert restored.offset == live.offset == 5
    assert not restored.is_trimmable()
    np.testing.assert_allclose(_np(actual), _np(expected), rtol=3e-5, atol=3e-5)
    for restored_state, live_state in zip(restored.state, live.state):
        assert restored_state is not None
        assert live_state is not None
        np.testing.assert_allclose(
            _np(restored_state), _np(live_state), rtol=3e-5, atol=3e-5
        )


def test_source_like_f32_kda_residents_install_as_bf16_compute() -> None:
    args = ModelArgs.tiny(
        vocab_size=16,
        hidden_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=6,
        moe_intermediate_size=3,
        routed_expert_hidden_size=2,
        num_experts=3,
        num_experts_per_token=2,
        num_shared_experts=2,
        q_lora_rank=3,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=2,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 2,
            "head_dim": 2,
            "short_conv_kernel_size": 3,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    model = Model(args)
    kda = model.model.layers[0].self_attn
    assert isinstance(kda, KimiDeltaAttention)
    for linear in (
        kda.q_proj,
        kda.k_proj,
        kda.v_proj,
        kda.f_a_proj,
        kda.f_b_proj,
        kda.b_proj,
        kda.g_proj,
        kda.o_proj,
    ):
        linear.weight = linear.weight.astype(mx.bfloat16)
    source = {
        f"language_model.model.layers.0.self_attn.{name}.weight": mx.full(
            (4, 1, 3), 0.125, dtype=mx.float32
        )
        for name in ("q_conv1d", "k_conv1d", "v_conv1d")
    }
    source["language_model.model.layers.0.self_attn.o_norm.weight"] = mx.ones(
        (2,), dtype=mx.float32
    )
    installed = model.sanitize(source)
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        value = installed[f"model.layers.0.self_attn.{name}.weight"]
        assert value.dtype == mx.bfloat16
        setattr(getattr(kda, name), "weight", value)
    kda.o_norm.weight = installed["model.layers.0.self_attn.o_norm.weight"]

    hidden = mx.array(
        [[[0.5, -0.25, 0.125, 0.75], [0.25, 0.5, -0.5, 0.125]]],
        dtype=mx.bfloat16,
    )
    cache = KDACache()
    output = kda(hidden, cache=cache)

    assert installed["model.layers.0.self_attn.o_norm.weight"].dtype == mx.bfloat16
    assert output.dtype == mx.bfloat16
    assert cache.q_conv is not None and cache.q_conv.dtype == mx.bfloat16
    assert cache.k_conv is not None and cache.k_conv.dtype == mx.bfloat16
    assert cache.v_conv is not None and cache.v_conv.dtype == mx.bfloat16
    assert cache.recurrent is not None and cache.recurrent.dtype == mx.float32
    assert kda.A_log.dtype == mx.float32
    assert kda.dt_bias.dtype == mx.float32
    router = model.model.layers[1].mlp.gate
    assert router.e_score_correction_bias.dtype == mx.float32


def test_kda_a_log_checkpoint_mapping_accepts_only_exact_128_to_96() -> None:
    source = mx.array(np.arange(128, dtype=np.float32))
    mapped = map_kda_a_log(source, num_heads=96)
    np.testing.assert_array_equal(_np(mapped), np.arange(96, dtype=np.float32))

    for bad_shape in ((127,), (129,), (1, 1, 128, 1), (96,)):
        with pytest.raises(ValueError, match="exact shape"):
            map_kda_a_log(mx.zeros(bad_shape), num_heads=96)
    with pytest.raises(ValueError, match="96 configured heads"):
        map_kda_a_log(source, num_heads=95)


def test_apply_attn_res_matches_independent_numpy_formula() -> None:
    norm = nn.RMSNorm(3, eps=1e-5)
    norm.weight = mx.array([0.75, 1.0, 1.25])
    proj = nn.Linear(3, 1, bias=False)
    proj.weight = mx.array([[0.2, -0.4, 0.6]])
    prefix = np.array([[0.5, -1.0, 0.25], [0.2, 0.4, -0.6]], dtype=np.float32)
    blocks = np.array(
        [
            [[1.0, 0.0, -0.5], [0.2, 0.3, 0.4]],
            [[-0.5, 0.7, 0.1], [0.9, -0.2, 0.3]],
        ],
        dtype=np.float32,
    )

    actual = _np(
        apply_attn_res(mx.array(prefix), mx.array(blocks), proj=proj, norm=norm)
    )
    values = np.concatenate([blocks, prefix[:, None]], axis=1)
    normalized = values / np.sqrt(
        np.mean(values * values, axis=-1, keepdims=True) + 1e-5
    )
    score_weight = _np(norm.weight) * _np(proj.weight).squeeze(0)
    scores = (normalized * score_weight).sum(-1)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    expected = (probs[..., None] * values).sum(axis=1)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("model_type", "not_kimi"),
        ("vocab_size", 163839),
        ("hidden_size", 7167),
        ("routed_expert_hidden_size", 3583),
        ("intermediate_size", 33791),
        ("moe_intermediate_size", 3071),
        ("num_experts", 895),
        ("num_experts_per_token", 15),
        ("num_shared_experts", 1),
        ("num_hidden_layers", 92),
        ("num_attention_heads", 95),
        ("num_key_value_heads", 95),
        ("q_lora_rank", 1535),
        ("kv_lora_rank", 511),
        ("qk_nope_head_dim", 127),
        ("qk_rope_head_dim", 63),
        ("v_head_dim", 127),
        ("attn_res_block_size", 11),
        ("first_k_dense_replace", 0),
        ("moe_layer_freq", 2),
        ("tie_word_embeddings", True),
        ("routed_scaling_factor", 1.5),
    ],
)
def test_production_model_args_fail_closed_on_non_pinned_geometry(
    field: str, invalid: object
) -> None:
    values = asdict(ModelArgs())
    values[field] = invalid
    with pytest.raises(ValueError):
        ModelArgs(**values)


def test_production_model_args_require_exact_one_based_attention_topology() -> None:
    values = asdict(ModelArgs())
    config = dict(values["linear_attn_config"])
    config["kda_layers"] = list(config["kda_layers"])
    config["kda_layers"][0] = 4
    values["linear_attn_config"] = config

    with pytest.raises(ValueError):
        ModelArgs(**values)


def test_artifact_from_dict_cannot_select_test_only_geometry() -> None:
    tiny = asdict(
        ModelArgs.tiny(
            vocab_size=16,
            hidden_size=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=6,
            moe_intermediate_size=3,
            routed_expert_hidden_size=2,
            num_experts=3,
            num_experts_per_token=2,
            num_shared_experts=2,
            q_lora_rank=3,
            kv_lora_rank=2,
            qk_nope_head_dim=2,
            qk_rope_head_dim=1,
            v_head_dim=2,
            linear_attn_config={
                "kda_layers": [1],
                "full_attn_layers": [2],
                "num_heads": 2,
                "head_dim": 2,
                "short_conv_kernel_size": 3,
                "use_full_rank_gate": True,
                "gate_lower_bound": -5.0,
            },
        )
    )
    tiny["_test_only_geometry"] = True

    with pytest.raises(ValueError, match="pinned Kimi K3"):
        ModelArgs.from_dict(tiny)


class _Identity(nn.Module):
    def __call__(self, value: mx.array) -> mx.array:
        return value


class _AttentionScale(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def __call__(self, value: mx.array, *, cache: object | None = None) -> mx.array:
        del cache
        return value * self.scale


class _MLPScale(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def __call__(self, value: mx.array) -> mx.array:
        return value * self.scale


def _attn_res_numpy(
    prefix: np.ndarray,
    blocks: np.ndarray,
    projection: np.ndarray,
    norm_weight: np.ndarray,
) -> np.ndarray:
    values = np.concatenate([blocks, prefix[:, None]], axis=1).astype(np.float32)
    normalized = values / np.sqrt(
        np.mean(values * values, axis=-1, keepdims=True) + 1e-5
    )
    scores = (normalized * (projection * norm_weight)).sum(-1)
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return (probabilities[..., None] * values).sum(axis=1)


def test_thirteen_layer_attn_res_state_machine_and_one_based_routing() -> None:
    kda_layers = [1, 3, 5, 7, 9, 11, 13]
    full_layers = [2, 4, 6, 8, 10, 12]
    args = ModelArgs.tiny(
        vocab_size=8,
        hidden_size=2,
        num_hidden_layers=13,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=3,
        moe_intermediate_size=2,
        routed_expert_hidden_size=2,
        num_experts=3,
        num_experts_per_token=2,
        num_shared_experts=2,
        q_lora_rank=2,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=1,
        linear_attn_config={
            "kda_layers": kda_layers,
            "full_attn_layers": full_layers,
            "num_heads": 1,
            "head_dim": 2,
            "short_conv_kernel_size": 3,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    model = Model(args)
    model.model.embed_tokens.weight = mx.array(
        [[0.6, -0.8]] + [[0.0, 0.0]] * 7, dtype=mx.float32
    )
    model.model.norm = _Identity()
    model.model.output_attn_res_norm.weight = mx.array([0.85, 1.15])
    model.model.output_attn_res_proj.weight = mx.array([[0.3, -0.2]])

    for index, layer in enumerate(model.model.layers):
        layer.input_layernorm = _Identity()
        layer.post_attention_layernorm = _Identity()
        layer.self_attn = _AttentionScale(0.01 * (index + 1))
        layer.mlp = _MLPScale(0.005 * (index + 1))
        layer.self_attention_res_norm.weight = mx.array([0.9, 1.1])
        layer.mlp_res_norm.weight = mx.array([1.05, 0.95])
        layer.self_attention_res_proj.weight = mx.array(
            [[0.02 * (index + 1), -0.015 * (index + 1)]]
        )
        layer.mlp_res_proj.weight = mx.array(
            [[-0.01 * (index + 1), 0.025 * (index + 1)]]
        )

    assert [
        index + 1
        for index, layer in enumerate(model.model.layers)
        if layer.is_linear_attn
    ] == kda_layers

    hidden = np.array([[0.6, -0.8]], dtype=np.float32)
    blocks = np.zeros((1, 0, 2), dtype=np.float32)
    block_snapshots: dict[int, np.ndarray] = {}
    for index, layer in enumerate(model.model.layers):
        prefix = hidden
        if blocks.shape[1]:
            hidden = _attn_res_numpy(
                prefix,
                blocks,
                _np(layer.self_attention_res_proj.weight).squeeze(0),
                _np(layer.self_attention_res_norm.weight),
            )
        if index % 12 == 0:
            blocks = np.concatenate([blocks, prefix[:, None]], axis=1)
            block_snapshots[index] = blocks.copy()
            prefix = None
        attention = hidden * (0.01 * (index + 1))
        prefix = attention if prefix is None else prefix + attention
        mlp_input = _attn_res_numpy(
            prefix,
            blocks,
            _np(layer.mlp_res_proj.weight).squeeze(0),
            _np(layer.mlp_res_norm.weight),
        )
        hidden = prefix + mlp_input * (0.005 * (index + 1))
    expected = _attn_res_numpy(
        hidden,
        blocks,
        _np(model.model.output_attn_res_proj.weight).squeeze(0),
        _np(model.model.output_attn_res_norm.weight),
    ).reshape(1, 1, 2)

    actual_hidden = model.model.embed_tokens(mx.array([[0]]))
    actual_blocks = mx.zeros((1, 0, 2), dtype=mx.float32)
    actual_block_snapshots: dict[int, np.ndarray] = {}
    for index, layer in enumerate(model.model.layers):
        actual_hidden, actual_blocks = layer(actual_hidden, actual_blocks, None)
        if index % 12 == 0:
            actual_block_snapshots[index] = _np(actual_blocks)
    actual = apply_attn_res(
        actual_hidden.reshape(-1, 2),
        actual_blocks,
        proj=model.model.output_attn_res_proj,
        norm=model.model.output_attn_res_norm,
    ).reshape(1, 1, 2)
    actual_model_forward = model.model(mx.array([[0]]), cache=[None] * 13)

    assert block_snapshots[0].shape[1] == 1
    np.testing.assert_array_equal(
        block_snapshots[0][:, 0], np.array([[0.6, -0.8]], dtype=np.float32)
    )
    assert block_snapshots[12].shape[1] == 2
    np.testing.assert_allclose(
        actual_block_snapshots[0], block_snapshots[0], rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        actual_block_snapshots[12], block_snapshots[12], rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(_np(actual), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        _np(actual_model_forward), expected, rtol=2e-5, atol=2e-5
    )


def test_tiny_model_exposes_streaming_switch_seam() -> None:
    args = ModelArgs.tiny(
        vocab_size=16,
        hidden_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=6,
        moe_intermediate_size=3,
        routed_expert_hidden_size=2,
        num_experts=3,
        num_experts_per_token=2,
        num_shared_experts=2,
        q_lora_rank=3,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=2,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 2,
            "head_dim": 2,
            "short_conv_kernel_size": 3,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    model = Model(args)

    assert hasattr(model.model.layers[1].mlp, "switch_mlp")
