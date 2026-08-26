from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.models.qwen4_exp import (
    Qwen4Cache,
    Qwen4GatedDeltaNet,
    Qwen4GatedResidual,
)


@dataclass(frozen=True)
class _TinyConfig:
    hidden_size: int = 5
    hc_count: int = 2
    hc_lowrank: int = 3
    rms_norm_eps: float = 1e-6
    linear_num_key_heads: int = 2
    linear_num_value_heads: int = 6
    linear_key_head_dim: int = 4
    linear_value_head_dim: int = 4
    linear_conv_kernel_dim: int = 4
    hidden_act: str = "silu"
    output_gate_type: str = "sigmoid"
    mamba_ssm_dtype: str = "float32"


@pytest.fixture
def mlx_cpu() -> None:
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous_device)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(x, np.float32(0.0))


def _linear(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return x @ weight.T


def _residual_weights(config: _TinyConfig) -> dict[str, np.ndarray]:
    width = config.hc_count * config.hidden_size
    return {
        "hc_norm.weight": np.linspace(-0.25, 0.2, width, dtype=np.float32),
        "input_mix_weight_down.weight": np.linspace(
            -0.21,
            0.27,
            config.hc_lowrank * width,
            dtype=np.float32,
        ).reshape(config.hc_lowrank, width),
        "input_mix_weight_up.weight": np.linspace(
            0.19,
            -0.17,
            width * config.hc_lowrank,
            dtype=np.float32,
        ).reshape(width, config.hc_lowrank),
        "block_inject_weight.weight": np.linspace(
            -0.13,
            0.23,
            config.hc_count * width,
            dtype=np.float32,
        ).reshape(config.hc_count, width),
    }


def _residual_reference(
    x: np.ndarray,
    weights: dict[str, np.ndarray],
    config: _TinyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    streams = x.reshape(*x.shape[:-1], config.hc_count, config.hidden_size)
    variance = np.mean(streams.astype(np.float32) ** 2, axis=-1, keepdims=True)
    normed = streams.astype(np.float32) / np.sqrt(variance + config.rms_norm_eps)
    normed *= (1.0 + weights["hc_norm.weight"]).reshape(
        config.hc_count, config.hidden_size
    )
    normed_flat = normed.reshape(x.shape)
    mix_hidden = _silu(
        _linear(normed_flat, weights["input_mix_weight_down.weight"]) / config.hc_count
    )
    mix = _sigmoid(_linear(mix_hidden, weights["input_mix_weight_up.weight"]))
    mixed = np.mean(
        mix.reshape(streams.shape) * normed,
        axis=-2,
    )
    inject = 2.0 * _sigmoid(
        _linear(normed_flat, weights["block_inject_weight.weight"]) / config.hc_count
    )
    written = streams + mixed[..., None, :] * inject[..., :, None]
    return mixed, written.reshape(x.shape)


def test_gated_residual_read_write_matches_fixed_numpy_reference(mlx_cpu) -> None:
    config = _TinyConfig()
    module = Qwen4GatedResidual(config)
    weights = _residual_weights(config)
    module.load_weights(
        [(name, mx.array(value)) for name, value in weights.items()], strict=True
    )
    x_np = np.linspace(
        -1.5,
        1.75,
        2 * 3 * config.hc_count * config.hidden_size,
        dtype=np.float32,
    ).reshape(2, 3, -1)
    x = mx.array(x_np)

    read, residual, inject = module.read(x)
    actual_write = module.write(residual, read, inject)
    expected_read, expected_write = _residual_reference(x_np, weights, config)
    mx.eval(read, actual_write)

    np.testing.assert_allclose(np.asarray(read), expected_read, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        np.asarray(actual_write), expected_write, rtol=2e-5, atol=2e-5
    )


def test_gated_residual_final_reduce_uses_same_trained_read_gate(mlx_cpu) -> None:
    config = _TinyConfig()
    module = Qwen4GatedResidual(config, use_combine=False)
    weights = _residual_weights(config)
    weights.pop("block_inject_weight.weight")
    module.load_weights(
        [(name, mx.array(value)) for name, value in weights.items()], strict=True
    )
    x_np = np.linspace(-0.8, 1.1, 20, dtype=np.float32).reshape(1, 2, 10)
    expected, _ = _residual_reference(
        x_np,
        {**weights, "block_inject_weight.weight": np.zeros((2, 10), np.float32)},
        config,
    )
    actual = module(x=mx.array(x_np))
    mx.eval(actual)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)


def _gdn_weights(config: _TinyConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260826)
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    conv_dim = 2 * key_dim + value_dim
    weights = {
        "in_proj_qkv.weight": rng.uniform(
            -0.17, 0.17, (conv_dim, config.hidden_size)
        ).astype(np.float32),
        "in_proj_z.weight": rng.uniform(
            -0.21, 0.21, (value_dim, config.hidden_size)
        ).astype(np.float32),
        "in_proj_b.weight": rng.uniform(
            -0.11, 0.11, (config.linear_num_value_heads, config.hidden_size)
        ).astype(np.float32),
        "in_proj_a.weight": rng.uniform(
            -0.13, 0.13, (config.linear_num_value_heads, config.hidden_size)
        ).astype(np.float32),
        "conv1d.weight": rng.uniform(
            -0.3,
            0.3,
            (conv_dim, config.linear_conv_kernel_dim, 1),
        ).astype(np.float32),
        "A_log": np.log(
            np.linspace(0.35, 1.4, config.linear_num_value_heads, dtype=np.float32)
        ),
        "dt_bias": np.linspace(
            -0.4, 0.35, config.linear_num_value_heads, dtype=np.float32
        ),
        "norm.weight": np.linspace(
            0.75, 1.25, config.linear_value_head_dim, dtype=np.float32
        ),
        "out_proj.weight": rng.uniform(
            -0.16, 0.16, (config.hidden_size, value_dim)
        ).astype(np.float32),
    }
    return weights


def _depthwise_causal_conv(
    x: np.ndarray,
    weight: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    batch, length, channels = x.shape
    padded = np.concatenate(
        (np.zeros((batch, kernel_size - 1, channels), dtype=x.dtype), x), axis=1
    )
    out = np.empty_like(x)
    per_channel = weight[..., 0]
    for token in range(length):
        window = padded[:, token : token + kernel_size, :]
        out[:, token, :] = np.sum(window * per_channel.T[None, :, :], axis=1)
    return out


def _gdn_reference(
    hidden: np.ndarray,
    weights: dict[str, np.ndarray],
    config: _TinyConfig,
) -> np.ndarray:
    batch, length, _ = hidden.shape
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    mixed = _linear(hidden, weights["in_proj_qkv.weight"])
    mixed = _silu(
        _depthwise_causal_conv(
            mixed, weights["conv1d.weight"], config.linear_conv_kernel_dim
        )
    )
    query, key, value = np.split(mixed, (key_dim, 2 * key_dim), axis=-1)
    query = query.reshape(
        batch,
        length,
        config.linear_num_key_heads,
        config.linear_key_head_dim,
    )
    key = key.reshape(query.shape)
    value = value.reshape(
        batch,
        length,
        config.linear_num_value_heads,
        config.linear_value_head_dim,
    )
    repeat = config.linear_num_value_heads // config.linear_num_key_heads
    query = np.repeat(query, repeat, axis=2).astype(np.float32)
    key = np.repeat(key, repeat, axis=2).astype(np.float32)
    query *= 1.0 / np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + 1e-6)
    query *= config.linear_key_head_dim**-0.5
    key *= 1.0 / np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + 1e-6)

    b = _linear(hidden, weights["in_proj_b.weight"]).astype(np.float32)
    a = _linear(hidden, weights["in_proj_a.weight"]).astype(np.float32)
    beta = _sigmoid(b)
    decay = -np.exp(weights["A_log"].astype(np.float32)) * _softplus(
        a + weights["dt_bias"].astype(np.float32)
    )
    state = np.zeros(
        (
            batch,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        ),
        dtype=np.float32,
    )
    recurrent_out = np.empty_like(value, dtype=np.float32)
    for token in range(length):
        q_t = query[:, token]
        k_t = key[:, token]
        v_t = value[:, token].astype(np.float32)
        state *= np.exp(decay[:, token])[..., None, None]
        prediction = np.sum(state * k_t[..., None], axis=-2)
        state += (
            beta[:, token, :, None, None]
            * k_t[..., None]
            * (v_t - prediction)[..., None, :]
        )
        recurrent_out[:, token] = np.sum(state * q_t[..., None], axis=-2)

    variance = np.mean(recurrent_out**2, axis=-1, keepdims=True)
    normalized = recurrent_out / np.sqrt(variance + config.rms_norm_eps)
    normalized *= weights["norm.weight"]
    z = _linear(hidden, weights["in_proj_z.weight"]).reshape(value.shape)
    normalized *= _sigmoid(z.astype(np.float32))
    return _linear(
        normalized.reshape(batch, length, value_dim), weights["out_proj.weight"]
    )


def _make_gdn(
    config: _TinyConfig,
    *,
    recurrent_lane: str | None = None,
) -> tuple[Qwen4GatedDeltaNet, dict[str, np.ndarray]]:
    module = Qwen4GatedDeltaNet(
        config,
        layer_idx=0,
        recurrent_lane=recurrent_lane,
    )
    weights = _gdn_weights(config)
    module.load_weights(
        [(name, mx.array(value)) for name, value in weights.items()], strict=True
    )
    return module, weights


def test_gdn_fixed_array_matches_independent_official_numpy_equations(mlx_cpu) -> None:
    config = _TinyConfig()
    module, weights = _make_gdn(config)
    hidden_np = np.linspace(-0.9, 1.2, 2 * 7 * 5, dtype=np.float32).reshape(2, 7, 5)
    expected = _gdn_reference(hidden_np, weights, config)
    actual = module(mx.array(hidden_np), cache=None)
    mx.eval(actual)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=3e-4, atol=3e-4)


def test_gdn_chunk_then_decode_matches_single_pass_and_rolls_back(mlx_cpu) -> None:
    config = _TinyConfig()
    module, _ = _make_gdn(config)
    hidden_np = np.linspace(-1.1, 1.3, 8 * 5, dtype=np.float32).reshape(1, 8, 5)

    expected = module(mx.array(hidden_np[:, :7]), cache=None)
    cache = Qwen4Cache.tiny(gdn_layers=(0,), qsa_layers=())
    prefix = module(mx.array(hidden_np[:, :6]), cache=cache)
    decode = module(mx.array(hidden_np[:, 6:7]), cache=cache)
    actual = mx.concatenate((prefix, decode), axis=1)
    mx.eval(expected, actual)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=3e-4, atol=3e-4
    )

    snapshot = cache.snapshot()
    digest = snapshot.digest()
    module(mx.array(hidden_np[:, 7:8]), cache=cache)
    advanced = cache.snapshot().digest()
    assert advanced != digest

    cache.restore(snapshot)
    assert cache.snapshot().digest() == digest

    module(mx.array(hidden_np[:, 7:8]), cache=cache)
    cache.trim(1, snapshot=snapshot)
    assert cache.snapshot().digest() == digest


def test_cache_snapshot_digest_covers_conv_recurrence_dtype_shape_and_offset(
    mlx_cpu,
) -> None:
    config = _TinyConfig()
    module, _ = _make_gdn(config)
    cache = Qwen4Cache.tiny(gdn_layers=(0,), qsa_layers=())
    module(mx.ones((1, 2, config.hidden_size), dtype=mx.float32), cache=cache)
    snapshot = cache.snapshot()
    mx.eval(*snapshot.arrays())

    digest = snapshot.digest()
    assert len(digest) == hashlib.sha256().digest_size * 2
    assert snapshot.gdn[0].conv_state.shape == (
        1,
        config.linear_conv_kernel_dim - 1,
        2 * config.linear_num_key_heads * config.linear_key_head_dim
        + config.linear_num_value_heads * config.linear_value_head_dim,
    )
    assert snapshot.gdn[0].recurrent_state.dtype == mx.float32
    assert snapshot.gdn[0].offset == 2


@pytest.mark.parametrize(
    ("head_k_dim", "num_k_heads", "num_v_heads", "head_v_dim"),
    [
        (32, 2, 6, 4),
        (128, 16, 48, 128),
    ],
)
def test_metal_recurrence_matches_logical_kv_ops_lane(
    head_k_dim: int,
    num_k_heads: int,
    num_v_heads: int,
    head_v_dim: int,
) -> None:
    config = replace(
        _TinyConfig(),
        linear_key_head_dim=head_k_dim,
        linear_num_key_heads=num_k_heads,
        linear_num_value_heads=num_v_heads,
        linear_value_head_dim=head_v_dim,
    )
    ops = Qwen4GatedDeltaNet(config, layer_idx=0, recurrent_lane="ops")
    metal = Qwen4GatedDeltaNet(config, layer_idx=0, recurrent_lane="metal")
    rng = np.random.default_rng(20260827 + head_k_dim)
    shape_k = (1, 3, num_k_heads, head_k_dim)
    query = rng.normal(0.0, 0.05, shape_k).astype(np.float32)
    key = rng.normal(0.0, 0.05, shape_k).astype(np.float32)
    query /= np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + 1e-6)
    query *= head_k_dim**-0.5
    key /= np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    value = rng.normal(
        0.0,
        0.1,
        (1, 3, num_v_heads, head_v_dim),
    ).astype(np.float32)
    decay = rng.uniform(0.7, 0.99, (1, 3, num_v_heads)).astype(np.float32)
    beta = rng.uniform(0.15, 0.85, (1, 3, num_v_heads)).astype(np.float32)
    state = rng.normal(
        0.0,
        0.03,
        (1, num_v_heads, head_k_dim, head_v_dim),
    ).astype(np.float32)
    mask = mx.array([[True, False, True]])
    inputs = tuple(mx.array(item) for item in (query, key, value, decay, beta, state))

    expected_output, expected_state = ops._recurrent(*inputs, mask)
    actual_output, actual_state = metal._recurrent(*inputs, mask)
    mx.eval(expected_output, expected_state, actual_output, actual_state)

    np.testing.assert_allclose(
        np.asarray(actual_output),
        np.asarray(expected_output),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_state),
        np.asarray(expected_state),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"linear_num_value_heads": 5}, "divisible"),
        ({"linear_conv_kernel_dim": 0}, "kernel"),
        ({"hidden_act": "gelu"}, "silu"),
        ({"output_gate_type": "silu"}, "sigmoid"),
        ({"mamba_ssm_dtype": "bfloat16"}, "float32"),
    ],
)
def test_gdn_rejects_invalid_topology_at_construction(
    changes: dict[str, object], match: str
) -> None:
    values = dict(_TinyConfig().__dict__)
    values.update(changes)
    with pytest.raises(ValueError, match=match):
        Qwen4GatedDeltaNet(_TinyConfig(**values), layer_idx=0)


def test_cache_rejects_ambiguous_or_duplicate_layer_ownership() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        Qwen4Cache.tiny(gdn_layers=(0,), qsa_layers=(0,))
    with pytest.raises(ValueError, match="unique"):
        Qwen4Cache.tiny(gdn_layers=(0, 0), qsa_layers=())
