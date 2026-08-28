from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.models.qwen4_omlx import (
    Attention,
    GatedDeltaNet,
    QSAKVCache,
    RotaryEmbedding,
    TextArgs,
)
from mtplx.proj_fusion import configure_fused_projections
from mtplx.qwen4_projection_fusion import install_qwen4_fused_projection_routes


@pytest.fixture(autouse=True)
def _cpu_stream():
    with mx.stream(mx.cpu):
        yield


def _args() -> TextArgs:
    return TextArgs(
        hidden_size=128,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=32,
    )


def test_qwen4_gdn_installs_one_exact_small_row_projection_route(monkeypatch):
    layer = GatedDeltaNet(_args())
    nn.quantize(layer, group_size=32, bits=4)
    names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    original = tuple(getattr(layer, name) for name in names)
    contiguous_calls = 0
    original_contiguous = mx.contiguous

    def counted_contiguous(value):
        nonlocal contiguous_calls
        contiguous_calls += 1
        return original_contiguous(value)

    report = install_qwen4_fused_projection_routes(layer, groups={"gdn"})
    monkeypatch.setattr(mx, "contiguous", counted_contiguous)

    assert report == {"gdn": 1, "attn": 0, "skipped": 0}
    assert type(layer).__name__ == "FusedProjectionGatedDeltaNet"
    assert all(isinstance(getattr(layer, name), nn.QuantizedLinear) for name in names)

    for rows in (1, 4, 8):
        x = mx.random.normal((1, rows, 128), dtype=mx.float16)
        expected = tuple(proj(x) for proj in original)
        actual = layer._project_inputs(x)
        mx.eval(*expected, *actual)
        assert all(mx.array_equal(got, want) for got, want in zip(actual, expected))
    assert contiguous_calls == 0


def test_generic_projection_configuration_selects_qwen4_route(monkeypatch):
    layer = GatedDeltaNet(_args())
    nn.quantize(layer, group_size=32, bits=4)
    monkeypatch.setenv("MTPLX_FUSE_PROJ", "gdn")
    monkeypatch.delenv("MTPLX_PACKED_PROJ_CONCATS", raising=False)

    report = configure_fused_projections(layer)

    assert report["gdn"] == 1
    assert type(layer).__name__ == "FusedProjectionGatedDeltaNet"


def test_qwen4_attention_installs_one_exact_small_row_projection_route():
    layer = Attention(_args())
    nn.quantize(layer, group_size=32, bits=4)
    names = ("q_proj", "k_proj", "v_proj")
    original = tuple(getattr(layer, name) for name in names)

    report = install_qwen4_fused_projection_routes(layer, groups={"attn"})

    assert report == {"gdn": 0, "attn": 1, "skipped": 0}
    assert type(layer).__name__ == "FusedProjectionAttention"
    for rows in (1, 4, 8):
        x = mx.random.normal((1, rows, 128), dtype=mx.float16)
        expected = tuple(proj(x) for proj in original)
        actual = layer._project_qkv(x)
        mx.eval(*expected, *actual)
        assert all(mx.array_equal(got, want) for got, want in zip(actual, expected))


def test_qwen4_attention_reuses_query_rope_with_sparse_indexer():
    args = _args()
    args.indexer_budget = 2
    args.indexer_compress_ratio = 1
    layer = Attention(args)
    cache = QSAKVCache()
    base_rope = RotaryEmbedding(args.head_dim, args.rope_theta)
    rope_calls = []

    def counted_rope(positions):
        rope_calls.append(positions)
        return base_rope(positions)

    x = mx.random.normal((1, 3, args.hidden_size), dtype=mx.float16)
    output = layer(x, counted_rope, None, cache, cache.indexer)
    mx.eval(output)

    # One call for pooled indexer blocks and one shared by indexer-query and
    # attention-query rotations.
    assert len(rope_calls) == 2
