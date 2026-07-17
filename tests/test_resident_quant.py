"""Load-time resident quantization: scope, config validation, flag plumbing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    parse_memory_bytes,
    resident_quant_plan_discount,
)
from mtplx.expert_streaming_models import (
    get_model_spec,
    plan_expert_memory,
    resident_quant_covers,
    resident_quant_kept_bytes,
)
from mtplx.models.hy3_mlx import Model as Hy3Model
from mtplx.models.hy3_mlx import ModelArgs as Hy3Args
from mtplx.resident_loader import ResidentLoadError, _runtime_quantize_resident


def _tiny_args() -> Hy3Args:
    return Hy3Args(
        model_type="hy_v3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        num_shared_experts=1,
        first_k_dense_replace=1,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=16,
        router_scaling_factor=2.0,
    )


def _load_benchmark_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_q2_mtp_depth_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("bench_depth_matrix_rq", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bench_depth_matrix_rq", module)
    spec.loader.exec_module(module)
    return module


def _config_kwargs(**overrides):
    kwargs = {
        "model_key": "hy3-expert-q2",
        "memory_limit_bytes": 96 * 1024**3,
        "max_live_kv_tokens": 4096,
    }
    kwargs.update(overrides)
    return kwargs


def test_resident_quant_config_accepts_supported_modes() -> None:
    for mode in (None, "q8", "q4"):
        config = ExpertStreamingConfig(**_config_kwargs(resident_quant=mode))
        assert config.resident_quant == mode


def test_resident_quant_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="resident_quant"):
        ExpertStreamingConfig(**_config_kwargs(resident_quant="int8"))


def test_runtime_quantize_scopes_attention_shared_and_dense_mlp() -> None:
    args = _tiny_args()
    model = Hy3Model(args)
    original_q_weight = model.model.layers[1].self_attn.q_proj.weight.astype(
        mx.float32
    )

    quantized = _runtime_quantize_resident(model, "q4")

    expected = set()
    for layer in range(args.num_hidden_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            expected.add(f"model.layers.{layer}.self_attn.{proj}")
    # Layer 0 is dense (first_k_dense_replace=1); layer 1 carries the
    # shared expert.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        expected.add(f"model.layers.0.mlp.{proj}")
        expected.add(f"model.layers.1.mlp.shared_mlp.{proj}")
    assert set(quantized) == expected

    for path in expected:
        module = model
        for part in path.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        assert isinstance(module, nn.QuantizedLinear), path
        assert module.bits == 4 and module.group_size == 64

    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, nn.QuantizedLinear)
    assert isinstance(model.model.embed_tokens, nn.Embedding)

    # Dequantized projection stays close to the BF16 original.
    x = mx.random.normal((1, 1, args.hidden_size)).astype(mx.bfloat16)
    reference = mx.matmul(x.astype(mx.float32), original_q_weight.T)
    actual = model.model.layers[1].self_attn.q_proj(x).astype(mx.float32)
    mx.eval(reference, actual)
    error = float(
        (mx.abs(actual - reference).mean() / (mx.abs(reference).mean() + 1e-6)).item()
    )
    assert error < 0.2, f"q4 dequantization drifted: {error}"


def test_runtime_quantize_rejects_matchless_model() -> None:
    class Bare(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(8, 8, bias=False)

    with pytest.raises(ResidentLoadError, match="matched no resident"):
        _runtime_quantize_resident(Bare(), "q4")


@pytest.mark.parametrize(
    ("path", "covered"),
    [
        ("model.layers.3.self_attn.q_proj", True),
        ("model.layers.3.self_attn.o_proj", True),
        ("model.layers.1.mlp.shared_mlp.gate_proj", True),
        ("model.layers.1.mlp.shared_mlp.gate_up_proj", True),
        ("model.layers.0.mlp.down_proj", True),
        ("model.layers.1.mlp.router.gate", False),
        ("model.layers.1.mlp.gate", False),
        ("lm_head", False),
        ("model.embed_tokens", False),
        ("model.layers.1.input_layernorm", False),
        ("model.layers.80.eh_proj", False),
    ],
)
def test_resident_quant_scope(path: str, covered: bool) -> None:
    assert resident_quant_covers(path) is covered


def test_resident_quant_kept_bytes_matches_group64_affine() -> None:
    # 1 MiB of BF16 = 512 Ki elements; q4 packs to 256 KiB + 32 KiB of
    # BF16 scales and biases (one pair per 64-element group).
    assert resident_quant_kept_bytes(1024 * 1024, "q4") == 256 * 1024 + 32 * 1024
    assert resident_quant_kept_bytes(1024 * 1024, "q8") == 512 * 1024 + 32 * 1024


def test_plan_discount_shrinks_fixed_bytes_exactly() -> None:
    spec = get_model_spec("hy3-expert-q2")
    base = plan_expert_memory(
        spec, total_limit_bytes=110 * 1024**3, context_tokens=4096
    )
    discounted = plan_expert_memory(
        spec,
        total_limit_bytes=110 * 1024**3,
        context_tokens=4096,
        resident_discount_bytes=10 * 1024**3,
    )
    assert base.fixed_bytes - discounted.fixed_bytes == 10 * 1024**3
    with pytest.raises(ValueError, match="exceeds the resident footprint"):
        plan_expert_memory(
            spec,
            total_limit_bytes=110 * 1024**3,
            context_tokens=4096,
            resident_discount_bytes=spec.resident_bytes,
        )


def test_manifest_plan_discount_scopes_and_prices_tensors() -> None:
    from types import SimpleNamespace

    tensors = (
        SimpleNamespace(
            tensor="model.layers.1.self_attn.q_proj.weight",
            dtype="BF16",
            length=1024 * 1024,
        ),
        SimpleNamespace(  # router stays exact
            tensor="model.layers.1.mlp.router.gate.weight",
            dtype="BF16",
            length=1024 * 1024,
        ),
        SimpleNamespace(  # non-BF16 never discounts
            tensor="model.layers.2.self_attn.q_proj.weight",
            dtype="F32",
            length=1024 * 1024,
        ),
        SimpleNamespace(  # biases keep loaded precision
            tensor="model.layers.1.self_attn.q_proj.bias",
            dtype="BF16",
            length=4096,
        ),
    )
    manifest = SimpleNamespace(resident_tensors=tensors)
    assert resident_quant_plan_discount(manifest, None) == 0
    expected = 1024 * 1024 - resident_quant_kept_bytes(1024 * 1024, "q4")
    assert resident_quant_plan_discount(manifest, "q4") == expected


def test_resident_quant_survives_benchmark_option_pipeline() -> None:
    """CLI vector -> parser -> runtime options -> config factory, mirroring
    the island-count guard (option keys have silently dropped between
    argparse and ExpertStreamingConfig before)."""

    from types import SimpleNamespace

    bench = _load_benchmark_module()
    parser = None
    import argparse as _argparse

    for name in dir(bench):
        fn = getattr(bench, name)
        if callable(fn) and name.endswith("parser"):
            candidate = fn()
            if isinstance(candidate, _argparse.ArgumentParser):
                parser = candidate
                break
    assert parser is not None, "benchmark parser factory not found"
    args = parser.parse_args(
        [
            "--model", "hy3-q2",
            "--hy3-q2-model-root", "/tmp",
            "--memory-limit", "96GiB",
            "--resident-quant", "q4",
        ]
    )
    options = {
        **bench.DEFAULT_RUNTIME_OPTIONS,
        "trace_routes": False,
        **bench._runtime_options_from_args(args),
    }
    assert options["resident_quant"] == "q4"
    apis = SimpleNamespace(
        config_factory=ExpertStreamingConfig,
        parse_memory_bytes=parse_memory_bytes,
    )
    config = bench._runtime_config(apis, "hy3-expert-q2", options)
    assert config.resident_quant == "q4"
