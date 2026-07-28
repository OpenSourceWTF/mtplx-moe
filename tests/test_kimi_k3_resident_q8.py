from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mtplx.expert_runtime import (
    ExpertStreamingRuntime,
    _resident_plan_adjustments,
    proj_quant_plan_discount,
)
from mtplx.expert_streaming_models import proj_quant_covers
from mtplx.resident_loader import _runtime_quantize_projections
from mtplx.runtime import _streaming_preflight_resident_discount


KIMI_K3_MODEL_KEY = "kimi-k3-q1t"


class _Router(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.ones((8, 64), dtype=mx.bfloat16)


class _TinyK3Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attention_res_proj = nn.Linear(64, 1, bias=False)
        self.mlp_res_proj = nn.Linear(64, 1, bias=False)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(64, 64, bias=False)
        self.self_attn.kv_b_proj = nn.Linear(64, 64, bias=False)
        self.self_attn.q_conv1d = nn.Conv1d(64, 64, 3, groups=64, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate = _Router()
        self.mlp.routed_expert_down_proj = nn.Linear(64, 64, bias=False)
        self.mlp.shared_experts = nn.Module()
        self.mlp.shared_experts.up_proj = nn.Linear(64, 64, bias=False)
        self.input_layernorm = nn.RMSNorm(64)


class _TinyK3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(128, 64)
        self.model.output_attn_res_proj = nn.Linear(64, 1, bias=False)
        self.model.layers = [_TinyK3Layer()]
        self.lm_head = nn.Linear(64, 128, bias=False)
        self.mtp = nn.Module()
        self.mtp.draft_proj = nn.Linear(64, 64, bias=False)


def test_kimi_k3_q8_scope_covers_compatible_residents_but_not_router_or_mtp() -> None:
    assert proj_quant_covers(
        "language_model.model.layers.1.block_sparse_moe.routed_expert_down_proj",
        model_key=KIMI_K3_MODEL_KEY,
    )
    assert proj_quant_covers(
        "model.layers.1.mlp.shared_experts.up_proj",
        model_key=KIMI_K3_MODEL_KEY,
    )
    assert proj_quant_covers("model.embed_tokens", model_key=KIMI_K3_MODEL_KEY)
    assert proj_quant_covers("lm_head", model_key=KIMI_K3_MODEL_KEY)
    assert not proj_quant_covers("model.layers.1.mlp.gate", model_key=KIMI_K3_MODEL_KEY)
    assert not proj_quant_covers(
        "model.layers.1.self_attn.q_conv1d", model_key=KIMI_K3_MODEL_KEY
    )
    assert not proj_quant_covers(
        "model.layers.1.self_attn.kv_b_proj",
        model_key=KIMI_K3_MODEL_KEY,
    )
    assert not proj_quant_covers(
        "model.layers.1.self_attention_res_proj",
        model_key=KIMI_K3_MODEL_KEY,
    )
    assert not proj_quant_covers(
        "model.layers.1.mlp_res_proj",
        model_key=KIMI_K3_MODEL_KEY,
    )
    assert not proj_quant_covers(
        "model.output_attn_res_proj",
        model_key=KIMI_K3_MODEL_KEY,
    )
    assert not proj_quant_covers("mtp.draft_proj", model_key=KIMI_K3_MODEL_KEY)


def test_kimi_k3_q8_runtime_quantizes_exact_installed_scope() -> None:
    model = _TinyK3()

    paths = _runtime_quantize_projections(
        model,
        "q8",
        model_key=KIMI_K3_MODEL_KEY,
    )

    assert paths == [
        "model.embed_tokens",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.routed_expert_down_proj",
        "model.layers.0.mlp.shared_experts.up_proj",
        "lm_head",
    ]
    assert isinstance(model.model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(
        model.model.layers[0].mlp.routed_expert_down_proj,
        nn.QuantizedLinear,
    )
    assert isinstance(model.model.layers[0].mlp.gate, _Router)
    assert isinstance(model.mtp.draft_proj, nn.Linear)


def test_kimi_k3_q8_memory_discount_matches_runtime_scope() -> None:
    manifest = SimpleNamespace(
        resident_tensors=(
            SimpleNamespace(
                tensor=(
                    "language_model.model.layers.1.block_sparse_moe."
                    "routed_expert_down_proj.weight"
                ),
                dtype="BF16",
                length=4096,
            ),
            SimpleNamespace(
                tensor="language_model.model.embed_tokens.weight",
                dtype="BF16",
                length=8192,
            ),
            SimpleNamespace(
                tensor="language_model.model.layers.1.block_sparse_moe.gate.weight",
                dtype="BF16",
                length=4096,
            ),
            SimpleNamespace(
                tensor="language_model.model.layers.1.self_attn.q_conv1d.weight",
                dtype="BF16",
                length=4096,
            ),
            SimpleNamespace(
                tensor="language_model.model.layers.1.self_attn.kv_b_proj.weight",
                dtype="BF16",
                length=4096,
            ),
            SimpleNamespace(
                tensor="language_model.model.layers.1.self_attention_res_proj.weight",
                dtype="BF16",
                length=4096,
            ),
            SimpleNamespace(
                tensor="language_model.model.layers.1.mlp_res_proj.weight",
                dtype="BF16",
                length=4096,
            ),
            SimpleNamespace(
                tensor="language_model.model.output_attn_res_proj.weight",
                dtype="BF16",
                length=4096,
            ),
            SimpleNamespace(
                tensor="mtp.draft_proj.weight",
                dtype="BF16",
                length=4096,
            ),
        )
    )

    assert (
        proj_quant_plan_discount(
            manifest,
            "q8",
            model_key=KIMI_K3_MODEL_KEY,
        )
        == 5760
    )
    config = SimpleNamespace(
        model_key=KIMI_K3_MODEL_KEY,
        proj_quant="q8",
        proj_requant=None,
    )
    assert _streaming_preflight_resident_discount(manifest, config) == 5760

    additional, discount = _resident_plan_adjustments(
        SimpleNamespace(is_mixed_official=False, resident_bytes=10_000),
        manifest,
        config,
        SimpleNamespace(resident_bytes=4_240),
        per_layer_record_bytes=None,
    )
    assert additional == 0
    assert discount == 5760


def test_kimi_k3_live_kv_replan_preserves_dynamic_q8_discount() -> None:
    calls: list[dict[str, object]] = []

    class _Config:
        def memory_plan(self, _spec: object, **kwargs: object) -> object:
            calls.append(kwargs)
            return object()

    runtime = object.__new__(ExpertStreamingRuntime)
    runtime.config = _Config()
    runtime.spec = object()
    runtime._additional_resident_bytes = 0
    runtime._resident_discount_bytes = 5760
    runtime._per_layer_record_bytes = None

    runtime._derived_expert_plan(12)

    assert calls == [
        {
            "additional_resident_bytes": 0,
            "resident_discount_bytes": 5760,
            "live_kv_tokens": 12,
            "layer_record_bytes": None,
        }
    ]
