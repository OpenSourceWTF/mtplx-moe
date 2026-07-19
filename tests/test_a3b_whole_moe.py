"""Correct-by-construction whole-MoE routing for exact A3B small rows."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.a3b_whole_moe import (
    A3BWholeMoeConfigError,
    prepare_a3b_whole_moe,
)


_LAYER_TYPES = tuple(
    "linear_attention" if index % 4 != 3 else "full_attention"
    for index in range(40)
)


class _ArraySpec:
    def __init__(self, shape, dtype=mx.bfloat16) -> None:
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype


class _Projection:
    def __init__(
        self,
        weight_shape,
        *,
        bits: int | None = None,
        group_size: int | None = None,
        scales_shape=None,
        biases_shape=None,
    ) -> None:
        self.weight = _ArraySpec(
            weight_shape, mx.uint32 if bits is not None else mx.bfloat16
        )
        if bits is not None:
            self.bits = bits
            self.group_size = group_size
            self.mode = "affine"
            self.scales = _ArraySpec(scales_shape)
            self.biases = _ArraySpec(biases_shape)

    def __contains__(self, name: str) -> bool:
        return hasattr(self, name)

    def __getitem__(self, name: str):
        return getattr(self, name)


def _target_block():
    return _sparse_block(
        gate=_Projection(
            (256, 512),
            bits=8,
            group_size=64,
            scales_shape=(256, 32),
            biases_shape=(256, 32),
        ),
        routed_group_size=64,
        shared_quantized=True,
        scalar_gate=_Projection(
            (1, 512),
            bits=8,
            group_size=64,
            scales_shape=(1, 32),
            biases_shape=(1, 32),
        ),
    )


def _mtp_block():
    return _sparse_block(
        gate=_Projection((256, 2048)),
        routed_group_size=32,
        shared_quantized=False,
        scalar_gate=_Projection((1, 2048)),
    )


def _quantized_routed_projection(output: int, *, group_size: int):
    return _Projection(
        (256, output, 2048 * 4 // 32),
        bits=4,
        group_size=group_size,
        scales_shape=(256, output, 2048 // group_size),
        biases_shape=(256, output, 2048 // group_size),
    )


def _sparse_block(*, gate, routed_group_size: int, shared_quantized: bool, scalar_gate):
    switch_mlp = SimpleNamespace(
        gate_proj=_quantized_routed_projection(512, group_size=routed_group_size),
        up_proj=_quantized_routed_projection(512, group_size=routed_group_size),
        down_proj=_Projection(
            (256, 2048, 512 * 4 // 32),
            bits=4,
            group_size=routed_group_size,
            scales_shape=(256, 2048, 512 // routed_group_size),
            biases_shape=(256, 2048, 512 // routed_group_size),
        ),
    )
    if shared_quantized:
        shared_expert = SimpleNamespace(
            gate_proj=_Projection(
                (512, 256), bits=4, group_size=64,
                scales_shape=(512, 32), biases_shape=(512, 32),
            ),
            up_proj=_Projection(
                (512, 256), bits=4, group_size=64,
                scales_shape=(512, 32), biases_shape=(512, 32),
            ),
            down_proj=_Projection(
                (2048, 64), bits=4, group_size=64,
                scales_shape=(2048, 8), biases_shape=(2048, 8),
            ),
        )
    else:
        shared_expert = SimpleNamespace(
            gate_proj=_Projection((512, 2048)),
            up_proj=_Projection((512, 2048)),
            down_proj=_Projection((2048, 512)),
        )
    return SimpleNamespace(
        gate=gate,
        switch_mlp=switch_mlp,
        shared_expert=shared_expert,
        shared_expert_gate=scalar_gate,
        num_experts=256,
        top_k=8,
        norm_topk_prob=True,
        sharding_group=None,
    )


def _exact_config():
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "layer_types": list(_LAYER_TYPES),
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "norm_topk_prob": True,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "mtp_num_hidden_layers": 1,
        },
    }


def _exact_model():
    target_blocks = [_target_block() for _ in range(40)]
    target_layers = []
    for index, block in enumerate(target_blocks):
        is_linear = _LAYER_TYPES[index] == "linear_attention"
        layer = SimpleNamespace(
            is_linear=is_linear,
            mlp=block,
            post_attention_layernorm=SimpleNamespace(
                weight=_ArraySpec((2048,))
            ),
        )
        if is_linear:
            layer.linear_attn = object()
        else:
            layer.self_attn = object()
        target_layers.append(layer)
    mtp_block = _mtp_block()
    mtp_layer = SimpleNamespace(
        mlp=mtp_block,
        self_attn=object(),
        post_attention_layernorm=SimpleNamespace(weight=_ArraySpec((2048,))),
    )
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=target_layers)),
        mtp=SimpleNamespace(layers=[mtp_layer]),
    )
    return model, target_blocks, [mtp_block]


def test_flag_off_returns_no_plan_and_preserves_all_block_classes(monkeypatch):
    monkeypatch.delenv("MTPLX_A3B_WHOLE_MOE_FUSION", raising=False)
    model, targets, mtp = _exact_model()
    original_classes = tuple(type(block) for block in (*targets, *mtp))

    assert prepare_a3b_whole_moe(model, config=_exact_config()) is None
    assert tuple(type(block) for block in (*targets, *mtp)) == original_classes


def test_exact_checkpoint_builds_40_target_and_one_mtp_binding(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    model, _, _ = _exact_model()

    plan = prepare_a3b_whole_moe(model, config=_exact_config())

    assert len(plan.target_bindings) == 40
    assert len(plan.mtp_bindings) == 1
    assert {binding.variant for binding in plan.target_bindings} == {
        "target_q8g64_q4g64"
    }
    assert plan.mtp_bindings[0].variant == "mtp_dense_q4g32_dense"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda config, model: config.update(model_type="qwen3_5"),
            "model architecture",
        ),
        (
            lambda config, model: config["text_config"].update(hidden_size=4096),
            "topology",
        ),
        (
            lambda config, model: model.language_model.model.layers.pop(),
            "40 target",
        ),
        (
            lambda config, model: model.mtp.layers.clear(),
            "one MTP",
        ),
        (
            lambda config, model: setattr(
                model.language_model.model.layers[0].mlp, "top_k", 4
            ),
            "top-k 8",
        ),
        (
            lambda config, model: setattr(
                model.language_model.model.layers[0].mlp,
                "norm_topk_prob",
                False,
            ),
            "normalization",
        ),
        (
            lambda config, model: setattr(
                model.language_model.model.layers[0].mlp,
                "sharding_group",
                object(),
            ),
            "sharding",
        ),
        (
            lambda config, model: setattr(
                model.language_model.model.layers[0].post_attention_layernorm.weight,
                "dtype",
                mx.float32,
            ),
            "BF16 hidden",
        ),
        (
            lambda config, model: setattr(
                model.language_model.model.layers[0].mlp.gate,
                "group_size",
                32,
            ),
            "target router",
        ),
        (
            lambda config, model: setattr(
                model.language_model.model.layers[0].mlp.switch_mlp.gate_proj.weight,
                "shape",
                (256, 512, 128),
            ),
            "target routed gate",
        ),
        (
            lambda config, model: setattr(
                model.mtp.layers[0].mlp.switch_mlp.down_proj,
                "group_size",
                64,
            ),
            "MTP routed down",
        ),
        (
            lambda config, model: setattr(
                model.mtp.layers[0].mlp.shared_expert.up_proj.weight,
                "dtype",
                mx.float32,
            ),
            "MTP shared up",
        ),
    ],
)
def test_external_contract_mismatch_fails_before_install(
    monkeypatch,
    mutate,
    match,
):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    config = _exact_config()
    model, targets, mtp = _exact_model()
    original_classes = tuple(type(block) for block in (*targets, *mtp))
    mutate(config, model)

    with pytest.raises(A3BWholeMoeConfigError, match=match):
        prepare_a3b_whole_moe(model, config=config)

    assert tuple(type(block) for block in (*targets, *mtp)) == original_classes
