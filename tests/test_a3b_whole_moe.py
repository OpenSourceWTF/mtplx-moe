"""Correct-by-construction whole-MoE routing for exact A3B small rows."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mtplx.a3b_whole_moe as whole_moe_module
from mtplx.a3b_whole_moe import (
    A3BWholeMoeConfigError,
    A3BWholeMoeRouteError,
    install_a3b_whole_moe,
    prepare_a3b_whole_moe,
)
from mtplx.kernels import a3b_whole_moe as kernel_module


_LAYER_TYPES = tuple(
    "linear_attention" if index % 4 != 3 else "full_attention"
    for index in range(40)
)


class _ArraySpec:
    def __init__(self, shape, dtype=mx.bfloat16) -> None:
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype

    def reshape(self, *shape):
        return _ArraySpec(shape, self.dtype)


class _ResultSpec(_ArraySpec):
    def __init__(self, label: str, shape) -> None:
        super().__init__(shape)
        self.label = label

    def reshape(self, *shape):
        return _ResultSpec(self.label, shape)


class _FakeSparseBlock:
    def __init__(self, **attributes) -> None:
        vars(self).update(attributes)
        self.stock_calls = []

    def __call__(self, value):
        self.stock_calls.append(value)
        return _ResultSpec("stock", value.shape)


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
        gate_up_proj=_quantized_routed_projection(
            1024, group_size=routed_group_size
        ),
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
            gate_up_proj=_Projection(
                (1024, 256), bits=4, group_size=64,
                scales_shape=(1024, 32), biases_shape=(1024, 32),
            ),
            down_proj=_Projection(
                (2048, 64), bits=4, group_size=64,
                scales_shape=(2048, 8), biases_shape=(2048, 8),
            ),
        )
    else:
        shared_expert = SimpleNamespace(
            gate_up_proj=_Projection((1024, 2048)),
            down_proj=_Projection((2048, 512)),
        )
    return _FakeSparseBlock(
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
    assert plan.target_bindings[0].routed_gate_up.weight.shape == (
        256,
        1024,
        256,
    )
    assert plan.target_bindings[0].shared_gate_up.weight.shape == (1024, 256)
    assert plan.mtp_bindings[0].shared_gate_up.weight.shape == (1024, 2048)


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
                model.language_model.model.layers[0].mlp.switch_mlp.gate_up_proj.weight,
                "shape",
                (256, 1024, 128),
            ),
            "target routed gate/up",
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
                model.mtp.layers[0].mlp.shared_expert.gate_up_proj.weight,
                "dtype",
                mx.float32,
            ),
            "MTP shared gate/up",
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


def test_fixed_kernel_sources_encode_exact_geometry_without_hot_validation():
    sources = kernel_module.all_whole_moe_sources()
    assert len(sources) == 9
    for source in sources.values():
        assert "constexpr uint HIDDEN = 2048" in source
        assert "constexpr uint EXPERTS = 256" in source
        assert "constexpr uint TOP_K = 8" in source
        assert "constexpr uint INTERMEDIATE = 512" in source
        for forbidden in (
            "getenv",
            "dtype",
            "shape",
            "eligible",
            "fallback",
            "lane_disabled",
            "record_",
            "counter",
        ):
            assert forbidden not in source
    for name, source in sources.items():
        if "stage1" not in name:
            assert "threadgroup_barrier" not in source


class _CapturedKernel:
    def __init__(self) -> None:
        self.call = None

    def __call__(self, **kwargs):
        self.call = kwargs
        return tuple(
            _ArraySpec(shape, dtype)
            for shape, dtype in zip(
                kwargs["output_shapes"], kwargs["output_dtypes"]
            )
        )


def test_target_m2_stage2_is_row_paired_with_fixed_288_threadgroups(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    kernel = _CapturedKernel()
    monkeypatch.setattr(
        kernel_module,
        "_build_target_m2_stage2_kernel",
        lambda: kernel,
    )
    model, _, _ = _exact_model()
    binding = prepare_a3b_whole_moe(
        model,
        config=_exact_config(),
    ).target_bindings[0]

    output = kernel_module.target_m2_stage2(
        _ArraySpec((2, 2048)),
        _ArraySpec((2, 8), mx.uint32),
        binding,
    )

    assert output.shape == (2, 9, 512)
    assert kernel.call["grid"] == (288 * 128, 1, 1)
    assert kernel.call["threadgroup"] == (128, 1, 1)
    assert kernel.call["output_shapes"] == [(2, 9, 512)]
    assert kernel.call["output_dtypes"] == [mx.bfloat16]


def test_fixed_entrypoint_launch_table(monkeypatch):
    expected = {
        "target_m1_stage1": ((256, 1, 1), (256, 1, 1)),
        "target_m2_stage1": ((512, 1, 1), (256, 1, 1)),
        "mtp_m1_stage1": ((256, 1, 1), (256, 1, 1)),
        "target_m1_stage2": ((288 * 128, 1, 1), (128, 1, 1)),
        "target_m2_stage2": ((288 * 128, 1, 1), (128, 1, 1)),
        "mtp_m1_stage2": ((288 * 128, 1, 1), (128, 1, 1)),
        "target_m1_stage3": ((128 * 128, 1, 1), (128, 1, 1)),
        "target_m2_stage3": ((128 * 128, 1, 1), (128, 1, 1)),
        "mtp_m1_stage3": ((128 * 128, 1, 1), (128, 1, 1)),
    }
    assert kernel_module.whole_moe_launch_table() == expected


def test_stage1_sources_encode_router_softmax_top8_and_score_rounding():
    sources = kernel_module.all_whole_moe_sources()
    for name in ("target_m1_stage1", "target_m2_stage1"):
        source = sources[name]
        assert "qdot8_affine" in source
        assert "constexpr uint ROUTER_GROUP = 64" in source
        assert "threadgroup bfloat router_logits[ROWS * EXPERTS]" in source
        assert "metal::exp" in source
        assert "simd_max" in source
        assert "bfloat rounded_denominator = bfloat(0.0f)" in source
        assert "candidate_probability == winner_probability" in source
    source = sources["mtp_m1_stage1"]
    assert "dense_bf16_dot" in source
    assert "threadgroup bfloat router_logits[ROWS * EXPERTS]" in source
    assert "metal::exp" in source
    assert "bfloat rounded_denominator = bfloat(0.0f)" in source


def test_stage2_sources_encode_selected_q4_and_exact_bf16_swiglu():
    sources = kernel_module.all_whole_moe_sources()
    for name in ("target_m1_stage2", "target_m2_stage2"):
        source = sources[name]
        assert "constexpr uint ROUTED_GROUP = 64" in source
        assert "qdot4_affine" in source
        assert "sigmoid_mlx_exact" in source
        assert "bfloat gate_value = bfloat(gate_sum)" in source
        assert "bfloat up_value = bfloat(up_sum)" in source
        assert "activations[output_index] = bfloat(silu * up_value)" in source
        assert "routed_gate_up_weight" in source
        assert "shared_gate_up_weight" in source
        assert "routed_gate_weight" not in source
        assert "routed_up_weight" not in source
    source = sources["mtp_m1_stage2"]
    assert "constexpr uint ROUTED_GROUP = 32" in source
    assert "qdot4_affine" in source
    assert "dense_shared_dot" in source
    assert "sigmoid_mlx_exact" in source
    assert "routed_gate_up_weight" in source
    assert "shared_gate_up_weight" in source
    assert "routed_gate_weight" not in source
    assert "routed_up_weight" not in source


def test_stage3_sources_encode_down_reduction_shared_gate_and_only_final_store():
    sources = kernel_module.all_whole_moe_sources()
    for name in ("target_m1_stage3", "target_m2_stage3"):
        source = sources[name]
        assert "constexpr uint ROUTED_GROUP = 64" in source
        assert "qdot4_affine" in source
        assert "bfloat down_value = bfloat(down_sum)" in source
        assert "bfloat route_product = bfloat(" in source
        assert "routed_accumulator[result_index] = bfloat(" in source
        assert "sigmoid_mlx_exact" in source
        assert "output[output_index] = bfloat(" in source
        assert "routed_outputs" not in source
        assert "shared_output" not in source
    source = sources["mtp_m1_stage3"]
    assert "constexpr uint ROUTED_GROUP = 32" in source
    assert "qdot4_affine" in source
    assert "dense_shared_down" in source
    assert "output[output_index] = bfloat(" in source


def test_all_fixed_entrypoints_launch_directly_without_runtime_validation(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    model, _, _ = _exact_model()
    plan = prepare_a3b_whole_moe(model, config=_exact_config())
    target = plan.target_bindings[0]
    mtp = plan.mtp_bindings[0]
    for name, (grid, threadgroup) in kernel_module.whole_moe_launch_table().items():
        rows = 2 if "m2" in name else 1
        binding = mtp if name.startswith("mtp") else target
        captured = _CapturedKernel()
        monkeypatch.setattr(
            kernel_module,
            f"_build_{name}_kernel",
            lambda captured=captured: captured,
        )
        entrypoint = getattr(kernel_module, name)
        if name.endswith("stage1"):
            result = entrypoint(_ArraySpec((rows, 2048)), binding)
            assert tuple(item.shape for item in result) == (
                (rows, 8),
                (rows, 8),
                (rows, 1),
            )
        elif name.endswith("stage2"):
            result = entrypoint(
                _ArraySpec((rows, 2048)),
                _ArraySpec((rows, 8), mx.uint32),
                binding,
            )
            assert result.shape == (rows, 9, 512)
        else:
            result = entrypoint(
                _ArraySpec((rows, 9, 512)),
                _ArraySpec((rows, 8), mx.uint32),
                _ArraySpec((rows, 8)),
                _ArraySpec((rows, 1)),
                binding,
            )
            assert result.shape == (rows, 2048)
        assert captured.call["grid"] == grid
        assert captured.call["threadgroup"] == threadgroup
        source = inspect.getsource(entrypoint)
        for forbidden in (
            "os.environ",
            "metal.is_available",
            ".dtype",
            ".shape",
            "eligible",
            "fallback",
            "lane_disabled",
            "try:",
            "except",
            "raise ",
        ):
            assert forbidden not in source


def _exact_selfcheck_report():
    return {
        "lanes": {
            "a3b_whole_moe_target_m1": "ok",
            "a3b_whole_moe_target_m2": "ok",
            "a3b_whole_moe_mtp_m1": "ok",
        }
    }


def _patch_whole_moe_stages(monkeypatch):
    def stage1(rows):
        return (
            _ArraySpec((rows, 8), mx.uint32),
            _ArraySpec((rows, 8)),
            _ArraySpec((rows, 1)),
        )

    monkeypatch.setattr(
        kernel_module, "target_m1_stage1", lambda value, binding: stage1(1)
    )
    monkeypatch.setattr(
        kernel_module, "target_m2_stage1", lambda value, binding: stage1(2)
    )
    monkeypatch.setattr(
        kernel_module, "mtp_m1_stage1", lambda value, binding: stage1(1)
    )
    monkeypatch.setattr(
        kernel_module,
        "target_m1_stage2",
        lambda value, ids, binding: _ArraySpec((1, 9, 512)),
    )
    monkeypatch.setattr(
        kernel_module,
        "target_m2_stage2",
        lambda value, ids, binding: _ArraySpec((2, 9, 512)),
    )
    monkeypatch.setattr(
        kernel_module,
        "mtp_m1_stage2",
        lambda value, ids, binding: _ArraySpec((1, 9, 512)),
    )
    monkeypatch.setattr(
        kernel_module,
        "target_m1_stage3",
        lambda *args: _ResultSpec("target_m1", (1, 2048)),
    )
    monkeypatch.setattr(
        kernel_module,
        "target_m2_stage3",
        lambda *args: _ResultSpec("target_m2", (2, 2048)),
    )
    monkeypatch.setattr(
        kernel_module,
        "mtp_m1_stage3",
        lambda *args: _ResultSpec("mtp_m1", (1, 2048)),
    )


def test_successful_selfcheck_atomically_installs_all_41_blocks(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    model, targets, mtp = _exact_model()
    original_classes = tuple(type(block) for block in (*targets, *mtp))
    plan = prepare_a3b_whole_moe(model, config=_exact_config())
    assert tuple(type(block) for block in (*targets, *mtp)) == original_classes

    report = install_a3b_whole_moe(plan, _exact_selfcheck_report())

    assert report["installation_status"] == "installed"
    assert report["target_blocks"] == 40
    assert report["mtp_blocks"] == 1
    assert all(
        type(block).__call__ is whole_moe_module._target_a3b_whole_moe_call
        for block in targets
    )
    assert type(mtp[0]).__call__ is whole_moe_module._mtp_a3b_whole_moe_call


def test_selfcheck_failure_prevents_every_installation(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    model, targets, mtp = _exact_model()
    original_classes = tuple(type(block) for block in (*targets, *mtp))
    plan = prepare_a3b_whole_moe(model, config=_exact_config())

    with pytest.raises(A3BWholeMoeConfigError, match="self-check"):
        install_a3b_whole_moe(
            plan,
            {"lanes": {"a3b_whole_moe_target_m1": "fallback"}},
        )

    assert tuple(type(block) for block in (*targets, *mtp)) == original_classes


def test_installed_route_uses_explicit_prefill_and_direct_small_row_calls(monkeypatch):
    monkeypatch.setenv("MTPLX_A3B_WHOLE_MOE_FUSION", "1")
    _patch_whole_moe_stages(monkeypatch)
    phase = {"value": "prefill"}
    monkeypatch.setattr(
        whole_moe_module,
        "current_attention_phase",
        lambda: phase["value"],
    )
    model, targets, mtp = _exact_model()
    plan = prepare_a3b_whole_moe(model, config=_exact_config())
    install_a3b_whole_moe(plan, _exact_selfcheck_report())

    assert targets[0](_ArraySpec((1, 64, 2048))).label == "stock"
    phase["value"] = "ar_decode"
    assert targets[0](_ArraySpec((1, 1, 2048))).label == "target_m1"
    assert mtp[0](_ArraySpec((1, 1, 2048))).label == "mtp_m1"
    phase["value"] = "decode_verify"
    assert targets[0](_ArraySpec((1, 2, 2048))).label == "target_m2"
    with pytest.raises(A3BWholeMoeRouteError, match="rows=3"):
        targets[0](_ArraySpec((1, 3, 2048)))
    assert len(targets[0].stock_calls) == 1


def test_installed_hot_call_only_routes_on_phase_and_logical_m():
    for hot_call in (
        whole_moe_module._target_a3b_whole_moe_call,
        whole_moe_module._mtp_a3b_whole_moe_call,
    ):
        source = inspect.getsource(hot_call)
        assert "current_attention_phase" in source
        assert "value.shape" in source
        for forbidden in (
            "os.environ",
            ".dtype",
            "bits",
            "group_size",
            "eligible",
            "selfcheck",
            "installed",
            "installation_status",
            "_STATS",
            "fallback",
            "lane_disabled",
            "try:",
            "except",
            "switch_mlp",
            "shared_expert",
            "mx.softmax",
            "m2_call is not None",
        ):
            assert forbidden not in source
