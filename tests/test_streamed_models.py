from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.activations import swiglu
from mlx_lm.models.deepseek_v32 import group_expert_select

from mtplx.expert_manifest import (
    build_expert_manifest,
    load_expert_manifest,
    save_expert_manifest,
)
from mtplx.expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime
from mtplx.expert_slots import ExpertSlotBinding
from mtplx.expert_streaming_models import ExpertStreamingModelSpec
from mtplx.models.expert_mlx import (
    _run_q4_expert,
    make_mlx_component_bank_allocator,
    make_mlx_slot_buffer_allocator,
)
from mtplx.models.glm52_mlx import FP32MoEGate
from mtplx.models.glm52_mlx import Model as GlmModel
from mtplx.models.glm52_mlx import ModelArgs as GlmArgs
from mtplx.models.hy3_mlx import Model as Hy3Model
from mtplx.models.hy3_mlx import ModelArgs as Hy3Args
from mtplx.resident_loader import construct_resident_model


def _hy3_args() -> Hy3Args:
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


def _glm_args(*, layers: int = 6, first_sparse: int = 1) -> GlmArgs:
    return GlmArgs(
        model_type="glm_moe_dsa",
        vocab_size=128,
        hidden_size=64,
        index_head_dim=8,
        index_n_heads=4,
        index_topk=4,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=24,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=first_sparse,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10_000.0},
        attention_bias=False,
        index_topk_pattern="FSFSFS" if layers == 6 else None,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )


def test_hy3_router_uses_unbiased_scores_for_weights() -> None:
    model = Hy3Model(_hy3_args())
    router = model.model.layers[1].mlp.router
    router.gate.weight = mx.array(
        [
            [1.0] + [0.0] * 63,
            [0.5] + [0.0] * 63,
        ],
        dtype=mx.float32,
    )
    router.expert_bias = mx.array([0.0, 10.0], dtype=mx.float32)
    hidden = mx.array([[[1.0] + [0.0] * 63]], dtype=mx.bfloat16)

    indices, weights = router(hidden)
    mx.eval(indices, weights)

    assert indices.item() == 1
    # Correction bias selects expert 1 but is not part of its returned weight.
    assert weights.item() == pytest.approx(2.0)


def test_hy3_non_fp32_combine_casts_routing_weights_to_activation_dtype() -> None:
    model = Hy3Model(_hy3_args())
    sparse_mlp = model.model.layers[1].mlp
    routed_values = [
        4.336133731530333,
        -3.7478681657261284,
        2.5238181218276177,
        4.55797767056557,
        -0.09106507450222434,
        -4.222918128185795,
        0.24504878855251455,
        3.055661890955095,
    ]
    routing_weights = [
        1.9165060684424067,
        0.1014428979156421,
        1.6294923886421544,
        1.2853930759868095,
        1.2728115717622752,
        0.5407021513938952,
        1.755765528776848,
        1.2906728107893897,
    ]

    class StubRouter:
        def __call__(self, _x):
            return (
                mx.zeros((1, 1, 8), dtype=mx.int32),
                mx.array([[routing_weights]], dtype=mx.float32),
            )

    class StubSwitch:
        def __call__(self, _x, _indices):
            rows = [[value] + [0.0] * 63 for value in routed_values]
            return mx.array([[rows]], dtype=mx.bfloat16)

    class StubShared:
        def __call__(self, x):
            return mx.zeros_like(x)

    sparse_mlp.router = StubRouter()
    sparse_mlp.switch_mlp = StubSwitch()
    sparse_mlp.shared_mlp = StubShared()
    hidden = mx.zeros((1, 1, 64), dtype=mx.bfloat16)

    output = sparse_mlp(hidden)
    mx.eval(output)

    # The pinned reference rounds routing weights to BF16 before the multiply.
    # Leaving them in FP32 produces 19.875 for this fixture instead of 20.0.
    assert output.dtype == mx.bfloat16
    assert output[0, 0, 0].item() == 20.0


def test_glm_router_fp32_projection_changes_a_near_tie_route() -> None:
    x = mx.array(
        [
            [
                0.0205078125,
                -0.006805419921875,
                -0.05859375,
                0.080078125,
                -0.21484375,
                -0.1875,
                -0.01318359375,
                0.1767578125,
                -0.01556396484375,
                0.10498046875,
                0.212890625,
                0.06640625,
                0.04541015625,
                0.058349609375,
                0.044921875,
                -0.021484375,
            ]
        ],
        dtype=mx.bfloat16,
    )
    weight = mx.array(
        [
            [
                0.0267333984375,
                0.134765625,
                0.22265625,
                -0.043212890625,
                -0.1826171875,
                -0.109375,
                0.06298828125,
                0.1552734375,
                0.01214599609375,
                -0.06591796875,
                -0.09814453125,
                0.1787109375,
                -0.02294921875,
                -0.12353515625,
                -0.050048828125,
                -0.00023937225341796875,
            ],
            [
                0.002044677734375,
                -0.01202392578125,
                -0.1279296875,
                0.04296875,
                -0.166015625,
                0.0634765625,
                -0.173828125,
                -0.072265625,
                -0.171875,
                -0.0294189453125,
                -0.057373046875,
                0.2001953125,
                0.039306640625,
                -0.0859375,
                -0.029541015625,
                -0.052734375,
            ],
            [
                -0.0888671875,
                0.205078125,
                0.12060546875,
                0.031005859375,
                -0.1337890625,
                -0.0220947265625,
                0.006866455078125,
                -0.0166015625,
                0.05810546875,
                0.076171875,
                -0.006439208984375,
                0.103515625,
                -0.031494140625,
                0.134765625,
                0.01409912109375,
                -0.06005859375,
            ],
            [
                0.060791015625,
                -0.022216796875,
                -0.0181884765625,
                -0.049072265625,
                -0.1259765625,
                0.1298828125,
                0.09375,
                0.0126953125,
                -0.126953125,
                -0.111328125,
                -0.1494140625,
                0.0869140625,
                -0.0034027099609375,
                0.015869140625,
                0.11572265625,
                -0.038330078125,
            ],
        ],
        dtype=mx.bfloat16,
    )
    original = SimpleNamespace(
        top_k=1,
        norm_topk_prob=True,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        weight=weight,
        e_score_correction_bias=mx.zeros((4,), dtype=mx.float32),
    )
    fp32_router = FP32MoEGate(original)

    fp32_indices, _ = fp32_router(x)
    legacy_indices, _ = group_expert_select(
        x @ weight.T,
        original.e_score_correction_bias,
        1,
        1,
        1,
        1.0,
        True,
    )
    mx.eval(fp32_indices, legacy_indices)

    assert fp32_indices.item() == 2
    assert legacy_indices.item() == 0


def test_glm_indexshare_schedule_and_asymmetric_caches_execute() -> None:
    args = _glm_args(first_sparse=6)
    model = GlmModel(args)

    assert args.indexer_types == ["full", "shared", "full", "shared", "full", "shared"]
    assert [layer.self_attn.indexer is not None for layer in model.model.layers] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    cache = model.make_cache()
    assert [len(item.caches) for item in cache] == [2, 1, 2, 1, 2, 1]
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    logits = model(prompt, cache=cache)
    mx.eval(logits)
    assert logits.shape == (1, 4, args.vocab_size)
    assert mx.all(mx.isfinite(logits)).item()


def _raw_array(value: mx.array, dtype: str) -> bytes:
    mx.eval(value)
    if dtype == "U32":
        return np.array(value, copy=True).astype("<u4", copy=False).tobytes()
    return (
        np.array(value.view(mx.uint16), copy=True).astype("<u2", copy=False).tobytes()
    )


def test_portable_q4_slot_execution_matches_direct_quantized_matmul() -> None:
    mx.random.seed(7)
    hidden = 64
    intermediate = 64
    gate_source = mx.random.normal((intermediate, hidden)).astype(mx.bfloat16)
    up_source = mx.random.normal((intermediate, hidden)).astype(mx.bfloat16)
    down_source = mx.random.normal((hidden, intermediate)).astype(mx.bfloat16)
    gate = mx.quantize(gate_source, group_size=64, bits=4)
    up = mx.quantize(up_source, group_size=64, bits=4)
    down = mx.quantize(down_source, group_size=64, bits=4)
    arrays = {
        "gate_proj.weight": (gate[0], "U32"),
        "gate_proj.scales": (gate[1], "BF16"),
        "gate_proj.biases": (gate[2], "BF16"),
        "up_proj.weight": (up[0], "U32"),
        "up_proj.scales": (up[1], "BF16"),
        "up_proj.biases": (up[2], "BF16"),
        "down_proj.weight": (down[0], "U32"),
        "down_proj.scales": (down[1], "BF16"),
        "down_proj.biases": (down[2], "BF16"),
    }
    from mtplx.expert_manifest import ExpertRecord, TensorSegment

    payload = bytearray()
    segments = []
    for component, (value, dtype) in arrays.items():
        raw = _raw_array(value, dtype)
        segments.append(
            TensorSegment(
                component=component,
                tensor=component,
                shard="fixture",
                offset=len(payload),
                length=len(raw),
                dtype=dtype,
                shape=tuple(value.shape),
            )
        )
        payload.extend(raw)
    record = ExpertRecord(
        layer=1,
        expert=0,
        logical_bytes=len(payload),
        segments=tuple(segments),
    )
    binding = ExpertSlotBinding(1, 0, 0, 1, record, payload)
    x = mx.random.normal((3, hidden)).astype(mx.bfloat16)

    actual = _run_q4_expert(x, binding, group_size=64)
    reference_gate = mx.quantized_matmul(
        x,
        gate[0],
        scales=gate[1],
        biases=gate[2],
        group_size=64,
        bits=4,
    )
    reference_up = mx.quantized_matmul(
        x,
        up[0],
        scales=up[1],
        biases=up[2],
        group_size=64,
        bits=4,
    )
    reference = mx.quantized_matmul(
        swiglu(reference_gate, reference_up),
        down[0],
        scales=down[1],
        biases=down[2],
        group_size=64,
        bits=4,
    )
    mx.eval(actual, reference)
    assert mx.allclose(actual, reference, atol=1e-5, rtol=1e-5).item()

def _integrated_hy3_artifact(tmp_path: Path):
    args = _hy3_args()
    model = Hy3Model(args)
    weights = dict(tree_flatten(model.parameters()))
    expert_shapes = {
        "gate_proj.weight": (2, 64, 8),
        "gate_proj.scales": (2, 64, 1),
        "gate_proj.biases": (2, 64, 1),
        "up_proj.weight": (2, 64, 8),
        "up_proj.scales": (2, 64, 1),
        "up_proj.biases": (2, 64, 1),
        "down_proj.weight": (2, 64, 8),
        "down_proj.scales": (2, 64, 1),
        "down_proj.biases": (2, 64, 1),
    }
    for component, shape in expert_shapes.items():
        dtype = mx.uint32 if component.endswith("weight") else mx.bfloat16
        value = mx.zeros(shape, dtype=dtype)
        if component.endswith("scales"):
            value = mx.ones(shape, dtype=dtype)
        weights[f"model.layers.1.mlp.switch_mlp.{component}"] = value
    mx.eval(weights)
    root = tmp_path / "hy3"
    root.mkdir()
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    config = asdict(args)
    config["model_type"] = "hy_v3"
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    routed_bytes = 2 * 6_912
    total_bytes = sum(int(value.nbytes) for value in weights.values())
    spec = ExpertStreamingModelSpec(
        key="tiny-hy3-q4",
        display_name="Tiny Hy3 Q4",
        source_model="test/tiny-hy3",
        source_revision="source",
        quant_model="test/tiny-hy3-q4",
        quant_revision="quant",
        total_tensor_bytes=total_bytes,
        total_layers=2,
        routed_layer_start=1,
        routed_layer_count=1,
        expert_count=2,
        top_k=1,
        hidden_size=64,
        expert_hidden_size=64,
        quant_bits=4,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="float32",
        router_matmul_dtype="float32",
        router_bytes=2 * 64 * 4 + 2 * 4,
        kv_bytes_per_token=0,
        mtp_layer_index=2,
        mtp_included=False,
    )
    assert spec.routed_expert_bytes == routed_bytes
    manifest = build_expert_manifest(root, spec)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    return root, config, spec, manifest_path


def test_resident_loader_runs_hy3_without_materializing_routed_parameters(
    tmp_path: Path,
) -> None:
    root, config, spec, manifest_path = _integrated_hy3_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_slot_buffer_allocator(
            stream_config.memory_plan(spec), spec
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    try:
        resident = construct_resident_model(root, runtime, config=config)
        parameter_names = {
            name for name, _ in tree_flatten(resident.model.parameters())
        }
        assert not any("switch_mlp" in name for name in parameter_names)
        assert resident.report.raw_tensor_bytes == spec.resident_bytes
        assert resident.report.bound_sparse_layers == 1

        logits = resident.model(mx.array([[1]], dtype=mx.int32))
        mx.eval(logits)
        assert logits.shape == (1, 1, config["vocab_size"])
        assert mx.all(mx.isfinite(logits)).item()
        snapshot = runtime.snapshot(mx_module=mx)
        assert snapshot["cache"]["expert_requests"] == 1
        assert snapshot["slots"]["pins"] == 0
        assert snapshot["slots"]["buffer_backend"] == "mlx-metal-direct-slots"
    finally:
        runtime.close()


def test_component_bank_hy3_executes_without_record_or_stack_copies(
    tmp_path: Path,
) -> None:
    root, config, spec, manifest_path = _integrated_hy3_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        slot_layout="component-banks",
    )
    plan = stream_config.memory_plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            plan,
            spec,
            load_expert_manifest(manifest_path),
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    try:
        resident = construct_resident_model(root, runtime, config=config)
        logits = resident.model(mx.array([[1, 2]], dtype=mx.int32))
        mx.eval(logits)
        assert logits.shape == (1, 2, config["vocab_size"])
        assert mx.all(mx.isfinite(logits)).item()
        snapshot = runtime.snapshot(mx_module=mx)
        assert snapshot["slots"]["buffer_backend"] == "mlx-metal-component-banks"
        assert snapshot["slots"]["pins"] == 0
        assert snapshot["slots"]["io"]["integrity_errors"] == 0
    finally:
        runtime.close()


def test_resident_loader_reads_extensionless_hugging_face_cache_blob(
    tmp_path: Path,
) -> None:
    source_parent = tmp_path / "source"
    source_parent.mkdir()
    source, config, spec, source_manifest = _integrated_hy3_artifact(source_parent)
    repository = tmp_path / "models--test--tiny-hy3"
    blobs = repository / "blobs"
    snapshot = repository / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / ("b" * 64)
    blob.hardlink_to(source / "model.safetensors")
    (snapshot / "model.safetensors").symlink_to(
        Path("..") / ".." / "blobs" / blob.name
    )
    (snapshot / "config.json").write_text(
        (source / "config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    manifest_path = snapshot / "expert-manifest.json"
    manifest_path.write_text(
        source_manifest.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    runtime = ExpertStreamingRuntime.open(
        snapshot,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_slot_buffer_allocator(
            stream_config.memory_plan(spec), spec
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    try:
        resident = construct_resident_model(snapshot, runtime, config=config)
        logits = resident.model(mx.array([[1]], dtype=mx.int32))
        mx.eval(logits)
        assert logits.shape == (1, 1, config["vocab_size"])
    finally:
        runtime.close()


def test_batched_single_token_decode_warms_persistent_expert_cache(
    tmp_path: Path,
) -> None:
    root, config, spec, manifest_path = _integrated_hy3_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_slot_buffer_allocator(
            stream_config.memory_plan(spec), spec
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    try:
        resident = construct_resident_model(root, runtime, config=config)
        logits = resident.model(mx.array([[1], [2]], dtype=mx.int32))
        mx.eval(logits)

        assert logits.shape == (2, 1, config["vocab_size"])
        assert runtime._banks[1].occupancy == 1
        assert runtime.snapshot(mx_module=mx)["cache"]["persistent_loads"] >= 1
    finally:
        runtime.close()


def _integrated_glm_artifact(tmp_path: Path):
    args = _glm_args(layers=6, first_sparse=1)
    model = GlmModel(args)
    weights = dict(tree_flatten(model.parameters()))
    expert_shapes = {
        "gate_proj.weight": (4, 64, 8),
        "gate_proj.scales": (4, 64, 1),
        "gate_proj.biases": (4, 64, 1),
        "up_proj.weight": (4, 64, 8),
        "up_proj.scales": (4, 64, 1),
        "up_proj.biases": (4, 64, 1),
        "down_proj.weight": (4, 64, 8),
        "down_proj.scales": (4, 64, 1),
        "down_proj.biases": (4, 64, 1),
    }
    for layer in range(1, 6):
        for component, shape in expert_shapes.items():
            dtype = mx.uint32 if component.endswith("weight") else mx.bfloat16
            value = mx.zeros(shape, dtype=dtype)
            if component.endswith("scales"):
                value = mx.ones(shape, dtype=dtype)
            weights[f"model.layers.{layer}.mlp.switch_mlp.{component}"] = value
    mx.eval(weights)
    root = tmp_path / "glm"
    root.mkdir()
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    config = asdict(args)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    total_bytes = sum(int(value.nbytes) for value in weights.values())
    spec = ExpertStreamingModelSpec(
        key="tiny-glm52-q4",
        display_name="Tiny GLM-5.2 Q4",
        source_model="test/tiny-glm",
        source_revision="source",
        quant_model="test/tiny-glm-q4",
        quant_revision="quant",
        total_tensor_bytes=total_bytes,
        total_layers=6,
        routed_layer_start=1,
        routed_layer_count=5,
        expert_count=4,
        top_k=2,
        hidden_size=64,
        expert_hidden_size=64,
        quant_bits=4,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="float32",
        router_matmul_dtype="float32",
        router_bytes=5 * (4 * 64 * 4 + 4 * 4),
        kv_bytes_per_token=0,
        mtp_layer_index=6,
        mtp_included=False,
        full_indexer_layers=(0, 2, 4),
    )
    manifest = build_expert_manifest(root, spec)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    return root, config, spec, manifest_path


def test_resident_loader_runs_indexshare_glm_with_streamed_sparse_layers(
    tmp_path: Path,
) -> None:
    root, config, spec, manifest_path = _integrated_glm_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        apply_memory_cap=False,
    )
    try:
        resident = construct_resident_model(root, runtime, config=config)
        parameter_names = {
            name for name, _ in tree_flatten(resident.model.parameters())
        }
        assert not any("switch_mlp" in name for name in parameter_names)
        assert [
            layer.self_attn.indexer is not None for layer in resident.model.model.layers
        ] == [True, False, True, False, True, False]

        logits = resident.model(mx.array([[1]], dtype=mx.int32))
        mx.eval(logits)
        assert logits.shape == (1, 1, config["vocab_size"])
        assert mx.all(mx.isfinite(logits)).item()
        assert runtime.snapshot(mx_module=mx)["cache"]["route_calls"] == 5
    finally:
        runtime.close()
