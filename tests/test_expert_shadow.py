"""Shadow-bank miss fallback (issue #51): codecs, kernel, planner, dispatch."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mtplx.expert_manifest import (
    build_expert_manifest,
    load_expert_manifest,
    save_expert_manifest,
)
from mtplx.expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime
from mtplx.expert_shadow import (
    SHADOW_GROUP,
    ShadowBankStore,
    ShadowCodecError,
    decode_shadow,
    encode_shadow,
    normalize_shadow_codec,
    shadow_record_bytes,
)
from mtplx.expert_streaming_models import (
    ExpertStreamingModelSpec,
    get_model_spec,
    plan_expert_memory,
)
from mtplx.kernels.shadow_gather import shadow_gather_mm
from mtplx.models.expert_mlx import (
    HotExpertSwitchGLU,
    make_mlx_component_bank_allocator,
)
from mtplx.models.hy3_mlx import Model as Hy3Model
from mtplx.models.hy3_mlx import ModelArgs as Hy3Args

REAL_HY3_Q2_ROOT = Path.home() / ".cache/huggingface/hy3-expert-only-mlx-q2"


# ---------------------------------------------------------------------------
# codec round-trip properties


@pytest.mark.parametrize("codec", ("b1", "t158"))
def test_codec_round_trip_is_stable_and_shape_exact(codec: str) -> None:
    rng = np.random.default_rng(11)
    weights = rng.standard_normal((9, 4 * SHADOW_GROUP)).astype(np.float32) * 0.03
    packed, scales = encode_shadow(codec, weights)
    decoded = decode_shadow(codec, packed, scales, weights.shape[1])
    assert decoded.shape == weights.shape
    # Re-encoding the decoded tensor must be a fixed point of the codec.
    packed_again, scales_again = encode_shadow(codec, decoded)
    assert np.array_equal(packed_again, packed)
    assert np.array_equal(scales_again, scales)
    # The decode must correlate with the source (sign structure survives).
    cosine = float(
        (weights * decoded).sum()
        / (np.linalg.norm(weights) * np.linalg.norm(decoded))
    )
    assert cosine > 0.7


def test_b1_codec_is_sign_times_group_mean() -> None:
    weights = np.array(
        [[0.5, -0.25] * (SHADOW_GROUP // 2) + [1.0] * SHADOW_GROUP],
        dtype=np.float32,
    )
    packed, scales = encode_shadow("b1", weights)
    decoded = decode_shadow("b1", packed, scales, weights.shape[1])
    # Group 1 is constant 1.0: mean |w| is exactly representable in bf16.
    assert np.all(decoded[0, SHADOW_GROUP:] == 1.0)
    # Group 0 alternates +0.5 / -0.25: signs survive, magnitude is 0.375.
    assert np.all(np.sign(decoded[0, :SHADOW_GROUP]) == np.sign(weights[0, :SHADOW_GROUP]))
    assert np.allclose(np.abs(decoded[0, :SHADOW_GROUP]), 0.375, rtol=1e-2)


def test_t158_codec_zeroes_small_weights_and_scales_nonzero_mean() -> None:
    rng = np.random.default_rng(3)
    weights = rng.standard_normal((2, 2 * SHADOW_GROUP)).astype(np.float32)
    packed, scales = encode_shadow("t158", weights)
    decoded = decode_shadow("t158", packed, scales, weights.shape[1])
    grouped_w = weights.reshape(2, -1, SHADOW_GROUP)
    grouped_d = decoded.reshape(2, -1, SHADOW_GROUP)
    threshold = 0.7 * np.abs(grouped_w).mean(axis=2, keepdims=True)
    assert np.all(grouped_d[np.abs(grouped_w) <= threshold] == 0.0)
    kept = np.abs(grouped_w) > threshold
    assert np.all(
        np.sign(grouped_d[kept]) == np.sign(grouped_w[kept])
    )
    # A constant group quantizes to all +1 at the group scale.
    flat = np.full((1, SHADOW_GROUP), 0.5, dtype=np.float32)
    flat_packed, flat_scales = encode_shadow("t158", flat)
    assert np.all(decode_shadow("t158", flat_packed, flat_scales, SHADOW_GROUP) == 0.5)
    # An all-zero group quantizes to zeros without dividing by zero.
    zero = np.zeros((1, SHADOW_GROUP), dtype=np.float32)
    zero_packed, zero_scales = encode_shadow("t158", zero)
    assert np.all(decode_shadow("t158", zero_packed, zero_scales, SHADOW_GROUP) == 0.0)


def test_codec_normalization_and_errors() -> None:
    assert normalize_shadow_codec(None) is None
    assert normalize_shadow_codec("off") is None
    assert normalize_shadow_codec("B1") == "b1"
    assert normalize_shadow_codec("t158") == "t158"
    with pytest.raises(ShadowCodecError):
        normalize_shadow_codec("q4")
    with pytest.raises(ShadowCodecError):
        encode_shadow("b1", np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ShadowCodecError):
        encode_shadow("nope", np.zeros((2, SHADOW_GROUP), dtype=np.float32))


def test_shadow_record_bytes_matches_packed_layout() -> None:
    params = 3 * 64 * 64
    weights = np.zeros((3 * 64, 64), dtype=np.float32)
    for codec in ("b1", "t158"):
        packed, scales = encode_shadow(codec, weights)
        assert shadow_record_bytes(codec, params) == packed.nbytes + scales.nbytes
    # b1: 10 bytes per 64 weights; t158: 15 bytes per 64 weights.
    assert shadow_record_bytes("b1", 64) == 10
    assert shadow_record_bytes("t158", 64) == 15
    with pytest.raises(ShadowCodecError):
        shadow_record_bytes("b1", 63)


# ---------------------------------------------------------------------------
# Metal kernel parity (synthetic + real experts)


def _bank_from_dense(codec: str, dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    packed_rows, scale_rows = zip(
        *(encode_shadow(codec, expert) for expert in dense)
    )
    return np.stack(packed_rows), np.stack(scale_rows)


def _numpy_reference(
    codec: str,
    packed: np.ndarray,
    scales: np.ndarray,
    x: np.ndarray,
    ids: np.ndarray,
) -> np.ndarray:
    in_dim = x.shape[1]
    return np.stack(
        [
            decode_shadow(codec, packed[expert], scales[expert], in_dim) @ x[row]
            for row, expert in enumerate(ids)
        ]
    )


@pytest.mark.parametrize("codec", ("b1", "t158"))
@pytest.mark.parametrize("rows", (1, 3, 8))
def test_shadow_gather_mm_matches_numpy_reference(codec: str, rows: int) -> None:
    rng = np.random.default_rng(rows * 31 + len(codec))
    experts, out_dim, in_dim = 4, 16, 2 * SHADOW_GROUP
    dense = rng.standard_normal((experts, out_dim, in_dim)).astype(np.float32) * 0.05
    packed, scales = _bank_from_dense(codec, dense)
    x = rng.standard_normal((rows, in_dim)).astype(np.float32)
    ids = rng.integers(0, experts, size=rows).astype(np.int32)
    reference = _numpy_reference(codec, packed, scales, x, ids)
    result = shadow_gather_mm(
        mx.array(x),
        mx.array(ids),
        mx.array(packed),
        mx.array(scales),
        codec=codec,
    )
    np.testing.assert_allclose(
        np.asarray(result), reference, rtol=1e-2, atol=1e-5
    )


@pytest.mark.parametrize("codec", ("b1", "t158"))
def test_shadow_gather_mm_bf16_parity(codec: str) -> None:
    rng = np.random.default_rng(5)
    experts, out_dim, in_dim = 3, 8, SHADOW_GROUP
    dense = rng.standard_normal((experts, out_dim, in_dim)).astype(np.float32) * 0.1
    packed, scales = _bank_from_dense(codec, dense)
    x32 = rng.standard_normal((2, in_dim)).astype(np.float32)
    x_bf16 = mx.array(x32).astype(mx.bfloat16)
    ids = np.array([2, 0], dtype=np.int32)
    reference = _numpy_reference(
        codec, packed, scales, np.asarray(x_bf16.astype(mx.float32)), ids
    )
    result = shadow_gather_mm(
        x_bf16, mx.array(ids), mx.array(packed), mx.array(scales), codec=codec
    )
    assert result.dtype == mx.bfloat16
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        reference,
        rtol=1e-2,
        atol=1e-2,
    )


def test_shadow_gather_mm_validates_shapes() -> None:
    packed = mx.zeros((1, 2, 2), dtype=mx.uint32)
    scales = mx.zeros((1, 2, 1), dtype=mx.uint16)
    with pytest.raises(ShadowCodecError):
        shadow_gather_mm(
            mx.zeros((1, SHADOW_GROUP + 1)),
            mx.array([0]),
            packed,
            scales,
            codec="b1",
        )
    with pytest.raises(ShadowCodecError):
        shadow_gather_mm(
            mx.zeros((1, SHADOW_GROUP)),
            mx.array([0]),
            mx.zeros((1, 2, 5), dtype=mx.uint32),
            scales,
            codec="b1",
        )


@pytest.mark.skipif(
    not (REAL_HY3_Q2_ROOT / "expert-manifest.json").exists(),
    reason="real hy3-q2 expert artifact not present",
)
@pytest.mark.parametrize("codec", ("b1", "t158"))
def test_shadow_gather_mm_parity_on_real_hy3_experts(codec: str) -> None:
    """Encode real Q2 experts on the CPU and gate the kernel on parity."""

    manifest = load_expert_manifest(REAL_HY3_Q2_ROOT / "expert-manifest.json")
    spec = get_model_spec(manifest.model_key)
    store = ShadowBankStore(
        spec, spec.routed_layer_indices[:1], codec=codec
    )
    records = [
        record
        for record in manifest.records
        if record.layer == spec.routed_layer_indices[0]
    ][:3]
    assert len(records) == 3
    from mtplx.expert_manifest import read_expert_record

    rng = np.random.default_rng(17)
    x = rng.standard_normal((2, spec.hidden_size)).astype(np.float32) * 0.02
    for record in records:
        blob = read_expert_record(
            manifest,
            REAL_HY3_Q2_ROOT,
            record.layer,
            record.expert,
            verify_hash=False,
        )
        dense = store._dequantize_record(
            mx,
            record,
            blob,
            bits=spec.quant_bits,
            group_size=spec.quant_group_size,
        )
        gate = dense["gate_proj"]
        packed, scales = encode_shadow(codec, gate)
        ids = np.zeros(2, dtype=np.int32)
        reference = _numpy_reference(codec, packed[None], scales[None], x, ids)
        result = shadow_gather_mm(
            mx.array(x),
            mx.array(ids),
            mx.array(packed[None]),
            mx.array(scales[None]),
            codec=codec,
        )
        np.testing.assert_allclose(
            np.asarray(result), reference, rtol=1e-2, atol=1e-4
        )


# ---------------------------------------------------------------------------
# planner pricing + config validation


def test_plan_prices_shadow_banks_on_the_fixed_side() -> None:
    spec = get_model_spec("hy3-expert-q2")
    base = dict(
        total_limit_bytes=100 * 1024**3,
        context_tokens=0,
        runtime_reserve_bytes=0,
        island_layer_count=4,
    )
    plain = plan_expert_memory(spec, **base)
    shadowed = plan_expert_memory(spec, miss_shadow="b1", **base)
    streamed = spec.routed_layer_count - 4
    expected = streamed * spec.expert_count * shadow_record_bytes(
        "b1", spec.expert_source_parameters
    )
    assert plain.miss_shadow is None and plain.shadow_bytes == 0
    assert shadowed.miss_shadow == "b1"
    assert shadowed.shadow_bytes == expected
    assert shadowed.fixed_bytes == plain.fixed_bytes + expected
    # The shadow cost must shrink the uniform slot budget, never grow it.
    assert shadowed.persistent_cache_bytes <= plain.persistent_cache_bytes
    heavier = plan_expert_memory(spec, miss_shadow="t158", **base)
    assert heavier.shadow_bytes == streamed * spec.expert_count * (
        shadow_record_bytes("t158", spec.expert_source_parameters)
    )
    assert heavier.shadow_bytes > shadowed.shadow_bytes
    with pytest.raises(ValueError):
        plan_expert_memory(spec, miss_shadow="fp8", **base)


def test_config_validates_and_forwards_miss_shadow() -> None:
    base = dict(
        model_key="hy3-expert-q2",
        memory_limit_bytes=100 * 1024**3,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        cache_scope="layer",
        slot_layout="component-banks",
    )
    config = ExpertStreamingConfig(**base, miss_shadow="t158")
    assert config.to_dict()["miss_shadow"] == "t158"
    plan = config.memory_plan(get_model_spec("hy3-expert-q2"))
    assert plan.miss_shadow == "t158" and plan.shadow_bytes > 0
    with pytest.raises(ValueError, match="miss_shadow must be"):
        ExpertStreamingConfig(**base, miss_shadow="b2")
    with pytest.raises(ValueError, match="cache_scope 'layer'"):
        ExpertStreamingConfig(
            **{**base, "cache_scope": "global"}, miss_shadow="b1"
        )
    with pytest.raises(ValueError, match="component-banks"):
        ExpertStreamingConfig(
            **{**base, "slot_layout": "direct-slots"}, miss_shadow="b1"
        )


def test_shadow_store_rejects_unpriced_plan() -> None:
    spec = get_model_spec("hy3-expert-q2")
    unpriced = plan_expert_memory(
        spec,
        total_limit_bytes=100 * 1024**3,
        context_tokens=0,
    )
    with pytest.raises(ShadowCodecError, match="did not price"):
        ShadowBankStore(
            spec, spec.routed_layer_indices, codec="b1", plan=unpriced
        )
    wrong_codec = plan_expert_memory(
        spec,
        total_limit_bytes=200 * 1024**3,
        context_tokens=0,
        miss_shadow="t158",
    )
    with pytest.raises(ShadowCodecError, match="did not price"):
        ShadowBankStore(
            spec, spec.routed_layer_indices, codec="b1", plan=wrong_codec
        )


# ---------------------------------------------------------------------------
# integrated dispatch on the tiny hy3 artifact


def _hy3_args(*, num_experts: int = 2, top_k: int = 1) -> Hy3Args:
    return Hy3Args(
        model_type="hy_v3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        num_shared_experts=1,
        first_k_dense_replace=1,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=16,
        router_scaling_factor=2.0,
    )


def _shadow_hy3_artifact(tmp_path: Path, *, expert_count: int = 2, top_k: int = 1):
    """The integrated tiny hy3 artifact with NON-degenerate expert weights.

    Mirrors ``_integrated_hy3_artifact`` (test_streamed_models) but fills
    the routed Q4 leaves with random bits and non-trivial scales/biases so
    experts dequantize to distinct non-zero matrices — required to compare
    shadow output against exact output by cosine.
    """

    args = _hy3_args(num_experts=expert_count, top_k=top_k)
    model = Hy3Model(args)
    weights = dict(tree_flatten(model.parameters()))
    rng = np.random.default_rng(23)
    expert_shapes = {
        "gate_proj.weight": (expert_count, 64, 8),
        "gate_proj.scales": (expert_count, 64, 1),
        "gate_proj.biases": (expert_count, 64, 1),
        "up_proj.weight": (expert_count, 64, 8),
        "up_proj.scales": (expert_count, 64, 1),
        "up_proj.biases": (expert_count, 64, 1),
        "down_proj.weight": (expert_count, 64, 8),
        "down_proj.scales": (expert_count, 64, 1),
        "down_proj.biases": (expert_count, 64, 1),
    }
    # Q4 rows dequantize to scale*(q - 7.5): zero-mean and sign-symmetric,
    # like trained expert weights (the shadow codecs encode sign structure;
    # a DC offset per group is unrepresentable and unrealistic).
    scale_values = {
        projection: rng.uniform(0.01, 0.03, size=(expert_count, 64, 1)).astype(
            np.float32
        )
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    for component, shape in expert_shapes.items():
        projection, leaf = component.split(".")
        if leaf == "weight":
            bits = rng.integers(0, 2**32, size=shape, dtype=np.uint64)
            value = mx.array(bits.astype(np.uint32))
        elif leaf == "scales":
            value = mx.array(scale_values[projection]).astype(mx.bfloat16)
        else:
            value = mx.array(-7.5 * scale_values[projection]).astype(mx.bfloat16)
        weights[f"model.layers.1.mlp.switch_mlp.{component}"] = value
    mx.eval(weights)
    root = tmp_path / "hy3"
    root.mkdir()
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    config = asdict(args)
    config["model_type"] = "hy_v3"
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
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
        expert_count=expert_count,
        top_k=top_k,
        hidden_size=64,
        expert_hidden_size=64,
        quant_bits=4,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="float32",
        router_matmul_dtype="float32",
        router_bytes=expert_count * 64 * 4 + expert_count * 4,
        kv_bytes_per_token=0,
        mtp_layer_index=2,
        mtp_included=False,
    )
    manifest = build_expert_manifest(root, spec)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    return root, spec, manifest_path


def _open_runtime(root, spec, manifest_path, *, miss_shadow, prefetch_slots=0):
    config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=spec.resident_bytes
        + spec.transient_scratch_bytes
        + 4 * 1024 * 1024,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        cache_scope="layer",
        slot_layout="component-banks",
        miss_shadow=miss_shadow,
        prefetch_slots=prefetch_slots,
    )
    return ExpertStreamingRuntime.open(
        root,
        manifest_path,
        config,
        spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            config.memory_plan(spec),
            spec,
            load_expert_manifest(manifest_path),
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )


def test_shadow_store_builds_at_open_for_streamed_layers(tmp_path: Path) -> None:
    root, spec, manifest_path = _shadow_hy3_artifact(tmp_path)
    runtime = _open_runtime(root, spec, manifest_path, miss_shadow="b1")
    try:
        store = runtime._shadow_store
        assert store is not None
        assert store.filled_layers == (1,)
        bank = runtime.shadow_bank_for_layer(1)
        assert bank is not None and bank.codec == "b1"
        assert runtime.shadow_bank_for_layer(0) is None
        expected = spec.expert_count * shadow_record_bytes(
            "b1", spec.expert_source_parameters
        )
        assert store.shadow_bytes == expected
        assert runtime.plan.shadow_bytes == expected
        snapshot = runtime.snapshot(mx_module=mx)
        assert snapshot["memory_plan"]["miss_shadow"] == "b1"
        assert snapshot["memory_plan"]["shadow_bytes"] == expected
        assert snapshot["shadow_experts"]["filled_layers"] == 1
        assert snapshot["shadow_experts"]["serve_routes"] == 0
    finally:
        runtime.close()


def test_shadow_bank_matches_dequantized_expert_weights(tmp_path: Path) -> None:
    root, spec, manifest_path = _shadow_hy3_artifact(tmp_path)
    runtime = _open_runtime(root, spec, manifest_path, miss_shadow="t158")
    try:
        bank = runtime.shadow_bank_for_layer(1)
        packed = np.asarray(bank.arrays["gate_proj.packed"])
        scales = np.asarray(bank.arrays["gate_proj.scales"])
        assert packed.shape[0] == spec.expert_count
        # The decoded shadow of a real expert row must correlate with its
        # exact dequantized weights.
        raw = mx.load(str(root / "model.safetensors"))
        dense = mx.dequantize(
            raw["model.layers.1.mlp.switch_mlp.gate_proj.weight"],
            raw["model.layers.1.mlp.switch_mlp.gate_proj.scales"],
            raw["model.layers.1.mlp.switch_mlp.gate_proj.biases"],
            bits=4,
            group_size=64,
            mode="affine",
        )
        for expert in range(spec.expert_count):
            exact = np.asarray(dense[expert].astype(mx.float32))
            decoded = decode_shadow(
                "t158", packed[expert], scales[expert], exact.shape[1]
            )
            cosine = float(
                (exact * decoded).sum()
                / (np.linalg.norm(exact) * np.linalg.norm(decoded))
            )
            assert cosine > 0.7
    finally:
        runtime.close()


def _decode_route(switch, x_row: mx.array, expert: int) -> mx.array:
    indices = mx.array([[[expert]]], dtype=mx.uint32)
    output = switch(x_row, indices)
    mx.eval(output)
    return output


# The quality bar is codec- and fixture-specific: on real trained experts
# the probe measured combine-cosine 0.90-0.91 (b1) and 0.946-0.952 (t158),
# but the tiny fixture's experts are uniform Q4 noise, the worst case for a
# sign-only codec (per-matrix cosine sqrt(3)/2 compounds across the three
# projections). t158 keeps magnitude structure and clears 0.8 even here.
@pytest.mark.parametrize(
    ("codec", "minimum_cosine"), (("t158", 0.8), ("b1", 0.4))
)
def test_decode_miss_is_served_from_shadow_without_ssd(
    tmp_path: Path, codec: str, minimum_cosine: float
) -> None:
    root, spec, manifest_path = _shadow_hy3_artifact(tmp_path)
    x_row = mx.array(
        np.random.default_rng(41).standard_normal((1, 1, 64)).astype(np.float32)
    )

    exact_runtime = _open_runtime(root, spec, manifest_path, miss_shadow=None)
    try:
        exact_switch = HotExpertSwitchGLU(exact_runtime, 1)
        exact = _decode_route(exact_switch, x_row, expert=1)
    finally:
        exact_runtime.close()

    runtime = _open_runtime(
        root, spec, manifest_path, miss_shadow=codec, prefetch_slots=1
    )
    try:
        switch = HotExpertSwitchGLU(runtime, 1)
        assert switch._shadow_bank is not None
        before = runtime.snapshot(mx_module=mx)["cache"]
        shadow_output = _decode_route(switch, x_row, expert=1)
        snapshot = runtime.snapshot(mx_module=mx)
        cache = snapshot["cache"]
        # The cold decode miss never touched the exact service tiers.
        assert cache["transient_loads"] == before["transient_loads"] == 0
        assert cache["persistent_loads"] == before["persistent_loads"] == 0
        assert snapshot["shadow_experts"]["serve_routes"] == 1
        assert snapshot["shadow_experts"]["served_assignments"] == 1
        assert snapshot["shadow_experts"]["served_experts"] == 1
        # Quality tier: close to exact, NOT bitwise.
        exact_vec = np.asarray(exact).reshape(-1)
        shadow_vec = np.asarray(shadow_output).reshape(-1)
        assert not np.array_equal(exact_vec, shadow_vec)
        cosine = float(
            (exact_vec * shadow_vec).sum()
            / (np.linalg.norm(exact_vec) * np.linalg.norm(shadow_vec))
        )
        assert cosine > minimum_cosine
        # The speculative admission fill lands asynchronously and turns
        # the same expert into an exact hit.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            counters = runtime.snapshot(mx_module=mx)["cache"]
            if counters["prefetch_committed"] >= 1:
                break
            time.sleep(0.02)
        else:
            pytest.fail("speculative admission fill never committed")
        warm = _decode_route(switch, x_row, expert=1)
        after = runtime.snapshot(mx_module=mx)
        assert after["shadow_experts"]["serve_routes"] == 1  # unchanged
        assert after["cache"]["expert_hits"] >= 1
        np.testing.assert_array_equal(np.asarray(warm), np.asarray(exact))
    finally:
        runtime.close()


def test_prefill_route_stays_exact_with_shadow_enabled(tmp_path: Path) -> None:
    root, spec, manifest_path = _shadow_hy3_artifact(tmp_path)
    runtime = _open_runtime(root, spec, manifest_path, miss_shadow="b1")
    try:
        switch = HotExpertSwitchGLU(runtime, 1)
        tokens = mx.array(
            np.random.default_rng(7).standard_normal((1, 3, 64)).astype(np.float32)
        )
        indices = mx.array([[[0], [1], [0]]], dtype=mx.uint32)
        output = switch(tokens, indices)
        mx.eval(output)
        snapshot = runtime.snapshot(mx_module=mx)
        cache = snapshot["cache"]
        # Prefill stays exact: misses are serviced from SSD (persistent or
        # transient tier depending on admission), never from the shadow.
        assert snapshot["shadow_experts"]["serve_routes"] == 0
        assert cache["persistent_loads"] + cache["transient_loads"] >= 1
        assert cache["bytes_read"] > 0
    finally:
        runtime.close()
