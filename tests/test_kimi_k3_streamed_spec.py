from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.expert_shadow import decode_shadow, encode_shadow
from mtplx.expert_streaming_models import get_model_spec, plan_expert_memory
from mtplx.models.expert_mlx import HotExpertSwitchGLU


def _bank_from_dense(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    packed_rows, scale_rows = zip(
        *(encode_shadow("t158", expert) for expert in dense),
        strict=True,
    )
    return np.stack(packed_rows), np.stack(scale_rows)


def test_kimi_k3_streaming_spec_has_exact_measured_geometry() -> None:
    spec = get_model_spec("kimi-k3-q1t")
    logical_mla_bytes_per_token = 24 * (512 + 64) * 2
    geometric_capacity_price = 2 * logical_mla_bytes_per_token

    assert spec.routed_layer_indices == tuple(range(1, 93))
    assert spec.expert_count == 896
    assert spec.top_k == 16
    assert spec.hidden_size == 3584
    assert spec.model_hidden_size == 7168
    assert spec.expert_hidden_size == 3072
    assert spec.expert_codec == "t158"
    assert spec.quant_bits == 2
    assert spec.expert_activation == "situ"
    assert spec.expert_record_bytes == 7_741_440
    assert spec.routed_expert_bytes == 638_142_382_080
    assert spec.resident_bytes == 113_509_540_864
    assert spec.total_tensor_bytes == 751_651_922_944
    assert logical_mla_bytes_per_token == 27_648
    assert spec.kv_bytes_per_token == geometric_capacity_price == 55_296
    assert spec.fixed_cache_bytes_per_batch == 449_372_160 + 7_077_888
    assert spec.mtp_layer_index is None
    assert spec.mtp_included is False


@pytest.mark.parametrize(
    ("context_tokens", "mla_capacity_tokens"),
    [(1, 256), (127, 256), (128, 256), (129, 256), (257, 512)],
)
def test_kimi_k3_cache_admission_covers_minimum_and_geometric_capacity(
    context_tokens: int,
    mla_capacity_tokens: int,
) -> None:
    spec = get_model_spec("kimi-k3-q1t")
    plan = plan_expert_memory(
        spec,
        total_limit_bytes=128 * 1024**3,
        context_tokens=context_tokens,
        transient_slots=16,
    )
    fixed_kda_bytes = 449_372_160
    minimum_mla_bytes = 24 * 256 * (512 + 64) * 2
    physical_mla_bytes = 24 * mla_capacity_tokens * (512 + 64) * 2

    assert minimum_mla_bytes == 7_077_888
    assert plan.fixed_cache_bytes == fixed_kda_bytes + minimum_mla_bytes
    assert plan.kv_bytes == context_tokens * 55_296
    assert plan.fixed_cache_bytes + plan.kv_bytes >= (
        fixed_kda_bytes + physical_mla_bytes
    )
    assert plan.fixed_bytes >= (
        plan.resident_bytes
        + plan.kv_bytes
        + plan.fixed_cache_bytes
        + plan.transient_bytes
    )


def test_t158_component_bank_uses_prebound_situ_arithmetic() -> None:
    experts, in_dim, middle_dim, rows = 1, 64, 64, 2
    gate_dense = np.zeros((experts, middle_dim, in_dim), dtype=np.float32)
    up_dense = np.zeros((experts, middle_dim, in_dim), dtype=np.float32)
    down_dense = np.zeros((experts, in_dim, middle_dim), dtype=np.float32)
    gate_dense[0, np.arange(middle_dim), np.arange(in_dim)] = 10.0
    up_dense[0, np.arange(middle_dim), np.arange(in_dim)] = 20.0
    down_dense[0, np.arange(in_dim), np.arange(middle_dim)] = 1.0
    gate_packed, gate_scales = _bank_from_dense(gate_dense)
    up_packed, up_scales = _bank_from_dense(up_dense)
    down_packed, down_scales = _bank_from_dense(down_dense)
    bank = SimpleNamespace(
        arrays={
            "gate_proj.packed": mx.array(gate_packed),
            "gate_proj.scales": mx.array(gate_scales),
            "up_proj.packed": mx.array(up_packed),
            "up_proj.scales": mx.array(up_scales),
            "down_proj.packed": mx.array(down_packed),
            "down_proj.scales": mx.array(down_scales),
        }
    )
    x = np.ones((rows, in_dim), dtype=np.float32)
    expert_rows = np.zeros(rows, dtype=np.int32)

    switch = HotExpertSwitchGLU(_switch_runtime(), layer_index=1)
    bindings = tuple(
        SimpleNamespace(
            buffer=SimpleNamespace(bank=bank, bank_index=int(expert)),
        )
        for expert in expert_rows
    )
    output = switch._dispatch_component_bank(
        mx.array(x).astype(mx.bfloat16),
        bindings,
    )

    x_bf16 = np.asarray(mx.array(x).astype(mx.bfloat16).astype(mx.float32))
    reference_rows: list[np.ndarray] = []
    for row, expert in enumerate(expert_rows):
        gate = (
            decode_shadow(
                "t158",
                gate_packed[expert],
                gate_scales[expert],
                in_dim,
            )
            @ x_bf16[row]
        )
        up = (
            decode_shadow(
                "t158",
                up_packed[expert],
                up_scales[expert],
                in_dim,
            )
            @ x_bf16[row]
        )
        hidden = (
            4.0
            * np.tanh(gate.astype(np.float32) / 4.0)
            * (1.0 / (1.0 + np.exp(-gate.astype(np.float32))))
            * 25.0
            * np.tanh(up.astype(np.float32) / 25.0)
        )
        reference_rows.append(
            decode_shadow(
                "t158",
                down_packed[expert],
                down_scales[expert],
                middle_dim,
            )
            @ np.asarray(mx.array(hidden).astype(mx.bfloat16).astype(mx.float32))
        )

    assert output.dtype == mx.bfloat16
    reference = np.stack(reference_rows)
    np.testing.assert_allclose(
        np.asarray(output.astype(mx.float32)),
        reference,
        rtol=0.002,
        atol=0.2,
    )
    wrong_swiglu = decode_shadow(
        "t158",
        down_packed[0],
        down_scales[0],
        middle_dim,
    ) @ (
        10.0
        * (1.0 / (1.0 + np.exp(-10.0)))
        * 20.0
        * np.ones(middle_dim, dtype=np.float32)
    )
    assert np.max(np.abs(reference[0] - wrong_swiglu)) > 100.0


def _switch_runtime(*, codec: str = "t158", slot_layout: str = "component-banks"):
    return SimpleNamespace(
        spec=SimpleNamespace(
            quant_group_size=64,
            quant_bits=2,
            expert_activation="situ",
            expert_codec=codec,
        ),
        config=SimpleNamespace(slot_layout=slot_layout),
    )


def test_kimi_situ_switch_binds_activation_once_at_construction() -> None:
    switch = HotExpertSwitchGLU(_switch_runtime(), layer_index=1)

    assert callable(switch._component_bank_executor)
    gate = mx.array([[10.0]], dtype=mx.bfloat16)
    up = mx.array([[20.0]], dtype=mx.bfloat16)
    actual = switch._expert_activation(gate, up)
    expected = (
        4.0
        * mx.tanh(gate.astype(mx.float32) / 4.0)
        * mx.sigmoid(gate.astype(mx.float32))
        * 25.0
        * mx.tanh(up.astype(mx.float32) / 25.0)
    ).astype(mx.bfloat16)

    assert mx.array_equal(actual, expected).item()


def test_kimi_situ_switch_fails_closed_without_t158_component_banks() -> None:
    for runtime in (
        _switch_runtime(codec="b1"),
        _switch_runtime(slot_layout="slots"),
    ):
        try:
            HotExpertSwitchGLU(runtime, layer_index=1)
        except ValueError as exc:
            assert "SITU streamed experts require t158 component-bank" in str(exc)
        else:
            raise AssertionError("invalid K3 expert lane was installed")
