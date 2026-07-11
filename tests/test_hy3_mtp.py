"""Hy3 layer-80 NextN head: structure, loading, and fail-closed integrity."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from mtplx.hy3_mtp_patch import (
    HY3_MTP_EXPERTS_FILE,
    HY3_MTP_RESIDENTS_FILE,
    Hy3MTPLoadError,
    build_hy3_mtp_module,
    expected_expert_names,
    expected_resident_names,
    load_hy3_mtp_weights,
)
from mtplx.models.hy3_mlx import Hy3MTP, Hy3MTPLayer
from mtplx.models.hy3_mlx import Model as Hy3Model
from mtplx.models.hy3_mlx import ModelArgs as Hy3Args

_RESIDENTS_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "quantize_mtp_layer80_residents.py"
)
TEST_REVISION = "test-revision"


def _load_residents_script():
    spec = importlib.util.spec_from_file_location(
        "quantize_mtp_layer80_residents", _RESIDENTS_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _tiny_source_residents(prefix: str) -> dict[str, mx.array]:
    def bf16(*shape):
        return mx.random.normal(shape).astype(mx.bfloat16)

    tensors = {
        prefix + "self_attn.q_proj.weight": bf16(64, 64),
        prefix + "self_attn.k_proj.weight": bf16(32, 64),
        prefix + "self_attn.v_proj.weight": bf16(32, 64),
        prefix + "self_attn.o_proj.weight": bf16(64, 64),
        prefix + "mlp.shared_mlp.gate_proj.weight": bf16(64, 64),
        prefix + "mlp.shared_mlp.up_proj.weight": bf16(64, 64),
        prefix + "mlp.shared_mlp.down_proj.weight": bf16(64, 64),
        prefix + "mlp.router.gate.weight": bf16(2, 64),
        prefix + "eh_proj.weight": bf16(64, 128),
        prefix + "enorm.weight": bf16(64),
        prefix + "hnorm.weight": bf16(64),
        prefix + "input_layernorm.weight": bf16(64),
        prefix + "post_attention_layernorm.weight": bf16(64),
        prefix + "final_layernorm.weight": bf16(64),
        prefix + "self_attn.q_norm.weight": bf16(16),
        prefix + "self_attn.k_norm.weight": bf16(16),
        prefix + "mlp.expert_bias": mx.zeros((2,), dtype=mx.float32),
    }
    mx.eval(list(tensors.values()))
    return tensors


def _write_tiny_artifacts(tmp_path: Path, *, revision: str = TEST_REVISION) -> Path:
    """Author both layer-80 fixture artifacts with the packaging scripts' formats."""

    mx.random.seed(11)
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    metadata = {"source_repo": "test/tiny-hy3", "source_revision": revision}
    residents_mod = _load_residents_script()
    residents = residents_mod.quantize_resident_tensors(
        _tiny_source_residents(prefix), layer_prefix=prefix
    )
    mx.eval(list(residents.values()))
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_RESIDENTS_FILE), residents, metadata=metadata
    )

    experts: dict[str, mx.array] = {}
    for expert in range(args.num_experts):
        for projection, shape in (
            ("gate_proj", (64, 64)),
            ("up_proj", (64, 64)),
            ("down_proj", (64, 64)),
        ):
            source = mx.random.normal(shape).astype(mx.bfloat16)
            weight, scales, biases = mx.quantize(
                source, group_size=64, bits=4, mode="affine"
            )
            base = f"{prefix}mlp.experts.{expert}.{projection}"
            experts[base + ".weight"] = weight
            experts[base + ".scales"] = scales
            experts[base + ".biases"] = biases
    # The real expert artifact also carries BF16 resident pass-through copies;
    # the loader must ignore them rather than double-load residents.
    experts[prefix + "eh_proj.weight"] = mx.zeros((64, 128), dtype=mx.bfloat16)
    mx.eval(list(experts.values()))
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_EXPERTS_FILE), experts, metadata=metadata
    )
    return tmp_path


def test_expected_name_tables_cover_the_packaged_artifacts(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path)
    assert len(expected_resident_names(args)) == 8 * 3 + 9
    assert len(expected_expert_names(args)) == args.num_experts * 3 * 3


def test_build_hy3_mtp_module_loads_quantized_resident_head(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path)
    mtp = build_hy3_mtp_module(tmp_path, args, expected_revision=TEST_REVISION)

    assert isinstance(mtp, Hy3MTP)
    assert len(mtp.layers) == 1
    assert mtp.start_layer == args.num_hidden_layers
    layer = mtp.layers[0]
    assert isinstance(layer, Hy3MTPLayer)

    # Experts are one resident stacked SwitchGLU in the pinned Q4 format.
    switch = layer.mtp_block.mlp.switch_mlp
    assert isinstance(switch.gate_proj, QuantizedSwitchLinear)
    assert switch.gate_proj.bits == 4 and switch.gate_proj.group_size == 64
    assert tuple(switch.gate_proj.weight.shape) == (2, 64, 8)
    assert tuple(switch.down_proj.scales.shape) == (2, 64, 1)

    # Residents mirror trunk conventions: Q4 attention, Q8 router gate,
    # BF16 eh_proj/norms, F32 correction bias.
    attn = layer.mtp_block.self_attn
    assert isinstance(attn.q_proj, nn.QuantizedLinear) and attn.q_proj.bits == 4
    router = layer.mtp_block.mlp.router
    assert isinstance(router.gate, nn.QuantizedLinear) and router.gate.bits == 8
    assert router.expert_bias.dtype == mx.float32
    assert isinstance(layer.eh_proj, nn.Linear)
    assert not isinstance(layer.eh_proj, nn.QuantizedLinear)
    assert layer.eh_proj.weight.dtype == mx.bfloat16
    assert layer.final_layernorm.weight.dtype == mx.bfloat16

    # 33 resident leaves + 9 stacked expert leaves, nothing else.
    parameters = dict(tree_flatten(mtp.parameters()))
    assert len(parameters) == 42
    assert all(name.startswith("layers.0.") for name in parameters)


def test_hy3_mtp_layer_forward_produces_finite_logits_and_hidden(
    tmp_path: Path,
) -> None:
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path)
    mtp = build_hy3_mtp_module(tmp_path, args, expected_revision=TEST_REVISION)
    trunk = Hy3Model(args)

    token_ids = mx.array([[3, 5]], dtype=mx.int32)
    previous_hidden = mx.random.normal((1, 2, args.hidden_size)).astype(mx.bfloat16)
    logits, hidden = mtp.layers[0](
        token_ids,
        previous_hidden,
        embed_tokens=trunk.model.embed_tokens,
        lm_head=trunk.lm_head,
    )
    mx.eval(logits, hidden)

    assert logits.shape == (1, 2, args.vocab_size)
    assert hidden.shape == (1, 2, args.hidden_size)
    assert mx.all(mx.isfinite(logits.astype(mx.float32))).item()
    assert mx.all(mx.isfinite(hidden.astype(mx.float32))).item()


def test_loader_rejects_revision_mismatch(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path, revision="unexpected")
    with pytest.raises(Hy3MTPLoadError, match="revision"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)


def test_loader_rejects_missing_artifact_files(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path)
    (tmp_path / HY3_MTP_EXPERTS_FILE).rename(tmp_path / "renamed.safetensors")
    with pytest.raises(Hy3MTPLoadError, match="missing Hy3 MTP artifact"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)


def test_loader_rejects_missing_and_unexpected_tensors(tmp_path: Path) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    _write_tiny_artifacts(tmp_path)

    residents = dict(mx.load(str(tmp_path / HY3_MTP_RESIDENTS_FILE)))
    del residents[prefix + "mlp.router.gate.scales"]
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_RESIDENTS_FILE),
        residents,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="missing tensors"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)

    _write_tiny_artifacts(tmp_path)
    experts = dict(mx.load(str(tmp_path / HY3_MTP_EXPERTS_FILE)))
    del experts[prefix + "mlp.experts.1.down_proj.weight"]
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_EXPERTS_FILE),
        experts,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="missing expert tensors"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)

    _write_tiny_artifacts(tmp_path)
    experts = dict(mx.load(str(tmp_path / HY3_MTP_EXPERTS_FILE)))
    experts[prefix + "mlp.experts.7.gate_proj.weight"] = mx.zeros(
        (64, 8), dtype=mx.uint32
    )
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_EXPERTS_FILE),
        experts,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="unexpected expert tensors"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)


def test_loader_rejects_wrong_leaf_dtypes(tmp_path: Path) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    _write_tiny_artifacts(tmp_path)
    residents = dict(mx.load(str(tmp_path / HY3_MTP_RESIDENTS_FILE)))
    residents[prefix + "mlp.expert_bias"] = mx.zeros((2,), dtype=mx.bfloat16)
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_RESIDENTS_FILE),
        residents,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="must be float32"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)


def test_loader_rejects_inconsistent_expert_shapes(tmp_path: Path) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    _write_tiny_artifacts(tmp_path)
    experts = dict(mx.load(str(tmp_path / HY3_MTP_EXPERTS_FILE)))
    experts[prefix + "mlp.experts.1.up_proj.weight"] = mx.zeros(
        (32, 8), dtype=mx.uint32
    )
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_EXPERTS_FILE),
        experts,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="differs from expert 0"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)


def test_stacked_expert_weights_match_artifact_order(tmp_path: Path) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    _write_tiny_artifacts(tmp_path)
    weights = load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)
    experts = mx.load(str(tmp_path / HY3_MTP_EXPERTS_FILE))
    stacked = weights["layers.0.mtp_block.mlp.switch_mlp.gate_proj.weight"]
    assert tuple(stacked.shape) == (2, 64, 8)
    for expert in range(2):
        source = experts[f"{prefix}mlp.experts.{expert}.gate_proj.weight"]
        assert mx.array_equal(stacked[expert], source).item()
    # Residents come from the residents artifact, not the expert artifact's
    # BF16 pass-through copy.
    assert not mx.array_equal(
        weights["layers.0.eh_proj.weight"],
        experts[prefix + "eh_proj.weight"],
    ).item()
