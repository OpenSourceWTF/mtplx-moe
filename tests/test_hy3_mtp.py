"""Hy3 layer-80 NextN head: structure, loading, and fail-closed integrity."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

import mtplx.hy3_mtp_patch as hy3_mtp_patch
from mtplx.hy3_mtp_patch import (
    HY3_MTP_BF16_FILE,
    HY3_MTP_EXPERTS_FILE,
    HY3_MTP_RESIDENTS_FILE,
    HY3_MTP_SOURCE_REPO,
    Hy3MTPLoadError,
    build_hy3_mtp_module,
    expected_bf16_names,
    expected_expert_names,
    expected_resident_names,
    load_hy3_mtp_bf16_weights,
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
TEST_METADATA = {
    "source_repo": HY3_MTP_SOURCE_REPO,
    "source_revision": TEST_REVISION,
}


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
    metadata = {"source_repo": HY3_MTP_SOURCE_REPO, "source_revision": revision}
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


def _write_tiny_bf16_artifact(tmp_path: Path, *, revision: str = TEST_REVISION) -> Path:
    """Author the layer-80 BF16 fixture, mirroring the Q4 artifact writer.

    The BF16 artifact is a straight extraction of the source checkpoint:
    every layer-80 tensor as one BF16 ``.weight`` leaf (per-expert, not
    stacked), except the F32 router correction bias.
    """

    mx.random.seed(13)
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    tensors = _tiny_source_residents(prefix)
    for expert in range(args.num_experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"{prefix}mlp.experts.{expert}.{projection}.weight"] = (
                mx.random.normal((64, 64)).astype(mx.bfloat16)
            )
    mx.eval(list(tensors.values()))
    metadata = {"source_repo": HY3_MTP_SOURCE_REPO, "source_revision": revision}
    mx.save_safetensors(str(tmp_path / HY3_MTP_BF16_FILE), tensors, metadata=metadata)
    return tmp_path


def test_expected_name_tables_cover_the_packaged_artifacts(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path)
    assert len(expected_resident_names(args)) == 8 * 3 + 9
    assert len(expected_expert_names(args)) == args.num_experts * 3 * 3


def test_build_hy3_mtp_module_loads_quantized_resident_head(tmp_path: Path) -> None:
    # precision="q4" must keep today's quantized head behavior unchanged.
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path)
    mtp = build_hy3_mtp_module(
        tmp_path, args, expected_revision=TEST_REVISION, precision="q4"
    )

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
    mtp = build_hy3_mtp_module(
        tmp_path, args, expected_revision=TEST_REVISION, precision="q4"
    )
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


@pytest.mark.parametrize("precision", ["q4", "bf16"])
@pytest.mark.parametrize("lm_head_fp32", [False, True])
def test_hy3_mtp_recycles_the_final_normalized_hidden(
    tmp_path: Path, precision: str, lm_head_fp32: bool
) -> None:
    """The reference NextN recurrence consumes the LM-head-normalized state."""

    args = _tiny_args()
    args.enable_lm_head_fp32 = lm_head_fp32
    if precision == "bf16":
        _write_tiny_bf16_artifact(tmp_path)
    else:
        _write_tiny_artifacts(tmp_path)
    mtp = build_hy3_mtp_module(
        tmp_path, args, expected_revision=TEST_REVISION, precision=precision
    )
    trunk = Hy3Model(args)
    captured_head_inputs: list[mx.array] = []

    class RecordingNorm(nn.Module):
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base = base
            self.output = None

        def __call__(self, hidden: mx.array) -> mx.array:
            self.output = self.base(hidden)
            return self.output

    recording_norm = RecordingNorm(mtp.layers[0].final_layernorm)
    mtp.layers[0].final_layernorm = recording_norm

    def capture_head(hidden: mx.array) -> mx.array:
        captured_head_inputs.append(hidden)
        return hidden

    token_ids = mx.array([[3, 5]], dtype=mx.int32)
    previous_hidden = mx.random.normal((1, 2, args.hidden_size)).astype(mx.bfloat16)
    logits, recurrent_hidden = mtp.layers[0](
        token_ids,
        previous_hidden,
        embed_tokens=trunk.model.embed_tokens,
        lm_head=capture_head,
    )
    mx.eval(logits, recurrent_hidden)

    assert len(captured_head_inputs) == 1
    assert recording_norm.output is not None
    assert recurrent_hidden.dtype == recording_norm.output.dtype
    assert mx.array_equal(recurrent_hidden, recording_norm.output).item()
    expected_head_dtype = mx.float32 if lm_head_fp32 else recurrent_hidden.dtype
    assert captured_head_inputs[0].dtype == expected_head_dtype
    assert mx.array_equal(
        captured_head_inputs[0], recurrent_hidden.astype(expected_head_dtype)
    ).item()


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
        metadata=TEST_METADATA,
    )
    with pytest.raises(Hy3MTPLoadError, match="missing tensors"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)

    _write_tiny_artifacts(tmp_path)
    experts = dict(mx.load(str(tmp_path / HY3_MTP_EXPERTS_FILE)))
    del experts[prefix + "mlp.experts.1.down_proj.weight"]
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_EXPERTS_FILE),
        experts,
        metadata=TEST_METADATA,
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
        metadata=TEST_METADATA,
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
        metadata=TEST_METADATA,
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
        metadata=TEST_METADATA,
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


def _module_leaves(mtp) -> list[tuple[str, object]]:
    leaves = []

    def visit(prefix, module):
        leaves.append((prefix, module))
        for name, child in module.children().items():
            if isinstance(child, list):
                for index, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        visit(f"{prefix}.{name}.{index}", item)
            elif isinstance(child, nn.Module):
                visit(f"{prefix}.{name}", child)

    visit("mtp", mtp)
    return leaves


def test_expected_bf16_names_cover_the_full_head(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_bf16_artifact(tmp_path)
    names = expected_bf16_names(args)
    # 8 dense projections + 8 BF16 norms/eh_proj + expert_bias, plus one
    # .weight leaf per expert projection.
    assert len(names) == 8 + 8 + 1 + args.num_experts * 3
    written = set(mx.load(str(tmp_path / HY3_MTP_BF16_FILE)))
    assert written == names


def test_default_build_precision_is_bf16_with_no_quantized_modules(
    tmp_path: Path,
) -> None:
    """Forge contract section 6: quantizing the MTP head collapses MoE
    acceptance to 5-11% (vs 79-85% BF16), so bf16 is the default and only
    the BF16 artifact is required."""

    args = _tiny_args()
    _write_tiny_bf16_artifact(tmp_path)  # no Q4 artifacts on disk at all
    mtp = build_hy3_mtp_module(tmp_path, args, expected_revision=TEST_REVISION)

    assert isinstance(mtp, Hy3MTP)
    layer = mtp.layers[0]
    assert isinstance(layer, Hy3MTPLayer)

    # The whole head is plain BF16: no quantized module anywhere.
    for path, module in _module_leaves(mtp):
        assert not isinstance(module, (nn.QuantizedLinear, QuantizedSwitchLinear)), path
    switch = layer.mtp_block.mlp.switch_mlp
    assert tuple(switch.gate_proj.weight.shape) == (2, 64, 64)
    assert switch.gate_proj.weight.dtype == mx.bfloat16
    attn = layer.mtp_block.self_attn
    assert isinstance(attn.q_proj, nn.Linear)
    assert attn.q_proj.weight.dtype == mx.bfloat16
    router = layer.mtp_block.mlp.router
    assert isinstance(router.gate, nn.Linear)
    assert router.gate.weight.dtype == mx.bfloat16
    assert router.expert_bias.dtype == mx.float32
    assert layer.eh_proj.weight.dtype == mx.bfloat16

    # 17 resident leaves + 3 stacked expert leaves, nothing else.
    parameters = dict(tree_flatten(mtp.parameters()))
    assert len(parameters) == 20
    assert all(name.startswith("layers.0.") for name in parameters)


def test_bf16_head_forward_produces_finite_logits_and_hidden(
    tmp_path: Path,
) -> None:
    args = _tiny_args()
    _write_tiny_bf16_artifact(tmp_path)
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


def test_bf16_stacked_expert_weights_match_artifact_order(tmp_path: Path) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    _write_tiny_bf16_artifact(tmp_path)
    weights = load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)
    source = mx.load(str(tmp_path / HY3_MTP_BF16_FILE))
    stacked = weights["layers.0.mtp_block.mlp.switch_mlp.gate_proj.weight"]
    assert tuple(stacked.shape) == (2, 64, 64)
    for expert in range(2):
        original = source[f"{prefix}mlp.experts.{expert}.gate_proj.weight"]
        assert mx.array_equal(stacked[expert], original).item()
    # Residents map onto the same Hy3MTPLayer paths as the Q4 loader.
    assert "layers.0.eh_proj.weight" in weights
    assert "layers.0.mtp_block.mlp.router.expert_bias" in weights
    assert "layers.0.mtp_block.self_attn.q_proj.weight" in weights


def test_bf16_loader_fails_closed(tmp_path: Path) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."

    # Missing artifact file (only the Q4 artifacts on disk).
    _write_tiny_artifacts(tmp_path)
    with pytest.raises(Hy3MTPLoadError, match="missing Hy3 MTP artifact"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)

    # Revision mismatch.
    _write_tiny_bf16_artifact(tmp_path, revision="unexpected")
    with pytest.raises(Hy3MTPLoadError, match="revision"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)

    # Missing tensor.
    _write_tiny_bf16_artifact(tmp_path)
    tensors = dict(mx.load(str(tmp_path / HY3_MTP_BF16_FILE)))
    del tensors[prefix + "mlp.experts.1.down_proj.weight"]
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_BF16_FILE),
        tensors,
        metadata=TEST_METADATA,
    )
    with pytest.raises(Hy3MTPLoadError, match="missing tensors"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)

    # Unexpected tensor.
    _write_tiny_bf16_artifact(tmp_path)
    tensors = dict(mx.load(str(tmp_path / HY3_MTP_BF16_FILE)))
    tensors[prefix + "mlp.experts.7.gate_proj.weight"] = mx.zeros(
        (64, 64), dtype=mx.bfloat16
    )
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_BF16_FILE),
        tensors,
        metadata=TEST_METADATA,
    )
    with pytest.raises(Hy3MTPLoadError, match="unexpected tensors"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)

    # Wrong dtypes: BF16-only weights and the F32 correction bias.
    _write_tiny_bf16_artifact(tmp_path)
    tensors = dict(mx.load(str(tmp_path / HY3_MTP_BF16_FILE)))
    tensors[prefix + "mlp.expert_bias"] = mx.zeros((2,), dtype=mx.bfloat16)
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_BF16_FILE),
        tensors,
        metadata=TEST_METADATA,
    )
    with pytest.raises(Hy3MTPLoadError, match="must be float32"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)

    _write_tiny_bf16_artifact(tmp_path)
    tensors = dict(mx.load(str(tmp_path / HY3_MTP_BF16_FILE)))
    tensors[prefix + "eh_proj.weight"] = mx.zeros((64, 128), dtype=mx.float32)
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_BF16_FILE),
        tensors,
        metadata=TEST_METADATA,
    )
    with pytest.raises(Hy3MTPLoadError, match="must be"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)

    # Inconsistent expert shapes.
    _write_tiny_bf16_artifact(tmp_path)
    tensors = dict(mx.load(str(tmp_path / HY3_MTP_BF16_FILE)))
    tensors[prefix + "mlp.experts.1.up_proj.weight"] = mx.zeros(
        (32, 64), dtype=mx.bfloat16
    )
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_BF16_FILE),
        tensors,
        metadata=TEST_METADATA,
    )
    with pytest.raises(Hy3MTPLoadError, match="differs from expert 0"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)


def test_build_rejects_unknown_precision(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_bf16_artifact(tmp_path)
    with pytest.raises(Hy3MTPLoadError, match="precision"):
        build_hy3_mtp_module(
            tmp_path, args, expected_revision=TEST_REVISION, precision="fp8"
        )


def _write_preflight_safetensors(
    path: Path,
    *,
    revision: object = TEST_REVISION,
    source_repo: object = HY3_MTP_SOURCE_REPO,
    tensors: tuple[tuple[str, str, list[int], bytes], ...] = (
        ("weight", "BF16", [3], b"\0" * 6),
    ),
) -> int:
    header: dict[str, object] = {
        "__metadata__": {
            "source_repo": source_repo,
            "source_revision": revision,
        }
    }
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return len(payload)


def test_preflight_bf16_returns_exact_payload_without_loading_mlx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _write_preflight_safetensors(tmp_path / HY3_MTP_BF16_FILE)
    monkeypatch.setattr(
        mx,
        "load",
        lambda *_args, **_kwargs: pytest.fail("preflight must not call MLX"),
    )

    assert (
        hy3_mtp_patch.preflight_hy3_mtp_artifacts(
            tmp_path,
            precision="bf16",
            expected_revision=TEST_REVISION,
        )
        == expected
    )


def test_preflight_q4_charges_both_complete_tensor_payloads(tmp_path: Path) -> None:
    resident_bytes = _write_preflight_safetensors(
        tmp_path / HY3_MTP_RESIDENTS_FILE,
        tensors=(
            ("resident", "BF16", [2], b"r" * 4),
            ("router", "F32", [1], b"g" * 4),
        ),
    )
    expert_bytes = _write_preflight_safetensors(
        tmp_path / HY3_MTP_EXPERTS_FILE,
        tensors=(
            ("expert", "U32", [3], b"e" * 12),
            # The real expert file carries resident pass-through tensors too;
            # charge the entire payload conservatively, even if loading skips one.
            ("resident_passthrough", "BF16", [2], b"p" * 4),
        ),
    )
    (tmp_path / HY3_MTP_BF16_FILE).write_bytes(b"not selected")

    assert (
        hy3_mtp_patch.preflight_hy3_mtp_artifacts(
            tmp_path,
            precision="q4",
            expected_revision=TEST_REVISION,
        )
        == resident_bytes + expert_bytes
    )


def test_preflight_reads_only_the_selected_precision_files(tmp_path: Path) -> None:
    expected = _write_preflight_safetensors(tmp_path / HY3_MTP_BF16_FILE)
    (tmp_path / HY3_MTP_RESIDENTS_FILE).write_bytes(b"malformed but unselected")
    (tmp_path / HY3_MTP_EXPERTS_FILE).mkdir()

    assert (
        hy3_mtp_patch.preflight_hy3_mtp_artifacts(
            tmp_path,
            precision="bf16",
            expected_revision=TEST_REVISION,
        )
        == expected
    )


@pytest.mark.parametrize("precision", ["", "fp8", "BF16"])
def test_preflight_rejects_unknown_precision(tmp_path: Path, precision: str) -> None:
    with pytest.raises(Hy3MTPLoadError, match="precision"):
        hy3_mtp_patch.preflight_hy3_mtp_artifacts(tmp_path, precision=precision)


def test_preflight_rejects_missing_nonregular_and_symlink_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(Hy3MTPLoadError, match="missing"):
        hy3_mtp_patch.preflight_hy3_mtp_artifacts(
            tmp_path,
            precision="bf16",
            expected_revision=TEST_REVISION,
        )

    (tmp_path / HY3_MTP_BF16_FILE).mkdir()
    with pytest.raises(Hy3MTPLoadError, match="regular file"):
        hy3_mtp_patch.preflight_hy3_mtp_artifacts(
            tmp_path,
            precision="bf16",
            expected_revision=TEST_REVISION,
        )

    (tmp_path / HY3_MTP_BF16_FILE).rmdir()
    target = tmp_path / "target.safetensors"
    _write_preflight_safetensors(target)
    (tmp_path / HY3_MTP_BF16_FILE).symlink_to(target)
    with pytest.raises(Hy3MTPLoadError, match="regular file"):
        hy3_mtp_patch.preflight_hy3_mtp_artifacts(
            tmp_path,
            precision="bf16",
            expected_revision=TEST_REVISION,
        )


@pytest.mark.parametrize("revision", [None, 7, "other-revision"])
def test_preflight_rejects_missing_malformed_or_wrong_revision(
    tmp_path: Path, revision: object
) -> None:
    _write_preflight_safetensors(
        tmp_path / HY3_MTP_BF16_FILE,
        revision=revision,
    )
    with pytest.raises(Hy3MTPLoadError, match="source_revision|revision"):
        hy3_mtp_patch.preflight_hy3_mtp_artifacts(
            tmp_path,
            precision="bf16",
            expected_revision=TEST_REVISION,
        )


def test_preflight_rejects_wrong_source_repository(tmp_path: Path) -> None:
    _write_preflight_safetensors(
        tmp_path / HY3_MTP_BF16_FILE,
        source_repo="attacker/repacked",
    )
    with pytest.raises(Hy3MTPLoadError, match="source_repo|repository"):
        hy3_mtp_patch.preflight_hy3_mtp_artifacts(
            tmp_path,
            precision="bf16",
            expected_revision=TEST_REVISION,
        )


def test_preflight_rejects_oversized_or_truncated_headers(tmp_path: Path) -> None:
    path = tmp_path / HY3_MTP_BF16_FILE
    path.write_bytes(struct.pack("<Q", 64 * 1024 * 1024 + 1))
    with pytest.raises(Hy3MTPLoadError, match="bounded"):
        hy3_mtp_patch.preflight_hy3_mtp_artifacts(
            tmp_path,
            precision="bf16",
            expected_revision=TEST_REVISION,
        )

    path.write_bytes(struct.pack("<Q", 128) + b"{}")
    with pytest.raises(Hy3MTPLoadError, match="truncated"):
        hy3_mtp_patch.preflight_hy3_mtp_artifacts(
            tmp_path,
            precision="bf16",
            expected_revision=TEST_REVISION,
        )


@pytest.mark.parametrize(
    ("header", "payload", "message"),
    [
        (
            {
                "__metadata__": TEST_METADATA,
                "weight": {
                    "dtype": "BF16",
                    "shape": [2],
                    "data_offsets": [0, 2],
                },
            },
            b"xx",
            "byte-count",
        ),
        (
            {
                "__metadata__": TEST_METADATA,
                "first": {
                    "dtype": "BF16",
                    "shape": [1],
                    "data_offsets": [0, 2],
                },
                "second": {
                    "dtype": "BF16",
                    "shape": [1],
                    "data_offsets": [3, 5],
                },
            },
            b"12345",
            "contiguous",
        ),
        (
            {
                "__metadata__": TEST_METADATA,
                "weight": {
                    "dtype": "BF16",
                    "shape": [1],
                    "data_offsets": [0, 2],
                },
            },
            b"xxtrailing",
            "trailing",
        ),
        (
            {
                "__metadata__": TEST_METADATA,
                "weight": {
                    "dtype": "BF16",
                    "shape": [True],
                    "data_offsets": [0, 2],
                },
            },
            b"xx",
            "shape",
        ),
    ],
)
def test_preflight_rejects_malformed_tensor_ranges(
    tmp_path: Path,
    header: dict[str, object],
    payload: bytes,
    message: str,
) -> None:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    (tmp_path / HY3_MTP_BF16_FILE).write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )

    with pytest.raises(Hy3MTPLoadError, match=message):
        hy3_mtp_patch.preflight_hy3_mtp_artifacts(
            tmp_path,
            precision="bf16",
            expected_revision=TEST_REVISION,
        )


def test_verified_bf16_handle_survives_atomic_path_replacement(
    tmp_path: Path,
) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    artifact_path = _write_tiny_bf16_artifact(tmp_path) / HY3_MTP_BF16_FILE
    original = mx.load(str(artifact_path))[prefix + "eh_proj.weight"]
    mx.eval(original)

    replacement_dir = tmp_path / "replacement"
    replacement_dir.mkdir()
    replacement_path = _write_tiny_bf16_artifact(replacement_dir) / HY3_MTP_BF16_FILE
    replacement = dict(mx.load(str(replacement_path)))
    replacement[prefix + "eh_proj.weight"] = mx.zeros_like(
        replacement[prefix + "eh_proj.weight"]
    )
    mx.save_safetensors(
        str(replacement_path),
        replacement,
        metadata=TEST_METADATA,
    )

    with hy3_mtp_patch.open_verified_hy3_mtp_artifacts(
        tmp_path,
        precision="bf16",
        expected_revision=TEST_REVISION,
    ) as verified:
        assert verified.root == tmp_path.resolve()
        assert verified.precision == "bf16"
        assert set(verified.files) == {HY3_MTP_BF16_FILE}
        assert verified.payload_bytes > 0

        os.replace(replacement_path, artifact_path)
        weights = load_hy3_mtp_bf16_weights(
            tmp_path,
            args,
            expected_revision=TEST_REVISION,
            verified_artifacts=verified,
        )
        loaded = weights["layers.0.eh_proj.weight"]
        mx.eval(loaded)
        assert mx.array_equal(loaded, original).item()
        assert not mx.array_equal(
            loaded,
            mx.load(str(artifact_path))[prefix + "eh_proj.weight"],
        ).item()


def test_verified_handle_allows_path_replacement_during_header_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / HY3_MTP_BF16_FILE
    expected = _write_preflight_safetensors(artifact_path)
    replacement_path = tmp_path / "replacement.safetensors"
    _write_preflight_safetensors(
        replacement_path,
        tensors=(("replacement", "BF16", [2], b"r" * 4),),
    )
    real_pread_exact = hy3_mtp_patch._pread_exact
    replaced = False

    def replace_after_read(fd, size, offset, *, label):
        nonlocal replaced
        data = real_pread_exact(fd, size, offset, label=label)
        if offset == 8 and not replaced:
            os.replace(replacement_path, artifact_path)
            replaced = True
        return data

    monkeypatch.setattr(hy3_mtp_patch, "_pread_exact", replace_after_read)
    with hy3_mtp_patch.open_verified_hy3_mtp_artifacts(
        tmp_path,
        precision="bf16",
        expected_revision=TEST_REVISION,
    ) as verified:
        assert verified.payload_bytes == expected
        assert replaced


def test_verified_handle_rejects_in_place_mutation_before_context_exit(
    tmp_path: Path,
) -> None:
    artifact_path = _write_tiny_bf16_artifact(tmp_path) / HY3_MTP_BF16_FILE

    with pytest.raises(Hy3MTPLoadError, match="changed while in use"):
        with hy3_mtp_patch.open_verified_hy3_mtp_artifacts(
            tmp_path,
            precision="bf16",
            expected_revision=TEST_REVISION,
        ):
            with artifact_path.open("ab") as artifact:
                artifact.write(b"mutation")
                artifact.flush()
                os.fsync(artifact.fileno())


def test_verified_handle_rejects_mutation_with_restored_mtime(tmp_path: Path) -> None:
    artifact_path = _write_tiny_bf16_artifact(tmp_path) / HY3_MTP_BF16_FILE
    before = artifact_path.stat()

    with pytest.raises(Hy3MTPLoadError, match="changed while in use"):
        with hy3_mtp_patch.open_verified_hy3_mtp_artifacts(
            tmp_path,
            precision="bf16",
            expected_revision=TEST_REVISION,
        ):
            with artifact_path.open("r+b") as artifact:
                artifact.seek(-1, os.SEEK_END)
                byte = artifact.read(1)
                artifact.seek(-1, os.SEEK_END)
                artifact.write(bytes([byte[0] ^ 0xFF]))
                artifact.flush()
                os.fsync(artifact.fileno())
            os.utime(
                artifact_path,
                ns=(before.st_atime_ns, before.st_mtime_ns),
            )


def test_verified_handle_checks_mutation_when_consumer_raises(tmp_path: Path) -> None:
    artifact_path = _write_tiny_bf16_artifact(tmp_path) / HY3_MTP_BF16_FILE

    with pytest.raises(Hy3MTPLoadError, match="changed while in use"):
        with hy3_mtp_patch.open_verified_hy3_mtp_artifacts(
            tmp_path,
            precision="bf16",
            expected_revision=TEST_REVISION,
        ):
            with artifact_path.open("ab") as artifact:
                artifact.write(b"mutation")
                artifact.flush()
                os.fsync(artifact.fileno())
            raise ValueError("consumer failed")


def test_bf16_loader_passes_held_file_object_to_mlx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _tiny_args()
    _write_tiny_bf16_artifact(tmp_path)
    real_load = mx.load
    observed = []

    def record_load(source, *args, **kwargs):
        observed.append(source)
        assert hasattr(source, "read") and not isinstance(source, (str, Path))
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(mx, "load", record_load)
    weights = load_hy3_mtp_bf16_weights(
        tmp_path,
        args,
        expected_revision=TEST_REVISION,
        mx_module=mx,
    )
    mx.eval(weights)
    assert len(observed) == 1


def test_build_accepts_borrowed_q4_handles(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path)

    with hy3_mtp_patch.open_verified_hy3_mtp_artifacts(
        tmp_path,
        precision="q4",
        expected_revision=TEST_REVISION,
    ) as verified:
        assert set(verified.files) == {
            HY3_MTP_RESIDENTS_FILE,
            HY3_MTP_EXPERTS_FILE,
        }
        mtp = build_hy3_mtp_module(
            tmp_path,
            args,
            expected_revision=TEST_REVISION,
            precision="q4",
            verified_artifacts=verified,
        )

    assert isinstance(mtp, Hy3MTP)
    assert all(handle.closed for handle in verified.files.values())
