"""Runtime MTP injection for streamed Hy3: the layer-80 NextN head.

The pinned pipenetwork/Hy3-4bit artifact omits checkpoint layer 80, so the
head is packaged separately from the official tencent/Hy3 BF16 weights:

    layer80-residents-q.safetensors  attention/router/shared/norms/eh_proj in
                                     the pinned resident conventions
                                     (scripts/quantize_mtp_layer80_residents.py)
    layer80-q4.safetensors           192 routed experts in the pinned affine
                                     Q4/gs64 expert segment format
                                     (scripts/quantize_mtp_layer80.py)

Unlike trunk layers 1-79 the head's experts are fully resident: they are
stacked into one quantized ``SwitchGLU`` at MTP-enable time.  Loading is
fail-closed: unexpected, missing, or wrongly typed tensors and revision
mismatches abort instead of degrading.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HY3_MTP_SOURCE_REPO = "tencent/Hy3"
HY3_MTP_SOURCE_REVISION = "716aa7241bd6d95896be4ebfc761162a9c4d49ef"
HY3_MTP_RESIDENTS_FILE = "layer80-residents-q.safetensors"
HY3_MTP_EXPERTS_FILE = "layer80-q4.safetensors"

_QUANTIZED_RESIDENT_BASES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.shared_mlp.gate_proj",
    "mlp.shared_mlp.up_proj",
    "mlp.shared_mlp.down_proj",
    "mlp.router.gate",
)
_BF16_RESIDENT_SUFFIXES = (
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "final_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)
_F32_RESIDENT_SUFFIXES = ("mlp.expert_bias",)
_HEAD_LOCAL_MODULES = ("enorm", "hnorm", "eh_proj", "final_layernorm")
_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_QUANT_LEAVES = ("weight", "scales", "biases")


class Hy3MTPLoadError(RuntimeError):
    pass


def read_safetensors_metadata(path: Path) -> dict[str, str]:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise Hy3MTPLoadError(f"{path} has a malformed __metadata__ block")
    return {str(key): str(value) for key, value in metadata.items()}


def _require_revision(path: Path, expected_revision: str) -> None:
    metadata = read_safetensors_metadata(path)
    revision = metadata.get("source_revision")
    if revision != expected_revision:
        raise Hy3MTPLoadError(
            f"{path.name} was packaged from revision {revision!r}; "
            f"expected {expected_revision!r}"
        )


def _layer_prefix(args: Any) -> str:
    return f"model.layers.{int(args.num_hidden_layers)}."


def expected_resident_names(args: Any) -> set[str]:
    prefix = _layer_prefix(args)
    names = {
        prefix + base + "." + leaf
        for base in _QUANTIZED_RESIDENT_BASES
        for leaf in _QUANT_LEAVES
    }
    names.update(prefix + suffix for suffix in _BF16_RESIDENT_SUFFIXES)
    names.update(prefix + suffix for suffix in _F32_RESIDENT_SUFFIXES)
    return names


def expected_expert_names(args: Any) -> set[str]:
    prefix = _layer_prefix(args)
    return {
        f"{prefix}mlp.experts.{expert}.{projection}.{leaf}"
        for expert in range(int(args.num_experts))
        for projection in _EXPERT_PROJECTIONS
        for leaf in _QUANT_LEAVES
    }


def _resident_target(suffix: str) -> str:
    """Map one layer-80 artifact suffix to its Hy3MTPLayer parameter path."""

    if suffix == "mlp.expert_bias":
        # The pinned artifact stores the router correction bias under the
        # router module; the source checkpoint keeps it on the MLP.
        return "mtp_block.mlp.router.expert_bias"
    if suffix.split(".", 1)[0] in _HEAD_LOCAL_MODULES:
        return suffix
    return "mtp_block." + suffix


def _validate_leaf_dtype(name: str, value: Any, mx: Any) -> None:
    if name.endswith((".scales", ".biases")):
        if value.dtype != mx.bfloat16:
            raise Hy3MTPLoadError(f"{name} must be bfloat16, found {value.dtype}")
    elif name.endswith("mlp.expert_bias"):
        if value.dtype != mx.float32:
            raise Hy3MTPLoadError(f"{name} must be float32, found {value.dtype}")
    elif name.endswith(".weight"):
        base = name.rsplit(".", 1)[0]
        quantized = any(
            base.endswith(candidate) for candidate in _QUANTIZED_RESIDENT_BASES
        ) or ".mlp.experts." in name
        wanted = mx.uint32 if quantized else mx.bfloat16
        if value.dtype != wanted:
            raise Hy3MTPLoadError(f"{name} must be {wanted}, found {value.dtype}")


def load_hy3_mtp_weights(
    artifact_dir: Path | str,
    args: Any,
    *,
    expected_revision: str = HY3_MTP_SOURCE_REVISION,
    mx_module: Any | None = None,
) -> dict[str, Any]:
    """Read and validate both layer-80 artifacts into module-path weights.

    Residents come only from the residents artifact and routed experts only
    from the expert artifact (whose BF16 resident pass-through copies are
    ignored).  Expert projections are stacked into ``switch_mlp`` tensors of
    shape ``[num_experts, ...]``.
    """

    if mx_module is None:
        import mlx.core as mx
    else:
        mx = mx_module
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    residents_path = artifact_dir / HY3_MTP_RESIDENTS_FILE
    experts_path = artifact_dir / HY3_MTP_EXPERTS_FILE
    for path in (residents_path, experts_path):
        if not path.exists():
            raise Hy3MTPLoadError(f"missing Hy3 MTP artifact {path}")
        _require_revision(path, expected_revision)

    prefix = _layer_prefix(args)
    mapped: dict[str, Any] = {}

    residents = mx.load(str(residents_path), format="safetensors")
    expected_residents = expected_resident_names(args)
    missing = expected_residents - set(residents)
    extra = set(residents) - expected_residents
    if missing:
        raise Hy3MTPLoadError(
            f"{residents_path.name} is missing tensors: {sorted(missing)[:4]}"
        )
    if extra:
        raise Hy3MTPLoadError(
            f"{residents_path.name} has unexpected tensors: {sorted(extra)[:4]}"
        )
    for name, value in residents.items():
        _validate_leaf_dtype(name, value, mx)
        mapped["layers.0." + _resident_target(name[len(prefix):])] = value

    experts = mx.load(str(experts_path), format="safetensors")
    expected_experts = expected_expert_names(args)
    missing = expected_experts - set(experts)
    if missing:
        raise Hy3MTPLoadError(
            f"{experts_path.name} is missing expert tensors: {sorted(missing)[:4]}"
        )
    unexpected = {
        name
        for name in set(experts) - expected_experts
        if ".mlp.experts." in name
    }
    if unexpected:
        raise Hy3MTPLoadError(
            f"{experts_path.name} has unexpected expert tensors: "
            f"{sorted(unexpected)[:4]}"
        )
    num_experts = int(args.num_experts)
    reference_shapes: dict[tuple[str, str], tuple[int, ...]] = {}
    for projection in _EXPERT_PROJECTIONS:
        for leaf in _QUANT_LEAVES:
            values = []
            for expert in range(num_experts):
                name = f"{prefix}mlp.experts.{expert}.{projection}.{leaf}"
                value = experts[name]
                _validate_leaf_dtype(name, value, mx)
                shape = tuple(int(dim) for dim in value.shape)
                reference = reference_shapes.setdefault((projection, leaf), shape)
                if shape != reference:
                    raise Hy3MTPLoadError(
                        f"{name} shape {shape} differs from expert 0 {reference}"
                    )
                values.append(value)
            mapped[f"layers.0.mtp_block.mlp.switch_mlp.{projection}.{leaf}"] = (
                mx.stack(values)
            )
    return mapped


def _quantization_spec_for(
    path: str,
    module: Any,
    weights: dict[str, Any],
    *,
    group_size: int,
) -> bool | dict[str, Any]:
    if not hasattr(module, "to_quantized"):
        return False
    packed = weights.get(f"{path}.weight")
    scales = weights.get(f"{path}.scales")
    if packed is None or scales is None:
        return False
    logical_in = int(module.weight.shape[-1])
    packed_words = int(packed.shape[-1])
    bits, remainder = divmod(packed_words * 32, logical_in)
    if remainder or bits not in {4, 8}:
        raise Hy3MTPLoadError(
            f"{path}.weight packs {packed_words} words for {logical_in} inputs; "
            "cannot derive a 4/8-bit affine layout"
        )
    derived_group, remainder = divmod(logical_in, int(scales.shape[-1]))
    if remainder or derived_group != group_size:
        raise Hy3MTPLoadError(
            f"{path}.scales implies group size {derived_group}; "
            f"expected {group_size}"
        )
    return {"bits": bits, "group_size": group_size, "mode": "affine"}


def build_hy3_mtp_module(
    artifact_dir: Path | str,
    args: Any,
    *,
    expected_revision: str = HY3_MTP_SOURCE_REVISION,
    group_size: int = 64,
) -> Any:
    """Construct, quantize, strictly load, and evaluate the Hy3 NextN head."""

    import mlx.core as mx
    import mlx.nn as nn

    from .models.hy3_mlx import Hy3MTP

    weights = load_hy3_mtp_weights(
        artifact_dir,
        args,
        expected_revision=expected_revision,
    )
    mtp = Hy3MTP(args, num_mtp_layers=1)

    def class_predicate(path: str, module: Any) -> bool | dict[str, Any]:
        return _quantization_spec_for(path, module, weights, group_size=group_size)

    nn.quantize(
        mtp,
        group_size=group_size,
        bits=4,
        mode="affine",
        class_predicate=class_predicate,
    )
    mtp.eval()
    try:
        mtp.load_weights(list(weights.items()), strict=True)
    except Exception as exc:
        raise Hy3MTPLoadError(f"Hy3 MTP weight validation failed: {exc}") from exc
    mx.eval(mtp.parameters())
    logger.info(
        "[Hy3 MTP] loaded %d tensors from %s",
        len(weights),
        Path(artifact_dir).expanduser(),
    )
    return mtp
