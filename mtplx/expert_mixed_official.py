"""Mixed-precision "official recipe" expert banks (issue #51, M1).

Tencent ships Hy3 through AngelSlim as a *mixed* GGUF: every routed MoE
layer's ``down`` projection is IQ3_XXS, while the ``gate``/``up`` pair is
either IQ1_M or IQ2_XXS on a per-layer schedule (see
``research/angelslim_mimic/official-recipe-map.json``).  This module mimics
that per-layer allocation with our *servable* encodings so the runtime can
open the bank through the machinery it already has:

============ =============== ============ ================ ==========
official     mimic tier      encoder      components       bpw (g64)
============ =============== ============ ================ ==========
IQ1_M   1.75 ``t158``        expert_shadow packed(U8)+scale 1.875
IQ2_XXS 2.06 ``affine2``     mx.quantize   w(U32)+s+b(BF16) 2.5
IQ3_XXS 3.06 ``affine3``     mx.quantize   w(U32)+s+b(BF16) 3.5
============ =============== ============ ================ ==========

The bank is one page-aligned sidecar bin plus a
``mtplx-expert-manifest-v1``-compatible manifest whose ``quantization``
block carries ``mode="mixed-official-v1"``, the per-layer ``layer_tier_map``
and a ``source_recipe`` provenance stamp.  Every record segment additionally
carries a ``quant_tier`` so a reader never has to re-derive the schedule.

This module is the single source of truth for the tier definitions, the
per-record segment layout and the manifest validation laws.  Both the
converter (``scripts/convert_expert_mixed_official.py``) and the future M2
runtime import it; nothing here touches the GPU or a whole safetensors
shard.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mtplx.expert_shadow import (
    SHADOW_GROUP,
    decode_t158,
    encode_t158,
)

MIXED_MANIFEST_FORMAT = "mtplx-expert-manifest-v1"
MIXED_MODE = "mixed-official-v1"
MIXED_ALIGNMENT = 16 * 1024
MIXED_GROUP_SIZE = SHADOW_GROUP  # 64
DEFAULT_MODEL_KEY = "hy3-expert-mixofficial"
DEFAULT_SIDECAR_NAME = "experts-mixofficial.bin"

# Component ordering inside every record (component-major, the island/q2
# convention). ``down`` is always the affine tier; ``gate``/``up`` share a
# tier per layer.
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
GATE_UP_PROJECTIONS = ("gate_proj", "up_proj")
DOWN_PROJECTION = "down_proj"

# GGUF ggml_type in the official recipe -> our servable mimic tier.
GGUF_TIER_TO_MIXED = {
    "IQ1_M": "t158",
    "IQ2_XXS": "affine2",
    "IQ3_XXS": "affine3",
}

# The only tiers a gate/up pair (resp. a down projection) may resolve to.
GATE_UP_TIERS = ("t158", "affine2")
DOWN_TIERS = ("affine3",)

_DTYPE_BYTES = {"U8": 1, "U16": 2, "BF16": 2, "U32": 4}


@dataclass(frozen=True)
class TierSpec:
    """Static description of one mimic tier."""

    name: str
    encoder: str  # "t158" | "affine"
    bits: int  # nominal bits (t158 carries the shadow nominal of 2)
    bpw: float  # bytes-per-weight * 8, at group size 64
    # ordered (leaf, dtype) the encoder emits per projection.
    leaves: tuple[tuple[str, str], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder,
            "bits": self.bits,
            "group_size": MIXED_GROUP_SIZE,
            "bpw": self.bpw,
            "components": [
                {"leaf": leaf, "dtype": dtype} for leaf, dtype in self.leaves
            ],
        }


TIERS: dict[str, TierSpec] = {
    "t158": TierSpec(
        name="t158",
        encoder="t158",
        bits=2,  # shadow nominal (byte math lives in the codec: 15 B / 64 w)
        bpw=1.875,
        leaves=(("packed", "U8"), ("scales", "U16")),
    ),
    "affine2": TierSpec(
        name="affine2",
        encoder="affine",
        bits=2,
        bpw=2.5,
        leaves=(("weight", "U32"), ("scales", "BF16"), ("biases", "BF16")),
    ),
    "affine3": TierSpec(
        name="affine3",
        encoder="affine",
        bits=3,
        bpw=3.5,
        leaves=(("weight", "U32"), ("scales", "BF16"), ("biases", "BF16")),
    ),
}


class MixedManifestError(ValueError):
    """Raised when a mixed-official bank or its manifest fails validation."""


# --------------------------------------------------------------------------
# lazy mlx (CPU-only; the converter/tests own the process default device)
# --------------------------------------------------------------------------
def _mx():
    import mlx.core as mx

    return mx


# --------------------------------------------------------------------------
# tier / recipe resolution
# --------------------------------------------------------------------------
def gguf_tier_to_mixed(gguf_type: str) -> str:
    try:
        return GGUF_TIER_TO_MIXED[gguf_type]
    except KeyError:
        raise MixedManifestError(
            f"official recipe uses ggml type {gguf_type!r} that has no mimic "
            f"tier (known: {sorted(GGUF_TIER_TO_MIXED)})"
        ) from None


def build_layer_tier_map(
    recipe_map: Mapping[str, Any],
    moe_layers: Iterable[int],
) -> dict[int, dict[str, str]]:
    """Translate the official ``per_layer_routed`` block to our tier map.

    ``moe_layers`` is the authoritative routed-layer set (our expert
    manifest).  The result is ``{layer: {"gate_up": tier, "down": tier}}``.
    Fails closed if any required layer is missing from the recipe, if a
    layer's gate/up disagree, or if a tier is unknown / out of contract.
    """

    per_layer = recipe_map.get("per_layer_routed")
    if not isinstance(per_layer, Mapping):
        raise MixedManifestError("recipe map lacks a per_layer_routed object")
    result: dict[int, dict[str, str]] = {}
    missing: list[int] = []
    for layer in sorted({int(value) for value in moe_layers}):
        entry = per_layer.get(str(layer))
        if not isinstance(entry, Mapping):
            missing.append(layer)
            continue
        gate = entry.get("gate")
        up = entry.get("up")
        down = entry.get("down")
        if gate is None or up is None or down is None:
            raise MixedManifestError(
                f"recipe layer {layer} is missing a gate/up/down assignment"
            )
        if gate != up:
            raise MixedManifestError(
                f"recipe layer {layer} has gate {gate!r} != up {up!r}; the "
                "mimic requires gate and up to share a tier"
            )
        gate_up_tier = gguf_tier_to_mixed(str(gate))
        down_tier = gguf_tier_to_mixed(str(down))
        if gate_up_tier not in GATE_UP_TIERS:
            raise MixedManifestError(
                f"recipe layer {layer} gate/up maps to {gate_up_tier!r}, not "
                f"one of {GATE_UP_TIERS}"
            )
        if down_tier not in DOWN_TIERS:
            raise MixedManifestError(
                f"recipe layer {layer} down maps to {down_tier!r}, not one of "
                f"{DOWN_TIERS}"
            )
        result[layer] = {"gate_up": gate_up_tier, "down": down_tier}
    if missing:
        raise MixedManifestError(
            "official recipe has no tier assignment for routed layer(s) "
            f"{missing}; refusing to convert an under-specified bank"
        )
    return result


def tier_for_projection(layer_tiers: Mapping[str, str], projection: str) -> str:
    if projection in GATE_UP_PROJECTIONS:
        return layer_tiers["gate_up"]
    if projection == DOWN_PROJECTION:
        return layer_tiers["down"]
    raise MixedManifestError(f"unknown projection {projection!r}")


# --------------------------------------------------------------------------
# per-tier encode / decode
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LeafPayload:
    leaf: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes


def encode_projection(
    tier: str, weights: np.ndarray, *, group_size: int = MIXED_GROUP_SIZE
) -> list[LeafPayload]:
    """Encode one projection ``(out, in)`` fp32 -> ordered leaf payloads.

    ``t158`` uses the shadow codec (numpy); ``affine2/affine3`` use
    ``mx.quantize`` exactly, storing the u32 weight and the bf16 scale/bias
    rows as their raw little-endian bytes (a bit-exact copy of what the
    runtime dequantizes).
    """

    spec = TIERS[tier]
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    if weights.ndim != 2:
        raise MixedManifestError(f"projection weights must be 2-D, got {weights.shape}")
    if spec.encoder == "t158":
        packed, scales = encode_t158(weights)
        return [
            LeafPayload("packed", "U8", tuple(packed.shape), packed.tobytes()),
            LeafPayload("scales", "U16", tuple(scales.shape), scales.tobytes()),
        ]
    if spec.encoder == "affine":
        mx = _mx()
        source = mx.array(weights).astype(mx.bfloat16)
        wq, scales, biases = mx.quantize(
            source, bits=spec.bits, group_size=group_size, mode="affine"
        )
        wq_np = np.array(wq, copy=False)
        scales_u16 = np.array(scales.view(mx.uint16), copy=False)
        biases_u16 = np.array(biases.view(mx.uint16), copy=False)
        if wq_np.dtype != np.uint32:
            raise MixedManifestError("mx.quantize weight must be u32")
        return [
            LeafPayload("weight", "U32", tuple(wq_np.shape), wq_np.tobytes()),
            LeafPayload("scales", "BF16", tuple(scales_u16.shape), scales_u16.tobytes()),
            LeafPayload("biases", "BF16", tuple(biases_u16.shape), biases_u16.tobytes()),
        ]
    raise MixedManifestError(f"unknown tier encoder {spec.encoder!r}")


def decode_projection(
    tier: str,
    leaves: Mapping[str, np.ndarray],
    *,
    group_size: int = MIXED_GROUP_SIZE,
) -> np.ndarray:
    """Inverse of :func:`encode_projection` -> fp32 ``(out, in)``.

    ``t158`` decodes with ``decode_t158``; affine decodes with the exact
    ``mx.dequantize`` the runtime executes.  Used by ``--verify-sample`` and
    the tests as the independent reference dequant.
    """

    spec = TIERS[tier]
    if spec.encoder == "t158":
        packed = np.ascontiguousarray(leaves["packed"], dtype=np.uint8)
        scales = np.ascontiguousarray(leaves["scales"], dtype=np.uint16)
        cols = int(scales.shape[1]) * SHADOW_GROUP
        return np.ascontiguousarray(decode_t158(packed, scales, cols), dtype=np.float32)
    if spec.encoder == "affine":
        mx = _mx()
        weight = mx.array(np.ascontiguousarray(leaves["weight"], dtype=np.uint32))
        scales = mx.array(
            np.ascontiguousarray(leaves["scales"], dtype=np.uint16)
        ).view(mx.bfloat16)
        biases = mx.array(
            np.ascontiguousarray(leaves["biases"], dtype=np.uint16)
        ).view(mx.bfloat16)
        dq = mx.dequantize(
            weight, scales, biases, bits=spec.bits, group_size=group_size, mode="affine"
        )
        return np.array(dq.astype(mx.float32), copy=False)
    raise MixedManifestError(f"unknown tier encoder {spec.encoder!r}")


# --------------------------------------------------------------------------
# pure-arithmetic segment layout (prediction, resume, law proofs)
# --------------------------------------------------------------------------
def leaf_shape(tier: str, leaf: str, rows: int, cols: int) -> tuple[int, int]:
    """The stored shape of one leaf for a projection ``(rows, cols)``."""

    groups = cols // MIXED_GROUP_SIZE
    if cols % MIXED_GROUP_SIZE:
        raise MixedManifestError(
            f"projection cols {cols} is not grouped in {MIXED_GROUP_SIZE}s"
        )
    spec = TIERS[tier]
    if spec.encoder == "t158":
        if leaf == "packed":
            return (rows, groups * 13)
        if leaf == "scales":
            return (rows, groups)
    else:  # affine
        if leaf == "weight":
            return (rows, cols * spec.bits // 32)
        if leaf in ("scales", "biases"):
            return (rows, groups)
    raise MixedManifestError(f"tier {tier!r} has no leaf {leaf!r}")


def _leaf_bytes(dtype: str, shape: Sequence[int]) -> int:
    length = _DTYPE_BYTES[dtype]
    for dimension in shape:
        length *= int(dimension)
    return length


@dataclass(frozen=True)
class PlannedSegment:
    component: str
    quant_tier: str
    dtype: str
    shape: tuple[int, ...]
    offset: int  # relative to the record start
    length: int


def plan_record_segments(
    layer_tiers: Mapping[str, str],
    dims: Mapping[str, tuple[int, int]],
) -> tuple[PlannedSegment, ...]:
    """Deterministic in-record segment table for one expert.

    ``dims`` is ``{projection: (rows, cols)}``.  Offsets are relative to the
    record start (contiguous, no inter-segment padding).  This is the single
    definition the converter, the resume scan and the manifest all agree on.
    """

    segments: list[PlannedSegment] = []
    cursor = 0
    for projection in PROJECTIONS:
        rows, cols = dims[projection]
        tier = tier_for_projection(layer_tiers, projection)
        for leaf, dtype in TIERS[tier].leaves:
            shape = leaf_shape(tier, leaf, rows, cols)
            length = _leaf_bytes(dtype, shape)
            segments.append(
                PlannedSegment(
                    component=f"{projection}.{leaf}",
                    quant_tier=tier,
                    dtype=dtype,
                    shape=shape,
                    offset=cursor,
                    length=length,
                )
            )
            cursor += length
    return tuple(segments)


def record_logical_bytes(
    layer_tiers: Mapping[str, str], dims: Mapping[str, tuple[int, int]]
) -> int:
    return sum(seg.length for seg in plan_record_segments(layer_tiers, dims))


def align_up(value: int, alignment: int = MIXED_ALIGNMENT) -> int:
    return (value + alignment - 1) & -alignment


# --------------------------------------------------------------------------
# manifest emission + validation
# --------------------------------------------------------------------------
def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def source_recipe_block(recipe_map: Mapping[str, Any], map_sha256: str) -> dict[str, Any]:
    return {
        "repo": str(recipe_map.get("repo", "AngelSlim/Hy3-GGUF")),
        "file": str(recipe_map.get("anchor_file", "Hy3-IQ1_M.gguf")),
        "map_sha256": map_sha256,
    }


def quantization_block(
    layer_tier_map: Mapping[int, Mapping[str, str]],
    source_recipe: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "mode": MIXED_MODE,
        "group_size": MIXED_GROUP_SIZE,
        "tiers": {name: spec.to_json() for name, spec in TIERS.items()},
        "layer_tier_map": {
            str(layer): {"gate_up": tiers["gate_up"], "down": tiers["down"]}
            for layer, tiers in sorted(layer_tier_map.items())
        },
        "source_recipe": dict(source_recipe),
    }


_SHA256_HEX = 64


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX
        and all(c in "0123456789abcdef" for c in value)
    )


def validate_mixed_manifest(
    manifest: Mapping[str, Any],
    *,
    alignment: int = MIXED_ALIGNMENT,
    require_segment_alignment: int | None = None,
    verify_digest: bool = True,
) -> None:
    """Fail-closed structural validation of a mixed-official manifest.

    Enforced universally (holds for tiny fixtures and the real bank):
      * format / mode / group size are the mixed-official contract;
      * every tier in the tier table is a known ``TierSpec``;
      * ``layer_tier_map`` assigns a tier to *every* routed layer present
        in the records (fail closed on a gap);
      * records are sorted & unique by ``(layer, expert)`` with
        non-overlapping sidecar ranges;
      * each record: ``sidecar_offset`` is ``alignment``-aligned, segments
        are contiguous from ``sidecar_offset`` and exactly cover
        ``logical_bytes`` (== ``sidecar_length``), the per-record sha256 is
        present, and each segment's dtype/shape/length agree and its
        ``quant_tier`` matches the layer_tier_map for its projection.

    ``require_segment_alignment`` (the real bank passes 16384) additionally
    asserts every *segment* offset is aligned -- a property the hy3 tensor
    dims guarantee but tiny fixture dims do not.
    """

    if manifest.get("format") != MIXED_MANIFEST_FORMAT:
        raise MixedManifestError("manifest format is not mtplx-expert-manifest-v1")
    quant = manifest.get("quantization")
    if not isinstance(quant, Mapping):
        raise MixedManifestError("manifest lacks a quantization block")
    if quant.get("mode") != MIXED_MODE:
        raise MixedManifestError(f"quantization mode must be {MIXED_MODE!r}")
    if int(quant.get("group_size", -1)) != MIXED_GROUP_SIZE:
        raise MixedManifestError(f"quantization group_size must be {MIXED_GROUP_SIZE}")
    tiers = quant.get("tiers")
    if not isinstance(tiers, Mapping) or set(tiers) != set(TIERS):
        raise MixedManifestError(
            f"quantization.tiers must define exactly {sorted(TIERS)}"
        )
    for name, spec in TIERS.items():
        declared = tiers.get(name)
        if not isinstance(declared, Mapping) or int(declared.get("bits", -1)) != spec.bits:
            raise MixedManifestError(f"tier {name!r} must declare bits={spec.bits}")
    recipe = quant.get("source_recipe")
    if not isinstance(recipe, Mapping) or not _is_sha256(recipe.get("map_sha256", "")):
        raise MixedManifestError("source_recipe must carry a sha256 map_sha256")
    if not recipe.get("repo") or not recipe.get("file"):
        raise MixedManifestError("source_recipe must carry repo and file")

    layer_tier_map = quant.get("layer_tier_map")
    if not isinstance(layer_tier_map, Mapping):
        raise MixedManifestError("quantization lacks a layer_tier_map")
    tier_map: dict[int, dict[str, str]] = {}
    for key, entry in layer_tier_map.items():
        if not isinstance(entry, Mapping):
            raise MixedManifestError("layer_tier_map entries must be objects")
        gate_up = entry.get("gate_up")
        down = entry.get("down")
        if gate_up not in GATE_UP_TIERS or down not in DOWN_TIERS:
            raise MixedManifestError(
                f"layer {key} tier assignment {entry!r} is out of contract"
            )
        tier_map[int(key)] = {"gate_up": gate_up, "down": down}

    records = manifest.get("records")
    if not isinstance(records, list):
        raise MixedManifestError("manifest records must be an array")

    if require_segment_alignment is not None and (
        require_segment_alignment <= 0
        or require_segment_alignment & (require_segment_alignment - 1)
    ):
        raise MixedManifestError("require_segment_alignment must be a power of two")

    keys: list[tuple[int, int]] = []
    ranges: list[tuple[int, int]] = []
    routed_bytes = 0
    covered_layers: set[int] = set()
    for record in records:
        layer = int(record["layer"])
        expert = int(record["expert"])
        keys.append((layer, expert))
        covered_layers.add(layer)
        if layer not in tier_map:
            raise MixedManifestError(
                f"record layer {layer} has no layer_tier_map entry"
            )
        logical = int(record["logical_bytes"])
        sidecar_offset = record.get("sidecar_offset")
        sidecar_length = record.get("sidecar_length")
        if sidecar_offset is None or sidecar_length is None:
            raise MixedManifestError(
                f"record ({layer}, {expert}) requires sidecar_offset/length"
            )
        sidecar_offset = int(sidecar_offset)
        sidecar_length = int(sidecar_length)
        if sidecar_offset % alignment:
            raise MixedManifestError(
                f"record ({layer}, {expert}) sidecar_offset not aligned to {alignment}"
            )
        if sidecar_length != logical:
            raise MixedManifestError(
                f"record ({layer}, {expert}) sidecar_length != logical_bytes"
            )
        if not _is_sha256(record.get("sha256", "")):
            raise MixedManifestError(f"record ({layer}, {expert}) lacks a sha256")

        segments = record.get("segments")
        if not isinstance(segments, list) or not segments:
            raise MixedManifestError(f"record ({layer}, {expert}) has no segments")
        expected = plan_record_segments(
            tier_map[layer],
            {
                seg_proj: _proj_dims_from_segments(segments, seg_proj)
                for seg_proj in PROJECTIONS
            },
        )
        expected_components = [seg.component for seg in expected]
        got_components = [seg.get("component") for seg in segments]
        if got_components != expected_components:
            raise MixedManifestError(
                f"record ({layer}, {expert}) component order {got_components} "
                f"!= expected {expected_components}"
            )
        cursor = sidecar_offset
        total = 0
        for seg, plan in zip(segments, expected):
            dtype = seg.get("dtype")
            if dtype not in _DTYPE_BYTES:
                raise MixedManifestError(f"segment dtype {dtype!r} unsupported")
            shape = tuple(int(v) for v in seg.get("shape", ()))
            length = int(seg["length"])
            offset = int(seg["offset"])
            if length != _leaf_bytes(dtype, shape):
                raise MixedManifestError(
                    f"segment {seg.get('component')} length/shape/dtype disagree"
                )
            if seg.get("quant_tier") != plan.quant_tier:
                raise MixedManifestError(
                    f"segment {seg.get('component')} quant_tier "
                    f"{seg.get('quant_tier')!r} != {plan.quant_tier!r}"
                )
            if dtype != plan.dtype or shape != plan.shape or length != plan.length:
                raise MixedManifestError(
                    f"segment {seg.get('component')} does not match the planned layout"
                )
            if offset != cursor:
                raise MixedManifestError(
                    f"record ({layer}, {expert}) segments are not contiguous"
                )
            if require_segment_alignment is not None and offset % require_segment_alignment:
                raise MixedManifestError(
                    f"segment {seg.get('component')} offset {offset} not aligned to "
                    f"{require_segment_alignment}"
                )
            cursor += length
            total += length
        if total != logical:
            raise MixedManifestError(
                f"record ({layer}, {expert}) segments cover {total} != {logical}"
            )
        if cursor != sidecar_offset + logical:
            raise MixedManifestError(
                f"record ({layer}, {expert}) segments do not end at the record end"
            )
        ranges.append((sidecar_offset, sidecar_offset + logical))
        routed_bytes += logical

    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise MixedManifestError("records must be sorted and unique by (layer, expert)")
    missing = covered_layers - set(tier_map)
    if missing:
        raise MixedManifestError(f"layers {sorted(missing)} lack a tier assignment")
    if ranges != sorted(ranges) or any(
        a[1] > b[0] for a, b in zip(ranges, ranges[1:])
    ):
        raise MixedManifestError("sidecar records overlap or are unsorted")

    sidecar = manifest.get("sidecar")
    if not isinstance(sidecar, Mapping):
        raise MixedManifestError("manifest lacks sidecar metadata")
    if int(sidecar.get("alignment", -1)) != alignment:
        raise MixedManifestError(f"sidecar alignment must be {alignment}")
    if ranges and int(sidecar.get("size", -1)) < ranges[-1][1]:
        raise MixedManifestError("sidecar size is smaller than the last record end")
    if not _is_sha256(sidecar.get("sha256", "")):
        raise MixedManifestError("sidecar requires a sha256")

    artifact = manifest.get("artifact")
    if not isinstance(artifact, Mapping):
        raise MixedManifestError("manifest lacks an artifact block")
    if int(artifact.get("routed_expert_bytes", -1)) != routed_bytes:
        raise MixedManifestError("artifact.routed_expert_bytes != sum of records")
    if int(artifact.get("resident_tensor_bytes", -1)) != 0:
        raise MixedManifestError("mixed-official banks carry no resident tensors")
    if int(artifact.get("record_count", -1)) != len(records):
        raise MixedManifestError("artifact.record_count != number of records")

    if verify_digest and "manifest_sha256" in manifest:
        if manifest_digest(manifest) != manifest["manifest_sha256"]:
            raise MixedManifestError("manifest_sha256 does not match the payload")


def _proj_dims_from_segments(
    segments: Sequence[Mapping[str, Any]], projection: str
) -> tuple[int, int]:
    """Recover ``(rows, cols)`` for a projection from its stored leaves.

    ``rows`` is the leaf row count; ``cols`` (the projection's ``in`` axis)
    comes from the scales group count (cols = groups * 64), which every tier
    stores identically.
    """

    rows: int | None = None
    groups: int | None = None
    for seg in segments:
        component = seg.get("component", "")
        if not component.startswith(projection + "."):
            continue
        leaf = component.split(".", 1)[1]
        shape = tuple(int(v) for v in seg.get("shape", ()))
        if len(shape) != 2:
            raise MixedManifestError(f"segment {component} must be 2-D")
        if leaf == "scales":
            rows, groups = shape[0], shape[1]
    if rows is None or groups is None:
        raise MixedManifestError(f"projection {projection} is missing its scales leaf")
    return rows, groups * MIXED_GROUP_SIZE
