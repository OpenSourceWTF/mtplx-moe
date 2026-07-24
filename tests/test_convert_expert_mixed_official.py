"""Fixture tests for the mixed-official expert bank converter (issue #51, M1).

Everything runs against synthetic bf16 safetensors shards written into
``tmp_path`` -- no GPU, no real bank, no network.  The laws exercised here
are the M1 contract: per-tier bitwise round trips against the reference
codecs (``decode_t158`` / ``mx.dequantize``), the 16384-alignment layout
laws, resume producing byte-identical output, and every fail-closed path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from mtplx.expert_mixed_official import (
    MIXED_ALIGNMENT,
    MIXED_GROUP_SIZE,
    MIXED_MANIFEST_FORMAT,
    MixedManifestError,
    PROJECTIONS,
    TIERS,
    build_layer_tier_map,
    decode_projection,
    encode_projection,
    manifest_digest,
    plan_record_segments,
    validate_mixed_manifest,
)
from mtplx.expert_shadow import decode_t158, encode_t158
from scripts.convert_expert_mixed_official import (
    ConvertError,
    _Interrupt,
    convert_mixed_official,
)

# Small dims keep most tests fast; their segments are NOT 16384-aligned, so
# converter tests over them pass allow_unaligned_segments=True explicitly.
SMALL_DIMS = {"gate_proj": (8, 128), "up_proj": (8, 128), "down_proj": (128, 64)}
# Aligned dims satisfy the hard law (every leaf length a 16384 multiple):
# rows * (cols/64) is a multiple of 16384 for both orientations.
ALIGNED_DIMS = {
    "gate_proj": (256, 4096),
    "up_proj": (256, 4096),
    "down_proj": (4096, 256),
}


# --------------------------------------------------------------------------
# fixture builders
# --------------------------------------------------------------------------
def _to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bit pattern (truncation; values are valid bf16)."""

    return (arr.astype(np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        raw = _to_bf16_bits(arr).tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(head)))
        handle.write(head)
        for blob in blobs:
            handle.write(blob)


def _expert_weights(layer: int, expert: int, dims: dict) -> dict[str, np.ndarray]:
    out = {}
    for projection in PROJECTIONS:
        rows, cols = dims[projection]
        rng = np.random.default_rng(seed=layer * 1000 + expert * 10 + len(projection))
        out[projection] = rng.standard_normal((rows, cols)).astype(np.float32)
    return out


def _build_fixture(
    root: Path,
    *,
    layers: dict[int, str],  # layer -> "IQ1_M" | "IQ2_XXS" gate/up tier
    experts: int,
    dims: dict,
) -> dict[str, Path]:
    """Write bf16 shard + index, coverage manifest and recipe map."""

    bf16_root = root / "bf16"
    bf16_root.mkdir(parents=True)
    tensors: dict[str, np.ndarray] = {}
    weight_map: dict[str, str] = {}
    shard_name = "model-00001-of-00001.safetensors"
    records = []
    for layer in sorted(layers):
        for expert in range(experts):
            for projection, arr in _expert_weights(layer, expert, dims).items():
                name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                tensors[name] = arr
                weight_map[name] = shard_name
            records.append({"layer": layer, "expert": expert})
    _write_safetensors(bf16_root / shard_name, tensors)
    (bf16_root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )

    coverage_path = root / "coverage-manifest.json"
    coverage_path.write_text(
        json.dumps(
            {
                "format": MIXED_MANIFEST_FORMAT,
                "source_repo": "fixture/hy3",
                "source_revision": "deadbeef",
                "manifest_sha256": "f" * 64,
                "records": records,
            }
        )
    )

    recipe_path = root / "recipe-map.json"
    recipe_path.write_text(
        json.dumps(
            {
                "repo": "AngelSlim/Hy3-GGUF",
                "anchor_file": "Hy3-IQ1_M.gguf",
                "per_layer_routed": {
                    str(layer): {
                        "gate": tier,
                        "up": tier,
                        "down": "IQ3_XXS",
                    }
                    for layer, tier in layers.items()
                },
            }
        )
    )
    return {
        "bf16_root": bf16_root,
        "coverage": coverage_path,
        "recipe": recipe_path,
    }


def _convert(fixture: dict[str, Path], output_root: Path, **kwargs):
    kwargs.setdefault("allow_unaligned_segments", True)
    kwargs.setdefault("verify_sample", 100)  # >= record count: verify everything
    return convert_mixed_official(
        source_manifest=fixture["coverage"],
        recipe_map=fixture["recipe"],
        bf16_root=fixture["bf16_root"],
        output_root=output_root,
        **kwargs,
    )


@pytest.fixture(scope="module")
def small_fixture(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("mixofficial-small")
    return _build_fixture(
        root, layers={1: "IQ1_M", 2: "IQ2_XXS"}, experts=2, dims=SMALL_DIMS
    )


@pytest.fixture(scope="module")
def converted(small_fixture, tmp_path_factory):
    """One shared conversion of the small fixture (read-only for tests)."""

    output_root = tmp_path_factory.mktemp("mixofficial-out")
    manifest = _convert(small_fixture, output_root)
    return {"manifest": manifest, "output_root": output_root}


# --------------------------------------------------------------------------
# per-tier bitwise round trips against the reference codecs
# --------------------------------------------------------------------------
def test_t158_tier_matches_shadow_codec_bitwise() -> None:
    w = np.random.default_rng(7).standard_normal((8, 128)).astype(np.float32)
    leaves = {p.leaf: p for p in encode_projection("t158", w)}
    packed_ref, scales_ref = encode_t158(w)
    assert leaves["packed"].data == packed_ref.tobytes()
    assert leaves["scales"].data == scales_ref.tobytes()
    assert leaves["packed"].shape == packed_ref.shape
    assert leaves["scales"].shape == scales_ref.shape

    decoded = decode_projection(
        "t158",
        {
            "packed": np.frombuffer(leaves["packed"].data, dtype=np.uint8).reshape(
                leaves["packed"].shape
            ),
            "scales": np.frombuffer(leaves["scales"].data, dtype=np.uint16).reshape(
                leaves["scales"].shape
            ),
        },
    )
    reference = decode_t158(packed_ref, scales_ref, 128)
    assert np.array_equal(decoded.view(np.uint32), reference.view(np.uint32))


@pytest.mark.parametrize("tier", ["affine2", "affine3"])
def test_affine_tiers_match_mx_quantize_bitwise(tier: str) -> None:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    spec = TIERS[tier]
    w = np.random.default_rng(11).standard_normal((8, 128)).astype(np.float32)
    leaves = {p.leaf: p for p in encode_projection(tier, w)}

    wq, scales, biases = mx.quantize(
        mx.array(w).astype(mx.bfloat16),
        bits=spec.bits,
        group_size=MIXED_GROUP_SIZE,
        mode="affine",
    )
    assert leaves["weight"].data == np.array(wq).tobytes()
    assert leaves["scales"].data == np.array(scales.view(mx.uint16)).tobytes()
    assert leaves["biases"].data == np.array(biases.view(mx.uint16)).tobytes()

    stored = decode_projection(
        tier,
        {
            leaf: np.frombuffer(
                payload.data,
                dtype={"U32": np.uint32, "BF16": np.uint16}[payload.dtype],
            ).reshape(payload.shape)
            for leaf, payload in leaves.items()
        },
    )
    reference = np.array(
        mx.dequantize(
            wq, scales, biases, bits=spec.bits, group_size=MIXED_GROUP_SIZE, mode="affine"
        ).astype(mx.float32)
    )
    assert np.array_equal(stored.view(np.uint32), reference.view(np.uint32))


# --------------------------------------------------------------------------
# tier map + layout laws
# --------------------------------------------------------------------------
def _recipe(entries: dict[int, dict[str, str]]) -> dict:
    return {"per_layer_routed": {str(k): v for k, v in entries.items()}}


def test_build_layer_tier_map_happy_path() -> None:
    recipe = _recipe(
        {
            1: {"gate": "IQ1_M", "up": "IQ1_M", "down": "IQ3_XXS"},
            2: {"gate": "IQ2_XXS", "up": "IQ2_XXS", "down": "IQ3_XXS"},
        }
    )
    tier_map = build_layer_tier_map(recipe, [1, 2])
    assert tier_map == {
        1: {"gate_up": "t158", "down": "affine3"},
        2: {"gate_up": "affine2", "down": "affine3"},
    }


@pytest.mark.parametrize(
    "entries, moe_layers, match",
    [
        ({1: {"gate": "IQ1_M", "up": "IQ1_M", "down": "IQ3_XXS"}}, [1, 2], "no tier assignment"),
        ({1: {"gate": "IQ1_M", "up": "IQ2_XXS", "down": "IQ3_XXS"}}, [1], "share a tier"),
        ({1: {"gate": "Q4_K", "up": "Q4_K", "down": "IQ3_XXS"}}, [1], "no mimic"),
        ({1: {"gate": "IQ3_XXS", "up": "IQ3_XXS", "down": "IQ3_XXS"}}, [1], "gate/up"),
        ({1: {"gate": "IQ1_M", "up": "IQ1_M", "down": "IQ1_M"}}, [1], "down"),
        ({1: {"gate": "IQ1_M", "up": "IQ1_M"}}, [1], "missing a gate/up/down"),
    ],
)
def test_build_layer_tier_map_fails_closed(entries, moe_layers, match) -> None:
    with pytest.raises(MixedManifestError, match=match):
        build_layer_tier_map(_recipe(entries), moe_layers)


def test_plan_record_segments_is_contiguous_and_ordered() -> None:
    tiers = {"gate_up": "t158", "down": "affine3"}
    planned = plan_record_segments(tiers, SMALL_DIMS)
    assert [seg.component for seg in planned] == [
        "gate_proj.packed",
        "gate_proj.scales",
        "up_proj.packed",
        "up_proj.scales",
        "down_proj.weight",
        "down_proj.scales",
        "down_proj.biases",
    ]
    cursor = 0
    for seg in planned:
        assert seg.offset == cursor
        cursor += seg.length
    gate = planned[0]
    assert gate.quant_tier == "t158"
    assert gate.shape == (8, (128 // 64) * 13)


# --------------------------------------------------------------------------
# converter over the small fixture
# --------------------------------------------------------------------------
def test_convert_emits_a_valid_bank(converted, small_fixture) -> None:
    manifest = converted["manifest"]
    output_root = converted["output_root"]
    validate_mixed_manifest(manifest, alignment=MIXED_ALIGNMENT)

    assert manifest["artifact"]["record_count"] == 4
    quant = manifest["quantization"]
    assert quant["mode"] == "mixed-official-v1"
    assert quant["layer_tier_map"] == {
        "1": {"gate_up": "t158", "down": "affine3"},
        "2": {"gate_up": "affine2", "down": "affine3"},
    }
    assert quant["source_recipe"]["repo"] == "AngelSlim/Hy3-GGUF"
    recipe_bytes = small_fixture["recipe"].read_bytes()
    assert quant["source_recipe"]["map_sha256"] == hashlib.sha256(recipe_bytes).hexdigest()
    assert quant["source_recipe"]["gguf_blk_to_manifest_layer_offset"] == 0

    bin_path = output_root / manifest["sidecar"]["file"]
    payload = bin_path.read_bytes()
    assert len(payload) == manifest["sidecar"]["size"]
    assert hashlib.sha256(payload).hexdigest() == manifest["sidecar"]["sha256"]
    for record in manifest["records"]:
        assert record["sidecar_offset"] % MIXED_ALIGNMENT == 0
        blob = payload[
            record["sidecar_offset"] : record["sidecar_offset"] + record["sidecar_length"]
        ]
        assert hashlib.sha256(blob).hexdigest() == record["sha256"]
        for seg in record["segments"]:
            assert seg["quant_tier"] == (
                manifest["quantization"]["layer_tier_map"][str(record["layer"])][
                    "gate_up" if not seg["component"].startswith("down_proj") else "down"
                ]
            )
    # No leftover journal/partial after a successful publish.
    assert not (output_root / "expert-manifest.json.journal").exists()
    assert not list(output_root.glob("*.partial"))


def test_stored_record_decodes_bitwise_like_a_fresh_encode(converted) -> None:
    manifest = converted["manifest"]
    output_root = converted["output_root"]
    payload = (output_root / manifest["sidecar"]["file"]).read_bytes()
    record = manifest["records"][0]  # layer 1 (t158 gate/up), expert 0
    base = record["sidecar_offset"]
    leaves: dict[str, np.ndarray] = {}
    for seg in record["segments"]:
        raw = payload[seg["offset"] : seg["offset"] + seg["length"]]
        dtype = {"U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "BF16": np.uint16}[
            seg["dtype"]
        ]
        leaves[seg["component"]] = np.frombuffer(raw, dtype=dtype).reshape(seg["shape"])
        assert seg["offset"] >= base

    source = _expert_weights(1, 0, SMALL_DIMS)
    tier_map = {"gate_proj": "t158", "up_proj": "t158", "down_proj": "affine3"}
    for projection, tier in tier_map.items():
        # The shard stores truncated-bf16 values; encode what the converter saw.
        seen = (
            (_to_bf16_bits(source[projection]).astype(np.uint32) << np.uint32(16))
            .view(np.float32)
            .reshape(source[projection].shape)
        )
        fresh = {p.leaf: p for p in encode_projection(tier, seen)}
        for leaf, payload_obj in fresh.items():
            assert leaves[f"{projection}.{leaf}"].tobytes() == payload_obj.data
        stored = decode_projection(
            tier,
            {
                leaf: leaves[f"{projection}.{leaf}"]
                for leaf, _ in TIERS[tier].leaves
            },
        )
        reference = decode_projection(
            tier,
            {
                leaf: np.frombuffer(
                    p.data,
                    dtype={"U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "BF16": np.uint16}[p.dtype],
                ).reshape(p.shape)
                for leaf, p in fresh.items()
            },
        )
        assert np.array_equal(stored.view(np.uint32), reference.view(np.uint32))


def test_aligned_dims_satisfy_the_hard_law_without_a_waiver(tmp_path) -> None:
    fixture = _build_fixture(
        tmp_path / "fx", layers={1: "IQ1_M"}, experts=1, dims=ALIGNED_DIMS
    )
    manifest = _convert(
        fixture,
        tmp_path / "out",
        allow_unaligned_segments=False,
        verify_sample=1,
    )
    for record in manifest["records"]:
        for seg in record["segments"]:
            assert seg["offset"] % MIXED_ALIGNMENT == 0
            assert seg["length"] % MIXED_ALIGNMENT == 0
    validate_mixed_manifest(
        manifest, alignment=MIXED_ALIGNMENT, require_segment_alignment=MIXED_ALIGNMENT
    )


# --------------------------------------------------------------------------
# fail-closed converter paths
# --------------------------------------------------------------------------
def test_unaligned_segments_refuse_to_publish(small_fixture, tmp_path) -> None:
    output_root = tmp_path / "out"
    with pytest.raises(ConvertError, match="alignment law"):
        _convert(small_fixture, output_root, allow_unaligned_segments=False)
    assert not (output_root / "expert-manifest.json").exists()
    assert not (output_root / "experts-mixofficial.bin").exists()


def test_recipe_missing_a_routed_layer_refuses(tmp_path) -> None:
    fixture = _build_fixture(
        tmp_path / "fx", layers={1: "IQ1_M", 2: "IQ2_XXS"}, experts=1, dims=SMALL_DIMS
    )
    recipe = json.loads(fixture["recipe"].read_text())
    del recipe["per_layer_routed"]["2"]
    fixture["recipe"].write_text(json.dumps(recipe))
    with pytest.raises(MixedManifestError, match="no tier assignment"):
        _convert(fixture, tmp_path / "out")


def test_empty_selection_refuses(small_fixture, tmp_path) -> None:
    with pytest.raises(ConvertError, match="selection is empty"):
        _convert(small_fixture, tmp_path / "out", limit_layers={99})


def test_existing_manifest_refuses_without_overwrite(converted, small_fixture) -> None:
    with pytest.raises(ConvertError, match="already exists"):
        _convert(small_fixture, converted["output_root"])


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------
def test_resume_is_byte_identical_to_a_single_shot_run(small_fixture, tmp_path) -> None:
    oneshot_root = tmp_path / "oneshot"
    resumed_root = tmp_path / "resumed"
    oneshot = _convert(small_fixture, oneshot_root)

    with pytest.raises(_Interrupt):
        _convert(small_fixture, resumed_root, fault_after=2)
    assert (resumed_root / "expert-manifest.json.journal").exists()
    assert not (resumed_root / "expert-manifest.json").exists()

    resumed = _convert(small_fixture, resumed_root, resume=True)
    assert resumed == oneshot
    assert resumed["manifest_sha256"] == manifest_digest(oneshot)
    one_bin = (oneshot_root / oneshot["sidecar"]["file"]).read_bytes()
    res_bin = (resumed_root / resumed["sidecar"]["file"]).read_bytes()
    assert one_bin == res_bin
    assert not (resumed_root / "expert-manifest.json.journal").exists()


def test_resume_refuses_a_different_selection(small_fixture, tmp_path) -> None:
    root = tmp_path / "out"
    with pytest.raises(_Interrupt):
        _convert(small_fixture, root, fault_after=1)
    with pytest.raises(ConvertError, match="does not match this selection"):
        _convert(small_fixture, root, resume=True, limit_experts={0})


def test_stale_partial_refuses_without_resume_or_overwrite(
    small_fixture, tmp_path
) -> None:
    root = tmp_path / "out"
    with pytest.raises(_Interrupt):
        _convert(small_fixture, root, fault_after=1)
    with pytest.raises(ConvertError, match="stale partial"):
        _convert(small_fixture, root)
    # --overwrite starts over cleanly.
    manifest = _convert(small_fixture, root, overwrite=True)
    validate_mixed_manifest(manifest, alignment=MIXED_ALIGNMENT)


# --------------------------------------------------------------------------
# manifest validator fail-closed mutations
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda m: m.__setitem__("format", "bogus"), "format"),
        (lambda m: m["quantization"].__setitem__("mode", "uniform"), "mode"),
        (lambda m: m["quantization"].__setitem__("group_size", 32), "group_size"),
        (lambda m: m["quantization"]["tiers"].pop("t158"), "tiers"),
        (
            lambda m: m["quantization"]["source_recipe"].__setitem__("map_sha256", "zz"),
            "map_sha256",
        ),
        (
            lambda m: m["quantization"]["layer_tier_map"].pop("2"),
            "no layer_tier_map entry",
        ),
        (
            lambda m: m["quantization"]["layer_tier_map"]["1"].__setitem__(
                "down", "affine2"
            ),
            "out of contract",
        ),
        (
            lambda m: m["records"][1].__setitem__(
                "sidecar_offset", m["records"][1]["sidecar_offset"] + 1
            ),
            "not aligned",
        ),
        (
            lambda m: m["records"][0].__setitem__("sha256", "0" * 64),
            "does not match the payload",  # record sha isn't checked structurally; digest breaks
        ),
        (
            lambda m: m["records"][0]["segments"][0].__setitem__("quant_tier", "affine2"),
            "quant_tier",
        ),
        (
            lambda m: m["records"][0]["segments"][1].__setitem__(
                "offset", m["records"][0]["segments"][1]["offset"] + 8
            ),
            "not contiguous",
        ),
        (
            lambda m: m["records"].insert(0, copy.deepcopy(m["records"][1])),
            "sorted and unique",
        ),
        (lambda m: m["sidecar"].__setitem__("size", 3), "smaller than the last"),
        (
            lambda m: m["artifact"].__setitem__("resident_tensor_bytes", 5),
            "no resident tensors",
        ),
        (lambda m: m.__setitem__("model_key", "tampered"), "manifest_sha256"),
    ],
)
def test_validator_fails_closed(converted, mutate, match) -> None:
    manifest = copy.deepcopy(converted["manifest"])
    mutate(manifest)
    with pytest.raises(MixedManifestError, match=match):
        validate_mixed_manifest(manifest, alignment=MIXED_ALIGNMENT)
