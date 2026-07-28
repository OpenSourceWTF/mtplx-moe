from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
import struct
from types import MappingProxyType

import numpy as np
import pytest

import mtplx.kimi_k3_gguf as kimi_gguf
from mtplx.kimi_k3_gguf import (
    GGML_TYPE_BF16,
    GGML_TYPE_F32,
    GGML_TYPE_Q2_K,
    KIMI_K3_SOURCE_REVISION,
    Q2_K_BYTES,
    GGUFTensor,
    GGUFFile,
    dequantize_q2_k,
    inspect_kimi_k3_source,
    read_gguf,
    tensor_nbytes,
)


def _scalar_q2_k(block: bytes) -> np.ndarray:
    scales = block[:16]
    qs = block[16:80]
    d = np.frombuffer(block[80:82], dtype="<f2")[0].astype(np.float32)
    dmin = np.frombuffer(block[82:84], dtype="<f2")[0].astype(np.float32)
    decoded = np.empty(256, dtype=np.float32)
    for group in range(16):
        half = group // 8
        lane = (group % 8) // 2
        byte_half = group % 2
        qbase = half * 32 + byte_half * 16
        shift = lane * 2
        scale = np.float32(scales[group] & 0x0F)
        minimum = np.float32(scales[group] >> 4)
        for value in range(16):
            q = np.float32((qs[qbase + value] >> shift) & 0x03)
            decoded[group * 16 + value] = d * scale * q - dmin * minimum
    return decoded


def test_dequantize_q2_k_matches_independent_scalar_golden() -> None:
    scales = bytes(((15 - index) << 4) | index for index in range(16))
    qs = bytes((index * 37 + 11) & 0xFF for index in range(64))
    block = scales + qs + struct.pack("<ee", 0.125, 0.03125)

    assert len(block) == Q2_K_BYTES
    decoded = dequantize_q2_k(block, value_count=256)

    assert decoded.shape == (256,)
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, _scalar_q2_k(block))


def test_dequantize_q2_k_rejects_partial_values_and_bytes() -> None:
    with pytest.raises(ValueError, match="multiple of 256"):
        dequantize_q2_k(bytes(Q2_K_BYTES), value_count=255)
    with pytest.raises(ValueError, match="byte length mismatch"):
        dequantize_q2_k(bytes(Q2_K_BYTES - 1), value_count=256)


@pytest.mark.parametrize("dims", [(1, 256), (128, 2)])
def test_tensor_nbytes_rejects_q2_k_when_first_dimension_is_not_blocked(
    dims: tuple[int, ...],
) -> None:
    tensor = GGUFTensor("bad-q2", dims, GGML_TYPE_Q2_K, 0)
    with pytest.raises(ValueError, match="first GGML dimension.*256"):
        tensor_nbytes(tensor)


def _gstr(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _u32_metadata(key: str, value: int) -> bytes:
    return _gstr(key) + struct.pack("<II", 4, value)


def _u16_metadata(key: str, value: int) -> bytes:
    return _gstr(key) + struct.pack("<IH", 2, value)


def _i32_metadata(key: str, value: int) -> bytes:
    return _gstr(key) + struct.pack("<Ii", 5, value)


def _u8_array_metadata(key: str, values: bytes) -> bytes:
    return _gstr(key) + struct.pack("<IIQ", 9, 0, len(values)) + values


def _gguf_bytes(
    tensors: list[tuple[str, tuple[int, ...], int, int]],
    *,
    alignment: int = 32,
    magic: bytes = b"GGUF",
    version: int = 3,
    tensor_count: int | None = None,
    metadata_count: int | None = None,
    metadata_entries: list[bytes] | None = None,
    payload_bytes: int | None = None,
) -> bytes:
    if metadata_entries is None:
        metadata_entries = [_u32_metadata("general.alignment", alignment)]
    metadata = b"".join(metadata_entries)
    header = bytearray(
        magic
        + struct.pack(
            "<IQQ",
            version,
            len(tensors) if tensor_count is None else tensor_count,
            len(metadata_entries) if metadata_count is None else metadata_count,
        )
        + metadata
    )
    for name, dims, ggml_type, offset in tensors:
        header += _gstr(name)
        header += struct.pack("<I", len(dims))
        header += struct.pack(f"<{len(dims)}Q", *dims)
        header += struct.pack("<IQ", ggml_type, offset)
    if alignment > 0:
        header += bytes((-len(header)) % alignment)
    if payload_bytes is None:
        payload_bytes = (
            max(
                (offset + tensor_nbytes(GGUFTensor(name, dims, ggml_type, offset)))
                for name, dims, ggml_type, offset in tensors
            )
            if tensors
            else 0
        )
    return bytes(header) + bytes(payload_bytes)


def test_read_gguf_v3_parses_typed_immutable_records(tmp_path: Path) -> None:
    path = tmp_path / "tiny-00001-of-00001.gguf"
    path.write_bytes(
        _gguf_bytes(
            [
                ("float.weight", (2, 3), GGML_TYPE_F32, 0),
                ("bf16.weight", (4,), GGML_TYPE_BF16, 32),
                ("quant.weight", (256,), GGML_TYPE_Q2_K, 64),
            ],
            payload_bytes=160,
        )
    )

    source = read_gguf(path)

    assert source.path == path
    assert source.version == 3
    assert source.alignment == 32
    assert [tensor.name for tensor in source.tensors] == [
        "float.weight",
        "bf16.weight",
        "quant.weight",
    ]
    assert source.tensor("float.weight").shape == (3, 2)
    assert tensor_nbytes(source.tensor("float.weight")) == 24
    assert tensor_nbytes(source.tensor("bf16.weight")) == 8
    assert tensor_nbytes(source.tensor("quant.weight")) == Q2_K_BYTES
    with pytest.raises(FrozenInstanceError):
        source.alignment = 64  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        source.tensors[0].offset = 32  # type: ignore[misc]
    with pytest.raises(TypeError):
        source.metadata["new"] = 1  # type: ignore[index]
    assert source.metadata_types["general.alignment"] == 4
    with pytest.raises(TypeError):
        source.metadata_types["general.alignment"] = 5  # type: ignore[index]


@pytest.mark.parametrize("mutation", ["modify", "replace"])
def test_verified_reopen_rejects_source_changed_after_inspection(
    tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / "identity.gguf"
    original = _gguf_bytes(
        [("float.weight", (4,), GGML_TYPE_F32, 0)],
        payload_bytes=32,
    )
    path.write_bytes(original)
    source = read_gguf(path)

    descriptor = kimi_gguf.open_verified_gguf(source)
    os.close(descriptor)
    if mutation == "modify":
        with path.open("r+b") as changed:
            changed.seek(-1, os.SEEK_END)
            changed.write(b"\x01")
            changed.flush()
            os.fsync(changed.fileno())
    else:
        path.unlink()
        path.write_bytes(original)

    with pytest.raises(ValueError, match="changed since inspection"):
        kimi_gguf.open_verified_gguf(source)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_gguf_bytes([], magic=b"NOPE"), "magic"),
        (_gguf_bytes([], version=2), "version"),
        (_gguf_bytes([], tensor_count=1_000_001), "tensor count"),
        (_gguf_bytes([], metadata_count=1_000_001), "metadata count"),
        (_gguf_bytes([], alignment=3), "power of two"),
        (
            _gguf_bytes([("unknown", (1,), 29, 0)], payload_bytes=32),
            "tensor type",
        ),
        (
            _gguf_bytes(
                [
                    ("same", (1,), GGML_TYPE_F32, 0),
                    ("same", (1,), GGML_TYPE_F32, 32),
                ],
                payload_bytes=64,
            ),
            "duplicate tensor",
        ),
        (
            _gguf_bytes(
                [("past-end", (16,), GGML_TYPE_F32, 32)],
                payload_bytes=64,
            )[:-1],
            "span",
        ),
        (
            _gguf_bytes(
                [
                    ("first", (16,), GGML_TYPE_F32, 0),
                    ("overlap", (16,), GGML_TYPE_F32, 32),
                ],
                payload_bytes=96,
            ),
            "overlap",
        ),
    ],
)
def test_read_gguf_rejects_malformed_headers(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    path = tmp_path / "bad.gguf"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=message):
        read_gguf(path)


def test_read_gguf_rejects_oversized_string_before_reading_it(tmp_path: Path) -> None:
    path = tmp_path / "oversized-name.gguf"
    path.write_bytes(
        b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + struct.pack("<Q", 16 * 1024 * 1024 + 1)
    )
    with pytest.raises(ValueError, match="string length"):
        read_gguf(path)


def test_read_gguf_accepts_measured_uint16_per_shard_split_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source-00007-of-00096.gguf"
    path.write_bytes(
        _gguf_bytes(
            [
                ("resident.f32", (4,), GGML_TYPE_F32, 0),
                ("resident.bf16", (4,), GGML_TYPE_BF16, 32),
            ],
            metadata_entries=[
                _u16_metadata("split.no", 6),
                _u16_metadata("split.count", 96),
                _u16_metadata("split.tensors.count", 2),
            ],
            payload_bytes=64,
        )
    )

    source = read_gguf(path)

    assert source.alignment == 32
    assert source.metadata["split.no"] == 6
    assert source.metadata["split.count"] == 96
    assert source.metadata["split.tensors.count"] == len(source.tensors) == 2
    assert source.metadata_types["split.no"] == 2
    assert source.metadata_types["split.count"] == 2
    assert source.metadata_types["split.tensors.count"] == 2


def test_read_gguf_rejects_wrong_per_shard_declared_tensor_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong-shard-count-00007-of-00096.gguf"
    path.write_bytes(
        _gguf_bytes(
            [("resident.f32", (4,), GGML_TYPE_F32, 0)],
            metadata_entries=[
                _u16_metadata("split.no", 6),
                _u16_metadata("split.count", 96),
                _u16_metadata("split.tensors.count", 2),
            ],
            payload_bytes=32,
        )
    )

    with pytest.raises(ValueError, match="split.tensors.count.*header tensor count"):
        read_gguf(path)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (
            _i32_metadata("general.alignment", 32),
            "general.alignment.*UINT32",
        ),
        (_u32_metadata("split.no", 0), "split.no.*UINT16"),
        (_u32_metadata("split.count", 96), "split.count.*UINT16"),
        (
            _u32_metadata("split.tensors.count", 23),
            "split.tensors.count.*UINT16",
        ),
    ],
)
def test_read_gguf_rejects_wrong_mandatory_metadata_types(
    tmp_path: Path, entry: bytes, message: str
) -> None:
    path = tmp_path / "wrong-metadata-type.gguf"
    path.write_bytes(
        _gguf_bytes(
            [],
            metadata_entries=(
                [entry]
                if message.startswith("general.alignment")
                else [_u32_metadata("general.alignment", 32), entry]
            ),
        )
    )
    with pytest.raises(ValueError, match=message):
        read_gguf(path)


def test_read_gguf_rejects_cumulative_metadata_object_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "too-many-metadata-objects.gguf"
    path.write_bytes(
        _gguf_bytes(
            [],
            metadata_entries=[
                _u32_metadata("general.alignment", 32),
                _u8_array_metadata("first", b"\x01\x02\x03\x04"),
                _u8_array_metadata("second", b"\x05\x06\x07\x08"),
            ],
        )
    )
    monkeypatch.setattr(kimi_gguf, "_MAX_METADATA_OBJECTS", 13)
    with pytest.raises(ValueError, match="metadata object budget"):
        read_gguf(path)


def _inventory_files(
    root: Path,
    *,
    wrong_shape: bool = False,
    resident_type: int | None = None,
    resident_count_delta: int = 0,
    resident_byte_delta: int = 0,
    declared_tensor_count: int | None = None,
) -> tuple[GGUFFile, ...]:
    per_shard: list[list[GGUFTensor]] = [[] for _ in range(96)]
    tensor_index = 0
    for layer in range(1, 93):
        for projection in ("gate", "up", "down"):
            dims = (3584, 3072, 896) if projection != "down" else (3072, 3584, 896)
            if wrong_shape and layer == 1 and projection == "gate":
                dims = (3584, 3071, 896)
            tensor = GGUFTensor(
                f"blk.{layer}.ffn_{projection}_exps.weight",
                dims,
                GGML_TYPE_Q2_K,
                0,
            )
            per_shard[tensor_index % 96].append(tensor)
            tensor_index += 1

    resident_specs = [
        (f"resident.bf16.{index:04d}", GGML_TYPE_BF16) for index in range(2122)
    ] + [(f"resident.f32.{index:04d}", GGML_TYPE_F32) for index in range(506)]
    if resident_count_delta == -1:
        resident_specs.pop()
    elif resident_count_delta == 1:
        resident_specs.append(("resident.extra", GGML_TYPE_BF16))
    elif resident_count_delta:
        raise AssertionError("test fixture supports only -1, 0, or +1 residents")

    target_resident_bytes = 114_404_258_816 + resident_byte_delta
    baseline_bytes = sum(
        2 if ggml_type == GGML_TYPE_BF16 else 4 for _, ggml_type in resident_specs
    )
    first_bf16_values = 1 + (target_resident_bytes - baseline_bytes) // 2
    for resident_index, (name, ggml_type) in enumerate(resident_specs):
        dims = (first_bf16_values,) if resident_index == 0 else (1,)
        if resident_type is not None and resident_index == 1:
            ggml_type = resident_type
        per_shard[tensor_index % 96].append(GGUFTensor(name, dims, ggml_type, 0))
        tensor_index += 1

    files: list[GGUFFile] = []
    for split, tensors in enumerate(per_shard):
        path = root / f"Kimi-K3-Q2_K-{split + 1:05d}-of-00096.gguf"
        path.touch()
        files.append(
            GGUFFile(
                path=path,
                version=3,
                alignment=32,
                data_offset=0,
                file_size=0,
                metadata=MappingProxyType(
                    {
                        "split.no": split,
                        "split.count": 96,
                        "split.tensors.count": (
                            len(tensors)
                            if declared_tensor_count is None
                            else declared_tensor_count
                        ),
                    }
                ),
                metadata_types=MappingProxyType(
                    {
                        "split.no": 2,
                        "split.count": 2,
                        "split.tensors.count": 2,
                    }
                ),
                tensors=tuple(tensors),
            )
        )
    return tuple(files)


def _install_inventory_fixture(
    monkeypatch: pytest.MonkeyPatch,
    files: tuple[GGUFFile, ...],
    *,
    expected_files: tuple[GGUFFile, ...] | None = None,
) -> None:
    by_path = {source.path: source for source in files}
    monkeypatch.setattr(kimi_gguf, "read_gguf", by_path.__getitem__)
    expected = files if expected_files is None else expected_files
    monkeypatch.setattr(
        kimi_gguf,
        "KIMI_K3_RESIDENT_DESCRIPTOR_SHA256",
        kimi_gguf.resident_descriptor_sha256(expected),
    )


def _mutate_resident_descriptor(
    files: tuple[GGUFFile, ...],
    mutation: str,
) -> tuple[GGUFFile, ...]:
    tensors = [list(source.tensors) for source in files]

    def locate(name: str) -> tuple[int, int]:
        for shard, owned in enumerate(tensors):
            for index, tensor in enumerate(owned):
                if tensor.name == name:
                    return shard, index
        raise AssertionError(name)

    bf16_shard, bf16_index = locate("resident.bf16.0000")
    other_shard, other_index = locate("resident.bf16.0001")
    f32_shard, f32_index = locate("resident.f32.0000")
    bf16 = tensors[bf16_shard][bf16_index]
    other = tensors[other_shard][other_index]
    f32 = tensors[f32_shard][f32_index]
    if mutation == "name":
        tensors[other_shard][other_index] = replace(other, name="resident.bf16.renamed")
    elif mutation == "dims":
        tensors[bf16_shard][bf16_index] = replace(bf16, dims=(bf16.dims[0] - 1,))
        tensors[other_shard][other_index] = replace(other, dims=(2,))
    elif mutation == "type":
        tensors[other_shard][other_index] = replace(other, ggml_type=GGML_TYPE_F32)
        tensors[f32_shard][f32_index] = replace(f32, ggml_type=GGML_TYPE_BF16)
    elif mutation == "span":
        tensors[other_shard][other_index] = replace(other, offset=other.offset + 32)
    elif mutation == "shard":
        moved = tensors[other_shard].pop(other_index)
        target_shard = (other_shard + 1) % len(tensors)
        tensors[target_shard].append(moved)
    else:
        raise AssertionError(mutation)

    changed: list[GGUFFile] = []
    for source, owned in zip(files, tensors, strict=True):
        metadata = {
            **source.metadata,
            "split.tensors.count": len(owned),
        }
        changed.append(
            replace(
                source,
                metadata=MappingProxyType(metadata),
                tensors=tuple(owned),
            )
        )
    return tuple(changed)


def test_inspect_kimi_k3_source_accepts_exact_pinned_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _inventory_files(tmp_path)
    _install_inventory_fixture(monkeypatch, files)

    inventory = inspect_kimi_k3_source(tmp_path, KIMI_K3_SOURCE_REVISION)

    assert inventory.revision == KIMI_K3_SOURCE_REVISION
    assert len(inventory.files) == 96
    assert len(inventory.expert_tensors) == 276
    assert len(inventory.resident_tensors) == 2628
    assert sum(len(source.tensors) for source in inventory.files) == 2904
    assert sum(tensor.nbytes for tensor in inventory.resident_tensors) == (
        114_404_258_816
    )
    assert sum(tensor.nbytes for tensor in inventory.expert_tensors) == (
        893_399_334_912
    )
    assert inventory.layers == tuple(range(1, 93))
    with pytest.raises(FrozenInstanceError):
        inventory.revision = "changed"  # type: ignore[misc]


def test_inspect_kimi_k3_source_stops_after_first_invalid_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _inventory_files(tmp_path)
    first = files[0]
    files = (
        replace(
            first,
            metadata=MappingProxyType(
                {
                    **first.metadata,
                    "split.no": 99,
                }
            ),
        ),
        *files[1:],
    )
    by_path = {source.path: source for source in files}
    calls: list[Path] = []

    def read_one(path: Path) -> GGUFFile:
        calls.append(path)
        return by_path[path]

    monkeypatch.setattr(kimi_gguf, "read_gguf", read_one)
    monkeypatch.setattr(
        kimi_gguf,
        "KIMI_K3_RESIDENT_DESCRIPTOR_SHA256",
        kimi_gguf.resident_descriptor_sha256(files),
    )
    with pytest.raises(ValueError, match="split.no"):
        inspect_kimi_k3_source(tmp_path, KIMI_K3_SOURCE_REVISION)
    assert calls == [files[0].path]


@pytest.mark.parametrize("mutation", ["shard", "name", "dims", "type", "span"])
def test_inspect_kimi_k3_source_rejects_resident_descriptor_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    baseline_root = tmp_path / "baseline"
    mutated_root = tmp_path / "mutated"
    baseline_root.mkdir()
    mutated_root.mkdir()
    baseline = _inventory_files(baseline_root)
    mutated = _mutate_resident_descriptor(_inventory_files(mutated_root), mutation)
    _install_inventory_fixture(
        monkeypatch,
        mutated,
        expected_files=baseline,
    )

    with pytest.raises(ValueError, match="resident descriptor digest"):
        inspect_kimi_k3_source(mutated_root, KIMI_K3_SOURCE_REVISION)


@pytest.mark.parametrize(
    ("wrong_shape", "resident_type", "message"),
    [
        (True, GGML_TYPE_BF16, "shape"),
        (False, 1, "resident tensor type"),
        (False, GGML_TYPE_F32, "tensor type counts"),
    ],
)
def test_inspect_kimi_k3_source_rejects_geometry_and_resident_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_shape: bool,
    resident_type: int,
    message: str,
) -> None:
    files = _inventory_files(
        tmp_path, wrong_shape=wrong_shape, resident_type=resident_type
    )
    _install_inventory_fixture(monkeypatch, files)
    with pytest.raises(ValueError, match=message):
        inspect_kimi_k3_source(tmp_path, KIMI_K3_SOURCE_REVISION)


@pytest.mark.parametrize(
    ("resident_count_delta", "message"),
    [
        (-1, "tensor count"),
        (1, "tensor count"),
    ],
)
def test_inspect_kimi_k3_source_rejects_missing_or_extra_residents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resident_count_delta: int,
    message: str,
) -> None:
    files = _inventory_files(tmp_path, resident_count_delta=resident_count_delta)
    _install_inventory_fixture(monkeypatch, files)
    with pytest.raises(ValueError, match=message):
        inspect_kimi_k3_source(tmp_path, KIMI_K3_SOURCE_REVISION)


def test_inspect_kimi_k3_source_rejects_wrong_resident_payload_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _inventory_files(tmp_path, resident_byte_delta=2)
    _install_inventory_fixture(monkeypatch, files)
    with pytest.raises(ValueError, match="resident payload"):
        inspect_kimi_k3_source(tmp_path, KIMI_K3_SOURCE_REVISION)


def test_inspect_kimi_k3_source_rejects_wrong_declared_tensor_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _inventory_files(tmp_path, declared_tensor_count=2903)
    _install_inventory_fixture(monkeypatch, files)
    with pytest.raises(ValueError, match="split.tensors.count"):
        inspect_kimi_k3_source(tmp_path, KIMI_K3_SOURCE_REVISION)


def test_inspect_kimi_k3_source_rejects_missing_shard_and_unpinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _inventory_files(tmp_path)
    files[-1].path.unlink()
    _install_inventory_fixture(monkeypatch, files)
    with pytest.raises(ValueError, match="96 GGUF shards"):
        inspect_kimi_k3_source(tmp_path, KIMI_K3_SOURCE_REVISION)
    with pytest.raises(ValueError, match="revision"):
        inspect_kimi_k3_source(tmp_path, "main")
