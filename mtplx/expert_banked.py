"""Banked expert sidecar: per-layer component-major banks (issue #51, C6).

The raw expert sidecar is record-major: each expert's nine component
segments sit together. ``gather_qmm`` wants the opposite — one contiguous
per-component bank whose row index is the expert id — so a banked sidecar
repacks selected layers component-major:

    layer region := [component 0 bank | component 1 bank | ...]
    component bank := [expert 0 segment | expert 1 segment | ...]

Every layer region starts on a 16 KiB boundary so it can be mapped into
Metal directly. The manifest carries a ``codec`` field; ``none`` stores
raw bank bytes, ``rans32x-v1`` is reserved for the entropy-coded variant
(issue #51, C7) and is rejected by loaders until its decoder ships.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from mtplx.expert_manifest import ExpertManifest

BANKED_FORMAT = "mtplx-banked-expert-banks-v1"
BANKED_ALIGNMENT = 16384
BANKED_CODECS = ("none", "rans32x-v1")
_DTYPE_ITEM_SIZE = {"U32": 4, "BF16": 2}


class BankedManifestError(ValueError):
    """Raised when a banked sidecar or its manifest is invalid."""


@dataclass(frozen=True)
class BankedComponent:
    component: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int

    def to_json(self) -> dict:
        return {
            "component": self.component,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "offset": self.offset,
            "length": self.length,
        }


@dataclass(frozen=True)
class BankedLayer:
    layer: int
    offset: int
    length: int
    sha256: str
    components: tuple[BankedComponent, ...]

    def to_json(self) -> dict:
        return {
            "layer": self.layer,
            "offset": self.offset,
            "length": self.length,
            "sha256": self.sha256,
            "components": [component.to_json() for component in self.components],
        }


@dataclass(frozen=True)
class BankedManifest:
    format: str
    model_key: str
    file: str
    codec: str
    alignment: int
    expert_count: int
    layers: tuple[BankedLayer, ...]
    path: Path | None = field(default=None, compare=False)

    def layer_entry(self, layer: int) -> BankedLayer:
        for entry in self.layers:
            if entry.layer == int(layer):
                return entry
        raise BankedManifestError(f"banked manifest has no layer {layer}")

    @property
    def layer_set(self) -> frozenset[int]:
        return frozenset(entry.layer for entry in self.layers)

    def bin_path(self) -> Path:
        if self.path is None:
            raise BankedManifestError("banked manifest has no on-disk location")
        return self.path.parent / self.file

    def to_json(self) -> dict:
        return {
            "format": self.format,
            "model_key": self.model_key,
            "file": self.file,
            "codec": self.codec,
            "alignment": self.alignment,
            "expert_count": self.expert_count,
            "layers": [entry.to_json() for entry in self.layers],
        }


def _component_order(record) -> tuple:
    seen = set()
    for segment in record.segments:
        if segment.component in seen:
            raise BankedManifestError(
                f"record ({record.layer}, {record.expert}) repeats component "
                f"{segment.component!r}"
            )
        seen.add(segment.component)
    return tuple(record.segments)


def write_banked_expert_banks(
    manifest: ExpertManifest,
    root: Path | str,
    layers: Iterable[int],
    *,
    output_bin: Path | str,
    output_manifest: Path | str,
    codec: str = "none",
    verify_record_hashes: bool = True,
) -> BankedManifest:
    """Repack selected layers of the raw sidecar into component-major banks."""

    if codec != "none":
        raise BankedManifestError(
            f"banked codec {codec!r} has no encoder yet; use 'none'"
        )
    if manifest.sidecar is None:
        raise BankedManifestError("banked repack requires a sidecar manifest")
    requested = tuple(sorted({int(layer) for layer in layers}))
    if not requested:
        raise BankedManifestError("banked repack requires at least one layer")
    records = {
        (record.layer, record.expert): record for record in manifest.records
    }
    expert_count = 1 + max(record.expert for record in manifest.records)
    reference = None
    output_bin = Path(output_bin)
    output_manifest = Path(output_manifest)
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = Path(root).resolve() / manifest.sidecar.file

    entries: list[BankedLayer] = []
    fd = os.open(sidecar_path, os.O_RDONLY)
    try:
        with output_bin.open("wb") as out:
            cursor = 0
            for layer in requested:
                banks: list[bytearray] | None = None
                segments = None
                for expert in range(expert_count):
                    record = records.get((layer, expert))
                    if record is None:
                        raise BankedManifestError(
                            f"manifest has no record for layer {layer} "
                            f"expert {expert}"
                        )
                    if record.sidecar_offset is None or record.sidecar_length is None:
                        raise BankedManifestError(
                            f"record ({layer}, {expert}) has no sidecar range"
                        )
                    segments = _component_order(record)
                    if reference is None:
                        reference = segments
                    if tuple(s.component for s in segments) != tuple(
                        s.component for s in reference
                    ):
                        raise BankedManifestError(
                            f"record ({layer}, {expert}) component order differs"
                        )
                    blob = os.pread(
                        fd, record.sidecar_length, record.sidecar_offset
                    )
                    if len(blob) != record.sidecar_length:
                        raise BankedManifestError(
                            f"short sidecar read for record ({layer}, {expert})"
                        )
                    if verify_record_hashes:
                        if record.sha256 is None:
                            raise BankedManifestError(
                                f"record ({layer}, {expert}) has no hash"
                            )
                        if hashlib.sha256(blob).hexdigest() != record.sha256:
                            raise BankedManifestError(
                                f"record hash mismatch: ({layer}, {expert})"
                            )
                    if banks is None:
                        banks = [bytearray() for _ in segments]
                    offset = 0
                    for index, segment in enumerate(segments):
                        banks[index].extend(
                            blob[offset : offset + segment.length]
                        )
                        offset += segment.length
                    if offset != record.logical_bytes:
                        raise BankedManifestError(
                            f"record ({layer}, {expert}) segments do not cover "
                            "its payload"
                        )
                assert banks is not None and segments is not None
                region_offset = cursor
                digest = hashlib.sha256()
                components: list[BankedComponent] = []
                component_cursor = 0
                for segment, bank in zip(segments, banks):
                    digest.update(bank)
                    components.append(
                        BankedComponent(
                            component=segment.component,
                            dtype=segment.dtype,
                            shape=tuple(segment.shape),
                            offset=component_cursor,
                            length=len(bank),
                        )
                    )
                    out.write(bank)
                    component_cursor += len(bank)
                    cursor += len(bank)
                entries.append(
                    BankedLayer(
                        layer=layer,
                        offset=region_offset,
                        length=component_cursor,
                        sha256=digest.hexdigest(),
                        components=tuple(components),
                    )
                )
                # mmap_u32 maps page-aligned extents only: pad the file so
                # every region (including the last) owns a full extent.
                padding = -cursor % BANKED_ALIGNMENT
                if padding:
                    out.write(b"\x00" * padding)
                    cursor += padding
    finally:
        os.close(fd)

    banked = BankedManifest(
        format=BANKED_FORMAT,
        model_key=manifest.model_key,
        file=output_bin.name,
        codec=codec,
        alignment=BANKED_ALIGNMENT,
        expert_count=expert_count,
        layers=tuple(entries),
        path=output_manifest,
    )
    output_manifest.write_text(json.dumps(banked.to_json(), indent=1))
    return banked


def load_banked_manifest(path: Path | str) -> BankedManifest:
    path = Path(path)
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise BankedManifestError(f"cannot read banked manifest: {exc}") from exc
    if not isinstance(obj, dict):
        raise BankedManifestError("banked manifest must be a JSON object")
    if obj.get("format") != BANKED_FORMAT:
        raise BankedManifestError(
            f"unsupported banked manifest format {obj.get('format')!r}"
        )
    codec = obj.get("codec")
    if codec not in BANKED_CODECS:
        raise BankedManifestError(f"unsupported banked codec {codec!r}")
    expert_count = obj.get("expert_count")
    if not isinstance(expert_count, int) or expert_count <= 0:
        raise BankedManifestError("banked expert_count must be positive")
    alignment = obj.get("alignment")
    if not isinstance(alignment, int) or alignment <= 0:
        raise BankedManifestError("banked alignment must be positive")
    layers_obj = obj.get("layers")
    if not isinstance(layers_obj, list) or not layers_obj:
        raise BankedManifestError("banked manifest lists no layers")
    layers: list[BankedLayer] = []
    for layer_obj in layers_obj:
        components: list[BankedComponent] = []
        cursor = 0
        for comp_obj in layer_obj.get("components", ()):
            dtype = comp_obj.get("dtype")
            item_size = _DTYPE_ITEM_SIZE.get(dtype)
            if item_size is None:
                raise BankedManifestError(
                    f"unsupported banked component dtype {dtype!r}"
                )
            shape = tuple(int(v) for v in comp_obj.get("shape", ()))
            length = comp_obj.get("length")
            offset = comp_obj.get("offset")
            if not isinstance(length, int) or not isinstance(offset, int):
                raise BankedManifestError("banked component range must be integral")
            elements = expert_count
            for value in shape:
                elements *= value
            if codec == "none" and length != elements * item_size:
                raise BankedManifestError(
                    f"banked component {comp_obj.get('component')!r} length "
                    f"{length} does not match shape {shape} x {expert_count}"
                )
            if offset != cursor:
                raise BankedManifestError(
                    f"banked component {comp_obj.get('component')!r} offset "
                    f"{offset} is not densely packed"
                )
            cursor += length
            components.append(
                BankedComponent(
                    component=str(comp_obj.get("component")),
                    dtype=str(dtype),
                    shape=shape,
                    offset=offset,
                    length=length,
                )
            )
        if not components:
            raise BankedManifestError("banked layer lists no components")
        entry_length = layer_obj.get("length")
        if entry_length != cursor:
            raise BankedManifestError(
                f"banked layer {layer_obj.get('layer')} length {entry_length} "
                f"does not cover its components ({cursor})"
            )
        entry_offset = layer_obj.get("offset")
        if not isinstance(entry_offset, int) or entry_offset % alignment:
            raise BankedManifestError(
                f"banked layer {layer_obj.get('layer')} offset is not aligned"
            )
        layers.append(
            BankedLayer(
                layer=int(layer_obj.get("layer")),
                offset=entry_offset,
                length=int(entry_length),
                sha256=str(layer_obj.get("sha256")),
                components=tuple(components),
            )
        )
    return BankedManifest(
        format=BANKED_FORMAT,
        model_key=str(obj.get("model_key")),
        file=str(obj.get("file")),
        codec=str(codec),
        alignment=alignment,
        expert_count=expert_count,
        layers=tuple(layers),
        path=path,
    )
