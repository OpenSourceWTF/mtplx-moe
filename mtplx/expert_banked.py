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
BANKED_CODECS = ("none", "huffman-l12-v1", "rans32x-v1")
BANKED_PER_LANE = 4096
_DTYPE_ITEM_SIZE = {"U32": 4, "BF16": 2}
_CLASS_OF_COMPONENT = {
    "weight": "weight",
    "scales": "scales",
    "biases": "biases",
}


def component_class(component: str) -> str:
    kind = component.rsplit(".", 1)[-1]
    try:
        return _CLASS_OF_COMPONENT[kind]
    except KeyError as exc:
        raise BankedManifestError(
            f"component {component!r} has no codec class"
        ) from exc


class BankedManifestError(ValueError):
    """Raised when a banked sidecar or its manifest is invalid."""


@dataclass(frozen=True)
class BankedComponent:
    component: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int
    lanes: int = 0  # per-expert decode lanes (compressed codecs only)

    @property
    def segment_length(self) -> int:
        item = _DTYPE_ITEM_SIZE[self.dtype]
        elements = 1
        for value in self.shape:
            elements *= value
        return elements * item

    def to_json(self) -> dict:
        result = {
            "component": self.component,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "offset": self.offset,
            "length": self.length,
        }
        if self.lanes:
            result["lanes"] = self.lanes
        return result


@dataclass(frozen=True)
class BankedLayer:
    layer: int
    offset: int
    length: int
    sha256: str
    components: tuple[BankedComponent, ...]
    directory_words: int = 0  # compressed codecs: u32 lane offsets before payload

    def to_json(self) -> dict:
        result = {
            "layer": self.layer,
            "offset": self.offset,
            "length": self.length,
            "sha256": self.sha256,
            "components": [component.to_json() for component in self.components],
        }
        if self.directory_words:
            result["directory_words"] = self.directory_words
        return result


@dataclass(frozen=True)
class BankedManifest:
    format: str
    model_key: str
    file: str
    codec: str
    alignment: int
    expert_count: int
    layers: tuple[BankedLayer, ...]
    tables: dict | None = None  # codec class -> 256 code lengths
    per_lane: int = 0
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
        result = {
            "format": self.format,
            "model_key": self.model_key,
            "file": self.file,
            "codec": self.codec,
            "alignment": self.alignment,
            "expert_count": self.expert_count,
            "layers": [entry.to_json() for entry in self.layers],
        }
        if self.tables is not None:
            result["tables"] = self.tables
        if self.per_lane:
            result["per_lane"] = self.per_lane
        return result


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
    """Repack selected layers of the raw sidecar into component-major banks.

    ``codec="huffman-l12-v1"`` writes each expert segment as independent
    length-limited-Huffman lane streams (weights raw-byte, BF16 scales and
    biases group-plane-split first), preceded per layer by a u32 directory
    of payload-relative lane word offsets. Tables are global per component
    class across the packed layers (measured: per-layer tables add 0.1%).
    """

    if codec == "rans32x-v1":
        raise BankedManifestError(
            f"banked codec {codec!r} has no encoder yet; use 'none'"
        )
    if codec == "huffman-l12-v1":
        return _write_compressed_banked_banks(
            manifest,
            root,
            layers,
            output_bin=Path(output_bin),
            output_manifest=Path(output_manifest),
            verify_record_hashes=verify_record_hashes,
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


def _read_layer_segments(
    manifest: ExpertManifest,
    sidecar_fd: int,
    records: dict,
    layer: int,
    expert_count: int,
    *,
    verify_record_hashes: bool,
):
    """Yield (expert, segments, blob-slices) for one layer, hash-verified."""

    for expert in range(expert_count):
        record = records.get((layer, expert))
        if record is None:
            raise BankedManifestError(
                f"manifest has no record for layer {layer} expert {expert}"
            )
        if record.sidecar_offset is None or record.sidecar_length is None:
            raise BankedManifestError(
                f"record ({layer}, {expert}) has no sidecar range"
            )
        blob = os.pread(sidecar_fd, record.sidecar_length, record.sidecar_offset)
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
        yield expert, record, blob


def _transform_segment(seg_bytes: bytes, dtype: str):
    import numpy as np

    from mtplx.expert_huffman import plane_split_groups

    data = np.frombuffer(seg_bytes, dtype=np.uint8)
    if dtype == "BF16":
        return plane_split_groups(data)
    return data


def _write_compressed_banked_banks(
    manifest: ExpertManifest,
    root: Path | str,
    layers,
    *,
    output_bin: Path,
    output_manifest: Path,
    verify_record_hashes: bool,
) -> BankedManifest:
    import numpy as np

    from mtplx.expert_huffman import build_table, encode_lanes_fast

    if manifest.sidecar is None:
        raise BankedManifestError("banked repack requires a sidecar manifest")
    requested = tuple(sorted({int(layer) for layer in layers}))
    if not requested:
        raise BankedManifestError("banked repack requires at least one layer")
    records = {
        (record.layer, record.expert): record for record in manifest.records
    }
    expert_count = 1 + max(record.expert for record in manifest.records)
    sidecar_path = Path(root).resolve() / manifest.sidecar.file
    output_bin.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: global per-class byte histograms over the transformed data.
    hists = {
        "weight": np.zeros(256, dtype=np.int64),
        "scales": np.zeros(256, dtype=np.int64),
        "biases": np.zeros(256, dtype=np.int64),
    }
    fd = os.open(sidecar_path, os.O_RDONLY)
    try:
        for layer in requested:
            for _expert, record, blob in _read_layer_segments(
                manifest, fd, records, layer, expert_count,
                verify_record_hashes=verify_record_hashes,
            ):
                cursor = 0
                for segment in record.segments:
                    kind = component_class(segment.component)
                    transformed = _transform_segment(
                        blob[cursor : cursor + segment.length], segment.dtype
                    )
                    hists[kind] += np.bincount(transformed, minlength=256)
                    cursor += segment.length
        tables = {
            kind: build_table(hist, max_bits=12)
            for kind, hist in hists.items()
        }

        # Pass 2: encode per (layer, expert, component) and write regions.
        entries: list[BankedLayer] = []
        with output_bin.open("wb") as out:
            cursor = 0
            for layer in requested:
                per_expert_streams: list[list] = []
                segments_ref = None
                for _expert, record, blob in _read_layer_segments(
                    manifest, fd, records, layer, expert_count,
                    verify_record_hashes=False,
                ):
                    segments_ref = record.segments
                    seg_streams = []
                    seg_cursor = 0
                    for segment in record.segments:
                        kind = component_class(segment.component)
                        transformed = _transform_segment(
                            blob[seg_cursor : seg_cursor + segment.length],
                            segment.dtype,
                        )
                        lane_bytes = min(BANKED_PER_LANE, segment.length)
                        if segment.length % lane_bytes:
                            raise BankedManifestError(
                                f"segment {segment.component} length "
                                f"{segment.length} is not lane-divisible"
                            )
                        seg_streams.append(
                            encode_lanes_fast(
                                transformed,
                                tables[kind],
                                per_lane=lane_bytes,
                            )
                        )
                        seg_cursor += segment.length
                    per_expert_streams.append(seg_streams)
                assert segments_ref is not None

                directory: list[np.ndarray] = []
                payload: list[np.ndarray] = []
                payload_words = 0
                for seg_streams in per_expert_streams:
                    for stream in seg_streams:
                        directory.append(
                            stream.word_offsets.astype(np.uint32)
                            + np.uint32(payload_words)
                        )
                        payload.append(stream.words)
                        payload_words += stream.words.size
                directory_arr = np.concatenate(directory).astype(np.uint32)
                dir_bytes = directory_arr.tobytes()
                # Pad the directory block to the mapping alignment so the
                # payload starts on its own page-aligned extent (directory
                # and payload map as separate clean regions).
                dir_pad = -len(dir_bytes) % BANKED_ALIGNMENT
                region = (
                    dir_bytes
                    + b"\x00" * dir_pad
                    + b"".join(w.tobytes() for w in payload)
                )
                components = []
                raw_cursor = 0
                for index, segment in enumerate(segments_ref):
                    lane_bytes = min(BANKED_PER_LANE, segment.length)
                    components.append(
                        BankedComponent(
                            component=segment.component,
                            dtype=segment.dtype,
                            shape=tuple(segment.shape),
                            offset=raw_cursor,
                            length=expert_count * segment.length,
                            lanes=segment.length // lane_bytes,
                        )
                    )
                    raw_cursor += expert_count * segment.length
                entries.append(
                    BankedLayer(
                        layer=layer,
                        offset=cursor,
                        length=len(region),
                        sha256=hashlib.sha256(region).hexdigest(),
                        components=tuple(components),
                        directory_words=directory_arr.size,
                    )
                )
                out.write(region)
                cursor += len(region)
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
        codec="huffman-l12-v1",
        alignment=BANKED_ALIGNMENT,
        expert_count=expert_count,
        layers=tuple(entries),
        tables={
            kind: table.lengths.astype(int).tolist()
            for kind, table in tables.items()
        },
        per_lane=BANKED_PER_LANE,
        path=output_manifest,
    )
    output_manifest.write_text(json.dumps(banked.to_json(), indent=1))
    return banked


def decode_banked_layer_reference(
    banked: BankedManifest,
    layer: int,
    region: bytes,
) -> dict[str, "np.ndarray"]:
    """Pure-numpy reference decode of one compressed layer region (tests)."""

    import numpy as np

    from mtplx.expert_huffman import (
        HuffmanTable,
        _canonical_codes,
        plane_unsplit_groups,
    )

    if banked.codec != "huffman-l12-v1":
        raise BankedManifestError("reference decode requires the huffman codec")
    entry = banked.layer_entry(layer)
    directory = np.frombuffer(
        region[: entry.directory_words * 4], dtype=np.uint32
    )
    payload_start = (
        (entry.directory_words * 4 + banked.alignment - 1)
        // banked.alignment
        * banked.alignment
    )
    payload = np.frombuffer(region[payload_start:], dtype=np.uint32)
    tables = {}
    luts = {}
    for kind, lengths_list in (banked.tables or {}).items():
        lengths = np.asarray(lengths_list, dtype=np.uint8)
        table = HuffmanTable(
            lengths=lengths, codes=_canonical_codes(lengths), max_bits=12
        )
        lut_sym = np.zeros(1 << 12, dtype=np.uint8)
        lut_len = np.zeros(1 << 12, dtype=np.uint8)
        for sym in range(256):
            length = int(lengths[sym])
            base = int(table.codes[sym]) << (12 - length)
            span = 1 << (12 - length)
            lut_sym[base : base + span] = sym
            lut_len[base : base + span] = length
        tables[kind] = table
        luts[kind] = (lut_sym, lut_len)

    raw_bytes = payload.byteswap().tobytes()
    banks: dict[str, np.ndarray] = {
        c.component: np.zeros(c.length, dtype=np.uint8)
        for c in entry.components
    }
    dir_cursor = 0
    for expert in range(banked.expert_count):
        for component in entry.components:
            seg_len = component.segment_length
            lanes = component.lanes
            lane_bytes = seg_len // lanes
            lut_sym, lut_len = luts[component_class(component.component)]
            decoded = np.zeros(seg_len, dtype=np.uint8)
            for lane in range(lanes):
                word_off = int(directory[dir_cursor + lane])
                byte_base = word_off * 4
                bitpos = 0
                for i in range(lane_bytes):
                    b0 = byte_base + (bitpos >> 3)
                    window = (
                        (raw_bytes[b0] << 24)
                        | (raw_bytes[b0 + 1] << 16)
                        | (raw_bytes[b0 + 2] << 8)
                        | raw_bytes[b0 + 3]
                    )
                    peek = (window >> (32 - (bitpos & 7) - 12)) & 0xFFF
                    decoded[lane * lane_bytes + i] = lut_sym[peek]
                    bitpos += int(lut_len[peek])
            dir_cursor += lanes
            if component.dtype == "BF16":
                decoded = plane_unsplit_groups(decoded)
            banks[component.component][
                expert * seg_len : (expert + 1) * seg_len
            ] = decoded
    return banks


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
    compressed = codec == "huffman-l12-v1"
    tables = obj.get("tables")
    per_lane = obj.get("per_lane", 0)
    if compressed:
        if not isinstance(tables, dict) or set(tables) != {
            "weight",
            "scales",
            "biases",
        }:
            raise BankedManifestError(
                "compressed banked manifest requires weight/scales/biases tables"
            )
        for kind, lengths in tables.items():
            if len(lengths) != 256 or not all(
                isinstance(v, int) and 1 <= v <= 12 for v in lengths
            ):
                raise BankedManifestError(
                    f"banked codec table {kind!r} must be 256 lengths in 1..12"
                )
        if not isinstance(per_lane, int) or per_lane <= 0:
            raise BankedManifestError(
                "compressed banked manifest requires a positive per_lane"
            )
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
            if codec != "rans32x-v1" and length != elements * item_size:
                raise BankedManifestError(
                    f"banked component {comp_obj.get('component')!r} length "
                    f"{length} does not match shape {shape} x {expert_count}"
                )
            if offset != cursor:
                raise BankedManifestError(
                    f"banked component {comp_obj.get('component')!r} offset "
                    f"{offset} is not densely packed"
                )
            lanes = int(comp_obj.get("lanes", 0))
            if compressed:
                seg_len = length // expert_count
                if lanes <= 0 or seg_len % lanes:
                    raise BankedManifestError(
                        f"banked component {comp_obj.get('component')!r} lane "
                        f"count {lanes} does not divide its segment"
                    )
            cursor += length
            components.append(
                BankedComponent(
                    component=str(comp_obj.get("component")),
                    dtype=str(dtype),
                    shape=shape,
                    offset=offset,
                    length=length,
                    lanes=lanes,
                )
            )
        if not components:
            raise BankedManifestError("banked layer lists no components")
        entry_length = layer_obj.get("length")
        directory_words = int(layer_obj.get("directory_words", 0))
        if compressed:
            if directory_words <= 0:
                raise BankedManifestError(
                    f"banked layer {layer_obj.get('layer')} lists no directory"
                )
            if not isinstance(entry_length, int) or entry_length <= (
                directory_words * 4
            ):
                raise BankedManifestError(
                    f"banked layer {layer_obj.get('layer')} region is smaller "
                    "than its directory"
                )
            expected_dir = sum(c.lanes for c in components) * expert_count
            if directory_words != expected_dir:
                raise BankedManifestError(
                    f"banked layer {layer_obj.get('layer')} directory has "
                    f"{directory_words} words; lanes require {expected_dir}"
                )
        elif entry_length != cursor:
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
                directory_words=directory_words,
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
        tables=tables if compressed else None,
        per_lane=int(per_lane) if compressed else 0,
        path=path,
    )
