"""Construction-owned GLM-5.2 Q1T fused-rANS expert route."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from collections import Counter
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import swiglu
import numpy as np

from mtplx.expert_rans import (
    LANES,
    M,
    RANS_CONTAINER_MAGIC,
    RANS_CONTAINER_VERSION,
    _HEADER_DTYPE,
    table_from_freq,
)
from mtplx.expert_streaming_models import GLM52_EXPERT_Q1T
from mtplx.glm52_q1t_over10 import (
    GLM52_Q1T_BASE_MANIFEST_SHA256,
    GLM52_Q1T_FUSED_RANS_CODEC,
    GLM52_Q1T_FUSED_RANS_GATE_UP_THREADGROUPS,
    GLM52_Q1T_PERSISTENT_SLOTS_PER_LAYER,
    GLM52_Q1T_TRANSIENT_SLOTS,
)
from mtplx.glm52_q1t_rans_artifact import (
    COMPONENT_ALIGNMENT,
    FUSED_RANS_FORMAT,
    FUSED_RANS_MODEL_KEY,
    FUSED_RANS_SOURCE_CODEC,
    FusedRansComponent,
    Glm52Q1TFusedRansManifest,
)
from mtplx.kernels.glm52_q1t_fused_rans import (
    _CACHED_EXPERT_ROUTE_TABLE_BYTES,
    bind_glm52_q1t_fused_rans_cached_bank,
)
from mtplx.mmap_mlx import allocate_metal_u8
from mtplx.models.expert_mlx import current_expert_routing_phase
from mtplx.expert_streaming import RoutingPhase


class Glm52Q1TFusedRansConstructionError(RuntimeError):
    """Raised before execution when the complete GLM fused route is invalid."""


_COMPONENT_GEOMETRY = (
    ("gate_proj.packed", "U8", (2048, 1248), 6144, 2048, 1248),
    ("gate_proj.scales", "U16", (2048, 96), 6144, 2048, 192),
    ("up_proj.packed", "U8", (2048, 1248), 6144, 2048, 1248),
    ("up_proj.scales", "U16", (2048, 96), 6144, 2048, 192),
    ("down_proj.packed", "U8", (6144, 416), 2048, 6144, 416),
    ("down_proj.scales", "U16", (6144, 32), 2048, 6144, 64),
)
_PROJECTION_NAMES = ("gate_proj", "up_proj", "down_proj")
# Guarded full-artifact 252-arm real-record receipt, layer 3, 2026-07-20.
# Candidate and unchanged shadow timings were interleaved for every arm;
# every arm was bitwise exact and emitted zero decoded-weight outputs.
_QUALIFIED_REAL_SHAPE_REPORT_SHA256 = (
    "5eb92c6462f1ff10a8edf19347362848e65d91a9d8340616e09bee3ed00262df"
)
_QUALIFIED_PROJECTION_THREADGROUPS = MappingProxyType(
    {"gate_proj": 64, "up_proj": 64, "down_proj": 64}
)
_QUALIFIED_GATE_UP_THREADGROUPS = GLM52_Q1T_FUSED_RANS_GATE_UP_THREADGROUPS
_QUALIFIED_ASSIGNMENT_COUNTS = (1, 2, 3, 8, 16, 24, 32)
_SELF_CHECK_FORMAT = "mtplx-glm52-q1t-fused-rans-selfcheck-v3"
_SELF_CHECK_SUFFIX = ".selfcheck-v3.json"


def _normalize_projection_threadgroups(
    value: Mapping[str, int],
) -> dict[str, int]:
    normalized = {str(name): int(threads) for name, threads in value.items()}
    if set(normalized) != set(_PROJECTION_NAMES):
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS launch geometry must cover gate/up/down exactly"
        )
    for name, out_dim in (("gate_proj", 2048), ("up_proj", 2048), ("down_proj", 6144)):
        threads = normalized[name]
        if threads not in (32, 64, 128, 256, 512, 1024) or out_dim % threads:
            raise Glm52Q1TFusedRansConstructionError(
                f"fused-rANS {name} threadgroup geometry is incompatible"
            )
    return normalized


def _qualified_projection_threadgroups() -> dict[str, int]:
    return _normalize_projection_threadgroups(_QUALIFIED_PROJECTION_THREADGROUPS)


def validate_glm52_q1t_fused_rans_manifest(
    manifest: Glm52Q1TFusedRansManifest,
) -> None:
    """Validate the complete immutable GLM identity and real component geometry."""

    if (
        manifest.format != FUSED_RANS_FORMAT
        or manifest.model_key != FUSED_RANS_MODEL_KEY
    ):
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS manifest model identity is not glm52-expert-q1t"
        )
    if manifest.codec != GLM52_Q1T_FUSED_RANS_CODEC:
        raise Glm52Q1TFusedRansConstructionError(
            f"fused-rANS manifest codec must be {GLM52_Q1T_FUSED_RANS_CODEC}"
        )
    if manifest.source_codec != FUSED_RANS_SOURCE_CODEC:
        raise Glm52Q1TFusedRansConstructionError("fused-rANS source codec must be t158")
    if manifest.source_manifest_sha256 != GLM52_Q1T_BASE_MANIFEST_SHA256:
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS source is not the authoritative GLM Q1T manifest"
        )
    if manifest.expert_count != GLM52_EXPERT_Q1T.expert_count:
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS expert geometry is incompatible"
        )
    if manifest.output_tile != 32 or manifest.alignment != COMPONENT_ALIGNMENT:
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS output tile or mapping alignment is incompatible"
        )
    if (
        manifest.routed_layers != GLM52_EXPERT_Q1T.routed_layer_indices
        or tuple(layer.layer for layer in manifest.layers)
        != GLM52_EXPERT_Q1T.routed_layer_indices
    ):
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS routed layer coverage is incomplete"
        )
    extent = 0
    for layer in manifest.layers:
        if len(layer.components) != len(_COMPONENT_GEOMETRY):
            raise Glm52Q1TFusedRansConstructionError(
                f"fused-rANS layer {layer.layer} component coverage is incomplete"
            )
        for component, expected in zip(
            layer.components, _COMPONENT_GEOMETRY, strict=True
        ):
            name, dtype, shape, in_dim, out_dim, row_bytes = expected
            tiles = out_dim // manifest.output_tile
            if (
                component.component != name
                or component.dtype != dtype
                or component.shape != shape
                or component.in_dim != in_dim
                or component.out_dim != out_dim
                or component.row_bytes != row_bytes
                or component.per_lane != row_bytes
                or component.record_count != manifest.expert_count * tiles
                or component.raw_length != manifest.expert_count * out_dim * row_bytes
                or component.lanes != manifest.output_tile
            ):
                raise Glm52Q1TFusedRansConstructionError(
                    f"fused-rANS component {name} geometry is incompatible"
                )
            if (
                component.offset != extent
                or component.offset % COMPONENT_ALIGNMENT
                or component.mapped_length % COMPONENT_ALIGNMENT
                or component.length > component.mapped_length
            ):
                raise Glm52Q1TFusedRansConstructionError(
                    f"fused-rANS component {name} mapped extent is incompatible"
                )
            extent += component.mapped_length
    if extent != manifest.file_bytes:
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS file extent does not cover every component"
        )


def verify_glm52_q1t_fused_rans_artifact(
    manifest: Glm52Q1TFusedRansManifest,
) -> float:
    """Hash every compressed component and the complete file once."""

    started = time.perf_counter()
    path = manifest.bin_path()
    try:
        if path.stat().st_size != manifest.file_bytes:
            raise Glm52Q1TFusedRansConstructionError(
                "fused-rANS binary size does not match its manifest"
            )
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            fcntl.fcntl(handle.fileno(), fcntl.F_NOCACHE, 1)
            for layer in manifest.layers:
                for component in layer.components:
                    if handle.tell() != component.offset:
                        raise Glm52Q1TFusedRansConstructionError(
                            "fused-rANS component file cursor is inconsistent"
                        )
                    component_digest = hashlib.sha256()
                    remaining = component.length
                    while remaining:
                        chunk = handle.read(min(remaining, 8 * 1024 * 1024))
                        if not chunk:
                            raise Glm52Q1TFusedRansConstructionError(
                                "fused-rANS component is truncated"
                            )
                        component_digest.update(chunk)
                        file_digest.update(chunk)
                        remaining -= len(chunk)
                    if component_digest.hexdigest() != component.sha256:
                        raise Glm52Q1TFusedRansConstructionError(
                            f"fused-rANS component {component.component} hash mismatch"
                        )
                    padding = component.mapped_length - component.length
                    while padding:
                        chunk = handle.read(min(padding, 8 * 1024 * 1024))
                        if not chunk or any(chunk):
                            raise Glm52Q1TFusedRansConstructionError(
                                "fused-rANS component padding is corrupt"
                            )
                        file_digest.update(chunk)
                        padding -= len(chunk)
            if handle.read(1):
                raise Glm52Q1TFusedRansConstructionError(
                    "fused-rANS binary has an unmanifested tail"
                )
        if file_digest.hexdigest() != manifest.file_sha256:
            raise Glm52Q1TFusedRansConstructionError(
                "fused-rANS complete artifact hash mismatch"
            )
    except OSError as exc:
        raise Glm52Q1TFusedRansConstructionError(
            f"cannot verify fused-rANS artifact: {exc}"
        ) from exc
    return time.perf_counter() - started


def _pack_rans_transition_table(
    cum2sym: np.ndarray,
    freq: np.ndarray,
    cum: np.ndarray,
) -> np.ndarray:
    symbols = np.asarray(cum2sym, dtype=np.uint32)
    frequencies = np.asarray(freq, dtype=np.uint32)
    cumulative = np.asarray(cum, dtype=np.uint32)
    if (
        symbols.shape != (M,)
        or frequencies.shape != (256,)
        or cumulative.shape != (256,)
    ):
        raise Glm52Q1TFusedRansConstructionError(
            "rANS transition table geometry is incompatible"
        )
    slot_frequency = frequencies[symbols]
    residue = np.arange(M, dtype=np.uint32) - cumulative[symbols]
    if (
        np.any(slot_frequency < 1)
        or np.any(slot_frequency > M)
        or np.any(residue >= slot_frequency)
    ):
        raise Glm52Q1TFusedRansConstructionError(
            "rANS transition table values are incompatible"
        )
    return (symbols | ((slot_frequency - 1) << 8) | (residue << 20)).astype(
        np.uint32, copy=False
    )


@dataclass(frozen=True)
class _PreparedFusedRansComponent:
    component: FusedRansComponent
    cum2sym: mx.array
    freq: mx.array
    cum: mx.array
    transition: mx.array


def _prepare_fused_rans_component(
    path: Path,
    component: FusedRansComponent,
    *,
    expert_count: int,
    require_uniform: bool = False,
) -> _PreparedFusedRansComponent:
    """Validate one container and construct its immutable small decode table."""

    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            metadata = os.pread(
                fd,
                component.directory_offset,
                component.offset,
            )
        finally:
            os.close(fd)
    except OSError as exc:
        raise Glm52Q1TFusedRansConstructionError(
            f"cannot inspect fused-rANS component {component.component}: {exc}"
        ) from exc
    if len(metadata) != component.directory_offset:
        raise Glm52Q1TFusedRansConstructionError(
            f"fused-rANS component {component.component} header is truncated"
        )
    header = np.frombuffer(metadata[: _HEADER_DTYPE.itemsize], dtype=_HEADER_DTYPE)[0]
    expected_seg_len = LANES * component.per_lane
    if (
        bytes(header["magic"]) != RANS_CONTAINER_MAGIC
        or int(header["version"]) != RANS_CONTAINER_VERSION
        or int(header["lanes"]) != LANES
        or int(header["expert_count"]) != component.record_count
        or int(header["seg_len"]) != expected_seg_len
        or int(header["payload_len"]) != component.payload_length
    ):
        raise Glm52Q1TFusedRansConstructionError(
            f"fused-rANS component {component.component} container header is incompatible"
        )
    frequency_bytes = metadata[component.frequency_offset : component.directory_offset]
    frequency = np.frombuffer(frequency_bytes, dtype="<u2").astype(np.uint32)
    if require_uniform and not np.array_equal(
        frequency,
        np.full(256, 16, dtype=np.uint32),
    ):
        raise Glm52Q1TFusedRansConstructionError(
            f"fused-rANS component {component.component} is not uniform freq-16"
        )
    try:
        table = table_from_freq(frequency)
    except ValueError as exc:
        raise Glm52Q1TFusedRansConstructionError(
            f"fused-rANS component {component.component} table is invalid: {exc}"
        ) from exc
    del expert_count
    return _PreparedFusedRansComponent(
        component=component,
        cum2sym=mx.array(table.cum2sym, dtype=mx.uint8),
        freq=mx.array(table.freq, dtype=mx.uint32),
        cum=mx.array(table.cum[:-1], dtype=mx.uint32),
        transition=mx.array(
            _pack_rans_transition_table(
                table.cum2sym,
                table.freq,
                table.cum[:-1],
            ),
            dtype=mx.uint32,
        ),
    )


def _read_component_expert_payload_offsets(
    path: Path,
    component: FusedRansComponent,
    *,
    expert_count: int,
) -> tuple[int, ...]:
    directory_bytes = component.record_count * LANES * 4
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            raw = os.pread(
                fd,
                directory_bytes,
                component.offset + component.directory_offset,
            )
        finally:
            os.close(fd)
    except OSError as exc:
        raise Glm52Q1TFusedRansConstructionError(
            f"cannot inspect fused-rANS component directory: {exc}"
        ) from exc
    if len(raw) != directory_bytes:
        raise Glm52Q1TFusedRansConstructionError(
            f"fused-rANS component {component.component} directory is truncated"
        )
    directory = np.frombuffer(raw, dtype="<u4")
    rows_per_expert = component.out_dim
    offsets = tuple(
        int(directory[expert * rows_per_expert]) for expert in range(expert_count)
    ) + (component.payload_length,)
    if offsets[0] != 0 or any(
        start >= stop for start, stop in zip(offsets, offsets[1:])
    ):
        raise Glm52Q1TFusedRansConstructionError(
            f"fused-rANS component {component.component} expert payloads overlap"
        )
    return offsets


@dataclass(frozen=True)
class _Glm52Q1TRansCacheCopy:
    source_offset: int
    destination_offset: int
    length: int


@dataclass(frozen=True)
class _Glm52Q1TRansCacheExpertSource:
    layer: int
    expert: int
    header: bytes
    copies: tuple[_Glm52Q1TRansCacheCopy, ...]
    image_bytes: int


def _build_glm52_q1t_rans_cache_source(
    layer: Any,
    expert: int,
    component_offsets: tuple[tuple[int, ...], ...],
    *,
    slot_bytes: int,
    max_read_chunk_bytes: int,
) -> _Glm52Q1TRansCacheExpertSource:
    header = [0] * 18
    copies: list[_Glm52Q1TRansCacheCopy] = []
    cursor = len(header) * 4
    for component_index, (component, payload_offsets) in enumerate(
        zip(layer.components, component_offsets, strict=True)
    ):
        cursor = (cursor + 3) & ~3
        directory_length = component.out_dim * 4
        directory_source = (
            component.offset + component.directory_offset + expert * directory_length
        )
        directory_destination = cursor
        cursor += directory_length
        payload_start = payload_offsets[expert]
        payload_stop = payload_offsets[expert + 1]
        payload_length = payload_stop - payload_start
        if (
            directory_length > max_read_chunk_bytes
            or payload_length > max_read_chunk_bytes
        ):
            raise Glm52Q1TFusedRansConstructionError(
                f"compressed rANS expert ({layer.layer}, {expert}) "
                "component exceeds the fixed read chunk"
            )
        payload_destination = cursor
        cursor += payload_length
        header_index = component_index * 3
        header[header_index] = directory_destination
        header[header_index + 1] = payload_destination
        header[header_index + 2] = payload_start
        copies.extend(
            (
                _Glm52Q1TRansCacheCopy(
                    source_offset=directory_source,
                    destination_offset=directory_destination,
                    length=directory_length,
                ),
                _Glm52Q1TRansCacheCopy(
                    source_offset=(
                        component.offset + component.payload_offset + payload_start
                    ),
                    destination_offset=payload_destination,
                    length=payload_length,
                ),
            )
        )
    usable_slot_bytes = slot_bytes - _CACHED_EXPERT_ROUTE_TABLE_BYTES
    if cursor > usable_slot_bytes:
        raise Glm52Q1TFusedRansConstructionError(
            f"compressed rANS expert ({layer.layer}, {expert}) needs "
            f"{cursor} cache bytes; fixed t158 slot leaves "
            f"{usable_slot_bytes} bytes before its routed-slot table"
        )
    return _Glm52Q1TRansCacheExpertSource(
        layer=layer.layer,
        expert=expert,
        header=np.asarray(header, dtype="<u4").tobytes(),
        copies=tuple(copies),
        image_bytes=cursor,
    )


def _layer_component_payload_offsets(
    artifact: Glm52Q1TFusedRansManifest,
    layer: Any,
) -> tuple[tuple[int, ...], ...]:
    path = artifact.bin_path()
    return tuple(
        _read_component_expert_payload_offsets(
            path,
            component,
            expert_count=artifact.expert_count,
        )
        for component in layer.components
    )


def _glm52_q1t_rans_cache_source(
    artifact: Glm52Q1TFusedRansManifest,
    *,
    layer: int,
    expert: int,
    slot_bytes: int,
    max_read_chunk_bytes: int,
) -> _Glm52Q1TRansCacheExpertSource:
    """Resolve one real expert image for focused cache-kernel qualification."""

    layer_entry = next(
        (entry for entry in artifact.layers if entry.layer == int(layer)),
        None,
    )
    if layer_entry is None or not 0 <= int(expert) < artifact.expert_count:
        raise Glm52Q1TFusedRansConstructionError(
            "focused compressed rANS cache source is outside the artifact"
        )
    return _build_glm52_q1t_rans_cache_source(
        layer_entry,
        int(expert),
        _layer_component_payload_offsets(artifact, layer_entry),
        slot_bytes=slot_bytes,
        max_read_chunk_bytes=max_read_chunk_bytes,
    )


def _glm52_q1t_rans_cache_sources(
    artifact: Glm52Q1TFusedRansManifest,
    *,
    slot_bytes: int,
    max_read_chunk_bytes: int,
) -> dict[tuple[int, int], _Glm52Q1TRansCacheExpertSource]:
    """Resolve every immutable component range before cache allocation."""

    sources: dict[tuple[int, int], _Glm52Q1TRansCacheExpertSource] = {}
    for layer in artifact.layers:
        component_offsets = _layer_component_payload_offsets(artifact, layer)
        for expert in range(artifact.expert_count):
            sources[(layer.layer, expert)] = _build_glm52_q1t_rans_cache_source(
                layer,
                expert,
                component_offsets,
                slot_bytes=slot_bytes,
                max_read_chunk_bytes=max_read_chunk_bytes,
            )
    expected = len(artifact.layers) * artifact.expert_count
    if len(sources) != expected:
        raise Glm52Q1TFusedRansConstructionError(
            "compressed rANS cache source coverage is incomplete"
        )
    return sources


class _Glm52Q1TRansCacheReader:
    """Exact positional copies from component containers into compressed slots."""

    def __init__(self, path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            self._fd = os.open(path, flags)
            fcntl.fcntl(self._fd, fcntl.F_NOCACHE, 1)
        except OSError as exc:
            try:
                os.close(self._fd)
            except (AttributeError, OSError):
                pass
            raise Glm52Q1TFusedRansConstructionError(
                f"cannot open fused-rANS cache source: {exc}"
            ) from exc

    def read_into(
        self,
        source: _Glm52Q1TRansCacheExpertSource,
        destination: memoryview,
    ) -> None:
        destination[: len(source.header)] = source.header
        for copy in source.copies:
            target = destination[
                copy.destination_offset : copy.destination_offset + copy.length
            ]
            received = os.preadv(self._fd, [target], copy.source_offset)
            if received != copy.length:
                raise OSError(
                    f"short fused-rANS cache read at {copy.source_offset}: "
                    f"{received} != {copy.length}"
                )

    def close(self) -> None:
        fd = self._fd
        self._fd = -1
        if fd >= 0:
            os.close(fd)


class Glm52Q1TFusedRansCacheStore:
    """The cache72 physical owner; every resident byte remains rANS-coded."""

    def __init__(
        self,
        artifact: Glm52Q1TFusedRansManifest,
        *,
        plan: Any,
        projection_threadgroups: Mapping[str, int],
        max_read_chunk_bytes: int,
    ) -> None:
        self.artifact = artifact
        self.plan = plan
        self.projection_threadgroups = MappingProxyType(
            _normalize_projection_threadgroups(projection_threadgroups)
        )
        self.slot_bytes = GLM52_EXPERT_Q1T.expert_record_bytes
        self.persistent_slots_per_layer = int(plan.slots_per_layer)
        self.transient_slots = int(plan.transient_slots)
        self.compressed_rans_persistent_cache_bytes = int(plan.persistent_cache_bytes)
        self.compressed_rans_transient_bytes = int(plan.transient_bytes)
        self.compressed_rans_allocated_bytes = (
            self.compressed_rans_persistent_cache_bytes
            + self.compressed_rans_transient_bytes
        )
        self.decoded_expert_cache_bytes = 0
        self.source_artifact_bytes = artifact.file_bytes
        self.table_bytes = len(artifact.layers) * 3 * M * 4
        self.metal_buffer_count = 0
        self.metal_slot_view_count = 0
        self.max_cache_image_bytes = 0
        self._max_read_chunk_bytes = int(max_read_chunk_bytes)
        self._sources: dict[tuple[int, int], _Glm52Q1TRansCacheExpertSource] = {}
        self._reader: _Glm52Q1TRansCacheReader | None = None
        self._metal_banks: list[mx.array] = []
        self._host_banks: list[memoryview] = []
        self._persistent_host: dict[int, tuple[memoryview, ...]] = {}
        self._expert_slot_tables: dict[int, memoryview] = {}
        self._transient_host: tuple[memoryview, ...] = ()
        self._layer_routes: dict[int, Callable[..., mx.array]] = {}

    def _host_slot_views(
        self,
        host: memoryview,
        count: int,
    ) -> tuple[memoryview, ...]:
        return tuple(
            host[slot * self.slot_bytes : (slot + 1) * self.slot_bytes]
            for slot in range(count)
        )

    def prepare(self) -> None:
        if self._layer_routes:
            return
        path = self.artifact.bin_path()
        prepared_by_layer: dict[int, tuple[_PreparedFusedRansComponent, ...]] = {}
        for layer in self.artifact.layers:
            prepared_by_layer[layer.layer] = tuple(
                _prepare_fused_rans_component(
                    path,
                    component,
                    expert_count=self.artifact.expert_count,
                    require_uniform=component.component.endswith(".packed"),
                )
                for component in layer.components
            )
        sources = _glm52_q1t_rans_cache_sources(
            self.artifact,
            slot_bytes=self.slot_bytes,
            max_read_chunk_bytes=self._max_read_chunk_bytes,
        )
        self.max_cache_image_bytes = max(
            source.image_bytes for source in sources.values()
        )
        if (
            self.max_cache_image_bytes
            > self.slot_bytes - _CACHED_EXPERT_ROUTE_TABLE_BYTES
        ):
            raise Glm52Q1TFusedRansConstructionError(
                "compressed cache images leave no bounded routed-slot table"
            )
        persistent_metal: dict[int, mx.array] = {}
        for layer in self.artifact.routed_layers:
            base = allocate_metal_u8(self.persistent_slots_per_layer * self.slot_bytes)
            host = memoryview(base).cast("B")
            host_slots = self._host_slot_views(
                host,
                self.persistent_slots_per_layer,
            )
            self._metal_banks.append(base)
            self._host_banks.append(host)
            persistent_metal[layer] = base
            self._persistent_host[layer] = host_slots
            self._expert_slot_tables[layer] = host[
                -_CACHED_EXPERT_ROUTE_TABLE_BYTES:
            ].cast("I")
        transient_base = allocate_metal_u8(self.transient_slots * self.slot_bytes)
        transient_host = memoryview(transient_base).cast("B")
        self._transient_host = self._host_slot_views(
            transient_host,
            self.transient_slots,
        )
        self._metal_banks.append(transient_base)
        self._host_banks.append(transient_host)
        self.metal_buffer_count = len(self._metal_banks)
        self.metal_slot_view_count = 0
        for layer in self.artifact.routed_layers:
            prepared = prepared_by_layer[layer]
            gate_transition = prepared[1].transition
            up_transition = prepared[3].transition
            down_transition = prepared[5].transition
            self._layer_routes[layer] = bind_glm52_q1t_fused_rans_cached_bank(
                persistent_bytes=persistent_metal[layer],
                transient_bytes=transient_base,
                slot_bytes=self.slot_bytes,
                persistent_slots=self.persistent_slots_per_layer,
                gate_scales_transition=gate_transition,
                up_scales_transition=up_transition,
                down_scales_transition=down_transition,
                hidden_size=GLM52_EXPERT_Q1T.hidden_size,
                expert_hidden_size=GLM52_EXPERT_Q1T.expert_hidden_size,
                gate_up_threads_per_tg=_QUALIFIED_GATE_UP_THREADGROUPS,
                down_threads_per_tg=self.projection_threadgroups["down_proj"],
                dtype=mx.bfloat16,
            )
        self._sources = sources
        self._reader = _Glm52Q1TRansCacheReader(path)

    def load(self, layer: int, expert: int, logical_slot: int) -> None:
        host = (
            self._persistent_host[layer][logical_slot]
            if logical_slot < self.persistent_slots_per_layer
            else self._transient_host[logical_slot - self.persistent_slots_per_layer]
        )
        self._reader.read_into(self._sources[(layer, expert)], host)

    def install_route(
        self,
        layer: int,
        experts: tuple[int, ...],
        slots: tuple[int, ...],
    ) -> None:
        table = self._expert_slot_tables[layer]
        for expert, logical_slot in zip(experts, slots, strict=True):
            table[expert] = logical_slot

    def route_for_layer(self, layer: int) -> Callable[..., mx.array]:
        return self._layer_routes[layer]

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._layer_routes.clear()
        self._sources.clear()
        self._persistent_host.clear()
        self._transient_host = ()
        for table in self._expert_slot_tables.values():
            table.release()
        self._expert_slot_tables.clear()
        for host in self._host_banks:
            host.release()
        self._host_banks.clear()
        self._metal_banks.clear()
        gc.collect()


@dataclass(frozen=True)
class _Glm52Q1TCacheLoad:
    expert: int
    slot: int


@dataclass(frozen=True)
class _Glm52Q1TCachePlan:
    experts: tuple[int, ...]
    slots: tuple[int, ...]
    hits: tuple[int, ...]
    misses: tuple[int, ...]
    loads: tuple[_Glm52Q1TCacheLoad, ...]


@dataclass
class _Glm52Q1TCacheHistory:
    score: float = 0.0
    score_epoch: int = 0
    last_used: int = -1


class _Glm52Q1TFrequencyCache:
    """Construction-fixed cache72 policy without hot-path instrumentation."""

    def __init__(self, *, persistent_slots: int, transient_slots: int) -> None:
        self.persistent_slots = int(persistent_slots)
        self.transient_slots = int(transient_slots)
        self.frequency_decay = 0.995
        self._decode_epoch = 0
        self._slot_to_expert: list[int | None] = [None] * self.persistent_slots
        self._expert_to_slot: dict[int, int] = {}
        self._history = [
            _Glm52Q1TCacheHistory() for _expert in range(GLM52_EXPERT_Q1T.expert_count)
        ]
        self._prefill_seed_candidates: set[int] = set()

    def reset(self) -> None:
        self._decode_epoch = 0
        self._slot_to_expert = [None] * self.persistent_slots
        self._expert_to_slot.clear()
        self._history = [
            _Glm52Q1TCacheHistory() for _expert in range(GLM52_EXPERT_Q1T.expert_count)
        ]
        self._prefill_seed_candidates.clear()

    def prepare_prefill(self, experts: tuple[int, ...]) -> None:
        empty = self.persistent_slots - len(self._expert_to_slot)
        counts = Counter(experts)
        ranked = sorted(counts, key=lambda expert: (-counts[expert], expert))
        chosen = tuple(
            expert for expert in ranked if expert not in self._expert_to_slot
        )[:empty]
        self._prefill_seed_candidates = set(chosen)

    def _score(self, expert: int) -> float:
        history = self._history[expert]
        age = self._decode_epoch - history.score_epoch
        return (
            history.score
            if age <= 0 or history.score == 0.0
            else history.score * (self.frequency_decay**age)
        )

    def _empty_slot(self) -> int | None:
        return next(
            (
                slot
                for slot, expert in enumerate(self._slot_to_expert)
                if expert is None
            ),
            None,
        )

    def _victim_slot(self, pinned: set[int]) -> int | None:
        candidates = [
            (self._score(expert), self._history[expert].last_used, slot)
            for slot, expert in enumerate(self._slot_to_expert)
            if expert is not None and expert not in pinned
        ]
        return min(candidates)[2] if candidates else None

    def plan(
        self,
        experts: tuple[int, ...],
        *,
        decode: bool,
    ) -> _Glm52Q1TCachePlan:
        unique_experts = tuple(dict.fromkeys(experts))
        if decode:
            self._decode_epoch += 1
            for expert in experts:
                history = self._history[expert]
                history.score = self._score(expert) + 1.0
                history.score_epoch = self._decode_epoch
                history.last_used = self._decode_epoch
        hit_set = {
            expert for expert in unique_experts if expert in self._expert_to_slot
        }
        misses = tuple(expert for expert in unique_experts if expert not in hit_set)
        resolved = {expert: self._expert_to_slot[expert] for expert in hit_set}
        loads: list[_Glm52Q1TCacheLoad] = []
        pinned = set(hit_set)
        transient: list[int] = []
        for expert in misses:
            persistent_slot = None
            if not decode and expert in self._prefill_seed_candidates:
                persistent_slot = self._empty_slot()
                self._prefill_seed_candidates.discard(expert)
            elif decode and self.persistent_slots:
                persistent_slot = self._empty_slot()
                if persistent_slot is None:
                    victim_slot = self._victim_slot(pinned)
                    if victim_slot is not None:
                        victim = self._slot_to_expert[victim_slot]
                        if self._score(expert) > max(1.0, self._score(victim)):
                            persistent_slot = victim_slot
            if persistent_slot is None:
                transient.append(expert)
                continue
            previous = self._slot_to_expert[persistent_slot]
            if previous is not None:
                del self._expert_to_slot[previous]
            self._slot_to_expert[persistent_slot] = expert
            self._expert_to_slot[expert] = persistent_slot
            pinned.add(expert)
            resolved[expert] = persistent_slot
            loads.append(_Glm52Q1TCacheLoad(expert, persistent_slot))
        for offset, expert in enumerate(transient):
            slot = self.persistent_slots + offset
            resolved[expert] = slot
            loads.append(_Glm52Q1TCacheLoad(expert, slot))
        return _Glm52Q1TCachePlan(
            experts=experts,
            slots=tuple(resolved[expert] for expert in experts),
            hits=tuple(expert for expert in unique_experts if expert in hit_set),
            misses=misses,
            loads=tuple(loads),
        )


@dataclass(frozen=True)
class Glm52Q1TFusedRansMlpRoute:
    """Three direct fused projection dispatches for one GLM routed layer."""

    gate: Callable[[mx.array, mx.array], mx.array]
    up: Callable[[mx.array, mx.array], mx.array]
    down: Callable[[mx.array, mx.array], mx.array]

    def __call__(self, x: mx.array, expert_ids: mx.array) -> mx.array:
        gate = self.gate(x, expert_ids)
        up = self.up(x, expert_ids)
        return self.down(swiglu(gate, up), expert_ids)


@dataclass(frozen=True)
class Glm52Q1TFusedRansGateUpDownRoute:
    """Two direct dispatches: exact fused gate/up/SwiGLU, then down."""

    gate_up: Callable[[mx.array, mx.array], mx.array]
    down: Callable[[mx.array, mx.array], mx.array]

    def __call__(self, x: mx.array, expert_ids: mx.array) -> mx.array:
        return self.down(self.gate_up(x, expert_ids), expert_ids)


@dataclass(frozen=True)
class Glm52Q1TFusedRansExpertBankRoute:
    """Dispatch only the construction-bound compressed experts selected now."""

    experts: tuple[Callable[[mx.array, mx.array], mx.array], ...]
    hidden_size: int
    top_k: int

    def __call__(self, x: mx.array, expert_ids: mx.array) -> mx.array:
        assignments = mx.broadcast_to(
            x.reshape(-1, 1, self.hidden_size),
            (int(x.shape[0]), self.top_k, self.hidden_size),
        ).reshape(-1, self.hidden_size)
        positions_by_expert: dict[int, list[int]] = {}
        for position, expert in enumerate(expert_ids.tolist()):
            positions_by_expert.setdefault(int(expert), []).append(position)
        grouped_positions: list[int] = []
        grouped_outputs: list[mx.array] = []
        for expert, positions in positions_by_expert.items():
            position_array = mx.array(positions, dtype=mx.uint32)
            grouped_positions.extend(positions)
            grouped_outputs.append(
                self.experts[expert](
                    mx.take(assignments, position_array, axis=0),
                    mx.take(expert_ids, position_array, axis=0),
                )
            )
        grouped = mx.concatenate(grouped_outputs, axis=0)
        inverse = [0] * len(grouped_positions)
        for grouped_position, original_position in enumerate(grouped_positions):
            inverse[original_position] = grouped_position
        return mx.take(grouped, mx.array(inverse, dtype=mx.uint32), axis=0)


class Glm52Q1TFusedRansSwitchGLU(nn.Module):
    """Installed GLM switch with one construction-selected direct route."""

    def __init__(
        self,
        *,
        route: Glm52Q1TFusedRansMlpRoute,
        hidden_size: int,
        top_k: int,
    ) -> None:
        super().__init__()
        if int(hidden_size) <= 0 or int(top_k) <= 0:
            raise Glm52Q1TFusedRansConstructionError(
                "fused-rANS hidden size and top-k must be positive"
            )
        self.route = route
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        tokens = x.reshape(-1, self.hidden_size)
        expert_ids = indices.reshape(-1)
        output = self.route(tokens, expert_ids)
        return output.reshape((*indices.shape, self.hidden_size))


class Glm52Q1TFusedRansBandEndSwitchGLU(Glm52Q1TFusedRansSwitchGLU):
    """Evaluate one fixed band so external residency cannot span later bands."""

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        output = super().__call__(x, indices)
        mx.eval(output)
        return output


class Glm52Q1TFusedRansCachedSwitchGLU(nn.Module):
    """The construction-installed cache72 route with one direct entrypoint."""

    def __init__(self, runtime: Glm52Q1TFusedRansRuntime, layer: int) -> None:
        super().__init__()
        self.runtime = runtime
        self.layer = int(layer)

    def _route_inputs(
        self,
        x: mx.array,
        indices: mx.array,
    ) -> tuple[mx.array, mx.array, tuple[int, ...], bool]:
        mx.eval(indices)
        routed_ids = indices.reshape(-1)
        expert_ids = tuple(int(value) for value in routed_ids.tolist())
        tokens = x.reshape(-1, self.runtime.spec.hidden_size)
        assignments = mx.broadcast_to(
            tokens.reshape(-1, 1, self.runtime.spec.hidden_size),
            (
                int(tokens.shape[0]),
                self.runtime.spec.top_k,
                self.runtime.spec.hidden_size,
            ),
        ).reshape(-1, self.runtime.spec.hidden_size)
        decode = (
            current_expert_routing_phase(token_count=int(x.shape[-2]))
            is RoutingPhase.DECODE
        )
        return assignments, routed_ids, expert_ids, decode

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        assignments, routed_ids, expert_ids, decode = self._route_inputs(x, indices)
        output = self.runtime.execute_layer(
            self.layer,
            assignments,
            routed_ids,
            expert_ids,
            decode=decode,
        )
        return output.reshape((*indices.shape, self.runtime.spec.hidden_size))

    def run_with_shared_overlap(
        self,
        x: mx.array,
        indices: mx.array,
        shared_work: Callable[[], mx.array],
    ) -> tuple[mx.array, mx.array]:
        assignments, routed_ids, expert_ids, decode = self._route_inputs(x, indices)
        output, shared = self.runtime.execute_layer_with_shared(
            self.layer,
            assignments,
            routed_ids,
            expert_ids,
            decode=decode,
            shared_work=shared_work,
        )
        return (
            output.reshape((*indices.shape, self.runtime.spec.hidden_size)),
            shared,
        )


@dataclass(frozen=True)
class _Glm52Q1TFusedRansKVAdmission:
    def __enter__(self) -> _Glm52Q1TFusedRansKVAdmission:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


_GLM52_Q1T_FUSED_RANS_KV_ADMISSION = _Glm52Q1TFusedRansKVAdmission()


class Glm52Q1TFusedRansRuntime:
    """Cache72 runtime whose only weight representation is compressed rANS."""

    def __init__(
        self,
        *,
        manifest: Any,
        fused_manifest: Glm52Q1TFusedRansManifest,
        config: Any,
        plan: Any,
        store: Glm52Q1TFusedRansCacheStore,
        integrity_seconds: float,
        self_check_seconds: float = 0.0,
    ) -> None:
        self.spec = GLM52_EXPERT_Q1T
        self.manifest = manifest
        self.fused_manifest = fused_manifest
        self.config = config
        self.store = store
        self.plan = plan
        self.island_layer_set = frozenset()
        self.integrity_seconds = float(integrity_seconds)
        self.self_check_seconds = float(self_check_seconds)
        self.qualification_receipt_sha256 = ""
        self.memory_cap_report: dict[str, int] = {}
        self._switches = {
            layer: Glm52Q1TFusedRansCachedSwitchGLU(self, layer)
            for layer in self.spec.routed_layer_indices
        }
        self._policies = {
            layer: _Glm52Q1TFrequencyCache(
                persistent_slots=self.plan.slots_per_layer,
                transient_slots=self.plan.transient_slots,
            )
            for layer in self.spec.routed_layer_indices
        }
        self._io = ThreadPoolExecutor(
            max_workers=self.plan.transient_slots,
            thread_name_prefix="mtplx-glm52-rans-cache",
        )
        self._async_eval = mx.async_eval
        self._closed = False

    def switch_for_layer(self, layer: int) -> Glm52Q1TFusedRansCachedSwitchGLU:
        try:
            return self._switches[int(layer)]
        except KeyError:
            raise Glm52Q1TFusedRansConstructionError(
                f"fused-rANS runtime has no routed layer {layer}"
            ) from None

    @staticmethod
    def _waves(
        experts: tuple[int, ...],
        *,
        capacity: int,
        decode: bool,
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        unique = tuple(dict.fromkeys(experts))
        ordered = unique if decode else tuple(sorted(unique))
        waves = []
        for start in range(0, len(ordered), capacity):
            selected = set(ordered[start : start + capacity])
            positions = tuple(
                position
                for position, expert in enumerate(experts)
                if expert in selected
            )
            waves.append(
                (positions, tuple(experts[position] for position in positions))
            )
        return tuple(waves)

    def _dispatch_assignments(
        self,
        layer: int,
        assignments: mx.array,
        routed_ids: mx.array,
        positions: tuple[int, ...],
        plan: _Glm52Q1TCachePlan,
        selected_experts: set[int],
    ) -> tuple[list[mx.array], list[int]]:
        local_positions = tuple(
            index
            for index, expert in enumerate(plan.experts)
            if expert in selected_experts
        )
        if not local_positions:
            return [], []
        global_positions = tuple(positions[index] for index in local_positions)
        if global_positions == tuple(range(int(assignments.shape[0]))):
            selected = assignments
            expert_ids = routed_ids
        else:
            position_array = mx.array(global_positions, dtype=mx.uint32)
            selected = mx.take(assignments, position_array, axis=0)
            expert_ids = mx.take(routed_ids, position_array, axis=0)
        output = self.store.route_for_layer(layer)(
            selected,
            expert_ids,
        )
        return [output], list(global_positions)

    @staticmethod
    def _no_overlap(_futures: tuple[Any, ...]) -> None:
        return None

    def _execute_layer(
        self,
        layer: int,
        assignments: mx.array,
        routed_ids: mx.array,
        experts: tuple[int, ...],
        *,
        decode: bool,
        overlap: Callable[[tuple[Any, ...]], None],
    ) -> mx.array:
        policy = self._policies[layer]
        if not decode:
            policy.prepare_prefill(experts)
        waves = self._waves(
            experts,
            capacity=self.plan.transient_slots,
            decode=decode,
        )
        outputs: list[mx.array] = []
        output_positions: list[int] = []
        for wave_index, (positions, wave_experts) in enumerate(waves):
            plan = policy.plan(wave_experts, decode=decode)
            self.store.install_route(layer, plan.experts, plan.slots)
            pending_loads = {
                self._io.submit(
                    self.store.load,
                    layer,
                    load.expert,
                    load.slot,
                ): load
                for load in plan.loads
            }
            futures = tuple(pending_loads)
            hit_outputs, hit_positions = self._dispatch_assignments(
                layer,
                assignments,
                routed_ids,
                positions,
                plan,
                set(plan.hits),
            )
            if hit_outputs:
                self._async_eval(hit_outputs)
            overlap(futures)
            if len(waves) == 1 and not futures:
                return hit_outputs[0]
            wave_outputs = [*hit_outputs]
            wave_positions = [*hit_positions]
            for future in as_completed(futures):
                future.result()
                ready_load = pending_loads[future]
                ready_outputs, ready_positions = self._dispatch_assignments(
                    layer,
                    assignments,
                    routed_ids,
                    positions,
                    plan,
                    {ready_load.expert},
                )
                self._async_eval(ready_outputs)
                wave_outputs.extend(ready_outputs)
                wave_positions.extend(ready_positions)
            outputs.extend(wave_outputs)
            output_positions.extend(wave_positions)
            if wave_index + 1 < len(waves):
                mx.eval(wave_outputs)
        grouped = mx.concatenate(outputs, axis=0)
        inverse = [0] * len(output_positions)
        for grouped_position, original_position in enumerate(output_positions):
            inverse[original_position] = grouped_position
        return mx.take(grouped, mx.array(inverse, dtype=mx.uint32), axis=0)

    def execute_layer(
        self,
        layer: int,
        assignments: mx.array,
        routed_ids: mx.array,
        experts: tuple[int, ...],
        *,
        decode: bool,
    ) -> mx.array:
        return self._execute_layer(
            layer,
            assignments,
            routed_ids,
            experts,
            decode=decode,
            overlap=self._no_overlap,
        )

    def execute_layer_with_shared(
        self,
        layer: int,
        assignments: mx.array,
        routed_ids: mx.array,
        experts: tuple[int, ...],
        *,
        decode: bool,
        shared_work: Callable[[], mx.array],
    ) -> tuple[mx.array, mx.array]:
        shared = None

        def overlap(futures: tuple[Any, ...]) -> None:
            nonlocal shared
            if decode and futures and shared is None:
                shared = shared_work()
                self._async_eval(shared)

        output = self._execute_layer(
            layer,
            assignments,
            routed_ids,
            experts,
            decode=decode,
            overlap=overlap,
        )
        if shared is None:
            shared = shared_work()
        return output, shared

    def admit_kv_tokens(self, tokens: int) -> _Glm52Q1TFusedRansKVAdmission:
        del tokens
        return _GLM52_Q1T_FUSED_RANS_KV_ADMISSION

    def snapshot(self, *, mx_module: Any | None = None) -> dict[str, Any]:
        del mx_module
        return {
            "model_key": self.spec.key,
            "expert_codec": self.fused_manifest.codec,
            "manifest_sha256": self.manifest.manifest_sha256,
            "fused_manifest_sha256": self.fused_manifest.file_sha256,
            "source_compressed_bytes": self.store.source_artifact_bytes,
            "table_bytes": self.store.table_bytes,
            "projection_threadgroups": dict(self.store.projection_threadgroups),
            "gate_up_threadgroups": _QUALIFIED_GATE_UP_THREADGROUPS,
            "compressed_rans_persistent_cache_bytes": (
                self.store.compressed_rans_persistent_cache_bytes
            ),
            "compressed_rans_transient_bytes": (
                self.store.compressed_rans_transient_bytes
            ),
            "compressed_rans_allocated_bytes": (
                self.store.compressed_rans_allocated_bytes
            ),
            "decoded_expert_cache_bytes": 0,
            "persistent_slots_per_layer": self.store.persistent_slots_per_layer,
            "transient_slots": self.store.transient_slots,
            "metal_buffer_count": self.store.metal_buffer_count,
            "metal_slot_view_count": self.store.metal_slot_view_count,
            "max_cache_image_bytes": self.store.max_cache_image_bytes,
            "memory_caps": dict(self.memory_cap_report),
            "memory_plan": {
                "decoded_expert_cache_bytes": 0,
                "compressed_rans_persistent_cache_bytes": (
                    self.plan.persistent_cache_bytes
                ),
                "compressed_rans_transient_bytes": self.plan.transient_bytes,
                "persistent_cache_bytes": self.plan.persistent_cache_bytes,
                "slots_per_layer": self.plan.slots_per_layer,
                "transient_slots": self.plan.transient_slots,
                "allocated_bytes": self.plan.allocated_bytes,
                "unallocated_bytes": self.plan.unallocated_bytes,
            },
            "construction": {
                "integrity_seconds": self.integrity_seconds,
                "self_check_seconds": self.self_check_seconds,
                "qualification_receipt_sha256": self.qualification_receipt_sha256,
            },
        }

    def resource_telemetry_snapshot(
        self, *, mx_module: Any | None = None
    ) -> dict[str, Any]:
        del mx_module
        return {
            "model_key": self.spec.key,
            "source_compressed_bytes": self.store.source_artifact_bytes,
            "table_bytes": self.store.table_bytes,
            "decoded_expert_cache_bytes": 0,
            "compressed_rans_allocated_bytes": (
                self.store.compressed_rans_allocated_bytes
            ),
        }

    def reset(self) -> None:
        for policy in self._policies.values():
            policy.reset()

    def close(self, *, timeout: float | None = None) -> None:
        del timeout
        if self._closed:
            return
        self._closed = True
        self._io.shutdown(wait=True, cancel_futures=True)
        self._switches.clear()
        self._policies.clear()
        self.store.close()


def _self_check_kernel_sha256() -> str:
    from mtplx.kernels import glm52_q1t_fused_rans as kernel_module

    return hashlib.sha256(Path(kernel_module.__file__).read_bytes()).hexdigest()


def _load_self_check_receipt(runtime: Glm52Q1TFusedRansRuntime) -> dict[str, Any]:
    manifest_path = runtime.fused_manifest.path
    if manifest_path is None:
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS self-check has no artifact manifest location"
        )
    path = manifest_path.with_name(manifest_path.stem + _SELF_CHECK_SUFFIX)
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise Glm52Q1TFusedRansConstructionError(
            f"cannot read fused-rANS self-check receipt: {exc}"
        ) from exc
    if not isinstance(receipt, dict):
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS self-check receipt must be an object"
        )
    expected_fields = {
        "format",
        "model_key",
        "artifact_sha256",
        "source_manifest_sha256",
        "kernel_sha256",
        "route_kind",
        "qualified_report_sha256",
        "qualified_assignment_counts",
        "launch_threadgroups",
        "vectors",
        "receipt_sha256",
    }
    if set(receipt) != expected_fields:
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS self-check receipt schema is incomplete"
        )
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if receipt["receipt_sha256"] != digest:
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS self-check receipt digest mismatch"
        )
    if (
        receipt["format"] != _SELF_CHECK_FORMAT
        or receipt["model_key"] != FUSED_RANS_MODEL_KEY
        or receipt["artifact_sha256"] != runtime.fused_manifest.file_sha256
        or receipt["source_manifest_sha256"]
        != runtime.fused_manifest.source_manifest_sha256
        or receipt["kernel_sha256"] != _self_check_kernel_sha256()
        or receipt["route_kind"] != "cached-rans-t158-routed-bank"
        or receipt["qualified_report_sha256"] != _QUALIFIED_REAL_SHAPE_REPORT_SHA256
        or tuple(receipt["qualified_assignment_counts"]) != _QUALIFIED_ASSIGNMENT_COUNTS
        or receipt["launch_threadgroups"]
        != {
            "gate_up": _QUALIFIED_GATE_UP_THREADGROUPS,
            "down": runtime.store.projection_threadgroups["down_proj"],
        }
    ):
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS self-check identity is incompatible"
        )
    vectors = receipt["vectors"]
    if (
        not isinstance(vectors, list)
        or tuple(vector.get("layer") for vector in vectors if isinstance(vector, dict))
        != runtime.spec.routed_layer_indices
    ):
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS self-check layer coverage is incomplete"
        )
    return receipt


def run_glm52_q1t_fused_rans_self_checks(
    runtime: Glm52Q1TFusedRansRuntime,
) -> float:
    """Run one exact final-output vector through every installed layer route."""

    started = time.perf_counter()
    receipt = _load_self_check_receipt(runtime)
    runtime.qualification_receipt_sha256 = str(receipt["receipt_sha256"])
    for vector in receipt["vectors"]:
        if not isinstance(vector, dict) or set(vector) != {
            "layer",
            "seed",
            "expert_ids",
            "output_sha256",
        }:
            raise Glm52Q1TFusedRansConstructionError(
                "fused-rANS self-check vector schema is invalid"
            )
        layer = int(vector["layer"])
        seed = int(vector["seed"])
        expert_ids = tuple(int(value) for value in vector["expert_ids"])
        if len(expert_ids) != runtime.spec.top_k or any(
            value < 0 or value >= runtime.spec.expert_count for value in expert_ids
        ):
            raise Glm52Q1TFusedRansConstructionError(
                f"fused-rANS self-check expert IDs are invalid for layer {layer}"
            )
        rng = np.random.default_rng(seed)
        host = rng.standard_normal((1, 1, runtime.spec.hidden_size), dtype=np.float32)
        x = mx.array(host, dtype=mx.bfloat16)
        indices = mx.array(expert_ids, dtype=mx.int32).reshape(1, 1, -1)
        output = runtime.switch_for_layer(layer)(x, indices)
        mx.eval(output)
        output_bits = np.asarray(mx.view(output, mx.uint16))
        digest = hashlib.sha256(output_bits.tobytes()).hexdigest()
        if digest != vector["output_sha256"]:
            raise Glm52Q1TFusedRansConstructionError(
                f"fused-rANS exact self-check failed for layer {layer}"
            )
    return time.perf_counter() - started


def construct_glm52_q1t_fused_rans_runtime(
    *,
    base_manifest: Any,
    fused_manifest: Glm52Q1TFusedRansManifest,
    config: Any,
    cache_plan: Any | None = None,
) -> Glm52Q1TFusedRansRuntime:
    """Validate, allocate cache72, and return the direct GLM-only route."""

    if (
        getattr(base_manifest, "model_key", None) != FUSED_RANS_MODEL_KEY
        or getattr(base_manifest, "manifest_sha256", None)
        != GLM52_Q1T_BASE_MANIFEST_SHA256
    ):
        raise Glm52Q1TFusedRansConstructionError(
            "authoritative expert manifest is not glm52-expert-q1t t158 g64"
        )
    if fused_manifest.source_manifest_sha256 != base_manifest.manifest_sha256:
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS source does not match the authoritative manifest"
    )
    validate_glm52_q1t_fused_rans_manifest(fused_manifest)
    projection_threadgroups = dict(_QUALIFIED_PROJECTION_THREADGROUPS)
    plan = (
        config.memory_plan(GLM52_EXPERT_Q1T)
        if cache_plan is None
        else cache_plan
    )
    if (
        getattr(plan, "slots_per_layer", None)
        != GLM52_Q1T_PERSISTENT_SLOTS_PER_LAYER
        or getattr(plan, "transient_slots", None) != GLM52_Q1T_TRANSIENT_SLOTS
    ):
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS cache plan is not the exact "
            "116-persistent/48-transient control geometry"
        )
    integrity_seconds = verify_glm52_q1t_fused_rans_artifact(fused_manifest)
    store = Glm52Q1TFusedRansCacheStore(
        fused_manifest,
        plan=plan,
        projection_threadgroups=projection_threadgroups,
        max_read_chunk_bytes=config.max_read_chunk_bytes,
    )
    try:
        store.prepare()
        runtime = Glm52Q1TFusedRansRuntime(
            manifest=base_manifest,
            fused_manifest=fused_manifest,
            config=config,
            plan=plan,
            store=store,
            integrity_seconds=integrity_seconds,
        )
        runtime.self_check_seconds = run_glm52_q1t_fused_rans_self_checks(runtime)
        return runtime
    except BaseException:
        store.close()
        raise


def bind_glm52_q1t_fused_rans_switches(model: Any, runtime: Any) -> int:
    """Install the already-qualified fused switch into every routed GLM layer."""

    spec = getattr(runtime, "spec", None)
    if getattr(spec, "key", None) != GLM52_EXPERT_Q1T.key:
        raise Glm52Q1TFusedRansConstructionError(
            "fused-rANS switches install only for glm52-expert-q1t"
        )
    routed_layers = tuple(getattr(spec, "routed_layer_indices", ()))
    if routed_layers != GLM52_EXPERT_Q1T.routed_layer_indices:
        raise Glm52Q1TFusedRansConstructionError(
            "glm52-expert-q1t routed layer geometry is incompatible"
        )
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise Glm52Q1TFusedRansConstructionError(
            "GLM model does not expose transformer layers"
        )
    switches: list[tuple[Any, Any]] = []
    for layer_index in routed_layers:
        try:
            mlp = layers[layer_index].mlp
        except (AttributeError, IndexError) as exc:
            raise Glm52Q1TFusedRansConstructionError(
                f"GLM routed layer {layer_index} has no MoE binding seam"
            ) from exc
        switch = runtime.switch_for_layer(layer_index)
        if switch is None:
            raise Glm52Q1TFusedRansConstructionError(
                f"GLM routed layer {layer_index} has no fused-rANS switch"
            )
        switches.append((mlp, switch))
    for mlp, switch in switches:
        mlp.switch_mlp = switch
    return len(switches)
