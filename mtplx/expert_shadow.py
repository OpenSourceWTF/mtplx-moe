"""Shadow expert banks: low-precision full-expert miss fallback (issue #51).

Streamed (non-island, non-mmap-band) sparse layers keep every expert
resident in a low-precision "shadow" bank so a decode cache miss can be
served instantly instead of waiting on SSD. Shadows are a quality tier:
outputs are close to the exact Q2 path, not bitwise — the benchmark's
ar_comparison / token-divergence diagnostics are the quality report.

Two codecs, both grouped along the input axis with group size 64:

``b1``
    Per-group sign bits packed into u32 (64 signs = 2 u32 per group)
    plus one bf16 scale per group (mean |w| over the group, computed
    from the DEQUANTIZED Q2 values). 10 bytes per 64 weights.

``t158``
    Ternary weight network: threshold 0.7 * mean|w| per group, scale =
    mean |w| over the nonzero elements (mean |w| over the group when the
    group quantizes to all zeros). Trits packed 5 per byte, base-3, 13
    bytes per group so every group stays byte-aligned and independently
    addressable, plus one bf16 scale. 15 bytes per 64 weights.

Encoders are numpy and run at load time from the expert sidecar records
(dynamic — no artifact rebuild). Banks are dense per layer: row index is
the expert id (island-bank convention), component-major (gate/up/down).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mtplx.expert_manifest import ExpertManifest
    from mtplx.expert_streaming_models import ExpertMemoryPlan

_LOGGER = logging.getLogger(__name__)

SHADOW_CODECS = ("b1", "t158")
SHADOW_GROUP = 64
_B1_WORDS_PER_GROUP = SHADOW_GROUP // 32  # 2 u32 sign words
_T158_BYTES_PER_GROUP = 13  # 5 trits/byte, 65 slots, last slot unused
_T158_POW3 = (1, 3, 9, 27, 81)


class ShadowCodecError(ValueError):
    """Raised when shadow codec inputs are malformed."""


def normalize_shadow_codec(value: object | None) -> str | None:
    """Canonicalize a --miss-shadow style value; None/off disable."""

    if value is None:
        return None
    raw = str(value).strip().lower().replace("-", "_")
    if raw in ("", "none", "off", "false", "0", "disabled", "disable"):
        return None
    if raw in SHADOW_CODECS:
        return raw
    choices = ", ".join(SHADOW_CODECS)
    raise ShadowCodecError(
        f"unsupported miss-shadow codec {value!r}; expected one of: {choices}"
    )


def shadow_bytes_per_weight(codec: str) -> float:
    """Shadow bank bytes per source weight for ``codec`` (g64 groups)."""

    if codec == "b1":
        return (4 * _B1_WORDS_PER_GROUP + 2) / SHADOW_GROUP  # 10 / 64
    if codec == "t158":
        return (_T158_BYTES_PER_GROUP + 2) / SHADOW_GROUP  # 15 / 64
    raise ShadowCodecError(f"unknown shadow codec {codec!r}")


def _require_grouped(w: np.ndarray) -> tuple[int, int, int]:
    if w.ndim != 2:
        raise ShadowCodecError(f"shadow codecs expect 2-D weights, got {w.shape}")
    rows, cols = w.shape
    if cols % SHADOW_GROUP:
        raise ShadowCodecError(
            f"shadow codecs group along the last axis in {SHADOW_GROUP}s; "
            f"got {cols} columns"
        )
    return rows, cols, cols // SHADOW_GROUP


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 stored as u16 (round to nearest even)."""

    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + rounding) >> np.uint32(16)).astype(np.uint16)


def _from_bf16_bits(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def encode_b1(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode fp32 weights (rows, cols) -> (packed u32 (rows, cols/32),
    bf16-bit scales u16 (rows, cols/64)). Zero weights count as +."""

    rows, cols, groups = _require_grouped(w)
    grp = np.ascontiguousarray(w, dtype=np.float32).reshape(rows, groups, SHADOW_GROUP)
    scales = np.abs(grp).mean(axis=2)
    signs = (grp >= 0).astype(np.uint32)
    lanes = signs.reshape(rows, groups, _B1_WORDS_PER_GROUP, 32)
    shifts = np.arange(32, dtype=np.uint32)
    packed = (lanes << shifts).sum(axis=3, dtype=np.uint64).astype(np.uint32)
    return packed.reshape(rows, groups * _B1_WORDS_PER_GROUP), _to_bf16_bits(scales)


def decode_b1(packed: np.ndarray, scales: np.ndarray, cols: int) -> np.ndarray:
    """Inverse of :func:`encode_b1` -> fp32 (rows, cols)."""

    rows = packed.shape[0]
    groups = cols // SHADOW_GROUP
    lanes = packed.reshape(rows, groups, _B1_WORDS_PER_GROUP, 1)
    shifts = np.arange(32, dtype=np.uint32)
    bits = (lanes >> shifts) & np.uint32(1)
    sign = bits.astype(np.float32) * 2.0 - 1.0
    scale = _from_bf16_bits(scales).reshape(rows, groups, 1, 1)
    return (sign * scale).reshape(rows, cols)


def encode_t158(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode fp32 weights (rows, cols) -> (packed u8 (rows, 13*cols/64),
    bf16-bit scales u16 (rows, cols/64))."""

    rows, cols, groups = _require_grouped(w)
    grp = np.ascontiguousarray(w, dtype=np.float32).reshape(rows, groups, SHADOW_GROUP)
    mean_abs = np.abs(grp).mean(axis=2, keepdims=True)
    threshold = 0.7 * mean_abs
    q = np.zeros(grp.shape, dtype=np.uint8)  # trit = q + 1
    q[grp > threshold] = 2
    q[grp < -threshold] = 0
    q[(grp <= threshold) & (grp >= -threshold)] = 1
    nonzero = q != 1
    nz_count = nonzero.sum(axis=2)
    abs_sum = np.where(nonzero, np.abs(grp), 0.0).sum(axis=2)
    scales = np.where(
        nz_count > 0,
        abs_sum / np.maximum(nz_count, 1),
        mean_abs[..., 0],
    )
    slots = np.zeros((rows, groups, _T158_BYTES_PER_GROUP * 5), dtype=np.uint8)
    slots[:, :, :SHADOW_GROUP] = q
    lanes = slots.reshape(rows, groups, _T158_BYTES_PER_GROUP, 5)
    weights = np.asarray(_T158_POW3, dtype=np.uint16)
    packed = (lanes * weights).sum(axis=3).astype(np.uint8)
    return packed.reshape(rows, groups * _T158_BYTES_PER_GROUP), _to_bf16_bits(scales)


def decode_t158(packed: np.ndarray, scales: np.ndarray, cols: int) -> np.ndarray:
    """Inverse of :func:`encode_t158` -> fp32 (rows, cols)."""

    rows = packed.shape[0]
    groups = cols // SHADOW_GROUP
    lanes = packed.reshape(rows, groups, _T158_BYTES_PER_GROUP, 1).astype(np.uint16)
    divisors = np.asarray(_T158_POW3, dtype=np.uint16)
    trits = (lanes // divisors) % 3
    q = trits.reshape(rows, groups, _T158_BYTES_PER_GROUP * 5)
    q = q[:, :, :SHADOW_GROUP].astype(np.float32) - 1.0
    scale = _from_bf16_bits(scales).reshape(rows, groups, 1)
    return (q * scale).reshape(rows, cols)


_ENCODERS: dict[str, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]] = {
    "b1": encode_b1,
    "t158": encode_t158,
}
_DECODERS: dict[str, Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = {
    "b1": decode_b1,
    "t158": decode_t158,
}


def encode_shadow(codec: str, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        return _ENCODERS[codec](w)
    except KeyError:
        raise ShadowCodecError(f"unknown shadow codec {codec!r}") from None


def decode_shadow(
    codec: str, packed: np.ndarray, scales: np.ndarray, cols: int
) -> np.ndarray:
    try:
        return _DECODERS[codec](packed, scales, cols)
    except KeyError:
        raise ShadowCodecError(f"unknown shadow codec {codec!r}") from None


def shadow_record_bytes(codec: str, source_parameters: int) -> int:
    """Shadow bank bytes for one expert with ``source_parameters`` weights."""

    if source_parameters % SHADOW_GROUP:
        raise ShadowCodecError(
            f"expert parameter count {source_parameters} is not grouped in "
            f"{SHADOW_GROUP}s"
        )
    groups = source_parameters // SHADOW_GROUP
    if codec == "b1":
        return groups * (4 * _B1_WORDS_PER_GROUP + 2)
    if codec == "t158":
        return groups * (_T158_BYTES_PER_GROUP + 2)
    raise ShadowCodecError(f"unknown shadow codec {codec!r}")


_SHADOW_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class ShadowLayerBank:
    """Dense per-layer shadow bank: row index is the expert id.

    ``arrays`` is component-major like the island banks: for each
    projection it holds ``{proj}.packed`` (experts, out, packed-words)
    and ``{proj}.scales`` (experts, out, in/64) u16 bf16 bits.
    """

    __slots__ = ("layer", "codec", "expert_count", "arrays")

    def __init__(
        self,
        layer: int,
        codec: str,
        expert_count: int,
        arrays: dict[str, "mx.array"],
    ) -> None:
        self.layer = int(layer)
        self.codec = codec
        self.expert_count = int(expert_count)
        self.arrays = arrays


class ShadowBankStore:
    """Low-precision full-expert banks for every streamed sparse layer.

    Built dynamically at runtime open from the expert sidecar records: each
    record's quantized projections are dequantized (the exact affine math
    the runtime executes) and re-encoded with the shadow codec, one record
    at a time so peak staging memory stays bounded by a single layer bank.
    """

    def __init__(
        self,
        spec,
        layers: Iterable[int],
        *,
        codec: str,
        plan: "ExpertMemoryPlan | None" = None,
    ) -> None:
        codec = normalize_shadow_codec(codec)
        if codec is None:
            raise ShadowCodecError("shadow store requires a codec")
        self.spec = spec
        self.codec = codec
        self.layers = tuple(sorted({int(layer) for layer in layers}))
        if not self.layers:
            raise ShadowCodecError("shadow store requires at least one layer")
        self.expert_count = int(spec.expert_count)
        self.record_bytes = shadow_record_bytes(codec, spec.expert_source_parameters)
        self.shadow_bytes = len(self.layers) * self.expert_count * self.record_bytes
        if plan is not None:
            planned_codec = getattr(plan, "miss_shadow", None)
            planned_bytes = getattr(plan, "shadow_bytes", 0)
            if planned_codec != codec or planned_bytes != self.shadow_bytes:
                raise ShadowCodecError(
                    "memory plan did not price the shadow banks: plan has "
                    f"miss_shadow={planned_codec!r} shadow_bytes={planned_bytes}, "
                    f"store needs {codec!r} / {self.shadow_bytes} bytes"
                )
        self._banks: dict[int, ShadowLayerBank] = {}
        self._fill_seconds = 0.0

    @property
    def filled_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self._banks))

    def bank_for_layer(self, layer: int) -> ShadowLayerBank:
        try:
            return self._banks[int(layer)]
        except KeyError:
            raise ShadowCodecError(
                f"shadow store holds no bank for layer {layer}"
            ) from None

    def snapshot(self) -> dict:
        return {
            "codec": self.codec,
            "shadow_bytes": self.shadow_bytes,
            "record_bytes": self.record_bytes,
            "filled_layers": len(self._banks),
            "fill_seconds": self._fill_seconds,
        }

    def fill(
        self,
        manifest: "ExpertManifest",
        root: Path | str,
        *,
        verify_hash: bool = True,
    ) -> None:
        """Encode shadow banks for every configured layer, record by record."""

        import mlx.core as mx

        from mtplx.expert_manifest import read_expert_record

        started = time.perf_counter()
        records = {
            (record.layer, record.expert): record for record in manifest.records
        }
        root_path = Path(root).resolve()
        bits = int(self.spec.quant_bits)
        group_size = int(self.spec.quant_group_size)
        fd = (
            os.open(root_path / manifest.sidecar.file, os.O_RDONLY)
            if manifest.sidecar is not None
            else None
        )
        try:
            for layer in self.layers:
                staging: dict[str, np.ndarray] | None = None
                for expert in range(self.expert_count):
                    record = records.get((layer, expert))
                    if record is None:
                        raise ShadowCodecError(
                            f"manifest has no record for layer {layer} "
                            f"expert {expert}"
                        )
                    if (
                        fd is not None
                        and record.sidecar_offset is not None
                        and record.sidecar_length is not None
                    ):
                        blob = os.pread(
                            fd, record.sidecar_length, record.sidecar_offset
                        )
                        if len(blob) != record.sidecar_length:
                            raise ShadowCodecError(
                                f"short sidecar read for record ({layer}, {expert})"
                            )
                        if verify_hash:
                            if record.sha256 is None:
                                raise ShadowCodecError(
                                    f"record ({layer}, {expert}) has no hash"
                                )
                            if hashlib.sha256(blob).hexdigest() != record.sha256:
                                raise ShadowCodecError(
                                    f"record hash mismatch: ({layer}, {expert})"
                                )
                    else:
                        # No sidecar: the manifest's source-safetensors
                        # segments are the fallback, as for slot reads.
                        blob = read_expert_record(
                            manifest,
                            root_path,
                            layer,
                            expert,
                            prefer_sidecar=False,
                            verify_hash=verify_hash and record.sha256 is not None,
                        )
                    dense = self._dequantize_record(
                        mx, record, blob, bits=bits, group_size=group_size
                    )
                    if staging is None:
                        staging = self._allocate_staging(dense)
                    for projection, weights in dense.items():
                        packed, scales = encode_shadow(self.codec, weights)
                        staging[f"{projection}.packed"][expert] = packed
                        staging[f"{projection}.scales"][expert] = scales
                assert staging is not None
                arrays = {
                    name: mx.array(value) for name, value in staging.items()
                }
                mx.eval(list(arrays.values()))
                self._banks[layer] = ShadowLayerBank(
                    layer, self.codec, self.expert_count, arrays
                )
        finally:
            if fd is not None:
                os.close(fd)
        self._fill_seconds += time.perf_counter() - started
        _LOGGER.info(
            "shadow banks ready: codec=%s layers=%d experts=%d bytes=%d "
            "(%.2f GiB) build=%.1fs",
            self.codec,
            len(self.layers),
            self.expert_count,
            self.shadow_bytes,
            self.shadow_bytes / 1024**3,
            self._fill_seconds,
        )

    def _dequantize_record(
        self,
        mx,
        record,
        blob: bytes,
        *,
        bits: int,
        group_size: int,
    ) -> dict[str, np.ndarray]:
        """Decode one sidecar record into fp32 (out, in) per projection.

        Record segments are cursor-sequential within the record blob; the
        segment ``shard``/``offset`` fields address the original
        safetensors and are ignored here.
        """

        leaves: dict[str, "mx.array"] = {}
        cursor = 0
        for segment in record.segments:
            view = memoryview(blob)[cursor : cursor + segment.length]
            cursor += segment.length
            shape = tuple(segment.shape)
            if segment.dtype == "U32":
                value = mx.array(
                    np.frombuffer(view, dtype="<u4").reshape(shape)
                )
            elif segment.dtype == "BF16":
                value = mx.array(
                    np.frombuffer(view, dtype="<u2").reshape(shape)
                ).view(mx.bfloat16)
            else:
                raise ShadowCodecError(
                    f"unsupported segment dtype {segment.dtype!r} for "
                    f"({record.layer}, {record.expert})"
                )
            leaves[segment.component] = value
        dense: dict[str, np.ndarray] = {}
        for projection in _SHADOW_PROJECTIONS:
            try:
                weight = leaves[f"{projection}.weight"]
                scales = leaves[f"{projection}.scales"]
                biases = leaves[f"{projection}.biases"]
            except KeyError as exc:
                raise ShadowCodecError(
                    f"record ({record.layer}, {record.expert}) lacks "
                    f"component {exc.args[0]!r}"
                ) from None
            dequantized = mx.dequantize(
                weight,
                scales,
                biases,
                bits=bits,
                group_size=group_size,
                mode="affine",
            )
            dense[projection] = np.asarray(
                dequantized.astype(mx.float32), dtype=np.float32
            )
        return dense

    def _allocate_staging(
        self, dense: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        staging: dict[str, np.ndarray] = {}
        for projection, weights in dense.items():
            rows, cols = weights.shape
            groups = cols // SHADOW_GROUP
            if self.codec == "b1":
                packed_shape = (rows, groups * _B1_WORDS_PER_GROUP)
                packed_dtype = np.uint32
            else:
                packed_shape = (rows, groups * _T158_BYTES_PER_GROUP)
                packed_dtype = np.uint8
            staging[f"{projection}.packed"] = np.zeros(
                (self.expert_count, *packed_shape), dtype=packed_dtype
            )
            staging[f"{projection}.scales"] = np.zeros(
                (self.expert_count, rows, groups), dtype=np.uint16
            )
        return staging
