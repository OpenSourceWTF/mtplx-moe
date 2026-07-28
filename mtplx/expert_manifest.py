"""Strict safetensors manifests for SSD-streamed MoE experts.

The manifest is the trust boundary between a revision-pinned checkpoint and
the native slot loader.  It maps every ``(layer, expert)`` to its streamed
record components without materializing tensors: nine affine Q2/Q4 leaves
(``weight``/``scales``/``biases`` per gate/up/down projection) or, for
shadow-codec (q1) artifacts, six leaves (``packed``/``scales`` per
projection).  ``quantization.mode`` selects the component schema: "affine"
or a shadow codec name ("b1", "t158") with nominal bits 1/2.

Only Python's standard library (plus numpy transitively) is used here so
manifests can be inspected and verified on machines without MLX.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping

from .expert_shadow import (
    SHADOW_CODECS,
    SHADOW_GROUP,
    _B1_WORDS_PER_GROUP,
    _T158_BYTES_PER_GROUP,
)
from .expert_streaming_models import (
    MIXED_OFFICIAL_CODEC,
    ExpertStreamingModelSpec,
    get_model_spec,
)


MANIFEST_FORMAT = "mtplx-expert-manifest-v1"
DEFAULT_ALIGNMENT = 16 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 128 * 1024 * 1024
MAX_INDEX_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_SHARDS = 4_096
MAX_MANIFEST_RESIDENT_TENSORS = 1_000_000
MAX_MANIFEST_RECORDS = 100_000
# A 226 GB bank at the ~15 GiB per-part size the hosting limit suggests is
# about 15 parts; the cap is loose enough to never bind in practice and tight
# enough that a hostile manifest cannot ask for unbounded descriptors.
MAX_SIDECAR_PARTS = 1_024
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_LEAVES = ("weight", "scales", "biases")
_SHADOW_LEAVES = ("packed", "scales")
_SHARD_KINDS = ("safetensors", "preserved-safetensors", "sidecar")
_COMPONENTS = tuple(
    f"{projection}.{leaf}" for projection in _PROJECTIONS for leaf in _LEAVES
)
# Shadow-codec (q1 lane, issue #51) records store six components per expert:
# per projection one packed sign/trit tensor plus one bf16-bit scale row.
_SHADOW_COMPONENTS = tuple(
    f"{projection}.{leaf}" for projection in _PROJECTIONS for leaf in _SHADOW_LEAVES
)
_KNOWN_COMPONENTS = frozenset(_COMPONENTS) | frozenset(_SHADOW_COMPONENTS)
# Nominal quantization bits carried by shadow-codec manifests; the byte
# math lives in the codec (10 or 15 bytes per g64 group), the nominal bits
# keep spec/manifest/pool guards single-valued.
_SHADOW_NOMINAL_BITS = {"b1": 1, "t158": 2}
_KIMI_K3_TEXT_MODEL_KEY = "kimi-k3-q1t"
_KIMI_K3_TEXT_PREFIX = "language_model."
_KIMI_K3_PRESERVED_NON_TEXT_PREFIXES = ("vision_tower.", "mm_projector.")

# Mixed-official (issue #51, M2): a per-layer tier schedule instead of one
# uniform record. down is always 3-bit affine; the gate/up pair is t158 (shadow
# packed+scales) or 2-bit affine, per layer. There is NO scalar ``bits`` — the
# schedule lives in ``quantization.layer_tier_map``. Each record therefore
# carries one of two component orders, and every segment a ``quant_tier`` stamp.
MIXED_MODE = MIXED_OFFICIAL_CODEC
_MIXED_GATE_UP_TIERS = ("t158", "affine2")
_MIXED_DOWN_TIER = "affine3"
# t158-tier layer: gate/up shadow (packed+scales), down 3-bit affine (7 leaves).
_MIXED_T158_COMPONENTS = (
    "gate_proj.packed",
    "gate_proj.scales",
    "up_proj.packed",
    "up_proj.scales",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
)
# affine2-tier layer: gate/up/down all affine leaves — identical to _COMPONENTS.
_MIXED_AFFINE2_COMPONENTS = _COMPONENTS
_MIXED_COMPONENT_ORDERS = {
    "t158": _MIXED_T158_COMPONENTS,
    "affine2": _MIXED_AFFINE2_COMPONENTS,
}


def expert_components_for_mode(quant_mode: str) -> tuple[str, ...]:
    """The ordered per-record component names for a manifest quant mode."""

    if quant_mode == "affine":
        return _COMPONENTS
    if quant_mode in SHADOW_CODECS:
        return _SHADOW_COMPONENTS
    raise ExpertManifestError(
        f"unsupported manifest quantization mode {quant_mode!r}; expected "
        f"'affine' or one of {', '.join(SHADOW_CODECS)}"
    )


_LAYER_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.mlp\."
    r"(?P<container>switch_mlp|experts)\.(?P<tail>.+)$"
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ExpertManifestError(ValueError):
    """Raised when artifact provenance or layout fails closed validation."""


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExpertManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_bytes(payload: bytes, *, source: str) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_strict_pairs)
    except ExpertManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExpertManifestError(f"invalid JSON in {source}: {exc}") from exc


def _load_json_file(path: Path, *, max_bytes: int = MAX_MANIFEST_BYTES) -> Any:
    try:
        payload = _read_file_nofollow(path, max_bytes=max_bytes)
    except ExpertManifestError:
        raise
    except OSError as exc:
        raise ExpertManifestError(f"could not read {path}: {exc}") from exc
    return _load_json_bytes(payload, source=str(path))


def _expect_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExpertManifestError(f"{label} must be an object")
    return value


def _expect_keys(
    value: dict[str, Any],
    *,
    label: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = required_set - set(value)
    unknown = set(value) - allowed
    if missing:
        raise ExpertManifestError(f"{label} is missing keys: {sorted(missing)}")
    if unknown:
        raise ExpertManifestError(f"{label} has unknown keys: {sorted(unknown)}")


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExpertManifestError(f"{label} must be an exact integer")
    if value < minimum:
        raise ExpertManifestError(f"{label} must be at least {minimum}")
    return value


def _string(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ExpertManifestError(f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, *, label: str) -> str:
    digest = _string(value, label=label)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ExpertManifestError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ExpertManifestError(f"{label} must be an integer array")
    return tuple(_integer(item, label=f"{label}[]", minimum=1) for item in value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _readonly_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _read_file_nofollow(
    path: Path,
    *,
    max_bytes: int,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> bytes:
    chunks: list[bytes] = []
    fd = os.open(path, _readonly_flags())
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ExpertManifestError(f"JSON source is not a regular file: {path}")
        if metadata.st_size > max_bytes:
            raise ExpertManifestError(
                f"JSON source {path} exceeds the {max_bytes}-byte limit"
            )
        total = 0
        while True:
            try:
                chunk = os.read(fd, min(chunk_bytes, max_bytes + 1 - total))
            except InterruptedError:
                continue
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ExpertManifestError(
                    f"JSON source {path} exceeds the {max_bytes}-byte limit"
                )
            chunks.append(chunk)
    finally:
        os.close(fd)
    return b"".join(chunks)


def _hash_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    fd: int | None = None
    try:
        fd = os.open(path, _readonly_flags())
        while True:
            try:
                chunk = os.read(fd, chunk_bytes)
            except InterruptedError:
                continue
            if not chunk:
                break
            digest.update(chunk)
    except OSError as exc:
        raise ExpertManifestError(f"could not hash {path}: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _safe_relative_name(name: str, *, label: str) -> str:
    value = _string(name, label=label)
    if "\\" in value:
        raise ExpertManifestError(f"{label} must use POSIX separators")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ExpertManifestError(f"unsafe {label}: {value!r}")
    normalized = pure.as_posix()
    if normalized in {"", "."}:
        raise ExpertManifestError(f"unsafe {label}: {value!r}")
    return normalized


def _resolve_member(root: Path, name: str, *, must_exist: bool = True) -> Path:
    relative = _safe_relative_name(name, label="artifact path")
    base = root.resolve()
    candidate = base.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise ExpertManifestError(
            f"could not resolve artifact member {relative}: {exc}"
        ) from exc
    if resolved != base and base not in resolved.parents:
        # Hugging Face's standard cache stores snapshots as symlinks of the
        # form ``snapshots/<revision>/<file> -> ../../blobs/<digest>``.  Keep
        # the general no-escape rule, but recognize that exact repository-local
        # content-addressed layout so a pinned cache snapshot is directly
        # usable without copying hundreds of gigabytes.  The blobs directory
        # itself must be a real directory, and only an actual snapshot symlink
        # may cross this boundary.
        snapshots = base.parent
        repository = snapshots.parent
        blob_root = repository / "blobs"
        try:
            resolved_blob_root = blob_root.resolve(strict=True)
        except OSError:
            resolved_blob_root = None
        hugging_face_blob = (
            snapshots.name == "snapshots"
            and candidate.is_symlink()
            and blob_root.is_dir()
            and not blob_root.is_symlink()
            and resolved_blob_root is not None
            and resolved_blob_root in resolved.parents
        )
        if not hugging_face_blob:
            raise ExpertManifestError(f"artifact member escapes root: {relative}")
    if must_exist and not resolved.is_file():
        raise ExpertManifestError(f"artifact member is not a file: {relative}")
    return resolved


def resolve_artifact_member(
    root: Path | str,
    name: str,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve a member inside its root or a repository-local HF cache blob."""

    return _resolve_member(Path(root), name, must_exist=must_exist)


@dataclass(frozen=True)
class ShardInfo:
    name: str
    size: int
    header_bytes: int
    header_sha256: str
    sha256: str | None = None
    kind: str = "safetensors"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "size": self.size,
            "header_bytes": self.header_bytes,
            "header_sha256": self.header_sha256,
        }
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        if self.kind != "safetensors":
            result["kind"] = self.kind
        return result

    @classmethod
    def from_dict(cls, value: Any) -> ShardInfo:
        obj = _expect_object(value, label="shard")
        _expect_keys(
            obj,
            label="shard",
            required=("name", "size", "header_bytes", "header_sha256"),
            optional=("sha256", "kind"),
        )
        kind = obj.get("kind", "safetensors")
        if kind not in _SHARD_KINDS:
            raise ExpertManifestError(f"unsupported shard kind {kind!r}")
        digest = obj.get("sha256")
        header_minimum = 0 if kind == "sidecar" else 1
        header_bytes = _integer(
            obj["header_bytes"],
            label="shard header_bytes",
            minimum=header_minimum,
        )
        header_sha256 = _string(obj["header_sha256"], label="shard header_sha256")
        if kind == "sidecar" and (header_bytes != 0 or header_sha256 != EMPTY_SHA256):
            raise ExpertManifestError(
                "sidecar shard must have an empty zero-byte header"
            )
        return cls(
            name=_safe_relative_name(obj["name"], label="shard name"),
            size=_integer(obj["size"], label="shard size", minimum=1),
            header_bytes=header_bytes,
            header_sha256=header_sha256,
            sha256=None if digest is None else _string(digest, label="shard sha256"),
            kind=kind,
        )


@dataclass(frozen=True)
class TensorSegment:
    component: str
    tensor: str
    shard: str
    offset: int
    length: int
    dtype: str
    shape: tuple[int, ...]
    # Mixed-official (issue #51, M2): the tier that produced this leaf
    # ("t158"/"affine2"/"affine3"). A reader never re-derives the schedule; it
    # is preserved verbatim from the converter (D5). None for uniform lanes.
    quant_tier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "component": self.component,
            "tensor": self.tensor,
            "shard": self.shard,
            "offset": self.offset,
            "length": self.length,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }
        if self.quant_tier is not None:
            result["quant_tier"] = self.quant_tier
        return result

    @classmethod
    def from_dict(cls, value: Any) -> TensorSegment:
        obj = _expect_object(value, label="segment")
        _expect_keys(
            obj,
            label="segment",
            required=(
                "component",
                "tensor",
                "shard",
                "offset",
                "length",
                "dtype",
                "shape",
            ),
            optional=("quant_tier",),
        )
        component = _string(obj["component"], label="segment component")
        if component not in _KNOWN_COMPONENTS:
            raise ExpertManifestError(f"unsupported expert component {component!r}")
        dtype = _string(obj["dtype"], label="segment dtype")
        if dtype not in _DTYPE_BYTES:
            raise ExpertManifestError(f"unsupported safetensors dtype {dtype!r}")
        raw_tier = obj.get("quant_tier")
        quant_tier = (
            None if raw_tier is None else _string(raw_tier, label="segment quant_tier")
        )
        return cls(
            component=component,
            tensor=_string(obj["tensor"], label="segment tensor"),
            shard=_safe_relative_name(obj["shard"], label="segment shard"),
            offset=_integer(obj["offset"], label="segment offset"),
            length=_integer(obj["length"], label="segment length", minimum=1),
            dtype=dtype,
            shape=_shape(obj["shape"], label="segment shape"),
            quant_tier=quant_tier,
        )


@dataclass(frozen=True)
class ResidentTensor:
    tensor: str
    shard: str
    offset: int
    length: int
    dtype: str
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor": self.tensor,
            "shard": self.shard,
            "offset": self.offset,
            "length": self.length,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ResidentTensor:
        obj = _expect_object(value, label="resident tensor")
        _expect_keys(
            obj,
            label="resident tensor",
            required=("tensor", "shard", "offset", "length", "dtype", "shape"),
        )
        dtype = _string(obj["dtype"], label="resident dtype")
        if dtype not in _DTYPE_BYTES:
            raise ExpertManifestError(f"unsupported safetensors dtype {dtype!r}")
        return cls(
            tensor=_string(obj["tensor"], label="resident tensor name"),
            shard=_safe_relative_name(obj["shard"], label="resident shard"),
            offset=_integer(obj["offset"], label="resident offset"),
            length=_integer(obj["length"], label="resident length", minimum=1),
            dtype=dtype,
            shape=_shape(obj["shape"], label="resident shape"),
        )


@dataclass(frozen=True)
class ExpertRecord:
    layer: int
    expert: int
    logical_bytes: int
    segments: tuple[TensorSegment, ...]
    sha256: str | None = None
    sidecar_offset: int | None = None
    sidecar_length: int | None = None
    # Index into ``SidecarInfo.parts``.  Part 0 is the only part a
    # single-file sidecar has, so the default keeps every pre-parts
    # manifest describing exactly what it described before.
    part: int = 0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "layer": self.layer,
            "expert": self.expert,
            "logical_bytes": self.logical_bytes,
            "segments": [segment.to_dict() for segment in self.segments],
        }
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        if self.sidecar_offset is not None:
            result["sidecar_offset"] = self.sidecar_offset
            result["sidecar_length"] = self.sidecar_length
        # Emitted only when non-default so single-part manifests serialize
        # byte-identically to what they were before parts existed -- the
        # manifest digest is taken over this JSON.
        if self.part:
            result["part"] = self.part
        return result

    @classmethod
    def from_dict(cls, value: Any) -> ExpertRecord:
        obj = _expect_object(value, label="expert record")
        _expect_keys(
            obj,
            label="expert record",
            required=("layer", "expert", "logical_bytes", "segments"),
            optional=("sha256", "sidecar_offset", "sidecar_length", "part"),
        )
        raw_segments = obj["segments"]
        if not isinstance(raw_segments, list):
            raise ExpertManifestError("expert record segments must be an array")
        _valid_lengths = {
            len(_COMPONENTS),
            len(_SHADOW_COMPONENTS),
            len(_MIXED_T158_COMPONENTS),
        }
        if len(raw_segments) not in _valid_lengths:
            raise ExpertManifestError(
                "expert record must contain the nine ordered affine components, "
                "the six ordered shadow-codec components, or a mixed-official "
                "seven-segment (t158 gate/up + affine3 down) record"
            )
        segments = tuple(TensorSegment.from_dict(item) for item in raw_segments)
        ordered = tuple(segment.component for segment in segments)
        # _MIXED_AFFINE2_COMPONENTS == _COMPONENTS (the affine order), so the
        # 9-segment mixed order is already covered by _COMPONENTS.
        _valid_orders = {_COMPONENTS, _SHADOW_COMPONENTS, _MIXED_T158_COMPONENTS}
        if ordered not in _valid_orders:
            raise ExpertManifestError(
                "expert record must contain the nine ordered affine components, "
                "the six ordered shadow-codec components, or the seven ordered "
                "mixed-official t158 components"
            )
        digest = obj.get("sha256")
        sidecar_offset = obj.get("sidecar_offset")
        sidecar_length = obj.get("sidecar_length")
        if (sidecar_offset is None) != (sidecar_length is None):
            raise ExpertManifestError(
                "sidecar_offset and sidecar_length must appear together"
            )
        return cls(
            layer=_integer(obj["layer"], label="record layer"),
            expert=_integer(obj["expert"], label="record expert"),
            logical_bytes=_integer(
                obj["logical_bytes"], label="record logical_bytes", minimum=1
            ),
            segments=segments,
            sha256=None if digest is None else _string(digest, label="record sha256"),
            sidecar_offset=(
                None
                if sidecar_offset is None
                else _integer(sidecar_offset, label="record sidecar_offset")
            ),
            sidecar_length=(
                None
                if sidecar_length is None
                else _integer(sidecar_length, label="record sidecar_length", minimum=1)
            ),
            part=_integer(obj.get("part", 0), label="record part", minimum=0),
        )


@dataclass(frozen=True)
class SidecarPart:
    """One file of an expert bank.

    ``data_start`` is where the record region begins inside the file.  A raw
    ``.bin`` bank starts at 0; a safetensors-framed part starts after its
    header, and record offsets stay relative to it so a record's position in
    the manifest never depends on how long the header happens to be.
    """

    file: str
    size: int
    sha256: str
    data_start: int = 0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "file": self.file,
            "size": self.size,
            "sha256": self.sha256,
        }
        if self.data_start:
            result["data_start"] = self.data_start
        return result

    @classmethod
    def from_dict(cls, value: Any) -> SidecarPart:
        obj = _expect_object(value, label="sidecar part")
        _expect_keys(
            obj,
            label="sidecar part",
            required=("file", "size", "sha256"),
            optional=("data_start",),
        )
        return cls(
            file=_safe_relative_name(obj["file"], label="sidecar file"),
            size=_integer(obj["size"], label="sidecar size", minimum=1),
            sha256=_string(obj["sha256"], label="sidecar sha256"),
            data_start=_integer(
                obj.get("data_start", 0), label="sidecar data_start", minimum=0
            ),
        )


@dataclass(frozen=True, init=False)
class SidecarInfo:
    """The expert bank: one or more parts holding whole expert records.

    ``pread`` does not care how many files the bank spans, so the bank may be
    split to stay under a hosting file-size limit.  A record never straddles a
    part, which keeps the split a matter of *which file* rather than of range
    arithmetic.

    The single-file spelling (``file``/``size``/``sha256`` as scalars) is
    still accepted on construction and still emitted by ``to_dict`` for
    one-part banks, so manifests written before parts existed parse and
    re-serialize unchanged.
    """

    alignment: int
    parts: tuple[SidecarPart, ...]

    def __init__(
        self,
        *,
        alignment: int,
        parts: Iterable[SidecarPart] | None = None,
        file: str | None = None,
        size: int | None = None,
        sha256: str | None = None,
        data_start: int = 0,
    ) -> None:
        overrides = {
            key: value
            for key, value in (
                ("file", file),
                ("size", size),
                ("sha256", sha256),
            )
            if value is not None
        }
        if data_start:
            overrides["data_start"] = data_start
        if parts is None:
            missing = {"file", "size", "sha256"} - set(overrides)
            if missing:
                raise ExpertManifestError(
                    "sidecar requires either parts or file/size/sha256"
                )
            resolved = (SidecarPart(**overrides),)
        else:
            resolved = tuple(parts)
            if overrides:
                # ``replace(sidecar, sha256=...)`` is the single-part spelling
                # applied to an existing bank.  It only means something when
                # there is exactly one part to apply it to.
                if len(resolved) != 1:
                    raise ExpertManifestError(
                        "the single-part file/size/sha256 spelling cannot "
                        "address a multi-part sidecar"
                    )
                resolved = (replace(resolved[0], **overrides),)
        if not resolved:
            raise ExpertManifestError("sidecar requires at least one part")
        object.__setattr__(self, "alignment", alignment)
        object.__setattr__(self, "parts", resolved)

    def _only(self) -> SidecarPart:
        if len(self.parts) != 1:
            raise ExpertManifestError(
                "this sidecar has multiple parts; ask for the part a record "
                "lives in instead of the whole-bank file/size/sha256"
            )
        return self.parts[0]

    @property
    def file(self) -> str:
        return self._only().file

    @property
    def size(self) -> int:
        return self._only().size

    @property
    def sha256(self) -> str:
        return self._only().sha256

    def part(self, index: int) -> SidecarPart:
        if not 0 <= index < len(self.parts):
            raise ExpertManifestError(f"sidecar has no part {index}")
        return self.parts[index]

    def part_for(self, record: ExpertRecord) -> SidecarPart:
        return self.part(record.part)

    def absolute_offset(self, record: ExpertRecord) -> int:
        """The record's byte offset inside its own part file."""

        if record.sidecar_offset is None:
            raise ExpertManifestError("manifest sidecar record is incomplete")
        return self.part_for(record).data_start + record.sidecar_offset

    def to_dict(self) -> dict[str, Any]:
        only = self.parts[0]
        if len(self.parts) == 1 and not only.data_start:
            # Byte-identical to the pre-parts serialization; the manifest
            # digest is taken over this JSON, so existing digests still verify.
            return {
                "file": only.file,
                "alignment": self.alignment,
                "size": only.size,
                "sha256": only.sha256,
            }
        return {
            "alignment": self.alignment,
            "parts": [part.to_dict() for part in self.parts],
        }

    @classmethod
    def from_dict(cls, value: Any) -> SidecarInfo:
        obj = _expect_object(value, label="sidecar")
        if "parts" in obj:
            _expect_keys(obj, label="sidecar", required=("alignment", "parts"))
            raw_parts = obj["parts"]
            if not isinstance(raw_parts, list):
                raise ExpertManifestError("sidecar parts must be an array")
            if not raw_parts:
                raise ExpertManifestError("sidecar requires at least one part")
            if len(raw_parts) > MAX_SIDECAR_PARTS:
                raise ExpertManifestError("sidecar part count exceeds the part limit")
            parts: Iterable[SidecarPart] | None = tuple(
                SidecarPart.from_dict(item) for item in raw_parts
            )
            scalar_kwargs: dict[str, Any] = {}
        else:
            _expect_keys(
                obj, label="sidecar", required=("file", "alignment", "size", "sha256")
            )
            parts = None
            scalar_kwargs = {
                "file": _safe_relative_name(obj["file"], label="sidecar file"),
                "size": _integer(obj["size"], label="sidecar size", minimum=1),
                "sha256": _string(obj["sha256"], label="sidecar sha256"),
            }
        alignment = _integer(obj["alignment"], label="sidecar alignment", minimum=1)
        if alignment & (alignment - 1):
            raise ExpertManifestError("sidecar alignment must be a power of two")
        return cls(alignment=alignment, parts=parts, **scalar_kwargs)


def _parse_mixed_layer_tier_map(value: Any) -> tuple[tuple[int, str, str], ...]:
    """Parse ``{layer: {gate_up, down}}`` into a sorted, validated tuple."""

    obj = _expect_object(value, label="layer_tier_map")
    if not obj:
        raise ExpertManifestError("mixed-official layer_tier_map is empty")
    entries: list[tuple[int, str, str]] = []
    for key, raw in obj.items():
        try:
            layer = int(key)
        except (TypeError, ValueError):
            raise ExpertManifestError(
                f"layer_tier_map layer {key!r} is not an integer"
            ) from None
        entry = _expect_object(raw, label="layer_tier_map entry")
        _expect_keys(entry, label="layer_tier_map entry", required=("gate_up", "down"))
        gate_up = _string(entry["gate_up"], label="gate_up tier")
        down = _string(entry["down"], label="down tier")
        if gate_up not in _MIXED_GATE_UP_TIERS or down != _MIXED_DOWN_TIER:
            raise ExpertManifestError(
                f"layer {layer} tier assignment (gate_up={gate_up!r}, "
                f"down={down!r}) is out of the mixed-official contract"
            )
        entries.append((layer, gate_up, down))
    entries.sort()
    layers = [layer for layer, _gate_up, _down in entries]
    if len(set(layers)) != len(layers):
        raise ExpertManifestError("layer_tier_map has duplicate layers")
    return tuple(entries)


@dataclass(frozen=True)
class ExpertManifest:
    model_key: str
    source_repo: str
    source_revision: str
    # None only for mixed-official banks (issue #51, M2): they carry no scalar
    # bits — the per-layer tier schedule lives in ``layer_tier_map`` (D5).
    quant_bits: int | None
    quant_group_size: int
    quant_mode: str
    artifact_tensor_bytes: int
    resident_tensor_bytes: int
    routed_expert_bytes: int
    shards: tuple[ShardInfo, ...]
    resident_tensors: tuple[ResidentTensor, ...]
    records: tuple[ExpertRecord, ...]
    sidecar: SidecarInfo | None = None
    manifest_sha256: str | None = None
    format: str = MANIFEST_FORMAT
    # Mixed-official per-layer schedule: sorted ``(layer, gate_up, down)`` tiers.
    # None for uniform lanes; required (and non-empty) for MIXED_MODE.
    layer_tier_map: tuple[tuple[int, str, str], ...] | None = None

    @property
    def is_mixed(self) -> bool:
        return self.quant_mode == MIXED_MODE

    def mixed_tier_for_layer(self, layer: int) -> tuple[str, str]:
        """Return ``(gate_up_tier, down_tier)`` for a mixed-official layer.

        Fail closed (D1): a routed layer with no tier entry is a bug, not a
        default.
        """

        for entry_layer, gate_up, down in self.layer_tier_map or ():
            if entry_layer == int(layer):
                return gate_up, down
        raise ExpertManifestError(
            f"mixed-official manifest has no tier entry for layer {layer}"
        )

    def record_bytes_by_layer(self) -> dict[int, int]:
        """Per-layer streamed record bytes (all experts in a layer are equal)."""

        result: dict[int, int] = {}
        for record in self.records:
            existing = result.get(record.layer)
            if existing is None:
                result[record.layer] = record.logical_bytes
            elif existing != record.logical_bytes:
                raise ExpertManifestError(
                    f"layer {record.layer} has non-uniform record bytes "
                    f"({existing} vs {record.logical_bytes})"
                )
        return result

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        if self.is_mixed:
            # Mixed-official carries the per-layer schedule instead of a scalar
            # ``bits``; round-trips through from_dict so the digest is stable.
            quantization: dict[str, Any] = {
                "group_size": self.quant_group_size,
                "mode": self.quant_mode,
                "layer_tier_map": {
                    str(layer): {"gate_up": gate_up, "down": down}
                    for layer, gate_up, down in (self.layer_tier_map or ())
                },
            }
        else:
            quantization = {
                "bits": self.quant_bits,
                "group_size": self.quant_group_size,
                "mode": self.quant_mode,
            }
        result: dict[str, Any] = {
            "format": self.format,
            "model_key": self.model_key,
            "source_repo": self.source_repo,
            "source_revision": self.source_revision,
            "quantization": quantization,
            "artifact": {
                "tensor_bytes": self.artifact_tensor_bytes,
                "resident_tensor_bytes": self.resident_tensor_bytes,
                "routed_expert_bytes": self.routed_expert_bytes,
                "shard_count": len(self.shards),
                "record_count": len(self.records),
            },
            "shards": [shard.to_dict() for shard in self.shards],
            "resident_tensors": [tensor.to_dict() for tensor in self.resident_tensors],
            "records": [record.to_dict() for record in self.records],
        }
        if self.sidecar is not None:
            result["sidecar"] = self.sidecar.to_dict()
        if include_digest and self.manifest_sha256 is not None:
            result["manifest_sha256"] = self.manifest_sha256
        return result

    def with_digest(self) -> ExpertManifest:
        digest = _sha256_bytes(_canonical_json(self.to_dict(include_digest=False)))
        return replace(self, manifest_sha256=digest)

    def record(self, layer: int, expert: int) -> ExpertRecord:
        for record in self.records:
            if record.layer == layer and record.expert == expert:
                return record
        raise ExpertManifestError(f"manifest has no expert record ({layer}, {expert})")

    @classmethod
    def from_dict(cls, value: Any, *, verify_digest: bool = True) -> ExpertManifest:
        obj = _expect_object(value, label="manifest")
        _expect_keys(
            obj,
            label="manifest",
            required=(
                "format",
                "model_key",
                "source_repo",
                "source_revision",
                "quantization",
                "artifact",
                "shards",
                "resident_tensors",
                "records",
            ),
            optional=("sidecar", "manifest_sha256"),
        )
        if obj["format"] != MANIFEST_FORMAT:
            raise ExpertManifestError(f"unsupported manifest format {obj['format']!r}")
        quant = _expect_object(obj["quantization"], label="quantization")
        quant_mode = _string(quant.get("mode"), label="quantization mode")
        mixed = quant_mode == MIXED_MODE
        if mixed:
            # Mixed-official carries the per-layer tier schedule and no scalar
            # bits (D5). Extra provenance keys the converter may emit (tiers,
            # source_recipe) are tolerated but ignored — only layer_tier_map is
            # load-bearing at serve time.
            _expect_keys(
                quant,
                label="quantization",
                required=("group_size", "mode", "layer_tier_map"),
                optional=("tiers", "source_recipe"),
            )
            layer_tier_map = _parse_mixed_layer_tier_map(quant["layer_tier_map"])
            quant_bits: int | None = None
        else:
            _expect_keys(
                quant, label="quantization", required=("bits", "group_size", "mode")
            )
            layer_tier_map = None
            quant_bits = _integer(quant["bits"], label="quantization bits", minimum=1)
        artifact = _expect_object(obj["artifact"], label="artifact")
        _expect_keys(
            artifact,
            label="artifact",
            required=(
                "tensor_bytes",
                "resident_tensor_bytes",
                "routed_expert_bytes",
                "shard_count",
                "record_count",
            ),
        )
        raw_shards = obj["shards"]
        raw_resident = obj["resident_tensors"]
        raw_records = obj["records"]
        if not all(
            isinstance(value, list) for value in (raw_shards, raw_resident, raw_records)
        ):
            raise ExpertManifestError(
                "shards, resident_tensors, and records must be arrays"
            )
        shard_count = _integer(artifact["shard_count"], label="artifact shard_count")
        record_count = _integer(artifact["record_count"], label="artifact record_count")
        if shard_count > MAX_MANIFEST_SHARDS or len(raw_shards) > MAX_MANIFEST_SHARDS:
            raise ExpertManifestError("manifest shard count exceeds the shard limit")
        if (
            record_count > MAX_MANIFEST_RECORDS
            or len(raw_records) > MAX_MANIFEST_RECORDS
        ):
            raise ExpertManifestError("manifest record count exceeds the record limit")
        if len(raw_resident) > MAX_MANIFEST_RESIDENT_TENSORS:
            raise ExpertManifestError(
                "manifest resident tensor count exceeds the resident limit"
            )
        if shard_count != len(raw_shards) or record_count != len(raw_records):
            raise ExpertManifestError("artifact counts do not match manifest arrays")
        shards = tuple(ShardInfo.from_dict(item) for item in raw_shards)
        resident_tensors = tuple(
            ResidentTensor.from_dict(item) for item in raw_resident
        )
        records = tuple(ExpertRecord.from_dict(item) for item in raw_records)
        digest = obj.get("manifest_sha256")
        manifest = cls(
            format=MANIFEST_FORMAT,
            model_key=_string(obj["model_key"], label="model_key"),
            source_repo=_string(obj["source_repo"], label="source_repo"),
            source_revision=_string(obj["source_revision"], label="source_revision"),
            quant_bits=quant_bits,
            quant_group_size=_integer(
                quant["group_size"], label="quantization group_size", minimum=1
            ),
            quant_mode=quant_mode,
            layer_tier_map=layer_tier_map,
            artifact_tensor_bytes=_integer(
                artifact["tensor_bytes"], label="artifact tensor_bytes", minimum=1
            ),
            resident_tensor_bytes=_integer(
                artifact["resident_tensor_bytes"],
                label="resident_tensor_bytes",
                minimum=0,
            ),
            routed_expert_bytes=_integer(
                artifact["routed_expert_bytes"], label="routed_expert_bytes", minimum=1
            ),
            shards=shards,
            resident_tensors=resident_tensors,
            records=records,
            sidecar=None
            if obj.get("sidecar") is None
            else SidecarInfo.from_dict(obj["sidecar"]),
            manifest_sha256=None
            if digest is None
            else _string(digest, label="manifest_sha256"),
        )
        manifest.validate_structure()
        if verify_digest:
            if manifest.manifest_sha256 is None:
                raise ExpertManifestError("manifest_sha256 is required")
            expected = manifest.with_digest().manifest_sha256
            if manifest.manifest_sha256 != expected:
                raise ExpertManifestError("manifest digest mismatch")
        return manifest

    def validate_structure(self) -> None:
        if self.quant_mode == "affine":
            if self.quant_bits not in {2, 4}:
                raise ExpertManifestError(
                    "only affine Q2 or Q4 expert manifests are supported"
                )
        elif self.quant_mode in SHADOW_CODECS:
            if self.quant_bits != _SHADOW_NOMINAL_BITS[self.quant_mode]:
                raise ExpertManifestError(
                    f"shadow codec {self.quant_mode!r} manifests carry nominal "
                    f"quantization bits {_SHADOW_NOMINAL_BITS[self.quant_mode]}"
                )
            if self.quant_group_size != SHADOW_GROUP:
                raise ExpertManifestError(
                    f"shadow codec manifests group in {SHADOW_GROUP}s"
                )
        elif self.quant_mode == MIXED_MODE:
            if self.quant_bits is not None:
                raise ExpertManifestError(
                    "mixed-official manifests carry no scalar quantization bits"
                )
            if self.quant_group_size != SHADOW_GROUP:
                raise ExpertManifestError(
                    f"mixed-official manifests group in {SHADOW_GROUP}s"
                )
            if not self.layer_tier_map:
                raise ExpertManifestError(
                    "mixed-official manifests require a layer_tier_map"
                )
        else:
            raise ExpertManifestError(
                "only affine Q2/Q4, shadow-codec (b1, t158), or mixed-official "
                "expert manifests are supported"
            )
        # Mixed-official records differ per layer, so the expected component
        # order is resolved per record below instead of once here.
        expected_components = (
            None if self.is_mixed else expert_components_for_mode(self.quant_mode)
        )
        mixed_tiers = (
            {layer: gate_up for layer, gate_up, _down in self.layer_tier_map}
            if self.is_mixed
            else {}
        )
        if len(self.shards) > MAX_MANIFEST_SHARDS:
            raise ExpertManifestError("manifest shard count exceeds the shard limit")
        if len(self.resident_tensors) > MAX_MANIFEST_RESIDENT_TENSORS:
            raise ExpertManifestError(
                "manifest resident tensor count exceeds the resident limit"
            )
        if len(self.records) > MAX_MANIFEST_RECORDS:
            raise ExpertManifestError("manifest record count exceeds the record limit")
        if len({shard.name for shard in self.shards}) != len(self.shards):
            raise ExpertManifestError("duplicate shard names")
        shard_by_name: dict[str, ShardInfo] = {}
        for shard in self.shards:
            if _safe_relative_name(shard.name, label="shard name") != shard.name:
                raise ExpertManifestError(f"unsafe shard name: {shard.name!r}")
            if shard.kind not in _SHARD_KINDS:
                raise ExpertManifestError(f"unsupported shard kind {shard.kind!r}")
            _integer(shard.size, label="shard size", minimum=1)
            _string(shard.header_sha256, label="shard header_sha256")
            if shard.sha256 is not None:
                _string(shard.sha256, label="shard sha256")
            header_minimum = 0 if shard.kind == "sidecar" else 1
            _integer(
                shard.header_bytes,
                label="shard header_bytes",
                minimum=header_minimum,
            )
            if shard.kind == "sidecar":
                if shard.header_bytes != 0 or shard.header_sha256 != EMPTY_SHA256:
                    raise ExpertManifestError(
                        "sidecar shard must have an empty zero-byte header"
                    )
            else:
                if shard.header_bytes > shard.size:
                    raise ExpertManifestError("safetensors header exceeds its shard")
            shard_by_name[shard.name] = shard
        if (
            self.artifact_tensor_bytes
            != self.resident_tensor_bytes + self.routed_expert_bytes
        ):
            raise ExpertManifestError(
                "resident plus routed bytes do not equal artifact bytes"
            )
        keys = [(record.layer, record.expert) for record in self.records]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ExpertManifestError("expert records must be sorted and unique")
        resident_names = [tensor.tensor for tensor in self.resident_tensors]
        if resident_names != sorted(resident_names) or len(resident_names) != len(
            set(resident_names)
        ):
            raise ExpertManifestError("resident tensor names must be sorted and unique")
        shard_sizes = {shard.name: shard.size for shard in self.shards}
        routed_bytes = 0
        for record in self.records:
            if self.is_mixed:
                gate_up_tier = mixed_tiers.get(record.layer)
                if gate_up_tier is None:
                    raise ExpertManifestError(
                        f"record layer {record.layer} has no layer_tier_map entry"
                    )
                record_expected = _MIXED_COMPONENT_ORDERS[gate_up_tier]
            else:
                record_expected = expected_components
            if (
                tuple(segment.component for segment in record.segments)
                != record_expected
            ):
                raise ExpertManifestError(
                    f"expert record components do not match quantization "
                    f"mode {self.quant_mode!r}"
                )
            if (
                sum(segment.length for segment in record.segments)
                != record.logical_bytes
            ):
                raise ExpertManifestError(
                    f"record ({record.layer}, {record.expert}) length mismatch"
                )
            if (
                record.sidecar_length is not None
                and record.sidecar_length != record.logical_bytes
            ):
                raise ExpertManifestError(
                    "sidecar record length must equal logical_bytes"
                )
            routed_bytes += record.logical_bytes
            for segment in record.segments:
                _safe_relative_name(segment.shard, label="segment shard")
                if segment.dtype not in _DTYPE_BYTES:
                    raise ExpertManifestError(
                        f"unsupported segment dtype {segment.dtype!r}"
                    )
                expected_length = _DTYPE_BYTES[segment.dtype]
                for dimension in segment.shape:
                    expected_length *= dimension
                if segment.length != expected_length:
                    raise ExpertManifestError(
                        f"segment for {segment.tensor} has inconsistent dtype/shape bytes"
                    )
                size = shard_sizes.get(segment.shard)
                if size is None or segment.offset + segment.length > size:
                    raise ExpertManifestError(
                        f"segment for {segment.tensor} exceeds its shard"
                    )
        if routed_bytes != self.routed_expert_bytes:
            raise ExpertManifestError("record bytes do not equal routed_expert_bytes")
        if (
            sum(tensor.length for tensor in self.resident_tensors)
            != self.resident_tensor_bytes
        ):
            raise ExpertManifestError(
                "resident tensor bytes do not match artifact metadata"
            )
        resident_shard_names: set[str] = set()
        for tensor in self.resident_tensors:
            shard = shard_by_name.get(tensor.shard)
            if shard is None or shard.kind != "safetensors":
                raise ExpertManifestError(
                    f"resident tensor {tensor.tensor} must reference a safetensors shard"
                )
            expected_length = _DTYPE_BYTES.get(tensor.dtype)
            if expected_length is None:
                raise ExpertManifestError(
                    f"unsupported resident dtype {tensor.dtype!r}"
                )
            for dimension in tensor.shape:
                expected_length *= dimension
            if tensor.length != expected_length:
                raise ExpertManifestError(
                    f"resident tensor {tensor.tensor} has inconsistent dtype/shape bytes"
                )
            if tensor.offset < shard.header_bytes:
                raise ExpertManifestError(
                    f"resident tensor {tensor.tensor} overlaps its shard header"
                )
            if tensor.offset + tensor.length > shard.size:
                raise ExpertManifestError(
                    f"resident tensor {tensor.tensor} exceeds its shard"
                )
            resident_shard_names.add(tensor.shard)
        if self.sidecar is None and any(
            record.sidecar_offset is not None for record in self.records
        ):
            raise ExpertManifestError("record sidecar offsets require sidecar metadata")
        if self.sidecar is not None:
            alignment = _integer(
                self.sidecar.alignment,
                label="sidecar alignment",
                minimum=1,
            )
            if alignment & (alignment - 1):
                raise ExpertManifestError("sidecar alignment must be a power of two")
            parts = self.sidecar.parts
            if not parts:
                raise ExpertManifestError("sidecar requires at least one part")
            if len(parts) > MAX_SIDECAR_PARTS:
                raise ExpertManifestError("sidecar part count exceeds the part limit")
            if len({part.file for part in parts}) != len(parts):
                raise ExpertManifestError("duplicate sidecar part files")
            for part in parts:
                _safe_relative_name(part.file, label="sidecar file")
                _integer(part.size, label="sidecar size", minimum=1)
                _string(part.sha256, label="sidecar sha256")
                _integer(part.data_start, label="sidecar data_start", minimum=0)
                if part.data_start % alignment:
                    raise ExpertManifestError("sidecar data_start is not aligned")
                if part.data_start >= part.size:
                    raise ExpertManifestError("sidecar part holds no record data")
            # Bounds and non-overlap are per-part: each part is its own address
            # space, so a record's range only has to make sense inside the file
            # it actually lives in.
            ranges_by_part: dict[int, list[tuple[int, int]]] = {}
            for record in self.records:
                if record.sidecar_offset is None or record.sidecar_length is None:
                    raise ExpertManifestError("every record requires a sidecar offset")
                if not 0 <= record.part < len(parts):
                    raise ExpertManifestError(
                        f"record ({record.layer}, {record.expert}) names sidecar "
                        f"part {record.part}, which does not exist"
                    )
                part = parts[record.part]
                if record.sidecar_offset % alignment:
                    raise ExpertManifestError("sidecar record is not aligned")
                end = record.sidecar_offset + record.sidecar_length
                # A record that would run past the end of its own part is a
                # record that straddles two parts.  Refuse it here: the read
                # path is allowed to assume one record, one file.
                if part.data_start + end > part.size:
                    raise ExpertManifestError("sidecar record exceeds file size")
                ranges_by_part.setdefault(record.part, []).append(
                    (record.sidecar_offset, end)
                )
            for ranges in ranges_by_part.values():
                if ranges != sorted(ranges) or any(
                    a[1] > b[0] for a, b in zip(ranges, ranges[1:])
                ):
                    raise ExpertManifestError("sidecar records overlap or are unsorted")

        sidecar_shards = [shard for shard in self.shards if shard.kind == "sidecar"]
        if sidecar_shards:
            if self.sidecar is None:
                raise ExpertManifestError(
                    "an authoritative sidecar shard requires sidecar metadata"
                )
            if len(sidecar_shards) != len(self.sidecar.parts):
                raise ExpertManifestError(
                    "an authoritative manifest requires one sidecar shard per "
                    "sidecar part"
                )
            part_by_file = {part.file: part for part in self.sidecar.parts}
            for sidecar_shard in sidecar_shards:
                part = part_by_file.get(sidecar_shard.name)
                if (
                    part is None
                    or sidecar_shard.size != part.size
                    or sidecar_shard.sha256 != part.sha256
                ):
                    raise ExpertManifestError("authoritative sidecar metadata mismatch")
                if sidecar_shard.sha256 is None:
                    raise ExpertManifestError("authoritative sidecar requires a hash")
                _sha256(sidecar_shard.sha256, label="authoritative sidecar sha256")
            safetensors_shards = {
                shard.name for shard in self.shards if shard.kind == "safetensors"
            }
            if safetensors_shards != resident_shard_names:
                raise ExpertManifestError(
                    "authoritative manifest contains unreferenced safetensors shards"
                )
            preserved_shards = {
                shard.name
                for shard in self.shards
                if shard.kind == "preserved-safetensors"
            }
            for name in resident_shard_names | preserved_shards:
                if shard_by_name[name].sha256 is None:
                    raise ExpertManifestError(
                        f"authoritative resident shard {name} requires a hash"
                    )
                _sha256(
                    shard_by_name[name].sha256,
                    label=f"authoritative resident shard {name} sha256",
                )
            for record in self.records:
                if record.sidecar_offset is None:
                    raise ExpertManifestError(
                        "authoritative record requires a sidecar offset"
                    )
                # Set membership over the parts: a record's components must
                # all sit in the one part the record claims.
                part = self.sidecar.part_for(record)
                start = part.data_start + record.sidecar_offset
                cursor = start
                for segment in record.segments:
                    if segment.shard != part.file or segment.offset != cursor:
                        raise ExpertManifestError(
                            "authoritative record components must be contiguous in the sidecar"
                        )
                    cursor += segment.length
                if cursor != start + record.logical_bytes:
                    raise ExpertManifestError(
                        "authoritative record components do not cover the record"
                    )


@dataclass(frozen=True)
class _TensorInfo:
    name: str
    shard: str
    offset: int
    length: int
    dtype: str
    shape: tuple[int, ...]


def load_expert_manifest(
    path: Path | str, *, verify_digest: bool = True
) -> ExpertManifest:
    value = _load_json_file(Path(path))
    return ExpertManifest.from_dict(value, verify_digest=verify_digest)


def save_expert_manifest(manifest: ExpertManifest, path: Path | str) -> ExpertManifest:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    finalized = manifest.with_digest()
    payload = json.dumps(finalized.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError as exc:
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ExpertManifestError(f"could not save manifest {target}: {exc}") from exc
    return finalized


def _read_safetensors_header(
    path: Path, *, relative_name: str
) -> tuple[ShardInfo, tuple[_TensorInfo, ...]]:
    try:
        fd = os.open(path, _readonly_flags())
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ExpertManifestError(
                    f"{relative_name} is not a regular safetensors file"
                )
            size = metadata.st_size
            length_raw = _pread_exact(fd, 0, 8)
            header_length = int.from_bytes(length_raw, "little")
            if not 1 <= header_length <= MAX_SAFETENSORS_HEADER_BYTES:
                raise ExpertManifestError(
                    f"{relative_name} has an invalid header length"
                )
            header_raw = _pread_exact(fd, 8, header_length)
        finally:
            os.close(fd)
    except OSError as exc:
        raise ExpertManifestError(f"could not inspect {relative_name}: {exc}") from exc
    header = _expect_object(
        _load_json_bytes(header_raw, source=relative_name),
        label=f"safetensors header {relative_name}",
    )
    if len(header) > MAX_MANIFEST_RESIDENT_TENSORS:
        raise ExpertManifestError(
            f"{relative_name} tensor count exceeds the header inventory limit"
        )
    data_start = 8 + header_length
    tensors: list[_TensorInfo] = []
    ranges: list[tuple[int, int, str]] = []
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        info = _expect_object(raw, label=f"tensor {name}")
        _expect_keys(
            info, label=f"tensor {name}", required=("dtype", "shape", "data_offsets")
        )
        dtype = _string(info["dtype"], label=f"tensor {name} dtype")
        if dtype not in _DTYPE_BYTES:
            raise ExpertManifestError(f"tensor {name} has unsupported dtype {dtype!r}")
        shape = _shape(info["shape"], label=f"tensor {name} shape")
        offsets = info["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ExpertManifestError(f"tensor {name} has invalid data_offsets")
        start = _integer(offsets[0], label=f"tensor {name} start")
        end = _integer(offsets[1], label=f"tensor {name} end")
        if end <= start:
            raise ExpertManifestError(f"tensor {name} has an empty or reversed range")
        length = end - start
        expected = _DTYPE_BYTES[dtype]
        for dimension in shape:
            expected *= dimension
        if length != expected:
            raise ExpertManifestError(
                f"tensor {name} byte length {length} does not match dtype/shape {expected}"
            )
        absolute_start = data_start + start
        absolute_end = data_start + end
        if absolute_end > size:
            raise ExpertManifestError(f"tensor {name} exceeds {relative_name}")
        tensors.append(
            _TensorInfo(
                name=name,
                shard=relative_name,
                offset=absolute_start,
                length=length,
                dtype=dtype,
                shape=shape,
            )
        )
        ranges.append((absolute_start, absolute_end, name))
    if not tensors:
        raise ExpertManifestError(f"{relative_name} contains no tensors")
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            raise ExpertManifestError(
                f"overlapping tensors in {relative_name}: {previous[2]} and {current[2]}"
            )
    if max(end for _start, end, _name in ranges) != size:
        raise ExpertManifestError(
            f"{relative_name} has trailing or unindexed payload bytes"
        )
    shard = ShardInfo(
        name=relative_name,
        size=size,
        header_bytes=data_start,
        header_sha256=_sha256_bytes(length_raw + header_raw),
    )
    return shard, tuple(tensors)


def _checkpoint_inventory(
    root: Path,
    *,
    hash_shards: bool,
) -> tuple[tuple[ShardInfo, ...], dict[str, _TensorInfo]]:
    index_path = root / "model.safetensors.index.json"
    weight_map: dict[str, str] | None = None
    declared_total_size: int | None = None
    if index_path.is_file() or index_path.is_symlink():
        index_path = _resolve_member(root, index_path.name)
        index = _expect_object(
            _load_json_file(index_path, max_bytes=MAX_INDEX_BYTES),
            label="safetensors index",
        )
        _expect_keys(
            index,
            label="safetensors index",
            required=("weight_map",),
            optional=("metadata",),
        )
        raw_map = _expect_object(index["weight_map"], label="safetensors weight_map")
        if len(raw_map) > MAX_MANIFEST_RESIDENT_TENSORS:
            raise ExpertManifestError(
                "safetensors weight_map exceeds the tensor inventory limit"
            )
        weight_map = {}
        for tensor, shard in raw_map.items():
            tensor_name = _string(tensor, label="weight_map tensor")
            shard_name = _safe_relative_name(shard, label="weight_map shard")
            weight_map[tensor_name] = shard_name
        if "metadata" in index:
            metadata = _expect_object(
                index["metadata"], label="safetensors index metadata"
            )
            if "total_size" in metadata:
                declared_total_size = _integer(
                    metadata["total_size"],
                    label="safetensors index metadata total_size",
                )
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = sorted(
            path.name for path in root.glob("model*.safetensors") if path.is_file()
        )
    if not shard_names:
        raise ExpertManifestError(f"no model safetensors found under {root}")

    shards: list[ShardInfo] = []
    tensors: dict[str, _TensorInfo] = {}
    for name in shard_names:
        path = _resolve_member(root, name)
        shard, shard_tensors = _read_safetensors_header(path, relative_name=name)
        if hash_shards:
            shard = replace(shard, sha256=_hash_file(path))
        shards.append(shard)
        for tensor in shard_tensors:
            if tensor.name in tensors:
                raise ExpertManifestError(
                    f"tensor {tensor.name!r} appears in multiple shards"
                )
            if weight_map is not None and weight_map.get(tensor.name) != name:
                raise ExpertManifestError(
                    f"index maps tensor {tensor.name!r} to the wrong shard"
                )
            tensors[tensor.name] = tensor
    if weight_map is not None and set(weight_map) != set(tensors):
        missing = sorted(set(weight_map) - set(tensors))
        extra = sorted(set(tensors) - set(weight_map))
        raise ExpertManifestError(
            f"index/header tensor mismatch; missing={missing[:4]}, extra={extra[:4]}"
        )
    if declared_total_size is not None and declared_total_size != sum(
        tensor.length for tensor in tensors.values()
    ):
        raise ExpertManifestError(
            "safetensors index total_size does not match header inventory"
        )
    return tuple(shards), tensors


def _expected_component_shape(
    spec: ExpertStreamingModelSpec,
    component: str,
) -> tuple[str, tuple[int, ...], int]:
    projection, leaf = component.split(".", 1)
    output = (
        spec.expert_hidden_size
        if projection in {"gate_proj", "up_proj"}
        else spec.hidden_size
    )
    input_size = (
        spec.hidden_size
        if projection in {"gate_proj", "up_proj"}
        else spec.expert_hidden_size
    )
    if spec.expert_codec != "affine":
        groups = input_size // SHADOW_GROUP
        if leaf == "packed":
            if spec.expert_codec == "b1":
                dtype = "U32"
                shape = (output, groups * _B1_WORDS_PER_GROUP)
            else:
                dtype = "U8"
                shape = (output, groups * _T158_BYTES_PER_GROUP)
        elif leaf == "scales":
            dtype = "U16"
            shape = (output, groups)
        else:
            raise ExpertManifestError(
                f"component {component!r} is not part of the "
                f"{spec.expert_codec!r} record layout"
            )
    elif leaf == "weight":
        dtype = "U32"
        shape = (output, input_size * spec.quant_bits // 32)
    else:
        dtype = "BF16"
        shape = (output, input_size // spec.quant_group_size)
    length = _DTYPE_BYTES[dtype]
    for dimension in shape:
        length *= dimension
    return dtype, shape, length


def _classify_expert_name(
    name: str,
    spec: ExpertStreamingModelSpec,
) -> tuple[int, int | None, str] | None:
    match = _LAYER_RE.search(name)
    if match is None:
        return None
    layer = int(match.group("layer"))
    if layer not in spec.routed_layer_indices:
        return None
    parts = match.group("tail").split(".")
    expert: int | None = None
    if len(parts) == 2 and parts[0] in _PROJECTIONS and parts[1] in _LEAVES:
        projection, leaf = parts
    elif (
        len(parts) == 3
        and parts[0].isdigit()
        and parts[1] in _PROJECTIONS
        and parts[2] in _LEAVES
    ):
        expert = int(parts[0])
        projection, leaf = parts[1:]
    else:
        return None
    return layer, expert, f"{projection}.{leaf}"


def _classify_expert_tensor(
    tensor: _TensorInfo,
    spec: ExpertStreamingModelSpec,
) -> tuple[int, int | None, str] | None:
    return _classify_expert_name(tensor.name, spec)


def _expert_segments(
    tensors: dict[str, _TensorInfo],
    spec: ExpertStreamingModelSpec,
) -> tuple[tuple[ExpertRecord, ...], set[str]]:
    if (
        spec.expert_codec != "affine"
        or spec.quant_bits not in {2, 4}
        or spec.quant_parameter_bytes != 2
    ):
        raise ExpertManifestError(
            "manifest builder supports affine Q2/Q4 metadata with BF16 "
            "parameters; shadow-codec (q1) manifests are emitted by "
            "the converter (mtplx.expert_q1)"
        )
    grouped: dict[tuple[int, int, str], TensorSegment] = {}
    expert_tensor_names: set[str] = set()
    for tensor in tensors.values():
        classified = _classify_expert_tensor(tensor, spec)
        if classified is None:
            continue
        layer, numbered_expert, component = classified
        expected_dtype, per_expert_shape, per_expert_length = _expected_component_shape(
            spec, component
        )
        if tensor.dtype != expected_dtype:
            raise ExpertManifestError(
                f"{tensor.name} dtype {tensor.dtype} does not match {expected_dtype}"
            )
        if numbered_expert is None:
            expected_shape = (spec.expert_count, *per_expert_shape)
            if (
                tensor.shape != expected_shape
                or tensor.length != per_expert_length * spec.expert_count
            ):
                raise ExpertManifestError(
                    f"{tensor.name} shape/length does not match stacked {expected_shape}"
                )
            experts = range(spec.expert_count)
        else:
            if not 0 <= numbered_expert < spec.expert_count:
                raise ExpertManifestError(
                    f"{tensor.name} expert index is outside the model"
                )
            if tensor.shape != per_expert_shape or tensor.length != per_expert_length:
                raise ExpertManifestError(
                    f"{tensor.name} shape/length does not match {per_expert_shape}"
                )
            experts = (numbered_expert,)
        expert_tensor_names.add(tensor.name)
        for expert in experts:
            key = (layer, expert, component)
            if key in grouped:
                raise ExpertManifestError(f"duplicate expert component {key}")
            relative_index = expert if numbered_expert is None else 0
            grouped[key] = TensorSegment(
                component=component,
                tensor=tensor.name,
                shard=tensor.shard,
                offset=tensor.offset + relative_index * per_expert_length,
                length=per_expert_length,
                dtype=tensor.dtype,
                shape=per_expert_shape,
            )

    records: list[ExpertRecord] = []
    missing: list[tuple[int, int, str]] = []
    for layer in spec.routed_layer_indices:
        for expert in range(spec.expert_count):
            segments: list[TensorSegment] = []
            for component in _COMPONENTS:
                segment = grouped.get((layer, expert, component))
                if segment is None:
                    missing.append((layer, expert, component))
                else:
                    segments.append(segment)
            if len(segments) == len(_COMPONENTS):
                logical_bytes = sum(segment.length for segment in segments)
                if logical_bytes != spec.expert_record_bytes:
                    raise ExpertManifestError(
                        f"record ({layer}, {expert}) has {logical_bytes} bytes, expected "
                        f"{spec.expert_record_bytes}"
                    )
                records.append(
                    ExpertRecord(
                        layer=layer,
                        expert=expert,
                        logical_bytes=logical_bytes,
                        segments=tuple(segments),
                    )
                )
    if missing:
        preview = ", ".join(str(item) for item in missing[:8])
        raise ExpertManifestError(
            f"checkpoint is missing {len(missing)} expert components: {preview}"
        )
    return tuple(records), expert_tensor_names


def _pread_exact(fd: int, offset: int, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    position = offset
    while remaining:
        try:
            chunk = os.pread(fd, remaining, position)
        except InterruptedError:
            continue
        if not chunk:
            raise ExpertManifestError(
                f"short positional read at offset {position}; wanted {remaining} more bytes"
            )
        chunks.append(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_source_record(root: Path, record: ExpertRecord) -> bytes:
    descriptors: dict[str, int] = {}
    chunks: list[bytes] = []
    try:
        for segment in record.segments:
            fd = descriptors.get(segment.shard)
            if fd is None:
                path = _resolve_member(root, segment.shard)
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(path, flags)
                descriptors[segment.shard] = fd
            chunks.append(_pread_exact(fd, segment.offset, segment.length))
    except OSError as exc:
        raise ExpertManifestError(
            f"could not read record ({record.layer}, {record.expert}): {exc}"
        ) from exc
    finally:
        for fd in descriptors.values():
            try:
                os.close(fd)
            except OSError:
                pass
    return b"".join(chunks)


def build_expert_manifest(
    root: Path | str,
    spec: ExpertStreamingModelSpec | str,
    *,
    source_repo: str | None = None,
    source_revision: str | None = None,
    hash_records: bool = True,
    hash_shards: bool = False,
    require_pinned_tensor_bytes: bool = True,
) -> ExpertManifest:
    """Inspect a checkpoint and build a strict deterministic expert manifest."""

    artifact_root = Path(root).resolve()
    if not artifact_root.is_dir():
        raise ExpertManifestError(f"artifact root is not a directory: {artifact_root}")
    model_spec = get_model_spec(spec) if isinstance(spec, str) else spec
    if not isinstance(model_spec, ExpertStreamingModelSpec):
        raise TypeError("spec must be a model key or ExpertStreamingModelSpec")
    shards, tensors = _checkpoint_inventory(artifact_root, hash_shards=hash_shards)
    records, expert_tensor_names = _expert_segments(tensors, model_spec)
    tensor_bytes = sum(tensor.length for tensor in tensors.values())
    if require_pinned_tensor_bytes and tensor_bytes != model_spec.total_tensor_bytes:
        raise ExpertManifestError(
            f"artifact tensor bytes {tensor_bytes} do not match pinned "
            f"{model_spec.total_tensor_bytes}"
        )
    routed_bytes = sum(record.logical_bytes for record in records)
    if routed_bytes != model_spec.routed_expert_bytes:
        raise ExpertManifestError(
            f"routed bytes {routed_bytes} do not match pinned {model_spec.routed_expert_bytes}"
        )
    resident_tensors = tuple(
        ResidentTensor(
            tensor=tensor.name,
            shard=tensor.shard,
            offset=tensor.offset,
            length=tensor.length,
            dtype=tensor.dtype,
            shape=tensor.shape,
        )
        for tensor in sorted(tensors.values(), key=lambda item: item.name)
        if tensor.name not in expert_tensor_names
    )
    resident_bytes = sum(tensor.length for tensor in resident_tensors)
    if resident_bytes + routed_bytes != tensor_bytes:
        raise ExpertManifestError(
            "expert/resident classification does not cover the artifact"
        )
    if require_pinned_tensor_bytes and resident_bytes != model_spec.resident_bytes:
        raise ExpertManifestError(
            f"resident bytes {resident_bytes} do not match pinned {model_spec.resident_bytes}"
        )
    if hash_records:
        hashed: list[ExpertRecord] = []
        for record in records:
            payload = _read_source_record(artifact_root, record)
            hashed.append(replace(record, sha256=_sha256_bytes(payload)))
        records = tuple(hashed)
    manifest = ExpertManifest(
        model_key=model_spec.key,
        source_repo=source_repo or model_spec.quant_model,
        source_revision=source_revision or model_spec.quant_revision,
        quant_bits=model_spec.quant_bits,
        quant_group_size=model_spec.quant_group_size,
        quant_mode="affine",
        artifact_tensor_bytes=tensor_bytes,
        resident_tensor_bytes=resident_bytes,
        routed_expert_bytes=routed_bytes,
        shards=shards,
        resident_tensors=resident_tensors,
        records=records,
    ).with_digest()
    manifest.validate_structure()
    return manifest


def build_mixed_official_manifest(
    root: Path | str,
    mixed_manifest: dict[str, Any],
    spec: ExpertStreamingModelSpec,
) -> ExpertManifest:
    """Unify a mixed-official bank into the streamed ``ExpertManifest`` schema.

    ``root`` is the assembled artifact directory: the resident-only safetensors
    (with ``model.safetensors.index.json``) plus the mixed-official record bin
    next to them. ``mixed_manifest`` is the dict emitted by
    ``scripts.convert_expert_mixed_official.convert_mixed_official``. The result
    is authoritative — the bin is the single sidecar shard, every expert record
    addresses it, ``quantization.mode`` is ``mixed-official-v1`` and the
    per-layer ``layer_tier_map`` is preserved — so the streamed runtime opens it
    through the ordinary ``load_expert_manifest`` + slot machinery (issue #51 M2,
    the q1 lane's equal for a non-uniform bank).
    """

    if not spec.is_mixed_official:
        raise ExpertManifestError(
            f"spec {spec.key!r} is not a mixed-official descriptor"
        )
    quant = mixed_manifest.get("quantization")
    if not isinstance(quant, dict) or quant.get("mode") != MIXED_MODE:
        raise ExpertManifestError("source manifest is not a mixed-official bank")
    sidecar_meta = mixed_manifest.get("sidecar")
    if not isinstance(sidecar_meta, dict):
        raise ExpertManifestError("mixed-official manifest lacks sidecar metadata")
    root = Path(root).resolve()
    bin_name = _safe_relative_name(sidecar_meta["file"], label="mixed bin name")
    bin_path = root / bin_name
    if not bin_path.is_file():
        raise ExpertManifestError(
            f"mixed-official record bin is not in the artifact: {bin_path}"
        )
    layer_tier_map = _parse_mixed_layer_tier_map(quant["layer_tier_map"])
    group_size = _integer(
        quant["group_size"], label="quantization group_size", minimum=1
    )

    shards, tensors = _checkpoint_inventory(root, hash_shards=True)
    routed = [name for name in tensors if _classify_expert_name(name, spec) is not None]
    if routed:
        raise ExpertManifestError(
            "mixed-official artifact safetensors must hold only resident tensors; "
            f"found routed expert tensors: {sorted(routed)[:4]}"
        )
    resident_tensors = tuple(
        ResidentTensor(
            tensor=tensor.name,
            shard=tensor.shard,
            offset=tensor.offset,
            length=tensor.length,
            dtype=tensor.dtype,
            shape=tensor.shape,
        )
        for tensor in sorted(tensors.values(), key=lambda item: item.name)
    )
    resident_bytes = sum(tensor.length for tensor in resident_tensors)
    bin_size = bin_path.stat().st_size
    bin_sha256 = _hash_file(bin_path)
    sidecar_shard = ShardInfo(
        name=bin_name,
        size=bin_size,
        header_bytes=0,
        header_sha256=EMPTY_SHA256,
        sha256=bin_sha256,
        kind="sidecar",
    )
    records: list[ExpertRecord] = []
    routed_bytes = 0
    for raw in mixed_manifest.get("records", ()):
        segments = tuple(
            TensorSegment(
                component=segment["component"],
                tensor=segment["tensor"],
                shard=bin_name,
                offset=_integer(segment["offset"], label="segment offset"),
                length=_integer(segment["length"], label="segment length", minimum=1),
                dtype=segment["dtype"],
                shape=_shape(segment["shape"], label="segment shape"),
                quant_tier=segment.get("quant_tier"),
            )
            for segment in raw["segments"]
        )
        logical = _integer(
            raw["logical_bytes"], label="record logical_bytes", minimum=1
        )
        records.append(
            ExpertRecord(
                layer=_integer(raw["layer"], label="record layer"),
                expert=_integer(raw["expert"], label="record expert"),
                logical_bytes=logical,
                segments=segments,
                sha256=raw.get("sha256"),
                sidecar_offset=_integer(
                    raw["sidecar_offset"], label="record sidecar_offset"
                ),
                sidecar_length=_integer(
                    raw["sidecar_length"], label="record sidecar_length", minimum=1
                ),
            )
        )
        routed_bytes += logical
    manifest = ExpertManifest(
        model_key=spec.key,
        source_repo=spec.quant_model,
        source_revision=spec.quant_revision,
        quant_bits=None,
        quant_group_size=group_size,
        quant_mode=MIXED_MODE,
        layer_tier_map=layer_tier_map,
        artifact_tensor_bytes=resident_bytes + routed_bytes,
        resident_tensor_bytes=resident_bytes,
        routed_expert_bytes=routed_bytes,
        shards=tuple(shards) + (sidecar_shard,),
        resident_tensors=resident_tensors,
        records=tuple(records),
        sidecar=SidecarInfo(
            file=bin_name,
            alignment=_integer(sidecar_meta["alignment"], label="sidecar alignment"),
            size=bin_size,
            sha256=bin_sha256,
        ),
    ).with_digest()
    manifest.validate_structure()
    validate_expert_manifest_spec(manifest, spec)
    return manifest


def _mixed_projection_dims(
    spec: ExpertStreamingModelSpec,
) -> dict[str, tuple[int, int]]:
    """(out, in) per projection for a mixed-official expert of this model."""

    return {
        "gate_proj": (spec.expert_hidden_size, spec.hidden_size),
        "up_proj": (spec.expert_hidden_size, spec.hidden_size),
        "down_proj": (spec.hidden_size, spec.expert_hidden_size),
    }


def _validate_mixed_manifest_spec(
    manifest: ExpertManifest,
    spec: ExpertStreamingModelSpec,
    *,
    require_pinned_tensor_bytes: bool,
) -> None:
    """Per-layer identity/geometry validation for a mixed-official bank (D5).

    The tier map is the single source of truth: every record's logical bytes
    and component layout must match ``plan_record_segments`` for that layer's
    tier, and every routed layer must have a tier (fail closed).
    """

    from mtplx.expert_mixed_official import plan_record_segments

    if manifest.quant_mode != MIXED_MODE:
        raise ExpertManifestError(
            f"manifest quantization mode {manifest.quant_mode!r} does not match "
            f"mixed-official descriptor codec {spec.expert_codec!r}"
        )
    if manifest.quant_group_size != spec.quant_group_size:
        raise ExpertManifestError(
            "manifest quantization group size does not match the descriptor"
        )
    if manifest.model_key != spec.key:
        raise ExpertManifestError(
            f"manifest model key {manifest.model_key!r} does not match "
            f"descriptor {spec.key!r}"
        )
    if (
        manifest.source_repo != spec.quant_model
        or manifest.source_revision != spec.quant_revision
    ):
        raise ExpertManifestError(
            "manifest source identity does not match the pinned descriptor"
        )

    expected_keys = tuple(
        (layer, expert)
        for layer in spec.routed_layer_indices
        for expert in range(spec.expert_count)
    )
    actual_keys = tuple((record.layer, record.expert) for record in manifest.records)
    if actual_keys != expected_keys:
        raise ExpertManifestError(
            "manifest record keys do not match the descriptor Cartesian product"
        )

    dims = _mixed_projection_dims(spec)
    # Cache the planned layout per (gate_up, down) tier so the per-record loop
    # is O(records) rather than O(records * segments) of layout math.
    planned_by_tiers: dict[tuple[str, str], tuple] = {}
    for record in manifest.records:
        gate_up_tier, down_tier = manifest.mixed_tier_for_layer(record.layer)
        key = (gate_up_tier, down_tier)
        planned = planned_by_tiers.get(key)
        if planned is None:
            planned = plan_record_segments(
                {"gate_up": gate_up_tier, "down": down_tier}, dims
            )
            planned_by_tiers[key] = planned
        if len(record.segments) != len(planned):
            raise ExpertManifestError(
                f"record ({record.layer}, {record.expert}) has "
                f"{len(record.segments)} segments; expected {len(planned)}"
            )
        for segment, plan in zip(record.segments, planned):
            if (
                segment.component != plan.component
                or segment.dtype != plan.dtype
                or segment.shape != tuple(plan.shape)
                or segment.length != plan.length
                or segment.quant_tier != plan.quant_tier
            ):
                raise ExpertManifestError(
                    f"record ({record.layer}, {record.expert}) component "
                    f"{plan.component} does not match the mixed-official layout"
                )
        expected_logical = sum(plan.length for plan in planned)
        if record.logical_bytes != expected_logical:
            raise ExpertManifestError(
                f"record ({record.layer}, {record.expert}) bytes "
                f"{record.logical_bytes} do not match the tier layout "
                f"{expected_logical}"
            )

    expected_routed = sum(record.logical_bytes for record in manifest.records)
    if manifest.routed_expert_bytes != expected_routed:
        raise ExpertManifestError(
            "manifest routed expert bytes do not match the record layout"
        )
    if require_pinned_tensor_bytes:
        expected_resident = spec.total_tensor_bytes - expected_routed
        if manifest.resident_tensor_bytes != expected_resident:
            raise ExpertManifestError(
                f"manifest resident bytes {manifest.resident_tensor_bytes} do not "
                f"match the descriptor {expected_resident}"
            )
        if manifest.artifact_tensor_bytes != spec.total_tensor_bytes:
            raise ExpertManifestError(
                "manifest total tensor bytes do not match the descriptor"
            )

    mtp_pattern = None
    if spec.mtp_layer_index is not None:
        mtp_pattern = re.compile(rf"(?:^|\.)layers\.{spec.mtp_layer_index}(?:\.|$)")
    for tensor in manifest.resident_tensors:
        if _classify_expert_name(tensor.tensor, spec) is not None:
            raise ExpertManifestError(
                f"resident tensor {tensor.tensor!r} is a routed expert tensor"
            )
        if mtp_pattern is not None and mtp_pattern.search(tensor.tensor):
            raise ExpertManifestError(
                f"resident tensor {tensor.tensor!r} is an MTP layer tensor"
            )


def validate_expert_manifest_spec(
    manifest: ExpertManifest,
    spec: ExpertStreamingModelSpec,
    *,
    require_pinned_tensor_bytes: bool = True,
) -> None:
    """Require exact identity, record keys, bytes, and component geometry."""

    if not isinstance(manifest, ExpertManifest):
        raise TypeError("manifest must be an ExpertManifest")
    if not isinstance(spec, ExpertStreamingModelSpec):
        raise TypeError("spec must be an ExpertStreamingModelSpec")
    manifest.validate_structure()
    preserved_safetensors = tuple(
        shard for shard in manifest.shards if shard.kind == "preserved-safetensors"
    )
    if preserved_safetensors:
        trusted_kimi_spec = get_model_spec(_KIMI_K3_TEXT_MODEL_KEY)
        if spec != trusted_kimi_spec:
            raise ExpertManifestError(
                "preserved safetensors shards require the trusted Kimi K3 descriptor"
            )
        if not any(shard.kind == "sidecar" for shard in manifest.shards):
            raise ExpertManifestError(
                "preserved safetensors shards require an authoritative manifest"
            )
    if spec.is_mixed_official:
        _validate_mixed_manifest_spec(
            manifest, spec, require_pinned_tensor_bytes=require_pinned_tensor_bytes
        )
        return
    if manifest.quant_bits != spec.quant_bits:
        raise ExpertManifestError(
            f"manifest bits {manifest.quant_bits} do not match descriptor bits "
            f"{spec.quant_bits}"
        )
    if manifest.quant_group_size != spec.quant_group_size:
        raise ExpertManifestError(
            "manifest quantization group size does not match the descriptor"
        )
    expected_mode = "affine" if spec.expert_codec == "affine" else spec.expert_codec
    if manifest.quant_mode != expected_mode:
        raise ExpertManifestError(
            f"manifest quantization mode {manifest.quant_mode!r} does not "
            f"match descriptor codec {expected_mode!r}"
        )
    if manifest.model_key != spec.key:
        raise ExpertManifestError(
            f"manifest model key {manifest.model_key!r} does not match "
            f"descriptor {spec.key!r}"
        )
    if (
        manifest.source_repo != spec.quant_model
        or manifest.source_revision != spec.quant_revision
    ):
        raise ExpertManifestError(
            "manifest source identity does not match the pinned descriptor"
        )

    expected_keys = tuple(
        (layer, expert)
        for layer in spec.routed_layer_indices
        for expert in range(spec.expert_count)
    )
    actual_keys = tuple((record.layer, record.expert) for record in manifest.records)
    if actual_keys != expected_keys:
        raise ExpertManifestError(
            "manifest record keys do not match the descriptor Cartesian product"
        )
    expected_components = expert_components_for_mode(expected_mode)
    for record in manifest.records:
        if record.logical_bytes != spec.expert_record_bytes:
            raise ExpertManifestError(
                f"record ({record.layer}, {record.expert}) bytes do not match "
                "the descriptor"
            )
        for component, segment in zip(
            expected_components, record.segments, strict=True
        ):
            expected_dtype, expected_shape, expected_length = _expected_component_shape(
                spec, component
            )
            if (
                segment.component != component
                or segment.dtype != expected_dtype
                or segment.shape != expected_shape
                or segment.length != expected_length
            ):
                raise ExpertManifestError(
                    f"record ({record.layer}, {record.expert}) component "
                    f"{component} does not match descriptor geometry"
                )
    if manifest.routed_expert_bytes != spec.routed_expert_bytes:
        raise ExpertManifestError(
            "manifest routed expert bytes do not match the descriptor"
        )
    if require_pinned_tensor_bytes and (
        manifest.resident_tensor_bytes != spec.resident_bytes
        or manifest.artifact_tensor_bytes != spec.total_tensor_bytes
    ):
        raise ExpertManifestError(
            "manifest resident or total tensor bytes do not match the descriptor"
        )

    mtp_pattern = None
    if spec.mtp_layer_index is not None:
        mtp_pattern = re.compile(rf"(?:^|\.)layers\.{spec.mtp_layer_index}(?:\.|$)")
    for tensor in manifest.resident_tensors:
        if _classify_expert_name(tensor.tensor, spec) is not None:
            raise ExpertManifestError(
                f"resident tensor {tensor.tensor!r} is a routed expert tensor"
            )
        if mtp_pattern is not None and mtp_pattern.search(tensor.tensor):
            raise ExpertManifestError(
                f"resident tensor {tensor.tensor!r} is an MTP layer tensor"
            )


def make_sidecar_authoritative(
    manifest: ExpertManifest,
    spec: ExpertStreamingModelSpec,
) -> ExpertManifest:
    """Retain referenced residents and rebase all expert segments to one sidecar."""

    if (
        manifest.manifest_sha256 is None
        or manifest.manifest_sha256 != manifest.with_digest().manifest_sha256
    ):
        raise ExpertManifestError("manifest digest mismatch")
    validate_expert_manifest_spec(manifest, spec)
    if manifest.sidecar is None:
        raise ExpertManifestError("authoritative conversion requires a sidecar")
    resident_names = {tensor.shard for tensor in manifest.resident_tensors}
    shard_by_name = {shard.name: shard for shard in manifest.shards}
    resident_shards: list[ShardInfo] = []
    for name in sorted(resident_names):
        shard = shard_by_name.get(name)
        if shard is None or shard.kind != "safetensors":
            raise ExpertManifestError(
                f"resident tensor shard {name!r} is not a safetensors shard"
            )
        if shard.sha256 is None:
            raise ExpertManifestError(
                f"authoritative resident shard {name} requires a full-file hash"
            )
        resident_shards.append(shard)
    if any(part.file in resident_names for part in manifest.sidecar.parts):
        raise ExpertManifestError("sidecar file conflicts with a resident shard")

    records: list[ExpertRecord] = []
    for record in manifest.records:
        if (
            record.sha256 is None
            or record.sidecar_offset is None
            or record.sidecar_length is None
        ):
            raise ExpertManifestError(
                "authoritative records require hashes and complete sidecar ranges"
            )
        _sha256(record.sha256, label="authoritative record sha256")
        part = manifest.sidecar.part_for(record)
        start = part.data_start + record.sidecar_offset
        cursor = start
        segments: list[TensorSegment] = []
        for segment in record.segments:
            segments.append(
                replace(
                    segment,
                    shard=part.file,
                    offset=cursor,
                )
            )
            cursor += segment.length
        if cursor != start + record.logical_bytes:
            raise ExpertManifestError(
                "record components do not exactly cover the sidecar record"
            )
        records.append(replace(record, segments=tuple(segments)))

    sidecar_shards = tuple(
        ShardInfo(
            name=part.file,
            size=part.size,
            header_bytes=0,
            header_sha256=EMPTY_SHA256,
            sha256=part.sha256,
            kind="sidecar",
        )
        for part in manifest.sidecar.parts
    )
    authoritative = replace(
        manifest,
        shards=tuple(resident_shards) + sidecar_shards,
        records=tuple(records),
        manifest_sha256=None,
    ).with_digest()
    validate_expert_manifest_spec(authoritative, spec)
    return authoritative


def _validate_authoritative_resident_inventory(
    expected_residents: Mapping[str, ResidentTensor],
    inventory_tensors: Mapping[str, ResidentTensor],
    *,
    trusted_spec: ExpertStreamingModelSpec | None = None,
) -> None:
    """Bind an authoritative index to the runtime's resident allowlist.

    K3 preserves its pinned multimodal tensors in the checkpoint inventory,
    but the dedicated streamed runtime is text-only.  Its allowlist therefore
    contains every ``language_model.*`` tensor and no vision/projector tensor.
    The two exact upstream non-text namespaces may remain indexed so their
    byte-preserved payload is not discarded.  Every other model retains the
    original exact-inventory contract.
    """

    expected_names = set(expected_residents)
    inventory_names = set(inventory_tensors)
    if trusted_spec is not None and trusted_spec.key == _KIMI_K3_TEXT_MODEL_KEY:
        if any(not name.startswith(_KIMI_K3_TEXT_PREFIX) for name in expected_names):
            raise ExpertManifestError("Kimi K3 resident allowlist must be text-only")
        indexed_text = {
            name for name in inventory_names if name.startswith(_KIMI_K3_TEXT_PREFIX)
        }
        extras = inventory_names - indexed_text
        if indexed_text != expected_names or any(
            not name.startswith(_KIMI_K3_PRESERVED_NON_TEXT_PREFIXES) for name in extras
        ):
            raise ExpertManifestError(
                "authoritative resident index/header inventory mismatch"
            )
    elif inventory_names != expected_names:
        raise ExpertManifestError(
            "authoritative resident index/header inventory mismatch"
        )

    for name, expected in expected_residents.items():
        tensor = inventory_tensors.get(name)
        if tensor is None:
            raise ExpertManifestError(
                "authoritative resident index/header inventory mismatch"
            )
        if (
            tensor.shard != expected.shard
            or tensor.offset != expected.offset
            or tensor.length != expected.length
            or tensor.dtype != expected.dtype
            or tensor.shape != expected.shape
        ):
            raise ExpertManifestError(
                f"authoritative resident metadata mismatch: {name}"
            )


def verify_expert_manifest(
    manifest: ExpertManifest,
    root: Path | str,
    *,
    verify_records: bool = False,
    verify_shard_hashes: bool = False,
    verify_sidecar_hash: bool = False,
) -> dict[str, int | bool | str]:
    """Verify provenance, bounds, optional payload hashes, and sidecar state."""

    artifact_root = Path(root).resolve()
    manifest.validate_structure()
    if manifest.manifest_sha256 != manifest.with_digest().manifest_sha256:
        raise ExpertManifestError("manifest digest mismatch")
    authoritative = any(shard.kind == "sidecar" for shard in manifest.shards)
    if authoritative:
        trusted_subset_spec = None
        if manifest.model_key == _KIMI_K3_TEXT_MODEL_KEY:
            trusted_subset_spec = get_model_spec(_KIMI_K3_TEXT_MODEL_KEY)
            validate_expert_manifest_spec(manifest, trusted_subset_spec)
        preserved_safetensors = {
            shard.name
            for shard in manifest.shards
            if shard.kind == "preserved-safetensors"
        }
        if preserved_safetensors and trusted_subset_spec is None:
            raise ExpertManifestError(
                "preserved safetensors shards require the trusted Kimi K3 descriptor"
            )
        if not (artifact_root / "model.safetensors.index.json").is_file():
            raise ExpertManifestError(
                "authoritative artifact requires a complete resident index"
            )
        expected_safetensors = {
            shard.name
            for shard in manifest.shards
            if shard.kind in {"safetensors", "preserved-safetensors"}
        }
        actual_safetensors = {path.name for path in artifact_root.glob("*.safetensors")}
        if actual_safetensors != expected_safetensors:
            extra = sorted(actual_safetensors - expected_safetensors)
            missing = sorted(expected_safetensors - actual_safetensors)
            raise ExpertManifestError(
                "authoritative resident inventory has unreferenced or missing "
                f"safetensors shards; extra={extra[:4]}, missing={missing[:4]}"
            )
        inventory_shards, inventory_tensors = _checkpoint_inventory(
            artifact_root,
            hash_shards=False,
        )
        if {shard.name for shard in inventory_shards} != expected_safetensors:
            raise ExpertManifestError("authoritative resident shard inventory mismatch")
        expected_residents = {
            tensor.tensor: tensor for tensor in manifest.resident_tensors
        }
        _validate_authoritative_resident_inventory(
            expected_residents,
            inventory_tensors,
            trusted_spec=trusted_subset_spec,
        )
    checked_shards = 0
    for shard in manifest.shards:
        path = _resolve_member(artifact_root, shard.name)
        if shard.kind == "sidecar":
            try:
                fd = os.open(path, _readonly_flags())
                try:
                    metadata = os.fstat(fd)
                finally:
                    os.close(fd)
            except OSError as exc:
                raise ExpertManifestError(
                    f"could not inspect sidecar {shard.name}: {exc}"
                ) from exc
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != shard.size:
                raise ExpertManifestError("sidecar size mismatch")
            checked_shards += 1
            continue
        current, _tensors = _read_safetensors_header(path, relative_name=shard.name)
        if (
            current.size != shard.size
            or current.header_bytes != shard.header_bytes
            or current.header_sha256 != shard.header_sha256
        ):
            raise ExpertManifestError(f"shard provenance mismatch: {shard.name}")
        if verify_shard_hashes:
            if shard.sha256 is None:
                raise ExpertManifestError(f"shard {shard.name} has no full-file hash")
            if _hash_file(path) != shard.sha256:
                raise ExpertManifestError(f"shard hash mismatch: {shard.name}")
        checked_shards += 1
    checked_records = 0
    if verify_records:
        for record in manifest.records:
            if record.sha256 is None:
                raise ExpertManifestError(
                    f"record ({record.layer}, {record.expert}) has no hash"
                )
            if (
                _sha256_bytes(_read_source_record(artifact_root, record))
                != record.sha256
            ):
                raise ExpertManifestError(
                    f"record hash mismatch: ({record.layer}, {record.expert})"
                )
            checked_records += 1
    sidecar_verified = False
    if manifest.sidecar is not None:
        for part in manifest.sidecar.parts:
            sidecar_path = _resolve_member(artifact_root, part.file)
            if sidecar_path.stat().st_size != part.size:
                raise ExpertManifestError("sidecar size mismatch")
            if verify_sidecar_hash:
                # Per-part digests: a mismatch names the file that is wrong
                # rather than condemning the whole bank.
                if _hash_file(sidecar_path) != part.sha256:
                    raise ExpertManifestError(f"sidecar hash mismatch: {part.file}")
        sidecar_verified = verify_sidecar_hash
    return {
        "valid": True,
        "model_key": manifest.model_key,
        "checked_shards": checked_shards,
        "checked_records": checked_records,
        "sidecar_verified": sidecar_verified,
    }


def read_expert_record(
    manifest: ExpertManifest,
    root: Path | str,
    layer: int,
    expert: int,
    *,
    prefer_sidecar: bool = True,
    verify_hash: bool = True,
) -> bytes:
    """Read one complete expert record from sidecar or source segments."""

    record = manifest.record(layer, expert)
    artifact_root = Path(root).resolve()
    if prefer_sidecar and manifest.sidecar is not None:
        if record.sidecar_offset is None or record.sidecar_length is None:
            raise ExpertManifestError("manifest sidecar record is incomplete")
        part = manifest.sidecar.part_for(record)
        path = _resolve_member(artifact_root, part.file)
        offset = part.data_start + record.sidecar_offset
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            try:
                payload = _pread_exact(fd, offset, record.sidecar_length)
            finally:
                os.close(fd)
        except OSError as exc:
            raise ExpertManifestError(f"could not read sidecar record: {exc}") from exc
    else:
        payload = _read_source_record(artifact_root, record)
    if len(payload) != record.logical_bytes:
        raise ExpertManifestError("expert record length mismatch")
    if verify_hash:
        if record.sha256 is None:
            raise ExpertManifestError("record has no payload hash")
        if _sha256_bytes(payload) != record.sha256:
            raise ExpertManifestError(
                f"record hash mismatch: ({record.layer}, {record.expert})"
            )
    return payload


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _relative_output(root: Path, output: Path) -> tuple[str, Path]:
    base = root.resolve()
    parent = output.parent.resolve()
    try:
        relative_parent = parent.relative_to(base)
    except ValueError as exc:
        raise ExpertManifestError(
            "sidecar must be created inside the artifact root"
        ) from exc
    relative = (relative_parent / output.name).as_posix()
    return _safe_relative_name(relative, label="sidecar output"), parent / output.name


def build_expert_sidecar(
    manifest: ExpertManifest,
    root: Path | str,
    output: Path | str,
    *,
    alignment: int = DEFAULT_ALIGNMENT,
    resume: bool = True,
    overwrite: bool = False,
) -> ExpertManifest:
    """Build a resumable expert-major sidecar and return an updated manifest."""

    if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment <= 0:
        raise ExpertManifestError("alignment must be a positive integer")
    if alignment & (alignment - 1):
        raise ExpertManifestError("alignment must be a power of two")
    artifact_root = Path(root).resolve()
    relative_output, final_path = _relative_output(artifact_root, Path(output))
    if final_path.exists() and not overwrite:
        raise ExpertManifestError(f"sidecar already exists: {final_path}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    partial = final_path.with_name(f".{final_path.name}.partial")
    if partial.exists() and not resume:
        partial.unlink()
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(partial, flags, 0o644)
    except OSError as exc:
        raise ExpertManifestError(
            f"could not open sidecar partial file: {exc}"
        ) from exc

    updated_records: list[ExpertRecord] = []
    cursor = 0
    try:
        for record in manifest.records:
            offset = _align_up(cursor, alignment)
            payload = read_expert_record(
                manifest,
                artifact_root,
                record.layer,
                record.expert,
                prefer_sidecar=False,
                verify_hash=record.sha256 is not None,
            )
            digest = _sha256_bytes(payload)
            current_size = os.fstat(fd).st_size
            reusable = False
            if resume and current_size >= offset + len(payload):
                reusable = (
                    _sha256_bytes(_pread_exact(fd, offset, len(payload))) == digest
                )
            if not reusable:
                os.ftruncate(fd, offset)
                if offset > os.fstat(fd).st_size:
                    os.ftruncate(fd, offset)
                position = offset
                view = memoryview(payload)
                while view:
                    try:
                        written = os.pwrite(fd, view, position)
                    except InterruptedError:
                        continue
                    if written <= 0:
                        raise ExpertManifestError("short positional sidecar write")
                    position += written
                    view = view[written:]
            updated_records.append(
                replace(
                    record,
                    sha256=digest,
                    sidecar_offset=offset,
                    sidecar_length=len(payload),
                )
            )
            cursor = offset + len(payload)
        os.ftruncate(fd, cursor)
        os.fsync(fd)
    except (OSError, ExpertManifestError) as exc:
        if isinstance(exc, ExpertManifestError):
            raise
        raise ExpertManifestError(f"sidecar build failed: {exc}") from exc
    finally:
        os.close(fd)

    sidecar_sha256 = _hash_file(partial)
    try:
        os.replace(partial, final_path)
        directory_fd = os.open(final_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise ExpertManifestError(f"could not publish sidecar: {exc}") from exc
    updated = replace(
        manifest,
        records=tuple(updated_records),
        sidecar=SidecarInfo(
            file=relative_output,
            alignment=alignment,
            size=cursor,
            sha256=sidecar_sha256,
        ),
        manifest_sha256=None,
    ).with_digest()
    updated.validate_structure()
    return updated


def iter_record_keys(manifest: ExpertManifest) -> Iterator[tuple[int, int]]:
    for record in manifest.records:
        yield record.layer, record.expert
