"""Exact CPU-side Qwen4 n-gram addressing and immutable shard manifests.

This module is deliberately independent of MLX.  Geometry and artifact
invariants are checked before an address plan or storage lane is used; the
planning path then contains only the official signed-int64 multiply/XOR and
head modulus operations.

N-gram installation has an explicit immutable-file finalization contract:
the installer closes every writer, removes all write permission bits, and
publishes each shard with exactly one hard link before verification.  The
verified owner then retains the hashed descriptors and nonblocking shared
``flock`` claims where the platform supplies them.  Readers reuse those
descriptors without hot-path pathname or metadata revalidation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
import unicodedata
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Self

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on non-POSIX hosts
    fcntl = None  # type: ignore[assignment]

import numpy as np

QWEN38_FLASH_NEXT_REPO = "Qwen/Qwen3.8-Flash-Next"
QWEN38_FLASH_NEXT_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
NGRAM_MANIFEST_FORMAT = "mtplx-qwen4-ngram-manifest-v1"
# The production artifact has exactly 128 n-gram shards.  At under 8 KiB of
# metadata per shard, 1 MiB leaves generous headroom without accepting an
# unbounded document at this trust boundary.
MAX_MANIFEST_BYTES = 1 * 1024 * 1024
MAX_MANIFEST_SHARDS = 128
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 4_096
MAX_JSON_COLLECTION_ITEMS = 256
MAX_JSON_STRING_CHARS = 4_096
MAX_JSON_INTEGER_DIGITS = 20

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_SEED_PRIME = 10_007
_MAX_SIGNED_INT64 = (1 << 63) - 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_STORAGES = frozenset(("bf16", "affine-q4-g32"))


class NGramManifestError(ValueError):
    """Raised when n-gram provenance, layout, or integrity is invalid."""


def _exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise NGramManifestError(f"{label} must be an exact integer")
    if value < minimum:
        raise NGramManifestError(f"{label} must be at least {minimum}")
    return value


def _valid_unicode(value: str, *, label: str) -> str:
    if len(value) > MAX_JSON_STRING_CHARS:
        raise NGramManifestError(
            f"{label} exceeds the {MAX_JSON_STRING_CHARS}-character limit"
        )
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise NGramManifestError(f"{label} contains an invalid Unicode scalar")
    return value


def _exact_string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise NGramManifestError(f"{label} must be a non-empty exact string")
    return _valid_unicode(value, label=label)


def _sha256(value: Any, *, label: str) -> str:
    digest = _exact_string(value, label=label)
    if _SHA256_RE.fullmatch(digest) is None:
        raise NGramManifestError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _multipliers(
    vocab_size: int, ngram_size: int, layer: int, seed: int
) -> tuple[int, ...]:
    multiplier_max = _MAX_SIGNED_INT64 // max(vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PLE_SEED_PRIME * layer
    return tuple(
        2
        * (
            _splitmix64((base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64)
            % half_bound
        )
        + 1
        for index in range(ngram_size)
    )


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def _token_matrix(tokens: Any, *, label: str, vocab_size: int) -> np.ndarray:
    if isinstance(tokens, np.ndarray):
        if tokens.ndim != 2:
            raise ValueError(f"{label} must have shape [batch, sequence]")
        if tokens.shape[0] == 0:
            raise ValueError(f"{label} must contain at least one batch row")
        if tokens.dtype.kind in "iu":
            if tokens.size and (
                bool(np.any(tokens < 0)) or bool(np.any(tokens >= vocab_size))
            ):
                raise ValueError(f"{label} token is outside the vocabulary range")
            return tokens.astype(np.int64, copy=False)
        if tokens.dtype.kind != "O":
            raise TypeError(f"{label} must contain exact integers")
        tokens = tokens.tolist()
    if not isinstance(tokens, (list, tuple)) or not tokens:
        raise TypeError(f"{label} must be a non-empty rectangular sequence")
    width: int | None = None
    rows: list[list[int]] = []
    for batch_index, row in enumerate(tokens):
        if not isinstance(row, (list, tuple)):
            raise TypeError(f"{label}[{batch_index}] must be a sequence")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{label} must be rectangular")
        parsed: list[int] = []
        for token in row:
            if type(token) is not int:
                raise TypeError(f"{label} must contain exact integers")
            if not 0 <= token < vocab_size:
                raise ValueError(f"{label} token is outside the vocabulary range")
            parsed.append(token)
        rows.append(parsed)
    try:
        return np.asarray(rows, dtype=np.int64)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} token is outside the vocabulary range") from exc


@dataclass(frozen=True)
class NGramGeometry:
    """Construction-validated immutable geometry for the pinned Qwen4 PLE."""

    vocab_size: int = field(default=248_320, init=False)
    eos_token_id: int = field(default=248_044, init=False)
    ngram_size: int = field(default=3, init=False)
    heads_per_ngram: int = field(default=8, init=False)
    ngram_vocab_size_base: int = field(default=20_000_000, init=False)
    divisor: int = field(default=128, init=False)
    ple_layer_index: int = field(default=0, init=False)
    seed: int = field(default=1234, init=False)
    multipliers: tuple[int, ...] = field(init=False)
    head_vocab_sizes: tuple[int, ...] = field(init=False)
    head_offsets: tuple[int, ...] = field(init=False)
    total_vocab_size: int = field(init=False)
    padded_rows: int = field(init=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("NGramGeometry is pinned and cannot be subclassed")

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "eos_token_id",
            "ngram_size",
            "heads_per_ngram",
            "ngram_vocab_size_base",
            "divisor",
            "ple_layer_index",
            "seed",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError("eos_token_id must be inside the unigram vocabulary")
        if self.ngram_size < 2 or self.heads_per_ngram < 1:
            raise ValueError("n-gram and head counts must be positive")
        if self.ngram_vocab_size_base < 2 or self.divisor < 1:
            raise ValueError("n-gram vocabulary base and divisor must be positive")
        if self.ple_layer_index < 0 or self.seed < 0:
            raise ValueError("PLE layer index and seed must be nonnegative")

        head_count = (self.ngram_size - 1) * self.heads_per_ngram
        sizes = tuple(
            _nth_prime_after(
                self.ngram_vocab_size_base - 1,
                self.ple_layer_index * head_count + head + 1,
            )
            for head in range(head_count)
        )
        offsets: list[int] = []
        total = 0
        for size in sizes:
            offsets.append(total)
            total += size
        object.__setattr__(
            self,
            "multipliers",
            _multipliers(
                self.vocab_size, self.ngram_size, self.ple_layer_index, self.seed
            ),
        )
        object.__setattr__(self, "head_vocab_sizes", sizes)
        object.__setattr__(self, "head_offsets", tuple(offsets))
        object.__setattr__(self, "total_vocab_size", total)
        object.__setattr__(
            self, "padded_rows", math.ceil(total / self.divisor) * self.divisor
        )

    @staticmethod
    def qwen38() -> NGramGeometry:
        """Return the exact revision-pinned Qwen3.8 Flash-Next geometry."""

        return NGramGeometry()

    @staticmethod
    def qwen38_flash_next() -> NGramGeometry:
        """Compatibility spelling for the pinned Qwen3.8 geometry."""

        return NGramGeometry()

    def _context_matrix(
        self, prior_context: Any, *, batch_size: int
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        if prior_context is None:
            context = tuple(
                (self.eos_token_id, self.eos_token_id) for _ in range(batch_size)
            )
            array = np.asarray(context, dtype=np.int64)
        else:
            array = _token_matrix(
                prior_context, label="prior_context", vocab_size=self.vocab_size
            )
            if array.shape[1] != 2:
                raise ValueError("prior_context must have shape [batch, 2]")
            context = tuple(tuple(int(token) for token in row) for row in array)
        if array.shape[0] != batch_size:
            raise ValueError("prior_context batch does not match new tokens")
        return array, context

    def row_ids(self, token_ids: Any) -> np.ndarray:
        """Plan global physical rows for a complete ``[B, T]`` token matrix."""

        rows, _ = self.plan_incremental(token_ids)
        return rows

    def plan_incremental(
        self,
        new_token_ids: Any,
        prior_context: Sequence[Sequence[int]] | None = None,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        """Plan only new rows and return the next raw two-token context."""

        tokens = _token_matrix(
            new_token_ids, label="new_token_ids", vocab_size=self.vocab_size
        )
        batch_size, token_count = tokens.shape
        context, old_context = self._context_matrix(
            prior_context, batch_size=batch_size
        )
        if token_count == 0:
            return np.empty(
                (batch_size, 0, len(self.head_vocab_sizes)), dtype=np.int64
            ), old_context

        history = np.concatenate((context, tokens), axis=1)
        sequence_length = history.shape[1]
        positions = np.arange(sequence_length, dtype=np.int64)
        eos_positions = np.where(history == self.eos_token_id, positions[None, :], -1)
        previous_eos_inclusive = np.maximum.accumulate(eos_positions, axis=1)
        previous_eos = np.concatenate(
            (
                np.full((batch_size, 1), -1, dtype=np.int64),
                previous_eos_inclusive[:, :-1],
            ),
            axis=1,
        )
        position_in_segment = positions[None, :] - (previous_eos + 1)

        shifted: list[np.ndarray] = []
        for shift in range(self.ngram_size):
            source_positions = positions - shift
            gathered = history[:, np.maximum(source_positions, 0)]
            valid = (position_in_segment >= shift) & (source_positions[None, :] >= 0)
            shifted.append(np.where(valid, gathered, self.eos_token_id))

        blocks: list[np.ndarray] = []
        for ngram in range(2, self.ngram_size + 1):
            mixed = shifted[0] * np.int64(self.multipliers[0])
            for position in range(1, ngram):
                mixed = np.bitwise_xor(
                    mixed,
                    shifted[position] * np.int64(self.multipliers[position]),
                )
            start = (ngram - 2) * self.heads_per_ngram
            stop = start + self.heads_per_ngram
            sizes = np.asarray(self.head_vocab_sizes[start:stop], dtype=np.int64)
            offsets = np.asarray(self.head_offsets[start:stop], dtype=np.int64)
            blocks.append(np.remainder(mixed[..., None], sizes) + offsets)

        planned = np.concatenate(blocks, axis=-1)[:, -token_count:]
        combined = np.concatenate((context, tokens), axis=1)
        next_context = tuple(
            tuple(int(value) for value in row[-2:]) for row in combined
        )
        return planned, next_context


def _safe_component(name: Any, *, label: str) -> str:
    value = _exact_string(name, label=label)
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise NGramManifestError(f"unsafe {label}: {value!r}")
    return value


def _scan_json_bounds(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    structural_nodes = 1
    collection_items: list[int] = []
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            structural_nodes += 1
            collection_items.append(1)
            if depth > MAX_JSON_DEPTH:
                raise NGramManifestError(
                    f"manifest JSON exceeds the depth limit {MAX_JSON_DEPTH}"
                )
        elif character in "]}":
            depth -= 1
            if collection_items:
                collection_items.pop()
        elif character == ",":
            structural_nodes += 1
            if collection_items:
                collection_items[-1] += 1
                if collection_items[-1] > MAX_JSON_COLLECTION_ITEMS:
                    raise NGramManifestError(
                        "manifest JSON collection exceeds the item limit"
                    )
        if structural_nodes > MAX_JSON_NODES:
            raise NGramManifestError(
                f"manifest JSON exceeds the {MAX_JSON_NODES}-node limit"
            )


def _bounded_json_int(value: str) -> int:
    digits = value.removeprefix("-")
    if len(digits) > MAX_JSON_INTEGER_DIGITS:
        raise NGramManifestError(
            f"manifest integer exceeds the {MAX_JSON_INTEGER_DIGITS}-digit limit"
        )
    try:
        return int(value)
    except ValueError as exc:
        raise NGramManifestError("manifest contains an invalid integer") from exc


def _reject_json_constant(value: str) -> None:
    raise NGramManifestError(f"manifest contains invalid constant {value!r}")


def _validate_json_tree(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise NGramManifestError(
                f"manifest JSON exceeds the {MAX_JSON_NODES}-node limit"
            )
        if depth > MAX_JSON_DEPTH:
            raise NGramManifestError(
                f"manifest JSON exceeds the depth limit {MAX_JSON_DEPTH}"
            )
        if type(current) is dict:
            if len(current) > MAX_JSON_COLLECTION_ITEMS:
                raise NGramManifestError("manifest JSON object exceeds the item limit")
            for key, item in current.items():
                if type(key) is not str:
                    raise NGramManifestError("manifest JSON object key must be a string")
                _valid_unicode(key, label="manifest JSON key")
                stack.append((item, depth + 1))
        elif type(current) in {list, tuple}:
            if len(current) > MAX_JSON_COLLECTION_ITEMS:
                raise NGramManifestError("manifest JSON array exceeds the item limit")
            stack.extend((item, depth + 1) for item in current)
        elif type(current) is str:
            _valid_unicode(current, label="manifest JSON string")
        elif current is None or type(current) in {bool, int, float}:
            continue
        else:
            raise NGramManifestError(
                f"manifest JSON contains unsupported type {type(current).__name__}"
            )


def _canonical_json(value: Any) -> bytes:
    _validate_json_tree(value)
    try:
        payload = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise NGramManifestError(f"could not canonicalize manifest JSON: {exc}") from exc
    if len(payload) > MAX_MANIFEST_BYTES:
        raise NGramManifestError(
            f"manifest exceeds the {MAX_MANIFEST_BYTES}-byte limit"
        )
    return payload


@dataclass(frozen=True)
class NGramShard:
    name: str
    tensor: str
    start_row: int
    row_count: int
    data_offset: int
    data_bytes: int
    file_size: int
    sha256: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _safe_component(self.name, label="shard name")
        _exact_string(self.tensor, label="shard tensor")
        _exact_int(self.start_row, label="shard start_row")
        _exact_int(self.row_count, label="shard row_count", minimum=1)
        _exact_int(self.data_offset, label="shard data_offset")
        _exact_int(self.data_bytes, label="shard data_bytes", minimum=1)
        _exact_int(self.file_size, label="shard file_size", minimum=1)
        _sha256(self.sha256, label="shard sha256")
        if self.data_offset + self.data_bytes > self.file_size:
            raise NGramManifestError("shard payload exceeds its exact file size")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tensor": self.tensor,
            "start_row": self.start_row,
            "row_count": self.row_count,
            "data_offset": self.data_offset,
            "data_bytes": self.data_bytes,
            "file_size": self.file_size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> NGramShard:
        obj = _object(value, label="shard")
        _keys(
            obj,
            label="shard",
            required=(
                "name",
                "tensor",
                "start_row",
                "row_count",
                "data_offset",
                "data_bytes",
                "file_size",
                "sha256",
            ),
        )
        shard = cls(
            name=_safe_component(obj["name"], label="shard name"),
            tensor=_exact_string(obj["tensor"], label="shard tensor"),
            start_row=_exact_int(obj["start_row"], label="shard start_row"),
            row_count=_exact_int(obj["row_count"], label="shard row_count", minimum=1),
            data_offset=_exact_int(obj["data_offset"], label="shard data_offset"),
            data_bytes=_exact_int(
                obj["data_bytes"], label="shard data_bytes", minimum=1
            ),
            file_size=_exact_int(obj["file_size"], label="shard file_size", minimum=1),
            sha256=_sha256(obj["sha256"], label="shard sha256"),
        )
        shard.validate()
        return shard


@dataclass(frozen=True)
class NGramManifest:
    source_repo: str
    source_revision: str
    storage: Literal["bf16", "affine-q4-g32"]
    row_width: int
    row_bytes: int
    padded_rows: int
    shards: tuple[NGramShard, ...]
    digest: str | None = None
    format: str = NGRAM_MANIFEST_FORMAT

    def __post_init__(self) -> None:
        self.validate_structure()

    def validate_structure(self) -> None:
        if type(self.format) is not str or self.format != NGRAM_MANIFEST_FORMAT:
            raise NGramManifestError(f"unsupported manifest format {self.format!r}")
        _exact_string(self.source_repo, label="source_repo")
        _exact_string(self.source_revision, label="source_revision")
        if type(self.storage) is not str or self.storage not in _STORAGES:
            raise NGramManifestError(f"unsupported n-gram storage {self.storage!r}")
        _exact_int(self.row_width, label="row_width", minimum=1)
        _exact_int(self.row_bytes, label="row_bytes", minimum=1)
        _exact_int(self.padded_rows, label="padded_rows", minimum=1)
        if type(self.shards) is not tuple or not self.shards:
            raise NGramManifestError("shards must be a non-empty exact tuple")
        if len(self.shards) > MAX_MANIFEST_SHARDS:
            raise NGramManifestError("manifest shard count exceeds the limit")
        expected_row_bytes = self.row_width * 2
        if self.storage == "affine-q4-g32":
            if self.row_width % 32:
                raise NGramManifestError("affine-q4-g32 row width must divide by 32")
            expected_row_bytes = self.row_width // 2 + (self.row_width // 32) * 4
        if self.row_bytes != expected_row_bytes:
            raise NGramManifestError(
                f"{self.storage} row_bytes must be {expected_row_bytes}"
            )
        if self.digest is not None:
            _sha256(self.digest, label="manifest digest")

        cursor = 0
        names: set[str] = set()
        tensors: set[str] = set()
        for shard in self.shards:
            if type(shard) is not NGramShard:
                raise NGramManifestError("shards must contain exact NGramShard values")
            shard.validate()
            if shard.name in names or shard.tensor in tensors:
                raise NGramManifestError(
                    "manifest contains a duplicate shard or tensor"
                )
            names.add(shard.name)
            tensors.add(shard.tensor)
            if shard.start_row != cursor:
                relation = "overlap" if shard.start_row < cursor else "gap"
                raise NGramManifestError(f"manifest row coverage has a {relation}")
            if shard.data_bytes != shard.row_count * self.row_bytes:
                raise NGramManifestError(
                    "shard data_bytes must equal row_count * row_bytes"
                )
            cursor += shard.row_count
        if cursor != self.padded_rows:
            raise NGramManifestError("manifest shards do not cover all padded rows")

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format": self.format,
            "source_repo": self.source_repo,
            "source_revision": self.source_revision,
            "storage": self.storage,
            "row_width": self.row_width,
            "row_bytes": self.row_bytes,
            "padded_rows": self.padded_rows,
            "shards": [shard.to_dict() for shard in self.shards],
        }
        if include_digest and self.digest is not None:
            result["digest"] = self.digest
        return result

    def with_digest(self) -> NGramManifest:
        self.validate_structure()
        digest = hashlib.sha256(
            _canonical_json(self.to_dict(include_digest=False))
        ).hexdigest()
        return replace(self, digest=digest)

    @classmethod
    def from_dict(cls, value: Any, *, verify_digest: bool = True) -> NGramManifest:
        obj = _object(value, label="manifest")
        _keys(
            obj,
            label="manifest",
            required=(
                "format",
                "source_repo",
                "source_revision",
                "storage",
                "row_width",
                "row_bytes",
                "padded_rows",
                "shards",
            ),
            optional=("digest",),
        )
        raw_shards = obj["shards"]
        if type(raw_shards) is not list:
            raise NGramManifestError("shards must be an array")
        if len(raw_shards) > MAX_MANIFEST_SHARDS:
            raise NGramManifestError("manifest shard count exceeds the limit")
        digest = obj.get("digest")
        manifest = cls(
            format=_exact_string(obj["format"], label="format"),
            source_repo=_exact_string(obj["source_repo"], label="source_repo"),
            source_revision=_exact_string(
                obj["source_revision"], label="source_revision"
            ),
            storage=_exact_string(obj["storage"], label="storage"),  # type: ignore[arg-type]
            row_width=_exact_int(obj["row_width"], label="row_width", minimum=1),
            row_bytes=_exact_int(obj["row_bytes"], label="row_bytes", minimum=1),
            padded_rows=_exact_int(obj["padded_rows"], label="padded_rows", minimum=1),
            shards=tuple(NGramShard.from_dict(item) for item in raw_shards),
            digest=None if digest is None else _sha256(digest, label="manifest digest"),
        )
        manifest.validate_structure()
        if verify_digest:
            if manifest.digest is None:
                raise NGramManifestError("manifest digest is required")
            if manifest.digest != manifest.with_digest().digest:
                raise NGramManifestError("manifest digest mismatch")
        return manifest

    def locate_row(self, row: int) -> tuple[NGramShard, int]:
        """Return the owning shard and exact byte offset for a global row."""

        if type(row) is not int:
            raise TypeError("row must be an exact integer")
        if not 0 <= row < self.padded_rows:
            raise IndexError(f"n-gram row {row} is out of range")
        starts = tuple(shard.start_row for shard in self.shards)
        shard = self.shards[bisect_right(starts, row) - 1]
        return shard, shard.data_offset + (row - shard.start_row) * self.row_bytes


def _object(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise NGramManifestError(f"{label} must be an object")
    return value


def _keys(
    value: dict[str, Any],
    *,
    label: str,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = required_set - set(value)
    unknown = set(value) - allowed
    if missing:
        raise NGramManifestError(f"{label} is missing keys: {sorted(missing)}")
    if unknown:
        raise NGramManifestError(f"{label} has unknown keys: {sorted(unknown)}")


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NGramManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _readonly_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _read_manifest(path: Path) -> bytes:
    fd: int | None = None
    try:
        fd = os.open(path, _readonly_flags())
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise NGramManifestError(f"manifest is not a regular file: {path}")
        if metadata.st_size > MAX_MANIFEST_BYTES:
            raise NGramManifestError("manifest exceeds the byte limit")
        chunks: list[bytes] = []
        remaining = MAX_MANIFEST_BYTES + 1
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > MAX_MANIFEST_BYTES:
            raise NGramManifestError("manifest exceeds the byte limit")
        return payload
    except NGramManifestError:
        raise
    except OSError as exc:
        raise NGramManifestError(f"could not read manifest {path}: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)


def load_ngram_manifest(
    path: Path | str, *, verify_digest: bool = True
) -> NGramManifest:
    payload = _read_manifest(Path(path))
    try:
        text = payload.decode("utf-8")
        _scan_json_bounds(text)
        value = json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_int=_bounded_json_int,
            parse_constant=_reject_json_constant,
        )
        _validate_json_tree(value)
    except NGramManifestError:
        raise
    except (RecursionError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise NGramManifestError(f"invalid manifest JSON: {exc}") from exc
    return NGramManifest.from_dict(value, verify_digest=verify_digest)


def save_ngram_manifest(manifest: NGramManifest, path: Path | str) -> NGramManifest:
    if type(manifest) is not NGramManifest:
        raise TypeError("manifest must be an exact NGramManifest")
    finalized = manifest.with_digest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = (
            json.dumps(
                finalized.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n"
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise NGramManifestError(f"could not encode manifest JSON: {exc}") from exc
    if len(payload) > MAX_MANIFEST_BYTES:
        raise NGramManifestError(
            f"manifest exceeds the {MAX_MANIFEST_BYTES}-byte limit"
        )
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise NGramManifestError(f"could not save manifest {target}: {exc}") from exc
    return finalized


def qwen38_ngram_manifest(
    storage: Literal["bf16", "affine-q4-g32"],
    shards: tuple[NGramShard, ...],
) -> NGramManifest:
    """Build a revision- and geometry-pinned Qwen3.8 n-gram manifest."""

    row_bytes = 320 if storage == "bf16" else 100
    manifest = NGramManifest(
        source_repo=QWEN38_FLASH_NEXT_REPO,
        source_revision=QWEN38_FLASH_NEXT_REVISION,
        storage=storage,
        row_width=160,
        row_bytes=row_bytes,
        padded_rows=320_001_536,
        shards=shards,
    )
    return manifest.with_digest()


@dataclass(frozen=True)
class NGramFileIdentity:
    """Stable identity captured from a verified open shard descriptor."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


class VerifiedNGramShard:
    """One verified immutable shard whose descriptor remains authoritative."""

    __slots__ = ("_fd", "_lock", "identity", "shard")

    def __init__(
        self, shard: NGramShard, descriptor: int, identity: NGramFileIdentity
    ) -> None:
        self.shard = shard
        self.identity = identity
        self._fd = descriptor
        self._lock = threading.RLock()

    def __copy__(self) -> None:
        raise TypeError("copy is forbidden for verified descriptor owners")

    def __deepcopy__(self, memo: dict[int, Any]) -> None:
        del memo
        raise TypeError("deepcopy is forbidden for verified descriptor owners")

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._fd < 0

    def fileno(self) -> int:
        with self._lock:
            if self.closed:
                raise NGramManifestError(
                    f"verified shard {self.shard.name} is closed"
                )
            return self._fd

    def pread(self, offset: int, length: int) -> bytes:
        """Read an exact descriptor-relative byte range from the retained file."""

        _exact_int(offset, label="pread offset")
        _exact_int(length, label="pread length")
        if offset + length > self.identity.size:
            raise NGramManifestError("pread range exceeds the verified shard size")
        with self._lock:
            descriptor = self.fileno()
            chunks: list[bytes] = []
            remaining = length
            cursor = offset
            try:
                while remaining:
                    chunk = os.pread(descriptor, remaining, cursor)
                    if not chunk:
                        raise NGramManifestError(
                            f"short descriptor read from shard {self.shard.name}"
                        )
                    chunks.append(chunk)
                    remaining -= len(chunk)
                    cursor += len(chunk)
            except NGramManifestError:
                raise
            except OSError as exc:
                raise NGramManifestError(
                    f"could not read verified shard {self.shard.name}: {exc}"
                ) from exc
            return b"".join(chunks)

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            descriptor = self._fd
            self._fd = -1
            try:
                os.close(descriptor)
            except OSError as exc:
                raise NGramManifestError(
                    f"could not close verified shard {self.shard.name}: {exc}"
                ) from exc


class VerifiedNGramArtifact:
    """Retained directory and shard descriptors for one verified artifact."""

    __slots__ = ("_lock", "_root_fd", "manifest", "report", "shards")

    def __init__(
        self,
        manifest: NGramManifest,
        root_descriptor: int,
        shards: tuple[VerifiedNGramShard, ...],
    ) -> None:
        self.manifest = manifest
        self._root_fd = root_descriptor
        self._lock = threading.RLock()
        self.shards = shards
        self.report = {
            "shards": len(shards),
            "rows": manifest.padded_rows,
            "bytes": sum(shard.shard.data_bytes for shard in shards),
        }

    def __copy__(self) -> None:
        raise TypeError("copy is forbidden for verified descriptor owners")

    def __deepcopy__(self, memo: dict[int, Any]) -> None:
        del memo
        raise TypeError("deepcopy is forbidden for verified descriptor owners")

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._root_fd < 0

    def root_fileno(self) -> int:
        with self._lock:
            if self.closed:
                raise NGramManifestError("verified n-gram artifact is closed")
            return self._root_fd

    def pread(self, shard_index: int, offset: int, length: int) -> bytes:
        _exact_int(shard_index, label="shard index")
        if shard_index >= len(self.shards):
            raise NGramManifestError("shard index is out of range")
        return self.shards[shard_index].pread(offset, length)

    def read_row(self, row: int) -> bytes:
        shard, offset = self.manifest.locate_row(row)
        for verified in self.shards:
            if verified.shard is shard or verified.shard.name == shard.name:
                return verified.pread(offset, self.manifest.row_bytes)
        raise NGramManifestError(f"verified artifact has no shard for row {row}")

    def close(self) -> None:
        with self._lock:
            first_error: NGramManifestError | None = None
            for shard in self.shards:
                try:
                    shard.close()
                except NGramManifestError as exc:
                    if first_error is None:
                        first_error = exc
            if not self.closed:
                descriptor = self._root_fd
                self._root_fd = -1
                try:
                    os.close(descriptor)
                except OSError as exc:
                    if first_error is None:
                        first_error = NGramManifestError(
                            f"could not close verified artifact directory: {exc}"
                        )
            if first_error is not None:
                raise first_error

    def __enter__(self) -> Self:
        if self.closed:
            raise NGramManifestError("verified n-gram artifact is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

def _directory_flags() -> int:
    return _readonly_flags() | getattr(os, "O_DIRECTORY", 0)


def _open_root_nofollow(root: Path | str) -> int:
    try:
        raw_root = os.fspath(root)
    except TypeError as exc:
        raise NGramManifestError(f"invalid artifact root: {exc}") from exc
    if type(raw_root) is not str or "\x00" in raw_root:
        raise NGramManifestError("artifact root must be a valid filesystem string")
    try:
        absolute = os.path.abspath(raw_root)
    except OSError as exc:
        raise NGramManifestError(f"could not normalize artifact root: {exc}") from exc
    components = [component for component in absolute.split(os.sep) if component]
    opened: list[int] = []
    try:
        current = os.open(os.sep, _directory_flags())
        opened.append(current)
        for component in components:
            current = os.open(
                component,
                _directory_flags(),
                dir_fd=current,
            )
            opened.append(current)
        metadata = os.fstat(current)
        if not stat.S_ISDIR(metadata.st_mode):
            raise NGramManifestError("artifact root is not a directory")
        for index in range(len(opened) - 1):
            os.close(opened[index])
            opened[index] = -1
        result = opened[-1]
        opened[-1] = -1
        return result
    except NGramManifestError:
        raise
    except OSError as exc:
        raise NGramManifestError(f"could not open artifact root {root}: {exc}") from exc
    finally:
        for descriptor in opened:
            if descriptor < 0:
                continue
            try:
                os.close(descriptor)
            except OSError:
                pass


def _hash_descriptor_payload(descriptor: int, shard: NGramShard) -> str:
    digest = hashlib.sha256()
    remaining = shard.data_bytes
    offset = shard.data_offset
    try:
        while remaining:
            chunk = os.pread(descriptor, min(8 * 1024 * 1024, remaining), offset)
            if not chunk:
                raise NGramManifestError(f"short payload in shard {shard.name}")
            digest.update(chunk)
            remaining -= len(chunk)
            offset += len(chunk)
    except NGramManifestError:
        raise
    except OSError as exc:
        raise NGramManifestError(
            f"could not hash shard {shard.name}: {exc}"
        ) from exc
    return digest.hexdigest()


def _require_verification_capabilities() -> None:
    """Fail closed before artifact access if immutable-FD primitives are absent."""

    missing: list[str] = []
    if fcntl is None:
        missing.append("fcntl")
    else:
        if not callable(getattr(fcntl, "flock", None)):
            missing.append("fcntl.flock")
        for name in ("LOCK_SH", "LOCK_NB", "LOCK_UN"):
            value = getattr(fcntl, name, None)
            if type(value) is not int:
                missing.append(f"fcntl.{name}")
    for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC"):
        value = getattr(os, name, None)
        if type(value) is not int or value == 0:
            missing.append(f"os.{name}")
    if type(getattr(os, "O_RDONLY", None)) is not int:
        missing.append("os.O_RDONLY")
    open_function = getattr(os, "open", None)
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    try:
        supports_descriptor_open = open_function in supports_dir_fd
    except TypeError:
        supports_descriptor_open = False
    if not callable(open_function) or not supports_descriptor_open:
        missing.append("descriptor-relative os.open")
    if not callable(getattr(os, "pread", None)):
        missing.append("os.pread")
    if not callable(getattr(os, "close", None)):
        missing.append("os.close")
    if not callable(getattr(os, "fstat", None)):
        missing.append("os.fstat")
    if missing:
        raise NGramManifestError(
            "n-gram verification lacks required platform capabilities: "
            + ", ".join(missing)
        )


def verify_ngram_manifest(
    root: Path | str, manifest: NGramManifest
) -> VerifiedNGramArtifact:
    """Verify and retain immutable, descriptor-anchored shard ownership.

    Installation must have closed every writer, removed all write permission
    bits, and reduced each shard to one hard link.  Verification acquires and
    retains a nonblocking shared ``flock`` on platforms that provide it.  A
    cooperating installer or mutator must acquire the exclusive counterpart;
    serving performs no repeated metadata checks.
    """

    _require_verification_capabilities()
    if type(manifest) is not NGramManifest:
        raise TypeError("manifest must be an exact NGramManifest")
    manifest.validate_structure()
    if manifest.digest is None or manifest.digest != manifest.with_digest().digest:
        raise NGramManifestError("manifest digest mismatch")
    root_descriptor: int | None = None
    retained: list[VerifiedNGramShard] = []
    try:
        root_descriptor = _open_root_nofollow(root)
        for shard in manifest.shards:
            name = _safe_component(shard.name, label="shard name")
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    name, _readonly_flags(), dir_fd=root_descriptor
                )
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise NGramManifestError(
                        f"shard is not a regular file: {shard.name}"
                    )
                if metadata.st_nlink != 1:
                    raise NGramManifestError(
                        f"shard {shard.name} must have exactly one hard link"
                    )
                if metadata.st_mode & (
                    stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                ):
                    raise NGramManifestError(
                        f"shard {shard.name} has write permission bits; "
                        "installation must finalize it read-only"
                    )
                if metadata.st_size != shard.file_size:
                    raise NGramManifestError(
                        f"shard size mismatch for {shard.name}: "
                        f"expected {shard.file_size}, got {metadata.st_size}"
                    )
                if fcntl is not None:
                    try:
                        fcntl.flock(
                            descriptor,
                            fcntl.LOCK_SH | fcntl.LOCK_NB,
                        )
                    except OSError as exc:
                        raise NGramManifestError(
                            f"could not acquire immutable shared lock for "
                            f"{shard.name}: {exc}"
                        ) from exc
                payload_digest = _hash_descriptor_payload(descriptor, shard)
                verified_metadata = os.fstat(descriptor)
                if (
                    verified_metadata.st_dev != metadata.st_dev
                    or verified_metadata.st_ino != metadata.st_ino
                    or verified_metadata.st_size != metadata.st_size
                    or verified_metadata.st_mode != metadata.st_mode
                    or verified_metadata.st_nlink != 1
                    or verified_metadata.st_mtime_ns != metadata.st_mtime_ns
                    or verified_metadata.st_ctime_ns != metadata.st_ctime_ns
                ):
                    raise NGramManifestError(
                        f"shard identity changed during verification: {shard.name}"
                    )
                if payload_digest != shard.sha256:
                    raise NGramManifestError(
                        f"payload digest mismatch for {shard.name}"
                    )
                retained.append(
                    VerifiedNGramShard(
                        shard,
                        descriptor,
                        NGramFileIdentity(
                            device=verified_metadata.st_dev,
                            inode=verified_metadata.st_ino,
                            size=verified_metadata.st_size,
                            mtime_ns=verified_metadata.st_mtime_ns,
                            ctime_ns=verified_metadata.st_ctime_ns,
                        ),
                    )
                )
                descriptor = None
            finally:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError as exc:
                        raise NGramManifestError(
                            f"could not close shard {shard.name}: {exc}"
                        ) from exc
        artifact = VerifiedNGramArtifact(
            manifest, root_descriptor, tuple(retained)
        )
        root_descriptor = None
        retained = []
        return artifact
    except NGramManifestError:
        raise
    except OSError as exc:
        raise NGramManifestError(f"could not verify n-gram artifact: {exc}") from exc
    finally:
        cleanup_error: NGramManifestError | None = None
        for verified in retained:
            try:
                verified.close()
            except NGramManifestError as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if root_descriptor is not None:
            try:
                os.close(root_descriptor)
            except OSError as exc:
                if cleanup_error is None:
                    cleanup_error = NGramManifestError(
                        f"could not close artifact root: {exc}"
                    )
        if cleanup_error is not None:
            raise cleanup_error


__all__ = [
    "QWEN38_FLASH_NEXT_REPO",
    "QWEN38_FLASH_NEXT_REVISION",
    "NGramFileIdentity",
    "NGramGeometry",
    "NGramManifest",
    "NGramManifestError",
    "NGramShard",
    "VerifiedNGramArtifact",
    "VerifiedNGramShard",
    "load_ngram_manifest",
    "qwen38_ngram_manifest",
    "save_ngram_manifest",
    "verify_ngram_manifest",
]
