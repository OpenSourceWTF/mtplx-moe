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
from array import array
from bisect import bisect_right
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field, replace
from heapq import nsmallest
from pathlib import Path
from time import monotonic
from typing import Any, Literal, Self

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on non-POSIX hosts
    fcntl = None  # type: ignore[assignment]

import numpy as np

QWEN38_FLASH_NEXT_REPO = "Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP"
QWEN38_FLASH_NEXT_REVISION = "43a82b3f0ff64fa417fd09ca046580f08d19b0d6"
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
_GIB = 1024**3
PRODUCTION_RUNTIME_TARGET_BYTES = 95 * _GIB
PRODUCTION_NGRAM_PAYLOAD_CEILING_BYTES = 10 * _GIB


class NGramManifestError(ValueError):
    """Raised when n-gram provenance, layout, or integrity is invalid."""


class NGramCacheError(RuntimeError):
    """Base error for the fixed exact-row cache."""


class NGramCacheClosed(NGramCacheError):
    """Raised when a cache or lease is no longer usable."""


class NGramCacheFull(NGramCacheError):
    """Raised when a complete request cannot be reserved atomically."""


class NGramCacheIOError(NGramCacheError):
    """Raised when an exact descriptor-relative read fails."""


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
class NGramComponent:
    """One exact tensor segment backing a source-native affine row."""

    component: Literal["weight", "scales", "biases"]
    name: str
    tensor: str
    data_offset: int
    row_bytes: int
    data_bytes: int
    file_size: int
    file_sha256: str
    dtype: Literal["U32", "BF16"]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.component) is not str or self.component not in {
            "weight",
            "scales",
            "biases",
        }:
            raise NGramManifestError("unsupported affine n-gram component")
        _safe_component(self.name, label="component shard name")
        _exact_string(self.tensor, label="component tensor")
        _exact_int(self.data_offset, label="component data_offset")
        _exact_int(self.row_bytes, label="component row_bytes", minimum=1)
        _exact_int(self.data_bytes, label="component data_bytes", minimum=1)
        _exact_int(self.file_size, label="component file_size", minimum=1)
        _sha256(self.file_sha256, label="component file_sha256")
        expected_dtype = "U32" if self.component == "weight" else "BF16"
        if type(self.dtype) is not str or self.dtype != expected_dtype:
            raise NGramManifestError(
                f"{self.component} component dtype must be {expected_dtype}"
            )
        if (
            type(self.shape) is not tuple
            or len(self.shape) != 2
            or any(type(size) is not int or size <= 0 for size in self.shape)
        ):
            raise NGramManifestError(
                "component shape must be an exact pair of positive integers"
            )
        dtype_bytes = 4 if self.dtype == "U32" else 2
        if self.row_bytes != self.shape[1] * dtype_bytes:
            raise NGramManifestError("component row_bytes does not match its shape")
        if self.data_bytes != self.shape[0] * self.row_bytes:
            raise NGramManifestError("component data_bytes does not match its shape")
        if self.data_offset + self.data_bytes > self.file_size:
            raise NGramManifestError("component payload exceeds its exact file size")

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "name": self.name,
            "tensor": self.tensor,
            "data_offset": self.data_offset,
            "row_bytes": self.row_bytes,
            "data_bytes": self.data_bytes,
            "file_size": self.file_size,
            "file_sha256": self.file_sha256,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, value: Any) -> NGramComponent:
        obj = _object(value, label="component")
        _keys(
            obj,
            label="component",
            required=(
                "component",
                "name",
                "tensor",
                "data_offset",
                "row_bytes",
                "data_bytes",
                "file_size",
                "file_sha256",
                "dtype",
                "shape",
            ),
        )
        raw_shape = obj["shape"]
        if type(raw_shape) is not list or len(raw_shape) != 2:
            raise NGramManifestError("component shape must contain two integers")
        return cls(
            component=_exact_string(  # type: ignore[arg-type]
                obj["component"], label="component kind"
            ),
            name=_safe_component(obj["name"], label="component shard name"),
            tensor=_exact_string(obj["tensor"], label="component tensor"),
            data_offset=_exact_int(obj["data_offset"], label="component data_offset"),
            row_bytes=_exact_int(
                obj["row_bytes"], label="component row_bytes", minimum=1
            ),
            data_bytes=_exact_int(
                obj["data_bytes"], label="component data_bytes", minimum=1
            ),
            file_size=_exact_int(
                obj["file_size"], label="component file_size", minimum=1
            ),
            file_sha256=_sha256(
                obj["file_sha256"], label="component file_sha256"
            ),
            dtype=_exact_string(obj["dtype"], label="component dtype"),  # type: ignore[arg-type]
            shape=(
                _exact_int(raw_shape[0], label="component shape[0]", minimum=1),
                _exact_int(raw_shape[1], label="component shape[1]", minimum=1),
            ),
        )


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
    components: tuple[NGramComponent, ...] = ()

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
        if type(self.components) is not tuple:
            raise NGramManifestError("shard components must be an exact tuple")
        if self.components:
            if tuple(component.component for component in self.components) != (
                "weight",
                "scales",
                "biases",
            ):
                raise NGramManifestError(
                    "segmented shard components must be weight, scales, biases"
                )
            for component in self.components:
                if type(component) is not NGramComponent:
                    raise NGramManifestError(
                        "shard components must contain exact NGramComponent values"
                    )
                component.validate()
                if component.shape[0] != self.row_count:
                    raise NGramManifestError(
                        "component row count does not match its logical shard"
                    )
            expected_bytes = self.row_count * sum(
                component.row_bytes for component in self.components
            )
            if self.data_offset != 0 or self.data_bytes != expected_bytes:
                raise NGramManifestError(
                    "segmented shard logical byte range does not match its components"
                )
            if self.file_size != self.data_bytes:
                raise NGramManifestError(
                    "segmented shard logical file_size must equal data_bytes"
                )
            component_digest = hashlib.sha256(
                _canonical_json(
                    [component.to_dict() for component in self.components]
                )
            ).hexdigest()
            if self.sha256 != component_digest:
                raise NGramManifestError("segmented shard component digest mismatch")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "tensor": self.tensor,
            "start_row": self.start_row,
            "row_count": self.row_count,
            "data_offset": self.data_offset,
            "data_bytes": self.data_bytes,
            "file_size": self.file_size,
            "sha256": self.sha256,
        }
        if self.components:
            result["components"] = [
                component.to_dict() for component in self.components
            ]
        return result

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
            optional=("components",),
        )
        raw_components = obj.get("components", [])
        if type(raw_components) is not list:
            raise NGramManifestError("shard components must be an array")
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
            components=tuple(
                NGramComponent.from_dict(component) for component in raw_components
            ),
        )
        shard.validate()
        return shard


def segmented_ngram_shard(
    *,
    name: str,
    tensor: str,
    start_row: int,
    row_count: int,
    components: tuple[NGramComponent, ...],
) -> NGramShard:
    """Create one logical row range backed by exact safetensors components."""

    if type(components) is not tuple:
        raise NGramManifestError("shard components must be an exact tuple")
    if any(type(component) is not NGramComponent for component in components):
        raise NGramManifestError(
            "shard components must contain exact NGramComponent values"
        )
    data_bytes = row_count * sum(component.row_bytes for component in components)
    digest = hashlib.sha256(
        _canonical_json([component.to_dict() for component in components])
    ).hexdigest()
    return NGramShard(
        name=name,
        tensor=tensor,
        start_row=start_row,
        row_count=row_count,
        data_offset=0,
        data_bytes=data_bytes,
        file_size=data_bytes,
        sha256=digest,
        components=components,
    )


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
        segmented_layout: bool | None = None
        component_tensors: set[str] = set()
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
            current_segmented = bool(shard.components)
            if segmented_layout is None:
                segmented_layout = current_segmented
            elif segmented_layout != current_segmented:
                raise NGramManifestError(
                    "manifest cannot mix contiguous and segmented shards"
                )
            if shard.components:
                if self.storage != "affine-q4-g32":
                    raise NGramManifestError(
                        "segmented n-gram shards require affine-q4-g32 storage"
                    )
                if sum(
                    component.row_bytes for component in shard.components
                ) != self.row_bytes:
                    raise NGramManifestError(
                        "segmented component row bytes do not match the manifest"
                    )
                for component in shard.components:
                    if component.tensor in component_tensors:
                        raise NGramManifestError(
                            "manifest contains a duplicate component tensor"
                        )
                    component_tensors.add(component.tensor)
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


class _VerifiedNGramFile:
    """One unique source safetensors descriptor shared by logical shards."""

    __slots__ = ("_fd", "_lock", "identity", "name")

    def __init__(
        self,
        name: str,
        descriptor: int,
        identity: NGramFileIdentity,
    ) -> None:
        self.name = name
        self.identity = identity
        self._fd = descriptor
        self._lock = threading.RLock()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._fd < 0

    def fileno(self) -> int:
        with self._lock:
            if self.closed:
                raise NGramManifestError(
                    f"verified source file {self.name} is closed"
                )
            return self._fd

    def pread(self, offset: int, length: int) -> bytes:
        if offset + length > self.identity.size:
            raise NGramManifestError("pread range exceeds the verified source file")
        descriptor = self.fileno()
        chunks: list[bytes] = []
        remaining = length
        cursor = offset
        while remaining:
            try:
                chunk = os.pread(descriptor, remaining, cursor)
            except OSError as exc:
                raise NGramManifestError(
                    f"could not read verified source file {self.name}: {exc}"
                ) from exc
            if not chunk:
                raise NGramManifestError(
                    f"short descriptor read from source file {self.name}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
            cursor += len(chunk)
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
                    f"could not close verified source file {self.name}: {exc}"
                ) from exc


@dataclass(frozen=True)
class _VerifiedNGramComponent:
    component: NGramComponent
    source: _VerifiedNGramFile


class VerifiedNGramShard:
    """One verified immutable shard whose descriptor remains authoritative."""

    __slots__ = (
        "_components",
        "_fd",
        "_lock",
        "_segmented_closed",
        "identity",
        "shard",
    )

    def __init__(
        self,
        shard: NGramShard,
        descriptor: int,
        identity: NGramFileIdentity,
        *,
        components: tuple[_VerifiedNGramComponent, ...] = (),
    ) -> None:
        self.shard = shard
        self.identity = identity
        self._fd = descriptor
        self._components = components
        self._segmented_closed = False
        self._lock = threading.RLock()

    def __copy__(self) -> None:
        raise TypeError("copy is forbidden for verified descriptor owners")

    def __deepcopy__(self, memo: dict[int, Any]) -> None:
        del memo
        raise TypeError("deepcopy is forbidden for verified descriptor owners")

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._segmented_closed if self._components else self._fd < 0

    def fileno(self) -> int:
        with self._lock:
            if self._components:
                raise NGramManifestError(
                    f"segmented shard {self.shard.name} has no single descriptor"
                )
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

    def read_row(self, local_row: int) -> bytes:
        _exact_int(local_row, label="local row")
        if not 0 <= local_row < self.shard.row_count:
            raise NGramManifestError("local row is outside its logical shard")
        if not self._components:
            row_bytes = self.shard.data_bytes // self.shard.row_count
            return self.pread(
                self.shard.data_offset + local_row * row_bytes,
                row_bytes,
            )
        with self._lock:
            if self.closed:
                raise NGramManifestError(
                    f"verified shard {self.shard.name} is closed"
                )
            return b"".join(
                verified.source.pread(
                    verified.component.data_offset
                    + local_row * verified.component.row_bytes,
                    verified.component.row_bytes,
                )
                for verified in self._components
            )

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            if self._components:
                self._segmented_closed = True
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

    __slots__ = (
        "_cache_owner",
        "_cache_reuse_error",
        "_lock",
        "_root_fd",
        "_source_files",
        "manifest",
        "report",
        "shards",
    )

    def __init__(
        self,
        manifest: NGramManifest,
        root_descriptor: int,
        shards: tuple[VerifiedNGramShard, ...],
        source_files: tuple[_VerifiedNGramFile, ...] = (),
    ) -> None:
        self.manifest = manifest
        self._root_fd = root_descriptor
        self._lock = threading.RLock()
        self._cache_owner: object | None = None
        self._cache_reuse_error: str | None = None
        self.shards = shards
        self._source_files = source_files
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
                if shard.components:
                    return verified.read_row(row - shard.start_row)
                return verified.pread(offset, self.manifest.row_bytes)
        raise NGramManifestError(f"verified artifact has no shard for row {row}")

    def close(self) -> None:
        with self._lock:
            if self._cache_owner is not None:
                raise NGramManifestError(
                    "cannot close verified n-gram artifact while its cache owns it"
                )
            first_error: NGramManifestError | None = None
            for shard in self.shards:
                try:
                    shard.close()
                except NGramManifestError as exc:
                    if first_error is None:
                        first_error = exc
            for source in self._source_files:
                try:
                    source.close()
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


def _hash_complete_descriptor(descriptor: int, size: int, *, name: str) -> str:
    digest = hashlib.sha256()
    remaining = size
    offset = 0
    try:
        while remaining:
            chunk = os.pread(descriptor, min(8 * 1024 * 1024, remaining), offset)
            if not chunk:
                raise NGramManifestError(f"short payload in source file {name}")
            digest.update(chunk)
            remaining -= len(chunk)
            offset += len(chunk)
    except NGramManifestError:
        raise
    except OSError as exc:
        raise NGramManifestError(f"could not hash source file {name}: {exc}") from exc
    return digest.hexdigest()


def _verify_segmented_ngram_manifest(
    root: Path | str,
    manifest: NGramManifest,
) -> VerifiedNGramArtifact:
    if any(not shard.components for shard in manifest.shards):
        raise NGramManifestError(
            "source-native n-gram manifests cannot mix contiguous and segmented shards"
        )
    expected_files: dict[str, tuple[int, str]] = {}
    for shard in manifest.shards:
        for component in shard.components:
            prior = expected_files.setdefault(
                component.name,
                (component.file_size, component.file_sha256),
            )
            if prior != (component.file_size, component.file_sha256):
                raise NGramManifestError(
                    f"conflicting source identity for {component.name}"
                )

    root_descriptor: int | None = None
    opened_files: list[_VerifiedNGramFile] = []
    try:
        root_descriptor = _open_root_nofollow(root)
        by_name: dict[str, _VerifiedNGramFile] = {}
        for name, (file_size, file_sha256) in sorted(expected_files.items()):
            safe_name = _safe_component(name, label="component shard name")
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    safe_name,
                    _readonly_flags(),
                    dir_fd=root_descriptor,
                )
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise NGramManifestError(
                        f"source component file is not regular: {name}"
                    )
                if metadata.st_nlink != 1:
                    raise NGramManifestError(
                        f"source component file {name} must have exactly one hard link"
                    )
                if metadata.st_mode & (
                    stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                ):
                    raise NGramManifestError(
                        f"source component file {name} must be read-only"
                    )
                if metadata.st_size != file_size:
                    raise NGramManifestError(
                        f"source component file size mismatch for {name}"
                    )
                if fcntl is not None:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except OSError as exc:
                        raise NGramManifestError(
                            f"could not acquire immutable shared lock for {name}: {exc}"
                        ) from exc
                if _hash_complete_descriptor(descriptor, file_size, name=name) != file_sha256:
                    raise NGramManifestError(
                        f"source component file digest mismatch for {name}"
                    )
                verified_metadata = os.fstat(descriptor)
                if (
                    verified_metadata.st_dev != metadata.st_dev
                    or verified_metadata.st_ino != metadata.st_ino
                    or verified_metadata.st_size != metadata.st_size
                    or verified_metadata.st_mode != metadata.st_mode
                    or verified_metadata.st_nlink != metadata.st_nlink
                    or verified_metadata.st_mtime_ns != metadata.st_mtime_ns
                    or verified_metadata.st_ctime_ns != metadata.st_ctime_ns
                ):
                    raise NGramManifestError(
                        f"source component file identity changed during verification: {name}"
                    )
                owner = _VerifiedNGramFile(
                    name,
                    descriptor,
                    NGramFileIdentity(
                        device=verified_metadata.st_dev,
                        inode=verified_metadata.st_ino,
                        size=verified_metadata.st_size,
                        mtime_ns=verified_metadata.st_mtime_ns,
                        ctime_ns=verified_metadata.st_ctime_ns,
                    ),
                )
                descriptor = None
                opened_files.append(owner)
                by_name[name] = owner
            finally:
                if descriptor is not None:
                    os.close(descriptor)

        retained: list[VerifiedNGramShard] = []
        for shard in manifest.shards:
            verified_components = tuple(
                _VerifiedNGramComponent(component, by_name[component.name])
                for component in shard.components
            )
            retained.append(
                VerifiedNGramShard(
                    shard,
                    -1,
                    verified_components[0].source.identity,
                    components=verified_components,
                )
            )
        artifact = VerifiedNGramArtifact(
            manifest,
            root_descriptor,
            tuple(retained),
            tuple(opened_files),
        )
        root_descriptor = None
        opened_files = []
        return artifact
    except NGramManifestError:
        raise
    except OSError as exc:
        raise NGramManifestError(
            f"could not verify source-native n-gram artifact: {exc}"
        ) from exc
    finally:
        cleanup_error: NGramManifestError | None = None
        for source in opened_files:
            try:
                source.close()
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


def _require_verification_capabilities() -> None:
    """Fail closed before artifact access if immutable-FD primitives are absent."""

    missing: list[str] = []
    if fcntl is None:
        missing.append("fcntl")
    else:
        if not callable(getattr(fcntl, "flock", None)):
            missing.append("fcntl.flock")
        lock_values: list[tuple[str, int]] = []
        for name in ("LOCK_SH", "LOCK_EX", "LOCK_NB", "LOCK_UN"):
            value = getattr(fcntl, name, None)
            if type(value) is not int or value <= 0:
                missing.append(f"fcntl.{name}")
            else:
                lock_values.append((name, value))
        if len(lock_values) == 4:
            overlaps = any(
                left_value & right_value
                for index, (_left_name, left_value) in enumerate(lock_values)
                for _right_name, right_value in lock_values[index + 1 :]
            )
            if overlaps:
                missing.append("distinct non-overlapping fcntl lock flags")
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
    if any(shard.components for shard in manifest.shards):
        return _verify_segmented_ngram_manifest(root, manifest)
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


@dataclass(frozen=True)
class NGramCacheConfig:
    """Construction-time resource and eviction policy for exact row streaming."""

    cache_limit_bytes: int
    transient_limit_bytes: int
    max_inflight_io_bytes: int
    max_open_files: int
    bypass_page_cache: bool
    eviction: Literal["lru", "frequency"]
    allocation_alignment_bytes: int = 1

    def __post_init__(self) -> None:
        for name in (
            "cache_limit_bytes",
            "transient_limit_bytes",
            "max_inflight_io_bytes",
            "max_open_files",
            "allocation_alignment_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if type(self.bypass_page_cache) is not bool:
            raise TypeError("bypass_page_cache must be an exact boolean")
        if type(self.eviction) is not str or self.eviction not in {
            "lru",
            "frequency",
        }:
            raise ValueError("eviction must be 'lru' or 'frequency'")
        if self.allocation_alignment_bytes & (self.allocation_alignment_bytes - 1):
            raise ValueError("allocation_alignment_bytes must be a power of two")


_EMPTY_ROW = (1 << 32) - 1
_EMPTY_SLOT = (1 << 32) - 1
_SLOT_METADATA_BYTES = 8 + 4 + 4 + 1 + 4 + 8
_ROUTE_ENTRY_BYTES = 4 + 4


@dataclass(frozen=True)
class NGramCachePlan:
    """Exact fixed-storage plan; Python container headers are not included."""

    requested_payload_bytes: int
    payload_bytes: int
    slot_count: int
    slot_metadata_bytes: int
    route_capacity: int
    route_table_bytes: int
    transient_bytes: int
    transient_metadata_bytes: int
    alignment_bytes: int
    total_reserved_bytes: int


@dataclass(frozen=True)
class NGramRuntimeBudget:
    """Measured construction-time inputs for the pinned production cache."""

    measured_base_residency_bytes: int
    kv_mtp_reserve_bytes: int
    metal_working_reserve_bytes: int
    safety_margin_bytes: int
    minimum_payload_bytes: int
    allocation_alignment_bytes: int
    target_residency_bytes: int = PRODUCTION_RUNTIME_TARGET_BYTES
    payload_ceiling_bytes: int = PRODUCTION_NGRAM_PAYLOAD_CEILING_BYTES

    def __post_init__(self) -> None:
        for name in (
            "measured_base_residency_bytes",
            "kv_mtp_reserve_bytes",
            "metal_working_reserve_bytes",
            "safety_margin_bytes",
            "minimum_payload_bytes",
            "allocation_alignment_bytes",
            "target_residency_bytes",
            "payload_ceiling_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.minimum_payload_bytes == 0:
            raise ValueError("minimum_payload_bytes must be positive")
        if self.allocation_alignment_bytes == 0 or self.allocation_alignment_bytes & (
            self.allocation_alignment_bytes - 1
        ):
            raise ValueError("allocation_alignment_bytes must be a positive power of two")
        if not 0 < self.target_residency_bytes <= PRODUCTION_RUNTIME_TARGET_BYTES:
            raise ValueError("target_residency_bytes exceeds the pinned 95 GiB target")
        if not 0 < self.payload_ceiling_bytes <= (
            PRODUCTION_NGRAM_PAYLOAD_CEILING_BYTES
        ):
            raise ValueError("payload_ceiling_bytes exceeds the pinned 10 GiB maximum")


@dataclass(frozen=True)
class NGramProductionCachePlan:
    """Pinned production capacity plus its complete residency accounting."""

    target_residency_bytes: int
    fixed_runtime_bytes: int
    available_cache_bytes: int
    payload_formula_ceiling_bytes: int
    projected_residency_bytes: int
    config: NGramCacheConfig
    cache: NGramCachePlan


def _checked_add_bytes(*values: int, label: str) -> int:
    total = 0
    for value in values:
        total += value
        if total > _MAX_SIGNED_INT64:
            raise OverflowError(f"{label} exceeds signed 64-bit byte accounting")
    return total


def _checked_mul_bytes(left: int, right: int, *, label: str) -> int:
    value = left * right
    if value > _MAX_SIGNED_INT64:
        raise OverflowError(f"{label} exceeds signed 64-bit byte accounting")
    return value


def _aligned_bytes(value: int, alignment: int, *, label: str) -> int:
    padded = _checked_add_bytes(value, alignment - 1, label=label)
    return padded & -alignment


def plan_ngram_cache(
    manifest: NGramManifest, config: NGramCacheConfig
) -> NGramCachePlan:
    if type(manifest) is not NGramManifest:
        raise TypeError("manifest must be an exact NGramManifest")
    if type(config) is not NGramCacheConfig:
        raise TypeError("config must be an exact NGramCacheConfig")
    if manifest.padded_rows >= _EMPTY_ROW:
        raise ValueError("manifest rows exceed the packed route-key domain")
    slot_count = config.cache_limit_bytes // manifest.row_bytes
    if slot_count == 0:
        raise ValueError("cache_limit_bytes must hold at least one complete row")
    if slot_count >= _EMPTY_SLOT:
        raise ValueError("cache payload exceeds the packed slot-index domain")
    transient_slots = config.transient_limit_bytes // manifest.row_bytes
    if transient_slots == 0:
        raise ValueError("transient_limit_bytes must hold at least one complete row")
    if transient_slots >= _EMPTY_SLOT:
        raise ValueError("transient storage exceeds the packed slot-index domain")
    route_capacity = 1 << max(1, (slot_count * 2 - 1).bit_length())
    payload_bytes = _checked_mul_bytes(
        slot_count, manifest.row_bytes, label="cache payload"
    )
    slot_metadata_bytes = _checked_mul_bytes(
        slot_count, _SLOT_METADATA_BYTES, label="slot metadata"
    )
    route_table_bytes = _checked_mul_bytes(
        route_capacity, _ROUTE_ENTRY_BYTES, label="route table"
    )
    transient_bytes = _checked_mul_bytes(
        transient_slots, manifest.row_bytes, label="transient buffer"
    )
    transient_metadata_bytes = transient_slots
    slot_backings = tuple(
        _checked_mul_bytes(slot_count, item_bytes, label="slot metadata backing")
        for item_bytes in (8, 4, 4, 1, 4, 8)
    )
    route_backings = (
        _checked_mul_bytes(route_capacity, 4, label="route-key backing"),
        _checked_mul_bytes(route_capacity, 4, label="route-slot backing"),
    )
    backing_bytes = (
        payload_bytes,
        *slot_backings,
        *route_backings,
        transient_bytes,
        transient_metadata_bytes,
    )
    logical_total = _checked_add_bytes(*backing_bytes, label="cache reservation")
    aligned_total = _checked_add_bytes(
        *(
            _aligned_bytes(
                value,
                config.allocation_alignment_bytes,
                label="aligned cache backing",
            )
            for value in backing_bytes
        ),
        label="aligned cache reservation",
    )
    alignment_bytes = aligned_total - logical_total
    return NGramCachePlan(
        requested_payload_bytes=config.cache_limit_bytes,
        payload_bytes=payload_bytes,
        slot_count=slot_count,
        slot_metadata_bytes=slot_metadata_bytes,
        route_capacity=route_capacity,
        route_table_bytes=route_table_bytes,
        transient_bytes=transient_bytes,
        transient_metadata_bytes=transient_metadata_bytes,
        alignment_bytes=alignment_bytes,
        total_reserved_bytes=aligned_total,
    )


def plan_production_ngram_cache(
    manifest: NGramManifest,
    budget: NGramRuntimeBudget,
    *,
    transient_limit_bytes: int,
    max_inflight_io_bytes: int,
    max_open_files: int,
    bypass_page_cache: bool,
    eviction: Literal["lru", "frequency"],
) -> NGramProductionCachePlan:
    """Solve the largest exact stored-row cache under the measured runtime budget."""

    if type(manifest) is not NGramManifest:
        raise TypeError("manifest must be an exact NGramManifest")
    if type(budget) is not NGramRuntimeBudget:
        raise TypeError("budget must be an exact NGramRuntimeBudget")
    if eviction != "lru":
        raise ValueError(
            "production eviction must remain 'lru' until a measured CLOCK lane "
            "is construction-approved"
        )
    fixed_runtime_bytes = _checked_add_bytes(
        budget.measured_base_residency_bytes,
        budget.kv_mtp_reserve_bytes,
        budget.metal_working_reserve_bytes,
        budget.safety_margin_bytes,
        label="fixed runtime reservation",
    )
    if fixed_runtime_bytes >= budget.target_residency_bytes:
        raise ValueError("minimum viable n-gram cache and runtime cannot fit")
    available_cache_bytes = budget.target_residency_bytes - fixed_runtime_bytes
    payload_formula_ceiling_bytes = min(
        budget.payload_ceiling_bytes,
        available_cache_bytes,
    )
    rounded_minimum_payload = _checked_add_bytes(
        budget.minimum_payload_bytes,
        manifest.row_bytes - 1,
        label="minimum payload rounding",
    )
    minimum_slots = rounded_minimum_payload // manifest.row_bytes
    maximum_slots = payload_formula_ceiling_bytes // manifest.row_bytes
    if maximum_slots < minimum_slots:
        raise ValueError("minimum viable n-gram cache and runtime cannot fit")

    def candidate(slot_count: int) -> tuple[NGramCacheConfig, NGramCachePlan]:
        config = NGramCacheConfig(
            cache_limit_bytes=_checked_mul_bytes(
                slot_count, manifest.row_bytes, label="candidate cache payload"
            ),
            transient_limit_bytes=transient_limit_bytes,
            max_inflight_io_bytes=max_inflight_io_bytes,
            max_open_files=max_open_files,
            bypass_page_cache=bypass_page_cache,
            eviction=eviction,
            allocation_alignment_bytes=budget.allocation_alignment_bytes,
        )
        return config, plan_ngram_cache(manifest, config)

    minimum_config, minimum_plan = candidate(minimum_slots)
    if minimum_plan.total_reserved_bytes > available_cache_bytes:
        raise ValueError("minimum viable n-gram cache and runtime cannot fit")

    low = minimum_slots
    high = maximum_slots
    selected_config = minimum_config
    selected_plan = minimum_plan
    while low <= high:
        middle = (low + high) // 2
        config, plan = candidate(middle)
        if plan.total_reserved_bytes <= available_cache_bytes:
            selected_config = config
            selected_plan = plan
            low = middle + 1
        else:
            high = middle - 1

    projected_residency_bytes = _checked_add_bytes(
        fixed_runtime_bytes,
        selected_plan.total_reserved_bytes,
        label="projected runtime residency",
    )
    return NGramProductionCachePlan(
        target_residency_bytes=budget.target_residency_bytes,
        fixed_runtime_bytes=fixed_runtime_bytes,
        available_cache_bytes=available_cache_bytes,
        payload_formula_ceiling_bytes=payload_formula_ceiling_bytes,
        projected_residency_bytes=projected_residency_bytes,
        config=selected_config,
        cache=selected_plan,
    )


@dataclass(frozen=True)
class SlotTicket:
    """Generation-qualified ownership of one fixed cache slot."""

    slot: int
    generation: int
    _owner: object | None = field(default=None, repr=False, compare=False)


class _PackedCacheIndex:
    """Fixed-capacity packed slot metadata and open-addressed row routes."""

    __slots__ = (
        "access",
        "frequency",
        "generations",
        "loaded",
        "pins",
        "route_keys",
        "route_mask",
        "route_slots",
        "rows",
        "slot_count",
    )

    def __init__(self, plan: NGramCachePlan) -> None:
        self.slot_count = plan.slot_count
        self.generations = array("Q", [0]) * plan.slot_count
        self.rows = array("I", [_EMPTY_ROW]) * plan.slot_count
        self.pins = array("I", [0]) * plan.slot_count
        self.loaded = bytearray(plan.slot_count)
        self.frequency = array("I", [0]) * plan.slot_count
        self.access = array("Q", [0]) * plan.slot_count
        self.route_keys = array("I", [_EMPTY_ROW]) * plan.route_capacity
        self.route_slots = array("I", [_EMPTY_SLOT]) * plan.route_capacity
        self.route_mask = plan.route_capacity - 1
        if (
            self.generations.itemsize != 8
            or self.rows.itemsize != 4
            or self.pins.itemsize != 4
            or self.frequency.itemsize != 4
            or self.access.itemsize != 8
            or self.route_keys.itemsize != 4
            or self.route_slots.itemsize != 4
        ):
            raise RuntimeError("platform packed integer widths do not match the cache plan")

    def _probe(self, row: int) -> tuple[int, bool]:
        index = (row * 2_654_435_761) & self.route_mask
        for _ in range(self.route_mask + 1):
            key = self.route_keys[index]
            if key == row:
                return index, True
            if key == _EMPTY_ROW:
                return index, False
            index = (index + 1) & self.route_mask
        raise NGramCacheFull("fixed n-gram route table is full")

    def lookup_with_probes(self, row: int) -> tuple[int | None, int]:
        index = (row * 2_654_435_761) & self.route_mask
        for probes in range(1, self.route_mask + 2):
            key = self.route_keys[index]
            if key == row:
                return int(self.route_slots[index]), probes
            if key == _EMPTY_ROW:
                return None, probes
            index = (index + 1) & self.route_mask
        raise NGramCacheFull("fixed n-gram route table is full")

    def lookup(self, row: int) -> int | None:
        index, found = self._probe(row)
        return int(self.route_slots[index]) if found else None

    def insert(self, row: int, slot: int) -> None:
        index, found = self._probe(row)
        if found:
            raise NGramCacheError("duplicate row route installation")
        self.route_keys[index] = row
        self.route_slots[index] = slot

    def remove(self, row: int) -> None:
        index, found = self._probe(row)
        if not found:
            return
        hole = index
        scan = (hole + 1) & self.route_mask
        while self.route_keys[scan] != _EMPTY_ROW:
            key = int(self.route_keys[scan])
            home = (key * 2_654_435_761) & self.route_mask
            if ((scan - home) & self.route_mask) >= ((hole - home) & self.route_mask):
                self.route_keys[hole] = key
                self.route_slots[hole] = self.route_slots[scan]
                hole = scan
            scan = (scan + 1) & self.route_mask
        self.route_keys[hole] = _EMPTY_ROW
        self.route_slots[hole] = _EMPTY_SLOT

    def clear(self) -> None:
        for slot in range(self.slot_count):
            self.generations[slot] += 1
            self.rows[slot] = _EMPTY_ROW
            self.pins[slot] = 0
            self.loaded[slot] = 0
            self.frequency[slot] = 0
            self.access[slot] = 0
        for index in range(self.route_mask + 1):
            self.route_keys[index] = _EMPTY_ROW
            self.route_slots[index] = _EMPTY_SLOT

    def release(self) -> None:
        self.generations = array("Q")
        self.rows = array("I")
        self.pins = array("I")
        self.loaded = bytearray()
        self.frequency = array("I")
        self.access = array("Q")
        self.route_keys = array("I")
        self.route_slots = array("I")
        self.slot_count = 0
        self.route_mask = 0


@dataclass(frozen=True)
class _InstalledNGramComponent:
    data_offset: int
    row_bytes: int
    source: _VerifiedNGramFile


@dataclass(frozen=True)
class _InstalledNGramShard:
    """Construction-proven physical route to one retained shard descriptor."""

    index: int
    start_row: int
    stop_row: int
    data_offset: int
    data_bytes: int
    retained: VerifiedNGramShard
    components: tuple[_InstalledNGramComponent, ...] = ()


def _install_ngram_shard_routes(
    artifact: VerifiedNGramArtifact,
) -> tuple[_InstalledNGramShard, ...]:
    """Seal manifest order, physical ranges, and live descriptor identities."""

    manifest_shards = artifact.manifest.shards
    retained_shards = artifact.shards
    if type(retained_shards) is not tuple:
        raise ValueError("verified n-gram shards must be an exact tuple")
    if len(retained_shards) != len(manifest_shards):
        raise ValueError("verified n-gram shard count does not match the manifest")
    routes: list[_InstalledNGramShard] = []
    try:
        root_descriptor = artifact.root_fileno()
        root_metadata = os.fstat(root_descriptor)
    except (NGramManifestError, OSError) as exc:
        raise NGramCacheIOError(
            f"could not prove retained artifact root: {exc}"
        ) from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise NGramCacheIOError("retained artifact root is not an open directory")
    descriptors: set[int] = {root_descriptor}
    identities: set[tuple[int, int]] = set()
    proven_sources: set[_VerifiedNGramFile] = set()
    for index, (expected, retained) in enumerate(
        zip(manifest_shards, retained_shards, strict=True)
    ):
        if type(retained) is not VerifiedNGramShard:
            raise TypeError("verified shard owner has the wrong exact type")
        if type(retained.shard) is not NGramShard or retained.shard != expected:
            raise ValueError(
                f"verified shard {index} metadata/order does not match the manifest"
            )
        if type(retained.identity) is not NGramFileIdentity:
            raise TypeError("verified shard identity has the wrong exact type")
        if retained.closed:
            raise NGramCacheClosed(f"verified shard {expected.name} is closed")
        if expected.components:
            if len(retained._components) != len(expected.components):
                raise ValueError(
                    f"verified shard {expected.name} component count changed"
                )
            installed_components: list[_InstalledNGramComponent] = []
            for component, verified_component in zip(
                expected.components,
                retained._components,
                strict=True,
            ):
                if verified_component.component != component:
                    raise ValueError(
                        f"verified shard {expected.name} component metadata changed"
                    )
                source = verified_component.source
                if source not in proven_sources:
                    try:
                        descriptor = source.fileno()
                        metadata = os.fstat(descriptor)
                    except (NGramManifestError, OSError) as exc:
                        raise NGramCacheIOError(
                            f"could not prove retained source {source.name}: {exc}"
                        ) from exc
                    current_identity = NGramFileIdentity(
                        device=metadata.st_dev,
                        inode=metadata.st_ino,
                        size=metadata.st_size,
                        mtime_ns=metadata.st_mtime_ns,
                        ctime_ns=metadata.st_ctime_ns,
                    )
                    if source.identity != current_identity:
                        raise NGramCacheIOError(
                            f"retained source {source.name} identity changed"
                        )
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                        raise NGramCacheIOError(
                            f"retained source {source.name} is not singly linked"
                        )
                    file_identity = (current_identity.device, current_identity.inode)
                    if descriptor in descriptors or file_identity in identities:
                        raise NGramCacheIOError(
                            "retained source descriptors are not unique"
                        )
                    descriptors.add(descriptor)
                    identities.add(file_identity)
                    proven_sources.add(source)
                installed_components.append(
                    _InstalledNGramComponent(
                        data_offset=component.data_offset,
                        row_bytes=component.row_bytes,
                        source=source,
                    )
                )
            routes.append(
                _InstalledNGramShard(
                    index=index,
                    start_row=expected.start_row,
                    stop_row=expected.start_row + expected.row_count,
                    data_offset=0,
                    data_bytes=expected.data_bytes,
                    retained=retained,
                    components=tuple(installed_components),
                )
            )
            continue
        try:
            descriptor = retained.fileno()
            metadata = os.fstat(descriptor)
        except (NGramManifestError, OSError) as exc:
            raise NGramCacheIOError(
                f"could not prove retained shard {expected.name}: {exc}"
            ) from exc
        current_identity = NGramFileIdentity(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
        )
        if retained.identity != current_identity:
            raise NGramCacheIOError(
                f"retained shard {expected.name} identity changed after verification"
            )
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise NGramCacheIOError(
                f"retained shard {expected.name} is not a singly-linked regular file"
            )
        if expected.file_size != current_identity.size:
            raise NGramCacheIOError(
                f"retained shard {expected.name} size does not match the manifest"
            )
        file_identity = (current_identity.device, current_identity.inode)
        if descriptor in descriptors or file_identity in identities:
            raise NGramCacheIOError("retained n-gram descriptors are not unique")
        descriptors.add(descriptor)
        identities.add(file_identity)
        routes.append(
            _InstalledNGramShard(
                index=index,
                start_row=expected.start_row,
                stop_row=expected.start_row + expected.row_count,
                data_offset=expected.data_offset,
                data_bytes=expected.data_bytes,
                retained=retained,
            )
        )
    return tuple(routes)


class _PageCachePolicyError(NGramCacheIOError):
    def __init__(self, message: str, *, rollback_failed: bool) -> None:
        super().__init__(message)
        self.rollback_failed = rollback_failed


@dataclass(slots=True)
class _PageCacheInstallState:
    rollback_failed: bool = False
    rollback_detail: str = ""


def _installed_source_descriptors(
    routes: tuple[_InstalledNGramShard, ...],
) -> tuple[int, ...]:
    descriptors: list[int] = []
    for route in routes:
        if route.components:
            descriptors.extend(
                component.source.fileno() for component in route.components
            )
        else:
            descriptors.append(route.retained.fileno())
    return tuple(dict.fromkeys(descriptors))


def _install_page_cache_policy(
    routes: tuple[_InstalledNGramShard, ...],
    bypass_page_cache: bool,
    state: _PageCacheInstallState,
) -> bool:
    if not bypass_page_cache:
        return False
    if (
        fcntl is None
        or type(getattr(fcntl, "F_NOCACHE", None)) is not int
        or not callable(getattr(fcntl, "fcntl", None))
    ):
        raise ValueError("bypass_page_cache requires Darwin fcntl.F_NOCACHE support")
    command = fcntl.F_NOCACHE
    installed: list[int] = []
    try:
        for descriptor in _installed_source_descriptors(routes):
            installed.append(descriptor)
            fcntl.fcntl(descriptor, command, 1)
    except BaseException as exc:
        rollback_error: BaseException | None = None
        for descriptor in reversed(installed):
            try:
                fcntl.fcntl(descriptor, command, 0)
            except BaseException as rollback_exc:  # noqa: BLE001 - try every FD
                if rollback_error is None:
                    rollback_error = rollback_exc
        suffix = "" if rollback_error is None else f"; rollback failed: {rollback_error}"
        state.rollback_failed = rollback_error is not None
        state.rollback_detail = suffix
        if isinstance(exc, Exception):
            raise _PageCachePolicyError(
                f"could not install F_NOCACHE on retained n-gram shards: {exc}{suffix}",
                rollback_failed=rollback_error is not None,
            ) from exc
        raise
    return True


def _restore_page_cache_policy(
    routes: tuple[_InstalledNGramShard, ...], installed: bool
) -> None:
    if not installed:
        return
    assert fcntl is not None
    command = fcntl.F_NOCACHE
    first_error: BaseException | None = None
    first_traceback: Any | None = None
    for descriptor in _installed_source_descriptors(routes):
        try:
            fcntl.fcntl(descriptor, command, 0)
        except BaseException as exc:  # noqa: BLE001 - restore every descriptor
            if first_error is None:
                first_error = exc
                first_traceback = exc.__traceback__
    if first_error is not None:
        if not isinstance(first_error, Exception):
            raise first_error.with_traceback(first_traceback)
        detail = f"{type(first_error).__name__}: {str(first_error)[:256]}"
        raise NGramCacheIOError(
            f"could not restore retained n-gram page-cache policy: {detail}"
        )


class _DescriptorReader:
    """Production descriptor reader with no runtime policy selection."""

    __slots__ = ("_iov_max", "_preadv")

    def __init__(self) -> None:
        preadv = getattr(os, "preadv", None)
        if not callable(preadv):
            raise TypeError("production n-gram streaming requires os.preadv")
        self._preadv = preadv
        try:
            iov_max = os.sysconf("SC_IOV_MAX")
        except (OSError, ValueError):
            iov_max = 1024
        self._iov_max = max(1, int(iov_max))

    def read_into(
        self,
        shard: VerifiedNGramShard,
        offset: int,
        target: memoryview,
    ) -> int:
        descriptor = shard.fileno()
        cursor = 0
        while cursor < target.nbytes:
            count = self._preadv(descriptor, [target[cursor:]], offset + cursor)
            if count <= 0:
                break
            cursor += count
        return cursor

    def read_segmented_into(
        self,
        route: _InstalledNGramShard,
        local_row: int,
        row_count: int,
        row_bytes: int,
        target: memoryview,
    ) -> int:
        output_offset = 0
        total = 0
        for component in route.components:
            completed = 0
            while completed < row_count:
                count = min(self._iov_max, row_count - completed)
                views = [
                    target[
                        (completed + row) * row_bytes + output_offset :
                        (completed + row) * row_bytes
                        + output_offset
                        + component.row_bytes
                    ]
                    for row in range(count)
                ]
                source_offset = component.data_offset + (
                    local_row + completed
                ) * component.row_bytes
                expected = count * component.row_bytes
                read = 0
                while read < expected:
                    skip = read
                    active: list[memoryview] = []
                    for view in views:
                        if skip >= view.nbytes:
                            skip -= view.nbytes
                            continue
                        active.append(view[skip:])
                        skip = 0
                    current = self._preadv(
                        component.source.fileno(),
                        active,
                        source_offset + read,
                    )
                    if current <= 0:
                        return total + read
                    read += current
                completed += count
                total += read
            output_offset += component.row_bytes
        return total


class _TransientPool:
    """Fixed row-granular staging storage for positional reads."""

    __slots__ = ("buffer", "row_bytes", "slots", "used", "view")

    def __init__(self, plan: NGramCachePlan, row_bytes: int) -> None:
        self.buffer = bytearray(plan.transient_bytes)
        self.view = memoryview(self.buffer)
        self.used = bytearray(plan.transient_metadata_bytes)
        self.slots = plan.transient_metadata_bytes
        self.row_bytes = row_bytes

    def reserve(self, row_counts: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        reserved: list[tuple[int, int]] = []
        for count in row_counts:
            run = 0
            start = 0
            found = False
            for slot in range(self.slots):
                if self.used[slot] == 0:
                    if run == 0:
                        start = slot
                    run += 1
                    if run == count:
                        found = True
                        break
                else:
                    run = 0
            if not found:
                for prior_start, prior_count in reserved:
                    self.release(prior_start, prior_count)
                raise NGramCacheFull("fixed n-gram transient pool is exhausted")
            for slot in range(start, start + count):
                self.used[slot] = 1
            reserved.append((start, count))
        return tuple(reserved)

    def reserve_fragmented(
        self, row_counts: tuple[int, ...]
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Return (group, group-row-offset, pool-start, count) segments."""

        needed = sum(row_counts)
        if sum(1 for value in self.used if value == 0) < needed:
            raise NGramCacheFull("fixed n-gram transient pool is exhausted")
        planned: list[tuple[int, int, int, int]] = []
        claimed: set[int] = set()
        for group_index, row_count in enumerate(row_counts):
            group_offset = 0
            while group_offset < row_count:
                start = next(
                    slot
                    for slot in range(self.slots)
                    if self.used[slot] == 0 and slot not in claimed
                )
                count = 0
                slot = start
                while (
                    slot < self.slots
                    and self.used[slot] == 0
                    and slot not in claimed
                    and group_offset + count < row_count
                ):
                    claimed.add(slot)
                    count += 1
                    slot += 1
                planned.append((group_index, group_offset, start, count))
                group_offset += count
        for slot in claimed:
            self.used[slot] = 1
        return tuple(planned)

    def release(self, start: int, count: int) -> None:
        for slot in range(start, start + count):
            self.used[slot] = 0

    def bytes_view(self, start: int, count: int) -> memoryview:
        begin = start * self.row_bytes
        return self.view[begin : begin + count * self.row_bytes]

    def close(self) -> None:
        self.view.release()
        self.buffer = bytearray()
        self.used = bytearray()
        self.slots = 0


_CACHE_CONSTRUCTION_KEY = object()


class NGramLease:
    """Pinned, generation-qualified access to exact packed row bytes."""

    __slots__ = ("_cache", "_released", "_tickets", "_token", "slot_ids")

    def __init__(
        self,
        cache: NGramRowCache,
        tickets: tuple[SlotTicket, ...],
        token: object,
        construction_key: object,
    ) -> None:
        if construction_key is not _CACHE_CONSTRUCTION_KEY or token is not cache._token:
            raise TypeError("NGramLease construction is private")
        self._cache = cache
        self._tickets = tickets
        self._token = token
        self.slot_ids = tuple(ticket.slot for ticket in tickets)
        self._released = False

    def __copy__(self) -> None:
        raise TypeError("copy is forbidden for n-gram leases")

    def __deepcopy__(self, memo: dict[int, Any]) -> None:
        del memo
        raise TypeError("deepcopy is forbidden for n-gram leases")

    @property
    def released(self) -> bool:
        return self._released

    def row_bytes(self, index: int) -> bytes:
        if type(index) is not int:
            raise TypeError("row index must be an exact integer")
        if not 0 <= index < len(self._tickets):
            raise IndexError("row index is out of range")
        return self._cache._lease_row_bytes(self, self._tickets[index])

    def release(self) -> None:
        self._cache._release_lease(self)

    def __enter__(self) -> Self:
        if self._released:
            raise NGramCacheClosed("n-gram lease has been released")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class NGramAcquireFuture:
    """A cancellable acquisition whose workers publish through slot tickets."""

    __slots__ = (
        "_cache",
        "_cancelled",
        "_failed",
        "_futures",
        "_lease",
        "_result_lock",
        "_tickets",
        "_token",
        "_unique_tickets",
    )

    def __init__(
        self,
        cache: NGramRowCache,
        tickets: tuple[SlotTicket, ...],
        futures: tuple[Future[None], ...],
        token: object,
        construction_key: object,
    ) -> None:
        if construction_key is not _CACHE_CONSTRUCTION_KEY or token is not cache._token:
            raise TypeError("NGramAcquireFuture construction is private")
        self._cache = cache
        self._token = token
        self._tickets = tickets
        self._unique_tickets = tuple(dict.fromkeys(tickets))
        self._futures = futures
        self._cancelled = False
        self._failed = False
        self._lease: NGramLease | None = None
        self._result_lock = threading.Lock()

    def __copy__(self) -> None:
        raise TypeError("copy is forbidden for n-gram acquisition futures")

    def __deepcopy__(self, memo: dict[int, Any]) -> None:
        del memo
        raise TypeError("deepcopy is forbidden for n-gram acquisition futures")

    def cancel(self) -> bool:
        return self._cache._cancel_request(self)

    def cancelled(self) -> bool:
        with self._cache._lock:
            return self._cancelled

    def done(self) -> bool:
        with self._cache._lock:
            if self._cancelled or self._failed or self._lease is not None:
                return True
        return all(future.done() for future in self._futures)

    def result(self, timeout: float | None = None) -> NGramLease:
        with self._result_lock:
            with self._cache._lock:
                poison = self._cache._poison_message
                cancelled = self._cancelled
            if poison is not None:
                self._cache._drain_poisoned_workers()
                raise NGramCacheIOError(
                    f"n-gram cache lane is poisoned: {poison}"
                )
            if self._lease is not None:
                return self._lease
            if cancelled:
                raise CancelledError()
            deadline = None if timeout is None else monotonic() + timeout
            try:
                for future in self._futures:
                    remaining = None if deadline is None else max(0.0, deadline - monotonic())
                    future.result(timeout=remaining)
            except TimeoutError:
                raise
            except CancelledError:
                self._cache._fail_request(self)
                raise
            except Exception as exc:
                self._cache._fail_request(self)
                self._cache._drain_poisoned_workers()
                if isinstance(exc, NGramCacheIOError):
                    raise
                raise NGramCacheIOError(f"n-gram row acquisition failed: {exc}") from exc
            except BaseException:
                self._cache._drain_poisoned_workers()
                raise
            return self._cache._finish_request(self)


class NGramRowCache:
    """Fixed-byte cache backed by retained descriptors owned by the caller."""

    def __init__(
        self,
        artifact: VerifiedNGramArtifact,
        config: NGramCacheConfig,
        *,
        reader: Any | None = None,
        allocator: Callable[[int], Any] | None = None,
    ) -> None:
        if type(artifact) is not VerifiedNGramArtifact:
            raise TypeError("artifact must be an exact VerifiedNGramArtifact")
        if type(config) is not NGramCacheConfig:
            raise TypeError("config must be an exact NGramCacheConfig")
        if artifact.closed:
            raise NGramCacheClosed("verified n-gram artifact is closed")
        manifest = artifact.manifest
        row_bytes = manifest.row_bytes
        segmented = bool(manifest.shards[0].components)
        if config.max_inflight_io_bytes < row_bytes:
            raise ValueError("max_inflight_io_bytes must hold at least one row")
        retained_file_count = (
            len(artifact._source_files) if segmented else len(artifact.shards)
        )
        if config.max_open_files < retained_file_count + 1:
            raise ValueError(
                "max_open_files must include the root and every retained source file"
            )
        plan = plan_ngram_cache(manifest, config)
        token = object()
        with artifact._lock:
            if artifact._cache_reuse_error is not None:
                raise NGramCacheIOError(
                    "verified n-gram artifact is poisoned for cache reuse: "
                    f"{artifact._cache_reuse_error}"
                )
            if artifact._cache_owner is not None:
                raise NGramCacheError("verified n-gram artifact already has a cache owner")
            artifact._cache_owner = token
        installed_routes: tuple[_InstalledNGramShard, ...] = ()
        page_cache_installed = False
        page_cache_install_state = _PageCacheInstallState()
        arena: memoryview | None = None
        packed: _PackedCacheIndex | None = None
        transient_pool: _TransientPool | None = None
        try:
            installed_routes = _install_ngram_shard_routes(artifact)
            page_cache_installed = _install_page_cache_policy(
                installed_routes,
                config.bypass_page_cache,
                page_cache_install_state,
            )
            selected_reader = reader if reader is not None else _DescriptorReader()
            if not callable(getattr(selected_reader, "read_into", None)):
                raise TypeError("reader must provide a callable read_into method")
            if segmented and not callable(
                getattr(selected_reader, "read_segmented_into", None)
            ):
                raise TypeError(
                    "source-native reader must provide read_segmented_into"
                )
            if allocator is None:
                from mtplx.mmap_mlx import allocate_metal_u8

                selected_allocator = allocate_metal_u8
            else:
                selected_allocator = allocator
            arena_object = selected_allocator(plan.payload_bytes)
            try:
                arena = memoryview(arena_object).cast("B")
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "allocator must return a writable byte-addressable arena"
                ) from exc
            if arena.readonly:
                raise TypeError("allocator returned a read-only arena")
            if arena.nbytes != plan.payload_bytes:
                raise ValueError("allocator returned an arena with the wrong byte size")
            packed = _PackedCacheIndex(plan)
            transient_pool = _TransientPool(plan, row_bytes)
            executor = ThreadPoolExecutor(
                max_workers=max(1, min(config.max_open_files - 1, len(installed_routes))),
                thread_name_prefix="qwen4-ngram",
            )
        except BaseException as construction_error:
            if arena is not None:
                arena.release()
            if packed is not None:
                packed.release()
            if transient_pool is not None:
                transient_pool.close()
            cleanup_safe = not (
                isinstance(construction_error, _PageCachePolicyError)
                and construction_error.rollback_failed
            ) and not page_cache_install_state.rollback_failed
            try:
                if cleanup_safe:
                    _restore_page_cache_policy(
                        installed_routes, page_cache_installed
                    )
            except BaseException as cleanup_error:
                cleanup_safe = False
                with artifact._lock:
                    artifact._cache_reuse_error = (
                        f"{type(cleanup_error).__name__}: "
                        f"{str(cleanup_error)[:448]}"
                    )
                raise
            finally:
                with artifact._lock:
                    if not cleanup_safe and artifact._cache_reuse_error is None:
                        artifact._cache_reuse_error = str(construction_error)[:512]
                    if artifact._cache_owner is token:
                        artifact._cache_owner = None
            raise

        assert packed is not None and transient_pool is not None

        self.artifact = artifact
        self.manifest = manifest
        self.config = config
        self.plan = plan
        self._token = token
        self._row_bytes = row_bytes
        self._arena_object = arena_object
        self._arena = arena
        self._packed = packed
        self._transient_pool = transient_pool
        self._loading_futures: dict[int, Future[None]] = {}
        self._physical_routes = installed_routes
        self._physical_starts = tuple(route.start_row for route in installed_routes)
        self._reader = selected_reader
        self._read_group = (
            self._read_segmented_group if segmented else self._read_contiguous_group
        )
        self._victim_key = self._lru_key if config.eviction == "lru" else self._frequency_key
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._executor = executor
        self._worker_futures: set[Future[None]] = set()
        self._pending: set[NGramAcquireFuture] = set()
        self._leases: set[NGramLease] = set()
        self._inflight_bytes = 0
        self._tick = 0
        self._state = "OPEN"
        self._resetting = False
        self._accepting = True
        self._poison_message: str | None = None
        self._fatal_cause: BaseException | None = None
        self._page_cache_installed = page_cache_installed
        self._close_error: str | None = None
        self._close_failure: BaseException | None = None

    @property
    def arena_bytes(self) -> int:
        with self._lock:
            return 0 if self._arena is None else self._arena.nbytes

    @property
    def metadata_bytes(self) -> int:
        with self._lock:
            return 0 if self._state == "CLOSED" else self.plan.slot_metadata_bytes

    @property
    def route_table_bytes(self) -> int:
        with self._lock:
            return 0 if self._state == "CLOSED" else self.plan.route_table_bytes

    @property
    def transient_storage_bytes(self) -> int:
        with self._lock:
            return 0 if self._state == "CLOSED" else self.plan.transient_bytes

    @property
    def total_reserved_bytes(self) -> int:
        with self._lock:
            return 0 if self._state == "CLOSED" else self.plan.total_reserved_bytes

    @property
    def inflight_bytes(self) -> int:
        with self._lock:
            return self._inflight_bytes

    @property
    def transient_bytes(self) -> int:
        with self._lock:
            return self._inflight_bytes

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._state == "CLOSED"

    def _lru_key(self, slot: int) -> tuple[int]:
        return (int(self._packed.access[slot]),)

    def _frequency_key(self, slot: int) -> tuple[int, int]:
        return (
            int(self._packed.frequency[slot]),
            int(self._packed.access[slot]),
        )

    def _touch(self, slot: int) -> None:
        self._tick += 1
        self._packed.access[slot] = self._tick
        if self._packed.frequency[slot] < (1 << 32) - 1:
            self._packed.frequency[slot] += 1

    def _ticket(self, slot: int) -> SlotTicket:
        return SlotTicket(slot, int(self._packed.generations[slot]), self._token)

    def _ticket_owned(self, ticket: SlotTicket) -> bool:
        return (
            type(ticket) is SlotTicket
            and ticket._owner is self._token
            and 0 <= ticket.slot < self._packed.slot_count
        )

    def _invalidate_ticket(self, ticket: SlotTicket) -> None:
        if not self._ticket_owned(ticket):
            return
        slot = ticket.slot
        if self._packed.generations[slot] != ticket.generation:
            return
        row = int(self._packed.rows[slot])
        if row != _EMPTY_ROW:
            self._packed.remove(row)
            self._loading_futures.pop(row, None)
        self._packed.generations[slot] += 1
        self._packed.rows[slot] = _EMPTY_ROW
        self._packed.pins[slot] = 0
        self._packed.loaded[slot] = 0
        self._packed.frequency[slot] = 0
        self._packed.access[slot] = 0

    def _release_ticket_pin(self, ticket: SlotTicket, *, cancel_loading: bool) -> None:
        if not self._ticket_owned(ticket):
            return
        slot = ticket.slot
        if (
            self._packed.generations[slot] != ticket.generation
            or self._packed.pins[slot] == 0
        ):
            return
        self._packed.pins[slot] -= 1
        if (
            cancel_loading
            and self._packed.pins[slot] == 0
            and not self._packed.loaded[slot]
        ):
            self._invalidate_ticket(ticket)

    def _read_contiguous_group(
        self,
        route: _InstalledNGramShard,
        offset: int,
        length: int,
        target: memoryview,
    ) -> int:
        del length
        return self._reader.read_into(route.retained, offset, target)

    def _read_segmented_group(
        self,
        route: _InstalledNGramShard,
        offset: int,
        length: int,
        target: memoryview,
    ) -> int:
        local_row = offset // self._row_bytes
        row_count = length // self._row_bytes
        return self._reader.read_segmented_into(
            route,
            local_row,
            row_count,
            self._row_bytes,
            target,
        )

    def _read_publish(
        self,
        shard_index: int,
        offset: int,
        length: int,
        transient_start: int,
        transient_rows: int,
        rows: tuple[tuple[int, SlotTicket], ...],
    ) -> None:
        route = self._physical_routes[shard_index]
        failure: str | None = None
        fatal: BaseException | None = None
        target: memoryview | None = None
        try:
            target = self._transient_pool.bytes_view(
                transient_start, transient_rows
            )
            count = self._read_group(route, offset, length, target)
            if type(count) is not int or count != length:
                raise NGramCacheIOError(
                    f"short n-gram read: expected {length} bytes, got "
                    f"{count if type(count) is int else 'invalid count'}"
                )
            with self._lock:
                if self._poison_message is not None:
                    return
                for index, (row, ticket) in enumerate(rows):
                    slot = ticket.slot
                    routed_slot = self._packed.lookup(row)
                    if (
                        not self._ticket_owned(ticket)
                        or self._packed.generations[slot] != ticket.generation
                        or self._packed.rows[slot] != row
                        or routed_slot != slot
                    ):
                        continue
                    start = ticket.slot * self._row_bytes
                    source = index * self._row_bytes
                    self._arena[start : start + self._row_bytes] = target[
                        source : source + self._row_bytes
                    ]
                    self._packed.loaded[slot] = 1
        except BaseException as exc:  # noqa: BLE001 - fatal workers must clean up
            with self._lock:
                poison_cause = exc if isinstance(exc, Exception) else RuntimeError(str(exc))
                self._poison_locked(poison_cause)
                failure = self._poison_message
                if not isinstance(exc, Exception):
                    fatal = exc
                    if self._fatal_cause is None:
                        self._fatal_cause = exc
        finally:
            if target is not None:
                target.release()
            with self._lock:
                self._inflight_bytes -= length
                self._transient_pool.release(transient_start, transient_rows)
        if failure is not None:
            if fatal is not None:
                raise fatal
            raise NGramCacheIOError(
                f"descriptor-relative n-gram read failed: {failure}"
            )

    def _poison_locked(self, cause: Exception) -> None:
        if self._poison_message is not None:
            return
        detail = str(cause).replace("\n", " ")[:256]
        self._poison_message = f"{type(cause).__name__}: {detail}"
        self._accepting = False
        for request in tuple(self._pending):
            request._cancelled = True
        self._pending.clear()
        for lease in tuple(self._leases):
            lease._released = True
        self._leases.clear()

    def _discard_worker(
        self, future: Future[None], rows: tuple[int, ...]
    ) -> None:
        with self._lock:
            self._worker_futures.discard(future)
            for row in rows:
                if self._loading_futures.get(row) is future:
                    del self._loading_futures[row]

    def _choose_slots(self, count: int, protected: set[int]) -> tuple[int, ...]:
        if count == 0:
            return ()
        selected: list[int] = []
        for slot in range(self._packed.slot_count):
            if self._packed.rows[slot] == _EMPTY_ROW and slot not in protected:
                selected.append(slot)
                if len(selected) == count:
                    return tuple(selected)
        needed = count - len(selected)
        candidates = (
            slot
            for slot in range(self._packed.slot_count)
            if self._packed.rows[slot] != _EMPTY_ROW
            and self._packed.pins[slot] == 0
            and slot not in protected
        )
        selected.extend(nsmallest(needed, candidates, key=self._victim_key))
        if len(selected) != count:
            raise NGramCacheFull("no unpinned slots for complete n-gram miss set")
        return tuple(selected)

    def _groups(
        self, misses: tuple[tuple[int, SlotTicket], ...]
    ) -> tuple[tuple[int, int, int, tuple[tuple[int, SlotTicket], ...]], ...]:
        located: list[tuple[int, int, int, SlotTicket]] = []
        for row, ticket in misses:
            route_index = bisect_right(self._physical_starts, row) - 1
            route = self._physical_routes[route_index]
            offset = route.data_offset + (row - route.start_row) * self._row_bytes
            located.append((route.index, offset, row, ticket))
        located.sort(key=lambda item: (item[0], item[1]))
        groups: list[tuple[int, int, int, tuple[tuple[int, SlotTicket], ...]]] = []
        for shard_index, offset, row, ticket in located:
            if groups:
                last_shard, last_offset, last_length, last_rows = groups[-1]
                if shard_index == last_shard and offset == last_offset + last_length:
                    groups[-1] = (
                        last_shard,
                        last_offset,
                        last_length + self._row_bytes,
                        last_rows + ((row, ticket),),
                    )
                    continue
            groups.append((shard_index, offset, self._row_bytes, ((row, ticket),)))
        return tuple(groups)

    def acquire_rows_async(self, row_ids: Sequence[int]) -> NGramAcquireFuture:
        if not isinstance(row_ids, Sequence):
            raise TypeError("row_ids must be a sequence")
        rows = tuple(row_ids)
        if not rows:
            raise ValueError("row_ids must not be empty")
        for row in rows:
            if type(row) is not int:
                raise TypeError("row_ids must contain exact integers")
            if not 0 <= row < self.manifest.padded_rows:
                raise IndexError(f"n-gram row {row} is out of range")
        unique_rows = tuple(dict.fromkeys(rows))
        with self._lock:
            poison = self._poison_message
        if poison is not None:
            self._drain_poisoned_workers()
            raise NGramCacheIOError(f"n-gram cache lane is poisoned: {poison}")
        with self._lock:
            if self._poison_message is not None:
                raise NGramCacheIOError(
                    f"n-gram cache lane is poisoned: {self._poison_message}"
                )
            if self._state != "OPEN" or not self._accepting:
                raise NGramCacheClosed("n-gram row cache is not accepting requests")
            routed = {row: self._packed.lookup(row) for row in unique_rows}
            missing = tuple(row for row in unique_rows if routed[row] is None)
            miss_bytes = len(missing) * self._row_bytes
            if self._inflight_bytes + miss_bytes > self.config.max_inflight_io_bytes:
                raise NGramCacheFull("complete n-gram miss set exceeds I/O budgets")
            protected = {slot for slot in routed.values() if slot is not None}
            selected = self._choose_slots(len(missing), protected)
            predicted_tickets = {
                row: SlotTicket(
                    slot,
                    int(self._packed.generations[slot])
                    + (2 if self._packed.rows[slot] != _EMPTY_ROW else 1),
                    self._token,
                )
                for row, slot in zip(missing, selected, strict=True)
            }
            groups = self._groups(
                tuple((row, predicted_tickets[row]) for row in missing)
            )
            reservations = self._transient_pool.reserve_fragmented(
                tuple(len(group_rows) for _shard, _offset, _length, group_rows in groups)
            )

            existing_tickets: dict[int, SlotTicket] = {}
            for row in unique_rows:
                slot = routed[row]
                if slot is None:
                    continue
                self._packed.pins[slot] += 1
                self._touch(slot)
                existing_tickets[row] = self._ticket(slot)

            new_tickets: dict[int, SlotTicket] = {}
            for row, slot_index in zip(missing, selected, strict=True):
                if self._packed.rows[slot_index] != _EMPTY_ROW:
                    self._invalidate_ticket(self._ticket(slot_index))
                self._packed.generations[slot_index] += 1
                self._packed.rows[slot_index] = row
                self._packed.pins[slot_index] = 1
                self._packed.loaded[slot_index] = 0
                self._packed.frequency[slot_index] = 0
                ticket = self._ticket(slot_index)
                if ticket != predicted_tickets[row]:
                    raise NGramCacheError("predicted slot generation changed")
                self._packed.insert(row, slot_index)
                self._touch(slot_index)
                new_tickets[row] = ticket

            self._inflight_bytes += miss_bytes
            submitted: list[Future[None]] = []
            submitted_bytes = 0
            submitted_reservations = 0
            try:
                for group_index, group_offset, group_start, group_count in reservations:
                    shard_index, base_offset, _base_length, all_group_rows = groups[
                        group_index
                    ]
                    group_rows = all_group_rows[
                        group_offset : group_offset + group_count
                    ]
                    offset = base_offset + group_offset * self._row_bytes
                    length = group_count * self._row_bytes
                    future = self._executor.submit(
                        self._read_publish,
                        shard_index,
                        offset,
                        length,
                        group_start,
                        group_count,
                        group_rows,
                    )
                    submitted.append(future)
                    submitted_bytes += length
                    submitted_reservations += 1
                    self._worker_futures.add(future)
                    group_row_ids = tuple(row for row, _ticket in group_rows)
                    future.add_done_callback(
                        lambda completed, owned=group_row_ids: self._discard_worker(
                            completed, owned
                        )
                    )
                    for row in group_row_ids:
                        self._loading_futures[row] = future
            except Exception as exc:
                unsent = miss_bytes - submitted_bytes
                self._inflight_bytes -= unsent
                for _group, _offset, start, count in reservations[
                    submitted_reservations:
                ]:
                    self._transient_pool.release(start, count)
                for ticket in existing_tickets.values():
                    self._release_ticket_pin(ticket, cancel_loading=False)
                for ticket in new_tickets.values():
                    self._invalidate_ticket(ticket)
                raise NGramCacheIOError(f"could not submit n-gram reads: {exc}") from exc

            waiting = list(submitted)
            for row in unique_rows:
                loading = self._loading_futures.get(row)
                if loading is not None:
                    waiting.append(loading)
            futures = tuple(dict.fromkeys(waiting))
            by_row = existing_tickets | new_tickets
            request = NGramAcquireFuture(
                self,
                tuple(by_row[row] for row in rows),
                futures,
                self._token,
                _CACHE_CONSTRUCTION_KEY,
            )
            self._pending.add(request)
            return request

    def acquire_rows(self, row_ids: Sequence[int]) -> NGramLease:
        return self.acquire_rows_async(row_ids).result()

    def _cancel_request(self, request: NGramAcquireFuture) -> bool:
        with self._lock:
            if (
                type(request) is not NGramAcquireFuture
                or request._cache is not self
                or request._token is not self._token
            ):
                raise TypeError("acquisition future does not belong to this cache")
            if request._lease is not None or request._cancelled or request._failed:
                return False
            if request not in self._pending:
                raise TypeError("acquisition future is not registered with this cache")
            request._cancelled = True
            self._pending.discard(request)
            for ticket in request._unique_tickets:
                self._release_ticket_pin(ticket, cancel_loading=True)
            return True

    def _fail_request(self, request: NGramAcquireFuture) -> None:
        with self._lock:
            if request._cache is not self or request._token is not self._token:
                raise TypeError("acquisition future does not belong to this cache")
            if request._failed or request._cancelled or request._lease is not None:
                return
            request._failed = True
            self._pending.discard(request)
            for ticket in request._unique_tickets:
                self._release_ticket_pin(ticket, cancel_loading=True)

    def _finish_request(self, request: NGramAcquireFuture) -> NGramLease:
        with self._lock:
            if request._cache is not self or request._token is not self._token:
                raise TypeError("acquisition future does not belong to this cache")
            if self._poison_message is not None:
                raise NGramCacheIOError(
                    f"n-gram cache lane is poisoned: {self._poison_message}"
                )
            if request._cancelled or self._state != "OPEN" or not self._accepting:
                raise CancelledError()
            if request._failed:
                raise NGramCacheIOError("n-gram row acquisition failed")
            for ticket in request._unique_tickets:
                if (
                    not self._ticket_owned(ticket)
                    or self._packed.generations[ticket.slot] != ticket.generation
                    or not self._packed.loaded[ticket.slot]
                ):
                    self._fail_request(request)
                    raise CancelledError()
            lease = NGramLease(
                self,
                request._tickets,
                self._token,
                _CACHE_CONSTRUCTION_KEY,
            )
            request._lease = lease
            self._pending.discard(request)
            self._leases.add(lease)
            return lease

    def _lease_row_bytes(self, lease: NGramLease, ticket: SlotTicket) -> bytes:
        with self._lock:
            if (
                type(lease) is not NGramLease
                or lease._cache is not self
                or lease._token is not self._token
                or not self._ticket_owned(ticket)
            ):
                raise TypeError("lease or ticket does not belong to this cache")
            if self._poison_message is not None:
                raise NGramCacheIOError(
                    f"n-gram cache lane is poisoned: {self._poison_message}"
                )
            if (
                lease._released
                or lease not in self._leases
                or self._state != "OPEN"
            ):
                raise NGramCacheClosed("n-gram lease is no longer active")
            if (
                self._packed.generations[ticket.slot] != ticket.generation
                or not self._packed.loaded[ticket.slot]
            ):
                raise NGramCacheClosed("n-gram lease no longer owns its slot")
            start = ticket.slot * self._row_bytes
            return bytes(self._arena[start : start + self._row_bytes])

    def _release_lease(self, lease: NGramLease) -> None:
        with self._lock:
            if (
                type(lease) is not NGramLease
                or lease._cache is not self
                or lease._token is not self._token
            ):
                raise TypeError("lease does not belong to this cache")
            if lease._released:
                return
            if lease not in self._leases:
                raise TypeError("lease is not registered with this cache")
            lease._released = True
            self._leases.discard(lease)
            for ticket in dict.fromkeys(lease._tickets):
                self._release_ticket_pin(ticket, cancel_loading=False)

    def _invalidate_all_locked(self) -> tuple[Future[None], ...]:
        for request in tuple(self._pending):
            self._cancel_request(request)
        for lease in tuple(self._leases):
            lease._released = True
        self._leases.clear()
        self._packed.clear()
        self._loading_futures.clear()
        return tuple(self._worker_futures)

    @staticmethod
    def _drain(futures: tuple[Future[None], ...]) -> BaseException | None:
        first_fatal: BaseException | None = None
        for future in futures:
            try:
                future.result()
            except (CancelledError, NGramCacheError):
                pass
            except BaseException as exc:  # noqa: BLE001 - drain before fatal re-raise
                if first_fatal is None:
                    first_fatal = exc
        return first_fatal

    def _drain_poisoned_workers(self) -> None:
        with self._lock:
            futures = tuple(self._worker_futures)
        fatal = self._drain(futures)
        if fatal is None:
            fatal = self._fatal_cause
        if fatal is not None:
            raise fatal

    def reset(self) -> None:
        with self._condition:
            while self._resetting and self._state == "OPEN":
                self._condition.wait()
            if self._state != "OPEN":
                raise NGramCacheClosed("n-gram row cache is closed")
            if self._poison_message is not None:
                raise NGramCacheIOError(
                    f"n-gram cache lane is poisoned: {self._poison_message}"
                )
            self._resetting = True
            self._accepting = False
            futures = self._invalidate_all_locked()
        fatal = self._drain(futures)
        if fatal is None:
            fatal = self._fatal_cause
        with self._condition:
            self._resetting = False
            if self._state == "OPEN" and self._poison_message is None:
                self._accepting = True
            self._condition.notify_all()
            poison = self._poison_message
        if poison is not None:
            if fatal is not None:
                raise fatal
            raise NGramCacheIOError(f"n-gram cache lane is poisoned: {poison}")
        if fatal is not None:
            raise fatal

    def close(self) -> None:
        with self._condition:
            if self._state == "CLOSED":
                if self._close_failure is not None:
                    raise self._close_failure
                if self._fatal_cause is not None:
                    raise self._fatal_cause
                if self._close_error is not None:
                    raise NGramCacheIOError(self._close_error)
                return
            if self._state == "CLOSING":
                while self._state != "CLOSED":
                    self._condition.wait()
                if self._close_failure is not None:
                    raise self._close_failure
                if self._fatal_cause is not None:
                    raise self._fatal_cause
                if self._close_error is not None:
                    raise NGramCacheIOError(self._close_error)
                return
            while self._resetting:
                self._condition.wait()
            self._state = "CLOSING"
            self._accepting = False
            futures = self._invalidate_all_locked()
        fatal = self._drain(futures)
        if fatal is None:
            fatal = self._fatal_cause
        first_failure: BaseException | None = fatal
        cleanup_uncertain = False
        try:
            try:
                self._executor.shutdown(wait=True, cancel_futures=False)
            except BaseException as exc:  # noqa: BLE001 - finalization must continue
                cleanup_uncertain = True
                if first_failure is None:
                    first_failure = exc
            try:
                _restore_page_cache_policy(
                    self._physical_routes, self._page_cache_installed
                )
            except BaseException as exc:  # noqa: BLE001 - finalization must continue
                cleanup_uncertain = True
                if first_failure is None:
                    first_failure = exc
            try:
                self._arena.release()
            except BaseException as exc:  # noqa: BLE001 - drop every fixed owner
                cleanup_uncertain = True
                if first_failure is None:
                    first_failure = exc
            finally:
                self._arena = None
                self._arena_object = None
            try:
                self._packed.release()
            except BaseException as exc:  # noqa: BLE001 - drop every fixed owner
                cleanup_uncertain = True
                if first_failure is None:
                    first_failure = exc
            try:
                self._transient_pool.close()
            except BaseException as exc:  # noqa: BLE001 - drop every fixed owner
                cleanup_uncertain = True
                if first_failure is None:
                    first_failure = exc
            self._physical_routes = ()
            self._physical_starts = ()
        finally:
            with self._condition:
                with self.artifact._lock:
                    if cleanup_uncertain:
                        detail_source = first_failure or RuntimeError(
                            "unknown cache close failure"
                        )
                        detail = (
                            f"{type(detail_source).__name__}: "
                            f"{str(detail_source)[:448]}"
                        )
                        self._close_error = detail
                        self.artifact._cache_reuse_error = detail
                    if self.artifact._cache_owner is self._token:
                        self.artifact._cache_owner = None
                self._close_failure = first_failure
                self._state = "CLOSED"
                self._condition.notify_all()
        if first_failure is not None:
            raise first_failure

    def __copy__(self) -> None:
        raise TypeError("copy is forbidden for n-gram row caches")

    def __deepcopy__(self, memo: dict[int, Any]) -> None:
        del memo
        raise TypeError("deepcopy is forbidden for n-gram row caches")

    def __enter__(self) -> Self:
        if self.closed:
            raise NGramCacheClosed("n-gram row cache is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = [
    "QWEN38_FLASH_NEXT_REPO",
    "QWEN38_FLASH_NEXT_REVISION",
    "NGramAcquireFuture",
    "NGramCacheClosed",
    "NGramCacheConfig",
    "NGramCacheError",
    "NGramCacheFull",
    "NGramCacheIOError",
    "NGramCachePlan",
    "NGramComponent",
    "NGramFileIdentity",
    "NGramGeometry",
    "NGramLease",
    "NGramManifest",
    "NGramManifestError",
    "NGramProductionCachePlan",
    "NGramRowCache",
    "NGramRuntimeBudget",
    "NGramShard",
    "SlotTicket",
    "VerifiedNGramArtifact",
    "VerifiedNGramShard",
    "load_ngram_manifest",
    "plan_ngram_cache",
    "plan_production_ngram_cache",
    "qwen38_ngram_manifest",
    "save_ngram_manifest",
    "segmented_ngram_shard",
    "verify_ngram_manifest",
]
