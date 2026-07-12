"""No-pickle tensor/tree codec for persistent SessionBank snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from mtplx.cache_state import CacheSnapshot


_MLX_TO_NUMPY_DTYPE: dict[str, Any] = {
    "bool": np.bool_,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}

_NUMPY_TO_MLX_DTYPE: dict[str, Any] = {
    "bool": mx.bool_,
    "uint8": mx.uint8,
    "uint16": mx.uint16,
    "uint32": mx.uint32,
    "uint64": mx.uint64,
    "int8": mx.int8,
    "int16": mx.int16,
    "int32": mx.int32,
    "int64": mx.int64,
    "float16": mx.float16,
    "float32": mx.float32,
    "float64": mx.float64,
}


@dataclass(frozen=True)
class EncodedPayload:
    spec: dict[str, Any]
    tensors: dict[str, bytes]
    nbytes: int


@dataclass(frozen=True)
class DecodedPayload:
    cache_snapshot: CacheSnapshot
    logits: Any
    hidden: Any | None
    mtp_history_snapshot: CacheSnapshot | None
    # v3: (token_count, recurrent-only CacheSnapshot, hidden_last|None)
    gdn_boundaries: tuple = ()
    has_recurrent: bool = False


class TreeCodec:
    """Flatten JSON-safe trees plus MLX arrays into raw tensor blobs."""

    def __init__(self, *, block_size: int = 256) -> None:
        self._next_tensor_id = 0
        self.tensors: dict[str, bytes] = {}
        self.block_size = max(1, int(block_size))

    def encode(self, value: Any) -> Any:
        if value is None:
            return {"kind": "none"}
        if isinstance(value, bool):
            return {"kind": "bool", "value": bool(value)}
        if isinstance(value, int):
            return {"kind": "int", "value": int(value)}
        if isinstance(value, float):
            return {"kind": "float", "value": float(value)}
        if isinstance(value, str):
            return {"kind": "str", "value": value}
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, Path):
            return {"kind": "str", "value": str(value)}
        if isinstance(value, mx.array):
            return self._encode_tensor(value)
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [self.encode(item) for item in value]}
        if isinstance(value, list):
            return {"kind": "list", "items": [self.encode(item) for item in value]}
        if isinstance(value, dict):
            items = []
            for key, item in value.items():
                if not isinstance(key, (str, int, float, bool)):
                    raise TypeError(f"unsupported dict key in SessionBank snapshot: {type(key)!r}")
                items.append([self.encode(key), self.encode(item)])
            return {"kind": "dict", "items": items}
        raise TypeError(f"unsupported SessionBank snapshot leaf: {type(value)!r}")

    def _encode_tensor(self, value: Any) -> dict[str, Any]:
        mx.eval(value)
        dtype = _dtype_name(value.dtype)
        shape = [int(dim) for dim in value.shape]
        if len(shape) >= 3 and shape[2] >= self.block_size * 2:
            return self._encode_tensor_blocks(value, dtype=dtype, shape=shape)
        name = f"tensor_{self._next_tensor_id:08d}"
        self._next_tensor_id += 1
        raw = bytes(memoryview(value))
        self.tensors[name] = raw
        return {
            "kind": "tensor",
            "name": name,
            "dtype": dtype,
            "shape": [int(dim) for dim in value.shape],
            "nbytes": len(raw),
        }

    def _encode_tensor_blocks(
        self,
        value: Any,
        *,
        dtype: str,
        shape: list[int],
    ) -> dict[str, Any]:
        axis = 2
        blocks: list[dict[str, Any]] = []
        total = 0
        for start in range(0, shape[axis], self.block_size):
            end = min(shape[axis], start + self.block_size)
            slices = [slice(None)] * len(shape)
            slices[axis] = slice(start, end)
            chunk = value[tuple(slices)]
            mx.eval(chunk)
            raw = bytes(memoryview(chunk))
            name = f"tensor_{self._next_tensor_id:08d}"
            self._next_tensor_id += 1
            self.tensors[name] = raw
            total += len(raw)
            blocks.append(
                {
                    "name": name,
                    "start": int(start),
                    "end": int(end),
                    "shape": [int(dim) for dim in chunk.shape],
                    "nbytes": len(raw),
                }
            )
        return {
            "kind": "tensor_blocks",
            "dtype": dtype,
            "shape": shape,
            "axis": axis,
            "block_size": self.block_size,
            "blocks": blocks,
            "nbytes": total,
        }


def decode_tree(spec: Any, read_tensor: Callable[[str], bytes]) -> Any:
    kind = spec.get("kind") if isinstance(spec, dict) else None
    if kind == "none":
        return None
    if kind == "bool":
        return bool(spec["value"])
    if kind == "int":
        return int(spec["value"])
    if kind == "float":
        return float(spec["value"])
    if kind == "str":
        return str(spec["value"])
    if kind == "tuple":
        return tuple(decode_tree(item, read_tensor) for item in spec.get("items", []))
    if kind == "list":
        return [decode_tree(item, read_tensor) for item in spec.get("items", [])]
    if kind == "dict":
        return {
            decode_tree(key, read_tensor): decode_tree(value, read_tensor)
            for key, value in spec.get("items", [])
        }
    if kind == "tensor":
        return _decode_tensor(spec, read_tensor)
    if kind == "tensor_blocks":
        return _decode_tensor_blocks(spec, read_tensor)
    raise ValueError(f"unsupported SessionBank payload spec kind: {kind!r}")


def encode_payload(
    *,
    cache_snapshot: CacheSnapshot,
    logits: Any,
    hidden: Any | None,
    mtp_history_snapshot: CacheSnapshot | None,
    gdn_boundaries: tuple | list | None = None,
    has_recurrent: bool | None = None,
    block_size: int = 256,
) -> EncodedPayload:
    codec = TreeCodec(block_size=block_size)
    spec = {
        "cache_snapshot": {
            "states": codec.encode(cache_snapshot.states),
            "meta_states": codec.encode(cache_snapshot.meta_states),
        },
        "logits": codec.encode(logits),
        "hidden": codec.encode(hidden),
        "mtp_history_snapshot": (
            None
            if mtp_history_snapshot is None
            else {
                "states": codec.encode(mtp_history_snapshot.states),
                "meta_states": codec.encode(mtp_history_snapshot.meta_states),
            }
        ),
        # v3: interior recurrent boundaries make persisted entries usable for
        # sub-prefix (partial) restores on hybrid models after a restart.
        "gdn_boundaries": [
            {
                "tokens": int(record[0]),
                "states": codec.encode(record[1].states),
                "meta_states": codec.encode(record[1].meta_states),
                "hidden_last": codec.encode(
                    record[2] if len(record) > 2 else None
                ),
            }
            for record in (gdn_boundaries or [])
        ],
        "has_recurrent": bool(
            has_recurrent
            if has_recurrent is not None
            else bool(gdn_boundaries)
        ),
    }
    return EncodedPayload(
        spec=spec,
        tensors=dict(codec.tensors),
        nbytes=sum(len(raw) for raw in codec.tensors.values()),
    )


def decode_payload(
    spec: dict[str, Any],
    read_tensor: Callable[[str], bytes],
    *,
    include_gdn_boundaries: bool = True,
) -> DecodedPayload:
    cache_spec = spec["cache_snapshot"]
    cache_snapshot = CacheSnapshot(
        states=tuple(decode_tree(cache_spec["states"], read_tensor)),
        meta_states=tuple(decode_tree(cache_spec["meta_states"], read_tensor)),
    )
    mtp_spec = spec.get("mtp_history_snapshot")
    mtp_history_snapshot = None
    if mtp_spec is not None:
        mtp_history_snapshot = CacheSnapshot(
            states=tuple(decode_tree(mtp_spec["states"], read_tensor)),
            meta_states=tuple(decode_tree(mtp_spec["meta_states"], read_tensor)),
        )
    gdn_boundaries = (
        decode_gdn_boundaries(spec, read_tensor)
        if include_gdn_boundaries
        else ()
    )
    decoded = DecodedPayload(
        cache_snapshot=cache_snapshot,
        logits=decode_tree(spec["logits"], read_tensor),
        hidden=decode_tree(spec["hidden"], read_tensor),
        mtp_history_snapshot=mtp_history_snapshot,
        gdn_boundaries=gdn_boundaries,
        has_recurrent=bool(spec.get("has_recurrent", False)),
    )
    _eval_decoded_arrays(decoded)
    return decoded


def _eval_decoded_arrays(decoded: DecodedPayload) -> None:
    """Single batched evaluation of every decoded array (vs per-tensor eval)."""
    arrays: list[Any] = []

    def collect(value: Any) -> None:
        if isinstance(value, mx.array):
            arrays.append(value)
        elif isinstance(value, CacheSnapshot):
            collect(value.states)
            collect(value.meta_states)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(decoded.cache_snapshot)
    collect(decoded.logits)
    collect(decoded.hidden)
    collect(decoded.mtp_history_snapshot)
    collect(decoded.gdn_boundaries)
    if arrays:
        mx.eval(*arrays)


def decode_gdn_boundaries(
    spec: dict[str, Any], read_tensor: Callable[[str], bytes]
) -> tuple:
    """Decode only the interior recurrent boundaries from a payload spec.

    Used lazily by SSD-restored entries: exact restores skip the MB-scale
    boundary payloads entirely, and partial restores load them on demand
    through this helper (batched single eval)."""
    boundaries = tuple(
        (
            int(record["tokens"]),
            CacheSnapshot(
                states=tuple(decode_tree(record["states"], read_tensor)),
                meta_states=tuple(decode_tree(record["meta_states"], read_tensor)),
            ),
            decode_tree(record.get("hidden_last") or {"kind": "none"}, read_tensor),
        )
        for record in (spec.get("gdn_boundaries") or [])
    )
    arrays: list[Any] = []

    def collect(value: Any) -> None:
        if isinstance(value, mx.array):
            arrays.append(value)
        elif isinstance(value, CacheSnapshot):
            collect(value.states)
            collect(value.meta_states)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(boundaries)
    if arrays:
        mx.eval(*arrays)
    return boundaries


def _decode_tensor(spec: dict[str, Any], read_tensor: Callable[[str], bytes]) -> Any:
    dtype = str(spec["dtype"])
    shape = tuple(int(dim) for dim in spec.get("shape") or [])
    raw = read_tensor(str(spec["name"]))
    if dtype == "bfloat16":
        arr = mx.array(np.frombuffer(raw, dtype=np.uint16)).view(mx.bfloat16)
    else:
        np_dtype = _MLX_TO_NUMPY_DTYPE.get(dtype)
        mlx_dtype = _NUMPY_TO_MLX_DTYPE.get(dtype)
        if np_dtype is None or mlx_dtype is None:
            raise ValueError(f"unsupported persisted tensor dtype: {dtype!r}")
        arr = mx.array(np.frombuffer(raw, dtype=np_dtype), dtype=mlx_dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def _decode_tensor_blocks(spec: dict[str, Any], read_tensor: Callable[[str], bytes]) -> Any:
    axis = int(spec.get("axis", 2))
    chunks = [_decode_tensor({**block, "dtype": spec["dtype"]}, read_tensor) for block in spec.get("blocks", [])]
    if not chunks:
        shape = tuple(int(dim) for dim in spec.get("shape") or [])
        return mx.zeros(shape)
    arr = mx.concatenate(chunks, axis=axis)
    shape = tuple(int(dim) for dim in spec.get("shape") or [])
    if shape:
        arr = arr.reshape(shape)
    return arr


def _dtype_name(dtype: Any) -> str:
    raw = str(dtype)
    if raw.startswith("mlx.core."):
        raw = raw.removeprefix("mlx.core.")
    if raw == "bfloat16":
        return raw
    if raw in _MLX_TO_NUMPY_DTYPE:
        return raw
    raise TypeError(f"unsupported MLX tensor dtype for SessionBank SSD cache: {dtype!r}")
