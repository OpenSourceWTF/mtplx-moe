"""Read a single tensor region out of a local safetensors shard via pread.

Never loads the whole 7 GB shard. Parses the shard's JSON header once
(8-byte length prefix + JSON), then os.pread's exactly the tensor's byte
span. BF16 is returned as fp32 (exact u16<<16 reinterpret).
"""
from __future__ import annotations

import json
import os
import struct
import hashlib

import numpy as np

_HEADER_CACHE: dict[str, tuple[int, dict]] = {}


def _load_header(path: str):
    cached = _HEADER_CACHE.get(path)
    if cached is not None:
        return cached
    fd = os.open(path, os.O_RDONLY)
    try:
        (hlen,) = struct.unpack("<Q", os.pread(fd, 8, 0))
        raw = os.pread(fd, hlen, 8)
    finally:
        os.close(fd)
    header = json.loads(raw)
    data_start = 8 + hlen
    _HEADER_CACHE[path] = (data_start, header)
    return data_start, header


def read_tensor_fp32(path: str, name: str):
    """Return (fp32 ndarray shaped as stored, meta dict with offsets/sha)."""
    data_start, header = _load_header(path)
    if name not in header:
        raise KeyError(f"{name} not in {path}")
    ent = header[name]
    dtype = ent["dtype"]
    shape = tuple(ent["shape"])
    b0, b1 = ent["data_offsets"]
    abs0 = data_start + b0
    nbytes = b1 - b0
    fd = os.open(path, os.O_RDONLY)
    try:
        raw = os.pread(fd, nbytes, abs0)
    finally:
        os.close(fd)
    if len(raw) != nbytes:
        raise IOError(f"short read {len(raw)} != {nbytes} for {name}")
    sha = hashlib.sha256(raw).hexdigest()
    if dtype in ("BF16", "bfloat16"):
        u16 = np.frombuffer(raw, dtype="<u2")
        f32 = (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)
        arr = f32.reshape(shape)
    elif dtype in ("F16", "float16"):
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
    elif dtype in ("F32", "float32"):
        arr = np.frombuffer(raw, dtype="<f4").reshape(shape)
    else:
        raise ValueError(f"unhandled dtype {dtype} for {name}")
    meta = {
        "shard_path": path,
        "dtype": dtype,
        "shape": list(shape),
        "data_start": data_start,
        "rel_begin": b0,
        "rel_end": b1,
        "abs_begin": abs0,
        "nbytes": nbytes,
        "sha256": sha,
    }
    return arr, meta
