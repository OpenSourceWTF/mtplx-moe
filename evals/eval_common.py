"""Shared, memory-safe helpers for the hy3 q2 Tier-1 correctness evals.

HARD RULES honored here:
  * CPU only. ``mx.set_default_device(mx.cpu)`` is called at import so no eval
    ever touches the Metal / GPU wired allocator. We never load the full model;
    every expert is read ONE record at a time (~5.9 MB q2 / ~10.6 MB q4) and the
    bf16 comparison tensors are pread'd one tensor at a time (~12.5 MB) directly
    out of their safetensors shard via the header byte-offsets -- no shard is
    ever fully materialized into an array.
  * No network. Everything resolves against the local HF cache.

All artifact reads go through the repository's OWN reader
(``mtplx.expert_manifest.read_expert_record``), which hash-verifies each record
against the manifest's recorded sha256. We do NOT hand-roll the bank parser.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

import mlx.core as mx

# ---- HARD RULE: never touch the GPU wired allocator. -----------------------
mx.set_default_device(mx.cpu)

HOME = Path(os.path.expanduser("~"))
Q2_DIR = HOME / ".cache/huggingface/hy3-expert-only-mlx-q2"
Q4_DIR = HOME / ".cache/huggingface/hy3-expert-only-mlx-q4"
BF16_DIR = HOME / ".cache/huggingface/hy3-bf16-and-mtp-layer80"

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_JSON = EVALS_DIR / "results.json"

# hy3 config constants (from the q2 config.json / target_descriptor).
HIDDEN_SIZE = 4096
EXPERT_HIDDEN_SIZE = 1536
GROUP_SIZE = 64
NUM_EXPERTS = 192
ROUTED_LAYERS = tuple(range(1, 80))  # first_k_dense_replace=1 -> layer 0 dense


def _np_dtype(dtype: str):
    if dtype == "U32":
        return "<u4"
    if dtype == "BF16":
        return "<u2"
    raise ValueError(f"unsupported segment dtype {dtype!r}")


def bytes_to_mx(buf: bytes, dtype: str, shape) -> mx.array:
    """Reinterpret raw little-endian bytes as an mx.array of the given dtype.

    Mirrors exactly the decode path the shipped converter uses
    (``mtplx.hy3_expert_q2.requantize_projection_q4_to_q2.decode_bf16`` and the
    U32 weight path): frombuffer -> mx.array(uint) -> ``.view`` for bf16.
    """
    arr = np.frombuffer(buf, dtype=_np_dtype(dtype)).copy().reshape(tuple(shape))
    if dtype == "U32":
        return mx.array(arr, dtype=mx.uint32)
    words = mx.array(arr, dtype=mx.uint16)
    return words.view(mx.bfloat16)


def split_record_segments(record, payload: bytes) -> dict[str, mx.array]:
    """Split a hash-verified record payload into {component: mx.array}.

    Segment offsets in the manifest are experts.bin-absolute; within a single
    record they are contiguous, so we slice by ``offset - base``.
    """
    base = record.segments[0].offset
    out: dict[str, mx.array] = {}
    for seg in record.segments:
        rel = seg.offset - base
        buf = payload[rel : rel + seg.length]
        if len(buf) != seg.length:
            raise ValueError(f"short slice for {seg.component}")
        out[seg.component] = bytes_to_mx(buf, seg.dtype, seg.shape)
    return out


def dequant_projection(arrs: dict[str, mx.array], projection: str, bits: int) -> mx.array:
    """Dequantize one projection with MLX's reference affine dequantizer."""
    w = arrs[f"{projection}.weight"]
    s = arrs[f"{projection}.scales"]
    b = arrs[f"{projection}.biases"]
    d = mx.dequantize(w, s, b, bits=bits, group_size=GROUP_SIZE, mode="affine")
    mx.eval(d)
    return d


# ---- bf16 single-tensor reader (no full-shard materialization) --------------

_ST_HEADER_CACHE: dict[Path, tuple[int, dict[str, Any]]] = {}


def _st_header(shard_path: Path) -> tuple[int, dict[str, Any]]:
    cached = _ST_HEADER_CACHE.get(shard_path)
    if cached is not None:
        return cached
    with open(shard_path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    _ST_HEADER_CACHE[shard_path] = (hlen, header)
    return hlen, header


def read_bf16_tensor(model_dir: Path, tensor_name: str) -> mx.array:
    """pread a single bf16 tensor out of its safetensors shard as mx.bfloat16."""
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][tensor_name]
    shard_path = model_dir / shard
    hlen, header = _st_header(shard_path)
    meta = header[tensor_name]
    if meta["dtype"] != "BF16":
        raise ValueError(f"{tensor_name} is {meta['dtype']}, expected BF16")
    start, end = meta["data_offsets"]
    fd = os.open(shard_path, os.O_RDONLY)
    try:
        raw = os.pread(fd, end - start, 8 + hlen + start)
    finally:
        os.close(fd)
    if len(raw) != end - start:
        raise ValueError(f"short read for {tensor_name}")
    words = mx.array(
        np.frombuffer(raw, dtype="<u2").copy().reshape(meta["shape"]),
        dtype=mx.uint16,
    )
    return words.view(mx.bfloat16)


# hy3 HF expert tensor name for a per-expert projection weight.
def bf16_expert_tensor(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


# ---- metrics ----------------------------------------------------------------

def cosine_and_relerr(a: mx.array, b: mx.array) -> tuple[float, float]:
    """Cosine similarity and relative Frobenius error of ``a`` vs reference ``b``."""
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    dot = float(mx.sum(af * bf).item())
    na = float(mx.linalg.norm(af).item())
    nb = float(mx.linalg.norm(bf).item())
    cos = dot / (na * nb) if na and nb else 0.0
    relerr = float(mx.linalg.norm(af - bf).item()) / nb if nb else float("inf")
    return cos, relerr


# ---- results.json aggregation ----------------------------------------------

def update_results(section: str, payload: dict[str, Any]) -> None:
    """Read-modify-write the shared results.json under ``section``."""
    data: dict[str, Any] = {}
    if RESULTS_JSON.exists():
        try:
            data = json.loads(RESULTS_JSON.read_text())
        except json.JSONDecodeError:
            data = {}
    data[section] = payload
    RESULTS_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
