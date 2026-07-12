#!/usr/bin/env python3
"""Extract Tencent Hy3 MTP (NextN) layer 80 tensors into one consolidated safetensors file.

Reads the BF16 shards downloaded from tencent/Hy3 (revision recorded in the
output metadata), pulls every ``model.layers.80.*`` tensor, and writes them to
``layer80-bf16.safetensors`` in the same directory.

CPU-only, numpy/stdlib-only. safetensors' numpy framework cannot represent
BF16 (numpy has no bfloat16 dtype), so this script parses the safetensors
container format directly and copies raw bytes verbatim -- no dtype
conversion of any kind. The output file therefore preserves the exact BF16
(and F32) bits of the source checkpoint.

Usage:
    python scripts/extract_mtp_layer80.py \
        [--src /Users/davidtai/.cache/huggingface/hy3-mtp-layer80] \
        [--out layer80-bf16.safetensors]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

LAYER_PREFIX = "model.layers.80."
SOURCE_REPO = "tencent/Hy3"
SOURCE_REVISION = "716aa7241bd6d95896be4ebfc761162a9c4d49ef"

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U32": 4, "I32": 4, "U8": 1, "I8": 1, "I64": 8}


def read_safetensors_header(path: Path):
    """Return (header_dict, data_start_offset) for a safetensors file."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def expected_shapes(cfg: dict) -> dict:
    """Expected shapes for layer-80 tensors, derived from the model config."""
    h = cfg["hidden_size"]                     # 4096
    moe_i = cfg["moe_intermediate_size"]       # 1536
    n_exp = cfg["num_experts"]                 # 192
    n_heads = cfg["num_attention_heads"]       # 64
    n_kv = cfg["num_key_value_heads"]          # 8
    hd = cfg["head_dim"]                       # 128

    exp = {
        "eh_proj.weight": (h, 2 * h),
        "enorm.weight": (h,),
        "hnorm.weight": (h,),
        "input_layernorm.weight": (h,),
        "post_attention_layernorm.weight": (h,),
        "final_layernorm.weight": (h,),
        "self_attn.q_proj.weight": (n_heads * hd, h),
        "self_attn.k_proj.weight": (n_kv * hd, h),
        "self_attn.v_proj.weight": (n_kv * hd, h),
        "self_attn.o_proj.weight": (h, n_heads * hd),
        "self_attn.q_norm.weight": (hd,),
        "self_attn.k_norm.weight": (hd,),
        "mlp.expert_bias": (n_exp,),
        "mlp.router.gate.weight": (n_exp, h),
        "mlp.shared_mlp.gate_proj.weight": (moe_i, h),
        "mlp.shared_mlp.up_proj.weight": (moe_i, h),
        "mlp.shared_mlp.down_proj.weight": (h, moe_i),
    }
    for i in range(n_exp):
        exp[f"mlp.experts.{i}.gate_proj.weight"] = (moe_i, h)
        exp[f"mlp.experts.{i}.up_proj.weight"] = (moe_i, h)
        exp[f"mlp.experts.{i}.down_proj.weight"] = (h, moe_i)
    return exp


def bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    """Reinterpret raw BF16 bytes as float32 (for sanity stats only)."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="/Users/davidtai/.cache/huggingface/hy3-mtp-layer80",
                    help="Directory holding the downloaded shards + index")
    ap.add_argument("--out", default="layer80-bf16.safetensors",
                    help="Output filename (relative to --src unless absolute)")
    args = ap.parse_args()

    src = Path(args.src)
    out_path = Path(args.out) if Path(args.out).is_absolute() else src / args.out

    index = json.loads((src / "model.safetensors.index.json").read_text())
    cfg = json.loads((src / "config.json").read_text())
    weight_map = index["weight_map"]

    layer80 = {name: shard for name, shard in weight_map.items() if name.startswith(LAYER_PREFIX)}
    if not layer80:
        print("ERROR: no model.layers.80.* tensors in index", file=sys.stderr)
        return 1

    shards = sorted(set(layer80.values()))
    print(f"layer-80 tensors in index : {len(layer80)}")
    print(f"shards involved           : {len(shards)}")
    for s in shards:
        if not (src / s).exists():
            print(f"ERROR: missing shard {s}", file=sys.stderr)
            return 1

    # Collect (name, dtype, shape, raw_bytes) by streaming each shard once.
    tensors: dict[str, tuple[str, list, bytes]] = {}
    for shard in shards:
        path = src / shard
        header, data_start = read_safetensors_header(path)
        wanted = [n for n, s in layer80.items() if s == shard]
        with open(path, "rb") as f:
            for name in wanted:
                meta = header.get(name)
                if meta is None:
                    print(f"ERROR: {name} not in header of {shard}", file=sys.stderr)
                    return 1
                start, end = meta["data_offsets"]
                nbytes = end - start
                expect = int(np.prod(meta["shape"], dtype=np.int64)) * DTYPE_BYTES[meta["dtype"]] if meta["shape"] else DTYPE_BYTES[meta["dtype"]]
                if nbytes != expect:
                    print(f"ERROR: byte-count mismatch for {name}: {nbytes} != {expect}", file=sys.stderr)
                    return 1
                f.seek(data_start + start)
                raw = f.read(nbytes)
                if len(raw) != nbytes:
                    print(f"ERROR: short read for {name}", file=sys.stderr)
                    return 1
                tensors[name] = (meta["dtype"], meta["shape"], raw)
        print(f"  read {len(wanted):4d} tensors from {shard}")

    # ---- shape sanity check against config ----
    exp = expected_shapes(cfg)
    surprises = []
    unchecked = []
    for name, (dtype, shape, _) in sorted(tensors.items()):
        suffix = name[len(LAYER_PREFIX):]
        want = exp.get(suffix)
        if want is None:
            unchecked.append(name)
        elif tuple(shape) != want:
            surprises.append(f"{name}: got {tuple(shape)}, expected {want}")
    print(f"shape-checked             : {len(tensors) - len(unchecked)}/{len(tensors)}")
    if unchecked:
        print("tensors without an expected-shape rule:")
        for n in unchecked:
            print(f"  ? {n} {tensors[n][0]} {tensors[n][1]}")
    if surprises:
        print("SHAPE SURPRISES:")
        for s in surprises:
            print(f"  ! {s}")
    else:
        print("shape surprises           : none")

    # ---- numeric sanity spot-check (BF16 -> F32 reinterpret) ----
    for probe in (LAYER_PREFIX + "mlp.experts.0.gate_proj.weight", LAYER_PREFIX + "eh_proj.weight"):
        dtype, shape, raw = tensors[probe]
        if dtype == "BF16":
            vals = bf16_bytes_to_f32(raw)
            finite = np.isfinite(vals).all()
            print(f"probe {probe}: finite={finite} absmax={np.abs(vals).max():.4f} mean={vals.mean():+.6f}")
            if not finite:
                print("ERROR: non-finite values in probe tensor", file=sys.stderr)
                return 1

    # ---- write consolidated safetensors (manual, exact raw-byte copy) ----
    names = sorted(tensors)
    out_header: dict = {"__metadata__": {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "extracted_prefix": LAYER_PREFIX,
        "producer": "scripts/extract_mtp_layer80.py",
    }}
    offset = 0
    for name in names:
        dtype, shape, raw = tensors[name]
        out_header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    header_bytes = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8  # align data section, spec allows trailing spaces
    header_bytes += b" " * pad

    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for name in names:
            f.write(tensors[name][2])

    total_bytes = offset
    by_dtype: dict[str, int] = {}
    for name in names:
        dtype, _, raw = tensors[name]
        by_dtype[dtype] = by_dtype.get(dtype, 0) + len(raw)
    print(f"\nwrote {out_path}")
    print(f"tensors                   : {len(names)}")
    print(f"tensor bytes              : {total_bytes:,} ({total_bytes / 2**30:.3f} GiB)")
    for dt, b in sorted(by_dtype.items()):
        print(f"  {dt:5s}: {b:,} bytes")

    # verify the file re-opens cleanly
    hdr, _ = read_safetensors_header(out_path)
    n_out = len([k for k in hdr if k != "__metadata__"])
    assert n_out == len(names), f"reopen mismatch: {n_out} != {len(names)}"
    print(f"reopen check              : OK ({n_out} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
