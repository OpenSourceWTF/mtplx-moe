#!/usr/bin/env python3
"""Quantize Hy3 MTP layer-80 expert projections to affine Q4 (MLX format).

Input : layer80-bf16.safetensors (produced by scripts/extract_mtp_layer80.py)
Output: layer80-q4.safetensors in the same directory.

Quantized (matching the pinned pipenetwork/Hy3-4bit expert segment format,
affine 4-bit group-size-64: packed U32 weight + BF16 scales/biases per group):
    model.layers.80.mlp.experts.{0..191}.{gate_proj,up_proj,down_proj}.weight
        -> .weight (uint32, [out, in/8]) + .scales/.biases (bfloat16, [out, in/64])

Left unquantized (copied through in source dtype, BF16/F32):
    eh_proj, enorm/hnorm/*layernorm, self_attn.* (incl. q/k_norm),
    mlp.router.gate, mlp.expert_bias, mlp.shared_mlp.*
NOTE: the pinned artifact quantizes resident attention/shared_mlp (4-bit) and
router gate (8-bit) too; per packaging instructions those stay BF16 here and
any resident-tensor quantization is left to the integration pass.

Guarded: refuses to start while benchmark_streamed_generation holds the GPU
(override with --skip-gate).
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

EXPERT_RE = re.compile(
    r"^model\.layers\.80\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
GROUP_SIZE = 64
MODE = "affine"
COSINE_MIN = 0.99


def gpu_gate_clear() -> bool:
    r = subprocess.run(["pgrep", "-f", "benchmark_streamed_generation"], capture_output=True)
    return r.returncode != 0  # pgrep exits 1 when nothing matches


def read_metadata(path: Path) -> dict:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    return header.get("__metadata__", {})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="/Users/davidtai/.cache/huggingface/hy3-mtp-layer80",
                    help="Directory holding layer80-bf16.safetensors")
    ap.add_argument("--infile", default="layer80-bf16.safetensors")
    ap.add_argument("--bits", type=int, choices=(4, 8), default=4,
                    help="Affine bit width for the routed expert projections")
    ap.add_argument("--out", default=None,
                    help="Output filename (default: layer80-q{bits}.safetensors)")
    ap.add_argument("--skip-gate", action="store_true", help="Skip the GPU-benchmark guard")
    args = ap.parse_args()
    bits = args.bits
    if args.out is None:
        args.out = f"layer80-q{bits}.safetensors"

    if not args.skip_gate and not gpu_gate_clear():
        print("ABORT: benchmark_streamed_generation is running; not touching the GPU.",
              file=sys.stderr)
        return 2

    import mlx.core as mx  # deferred so the gate check runs first

    src = Path(args.src)
    in_path = src / args.infile
    out_path = Path(args.out) if Path(args.out).is_absolute() else src / args.out

    src_meta = read_metadata(in_path)
    tensors = mx.load(str(in_path))
    print(f"loaded {len(tensors)} tensors from {in_path}")

    out: dict[str, "mx.array"] = {}
    n_quant = 0
    for name in sorted(tensors):
        w = tensors[name]
        m = EXPERT_RE.match(name)
        if m is None:
            out[name] = w  # pass through untouched (BF16 / F32)
            continue
        wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=bits, mode=MODE)
        base = name[: -len(".weight")]
        out[base + ".weight"] = wq
        out[base + ".scales"] = scales
        out[base + ".biases"] = biases
        n_quant += 1
    mx.eval(list(out.values()))
    print(f"quantized {n_quant} expert projection tensors "
          f"(mode={MODE}, bits={bits}, group_size={GROUP_SIZE})")

    # ---- format assertions vs pinned pipenetwork/Hy3-4bit expert segments ----
    # Packed last dim scales with bits (in_features * bits / 32); the scales/
    # biases last dim depends only on group size (in_features / GROUP_SIZE).
    gu_packed = 4096 * bits // 32   # gate_proj / up_proj read 4096 inputs
    dn_packed = 1536 * bits // 32   # down_proj reads 1536 inputs
    checks = [
        ("model.layers.80.mlp.experts.0.gate_proj", (1536, gu_packed), (1536, 64)),
        ("model.layers.80.mlp.experts.0.up_proj", (1536, gu_packed), (1536, 64)),
        ("model.layers.80.mlp.experts.0.down_proj", (4096, dn_packed), (4096, 24)),
        ("model.layers.80.mlp.experts.191.down_proj", (4096, dn_packed), (4096, 24)),
    ]
    for base, wshape, gshape in checks:
        wq, sc, bi = out[base + ".weight"], out[base + ".scales"], out[base + ".biases"]
        assert wq.dtype == mx.uint32 and tuple(wq.shape) == wshape, (base, wq.dtype, wq.shape)
        assert sc.dtype == mx.bfloat16 and tuple(sc.shape) == gshape, (base, sc.dtype, sc.shape)
        assert bi.dtype == mx.bfloat16 and tuple(bi.shape) == gshape, (base, bi.dtype, bi.shape)
    print("format assertions         : OK (U32 packed + BF16 scales/biases, pinned shapes)")

    # ---- roundtrip verification ----
    worst = 1.0
    for base, _, _ in checks[:3]:
        orig = tensors[base + ".weight"].astype(mx.float32).flatten()
        deq = mx.dequantize(out[base + ".weight"], out[base + ".scales"], out[base + ".biases"],
                            group_size=GROUP_SIZE, bits=bits, mode=MODE).astype(mx.float32).flatten()
        cos = (mx.sum(orig * deq) / (mx.linalg.norm(orig) * mx.linalg.norm(deq))).item()
        print(f"roundtrip cosine {base.split('.experts.')[-1]:14s}: {cos:.6f}")
        worst = min(worst, cos)
    if worst <= COSINE_MIN:
        print(f"ERROR: roundtrip cosine {worst:.6f} <= {COSINE_MIN}", file=sys.stderr)
        return 1

    metadata = {
        **{k: str(v) for k, v in src_meta.items()},
        "quantization": json.dumps({"mode": MODE, "bits": bits, "group_size": GROUP_SIZE,
                                    "scope": "mlp.experts.*.{gate,up,down}_proj only"}),
        "producer": "scripts/quantize_mtp_layer80.py",
    }
    mx.save_safetensors(str(out_path), out, metadata=metadata)

    total = sum(t.nbytes for t in out.values())
    by_dtype: dict[str, int] = {}
    for t in out.values():
        by_dtype[str(t.dtype)] = by_dtype.get(str(t.dtype), 0) + t.nbytes
    print(f"\nwrote {out_path}")
    print(f"tensors                   : {len(out)}")
    print(f"tensor bytes              : {total:,} ({total / 2**30:.3f} GiB)")
    for dt, b in sorted(by_dtype.items()):
        print(f"  {dt}: {b:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
