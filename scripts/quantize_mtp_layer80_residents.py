#!/usr/bin/env python3
"""Quantize Hy3 MTP layer-80 resident tensors to the pinned artifact format.

Input : layer80-bf16.safetensors (produced by scripts/extract_mtp_layer80.py)
Output: layer80-residents-q.safetensors in the same directory.

The pinned pipenetwork/Hy3-4bit artifact stores every resident (non-routed)
projection quantized: attention q/k/v/o and mlp.shared_mlp gate/up/down as
affine Q4 group-size-64 and mlp.router.gate as affine Q8 group-size-64, all
with BF16 scales/biases. Norm vectors stay BF16 and the router correction
bias stays F32. This script mirrors those conventions exactly for layer 80 so
the MTP head loads with the same resident precision as trunk layers 1-79:

    Q4 gs64 : self_attn.{q,k,v,o}_proj.weight,
              mlp.shared_mlp.{gate,up,down}_proj.weight
    Q8 gs64 : mlp.router.gate.weight
    BF16    : eh_proj, enorm, hnorm, input_layernorm,
              post_attention_layernorm, final_layernorm, q_norm, k_norm
    F32     : mlp.expert_bias

Routed expert projections are NOT written here; they live in
layer80-q4.safetensors (scripts/quantize_mtp_layer80.py).

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

LAYER_PREFIX = "model.layers.80."
GROUP_SIZE = 64
MODE = "affine"
Q4_COSINE_MIN = 0.99
Q8_COSINE_MIN = 0.999

EXPERT_RE = re.compile(
    r"^model\.layers\.80\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)

Q4_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.shared_mlp.gate_proj.weight",
    "mlp.shared_mlp.up_proj.weight",
    "mlp.shared_mlp.down_proj.weight",
)
Q8_SUFFIXES = ("mlp.router.gate.weight",)
BF16_PASS_SUFFIXES = (
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "final_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)
F32_PASS_SUFFIXES = ("mlp.expert_bias",)

RESIDENT_SUFFIXES = (
    Q4_SUFFIXES + Q8_SUFFIXES + BF16_PASS_SUFFIXES + F32_PASS_SUFFIXES
)

# Expected quantized shapes for the real tencent/Hy3 layer-80 geometry,
# mirroring the pinned artifact's layer 1-79 resident segments exactly.
PINNED_RESIDENT_Q_SHAPES = {
    "self_attn.q_proj": ((8192, 512), (8192, 64)),
    "self_attn.k_proj": ((1024, 512), (1024, 64)),
    "self_attn.v_proj": ((1024, 512), (1024, 64)),
    "self_attn.o_proj": ((4096, 1024), (4096, 128)),
    "mlp.shared_mlp.gate_proj": ((1536, 512), (1536, 64)),
    "mlp.shared_mlp.up_proj": ((1536, 512), (1536, 64)),
    "mlp.shared_mlp.down_proj": ((4096, 192), (4096, 24)),
    "mlp.router.gate": ((192, 1024), (192, 64)),
}


def gpu_gate_clear() -> bool:
    r = subprocess.run(["pgrep", "-f", "benchmark_streamed_generation"], capture_output=True)
    return r.returncode != 0  # pgrep exits 1 when nothing matches


def read_metadata(path: Path) -> dict:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    return header.get("__metadata__", {})


def classify_resident(name: str, *, layer_prefix: str = LAYER_PREFIX) -> str:
    """Return the pinned-format treatment for one layer-80 resident tensor."""

    if EXPERT_RE.match(name):
        raise ValueError(f"{name} is a routed expert tensor, not a resident")
    if not name.startswith(layer_prefix):
        raise ValueError(f"{name} is outside {layer_prefix}")
    suffix = name[len(layer_prefix):]
    if suffix in Q4_SUFFIXES:
        return "q4"
    if suffix in Q8_SUFFIXES:
        return "q8"
    if suffix in BF16_PASS_SUFFIXES:
        return "bf16"
    if suffix in F32_PASS_SUFFIXES:
        return "f32"
    raise ValueError(f"unexpected layer-80 resident tensor {name}")


def quantize_resident_tensors(
    tensors: dict,
    *,
    layer_prefix: str = LAYER_PREFIX,
    group_size: int = GROUP_SIZE,
) -> dict:
    """Quantize a complete layer-80 resident set to pinned conventions.

    ``tensors`` must contain exactly the resident tensors (no routed
    experts); anything unexpected fails closed.
    """

    import mlx.core as mx

    expected = {layer_prefix + suffix for suffix in RESIDENT_SUFFIXES}
    missing = expected - set(tensors)
    extra = set(tensors) - expected
    if missing:
        raise ValueError(f"missing resident tensors: {sorted(missing)[:4]}")
    if extra:
        raise ValueError(f"unexpected resident tensors: {sorted(extra)[:4]}")

    out: dict = {}
    for name in sorted(tensors):
        value = tensors[name]
        treatment = classify_resident(name, layer_prefix=layer_prefix)
        if treatment in {"bf16", "f32"}:
            wanted = mx.bfloat16 if treatment == "bf16" else mx.float32
            if value.dtype != wanted:
                raise ValueError(
                    f"{name} must stay {wanted}, found {value.dtype}"
                )
            out[name] = value
            continue
        bits = 4 if treatment == "q4" else 8
        if value.dtype != mx.bfloat16:
            raise ValueError(f"{name} must be bfloat16 source, found {value.dtype}")
        weight, scales, biases = mx.quantize(
            value, group_size=group_size, bits=bits, mode=MODE
        )
        base = name[: -len(".weight")]
        out[base + ".weight"] = weight
        out[base + ".scales"] = scales
        out[base + ".biases"] = biases
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="/Users/davidtai/.cache/huggingface/hy3-mtp-layer80",
                    help="Directory holding layer80-bf16.safetensors")
    ap.add_argument("--infile", default="layer80-bf16.safetensors")
    ap.add_argument("--out", default="layer80-residents-q.safetensors")
    ap.add_argument("--skip-gate", action="store_true", help="Skip the GPU-benchmark guard")
    args = ap.parse_args()

    if not args.skip_gate and not gpu_gate_clear():
        print("ABORT: benchmark_streamed_generation is running; not touching the GPU.",
              file=sys.stderr)
        return 2

    import mlx.core as mx  # deferred so the gate check runs first

    src = Path(args.src)
    in_path = src / args.infile
    out_path = Path(args.out) if Path(args.out).is_absolute() else src / args.out

    src_meta = read_metadata(in_path)
    all_tensors = mx.load(str(in_path))
    residents = {
        name: value
        for name, value in all_tensors.items()
        if EXPERT_RE.match(name) is None
    }
    print(f"loaded {len(all_tensors)} tensors from {in_path}; "
          f"{len(residents)} residents selected")

    out = quantize_resident_tensors(residents)
    mx.eval(list(out.values()))
    n_quant = sum(1 for name in out if name.endswith(".scales"))
    print(f"quantized {n_quant} resident projections "
          f"(mode={MODE}, group_size={GROUP_SIZE}; router gate 8-bit, rest 4-bit)")

    # ---- format assertions vs pinned pipenetwork/Hy3-4bit resident segments ----
    for suffix, (wshape, gshape) in PINNED_RESIDENT_Q_SHAPES.items():
        base = LAYER_PREFIX + suffix
        wq, sc, bi = out[base + ".weight"], out[base + ".scales"], out[base + ".biases"]
        assert wq.dtype == mx.uint32 and tuple(wq.shape) == wshape, (base, wq.dtype, wq.shape)
        assert sc.dtype == mx.bfloat16 and tuple(sc.shape) == gshape, (base, sc.dtype, sc.shape)
        assert bi.dtype == mx.bfloat16 and tuple(bi.shape) == gshape, (base, bi.dtype, bi.shape)
    assert out[LAYER_PREFIX + "eh_proj.weight"].dtype == mx.bfloat16
    assert tuple(out[LAYER_PREFIX + "eh_proj.weight"].shape) == (4096, 8192)
    assert out[LAYER_PREFIX + "mlp.expert_bias"].dtype == mx.float32
    print("format assertions         : OK (U32 packed + BF16 scales/biases, pinned shapes)")

    # ---- roundtrip verification ----
    worst_by_bits = {4: 1.0, 8: 1.0}
    for suffix in Q4_SUFFIXES + Q8_SUFFIXES:
        base = LAYER_PREFIX + suffix[: -len(".weight")]
        bits = 8 if suffix in Q8_SUFFIXES else 4
        orig = residents[base + ".weight"].astype(mx.float32).flatten()
        deq = mx.dequantize(out[base + ".weight"], out[base + ".scales"], out[base + ".biases"],
                            group_size=GROUP_SIZE, bits=bits, mode=MODE).astype(mx.float32).flatten()
        cos = (mx.sum(orig * deq) / (mx.linalg.norm(orig) * mx.linalg.norm(deq))).item()
        print(f"roundtrip cosine q{bits} {suffix:32s}: {cos:.6f}")
        worst_by_bits[bits] = min(worst_by_bits[bits], cos)
    if worst_by_bits[4] <= Q4_COSINE_MIN or worst_by_bits[8] <= Q8_COSINE_MIN:
        print(f"ERROR: roundtrip cosine too low (q4={worst_by_bits[4]:.6f}, "
              f"q8={worst_by_bits[8]:.6f})", file=sys.stderr)
        return 1

    metadata = {
        **{k: str(v) for k, v in src_meta.items()},
        "quantization": json.dumps({
            "mode": MODE,
            "group_size": GROUP_SIZE,
            "bits": 4,
            "model.layers.80.mlp.router.gate": {"bits": 8, "group_size": GROUP_SIZE},
            "scope": "layer-80 residents only (attention, router gate, shared_mlp)",
        }),
        "producer": "scripts/quantize_mtp_layer80_residents.py",
    }
    mx.save_safetensors(str(out_path), out, metadata=metadata)

    total = sum(t.nbytes for t in out.values())
    by_dtype: dict[str, int] = {}
    for t in out.values():
        by_dtype[str(t.dtype)] = by_dtype.get(str(t.dtype), 0) + t.nbytes
    print(f"\nwrote {out_path}")
    print(f"tensors                   : {len(out)}")
    print(f"tensor bytes              : {total:,} ({total / 2**20:.1f} MiB)")
    for dt, b in sorted(by_dtype.items()):
        print(f"  {dt}: {b:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
