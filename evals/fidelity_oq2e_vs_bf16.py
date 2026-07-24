#!/usr/bin/env python3
"""Weight-space fidelity of mlx-community Hy3-oQ2e experts (gs128, 2-bit) vs bf16.

Mirrors fidelity_vs_bf16.py (same 9-layer x 5-expert sample, same aggregation)
so the numbers are directly comparable to the shipped q2/gs64 bank's Tier-1
result (median cosine 0.9148 on this exact sample). Differences:

  - manifest/records come from ~/.cache/huggingface/hy3-oq2e-mlx (sidecar built
    2026-07-21, sha-verified reads);
  - dequantize uses group_size=128 (the oQ2e expert format);
  - runs on CPU: dequantize is exact unpacking arithmetic (int levels * bf16
    scale + bias), so no GPU window is needed and the flock law is honored.

Run:  PYTHONPATH=<worktree> .venv/bin/python evals/fidelity_oq2e_vs_bf16.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import mlx.core as mx

mx.set_default_device(mx.cpu)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import (  # noqa: E402
    BF16_DIR,
    bf16_expert_tensor,
    bytes_to_mx,
    cosine_and_relerr,
    read_bf16_tensor,
    update_results,
)

from mtplx.expert_manifest import load_expert_manifest, read_expert_record  # noqa: E402

OQ2E_DIR = Path.home() / ".cache/huggingface/hy3-oq2e-mlx"
GROUP_SIZE = 128
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# Shipped q2/gs64 Tier-1 result on this exact sample — the comparison target.
SHIPPED_Q2_MEDIAN_COSINE = 0.9148

LAYER_BUCKETS = {
    "low": [1, 6, 13],
    "mid": [27, 40, 53],
    "high": [66, 73, 79],
}
SAMPLE_EXPERTS = [0, 48, 96, 144, 191]


def self_check_dequant_roundtrip() -> dict:
    """Byte decode path == MLX reference dequantize at gs128, bitwise (CPU)."""
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((256, 512)).astype("float32")).astype(mx.bfloat16)
    w, s, b = mx.quantize(x, bits=2, group_size=GROUP_SIZE, mode="affine")
    mx.eval(w, s, b)
    reference = mx.dequantize(w, s, b, bits=2, group_size=GROUP_SIZE, mode="affine")
    mx.eval(reference)
    w_bytes = np.array(w, copy=True).astype("<u4", copy=False).tobytes()
    s_bytes = np.array(s.view(mx.uint16), copy=True).astype("<u2", copy=False).tobytes()
    b_bytes = np.array(b.view(mx.uint16), copy=True).astype("<u2", copy=False).tobytes()
    w2 = bytes_to_mx(w_bytes, "U32", w.shape)
    s2 = bytes_to_mx(s_bytes, "BF16", s.shape)
    b2 = bytes_to_mx(b_bytes, "BF16", b.shape)
    got = mx.dequantize(w2, s2, b2, bits=2, group_size=GROUP_SIZE, mode="affine")
    mx.eval(got)
    max_abs = float(mx.max(mx.abs(got - reference)).item())
    bitwise = bool(mx.all(got == reference).item())
    return {"bitwise_identical": bitwise, "max_abs_diff": max_abs, "device": "cpu"}


def split_packed_record(record, payload: bytes) -> dict[str, mx.array]:
    """Split a sidecar record payload by cumulative segment lengths.

    Unlike the q2 bank (whose manifest segments point INTO experts.bin, so
    offset deltas work), the oq2e manifest's segments keep shard-absolute
    offsets into the stock checkpoint; the sidecar packs the same segments
    contiguously in manifest order, so the payload is sliced by running
    length, not by offset arithmetic.
    """
    out: dict[str, mx.array] = {}
    pos = 0
    for seg in record.segments:
        buf = payload[pos : pos + seg.length]
        if len(buf) != seg.length:
            raise ValueError(f"short slice for {seg.component}")
        out[seg.component] = bytes_to_mx(buf, seg.dtype, seg.shape)
        pos += seg.length
    if pos != len(payload):
        raise ValueError(f"record payload has {len(payload) - pos} trailing bytes")
    return out


def dequant_projection_gs128(arrs: dict[str, mx.array], projection: str) -> mx.array:
    w = arrs[f"{projection}.weight"]
    s = arrs[f"{projection}.scales"]
    b = arrs[f"{projection}.biases"]
    d = mx.dequantize(w, s, b, bits=2, group_size=GROUP_SIZE, mode="affine")
    mx.eval(d)
    return d


def main() -> int:
    t0 = time.time()
    self_check = self_check_dequant_roundtrip()
    if not self_check["bitwise_identical"]:
        print("SELF-CHECK FAILED: decode path diverges from MLX reference", file=sys.stderr)

    manifest = load_expert_manifest(OQ2E_DIR / "expert-manifest.json")

    sample = []
    for bucket, layers in LAYER_BUCKETS.items():
        for layer in layers:
            for expert in SAMPLE_EXPERTS:
                sample.append((bucket, layer, expert))

    per_tensor: list[dict] = []
    per_expert_cos: list[float] = []
    n_records = 0
    for bucket, layer, expert in sample:
        payload = read_expert_record(
            manifest, OQ2E_DIR, layer, expert, prefer_sidecar=True, verify_hash=True
        )
        n_records += 1
        arrs = split_packed_record(manifest.record(layer, expert), payload)
        expert_cos = []
        for proj in PROJECTIONS:
            deq = dequant_projection_gs128(arrs, proj)
            ref = read_bf16_tensor(BF16_DIR, bf16_expert_tensor(layer, expert, proj))
            if tuple(deq.shape) != tuple(ref.shape):
                raise SystemExit(
                    f"shape mismatch {proj} L{layer}E{expert}: "
                    f"oq2e {deq.shape} vs bf16 {ref.shape}"
                )
            cos, relerr = cosine_and_relerr(deq, ref)
            per_tensor.append(
                {
                    "bucket": bucket,
                    "layer": layer,
                    "expert": expert,
                    "projection": proj,
                    "cosine": cos,
                    "relerr": relerr,
                }
            )
            expert_cos.append(cos)
            del deq, ref
        per_expert_cos.append(sum(expert_cos) / len(expert_cos))
        mx.clear_cache()

    cosines = [r["cosine"] for r in per_tensor]
    relerrs = [r["relerr"] for r in per_tensor]

    def agg(vals):
        return {
            "min": min(vals),
            "median": statistics.median(vals),
            "mean": statistics.fmean(vals),
            "max": max(vals),
        }

    by_proj = {
        proj: agg([r["cosine"] for r in per_tensor if r["projection"] == proj])
        for proj in PROJECTIONS
    }
    cos_agg = agg(cosines)

    result = {
        "eval": "oq2e weight-space fidelity vs bf16",
        "self_check_dequant_roundtrip": self_check,
        "sample": {
            "n_experts": n_records,
            "n_tensors": len(per_tensor),
            "layer_buckets": LAYER_BUCKETS,
            "expert_ids": SAMPLE_EXPERTS,
        },
        "per_tensor_cosine": cos_agg,
        "per_tensor_relerr": agg(relerrs),
        "per_expert_mean_cosine": agg(per_expert_cos),
        "per_projection_cosine": by_proj,
        "comparison": {
            "shipped_q2_gs64_median_cosine": SHIPPED_Q2_MEDIAN_COSINE,
            "delta_median": cos_agg["median"] - SHIPPED_Q2_MEDIAN_COSINE,
        },
        "verdict": "PASS" if self_check["bitwise_identical"] else "FAIL",
        "verdict_basis": (
            "harness validated by bitwise dequant self-check at gs128; the "
            "cosine/relerr numbers are the fidelity measurement"
        ),
        "seconds": round(time.time() - t0, 1),
    }
    update_results("eval_oq2e_fidelity_vs_bf16", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
