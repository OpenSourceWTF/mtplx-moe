"""Decode-rate microbench for the static byte-rANS expert lane (issue #51, C7).

THE go/no-go number: decoded GiB/s on the QUEUED Metal lane over a batch of
real Hy3-Q2 expert records. The bar is the SSD's raw-equivalent delivery
rate: the machine SSD reads ~12.5 GiB/s of *compressed* bytes, which at the
~1.29x order-0 ratio expands to ~16 GiB/s of *raw* bytes. The decoder must
produce raw bytes at least that fast or it becomes the bottleneck.

Tiny by construction: encodes a handful of real experts once, decodes many
routed assignments (experts repeat), a few seconds of GPU. No model load, no
locks, no exclusive device.

    python scripts/benchmark_rans_decode.py --experts 32 --assignments 128 --iters 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mlx.core as mx

import mtplx.expert_rans as R
import mtplx.expert_rans_metal as RM

DEFAULT_ROOT = Path("~/.cache/huggingface/hy3-expert-only-mlx-q2").expanduser()


def load_component_banks(root: Path, layer: int, experts: int):
    """Read ``experts`` experts of ``layer`` and group bytes per component.

    Returns an ordered dict ``{component: (segments[E, seg_len], dtype)}``
    using the record ``sidecar_offset``/``sidecar_length`` and the
    shard-absolute segment offsets.
    """

    manifest = json.loads((root / "expert-manifest.json").read_text())
    records = {(r["layer"], r["expert"]): r for r in manifest["records"]}
    sidecar = root / manifest["sidecar"]["file"]
    fd = os.open(sidecar, os.O_RDONLY)
    banks: dict[str, list[np.ndarray]] = {}
    dtypes: dict[str, str] = {}
    order: list[str] = []
    try:
        for expert in range(experts):
            record = records[(layer, expert)]
            blob = os.pread(fd, record["sidecar_length"], record["sidecar_offset"])
            for seg in record["segments"]:
                comp = seg["component"]
                rel = seg["offset"] - record["sidecar_offset"]
                chunk = np.frombuffer(blob[rel : rel + seg["length"]], dtype=np.uint8)
                banks.setdefault(comp, []).append(chunk)
                if comp not in dtypes:
                    dtypes[comp] = seg["dtype"]
                    order.append(comp)
    finally:
        os.close(fd)
    return {comp: (np.stack(banks[comp]), dtypes[comp]) for comp in order}


def build_encoded(banks):
    """Encode every component bank; return per-component decode state."""

    encoded = {}
    total_raw = 0
    total_comp = 0
    for comp, (segments, dtype) in banks.items():
        table = R.build_table(R.histogram(segments.reshape(-1)))
        streams = R.encode_bank(segments, table)
        cum2sym, freq, cum = RM.table_device_arrays(table)
        encoded[comp] = {
            "segments": segments,
            "streams": streams,
            "table": table,
            "payload": mx.array(streams.payload),
            "directory": mx.array(streams.directory.reshape(-1)),
            "cum2sym": cum2sym,
            "freq": freq,
            "cum": cum,
        }
        total_raw += int(segments.nbytes)
        total_comp += int(streams.payload.size)
        mx.eval(encoded[comp]["payload"], encoded[comp]["directory"])
    return encoded, total_raw, total_comp


def decode_all(encoded, indices, assignments):
    outs = []
    for comp, e in encoded.items():
        s = e["streams"]
        outs.append(
            RM.decode_component(
                e["payload"], e["directory"], indices,
                e["cum2sym"], e["freq"], e["cum"],
                lanes=s.lanes, per_lane=s.per_lane, seg_len=s.seg_len,
                assignments=assignments,
            )
        )
    return outs


def verify_roundtrip(encoded, experts):
    """Bitwise: kernel decode of every real expert == its raw bytes."""

    indices = mx.array(np.arange(experts, dtype=np.int32))
    ok = True
    for comp, e in encoded.items():
        s = e["streams"]
        out = RM.decode_component(
            e["payload"], e["directory"], indices,
            e["cum2sym"], e["freq"], e["cum"],
            lanes=s.lanes, per_lane=s.per_lane, seg_len=s.seg_len,
            assignments=experts,
        )
        mx.eval(out)
        got = np.array(out).reshape(experts, s.seg_len)
        if not np.array_equal(got, e["segments"]):
            ok = False
            print(f"  ROUND-TRIP MISMATCH in component {comp}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--assignments", type=int, default=128)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--ssd-gibs", type=float, default=12.5)
    args = ap.parse_args()

    print(f"loading {args.experts} experts of layer {args.layer} from {args.root}")
    banks = load_component_banks(args.root, args.layer, args.experts)
    encoded, total_raw, total_comp = build_encoded(banks)
    ratio = total_raw / total_comp
    print(
        f"encoded {len(encoded)} components: raw {total_raw/2**20:.1f} MiB -> "
        f"compressed {total_comp/2**20:.1f} MiB (ratio {ratio:.4f})"
    )

    print("verifying bitwise round-trip on every real record ...")
    ok = verify_roundtrip(encoded, args.experts)
    print(f"  round-trip bitwise-exact: {ok}")
    if not ok:
        return 1

    rng = np.random.default_rng(0)
    idx = rng.integers(0, args.experts, size=args.assignments).astype(np.int32)
    indices = mx.array(idx)
    record_bytes = sum(e["streams"].seg_len for e in encoded.values())
    decoded_per_iter = args.assignments * record_bytes

    # Warmup (compiles kernels, warms caches) — evaluated, discarded.
    mx.eval(decode_all(encoded, indices, args.assignments))
    mx.synchronize()

    # QUEUED lane: submit every iteration's dispatches, hold refs, sync once.
    t0 = time.perf_counter()
    held = []
    for _ in range(args.iters):
        held.extend(decode_all(encoded, indices, args.assignments))
    mx.eval(held)
    mx.synchronize()
    elapsed = time.perf_counter() - t0
    total_decoded = decoded_per_iter * args.iters
    gibs = total_decoded / 2**30 / elapsed

    ssd_raw_equiv = args.ssd_gibs * ratio
    print()
    print("=" * 60)
    print(f"DECODE RATE (queued): {gibs:.2f} GiB/s decoded raw bytes")
    print(f"  {args.assignments} assignments x {args.iters} iters, "
          f"{total_decoded/2**30:.2f} GiB in {elapsed*1e3:.1f} ms")
    print(f"  SSD delivers {args.ssd_gibs:.1f} GiB/s compressed "
          f"= {ssd_raw_equiv:.1f} GiB/s raw-equivalent (bar)")
    verdict = "CLEARS" if gibs >= ssd_raw_equiv else "MISSES"
    print(f"  VERDICT: decode {verdict} the {ssd_raw_equiv:.1f} GiB/s bar")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
