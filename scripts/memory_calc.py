"""Memory calculator for expert-streaming configurations (issue #46/#51).

Prints every memory piece individually — resident model, MTP/router
additions, wired islands, compressed band (actual manifest bytes),
streamed slot cache, KV, transient — against the machine's GPU wired
limit, so the user composes explicit flags with full visibility instead
of trusting derived envelope math.

    python scripts/memory_calc.py \
        --model-key hy3-expert-q2 \
        --island-layers 1-20 \
        --mmap-island-layers 21-79 \
        --banked-manifest ~/.cache/.../experts-huff-all-manifest.json \
        --cache-bytes 20GiB --kv-tokens 2560 --transient-slots 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtplx.expert_streaming_models import get_model_spec  # noqa: E402


def parse_layers(text: str) -> tuple[int, ...]:
    layers: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = (int(v) for v in part.split("-", 1))
            if high < low:
                raise ValueError(f"layer range {part!r} is inverted")
            layers.update(range(low, high + 1))
        else:
            layers.add(int(part))
    return tuple(sorted(layers))


def parse_bytes(text: str) -> int:
    text = text.strip()
    units = {"GiB": 1 << 30, "MiB": 1 << 20, "KiB": 1 << 10, "B": 1}
    for unit, scale in units.items():
        if text.endswith(unit):
            return int(float(text[: -len(unit)]) * scale)
    return int(text)


def gib(value: float) -> str:
    return f"{value / (1 << 30):7.2f} GiB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default="hy3-expert-q2")
    parser.add_argument("--island-layers", default="")
    parser.add_argument("--mmap-island-layers", default="")
    parser.add_argument("--banked-manifest", type=Path)
    parser.add_argument("--cache-bytes", default="0B",
                        help="Streamed slot-cache budget, e.g. '20GiB'")
    parser.add_argument("--kv-tokens", type=int, default=2560)
    parser.add_argument(
        "--kv-quant",
        default="bf16",
        choices=["bf16", "q8", "q4"],
        help=(
            "KV cache storage pricing. Quantized KV is NOT wired into the "
            "Hy3 runtime yet (campaign item; QuantizedKVCache exists in "
            "mlx_lm); this prices the target shape."
        ),
    )
    parser.add_argument("--transient-slots", type=int, default=8)
    parser.add_argument("--additional-resident", default="7.1GiB",
                        help="MTP head + router kernels (measured 7.1GiB for hy3-q2 K3)")
    parser.add_argument("--headroom", default="4GiB",
                        help="Slack to leave under the GPU wired limit")
    args = parser.parse_args()

    spec = get_model_spec(args.model_key)
    islands = parse_layers(args.island_layers)
    band = parse_layers(args.mmap_island_layers)
    overlap = set(islands) & set(band)
    if overlap:
        raise SystemExit(f"island/band overlap: {sorted(overlap)}")
    routed = set(spec.routed_layer_indices)
    streamed = sorted(routed - set(islands) - set(band))

    record = spec.expert_record_bytes
    resident = spec.resident_bytes
    additional = parse_bytes(args.additional_resident)
    island_bytes = len(islands) * spec.expert_count * record
    cache_bytes = parse_bytes(args.cache_bytes)
    kv_scale = {
        "bf16": 1.0,
        # bits/16 plus one bf16 scale+bias pair per 64-value group.
        "q8": (8 + 64 / 64) / 16,
        "q4": (4 + 64 / 64) / 16,
    }[args.kv_quant]
    kv_bytes = int(args.kv_tokens * spec.kv_bytes_per_token * kv_scale)
    transient_bytes = max(args.transient_slots, spec.top_k) * record

    band_bytes = 0
    band_source = "none"
    if band:
        if not args.banked_manifest:
            raise SystemExit("--mmap-island-layers requires --banked-manifest")
        from mtplx.expert_banked import load_banked_manifest

        banked = load_banked_manifest(args.banked_manifest)
        missing = [layer for layer in band if layer not in banked.layer_set]
        if missing:
            raise SystemExit(f"banked manifest missing layers {missing}")
        band_bytes = sum(banked.layer_entry(layer).length for layer in band)
        band_source = f"{banked.codec} manifest (actual bytes)"

    slots_per_layer = (
        min(spec.expert_count, cache_bytes // (len(streamed) * record))
        if streamed
        else 0
    )
    wired_total = (
        resident + additional + island_bytes + band_bytes
        + cache_bytes + kv_bytes + transient_bytes
    )

    try:
        limit_mb = int(
            subprocess.run(
                ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"],
                capture_output=True, text=True,
            ).stdout.strip()
        )
        gpu_limit = limit_mb << 20
    except Exception:
        gpu_limit = None
    headroom = parse_bytes(args.headroom)

    print(f"model {spec.key}: {spec.routed_layer_count} routed layers x "
          f"{spec.expert_count} experts, record {record / (1 << 20):.3f} MiB")
    print()
    print(f"  resident model (non-expert)      {gib(resident)}")
    print(f"  + MTP head / router kernels      {gib(additional)}")
    print(f"  + wired islands       {len(islands):3d} layers {gib(island_bytes)}")
    print(f"  + compressed band     {len(band):3d} layers {gib(band_bytes)}   [{band_source}]")
    print(f"  + streamed slot cache {len(streamed):3d} layers {gib(cache_bytes)}"
          f"   ({slots_per_layer}/{spec.expert_count} slots/layer)")
    print(f"  + KV cache ({args.kv_tokens} tokens, {args.kv_quant})"
          f"  {gib(kv_bytes)}")
    print(f"  + transient slots                {gib(transient_bytes)}")
    print(f"  = WIRED TOTAL                    {gib(wired_total)}")
    print()
    if gpu_limit:
        budget = gpu_limit - headroom
        verdict = "FITS" if wired_total <= budget else "OVER"
        print(f"  GPU wired limit (iogpu)          {gib(gpu_limit)}")
        print(f"  - requested headroom             {gib(headroom)}")
        print(f"  = usable budget                  {gib(budget)}   -> {verdict}"
              + ("" if wired_total <= budget
                 else f" by {gib(wired_total - budget)}"))
    print()
    reserve = 10 << 30
    suggested_limit = wired_total + reserve
    print("suggested explicit flags:")
    print(f"  --memory-limit {suggested_limit / (1 << 30):.0f}GiB "
          f"--runtime-reserve 10GiB")
    print(f"  --expert-cache-limit {max(cache_bytes, 1 << 20) / (1 << 30):.0f}GiB "
          f"--max-live-kv-tokens {args.kv_tokens} "
          f"--transient-slots {args.transient_slots}")
    if islands:
        print(f"  --island-layers {args.island_layers}")
    if band:
        print(f"  --mmap-island-layers {args.mmap_island_layers}")
        print(f"  --banked-manifest {args.banked_manifest}")
        print(f"  --banked-codec huffman-l12-v1 "
              f"--banked-band-bytes {band_bytes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
