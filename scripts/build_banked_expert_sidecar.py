"""Repack raw expert sidecar layers into component-major banked regions (C6).

The banked sidecar serves mmap island layers: one page-aligned region per
layer whose per-component banks have bank row == expert id, mappable into
Metal without copies. Example:

    python scripts/build_banked_expert_sidecar.py \
        --model-root ~/.cache/huggingface/hy3-expert-only-mlx-q2 \
        --layers 39-59
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtplx.expert_banked import write_banked_expert_banks
from mtplx.expert_manifest import load_expert_manifest


def parse_layers(text: str) -> tuple[int, ...]:
    layers: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low_text, high_text = part.split("-", 1)
            low, high = int(low_text), int(high_text)
            if high < low:
                raise ValueError(f"layer range {part!r} is inverted")
            layers.update(range(low, high + 1))
        else:
            layers.add(int(part))
    if not layers:
        raise ValueError("at least one layer is required")
    return tuple(sorted(layers))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Expert manifest path (default: <model-root>/expert-manifest.json)",
    )
    parser.add_argument("--layers", required=True, help="e.g. '39-59' or '1,4,7'")
    parser.add_argument(
        "--output-bin",
        type=Path,
        help="Banked bin path (default: <model-root>/experts-banked.bin)",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        help=(
            "Banked manifest path "
            "(default: <model-root>/experts-banked-manifest.json)"
        ),
    )
    parser.add_argument(
        "--codec",
        default="none",
        choices=["none", "huffman-l12-v1", "rans32x-v1"],
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--no-verify-record-hashes",
        action="store_true",
        help="Skip per-record hash checks while reading (not recommended).",
    )
    args = parser.parse_args()

    root = args.model_root.expanduser().resolve()
    manifest_path = args.manifest or root / "expert-manifest.json"
    output_bin = args.output_bin or root / "experts-banked.bin"
    output_manifest = args.output_manifest or root / "experts-banked-manifest.json"
    layers = parse_layers(args.layers)

    print(f"loading manifest {manifest_path} ...", flush=True)
    manifest = load_expert_manifest(manifest_path)
    print(
        f"repacking {len(layers)} layers ({layers[0]}..{layers[-1]}) "
        f"codec={args.codec} -> {output_bin}",
        flush=True,
    )
    started = time.perf_counter()
    banked = write_banked_expert_banks(
        manifest,
        root,
        layers,
        output_bin=output_bin,
        output_manifest=output_manifest,
        codec=args.codec,
        verify_record_hashes=not args.no_verify_record_hashes,
        workers=args.workers,
    )
    elapsed = time.perf_counter() - started
    total = sum(entry.length for entry in banked.layers)
    print(
        f"done: {len(banked.layers)} layers, {total / 2**30:.2f} GiB banked "
        f"in {elapsed:.1f}s ({total / 2**30 / max(elapsed, 1e-9):.2f} GiB/s); "
        f"manifest -> {output_manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
