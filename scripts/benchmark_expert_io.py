#!/usr/bin/env python3
"""Benchmark checked expert-record reads into fixed MLX/Metal slot buffers."""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_io import PositionalExpertReader  # noqa: E402
from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_runtime import mlx_memory_telemetry  # noqa: E402
from mtplx.expert_streaming_models import get_model_spec  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", name], text=True, timeout=2
        ).strip()
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--operations", type=_positive_int, default=64)
    parser.add_argument("--warmup-operations", type=int, default=8)
    parser.add_argument("--queue-depth", type=_positive_int, default=4)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--no-verify-record-hashes", action="store_true")
    parser.add_argument(
        "--cache-state",
        choices=["unknown", "cold", "warm", "steady"],
        default="unknown",
        help="Provenance label only; this script never claims to purge macOS caches.",
    )
    parser.add_argument("--ssd-label", default="unspecified")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup_operations < 0:
        raise SystemExit("--warmup-operations must be non-negative")
    root = args.model_root.expanduser().resolve()
    manifest = load_expert_manifest(args.manifest)
    spec = get_model_spec(manifest.model_key)
    records = [
        record
        for record in manifest.records
        if args.layer is None or record.layer == args.layer
    ]
    if not records:
        raise SystemExit("no manifest records match the requested layer")
    if args.queue_depth > len(records):
        raise SystemExit("--queue-depth cannot exceed the selected record count")

    import mlx.core as mx

    slot_bank = mx.zeros(
        (args.queue_depth, spec.expert_record_bytes), dtype=mx.uint8
    )
    mx.eval(slot_bank)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    sequence = [
        records[index % len(records)]
        for index in range(args.warmup_operations + args.operations)
    ]
    verify_hash = not args.no_verify_record_hashes
    reader = PositionalExpertReader(
        root,
        max_open_files=max(2, args.queue_depth * 2),
    )

    def read_one(index: int, record) -> float:
        slot = slot_bank[index % args.queue_depth]
        started = time.perf_counter_ns()
        reader.read_record_into(
            manifest,
            record,
            slot,
            prefer_sidecar=not args.source_only,
            verify_hash=verify_hash,
        )
        return (time.perf_counter_ns() - started) / 1e6

    try:
        for start in range(0, args.warmup_operations, args.queue_depth):
            batch = sequence[start : min(args.warmup_operations, start + args.queue_depth)]
            with ThreadPoolExecutor(max_workers=args.queue_depth) as pool:
                list(pool.map(lambda pair: read_one(*pair), enumerate(batch)))

        measured = sequence[args.warmup_operations :]
        latencies_ms: list[float] = []
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.queue_depth) as pool:
            for batch_start in range(0, len(measured), args.queue_depth):
                batch = measured[batch_start : batch_start + args.queue_depth]
                indexed = [
                    (index, record) for index, record in enumerate(batch)
                ]
                latencies_ms.extend(pool.map(lambda pair: read_one(*pair), indexed))
        elapsed = time.perf_counter() - started
        metrics = reader.metrics.as_dict()
    finally:
        reader.close()

    total_bytes = args.operations * spec.expert_record_bytes
    payload = {
        "schema": "mtplx-expert-io-benchmark-v1",
        "model_key": spec.key,
        "source_repo": manifest.source_repo,
        "source_revision": manifest.source_revision,
        "manifest_sha256": manifest.manifest_sha256,
        "sidecar_sha256": manifest.sidecar.sha256 if manifest.sidecar else None,
        "sidecar_used": bool(manifest.sidecar and not args.source_only),
        "record_bytes": spec.expert_record_bytes,
        "operations": args.operations,
        "queue_depth": args.queue_depth,
        "hash_verification": verify_hash,
        "cache_state": args.cache_state,
        "ssd_label": args.ssd_label,
        "elapsed_seconds": elapsed,
        "bytes": total_bytes,
        "gib_per_second": total_bytes / 1024**3 / elapsed,
        "records_per_second": args.operations / elapsed,
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms),
            "p50": _percentile(latencies_ms, 0.50),
            "p95": _percentile(latencies_ms, 0.95),
            "max": max(latencies_ms),
        },
        "reader": {**metrics, "backend": reader.backend},
        "mlx_memory": mlx_memory_telemetry(mx),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "chip": _sysctl("machdep.cpu.brand_string"),
            "memory_bytes": _sysctl("hw.memsize"),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
