#!/usr/bin/env python3
"""Replay routed-expert JSONL through MTPLX's proposed SSD slot-bank policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mtplx.expert_streaming import ExpertCacheSimulation, RoutingPhase


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay JSONL routes shaped as "
            "{phase, layer, experts} and estimate expert-cache I/O."
        )
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expert-count", type=_positive_int, default=192)
    parser.add_argument("--persistent-slots-per-layer", type=int, required=True)
    parser.add_argument("--transient-slots", type=_positive_int, default=8)
    parser.add_argument("--expert-record-bytes", type=_positive_int, required=True)
    parser.add_argument(
        "--ssd-gib-per-second",
        type=float,
        default=5.5,
        help="Effective random-read throughput, not headline sequential speed.",
    )
    parser.add_argument("--frequency-decay", type=float, default=0.995)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.persistent_slots_per_layer < 0:
        raise SystemExit("--persistent-slots-per-layer must be non-negative")
    simulation = ExpertCacheSimulation(
        expert_count=args.expert_count,
        persistent_slots=args.persistent_slots_per_layer,
        transient_slots=args.transient_slots,
        expert_record_bytes=args.expert_record_bytes,
        frequency_decay=args.frequency_decay,
    )

    with args.trace.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                phase = RoutingPhase(event["phase"])
                layer = int(event["layer"])
                experts = event["experts"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SystemExit(f"{args.trace}:{line_number}: invalid route: {exc}")
            simulation.observe(
                layer_index=layer,
                expert_ids=experts,
                phase=phase,
            )

    gib = 1024**3
    summary = simulation.summary(
        effective_ssd_bytes_per_second=args.ssd_gib_per_second * gib
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
