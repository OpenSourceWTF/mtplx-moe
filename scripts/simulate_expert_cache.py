#!/usr/bin/env python3
"""Replay routed-expert JSONL through MTPLX's proposed SSD slot-bank policy."""

from __future__ import annotations

import argparse
import json
import sys
from math import isfinite
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_streaming import ExpertCacheSimulation, RoutingPhase  # noqa: E402
from mtplx.expert_streaming_models import (  # noqa: E402
    MODEL_SPECS,
    get_model_spec,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _frequency_decay(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be finite and in (0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay JSONL routes shaped as "
            "{phase, layer, experts} and estimate expert-cache I/O."
        )
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS),
        help="Derive expert geometry and validate sparse-layer indices.",
    )
    parser.add_argument("--expert-count", type=_positive_int)
    parser.add_argument(
        "--persistent-slots-per-layer", type=_nonnegative_int, required=True
    )
    parser.add_argument("--transient-slots", type=_positive_int)
    parser.add_argument("--expert-record-bytes", type=_positive_int)
    parser.add_argument(
        "--allocated-layer-count",
        type=_positive_int,
        help="Required for full-bank sizing in generic mode; derived with --model.",
    )
    parser.add_argument(
        "--ssd-gib-per-second",
        type=_positive_float,
        default=5.5,
        help="Effective random-read throughput, not headline sequential speed.",
    )
    parser.add_argument("--frequency-decay", type=_frequency_decay, default=0.995)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec = get_model_spec(args.model) if args.model else None
    if spec is not None and (
        args.expert_count is not None
        or args.expert_record_bytes is not None
        or args.allocated_layer_count is not None
    ):
        raise SystemExit(
            "--model derives expert count, record bytes, and allocated layer count; "
            "do not override them"
        )
    expert_count = args.expert_count or (spec.expert_count if spec else 192)
    transient_slots = args.transient_slots or (spec.top_k if spec else 8)
    expert_record_bytes = args.expert_record_bytes or (
        spec.expert_record_bytes if spec else None
    )
    if expert_record_bytes is None:
        raise SystemExit("--expert-record-bytes is required when --model is omitted")
    if spec is None and args.allocated_layer_count is None:
        raise SystemExit("--allocated-layer-count is required when --model is omitted")
    if spec is not None and transient_slots < spec.top_k:
        raise SystemExit(f"--transient-slots must be at least {spec.top_k}")
    if args.persistent_slots_per_layer > expert_count:
        raise SystemExit("--persistent-slots-per-layer cannot exceed the expert count")
    simulation = ExpertCacheSimulation(
        expert_count=expert_count,
        persistent_slots=args.persistent_slots_per_layer,
        transient_slots=transient_slots,
        expert_record_bytes=expert_record_bytes,
        allocated_layer_count=(
            spec.routed_layer_count if spec else args.allocated_layer_count
        ),
        frequency_decay=args.frequency_decay,
    )

    with args.trace.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                phase = RoutingPhase(event["phase"])
                layer = event["layer"]
                if isinstance(layer, bool) or not isinstance(layer, int):
                    raise TypeError("layer must be an exact integer")
                experts = event["experts"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise SystemExit(f"{args.trace}:{line_number}: invalid route: {exc}")
            if spec is not None and layer not in spec.routed_layer_indices:
                raise SystemExit(
                    f"{args.trace}:{line_number}: layer {layer} is not a routed "
                    f"{spec.key} layer"
                )
            try:
                simulation.observe(
                    layer_index=layer,
                    expert_ids=experts,
                    phase=phase,
                )
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"{args.trace}:{line_number}: invalid route: {exc}")

    gib = 1024**3
    summary = simulation.summary(
        effective_ssd_bytes_per_second=args.ssd_gib_per_second * gib
    )
    if spec is not None:
        summary["model_key"] = spec.key
    summary["effective_expert_count"] = expert_count
    summary["effective_expert_record_bytes"] = expert_record_bytes
    summary["effective_transient_slots"] = transient_slots
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
