#!/usr/bin/env python3
"""Size an SSD-backed MoE expert bank under a total memory threshold."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_streaming_models import (  # noqa: E402
    MODEL_SPECS,
    get_model_spec,
    plan_expert_memory,
)


GIB = 1024**3


def _decimal(value: str, *, allow_zero: bool) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("value must be a decimal number") from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("value must be finite")
    if parsed < 0 or (not allow_zero and parsed == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise argparse.ArgumentTypeError(f"value must be {qualifier}")
    return parsed


def _positive_decimal(value: str) -> Decimal:
    return _decimal(value, allow_zero=False)


def _nonnegative_decimal(value: str) -> Decimal:
    return _decimal(value, allow_zero=True)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _as_gib(value: int) -> float:
    return value / GIB


def _gib_to_bytes(value: Decimal) -> int:
    """Convert exactly parsed GiB to a conservative whole-byte floor."""

    return int(value * GIB)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reserve resident weights, KV cache, runtime headroom, and top-k "
            "scratch before assigning persistent routed-expert slots."
        )
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument(
        "--memory-limit-gib",
        type=_positive_decimal,
        required=True,
        help="Total memory ceiling for resident state, KV, scratch, and hot experts.",
    )
    parser.add_argument(
        "--context-tokens",
        type=_nonnegative_int,
        required=True,
        help="Required maximum live KV tokens across admitted sequences; use 0 only for load-only planning.",
    )
    parser.add_argument(
        "--runtime-reserve-gib",
        type=_nonnegative_decimal,
        default=Decimal("16"),
        help="Headroom for activations, Metal, MTPLX, and macOS (default: 16).",
    )
    parser.add_argument(
        "--expert-cache-limit-gib",
        type=_nonnegative_decimal,
        help="Optional stricter cap for persistent experts alone.",
    )
    parser.add_argument(
        "--transient-slots",
        type=_nonnegative_int,
        help="Global miss-service slots; defaults to the model's top-k.",
    )
    parser.add_argument(
        "--io-staging-gib",
        type=_nonnegative_decimal,
        default=Decimal("0"),
        help="Known staging/in-flight I/O buffers not aliased to expert slots.",
    )
    parser.add_argument(
        "--execution-workspace-gib",
        type=_nonnegative_decimal,
        default=Decimal("0"),
        help="Known router/kernel workspace outside resident and slot buffers.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec = get_model_spec(args.model)
    cache_cap = (
        None
        if args.expert_cache_limit_gib is None
        else _gib_to_bytes(args.expert_cache_limit_gib)
    )
    try:
        plan = plan_expert_memory(
            spec,
            total_limit_bytes=_gib_to_bytes(args.memory_limit_gib),
            context_tokens=args.context_tokens,
            runtime_reserve_bytes=_gib_to_bytes(args.runtime_reserve_gib),
            expert_cache_limit_bytes=cache_cap,
            transient_slots=args.transient_slots,
            io_staging_bytes=_gib_to_bytes(args.io_staging_gib),
            execution_workspace_bytes=_gib_to_bytes(args.execution_workspace_gib),
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    payload = {
        "model": {
            "key": spec.key,
            "display_name": spec.display_name,
            "source_model": spec.source_model,
            "source_revision": spec.source_revision,
            "quant_model": spec.quant_model,
            "quant_revision": spec.quant_revision,
            "routed_layers": spec.routed_layer_count,
            "experts_per_layer": spec.expert_count,
            "top_k": spec.top_k,
            "expert_record_bytes": spec.expert_record_bytes,
            "expert_record_mib": spec.expert_record_bytes / 1024**2,
        },
        "plan": {
            "fits_fixed": plan.fits_fixed,
            "scope": (
                "planned MTPLX buffers and explicit reserves; runtime must also "
                "enforce the MLX cap and reject unplanned context growth"
            ),
            "memory_limit_gib_input": str(args.memory_limit_gib),
            "memory_limit_bytes": plan.total_limit_bytes,
            "memory_limit_gib": _as_gib(plan.total_limit_bytes),
            "context_tokens": plan.context_tokens,
            "kv_bytes_per_token": spec.kv_bytes_per_token,
            "resident_bytes": plan.resident_bytes,
            "resident_gib": _as_gib(plan.resident_bytes),
            "kv_bytes": plan.kv_bytes,
            "kv_gib": _as_gib(plan.kv_bytes),
            "runtime_reserve_bytes": plan.runtime_reserve_bytes,
            "runtime_reserve_gib": _as_gib(plan.runtime_reserve_bytes),
            "runtime_reserve_gib_input": str(args.runtime_reserve_gib),
            "io_staging_bytes": plan.io_staging_bytes,
            "io_staging_gib": _as_gib(plan.io_staging_bytes),
            "io_staging_gib_input": str(args.io_staging_gib),
            "execution_workspace_bytes": plan.execution_workspace_bytes,
            "execution_workspace_gib": _as_gib(plan.execution_workspace_bytes),
            "execution_workspace_gib_input": str(args.execution_workspace_gib),
            "transient_slots": plan.transient_slots,
            "transient_bytes": plan.transient_bytes,
            "transient_gib": _as_gib(plan.transient_bytes),
            "expert_cache_limit_bytes": plan.expert_cache_limit_bytes,
            "expert_cache_limit_gib_input": (
                None
                if args.expert_cache_limit_gib is None
                else str(args.expert_cache_limit_gib)
            ),
            "fixed_bytes": plan.fixed_bytes,
            "fixed_gib": _as_gib(plan.fixed_bytes),
            "available_after_fixed_bytes": max(
                0, plan.total_limit_bytes - plan.fixed_bytes
            ),
            "persistent_budget_bytes": plan.persistent_budget_bytes,
            "persistent_budget_gib": _as_gib(plan.persistent_budget_bytes),
            "persistent_slots_per_layer": plan.slots_per_layer,
            "persistent_cache_bytes": plan.persistent_cache_bytes,
            "persistent_cache_gib": _as_gib(plan.persistent_cache_bytes),
            "accounted_bytes": plan.allocated_bytes,
            "accounted_gib": _as_gib(plan.allocated_bytes),
            "unallocated_bytes": plan.unallocated_bytes,
            "unallocated_gib": _as_gib(plan.unallocated_bytes),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if plan.fits_fixed else 2


if __name__ == "__main__":
    raise SystemExit(main())
