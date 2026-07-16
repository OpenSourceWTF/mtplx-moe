#!/usr/bin/env python3
"""Device-scope no-initialization litmus for fused Hy3 router election."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mtplx.hy3_router_last_arrival import (
    TaggedArrivalLayout,
    tagged_arrival_checksums,
    tagged_arrival_litmus_source,
    tagged_arrival_payload,
    tagged_arrival_tag,
)
from mtplx.qwen_guard import DEFAULT_MLX_LOCK_PATH, exclusive_mlx_window


@lru_cache(maxsize=4)
def build_litmus_kernel(layout: TaggedArrivalLayout):
    """Build one exact-size lazy MLX custom kernel."""

    return mx.fast.metal_kernel(
        name=(
            "mtplx_hy3_router_tagged_arrival_"
            f"t{layout.threadgroups}_e{layout.elections}"
        ),
        input_names=["base_event", "seed"],
        output_names=["scratch"],
        source=tagged_arrival_litmus_source(layout),
        ensure_row_contiguous=True,
    )


def dispatch_litmus(
    *,
    layout: TaggedArrivalLayout,
    base_event: int,
    seed: int,
):
    """Dispatch without `init_value`; uninitialized scratch is the test subject."""

    kernel = build_litmus_kernel(layout)
    (scratch,) = kernel(
        inputs=[
            mx.array(int(base_event) & 0xFFFFFFFF, dtype=mx.uint32),
            mx.array(int(seed) & 0xFFFFFFFF, dtype=mx.uint32),
        ],
        grid=(layout.threadgroups * layout.elections * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(layout.total_words,)],
        output_dtypes=[mx.uint32],
    )
    return scratch


def validate_litmus_scratch(
    scratch: np.ndarray,
    *,
    layout: TaggedArrivalLayout,
    base_event: int,
    seed: int,
) -> dict[str, Any]:
    """Validate claim, publication, payload, checksum, and winner invariants."""

    values = np.asarray(scratch, dtype=np.uint32).reshape(-1)
    if values.size != layout.total_words:
        raise ValueError(
            f"litmus scratch has {values.size} words, expected {layout.total_words}"
        )
    rows = values.reshape(layout.elections, layout.words_per_election)
    counters = {
        "successful_elections": 0,
        "failed_elections": 0,
        "flag_failures": 0,
        "payload_failures": 0,
        "checksum_failures": 0,
        "winner_failures": 0,
    }
    first_failure: dict[str, Any] | None = None
    metadata_offset = layout.flag_words + layout.payload_words
    for election, row in enumerate(rows):
        event = int(base_event) + election
        tag = tagged_arrival_tag(event)
        expected_claim = (~tag) & 0xFFFFFFFF
        ready_failure = int(row[0]) != expected_claim or any(
            int(value) != tag for value in row[1 : layout.ready_words]
        )
        check_failure = any(
            int(value) != expected_claim
            for value in row[layout.ready_words : layout.flag_words]
        )
        flag_failure = ready_failure or check_failure
        payload_failure = False
        for group in range(layout.threadgroups):
            observed = int(row[layout.flag_words + group])
            expected = tagged_arrival_payload(event=event, group=group, seed=seed)
            payload_failure = payload_failure or observed != expected
        checksum_sum, checksum_xor = tagged_arrival_checksums(
            event=event,
            seed=seed,
            threadgroups=layout.threadgroups,
        )
        winner = int(row[metadata_offset])
        winner_failure = not 0 <= winner < layout.threadgroups
        checksum_failure = (
            int(row[metadata_offset + 1]) != checksum_sum
            or int(row[metadata_offset + 2]) != checksum_xor
        )
        failures = {
            "flag": flag_failure,
            "payload": payload_failure,
            "checksum": checksum_failure,
            "winner": winner_failure,
        }
        for name, failed in failures.items():
            counters[f"{name}_failures"] += int(failed)
        if any(failures.values()):
            counters["failed_elections"] += 1
            if first_failure is None:
                first_failure = {
                    "event": event,
                    "election_in_batch": election,
                    "failures": failures,
                    "observed_flag0": int(row[0]),
                    "expected_claim": expected_claim,
                    "winner": winner,
                }
        else:
            counters["successful_elections"] += 1
    return {**counters, "first_failure": first_failure}


def run_litmus(
    *,
    total_elections: int,
    elections_per_dispatch: int,
    seed: int,
    threadgroups: int = 16,
) -> dict[str, Any]:
    """Run and validate the complete device litmus campaign."""

    if total_elections <= 0 or elections_per_dispatch <= 0:
        raise ValueError("litmus election counts must be positive")
    if total_elections % elections_per_dispatch:
        raise ValueError("total elections must divide evenly into dispatch batches")
    layout = TaggedArrivalLayout(
        threadgroups=threadgroups,
        elections=elections_per_dispatch,
    )
    source = tagged_arrival_litmus_source(layout)
    aggregate: dict[str, Any] = {
        "successful_elections": 0,
        "failed_elections": 0,
        "flag_failures": 0,
        "payload_failures": 0,
        "checksum_failures": 0,
        "winner_failures": 0,
        "first_failure": None,
    }
    batches = total_elections // elections_per_dispatch
    started = time.perf_counter()
    completed_batches = 0
    for batch in range(batches):
        base_event = batch * elections_per_dispatch
        batch_seed = (int(seed) ^ ((batch + 1) * 0xA511E9B3)) & 0xFFFFFFFF
        scratch = dispatch_litmus(
            layout=layout,
            base_event=base_event,
            seed=batch_seed,
        )
        mx.eval(scratch)
        validation = validate_litmus_scratch(
            np.asarray(scratch),
            layout=layout,
            base_event=base_event,
            seed=batch_seed,
        )
        completed_batches += 1
        for name in (
            "successful_elections",
            "failed_elections",
            "flag_failures",
            "payload_failures",
            "checksum_failures",
            "winner_failures",
        ):
            aggregate[name] += int(validation[name])
        if aggregate["first_failure"] is None:
            aggregate["first_failure"] = validation["first_failure"]
        if validation["failed_elections"]:
            break
        if completed_batches % 16 == 0 or completed_batches == batches:
            print(
                f"validated {aggregate['successful_elections']} tagged elections",
                file=sys.stderr,
                flush=True,
            )
    elapsed = time.perf_counter() - started
    return {
        "schema": "mtplx-issue58-tagged-last-arrival-litmus-v2",
        "status": "pass" if aggregate["failed_elections"] == 0 else "fail",
        "requested_elections": int(total_elections),
        "completed_batches": completed_batches,
        "requested_batches": batches,
        "elections_per_dispatch": int(elections_per_dispatch),
        "threadgroups_per_election": layout.threadgroups,
        "threads_per_threadgroup": 32,
        "scratch_bytes_per_dispatch": layout.total_bytes,
        "seed": int(seed) & 0xFFFFFFFF,
        "elapsed_seconds": elapsed,
        "elections_per_second": aggregate["successful_elections"] / elapsed,
        "validation": aggregate,
        "kernel": {
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "output_initialized": False,
            "readiness_words_per_producer": 2,
            "readiness_encoding": "tag-and-bitwise-complement",
            "ordinary_atomic_order": "relaxed",
            "publication_fence": "seq_cst mem_device thread_scope_device",
            "readiness_spin": False,
        },
        "environment": {"device": mx.metal.device_info()},
    }


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-elections", type=_positive_int, default=1_048_576)
    parser.add_argument(
        "--elections-per-dispatch",
        type=_positive_int,
        default=4096,
    )
    parser.add_argument("--seed", type=int, default=58_051)
    parser.add_argument(
        "--threadgroups",
        type=int,
        choices=(16, 24, 32, 48),
        default=16,
    )
    parser.add_argument(
        "--qwen-plist",
        type=Path,
        default=Path.home() / "Library/LaunchAgents/com.tea.qwen.plist",
    )
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_MLX_LOCK_PATH)
    parser.add_argument("--lock-timeout-seconds", type=float, default=21_600.0)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started_at = datetime.now(UTC).isoformat()
    with exclusive_mlx_window(
        plist=args.qwen_plist,
        lock_path=args.lock_path,
        lock_timeout_seconds=args.lock_timeout_seconds,
    ) as receipt:
        print(
            f"acquired exclusive MLX window at {receipt.lock_path}",
            file=sys.stderr,
            flush=True,
        )
        result = run_litmus(
            total_elections=args.total_elections,
            elections_per_dispatch=args.elections_per_dispatch,
            seed=args.seed,
            threadgroups=args.threadgroups,
        )
        result["exclusive_window"] = {
            "lock_path": str(receipt.lock_path),
            "lock_holder_pid": os.getpid(),
            "qwen_loaded_before": receipt.qwen_state.loaded,
            "qwen_models_before": list(receipt.qwen_state.models),
        }
    result["started_at"] = started_at
    result["finished_at"] = datetime.now(UTC).isoformat()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
