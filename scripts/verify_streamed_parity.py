#!/usr/bin/env python3
"""Verify streamed AR token/logit probes against a checked-in golden JSONL."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_runtime import ExpertStreamingConfig, parse_memory_bytes  # noqa: E402
from mtplx.runtime import load  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("probes", type=Path)
    parser.add_argument(
        "--model-key",
        choices=[
            "hy3-q4",
            "glm52-q4",
            "hy3-expert-only-q4",
            "hy3-expert-q2",
        ],
        required=True,
    )
    parser.add_argument("--memory-limit", required=True)
    parser.add_argument("--max-live-kv-tokens", type=_positive_int, required=True)
    parser.add_argument("--runtime-reserve", default="16GiB")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    return parser


def _load_probes(path: Path) -> list[dict]:
    probes = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise SystemExit(f"{path}:{line_number}: probe must be an object")
        ids = value.get("input_ids")
        expected_ids = value.get("topk_ids")
        expected_logits = value.get("topk_logits")
        if (
            not isinstance(ids, list)
            or not ids
            or not isinstance(expected_ids, list)
            or not expected_ids
            or not isinstance(expected_logits, list)
            or len(expected_ids) != len(expected_logits)
        ):
            raise SystemExit(
                f"{path}:{line_number}: input_ids/topk_ids/topk_logits are required"
            )
        probes.append(value)
    if not probes:
        raise SystemExit("probe file is empty")
    return probes


def main() -> int:
    args = build_parser().parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        raise SystemExit("--atol must be finite and non-negative")
    if not math.isfinite(args.rtol) or args.rtol < 0:
        raise SystemExit("--rtol must be finite and non-negative")
    probes = _load_probes(args.probes)
    config = ExpertStreamingConfig(
        model_key=args.model_key,
        memory_limit_bytes=parse_memory_bytes(args.memory_limit),
        max_live_kv_tokens=args.max_live_kv_tokens,
        runtime_reserve_bytes=parse_memory_bytes(args.runtime_reserve),
    )
    runtime = load(
        args.model_root,
        mtp=False,
        expert_streaming_config=config,
        expert_manifest=args.manifest,
    )
    results = []
    try:
        import mlx.core as mx

        for index, probe in enumerate(probes):
            input_ids = [int(token) for token in probe["input_ids"]]
            expected_ids = [int(token) for token in probe["topk_ids"]]
            expected_logits = [float(value) for value in probe["topk_logits"]]
            runtime.expert_streaming.reset()
            cache = runtime.make_cache()
            with runtime.admit_kv_tokens(len(input_ids)):
                logits = runtime.forward_ar(
                    mx.array([input_ids], dtype=mx.int32), cache=cache
                )
                last = logits[0, -1].astype(mx.float32)
                order = mx.argsort(last)[-len(expected_ids) :][::-1]
                actual = mx.take(last, order)
                mx.eval(order, actual)
            actual_ids = [int(value) for value in order.tolist()]
            actual_logits = [float(value) for value in actual.tolist()]
            ids_match = actual_ids == expected_ids
            logits_match = all(
                math.isclose(
                    actual_value, expected_value, rel_tol=args.rtol, abs_tol=args.atol
                )
                for actual_value, expected_value in zip(
                    actual_logits, expected_logits, strict=True
                )
            )
            results.append(
                {
                    "index": index,
                    "name": probe.get("name") or f"probe-{index}",
                    "passed": ids_match and logits_match,
                    "expected_topk_ids": expected_ids,
                    "actual_topk_ids": actual_ids,
                    "max_abs_logit_error": max(
                        abs(actual_value - expected_value)
                        for actual_value, expected_value in zip(
                            actual_logits, expected_logits, strict=True
                        )
                    ),
                }
            )
    finally:
        snapshot = runtime.expert_streaming_snapshot()
        runtime.close(timeout=10.0)

    passed = all(result["passed"] for result in results)
    print(
        json.dumps(
            {
                "schema": "mtplx-streamed-parity-v1",
                "passed": passed,
                "model_key": args.model_key,
                "probe_count": len(results),
                "atol": args.atol,
                "rtol": args.rtol,
                "results": results,
                "streaming": snapshot,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
