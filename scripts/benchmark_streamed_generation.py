#!/usr/bin/env python3
"""Run provenance-rich deterministic AR benchmarks through streamed experts."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
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


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True, timeout=2
        ).strip()
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model-key", choices=["hy3-q4", "glm52-q4"], required=True)
    parser.add_argument("--memory-limit", required=True)
    parser.add_argument("--max-live-kv-tokens", type=_positive_int, required=True)
    parser.add_argument("--runtime-reserve", default="16GiB")
    parser.add_argument("--expert-cache-limit")
    parser.add_argument("--prompt", default="Explain why the sky is blue in one paragraph.")
    parser.add_argument("--max-tokens", type=_positive_int, default=64)
    parser.add_argument("--repeats", type=_positive_int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-between", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.model_root.expanduser().resolve()
    config = ExpertStreamingConfig(
        model_key=args.model_key,
        memory_limit_bytes=parse_memory_bytes(args.memory_limit),
        max_live_kv_tokens=args.max_live_kv_tokens,
        runtime_reserve_bytes=parse_memory_bytes(args.runtime_reserve),
        expert_cache_limit_bytes=(
            parse_memory_bytes(args.expert_cache_limit)
            if args.expert_cache_limit
            else None
        ),
    )
    runtime = load(
        root,
        mtp=False,
        expert_streaming_config=config,
        expert_manifest=args.manifest,
    )
    rows = []
    try:
        from mtplx.generation import generate_ar
        from mtplx.sampling import SamplerConfig

        prompt_ids = runtime.tokenizer.encode(args.prompt)
        sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=1)
        for repeat in range(args.repeats):
            if args.reset_between and repeat:
                runtime.expert_streaming.reset()
            before = runtime.expert_streaming_snapshot()
            started = time.perf_counter()
            with runtime.admit_kv_tokens(len(prompt_ids) + args.max_tokens):
                result = generate_ar(
                    runtime,
                    prompt_ids,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    seed=args.seed,
                )
            elapsed = time.perf_counter() - started
            after = runtime.expert_streaming_snapshot()
            token_ids = [int(token) for token in result.tokens]
            rows.append(
                {
                    "repeat": repeat,
                    "elapsed_seconds": elapsed,
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(token_ids),
                    "completion_tokens_per_second": len(token_ids) / elapsed,
                    "token_ids": token_ids,
                    "text": runtime.tokenizer.decode(token_ids),
                    "streaming_before": before,
                    "streaming_after": after,
                    "generation_stats": result.stats.to_dict(),
                }
            )
    finally:
        runtime.close(timeout=10.0)

    payload = {
        "schema": "mtplx-streamed-generation-benchmark-v1",
        "git_commit": _git_commit(),
        "model_root": str(root),
        "model_key": args.model_key,
        "manifest": str(args.manifest.resolve()),
        "config": config.to_dict(),
        "seed": args.seed,
        "reset_between": args.reset_between,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "runs": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
