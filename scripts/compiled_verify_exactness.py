#!/usr/bin/env python3
"""Compiled-verify exactness gate (W2).

Runs the real speculative pipeline with ``MTPLX_COMPILED_VERIFY=parity``: the
CompiledVerifyBank double-runs every verify call — compiled pure step first
(committing nothing), then today's eager forward on the real cache as the
authority — and asserts exact equality (``np.array_equal``) on logits, hidden,
and every capture/state leaf.  Any mismatch aborts the stream with a
``CompiledVerifyParityError`` diff report, which this script records.

Modeled on ``scripts/phase0h_paged_verifier_exactness.py``: loads a model,
sweeps D1/D2/D3 with greedy and temperature samplers over a short and a
>4096-token context, and reports a zero-mismatch verdict.

NOTE: this script loads a real model and owns the GPU while it runs.  Do not
launch it while another workstream is using the machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_PROMPT = (
    "Create a single-file HTML5 Canvas flappy bird game. All visuals drawn "
    "procedurally. Animated bird with distinct up-stroke and down-stroke wing "
    "shapes, body tilt, squash-and-stretch on flap, feather particles from "
    "wing tips. Pipes with gradient shading, cap/lip, cylindrical highlight. "
    "Three-layer parallax background: sky with day/night colour cycle and "
    "stars, clouds with bobbing, rolling hills. Death explosion, +1 score pop, "
    "ambient floating motes. Start screen, death screen with best score in "
    "localStorage. Delta-time physics. Make it gorgeous."
)


@contextmanager
def patched_env(updates: dict[str, str | None]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _csv_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out or any(item < 1 for item in out):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return out


def _repeat_tokens(token_ids: list[int], needed: int) -> tuple[list[int], bool]:
    if not token_ids:
        raise ValueError("prompt encoded to no tokens")
    out = list(token_ids)
    while len(out) < needed:
        out.extend(token_ids)
    return out[:needed], len(token_ids) < needed


def _bank_stats(out: Any) -> dict[str, Any]:
    graphbank = getattr(out.stats, "graphbank", None) or {}
    return dict(graphbank.get("compiled_verify") or {})


def _run_case(
    rt: Any,
    prompt_ids: list[int],
    *,
    depth: int,
    sampler_name: str,
    sampler: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from mtplx.generation import generate_mtpk
    from mtplx.graphbank import CompiledVerifyParityError

    row: dict[str, Any] = {
        "context_len": len(prompt_ids),
        "depth": depth,
        "sampler": sampler_name,
        "max_tokens": int(args.max_tokens),
    }
    started = time.perf_counter()
    try:
        with patched_env({"MTPLX_COMPILED_VERIFY": "parity"}):
            out = generate_mtpk(
                rt,
                prompt_ids,
                max_tokens=int(args.max_tokens),
                sampler=sampler,
                speculative_depth=int(depth),
                verify_strategy="capture_commit",
                stop_token_ids=set(),
                seed=int(args.seed),
            )
    except CompiledVerifyParityError as exc:
        row["elapsed_s"] = time.perf_counter() - started
        row["mismatch"] = True
        row["mismatch_report"] = list(exc.report)
        row["passed"] = False
        return row
    row["elapsed_s"] = time.perf_counter() - started
    row["mismatch"] = False
    stats = _bank_stats(out)
    row["bank"] = stats
    row["generated_tokens"] = int(out.stats.generated_tokens)
    parity_checks = int(stats.get("parity_checks", 0))
    parity_failures = int(stats.get("parity_failures", 0))
    fallback_calls = int(stats.get("fallback_calls", 0))
    row["parity_checks"] = parity_checks
    row["parity_failures"] = parity_failures
    row["fallback_calls"] = fallback_calls
    row["fallback_reasons"] = dict(stats.get("fallback_reasons") or {})
    # A run that never reached the compiled path proves nothing: require the
    # configured number of double-run verify calls before calling it a pass.
    row["passed"] = (
        parity_failures == 0
        and parity_checks >= int(args.min_verify_calls)
    )
    if parity_checks < int(args.min_verify_calls):
        row["verdict_note"] = (
            f"inconclusive: only {parity_checks} parity-checked verify calls "
            f"(need >= {int(args.min_verify_calls)}); "
            f"fallback_reasons={row['fallback_reasons']}"
        )
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    from mtplx.runtime import load
    from mtplx.sampling import SamplerConfig

    rt = load(args.model, mtp=not args.no_mtp)
    base_ids = list(rt.tokenizer.encode(args.prompt))
    samplers = [
        ("greedy", SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)),
        (
            "temperature",
            SamplerConfig(
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for context_len in args.contexts:
        token_ids, synthetic_repeat = _repeat_tokens(base_ids, context_len)
        for depth in args.depths:
            for sampler_name, sampler in samplers:
                row = _run_case(
                    rt,
                    token_ids,
                    depth=depth,
                    sampler_name=sampler_name,
                    sampler=sampler,
                    args=args,
                )
                row["synthetic_repeat"] = synthetic_repeat
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                mx.clear_cache()

    passed = all(row["passed"] for row in rows)
    mismatches = [row for row in rows if row.get("mismatch")]
    return {
        "run_id": f"compiled-verify-exactness-{time.strftime('%Y%m%d-%H%M%S')}",
        "model": str(args.model),
        "mtp_enabled": not args.no_mtp,
        "contexts": args.contexts,
        "depths": args.depths,
        "samplers": [name for name, _ in samplers],
        "max_tokens": int(args.max_tokens),
        "min_verify_calls": int(args.min_verify_calls),
        "seed": int(args.seed),
        "mismatch_rows": len(mismatches),
        "verdict": "zero-mismatch" if passed and not mismatches else "FAILED",
        "passed": passed,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3.6-27B-MLXCommunity-4bit-CyanKiwiMTP"),
    )
    parser.add_argument(
        "--contexts",
        type=_csv_ints,
        default=_csv_ints("512,6144"),
        help="prompt lengths; keep one short and one >4096",
    )
    parser.add_argument(
        "--depths",
        type=_csv_ints,
        default=_csv_ints("1,2,3"),
        help="speculative depths to sweep (D1/D2/D3)",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument(
        "--min-verify-calls",
        type=int,
        default=8,
        help="minimum parity-checked verify calls per row for a conclusive pass",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "verdict": result["verdict"],
                "mismatch_rows": result["mismatch_rows"],
                "output": str(args.output) if args.output else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
