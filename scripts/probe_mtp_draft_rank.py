#!/usr/bin/env python3
"""Teacher-forced Hy3 MTP draft-rank probe.

Measures WHERE the true next-next token ranks in the layer-80 NextN head's
draft distribution, position by position, without any sampling loop in the
way: the full prompt is prefilled teacher-forced through the trunk, and for
each probed position ``i`` the head drafts from the trunk hidden ``h_i`` plus
the true next token ``t_{i+1}`` (exactly the generate_mtp1 draft geometry,
fresh MTP cache per draft) while the true target ``t_{i+2}`` is known.

Both trunk hidden variants (``post_norm`` and ``pre_norm``) are probed from
the same prefill pass, so one run answers whether low acceptance is a
ranking problem, an echo problem (the head parroting its own input token),
or a hidden-variant problem.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_runtime import ExpertStreamingConfig, parse_memory_bytes  # noqa: E402
from mtplx.runtime import load  # noqa: E402

HIDDEN_VARIANTS = ("post_norm", "pre_norm")
SCHEMA = "mtplx-mtp-draft-rank-probe-v1"


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
    parser.add_argument(
        "--cache-policy",
        choices=["frequency", "lru"],
        default="frequency",
    )
    parser.add_argument(
        "--cache-scope",
        choices=["layer", "global"],
        default="layer",
        help="Partition persistent expert records by layer or share them globally.",
    )
    parser.add_argument(
        "--transient-slots",
        type=_positive_int,
        help="Global miss-service/I/O slots (default: model top-k).",
    )
    parser.add_argument(
        "--read-chunk",
        default="8MiB",
        help="Maximum native positional-read chunk (default: 8MiB).",
    )
    parser.add_argument(
        "--f-nocache",
        action="store_true",
        help="Use macOS F_NOCACHE reads directly into shared expert slots.",
    )
    parser.add_argument(
        "--slot-layout",
        choices=["direct-slots", "component-banks", "metal-mmap"],
        default="direct-slots",
    )
    parser.add_argument(
        "--verified-sidecar",
        action="store_true",
        help="Verify the full sidecar once at open, then skip repeated record hashes.",
    )
    parser.add_argument(
        "--trust-sidecar",
        action="store_true",
        help=(
            "Explicitly trust the manifest's sidecar digest and skip both "
            "startup and per-record hashing. Intended for an unchanged local "
            "sidecar that was fully verified earlier."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="Prompt text whose token stream is teacher-forced through the trunk.",
    )
    parser.add_argument(
        "--chat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Encode the prompt as a user turn with the artifact chat template.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a precise senior software engineer. Give a complete, self-contained answer.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--reasoning-effort")
    parser.add_argument(
        "--probe-positions",
        type=_positive_int,
        default=200,
        help=(
            "Probe the last N usable positions (capped at prompt_tokens - 2: "
            "each probe needs a trunk hidden, the head's input token, and a "
            "known next-next target)."
        ),
    )
    parser.add_argument(
        "--prefill-chunk",
        type=_positive_int,
        default=256,
        help="Teacher-forced trunk prefill chunk size (bounds live graph memory).",
    )
    parser.add_argument(
        "--draft-batch",
        type=_positive_int,
        default=32,
        help=(
            "Positions drafted per batched head call. Each position is an "
            "independent single-token draft on a fresh MTP cache; batching "
            "along the batch axis is mathematically identical to one-at-a-time."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write the complete probe payload (per-position records) here.",
    )
    parser.add_argument(
        "--enable-mtp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Required acknowledgment that the layer-80 NextN head will be "
            "loaded (hy3-q4 only; requires --mtp-artifacts). The probe cannot "
            "run without it."
        ),
    )
    parser.add_argument(
        "--mtp-artifacts",
        type=Path,
        help=(
            "Directory holding the layer-80 Hy3 MTP head artifacts "
            "(layer80-bf16.safetensors for bf16; layer80-residents-q"
            ".safetensors and layer80-q4.safetensors for q4)."
        ),
    )
    parser.add_argument(
        "--mtp-precision",
        choices=("bf16", "q4"),
        help=(
            "Layer-80 NextN head precision (default bf16, the bit-exact head "
            "per docs/FORGE_BACKEND_CONTRACT.md section 6; ~7.5 GB resident "
            "- budget it against --expert-cache-limit). Requires --enable-mtp."
        ),
    )
    return parser


def validate_probe_flags(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Mirror benchmark_streamed_generation.validate_mtp_flags, tightened.

    The benchmark treats MTP as opt-in; this probe measures the head itself,
    so --enable-mtp is mandatory rather than optional, and the artifact
    directory must be supplied (loading then fails closed on missing or
    mismatched artifact files).
    """

    if not args.enable_mtp:
        parser.error(
            "this probe measures the layer-80 NextN head; pass --enable-mtp "
            "with --mtp-artifacts"
        )
    if args.model_key != "hy3-q4":
        parser.error("--enable-mtp is packaged for --model-key hy3-q4 only")
    if args.mtp_artifacts is None:
        parser.error("--enable-mtp requires --mtp-artifacts")
    if args.mtp_precision is None:
        args.mtp_precision = "bf16"


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    ranks = [record["rank"] for record in records]
    return {
        "positions": count,
        "acceptance_at_1": sum(rank == 0 for rank in ranks) / count,
        "top5_rate": sum(rank < 5 for rank in ranks) / count,
        "top20_rate": sum(rank < 20 for rank in ranks) / count,
        "median_rank": float(statistics.median(ranks)),
        "echo_rate": sum(record["echo"] for record in records) / count,
        "mean_true_prob": sum(record["true_prob"] for record in records) / count,
    }


def probe_draft_ranks(
    runtime: Any,
    token_ids: Sequence[int],
    *,
    probe_positions: int,
    prefill_chunk: int = 256,
    draft_batch: int = 32,
    variants: Sequence[str] = HIDDEN_VARIANTS,
) -> dict[str, Any]:
    """Teacher-forced draft-rank measurement over the last N usable positions.

    For probed trunk position ``i`` (hidden after processing ``t_i``) the
    head drafts with input token ``t_{i+1}`` on a fresh MTP cache — exactly
    generate_mtp1's draft call — and the record reports where the true
    ``t_{i+2}`` ranks in that draft distribution (rank 0 == argmax match),
    the true token's draft probability, the draft argmax probability, and
    whether the argmax merely echoes the head's own input token.

    Both trunk hidden variants come from one prefill pass via
    ``model.model(chunk, cache, return_pre_norm=True)``; only the hiddens for
    the probed tail are retained, so long prompts stay memory-bounded.
    """

    import mlx.core as mx

    from mtplx.attention_context import attention_phase

    if not getattr(runtime, "mtp_enabled", False):
        raise RuntimeError("probe_draft_ranks requires an MTP-enabled runtime")
    model = runtime.model
    if not hasattr(model, "mtp_forward"):
        raise RuntimeError("runtime model exposes no mtp_forward surface")
    unknown = set(variants) - set(HIDDEN_VARIANTS)
    if unknown:
        raise ValueError(f"unknown hidden variants: {sorted(unknown)}")

    token_ids = [int(token) for token in token_ids]
    total = len(token_ids)
    usable = total - 2
    if usable < 1:
        raise ValueError(
            f"prompt has {total} tokens; probing needs at least 3 "
            "(trunk hidden, head input token, next-next target)"
        )
    positions = min(int(probe_positions), usable)
    first_position = usable - positions

    trunk = getattr(model, "language_model", model).model
    cache = runtime.make_cache()
    kept: dict[str, list[Any]] = {"post_norm": [], "pre_norm": []}
    start = 0
    while start < total:
        stop = min(start + int(prefill_chunk), total)
        chunk = mx.array([token_ids[start:stop]])
        with attention_phase("prefill"), runtime._expert_routing_context(chunk):
            post_norm, pre_norm = trunk(chunk, cache, return_pre_norm=True)
        # Materialize per chunk: bounds the live graph and forces the KV
        # cache writes for this chunk before the next one builds on them.
        mx.eval(post_norm, pre_norm)
        keep_from = max(start, first_position)
        if keep_from < stop:
            offset = keep_from - start
            kept["post_norm"].append(post_norm[:, offset:, :])
            kept["pre_norm"].append(pre_norm[:, offset:, :])
        start = stop
    hiddens = {
        variant: mx.concatenate(slices, axis=1) for variant, slices in kept.items()
    }

    last_probe = total - 3  # inclusive: the target t_{i+2} must exist
    variant_payloads: dict[str, Any] = {}
    for variant in variants:
        hidden_all = hiddens[variant]
        records: list[dict[str, Any]] = []
        for batch_start in range(first_position, last_probe + 1, int(draft_batch)):
            batch_end = min(batch_start + int(draft_batch), last_probe + 1)
            index = batch_start - first_position
            width = batch_end - batch_start
            # (1, width, H) -> (width, 1, H): independent single-token drafts
            # stacked on the batch axis, each on the same fresh cache batch.
            hidden = hidden_all[0, index : index + width, :][:, None, :]
            inputs = mx.array(
                [[token_ids[i + 1]] for i in range(batch_start, batch_end)]
            )
            mtp_cache = runtime.make_mtp_cache()
            logits = model.mtp_forward(hidden, inputs, mtp_cache=mtp_cache)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, -1, :].astype(mx.float32)
            probs = mx.softmax(logits, axis=-1)
            true_tokens = mx.array(
                [token_ids[i + 2] for i in range(batch_start, batch_end)]
            )
            true_logits = mx.take_along_axis(logits, true_tokens[:, None], axis=-1)
            ranks = mx.sum(logits > true_logits, axis=-1)
            true_probs = mx.take_along_axis(probs, true_tokens[:, None], axis=-1)
            argmaxes = mx.argmax(logits, axis=-1)
            argmax_probs = mx.max(probs, axis=-1)
            mx.eval(ranks, true_probs, argmaxes, argmax_probs)
            for row, position in enumerate(range(batch_start, batch_end)):
                input_token = token_ids[position + 1]
                argmax_token = int(argmaxes[row].item())
                records.append(
                    {
                        "position": position,
                        "input_token": input_token,
                        "true_token": token_ids[position + 2],
                        "rank": int(ranks[row].item()),
                        "true_prob": float(true_probs[row, 0].item()),
                        "argmax_token": argmax_token,
                        "argmax_prob": float(argmax_probs[row].item()),
                        "echo": bool(argmax_token == input_token),
                    }
                )
        variant_payloads[variant] = {
            "records": records,
            "summary": _summarize(records),
        }
    return {
        "prompt_tokens": total,
        "probe_positions_requested": int(probe_positions),
        "probe_positions": positions,
        "probe_positions_capped": positions < int(probe_positions),
        "first_position": first_position,
        "last_position": last_probe,
        "prefill_chunk": int(prefill_chunk),
        "draft_batch": int(draft_batch),
        "variants": variant_payloads,
    }


def format_summary_table(variants: dict[str, Any]) -> str:
    header = (
        f"{'variant':<10} {'n':>5} {'accept@1':>9} {'top5':>7} {'top20':>7} "
        f"{'median_rank':>12} {'echo':>7} {'mean_p(true)':>13}"
    )
    lines = [header, "-" * len(header)]
    for variant, payload in variants.items():
        summary = payload["summary"]
        lines.append(
            f"{variant:<10} {summary['positions']:>5} "
            f"{summary['acceptance_at_1']:>9.3f} {summary['top5_rate']:>7.3f} "
            f"{summary['top20_rate']:>7.3f} {summary['median_rank']:>12.1f} "
            f"{summary['echo_rate']:>7.3f} {summary['mean_true_prob']:>13.4f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.verified_sidecar and args.trust_sidecar:
        parser.error("--verified-sidecar and --trust-sidecar are mutually exclusive")
    validate_probe_flags(parser, args)
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
        cache_policy=args.cache_policy,
        cache_scope=args.cache_scope,
        transient_slots=args.transient_slots,
        max_read_chunk_bytes=parse_memory_bytes(args.read_chunk),
        bypass_page_cache=args.f_nocache,
        slot_layout=args.slot_layout,
        verify_record_hashes=not (
            args.trust_sidecar or args.slot_layout == "metal-mmap"
        ),
        verify_sidecar_hash_at_open=args.verified_sidecar,
    )
    # Loading fails closed: missing/mismatched layer-80 artifacts raise
    # Hy3MTPLoadError inside load() instead of degrading to AR.
    runtime = load(
        root,
        mtp=True,
        expert_streaming_config=config,
        expert_manifest=args.manifest,
        mtp_artifacts=args.mtp_artifacts.expanduser().resolve(),
        mtp_precision=args.mtp_precision,
    )
    started = time.perf_counter()
    try:
        prompt_text = args.prompt_file.expanduser().read_text(encoding="utf-8")
        if args.chat:
            from mtplx.chat_encoding import encode_chat_messages

            token_ids = encode_chat_messages(
                runtime.tokenizer,
                [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                enable_thinking=args.enable_thinking,
                reasoning_effort=args.reasoning_effort,
            )
        else:
            token_ids = runtime.tokenizer.encode(prompt_text)
        with runtime.admit_kv_tokens(len(token_ids) + args.draft_batch):
            probe = probe_draft_ranks(
                runtime,
                token_ids,
                probe_positions=args.probe_positions,
                prefill_chunk=args.prefill_chunk,
                draft_batch=args.draft_batch,
            )
    finally:
        runtime.close(timeout=10.0)
    elapsed = time.perf_counter() - started
    payload = {
        "schema": SCHEMA,
        "git_commit": _git_commit(),
        "model_root": str(root),
        "model_key": args.model_key,
        "manifest": str(args.manifest.resolve()),
        "config": config.to_dict(),
        "mtp": {
            "enabled": True,
            "artifacts": str(args.mtp_artifacts.expanduser().resolve()),
            "precision": args.mtp_precision,
        },
        "chat": args.chat,
        "enable_thinking": args.enable_thinking,
        "reasoning_effort": args.reasoning_effort,
        "prompt_file": str(args.prompt_file.expanduser().resolve()),
        "probe_seconds": elapsed,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        **probe,
    }
    print(format_summary_table(payload["variants"]))
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        output_json = args.output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {output_json}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
