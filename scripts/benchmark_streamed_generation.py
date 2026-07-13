#!/usr/bin/env python3
"""Run provenance-rich deterministic AR benchmarks through streamed experts."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_runtime import ExpertStreamingConfig, parse_memory_bytes  # noqa: E402
from mtplx.benchmarks.resource_telemetry import (  # noqa: E402
    PowermetricsCollector,
    ResourceRun,
    ResourceTelemetrySampler,
)
from mtplx.runtime import load  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
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
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument(
        "--prompt",
    )
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument(
        "--context-tokens",
        type=_positive_int,
        help="Build an exact-size MTPLX prefill-ladder prompt for comparison runs.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["coding-agent", "legacy-repeat"],
        default="coding-agent",
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
        default=None,
    )
    parser.add_argument("--reasoning-effort")
    parser.add_argument(
        "--generation-profile",
        choices=["model-default", "deterministic", "qwen36-comparable"],
        default="deterministic",
        help=(
            "Sampling profile. Defaults to deterministic greedy so that "
            "unflagged runs are reproducible and comparable; pass "
            "model-default explicitly for vendor sampling."
        ),
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--window-tokens",
        type=_positive_int,
        default=32,
        help="Capture rolling decode/cache telemetry every N generated tokens.",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=256,
        help=(
            "Maximum generated tokens; generation still stops naturally at EOS. "
            "Defaults to a bounded 256 so an unflagged run cannot decode for "
            "hours; pass the documented model ceiling explicitly for "
            "full-response lanes (65,536 for both profiles; GLM-5.2's hard "
            "output max is 131,072)."
        ),
    )
    parser.add_argument(
        "--window-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Capture a full streaming snapshot at each rolling window. "
            "Disable for headline runs: the snapshot walks every slot "
            "condition and contends with in-flight miss loads."
        ),
    )
    parser.add_argument(
        "--resource-telemetry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Capture diagnostic resource throughput, queue occupancy, and "
            "I/O/Metal overlap. Off by default; enabled runs are diagnostic, "
            "not headline timing lanes."
        ),
    )
    parser.add_argument(
        "--resource-sample-interval",
        type=_positive_float,
        default=0.25,
        help="Seconds between cheap resource snapshots (default: 0.25).",
    )
    parser.add_argument(
        "--resource-max-samples",
        type=_positive_int,
        default=4096,
        help="Bounded resource timeline length (default: 4096 samples).",
    )
    parser.add_argument(
        "--ssd-ceiling-gib-s",
        type=_positive_float,
        help="Measured SSD ceiling used only for saturation evidence.",
    )
    parser.add_argument(
        "--powermetrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add non-interactive per-process CPU/GPU/wait/I/O samples. "
            "Requires --resource-telemetry and passwordless sudo authorization."
        ),
    )
    parser.add_argument("--repeats", type=_positive_int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-between", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=1,
        help=(
            "Saturation-lane stream count: decode N identical prompts through "
            "the streamed continuous-batch runner and report aggregate and "
            "per-stream tok/s. 1 keeps the reference single-stream path. "
            "Streams whose combined prompt+max-tokens KV exceeds "
            "--max-live-kv-tokens are serialized at step boundaries. "
            "Outputs at different concurrencies are not token-comparable: "
            "batch size is part of the run configuration label."
        ),
    )
    parser.add_argument(
        "--max-prefills-per-step",
        type=_positive_int,
        default=1,
        help=(
            "Joining prefills allowed per decode step boundary while other "
            "streams are actively decoding (concurrency > 1 only)."
        ),
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
        "--output-dir",
        type=Path,
        help="Save each generated response as Markdown in this directory.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Also write the complete benchmark payload to this JSON path.",
    )
    parser.add_argument(
        "--run-label",
        help="Filesystem-safe label used in saved response filenames.",
    )
    parser.add_argument(
        "--route-trace-json",
        type=Path,
        help="Save per-layer routed expert IDs for cache/prefetch simulation.",
    )
    parser.add_argument(
        "--enable-mtp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Speculative decoding through the packaged layer-80 NextN head "
            "(hy3-q4 only; requires --mtp-artifacts). Default off: the AR "
            "path is unchanged unless this flag is passed."
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
            "Layer-80 NextN head precision (default bf16). bf16 loads the "
            "bit-exact BF16 head (~7.5 GB resident; quantized MTP heads "
            "collapse acceptance, docs/FORGE_BACKEND_CONTRACT.md section 6) "
            "- budget it against --expert-cache-limit. q4 loads the pinned "
            "quantized artifacts (~1.94 GiB expert bank). Requires "
            "--enable-mtp."
        ),
    )
    return parser


def validate_mtp_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.enable_mtp:
        if args.model_key != "hy3-q4":
            parser.error("--enable-mtp is packaged for --model-key hy3-q4 only")
        if args.mtp_artifacts is None:
            parser.error("--enable-mtp requires --mtp-artifacts")
        if getattr(args, "concurrency", 1) > 1:
            parser.error(
                "--enable-mtp is single-stream; the batch runner is AR-only "
                "and concurrent MTP requests would only queue"
            )
        if args.mtp_precision is None:
            args.mtp_precision = "bf16"
    elif args.mtp_artifacts is not None:
        parser.error("--mtp-artifacts requires --enable-mtp")
    elif args.mtp_precision is not None:
        parser.error("--mtp-precision requires --enable-mtp")


def validate_resource_flags(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.resource_max_samples < 2:
        parser.error("--resource-max-samples must be at least 2")
    if args.powermetrics and not args.resource_telemetry:
        parser.error("--powermetrics requires --resource-telemetry")
    if args.ssd_ceiling_gib_s is not None and not args.resource_telemetry:
        parser.error("--ssd-ceiling-gib-s requires --resource-telemetry")


class _ConcurrentTokenCounter:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def observe(self, state) -> None:
        for stream in state.live:
            request_id = str(stream.request_id)
            self._counts[request_id] = max(
                self._counts.get(request_id, 0),
                int(stream.generated_tokens),
            )

    def finish(self, results) -> None:
        for result in results:
            request_id = str(result.request_id)
            self._counts[request_id] = max(
                self._counts.get(request_id, 0),
                len(result.tokens),
            )

    def count(self) -> int:
        return sum(self._counts.values())


@contextmanager
def _resource_telemetry(args, runtime, token_count):
    if not args.resource_telemetry:
        yield None
        return
    sampler = ResourceTelemetrySampler(
        runtime.expert_resource_telemetry_snapshot,
        token_count=token_count,
        interval_s=args.resource_sample_interval,
        max_samples=args.resource_max_samples,
    )
    power = PowermetricsCollector(
        enabled=args.powermetrics,
        pid=os.getpid(),
        interval_ms=max(100, int(args.resource_sample_interval * 1000)),
    )
    with power, sampler:
        yield ResourceRun(sampler=sampler, powermetrics=power)


def _attach_resource_report(
    row: dict,
    resource_run: ResourceRun | None,
    *,
    ssd_ceiling_gib_s: float | None,
    generation_thread_cpu_ns: int,
    generation_elapsed_ns: int,
    final_completion_tokens: int,
) -> None:
    if resource_run is None:
        return
    row["diagnostic_run"] = True
    row["resource_telemetry"] = resource_run.report(
        ssd_ceiling_gib_s=ssd_ceiling_gib_s,
        generation_thread_cpu_ns=generation_thread_cpu_ns,
        generation_elapsed_ns=generation_elapsed_ns,
        final_completion_tokens=final_completion_tokens,
    )


def build_concurrent_requests(
    prompt_ids,
    *,
    concurrency: int,
    max_tokens: int,
    sampler,
    seed: int,
):
    """Build the saturation lane's N identical prompts as batch requests.

    Prompts are identical across streams; per-stream seeds are ``seed + i``
    so sampled profiles produce distinct streams while the deterministic
    profile stays seed-independent.
    """

    from mtplx.streamed_batch import StreamedBatchRequest

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    return [
        StreamedBatchRequest(
            request_id=f"stream-{index:02d}",
            prompt_ids=tuple(int(token) for token in prompt_ids),
            max_tokens=max_tokens,
            sampler=sampler,
            seed=seed + index,
        )
        for index in range(concurrency)
    ]


def _run_concurrent_repeats(
    args,
    runtime,
    *,
    prompt_ids,
    sampler,
    max_tokens: int,
    run_label: str,
) -> list[dict]:
    """Saturation lane: N identical prompts, aggregate and per-stream tok/s."""

    from mtplx.streamed_batch import StreamedBatchRunner

    rows: list[dict] = []
    for repeat in range(args.repeats):
        if args.reset_between and repeat:
            runtime.expert_streaming.reset()
        requests = build_concurrent_requests(
            prompt_ids,
            concurrency=args.concurrency,
            max_tokens=max_tokens,
            sampler=sampler,
            seed=args.seed,
        )
        before = runtime.expert_streaming_snapshot()
        token_counter = _ConcurrentTokenCounter()
        runner = StreamedBatchRunner(
            runtime,
            max_concurrency=args.concurrency,
            max_prefills_per_step=args.max_prefills_per_step,
            on_step=token_counter.observe,
        )
        for request in requests:
            runner.submit(request)
        with _resource_telemetry(
            args,
            runtime,
            token_counter.count,
        ) as resource_run:
            started = time.perf_counter()
            thread_cpu_started = time.thread_time_ns()
            results = runner.run()
            token_counter.finish(results)
            thread_cpu_finished = time.thread_time_ns()
            finished = time.perf_counter()
        elapsed = finished - started
        after = runtime.expert_streaming_snapshot()
        streams = []
        for result in results:
            completion_tokens = len(result.tokens)
            stream_elapsed = result.last_token_s - result.admitted_s
            decode_elapsed = result.last_token_s - result.first_token_s
            response_path = None
            if args.output_dir is not None:
                output_dir = args.output_dir.expanduser().resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                response_path = output_dir / (
                    f"{args.model_key}-{run_label}-repeat-{repeat}"
                    f"-{result.request_id}.md"
                )
                response_path.write_text(result.text + "\n", encoding="utf-8")
            streams.append(
                {
                    "request_id": result.request_id,
                    "seed": args.seed + len(streams),
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "completion_tokens_per_second": (
                        completion_tokens / stream_elapsed
                        if stream_elapsed > 0.0
                        else 0.0
                    ),
                    "decode_tokens_per_second": (
                        (completion_tokens - 1) / decode_elapsed
                        if completion_tokens > 1 and decode_elapsed > 0.0
                        else 0.0
                    ),
                    "finish_reason": result.finish_reason,
                    "admitted_step": result.admitted_step,
                    "finished_step": result.finished_step,
                    "decode_steps": result.decode_steps,
                    "prefill_seconds": result.prefill_seconds,
                    "token_times_s": [
                        token_time - started for token_time in result.token_times_s
                    ],
                    "token_ids": list(result.tokens),
                    "text": result.text,
                    "response_path": (
                        str(response_path) if response_path is not None else None
                    ),
                }
            )
        aggregate_tokens = sum(stream["completion_tokens"] for stream in streams)
        row = {
            "repeat": repeat,
            "elapsed_seconds": elapsed,
            "prompt_tokens": len(prompt_ids),
            "concurrency": args.concurrency,
            "aggregate_completion_tokens": aggregate_tokens,
            "aggregate_completion_tokens_per_second": (
                aggregate_tokens / elapsed if elapsed > 0.0 else 0.0
            ),
            "scheduler": runner.stats(),
            "streams": streams,
            "streaming_before": before,
            "streaming_after": after,
        }
        _attach_resource_report(
            row,
            resource_run,
            ssd_ceiling_gib_s=args.ssd_ceiling_gib_s,
            generation_thread_cpu_ns=thread_cpu_finished - thread_cpu_started,
            generation_elapsed_ns=int(elapsed * 1e9),
            final_completion_tokens=aggregate_tokens,
        )
        rows.append(row)
    return rows


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_resource_flags(parser, args)
    root = args.model_root.expanduser().resolve()
    model_defaults = {
        "glm52-q4": {
            "max_tokens": 65_536,
            "max_output_tokens": 131_072,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 0,
            "enable_thinking": True,
            "reasoning_effort": "max",
        },
        "hy3-q4": {
            "max_tokens": 65_536,
            "max_output_tokens": 262_144,
            "temperature": 0.9,
            "top_p": 1.0,
            "top_k": 0,
            "enable_thinking": False,
            "reasoning_effort": None,
        },
    }[args.model_key]
    if args.generation_profile == "deterministic":
        profile_defaults = {
            **model_defaults,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "enable_thinking": False,
            "reasoning_effort": None,
        }
    elif args.generation_profile == "qwen36-comparable":
        profile_defaults = {
            **model_defaults,
            "max_tokens": 128,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "enable_thinking": False,
            "reasoning_effort": None,
        }
    else:
        profile_defaults = model_defaults
    max_tokens = args.max_tokens or int(profile_defaults["max_tokens"])
    benchmark_context_limit = 262_144
    if args.max_live_kv_tokens > benchmark_context_limit:
        parser.error(
            f"--max-live-kv-tokens {args.max_live_kv_tokens} exceeds the current "
            f"benchmark ceiling of {benchmark_context_limit}"
        )
    if max_tokens > int(model_defaults["max_output_tokens"]):
        parser.error(
            f"--max-tokens {max_tokens} exceeds {args.model_key}'s documented "
            f"maximum output of {model_defaults['max_output_tokens']}"
        )
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(profile_defaults["temperature"])
    )
    top_p = args.top_p if args.top_p is not None else float(profile_defaults["top_p"])
    top_k = args.top_k if args.top_k is not None else int(profile_defaults["top_k"])
    enable_thinking = (
        args.enable_thinking
        if args.enable_thinking is not None
        else bool(profile_defaults["enable_thinking"])
    )
    reasoning_effort = args.reasoning_effort or profile_defaults["reasoning_effort"]
    run_label = args.run_label or args.manifest.stem
    if not run_label or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in run_label
    ):
        parser.error("--run-label may contain only letters, digits, '-' and '_'")
    if args.verified_sidecar and args.trust_sidecar:
        parser.error("--verified-sidecar and --trust-sidecar are mutually exclusive")
    validate_mtp_flags(parser, args)
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
        trace_routes=args.route_trace_json is not None,
    )
    runtime = load(
        root,
        mtp=args.enable_mtp,
        expert_streaming_config=config,
        expert_manifest=args.manifest,
        mtp_artifacts=(
            args.mtp_artifacts.expanduser().resolve()
            if args.mtp_artifacts is not None
            else None
        ),
        mtp_precision=(args.mtp_precision or "bf16"),
    )
    rows = []
    try:
        from mtplx.generation import generate_ar, generate_mtp1
        from mtplx.sampling import SamplerConfig

        prompt_text = (
            args.prompt_file.expanduser().read_text(encoding="utf-8")
            if args.prompt_file is not None
            else args.prompt
        )
        prompt_metadata = None
        if args.context_tokens is not None:
            from mtplx.prefill_bench import _prompt_build_for_context

            prompt_build = _prompt_build_for_context(
                runtime.tokenizer,
                args.context_tokens,
                prompt_style=args.prompt_style,
                prompt_tail=prompt_text,
                prompt_format="chat" if args.chat else "raw",
                enable_thinking=enable_thinking,
            )
            prompt_ids = prompt_build.token_ids
            prompt_metadata = prompt_build.metadata
        elif args.chat:
            from mtplx.chat_encoding import encode_chat_messages

            prompt_ids = encode_chat_messages(
                runtime.tokenizer,
                [
                    {"role": "system", "content": args.system_prompt},
                    {
                        "role": "user",
                        "content": prompt_text
                        or "Explain why the sky is blue in one paragraph.",
                    },
                ],
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
        else:
            prompt_ids = runtime.tokenizer.encode(
                prompt_text or "Explain why the sky is blue in one paragraph."
            )
        sampler = SamplerConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        if args.concurrency > 1:
            rows.extend(
                _run_concurrent_repeats(
                    args,
                    runtime,
                    prompt_ids=prompt_ids,
                    sampler=sampler,
                    max_tokens=max_tokens,
                    run_label=run_label,
                )
            )
        # The single-stream reference lane runs only at concurrency 1.
        for repeat in range(args.repeats if args.concurrency == 1 else 0):
            if args.reset_between and repeat:
                runtime.expert_streaming.reset()
            before = runtime.expert_streaming_snapshot()
            decode_points = []
            decoded_count = 0

            def token_callback(token_ids):
                nonlocal decoded_count
                decoded_count += len(token_ids)
                if decoded_count == 1 or (decoded_count - 1) % args.window_tokens == 0:
                    decode_points.append(
                        {
                            "completion_tokens": decoded_count,
                            "time": time.perf_counter(),
                            "streaming": (
                                runtime.expert_streaming_snapshot()
                                if args.window_telemetry
                                else None
                            ),
                        }
                    )

            with _resource_telemetry(
                args,
                runtime,
                lambda: decoded_count,
            ) as resource_run:
                started = time.perf_counter()
                thread_cpu_started = time.thread_time_ns()
                with runtime.admit_kv_tokens(len(prompt_ids) + max_tokens):
                    if args.enable_mtp:
                        # generate_mtp1 exposes accept/reject telemetry through
                        # generation_stats instead of a token callback.
                        result = generate_mtp1(
                            runtime,
                            prompt_ids,
                            max_tokens=max_tokens,
                            sampler=sampler,
                            seed=args.seed,
                        )
                    else:
                        result = generate_ar(
                            runtime,
                            prompt_ids,
                            max_tokens=max_tokens,
                            sampler=sampler,
                            seed=args.seed,
                            token_callback=token_callback,
                        )
                decoded_count = len(result.tokens)
                thread_cpu_finished = time.thread_time_ns()
                finished = time.perf_counter()
            elapsed = finished - started
            after = runtime.expert_streaming_snapshot()
            token_ids = [int(token) for token in result.tokens]
            if token_ids and (
                not decode_points
                or decode_points[-1]["completion_tokens"] != len(token_ids)
            ):
                # Stamp the final window with the generation end time, not a
                # timestamp taken after the full-slot snapshot walk above.
                decode_points.append(
                    {
                        "completion_tokens": len(token_ids),
                        "time": finished,
                        "streaming": after,
                    }
                )
            rolling_decode = []
            for left, right in zip(decode_points, decode_points[1:], strict=False):
                window_elapsed = right["time"] - left["time"]
                window_tokens = (
                    right["completion_tokens"] - left["completion_tokens"]
                )
                rolling_decode.append(
                    {
                        "from_completion_token": left["completion_tokens"],
                        "to_completion_token": right["completion_tokens"],
                        "decode_tokens": window_tokens,
                        "elapsed_seconds": window_elapsed,
                        "decode_tokens_per_second": (
                            window_tokens / window_elapsed
                            if window_elapsed > 0.0
                            else 0.0
                        ),
                        "streaming_before": left["streaming"],
                        "streaming_after": right["streaming"],
                    }
                )
            response_path = None
            if args.output_dir is not None:
                output_dir = args.output_dir.expanduser().resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                response_path = output_dir / (
                    f"{args.model_key}-{run_label}-repeat-{repeat}.md"
                )
                response_path.write_text(result.text + "\n", encoding="utf-8")
            row = {
                "repeat": repeat,
                "elapsed_seconds": elapsed,
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(token_ids),
                "completion_tokens_per_second": len(token_ids) / elapsed,
                "token_ids": token_ids,
                "text": result.text,
                "response_path": (
                    str(response_path) if response_path is not None else None
                ),
                "finish_reason": result.finish_reason,
                "rolling_decode": rolling_decode,
                "streaming_before": before,
                "streaming_after": after,
                "generation_stats": result.stats.to_dict(),
            }
            _attach_resource_report(
                row,
                resource_run,
                ssd_ceiling_gib_s=args.ssd_ceiling_gib_s,
                generation_thread_cpu_ns=thread_cpu_finished - thread_cpu_started,
                generation_elapsed_ns=int(elapsed * 1e9),
                final_completion_tokens=len(token_ids),
            )
            rows.append(row)
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
        "chat": args.chat,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "mtp": {
            "enabled": args.enable_mtp,
            "artifacts": (
                str(args.mtp_artifacts.expanduser().resolve())
                if args.mtp_artifacts is not None
                else None
            ),
            "precision": args.mtp_precision,
        },
        "generation_profile": args.generation_profile,
        "run_label": run_label,
        "concurrency": args.concurrency,
        "max_prefills_per_step": args.max_prefills_per_step,
        "generation": {
            "max_tokens": max_tokens,
            "documented_max_output_tokens": model_defaults["max_output_tokens"],
            "benchmark_context_limit": benchmark_context_limit,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
        "prompt_file": (
            str(args.prompt_file.expanduser().resolve())
            if args.prompt_file is not None
            else None
        ),
        "context_tokens": args.context_tokens,
        "prompt_style": args.prompt_style if args.context_tokens is not None else None,
        "prompt_metadata": prompt_metadata,
        "reset_between": args.reset_between,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "runs": rows,
    }
    if args.route_trace_json is not None:
        route_trace_json = args.route_trace_json.expanduser().resolve()
        route_trace_json.parent.mkdir(parents=True, exist_ok=True)
        route_trace_payload = {
            "schema": "mtplx-expert-route-trace-v1",
            "model_key": args.model_key,
            "manifest_sha256": runtime.expert_streaming.manifest.manifest_sha256,
            "entries": runtime.expert_streaming.route_trace(),
        }
        route_trace_json.write_text(
            json.dumps(route_trace_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload["route_trace_json"] = str(route_trace_json)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        output_json = args.output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
