#!/usr/bin/env python3
"""Benchmark real expert-Q2 trunks with their resident BF16 MTP heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


SCHEMA = "mtplx-q2-bf16-mtp-depth-matrix-v1"
DEFAULT_CONTEXTS = (1024, 2048)
OUTPUT_TOKENS = 128
WARMUP_TOKENS = 8
SEED = 0

MODEL_SPECS = {
    "hy3-q2": {
        "model_key": "hy3-expert-q2",
        "depths": (1, 2, 3, 4),
        "model_root": Path("~/.cache/huggingface/hy3-expert-only-mlx-q2"),
        "mtp_artifacts": Path("~/.cache/huggingface/hy3-mtp-layer80"),
        "prompt_tail": _ROOT
        / "benchmarks"
        / "fixtures"
        / "hy3-q2-benchmark-prompt.txt",
    },
    "glm52-q2": {
        "model_key": "glm52-expert-q2",
        "depths": (1, 2, 3, 4, 5),
        "model_root": Path("~/.cache/huggingface/glm52-expert-only-mlx-q2"),
        "mtp_artifacts": Path("~/.cache/huggingface/glm52-mtp-layer78"),
        "prompt_tail": _ROOT
        / "benchmarks"
        / "fixtures"
        / "glm52-q2-benchmark-prompt.txt",
    },
}

DEFAULT_RUNTIME_OPTIONS = {
    "memory_limit": "112GiB",
    "runtime_reserve": "12GiB",
    "expert_cache_limit": "64GiB",
    "max_live_kv_tokens": 4096,
    "cache_policy": "frequency",
    "cache_scope": "layer",
    "slot_layout": "component-banks",
    "transient_slots": 8,
    "read_chunk": "8MiB",
    "bypass_page_cache": True,
    "resource_telemetry": False,
}


class BenchmarkConfigurationError(ValueError):
    """Raised before model loading when the requested matrix is invalid."""


class BenchmarkGateError(RuntimeError):
    """Raised as soon as a correctness or metric gate fails."""


class RunnerAPIs:
    """Injectable API surface so unit tests never load MLX or model artifacts."""

    def __init__(
        self,
        *,
        load,
        config_factory,
        parse_memory_bytes,
        prompt_builder,
        sampler_factory,
        generate_ar,
        generate_mtpk,
        reset_peak_memory,
        get_peak_memory,
        synchronize,
    ) -> None:
        self.load = load
        self.config_factory = config_factory
        self.parse_memory_bytes = parse_memory_bytes
        self.prompt_builder = prompt_builder
        self.sampler_factory = sampler_factory
        self.generate_ar = generate_ar
        self.generate_mtpk = generate_mtpk
        self.reset_peak_memory = reset_peak_memory
        self.get_peak_memory = get_peak_memory
        self.synchronize = synchronize


def _default_apis() -> RunnerAPIs:
    # Keep all MLX-bearing imports behind execution so importing the CLI is cheap
    # and fake-runtime tests cannot allocate a model accidentally.
    import mlx.core as mx

    from mtplx.expert_runtime import ExpertStreamingConfig, parse_memory_bytes
    from mtplx.generation import generate_ar, generate_mtpk
    from mtplx.prefill_bench import _prompt_build_for_context
    from mtplx.runtime import load
    from mtplx.sampling import SamplerConfig

    return RunnerAPIs(
        load=load,
        config_factory=ExpertStreamingConfig,
        parse_memory_bytes=parse_memory_bytes,
        prompt_builder=_prompt_build_for_context,
        sampler_factory=SamplerConfig,
        generate_ar=generate_ar,
        generate_mtpk=generate_mtpk,
        reset_peak_memory=mx.reset_peak_memory,
        get_peak_memory=mx.get_peak_memory,
        synchronize=mx.synchronize,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _integer_csv(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for piece in value.replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            parsed = int(piece)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "value must be a comma-separated integer list"
            ) from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError("values must be positive")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must not repeat")
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        choices=tuple(MODEL_SPECS),
        help="Model to run; repeat for both. Defaults to both models.",
    )
    parser.add_argument(
        "--contexts",
        type=_integer_csv,
        default=DEFAULT_CONTEXTS,
        help="Comma-separated exact prompt sizes (default: 1024,2048).",
    )
    parser.add_argument(
        "--hy3-depths",
        type=_integer_csv,
        default=MODEL_SPECS["hy3-q2"]["depths"],
    )
    parser.add_argument(
        "--glm52-depths",
        type=_integer_csv,
        default=MODEL_SPECS["glm52-q2"]["depths"],
    )

    parser.add_argument(
        "--hy3-q2-model-root",
        type=Path,
        default=MODEL_SPECS["hy3-q2"]["model_root"],
    )
    parser.add_argument("--hy3-q2-manifest", type=Path)
    parser.add_argument(
        "--hy3-q2-mtp-artifacts",
        type=Path,
        default=MODEL_SPECS["hy3-q2"]["mtp_artifacts"],
    )
    parser.add_argument(
        "--hy3-q2-prompt-tail",
        type=Path,
        default=MODEL_SPECS["hy3-q2"]["prompt_tail"],
    )
    parser.add_argument(
        "--glm52-q2-model-root",
        type=Path,
        default=MODEL_SPECS["glm52-q2"]["model_root"],
    )
    parser.add_argument("--glm52-q2-manifest", type=Path)
    parser.add_argument(
        "--glm52-q2-mtp-artifacts",
        type=Path,
        default=MODEL_SPECS["glm52-q2"]["mtp_artifacts"],
    )
    parser.add_argument(
        "--glm52-q2-prompt-tail",
        type=Path,
        default=MODEL_SPECS["glm52-q2"]["prompt_tail"],
    )

    parser.add_argument("--memory-limit", default="112GiB")
    parser.add_argument("--runtime-reserve", default="12GiB")
    parser.add_argument("--expert-cache-limit", default="64GiB")
    parser.add_argument("--max-live-kv-tokens", type=_positive_int, default=4096)
    parser.add_argument(
        "--cache-policy", choices=("frequency", "lru"), default="frequency"
    )
    parser.add_argument("--cache-scope", choices=("layer", "global"), default="layer")
    parser.add_argument(
        "--slot-layout",
        choices=("direct-slots", "component-banks", "metal-mmap"),
        default="component-banks",
    )
    parser.add_argument("--transient-slots", type=_positive_int, default=8)
    parser.add_argument("--read-chunk", default="8MiB")
    parser.add_argument(
        "--f-nocache",
        dest="bypass_page_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resource-telemetry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable instrumented per-row resource snapshots. This changes hot-path "
            "cost, so these rows are diagnostic rather than headline throughput."
        ),
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def _expand(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _requests_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = list(args.models or MODEL_SPECS)
    if len(set(selected)) != len(selected):
        raise BenchmarkConfigurationError("--model values must not repeat")
    requests: list[dict[str, Any]] = []
    for model in selected:
        if model == "hy3-q2":
            model_root = args.hy3_q2_model_root
            manifest = args.hy3_q2_manifest
            mtp_artifacts = args.hy3_q2_mtp_artifacts
            prompt_tail = args.hy3_q2_prompt_tail
            depths = args.hy3_depths
        else:
            model_root = args.glm52_q2_model_root
            manifest = args.glm52_q2_manifest
            mtp_artifacts = args.glm52_q2_mtp_artifacts
            prompt_tail = args.glm52_q2_prompt_tail
            depths = args.glm52_depths
        model_root = _expand(model_root)
        requests.append(
            {
                "model": model,
                "model_root": model_root,
                "manifest": _expand(manifest or model_root / "expert-manifest.json"),
                "mtp_artifacts": _expand(mtp_artifacts),
                "prompt_tail": _expand(prompt_tail),
                "depths": tuple(depths),
            }
        )
    return requests


def _runtime_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "memory_limit": args.memory_limit,
        "runtime_reserve": args.runtime_reserve,
        "expert_cache_limit": args.expert_cache_limit,
        "max_live_kv_tokens": args.max_live_kv_tokens,
        "cache_policy": args.cache_policy,
        "cache_scope": args.cache_scope,
        "slot_layout": args.slot_layout,
        "transient_slots": args.transient_slots,
        "read_chunk": args.read_chunk,
        "bypass_page_cache": args.bypass_page_cache,
        "resource_telemetry": args.resource_telemetry,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _required_number(stats: Any, name: str) -> float:
    value = _field(stats, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkGateError(f"generation stats are missing numeric {name}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise BenchmarkGateError(f"generation stat {name} is not finite")
    return parsed


def _required_int(stats: Any, name: str) -> int:
    value = _field(stats, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkGateError(f"generation stats are missing integer {name}")
    return int(value)


def _optional_number(stats: Any, name: str, default: float = 0.0) -> float:
    value = _field(stats, name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkGateError(f"generation stat {name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise BenchmarkGateError(f"generation stat {name} is not finite")
    return parsed


def _optional_int(stats: Any, name: str, default: int = 0) -> int:
    value = _field(stats, name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkGateError(f"generation stat {name} must be an integer")
    return int(value)


def _integer_list(stats: Any, name: str) -> list[int]:
    value = _field(stats, name, [])
    if not isinstance(value, (list, tuple)):
        raise BenchmarkGateError(f"generation stat {name} must be a list")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise BenchmarkGateError(f"generation stat {name} has invalid counts")
        result.append(int(item))
    return result


def _number_list(stats: Any, name: str) -> list[float | None]:
    value = _field(stats, name, [])
    if not isinstance(value, (list, tuple)):
        raise BenchmarkGateError(f"generation stat {name} must be a list")
    result: list[float | None] = []
    for item in value:
        if item is None:
            result.append(None)
        elif isinstance(item, bool) or not isinstance(item, (int, float)):
            raise BenchmarkGateError(f"generation stat {name} has invalid values")
        else:
            parsed = float(item)
            if not math.isfinite(parsed):
                raise BenchmarkGateError(
                    f"generation stat {name} has non-finite values"
                )
            result.append(parsed)
    return result


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _prompt_identity(token_ids: Sequence[int], prompt_tail: str) -> dict[str, Any]:
    tokens = [int(token) for token in token_ids]
    encoded = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return {
        "token_count": len(tokens),
        "token_sha256": hashlib.sha256(encoded).hexdigest(),
        "tail_sha256": hashlib.sha256(prompt_tail.encode("utf-8")).hexdigest(),
    }


def _first_divergence(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if int(left_token) != int(right_token):
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def _guards_disabled(stats: Any) -> bool:
    if bool(_field(stats, "repetition_stop_triggered", False)):
        return False
    loop_guard = _field(stats, "loop_guard", {})
    if not isinstance(loop_guard, Mapping):
        return not bool(loop_guard)
    return not any(
        bool(loop_guard.get(key))
        for key in ("triggered", "stopped", "loop_detected", "abort")
    )


def _metrics(stats: Any, *, completion_tokens: int, depth: int) -> dict[str, Any]:
    generated_tokens = _required_int(stats, "generated_tokens")
    new_prefill_tokens = _required_int(stats, "new_prefill_tokens")
    prompt_eval_time = _required_number(stats, "prompt_eval_time_s")
    target_prefill_time = _required_number(stats, "prompt_target_prefill_time_s")
    decode_elapsed = _required_number(stats, "decode_elapsed_s")
    elapsed = _required_number(stats, "elapsed_s")
    if min(prompt_eval_time, target_prefill_time, decode_elapsed, elapsed) <= 0.0:
        raise BenchmarkGateError("generation timing denominators must be positive")
    if generated_tokens != completion_tokens:
        raise BenchmarkGateError(
            "generation stats count does not match the emitted token count"
        )

    accepted = _optional_int(stats, "accepted_drafts")
    evaluated = _optional_int(stats, "evaluated_drafts")
    drafted = _optional_int(stats, "drafted_tokens")
    verify_calls = _optional_int(stats, "verify_calls")
    fully_accepted = _optional_int(stats, "fully_accepted_verify_calls")
    accepted_by_depth = _integer_list(stats, "accepted_by_depth")
    evaluated_by_depth = _integer_list(stats, "evaluated_by_depth")
    drafted_by_depth = _integer_list(stats, "drafted_by_depth")
    mean_probability = _number_list(stats, "mean_accept_probability_by_depth")
    peak_memory_bytes = _required_int(stats, "peak_memory_bytes")
    if peak_memory_bytes < 0:
        raise BenchmarkGateError("peak_memory_bytes must be non-negative")

    if depth:
        arrays = (accepted_by_depth, evaluated_by_depth, drafted_by_depth)
        if any(len(values) != depth for values in arrays):
            raise BenchmarkGateError(
                f"depth {depth} row has the wrong number of acceptance counters"
            )
        if len(mean_probability) != depth:
            raise BenchmarkGateError(
                f"depth {depth} row has the wrong number of probability counters"
            )
        if any(
            not (accepted_count <= evaluated_count <= drafted_count)
            for accepted_count, evaluated_count, drafted_count in zip(
                accepted_by_depth,
                evaluated_by_depth,
                drafted_by_depth,
                strict=True,
            )
        ):
            raise BenchmarkGateError(
                "per-depth counters must satisfy accepted <= evaluated <= drafted"
            )
        if sum(accepted_by_depth) != accepted:
            raise BenchmarkGateError(
                "accepted_by_depth does not sum to accepted_drafts"
            )
        if sum(evaluated_by_depth) != evaluated:
            raise BenchmarkGateError(
                "evaluated_by_depth does not sum to evaluated_drafts"
            )
        if sum(drafted_by_depth) != drafted:
            raise BenchmarkGateError("drafted_by_depth does not sum to drafted_tokens")
        if verify_calls <= 0:
            raise BenchmarkGateError("MTP observations must report verification calls")
        if fully_accepted < 0 or fully_accepted > verify_calls:
            raise BenchmarkGateError(
                "fully accepted verification calls exceed total verification calls"
            )

    acceptance_by_depth = []
    for index in range(depth):
        mean = mean_probability[index] if index < len(mean_probability) else None
        acceptance_by_depth.append(
            {
                "depth": index + 1,
                "drafted": drafted_by_depth[index],
                "evaluated": evaluated_by_depth[index],
                "accepted": accepted_by_depth[index],
                "conditional_hit_rate": _ratio(
                    accepted_by_depth[index], evaluated_by_depth[index]
                ),
                "cumulative_accepted_drafted_yield": _ratio(
                    accepted_by_depth[index], drafted_by_depth[index]
                ),
                "mean_accept_probability": mean,
            }
        )

    return {
        "generated_tokens": generated_tokens,
        "elapsed_s": elapsed,
        "decode_elapsed_s": decode_elapsed,
        "new_prefill_tokens": new_prefill_tokens,
        "prompt_eval_time_s": prompt_eval_time,
        "ingestion_tok_s": new_prefill_tokens / prompt_eval_time,
        "prompt_target_prefill_time_s": target_prefill_time,
        "prompt_target_prefill_tok_s": new_prefill_tokens / target_prefill_time,
        "prompt_mtp_history_time_s": _optional_number(
            stats, "prompt_mtp_history_time_s"
        ),
        "prompt_mtp_history_tokens": _optional_int(stats, "prompt_mtp_history_tokens"),
        "decode_tok_s": completion_tokens / decode_elapsed,
        "end_to_end_tok_s": completion_tokens / elapsed,
        "reported_decode_tok_s": _optional_number(stats, "decode_tok_s"),
        "reported_prompt_target_prefill_tok_s": _optional_number(
            stats, "prompt_target_prefill_tok_s"
        ),
        "peak_memory_bytes": peak_memory_bytes,
        "accepted_drafts": accepted,
        "evaluated_drafts": evaluated,
        "drafted_tokens": drafted,
        "conditional_hit_rate": _ratio(accepted, evaluated),
        "cumulative_accepted_drafted_yield": _ratio(accepted, drafted),
        "verify_calls": verify_calls,
        "accepted_per_verify": _ratio(accepted, verify_calls),
        "fully_accepted_verify_calls": fully_accepted,
        "fully_accepted_verify_ratio": _ratio(fully_accepted, verify_calls),
        "acceptance_by_depth": acceptance_by_depth,
    }


def _reset_expert_streaming(runtime: Any) -> None:
    streaming = getattr(runtime, "expert_streaming", None)
    if streaming is None:
        raise BenchmarkGateError("Q2 matrix requires an expert-streamed runtime")
    reset_counters = getattr(streaming, "reset_counters", None)
    if callable(reset_counters):
        reset_counters()
        return
    reset = getattr(streaming, "reset", None)
    if not callable(reset):
        raise BenchmarkGateError("expert streaming runtime has no reset API")
    # The current public reset API clears row-local counters together with the
    # expert-cache state, which is the required cold observation boundary.
    reset()


def _streaming_counters(runtime: Any) -> dict[str, Any] | None:
    snapshot_fn = getattr(runtime, "expert_streaming_snapshot", None)
    if not callable(snapshot_fn):
        return None
    snapshot = snapshot_fn()
    if not isinstance(snapshot, Mapping):
        return None
    cache = snapshot.get("cache")
    return _jsonable(cache) if isinstance(cache, Mapping) else None


def _resource_telemetry(runtime: Any) -> dict[str, Any] | None:
    snapshot_fn = getattr(runtime, "expert_resource_telemetry_snapshot", None)
    if not callable(snapshot_fn):
        return None
    snapshot = snapshot_fn()
    return _jsonable(snapshot) if isinstance(snapshot, Mapping) else None


def _numeric_delta(before: Any, after: Any) -> Any:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        result = {}
        for key in before.keys() & after.keys():
            delta = _numeric_delta(before[key], after[key])
            if delta is not None:
                result[str(key)] = delta
        return result
    if (
        isinstance(before, (int, float))
        and not isinstance(before, bool)
        and isinstance(after, (int, float))
        and not isinstance(after, bool)
    ):
        return after - before
    return None


def _result_tokens(result: Any) -> list[int]:
    value = _field(result, "tokens")
    if not isinstance(value, (list, tuple)):
        raise BenchmarkGateError("generation result tokens must be a sequence")
    return [int(token) for token in value]


def _run_observation(
    *,
    apis: RunnerAPIs,
    runtime: Any,
    model: str,
    context_tokens: int,
    prompt_ids: tuple[int, ...],
    prompt_identity: Mapping[str, Any],
    sampler: Any,
    depth: int,
    position: int,
    max_tokens: int,
    resource_telemetry_enabled: bool,
    ar_tokens: Sequence[int] | None,
    ar_finish_reason: str | None,
) -> tuple[dict[str, Any], list[int], str | None]:
    _reset_expert_streaming(runtime)
    resource_before = (
        _resource_telemetry(runtime) if resource_telemetry_enabled else None
    )
    apis.synchronize()
    apis.reset_peak_memory()
    prompt_copy = list(prompt_ids)
    admission_fn = getattr(runtime, "admit_kv_tokens", None)
    admission = (
        admission_fn(len(prompt_ids) + max_tokens)
        if callable(admission_fn)
        else nullcontext()
    )
    common = {
        "max_tokens": max_tokens,
        "sampler": sampler,
        "seed": SEED,
        "stop_token_ids": set(),
        "repetition_stop": False,
        "loop_guard": False,
    }
    with admission:
        if depth == 0:
            result = apis.generate_ar(runtime, prompt_copy, **common)
        else:
            result = apis.generate_mtpk(
                runtime,
                prompt_copy,
                speculative_depth=depth,
                mtp_cache_policy="persistent",
                mtp_history_policy="committed",
                **common,
            )
    apis.synchronize()
    resource_after = (
        _resource_telemetry(runtime) if resource_telemetry_enabled else None
    )

    if tuple(int(token) for token in prompt_copy) != prompt_ids:
        raise BenchmarkGateError("generation mutated the exact benchmark prompt")
    tokens = _result_tokens(result)
    finish_reason = _field(result, "finish_reason")
    stats = _field(result, "stats")
    if stats is None:
        raise BenchmarkGateError("generation result is missing stats")
    if len(tokens) != max_tokens:
        raise BenchmarkGateError(
            f"{model} d{depth} emitted {len(tokens)} tokens; expected exactly "
            f"{max_tokens}"
        )
    if finish_reason != "length":
        raise BenchmarkGateError(
            f"{model} d{depth} finished as {finish_reason!r}; expected 'length'"
        )

    requested_depth = _optional_int(stats, "requested_speculative_depth")
    effective_depth = _optional_int(stats, "speculative_depth")
    history_policy = _field(stats, "mtp_history_policy", "none")
    ar_token_parity = depth == 0 or tokens == list(ar_tokens or [])
    ar_finish_parity = depth == 0 or finish_reason == ar_finish_reason
    requested_exact = requested_depth == depth
    effective_exact = effective_depth == depth
    committed_history = depth == 0 or history_policy == "committed"
    guards_disabled = _guards_disabled(stats)
    if not ar_token_parity:
        divergence = _first_divergence(tokens, ar_tokens or [])
        raise BenchmarkGateError(
            f"{model} d{depth} diverged from AR at output token {divergence}"
        )
    if not ar_finish_parity:
        raise BenchmarkGateError(f"{model} d{depth} finish reason diverged from AR")
    if not requested_exact or not effective_exact:
        raise BenchmarkGateError(
            f"{model} d{depth} reported requested/effective depth "
            f"{requested_depth}/{effective_depth}"
        )
    if not committed_history:
        raise BenchmarkGateError(f"{model} d{depth} did not use committed MTP history")
    if not guards_disabled:
        raise BenchmarkGateError(f"{model} d{depth} triggered a generation guard")

    row = {
        "model": model,
        "context_tokens": context_tokens,
        "prompt_tokens": len(prompt_ids),
        "prompt_identity": dict(prompt_identity),
        "replicate": 1,
        "cell": "ar" if depth == 0 else f"d{depth}",
        "cell_position": position,
        "requested_depth": depth,
        "effective_depth": effective_depth,
        "mtp_cache_policy": None if depth == 0 else "persistent",
        "mtp_history_policy": None if depth == 0 else "committed",
        "finish_reason": finish_reason,
        "token_ids": tokens,
        **_metrics(stats, completion_tokens=len(tokens), depth=depth),
        "expert_streaming_counters": _streaming_counters(runtime),
        "expert_resource_telemetry": (
            {
                "before": resource_before,
                "after": resource_after,
                "numeric_delta": _numeric_delta(resource_before, resource_after),
            }
            if resource_telemetry_enabled
            else None
        ),
        "gates": {
            "prompt_length_exact": len(prompt_ids) == context_tokens,
            "new_prefill_tokens_exact": _field(stats, "new_prefill_tokens")
            == len(prompt_ids),
            "output_tokens_exact": len(tokens) == max_tokens,
            "generated_count_consistent": _field(stats, "generated_tokens")
            == len(tokens),
            "length_finish": finish_reason == "length",
            "ar_token_parity": ar_token_parity,
            "ar_finish_reason_parity": ar_finish_parity,
            "requested_depth_exact": requested_exact,
            "effective_depth_exact": effective_exact,
            "committed_history": committed_history,
            "guards_disabled": guards_disabled,
        },
    }
    if not row["gates"]["new_prefill_tokens_exact"]:
        raise BenchmarkGateError(
            f"{model} d{depth} did not ingest the full exact prompt"
        )
    return row, tokens, finish_reason


def _runtime_config(
    apis: RunnerAPIs,
    model_key: str,
    options: Mapping[str, Any],
) -> Any:
    return apis.config_factory(
        model_key=model_key,
        memory_limit_bytes=apis.parse_memory_bytes(options["memory_limit"]),
        max_live_kv_tokens=int(options["max_live_kv_tokens"]),
        runtime_reserve_bytes=apis.parse_memory_bytes(options["runtime_reserve"]),
        expert_cache_limit_bytes=apis.parse_memory_bytes(options["expert_cache_limit"]),
        cache_policy=str(options["cache_policy"]),
        cache_scope=str(options["cache_scope"]),
        slot_layout=str(options["slot_layout"]),
        transient_slots=int(options["transient_slots"]),
        max_read_chunk_bytes=apis.parse_memory_bytes(options["read_chunk"]),
        bypass_page_cache=bool(options["bypass_page_cache"]),
        resource_telemetry=bool(options["resource_telemetry"]),
    )


def _normalized_request(request: Mapping[str, Any]) -> dict[str, Any]:
    model = request.get("model")
    if model not in MODEL_SPECS:
        raise BenchmarkConfigurationError(f"unsupported model {model!r}")
    spec = MODEL_SPECS[model]
    depths = tuple(int(value) for value in request.get("depths", spec["depths"]))
    if not depths or len(set(depths)) != len(depths):
        raise BenchmarkConfigurationError(f"{model} depths must be unique and nonempty")
    permitted = set(spec["depths"])
    if any(depth not in permitted for depth in depths):
        raise BenchmarkConfigurationError(
            f"{model} depths must be selected from {tuple(spec['depths'])}"
        )
    required_paths = ("model_root", "manifest", "mtp_artifacts")
    missing = [name for name in required_paths if request.get(name) is None]
    if missing:
        raise BenchmarkConfigurationError(
            f"{model} is missing required paths: {', '.join(missing)}"
        )
    if "prompt_tail_text" in request:
        prompt_tail_text = str(request["prompt_tail_text"])
        prompt_tail_path = request.get("prompt_tail")
    else:
        prompt_tail_path = request.get("prompt_tail", spec["prompt_tail"])
        prompt_tail_text = _expand(prompt_tail_path).read_text(encoding="utf-8")
    if not prompt_tail_text:
        raise BenchmarkConfigurationError(f"{model} prompt tail must not be empty")
    return {
        "model": model,
        "model_key": spec["model_key"],
        "depths": depths,
        "model_root": _expand(request["model_root"]),
        "manifest": _expand(request["manifest"]),
        "mtp_artifacts": _expand(request["mtp_artifacts"]),
        "prompt_tail": _expand(prompt_tail_path) if prompt_tail_path else None,
        "prompt_tail_text": prompt_tail_text,
    }


def run_depth_matrix(
    model_requests: Sequence[Mapping[str, Any]],
    *,
    contexts: Sequence[int] = DEFAULT_CONTEXTS,
    runtime_options: Mapping[str, Any] | None = None,
    apis: RunnerAPIs | None = None,
) -> dict[str, Any]:
    """Run one AR/depth matrix per model, loading each selected model once."""

    if not model_requests:
        raise BenchmarkConfigurationError("at least one model must be selected")
    normalized = [_normalized_request(request) for request in model_requests]
    names = [request["model"] for request in normalized]
    if len(set(names)) != len(names):
        raise BenchmarkConfigurationError("models must not repeat")
    context_values = tuple(int(value) for value in contexts)
    if (
        not context_values
        or any(value <= 0 for value in context_values)
        or len(set(context_values)) != len(context_values)
    ):
        raise BenchmarkConfigurationError("contexts must be unique positive integers")
    options = {**DEFAULT_RUNTIME_OPTIONS, **dict(runtime_options or {})}
    if max(context_values) + OUTPUT_TOKENS > int(options["max_live_kv_tokens"]):
        raise BenchmarkConfigurationError(
            "context plus 128 output tokens exceeds max_live_kv_tokens"
        )
    apis = apis or _default_apis()
    sampler = apis.sampler_factory(temperature=0.0, top_p=1.0, top_k=1)

    models = []
    for request in normalized:
        config = _runtime_config(apis, request["model_key"], options)
        runtime = apis.load(
            request["model_root"],
            mtp=True,
            expert_streaming_config=config,
            expert_manifest=request["manifest"],
            mtp_artifacts=request["mtp_artifacts"],
            mtp_precision="bf16",
        )
        try:
            if getattr(runtime, "mtp_enabled", False) is not True:
                raise BenchmarkGateError(
                    f"{request['model']} runtime did not load its BF16 MTP head"
                )
            apis.synchronize()
            load_peak_memory_bytes = apis.get_peak_memory()
            if (
                isinstance(load_peak_memory_bytes, bool)
                or not isinstance(load_peak_memory_bytes, int)
                or load_peak_memory_bytes < 0
            ):
                raise BenchmarkGateError("MLX load peak must be a non-negative integer")
            hard_peak_memory_bytes = int(load_peak_memory_bytes)
            observations = []
            prompts = []
            discarded_warmup_count = 0
            for context_tokens in context_values:
                prompt = apis.prompt_builder(
                    runtime.tokenizer,
                    context_tokens,
                    prompt_style="coding-agent",
                    prompt_tail=request["prompt_tail_text"],
                    prompt_format="raw",
                    enable_thinking=False,
                )
                prompt_ids = tuple(int(token) for token in prompt.token_ids)
                if len(prompt_ids) != context_tokens:
                    raise BenchmarkGateError(
                        f"{request['model']} prompt builder returned {len(prompt_ids)} "
                        f"tokens for requested context {context_tokens}"
                    )
                metadata = _jsonable(getattr(prompt, "metadata", {}))
                if (
                    isinstance(metadata, Mapping)
                    and metadata.get("prompt_tail_preserved") is False
                ):
                    raise BenchmarkGateError(
                        f"{request['model']} prompt builder did not preserve the tail"
                    )
                identity = _prompt_identity(prompt_ids, request["prompt_tail_text"])
                prompts.append(
                    {
                        "context_tokens": context_tokens,
                        **identity,
                        "builder_metadata": metadata,
                    }
                )
                _warmup_ar_row, warmup_ar_tokens, warmup_ar_finish = _run_observation(
                    apis=apis,
                    runtime=runtime,
                    model=request["model"],
                    context_tokens=context_tokens,
                    prompt_ids=prompt_ids,
                    prompt_identity=identity,
                    sampler=sampler,
                    depth=0,
                    position=1,
                    max_tokens=WARMUP_TOKENS,
                    resource_telemetry_enabled=bool(options["resource_telemetry"]),
                    ar_tokens=None,
                    ar_finish_reason=None,
                )
                hard_peak_memory_bytes = max(
                    hard_peak_memory_bytes,
                    int(_warmup_ar_row["peak_memory_bytes"]),
                )
                discarded_warmup_count += 1
                ar_row, ar_tokens, ar_finish = _run_observation(
                    apis=apis,
                    runtime=runtime,
                    model=request["model"],
                    context_tokens=context_tokens,
                    prompt_ids=prompt_ids,
                    prompt_identity=identity,
                    sampler=sampler,
                    depth=0,
                    position=1,
                    max_tokens=OUTPUT_TOKENS,
                    resource_telemetry_enabled=bool(options["resource_telemetry"]),
                    ar_tokens=None,
                    ar_finish_reason=None,
                )
                hard_peak_memory_bytes = max(
                    hard_peak_memory_bytes,
                    int(ar_row["peak_memory_bytes"]),
                )
                pair_id = f"{request['model']}-c{context_tokens}-ar"
                ar_row["pair_id"] = pair_id
                observations.append(ar_row)
                for position, depth in enumerate(request["depths"], start=2):
                    _warmup_row, _warmup_tokens, _warmup_finish = _run_observation(
                        apis=apis,
                        runtime=runtime,
                        model=request["model"],
                        context_tokens=context_tokens,
                        prompt_ids=prompt_ids,
                        prompt_identity=identity,
                        sampler=sampler,
                        depth=depth,
                        position=position,
                        max_tokens=WARMUP_TOKENS,
                        resource_telemetry_enabled=bool(options["resource_telemetry"]),
                        ar_tokens=warmup_ar_tokens,
                        ar_finish_reason=warmup_ar_finish,
                    )
                    hard_peak_memory_bytes = max(
                        hard_peak_memory_bytes,
                        int(_warmup_row["peak_memory_bytes"]),
                    )
                    discarded_warmup_count += 1
                    row, _tokens, _finish = _run_observation(
                        apis=apis,
                        runtime=runtime,
                        model=request["model"],
                        context_tokens=context_tokens,
                        prompt_ids=prompt_ids,
                        prompt_identity=identity,
                        sampler=sampler,
                        depth=depth,
                        position=position,
                        max_tokens=OUTPUT_TOKENS,
                        resource_telemetry_enabled=bool(options["resource_telemetry"]),
                        ar_tokens=ar_tokens,
                        ar_finish_reason=ar_finish,
                    )
                    hard_peak_memory_bytes = max(
                        hard_peak_memory_bytes,
                        int(row["peak_memory_bytes"]),
                    )
                    row["pair_id"] = pair_id
                    observations.append(row)
            models.append(
                {
                    "model": request["model"],
                    "model_key": request["model_key"],
                    "model_root": str(request["model_root"]),
                    "manifest": str(request["manifest"]),
                    "mtp_artifacts": str(request["mtp_artifacts"]),
                    "mtp_precision": "bf16",
                    "depths": list(request["depths"]),
                    "load_count": 1,
                    "measurement_lane": (
                        "diagnostic-resource-instrumented"
                        if options["resource_telemetry"]
                        else "headline-uninstrumented"
                    ),
                    "load_peak_memory_bytes": load_peak_memory_bytes,
                    "hard_peak_memory_bytes": hard_peak_memory_bytes,
                    "discarded_warmup_count": discarded_warmup_count,
                    "warmup_output_tokens": WARMUP_TOKENS,
                    "runtime_config": _jsonable(config),
                    "prompts": prompts,
                    "observations": observations,
                    "passed": True,
                }
            )
        finally:
            close = getattr(runtime, "close", None)
            if callable(close):
                close()

    return {
        "schema": SCHEMA,
        "passed": True,
        "configuration": {
            "measurement_lane": (
                "diagnostic-resource-instrumented"
                if options["resource_telemetry"]
                else "headline-uninstrumented"
            ),
            "contexts": list(context_values),
            "output_tokens": OUTPUT_TOKENS,
            "warmup_output_tokens": WARMUP_TOKENS,
            "retained_replicates": 1,
            "sampler": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "seed": SEED,
            },
            "generation": {
                "ar_api": "generate_ar",
                "mtp_api": "generate_mtpk",
                "stop_token_ids": [],
                "repetition_stop": False,
                "loop_guard": False,
                "mtp_cache_policy": "persistent",
                "mtp_history_policy": "committed",
            },
            "runtime": _jsonable(options),
        },
        "models": models,
    }


def _write_json_atomic(path: Path, rendered: str) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None, *, apis: RunnerAPIs | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run_depth_matrix(
            _requests_from_args(args),
            contexts=args.contexts,
            runtime_options=_runtime_options_from_args(args),
            apis=apis,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "passed": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 1
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output_json is not None:
        _write_json_atomic(args.output_json, rendered)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
