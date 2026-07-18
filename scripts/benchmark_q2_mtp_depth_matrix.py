#!/usr/bin/env python3
"""Benchmark real expert-Q2 trunks with their resident BF16 MTP heads."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.benchmarks.resource_telemetry import (  # noqa: E402
    PowermetricsCollector,
    ResourceTelemetrySampler,
)
from mtplx.optimization_profiles import (  # noqa: E402
    profile_conflict_warnings,
)


SCHEMA = "mtplx-q2-bf16-mtp-depth-matrix-v3"
DEFAULT_CONTEXTS = (1024, 2048)
OUTPUT_TOKENS = 128
WARMUP_TOKENS = 8
SEED = 0
VERIFY_STRATEGIES = ("batched", "capture_commit")
COMPILED_VERIFY_MODES = ("off", "parity", "on")
DRAFT_CORES = ("stock", "device-k", "device-d2")

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_FIXED_DEPTH_ENV_KEYS = (
    "MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS",
    "MTPLX_LATE_DEPTH_BEFORE",
    "MTPLX_LATE_DEPTH_AFTER",
)

MODEL_SPECS = {
    "hy3-q2": {
        "model_key": "hy3-expert-q2",
        "depths": (1, 2, 3, 4, 5, 6, 7),
        "model_root": Path("~/.cache/huggingface/hy3-expert-only-mlx-q2"),
        "mtp_artifacts": Path("~/.cache/huggingface/hy3-mtp-layer80"),
        "prompt_tail": None,
    },
    "glm52-q2": {
        "model_key": "glm52-expert-q2",
        "depths": (1, 2, 3, 4, 5),
        "model_root": Path("~/.cache/huggingface/glm52-expert-only-mlx-q2"),
        "mtp_artifacts": Path("~/.cache/huggingface/glm52-mtp-layer78"),
        "prompt_tail": None,
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
    "q2_expert_kernel": "stock",
    "hy3_router_kernel": "mpp-r1-fused-r2",
    "hy3_router_sigmoid": "precise",
    "hy3_mtp_shared_kernel": "stock",
    "hy3_mtp_shared_kernel_depth": 3,
    "deferred_pin_release": True,
    "island_layers": "",
    "island_layer_count": None,
    "resident_quant": None,
    "kv_quant": None,
    "miss_shadow": None,
    "miss_shadow_layers": None,
    "expert_integrity": "per-read",
    "prefetch_slots": 0,
    "split_route_release": "fenced",
    "mmap_island_layers": "",
    "banked_manifest": "",
    "banked_codec": "none",
    "streamed_codec": "none",
    "streamed_codec_manifest": "",
    "streamed_codec_verify": True,
    "mmap_island_wired": True,
    "read_chunk": "8MiB",
    "bypass_page_cache": True,
    "resource_telemetry": False,
    "resource_sample_interval": 0.25,
    "resource_max_samples": 4096,
    "ssd_ceiling_gib_s": None,
    "powermetrics": False,
    "powermetrics_interval_ms": 250,
}


class BenchmarkConfigurationError(ValueError):
    """Raised before model loading when the requested matrix is invalid."""


class BenchmarkGateError(RuntimeError):
    """Raised as soon as a correctness or metric gate fails."""

    def __init__(self, message: str, *, evidence: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.evidence = dict(evidence) if evidence is not None else None


def _generation_environment() -> dict[str, str | None]:
    sustained_prefill = os.environ.get("MTPLX_SUSTAINED_PREFILL")
    if (sustained_prefill or "").strip().lower() not in _TRUTHY_ENV_VALUES:
        raise BenchmarkConfigurationError(
            "MTPLX_SUSTAINED_PREFILL=1 is required for the retained matrix"
        )
    for name in _FIXED_DEPTH_ENV_KEYS:
        value = os.environ.get(name)
        if value is not None and value.strip():
            raise BenchmarkConfigurationError(f"fixed-depth matrix forbids {name}")
    return {
        "MTPLX_SUSTAINED_PREFILL": sustained_prefill,
        "MTPLX_SUSTAINED_PREFILL_LAYOUT": os.environ.get(
            "MTPLX_SUSTAINED_PREFILL_LAYOUT"
        ),
        **{name: os.environ.get(name) for name in _FIXED_DEPTH_ENV_KEYS},
    }


def _compiled_verify_environment_mode() -> str:
    raw = (os.environ.get("MTPLX_COMPILED_VERIFY") or "off").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return "off"
    if raw in _TRUTHY_ENV_VALUES:
        return "on"
    return raw


Checkpoint = Callable[[Mapping[str, Any]], None]


def _emit_checkpoint(payload: Mapping[str, Any], checkpoint: Checkpoint | None) -> None:
    if checkpoint is not None:
        checkpoint(copy.deepcopy(dict(payload)))


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
        resource_sampler_factory=ResourceTelemetrySampler,
        powermetrics_factory=PowermetricsCollector,
        thread_time_ns=time.thread_time_ns,
        monotonic_ns=time.monotonic_ns,
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
        self.resource_sampler_factory = resource_sampler_factory
        self.powermetrics_factory = powermetrics_factory
        self.thread_time_ns = thread_time_ns
        self.monotonic_ns = monotonic_ns


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
        "--output-tokens",
        type=_positive_int,
        default=OUTPUT_TOKENS,
        help="Exact retained output length per matrix row (default: 128).",
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
    parser.add_argument(
        "--q2-expert-kernel",
        choices=("stock", "nax", "fused", "fused-nax"),
        default="stock",
        help="Issue 51 Q2 expert execution arm (default: stock).",
    )
    parser.add_argument(
        "--hy3-router-kernel",
        choices=(
            "stock",
            "steel-r1-fused-r2",
            "mpp-r1-fused-r2",
            "mpp-fp32-splitk-r1-fused-r2",
            "mpp-r1-last-arrival-fused-r2",
            "mpp-row-owned-fused",
        ),
        default="mpp-r1-fused-r2",
        help=(
            "Issue 51 authoritative Hy3 router arithmetic mode "
            "(default: mpp-r1-fused-r2)."
        ),
    )
    parser.add_argument(
        "--deferred-pin-release",
        dest="deferred_pin_release",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "All-hit pin release defers to the next generation eval "
            "(C3-promoted; --no-deferred-pin-release restores the fence)."
        ),
    )
    parser.add_argument(
        "--hy3-router-sigmoid",
        choices=("precise", "fast"),
        default="precise",
        help=(
            "Row-owned router finalizer exponential "
            "(fast is a selectable experiment; default: precise)."
        ),
    )
    parser.add_argument(
        "--hy3-mtp-shared-kernel",
        choices=("stock", "metal-exact"),
        default="stock",
        help="Issue 69 Hy3 MTP shared-MLP execution arm (default: stock).",
    )
    parser.add_argument(
        "--hy3-mtp-shared-kernel-depth",
        type=_positive_int,
        default=3,
        help="Only fixed MTP depth that selects metal-exact (default: 3).",
    )
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
    parser.add_argument(
        "--resource-sample-interval",
        type=float,
        default=0.25,
        help="Decode resource-sampling interval in seconds (default: 0.25).",
    )
    parser.add_argument(
        "--resource-max-samples",
        type=_positive_int,
        default=4096,
        help="Maximum retained decode resource samples per row (default: 4096).",
    )
    parser.add_argument(
        "--ssd-ceiling-gib-s",
        type=float,
        help=(
            "Measured uncached SSD ceiling used only to calculate utilization; "
            "requires --resource-telemetry and --f-nocache."
        ),
    )
    parser.add_argument(
        "--powermetrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Collect non-interactive process CPU/GPU evidence during decode. "
            "Unavailable authorization is recorded, never treated as zero use."
        ),
    )
    parser.add_argument(
        "--powermetrics-interval-ms",
        type=_positive_int,
        default=250,
    )
    parser.add_argument(
        "--mtp-disabled-baseline",
        action="store_true",
        help=(
            "Load the streamed Q2 target without an MTP artifact and run only "
            "the AR warmup/retained baseline rows."
        ),
    )
    parser.add_argument(
        "--verify-strategy",
        choices=VERIFY_STRATEGIES,
        default="batched",
    )
    parser.add_argument(
        "--compiled-verify-mode",
        choices=COMPILED_VERIFY_MODES,
        default="off",
    )
    parser.add_argument(
        "--draft-core",
        choices=DRAFT_CORES,
        default="stock",
        help=(
            "MTP draft execution core. device-k is the D1...D7 fixed compiled "
            "committed-history lane (primary K3/M4)."
        ),
    )
    parser.add_argument(
        "--trace-routes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--island-layers",
        default="",
        help=(
            "Dense island layers holding every expert resident (issue #63 "
            "C5), e.g. '1-38,60-79'. Empty disables islands."
        ),
    )
    parser.add_argument(
        "--mmap-island-layers",
        default="",
        help=(
            "Banked mmap island layers served from a repacked component-major "
            "sidecar via the page cache (issue #51 C6), e.g. '39-59'. "
            "Requires --banked-manifest. Empty disables the mmap band."
        ),
    )
    parser.add_argument(
        "--banked-manifest",
        default="",
        help="Path to the banked expert sidecar manifest JSON (C6).",
    )
    parser.add_argument(
        "--banked-codec",
        default="none",
        choices=["none", "rans32x-v1"],
        help=(
            "Lossless codec of the banked sidecar. 'rans32x-v1' stores each "
            "component bank as a static order-0 byte-rANS container that the "
            "in-kernel decoder rebuilds at load (issue #51 C7)."
        ),
    )
    parser.add_argument(
        "--streamed-codec",
        default="none",
        choices=["none", "rans32x-v1"],
        help=(
            "Lossless codec of the streamed miss-read sidecar (issue #113). "
            "'rans32x-v1' reads per-record rANS containers and decodes them "
            "in-kernel on the miss path."
        ),
    )
    parser.add_argument(
        "--streamed-codec-manifest",
        default="",
        help="Path to the streamed codec sidecar manifest JSON.",
    )
    parser.add_argument(
        "--streamed-codec-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify decoded record payload hashes on the streamed codec path.",
    )
    parser.add_argument(
        "--mtp-precision",
        default="bf16",
        choices=["bf16", "q4"],
        help=(
            "Streamed external MTP head precision. 'bf16' is the bit-exact "
            "default; 'q4' loads the quantized sibling artifact (speculative-"
            "only head: costs acceptance, never correctness). Point the "
            "matching --*-mtp-artifacts flag at the artifact directory of the "
            "chosen precision."
        ),
    )
    parser.add_argument(
        "--mmap-island-wired",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Register banked mmap regions in MLX's residency set (permanently "
            "resident, counted on the plan's fixed side). --no-mmap-island-wired "
            "leaves residency to the pager for bands larger than RAM."
        ),
    )
    parser.add_argument(
        "--island-layer-count",
        type=int,
        default=None,
        help=(
            "Pin the N worst-streaming layers as dense islands using the "
            "model's measured pin order. Mutually exclusive with "
            "--island-layers. The experts cache stays independently sized "
            "via --expert-cache-limit."
        ),
    )
    parser.add_argument(
        "--resident-quant",
        choices=("q8", "q4"),
        default=None,
        help=(
            "Quantize the resident attention/shared/dense-MLP Linears at "
            "load time (group 64 affine). Routers, embeddings, and the LM "
            "head keep their loaded precision. No artifact rebuild."
        ),
    )
    parser.add_argument(
        "--split-route-release",
        choices=("fenced", "deferred"),
        default="fenced",
        help=(
            "Split-route slot release. fenced: per-part synchronous GPU "
            "fences (3-6 pipeline drains per streamed layer). deferred: "
            "lease bookkeeping replays at the next generation eval "
            "(requires deferred pin release; decode only)."
        ),
    )
    parser.add_argument(
        "--expert-integrity",
        choices=("per-read", "at-open", "headers-only"),
        default="per-read",
        help=(
            "Expert record integrity mode. per-read: sha256 every record "
            "read (measured ~10 ms/token of decode CPU at 27%% miss). "
            "at-open: verify the sidecar hash once at load, no per-read "
            "hashing. headers-only: artifact header checks only."
        ),
    )
    parser.add_argument(
        "--kv-quant",
        choices=("q8", "q4"),
        default=None,
        help=(
            "Quantized trunk KV cache (group 64 affine). The MTP draft "
            "cache stays BF16; use --verify-strategy batched — "
            "capture_commit adapts dense trunk KV state."
        ),
    )
    parser.add_argument(
        "--miss-shadow-layers",
        type=int,
        default=None,
        help=(
            "Cap shadow coverage to the N worst streamed layers (pin-order-"
            "first: flattest routing, lowest exact-cache hit). Requires "
            "--miss-shadow. Default: all streamed layers."
        ),
    )
    parser.add_argument(
        "--miss-shadow",
        choices=("b1", "t158"),
        default=None,
        help=(
            "Resident low-precision shadow banks for every streamed layer "
            "(built at load from the sidecar, no artifact rebuild). Decode "
            "cache misses are served instantly from the shadow instead of "
            "waiting on SSD; the exact tier warms asynchronously. Output "
            "is a quality tier, not bitwise — read ar_comparison / "
            "token-divergence for the quality report. Requires cache-scope "
            "layer and the component-banks slot layout. Use t158: b1 "
            "collapses when encoded from Q2 sources (the Q2 grid's "
            "exact-zero level; probe cosine 0.02-0.09) and is only a "
            "mechanism-speed arm there."
        ),
    )
    parser.add_argument(
        "--prefetch-slots",
        type=int,
        default=0,
        help=(
            "Speculative expert-prefetch ring slots per streamed layer "
            "(default 0: disabled). Decode applies the next sparse layers' "
            "routers to the current hidden state and overlaps predicted "
            "miss reads with compute. Requires layer cache scope."
        ),
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def parse_island_layers(text: str) -> tuple[int, ...]:
    """Parse '1-38,60-79' style layer ranges into a sorted unique tuple."""

    layers: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low_text, high_text = part.split("-", 1)
            low, high = int(low_text), int(high_text)
            if high < low:
                raise ValueError(f"island layer range {part!r} is inverted")
            layers.update(range(low, high + 1))
        else:
            layers.add(int(part))
    return tuple(sorted(layers))


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
        request = {
            "model": model,
            "model_root": model_root,
            "manifest": _expand(manifest or model_root / "expert-manifest.json"),
            "depths": tuple(depths),
        }
        if prompt_tail is not None:
            request["prompt_tail"] = _expand(prompt_tail)
        if not args.mtp_disabled_baseline:
            request["mtp_artifacts"] = _expand(mtp_artifacts)
            request["mtp_precision"] = args.mtp_precision
        requests.append(request)
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
        "q2_expert_kernel": args.q2_expert_kernel,
        "hy3_router_kernel": args.hy3_router_kernel,
        "hy3_router_sigmoid": args.hy3_router_sigmoid,
        "hy3_mtp_shared_kernel": args.hy3_mtp_shared_kernel,
        "hy3_mtp_shared_kernel_depth": args.hy3_mtp_shared_kernel_depth,
        "deferred_pin_release": bool(args.deferred_pin_release),
        "island_layers": args.island_layers,
        "island_layer_count": args.island_layer_count,
        "resident_quant": args.resident_quant,
        "kv_quant": args.kv_quant,
        "miss_shadow": args.miss_shadow,
        "miss_shadow_layers": args.miss_shadow_layers,
        "expert_integrity": args.expert_integrity,
        "prefetch_slots": args.prefetch_slots,
        "split_route_release": args.split_route_release,
        "mmap_island_layers": args.mmap_island_layers,
        "banked_manifest": args.banked_manifest,
        "banked_codec": args.banked_codec,
        "streamed_codec": args.streamed_codec,
        "streamed_codec_manifest": args.streamed_codec_manifest,
        "streamed_codec_verify": bool(args.streamed_codec_verify),
        "mmap_island_wired": bool(args.mmap_island_wired),
        "read_chunk": args.read_chunk,
        "bypass_page_cache": args.bypass_page_cache,
        "resource_telemetry": args.resource_telemetry,
        "resource_sample_interval": args.resource_sample_interval,
        "resource_max_samples": args.resource_max_samples,
        "ssd_ceiling_gib_s": args.ssd_ceiling_gib_s,
        "powermetrics": args.powermetrics,
        "powermetrics_interval_ms": args.powermetrics_interval_ms,
    }


def _profile_advisories(
    requests: Sequence[Mapping[str, Any]],
    runtime_options: Mapping[str, Any],
    *,
    verify_strategy: str,
) -> list[str]:
    """Optimization-profile advisories for the requested matrix (#99).

    Compares the effective flags against each model's measured-default
    profile and returns "profile suggests" lines. Warning only — nothing
    here changes what runs.
    """

    lines: list[str] = []
    for request in requests:
        model_key = MODEL_SPECS[request["model"]]["model_key"]
        observed = {
            "resident_quant": runtime_options.get("resident_quant") or None,
            "kv_quant": runtime_options.get("kv_quant") or None,
            "expert_integrity": runtime_options.get("expert_integrity"),
            "split_route_release": runtime_options.get("split_route_release"),
            "prefetch_slots": int(runtime_options.get("prefetch_slots") or 0),
            "miss_shadow": runtime_options.get("miss_shadow") or None,
            "miss_shadow_layers": runtime_options.get("miss_shadow_layers"),
            "island_layer_count": runtime_options.get("island_layer_count"),
            "mtp_depth": tuple(request["depths"]),
            "verify_strategy": verify_strategy,
        }
        lines.extend(profile_conflict_warnings(model_key, observed))
    return lines


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


def _prompt_identity(
    token_ids: Sequence[int], prompt_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    tokens = [int(token) for token in token_ids]
    encoded = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return {
        "token_count": len(tokens),
        "token_sha256": hashlib.sha256(encoded).hexdigest(),
        "prompt_policy": prompt_metadata.get("prompt_policy"),
        "prompt_format": prompt_metadata.get("prompt_format"),
        "prompt_release_valid": prompt_metadata.get("prompt_release_valid"),
        "prompt_tail_sha256": prompt_metadata.get("prompt_tail_sha256"),
        "prompt_filler_sha256": prompt_metadata.get("prompt_filler_sha256"),
        "prompt_artifact_kinds": prompt_metadata.get("prompt_artifact_kinds"),
    }


def _first_divergence(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if int(left_token) != int(right_token):
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def _token_sha256(tokens: Sequence[int]) -> str:
    encoded = json.dumps(
        [int(token) for token in tokens], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ar_comparison(
    tokens: Sequence[int],
    reference_tokens: Sequence[int] | None,
    *,
    is_reference: bool,
) -> dict[str, Any]:
    observed = [int(token) for token in tokens]
    if is_reference:
        reference = observed
        status = "reference"
    else:
        reference = [int(token) for token in (reference_tokens or [])]
        status = "exact" if observed == reference else "token_divergence"
    first = _first_divergence(observed, reference)
    differing = sum(
        left != right for left, right in zip(observed, reference, strict=False)
    ) + abs(len(observed) - len(reference))
    return {
        "status": status,
        "token_parity": observed == reference,
        "first_divergence": first,
        "differing_token_count": differing,
        "reference_token_at_first_divergence": (
            reference[first] if first is not None and first < len(reference) else None
        ),
        "observed_token_at_first_divergence": (
            observed[first] if first is not None and first < len(observed) else None
        ),
        "reference_token_sha256": _token_sha256(reference),
        "observed_token_sha256": _token_sha256(observed),
        "divergence_attribution": (None if observed == reference else "unclassified"),
    }


def _validate_speculative_event_contract(
    stats: Any,
    *,
    model: str,
    depth: int,
    tokens: Sequence[int],
) -> dict[str, int] | None:
    if depth == 0:
        return None
    events = _field(stats, "events")
    if not isinstance(events, (list, tuple)) or not events:
        raise BenchmarkGateError(
            f"{model} d{depth} did not expose speculative verification events"
        )

    drafted_records = 0
    evaluated_records = 0
    accepted_records = 0
    verify_events = 0
    drafted_by_depth = [0 for _ in range(depth)]
    evaluated_by_depth = [0 for _ in range(depth)]
    accepted_by_depth = [0 for _ in range(depth)]
    accept_probability_sums = [0.0 for _ in range(depth)]
    rejected_records = 0
    correction_records = 0
    bonus_records = 0
    fully_accepted_events = 0
    reconstructed_tokens: list[int] = []
    expected_pending_primary: int | None = None
    for event_index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise BenchmarkGateError(
                f"{model} d{depth} event {event_index} is not a mapping"
            )
        requested = event.get("requested_depth")
        if isinstance(requested, bool) or not isinstance(requested, int):
            raise BenchmarkGateError(
                f"{model} d{depth} event {event_index} has invalid requested depth"
            )
        if requested != depth:
            raise BenchmarkGateError(
                f"{model} d{depth} event {event_index} requested depth {requested}"
            )
        primary = event.get("primary")
        if isinstance(primary, bool) or not isinstance(primary, int):
            raise BenchmarkGateError(
                f"{model} d{depth} event {event_index} has invalid primary token"
            )
        primary_already_emitted = event.get("primary_already_emitted")
        if not isinstance(primary_already_emitted, bool):
            raise BenchmarkGateError(
                f"{model} d{depth} event {event_index} has invalid primary emission state"
            )
        if primary_already_emitted:
            if expected_pending_primary is None:
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} marks its primary as "
                    "already emitted without a prior pending verifier decision"
                )
            if expected_pending_primary != primary:
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} primary token does not "
                    "continue the prior pending verifier decision"
                )
            expected_pending_primary = None
        else:
            if expected_pending_primary is not None:
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} did not consume the "
                    "prior pending verifier decision"
                )
            reconstructed_tokens.append(primary)
        event_depth = event.get("depth")
        if isinstance(event_depth, bool) or not isinstance(event_depth, int):
            raise BenchmarkGateError(
                f"{model} d{depth} event {event_index} has invalid effective depth"
            )
        if event_depth != depth:
            raise BenchmarkGateError(
                f"{model} d{depth} event {event_index} has effective depth "
                f"{event_depth}; fixed-depth rows require {depth}"
            )
        drafts = event.get("drafts")
        if not isinstance(drafts, (list, tuple)):
            raise BenchmarkGateError(
                f"{model} d{depth} event {event_index} has invalid draft records"
            )
        if not drafts:
            if (
                event_index != len(events) - 1
                or event.get("pending_primary") != primary
            ):
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} has an empty nonterminal "
                    "verification decision"
                )
            expected_pending_primary = primary
            continue
        if len(drafts) > event_depth:
            raise BenchmarkGateError(
                f"{model} d{depth} event {event_index} exceeds its effective depth"
            )

        verify_events += 1
        drafted_records += len(drafts)
        for draft_index in range(len(drafts)):
            drafted_by_depth[draft_index] += 1
        accepted_prefix = 0
        first_rejection: int | None = None
        evaluation_closed = False
        for draft_index, draft in enumerate(drafts, start=1):
            if not isinstance(draft, Mapping):
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} has invalid draft record"
                )
            token = draft.get("token")
            if isinstance(token, bool) or not isinstance(token, int):
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} has invalid draft token"
                )
            declared_depth = draft.get("depth")
            if (
                isinstance(declared_depth, bool)
                or not isinstance(declared_depth, int)
                or declared_depth != draft_index
            ):
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} draft {draft_index} "
                    "has invalid depth metadata"
                )
            accepted = draft.get("accepted")
            if accepted is None:
                evaluation_closed = True
                continue
            if evaluation_closed or not isinstance(accepted, bool):
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} has non-prefix evaluation"
                )
            correction = draft.get("correction")
            if isinstance(correction, bool) or not isinstance(correction, int):
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} has invalid target correction"
                )
            accept_probability = draft.get("accept_probability")
            if (
                isinstance(accept_probability, bool)
                or not isinstance(accept_probability, (int, float))
                or not math.isfinite(float(accept_probability))
                or float(accept_probability) not in {0.0, 1.0}
                or float(accept_probability) != (1.0 if accepted else 0.0)
            ):
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} draft {draft_index} "
                    "has invalid greedy acceptance probability"
                )
            evaluated_records += 1
            evaluated_by_depth[draft_index - 1] += 1
            accept_probability_sums[draft_index - 1] += float(accept_probability)
            if accepted:
                if first_rejection is not None:
                    raise BenchmarkGateError(
                        f"{model} d{depth} event {event_index} accepts after rejection"
                    )
                if token != correction:
                    raise BenchmarkGateError(
                        f"{model} d{depth} accepted draft token {token} without "
                        f"matching target correction {correction}"
                    )
                accepted_prefix += 1
                accepted_records += 1
                accepted_by_depth[draft_index - 1] += 1
                reconstructed_tokens.append(token)
            else:
                if token == correction:
                    raise BenchmarkGateError(
                        f"{model} d{depth} rejected draft token {token} but its "
                        "greedy target correction is identical"
                    )
                first_rejection = draft_index
                evaluation_closed = True
                rejected_records += 1
                correction_records += 1
                reconstructed_tokens.append(correction)

        if event.get("accepted_depths") != accepted_prefix:
            raise BenchmarkGateError(
                f"{model} d{depth} event {event_index} accepted-depth count disagrees"
            )
        if event.get("rejected_at_depth") != first_rejection:
            raise BenchmarkGateError(
                f"{model} d{depth} event {event_index} rejection depth disagrees"
            )
        if accepted_prefix == len(drafts):
            fully_accepted_events += 1
        if first_rejection is not None and event.get("bonus_token") is not None:
            raise BenchmarkGateError(
                f"{model} d{depth} event {event_index} is a rejected event with "
                "a bonus token"
            )
        if first_rejection is None and event.get("bonus_token") is not None:
            bonus = event["bonus_token"]
            if isinstance(bonus, bool) or not isinstance(bonus, int):
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} has invalid bonus token"
                )
            reconstructed_tokens.append(bonus)
            bonus_records += 1
            expected_pending_primary = bonus
        pending = event.get("pending_primary")
        if pending is not None:
            if isinstance(pending, bool) or not isinstance(pending, int):
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} has invalid pending primary"
                )
            if not reconstructed_tokens or reconstructed_tokens[-1] != pending:
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} pending primary does not "
                    "match the verifier decision"
                )
            if (
                expected_pending_primary is not None
                and expected_pending_primary != pending
            ):
                raise BenchmarkGateError(
                    f"{model} d{depth} event {event_index} has conflicting pending "
                    "verifier decisions"
                )
            expected_pending_primary = pending

    expected = {
        "drafted_records": _required_int(stats, "drafted_tokens"),
        "evaluated_records": _required_int(stats, "evaluated_drafts"),
        "accepted_records": _required_int(stats, "accepted_drafts"),
    }
    actual = {
        "drafted_records": drafted_records,
        "evaluated_records": evaluated_records,
        "accepted_records": accepted_records,
    }
    if actual != expected:
        raise BenchmarkGateError(
            f"{model} d{depth} speculative event counts disagree with generation stats"
        )
    expected_depth_counts = {
        "drafted_by_depth": _integer_list(stats, "drafted_by_depth"),
        "evaluated_by_depth": _integer_list(stats, "evaluated_by_depth"),
        "accepted_by_depth": _integer_list(stats, "accepted_by_depth"),
    }
    actual_depth_counts = {
        "drafted_by_depth": drafted_by_depth,
        "evaluated_by_depth": evaluated_by_depth,
        "accepted_by_depth": accepted_by_depth,
    }
    if actual_depth_counts != expected_depth_counts:
        raise BenchmarkGateError(
            f"{model} d{depth} per-depth event counts disagree with generation stats"
        )
    if verify_events != _required_int(stats, "verify_calls"):
        raise BenchmarkGateError(
            f"{model} d{depth} verification-event count disagrees with generation stats"
        )
    derived_totals = {
        "rejected_drafts": rejected_records,
        "correction_tokens": correction_records,
        "bonus_tokens": bonus_records,
        "fully_accepted_verify_calls": fully_accepted_events,
    }
    reported_totals = {name: _required_int(stats, name) for name in derived_totals}
    if reported_totals != derived_totals:
        raise BenchmarkGateError(
            f"{model} d{depth} derived event totals disagree with generation stats"
        )
    reported_mean_probabilities = _number_list(
        stats, "mean_accept_probability_by_depth"
    )
    derived_mean_probabilities = [
        (total / count if count else None)
        for total, count in zip(
            accept_probability_sums,
            evaluated_by_depth,
            strict=True,
        )
    ]
    if len(reported_mean_probabilities) != depth or any(
        (reported is None) != (derived is None)
        or (
            reported is not None
            and derived is not None
            and not math.isclose(reported, derived, rel_tol=1e-12, abs_tol=1e-12)
        )
        for reported, derived in zip(
            reported_mean_probabilities,
            derived_mean_probabilities,
            strict=False,
        )
    ):
        raise BenchmarkGateError(
            f"{model} d{depth} acceptance probabilities disagree with event trace"
        )
    observed_tokens = [int(token) for token in tokens]
    if reconstructed_tokens != observed_tokens:
        divergence = _first_divergence(reconstructed_tokens, observed_tokens)
        raise BenchmarkGateError(
            f"{model} d{depth} verifier decision trace does not reconstruct output "
            f"at token {divergence}"
        )
    return {"events": len(events), "verify_events": verify_events, **actual}


def _cache_offsets(cache: Any, *, label: str) -> list[int]:
    if not isinstance(cache, (list, tuple)) or not cache:
        raise BenchmarkGateError(f"generation final state has no {label} cache")
    offsets: list[int] = []
    for index, entry in enumerate(cache):
        if entry is None:
            raise BenchmarkGateError(
                f"generation final state {label} cache {index} is missing"
            )
        value = getattr(entry, "offset", None)
        if value is None:
            size = getattr(entry, "size", None)
            if callable(size):
                value = size()
        try:
            parsed = int(value.item()) if hasattr(value, "item") else int(value)
        except (TypeError, ValueError):
            raise BenchmarkGateError(
                f"generation final state {label} cache {index} has no logical offset"
            ) from None
        if parsed < 0:
            raise BenchmarkGateError(
                f"generation final state {label} cache {index} has a negative offset"
            )
        offsets.append(parsed)
    if not offsets:
        raise BenchmarkGateError(f"generation final state has no {label} offsets")
    return offsets


def _validate_final_state_contract(
    result: Any,
    stats: Any,
    *,
    model: str,
    depth: int,
    prompt_tokens: int,
    tokens: Sequence[int],
    finish_reason: str | None,
) -> dict[str, Any] | None:
    if depth == 0:
        return None
    final_state = _field(result, "final_state")
    if final_state is None:
        raise BenchmarkGateError(f"{model} d{depth} did not capture final state")
    if _field(final_state, "safe_to_commit") is not True:
        raise BenchmarkGateError(f"{model} d{depth} final state is unsafe to commit")
    generated = _field(final_state, "generated_token_ids")
    if not isinstance(generated, (list, tuple)) or [
        int(item) for item in generated
    ] != [int(item) for item in tokens]:
        raise BenchmarkGateError(
            f"{model} d{depth} final state token history disagrees with output"
        )
    if _field(final_state, "finish_reason") != finish_reason:
        raise BenchmarkGateError(
            f"{model} d{depth} final state finish reason disagrees with output"
        )
    expected_prompt_history = max(0, prompt_tokens - 1)
    if _required_int(stats, "prompt_mtp_history_tokens") != expected_prompt_history:
        raise BenchmarkGateError(
            f"{model} d{depth} prompt MTP history does not contain N-1 rows"
        )
    if (
        _required_int(stats, "mtp_history_position_base") != 0
        or _field(final_state, "mtp_history_position_base") != 0
    ):
        raise BenchmarkGateError(
            f"{model} d{depth} committed MTP history position base is not zero"
        )
    target_offsets = _cache_offsets(
        _field(final_state, "final_trunk_cache"),
        label="target",
    )
    mtp_offsets = _cache_offsets(
        _field(final_state, "final_committed_mtp_cache"),
        label="committed MTP",
    )
    expected_target_offset = prompt_tokens + len(tokens)
    expected_mtp_offset = expected_target_offset - 1
    if any(offset != expected_target_offset for offset in target_offsets):
        raise BenchmarkGateError(
            f"{model} d{depth} target cache offsets do not match prompt plus output"
        )
    if any(offset != expected_mtp_offset for offset in mtp_offsets):
        raise BenchmarkGateError(
            f"{model} d{depth} committed MTP cache offsets do not match N-1"
        )
    return {
        "safe_to_commit": True,
        "generated_token_ids_match": True,
        "finish_reason_match": True,
        "prompt_mtp_history_tokens": expected_prompt_history,
        "mtp_history_position_base": 0,
        "target_cache_offsets": target_offsets,
        "committed_mtp_cache_offsets": mtp_offsets,
    }


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
    final_state_capture_time = _optional_number(stats, "final_state_capture_time_s")
    if min(prompt_eval_time, target_prefill_time, decode_elapsed, elapsed) <= 0.0:
        raise BenchmarkGateError("generation timing denominators must be positive")
    if final_state_capture_time < 0.0:
        raise BenchmarkGateError("final-state capture time must be non-negative")
    if depth == 0 and final_state_capture_time != 0.0:
        raise BenchmarkGateError(
            "AR timing reports unexpected final-state capture work"
        )
    if final_state_capture_time >= min(decode_elapsed, elapsed):
        raise BenchmarkGateError(
            "final-state capture time must be smaller than measured generation time"
        )
    raw_decode_elapsed = decode_elapsed
    raw_elapsed = elapsed
    decode_elapsed -= final_state_capture_time
    elapsed -= final_state_capture_time
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
        "raw_elapsed_s": raw_elapsed,
        "raw_decode_elapsed_s": raw_decode_elapsed,
        "final_state_capture_time_s": final_state_capture_time,
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


def _streaming_cache_metrics(
    runtime: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    snapshot_fn = getattr(runtime, "expert_streaming_snapshot", None)
    if not callable(snapshot_fn):
        return None, None
    snapshot = snapshot_fn()
    if not isinstance(snapshot, Mapping):
        return None, None
    cache = snapshot.get("cache")
    cache_by_phase = snapshot.get("cache_by_phase")
    return (
        _jsonable(cache) if isinstance(cache, Mapping) else None,
        _jsonable(cache_by_phase) if isinstance(cache_by_phase, Mapping) else None,
    )


def _streaming_counters(runtime: Any) -> dict[str, Any] | None:
    """Return aggregate counters for compatibility with existing artifacts."""

    return _streaming_cache_metrics(runtime)[0]


def _cache_hit_rate(counters: Any) -> float | None:
    if not isinstance(counters, Mapping):
        return None
    hit_rate = counters.get("hit_rate")
    if isinstance(hit_rate, (int, float)) and not isinstance(hit_rate, bool):
        return float(hit_rate)
    hits = counters.get("expert_hits")
    misses = counters.get("expert_misses")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (hits, misses)
    ):
        return None
    return _ratio(hits, hits + misses)


def _require_decode_cache_metrics(
    cache_by_phase: Any,
    *,
    model: str,
    depth: int,
    fully_islanded: bool = False,
) -> tuple[dict[str, Any], float | None]:
    decode = (
        cache_by_phase.get("decode") if isinstance(cache_by_phase, Mapping) else None
    )
    if not isinstance(decode, Mapping):
        raise BenchmarkGateError(
            f"{model} d{depth} did not expose decode expert-cache counters"
        )
    for name in ("expert_hits", "expert_misses"):
        value = decode.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BenchmarkGateError(
                f"{model} d{depth} decode expert-cache {name} is invalid"
            )
    hits = int(decode["expert_hits"])
    misses = int(decode["expert_misses"])
    denominator = hits + misses
    if fully_islanded:
        # Every routed layer executes on an island bank; a streamed cache
        # assignment here is a wiring bug, not routing evidence.
        if denominator != 0:
            raise BenchmarkGateError(
                f"{model} d{depth} routed {denominator} assignments through "
                "the streamed expert cache despite full island coverage"
            )
        return dict(decode), None
    if denominator <= 0:
        raise BenchmarkGateError(
            f"{model} d{depth} decode expert-cache has no routed assignments"
        )
    hit_rate = hits / denominator
    reported_hit_rate = decode.get("hit_rate")
    if (
        isinstance(reported_hit_rate, bool)
        or not isinstance(reported_hit_rate, (int, float))
        or not math.isfinite(float(reported_hit_rate))
    ):
        raise BenchmarkGateError(
            f"{model} d{depth} decode expert-cache hit rate is invalid"
        )
    if not math.isclose(float(reported_hit_rate), hit_rate, abs_tol=1e-12):
        raise BenchmarkGateError(
            f"{model} d{depth} decode expert-cache hit rate disagrees with counts"
        )
    return dict(decode), hit_rate


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


def _reader_pool_interval(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, float | int]:
    before_pool = before.get("reader_pool")
    after_pool = after.get("reader_pool")
    if not isinstance(before_pool, Mapping) or not isinstance(after_pool, Mapping):
        raise BenchmarkGateError("resource telemetry is missing reader_pool counters")
    capacity = after_pool.get("worker_capacity")
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or capacity <= 0
        or before_pool.get("worker_capacity") != capacity
    ):
        raise BenchmarkGateError("reader_pool worker capacity is invalid or changed")

    def interval(name: str) -> int:
        start = before_pool.get(name)
        end = after_pool.get(name)
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end < start
        ):
            raise BenchmarkGateError(f"reader_pool {name} interval is invalid")
        return end - start

    elapsed_ns = interval("observation_ns")
    if elapsed_ns <= 0:
        raise BenchmarkGateError("reader_pool decode interval is empty")
    active_work_ns = interval("active_work_ns")
    queued_work_ns = interval("queued_work_ns")
    before_histogram = before_pool.get("active_work_histogram_ns")
    after_histogram = after_pool.get("active_work_histogram_ns")
    if not isinstance(before_histogram, Mapping) or not isinstance(
        after_histogram, Mapping
    ):
        raise BenchmarkGateError("reader_pool active occupancy histogram is missing")

    histogram_ns: dict[int, int] = {}
    for active_readers in range(capacity + 1):
        key = str(active_readers)
        if key not in before_histogram and key not in after_histogram:
            histogram_ns[active_readers] = 0
            continue
        start = before_histogram.get(key)
        end = after_histogram.get(key)
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end < start
        ):
            raise BenchmarkGateError(
                "reader_pool active occupancy histogram interval is invalid"
            )
        histogram_ns[active_readers] = end - start
    if sum(histogram_ns.values()) != elapsed_ns:
        raise BenchmarkGateError(
            "reader_pool active occupancy histogram disagrees with elapsed time"
        )
    integrated_active_ns = sum(
        active_readers * duration_ns
        for active_readers, duration_ns in histogram_ns.items()
    )
    if integrated_active_ns != active_work_ns:
        raise BenchmarkGateError(
            "reader_pool active occupancy histogram disagrees with active time"
        )
    mean_active = active_work_ns / elapsed_ns
    mean_queued = queued_work_ns / elapsed_ns
    peak_active = max(
        (
            active_readers
            for active_readers, duration_ns in histogram_ns.items()
            if duration_ns > 0
        ),
        default=0,
    )
    return {
        "worker_capacity": capacity,
        "elapsed_seconds": elapsed_ns / 1_000_000_000,
        "mean_active_readers": mean_active,
        "active_capacity_fraction": mean_active / capacity,
        "peak_active_readers": peak_active,
        "full_capacity_fraction": histogram_ns[capacity] / elapsed_ns,
        "mean_queued_reads": mean_queued,
    }


def _result_tokens(result: Any) -> list[int]:
    value = _field(result, "tokens")
    if not isinstance(value, (list, tuple)):
        raise BenchmarkGateError("generation result tokens must be a sequence")
    return [int(token) for token in value]


def _require_compiled_draft_evidence(
    stats: Any,
    *,
    model: str,
    depth: int,
    draft_core: str,
) -> dict[str, Any] | None:
    evidence = _field(stats, "draft_core")
    if draft_core == "stock":
        return dict(evidence) if isinstance(evidence, Mapping) else None
    if not isinstance(evidence, Mapping):
        raise BenchmarkGateError(f"{model} d{depth} compiled draft emitted no evidence")
    report = dict(evidence)
    if report.get("requested") != draft_core:
        raise BenchmarkGateError(
            f"{model} d{depth} compiled draft request evidence disagrees"
        )
    if report.get("selected") != "device-k":
        raise BenchmarkGateError(
            f"{model} d{depth} silently selected a non-compiled draft core"
        )
    compiled_calls = _optional_int(report, "compiled_calls")
    if compiled_calls is None or compiled_calls <= 0:
        raise BenchmarkGateError(
            f"{model} d{depth} compiled draft emitted no measured calls"
        )
    if _optional_int(report, "organic_compile_calls") != 0:
        raise BenchmarkGateError(
            f"{model} d{depth} compiled draft compiled during measured dispatch"
        )
    fallbacks = _optional_int(report, "fallbacks")
    if fallbacks != 0 or report.get("fallback_reasons"):
        raise BenchmarkGateError(f"{model} d{depth} compiled draft used fallback calls")
    if report.get("history_policy") != "committed":
        raise BenchmarkGateError(
            f"{model} d{depth} compiled draft did not use committed history"
        )
    if report.get("cache_state_mode") != "explicit-live-io":
        raise BenchmarkGateError(
            f"{model} d{depth} compiled draft did not expose live cache state"
        )
    prewarm = report.get("prewarm")
    if not isinstance(prewarm, Mapping) or prewarm.get("compiled") is not True:
        raise BenchmarkGateError(f"{model} d{depth} compiled draft was not prewarmed")
    per_depth = report.get("per_depth")
    depth_evidence = (
        per_depth.get(str(depth)) if isinstance(per_depth, Mapping) else None
    )
    if not isinstance(depth_evidence, Mapping):
        raise BenchmarkGateError(
            f"{model} d{depth} compiled draft omitted per-depth evidence"
        )
    depth_calls = _optional_int(depth_evidence, "calls")
    live_commits = _optional_int(depth_evidence, "live_cache_commits")
    if depth_calls is None or depth_calls <= 0 or live_commits != depth_calls:
        raise BenchmarkGateError(
            f"{model} d{depth} compiled draft live-cache commits disagree"
        )
    if draft_core == "device-k":
        if report.get("qualification_eligible") is not True:
            raise BenchmarkGateError(
                f"{model} d{depth} compiled draft is not qualification eligible"
            )
        if report.get("primary_depth") != 3 or report.get("primary_width") != 4:
            raise BenchmarkGateError(
                f"{model} d{depth} compiled draft K3/M4 marker disagrees"
            )
        supported = [int(value) for value in report.get("supported_depths", ())]
        if supported != list(range(1, 8)):
            raise BenchmarkGateError(
                f"{model} d{depth} compiled draft D1...D7 support disagrees"
            )
        prewarmed = [int(value) for value in report.get("prewarmed_depths", ())]
        if prewarmed != list(range(1, depth + 1)):
            raise BenchmarkGateError(
                f"{model} d{depth} compiled draft prewarm coverage disagrees"
            )
    elif depth != 2:
        raise BenchmarkGateError("legacy device-d2 requires an exact D2 row")
    return report


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
    resource_options: Mapping[str, Any],
    ar_tokens: Sequence[int] | None,
    ar_finish_reason: str | None,
    verify_strategy: str,
    compiled_verify_mode: str,
    draft_core: str,
    draft_observer: Callable[[Mapping[str, Any]], None] | None,
) -> tuple[dict[str, Any], list[int], str | None]:
    _reset_expert_streaming(runtime)
    resource_telemetry_enabled = bool(resource_options["resource_telemetry"])
    resource_before = (
        _resource_telemetry(runtime) if resource_telemetry_enabled else None
    )
    resource_prefill_complete = None
    resource_sampler: ResourceTelemetrySampler | None = None
    power_collector: PowermetricsCollector | None = None
    resource_completion_tokens = 0
    generation_thread_started_ns: int | None = None
    generation_thread_finished_ns: int | None = None
    generation_started_ns: int | None = None
    generation_finished_ns: int | None = None

    def capture_tokens(token_ids: Sequence[int]) -> None:
        nonlocal resource_completion_tokens
        resource_completion_tokens += len(token_ids)

    def capture_prefill_boundary(event: Mapping[str, Any]) -> None:
        nonlocal resource_prefill_complete
        nonlocal resource_sampler, power_collector
        nonlocal generation_thread_started_ns, generation_started_ns
        if resource_prefill_complete is None and event.get("phase") == "completed":
            power_collector = apis.powermetrics_factory(
                enabled=bool(resource_options["powermetrics"]),
                pid=os.getpid(),
                interval_ms=int(resource_options["powermetrics_interval_ms"]),
            )
            power_collector.start()
            resource_sampler = apis.resource_sampler_factory(
                lambda: _resource_telemetry(runtime),
                token_count=lambda: resource_completion_tokens,
                interval_s=float(resource_options["resource_sample_interval"]),
                max_samples=int(resource_options["resource_max_samples"]),
            )
            resource_sampler.__enter__()
            ticks = resource_sampler.ticks
            if not ticks:
                raise BenchmarkGateError(
                    "resource sampler did not capture the prefill boundary"
                )
            resource_prefill_complete = _jsonable(ticks[0].snapshot)
            generation_thread_started_ns = int(apis.thread_time_ns())
            generation_started_ns = int(apis.monotonic_ns())

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
    if resource_telemetry_enabled:
        common["prefill_callback"] = capture_prefill_boundary
        common["token_callback"] = capture_tokens
    try:
        with admission:
            if depth == 0:
                result = apis.generate_ar(runtime, prompt_copy, **common)
            else:
                mtpk_options: dict[str, Any] = {
                    "speculative_depth": depth,
                    "mtp_cache_policy": "persistent",
                    "mtp_history_policy": "committed",
                    "draft_core": draft_core,
                    "capture_final_state": True,
                    "verify_strategy": verify_strategy,
                }
                if draft_observer is not None:
                    mtpk_options["draft_observer"] = draft_observer
                result = apis.generate_mtpk(
                    runtime,
                    prompt_copy,
                    **mtpk_options,
                    **common,
                )
        apis.synchronize()
    finally:
        if resource_sampler is not None:
            generation_finished_ns = int(apis.monotonic_ns())
            generation_thread_finished_ns = int(apis.thread_time_ns())
            resource_sampler.__exit__(None, None, None)
        if power_collector is not None:
            power_collector.stop()
    resource_after = None
    if resource_sampler is not None and resource_sampler.ticks:
        resource_after = _jsonable(resource_sampler.ticks[-1].snapshot)
    if resource_telemetry_enabled and resource_prefill_complete is None:
        raise BenchmarkGateError(
            "resource telemetry did not observe the prefill-complete boundary"
        )
    resource_evidence = None
    if resource_telemetry_enabled:
        assert resource_before is not None
        assert resource_prefill_complete is not None
        assert resource_after is not None
        resource_evidence = {
            "before": resource_before,
            "prefill_complete": resource_prefill_complete,
            "after": resource_after,
            "numeric_delta": _numeric_delta(resource_before, resource_after),
            "decode": {
                "numeric_delta": _numeric_delta(
                    resource_prefill_complete, resource_after
                ),
                "reader_pool": _reader_pool_interval(
                    resource_prefill_complete, resource_after
                ),
            },
        }

    if tuple(int(token) for token in prompt_copy) != prompt_ids:
        raise BenchmarkGateError("generation mutated the exact benchmark prompt")
    tokens = _result_tokens(result)
    finish_reason = _field(result, "finish_reason")
    stats = _field(result, "stats")
    if stats is None:
        raise BenchmarkGateError("generation result is missing stats")
    streaming_counters, streaming_counters_by_phase = _streaming_cache_metrics(runtime)
    failure_evidence = {
        "model": model,
        "depth": depth,
        "context_tokens": context_tokens,
        "token_ids": tokens,
        "finish_reason": finish_reason,
        "generation_events": _jsonable(_field(stats, "events", [])),
        "generation_stats": _jsonable(stats),
        "expert_streaming_counters": streaming_counters,
        "expert_streaming_counters_by_phase": streaming_counters_by_phase,
        "expert_resource_telemetry": (
            resource_evidence if resource_telemetry_enabled else None
        ),
    }
    try:
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
        ar_comparison = _ar_comparison(
            tokens,
            ar_tokens,
            is_reference=depth == 0,
        )
        ar_finish_parity = depth == 0 or finish_reason == ar_finish_reason
        requested_exact = requested_depth == depth
        effective_exact = effective_depth == depth
        committed_history = depth == 0 or history_policy == "committed"
        guards_disabled = _guards_disabled(stats)
        compiled_draft = (
            _require_compiled_draft_evidence(
                stats,
                model=model,
                depth=depth,
                draft_core=draft_core,
            )
            if depth > 0
            else None
        )
        compiled_verify = None
        graphbank = _field(stats, "graphbank", {})
        if isinstance(graphbank, Mapping):
            candidate_evidence = graphbank.get("compiled_verify")
            if isinstance(candidate_evidence, Mapping):
                compiled_verify = dict(candidate_evidence)
        compiled_verify_evidence = depth == 0
        if depth > 0 and compiled_verify_mode in {"parity", "on"}:
            if compiled_verify is None:
                raise BenchmarkGateError("compiled verifier emitted no evidence")
            calls = _optional_int(compiled_verify, "calls")
            compiled_calls = _optional_int(compiled_verify, "compiled_calls")
            fallback_calls = _optional_int(compiled_verify, "fallback_calls")
            if calls is None or calls <= 0:
                raise BenchmarkGateError("compiled verifier emitted no calls")
            if fallback_calls != 0:
                raise BenchmarkGateError("compiled verifier used fallback calls")
            if compiled_calls != calls:
                raise BenchmarkGateError(
                    "compiled verifier calls were not fully compiled"
                )
            if compiled_verify.get("mode") != compiled_verify_mode:
                raise BenchmarkGateError("compiled verifier mode evidence disagrees")
            compiled_verify_evidence = True
        elif depth > 0:
            compiled_calls = (
                _optional_int(compiled_verify, "compiled_calls")
                if compiled_verify is not None
                else 0
            )
            if compiled_calls not in {None, 0}:
                raise BenchmarkGateError(
                    "compiled verifier ran while declared mode was off"
                )
            compiled_verify_evidence = True
        if not ar_finish_parity:
            raise BenchmarkGateError(f"{model} d{depth} finish reason diverged from AR")
        if not requested_exact or not effective_exact:
            raise BenchmarkGateError(
                f"{model} d{depth} reported requested/effective depth "
                f"{requested_depth}/{effective_depth}"
            )
        if not committed_history:
            raise BenchmarkGateError(
                f"{model} d{depth} did not use committed MTP history"
            )
        if not guards_disabled:
            raise BenchmarkGateError(f"{model} d{depth} triggered a generation guard")

        expert_streaming = getattr(runtime, "expert_streaming", None)
        fully_islanded = False
        if expert_streaming is not None:
            streaming_config = expert_streaming.config
            island_total = len(streaming_config.island_layers) + len(
                getattr(streaming_config, "mmap_island_layers", ())
            )
            fully_islanded = island_total == len(
                expert_streaming.spec.routed_layer_indices
            )
        _decode_streaming_counters, decode_cache_hit_rate = (
            _require_decode_cache_metrics(
                streaming_counters_by_phase,
                model=model,
                depth=depth,
                fully_islanded=fully_islanded,
            )
        )
        speculative_event_contract = _validate_speculative_event_contract(
            stats,
            model=model,
            depth=depth,
            tokens=tokens,
        )
        final_state_contract = _validate_final_state_contract(
            result,
            stats,
            model=model,
            depth=depth,
            prompt_tokens=len(prompt_ids),
            tokens=tokens,
            finish_reason=finish_reason,
        )
        metrics = _metrics(stats, completion_tokens=len(tokens), depth=depth)
    except BenchmarkGateError as exc:
        if exc.evidence is None:
            exc.evidence = failure_evidence
        raise
    resource_report = None
    if resource_telemetry_enabled:
        assert resource_sampler is not None
        assert power_collector is not None
        assert generation_thread_started_ns is not None
        assert generation_thread_finished_ns is not None
        assert generation_started_ns is not None
        assert generation_finished_ns is not None
        resource_report = resource_sampler.report(
            ssd_ceiling_gib_s=resource_options["ssd_ceiling_gib_s"],
            powermetrics=power_collector.report(),
            generation_thread_cpu_ns=(
                generation_thread_finished_ns - generation_thread_started_ns
            ),
            generation_elapsed_ns=generation_finished_ns - generation_started_ns,
            final_completion_tokens=len(tokens),
        )
        memory_limit_bytes = int(
            apis.parse_memory_bytes(resource_options["memory_limit"])
        )
        resource_report["memory"] = {
            "peak_memory_bytes": metrics["peak_memory_bytes"],
            "limit_bytes": memory_limit_bytes,
            "utilization_of_limit": (
                metrics["peak_memory_bytes"] / memory_limit_bytes
                if memory_limit_bytes > 0
                else None
            ),
        }

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
        "verify_strategy": None if depth == 0 else verify_strategy,
        "finish_reason": finish_reason,
        "token_ids": tokens,
        "generation_events": _jsonable(_field(stats, "events", [])),
        "compiled_verify": _jsonable(compiled_verify),
        "draft_core": None if depth == 0 else draft_core,
        "compiled_draft": _jsonable(compiled_draft),
        "ar_comparison": ar_comparison,
        "speculative_event_contract": speculative_event_contract,
        "final_state_contract": final_state_contract,
        **metrics,
        "expert_streaming_counters": streaming_counters,
        "expert_streaming_counters_by_phase": streaming_counters_by_phase,
        "decode_expert_cache_hit_rate": decode_cache_hit_rate,
        "expert_resource_telemetry": (
            resource_evidence if resource_telemetry_enabled else None
        ),
        "resource_telemetry": resource_report,
        "gates": {
            "prompt_length_exact": len(prompt_ids) == context_tokens,
            "new_prefill_tokens_exact": _field(stats, "new_prefill_tokens")
            == len(prompt_ids),
            "output_tokens_exact": len(tokens) == max_tokens,
            "generated_count_consistent": _field(stats, "generated_tokens")
            == len(tokens),
            "length_finish": finish_reason == "length",
            "requested_depth_exact": requested_exact,
            "effective_depth_exact": effective_exact,
            "committed_history": committed_history,
            "guards_disabled": guards_disabled,
            "decode_expert_cache_metrics": True,
            "speculative_event_contract": depth == 0
            or speculative_event_contract is not None,
            "final_state_contract": depth == 0 or final_state_contract is not None,
            "compiled_verify_evidence": compiled_verify_evidence,
            "compiled_draft_evidence": depth == 0
            or draft_core == "stock"
            or compiled_draft is not None,
        },
    }
    if not row["gates"]["new_prefill_tokens_exact"]:
        raise BenchmarkGateError(
            f"{model} d{depth} did not ingest the full exact prompt"
        )
    return row, tokens, finish_reason


def _record_token_divergence(
    model_payload: dict[str, Any],
    row: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    comparison = row.get("ar_comparison")
    if not isinstance(comparison, Mapping):
        return
    if comparison.get("status") != "token_divergence":
        return
    model_payload["token_divergence_observations"].append(
        {
            "model": row["model"],
            "context_tokens": row["context_tokens"],
            "depth": row["requested_depth"],
            "cell": row["cell"],
            "phase": phase,
            **dict(comparison),
        }
    )


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
        q2_expert_kernel=str(options["q2_expert_kernel"]),
        hy3_router_kernel=str(options["hy3_router_kernel"]),
        hy3_router_sigmoid=str(options["hy3_router_sigmoid"]),
        hy3_mtp_shared_kernel=str(options["hy3_mtp_shared_kernel"]),
        hy3_mtp_shared_kernel_depth=int(
            options["hy3_mtp_shared_kernel_depth"]
        ),
        deferred_pin_release=bool(options["deferred_pin_release"]),
        max_read_chunk_bytes=apis.parse_memory_bytes(options["read_chunk"]),
        bypass_page_cache=bool(options["bypass_page_cache"]),
        resource_telemetry=bool(options["resource_telemetry"]),
        trace_routes=bool(options["trace_routes"]),
        island_layers=parse_island_layers(options.get("island_layers", "")),
        island_layer_count=options.get("island_layer_count"),
        resident_quant=options.get("resident_quant") or None,
        kv_quant=options.get("kv_quant") or None,
        miss_shadow=options.get("miss_shadow") or None,
        miss_shadow_layers=options.get("miss_shadow_layers"),
        prefetch_slots=int(options.get("prefetch_slots") or 0),
        verify_record_hashes=(
            options.get("expert_integrity", "per-read") == "per-read"
        ),
        verify_sidecar_hash_at_open=(
            options.get("expert_integrity") == "at-open"
        ),
        split_route_release=options.get("split_route_release", "fenced"),
        mmap_island_layers=parse_island_layers(
            options.get("mmap_island_layers", "")
        ),
        banked_manifest=(
            str(options["banked_manifest"])
            if options.get("banked_manifest")
            else None
        ),
        banked_codec=str(options.get("banked_codec", "none")),
        mmap_island_wired=bool(options.get("mmap_island_wired", True)),
        streamed_codec=str(options.get("streamed_codec", "none")),
        streamed_codec_manifest=(
            str(options["streamed_codec_manifest"])
            if options.get("streamed_codec_manifest")
            else None
        ),
        streamed_codec_verify=bool(options.get("streamed_codec_verify", True)),
    )


def _normalized_request(
    request: Mapping[str, Any],
    *,
    mtp_disabled_baseline: bool,
) -> dict[str, Any]:
    model = request.get("model")
    if model not in MODEL_SPECS:
        raise BenchmarkConfigurationError(f"unsupported model {model!r}")
    spec = MODEL_SPECS[model]
    if mtp_disabled_baseline:
        depths: tuple[int, ...] = ()
    else:
        depths = tuple(int(value) for value in request.get("depths", spec["depths"]))
        if not depths or len(set(depths)) != len(depths):
            raise BenchmarkConfigurationError(
                f"{model} depths must be unique and nonempty"
            )
        permitted = set(spec["depths"])
        if any(depth not in permitted for depth in depths):
            raise BenchmarkConfigurationError(
                f"{model} depths must be selected from {tuple(spec['depths'])}"
            )
    required_paths = (
        ("model_root", "manifest")
        if mtp_disabled_baseline
        else ("model_root", "manifest", "mtp_artifacts")
    )
    missing = [name for name in required_paths if request.get(name) is None]
    if missing:
        raise BenchmarkConfigurationError(
            f"{model} is missing required paths: {', '.join(missing)}"
        )
    if request.get("prompt_tail_text") is not None:
        prompt_tail_text: str | None = str(request["prompt_tail_text"])
        prompt_tail_path = request.get("prompt_tail")
    elif request.get("prompt_tail") is not None:
        prompt_tail_path = request["prompt_tail"]
        prompt_tail_text = _expand(prompt_tail_path).read_text(encoding="utf-8")
    else:
        prompt_tail_path = None
        prompt_tail_text = None
    if prompt_tail_text is not None and not prompt_tail_text:
        raise BenchmarkConfigurationError(f"{model} prompt tail must not be empty")
    normalized = {
        "model": model,
        "model_key": spec["model_key"],
        "depths": depths,
        "model_root": _expand(request["model_root"]),
        "manifest": _expand(request["manifest"]),
        "prompt_tail": _expand(prompt_tail_path) if prompt_tail_path else None,
        "prompt_tail_text": prompt_tail_text,
    }
    if not mtp_disabled_baseline:
        normalized["mtp_artifacts"] = _expand(request["mtp_artifacts"])
        precision = str(request.get("mtp_precision", "bf16"))
        if precision not in ("bf16", "q4"):
            raise BenchmarkConfigurationError(
                f"{model} mtp_precision must be 'bf16' or 'q4'; got {precision!r}"
            )
        normalized["mtp_precision"] = precision
    return normalized


def _checkpoint_skeleton(
    *,
    contexts: Sequence[int],
    output_tokens: int,
    runtime_options: Mapping[str, Any] | None,
    mtp_disabled_baseline: bool,
    verify_strategy: str,
    compiled_verify_mode: str,
    draft_core: str,
    trace_routes: bool,
) -> dict[str, Any]:
    """Return a uniform payload for failures before validated setup exists."""

    baseline = (
        mtp_disabled_baseline if isinstance(mtp_disabled_baseline, bool) else None
    )
    lane = (
        "mtp-disabled-ar-baseline"
        if baseline is True
        else "mtp-resident-depth-matrix"
        if baseline is False
        else "invalid-configuration"
    )
    return {
        "schema": SCHEMA,
        "status": "running",
        "passed": False,
        "active_cell": {"phase": "configuration"},
        "failure": None,
        "lane": lane,
        "mtp_resident": None if baseline is None else not baseline,
        "configuration": {
            "lane": lane,
            "mtp_resident": None if baseline is None else not baseline,
            "requested_contexts": _jsonable(list(contexts)),
            "requested_output_tokens": _jsonable(output_tokens),
            "requested_runtime": _jsonable(dict(runtime_options or {})),
            "requested_candidate": {
                "verify_strategy": verify_strategy,
                "compiled_verify_mode": compiled_verify_mode,
                "draft_core": draft_core,
                "trace_routes": trace_routes,
            },
        },
        "models": [],
    }


def run_depth_matrix(
    model_requests: Sequence[Mapping[str, Any]],
    *,
    contexts: Sequence[int] = DEFAULT_CONTEXTS,
    output_tokens: int = OUTPUT_TOKENS,
    runtime_options: Mapping[str, Any] | None = None,
    mtp_disabled_baseline: bool = False,
    verify_strategy: str = "batched",
    compiled_verify_mode: str = "off",
    draft_core: str = "stock",
    trace_routes: bool = False,
    draft_observer: Callable[[Mapping[str, Any]], None] | None = None,
    checkpoint: Checkpoint | None = None,
    apis: RunnerAPIs | None = None,
) -> dict[str, Any]:
    """Run one AR/depth matrix or true MTP-disabled AR baseline per model."""

    live_payload = {
        "payload": _checkpoint_skeleton(
            contexts=contexts,
            output_tokens=output_tokens,
            runtime_options=runtime_options,
            mtp_disabled_baseline=mtp_disabled_baseline,
            verify_strategy=verify_strategy,
            compiled_verify_mode=compiled_verify_mode,
            draft_core=draft_core,
            trace_routes=trace_routes,
        )
    }
    try:
        return _run_depth_matrix_impl(
            model_requests,
            contexts=contexts,
            output_tokens=output_tokens,
            runtime_options=runtime_options,
            mtp_disabled_baseline=mtp_disabled_baseline,
            verify_strategy=verify_strategy,
            compiled_verify_mode=compiled_verify_mode,
            draft_core=draft_core,
            trace_routes=trace_routes,
            draft_observer=draft_observer,
            checkpoint=checkpoint,
            apis=apis,
            _live_payload=live_payload,
        )
    except Exception as exc:
        failed = live_payload["payload"]
        active_cell = copy.deepcopy(failed.get("active_cell"))
        failed["status"] = "failed"
        failed["passed"] = False
        failed["failure"] = {
            "error": str(exc),
            "error_type": type(exc).__name__,
            "active_cell": active_cell,
        }
        evidence = getattr(exc, "evidence", None)
        if isinstance(evidence, Mapping):
            failed["failure"]["evidence"] = _jsonable(evidence)
        _emit_checkpoint(failed, checkpoint)
        raise


def _run_depth_matrix_impl(
    model_requests: Sequence[Mapping[str, Any]],
    *,
    contexts: Sequence[int] = DEFAULT_CONTEXTS,
    output_tokens: int = OUTPUT_TOKENS,
    runtime_options: Mapping[str, Any] | None = None,
    mtp_disabled_baseline: bool = False,
    verify_strategy: str = "batched",
    compiled_verify_mode: str = "off",
    draft_core: str = "stock",
    trace_routes: bool = False,
    draft_observer: Callable[[Mapping[str, Any]], None] | None = None,
    checkpoint: Checkpoint | None = None,
    apis: RunnerAPIs | None = None,
    _live_payload: dict[str, Any],
) -> dict[str, Any]:
    """Validated implementation; ``run_depth_matrix`` owns terminal failure state."""

    if not model_requests:
        raise BenchmarkConfigurationError("at least one model must be selected")
    if not isinstance(mtp_disabled_baseline, bool):
        raise BenchmarkConfigurationError("mtp_disabled_baseline must be bool")
    if (
        isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens <= 0
    ):
        raise BenchmarkConfigurationError("output_tokens must be a positive integer")
    if verify_strategy not in VERIFY_STRATEGIES:
        raise BenchmarkConfigurationError("unsupported verify strategy")
    if compiled_verify_mode not in COMPILED_VERIFY_MODES:
        raise BenchmarkConfigurationError("unsupported compiled verify mode")
    if draft_core not in DRAFT_CORES:
        raise BenchmarkConfigurationError("unsupported draft core")
    if mtp_disabled_baseline and draft_core != "stock":
        raise BenchmarkConfigurationError(
            "MTP-disabled baseline requires the stock draft core"
        )
    if compiled_verify_mode != "off" and verify_strategy != "capture_commit":
        raise BenchmarkConfigurationError(
            "compiled verify requires capture_commit verify strategy"
        )
    observed_compiled_mode = _compiled_verify_environment_mode()
    if observed_compiled_mode != compiled_verify_mode:
        raise BenchmarkConfigurationError(
            "declared compiled verify mode differs from process environment"
        )
    normalized = [
        _normalized_request(
            request,
            mtp_disabled_baseline=mtp_disabled_baseline,
        )
        for request in model_requests
    ]
    if draft_core == "device-d2" and any(
        request["depths"] != (2,) for request in normalized
    ):
        raise BenchmarkConfigurationError(
            "legacy device-d2 requires every requested matrix to contain only D2"
        )
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
    options = {
        **DEFAULT_RUNTIME_OPTIONS,
        **dict(runtime_options or {}),
        "trace_routes": bool(trace_routes),
    }
    if options["powermetrics"] and not options["resource_telemetry"]:
        raise BenchmarkConfigurationError("powermetrics requires resource telemetry")
    if options["ssd_ceiling_gib_s"] is not None:
        if not options["resource_telemetry"]:
            raise BenchmarkConfigurationError("SSD ceiling requires resource telemetry")
        if not options["bypass_page_cache"]:
            raise BenchmarkConfigurationError("SSD ceiling requires F_NOCACHE")
        if float(options["ssd_ceiling_gib_s"]) <= 0:
            raise BenchmarkConfigurationError("SSD ceiling must be positive")
    if float(options["resource_sample_interval"]) <= 0:
        raise BenchmarkConfigurationError("resource sample interval must be positive")
    if int(options["resource_max_samples"]) < 2:
        raise BenchmarkConfigurationError("resource max samples must be at least 2")
    if max(context_values) + output_tokens > int(options["max_live_kv_tokens"]):
        raise BenchmarkConfigurationError(
            f"context plus {output_tokens} output tokens exceeds max_live_kv_tokens"
        )
    generation_environment = _generation_environment()
    apis = apis or _default_apis()
    sampler = apis.sampler_factory(temperature=0.0, top_p=1.0, top_k=1)
    lane = (
        "mtp-disabled-ar-baseline"
        if mtp_disabled_baseline
        else "mtp-resident-depth-matrix"
    )

    models: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "passed": False,
        "active_cell": {"phase": "setup"},
        "failure": None,
        "lane": lane,
        "mtp_resident": not mtp_disabled_baseline,
        "configuration": {
            "lane": lane,
            "mtp_resident": not mtp_disabled_baseline,
            "measurement_lane": (
                "diagnostic-resource-instrumented"
                if options["resource_telemetry"]
                else "headline-uninstrumented"
            ),
            "contexts": list(context_values),
            "output_tokens": output_tokens,
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
                "mtp_api": None if mtp_disabled_baseline else "generate_mtpk",
                "stop_token_ids": [],
                "repetition_stop": False,
                "loop_guard": False,
                "mtp_cache_policy": (None if mtp_disabled_baseline else "persistent"),
                "mtp_history_policy": (None if mtp_disabled_baseline else "committed"),
                "draft_core": None if mtp_disabled_baseline else draft_core,
                "capture_final_state": not mtp_disabled_baseline,
            },
            "generation_environment": generation_environment,
            "correctness_contract": {
                "ar_token_parity": "diagnostic",
                "cross_shape_token_divergence": "retained_unclassified_warning",
                "speculative_event_integrity": "hard_gate",
                "safe_final_state_and_cache_offsets": "hard_gate",
                "prompt_output_depth_counters_and_cache_metrics": "hard_gate",
            },
            "runtime": _jsonable(options),
            "candidate": {
                "verify_strategy": verify_strategy,
                "compiled_verify_mode": compiled_verify_mode,
                "draft_core": draft_core,
                "trace_routes": bool(trace_routes),
            },
        },
        "models": models,
    }
    _live_payload["payload"] = payload
    _emit_checkpoint(payload, checkpoint)
    for request in normalized:
        payload["active_cell"] = {
            "model": request["model"],
            "context_tokens": None,
            "depth": None,
            "cell": None,
            "phase": "configuration",
        }
        config = _runtime_config(apis, request["model_key"], options)
        observations: list[dict[str, Any]] = []
        prompts: list[dict[str, Any]] = []
        model_payload: dict[str, Any] = {
            "model": request["model"],
            "model_key": request["model_key"],
            "model_root": str(request["model_root"]),
            "manifest": str(request["manifest"]),
            "lane": lane,
            "mtp_resident": not mtp_disabled_baseline,
            "depths": list(request["depths"]),
            "load_count": 0,
            "measurement_lane": (
                "diagnostic-resource-instrumented"
                if options["resource_telemetry"]
                else "headline-uninstrumented"
            ),
            "load_peak_memory_bytes": None,
            "hard_peak_memory_bytes": None,
            "discarded_warmup_count": 0,
            "warmup_output_tokens": WARMUP_TOKENS,
            "runtime_config": _jsonable(config),
            "prompts": prompts,
            "observations": observations,
            "token_divergence_observations": [],
            "passed": False,
        }
        if not mtp_disabled_baseline:
            model_payload.update(
                {
                    "mtp_artifacts": str(request["mtp_artifacts"]),
                    "mtp_precision": request.get("mtp_precision", "bf16"),
                }
            )
        models.append(model_payload)
        payload["active_cell"] = {
            "model": request["model"],
            "context_tokens": None,
            "depth": None,
            "cell": None,
            "phase": "load",
        }
        load_kwargs = {
            "mtp": not mtp_disabled_baseline,
            "expert_streaming_config": config,
            "expert_manifest": request["manifest"],
        }
        if not mtp_disabled_baseline:
            load_kwargs.update(
                {
                    "mtp_artifacts": request["mtp_artifacts"],
                    "mtp_precision": request.get("mtp_precision", "bf16"),
                }
            )
        runtime = apis.load(request["model_root"], **load_kwargs)
        model_payload["load_count"] = 1
        try:
            runtime_mtp_enabled = getattr(runtime, "mtp_enabled", None)
            if mtp_disabled_baseline:
                if runtime_mtp_enabled is not False:
                    raise BenchmarkGateError(
                        f"{request['model']} baseline runtime unexpectedly enabled MTP"
                    )
            elif runtime_mtp_enabled is not True:
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
            model_payload["load_peak_memory_bytes"] = load_peak_memory_bytes
            model_payload["hard_peak_memory_bytes"] = hard_peak_memory_bytes
            discarded_warmup_count = 0
            for context_tokens in context_values:
                payload["active_cell"] = {
                    "model": request["model"],
                    "context_tokens": context_tokens,
                    "depth": None,
                    "cell": None,
                    "phase": "prompt",
                }
                prompt_options: dict[str, Any] = {
                    "prompt_style": "coding-agent",
                    "prompt_format": "chat",
                    "enable_thinking": False,
                }
                if request["prompt_tail_text"] is not None:
                    prompt_options["prompt_tail"] = request["prompt_tail_text"]
                prompt = apis.prompt_builder(
                    runtime.tokenizer,
                    context_tokens,
                    **prompt_options,
                )
                prompt_ids = tuple(int(token) for token in prompt.token_ids)
                if len(prompt_ids) != context_tokens:
                    raise BenchmarkGateError(
                        f"{request['model']} prompt builder returned {len(prompt_ids)} "
                        f"tokens for requested context {context_tokens}"
                    )
                metadata = _jsonable(getattr(prompt, "metadata", {}))
                if not isinstance(metadata, Mapping):
                    raise BenchmarkGateError(
                        f"{request['model']} prompt builder emitted no metadata"
                    )
                if metadata.get("prompt_tail_preserved") is not True:
                    raise BenchmarkGateError(
                        f"{request['model']} prompt builder did not preserve the tail"
                    )
                if (
                    metadata.get("prompt_policy") != "realistic_programming_v1"
                    or metadata.get("prompt_format") != "chat"
                    or metadata.get("prompt_release_valid") is not True
                ):
                    raise BenchmarkGateError(
                        f"{request['model']} prompt builder did not use the release "
                        "realistic programming chat policy"
                    )
                identity = _prompt_identity(prompt_ids, metadata)
                prompts.append(
                    {
                        "context_tokens": context_tokens,
                        **identity,
                        "builder_metadata": metadata,
                    }
                )
                payload["active_cell"] = {
                    "model": request["model"],
                    "context_tokens": context_tokens,
                    "depth": 0,
                    "cell": "ar",
                    "phase": "warmup",
                }
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
                    resource_options=options,
                    ar_tokens=None,
                    ar_finish_reason=None,
                    verify_strategy=verify_strategy,
                    compiled_verify_mode=compiled_verify_mode,
                    draft_core=draft_core,
                    draft_observer=draft_observer,
                )
                hard_peak_memory_bytes = max(
                    hard_peak_memory_bytes,
                    int(_warmup_ar_row["peak_memory_bytes"]),
                )
                discarded_warmup_count += 1
                model_payload["discarded_warmup_count"] = discarded_warmup_count
                model_payload["hard_peak_memory_bytes"] = hard_peak_memory_bytes
                payload["active_cell"] = {
                    "model": request["model"],
                    "context_tokens": context_tokens,
                    "depth": 0,
                    "cell": "ar",
                    "phase": "retained",
                }
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
                    max_tokens=output_tokens,
                    resource_options=options,
                    ar_tokens=None,
                    ar_finish_reason=None,
                    verify_strategy=verify_strategy,
                    compiled_verify_mode=compiled_verify_mode,
                    draft_core=draft_core,
                    draft_observer=draft_observer,
                )
                hard_peak_memory_bytes = max(
                    hard_peak_memory_bytes,
                    int(ar_row["peak_memory_bytes"]),
                )
                pair_id = f"{request['model']}-c{context_tokens}-ar"
                ar_row["pair_id"] = pair_id
                observations.append(ar_row)
                model_payload["hard_peak_memory_bytes"] = hard_peak_memory_bytes
                _emit_checkpoint(payload, checkpoint)
                for position, depth in enumerate(request["depths"], start=2):
                    payload["active_cell"] = {
                        "model": request["model"],
                        "context_tokens": context_tokens,
                        "depth": depth,
                        "cell": f"d{depth}",
                        "phase": "warmup",
                    }
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
                        resource_options=options,
                        ar_tokens=warmup_ar_tokens,
                        ar_finish_reason=warmup_ar_finish,
                        verify_strategy=verify_strategy,
                        compiled_verify_mode=compiled_verify_mode,
                        draft_core=draft_core,
                        draft_observer=draft_observer,
                    )
                    _record_token_divergence(
                        model_payload,
                        _warmup_row,
                        phase="warmup",
                    )
                    hard_peak_memory_bytes = max(
                        hard_peak_memory_bytes,
                        int(_warmup_row["peak_memory_bytes"]),
                    )
                    discarded_warmup_count += 1
                    model_payload["discarded_warmup_count"] = discarded_warmup_count
                    model_payload["hard_peak_memory_bytes"] = hard_peak_memory_bytes
                    payload["active_cell"] = {
                        "model": request["model"],
                        "context_tokens": context_tokens,
                        "depth": depth,
                        "cell": f"d{depth}",
                        "phase": "retained",
                    }
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
                        max_tokens=output_tokens,
                        resource_options=options,
                        ar_tokens=ar_tokens,
                        ar_finish_reason=ar_finish,
                        verify_strategy=verify_strategy,
                        compiled_verify_mode=compiled_verify_mode,
                        draft_core=draft_core,
                        draft_observer=draft_observer,
                    )
                    _record_token_divergence(
                        model_payload,
                        row,
                        phase="retained",
                    )
                    hard_peak_memory_bytes = max(
                        hard_peak_memory_bytes,
                        int(row["peak_memory_bytes"]),
                    )
                    row["pair_id"] = pair_id
                    observations.append(row)
                    model_payload["hard_peak_memory_bytes"] = hard_peak_memory_bytes
                    _emit_checkpoint(payload, checkpoint)
            model_payload["discarded_warmup_count"] = discarded_warmup_count
            model_payload["hard_peak_memory_bytes"] = hard_peak_memory_bytes
            model_payload["passed"] = True
            payload["active_cell"] = {
                "model": request["model"],
                "context_tokens": None,
                "depth": None,
                "cell": None,
                "phase": "model_complete",
            }
            _emit_checkpoint(payload, checkpoint)
        finally:
            close = getattr(runtime, "close", None)
            if callable(close):
                primary_error = sys.exc_info()[1]
                if primary_error is None:
                    payload["active_cell"] = {
                        "model": request["model"],
                        "context_tokens": None,
                        "depth": None,
                        "cell": None,
                        "phase": "close",
                    }
                try:
                    close()
                except Exception as close_error:
                    if primary_error is None:
                        raise
                    payload.setdefault("cleanup_errors", []).append(
                        {
                            "error": str(close_error),
                            "error_type": type(close_error).__name__,
                            "model": request["model"],
                        }
                    )

    payload["status"] = "passed"
    payload["passed"] = True
    payload["active_cell"] = None
    _emit_checkpoint(payload, checkpoint)
    return payload


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
    if args.kv_quant and args.verify_strategy == "capture_commit":
        parser.error(
            "--kv-quant requires --verify-strategy batched: capture_commit "
            "adapts dense trunk KV state and cannot wrap a quantized cache"
        )
    latest_checkpoint: dict[str, Any] | None = None

    def persist_checkpoint(snapshot: Mapping[str, Any]) -> None:
        nonlocal latest_checkpoint
        latest_checkpoint = copy.deepcopy(dict(snapshot))
        if args.output_json is not None:
            rendered_checkpoint = json.dumps(
                latest_checkpoint,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            _write_json_atomic(args.output_json, rendered_checkpoint)

    try:
        requests = _requests_from_args(args)
        runtime_options = _runtime_options_from_args(args)
        try:
            advisories = _profile_advisories(
                requests, runtime_options, verify_strategy=args.verify_strategy
            )
        except Exception as exc:  # advisory must never block a run
            advisories = [f"profile advisory computation failed: {exc}"]
        for line in advisories:
            print(line, file=sys.stderr)
        payload = run_depth_matrix(
            requests,
            contexts=args.contexts,
            output_tokens=args.output_tokens,
            runtime_options=runtime_options,
            mtp_disabled_baseline=args.mtp_disabled_baseline,
            verify_strategy=args.verify_strategy,
            compiled_verify_mode=args.compiled_verify_mode,
            draft_core=args.draft_core,
            trace_routes=args.trace_routes,
            checkpoint=persist_checkpoint,
            apis=apis,
        )
    except Exception as exc:
        if latest_checkpoint is None:
            failed = {
                "schema": SCHEMA,
                "passed": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        else:
            failed = latest_checkpoint
            active_cell = copy.deepcopy(failed.get("active_cell"))
            checkpoint_failure = failed.get("failure")
            checkpoint_evidence = (
                checkpoint_failure.get("evidence")
                if isinstance(checkpoint_failure, Mapping)
                else None
            )
            failed["status"] = "failed"
            failed["passed"] = False
            failed["failure"] = {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "active_cell": active_cell,
            }
            if isinstance(checkpoint_evidence, Mapping):
                failed["failure"]["evidence"] = _jsonable(checkpoint_evidence)
        rendered_failure = json.dumps(
            failed,
            indent=2 if latest_checkpoint is not None else None,
            sort_keys=True,
            allow_nan=False,
        )
        if args.output_json is not None:
            _write_json_atomic(args.output_json, rendered_failure)
        print(rendered_failure)
        return 1
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output_json is not None and latest_checkpoint is None:
        _write_json_atomic(args.output_json, rendered)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
