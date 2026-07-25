"""Shared CLI plumbing for opt-in SSD expert streaming."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .expert_profiles import ExpertServeProfile
    from .expert_runtime import ExpertStreamingConfig
    from .expert_streaming_models import ExpertMemoryPlan


_BYTE_FIELDS = {
    "memory_limit_bytes",
    "runtime_reserve_bytes",
    "expert_cache_limit_bytes",
    "io_staging_bytes",
    "execution_workspace_bytes",
    "max_inflight_io_bytes",
    "max_read_chunk_bytes",
}

# The manifest's top-level ``model_key`` is a small scalar that precedes the
# large ``records`` array in every serialization order, so a bounded head scan
# lifts it out without a full parse or the digest/record verification pass.
_MODEL_KEY_RE = re.compile(rb'"model_key"\s*:\s*"([^"]+)"')
_MANIFEST_MODEL_KEY_SCAN_BYTES = 1 * 1024 * 1024

# RAM-derived memory-limit policy, mirrored from the non-streaming Metal caps
# in mtplx/server/openai.py: 75% of installed RAM, floored at 8GiB and capped
# at 192GiB. The GPU wired budget is engineered UNDER installed RAM and must
# never exceed it, so this fraction/cap is a hard ceiling, not a suggestion.
_DEFAULT_MEMORY_FRACTION = 0.75
_MIN_MEMORY_LIMIT_BYTES = 8 * 1024**3
_MAX_MEMORY_LIMIT_BYTES = 192 * 1024**3

# Sane KV-admission ceiling when the artifact's config.json declares no
# context window. Derived max_live_kv_tokens is clamped to this so a very large
# unified-memory box does not reserve an absurd admission budget.
_DEFAULT_CONTEXT_CEILING = 131072
_CONTEXT_WINDOW_CONFIG_KEYS = (
    "max_position_embeddings",
    "max_sequence_length",
    "seq_length",
)

# Out-of-box KV-admission target for the derived default. For an SSD-streaming
# MoE the per-layer expert-slot count is the dominant decode-throughput lever,
# so the default deliberately caps KV reservation here instead of maximizing it:
# maximizing KV would spend nearly the whole envelope on KV and starve the
# expert cache down to a single slot per layer. This value matches the
# README-tested streaming profile (156 slots/layer at a 96 GiB envelope for
# hy3-expert-oq2e) and trades context headroom for expert-cache residency.
# Users who need more context can pass --expert-max-live-kv-tokens explicitly
# (which reserves more KV and yields slower decode).
_DEFAULT_KV_TOKENS = 32768
EXPERT_PROFILE_CHOICES = (
    "auto",
    "hy3-oq2e-64",
    "hy3-oq2e-88",
    "hy3-oq2e-96",
)
_PROFILE_EFFECTIVE_FIELDS = (
    "memory_limit_bytes",
    "max_live_kv_tokens",
    "runtime_reserve_bytes",
    "expert_cache_limit_bytes",
    "transient_slots",
    "io_staging_bytes",
    "execution_workspace_bytes",
    "max_inflight_io_bytes",
    "max_open_files",
    "max_read_chunk_bytes",
    "frequency_decay",
    "prefer_sidecar",
    "verify_record_hashes",
    "verify_artifact_headers",
    "verify_sidecar_hash_at_open",
    "prefill_admission",
    "slot_layout",
    "trace_routes",
    "cache_policy",
    "cache_scope",
    "bypass_page_cache",
    "resource_telemetry",
    "q2_expert_kernel",
    "hy3_router_kernel",
    "hy3_router_sigmoid",
    "hy3_mtp_shared_kernel",
    "hy3_mtp_shared_kernel_depth",
    "proj_quant",
    "proj_requant",
    "kv_quant",
    "split_route_release",
    "deferred_pin_release",
    "island_layers",
    "island_layer_count",
    "mmap_island_layers",
    "banked_codec",
    "streamed_codec",
    "streamed_codec_verify",
    "mmap_island_wired",
    "overlap_miss_reads",
    "prefetch_slots",
    "speculative_io_fraction",
    "route_census",
    "miss_shadow",
    "miss_shadow_layers",
)
_EXPERT_STREAMING_POSITIVE_FLAGS = (
    "expert-streaming",
    "expert-streaming-config",
    "expert-manifest",
    "expert-model-key",
    "expert-memory-limit",
    "expert-max-live-kv-tokens",
    "expert-runtime-reserve",
    "expert-cache-limit",
    "expert-cache-policy",
    "expert-cache-scope",
    "expert-transient-slots",
    "expert-io-staging",
    "expert-execution-workspace",
    "expert-max-inflight-io",
    "expert-max-open-files",
    "expert-read-chunk",
    "expert-f-nocache",
    "expert-slot-layout",
    "expert-frequency-decay",
    "expert-prefer-sidecar",
    "expert-verify-record-hashes",
    "expert-verify-headers",
    "expert-verify-sidecar-at-open",
)


def ensure_expert_admitted(root: Path) -> dict[str, Any]:
    """Load the heavy artifact verifier only for an enabled streaming route."""

    from .expert_admission import ensure_expert_admitted as admit

    return admit(root)


def load_expert_profiles() -> dict[str, ExpertServeProfile]:
    """Load profile/runtime modules only when profile resolution is requested."""

    from .expert_profiles import load_expert_profiles as load

    return load()


def select_expert_profile(
    requested: str,
    *,
    model_key: str,
) -> ExpertServeProfile:
    """Resolve a profile without loading runtime modules during parser setup."""

    from .expert_profiles import select_expert_profile as select

    return select(requested, model_key=model_key)


def add_expert_streaming_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("SSD expert streaming")
    group.add_argument(
        "--expert-profile",
        choices=EXPERT_PROFILE_CHOICES,
        default="auto",
        help="Promoted SSD expert memory profile (default: auto).",
    )
    group.add_argument(
        "--expert-streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load only resident weights and stream routed quantized experts from SSD. "
            "This selects target-only AR for the pinned Hy3/GLM artifacts."
        ),
    )
    group.add_argument(
        "--expert-streaming-config",
        help="JSON ExpertStreamingConfig; explicit flags below override its values.",
    )
    group.add_argument(
        "--expert-manifest",
        help="Manifest path (default: MODEL/expert-manifest.json).",
    )
    group.add_argument(
        "--expert-model-key",
        choices=[
            "hy3-q4",
            "glm52-q4",
            "hy3-expert-only-q4",
            "hy3-expert-q2",
            "hy3-expert-oq2e",
            "hy3-expert-mixofficial",
            "glm52-expert-q2",
            "glm52-expert-q1t",
            "glm52-expert-q1b1",
        ],
        help="Pinned streamed model descriptor; inferred from config.json by default.",
    )
    group.add_argument(
        "--expert-memory-limit",
        help="Total process memory ceiling, for example 96GiB or 320GiB.",
    )
    group.add_argument(
        "--expert-max-live-kv-tokens",
        type=int,
        help="Aggregate live KV-token admission ceiling reserved in the memory plan.",
    )
    group.add_argument(
        "--expert-runtime-reserve", help="Runtime/OS headroom (default 16GiB)."
    )
    group.add_argument(
        "--expert-cache-limit",
        help=(
            "Optional static persistent expert-cache cap. When omitted, the "
            "cache allowance is derived from --expert-memory-limit at every "
            "KV admission boundary."
        ),
    )
    group.add_argument(
        "--expert-cache-policy",
        choices=["frequency", "lru"],
        help="Decode expert-cache replacement policy.",
    )
    group.add_argument(
        "--expert-cache-scope",
        choices=["layer", "global"],
        help="Use fixed per-layer banks or one global expert-record pool.",
    )
    group.add_argument("--expert-transient-slots", type=int)
    group.add_argument("--expert-io-staging", help="Host I/O staging reserve.")
    group.add_argument(
        "--expert-execution-workspace", help="Execution workspace reserve."
    )
    group.add_argument(
        "--expert-max-inflight-io", help="Bound concurrent expert-read bytes."
    )
    group.add_argument("--expert-max-open-files", type=int)
    group.add_argument("--expert-read-chunk", help="Maximum positional read chunk.")
    group.add_argument(
        "--expert-f-nocache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Bypass the macOS page cache for expert reads.",
    )
    group.add_argument(
        "--expert-slot-layout",
        choices=["direct-slots", "component-banks", "metal-mmap"],
    )
    group.add_argument("--expert-frequency-decay", type=float)
    group.add_argument(
        "--expert-prefer-sidecar",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    group.add_argument(
        "--expert-verify-record-hashes",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    group.add_argument(
        "--expert-verify-headers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    group.add_argument(
        "--expert-verify-sidecar-at-open",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def expert_streaming_requested(args: Any) -> bool:
    profile = str(getattr(args, "expert_profile", "auto") or "auto")
    cli_flags = set(getattr(args, "_cli_flags", set()) or set())
    if "no-expert-streaming" in cli_flags:
        positive_selector = None
        if (
            "expert-streaming" in cli_flags
            or getattr(args, "expert_streaming", False)
        ):
            positive_selector = "--expert-streaming"
        elif profile != "auto":
            positive_selector = "--expert-profile"
        elif (
            "expert-streaming-config" in cli_flags
            or getattr(args, "expert_streaming_config", None)
        ):
            positive_selector = "--expert-streaming-config"
        elif (
            "expert-manifest" in cli_flags
            or getattr(args, "expert_manifest", None)
        ):
            positive_selector = "--expert-manifest"
        if positive_selector is None:
            positive_flag = next(
                (
                    flag
                    for flag in _EXPERT_STREAMING_POSITIVE_FLAGS
                    if flag in cli_flags
                ),
                None,
            )
            if positive_flag is not None:
                positive_selector = f"--{positive_flag}"
        if positive_selector is not None:
            raise ValueError(
                f"{positive_selector} cannot be combined with "
                "--no-expert-streaming"
            )
    return bool(
        getattr(args, "expert_streaming", False)
        or getattr(args, "expert_streaming_config", None)
        or getattr(args, "expert_manifest", None)
        or profile != "auto"
    )


def _read_model_key(model_path: Path) -> str:
    config_path = model_path / "config.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            "--expert-model-key is required when config.json cannot be read"
        ) from exc
    model_type = str(data.get("model_type") or "")
    try:
        return {"hy_v3": "hy3-q4", "glm_moe_dsa": "glm52-q4"}[model_type]
    except KeyError as exc:
        raise ValueError(
            f"cannot infer streamed model key from model_type={model_type!r}"
        ) from exc


def _read_manifest_model_key(manifest_path: Path) -> str | None:
    """Return the manifest's top-level ``model_key``, or ``None`` if absent.

    The manifest is read once through the shared bounded artifact descriptor
    path. A head scan handles published ordering cheaply; ``json.loads`` over
    those already-bounded bytes is the fallback for unusual field ordering.
    Neither path triggers digest/record verification — this only lifts one
    field.
    """

    from .expert_manifest import MAX_MANIFEST_BYTES
    from .hf_loader import read_bounded_artifact_member

    try:
        payload = read_bounded_artifact_member(
            manifest_path.parent,
            manifest_path.name,
            max_bytes=MAX_MANIFEST_BYTES,
        )
    except (OSError, ValueError):
        return None
    head = payload[:_MANIFEST_MODEL_KEY_SCAN_BYTES]
    match = _MODEL_KEY_RE.search(head)
    if match is not None:
        return match.group(1).decode("utf-8")
    try:
        data = json.loads(payload)
    except (UnicodeDecodeError, ValueError):
        return None
    if isinstance(data, dict):
        value = data.get("model_key")
        if isinstance(value, str) and value:
            return value
    return None


def _resolve_model_key(model_root: Path, manifest_path: Path) -> str:
    """Resolve the streamed spec key: manifest-authoritative, config fallback.

    The published streaming artifacts declare the exact registered spec key in
    the manifest's top-level ``model_key``. It is authoritative when present —
    it disambiguates banks (oQ2e, t158) that share a ``config.json``
    ``model_type`` with the production q4 default. Only when no manifest
    ``model_key`` can be read does the coarse ``model_type`` map apply, which
    preserves the safe q4 default for bare artifacts.
    """

    manifest_key = _read_manifest_model_key(manifest_path)
    if manifest_key:
        return manifest_key
    return _read_model_key(model_root)


def _installed_ram_bytes() -> int:
    """Installed unified memory in bytes; ``0`` when it cannot be determined."""

    from .hardware import total_memory_gib

    return int(total_memory_gib() * (1024**3))


def _derive_memory_limit_bytes() -> int:
    """Default the process memory ceiling from installed RAM."""

    total = _installed_ram_bytes()
    if total <= 0:
        raise ValueError(
            "could not detect installed RAM to derive an expert-streaming "
            "memory limit; pass --expert-memory-limit explicitly"
        )
    return min(
        total,
        max(_MIN_MEMORY_LIMIT_BYTES, int(total * _DEFAULT_MEMORY_FRACTION)),
        _MAX_MEMORY_LIMIT_BYTES,
    )


def _declared_context_window(model_root: Path) -> int:
    """Artifact-declared context window, or a sane fallback ceiling."""

    try:
        data = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _DEFAULT_CONTEXT_CEILING
    if isinstance(data, dict):
        for key in _CONTEXT_WINDOW_CONFIG_KEYS:
            value = data.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return _DEFAULT_CONTEXT_CEILING


def _derive_max_live_kv_tokens(
    values: Mapping[str, Any],
    model_root: Path,
) -> int:
    """Largest KV admission ceiling that still fits under the memory limit.

    Runs the authoritative memory planner backwards from the resolved memory
    limit: the derived ceiling is the largest ``max_live_kv_tokens`` whose plan
    still leaves resident weights, the runtime reserve, transient service, and
    at least a minimal expert slot bank inside the limit, clamped to the smaller
    of the artifact's declared context window and ``_DEFAULT_KV_TOKENS``. The
    ``_DEFAULT_KV_TOKENS`` clamp keeps the out-of-box default from spending the
    whole envelope on KV and starving the expert cache to one slot per layer;
    on a normal machine the search therefore returns ``_DEFAULT_KV_TOKENS`` and
    the planner instead maximizes the expert slot bank, while a memory-tight
    machine still gets the largest viable value below it. Reuses
    :func:`plan_expert_memory` so it can never disagree with the plan that
    actually admits KV at serve time.
    """

    from .expert_streaming_models import get_model_spec, plan_expert_memory

    spec = get_model_spec(str(values["model_key"]))
    memory_limit_bytes = int(values["memory_limit_bytes"])
    runtime_reserve_bytes = int(values.get("runtime_reserve_bytes", 16 * 1024**3))
    io_staging_bytes = int(values.get("io_staging_bytes", 0))
    execution_workspace_bytes = int(values.get("execution_workspace_bytes", 0))
    transient_slots = values.get("transient_slots")
    kv_quant = values.get("kv_quant")
    cache_scope = values.get("cache_scope", "layer")

    def _plan(context_tokens: int) -> ExpertMemoryPlan:
        return plan_expert_memory(
            spec,
            total_limit_bytes=memory_limit_bytes,
            context_tokens=context_tokens,
            runtime_reserve_bytes=runtime_reserve_bytes,
            transient_slots=transient_slots,
            io_staging_bytes=io_staging_bytes,
            execution_workspace_bytes=execution_workspace_bytes,
            kv_quant=kv_quant,
            cache_scope=cache_scope,
        )

    def _viable(plan: ExpertMemoryPlan) -> bool:
        # Fits the fixed footprint AND leaves room for at least one persistent
        # expert slot per streamed layer (a minimal, usable cache bank).
        return plan.fits_fixed and plan.slots_per_layer >= 1

    floor_plan = _plan(0)
    if not _viable(floor_plan):
        raise ValueError(
            f"the derived expert-streaming memory limit ({memory_limit_bytes} "
            f"bytes) is too small for {spec.key}: resident weights "
            f"({floor_plan.resident_bytes} bytes) + runtime reserve "
            f"({runtime_reserve_bytes} bytes) + transient service already "
            f"require {floor_plan.fixed_bytes} bytes with no room left for a "
            "minimal expert slot bank. Free RAM or pass --expert-memory-limit "
            "and --expert-max-live-kv-tokens explicitly."
        )

    ceiling = min(_declared_context_window(model_root), _DEFAULT_KV_TOKENS)
    low, high = 0, ceiling
    while low < high:
        mid = (low + high + 1) // 2
        if _viable(_plan(mid)):
            low = mid
        else:
            high = mid - 1
    if low < 1:
        raise ValueError(
            f"the derived expert-streaming memory limit ({memory_limit_bytes} "
            f"bytes) leaves no KV admission budget for {spec.key} after "
            "resident weights, runtime reserve, and a minimal expert slot "
            "bank. This machine is too small; free RAM or pass "
            "--expert-max-live-kv-tokens explicitly."
        )
    return low


def _load_config_object(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"could not read expert streaming config {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("expert streaming config must contain one JSON object")
    return dict(value)


def _normalize_byte_fields(values: Mapping[str, Any]) -> dict[str, Any]:
    from .expert_runtime import parse_memory_bytes

    normalized = dict(values)
    for field in _BYTE_FIELDS:
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = parse_memory_bytes(value)
    return normalized


def _authoritative_manifest_path(root: Path) -> Path:
    from .expert_manifest import resolve_artifact_member

    return resolve_artifact_member(root, "expert-manifest.json")


def _validate_explicit_manifest(
    root_manifest: Path,
    explicit_manifest: Path,
) -> None:
    from .expert_manifest import load_expert_manifest

    if explicit_manifest == root_manifest:
        return
    try:
        root_digest = load_expert_manifest(root_manifest).manifest_sha256
        explicit_digest = load_expert_manifest(explicit_manifest).manifest_sha256
    except Exception as exc:
        raise ValueError(
            "--expert-manifest must be exactly or digest-equivalent to the "
            "admitted root manifest"
        ) from exc
    if not root_digest or explicit_digest != root_digest:
        raise ValueError(
            "--expert-manifest must be exactly or digest-equivalent to the "
            "admitted root manifest"
        )


def _explicit_overrides(args: Any) -> dict[str, Any]:
    overrides = {
        "model_key": getattr(args, "expert_model_key", None),
        "memory_limit_bytes": getattr(args, "expert_memory_limit", None),
        "max_live_kv_tokens": getattr(args, "expert_max_live_kv_tokens", None),
        "runtime_reserve_bytes": getattr(args, "expert_runtime_reserve", None),
        "expert_cache_limit_bytes": getattr(args, "expert_cache_limit", None),
        "cache_policy": getattr(args, "expert_cache_policy", None),
        "cache_scope": getattr(args, "expert_cache_scope", None),
        "transient_slots": getattr(args, "expert_transient_slots", None),
        "io_staging_bytes": getattr(args, "expert_io_staging", None),
        "execution_workspace_bytes": getattr(
            args, "expert_execution_workspace", None
        ),
        "max_inflight_io_bytes": getattr(args, "expert_max_inflight_io", None),
        "max_open_files": getattr(args, "expert_max_open_files", None),
        "max_read_chunk_bytes": getattr(args, "expert_read_chunk", None),
        "bypass_page_cache": getattr(args, "expert_f_nocache", None),
        "slot_layout": getattr(args, "expert_slot_layout", None),
        "frequency_decay": getattr(args, "expert_frequency_decay", None),
        "prefer_sidecar": getattr(args, "expert_prefer_sidecar", None),
        "verify_record_hashes": getattr(args, "expert_verify_record_hashes", None),
        "verify_artifact_headers": getattr(args, "expert_verify_headers", None),
        "verify_sidecar_hash_at_open": getattr(
            args, "expert_verify_sidecar_at_open", None
        ),
    }
    return {key: value for key, value in overrides.items() if value is not None}


def _profile_for_model_key(
    args: Any,
    *,
    model_key: str,
) -> ExpertServeProfile | None:
    requested = str(getattr(args, "expert_profile", "auto") or "auto")
    promoted_model_keys = {
        profile.model_key for profile in load_expert_profiles().values()
    }
    if requested == "auto" and model_key not in promoted_model_keys:
        return None
    return select_expert_profile(requested, model_key=model_key)


def resolve_expert_profile_for_args(
    args: Any,
    model_path: Path | str,
    *,
    model_key: str | None = None,
) -> ExpertServeProfile:
    root = Path(model_path).resolve()
    resolved_model_key = model_key
    if resolved_model_key is None:
        resolved_model_key = _resolve_model_key(
            root, _authoritative_manifest_path(root)
        )
    return select_expert_profile(
        str(getattr(args, "expert_profile", "auto") or "auto"),
        model_key=resolved_model_key,
    )


def apply_expert_profile_child_env(
    args: Any,
    environ: dict[str, str],
) -> None:
    profile = getattr(args, "_resolved_expert_profile", None)
    if profile is not None:
        environ.update(dict(profile.child_env))


def _apply_diagnostic_hash_policy(
    args: Any,
    config: ExpertStreamingConfig,
) -> ExpertStreamingConfig:
    cli_flags = set(getattr(args, "_cli_flags", set()) or set())
    record_hashes = (
        "expert-verify-record-hashes" in cli_flags
        and getattr(args, "expert_verify_record_hashes", None) is True
    )
    sidecar_hash = (
        "expert-verify-sidecar-at-open" in cli_flags
        and getattr(args, "expert_verify_sidecar_at_open", None) is True
    )
    changes: dict[str, Any] = {
        "verify_record_hashes": record_hashes,
        "verify_sidecar_hash_at_open": sidecar_hash,
    }
    if config.island_layers:
        changes["island_layer_count"] = None
    return dataclasses.replace(config, **changes)


def _profile_customization(
    profile: ExpertServeProfile,
    config: ExpertStreamingConfig,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    from .expert_profiles import build_expert_streaming_config

    baseline = build_expert_streaming_config(profile)
    baseline_changes: dict[str, Any] = {
        "verify_record_hashes": False,
        "verify_sidecar_hash_at_open": False,
    }
    if baseline.island_layers:
        baseline_changes["island_layer_count"] = None
    baseline = dataclasses.replace(baseline, **baseline_changes)
    changed = tuple(
        field.name
        for field in dataclasses.fields(config)
        if getattr(config, field.name) != getattr(baseline, field.name)
    )
    if not changed:
        return (), {}
    effective: dict[str, Any] = {}
    for name in _PROFILE_EFFECTIVE_FIELDS:
        value = getattr(config, name)
        effective[name] = list(value) if isinstance(value, tuple) else value
    return changed, effective


def expert_streaming_load_kwargs(
    args: Any,
    model_path: Path | str,
) -> dict[str, Any]:
    """Build validated ``runtime.load`` kwargs or return an empty mapping."""

    if not expert_streaming_requested(args):
        return {}
    from .expert_profiles import build_expert_streaming_config
    from .expert_runtime import ExpertStreamingConfig, resolve_island_placement

    cli_flags = set(getattr(args, "_cli_flags", set()) or set())
    if (
        (
            "generation-mode" in cli_flags
            and str(getattr(args, "generation_mode", "") or "").strip().lower()
            == "mtp"
        )
        or "mtp" in cli_flags
        or (
            "load-mtp" in cli_flags
            and getattr(args, "load_mtp", True) is True
        )
    ):
        raise ValueError(
            "promoted streamed profiles are AR-only in the OpenSourceWTF "
            "MTPLX-MOE fork"
        )
    root = Path(model_path).resolve()
    receipt = ensure_expert_admitted(root)
    manifest = _authoritative_manifest_path(root)
    explicit_manifest = getattr(args, "expert_manifest", None)
    if explicit_manifest:
        _validate_explicit_manifest(
            manifest,
            Path(explicit_manifest).expanduser().resolve(),
        )
    model_key = _resolve_model_key(root, manifest)
    values = _load_config_object(getattr(args, "expert_streaming_config", None))
    overrides = _explicit_overrides(args)
    configured_model_key = overrides.pop(
        "model_key", values.pop("model_key", None)
    )
    if configured_model_key is not None and configured_model_key != model_key:
        raise ValueError(
            f"expert model key {configured_model_key!r} does not match "
            f"admitted manifest model key {model_key!r}"
        )
    profile = _profile_for_model_key(args, model_key=model_key)
    customized_fields: tuple[str, ...] = ()
    effective_config: dict[str, Any] = {}
    if profile is not None:
        profile_overrides = dict(values)
        profile_overrides.update(overrides)
        try:
            config = build_expert_streaming_config(
                profile,
                overrides=profile_overrides,
            )
        except TypeError as exc:
            raise ValueError(
                f"invalid expert profile overrides: {exc}"
            ) from exc
        setattr(args, "expert_profile", profile.name)
    else:
        values["model_key"] = model_key
        values.update(overrides)
        values.setdefault("runtime_reserve_bytes", 16 * 1024**3)
        values.setdefault("io_staging_bytes", 0)
        values.setdefault("execution_workspace_bytes", 0)
        values = _normalize_byte_fields(values)
        if "memory_limit_bytes" not in values:
            values["memory_limit_bytes"] = _derive_memory_limit_bytes()
        if "max_live_kv_tokens" not in values:
            values["max_live_kv_tokens"] = _derive_max_live_kv_tokens(values, root)
        try:
            config = ExpertStreamingConfig(**values)
        except TypeError as exc:
            raise ValueError(f"invalid expert streaming config: {exc}") from exc
        config = resolve_island_placement(config, root)
    config = _apply_diagnostic_hash_policy(args, config)
    if profile is not None:
        customized_fields, effective_config = _profile_customization(
            profile,
            config,
        )
    if not manifest.is_file():
        raise ValueError(f"expert manifest does not exist: {manifest}")
    setattr(args, "_resolved_expert_profile", profile)
    setattr(
        args,
        "_resolved_expert_profile_customized",
        bool(customized_fields),
    )
    setattr(
        args,
        "_resolved_expert_profile_customized_fields",
        customized_fields,
    )
    setattr(args, "_resolved_expert_effective_config", effective_config)
    setattr(args, "_expert_admission_receipt", receipt)
    return {
        "mtp": False,
        "expert_streaming_config": config,
        "expert_manifest": manifest,
        "expert_admission_receipt": receipt,
    }


def append_expert_streaming_child_args(command: list[str], args: Any) -> None:
    """Forward public ``mtplx serve`` expert flags to the daemon child."""

    if not expert_streaming_requested(args):
        return
    command.append("--expert-streaming")
    mappings = (
        ("expert_profile", "--expert-profile"),
        ("expert_streaming_config", "--expert-streaming-config"),
        ("expert_manifest", "--expert-manifest"),
        ("expert_model_key", "--expert-model-key"),
        ("expert_memory_limit", "--expert-memory-limit"),
        ("expert_max_live_kv_tokens", "--expert-max-live-kv-tokens"),
        ("expert_runtime_reserve", "--expert-runtime-reserve"),
        ("expert_cache_limit", "--expert-cache-limit"),
        ("expert_cache_policy", "--expert-cache-policy"),
        ("expert_cache_scope", "--expert-cache-scope"),
        ("expert_transient_slots", "--expert-transient-slots"),
        ("expert_io_staging", "--expert-io-staging"),
        ("expert_execution_workspace", "--expert-execution-workspace"),
        ("expert_max_inflight_io", "--expert-max-inflight-io"),
        ("expert_max_open_files", "--expert-max-open-files"),
        ("expert_read_chunk", "--expert-read-chunk"),
        ("expert_slot_layout", "--expert-slot-layout"),
        ("expert_frequency_decay", "--expert-frequency-decay"),
    )
    for attribute, flag in mappings:
        value = getattr(args, attribute, None)
        if value is not None and not (
            attribute == "expert_profile" and value == "auto"
        ):
            if attribute in {"expert_streaming_config", "expert_manifest"}:
                value = Path(value).expanduser().resolve()
            command.extend([flag, str(value)])
    for attribute, positive, negative in (
        (
            "expert_prefer_sidecar",
            "--expert-prefer-sidecar",
            "--no-expert-prefer-sidecar",
        ),
        (
            "expert_verify_record_hashes",
            "--expert-verify-record-hashes",
            "--no-expert-verify-record-hashes",
        ),
        (
            "expert_verify_headers",
            "--expert-verify-headers",
            "--no-expert-verify-headers",
        ),
        (
            "expert_verify_sidecar_at_open",
            "--expert-verify-sidecar-at-open",
            "--no-expert-verify-sidecar-at-open",
        ),
        (
            "expert_f_nocache",
            "--expert-f-nocache",
            "--no-expert-f-nocache",
        ),
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            command.append(positive if value else negative)
