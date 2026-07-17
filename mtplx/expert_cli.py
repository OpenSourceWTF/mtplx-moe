"""Shared CLI plumbing for opt-in SSD expert streaming."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .expert_runtime import ExpertStreamingConfig, parse_memory_bytes


_BYTE_FIELDS = {
    "memory_limit_bytes",
    "runtime_reserve_bytes",
    "expert_cache_limit_bytes",
    "io_staging_bytes",
    "execution_workspace_bytes",
    "max_inflight_io_bytes",
    "max_read_chunk_bytes",
}


def add_expert_streaming_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("SSD expert streaming")
    group.add_argument(
        "--expert-streaming",
        action="store_true",
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
            "glm52-expert-q2",
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
    return bool(
        getattr(args, "expert_streaming", False)
        or getattr(args, "expert_streaming_config", None)
        or getattr(args, "expert_manifest", None)
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
    normalized = dict(values)
    for field in _BYTE_FIELDS:
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = parse_memory_bytes(value)
    return normalized


def expert_streaming_load_kwargs(
    args: Any,
    model_path: Path | str,
) -> dict[str, Any]:
    """Build validated ``runtime.load`` kwargs or return an empty mapping."""

    if not expert_streaming_requested(args):
        return {}
    root = Path(model_path).resolve()
    values = _load_config_object(getattr(args, "expert_streaming_config", None))
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
        "execution_workspace_bytes": getattr(args, "expert_execution_workspace", None),
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
    values.update({key: value for key, value in overrides.items() if value is not None})
    if "model_key" not in values:
        values["model_key"] = _read_model_key(root)
    values.setdefault("runtime_reserve_bytes", 16 * 1024**3)
    values.setdefault("io_staging_bytes", 0)
    values.setdefault("execution_workspace_bytes", 0)
    missing = [
        name
        for name in ("memory_limit_bytes", "max_live_kv_tokens")
        if name not in values
    ]
    if missing:
        flags = ", ".join(name.replace("_", "-") for name in missing)
        raise ValueError(
            "expert streaming requires explicit memory/KV admission limits; "
            f"missing {flags}"
        )
    values = _normalize_byte_fields(values)
    try:
        config = ExpertStreamingConfig(**values)
    except TypeError as exc:
        raise ValueError(f"invalid expert streaming config: {exc}") from exc
    manifest = Path(
        getattr(args, "expert_manifest", None) or root / "expert-manifest.json"
    ).resolve()
    if not manifest.is_file():
        raise ValueError(f"expert manifest does not exist: {manifest}")
    return {
        "mtp": False,
        "expert_streaming_config": config,
        "expert_manifest": manifest,
    }


def append_expert_streaming_child_args(command: list[str], args: Any) -> None:
    """Forward public ``mtplx serve`` expert flags to the daemon child."""

    if not expert_streaming_requested(args):
        return
    command.append("--expert-streaming")
    mappings = (
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
        if value is not None:
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
