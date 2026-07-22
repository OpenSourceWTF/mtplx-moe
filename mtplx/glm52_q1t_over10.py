"""Construction-only settings contract for GLM-5.2 Q1T fused rANS.

The module stays MLX-free so configuration can fail before model or GPU
allocation.  It deliberately contains no identity for the discarded
per-record rANS transport artifact and no benchmark-qualification receipt.
Artifact identity, component geometry, mapped-container integrity, kernel
support, and exact output self-checks are validated by the fused runtime
constructor.
"""

from __future__ import annotations

import os
from types import MappingProxyType
from typing import Any


GLM52_Q1T_BASE_MANIFEST_SHA256 = (
    "5017b201e858931cfeafa466a8173e685af8dfcd3a47679bccd491d6a7122dd1"
)
GLM52_Q1T_FUSED_RANS_CODEC = "rans32x-uniform-packed-v1"
GLM52_Q1T_MAX_TOTAL_MEMORY_BYTES = 96 * 1024**3
GLM52_Q1T_RUNTIME_RESERVE_BYTES = 12 * 1024**3
GLM52_Q1T_EXPERT_CACHE_BYTES = 72 * 1024**3
GLM52_Q1T_TRANSIENT_SLOTS = 48
GLM52_Q1T_PERSISTENT_SLOTS_PER_LAYER = 116
GLM52_Q1T_MAX_LIVE_KV_TOKENS = 4096
GLM52_Q1T_READ_CHUNK_BYTES = 8 * 1024**2
GLM52_Q1T_FUSED_RANS_GATE_UP_THREADGROUPS = 64

_REQUIRED_CONFIG = MappingProxyType(
    {
        "model_key": "glm52-expert-q1t",
        "slot_layout": "fused-rans",
        "memory_limit_bytes": GLM52_Q1T_MAX_TOTAL_MEMORY_BYTES,
        "runtime_reserve_bytes": GLM52_Q1T_RUNTIME_RESERVE_BYTES,
        "expert_cache_limit_bytes": GLM52_Q1T_EXPERT_CACHE_BYTES,
        "transient_slots": GLM52_Q1T_TRANSIENT_SLOTS,
        "max_live_kv_tokens": GLM52_Q1T_MAX_LIVE_KV_TOKENS,
        "cache_policy": "frequency",
        "cache_scope": "layer",
        "deferred_pin_release": True,
        "split_route_release": "deferred",
        "max_read_chunk_bytes": GLM52_Q1T_READ_CHUNK_BYTES,
        "banked_codec": GLM52_Q1T_FUSED_RANS_CODEC,
        "streamed_codec": "none",
        "streamed_codec_manifest": None,
        "streamed_codec_verify": False,
        "verify_record_hashes": False,
        "verify_sidecar_hash_at_open": False,
        "island_layers": (),
        "mmap_island_layers": (),
        "miss_shadow": None,
        "prefetch_slots": 0,
        "route_census": False,
        "resource_telemetry": False,
        "trace_routes": False,
        "q2_expert_kernel": "stock",
    }
)


def validate_glm52_q1t_fused_rans_config(config: Any) -> None:
    """Reject an incomplete or cross-family fused route before allocation."""

    mismatches = [
        f"{name}={getattr(config, name, None)!r}, expected {expected!r}"
        for name, expected in _REQUIRED_CONFIG.items()
        if getattr(config, name, None) != expected
    ]
    if not getattr(config, "banked_manifest", None):
        mismatches.append("banked_manifest is required")
    if mismatches:
        raise ValueError(
            "glm52-expert-q1t fused-rans construction contract failed: "
            + "; ".join(mismatches)
        )


def _memory_limit_setter(mx_module: Any, name: str) -> Any:
    setter = getattr(mx_module, name, None)
    if callable(setter):
        return setter
    setter = getattr(getattr(mx_module, "metal", None), name, None)
    if callable(setter):
        return setter
    raise RuntimeError(f"GLM fused-rANS requires MLX {name}")


def apply_glm52_q1t_fused_rans_memory_caps(
    plan: Any,
    *,
    mx_module: Any,
    env: dict[str, str] | None = None,
    external_residency_bytes: int = 0,
) -> dict[str, int]:
    """Apply the GLM lane's allocator and wired caps before GPU allocation."""

    if getattr(plan, "model_key", None) != "glm52-expert-q1t":
        raise ValueError("GLM fused-rANS memory caps require glm52-expert-q1t")
    total_limit_bytes = getattr(plan, "total_limit_bytes", None)
    if (
        isinstance(total_limit_bytes, bool)
        or not isinstance(total_limit_bytes, int)
        or total_limit_bytes > GLM52_Q1T_MAX_TOTAL_MEMORY_BYTES
    ):
        raise ValueError("GLM fused-rANS total memory exceeds 96 GiB")
    if (
        isinstance(external_residency_bytes, bool)
        or not isinstance(external_residency_bytes, int)
        or external_residency_bytes < 0
    ):
        raise ValueError("GLM fused-rANS external residency must be non-negative")

    from .expert_runtime import parse_memory_bytes, reconcile_mlx_memory_cap

    target_env = os.environ if env is None else env
    base_mlx_limit = reconcile_mlx_memory_cap(plan, env=target_env)
    mlx_limit = base_mlx_limit - external_residency_bytes
    required_mlx_bytes = (
        plan.fixed_bytes
        - plan.runtime_reserve_bytes
        - plan.io_staging_bytes
        - plan.transient_bytes
    )
    if mlx_limit < required_mlx_bytes:
        raise ValueError(
            "GLM fused-rANS mapping residency leaves too little MLX memory: "
            f"cap={mlx_limit}, required={required_mlx_bytes}"
        )
    existing_wired = target_env.get("MTPLX_WIRED_LIMIT_BYTES")
    if existing_wired and parse_memory_bytes(existing_wired) not in {
        base_mlx_limit,
        mlx_limit,
    }:
        raise ValueError(
            "MTPLX_WIRED_LIMIT_BYTES conflicts with GLM fused-rANS plan: "
            f"env={parse_memory_bytes(existing_wired)}, planned={mlx_limit}"
        )

    _memory_limit_setter(mx_module, "set_memory_limit")(mlx_limit)
    _memory_limit_setter(mx_module, "set_wired_limit")(mlx_limit)
    target_env["MTPLX_MEMORY_LIMIT_BYTES"] = str(mlx_limit)
    target_env["MTPLX_WIRED_LIMIT_BYTES"] = str(mlx_limit)
    return {
        "total_limit_bytes": total_limit_bytes,
        "runtime_reserve_bytes": int(plan.runtime_reserve_bytes),
        "external_residency_bytes": external_residency_bytes,
        "mlx_memory_limit_bytes": mlx_limit,
        "mlx_wired_limit_bytes": mlx_limit,
    }


__all__ = [
    "GLM52_Q1T_BASE_MANIFEST_SHA256",
    "GLM52_Q1T_FUSED_RANS_CODEC",
    "GLM52_Q1T_FUSED_RANS_GATE_UP_THREADGROUPS",
    "GLM52_Q1T_EXPERT_CACHE_BYTES",
    "GLM52_Q1T_MAX_TOTAL_MEMORY_BYTES",
    "GLM52_Q1T_MAX_LIVE_KV_TOKENS",
    "GLM52_Q1T_PERSISTENT_SLOTS_PER_LAYER",
    "GLM52_Q1T_READ_CHUNK_BYTES",
    "GLM52_Q1T_RUNTIME_RESERVE_BYTES",
    "GLM52_Q1T_TRANSIENT_SLOTS",
    "apply_glm52_q1t_fused_rans_memory_caps",
    "validate_glm52_q1t_fused_rans_config",
]
