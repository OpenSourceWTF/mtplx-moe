"""CPU-only memory gate for the fully resident Qwen4 oQ4 runtime."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .qwen4_ngram import (
    NGramManifest,
    NGramRuntimeBudget,
    QWEN38_FLASH_NEXT_REPO,
    QWEN38_FLASH_NEXT_REVISION,
    plan_production_ngram_cache,
    qwen4_kv_mtp_reserve_bytes,
)

_GIB = 1024**3
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024**2
_MINIMUM_NGRAM_PAYLOAD_BYTES = 1 * _GIB
_METAL_WORKING_RESERVE_BYTES = 2 * _GIB
_SAFETY_MARGIN_BYTES = 2 * _GIB
_TRANSIENT_BYTES = 16 * 2048 * 100
_ALIGNMENT_BYTES = 16 * 1024


@dataclass(frozen=True)
class Qwen4WeightInventory:
    total_bytes: int
    resident_bytes: int
    ngram_bytes: int
    resident_moe_bytes: int
    mtp_bytes: int
    tensor_count: int


@dataclass(frozen=True)
class Qwen4ResidentPreflight:
    resident_weight_bytes: int
    kv_mtp_reserve_bytes: int
    cache_payload_bytes: int
    cache_overhead_bytes: int
    metal_working_reserve_bytes: int
    safety_margin_bytes: int
    projected_residency_bytes: int
    target_residency_bytes: int


def validate_qwen4_oq4_contract(
    config: dict[str, Any], manifest: NGramManifest
) -> None:
    """Reject every artifact except the pinned published affine oQ4."""

    quantization = config.get("quantization")
    valid_quantization = (
        type(quantization) is dict
        and quantization.get("bits") == 4
        and quantization.get("group_size") == 32
        and quantization.get("mode", "affine") == "affine"
    )
    if (
        str(config.get("model_type") or "") != "qwen4_exp"
        or not valid_quantization
        or manifest.source_repo != QWEN38_FLASH_NEXT_REPO
        or manifest.source_revision != QWEN38_FLASH_NEXT_REVISION
        or manifest.storage != "affine-q4-g32"
    ):
        raise ValueError(
            "Qwen4 runtime requires the pinned published oQ4 artifact and "
            "its source-native affine-Q4 n-gram rows"
        )


def _exact_offset(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative exact integer")
    return value


def scan_qwen4_weight_bytes(model_root: Path | str) -> Qwen4WeightInventory:
    """Read safetensors headers only and count the sole externalized payload."""

    root = Path(model_root)
    files = sorted(root.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model safetensors found under {root}")
    total = resident = ngram = resident_moe = mtp = tensor_count = 0
    names: set[str] = set()
    for path in files:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"truncated safetensors header length: {path.name}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if not 0 < header_length <= _MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError(f"unsafe safetensors header length: {path.name}")
            if header_length > file_size - 8:
                raise ValueError(f"safetensors header exceeds file: {path.name}")
            try:
                header = json.loads(handle.read(header_length))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid safetensors header JSON: {path.name}"
                ) from exc
        if type(header) is not dict:
            raise ValueError(f"safetensors header must be an object: {path.name}")
        payload_bytes = file_size - 8 - header_length
        for name, descriptor in header.items():
            if name == "__metadata__":
                continue
            if type(name) is not str or name in names:
                raise ValueError("safetensors tensor names must be unique strings")
            names.add(name)
            if type(descriptor) is not dict:
                raise ValueError(f"invalid tensor descriptor: {name}")
            offsets = descriptor.get("data_offsets")
            if type(offsets) is not list or len(offsets) != 2:
                raise ValueError(f"invalid tensor offsets: {name}")
            start = _exact_offset(offsets[0], label=f"{name} start")
            stop = _exact_offset(offsets[1], label=f"{name} stop")
            if stop < start or stop > payload_bytes:
                raise ValueError(f"tensor payload exceeds safetensors file: {name}")
            size = stop - start
            total += size
            tensor_count += 1
            if ".ngram_embedding." in name:
                ngram += size
                continue
            resident += size
            if ".mlp.switch_mlp." in name:
                resident_moe += size
            if ".mtp." in name:
                mtp += size
    if ngram == 0 or resident == 0:
        raise ValueError("Qwen4 artifact must contain resident and n-gram tensors")
    return Qwen4WeightInventory(
        total_bytes=total,
        resident_bytes=resident,
        ngram_bytes=ngram,
        resident_moe_bytes=resident_moe,
        mtp_bytes=mtp,
        tensor_count=tensor_count,
    )


def plan_qwen4_resident_preflight(
    *,
    resident_weight_bytes: int,
    manifest: NGramManifest,
    context_tokens: int,
    payload_ceiling_bytes: int,
    target_residency_bytes: int,
    available_memory_bytes: int | None = None,
) -> Qwen4ResidentPreflight:
    """Reject an unsafe full-resident load before MLX creates model arrays."""

    if type(resident_weight_bytes) is not int or resident_weight_bytes <= 0:
        raise ValueError("resident_weight_bytes must be a positive exact integer")
    effective_target = target_residency_bytes
    if available_memory_bytes is not None:
        if type(available_memory_bytes) is not int or available_memory_bytes <= 0:
            raise ValueError("available_memory_bytes must be a positive exact integer")
        effective_target = min(effective_target, available_memory_bytes)
    kv_mtp = qwen4_kv_mtp_reserve_bytes(context_tokens)
    budget = NGramRuntimeBudget(
        measured_base_residency_bytes=resident_weight_bytes,
        kv_mtp_reserve_bytes=kv_mtp,
        metal_working_reserve_bytes=_METAL_WORKING_RESERVE_BYTES,
        safety_margin_bytes=_SAFETY_MARGIN_BYTES,
        minimum_payload_bytes=_MINIMUM_NGRAM_PAYLOAD_BYTES,
        allocation_alignment_bytes=_ALIGNMENT_BYTES,
        target_residency_bytes=effective_target,
        payload_ceiling_bytes=payload_ceiling_bytes,
    )
    try:
        production = plan_production_ngram_cache(
            manifest,
            budget,
            transient_limit_bytes=_TRANSIENT_BYTES,
            max_inflight_io_bytes=_TRANSIENT_BYTES,
            max_open_files=129,
            bypass_page_cache=True,
            eviction="lru",
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "Qwen4 full-resident runtime rejected before MLX load: " + str(exc)
        ) from exc
    cache = production.cache
    return Qwen4ResidentPreflight(
        resident_weight_bytes=resident_weight_bytes,
        kv_mtp_reserve_bytes=kv_mtp,
        cache_payload_bytes=cache.payload_bytes,
        cache_overhead_bytes=cache.total_reserved_bytes - cache.payload_bytes,
        metal_working_reserve_bytes=_METAL_WORKING_RESERVE_BYTES,
        safety_margin_bytes=_SAFETY_MARGIN_BYTES,
        projected_residency_bytes=production.projected_residency_bytes,
        target_residency_bytes=production.target_residency_bytes,
    )


__all__ = [
    "Qwen4ResidentPreflight",
    "Qwen4WeightInventory",
    "plan_qwen4_resident_preflight",
    "scan_qwen4_weight_bytes",
    "validate_qwen4_oq4_contract",
]
