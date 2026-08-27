"""Construction-time runtime wiring for the pinned Qwen3.8 oQ4 n-gram lane."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .qwen4_ngram import (
    NGramGeometry,
    NGramRowCache,
    NGramRuntimeBudget,
    PRODUCTION_NGRAM_PAYLOAD_CEILING_BYTES,
    QWEN38_FLASH_NEXT_REPO,
    QWEN38_FLASH_NEXT_REVISION,
    load_ngram_manifest,
    plan_production_ngram_cache,
    qwen4_kv_mtp_reserve_bytes,
    verify_ngram_manifest,
)

_GIB = 1024**3
QWEN4_NGRAM_MANIFEST_NAME = "ngram-manifest.json"
QWEN4_NGRAM_MINIMUM_PAYLOAD_BYTES = 1 * _GIB
QWEN4_NGRAM_METAL_WORKING_RESERVE_BYTES = 2 * _GIB
QWEN4_NGRAM_SAFETY_MARGIN_BYTES = 2 * _GIB
QWEN4_NGRAM_TRANSIENT_LIMIT_BYTES = 16 * 2048 * 100
QWEN4_NGRAM_MAX_INFLIGHT_IO_BYTES = QWEN4_NGRAM_TRANSIENT_LIMIT_BYTES
QWEN4_NGRAM_MAX_OPEN_FILES = 129
QWEN4_NGRAM_ALLOCATION_ALIGNMENT_BYTES = 16 * 1024


@dataclass(frozen=True)
class Qwen4NGramRuntimeResources:
    cache: Any
    artifact: Any
    report: dict[str, Any]


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, Mapping) else config


def _exact_config_int(
    text: Mapping[str, Any], name: str, expected: int
) -> None:
    value = text.get(name, 1234 if name == "seed" else None)
    if type(value) is not int:
        raise TypeError(f"Qwen4 {name} must be an exact integer")
    if value != expected:
        raise ValueError(f"Qwen4 {name} differs from the pinned n-gram contract")


def _validate_pinned_model_geometry(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise TypeError("Qwen4 config must be a mapping")
    if str(config.get("model_type") or "") != "qwen4_exp":
        raise ValueError("Qwen4 root model_type differs from the pinned contract")
    text = _text_config(config)
    if str(text.get("model_type") or "") != "qwen4_exp_text":
        raise ValueError("Qwen4 text model_type differs from the pinned contract")
    geometry = NGramGeometry.qwen38()
    for name, expected in (
        ("vocab_size", geometry.vocab_size),
        ("eos_token_id", geometry.eos_token_id),
        ("seed", geometry.seed),
        ("ngram_size", geometry.ngram_size),
        ("heads_per_ngram", geometry.heads_per_ngram),
        ("ngram_vocab_size_base", geometry.ngram_vocab_size_base),
        ("make_ngram_vocab_size_divisible_by", geometry.divisor),
        ("split_ngram_parts", 128),
        ("ple_embed_dim", 2560),
    ):
        _exact_config_int(text, name, expected)
    ple_layers = text.get("ple_layer_ids")
    if not isinstance(ple_layers, (list, tuple)) or tuple(ple_layers) != (2,):
        raise ValueError("Qwen4 PLE layer differs from the pinned n-gram contract")


def _validate_pinned_manifest(manifest: Any) -> None:
    expected = {
        "source_repo": QWEN38_FLASH_NEXT_REPO,
        "source_revision": QWEN38_FLASH_NEXT_REVISION,
        "storage": "affine-q4-g32",
        "row_width": 160,
        "row_bytes": 100,
        "padded_rows": NGramGeometry.qwen38().padded_rows,
    }
    for name, value in expected.items():
        if getattr(manifest, name, None) != value:
            if name == "storage":
                raise ValueError(
                    "Qwen4 n-gram runtime requires the published affine-Q4 rows"
                )
            raise ValueError(
                f"Qwen4 n-gram manifest {name} differs from the pinned contract"
            )


def _bind_streamed_ngram_rows(model: Any, cache: Any) -> int:
    from .models.qwen4_ngram_mlx import bind_streamed_ngram_rows

    return bind_streamed_ngram_rows(model, cache)


def _close_construction_resources(cache: Any | None, artifact: Any | None) -> None:
    first_failure: BaseException | None = None
    for owner in (cache, artifact):
        if owner is None:
            continue
        try:
            owner.close()
        except BaseException as exc:  # noqa: BLE001 - close every acquired owner
            if first_failure is None:
                first_failure = exc
    if first_failure is not None:
        raise first_failure


def construct_qwen4_ngram_runtime(
    model_root: Path | str,
    model: Any,
    *,
    config: Mapping[str, Any],
    context_tokens: int,
    payload_ceiling_bytes: int,
    target_residency_bytes: int,
    mx_module: Any,
) -> Qwen4NGramRuntimeResources:
    """Verify, size, allocate, and bind the exact source-native row cache once."""

    if type(context_tokens) is not int or context_tokens < 1:
        raise ValueError("Qwen4 context_tokens must be a positive exact integer")
    if (
        type(payload_ceiling_bytes) is not int
        or not 0 < payload_ceiling_bytes <= PRODUCTION_NGRAM_PAYLOAD_CEILING_BYTES
    ):
        raise ValueError("Qwen4 n-gram payload ceiling must be within (0, 10 GiB]")
    get_active_memory = getattr(mx_module, "get_active_memory", None)
    if not callable(get_active_memory):
        raise TypeError("MLX must expose get_active_memory for Qwen4 construction")
    set_cache_limit = getattr(mx_module, "set_cache_limit", None)
    clear_cache = getattr(mx_module, "clear_cache", None)
    if not callable(set_cache_limit) or not callable(clear_cache):
        raise TypeError(
            "MLX must expose set_cache_limit and clear_cache for Qwen4 construction"
        )
    _validate_pinned_model_geometry(config)
    root = Path(model_root).resolve()
    manifest = load_ngram_manifest(root / QWEN4_NGRAM_MANIFEST_NAME)
    _validate_pinned_manifest(manifest)

    artifact = None
    cache = None
    try:
        artifact = verify_ngram_manifest(
            root,
            manifest,
            bypass_page_cache=True,
        )
        previous_cache_limit = set_cache_limit(
            QWEN4_NGRAM_METAL_WORKING_RESERVE_BYTES
        )
        if type(previous_cache_limit) is not int or previous_cache_limit < 0:
            raise RuntimeError("MLX returned an invalid previous cache limit")
        clear_cache()
        measured_base = get_active_memory()
        if type(measured_base) is not int or measured_base < 0:
            raise RuntimeError("MLX returned an invalid active-memory measurement")
        kv_mtp_reserve = qwen4_kv_mtp_reserve_bytes(context_tokens)
        budget = NGramRuntimeBudget(
            measured_base_residency_bytes=measured_base,
            kv_mtp_reserve_bytes=kv_mtp_reserve,
            metal_working_reserve_bytes=QWEN4_NGRAM_METAL_WORKING_RESERVE_BYTES,
            safety_margin_bytes=QWEN4_NGRAM_SAFETY_MARGIN_BYTES,
            minimum_payload_bytes=QWEN4_NGRAM_MINIMUM_PAYLOAD_BYTES,
            allocation_alignment_bytes=QWEN4_NGRAM_ALLOCATION_ALIGNMENT_BYTES,
            target_residency_bytes=target_residency_bytes,
            payload_ceiling_bytes=payload_ceiling_bytes,
        )
        production = plan_production_ngram_cache(
            manifest,
            budget,
            transient_limit_bytes=QWEN4_NGRAM_TRANSIENT_LIMIT_BYTES,
            max_inflight_io_bytes=QWEN4_NGRAM_MAX_INFLIGHT_IO_BYTES,
            max_open_files=QWEN4_NGRAM_MAX_OPEN_FILES,
            bypass_page_cache=True,
            eviction="lru",
        )
        cache = NGramRowCache(artifact, production.config)
        bound = _bind_streamed_ngram_rows(model, cache)
        if bound != 1:
            raise RuntimeError(
                f"Qwen4 runtime must bind exactly one n-gram seam; bound {bound}"
            )
        planned = production.cache
        report = {
            "source_repo": manifest.source_repo,
            "source_revision": manifest.source_revision,
            "manifest_sha256": manifest.digest,
            "storage": manifest.storage,
            "eviction": "lru",
            "target_residency_bytes": production.target_residency_bytes,
            "measured_base_residency_bytes": measured_base,
            "kv_mtp_reserve_bytes": kv_mtp_reserve,
            "metal_working_reserve_bytes": (
                QWEN4_NGRAM_METAL_WORKING_RESERVE_BYTES
            ),
            "mlx_cache_limit_bytes": QWEN4_NGRAM_METAL_WORKING_RESERVE_BYTES,
            "previous_mlx_cache_limit_bytes": previous_cache_limit,
            "safety_margin_bytes": QWEN4_NGRAM_SAFETY_MARGIN_BYTES,
            "requested_payload_ceiling_bytes": payload_ceiling_bytes,
            "payload_formula_ceiling_bytes": (
                production.payload_formula_ceiling_bytes
            ),
            "cache_payload_bytes": planned.payload_bytes,
            "cache_slot_count": planned.slot_count,
            "cache_metadata_bytes": planned.slot_metadata_bytes,
            "cache_route_table_bytes": planned.route_table_bytes,
            "cache_transient_bytes": planned.transient_bytes,
            "cache_transient_metadata_bytes": planned.transient_metadata_bytes,
            "cache_alignment_bytes": planned.alignment_bytes,
            "cache_overhead_bytes": planned.total_reserved_bytes
            - planned.payload_bytes,
            "cache_total_reserved_bytes": planned.total_reserved_bytes,
            "projected_residency_bytes": production.projected_residency_bytes,
        }
        return Qwen4NGramRuntimeResources(
            cache=cache,
            artifact=artifact,
            report=report,
        )
    except BaseException:
        _close_construction_resources(cache, artifact)
        raise


__all__ = [
    "Qwen4NGramRuntimeResources",
    "construct_qwen4_ngram_runtime",
]
