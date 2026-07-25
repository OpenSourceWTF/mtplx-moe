"""Package-owned, benchmark-promoted expert serve profiles."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .expert_runtime import (
    ExpertStreamingConfig,
    parse_memory_bytes,
    resolve_island_placement,
)


_BYTE_FIELDS = frozenset(
    {
        "memory_limit_bytes",
        "runtime_reserve_bytes",
        "expert_cache_limit_bytes",
        "io_staging_bytes",
        "execution_workspace_bytes",
        "max_inflight_io_bytes",
        "max_read_chunk_bytes",
    }
)
_VM_STAT_PAGE_SIZE = re.compile(r"page size of ([0-9]+) bytes")
_VM_STAT_PAGE_COUNT = re.compile(r"^(Pages [^:]+):\s*([0-9]+)\.$")
_AVAILABLE_VM_STAT_PAGES = frozenset(
    {
        "Pages free",
        "Pages inactive",
        "Pages speculative",
        "Pages purgeable",
    }
)
_PROFILE_IDENTITY_FIELDS = frozenset(
    {
        "name",
        "model_key",
        "process_ceiling_bytes",
        "weight_envelope_bytes",
        "generation_mode",
        "evidence_commit",
        "evidence_receipts",
        "config",
        "child_env",
    }
)


@dataclass(frozen=True)
class ExpertServeProfile:
    name: str
    model_key: str
    process_ceiling_bytes: int
    weight_envelope_bytes: int
    generation_mode: str
    evidence_commit: str
    evidence_receipts: tuple[str, ...]
    config: Mapping[str, Any]
    child_env: Mapping[str, str]


def _profile_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"expert profile {key!r} must be a non-empty string")
    return value


def _profile_positive_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"expert profile {key!r} must be a positive integer")
    return value


def _profile_string_tuple(row: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"expert profile {key!r} must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"expert profile {key!r} must contain only strings")
    return tuple(value)


def _profile_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"expert profile {key!r} must be an object")
    return MappingProxyType(dict(value))


def _parse_profile(row: object) -> ExpertServeProfile:
    if not isinstance(row, dict):
        raise ValueError("each expert profile must be an object")
    name = _profile_string(row, "name")
    process_ceiling_bytes = _profile_positive_int(
        row, "process_ceiling_bytes"
    )
    weight_envelope_bytes = _profile_positive_int(
        row, "weight_envelope_bytes"
    )
    generation_mode = _profile_string(row, "generation_mode")
    if generation_mode != "ar":
        raise ValueError("promoted expert profiles must use generation_mode 'ar'")
    config = _profile_mapping(row, "config")
    shadowed_identity = _PROFILE_IDENTITY_FIELDS.intersection(config)
    if shadowed_identity:
        names = ", ".join(sorted(shadowed_identity))
        raise ValueError(
            f"expert profile {name!r} config shadows profile identity: {names}"
        )
    config_ceiling = config.get("memory_limit_bytes")
    if (
        isinstance(config_ceiling, bool)
        or not isinstance(config_ceiling, int)
        or config_ceiling <= 0
    ):
        raise ValueError(
            f"expert profile {name!r} config.memory_limit_bytes must be "
            "a positive integer"
        )
    runtime_reserve = config.get("runtime_reserve_bytes")
    if (
        isinstance(runtime_reserve, bool)
        or not isinstance(runtime_reserve, int)
        or runtime_reserve < 0
    ):
        raise ValueError(
            f"expert profile {name!r} config.runtime_reserve_bytes must be "
            "a non-negative integer"
        )
    if config_ceiling != process_ceiling_bytes:
        raise ValueError(
            f"expert profile {name!r} config.memory_limit_bytes "
            f"{config_ceiling} does not equal process_ceiling_bytes "
            f"{process_ceiling_bytes}"
        )
    if weight_envelope_bytes + runtime_reserve != process_ceiling_bytes:
        raise ValueError(
            f"expert profile {name!r} weight_envelope_bytes "
            f"{weight_envelope_bytes} plus config.runtime_reserve_bytes "
            f"{runtime_reserve} does not equal process_ceiling_bytes "
            f"{process_ceiling_bytes}"
        )
    child_env = _profile_mapping(row, "child_env")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in child_env.items()
    ):
        raise ValueError("expert profile child_env must map strings to strings")
    return ExpertServeProfile(
        name=name,
        model_key=_profile_string(row, "model_key"),
        process_ceiling_bytes=process_ceiling_bytes,
        weight_envelope_bytes=weight_envelope_bytes,
        generation_mode=generation_mode,
        evidence_commit=_profile_string(row, "evidence_commit"),
        evidence_receipts=_profile_string_tuple(row, "evidence_receipts"),
        config=config,
        child_env=child_env,
    )


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON key {key!r}")
        parsed[key] = value
    return parsed


def _parse_expert_profiles_resource(
    resource_text: str,
) -> Mapping[str, ExpertServeProfile]:
    try:
        document = json.loads(
            resource_text,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid expert profiles JSON: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != 1:
        raise ValueError("expert profiles resource must use schema 1")
    rows = document.get("profiles")
    if not isinstance(rows, list):
        raise ValueError("expert profiles resource must contain a profiles list")
    parsed: dict[str, ExpertServeProfile] = {}
    for row in rows:
        profile = _parse_profile(row)
        if profile.name in parsed:
            raise ValueError(f"duplicate expert profile {profile.name!r}")
        parsed[profile.name] = profile
    return MappingProxyType(parsed)


@lru_cache(maxsize=1)
def load_expert_profiles() -> Mapping[str, ExpertServeProfile]:
    """Load immutable production profiles from the installed package."""

    resource = files("mtplx").joinpath("data/expert_profiles.json")
    return _parse_expert_profiles_resource(
        resource.read_text(encoding="utf-8")
    )


def _installed_ram_bytes() -> int:
    result = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "hw.memsize"],
        check=True,
        capture_output=True,
        text=True,
        timeout=2.0,
    )
    try:
        total = int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("could not parse installed RAM from hw.memsize") from exc
    if total <= 0:
        raise RuntimeError("installed RAM from hw.memsize must be positive")
    return total


def available_memory_bytes() -> int:
    """Return launch-time reclaimable memory reported by macOS ``vm_stat``."""

    try:
        result = subprocess.run(
            ["/usr/bin/vm_stat"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"vm_stat preflight failed: {exc}") from exc
    lines = result.stdout.splitlines()
    if not lines:
        raise RuntimeError("vm_stat returned no output")
    size_match = _VM_STAT_PAGE_SIZE.search(lines[0])
    if size_match is None:
        raise RuntimeError("could not parse vm_stat page size")
    page_size = int(size_match.group(1))
    if page_size <= 0:
        raise RuntimeError("vm_stat page size must be positive")
    counts: dict[str, int] = {}
    for line in lines[1:]:
        match = _VM_STAT_PAGE_COUNT.match(line.strip())
        if match is not None and match.group(1) in _AVAILABLE_VM_STAT_PAGES:
            counts[match.group(1)] = int(match.group(2))
    missing = _AVAILABLE_VM_STAT_PAGES.difference(counts)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise RuntimeError(f"vm_stat omitted required page counts: {missing_text}")
    return page_size * sum(counts.values())


def _memory_measurement(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def select_expert_profile(
    requested: str,
    *,
    model_key: str,
    installed_ram_bytes: int | None = None,
    available_bytes: int | None = None,
) -> ExpertServeProfile:
    """Select a promoted profile once, before runtime construction."""

    profiles = load_expert_profiles()
    installed = _memory_measurement(
        "installed_ram_bytes",
        _installed_ram_bytes()
        if installed_ram_bytes is None
        else installed_ram_bytes,
    )
    available = _memory_measurement(
        "available_bytes",
        available_memory_bytes() if available_bytes is None else available_bytes,
    )

    if requested != "auto":
        try:
            selected = profiles[requested]
        except KeyError as exc:
            names = ", ".join(profiles)
            raise ValueError(
                f"unknown expert profile {requested!r}; choose from {names}"
            ) from exc
        if selected.model_key != model_key:
            raise ValueError(
                f"expert profile {selected.name!r} requires model key "
                f"{selected.model_key!r}, not {model_key!r}"
            )
        if (
            selected.process_ceiling_bytes > installed
            or selected.process_ceiling_bytes > available
        ):
            raise ValueError(
                f"expert profile {selected.name!r}: required "
                f"{selected.process_ceiling_bytes} installed bytes and "
                f"{selected.process_ceiling_bytes} available bytes; detected "
                f"{installed} installed bytes and {available} available bytes"
            )
        return selected

    candidates = sorted(
        (profile for profile in profiles.values() if profile.model_key == model_key),
        key=lambda profile: (
            profile.weight_envelope_bytes,
            profile.process_ceiling_bytes,
        ),
        reverse=True,
    )
    for profile in candidates:
        if (
            profile.process_ceiling_bytes <= installed
            and profile.process_ceiling_bytes <= available
        ):
            return profile
    if not candidates:
        raise ValueError(f"no promoted expert profiles match model key {model_key!r}")
    smallest = candidates[-1]
    names = ", ".join(profile.name for profile in reversed(candidates))
    raise ValueError(
        "no promoted expert profile fits: minimum profile "
        f"{smallest.name!r} requires {smallest.process_ceiling_bytes} installed "
        f"bytes and {smallest.process_ceiling_bytes} available bytes; detected "
        f"{installed} installed bytes and {available} available bytes; "
        f"promoted profiles: {names}"
    )


def _normalize_overrides(
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized = dict(overrides or {})
    for field in _BYTE_FIELDS:
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = parse_memory_bytes(value)
    return normalized


def build_expert_streaming_config(
    profile: ExpertServeProfile,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ExpertStreamingConfig:
    """Construct one validated immutable streaming configuration."""

    normalized_overrides = _normalize_overrides(overrides)
    if "model_key" in normalized_overrides:
        raise ValueError("model_key cannot override expert profile identity")
    values = {
        "model_key": profile.model_key,
        **profile.config,
        **normalized_overrides,
    }
    config = ExpertStreamingConfig(**values)
    if config.model_key != profile.model_key:
        raise ValueError(
            f"completed config model_key {config.model_key!r} does not match "
            f"expert profile model_key {profile.model_key!r}"
        )
    if config.memory_limit_bytes > profile.process_ceiling_bytes:
        raise ValueError(
            f"completed config memory_limit_bytes {config.memory_limit_bytes} "
            "exceeds the admitted expert profile process ceiling "
            f"{profile.process_ceiling_bytes}"
        )
    if config.island_layer_count is not None:
        config = resolve_island_placement(config, Path())
    return config
