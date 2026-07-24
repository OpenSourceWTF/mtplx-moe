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
    generation_mode = _profile_string(row, "generation_mode")
    if generation_mode != "ar":
        raise ValueError("promoted expert profiles must use generation_mode 'ar'")
    child_env = _profile_mapping(row, "child_env")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in child_env.items()
    ):
        raise ValueError("expert profile child_env must map strings to strings")
    return ExpertServeProfile(
        name=_profile_string(row, "name"),
        model_key=_profile_string(row, "model_key"),
        process_ceiling_bytes=_profile_positive_int(
            row, "process_ceiling_bytes"
        ),
        weight_envelope_bytes=_profile_positive_int(
            row, "weight_envelope_bytes"
        ),
        generation_mode=generation_mode,
        evidence_commit=_profile_string(row, "evidence_commit"),
        evidence_receipts=_profile_string_tuple(row, "evidence_receipts"),
        config=_profile_mapping(row, "config"),
        child_env=child_env,
    )


@lru_cache(maxsize=1)
def load_expert_profiles() -> Mapping[str, ExpertServeProfile]:
    """Load immutable production profiles from the installed package."""

    resource = files("mtplx").joinpath("data/expert_profiles.json")
    document = json.loads(resource.read_text(encoding="utf-8"))
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

    result = subprocess.run(
        ["/usr/bin/vm_stat"],
        check=True,
        capture_output=True,
        text=True,
        timeout=2.0,
    )
    lines = result.stdout.splitlines()
    if not lines:
        raise RuntimeError("vm_stat returned no output")
    size_match = _VM_STAT_PAGE_SIZE.search(lines[0])
    if size_match is None:
        raise RuntimeError("could not parse vm_stat page size")
    page_size = int(size_match.group(1))
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

    values = {
        "model_key": profile.model_key,
        **profile.config,
        **_normalize_overrides(overrides),
    }
    config = ExpertStreamingConfig(**values)
    if config.island_layer_count is not None:
        config = resolve_island_placement(config, Path())
    return config
