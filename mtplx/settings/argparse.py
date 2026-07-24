"""Bridge canonical settings to the legacy argparse Namespace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from mtplx.config import user_config_path

from .builtins import default_setting_catalog
from .bundles import load_settings_bundle
from .resolver import ResolvedSettings, SettingSource, SettingsResolver
from .storage import load_user_settings


ARG_DEST_BY_SETTING: dict[str, str] = {
    "model.ref": "model",
    "model.cache_dir": "cache_dir",
    "runtime.profile": "profile",
    "runtime.mtp.enabled": "no_mtp",
    "runtime.mtp.depth": "depth",
    "thermal.control": "thermal_control",
    "memory.kv.quantization": "paged_kv_quantization",
    "runtime.scheduler.mode": "scheduler_mode",
    "runtime.batching.preset": "batching_preset",
    "runtime.requests.max_active": "max_active_requests",
    "runtime.batching.decode_max": "decode_batch_max",
    "runtime.batching.wait_ms": "batch_wait_ms",
    "runtime.prefill.chunk_tokens": "prefill_chunk_tokens",
    "runtime.mtp.cohorts.enabled": "experimental_mtp_cohorts",
    "cache.session.ssd.mode": "ssd_session_cache",
    "cache.session.ssd.directory": "ssd_session_cache_dir",
    "cache.session.ssd.max_size": "ssd_session_cache_max_size",
    "cache.session.ssd.min_prefix_tokens": "ssd_session_cache_min_prefix_tokens",
    "cache.session.ram.policy": "ram_session_cache_policy",
    "cache.session.ram.max_entries": "ram_session_cache_max_entries",
    "cache.session.ram.max_size": "ram_session_cache_max_size",
    "cache.session.ram.per_session_max_size": (
        "ram_session_cache_per_session_max_size"
    ),
    "cache.session.ram.block_prefix_restore": "ram_session_block_prefix_restore",
    "model.context_window": "context_window",
    "generation.reasoning": "reasoning",
    "generation.reasoning_effort": "reasoning_effort",
    "generation.temperature": "temperature",
    "generation.top_p": "top_p",
    "generation.top_k": "top_k",
    "server.api_key_file": "api_key_file",
}

_EXPLICIT_SOURCES = frozenset(
    {
        SettingSource.USER,
        SettingSource.ENV,
        SettingSource.BUNDLE,
        SettingSource.LEGACY_CLI,
        SettingSource.CLI_SET,
        SettingSource.CONSTRAINT,
    }
)


def add_settings_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        dest="setting_overrides",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override one canonical setting; repeat for multiple values.",
    )
    parser.add_argument(
        "--settings",
        dest="settings_bundles",
        action="append",
        default=[],
        metavar="PATH",
        help="Load a data-only settings bundle; repeat to layer bundles.",
    )


def _parse_overrides(pairs: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    malformed: list[str] = []
    for pair in pairs:
        name, separator, raw = str(pair).partition("=")
        name = name.strip()
        if not separator or not name:
            malformed.append(str(pair))
            continue
        try:
            values[name] = json.loads(raw)
        except json.JSONDecodeError:
            values[name] = raw
    if malformed:
        raise ValueError(
            "expected NAME=VALUE: " + ", ".join(repr(item) for item in malformed)
        )
    return values


def _environment_values(
    environ: Mapping[str, str],
) -> dict[str, Any]:
    catalog = default_setting_catalog()
    values: dict[str, Any] = {}
    for (source, alias), spec in catalog.by_alias.items():
        if source == "env" and alias in environ:
            values[spec.name] = environ[alias]
    return values


def _legacy_cli_values(args: Any) -> dict[str, Any]:
    catalog = default_setting_catalog()
    flags = set(getattr(args, "_cli_flags", set()) or set())
    values: dict[str, Any] = {}
    for spec in catalog.by_name.values():
        aliases = {
            alias.name for alias in spec.aliases if alias.source == "cli"
        }
        if not aliases.intersection(flags):
            continue
        dest = ARG_DEST_BY_SETTING.get(spec.name)
        if dest is None or not hasattr(args, dest):
            continue
        value = getattr(args, dest)
        if spec.name == "runtime.mtp.enabled":
            value = not bool(value)
        values[spec.name] = value
    return values


def _bundle_values(
    paths: list[str], *, model_family: str | None
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    values: dict[str, Any] = {}
    provenance: list[Any] = []
    for path in paths:
        loaded = load_settings_bundle(path, model_family=model_family)
        values.update(loaded.settings)
        provenance.append(loaded)
    return values, tuple(provenance)


def _apply_to_namespace(args: Any, resolved: ResolvedSettings) -> None:
    for name, dest in ARG_DEST_BY_SETTING.items():
        record = resolved.provenance[name]
        if record.source not in _EXPLICIT_SOURCES:
            continue
        value = resolved[name]
        if name == "runtime.mtp.enabled":
            value = not bool(value)
        setattr(args, dest, value)


def apply_args_constraints(
    args: Any, constraints: Mapping[str, tuple[Any, str]]
) -> ResolvedSettings | None:
    """Attach explanatory constraints to an existing settings snapshot."""

    resolved = getattr(args, "mtplx_settings", None)
    if not isinstance(resolved, ResolvedSettings):
        return None
    constrained = resolved.with_constraints(constraints)
    args.mtplx_settings = constrained
    _apply_to_namespace(args, constrained)
    return constrained


def resolve_args_settings(
    args: Any,
    *,
    environ: Mapping[str, str] | None = None,
    user_path: str | Path | None = None,
    model_family: str | None = None,
) -> ResolvedSettings:
    """Resolve registered sources and apply explicit winners to ``args``."""

    catalog = default_setting_catalog()
    source_environ = os.environ if environ is None else environ
    path = user_config_path(user_path)
    sources: dict[SettingSource, Mapping[str, Any]] = {}
    if getattr(args, "profile", None) is not None:
        sources[SettingSource.PROFILE] = {
            "runtime.profile": getattr(args, "profile")
        }
    user = load_user_settings(path, catalog=catalog)
    if user:
        sources[SettingSource.USER] = user
    environment = _environment_values(source_environ)
    if environment:
        sources[SettingSource.ENV] = environment
    bundles, bundle_provenance = _bundle_values(
        list(getattr(args, "settings_bundles", []) or []),
        model_family=model_family,
    )
    if bundles:
        sources[SettingSource.BUNDLE] = bundles
    legacy = _legacy_cli_values(args)
    if legacy:
        sources[SettingSource.LEGACY_CLI] = legacy
    overrides = _parse_overrides(
        list(getattr(args, "setting_overrides", []) or [])
    )
    if overrides:
        sources[SettingSource.CLI_SET] = overrides

    resolved = SettingsResolver(catalog).resolve(
        sources,
        bundle_provenance=bundle_provenance,
    )
    args.mtplx_settings = resolved
    _apply_to_namespace(args, resolved)
    return resolved
