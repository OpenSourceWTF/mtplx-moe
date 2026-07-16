"""No-MLX settings inspection and persistence commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mtplx.config import user_config_path
from mtplx.settings.builtins import default_setting_catalog
from mtplx.settings.resolver import SettingSource, SettingsResolver
from mtplx.settings.storage import (
    load_user_settings,
    unset_user_setting,
    update_user_setting,
)


def _config_path(args: Any) -> Path:
    return user_config_path(getattr(args, "config", None))


def _environment_settings() -> dict[str, Any]:
    catalog = default_setting_catalog()
    values: dict[str, Any] = {}
    for (source, alias), spec in catalog.by_alias.items():
        if source == "env" and alias in os.environ:
            values[spec.name] = os.environ[alias]
    return values


def _effective_settings(args: Any):
    catalog = default_setting_catalog()
    return catalog, SettingsResolver(catalog).resolve(
        {
            SettingSource.USER: load_user_settings(
                _config_path(args), catalog=catalog
            ),
            SettingSource.ENV: _environment_settings(),
        }
    )


def _parse_pairs(pairs: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    malformed: list[str] = []
    for pair in pairs:
        name, separator, raw = pair.partition("=")
        name = name.strip()
        if not separator or not name:
            malformed.append(pair)
            continue
        try:
            parsed[name] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[name] = raw
    if malformed:
        raise ValueError(
            "expected NAME=VALUE: " + ", ".join(repr(item) for item in malformed)
        )
    return parsed


def _emit(payload: Any, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if isinstance(payload, dict):
        for name, value in payload.items():
            print(f"{name} = {json.dumps(value, default=str)}")


def _redact_user_settings(catalog: Any, settings: dict[str, Any]) -> dict[str, Any]:
    return {
        name: "[redacted]"
        if catalog.require(name).secret and value
        else value
        for name, value in settings.items()
    }


def _show_effective(args: Any) -> int:
    _, resolved = _effective_settings(args)
    payload = {
        name: {
            "value": resolved.provenance[name].display_value,
            "source": resolved.provenance[name].source.name,
        }
        for name in sorted(resolved.provenance)
    }
    if getattr(args, "json", False):
        _emit({"settings": payload}, json_output=True)
    else:
        for name, item in payload.items():
            print(
                f"{name} = {json.dumps(item['value'], default=str)} "
                f"({item['source']})"
            )
    return 0


def _list_catalog(args: Any) -> int:
    catalog = default_setting_catalog()
    specs = catalog.list(
        domain=getattr(args, "settings_group", None),
        visibility=getattr(args, "visibility", None),
    )
    payload = [
        {
            "name": spec.name,
            "type": spec.value_type.value,
            "default": "[redacted]" if spec.secret else spec.default,
            "group": spec.domain,
            "visibility": spec.visibility.value,
            "lifecycle": spec.lifecycle.value,
            "description": spec.description,
            "live_mutable": spec.live_mutable,
        }
        for spec in specs
    ]
    if getattr(args, "json", False):
        _emit({"settings": payload}, json_output=True)
    else:
        for item in payload:
            print(
                f"{item['name']}  {item['type']}  "
                f"{item['visibility']}  {item['group']}"
            )
    return 0


def _explain(args: Any) -> int:
    catalog, resolved = _effective_settings(args)
    spec = catalog.require(args.name)
    record = resolved.provenance[spec.name]
    payload = {
        "name": spec.name,
        "value": record.display_value,
        "requested_value": record.requested_value,
        "source": record.source.name,
        "reason": record.reason,
        "type": spec.value_type.value,
        "group": spec.domain,
        "visibility": spec.visibility.value,
        "lifecycle": spec.lifecycle.value,
        "description": spec.description,
        "aliases": [
            {"source": alias.source, "name": alias.name}
            for alias in spec.aliases
        ],
        "live_mutable": spec.live_mutable,
        "shadowed": [
            {"source": item.source.name, "value": item.value}
            for item in record.shadowed
        ],
    }
    if getattr(args, "json", False):
        _emit(payload, json_output=True)
    else:
        print(f"{spec.name} = {json.dumps(record.display_value, default=str)}")
        print(f"source: {record.source.name}")
        print(f"type: {spec.value_type.value}")
    return 0


def _user(args: Any) -> int:
    catalog = default_setting_catalog()
    path = _config_path(args)
    operation = args.settings_operation
    if operation == "show":
        settings = _redact_user_settings(
            catalog, load_user_settings(path, catalog=catalog)
        )
        _emit(
            {"settings": settings} if getattr(args, "json", False) else settings,
            json_output=bool(getattr(args, "json", False)),
        )
        return 0

    if operation == "set":
        updates = _parse_pairs(list(args.pairs))
        # Validate the whole request before opening the destination file.
        validated = {
            catalog.require(name).name: catalog.require(name).parse(value)
            for name, value in updates.items()
        }
        for name, value in validated.items():
            update_user_setting(path, name, value, catalog=catalog)
        settings = _redact_user_settings(
            catalog, load_user_settings(path, catalog=catalog)
        )
        if getattr(args, "json", False):
            _emit({"settings": settings}, json_output=True)
        else:
            for name in validated:
                print(f"saved: {name}")
        return 0

    names = list(args.names)
    for name in names:
        catalog.require(name)
    for name in names:
        unset_user_setting(path, name, catalog=catalog)
    if getattr(args, "json", False):
        _emit(
            {
                "settings": _redact_user_settings(
                    catalog, load_user_settings(path, catalog=catalog)
                )
            },
            json_output=True,
        )
    else:
        for name in names:
            print(f"removed: {name}")
    return 0


def _live(args: Any) -> int:
    args.settings_action = "get" if args.settings_operation == "show" else "set"
    if not hasattr(args, "pairs"):
        args.pairs = []
    return cmd_settings_public(args)


def cmd_settings(args: Any) -> int:
    """Dispatch a scoped settings operation without importing MLX."""

    try:
        scope = args.settings_scope
        if scope == "effective" and args.settings_operation == "show":
            return _show_effective(args)
        if scope == "catalog":
            return _list_catalog(args)
        if scope == "effective":
            return _explain(args)
        if scope == "user":
            return _user(args)
        if scope == "live":
            return _live(args)
        raise ValueError(f"unsupported settings scope: {scope}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2


def _parse_settings_pairs(pairs: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Parse ``key=value`` pairs; values decode as JSON with string fallback."""

    parsed: dict[str, Any] = {}
    errors: list[str] = []
    for pair in pairs:
        key, separator, raw_value = str(pair).partition("=")
        key = key.strip()
        if not separator or not key:
            errors.append(pair)
            continue
        value_text = raw_value.strip()
        try:
            parsed[key] = json.loads(value_text)
        except json.JSONDecodeError:
            parsed[key] = value_text
    return parsed, errors


def cmd_settings_public(args: Any) -> int:
    """Read or change live server settings over /v1/mtplx/settings."""

    from mtplx.commands import public as compatibility

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    base = compatibility._server_url(host, port)
    json_output = bool(getattr(args, "json", False))
    action = str(getattr(args, "settings_action", None) or "get")
    pairs = list(getattr(args, "pairs", None) or [])
    if action == "get" and pairs:
        action = "set"

    def fail_unreachable() -> int:
        print(f"No MTPLX server is responding on {base}.")
        print("Start one with the MTPLX app or: mtplx start")
        return 1

    if action == "get":
        payload = compatibility._http_json(
            base + "/v1/mtplx/settings", timeout=5.0
        )
        if not payload.get("ok"):
            return fail_unreachable()
        if json_output:
            compatibility._print(payload)
            return 0
        print(f"MTPLX server settings  ·  {base}")
        for key in sorted(payload):
            if key != "ok":
                print(f"  {key} = {json.dumps(payload[key], default=str)}")
        return 0

    update, malformed = _parse_settings_pairs(pairs)
    if malformed or not update:
        for pair in malformed:
            print(f"error: not a key=value pair: {pair!r}")
        if not update:
            print("usage: mtplx settings set key=value [key=value ...]")
            print("example: mtplx settings set depth=2 reasoning=off")
        return 2
    response = compatibility._http_post_json(
        base + "/v1/mtplx/settings", update, timeout=10.0
    )
    if response.get("ok"):
        body = response.get("json") or {}
        applied = body.get("applied") or {}
        if json_output:
            compatibility._print(body)
            return 0
        if applied:
            for key in sorted(applied):
                print(f"applied: {key} = {json.dumps(applied[key], default=str)}")
        else:
            print("nothing to apply")
        return 0
    error = response.get("error")
    if isinstance(error, dict) and isinstance(error.get("error"), dict):
        error = error["error"]
    detail = error.get("detail") if isinstance(error, dict) else None
    if isinstance(detail, dict):
        kind = detail.get("error")
        keys = detail.get("keys") or []
        if kind == "restart_required":
            print(
                "error: these settings need a server restart: "
                + ", ".join(str(key) for key in keys)
            )
            print(
                "Change them in the MTPLX app's settings, or restart "
                "`mtplx serve` with the matching flags."
            )
            return 2
        if kind == "unknown_settings":
            print("error: unknown settings: " + ", ".join(map(str, keys)))
            supported = detail.get("supported") or []
            if supported:
                print("supported: " + ", ".join(map(str, supported)))
            return 2
    if isinstance(detail, str) and detail:
        print(f"error: {detail}")
        return 2
    if response.get("status") is None:
        return fail_unreachable()
    print(f"error: settings update failed ({response.get('status')})")
    return 1
