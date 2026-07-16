from __future__ import annotations

import json
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Mapping

from .builtins import default_setting_catalog
from .catalog import SettingCatalog


def _catalog(catalog: SettingCatalog | None) -> SettingCatalog:
    return catalog or default_setting_catalog()


def _read_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a TOML document")
    return payload


def load_user_settings(
    path: str | Path, *, catalog: SettingCatalog | None = None
) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    document = _read_document(resolved)
    active_catalog = _catalog(catalog)
    values: dict[str, Any] = {}

    settings_table = document.get("settings", {})
    if settings_table and not isinstance(settings_table, dict):
        raise ValueError(f"{resolved}: [settings] must be a table")
    for name, raw in settings_table.items():
        spec = active_catalog.require(name)
        values[spec.name] = spec.parse(raw)

    for key, raw in document.items():
        if key == "settings" or isinstance(raw, dict):
            continue
        spec = active_catalog.resolve_alias("config", key)
        if spec is not None and spec.name not in values:
            values[spec.name] = spec.parse(raw)
    return values


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise ValueError(f"unsupported settings value: {value!r}")


def _render_document(
    legacy: Mapping[str, Any], settings: Mapping[str, Any]
) -> str:
    lines: list[str] = []
    for key in sorted(legacy):
        value = legacy[key]
        if isinstance(value, dict):
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    if lines and settings:
        lines.append("")
    if settings:
        lines.append("[settings]")
        for name in sorted(settings):
            lines.append(f"{json.dumps(name)} = {_toml_value(settings[name])}")
    return "\n".join(lines) + "\n"


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


def _candidate_document(
    path: Path,
    *,
    name: str,
    value: Any | None,
    remove: bool,
    catalog: SettingCatalog,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = catalog.require(name)
    document = _read_document(path)
    current = load_user_settings(path, catalog=catalog)
    if remove:
        current.pop(spec.name, None)
    else:
        current[spec.name] = spec.parse(value)

    for candidate_name, candidate_value in current.items():
        catalog.require(candidate_name).parse(candidate_value)

    legacy = {
        key: raw
        for key, raw in document.items()
        if key != "settings"
        and catalog.resolve_alias("config", key) is not spec
    }
    return legacy, current


def update_user_setting(
    path: str | Path,
    name: str,
    value: Any,
    *,
    catalog: SettingCatalog | None = None,
) -> None:
    resolved = Path(path).expanduser()
    active_catalog = _catalog(catalog)
    legacy, settings = _candidate_document(
        resolved,
        name=name,
        value=value,
        remove=False,
        catalog=active_catalog,
    )
    _write_atomic(resolved, _render_document(legacy, settings))


def unset_user_setting(
    path: str | Path,
    name: str,
    *,
    catalog: SettingCatalog | None = None,
) -> None:
    resolved = Path(path).expanduser()
    active_catalog = _catalog(catalog)
    legacy, settings = _candidate_document(
        resolved,
        name=name,
        value=None,
        remove=True,
        catalog=active_catalog,
    )
    _write_atomic(resolved, _render_document(legacy, settings))
