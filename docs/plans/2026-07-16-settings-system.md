# Hierarchical Settings System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace reusable runtime flags with a typed, hierarchical settings system while preserving current CLI and environment behavior through compatibility adapters.

**Architecture:** A no-MLX `mtplx.settings` package owns the schema, catalog, resolver, provenance, bundle loading, and user TOML storage. Existing config, CLI flags, profiles, and environment variables feed that resolver; runtime commands receive resolved values through an explicit argparse adapter. Settings inspection and persistence are handled by a focused command module, while the historical live-daemon syntax remains a compatibility path.

**Tech Stack:** Python 3.11+, stdlib `argparse`, `dataclasses`, `enum`, `tomllib`, `json`, `pathlib`, `tempfile`; pytest; Ruff.

**Assumptions:**

- Assumes `f3e08cb` behavior is the compatibility baseline — will NOT normalize inconsistent legacy precedence until a separately reviewed migration.
- Assumes user TOML contains scalar settings only — will NOT serialize arbitrary arrays or nested application payloads.
- Assumes individual action operands remain argparse options — will NOT convert output paths, prompt files, confirmation, help, JSON, or dry-run controls into settings.
- Assumes issue #90 remains the umbrella tracker — will NOT open a replacement issue for each plan task.

---

## File Structure

- `mtplx/settings/schema.py` — setting types, visibility/lifecycle enums, alias metadata, parsing, and validation errors.
- `mtplx/settings/catalog.py` — immutable lookup, uniqueness checks, aliases, and typo suggestions.
- `mtplx/settings/builtins.py` — canonical product settings and compatibility aliases.
- `mtplx/settings/resolver.py` — ordered source merge and immutable provenance.
- `mtplx/settings/storage.py` — user TOML read/write with atomic replacement and permissions.
- `mtplx/settings/bundles.py` — generic `[settings]` TOML loading.
- `mtplx/settings/argparse.py` — `--set`/`--settings` registration and resolved-value application.
- `mtplx/commands/settings.py` — show/list/explain/user/live commands.
- `mtplx/settings/legacy_env.py` — generated legacy environment inventory and classifications.
- `scripts/audit_settings_catalog.py` — source/catalog drift check and inventory generator.
- `tests/test_settings_schema.py`, `tests/test_settings_resolver.py`, `tests/test_settings_storage.py`, `tests/test_settings_cli.py`, `tests/test_settings_audit.py` — behavior and migration locks.

### Task 1: Lock Existing Configuration and Settings Behavior

**Files:**
- Create: `tests/test_settings_compatibility_baseline.py`
- Modify: none
- Test: `tests/test_settings_compatibility_baseline.py`

**Security flag:** `security`

- [x] **Step 1: Add characterization tests for current precedence and live settings parsing**

```python
from __future__ import annotations

from mtplx.cli import build_parser
from mtplx.config import apply_user_config


def test_legacy_profile_flag_beats_user_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('profile = "exact"\n', encoding="utf-8")
    args = build_parser().parse_args(["serve", "--profile", "sustained"])
    apply_user_config(args, config_path=path)
    assert args.profile == "sustained"


def test_legacy_user_config_applies_without_explicit_flag(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('profile = "exact"\n', encoding="utf-8")
    args = build_parser().parse_args(["serve"])
    apply_user_config(args, config_path=path)
    assert args.profile == "exact"


def test_historical_settings_set_shape_remains_live_daemon_compatible():
    args = build_parser().parse_args(["settings", "set", "depth=2"])
    assert args.settings_action == "set"
    assert args.pairs == ["depth=2"]
    assert args.func.__name__ == "cmd_settings_public"
```

- [x] **Step 2: Run the characterization tests**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_compatibility_baseline.py tests/test_config.py tests/test_config_profile_precedence.py tests/test_cli_parity_tools.py`

Expected: PASS; these tests capture behavior rather than introduce new behavior.

- [x] **Step 3: Commit the behavior lock**

```bash
git add tests/test_settings_compatibility_baseline.py
git commit -m "test: lock settings compatibility baseline"
gh issue comment 90 --repo davidtai/MTPLX --body "Settings task 1 complete: legacy config precedence and live settings syntax are behavior-locked. Commit: $(git rev-parse --short HEAD)."
```

### Task 2: Add Typed Setting Schema

**Files:**
- Create: `mtplx/settings/__init__.py`
- Create: `mtplx/settings/schema.py`
- Create: `tests/test_settings_schema.py`

**Security flag:** `security`

**Does NOT cover:** Source precedence, persistence, CLI application, live mutation, or environment discovery.

- [x] **Step 1: Write failing schema tests**

```python
from __future__ import annotations

import pytest

from mtplx.settings.schema import (
    Lifecycle,
    SettingAlias,
    SettingSpec,
    SettingType,
    Visibility,
)


def test_setting_spec_parses_bool_int_float_and_choice():
    enabled = SettingSpec("runtime.mtp.enabled", SettingType.BOOL, False)
    depth = SettingSpec("runtime.mtp.depth", SettingType.INT, 3, minimum=1)
    temperature = SettingSpec("generation.temperature", SettingType.FLOAT, 0.6, minimum=0.0)
    profile = SettingSpec(
        "runtime.profile",
        SettingType.STRING,
        "sustained",
        choices=("stable", "sustained", "turbo"),
    )
    assert enabled.parse("yes") is True
    assert depth.parse("4") == 4
    assert temperature.parse("0.7") == 0.7
    assert profile.parse("turbo") == "turbo"


def test_setting_spec_rejects_invalid_values_with_canonical_name():
    spec = SettingSpec("runtime.mtp.depth", SettingType.INT, 3, minimum=1)
    with pytest.raises(ValueError, match=r"runtime\.mtp\.depth.*>= 1"):
        spec.parse("0")


def test_alias_metadata_is_orthogonal_to_visibility_and_lifecycle():
    spec = SettingSpec(
        "runtime.profile",
        SettingType.STRING,
        "sustained",
        visibility=Visibility.PUBLIC,
        lifecycle=Lifecycle.ACTIVE,
        aliases=(SettingAlias("profile", "cli"), SettingAlias("MTPLX_PROFILE", "env")),
    )
    assert spec.aliases[0].source == "cli"
    assert spec.visibility is Visibility.PUBLIC
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_schema.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'mtplx.settings'`.

- [x] **Step 3: Implement schema primitives**

```python
# mtplx/settings/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class SettingType(str, Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"


class Visibility(str, Enum):
    PUBLIC = "public"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"
    INTERNAL = "internal"


class Lifecycle(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    COMPATIBILITY = "compatibility"
    RETIRED = "retired"


@dataclass(frozen=True)
class SettingAlias:
    name: str
    source: str
    replacement: str | None = None


@dataclass(frozen=True)
class SettingSpec:
    name: str
    value_type: SettingType
    default: Any
    domain: str = "runtime"
    visibility: Visibility = Visibility.ADVANCED
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    description: str = ""
    aliases: tuple[SettingAlias, ...] = field(default_factory=tuple)
    choices: tuple[str, ...] = field(default_factory=tuple)
    minimum: int | float | None = None
    maximum: int | float | None = None
    secret: bool = False
    live_mutable: bool = False
    validator: Callable[[Any], Any] | None = None

    def parse(self, raw: Any) -> Any:
        if self.value_type is SettingType.BOOL:
            if isinstance(raw, bool):
                value = raw
            else:
                text = str(raw).strip().lower()
                if text in {"1", "true", "yes", "on"}:
                    value = True
                elif text in {"0", "false", "no", "off"}:
                    value = False
                else:
                    raise ValueError(f"{self.name}: expected boolean")
        elif self.value_type is SettingType.INT:
            value = int(raw)
        elif self.value_type is SettingType.FLOAT:
            value = float(raw)
        else:
            value = str(raw)
        if self.choices and value not in self.choices:
            raise ValueError(f"{self.name}: expected one of {', '.join(self.choices)}")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.name}: expected >= {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.name}: expected <= {self.maximum}")
        return self.validator(value) if self.validator else value
```

`mtplx/settings/__init__.py` re-exports the five public schema types.

- [x] **Step 4: Verify GREEN**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_schema.py`

Expected: PASS.

- [x] **Step 5: Commit and update issue**

```bash
git add mtplx/settings tests/test_settings_schema.py
git commit -m "feat: add typed settings schema"
gh issue comment 90 --repo davidtai/MTPLX --body "Settings task 2 complete: typed schema and validation primitives landed. Commit: $(git rev-parse --short HEAD)."
```

### Task 3: Add Catalog, Resolver, and Provenance

**Files:**
- Create: `mtplx/settings/catalog.py`
- Create: `mtplx/settings/builtins.py`
- Create: `mtplx/settings/resolver.py`
- Create: `tests/test_settings_resolver.py`
- Modify: `mtplx/settings/__init__.py`

**Security flag:** `security`

**Does NOT cover:** TOML I/O, argparse wiring, environment source scanning, model safety constraints, or live daemon writes.

- [x] **Step 1: Write failing catalog and precedence tests**

```python
from __future__ import annotations

import pytest

from mtplx.settings.builtins import BUILTIN_SETTINGS
from mtplx.settings.catalog import SettingCatalog
from mtplx.settings.resolver import SettingSource, SettingsResolver
from mtplx.settings.schema import SettingSpec, SettingType


def test_catalog_rejects_duplicate_canonical_names_and_aliases():
    spec = SettingSpec("runtime.profile", SettingType.STRING, "sustained")
    with pytest.raises(ValueError, match="duplicate setting"):
        SettingCatalog((spec, spec))


def test_catalog_suggests_close_canonical_name():
    catalog = SettingCatalog(BUILTIN_SETTINGS)
    assert catalog.suggest("generation.temprature")[0] == "generation.temperature"


def test_resolver_uses_documented_source_order_and_keeps_provenance():
    catalog = SettingCatalog(BUILTIN_SETTINGS)
    resolved = SettingsResolver(catalog).resolve(
        {
            SettingSource.PROFILE: {"generation.temperature": 0.5},
            SettingSource.USER: {"generation.temperature": 0.6},
            SettingSource.BUNDLE: {"generation.temperature": 0.7},
            SettingSource.CLI_SET: {"generation.temperature": 0.8},
        }
    )
    assert resolved["generation.temperature"] == 0.8
    record = resolved.provenance["generation.temperature"]
    assert record.source is SettingSource.CLI_SET
    assert [item.source for item in record.shadowed] == [
        SettingSource.BUNDLE,
        SettingSource.USER,
        SettingSource.PROFILE,
    ]


def test_resolver_redacts_secret_provenance():
    catalog = SettingCatalog((SettingSpec(
        "server.api_key",
        SettingType.STRING,
        "",
        secret=True,
    ),))
    resolved = SettingsResolver(catalog).resolve(
        {SettingSource.CLI_SET: {"server.api_key": "top-secret"}}
    )
    assert resolved.provenance["server.api_key"].display_value == "[redacted]"
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_resolver.py`

Expected: FAIL because catalog, builtins, and resolver do not exist.

- [x] **Step 3: Implement catalog lookup and suggestions**

`SettingCatalog` stores `by_name` and `by_alias`, rejects collisions in its
constructor, resolves aliases to a canonical spec, returns domain/visibility
filtered lists, and uses `difflib.get_close_matches(name, by_name, n=3,
cutoff=0.55)` for suggestions.

```python
# mtplx/settings/catalog.py
from __future__ import annotations

from difflib import get_close_matches
from typing import Iterable

from .schema import SettingSpec


class SettingCatalog:
    def __init__(self, specs: Iterable[SettingSpec]):
        self.by_name: dict[str, SettingSpec] = {}
        self.by_alias: dict[tuple[str, str], SettingSpec] = {}
        for spec in specs:
            if spec.name in self.by_name:
                raise ValueError(f"duplicate setting: {spec.name}")
            self.by_name[spec.name] = spec
            for alias in spec.aliases:
                key = (alias.source, alias.name)
                if key in self.by_alias:
                    raise ValueError(f"duplicate setting alias: {alias.source}:{alias.name}")
                self.by_alias[key] = spec

    def require(self, name: str) -> SettingSpec:
        try:
            return self.by_name[name]
        except KeyError as exc:
            suggestions = self.suggest(name)
            suffix = f"; did you mean {', '.join(suggestions)}" if suggestions else ""
            raise ValueError(f"unknown setting: {name}{suffix}") from exc

    def suggest(self, name: str) -> list[str]:
        return get_close_matches(name, self.by_name, n=3, cutoff=0.55)
```

- [x] **Step 4: Define the product settings used by current `UserConfig`**

`BUILTIN_SETTINGS` must contain canonical entries and aliases for all existing
`CONFIG_VALUE_KEYS`: model/model_dir/profile/thermal control, paged KV,
scheduler/batching, active/decode/wait/prefill controls, experimental cohorts,
SSD and RAM session cache controls, context/reasoning/sampling, and API-key
file. Use canonical names from the design, exact current defaults, and each
existing `_RUNTIME_DEFAULTS` CLI alias. Mark `server.api_key_file` secret.

```python
# representative constructor used for every CONFIG_VALUE_KEYS entry
def _setting(
    name: str,
    value_type: SettingType,
    default: object,
    *,
    domain: str,
    config_key: str,
    cli: tuple[str, ...] = (),
    env: tuple[str, ...] = (),
    **kwargs: object,
) -> SettingSpec:
    aliases = (
        SettingAlias(config_key, "config"),
        *(SettingAlias(value, "cli") for value in cli),
        *(SettingAlias(value, "env") for value in env),
    )
    return SettingSpec(name, value_type, default, domain=domain, aliases=aliases, **kwargs)
```

The completed tuple must have exactly 28 canonical settings and a test asserting
that `{alias.name for spec in BUILTIN_SETTINGS for alias in spec.aliases if
alias.source == "config"} == set(CONFIG_VALUE_KEYS)`.

Expose one constructor used by all callers:

```python
def default_setting_catalog() -> SettingCatalog:
    return SettingCatalog(BUILTIN_SETTINGS)
```

- [x] **Step 5: Implement resolver and provenance**

```python
# mtplx/settings/resolver.py
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Mapping

from .catalog import SettingCatalog


class SettingSource(IntEnum):
    BUILTIN = 10
    PROFILE = 20
    MODEL = 30
    USER = 40
    ENV = 50
    BUNDLE = 60
    LEGACY_CLI = 70
    CLI_SET = 80
    CONSTRAINT = 90


@dataclass(frozen=True)
class SourceValue:
    source: SettingSource
    value: Any


@dataclass(frozen=True)
class ProvenanceRecord:
    source: SettingSource
    display_value: Any
    shadowed: tuple[SourceValue, ...]


class ResolvedSettings:
    def __init__(self, values: dict[str, Any], provenance: dict[str, ProvenanceRecord]):
        self._values = MappingProxyType(dict(values))
        self.provenance = MappingProxyType(dict(provenance))

    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        if not redact:
            return dict(self._values)
        return {name: record.display_value for name, record in self.provenance.items()}


class SettingsResolver:
    def __init__(self, catalog: SettingCatalog):
        self.catalog = catalog

    def resolve(self, sources: Mapping[SettingSource, Mapping[str, Any]]) -> ResolvedSettings:
        candidates: dict[str, list[SourceValue]] = {}
        for spec in self.catalog.by_name.values():
            candidates[spec.name] = [SourceValue(SettingSource.BUILTIN, spec.default)]
        for source, mapping in sources.items():
            for name, raw in mapping.items():
                spec = self.catalog.require(name)
                candidates[spec.name].append(SourceValue(source, spec.parse(raw)))
        values: dict[str, Any] = {}
        provenance: dict[str, ProvenanceRecord] = {}
        for name, items in candidates.items():
            ordered = sorted(items, key=lambda item: item.source, reverse=True)
            winner = ordered[0]
            spec = self.catalog.require(name)
            values[name] = winner.value
            provenance[name] = ProvenanceRecord(
                winner.source,
                "[redacted]" if spec.secret and winner.value else winner.value,
                tuple(ordered[1:]),
            )
        return ResolvedSettings(values, provenance)
```

- [x] **Step 6: Verify GREEN and broader config tests**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_schema.py tests/test_settings_resolver.py tests/test_config.py tests/test_profiles.py`

Expected: PASS.

- [x] **Step 7: Commit and update issue**

```bash
git add mtplx/settings tests/test_settings_resolver.py
git commit -m "feat: resolve hierarchical settings with provenance"
gh issue comment 90 --repo davidtai/MTPLX --body "Settings task 3 complete: canonical product catalog, source precedence, and redacted provenance landed. Commit: $(git rev-parse --short HEAD)."
```

### Task 4: Add Bundle Loading and Atomic User Storage

**Files:**
- Create: `mtplx/settings/bundles.py`
- Create: `mtplx/settings/storage.py`
- Create: `tests/test_settings_storage.py`

**Security flag:** `security`

**Does NOT cover:** Lab URIs, arrays, arbitrary nested payloads, shell expansion, secrets, or live daemon settings.

- [x] **Step 1: Write failing storage and bundle tests**

```python
from __future__ import annotations

import os

import pytest

from mtplx.settings.bundles import load_settings_bundle
from mtplx.settings.storage import load_user_settings, update_user_setting


def test_bundle_loads_only_settings_table(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('[settings]\n"runtime.profile" = "turbo"\n', encoding="utf-8")
    assert load_settings_bundle(path) == {"runtime.profile": "turbo"}


def test_bundle_rejects_unknown_top_level_tables(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('[shell]\ncommand = "rm -rf /"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="only \[settings\]"):
        load_settings_bundle(path)


def test_user_update_is_atomic_private_and_round_trips(tmp_path):
    path = tmp_path / "config.toml"
    update_user_setting(path, "runtime.profile", "sustained")
    update_user_setting(path, "generation.temperature", 0.7)
    assert load_user_settings(path) == {
        "runtime.profile": "sustained",
        "generation.temperature": 0.7,
    }
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_api_key_file_reference_can_be_persisted_but_raw_key_cannot(tmp_path):
    path = tmp_path / "config.toml"
    update_user_setting(path, "server.api_key_file", "~/.mtplx/api-key")
    assert load_user_settings(path)["server.api_key_file"] == "~/.mtplx/api-key"
    with pytest.raises(ValueError, match="unknown setting: server.api_key"):
        update_user_setting(path, "server.api_key", "top-secret")
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_storage.py`

Expected: FAIL because bundle/storage modules do not exist.

- [x] **Step 3: Implement strict bundle loading**

`load_settings_bundle` uses `tomllib`, requires exactly the top-level
`settings` table, requires string keys and scalar bool/int/float/string values,
and returns a plain dictionary. It never expands environment variables or
executes content.

- [x] **Step 4: Implement deterministic TOML and atomic writes**

`load_user_settings` reads canonical `[settings]` first and maps existing flat
config aliases through `SettingCatalog.by_alias`. `update_user_setting` and
`unset_user_setting` validate the complete candidate mapping, render sorted
quoted dotted keys under `[settings]`, create a same-directory temporary file
with mode `0o600`, flush and `os.fsync`, then `os.replace`. Preserve unrelated
legacy config keys until their canonical replacement is written. These
functions use `default_setting_catalog()` unless a catalog is explicitly
injected by a test.

- [x] **Step 5: Verify GREEN**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_storage.py tests/test_config.py`

Expected: PASS.

- [x] **Step 6: Commit and update issue**

```bash
git add mtplx/settings tests/test_settings_storage.py
git commit -m "feat: persist and load settings bundles safely"
gh issue comment 90 --repo davidtai/MTPLX --body "Settings task 4 complete: strict data-only bundles and atomic private user settings storage landed. Commit: $(git rev-parse --short HEAD)."
```

### Task 5: Add Settings Inspection, User Scope, and Explicit Live Scope

**Files:**
- Create: `mtplx/commands/settings.py`
- Create: `tests/test_settings_cli.py`
- Modify: `mtplx/cli.py`
- Modify: `mtplx/commands/public.py`

**Security flag:** `security`

**Does NOT cover:** Applying settings to runtime commands or changing the live server API's mutable-key policy.

- [x] **Step 1: Write failing parser and handler tests**

```python
from __future__ import annotations

import json

from mtplx.cli import build_parser, main


def test_settings_parser_has_unambiguous_scopes():
    parser = build_parser()
    assert parser.parse_args(["settings", "show"]).settings_operation == "show"
    user = parser.parse_args(["settings", "user", "set", "runtime.profile=sustained"])
    assert user.settings_scope == "user"
    assert user.settings_operation == "set"
    live = parser.parse_args(["settings", "live", "show"])
    assert live.settings_scope == "live"


def test_settings_explain_is_no_mlx_and_reports_source(capsys, tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[settings]\n"runtime.profile" = "turbo"\n', encoding="utf-8")
    monkeypatch.setenv("MTPLX_CONFIG", str(path))
    assert main(["settings", "explain", "runtime.profile", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "runtime.profile"
    assert payload["value"] == "turbo"
    assert payload["source"] == "USER"


def test_settings_user_rejects_unknown_key_without_writing(capsys, tmp_path):
    path = tmp_path / "config.toml"
    assert main(["settings", "user", "set", "runtime.proflie=turbo", "--config", str(path)]) == 2
    assert not path.exists()
    assert "runtime.profile" in capsys.readouterr().out
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_cli.py`

Expected: FAIL because new settings subcommands are not registered.

- [x] **Step 3: Register the settings command tree**

Add parsers for `show`, `list`, `explain NAME`, `user show`, `user set PAIR...`,
`user unset NAME...`, `live show`, and `live set PAIR...`. Keep root `get` and
`set` compatibility parsers wired to the old handler. Inspection/user parsers
use `mtplx.commands.settings.cmd_settings`; live parsers adapt their namespace
and call the existing `cmd_settings_public`.

- [x] **Step 4: Implement no-MLX handlers**

`cmd_settings` loads catalog, user storage, environment aliases, and profiles;
resolves effective values; renders human tables or JSON; mutates user storage
only for the explicit user scope. `explain` includes canonical name, effective
and requested values, source, shadowed sources, type, visibility, lifecycle,
aliases, and restart/live metadata with secrets redacted.

- [x] **Step 5: Verify GREEN and compatibility**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_cli.py tests/test_cli_parity_tools.py tests/test_settings_compatibility_baseline.py`

Expected: PASS, including historical `settings get/set` parsing.

- [x] **Step 6: Commit and update issue**

```bash
git add mtplx/cli.py mtplx/commands/public.py mtplx/commands/settings.py tests/test_settings_cli.py
git commit -m "feat: add scoped settings commands"
gh issue comment 90 --repo davidtai/MTPLX --body "Settings task 5 complete: no-MLX show/list/explain, explicit user scope, explicit live scope, and legacy live aliases landed. Commit: $(git rev-parse --short HEAD)."
```

### Task 6: Apply `--set` and `--settings` to Runtime Commands

**Files:**
- Create: `mtplx/settings/argparse.py`
- Create: `tests/test_runtime_settings_args.py`
- Modify: `mtplx/cli.py`
- Modify: `mtplx/config.py`
- Modify: `mtplx/settings/builtins.py`

**Security flag:** `security`

**Does NOT cover:** Benchmark lab URIs, model contract constraints, live daemon mutation, or removal of legacy flags.

- [x] **Step 1: Write failing runtime adapter tests**

```python
from __future__ import annotations

import pytest

from mtplx.cli import build_parser
from mtplx.settings.argparse import resolve_args_settings


@pytest.mark.parametrize("command", ["start", "ask", "run", "chat", "serve", "quickstart"])
def test_runtime_command_accepts_generic_set(command):
    args = build_parser().parse_args([
        command,
        "--set", "runtime.profile=turbo",
        "--set", "generation.temperature=0.7",
    ])
    resolved = resolve_args_settings(args, environ={})
    assert resolved["runtime.profile"] == "turbo"
    assert resolved["generation.temperature"] == 0.7
    assert args.profile == "turbo"
    assert args.temperature == 0.7


def test_explicit_generic_set_beats_legacy_flag():
    args = build_parser().parse_args([
        "serve", "--profile", "sustained", "--set", "runtime.profile=turbo"
    ])
    resolved = resolve_args_settings(args, environ={})
    assert resolved["runtime.profile"] == "turbo"
    assert resolved.provenance["runtime.profile"].source.name == "CLI_SET"


def test_settings_bundle_applies_without_mutating_user_config(tmp_path):
    bundle = tmp_path / "run.toml"
    bundle.write_text('[settings]\n"runtime.mtp.depth" = 2\n', encoding="utf-8")
    args = build_parser().parse_args(["run", "--settings", str(bundle)])
    resolve_args_settings(args, environ={})
    assert args.depth == 2
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_runtime_settings_args.py`

Expected: FAIL because generic settings options are not registered.

- [x] **Step 3: Add shared parser options and canonical-to-namespace mapping**

`add_settings_options(parser)` registers repeatable `--set` into
`setting_overrides` and repeatable `--settings` into `settings_bundles`.
`resolve_args_settings` parses `key=value`, loads bundles in order, maps legacy
CLI aliases from `args._cli_flags`, maps recognized environment aliases, loads
user storage and the selected profile, resolves, stores the result on
`args.mtplx_settings`, and applies canonical values through an explicit
`ARG_DEST_BY_SETTING` mapping.

The mapping includes all existing runtime config destinations and must not use
name-shape guessing. It applies only values whose winning source is explicit
(`USER`, `ENV`, `BUNDLE`, `LEGACY_CLI`, `CLI_SET`, or `CONSTRAINT`); catalog
defaults never overwrite a command's existing argparse default.

- [x] **Step 4: Call resolution from `main` after legacy config loading**

Keep `apply_user_config(args)` for compatibility, then call
`resolve_args_settings(args)` for commands that registered settings options.
Settings-native sources may override compatibility values according to the
documented order; legacy-only invocations retain characterized values.

- [x] **Step 5: Verify GREEN and runtime parser compatibility**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_runtime_settings_args.py tests/test_settings_compatibility_baseline.py tests/test_public_cli.py tests/test_config.py tests/test_config_profile_precedence.py`

Expected: PASS.

- [x] **Step 6: Commit and update issue**

```bash
git add mtplx/settings/argparse.py mtplx/settings/builtins.py mtplx/cli.py mtplx/config.py tests/test_runtime_settings_args.py
git commit -m "feat: apply generic settings to runtime commands"
gh issue comment 90 --repo davidtai/MTPLX --body "Settings task 6 complete: runtime commands accept generic --set/--settings with compatibility-preserving resolution. Commit: $(git rev-parse --short HEAD)."
```

### Task 7: Record Hard Constraints Without Hiding Requested Values

**Files:**
- Modify: `mtplx/settings/resolver.py`
- Modify: `mtplx/settings/argparse.py`
- Modify: `mtplx/commands/public.py`
- Create: `tests/test_settings_constraints.py`

**Security flag:** `security`

**Does NOT cover:** Relaxing existing model gates, forcing unsupported MTP, or changing profile defaults.

- [x] **Step 1: Write failing constraint tests**

```python
from __future__ import annotations

from mtplx.settings.builtins import default_setting_catalog
from mtplx.settings.resolver import SettingSource, SettingsResolver


def test_constraint_keeps_requested_and_effective_values():
    resolved = SettingsResolver(default_setting_catalog()).resolve(
        {SettingSource.CLI_SET: {"runtime.mtp.enabled": True}}
    )
    constrained = resolved.with_constraints({
        "runtime.mtp.enabled": (False, "model has no compatible MTP heads")
    })
    assert constrained["runtime.mtp.enabled"] is False
    record = constrained.provenance["runtime.mtp.enabled"]
    assert record.source is SettingSource.CONSTRAINT
    assert record.requested_value is True
    assert record.reason == "model has no compatible MTP heads"


def test_compatible_value_is_not_relabelled_as_constraint():
    resolved = SettingsResolver(default_setting_catalog()).resolve(
        {SettingSource.CLI_SET: {"runtime.mtp.depth": 2}}
    )
    constrained = resolved.with_constraints({
        "runtime.mtp.depth": (2, "model maximum depth is 2")
    })
    assert constrained.provenance["runtime.mtp.depth"].source is SettingSource.CLI_SET
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_constraints.py`

Expected: FAIL because `with_constraints` and requested/reason provenance do not exist.

- [x] **Step 3: Implement immutable constraint application**

`ResolvedSettings.with_constraints` returns a new snapshot. For each changed
value it creates a `CONSTRAINT` provenance record with effective value,
requested value, reason, and the previous winner in `shadowed`. Equal values
preserve their original source.

- [x] **Step 4: Integrate existing model decisions**

After existing inspection/backend checks resolve whether MTP is loadable and
the effective maximum depth, update `args.mtplx_settings` with constraints for
`runtime.mtp.enabled` and `runtime.mtp.depth`. Existing refusal/fallback behavior
remains authoritative; the settings snapshot only explains it. Add assertions
to the existing public CLI model-gate tests that requested/effective values and
reasons are present.

- [x] **Step 5: Verify GREEN and model gates**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_constraints.py tests/test_public_cli.py tests/test_model_catalog.py`

Expected: PASS with unchanged model-gate exit behavior.

- [x] **Step 6: Commit and update issue**

```bash
git add mtplx/settings/resolver.py mtplx/settings/argparse.py mtplx/commands/public.py tests/test_settings_constraints.py tests/test_public_cli.py
git commit -m "feat: explain model constraints in resolved settings"
gh issue comment 90 --repo davidtai/MTPLX --body "Settings task 7 complete: hard model constraints now preserve requested/effective values and reasons without changing existing gates. Commit: $(git rev-parse --short HEAD)."
```

### Task 8: Inventory and Enforce Every Production `MTPLX_*` Name

**Files:**
- Create: `mtplx/settings/legacy_env.py`
- Create: `scripts/audit_settings_catalog.py`
- Create: `tests/test_settings_audit.py`
- Modify: `mtplx/settings/builtins.py`

**Security flag:** `none`

**Does NOT cover:** Migrating all direct environment reads in one change or deciding that every discovered name is user-settable.

- [x] **Step 1: Write failing inventory audit**

```python
from __future__ import annotations

from pathlib import Path

from scripts.audit_settings_catalog import audit_source_settings


def test_every_production_mtplx_name_is_classified():
    root = Path(__file__).resolve().parents[1]
    report = audit_source_settings(root / "mtplx")
    assert report.unclassified == ()
    assert report.duplicate_aliases == ()


def test_new_direct_setting_reads_are_confined_to_compatibility_boundary():
    root = Path(__file__).resolve().parents[1]
    report = audit_source_settings(root / "mtplx")
    assert report.unauthorized_direct_reads == ()
```

- [x] **Step 2: Verify RED**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_audit.py`

Expected: FAIL because the audit script and classification inventory do not exist.

- [x] **Step 3: Implement deterministic source scanning**

The audit script scans Python text with `MTPLX_[A-Z0-9_]+`, reports source
paths/lines, compares names against catalog environment aliases and
`INTERNAL_ENV_SPECS`, and detects direct `os.environ.get`, `os.getenv`, and
`os.environ[...]` reads for canonical registered settings outside explicit
compatibility modules.

Classify all discovered names into domains using reviewed prefix groups:
`MODEL`, `MTP`, `VERIFY`/`COMPILED`, `VLLM_METAL`/`PAGED`, `PREFILL`, `SESSION`,
`CACHE`, `SERVER`/`API`, `OPENCODE`/`PI`/`HERMES`, `THERMAL`, `FORGE`,
`BENCH`, `TRACE`/`DEBUG`, and `PROCESS`. Names without a reviewed prefix are
listed explicitly. Every classification records visibility and lifecycle.

- [x] **Step 4: Generate and review the inventory**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python scripts/audit_settings_catalog.py --write mtplx/settings/legacy_env.py`

Expected: writes a sorted deterministic inventory containing every discovered
name, then exits 0 after a second `--check` run.

Review each `public` or `advanced` classification manually; experimental and
internal names must not appear in public settings help.

- [x] **Step 5: Verify GREEN and catalog drift**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_audit.py tests/test_settings_resolver.py && /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python scripts/audit_settings_catalog.py --check`

Expected: PASS with zero unclassified names, duplicate aliases, or unauthorized
new direct reads. Pre-existing direct reads are enumerated in the compatibility
boundary rather than hidden.

- [x] **Step 6: Commit and update issue**

```bash
git add mtplx/settings/legacy_env.py mtplx/settings/builtins.py scripts/audit_settings_catalog.py tests/test_settings_audit.py
git commit -m "chore: classify and audit MTPLX settings environment"
gh issue comment 90 --repo davidtai/MTPLX --body "Settings task 8 complete: every production MTPLX_* name is classified and catalog drift is test-enforced. Commit: $(git rev-parse --short HEAD). Additional cleanup candidates discovered by the inventory are recorded in follow-up comments."
```

### Task 9: Verify the Settings System Phase

**Files:**
- Modify: `docs/plans/2026-07-16-settings-system.md` (checkboxes only)
- Test: all settings/config/CLI tests and full suite

**Security flag:** `security`

- [x] **Step 1: Run focused settings/config/CLI verification**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q tests/test_settings_schema.py tests/test_settings_resolver.py tests/test_settings_storage.py tests/test_settings_cli.py tests/test_runtime_settings_args.py tests/test_settings_constraints.py tests/test_settings_audit.py tests/test_settings_compatibility_baseline.py tests/test_config.py tests/test_config_profile_precedence.py tests/test_cli_parity_tools.py tests/test_public_cli.py`

Expected: PASS.

- [x] **Step 2: Run Ruff and stub scan**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/ruff check mtplx/settings mtplx/commands/settings.py scripts/audit_settings_catalog.py tests/test_settings_*.py tests/test_runtime_settings_args.py && ! rg -n 'TODO|FIXME|placeholder|NotImplementedError' mtplx/settings mtplx/commands/settings.py scripts/audit_settings_catalog.py`

Expected: PASS and no stub matches.

- [x] **Step 3: Run full suite**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -m pytest -q`

Expected: exit 0 with no new skips or failures.

- [x] **Step 4: Smoke the no-MLX interface**

Run: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python -c 'import sys; from mtplx.cli import main; code = main(["settings", "list", "--json"]); assert code == 0; assert "mlx" not in sys.modules'`

Expected: exit 0 and JSON output without importing `mlx`.

- [x] **Step 5: Record checkpoint**

```bash
gh issue comment 90 --repo davidtai/MTPLX --body "Settings system phase verified: focused settings/config/CLI suite, Ruff, source audit, no-MLX smoke, and full repository suite all pass on $(git rev-parse --short HEAD)."
```
