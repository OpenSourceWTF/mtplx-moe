"""Named shorthand configurations for the benchmark runners.

The runners carry ~55 flags each and a real run is 25 of them. Nobody should
have to remember that, and a mistyped flag in a 25-flag command is a silently
different experiment. A preset is a named bundle of flag values, defined in
TOML, that expands to exactly those flags.

Presets are applied as argparse **defaults**, not as parsed values. That is
the whole trick: argparse already gives an explicitly-typed flag precedence
over a default, so ``--preset championship --proj-quant q8`` runs the
championship bundle with q8 substituted, and no explicit-flag detection is
needed. (The repo's existing raw-token detection is defeated by argparse
abbreviation — see docs/AUDIT_2026-07-18.md — so avoiding it entirely is
deliberate.)

Precedence, lowest to highest:

    runner defaults  <  preset (through its ``extends`` chain)  <  typed flags

File layout::

    [preset.full-residency]
    description = "All 79 layers resident; nothing streams."
    island-layer-count = 79
    proj-quant = "q4"

    [preset.full-residency.env]
    MTPLX_SUSTAINED_PREFILL = "1"

    [preset.championship]
    description = "The 40.59 tok/s router A/B winner."
    extends = "full-residency"
    hy3-router-kernel = "mpp-fp32-splitk-r1-fused-r2"

Keys are flag spellings (``proj-quant``) or dest spellings (``proj_quant``);
both resolve to the same option. Unknown keys are a hard error rather than a
silent no-op, because a preset that quietly drops ``--proj-quant`` is worse
than one that fails to load.

No MLX imports here: ``--list-presets`` must work in a fresh environment.
"""

from __future__ import annotations

import argparse
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

__all__ = [
    "PresetError",
    "Preset",
    "ResolvedPreset",
    "DEFAULT_PRESET_FILES",
    "REPO_PRESET_FILE",
    "USER_PRESET_FILE",
    "load_preset_files",
    "resolve_preset",
    "apply_preset_defaults",
    "format_preset_list",
    "format_resolved_preset",
    "add_preset_arguments",
    "preselect_preset",
]

# Shipped presets live in the repo so they are reviewable in a diff; the user
# file is layered on top so a machine-local bundle never needs a commit.
REPO_PRESET_FILE = Path(__file__).resolve().parents[2] / "benchmarks" / "presets.toml"
USER_PRESET_FILE = Path("~/.mtplx/benchmark-presets.toml").expanduser()
DEFAULT_PRESET_FILES: tuple[Path, ...] = (REPO_PRESET_FILE, USER_PRESET_FILE)

_RESERVED_KEYS = frozenset({"description", "extends", "env"})


class PresetError(ValueError):
    """Raised for a malformed, missing, or self-referential preset."""


@dataclass(frozen=True)
class Preset:
    """One preset exactly as written in TOML, before inheritance."""

    name: str
    description: str = ""
    extends: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    source: Path | None = None


@dataclass(frozen=True)
class ResolvedPreset:
    """A preset flattened through its ``extends`` chain."""

    name: str
    description: str
    options: Mapping[str, Any]
    env: Mapping[str, str]
    chain: tuple[str, ...]
    sources: tuple[Path, ...]


def _normalize_key(key: str) -> str:
    return key.strip().replace("-", "_").lstrip("_")


def _parse_preset_table(name: str, table: Any, source: Path) -> Preset:
    if not isinstance(table, dict):
        raise PresetError(f"{source}: preset {name!r} must be a table")

    description = table.get("description", "")
    if not isinstance(description, str):
        raise PresetError(f"{source}: preset {name!r} description must be a string")

    extends = table.get("extends")
    if extends is not None and not isinstance(extends, str):
        raise PresetError(f"{source}: preset {name!r} extends must be a string")

    raw_env = table.get("env", {})
    if not isinstance(raw_env, dict):
        raise PresetError(f"{source}: preset {name!r} env must be a table")
    env: dict[str, str] = {}
    for env_key, env_value in raw_env.items():
        if isinstance(env_value, bool):
            # TOML true/false would stringify as "True"/"False", which no
            # MTPLX_* reader accepts. Fail loudly instead.
            raise PresetError(
                f"{source}: preset {name!r} env {env_key!r} is a TOML boolean; "
                'use the string "1" or "0" -- env vars are text'
            )
        if not isinstance(env_value, (str, int)):
            raise PresetError(
                f"{source}: preset {name!r} env {env_key!r} must be a string"
            )
        env[str(env_key)] = str(env_value)

    options = {
        _normalize_key(key): value
        for key, value in table.items()
        if key not in _RESERVED_KEYS
    }
    return Preset(
        name=name,
        description=description,
        extends=extends,
        options=options,
        env=env,
        source=source,
    )


def load_preset_files(
    paths: Iterable[Path | str] | None = None,
) -> dict[str, Preset]:
    """Load presets from each path in order; later files override earlier.

    Missing files are skipped, so a user file is optional. A malformed file is
    an error: silently ignoring it would run the wrong configuration.
    """

    if paths is None:
        paths = DEFAULT_PRESET_FILES
    presets: dict[str, Preset] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            continue
        try:
            document = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise PresetError(f"could not read presets from {path}: {exc}") from exc
        table = document.get("preset", {})
        if not isinstance(table, dict):
            raise PresetError(f"{path}: top-level [preset] must be a table")
        for name, body in table.items():
            presets[str(name)] = _parse_preset_table(str(name), body, path)
    return presets


def resolve_preset(name: str, presets: Mapping[str, Preset]) -> ResolvedPreset:
    """Flatten ``name`` through its ``extends`` chain, base first."""

    chain: list[str] = []
    seen: set[str] = set()
    cursor: str | None = name
    while cursor is not None:
        if cursor in seen:
            cycle = " -> ".join([*chain, cursor])
            raise PresetError(f"preset {name!r} has a circular extends chain: {cycle}")
        preset = presets.get(cursor)
        if preset is None:
            known = ", ".join(sorted(presets)) or "(none defined)"
            if cursor == name:
                raise PresetError(f"unknown preset {cursor!r}; available: {known}")
            raise PresetError(
                f"preset {chain[-1]!r} extends unknown preset {cursor!r}; "
                f"available: {known}"
            )
        seen.add(cursor)
        chain.append(cursor)
        cursor = preset.extends

    options: dict[str, Any] = {}
    env: dict[str, str] = {}
    sources: list[Path] = []
    descriptions: list[str] = []
    # Walk base-first so the named preset wins over what it extends.
    for entry in reversed(chain):
        preset = presets[entry]
        options.update(preset.options)
        env.update(preset.env)
        if preset.source is not None and preset.source not in sources:
            sources.append(preset.source)
        if preset.description:
            descriptions.append(preset.description)
    return ResolvedPreset(
        name=name,
        description=presets[name].description or (descriptions[-1] if descriptions else ""),
        options=options,
        env=env,
        chain=tuple(reversed(chain)),
        sources=tuple(sources),
    )


def _parser_destinations(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {
        action.dest: action
        for action in parser._actions  # noqa: SLF001 - argparse exposes no public API
        if action.dest and action.dest != "help"
    }


def apply_preset_defaults(
    parser: argparse.ArgumentParser,
    resolved: ResolvedPreset,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    """Push a resolved preset into ``parser`` as defaults, and export its env.

    Returns the option mapping that was applied. Raises ``PresetError`` if the
    preset names an option this runner does not have -- that is almost always
    a typo, and a silently-dropped flag is a silently different experiment.

    Env vars already present in the environment are left alone, so an explicit
    ``MTPLX_FOO=1 ./runner --preset x`` still beats the preset.
    """

    known = _parser_destinations(parser)
    unknown = sorted(key for key in resolved.options if key not in known)
    if unknown:
        available = ", ".join(sorted(known))
        raise PresetError(
            f"preset {resolved.name!r} sets unknown option(s): "
            f"{', '.join(unknown)}.\nKnown options: {available}"
        )

    applied = dict(resolved.options)
    for dest, value in applied.items():
        action = known[dest]
        choices = getattr(action, "choices", None)
        if choices is None or value is None:
            continue
        # `append`/`nargs` actions validate each ELEMENT against choices, not
        # the collection, so a preset for them must supply a list and be
        # checked the same way argparse would.
        multi = isinstance(action, argparse._AppendAction) or action.nargs not in (
            None,
            0,
        )
        candidates = value if multi and isinstance(value, (list, tuple)) else [value]
        for candidate in candidates:
            if candidate not in choices:
                raise PresetError(
                    f"preset {resolved.name!r} sets {dest}={candidate!r}, which "
                    f"is not one of {tuple(choices)!r}"
                )
    parser.set_defaults(**applied)

    target = os.environ if environ is None else environ
    for key, value in resolved.env.items():
        target.setdefault(key, value)
    return applied



def format_preset_list(presets: Mapping[str, Preset]) -> str:
    if not presets:
        return "No presets defined. Expected a [preset.<name>] table in:\n  " + "\n  ".join(
            str(path) for path in DEFAULT_PRESET_FILES
        )
    width = max(len(name) for name in presets)
    lines = ["Available presets:", ""]
    for name in sorted(presets):
        preset = presets[name]
        suffix = f"  (extends {preset.extends})" if preset.extends else ""
        lines.append(f"  {name.ljust(width)}  {preset.description}{suffix}")
    lines.append("")
    lines.append("Show the expanded flags for one:  --show-preset <name>")
    return "\n".join(lines)


def format_resolved_preset(resolved: ResolvedPreset) -> str:
    lines = [f"preset: {resolved.name}"]
    if resolved.description:
        lines.append(f"  {resolved.description}")
    if len(resolved.chain) > 1:
        lines.append(f"  chain: {' -> '.join(resolved.chain)}")
    for source in resolved.sources:
        lines.append(f"  from:  {source}")
    lines.append("")
    if resolved.options:
        lines.append("flags:")
        width = max(len(key) for key in resolved.options)
        for key in sorted(resolved.options):
            flag = "--" + key.replace("_", "-")
            lines.append(f"  {flag.ljust(width + 2)}  {resolved.options[key]!r}")
    else:
        lines.append("flags: (none)")
    if resolved.env:
        lines.append("")
        lines.append("env:")
        for key in sorted(resolved.env):
            lines.append(f"  {key}={resolved.env[key]}")
    return "\n".join(lines)


def add_preset_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the preset flags on a runner's parser."""

    group = parser.add_argument_group("presets")
    group.add_argument(
        "--preset",
        default=None,
        help=(
            "Named shorthand configuration to apply before other flags. Any "
            "flag typed explicitly overrides the preset. See --list-presets."
        ),
    )
    group.add_argument(
        "--presets-file",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Preset TOML to load instead of the defaults "
            f"({REPO_PRESET_FILE} then {USER_PRESET_FILE}). Repeatable; "
            "later files override earlier ones."
        ),
    )
    group.add_argument(
        "--list-presets",
        action="store_true",
        help="Print the available presets and exit.",
    )
    group.add_argument(
        "--show-preset",
        default=None,
        metavar="NAME",
        help="Print the fully expanded flags for one preset and exit.",
    )


def _scan_option(argv: Sequence[str], option: str) -> str | None:
    """Find ``--option value`` or ``--option=value`` in a raw argv."""

    for index, token in enumerate(argv):
        if token == option and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(option + "="):
            return token.split("=", 1)[1]
    return None


def _scan_repeated(argv: Sequence[str], option: str) -> list[str]:
    found: list[str] = []
    for index, token in enumerate(argv):
        if token == option and index + 1 < len(argv):
            found.append(argv[index + 1])
        elif token.startswith(option + "="):
            found.append(token.split("=", 1)[1])
    return found


def preselect_preset(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
    *,
    environ: MutableMapping[str, str] | None = None,
) -> ResolvedPreset | None:
    """Handle the preset flags before the main parse.

    Scans ``argv`` for ``--preset`` / ``--presets-file`` / ``--list-presets`` /
    ``--show-preset``, applies the preset to ``parser`` as defaults, and
    returns it. ``--list-presets`` and ``--show-preset`` print and raise
    ``SystemExit(0)``.

    The raw scan is intentional: the preset must be applied *before*
    ``parse_args`` so argparse resolves precedence, which means it cannot come
    from the parsed namespace. Only exact long-form spellings are honoured --
    abbreviations are not, which keeps this immune to the abbreviation
    ambiguity that breaks token-based explicit-flag detection elsewhere.
    """

    files = _scan_repeated(argv, "--presets-file") or None
    presets = load_preset_files(files)

    if "--list-presets" in argv:
        print(format_preset_list(presets))
        raise SystemExit(0)

    show = _scan_option(argv, "--show-preset")
    if show is not None:
        print(format_resolved_preset(resolve_preset(show, presets)))
        raise SystemExit(0)

    name = _scan_option(argv, "--preset")
    if name is None:
        return None
    resolved = resolve_preset(name, presets)
    apply_preset_defaults(parser, resolved, environ=environ)
    return resolved
