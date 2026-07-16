#!/usr/bin/env python3
"""Audit and generate the production ``MTPLX_*`` compatibility inventory."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ENV_NAME_RE = re.compile(r"\bMTPLX_[A-Z0-9_]+\b")
DIRECT_READ_PATTERNS = (
    re.compile(
        r"os\.(?:environ\.get|getenv)\(\s*[\"'](?P<name>MTPLX_[A-Z0-9_]+)[\"']"
    ),
    re.compile(
        r"os\.environ\[\s*[\"'](?P<name>MTPLX_[A-Z0-9_]+)[\"']\s*\]"
    ),
)
GENERATED_PATH = "mtplx/settings/legacy_env.py"


@dataclass(frozen=True, order=True)
class DirectRead:
    path: str
    line: int
    name: str


@dataclass(frozen=True)
class SettingsAuditReport:
    discovered: tuple[str, ...]
    unclassified: tuple[str, ...]
    duplicate_aliases: tuple[str, ...]
    unauthorized_direct_reads: tuple[str, ...]
    direct_reads: tuple[DirectRead, ...]


def _python_sources(root: Path):
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root.parent).as_posix()
        if relative == GENERATED_PATH:
            continue
        yield path, relative


def _scan(root: Path) -> tuple[set[str], tuple[DirectRead, ...]]:
    discovered: set[str] = set()
    direct_reads: set[DirectRead] = set()
    for path, relative in _python_sources(root):
        text = path.read_text(encoding="utf-8")
        discovered.update(ENV_NAME_RE.findall(text))
        for pattern in DIRECT_READ_PATTERNS:
            for match in pattern.finditer(text):
                direct_reads.add(
                    DirectRead(
                        relative,
                        text.count("\n", 0, match.start()) + 1,
                        match.group("name"),
                    )
                )
    return discovered, tuple(sorted(direct_reads))


def _catalog_env_aliases() -> tuple[str, ...]:
    from mtplx.settings.builtins import default_setting_catalog

    return tuple(
        alias.name
        for spec in default_setting_catalog().by_name.values()
        for alias in spec.aliases
        if alias.source == "env"
    )


def audit_source_settings(root: str | Path) -> SettingsAuditReport:
    from mtplx.settings.legacy_env import (
        DIRECT_READ_ALLOWLIST,
        INTERNAL_ENV_SPECS,
    )

    source_root = Path(root).resolve()
    discovered, direct_reads = _scan(source_root)
    catalog_aliases = _catalog_env_aliases()
    classified = (*catalog_aliases, *INTERNAL_ENV_SPECS)
    counts = Counter(classified)
    duplicates = tuple(sorted(name for name, count in counts.items() if count > 1))
    unclassified = tuple(sorted(discovered.difference(classified)))
    unauthorized = tuple(
        f"{read.path}:{read.line}:{read.name}"
        for read in direct_reads
        if (read.path, read.name) not in DIRECT_READ_ALLOWLIST
    )
    return SettingsAuditReport(
        tuple(sorted(discovered)),
        unclassified,
        duplicates,
        unauthorized,
        direct_reads,
    )


_PREFIX_DOMAINS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("MODEL",), "model", "MODEL"),
    (("MTP",), "runtime", "MTP"),
    (("VERIFY", "COMPILED"), "verify", "VERIFY_COMPILED"),
    (
        ("VLLM_METAL", "PAGED", "DYNAMIC_PAGED"),
        "attention",
        "VLLM_METAL_PAGED",
    ),
    (("PREFILL", "SUSTAINED", "TARGET"), "runtime", "PREFILL"),
    (("SESSION",), "cache", "SESSION"),
    (("CACHE", "CLEAR", "KV"), "cache", "CACHE"),
    (("SERVER", "API", "DAEMON", "RATE"), "server", "SERVER_API"),
    (
        ("OPENCODE", "PI", "HERMES", "SWIVAL", "OPENWEBUI", "ANDROID"),
        "integration",
        "INTEGRATION",
    ),
    (("THERMAL", "FAN"), "thermal", "THERMAL"),
    (("FORGE",), "thermal", "FORGE"),
    (("BENCH", "TUNE", "PROFILE"), "benchmark", "BENCH"),
    (
        ("TRACE", "DEBUG", "LOG", "DROP", "EVAL"),
        "diagnostics",
        "TRACE_DEBUG",
    ),
    (("PROCESS", "PYTHON"), "process", "PROCESS"),
)


def _classify(name: str) -> tuple[str, str]:
    suffix = name.removeprefix("MTPLX_")
    for prefixes, domain, label in _PREFIX_DOMAINS:
        if any(suffix == prefix or suffix.startswith(prefix + "_") for prefix in prefixes):
            return domain, f"prefix:{label}"
    # Names outside a reviewed prefix are intentionally listed one-by-one in
    # the generated inventory and remain internal until separately promoted.
    return "internal", "explicit"


def render_inventory(root: str | Path) -> str:
    source_root = Path(root).resolve()
    discovered, direct_reads = _scan(source_root)
    catalog_aliases = set(_catalog_env_aliases())
    internal = sorted(discovered.difference(catalog_aliases))
    lines = [
        '"""Generated compatibility inventory. Run scripts/audit_settings_catalog.py."""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "",
        "",
        "@dataclass(frozen=True)",
        "class LegacyEnvSpec:",
        "    domain: str",
        '    visibility: str = "internal"',
        '    lifecycle: str = "compatibility"',
        '    classification: str = "explicit"',
        "",
        "",
        "INTERNAL_ENV_SPECS: dict[str, LegacyEnvSpec] = {",
    ]
    for name in internal:
        domain, classification = _classify(name)
        lines.append(
            f'    "{name}": LegacyEnvSpec("{domain}", classification="{classification}"),'
        )
    lines.extend(
        [
            "}",
            "",
            "# Existing direct reads are a frozen compatibility boundary. New reads",
            "# must use the resolver or be explicitly reviewed into this inventory.",
            "DIRECT_READ_ALLOWLIST = frozenset(",
            "    {",
        ]
    )
    for path, name in sorted({(read.path, read.name) for read in direct_reads}):
        lines.append(f'        ("{path}", "{name}"),')
    lines.extend(["    }", ")", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", metavar="PATH")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    source_root = REPO_ROOT / "mtplx"
    expected = render_inventory(source_root)
    destination = (
        Path(args.write).resolve()
        if args.write
        else source_root / "settings" / "legacy_env.py"
    )
    if args.write:
        destination.write_text(expected, encoding="utf-8")
        print(f"wrote {destination}")
        return 0
    if not destination.exists() or destination.read_text(encoding="utf-8") != expected:
        print(f"error: stale settings inventory: {destination}")
        return 1
    report = audit_source_settings(source_root)
    if report.unclassified or report.duplicate_aliases or report.unauthorized_direct_reads:
        print(report)
        return 1
    print(
        f"settings inventory ok: {len(report.discovered)} names, "
        f"{len(report.direct_reads)} direct reads"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
