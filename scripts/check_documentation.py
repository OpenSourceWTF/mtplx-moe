#!/usr/bin/env python3
"""Validate repository documentation without executing model commands."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"```(?:bash|sh|shell)\s*\n(.*?)```", re.DOTALL)
SHELL_OPERATORS = {"&&", "||", ";", "|"}
LEGACY_NORMAL_FLAGS = (
    "--profile sustained",
    "--default-temperature",
    "--default-top-p",
    "--adaptive-policy",
    "MTPLX_COMPILED_VERIFY=",
    "MTPLX_NAX_VERIFY=",
)


@dataclass(frozen=True)
class DocumentationReport:
    missing_links: tuple[str, ...]
    invalid_shell_blocks: tuple[str, ...]
    unknown_commands: tuple[str, ...]
    legacy_normal_path_flags: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not any(asdict(self).values())


def _markdown_files(root: Path) -> tuple[Path, ...]:
    files = [root / "README.md"]
    files.extend(sorted((root / "docs").rglob("*.md")))
    return tuple(path for path in files if path.exists())


def _user_documentation_files(
    root: Path, files: tuple[Path, ...]
) -> tuple[Path, ...]:
    return tuple(
        path
        for path in files
        if path.relative_to(root).parts[:2] != ("docs", "plans")
    )


def _link_target(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    # Markdown permits an optional quoted link title after the destination.
    try:
        return shlex.split(target)[0]
    except (ValueError, IndexError):
        return target


def _missing_links(root: Path, files: tuple[Path, ...]) -> tuple[str, ...]:
    missing: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            target = _link_target(match.group(1))
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative = target.split("#", 1)[0]
            if not relative:
                continue
            destination = (path.parent / relative).resolve()
            if not destination.exists():
                line = text.count("\n", 0, match.start()) + 1
                missing.append(f"{path.relative_to(root)}:{line}:{target}")
    return tuple(sorted(missing))


def _logical_shell_lines(block: str) -> tuple[tuple[int, str], ...]:
    logical: list[tuple[int, str]] = []
    current = ""
    start = 1
    for number, raw in enumerate(block.splitlines(), 1):
        stripped = raw.strip()
        if not current:
            start = number
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        current += stripped
        if current and not current.lstrip().startswith("#"):
            logical.append((start, current))
        current = ""
    if current:
        logical.append((start, current))
    return tuple(logical)


def _mtplx_argv(tokens: list[str]) -> list[str] | None:
    for index, token in enumerate(tokens):
        if token != "mtplx":
            continue
        argv: list[str] = []
        for value in tokens[index + 1 :]:
            if value in SHELL_OPERATORS:
                break
            argv.append(value)
        return argv
    return None


def _shell_checks(
    root: Path, files: tuple[Path, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    from mtplx.cli import build_parser

    invalid: list[str] = []
    unknown: list[str] = []
    parser = build_parser()
    for path in files:
        text = path.read_text(encoding="utf-8")
        for fence in FENCE_RE.finditer(text):
            fence_line = text.count("\n", 0, fence.start()) + 1
            for relative_line, command in _logical_shell_lines(fence.group(1)):
                location = f"{path.relative_to(root)}:{fence_line + relative_line}"
                try:
                    tokens = shlex.split(command, comments=True)
                except ValueError as exc:
                    invalid.append(f"{location}:{exc}")
                    continue
                argv = _mtplx_argv(tokens)
                if argv is None or not argv:
                    continue
                if "..." in command:
                    # Deliberate illustrative tokens are syntax-checked by
                    # shlex above but cannot be meaningfully argparse-checked.
                    continue
                try:
                    with contextlib.redirect_stderr(io.StringIO()):
                        parser.parse_known_args(argv)
                except (SystemExit, ValueError) as exc:
                    unknown.append(f"{location}:{' '.join(argv)}:{exc}")
    return tuple(sorted(invalid)), tuple(sorted(unknown))


def check_documentation(root: str | Path) -> DocumentationReport:
    resolved = Path(root).resolve()
    files = _markdown_files(resolved)
    invalid, unknown = _shell_checks(
        resolved, _user_documentation_files(resolved, files)
    )
    readme = (resolved / "README.md").read_text(encoding="utf-8")
    normal = readme.split("## Advanced and compatibility", 1)[0]
    legacy = tuple(flag for flag in LEGACY_NORMAL_FLAGS if flag in normal)
    return DocumentationReport(
        missing_links=_missing_links(resolved, files),
        invalid_shell_blocks=invalid,
        unknown_commands=unknown,
        legacy_normal_path_flags=legacy,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = check_documentation(REPO_ROOT)
    payload = asdict(report) | {"ok": report.ok}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif report.ok:
        print("documentation ok")
    else:
        for category, items in asdict(report).items():
            for item in items:
                print(f"{category}: {item}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
