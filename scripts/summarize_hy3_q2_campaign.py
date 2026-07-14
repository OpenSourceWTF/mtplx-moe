#!/usr/bin/env python3
"""Write one strict Hy3 expert-Q2 campaign summary outside the worktree."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Callable, Sequence


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.benchmarks.hy3_q2_campaign import (  # noqa: E402
    summarize_hy3_q2_campaign,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize paired Hy3 Q4/Q2 benchmark evidence."
    )
    parser.add_argument(
        "--resource",
        nargs="+",
        type=Path,
        required=True,
        help="Four resource payloads in physical ABBA order.",
    )
    parser.add_argument(
        "--headline",
        nargs="+",
        type=Path,
        required=True,
        help="Eight headline payloads in physical paired ABBAABBA order.",
    )
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def validate_external_output_path(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output == _ROOT or output.is_relative_to(_ROOT):
        raise ValueError("--output-json must remain outside the Git worktree")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite campaign evidence: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output.parent}")
    return output


def _load_payload(path: Path, *, label: str) -> dict:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} payload must be a JSON object: {path}")
    return payload


def _load_payloads(paths: Sequence[Path], *, label: str) -> list[dict]:
    return [_load_payload(path, label=label) for path in paths]


class _OutputBinding:
    __slots__ = ("path", "parent_path", "final_name", "parent_fd", "parent_identity")

    def __init__(
        self,
        *,
        path: Path,
        parent_path: Path,
        final_name: str,
        parent_fd: int,
        parent_identity: tuple[int, int],
    ) -> None:
        self.path = path
        self.parent_path = parent_path
        self.final_name = final_name
        self.parent_fd = parent_fd
        self.parent_identity = parent_identity


def _inode_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _bind_output_parent(path: Path) -> _OutputBinding:
    parent_path = path.expanduser().parent.resolve(strict=True)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd = os.open(parent_path, flags)
    try:
        identity = _inode_identity(os.fstat(parent_fd))
        if _inode_identity(os.stat(parent_path, follow_symlinks=False)) != identity:
            raise RuntimeError("output parent was substituted while binding")
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite campaign evidence: {path}")
        return _OutputBinding(
            path=path,
            parent_path=parent_path,
            final_name=path.name,
            parent_fd=parent_fd,
            parent_identity=identity,
        )
    except BaseException:
        os.close(parent_fd)
        raise


def _unlink_if_identity(parent_fd: int, name: str, expected: tuple[int, int]) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if _inode_identity(current) == expected:
        os.unlink(name, dir_fd=parent_fd)


def _revalidate_advertised_output(
    parent_path: Path,
    final_name: str,
    *,
    parent_identity: tuple[int, int],
    final_identity: tuple[int, int],
) -> None:
    try:
        current_parent = _inode_identity(os.stat(parent_path, follow_symlinks=False))
    except FileNotFoundError as exc:
        raise RuntimeError("output parent was substituted or removed") from exc
    if current_parent != parent_identity:
        raise RuntimeError("output parent was substituted after validation")
    try:
        current_final = _inode_identity(
            os.stat(parent_path / final_name, follow_symlinks=False)
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "advertised output final was substituted or removed"
        ) from exc
    if current_final != final_identity:
        raise RuntimeError("advertised output final inode was substituted")


def _write_json_exclusive(
    path: Path,
    payload: dict,
    *,
    binding: _OutputBinding | None = None,
) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    owned_binding = binding is None
    bound = binding or _bind_output_parent(path)
    parent_fd = bound.parent_fd
    temporary_name = f".{bound.final_name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    temporary_fd = -1
    final_fd = -1
    temporary_identity: tuple[int, int] | None = None
    linked = False
    succeeded = False
    try:
        try:
            os.stat(bound.final_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite campaign evidence: {path}")
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        temporary_identity = _inode_identity(os.fstat(temporary_fd))
        view = memoryview(encoded)
        while view:
            written = os.write(temporary_fd, view)
            if written <= 0:
                raise OSError("short write while recording campaign evidence")
            view = view[written:]
        os.fsync(temporary_fd)
        if _inode_identity(os.fstat(temporary_fd)) != temporary_identity:
            raise RuntimeError("output temporary inode was substituted while open")
        os.link(
            temporary_name,
            bound.final_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        final_fd = os.open(
            bound.final_name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        final_stat = os.fstat(final_fd)
        if _inode_identity(
            final_stat
        ) != temporary_identity or final_stat.st_size != len(encoded):
            raise RuntimeError("output final inode was substituted before validation")
        _revalidate_advertised_output(
            bound.parent_path,
            bound.final_name,
            parent_identity=bound.parent_identity,
            final_identity=temporary_identity,
        )
        _unlink_if_identity(parent_fd, temporary_name, temporary_identity)
        os.fsync(parent_fd)
        _revalidate_advertised_output(
            bound.parent_path,
            bound.final_name,
            parent_identity=bound.parent_identity,
            final_identity=temporary_identity,
        )
        succeeded = True
    finally:
        if final_fd >= 0:
            os.close(final_fd)
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_identity is not None:
            if linked and not succeeded:
                _unlink_if_identity(parent_fd, bound.final_name, temporary_identity)
            _unlink_if_identity(parent_fd, temporary_name, temporary_identity)
        if not succeeded:
            os.fsync(parent_fd)
        if owned_binding:
            os.close(parent_fd)


def main(
    argv: Sequence[str] | None = None,
    *,
    _summarize: Callable[..., dict] = summarize_hy3_q2_campaign,
) -> int:
    args = build_parser().parse_args(argv)
    binding: _OutputBinding | None = None
    try:
        output = validate_external_output_path(args.output_json)
        binding = _bind_output_parent(output)
        resource = _load_payloads(args.resource, label="resource")
        headline = _load_payloads(args.headline, label="headline")
        quality = _load_payload(args.quality, label="quality")
        summary = _summarize(resource, headline, quality_payload=quality)
        _write_json_exclusive(output, summary, binding=binding)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"campaign summary failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if binding is not None:
            os.close(binding.parent_fd)
    return 0 if summary["decision"]["eligible"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
