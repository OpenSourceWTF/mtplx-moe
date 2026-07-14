#!/usr/bin/env python3
"""Run checkpointed, process-isolated Issue #51 Hy3-Q2 campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.benchmarks.issue51 import (  # noqa: E402
    A1_CANDIDATES,
    A1_PROCESS_CONFIG,
    build_abba_schedule,
    validate_a1_child,
)


A1_SCHEMA = "mtplx-issue51-a1-campaign-v1"
A1_COMPARISONS = (
    ("stock-vs-eager", "batched-stock", "capture-eager"),
    ("eager-vs-compiled", "capture-eager", "capture-compiled"),
)


def _integer_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(
            int(piece.strip()) for piece in value.split(",") if piece.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if (
        not parsed
        or any(item <= 0 for item in parsed)
        or len(set(parsed)) != len(parsed)
    ):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"child artifact must be a JSON object: {path}")
    return payload


def _write_json_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_a1_child_invocation(
    *,
    arm: str,
    contexts: Sequence[int],
    depths: Sequence[int],
    output_tokens: int,
    output_path: Path,
    python_executable: str | Path = sys.executable,
) -> tuple[list[str], dict[str, str]]:
    """Build one immutable candidate process and its exact verifier environment."""

    if arm not in A1_PROCESS_CONFIG:
        raise ValueError(f"unsupported A1 candidate arm: {arm!r}")
    if tuple(contexts) != (1024, 2048):
        raise ValueError("A1 contexts must be exactly 1024,2048")
    if tuple(depths) != (1, 2):
        raise ValueError("A1 depths must be exactly 1,2")
    if output_tokens != 128:
        raise ValueError("A1 output_tokens must be exactly 128")
    mode, strategy = A1_PROCESS_CONFIG[arm]
    command = [
        str(python_executable),
        str(_ROOT / "scripts" / "benchmark_q2_mtp_depth_matrix.py"),
        "--model",
        "hy3-q2",
        "--contexts",
        ",".join(str(item) for item in contexts),
        "--hy3-depths",
        ",".join(str(item) for item in depths),
        "--verify-strategy",
        strategy,
        "--compiled-verify-mode",
        mode,
        "--no-trace-routes",
        "--output-json",
        str(output_path),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "MTPLX_COMPILED_VERIFY": mode,
            "MTPLX_SUSTAINED_PREFILL": "1",
        }
    )
    for name in (
        "MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS",
        "MTPLX_LATE_DEPTH_BEFORE",
        "MTPLX_LATE_DEPTH_AFTER",
    ):
        environment.pop(name, None)
    return command, environment


def run_a1_child(
    *,
    arm: str,
    contexts: Sequence[int],
    depths: Sequence[int],
    output_tokens: int,
    output_path: Path,
    python_executable: str | Path = sys.executable,
    run_process: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Run and validate exactly one candidate without replacing prior evidence."""

    output_path = output_path.expanduser().resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite child artifact: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"child output directory does not exist: {output_path.parent}"
        )
    command, environment = build_a1_child_invocation(
        arm=arm,
        contexts=contexts,
        depths=depths,
        output_tokens=output_tokens,
        output_path=output_path,
        python_executable=python_executable,
    )
    result = run_process(
        command,
        cwd=_ROOT,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    returncode = getattr(result, "returncode", None)
    if returncode != 0:
        detail = (
            getattr(result, "stderr", "") or getattr(result, "stdout", "")
        ).strip()
        raise RuntimeError(
            f"A1 child {arm} failed with exit {returncode}: {detail or 'no output'}"
        )
    if not output_path.is_file() or output_path.is_symlink():
        raise RuntimeError(f"A1 child did not create a regular artifact: {output_path}")
    payload = _load_json_object(output_path)
    validate_a1_child(payload, arm=arm)
    return payload


def _artifact_entry(path: Path, *, root: Path, arm: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _file_sha256(path),
        "arm": arm,
    }


def run_a1_campaign(
    *,
    contexts: Sequence[int],
    depths: Sequence[int],
    output_tokens: int,
    retained_pairs: int,
    output_dir: Path,
    python_executable: str | Path = sys.executable,
    child_runner: Callable[..., dict[str, Any]] = run_a1_child,
) -> dict[str, Any]:
    """Run qualification and both A1 ABBA comparisons with per-child checkpoints."""

    # Validate all fixed campaign inputs before claiming the output directory.
    build_a1_child_invocation(
        arm=A1_CANDIDATES[0],
        contexts=contexts,
        depths=depths,
        output_tokens=output_tokens,
        output_path=Path("unused.json"),
        python_executable=python_executable,
    )
    build_abba_schedule(control="a", candidate="b", retained_pairs=retained_pairs)
    root = output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    index_path = root / "index.json"
    index: dict[str, Any] = {
        "schema": A1_SCHEMA,
        "stage": "a1",
        "status": "running",
        "failure": None,
        "configuration": {
            "contexts": list(contexts),
            "depths": list(depths),
            "output_tokens": output_tokens,
            "retained_pairs": retained_pairs,
        },
        "qualifications": [],
        "comparisons": [],
    }
    _write_json_checkpoint(index_path, index)

    try:
        qualification_dir = root / "qualification"
        qualification_dir.mkdir()
        qualifications = index["qualifications"]
        for position, arm in enumerate(A1_CANDIDATES):
            path = qualification_dir / f"{position:02d}-{arm}.json"
            child_runner(
                arm=arm,
                contexts=contexts,
                depths=depths,
                output_tokens=output_tokens,
                output_path=path,
                python_executable=python_executable,
            )
            qualifications.append(_artifact_entry(path, root=root, arm=arm))
            _write_json_checkpoint(index_path, index)

        performance_dir = root / "performance"
        performance_dir.mkdir()
        comparisons = index["comparisons"]
        for name, control, candidate in A1_COMPARISONS:
            schedule = build_abba_schedule(
                control=control,
                candidate=candidate,
                retained_pairs=retained_pairs,
            )
            comparison = {
                "name": name,
                "control": control,
                "candidate": candidate,
                "schedule": [asdict(row) for row in schedule],
                "artifacts": [],
            }
            comparisons.append(comparison)
            _write_json_checkpoint(index_path, index)
            comparison_dir = performance_dir / name
            comparison_dir.mkdir()
            for row in schedule:
                path = comparison_dir / f"{row.index:02d}-{row.arm}.json"
                child_runner(
                    arm=row.arm,
                    contexts=contexts,
                    depths=depths,
                    output_tokens=output_tokens,
                    output_path=path,
                    python_executable=python_executable,
                )
                entry = _artifact_entry(path, root=root, arm=row.arm)
                entry.update(
                    {
                        "schedule_index": row.index,
                        "block": row.block,
                        "pair_slot": row.pair_slot,
                    }
                )
                comparison["artifacts"].append(entry)
                _write_json_checkpoint(index_path, index)
    except BaseException as exc:
        index["status"] = "failed"
        index["failure"] = {"error": str(exc), "error_type": type(exc).__name__}
        _write_json_checkpoint(index_path, index)
        raise
    index["status"] = "passed"
    _write_json_checkpoint(index_path, index)
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    a1 = subparsers.add_parser("a1", help="Run compiled-verifier A1 campaign")
    a1.add_argument("--contexts", type=_integer_csv, default=(1024, 2048))
    a1.add_argument("--depths", type=_integer_csv, default=(1, 2))
    a1.add_argument("--output-tokens", type=_positive_int, default=128)
    a1.add_argument("--retained-pairs", type=_positive_int, default=8)
    a1.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage != "a1":
        raise AssertionError(f"unsupported stage: {args.stage}")
    payload = run_a1_campaign(
        contexts=args.contexts,
        depths=args.depths,
        output_tokens=args.output_tokens,
        retained_pairs=args.retained_pairs,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
