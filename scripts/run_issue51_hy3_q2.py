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


def validate_depth_authorization(
    depths: Sequence[int], k1_summary: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    requested = tuple(depths)
    if len(requested) != 1 or requested[0] not in {1, 2}:
        raise ValueError("Issue #51 requires exactly one depth, K=1 or K=2")
    if requested == (1,):
        return None
    if k1_summary is None:
        raise ValueError("K=2 requires a passing K=1 summary")
    if k1_summary.get("schema") != "mtplx-issue51-a1-summary-v2":
        raise ValueError("K=2 requires a passing K=1 summary with the v2 schema")
    gate = k1_summary.get("next_k_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("K=1 summary is missing the Next-K gate")
    authorization = {
        "tested_depth": gate.get("tested_depth"),
        "max_depth": gate.get("max_depth"),
        "advance_to_k2": gate.get("advance_to_k2"),
    }
    if authorization != {
        "tested_depth": 1,
        "max_depth": 2,
        "advance_to_k2": True,
    }:
        raise ValueError("K=1 summary does not authorize K=2")
    return authorization


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
    resource_telemetry: bool = False,
    python_executable: str | Path = sys.executable,
) -> tuple[list[str], dict[str, str]]:
    """Build one immutable candidate process and its exact verifier environment."""

    if arm not in A1_PROCESS_CONFIG:
        raise ValueError(f"unsupported A1 candidate arm: {arm!r}")
    if tuple(contexts) != (1024, 2048):
        raise ValueError("A1 contexts must be exactly 1024,2048")
    if len(tuple(depths)) != 1:
        raise ValueError("A1 runs one depth at a time")
    if tuple(depths)[0] not in {1, 2}:
        raise ValueError("A1 depth must be K=1 or K=2")
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
        "--resource-telemetry" if resource_telemetry else "--no-resource-telemetry",
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
    if mode in {"parity", "on"}:
        # Q2 is deliberately outside the production measured-win allowlist.
        # Issue #51 is its isolated promotion experiment, so force only these
        # compiled children and prove actual compiled calls in their payloads.
        environment["MTPLX_COMPILED_VERIFY_FORCE"] = "1"
    else:
        environment.pop("MTPLX_COMPILED_VERIFY_FORCE", None)
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
    resource_telemetry: bool = False,
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
        resource_telemetry=resource_telemetry,
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
    validate_a1_child(payload, arm=arm, depths=depths)
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
    diagnostic_repeats: int,
    output_dir: Path,
    k1_summary: Mapping[str, Any] | None = None,
    python_executable: str | Path = sys.executable,
    child_runner: Callable[..., dict[str, Any]] = run_a1_child,
) -> dict[str, Any]:
    """Run qualification and both A1 ABBA comparisons with per-child checkpoints."""

    # Validate all fixed campaign inputs before claiming the output directory.
    depth_authorization = validate_depth_authorization(depths, k1_summary)
    build_a1_child_invocation(
        arm=A1_CANDIDATES[0],
        contexts=contexts,
        depths=depths,
        output_tokens=output_tokens,
        output_path=Path("unused.json"),
        python_executable=python_executable,
    )
    build_abba_schedule(control="a", candidate="b", retained_pairs=retained_pairs)
    if diagnostic_repeats < 2:
        raise ValueError("diagnostic_repeats must be at least two")
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
            "diagnostic_repeats": diagnostic_repeats,
            "depth_authorization": depth_authorization,
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
        diagnostic_dir = root / "diagnostics"
        diagnostic_dir.mkdir()
        diagnostics = []
        index["diagnostics"] = diagnostics
        _write_json_checkpoint(index_path, index)
        for position in range(diagnostic_repeats):
            arm = "capture-compiled"
            path = diagnostic_dir / f"{position:02d}-{arm}.json"
            child_runner(
                arm=arm,
                contexts=contexts,
                depths=depths,
                output_tokens=output_tokens,
                output_path=path,
                resource_telemetry=True,
                python_executable=python_executable,
            )
            diagnostics.append(_artifact_entry(path, root=root, arm=arm))
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
    a1.add_argument("--depths", type=_integer_csv, default=(1,))
    a1.add_argument("--output-tokens", type=_positive_int, default=128)
    a1.add_argument("--retained-pairs", type=_positive_int, default=8)
    a1.add_argument("--diagnostic-repeats", type=_positive_int, default=4)
    a1.add_argument("--k1-summary", type=Path)
    a1.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage != "a1":
        raise AssertionError(f"unsupported stage: {args.stage}")
    k1_summary = (
        _load_json_object(args.k1_summary.expanduser().resolve())
        if args.k1_summary is not None
        else None
    )
    payload = run_a1_campaign(
        contexts=args.contexts,
        depths=args.depths,
        output_tokens=args.output_tokens,
        retained_pairs=args.retained_pairs,
        diagnostic_repeats=args.diagnostic_repeats,
        output_dir=args.output_dir,
        k1_summary=k1_summary,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
