#!/usr/bin/env python3
"""Run the Qwen 27B K2 cohort construction gate on the actual model.

Normal module import is standard-library-only. MLX and MTPLX are imported
after the selected worktree and exact local model path have been validated.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


EXPECTED_MODEL_DIRECTORY = "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed"


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    worktree = script_path.parent.parent.resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if model_path.name != EXPECTED_MODEL_DIRECTORY or not model_path.is_dir():
        raise RuntimeError(f"exact local Qwen model is unavailable: {model_path}")

    os.environ["PYTHONPATH"] = str(worktree)
    sys.path.insert(0, str(worktree))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    import mtplx

    mtplx_source = Path(mtplx.__file__).resolve()
    if not mtplx_source.is_relative_to(worktree):
        raise RuntimeError(
            f"mtplx resolved outside the self-check worktree: {mtplx_source}"
        )

    import mlx.core as mx

    from mtplx.backends.descriptors import descriptor_from_runtime
    from mtplx.profiles import apply_profile_env
    from mtplx.qwen27b_mtp_cohort import (
        EXPECTED_BACKEND_ID,
        EXPECTED_DEPTH,
        EXPECTED_VERIFY_CORE,
        EXPECTED_VERIFY_STRATEGY,
        install_qwen27b_k2_dual_lane,
        validate_qwen27b_mtp_cohort_selfcheck_report,
    )
    from mtplx.runtime import load

    mlx_version = importlib.metadata.version("mlx")
    if not mlx_version.startswith("0.32."):
        raise RuntimeError(f"expected MLX 0.32.x, got {mlx_version}")

    started = time.perf_counter()
    apply_profile_env("turbo")
    runtime = load(model_path, mtp=True)
    descriptor = descriptor_from_runtime(runtime)
    if descriptor.backend_id != EXPECTED_BACKEND_ID:
        raise RuntimeError(
            f"backend must be {EXPECTED_BACKEND_ID!r}, "
            f"got {descriptor.backend_id!r}"
        )
    lane = install_qwen27b_k2_dual_lane(
        runtime,
        backend_id=descriptor.backend_id,
        depth=EXPECTED_DEPTH,
        verify_strategy=EXPECTED_VERIFY_STRATEGY,
        verify_core=EXPECTED_VERIFY_CORE,
    )
    raw_report = lane.construction_receipt.get("actual_model_selfcheck")
    report = validate_qwen27b_mtp_cohort_selfcheck_report(raw_report)
    receipt = {
        **_json_value(report),
        "model_path": str(model_path),
        "worktree": str(worktree),
        "mtplx_source": str(mtplx_source),
        "mlx_version": mlx_version,
        "backend_id": descriptor.backend_id,
        "construction_elapsed_s": time.perf_counter() - started,
        "construction_receipt": _json_value(lane.construction_receipt),
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }
    _atomic_json_write(output_path, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output_path),
                "qlinear_routes": receipt["qlinear"]["tested_module_count"],
                "construction_elapsed_s": receipt["construction_elapsed_s"],
                "peak_memory_bytes": receipt["peak_memory_bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the actual-model Qwen 27B B2 cohort self-check",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
