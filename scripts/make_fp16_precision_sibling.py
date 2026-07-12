#!/usr/bin/env python3
"""Build an FP16 precision sibling of an MTPLX model artifact.

Rewrite (2026-07-07) of the lost 2026-05-09 converter that produced
``Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16``. The policy is fully
specified by that artifact's ``MTPLX_FP16_CONVERSION_MANIFEST.json``:

- convert bf16 floating tensors to fp16 (weights, scales, biases, norms);
- byte-preserve packed integer/quantized tensors (uint32 packs) and any
  non-bf16 float tensors;
- apply the same policy to the MTP sidecar (``mtp/weights.safetensors``
  and/or root ``mtp.safetensors``), preserving safetensors metadata;
- copy every other file verbatim;
- emit ``MTPLX_FP16_CONVERSION_MANIFEST.json`` with per-file source/output
  sha256 + per-tensor converted/preserved lists (same schema as the
  Speed-FP16 manifest);
- patch ``mtplx_runtime.json``: ``precision_variant: fp16``, an m1/m2
  ``precision_policy`` routing note, and ``recommended_profile``
  (default turbo — the 2.0.1 promotion).

Usage:
  python scripts/make_fp16_precision_sibling.py \
    --source ~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality \
    --output ~/Documents/MTPLX/hf-staging/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16 \
    --repo-id Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_safetensors(source: Path, output: Path) -> dict[str, Any]:
    import mlx.core as mx

    tensors, metadata = mx.load(str(source), return_metadata=True)
    tensor_rows: list[dict[str, Any]] = []
    converted = 0
    preserved = 0
    out_tensors: dict[str, Any] = {}
    for name in sorted(tensors):
        value = tensors[name]
        old_dtype = str(value.dtype).removeprefix("mlx.core.")
        if value.dtype == mx.bfloat16:
            value = value.astype(mx.float16)
            converted += 1
            was_converted = True
        else:
            preserved += 1
            was_converted = False
        out_tensors[name] = value
        tensor_rows.append(
            {
                "name": name,
                "converted": was_converted,
                "old_dtype": old_dtype,
                "new_dtype": str(value.dtype).removeprefix("mlx.core."),
                "shape": list(value.shape),
            }
        )
    mx.eval(list(out_tensors.values()))
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output), out_tensors, metadata=dict(metadata or {}))
    del out_tensors, tensors
    mx.clear_cache()
    return {
        "name": str(output.name),
        "tensor_count": len(tensor_rows),
        "converted_bf16_to_fp16": converted,
        "preserved": preserved,
        "source_sha256": _sha256(source),
        "source_size_bytes": source.stat().st_size,
        "output_sha256": _sha256(output),
        "output_size_bytes": output.stat().st_size,
        "tensors": tensor_rows,
    }


def patch_runtime_manifest(
    path: Path,
    *,
    recommended_profile: str,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["precision_variant"] = "fp16"
    manifest["precision_policy"] = {
        "variant": "fp16",
        "intended_default_for": ["m1", "m2"],
        "routing": "mtplx start auto-selects this artifact on M1/M2 Apple Silicon",
        "note": (
            "This is a sibling precision variant; it is not a universal "
            "speed claim."
        ),
    }
    manifest["recommended_profile"] = recommended_profile
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--recommended-profile", default="turbo")
    args = parser.parse_args()

    source: Path = args.source.expanduser().resolve()
    output: Path = args.output.expanduser()
    if not source.is_dir():
        raise SystemExit(f"source is not a directory: {source}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    shard_reports: list[dict[str, Any]] = []
    copied: list[dict[str, Any]] = []
    for item in sorted(source.rglob("*")):
        rel = item.relative_to(source)
        target = output / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if item.suffix == ".safetensors":
            print(f"[convert] {rel}", flush=True)
            shard_reports.append(convert_safetensors(item, target))
        elif item.name == "MTPLX_FP16_CONVERSION_MANIFEST.json":
            continue  # never inherit a parent conversion manifest
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            copied.append(
                {"kind": "file-copy", "name": str(rel), "size_bytes": item.stat().st_size}
            )

    runtime_path = output / "mtplx_runtime.json"
    if runtime_path.exists():
        patch_runtime_manifest(
            runtime_path,
            recommended_profile=str(args.recommended_profile),
        )
        print(f"[patch] mtplx_runtime.json -> precision_variant=fp16, "
              f"recommended_profile={args.recommended_profile}", flush=True)

    # config.json: neither the Quality parent nor the Speed-FP16 sibling
    # carries a torch_dtype/dtype field; patch only if one exists.
    config_path = output / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        touched = False
        for key in ("torch_dtype", "dtype"):
            if str(config.get(key, "")) == "bfloat16":
                config[key] = "float16"
                touched = True
        if touched:
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            print("[patch] config.json dtype -> float16", flush=True)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "policy": (
            "convert bf16 floating tensors to fp16; preserve packed "
            "integer/quantized tensors"
        ),
        "repo_id": str(args.repo_id),
        "source_path": str(source),
        "source_repo": None,
        "output_path": str(output),
        "files_copied": copied,
        "safetensors": shard_reports,
        "summary": {
            "shards": len(shard_reports),
            "tensors_total": sum(r["tensor_count"] for r in shard_reports),
            "converted_bf16_to_fp16": sum(
                r["converted_bf16_to_fp16"] for r in shard_reports
            ),
            "preserved": sum(r["preserved"] for r in shard_reports),
        },
    }
    (output / "MTPLX_FP16_CONVERSION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    print(f"done: {output}", flush=True)


if __name__ == "__main__":
    main()
