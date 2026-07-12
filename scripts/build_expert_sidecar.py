#!/usr/bin/env python3
"""Build a resumable aligned expert-major sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_manifest import (  # noqa: E402
    DEFAULT_ALIGNMENT,
    ExpertManifestError,
    build_expert_sidecar,
    load_expert_manifest,
    save_expert_manifest,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pack manifest-described source slices into aligned expert records."
    )
    parser.add_argument("model_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--alignment", type=_positive_int, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.model_root.expanduser().resolve()
    manifest_output = args.manifest_output or args.manifest
    try:
        manifest = load_expert_manifest(args.manifest)
        manifest = build_expert_sidecar(
            manifest,
            root,
            args.output,
            alignment=args.alignment,
            resume=not args.no_resume,
            overwrite=args.overwrite,
        )
        manifest = save_expert_manifest(manifest, manifest_output)
    except ExpertManifestError as exc:
        raise SystemExit(f"expert sidecar build failed: {exc}") from exc
    assert manifest.sidecar is not None
    print(
        json.dumps(
            {
                "manifest": str(manifest_output.resolve()),
                "manifest_sha256": manifest.manifest_sha256,
                "sidecar": str((root / manifest.sidecar.file).resolve()),
                "sidecar_sha256": manifest.sidecar.sha256,
                "sidecar_bytes": manifest.sidecar.size,
                "alignment": manifest.sidecar.alignment,
                "records": len(manifest.records),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
