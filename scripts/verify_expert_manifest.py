#!/usr/bin/env python3
"""Verify an expert manifest against its checkpoint and optional sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_manifest import (  # noqa: E402
    ExpertManifestError,
    load_expert_manifest,
    verify_expert_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed expert artifact verification."
    )
    parser.add_argument("model_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--records", action="store_true")
    parser.add_argument("--shards", action="store_true")
    parser.add_argument("--sidecar", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = load_expert_manifest(args.manifest)
        report = verify_expert_manifest(
            manifest,
            args.model_root,
            verify_records=args.records,
            verify_shard_hashes=args.shards,
            verify_sidecar_hash=args.sidecar,
        )
    except ExpertManifestError as exc:
        raise SystemExit(f"expert manifest verification failed: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
