#!/usr/bin/env python3
"""Build a revision-pinned safetensors expert manifest."""

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
    build_expert_manifest,
    save_expert_manifest,
)
from mtplx.expert_streaming_models import MODEL_SPECS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect safetensors headers, validate all affine-Q4 expert slices, "
            "and emit a deterministic manifest."
        )
    )
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-repo")
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--hash-records",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hash every reconstructed expert record (default: enabled).",
    )
    parser.add_argument(
        "--hash-shards",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also hash whole shard files; expensive and redundant with record hashes.",
    )
    parser.add_argument(
        "--allow-unpinned-size",
        action="store_true",
        help="Development fixtures only: do not require the pinned total tensor bytes.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.model_root.expanduser().resolve()
    output = args.output or (root / "expert-manifest.json")
    try:
        manifest = build_expert_manifest(
            root,
            args.model,
            source_repo=args.source_repo,
            source_revision=args.source_revision,
            hash_records=args.hash_records,
            hash_shards=args.hash_shards,
            require_pinned_tensor_bytes=not args.allow_unpinned_size,
        )
        manifest = save_expert_manifest(manifest, output)
    except ExpertManifestError as exc:
        raise SystemExit(f"expert manifest build failed: {exc}") from exc
    print(
        json.dumps(
            {
                "manifest": str(output.resolve()),
                "manifest_sha256": manifest.manifest_sha256,
                "model_key": manifest.model_key,
                "shards": len(manifest.shards),
                "records": len(manifest.records),
                "artifact_tensor_bytes": manifest.artifact_tensor_bytes,
                "resident_tensor_bytes": manifest.resident_tensor_bytes,
                "routed_expert_bytes": manifest.routed_expert_bytes,
                "record_hashes": sum(
                    record.sha256 is not None for record in manifest.records
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
