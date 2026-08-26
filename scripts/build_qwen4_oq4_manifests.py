#!/usr/bin/env python3
"""Build MTPLX manifests for the unchanged pinned Qwen3.8 oQ4-MTP files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_manifest import save_expert_manifest
from mtplx.qwen4_ngram import save_ngram_manifest
from mtplx.qwen4_oq4 import build_qwen4_oq4_manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hash the pinned source safetensors and emit direct expert/ngram "
            "streaming manifests without quantizing or repacking weights."
        )
    )
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--expert-output", type=Path)
    parser.add_argument("--ngram-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.model_root.expanduser().resolve()
    expert_output = args.expert_output or root / "expert-manifest.json"
    ngram_output = args.ngram_output or root / "ngram-manifest.json"
    manifests = build_qwen4_oq4_manifests(root)
    expert = save_expert_manifest(manifests.expert, expert_output)
    ngram = save_ngram_manifest(manifests.ngram, ngram_output)
    print(
        json.dumps(
            {
                "source": str(root),
                "expert_manifest": str(expert_output.resolve()),
                "expert_manifest_sha256": expert.manifest_sha256,
                "expert_records": len(expert.records),
                "resident_tensor_bytes": expert.resident_tensor_bytes,
                "routed_expert_bytes": expert.routed_expert_bytes,
                "ngram_manifest": str(ngram_output.resolve()),
                "ngram_manifest_sha256": ngram.digest,
                "ngram_rows": ngram.padded_rows,
                "ngram_bytes": sum(shard.data_bytes for shard in ngram.shards),
                "weight_transform": "none",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
