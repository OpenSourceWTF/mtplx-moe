#!/usr/bin/env python3
"""Compare streamed model parameter keys with a pinned Hub Q4 index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from huggingface_hub import hf_hub_download  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from mtplx.expert_streaming_models import MODEL_SPECS, get_model_spec  # noqa: E402
from mtplx.resident_loader import (  # noqa: E402
    _quantize_resident_model,
    get_streaming_model_classes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download only config/index metadata, construct the parameter-free "
            "streamed model, and require an exact resident key match."
        )
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec = get_model_spec(args.model)
    config_path = hf_hub_download(
        spec.quant_model,
        "config.json",
        revision=spec.quant_revision,
    )
    index_path = hf_hub_download(
        spec.quant_model,
        "model.safetensors.index.json",
        revision=spec.quant_revision,
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))["weight_map"]
    model_class, args_class = get_streaming_model_classes(config)
    model = model_class(args_class.from_dict(config))
    resident = {key: None for key in index if ".switch_mlp." not in key}
    if hasattr(model, "sanitize"):
        resident = model.sanitize(resident)
    _quantize_resident_model(model, config, resident)
    parameter_keys = {key for key, _value in tree_flatten(model.parameters())}
    resident_keys = set(resident)
    missing = sorted(parameter_keys - resident_keys)
    extra = sorted(resident_keys - parameter_keys)
    routed = sorted(set(index) - set(resident))
    expected_routed_tensors = spec.routed_layer_count * 9
    report = {
        "model_key": spec.key,
        "quant_model": spec.quant_model,
        "quant_revision": spec.quant_revision,
        "parameter_keys": len(parameter_keys),
        "resident_index_keys": len(resident_keys),
        "routed_index_keys": len(routed),
        "expected_routed_index_keys": expected_routed_tensors,
        "missing_parameter_keys": missing,
        "extra_resident_keys": extra,
        "valid": not missing and not extra and len(routed) == expected_routed_tensors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
