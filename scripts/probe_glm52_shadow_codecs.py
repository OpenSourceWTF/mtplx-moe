#!/usr/bin/env python
"""Probe shadow-codec quality on real GLM-5.2 Q2 experts (issue #51 q1 lane).

Reads sampled expert records from the glm52-expert-only-mlx-q2 artifact,
dequantizes them (the exact affine math the runtime executes), encodes the
b1 / t158 shadow codecs, and measures how close the shadow expert MLP
output is to the exact dequantized output on SYNTHETIC hiddens (standard
normal, unit RMS — the pre-MLP operating point after RMSNorm; no captured
GLM decode hiddens exist yet, so treat absolute numbers as a lower-bound
sanity gate, not a decode-loop measurement).

Reported per codec:
- per-projection weight cosine / rel-L2 (gate, up, down)
- single-expert MLP output cosine / rel-L2 over the hidden draws
- top-k combine cosine: softmax-weighted sum over ``--top-k`` sampled
  experts per layer, the metric that gated the hy3 shadow lane
  (hy3 probe on captured decode hiddens: b1 0.90-0.91, t158 0.946-0.952).

This is the go/no-go evidence for the full GLM q1 conversion burn.
Bounded: reads only the sampled records (a few hundred MiB of a 221 GiB
artifact), one expert in memory at a time. CPU-dominant; MLX use is
per-record dequantization only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_ROOT = Path.home() / ".cache/huggingface/glm52-expert-only-mlx-q2"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0.0:
        return float("nan")
    return float((a * b).sum() / denominator)


def _rel_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = reference.reshape(-1).astype(np.float64)
    candidate = candidate.reshape(-1).astype(np.float64)
    norm = np.linalg.norm(reference)
    if norm == 0.0:
        return float("nan")
    return float(np.linalg.norm(reference - candidate) / norm)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _expert_mlp(weights: dict[str, np.ndarray], hiddens: np.ndarray) -> np.ndarray:
    """Exact SwiGLU expert MLP in fp64-free fp32: (draws, H) -> (draws, H)."""

    gate = hiddens @ weights["gate_proj"].T
    up = hiddens @ weights["up_proj"].T
    return (_silu(gate) * up) @ weights["down_proj"].T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--layers",
        type=str,
        default="",
        help="Comma-separated routed layers to sample (default: first, middle, last).",
    )
    parser.add_argument("--experts-per-layer", type=int, default=8)
    parser.add_argument("--hidden-draws", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--codecs", type=str, default="b1,t158")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import mlx.core as mx

    from mtplx.expert_manifest import load_expert_manifest, read_expert_record
    from mtplx.expert_shadow import ShadowBankStore, decode_shadow, encode_shadow
    from mtplx.expert_streaming_models import get_model_spec

    manifest_path = args.manifest or args.root / "expert-manifest.json"
    manifest = load_expert_manifest(manifest_path)
    spec = get_model_spec(manifest.model_key)
    routed = spec.routed_layer_indices
    if args.layers:
        layers = tuple(int(value) for value in args.layers.split(","))
    else:
        layers = (routed[0], routed[len(routed) // 2], routed[-1])
    codecs = tuple(codec.strip() for codec in args.codecs.split(",") if codec.strip())
    experts_per_layer = max(args.experts_per_layer, args.top_k)
    rng = np.random.default_rng(args.seed)
    hiddens = rng.standard_normal(
        (args.hidden_draws, spec.hidden_size)
    ).astype(np.float32)
    # Probe reuses the runtime's record decode; the store itself is unused.
    store = ShadowBankStore(spec, layers, codec=codecs[0])

    results: dict[str, dict] = {codec: {"experts": []} for codec in codecs}
    combine_weights = rng.dirichlet(np.ones(args.top_k), size=1)[0].astype(np.float32)
    started = time.perf_counter()
    exact_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    shadow_by_layer: dict[str, dict[int, list[np.ndarray]]] = {
        codec: {layer: [] for layer in layers} for codec in codecs
    }
    for layer in layers:
        expert_ids = rng.choice(
            spec.expert_count, size=experts_per_layer, replace=False
        )
        for expert in sorted(int(e) for e in expert_ids):
            record = manifest.record(layer, expert)
            blob = read_expert_record(
                manifest, args.root, layer, expert, verify_hash=False
            )
            dense = store._dequantize_record(
                mx,
                record,
                blob,
                bits=spec.quant_bits,
                group_size=spec.quant_group_size,
            )
            exact = _expert_mlp(dense, hiddens)
            exact_by_layer[layer].append(exact)
            for codec in codecs:
                shadow_weights = {}
                weight_stats = {}
                for projection, weights in dense.items():
                    packed, scales = encode_shadow(codec, weights)
                    decoded = decode_shadow(codec, packed, scales, weights.shape[1])
                    shadow_weights[projection] = decoded
                    weight_stats[projection] = {
                        "cosine": _cosine(weights, decoded),
                        "rel_l2": _rel_l2(weights, decoded),
                        # Q2's affine grid includes an exact-zero level; a
                        # 1-bit sign code cannot represent it, so the zero
                        # mass bounds b1 quality from above.
                        "fraction_zero": float((weights == 0.0).mean()),
                    }
                shadow = _expert_mlp(shadow_weights, hiddens)
                shadow_by_layer[codec][layer].append(shadow)
                results[codec]["experts"].append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "weights": weight_stats,
                        "output_cosine": _cosine(exact, shadow),
                        "output_rel_l2": _rel_l2(exact, shadow),
                    }
                )
            print(
                f"layer {layer} expert {expert}: "
                + " ".join(
                    f"{codec}={results[codec]['experts'][-1]['output_cosine']:.4f}"
                    for codec in codecs
                ),
                flush=True,
            )

    for codec in codecs:
        expert_cosines = [e["output_cosine"] for e in results[codec]["experts"]]
        combine_cosines = []
        for layer in layers:
            exact_stack = np.stack(exact_by_layer[layer][: args.top_k])
            shadow_stack = np.stack(shadow_by_layer[codec][layer][: args.top_k])
            exact_combined = np.tensordot(combine_weights, exact_stack, axes=1)
            shadow_combined = np.tensordot(combine_weights, shadow_stack, axes=1)
            combine_cosines.append(_cosine(exact_combined, shadow_combined))
        results[codec]["summary"] = {
            "expert_output_cosine_mean": float(np.mean(expert_cosines)),
            "expert_output_cosine_min": float(np.min(expert_cosines)),
            "expert_output_rel_l2_mean": float(
                np.mean([e["output_rel_l2"] for e in results[codec]["experts"]])
            ),
            "combine_cosine_by_layer": {
                str(layer): value
                for layer, value in zip(layers, combine_cosines, strict=True)
            },
            "combine_cosine_mean": float(np.mean(combine_cosines)),
            "combine_cosine_min": float(np.min(combine_cosines)),
        }

    payload = {
        "artifact": str(args.root),
        "model_key": manifest.model_key,
        "layers": list(layers),
        "experts_per_layer": experts_per_layer,
        "hidden_draws": args.hidden_draws,
        "top_k": args.top_k,
        "seed": args.seed,
        "hiddens": "synthetic standard normal (no captured GLM decode hiddens)",
        "elapsed_seconds": time.perf_counter() - started,
        "results": {
            codec: results[codec]["summary"] for codec in codecs
        },
        "experts": {codec: results[codec]["experts"] for codec in codecs},
    }
    print(json.dumps({c: results[c]["summary"] for c in codecs}, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
