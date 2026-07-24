"""O8 step 1: extract the OFFICIAL per-tensor quant assignment from the
AngelSlim/Hy3-GGUF recipe.

The anchor file is `Hy3-IQ1_M.gguf` (89,446,312,384 B = 89.446 GB, single-file
GGUF, NOT split). We also parse `Hy3-IQ1_M-mtp.gguf` header purely to recover
the extra MTP (next-token-predict) block's assignment. Header-only range
fetches; no tensor data downloaded here.

Emits official-recipe-map.json: every tensor (name, ggml type, shape, bytes,
bpw) + per-layer routed-expert summary + role rollups + anchor check.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

WORKTREE = "/Users/davidtai/projects/OpenSourceWTF/.worktrees/51-c6-mmap-band"
IQ = os.path.join(WORKTREE, "research", "iq_transcode")
sys.path.insert(0, IQ)
sys.path.insert(0, os.path.dirname(__file__))

import gguf_range as G  # noqa: E402
from gguf.constants import GGMLQuantizationType as GQ  # noqa: E402
from gguf import GGML_QUANT_SIZES  # noqa: E402

REPO = "AngelSlim/Hy3-GGUF"
ANCHOR_FILE = "Hy3-IQ1_M.gguf"
ANCHOR_BYTES = 89446312384
MTP_FILE = "Hy3-IQ1_M-mtp.gguf"
MTP_BYTES = 91756066272
OUT = os.path.join(os.path.dirname(__file__), "official-recipe-map.json")


def tensor_nbytes(ti) -> int:
    nelem = 1
    for d in ti.dims:
        nelem *= d
    be, bb = GGML_QUANT_SIZES[GQ(ti.ggml_type)]
    assert nelem % be == 0, (ti.name, nelem, be)
    return (nelem // be) * bb


def nelem(ti) -> int:
    n = 1
    for d in ti.dims:
        n *= d
    return n


def role_of(name: str):
    """Return (role, layer_or_None, proj_or_None)."""
    if name == "token_embd.weight":
        return "embed", None, None
    if name == "output.weight":
        return "head", None, None
    if name == "output_norm.weight":
        return "norm_final", None, None
    if not name.startswith("blk."):
        return "other", None, None
    parts = name.split(".")
    L = int(parts[1])
    rest = ".".join(parts[2:])
    # attention
    if rest.startswith("attn_"):
        if rest.endswith("_norm.weight"):
            return "attn_norm", L, rest
        return "attention", L, rest.replace(".weight", "")
    if rest in ("attn_norm.weight", "ffn_norm.weight"):
        return "norm", L, rest
    if rest == "ffn_gate_inp.weight":
        return "router", L, "gate_inp"
    if rest in ("exp_probs_b", "exp_probs_b.weight"):
        return "router_bias", L, "exp_probs_b"
    # routed experts (fused 3D)
    for p in ("gate", "up", "down"):
        if rest == f"ffn_{p}_exps.weight":
            return "routed_expert", L, p
        if rest == f"ffn_{p}_shexp.weight":
            return "shared_expert", L, p
        if rest == f"ffn_{p}.weight":
            return "dense_mlp", L, p
    return "other", L, rest


def parse_recipe(session, path, initial):
    sf = G.SplitFile(REPO, path, session)
    h = sf.load_header(initial=initial, cap=96 << 20)
    tensors = {}
    for name, ti in h.tensors.items():
        role, L, proj = role_of(name)
        nb = tensor_nbytes(ti)
        ne = nelem(ti)
        tensors[name] = {
            "ggml_type": GQ(ti.ggml_type).name,
            "shape_ne": list(ti.dims),
            "nelem": ne,
            "nbytes": nb,
            "bpw": round(nb * 8 / ne, 5),
            "role": role,
            "layer": L,
            "proj": proj,
        }
    return sf, h, tensors


def main():
    t0 = time.time()
    session = G.new_session()
    dl = 0

    sf, h, tensors = parse_recipe(session, ANCHOR_FILE, 16 << 20)
    dl += sf.header_bytes

    # ---- MTP: parse the -mtp variant, diff for the extra block ----
    sf_m, h_m, tensors_m = parse_recipe(session, MTP_FILE, 16 << 20)
    dl += sf_m.header_bytes
    extra_names = [n for n in tensors_m if n not in tensors]
    mtp_block = {n: tensors_m[n] for n in extra_names}

    # ---- rollups ----
    type_hist = Counter(t["ggml_type"] for t in tensors.values())
    role_type = defaultdict(Counter)
    role_bytes = defaultdict(int)
    role_elems = defaultdict(int)
    for t in tensors.values():
        role_type[t["role"]][t["ggml_type"]] += 1
        role_bytes[t["role"]] += t["nbytes"]
        role_elems[t["role"]] += t["nelem"]

    # per-layer routed expert assignment
    per_layer = {}
    for name, t in tensors.items():
        if t["role"] == "routed_expert":
            L = t["layer"]
            per_layer.setdefault(L, {})[t["proj"]] = t["ggml_type"]
    # layer-level gate/up tier + down tier
    layer_summary = {}
    gateup_tier_hist = Counter()
    down_tier_hist = Counter()
    for L in sorted(per_layer):
        d = per_layer[L]
        gu = d.get("gate"), d.get("up")
        gateup_tier_hist[gu[0]] += 1  # gate tier
        down_tier_hist[d.get("down")] += 1
        layer_summary[str(L)] = {
            "gate": d.get("gate"), "up": d.get("up"), "down": d.get("down"),
            "gate_up_same": gu[0] == gu[1],
        }

    # trunk (attention / shared / embed / head) assignment table
    trunk = {}
    for name, t in tensors.items():
        if t["role"] in ("attention", "shared_expert", "embed", "head"):
            key = t["role"] if t["role"] in ("embed", "head") else f"{t['role']}:{t['proj']}"
            trunk.setdefault(key, Counter())[t["ggml_type"]] += 1
    trunk = {k: dict(v) for k, v in trunk.items()}

    # ---- anchor check ----
    data_bytes = sum(t["nbytes"] for t in tensors.values())
    total_with_header = data_bytes + h.data_offset
    # gguf pads each tensor's data to alignment; account for inter-tensor pad
    # by computing from declared file size vs summed payload
    anchor_delta = ANCHOR_BYTES - total_with_header

    recipe = {
        "task": "O8 AngelSlim-recipe extraction",
        "date": "2026-07-21",
        "repo": REPO,
        "anchor_file": ANCHOR_FILE,
        "anchor_file_bytes": ANCHOR_BYTES,
        "mtp_file": MTP_FILE,
        "mtp_file_bytes": MTP_BYTES,
        "model_config": {
            "num_hidden_layers": 80, "first_k_dense_replace": 1,
            "num_experts": 192, "num_experts_per_tok": 8,
            "num_shared_experts": 1, "hidden_size": 4096,
            "moe_intermediate_size": 1536, "intermediate_size": 13312,
            "num_attention_heads": 64, "num_key_value_heads": 8,
            "head_dim": 128, "vocab_size": 120832,
            "num_nextn_predict_layers": 1,
        },
        "counts": {
            "tensor_count_anchor": len(tensors),
            "tensor_count_mtp": len(tensors_m),
            "type_histogram": dict(type_hist),
        },
        "role_rollup": {
            r: {
                "types": dict(role_type[r]),
                "nbytes": role_bytes[r],
                "gib": round(role_bytes[r] / (1 << 30), 3),
                "nelem": role_elems[r],
            } for r in sorted(role_bytes)
        },
        "routed_expert_tiers": {
            "gate_tier_per_layer_hist": dict(gateup_tier_hist),
            "down_tier_per_layer_hist": dict(down_tier_hist),
            "note": "gate and up share a tier per layer; down independent",
        },
        "trunk_assignment": trunk,
        "per_layer_routed": layer_summary,
        "mtp_block_tensors": mtp_block,
        "anchor_check": {
            "summed_payload_bytes": data_bytes,
            "header_data_offset": h.data_offset,
            "summed_plus_header": total_with_header,
            "declared_file_bytes": ANCHOR_BYTES,
            "delta_bytes": anchor_delta,
            "delta_note": ("positive delta = per-tensor alignment padding not "
                           "in payload sum; should be a small fraction of total"),
            "summed_gib": round(data_bytes / (1 << 30), 3),
            "summed_gb": round(data_bytes / 1e9, 3),
            "declared_gb": round(ANCHOR_BYTES / 1e9, 3),
            "match_within_1pct": abs(anchor_delta) < 0.01 * ANCHOR_BYTES,
        },
        "download_bytes_headers": dl,
        "all_tensors": tensors,
    }

    with open(OUT, "w") as f:
        json.dump(recipe, f, indent=2)

    print("wrote", OUT)
    print("tensors:", len(tensors), "| types:", dict(type_hist))
    print("gate/up tier per-layer:", dict(gateup_tier_hist))
    print("down tier per-layer:", dict(down_tier_hist))
    print("MTP extra tensors:", len(mtp_block))
    print(f"payload {data_bytes/1e9:.3f} GB + header -> {total_with_header/1e9:.3f} GB "
          f"vs declared {ANCHOR_BYTES/1e9:.3f} GB (delta {anchor_delta/1e9:+.4f} GB, "
          f"{100*anchor_delta/ANCHOR_BYTES:+.3f}%)")
    print("header download:", round(dl / 1e6, 2), "MB, elapsed", round(time.time()-t0,1), "s")
    return recipe


if __name__ == "__main__":
    main()
