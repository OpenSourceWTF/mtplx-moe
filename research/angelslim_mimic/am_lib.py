"""AngelSlim single-file GGUF access + codec analogs for the O8 mimic pilot.

Reuses O7 tooling (gguf_range, st_range, pilot_lib metrics, mtplx t158).
AngelSlim/Hy3-GGUF ships single-file (not split) GGUFs, so one SplitFile with a
loaded header suffices; expert slices are range-fetched one expert at a time.
"""
from __future__ import annotations

import os
import sys

import numpy as np

WORKTREE = "/Users/davidtai/projects/OpenSourceWTF/.worktrees/51-c6-mmap-band"
IQ = os.path.join(WORKTREE, "research", "iq_transcode")
sys.path.insert(0, IQ)
sys.path.insert(0, WORKTREE)  # local mtplx (with expert_shadow) shadows site-packages

import mlx.core as mx  # noqa: E402
mx.set_default_device(mx.cpu)

import gguf_range as G  # noqa: E402
import gguf.quants as gq  # noqa: E402
from gguf.constants import GGMLQuantizationType as GQ  # noqa: E402
import pilot_lib as P  # noqa: E402  (cosine, rel_err, _proj_out_in, bf16_shard_path)
from mtplx.expert_shadow import encode_t158, decode_t158  # noqa: E402

REPO = "AngelSlim/Hy3-GGUF"
ANCHOR_FILE = "Hy3-IQ1_M.gguf"
N_EXPERT = 192

# bpw accounting convention (task-mandated).
#   official IQx : exact ggml block bpw
#   t158         : 13 B/64w trits + bf16 scale = 1.875
#   affine qN gs64: N bits + (fp16 scale + fp16 bias)/64 = N + 0.5
BPW = {
    "IQ1_M": 1.75, "IQ2_XXS": 2.0625, "IQ3_XXS": 3.0625,
    "t158": 1.875,
    "q2_gs64": 2.5, "q3_gs64": 3.5, "q4_gs64": 4.5,
    "q5_gs64": 5.5, "q6_gs64": 6.5, "q8_gs64": 8.5,
}

PROJ_SUFFIX = {"gate": "ffn_gate_exps.weight",
               "up": "ffn_up_exps.weight",
               "down": "ffn_down_exps.weight"}


class AngelSlimIndex:
    def __init__(self, session, path=ANCHOR_FILE, initial=16 << 20):
        self.sf = G.SplitFile(REPO, path, session)
        self.sf.load_header(initial=initial, cap=96 << 20)
        self.header_bytes = self.sf.header_bytes

    def official_type(self, L, proj):
        name = f"blk.{L}.{PROJ_SUFFIX[proj]}"
        ti = self.sf.header.tensors[name]
        return GQ(ti.ggml_type).name

    def fetch_expert(self, L, proj, expert):
        """Range-fetch one expert slice, dequantize official IQx -> (out,in)."""
        name = f"blk.{L}.{PROJ_SUFFIX[proj]}"
        raw, ti, start, per, sha = self.sf.fetch_expert_slice(name, expert, N_EXPERT)
        out_dim, in_dim = P._proj_out_in(proj)
        deq = gq.dequantize(np.frombuffer(raw, dtype=np.uint8).copy(),
                            GQ(ti.ggml_type)).astype(np.float32)
        assert deq.size == out_dim * in_dim, (deq.size, out_dim, in_dim)
        deq = deq.reshape(out_dim, in_dim)
        meta = {
            "gguf_tensor": name, "ggml_type": GQ(ti.ggml_type).name,
            "dims_ne": list(ti.dims), "expert": expert,
            "byte_start": start, "byte_len": per, "sha256": sha,
            "sliced_out_in": [out_dim, in_dim],
        }
        return deq, meta


# ---- MLX-servable analog codecs, encoded FROM bf16 ----

def affine_from_bf16(w: np.ndarray, bits: int, group_size: int = 64) -> np.ndarray:
    wm = mx.array(np.ascontiguousarray(w, dtype=np.float32))
    wq, sc, bi = mx.quantize(wm, group_size=group_size, bits=bits)
    deq = mx.dequantize(wq, sc, bi, group_size=group_size, bits=bits)
    return np.array(deq.astype(mx.float32), dtype=np.float32)


def t158_from_bf16(w: np.ndarray) -> np.ndarray:
    packed, scales = encode_t158(w.astype(np.float32))
    return decode_t158(packed, scales, w.shape[1]).astype(np.float32)


# tier -> (mimic codec, one-tier-up codec)
TIER_CODECS = {
    "IQ1_M":   ("t158",    "q2_gs64"),
    "IQ2_XXS": ("q2_gs64", "q3_gs64"),
    "IQ3_XXS": ("q3_gs64", "q4_gs64"),
}


def apply_codec(name: str, w: np.ndarray) -> np.ndarray:
    if name == "t158":
        return t158_from_bf16(w)
    if name.startswith("q") and name.endswith("_gs64"):
        return affine_from_bf16(w, int(name[1:-5]))
    raise ValueError(name)


def stable_sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def swiglu(gate_w, up_w, down_w, X):
    """X: (n, in). Returns (n, out) fp64. y = down( silu(gate@x) * (up@x) )."""
    g = X @ gate_w.T.astype(np.float64)          # (n, ff)
    u = X @ up_w.T.astype(np.float64)            # (n, ff)
    h = (g * stable_sigmoid(g)) * u              # (n, ff)
    return h @ down_w.T.astype(np.float64)       # (n, out)
