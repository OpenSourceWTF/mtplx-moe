"""Shared pieces for the IQ-transcode pilot."""
from __future__ import annotations

import json
import os
import numpy as np

import gguf_range as G
from gguf.constants import GGMLQuantizationType as GQ
import gguf.quants as gq

REPO = "bartowski/Hy3-GGUF"
BF16_DIR = os.path.expanduser("~/.cache/huggingface/hy3-mtp-layer80")
BF16_INDEX = os.path.join(BF16_DIR, "model.safetensors.index.json")

FORMAT_SPLITS = {
    "IQ1_M": [f"Hy3-IQ1_M/Hy3-IQ1_M-{i:05d}-of-00002.gguf" for i in range(1, 3)],
    "IQ2_XXS": [f"Hy3-IQ2_XXS/Hy3-IQ2_XXS-{i:05d}-of-00003.gguf" for i in range(1, 4)],
    "IQ3_XXS": [f"Hy3-IQ3_XXS/Hy3-IQ3_XXS-{i:05d}-of-00004.gguf" for i in range(1, 5)],
}

N_EXPERT = 192
# GGUF ggml order dims per proj: expert-slice = (ne1, ne0)
# gate/up_exps ne=(4096,1536,192) -> slice (1536,4096)=(out,in)
# down_exps    ne=(1536,4096,192) -> slice (4096,1536)=(out,in)
PROJ_TENSOR = {
    "gate": "ffn_gate_exps.weight",
    "up": "ffn_up_exps.weight",
    "down": "ffn_down_exps.weight",
}


def _weight_map():
    with open(BF16_INDEX) as f:
        return json.load(f)["weight_map"]


_WM = None


def bf16_shard_path(layer: int, expert: int, proj: str) -> tuple[str, str]:
    global _WM
    if _WM is None:
        _WM = _weight_map()
    name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}_proj.weight"
    shard = _WM[name]
    return os.path.join(BF16_DIR, shard), name


class FormatIndex:
    """Parse all split headers for one format; map tensor -> owning split."""

    def __init__(self, fmt: str, session: G.requests.Session):
        self.fmt = fmt
        self.session = session
        self.splits: list[G.SplitFile] = []
        self.owner: dict[str, G.SplitFile] = {}
        self.header_bytes = 0
        for i, path in enumerate(FORMAT_SPLITS[fmt]):
            sf = G.SplitFile(REPO, path, session)
            # first split has the big tokenizer KV; later ones are tiny
            sf.load_header(initial=(9 << 20) if i == 0 else (1 << 19))
            self.header_bytes += sf.header_bytes
            self.splits.append(sf)
            for n in sf.header.tensors:
                self.owner[n] = sf

    def ttype(self, name: str) -> str:
        sf = self.owner[name]
        return GQ(sf.header.tensors[name].ggml_type).name

    def exps_type_map(self) -> dict:
        """{layer: {proj: type_name}} for exps tensors."""
        out: dict[int, dict[str, str]] = {}
        for name, sf in self.owner.items():
            if "exps" not in name:
                continue
            parts = name.split(".")
            layer = int(parts[1])
            for proj, suf in PROJ_TENSOR.items():
                if name.endswith(suf):
                    out.setdefault(layer, {})[proj] = self.ttype(name)
        return out

    def fetch_expert(self, layer: int, proj: str, expert: int):
        name = f"blk.{layer}.{PROJ_TENSOR[proj]}"
        sf = self.owner[name]
        raw, ti, start, per, sha = sf.fetch_expert_slice(name, expert, N_EXPERT)
        # dequantize -> (out, in)
        out_dim, in_dim = _proj_out_in(proj)
        deq = gq.dequantize(np.frombuffer(raw, dtype=np.uint8).copy(),
                            GQ(ti.ggml_type)).astype(np.float32)
        assert deq.size == out_dim * in_dim, (deq.size, out_dim, in_dim)
        deq = deq.reshape(out_dim, in_dim)
        meta = {
            "gguf_tensor": name,
            "split_file": sf.path,
            "ggml_type": GQ(ti.ggml_type).name,
            "dims_ne": list(ti.dims),
            "expert": expert,
            "byte_start": start,
            "byte_len": per,
            "sha256": sha,
        }
        return deq, meta


def _proj_out_in(proj: str) -> tuple[int, int]:
    if proj in ("gate", "up"):
        return 1536, 4096
    return 4096, 1536  # down


# ---- metrics (fp64) ----

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def rel_err(ref: np.ndarray, x: np.ndarray) -> float:
    ref = ref.astype(np.float64).ravel()
    x = x.astype(np.float64).ravel()
    nr = np.linalg.norm(ref)
    if nr == 0:
        return float("nan")
    return float(np.linalg.norm(x - ref) / nr)
