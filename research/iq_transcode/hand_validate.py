"""Independent hand-validation of IQ2_XXS dequant against gguf.quants.

We re-implement the block-unpack algorithm from the llama.cpp/ggml spec in a
plain scalar loop, deriving the per-element sign from the *even-parity rule*
(NOT gguf's ksigns constant), the per-subgroup scale from the top 4 bits, and
indexing the 256x8 codebook. The codebook itself is an irreducible spec
constant; we decode it from the packed grid_hex independently. If our scalar
output matches gguf's vectorized dequant bitwise on a real fetched block, both
the layout understanding and gguf's decoder are confirmed.
"""
from __future__ import annotations

import struct
import numpy as np

import gguf.quants as gq
from gguf.constants import GGMLQuantizationType as GQ


def decode_grid_iq2xxs() -> np.ndarray:
    """Return the (256, 8) IQ2_XXS codebook as float32, decoded from grid_hex.

    Independent of gguf.init_grid: each hex pair -> a byte; each byte packs 4
    2-bit indices (little end first) into grid_map {0x08,0x19,0x2b}.
    """
    grid_map = (0x08, 0x19, 0x2b)
    hex_bytes = bytes.fromhex(gq.IQ2_XXS.grid_hex.decode("ascii"))
    # 256 entries * 8 elems, 2 bits/elem => 2 bits*8 = 16 bits = 2 bytes/entry
    out = np.zeros((256, 8), dtype=np.float32)
    bi = 0
    for e in range(256):
        for byte_in_entry in range(2):  # 2 bytes per entry, 4 elems each
            b = hex_bytes[bi]; bi += 1
            for k in range(4):
                idx2 = (b >> (2 * k)) & 0x3
                out[e, byte_in_entry * 4 + k] = grid_map[idx2]
    return out


def parity_sign(index7: int, j: int) -> int:
    """Even-parity sign rule: elements 0..6 = bit j of index; element 7 = parity.
    Returns +1 or -1. This reproduces ggml's ksigns table without using it."""
    if j < 7:
        bit = (index7 >> j) & 1
    else:
        bit = bin(index7 & 0x7F).count("1") & 1  # parity of low 7 bits
    return -1 if bit else 1


def hand_dequant_iq2xxs_block(block66: bytes, grid: np.ndarray) -> np.ndarray:
    """Scalar reimpl of dequantize_row_iq2_xxs for one 66-byte block."""
    assert len(block66) == 66
    d = np.frombuffer(block66[:2], dtype="<f2").astype(np.float64)[0]
    qs = np.frombuffer(block66[2:], dtype="<u2")  # 32 uint16
    # regroup as 8 subgroups x 2 uint32
    u32 = qs.view("<u4").reshape(8, 2)
    y = np.zeros(256, dtype=np.float64)
    pos = 0
    for ib32 in range(8):
        grid_word = int(u32[ib32, 0])   # 4 grid indices (bytes 0..3)
        sign_word = int(u32[ib32, 1])   # scale(top4) + 4x7bit sign indices
        mult = sign_word >> 28
        db = d * 0.25 * (0.5 + mult)
        gi = [(grid_word >> (8 * l)) & 0xFF for l in range(4)]
        si = [(sign_word >> (7 * l)) & 0x7F for l in range(4)]
        for l in range(4):
            gentry = grid[gi[l]]
            for j in range(8):
                y[pos] = db * gentry[j] * parity_sign(si[l], j)
                pos += 1
    return y


def validate(block66: bytes) -> dict:
    grid = decode_grid_iq2xxs()
    mine = hand_dequant_iq2xxs_block(block66, grid)
    ref = gq.dequantize(np.frombuffer(block66, dtype=np.uint8).copy(), GQ.IQ2_XXS).astype(np.float64)
    max_abs_diff = float(np.max(np.abs(mine - ref)))
    exact = bool(np.array_equal(mine.astype(np.float32), ref.astype(np.float32)))
    return {
        "max_abs_diff_fp64": max_abs_diff,
        "bitwise_equal_fp32": exact,
        "mine_min": float(mine.min()), "mine_max": float(mine.max()),
        "ref_min": float(ref.min()), "ref_max": float(ref.max()),
        "n_nonzero": int(np.count_nonzero(ref)),
    }
