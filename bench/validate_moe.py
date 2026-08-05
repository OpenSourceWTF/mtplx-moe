#!/usr/bin/env python3
"""Validate the fused EschaSwitchGLU (grouped, no dense W) against a per-slot reference. Tiny flocked run."""
import os, sys, fcntl
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.escha_prefill_bench import EschaSwitchGLU
from mtplx.eschamoe import decode_expert_weights_fast, t128

lockf = open("/tmp/mtplx-gpu-exclusive.lock", "a+")
try:
    fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); print("flock acquired")
except BlockingIOError:
    print("flock held — abort"); sys.exit(1)

D = np.load(os.path.join(os.path.dirname(__file__), "..", ".escha_fixtures", "escha_goldens.npz"))
H, I, E = 2048, 512, 4
gu_code = mx.array(np.stack([D[f"real_gate_up_proj_e{e}_code"] for e in range(E)]).astype(np.int16))
gu_rin = mx.array(np.stack([D[f"real_gate_up_proj_e{e}_rin"] for e in range(E)]))
gu_rout = mx.array(np.stack([D[f"real_gate_up_proj_e{e}_rout"] for e in range(E)]))
dn_code = mx.array(np.stack([D[f"real_down_proj_e{e}_code"] for e in range(E)]).astype(np.int16))
dn_rin = mx.array(np.stack([D[f"real_down_proj_e{e}_rin"] for e in range(E)]))
dn_rout = mx.array(np.stack([D[f"real_down_proj_e{e}_rout"] for e in range(E)]))
mod = EschaSwitchGLU(gu_code, gu_rin, gu_rout, dn_code, dn_rin, dn_rout, H, I)

# decode reference weights once
Wgu = [decode_expert_weights_fast(gu_code[e:e+1], 2)[0] for e in range(E)]  # [H,2I]
Wdn = [decode_expert_weights_fast(dn_code[e:e+1], 3)[0] for e in range(E)]  # [I,H]

T, top_k = 6, 2
mx.random.seed(0)
x = (mx.random.normal((T, H)) * 0.1)
idx = mx.array([[0, 1], [2, 3], [1, 0], [3, 2], [0, 3], [2, 1]], dtype=mx.uint32)

out = np.array(mod(x, idx).astype(mx.float32)).astype(np.float32)   # [T, top_k, H]

ref = np.zeros((T, top_k, H), np.float32)
for t in range(T):
    for j in range(top_k):
        e = int(np.array(idx[t, j]))
        xh = t128(x[t:t+1] * gu_rin[e])
        ygu = xh @ Wgu[e]
        ygu = t128(ygu) * gu_rout[e]
        gated = nn.silu(ygu[:, :I]) * ygu[:, I:]
        xhd = t128(gated * dn_rin[e])
        y = xhd @ Wdn[e]
        y = t128(y) * dn_rout[e]
        ref[t, j] = np.array(y).astype(np.float32)[0]

err = np.abs(out - ref); rel = err.max() / (np.abs(ref).max() + 1e-9)
print(f"fused MoE vs per-slot reference: max_abs_err={err.max():.5f} rel={rel:.5f} ref_std={ref.std():.4f}")
print("MoE integration:", "OK ✅" if rel < 5e-3 else "MISMATCH ❌")
