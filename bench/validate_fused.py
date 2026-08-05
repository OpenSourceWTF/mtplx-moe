#!/usr/bin/env python3
"""Validate fused decode+GEMM (no dense W) == xh @ decode(code). Tiny flocked run."""
import os, sys, fcntl
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtplx.eschamoe import fused_decode_matmul, decode_expert_weights_fast

lockf = open("/tmp/mtplx-gpu-exclusive.lock", "a+")
try:
    fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); print("flock acquired")
except BlockingIOError:
    print("flock held — abort"); sys.exit(1)

D = np.load(os.path.join(os.path.dirname(__file__), "..", ".escha_fixtures", "escha_goldens.npz"))
for proj, K, IN, OUT in (("gate_up_proj", 2, 2048, 1024), ("down_proj", 3, 512, 2048)):
    code = mx.array(D[f"real_{proj}_e0_code"].astype(np.int16))          # [IN/16, OUT/16, NW]
    W = decode_expert_weights_fast(code[None], K)[0]                     # [IN, OUT] fp16 reference
    for M in (1, 8, 37):
        xh = (mx.random.normal((M, IN)) * 0.1).astype(mx.float16)
        y_ref = (xh @ W).astype(mx.float32)
        y_fused = fused_decode_matmul(xh, code, K, OUT).astype(mx.float32)
        err = float(mx.max(mx.abs(y_ref - y_fused)))
        rel = err / (float(mx.max(mx.abs(y_ref))) + 1e-9)
        print(f"{proj} K={K} M={M}: max_abs_err={err:.5f} rel={rel:.5f} "
              f"-> {'OK' if rel < 5e-3 else 'CHECK'}")
