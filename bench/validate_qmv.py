#!/usr/bin/env python3
"""Validate ported ESCHA_QMV (fast matvec) == xh @ decode(code) per row's expert. Flock only."""
import os, sys, fcntl
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtplx.eschamoe import escha_qmv, decode_expert_weights_fast

lockf = open("/tmp/mtplx-gpu-exclusive.lock", "a+")
try:
    fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); print("flock acquired")
except BlockingIOError:
    print("flock held"); sys.exit(1)

D = np.load(os.path.join(os.path.dirname(__file__), "..", ".escha_fixtures", "escha_goldens.npz"))
rng = np.random.RandomState(0)
for proj, K, IN, OUT in (("gate_up_proj", 2, 2048, 1024), ("down_proj", 3, 512, 2048)):
    E = 4
    code = mx.array(np.stack([D[f"real_{proj}_e{e}_code"] for e in range(E)]).astype(np.int16))
    W = [decode_expert_weights_fast(code[e:e + 1], K)[0] for e in range(E)]   # [IN, OUT] each
    rows = 8
    xh = (mx.array(rng.randn(rows, IN).astype(np.float32)) * 0.1)
    eids = mx.array(rng.randint(0, E, rows).astype(np.uint32))
    y = np.array(escha_qmv(xh, eids, code, K, OUT))
    ref = np.stack([np.array(xh[r:r + 1] @ W[int(np.array(eids[r]))])[0] for r in range(rows)])
    err = np.abs(y - ref); rel = err.max() / (np.abs(ref).max() + 1e-9)
    print(f"{proj} K={K}: escha_qmv vs xh@W  rel={rel:.5f}  -> {'OK ✅' if rel < 5e-3 else 'MISMATCH ❌'}")
