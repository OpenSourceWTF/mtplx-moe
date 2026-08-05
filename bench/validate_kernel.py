#!/usr/bin/env python3
"""Validate the Metal decompression kernel is BIT-EXACT vs goldens + the vectorized decode.
Tiny GPU run (4 experts/proj); holds the GPU flock per policy."""
import os, sys, fcntl
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtplx.eschamoe import decode_expert_weights, decode_expert_weights_fast

lockf = open("/tmp/mtplx-gpu-exclusive.lock", "a+")
try:
    fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    print("flock acquired")
except BlockingIOError:
    print("flock held by another process — aborting"); sys.exit(1)

D = np.load(os.path.join(os.path.dirname(__file__), "..", ".escha_fixtures", "escha_goldens.npz"))
for proj, K in (("gate_up_proj", 2), ("down_proj", 3)):
    bad = tot = badv = 0
    for e in range(4):
        code = mx.array(D[f"real_{proj}_e{e}_code"].astype(np.int16))[None]   # [1,ni,nj,nw]
        Wf = np.array(decode_expert_weights_fast(code, K))[0].astype(np.float16)
        Wv = np.array(decode_expert_weights(code, K))[0].astype(np.float16)
        g = D[f"real_{proj}_e{e}_W"]
        bad += int((Wf != g).sum()); badv += int((Wf != Wv).sum()); tot += Wf.size
    print(f"{proj} K={K}: kernel-vs-golden {bad}/{tot}  kernel-vs-vectorized {badv}/{tot}  "
          f"-> {'BIT-EXACT ✅' if bad == 0 else 'FAIL'}")
