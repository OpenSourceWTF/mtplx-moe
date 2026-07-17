"""Offline verify-union gate — REAL lookahead predictions via actual routers.

For each token t and streamed target layer M, the runtime lookahead predicts
layer M's route from the up-to-3 preceding sparse layers' hiddens:
    pred_M(t) = U_{S in {M-1,M-2,M-3}} top8( router_M( h_S(t) ) )
The verify-union bet: pred_M(t) (loads issued while processing token t) covers
the NEXT token's actual route route_M(t+1) that the verify forward will need.

Reconstructs router_M exactly as mtplx.models.hy3_mlx.Router.__call__ does for
the affine-Q8 gate: quantized_matmul (x kept in activation dtype), sigmoid,
+expert_bias(fp32), top-8. Validates against the trace before reporting.
"""

import glob
import json
import os

import mlx.core as mx
import numpy as np

Q4 = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--pipenetwork--Hy3-4bit/snapshots/*/"))[0]
idx = json.load(open(os.path.join(Q4, "model.safetensors.index.json")))["weight_map"]

L = 79
SUFS = ("router.gate.weight", "router.gate.scales",
        "router.gate.biases", "router.expert_bias")

# Group the tensors we need by shard so each shard is opened once.
by_shard = {}
for layer in range(1, L + 1):
    for suf in SUFS:
        k = f"model.layers.{layer}.mlp.{suf}"
        by_shard.setdefault(idx[k], []).append((layer, suf, k))

gate = {}      # layer -> (weight, scales, biases)  mx.arrays
ebias = {}     # layer -> expert_bias (fp32)
tmp = {}
for shard, items in by_shard.items():
    # mx.load memory-maps the shard lazily; evaluating only the small router
    # tensors keeps the multi-GB expert-weight pages from ever faulting in.
    mapping = mx.load(os.path.join(Q4, shard))
    for layer, suf, k in items:
        arr = mapping[k]
        mx.eval(arr)
        tmp[(layer, suf)] = arr
    del mapping
for layer in range(1, L + 1):
    gate[layer] = (
        tmp[(layer, "router.gate.weight")],
        tmp[(layer, "router.gate.scales")],
        tmp[(layer, "router.gate.biases")],
    )
    ebias[layer] = tmp[(layer, "router.expert_bias")].astype(mx.float32)

w0, s0, b0 = gate[5]
print(f"gate[5] weight {w0.shape} {w0.dtype}  scales {s0.shape}  ebias {ebias[5].shape}")


def route_from_hidden(layer, x):
    """Return top-8 expert ids exactly like Router.__call__ (affine-Q8 gate)."""
    w, s, b = gate[layer]
    logits = mx.quantized_matmul(
        x, w, scales=s, biases=b, transpose=True, group_size=64, bits=8
    ).astype(mx.float32)
    scores = mx.sigmoid(logits)
    sel = scores + ebias[layer]
    idxs = mx.argpartition(sel, kth=-8, axis=-1)[..., -8:]
    return idxs


d = np.load("/tmp/issue51-lookahead.npz", allow_pickle=True)
eids = d["expert_ids"].astype(np.int64).reshape(400, L, 8)
hid = mx.array(d["hiddens"]).reshape(400, L, 4096)  # fp16 router inputs
T = 400

# ---- validation 1: router_M(h_M) must reproduce the trace route_M exactly ----
same = []
for layer in range(1, L + 1):
    pred = np.asarray(route_from_hidden(layer, hid[:, layer - 1, :]))  # (T,8)
    for t in range(T):
        same.append(len(set(pred[t].tolist()) & set(eids[t, layer - 1].tolist())) / 8)
print(f"\n[validate] router_M(h_M) vs trace route_M (expect ~100%): "
      f"{np.mean(same)*100:.2f}%")

# ---- validation 2: within-token lookahead overlap (expect ~74/66/61%) ----
for depth in (1, 2, 3):
    covs = []
    for layer in range(1 + depth, L + 1):
        src = hid[:, layer - 1 - depth, :]       # h_{M-depth}(t)
        pred = np.asarray(route_from_hidden(layer, src))
        for t in range(T):
            tgt = set(eids[t, layer - 1].tolist())   # SAME token t route_M
            covs.append(len(set(pred[t].tolist()) & tgt) / len(tgt))
    print(f"[validate] router_M(h_M-{depth}) vs SAME-token route_M "
          f"(expect ~{[74,66,61][depth-1]}%): {np.mean(covs)*100:.1f}%")

# Precompute all lookahead predictions pred[depth][layer] = (T,8) once.
pred_cache = {}
for depth in (1, 2, 3):
    for layer in range(1 + depth, L + 1):
        pred_cache[(depth, layer)] = np.asarray(
            route_from_hidden(layer, hid[:, layer - 1 - depth, :]))

# ---- REAL verify-union coverage: union of L=1..3 predictions from token t
#      vs NEXT token t+1's actual route ----
print("\n== REAL verify-union coverage (union of L=1..maxL lookahead preds "
      "from token t) vs route_M(t+1) ==")
for maxL in (1, 2, 3):
    covs = []
    cand_sizes = []
    for layer in range(1 + maxL, L + 1):
        preds = [pred_cache[(dd, layer)] for dd in range(1, maxL + 1)]
        for t in range(T - 1):
            union = set()
            for p in preds:
                union |= set(p[t].tolist())
            tgt = set(eids[t + 1, layer - 1].tolist())
            covs.append(len(union & tgt) / len(tgt))
            cand_sizes.append(len(union))
    print(f"  union L=1..{maxL}: coverage {np.mean(covs)*100:5.1f}%   "
          f"avg candidates/layer {np.mean(cand_sizes):4.1f}")

# For reference: union of L=1..3 preds vs the SAME token t (upper bound of what
# the wider net can do within-token) and vs t+2 / t+3 (deeper draft rows).
print("\n== reference: union L=1..3 preds from token t vs route at t+k ==")
for k in (0, 1, 2, 3):
    covs = []
    for layer in range(4, L + 1):
        preds = [pred_cache[(dd, layer)] for dd in (1, 2, 3)]
        for t in range(T - k):
            union = set()
            for p in preds:
                union |= set(p[t].tolist())
            tgt = set(eids[t + k, layer - 1].tolist())
            covs.append(len(union & tgt) / len(tgt))
    lbl = "SAME token t" if k == 0 else f"t+{k}"
    print(f"  vs {lbl:12s}: {np.mean(covs)*100:5.1f}%")

# ---- MOST GENEROUS real variant: union of L=1..3 lookahead preds from the
#      last `W` committed tokens {t, t-1, ...} vs route_M(t+1). This is every
#      cheap prediction we could possibly have queued at verify launch. It
#      widens the candidate set (and the speculative load cost) proportionally.
print("\n== ceiling: union L=1..3 preds from last W committed tokens vs route_M(t+1) ==")
for W in (1, 2, 3, 4):
    covs = []
    cand_sizes = []
    for layer in range(4, L + 1):
        preds = [pred_cache[(dd, layer)] for dd in (1, 2, 3)]
        for t in range(W - 1, T - 1):
            union = set()
            for j in range(W):
                for p in preds:
                    union |= set(p[t - j].tolist())
            tgt = set(eids[t + 1, layer - 1].tolist())
            covs.append(len(union & tgt) / len(tgt))
            cand_sizes.append(len(union))
    print(f"  W={W}: coverage {np.mean(covs)*100:5.1f}%   "
          f"avg candidates/layer {np.mean(cand_sizes):4.1f}  "
          f"(load cost ~{np.mean(cand_sizes)/8:.1f}x an 8-expert route)")
