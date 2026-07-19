"""Offline verify-union coverage gate — trace-only ceiling (no router weights).

Measures cross-token expert-route PERSISTENCE, which is the ceiling of the
verify-union prefetch: how much of the NEXT token's per-layer routed experts is
already covered by the PREVIOUS token's route (K=1) or a union of the previous
W tokens' routes (approximates deeper K per the design's 'union 2-3 consecutive
tokens' hint).

The real lookahead applies router_M to h_{M-1..M-3} to approximate route_M; the
best it can do is reproduce route_M(t). So prev-token actual route_M(t) is a
strict upper bound on what the lookahead can predict for token t, and thus on
verify-union coverage of route_M(t+1). If even this ceiling is < ~50% the
feature dies here.
"""

import numpy as np

d = np.load("/tmp/issue51-lookahead.npz", allow_pickle=True)
layers = d["layers"]
eids = d["expert_ids"].astype(np.int64)  # (N, 8)

L = 79
N = eids.shape[0]
assert N % L == 0, N
T = N // L
print(f"tokens={T} layers={L} topk={eids.shape[1]} experts_seen={eids.max()+1}")

# Confirm token-major ordering: layers reshape to (T, L) must be constant cols.
lay = layers.reshape(T, L)
assert np.all(lay == lay[0][None, :]), "layer ordering is not token-major 1..79"
assert list(lay[0]) == list(range(1, L + 1))

routes = eids.reshape(T, L, 8)  # routes[t, m, :] = top-8 experts, layer m+1


def coverage_union_prev(window):
    """Mean fraction of token t+1's layer-m route covered by union of the prev
    `window` tokens' routes at the SAME layer m. window=1 => pure K=1."""
    per_layer = np.zeros(L)
    counts = 0
    covs = []
    for t in range(window - 1, T - 1):
        tgt = routes[t + 1]  # (L, 8)
        for m in range(L):
            target = set(tgt[m].tolist())
            pred = set()
            for j in range(window):
                pred |= set(routes[t - j, m].tolist())
            cov = len(pred & target) / len(target)
            per_layer[m] += cov
            covs.append(cov)
        counts += 1
    return np.mean(covs), per_layer / counts, counts


def identity_baseline():
    """Random-chance floor: a fixed arbitrary 8 experts vs next token's route."""
    rng = np.random.default_rng(0)
    covs = []
    for t in range(T - 1):
        for m in range(L):
            target = set(routes[t + 1, m].tolist())
            pred = set(rng.choice(192, 8, replace=False).tolist())
            covs.append(len(pred & target) / len(target))
    return np.mean(covs)


print("\n== identity/random baseline (sanity, expect ~4.2%) ==")
print(f"  {identity_baseline()*100:.2f}%")

print("\n== cross-token persistence coverage (verify-union CEILING) ==")
for w in (1, 2, 3, 4):
    mean_cov, per_layer, cnt = coverage_union_prev(w)
    tag = "K=1 (prev token only)" if w == 1 else f"union of prev {w} tokens (~K={w})"
    print(f"  {tag:34s}: {mean_cov*100:5.1f}%  (pairs={cnt})")

# Depth-resolved: coverage of the specific draft-depth token. For K spec, verify
# row r verifies the r-th drafted token = position t+r. Report how prev-token
# route covers each of the next up to 3 positions.
print("\n== coverage of the k-th-ahead token by token t's route (single prev) ==")
for k in (1, 2, 3):
    covs = []
    for t in range(T - k):
        for m in range(L):
            target = set(routes[t + k, m].tolist())
            pred = set(routes[t, m].tolist())
            covs.append(len(pred & target) / len(target))
    print(f"  next+{k}: {np.mean(covs)*100:5.1f}%")

# Per-layer detail for K=1 to see if early streamed layers (the miss-dominant
# ones) behave differently from late layers.
mean_cov, per_layer, cnt = coverage_union_prev(1)
print("\n== per-layer K=1 coverage (layer: cov%) — first/last 10 ==")
order = [(m + 1, per_layer[m]) for m in range(L)]
for m, c in order[:10]:
    print(f"  L{m:2d}: {c*100:5.1f}%")
print("  ...")
for m, c in order[-10:]:
    print(f"  L{m:2d}: {c*100:5.1f}%")
print(f"\n  min layer cov: {min(c for _,c in order)*100:.1f}%"
      f"  max: {max(c for _,c in order)*100:.1f}%")
