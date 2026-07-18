"""Env-gated per-component GPU roofline for hy3 DECODE, QUEUED lane (issue #51).

MTPLX_ROOFLINE_PROFILE=1 turns it on. REAL modules, no synthetic shapes, and no
per-call barrier (the eager barrier adds ~0.2 ms host sync that inverts small-op
verdicts — the queued-vs-eager law). Method:

  - during the real forward, CAPTURE a handful of real (module, concrete input)
    samples per component across the first few layers (cheap; no timing there, so
    the forward is undistorted);
  - at exit, bench each component on the QUEUED lane: submit N ops, ONE eval. The
    L2 cache is defeated the way real decode defeats it — experts gather DISTINCT
    random slots each iter, and dense components CYCLE several layers' weights so
    the working set exceeds L2 and every weight is re-read from DRAM;
  - achieved GB/s = bytes_moved / queued_time, vs a measured DRAM ceiling.
    >=70% ceiling => memory-bound; <40% & GPU busy => compute/occupancy-bound.
"""

from __future__ import annotations

import atexit
import copy
import os
import time

import mlx.core as mx
from mlx.utils import tree_flatten

_SAMPLES: dict[str, list] = {}   # name -> list of rebuild thunks () -> lazy out
_BYTES: dict[str, int] = {}      # name -> bytes moved per call
_RESULTS: dict[str, tuple] = {}  # name -> (secs_per_call | None, bytes | errmsg)
_ORDER: list[str] = []
_MAX = 8                         # layers cycled per dense component (defeat L2)
_N = 60                          # queued iterations
_TOP_K = 8
_EXPERTS = 192


def enabled() -> bool:
    return os.environ.get("MTPLX_ROOFLINE_PROFILE") == "1"


def module_nbytes(module) -> int:
    return sum(int(p.nbytes) for _, p in tree_flatten(module.parameters()))


def _concrete(x):
    """A standalone concrete copy safe to replay after the forward frees x."""
    c = x + 0.0
    mx.eval(c)
    return c


def _bench_now(name: str) -> None:
    """Queued-bench a full sample set immediately, while every resident bank/weight
    is still live (the MoE bank is cleared at teardown, so atexit is too late)."""
    thunks = _SAMPLES.pop(name, [])
    try:
        probe = thunks[0]()
        if probe is None:
            _RESULTS[name] = (None, "ineligible (wave declined shape)")
            return
        _RESULTS[name] = (_queued_seconds(thunks), _BYTES.get(name, 0))
    except Exception as exc:  # pragma: no cover - diagnostic only
        _RESULTS[name] = (None, repr(exc))


def _reg(name: str, thunk, nbytes: int) -> None:
    if name in _RESULTS:
        return
    if name not in _SAMPLES:
        _SAMPLES[name] = []
        _ORDER.append(name)
        _BYTES[name] = int(nbytes)
    _SAMPLES[name].append(thunk)
    if len(_SAMPLES[name]) >= _MAX:
        _bench_now(name)


def capture_dense(name: str, module, x, nbytes: int) -> None:
    """A per-layer matvec module (router / shared / lm_head): cycle layers."""
    if not enabled():
        return
    xc = _concrete(x)
    _reg(name, lambda m=module, xx=xc: m(xx), nbytes)


def snapshot_cache(cache):
    """An INDEPENDENT copy of a KV cache, or None if one cannot be made safely.

    capture_attention replays its thunk ~61 times (1 warm + _N=60). Attention's
    forward calls cache.update_and_fetch, which APPENDS a token per call. Against
    the live cache that injects ~61 phantom tokens into layers 0-7 only (the
    captured ones), which:
      * corrupts the run's generated text from that point on, so tok/s and token
        hashes from any MTPLX_ROOFLINE_PROFILE=1 run are not trustworthy, and
      * makes the measurement non-idempotent (offset crosses KVCache.step=256
        boundaries, triggering whole-buffer reallocation mid-bench).
    Copying arrays via an arithmetic op (arr + 0) rather than a constructor
    guarantees a distinct buffer regardless of MLX's aliasing rules; this is
    asserted by tests/test_roofline_cache_isolation.py rather than assumed.

    Returning None makes the caller SKIP the capture. Losing one component's
    number is strictly better than silently corrupting the run producing it.
    """
    if cache is None:
        return None
    try:
        snap = copy.copy(cache)
        copied = False
        for attr in ("keys", "values"):
            arr = getattr(cache, attr, None)
            if isinstance(arr, mx.array):
                setattr(snap, attr, arr + 0)  # forces an independent buffer
                copied = True
            elif arr is not None:
                # Quantized caches hold tuples of arrays; not handled -> skip
                # rather than risk sharing a buffer with the live cache.
                return None
        if not copied:
            return None
        mx.eval(snap.keys, snap.values)
        return snap
    except Exception:  # pragma: no cover - never corrupt a run over a profile
        return None


def capture_attention(module, normed, mask, cache, nbytes: int) -> None:
    if not enabled():
        return
    # Replay against an isolated copy so the live cache is never advanced.
    snap = snapshot_cache(cache)
    if cache is not None and snap is None:
        return
    nc = _concrete(normed)
    _reg("attention", lambda m=module, n=nc, c=snap: m(n, mask, c), nbytes)


def capture_moe(switch_mlp, x, indices, nbytes: int) -> None:
    """Routed experts: replay each captured layer's real gather. Cache is defeated
    by cycling several DISTINCT layers' banks (each ~1 GB resident), so a layer's
    experts are evicted before its thunk comes round again -> streamed from DRAM."""
    if not enabled():
        return
    xc = _concrete(x)
    idx = mx.array(indices)
    mx.eval(idx)
    _reg("moe_experts", lambda m=switch_mlp, xx=xc, ii=idx: m(xx, ii), nbytes)


def capture_moe_wave(switch_mlp, x, nbytes_32: int) -> None:
    """T0a: bench the 32-assignment M4 wave (the real MTP-depth hot path shape)
    vs the single-token 8-assignment gather. If the wave runs materially above
    the single-token 44%, MoE occupancy starvation is a batch=1-only artifact.
    Weights/kernel/shape are real; only x values are broadcast (they don't affect
    bandwidth), and 32 DISTINCT experts are gathered so the bank streams from DRAM."""
    if not enabled():
        return
    xc = _concrete(x)

    def thunk(m=switch_mlp, xx=xc):
        wave_call = getattr(m, "wave_call", None)
        if wave_call is None:
            return None
        x4 = mx.broadcast_to(xx.reshape(1, 1, -1), (1, 4, xx.shape[-1])) + 0.0
        idx = (mx.arange(32, dtype=mx.uint32) % _EXPERTS).reshape(1, 4, 8)
        scores = mx.ones((1, 4, 8), dtype=mx.bfloat16)
        return wave_call(x4, idx, scores)

    _reg("moe_wave(32)", thunk, nbytes_32)


_DTYPES: dict[str, str] = {}


def note_dtype(key: str, arr) -> None:
    """T0b: record a tensor's dtype once (norm weights / cache keys) to prove the
    bf16 invariant and surface the latent fp32-KV trap."""
    if not enabled() or key in _DTYPES:
        return
    dtype = getattr(arr, "dtype", None)
    if dtype is not None:
        _DTYPES[key] = str(dtype)


def _measure_ceiling() -> float:
    big = mx.random.normal((512, 1024, 1024)).astype(mx.float32)
    mx.eval(big)
    mx.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        mx.eval(big.sum())
    mx.synchronize()
    return big.nbytes / ((time.perf_counter() - start) / 5) / 1e9


def _queued_seconds(thunks) -> float:
    """Per-op seconds on the queued lane: build N ops cycling the samples, ONE
    eval. No per-call barrier -> no host-sync inflation."""
    def _flat(out, acc):
        if isinstance(out, tuple):
            acc.extend(o for o in out if isinstance(o, mx.array))
        else:
            acc.append(out)

    warm = []
    _flat(thunks[0](), warm)
    mx.eval(warm)
    mx.synchronize()
    start = time.perf_counter()
    outs: list = []
    for i in range(_N):
        _flat(thunks[i % len(thunks)](), outs)
    mx.eval(outs)
    mx.synchronize()
    return (time.perf_counter() - start) / _N


@atexit.register
def _dump() -> None:
    if not _RESULTS and not _SAMPLES:
        return
    try:
        ceiling = _measure_ceiling()
    except Exception:  # pragma: no cover
        ceiling = float("nan")
    print("\n" + "=" * 76, flush=True)
    print("PER-COMPONENT DECODE ROOFLINE — QUEUED lane, real modules", flush=True)
    print(f"DRAM read ceiling (measured): {ceiling:.0f} GB/s", flush=True)
    print("=" * 76, flush=True)
    # Bench any component that never reached _MAX (persistent-weight ones only;
    # the MoE bank is gone by now, but it always fills within the first token).
    for name in list(_SAMPLES):
        if name not in _RESULTS:
            _bench_now(name)
    layers = {"attention": 80, "router": 79, "moe_experts": 79,
              "shared_expert": 79, "lm_head": 1}
    print(f"{'component':16s} {'us/call':>8s} {'MB/call':>8s} {'GB/s':>7s} {'%ceil':>6s} "
          f"{'ms/tok':>7s}  bound", flush=True)
    tot_ms = 0.0
    for name in _ORDER:
        secs, payload = _RESULTS.get(name, (None, "not benched"))
        if secs is None:
            print(f"{name:16s} FAILED: {payload}", flush=True)
            continue
        nbytes = int(payload)
        gbs = (nbytes / secs / 1e9) if secs and nbytes else 0.0
        pct = (gbs / ceiling * 100) if ceiling and gbs else 0.0
        ms_tok = secs * 1e3 * layers.get(name, 1)
        tot_ms += ms_tok
        bound = "MEMORY" if pct >= 70 else ("compute/occ" if 0 < pct < 40 else ("mixed" if pct else "-"))
        print(f"{name:16s} {secs*1e6:8.1f} {nbytes/1e6:8.2f} {gbs:7.0f} {pct:5.0f}% "
              f"{ms_tok:7.2f}  {bound}", flush=True)
    print("-" * 76, flush=True)
    print(f"reconstructed decode ~{tot_ms:.1f} ms/token  ({1000/tot_ms if tot_ms else 0:.1f} tok/s)"
          f"  vs measured ~35 tok/s (~28.4 ms)", flush=True)
    print("per-call = one layer; ms/tok = per-call x layers (attention 80, MoE/shared/router 79).", flush=True)
    if "moe_experts" in _RESULTS and "moe_wave(32)" in _RESULTS:
        s8, b8 = _RESULTS["moe_experts"]
        sw, bw = _RESULTS["moe_wave(32)"]
        if s8 and sw:
            g8 = b8 / s8 / 1e9
            gw = int(bw) / sw / 1e9
            print(f"\nT0a: single-token 8-assign {g8:.0f} GB/s vs 32-assign wave {gw:.0f} GB/s "
                  f"({gw/g8:.2f}x) -> occupancy starvation is "
                  f"{'batch=1-ONLY (MoE deprioritizes)' if gw > 1.3*g8 else 'STRUCTURAL (kernel work justified)'}", flush=True)
    if _DTYPES:
        print("\nT0b dtype invariants (bf16 expected; fp32 = latent KV trap):", flush=True)
        for key, dtype in _DTYPES.items():
            flag = "  <-- FP32 TRAP" if "float32" in dtype else ""
            print(f"  {key:24s} {dtype}{flag}", flush=True)
