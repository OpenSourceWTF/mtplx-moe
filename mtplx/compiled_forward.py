"""Compiled AR decode forward (issue #51, 70 tps full-residency goal).

The target forward `self.model(input_ids, cache=cache)` (runtime.py forward_ar) is
NOT compiled — the decode loop rebuilds the whole 79-layer MLX graph in Python
every step (~550 stock ops = the ~12 ms host overhead that caps full residency at
40.5 tok/s against a ~13 ms memory floor). This compiles that single call so the
graph traces ONCE and replays.

Mechanism (the draft core's proven pattern, generation.py `_build_core` /
`_trace_depth`, scaled from 1 cache to N layers): mx.compile cannot trace a Python
object whose arrays grow each step, so each layer's KV cache is converted to a
fixed-buffer `TensorOffsetKVCache` (stable graph shape via a reserved buffer +
offset), and its 3 state leaves (keys, values, offset) are threaded as explicit
compile inputs and outputs. N layers -> 3N threaded leaves.

Correctness note: mx.compile is EXACT for fp32 matmul; the quantized gather_qmm
can select a different fused kernel (~0.1-0.6% divergence, non-associative FP,
tagged per David 2026-07-18). Whether that flips tokens end-to-end is the A/B
gate; the compiled-island A/B held token parity, so it is expected benign.

Flag-gated (MTPLX_HY3_COMPILE_FORWARD), full-residency only (no per-layer host
syncs to break the compiled region — islands pass indices device-side). Emits an
engagement counter so a null is never credited as control-vs-control.
"""

from __future__ import annotations

import atexit
import os
from typing import Any, Callable

import mlx.core as mx

from mtplx.compile_state import compile_trace
from mtplx.graphbank import TensorOffsetKVCache

# Engagement proof: incremented every time the compiled forward actually runs, so
# an A/B can assert the compiled path fired rather than inferring it from a null.
_COMPILED_FORWARD_CALLS = 0


def compiled_forward_calls() -> int:
    return _COMPILED_FORWARD_CALLS


def _write_engagement_count_file() -> None:
    """Persist the final call count to a file so an out-of-process A/B driver can
    verify engagement (arm-off must read 0, arm-on > 0). The in-memory counter and
    the runtime's diagnostic_counters never cross the subprocess boundary; the
    benchmark does not serialize either into its JSON, so a file is the only
    channel the driver can read. Registered at import; fires on normal exit."""
    path = os.environ.get("MTPLX_HY3_COMPILE_FORWARD_COUNT_FILE")
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(str(_COMPILED_FORWARD_CALLS))
    except OSError:
        pass


atexit.register(_write_engagement_count_file)


def compile_forward_enabled() -> bool:
    return os.environ.get("MTPLX_HY3_COMPILE_FORWARD") == "1"


class CompiledARForward:
    """Compiles `model(input_ids, cache)` with per-layer cache state threaded.

    Stateful across steps: the fixed-buffer `TensorOffsetKVCache` entries are
    built once from the live cache and advanced in place, exactly like the live
    KV cache would be. The compiled function reads and returns the 3N leaves so
    mx.compile sees a pure array->array graph.
    """

    def __init__(self, model: Any, *, reserve_tokens: int) -> None:
        self._model = model
        self._reserve = int(reserve_tokens)
        self._compiled: Callable[..., tuple] | None = None
        self._n: int = 0
        # The PERSISTENT decode state (3N leaves), owned by the wrapper — NOT
        # by the scratch caches. The compiled fn's scratch caches are overwritten
        # by these leaves every call, so state ownership stays unambiguous (the
        # tangle that flipped tokens when the caches held state and were also
        # mutated in-graph). Same discipline as the draft core's `_trace_depth`.
        self._state: list[mx.array] | None = None

    def _ensure_compiled(self, live_cache: list[Any]) -> None:
        if self._compiled is not None:
            return
        self._n = len(live_cache)
        # Scratch fixed-buffer caches, seeded from the live (primed) cache.
        caches = [
            TensorOffsetKVCache.from_kv_cache(entry, reserve_tokens=self._reserve)
            for entry in live_cache
        ]
        # Seed the persistent state from the primed caches' current contents.
        self._state = []
        for cache in caches:
            self._state.extend([cache.cache[0], cache.cache[1], cache.cache[2]])
        model = self._model
        n = self._n

        def forward(input_ids: mx.array, *state: mx.array) -> tuple:
            if len(state) != 3 * n:
                raise ValueError(f"expected {3 * n} state leaves, got {len(state)}")
            for i in range(n):
                caches[i].cache[0] = state[3 * i]
                caches[i].cache[1] = state[3 * i + 1]
                caches[i].cache[2] = state[3 * i + 2]
            logits = model(input_ids, cache=caches)
            out: list[mx.array] = []
            for i in range(n):
                out.append(caches[i].cache[0])
                out.append(caches[i].cache[1])
                out.append(caches[i].cache[2])
            return (logits, *out)

        self._compiled = mx.compile(forward)

    def __call__(self, input_ids: mx.array, cache: list[Any]) -> mx.array:
        global _COMPILED_FORWARD_CALLS
        self._ensure_compiled(cache)
        try:
            # Mark the trace so the model forward suppresses its per-layer
            # async_eval submit cadence (illegal inside a graph transformation,
            # and obsolete once the whole forward is one traced submission).
            with compile_trace():
                result = self._compiled(input_ids, *self._state)  # type: ignore[misc]
        except Exception:
            # The trace fires on the first call; a host-sync buried in the model
            # forward (async_eval/eval) only surfaces here. Dump the full stack
            # so the offending line is visible in the driver log, then re-raise.
            if os.environ.get("MTPLX_HY3_COMPILE_FORWARD_DEBUG") == "1":
                import sys
                import traceback

                traceback.print_exc(file=sys.stderr)
            raise
        self._state = list(result[1:])
        _COMPILED_FORWARD_CALLS += 1
        return result[0]
