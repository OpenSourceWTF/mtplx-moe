"""Runtime Belady oracle — the eviction-ceiling analog of the route census.

The route census (issue #98) accumulates decode routes online and derives island
PLACEMENT at close. This does the same shape of thing for the eviction CEILING:
it records the ordered decode access sequence per streamed layer during a real
run, and at close computes the Belady-optimal (clairvoyant) fetch count at the
actual per-layer slot budget. That yields the true achievable floor on EVERY run
over the full decode window, replacing the offline 64-token-trace estimate whose
compulsory-miss inflation was the source of the cold-vs-steady-state discrepancy
(see research/hy3-per-fetch-overhead/REPORT.md C4).

Belady is optimal for equal-size objects (Mattson 1970 / Belady 1966), which the
MoE expert records are — so this is the exact floor, not a heuristic. It is
diagnostic only: it records host-side and never touches the decode data path.
Enabled by env MTPLX_BELADY_ORACLE=1 (zero cost when unset), mirroring how
trace_routes / route_census gate their recording.

The recorded sequence is per (layer, decode step). True Belady needs the whole
future, so the fetch count is computed at close from the complete sequence, not
incrementally — the same reason the census derives placement at close, not per
observation.
"""

from __future__ import annotations

import os
from collections import defaultdict


def belady_fetches(sequence: list[tuple[int, ...]], slots: int) -> int:
    """Clairvoyant fetch count for one layer at `slots` capacity.

    Set semantics: at each decode step the routed experts must all be resident;
    absent ones are fetched. To make room, evict the resident expert whose next
    use is farthest in the future, never one needed at the current step. Equal
    object sizes make this optimal.
    """
    if slots <= 0:
        return sum(len(set(step)) for step in sequence)
    resident: set[int] = set()
    fetches = 0
    n = len(sequence)
    for t, step in enumerate(sequence):
        needed = set(step)
        absent = needed - resident
        fetches += len(absent)
        resident |= absent
        if len(resident) <= slots:
            continue

        def next_use(expert: int) -> int:
            for future in range(t + 1, n):
                if expert in sequence[future]:
                    return future
            return n + 1  # never used again -> evict first

        evictable = sorted(
            (e for e in resident if e not in needed),
            key=next_use,
            reverse=True,
        )
        for expert in evictable[: len(resident) - slots]:
            resident.discard(expert)
    return fetches


class BeladyOracle:
    """Records per-streamed-layer decode access sequences; reports the floor.

    ``island_layers`` are excluded — they are fully resident and never fetch, so
    they carry no eviction ceiling (the census is blind to them for the same
    reason; here it is correct rather than a limitation).
    """

    __slots__ = ("_sequences", "_island_layers", "_steps")

    def __init__(self, island_layers: frozenset[int] = frozenset()) -> None:
        self._sequences: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        self._island_layers = frozenset(island_layers)
        self._steps = 0

    @staticmethod
    def enabled() -> bool:
        return os.environ.get("MTPLX_BELADY_ORACLE") == "1"

    def observe(self, layer: int, expert_ids: list[int] | tuple[int, ...]) -> None:
        if layer in self._island_layers:
            return
        # de-dup within a step, preserve order (a step needs each unique expert)
        self._sequences[int(layer)].append(tuple(dict.fromkeys(int(e) for e in expert_ids)))

    def report(self, slots_by_layer: dict[int, int] | int) -> dict:
        """Belady fetches/token, per streamed layer and summed.

        ``slots_by_layer`` is either the uniform per-layer slot budget or a
        per-layer map. Returns a JSON-safe dict for the benchmark record.
        """
        per_layer = {}
        total_fetches = 0
        total_steps = 0
        for layer, seq in sorted(self._sequences.items()):
            slots = (
                slots_by_layer
                if isinstance(slots_by_layer, int)
                else int(slots_by_layer.get(layer, 0))
            )
            fetches = belady_fetches(seq, slots)
            steps = len(seq)
            uniq = len(set().union(*(set(s) for s in seq))) if seq else 0
            per_layer[str(layer)] = {
                "belady_fetches_per_token": round(fetches / steps, 4) if steps else 0.0,
                "unique_experts": uniq,
                "slots": slots,
                "steps": steps,
            }
            total_fetches += fetches
            total_steps = max(total_steps, steps)
        return {
            "schema": "belady-oracle-v1",
            "streamed_layers": len(self._sequences),
            "belady_fetches_per_token": (
                round(total_fetches / total_steps, 4) if total_steps else 0.0
            ),
            "per_layer": per_layer,
        }
