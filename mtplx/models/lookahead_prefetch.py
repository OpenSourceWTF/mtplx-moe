"""Residual-stream lookahead prefetch, shared by the streamed MoE trunks.

At a streamed sparse layer N this applies the next few sparse layers' routers to
N's hidden state and hands the predicted expert ids to the runtime's speculative
ring, so their misses stream from SSD while layer N is still executing.  The
mechanism was validated offline on hy3: top-k overlap 74.3/66.1/60.8% at
L=1/2/3 against a 4.2% identity baseline (400 tokens).

Extracted so glm52 can share hy3's implementation instead of growing a second
copy (#99 records GLM as "needs GLM wiring").  The only model coupling is the
router contract: a router must be callable as ``router(x) -> (indices, scores)``.
Both ``hy3_mlx.Router`` and ``glm52_mlx.FP32MoEGate`` satisfy it.

Three properties this file exists to preserve, each measured or paid for:

* **No new host sync.** The predictions are not ancestors of the layer's own
  output, so they would not ride along with the switch's eval.  They are
  materialized together with the layer's own indices in the single sync the
  streamed switch would perform anyway.
* **Streamed sources, streamed targets only.** Island layers have no per-layer
  host sync of their own, so firing there adds a brand-new sync per island per
  token — measured d0 12.5 -> 9.2 tok/s at the 90 GiB config.
* **Nearest first.** Overlap decays with depth, so when the speculative lane
  saturates mid-set, the farther and lower-value layers are the ones dropped.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import mlx.core as mx

# Reaching past intervening island layers needs a window wider than the number
# of targets actually fired; the hook filters islands out and caps at fire time.
DEFAULT_LOOKAHEAD_WINDOW = 8
MAX_LOOKAHEAD_TARGETS = 3


class LookaheadRouters:
    """Plain holder for sibling router references; deliberately NOT a Module.

    ``mlx.nn.Module.__setattr__`` registers array/dict/list/tuple values as
    children, so assigning the sibling routers directly would re-register their
    parameters under a second path in the tree.  A plain object rides outside
    the parameter tree entirely.
    """

    __slots__ = ("entries",)

    def __init__(self, entries: tuple) -> None:
        # tuple of (layer_index, router) for the following sparse layers.
        self.entries = tuple(entries)


def install_lookahead_routers(
    sparse_layers: Sequence[tuple[int, Any, Any]],
    *,
    window: int = DEFAULT_LOOKAHEAD_WINDOW,
) -> int:
    """Give each sparse MoE module references to the following routers.

    ``sparse_layers`` is an ordered sequence of ``(layer_index, module,
    router)`` in trunk order.  Returns the number of modules wired.
    """

    entries = tuple(sparse_layers)
    for position, (_layer_index, module, _router) in enumerate(entries):
        module._mtplx_next_routers = LookaheadRouters(
            tuple(
                (next_index, next_router)
                for next_index, _next_module, next_router in entries[
                    position + 1 : position + 1 + window
                ]
            )
        )
    return len(entries)


def _layer_index_of(module: Any) -> int | None:
    index = getattr(module, "layer_index", None)
    if index is not None:
        return int(index)
    switch = getattr(module, "switch_mlp", None)
    index = getattr(switch, "layer_index", None)
    return None if index is None else int(index)


def maybe_prefetch_lookahead(
    module: Any,
    x: mx.array,
    indices: mx.array,
) -> None:
    """Predict the next streamed layers' routes and queue their loads.

    Inert unless a bound runtime has ``prefetch_slots`` enabled, and inert
    outside single-token decode.  Speculation never raises into the model: a
    failed prediction only costs the prediction.
    """

    lookahead = getattr(module, "_mtplx_next_routers", None)
    if lookahead is None or not lookahead.entries:
        return
    switch = getattr(module, "switch_mlp", None)
    runtime = getattr(switch, "runtime", None)
    if runtime is None:
        return
    config = getattr(runtime, "config", None)
    if config is None or getattr(config, "prefetch_slots", 0) <= 0:
        return
    prefetch = getattr(runtime, "prefetch_experts", None)
    if prefetch is None or int(x.shape[-2]) != 1:
        return
    layer_index = _layer_index_of(module)
    if layer_index is None:
        return
    # The island filter is config-dependent, so cache it per module as plain
    # ints — a tuple of modules would re-register parameters under
    # Module.__setattr__.
    selection = getattr(module, "_mtplx_lookahead_selection", None)
    if selection is None:
        islands = getattr(runtime, "island_layer_set", frozenset())
        if layer_index in islands:
            selection = ()
        else:
            selection = tuple(
                position
                for position, (next_layer, _router) in enumerate(lookahead.entries)
                if next_layer not in islands
            )[:MAX_LOOKAHEAD_TARGETS]
        module._mtplx_lookahead_selection = selection
    if not selection:
        return
    # Admission pressure: skip the router compute entirely rather than predict
    # into a saturated lane.
    if getattr(runtime, "speculation_saturated", False):
        return
    predictions = [
        (lookahead.entries[position][0], lookahead.entries[position][1](x)[0])
        for position in selection
    ]
    # One host sync covering the layer's own indices and every prediction.
    mx.eval(indices, *(predicted for _layer, predicted in predictions))
    for next_layer, predicted in predictions:
        prefetch(next_layer, [int(value) for value in predicted.reshape(-1).tolist()])
        if getattr(runtime, "speculation_saturated", False):
            break


def sparse_layer_entries(
    layers: Iterable[Any],
    *,
    module_type: type,
    router_attr: str,
) -> tuple[tuple[int, Any, Any], ...]:
    """Collect ``(layer_index, mlp, router)`` for each sparse layer in order."""

    entries = []
    for layer_index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, module_type):
            continue
        router = getattr(mlp, router_attr, None)
        if router is None:
            continue
        entries.append((layer_index, mlp, router))
    return tuple(entries)
