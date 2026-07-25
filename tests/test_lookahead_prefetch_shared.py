"""The shared lookahead hook, and GLM's wiring onto it (#99 "needs GLM wiring").

The hook must stay inert unless a bound runtime enables prefetch, must fire only
from streamed sources to streamed targets (firing at islands adds a per-island
host sync: measured d0 12.5 -> 9.2 at 90 GiB), must spend exactly ONE host sync
covering the layer's own indices plus every prediction, and must drop the
farther, lower-overlap targets first when the speculative lane saturates.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mtplx.models import lookahead_prefetch
from mtplx.models.lookahead_prefetch import (
    LookaheadRouters,
    install_lookahead_routers,
    maybe_prefetch_lookahead,
    sparse_layer_entries,
)


class _Router:
    """Router contract: ``router(x) -> (indices, scores)``."""

    def __init__(self, experts: tuple[int, ...]) -> None:
        self.experts = experts
        self.calls = 0

    def __call__(self, x: mx.array):
        self.calls += 1
        return mx.array([list(self.experts)], dtype=mx.int32), None


class _Module:
    def __init__(self, layer_index: int, runtime=None) -> None:
        self.switch_mlp = SimpleNamespace(layer_index=layer_index, runtime=runtime)


def _runtime(
    *,
    prefetch_slots: int = 8,
    islands: frozenset = frozenset(),
    saturate_after: int | None = None,
):
    issued: list[tuple[int, list[int]]] = []
    state = {"calls": 0}

    class _Runtime:
        config = SimpleNamespace(prefetch_slots=prefetch_slots)
        island_layer_set = islands

        @property
        def speculation_saturated(self) -> bool:
            return (
                saturate_after is not None and state["calls"] >= saturate_after
            )

        def prefetch_experts(self, layer, expert_ids):
            state["calls"] += 1
            issued.append((layer, list(expert_ids)))
            return len(expert_ids)

    return _Runtime(), issued


def _wire(module, targets: list[tuple[int, _Router]]) -> None:
    module._mtplx_next_routers = LookaheadRouters(tuple(targets))


def _x() -> mx.array:
    return mx.zeros((1, 1, 4), dtype=mx.float32)


def test_predicts_the_next_streamed_layers_and_queues_their_experts() -> None:
    runtime, issued = _runtime()
    module = _Module(1, runtime)
    _wire(module, [(2, _Router((5, 6))), (3, _Router((7,)))])

    maybe_prefetch_lookahead(module, _x(), mx.array([[0]], dtype=mx.int32))

    assert issued == [(2, [5, 6]), (3, [7])]


def test_caps_at_three_targets_even_with_a_wider_window() -> None:
    runtime, issued = _runtime()
    module = _Module(1, runtime)
    _wire(module, [(index, _Router((index,))) for index in range(2, 9)])

    maybe_prefetch_lookahead(module, _x(), mx.array([[0]], dtype=mx.int32))

    assert [layer for layer, _experts in issued] == [2, 3, 4]


def test_island_sources_and_island_targets_are_skipped() -> None:
    # Source layer 1 is an island: firing there would add a brand-new host sync.
    runtime, issued = _runtime(islands=frozenset({1}))
    module = _Module(1, runtime)
    _wire(module, [(2, _Router((5,)))])
    maybe_prefetch_lookahead(module, _x(), mx.array([[0]], dtype=mx.int32))
    assert issued == []

    # Streamed source, but layer 2 is an island target: skip it, take 3.
    runtime, issued = _runtime(islands=frozenset({2}))
    module = _Module(1, runtime)
    _wire(module, [(2, _Router((5,))), (3, _Router((6,)))])
    maybe_prefetch_lookahead(module, _x(), mx.array([[0]], dtype=mx.int32))
    assert issued == [(3, [6])]


def test_saturation_stops_further_issues_and_skips_router_compute() -> None:
    # Saturated before the first issue: never spend the router matmuls.
    routers = [_Router((5,)), _Router((6,))]
    runtime, issued = _runtime(saturate_after=0)
    module = _Module(1, runtime)
    _wire(module, list(enumerate(routers, start=2)))
    maybe_prefetch_lookahead(module, _x(), mx.array([[0]], dtype=mx.int32))
    assert issued == []
    assert [router.calls for router in routers] == [0, 0]

    # Saturating mid-set drops the farther, lower-overlap target.
    runtime, issued = _runtime(saturate_after=1)
    module = _Module(1, runtime)
    _wire(module, [(2, _Router((5,))), (3, _Router((6,)))])
    maybe_prefetch_lookahead(module, _x(), mx.array([[0]], dtype=mx.int32))
    assert issued == [(2, [5])]


def test_inert_without_runtime_prefetch_or_during_prefill() -> None:
    # No bound runtime.
    module = _Module(1, None)
    _wire(module, [(2, _Router((5,)))])
    assert maybe_prefetch_lookahead(module, _x(), mx.array([[0]])) is None

    # Prefetch disabled.
    runtime, issued = _runtime(prefetch_slots=0)
    module = _Module(1, runtime)
    _wire(module, [(2, _Router((5,)))])
    maybe_prefetch_lookahead(module, _x(), mx.array([[0]], dtype=mx.int32))
    assert issued == []

    # Multi-token (prefill / wide verify) is not the decode single-token path.
    runtime, issued = _runtime()
    module = _Module(1, runtime)
    _wire(module, [(2, _Router((5,)))])
    maybe_prefetch_lookahead(
        module,
        mx.zeros((1, 4, 4), dtype=mx.float32),
        mx.array([[0]], dtype=mx.int32),
    )
    assert issued == []


def test_predictions_ride_one_host_sync_with_the_layer_indices(monkeypatch) -> None:
    """No NEW sync: one eval covers the indices and every prediction."""

    evals: list[int] = []
    monkeypatch.setattr(
        lookahead_prefetch.mx, "eval", lambda *values: evals.append(len(values))
    )
    runtime, issued = _runtime()
    module = _Module(1, runtime)
    _wire(module, [(2, _Router((5,))), (3, _Router((6,)))])

    maybe_prefetch_lookahead(module, _x(), mx.array([[0]], dtype=mx.int32))

    assert evals == [3]  # one call: the indices plus two predictions


def test_installer_windows_forward_only_and_stays_out_of_the_parameter_tree() -> None:
    class _Sparse(nn.Module):
        def __init__(self, layer_index: int) -> None:
            super().__init__()
            self.gate = nn.Linear(4, 4)
            self.switch_mlp = SimpleNamespace(layer_index=layer_index, runtime=None)

    modules = [_Sparse(index) for index in range(4)]
    entries = [(index, mlp, mlp.gate) for index, mlp in enumerate(modules)]
    assert install_lookahead_routers(entries, window=8) == 4

    # Forward-only windows; the tail layer has no targets.
    assert [
        layer for layer, _router in modules[0]._mtplx_next_routers.entries
    ] == [1, 2, 3]
    assert modules[3]._mtplx_next_routers.entries == ()

    # The sibling routers must not re-register as children of this module.
    assert "_mtplx_next_routers" in vars(modules[0])
    assert "_mtplx_next_routers" not in modules[0]
    names = [name for name, _value in modules[0].named_modules()]
    assert not any("_mtplx_next_routers" in name for name in names)


def test_sparse_layer_entries_selects_only_sparse_layers_in_order() -> None:
    class _Sparse:
        def __init__(self) -> None:
            self.gate = object()

    class _Dense:
        pass

    layers = [
        SimpleNamespace(mlp=_Dense()),
        SimpleNamespace(mlp=_Sparse()),
        SimpleNamespace(mlp=_Dense()),
        SimpleNamespace(mlp=_Sparse()),
    ]
    entries = sparse_layer_entries(layers, module_type=_Sparse, router_attr="gate")
    assert [index for index, _mlp, _router in entries] == [1, 3]


def test_glm_streamed_moe_is_wired_and_calls_the_hook() -> None:
    """GLM's trunk installs the references and its MoE calls the hook."""

    import inspect

    from mtplx.models import glm52_mlx

    source = inspect.getsource(glm52_mlx.StreamedMoE.__call__)
    assert "maybe_prefetch_lookahead(self, x, indices)" in source
    # The hook must run on the routed indices, before the expert work.
    assert source.index("self.gate(x)") < source.index("maybe_prefetch_lookahead")

    trunk_source = inspect.getsource(glm52_mlx)
    assert "install_lookahead_routers(" in trunk_source
    assert 'router_attr="gate"' in trunk_source


def test_glm_gate_matches_the_router_contract() -> None:
    """FP32MoEGate returns (indices, scores) — index 0 is what the hook uses."""

    import inspect

    from mtplx.models.glm52_mlx import FP32MoEGate

    signature = inspect.signature(FP32MoEGate.__call__)
    assert str(signature.return_annotation).replace(" ", "").endswith(
        "tuple[mx.array,mx.array]"
    )
