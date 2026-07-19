"""Speculative expert-prefetch wiring: config knob, runtime API, lookahead."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mtplx.expert_runtime import ExpertStreamingConfig
from mtplx.expert_streaming_models import get_model_spec
from mtplx.models.hy3_mlx import Model as Hy3Model
from mtplx.models.hy3_mlx import ModelArgs as Hy3Args
from mtplx.models.hy3_mlx import SparseMLP


def _config_kwargs(**overrides):
    kwargs = {
        "model_key": "hy3-expert-q2",
        "memory_limit_bytes": 96 * 1024**3,
        "max_live_kv_tokens": 4096,
    }
    kwargs.update(overrides)
    return kwargs


def test_prefetch_slots_default_zero_and_validation() -> None:
    config = ExpertStreamingConfig(**_config_kwargs())
    assert config.prefetch_slots == 0

    config = ExpertStreamingConfig(**_config_kwargs(prefetch_slots=8))
    assert config.prefetch_slots == 8

    with pytest.raises(ValueError, match="prefetch_slots"):
        ExpertStreamingConfig(**_config_kwargs(prefetch_slots=-1))
    with pytest.raises(TypeError, match="prefetch_slots"):
        ExpertStreamingConfig(**_config_kwargs(prefetch_slots=True))
    with pytest.raises(TypeError, match="prefetch_slots"):
        ExpertStreamingConfig(**_config_kwargs(prefetch_slots="8"))


def test_prefetch_slots_require_layer_cache_scope() -> None:
    with pytest.raises(ValueError, match="cache_scope 'layer'"):
        ExpertStreamingConfig(
            **_config_kwargs(prefetch_slots=8, cache_scope="global")
        )
    # Zero slots never constrain the scope.
    config = ExpertStreamingConfig(
        **_config_kwargs(prefetch_slots=0, cache_scope="global")
    )
    assert config.prefetch_slots == 0


def test_speculative_io_fraction_validation() -> None:
    config = ExpertStreamingConfig(**_config_kwargs())
    assert config.speculative_io_fraction == 0.25

    for fraction in (0.1, 0.5, 1.0):
        config = ExpertStreamingConfig(
            **_config_kwargs(speculative_io_fraction=fraction)
        )
        assert config.speculative_io_fraction == fraction

    for bad in (0.0, -0.25, 1.01, 2):
        with pytest.raises(ValueError, match="speculative_io_fraction"):
            ExpertStreamingConfig(**_config_kwargs(speculative_io_fraction=bad))
    for wrong_type in (True, "0.5", None):
        with pytest.raises(TypeError, match="speculative_io_fraction"):
            ExpertStreamingConfig(
                **_config_kwargs(speculative_io_fraction=wrong_type)
            )


def test_memory_plan_carries_prefetch_slots_per_layer() -> None:
    spec = get_model_spec("hy3-expert-q2")
    config = ExpertStreamingConfig(**_config_kwargs(prefetch_slots=8))
    plan = config.memory_plan(spec)
    assert plan.prefetch_slots_per_layer == 8
    assert plan.prefetch_bytes == (
        spec.routed_layer_count * 8 * spec.expert_record_bytes
    )

    baseline = ExpertStreamingConfig(**_config_kwargs()).memory_plan(spec)
    assert baseline.prefetch_slots_per_layer == 0
    assert baseline.prefetch_bytes == 0


def _model_args(*, layers: int = 6, first_dense: int = 2) -> Hy3Args:
    return Hy3Args(
        model_type="hy_v3",
        hidden_size=64,
        num_hidden_layers=layers,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        first_k_dense_replace=first_dense,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=16,
        router_scaling_factor=2.0,
    )


def test_lookahead_router_refs_stay_out_of_parameter_tree() -> None:
    args = _model_args()
    model = Hy3Model(args)
    layers = model.model.layers
    sparse_indices = [
        index
        for index, layer in enumerate(layers)
        if isinstance(layer.mlp, SparseMLP)
    ]
    assert sparse_indices == [2, 3, 4, 5]

    # Each sparse layer sees the next up-to-3 sparse layers' routers, by
    # identity, and the tail sees fewer.
    for position, index in enumerate(sparse_indices):
        entries = layers[index].mlp._mtplx_next_routers.entries
        expected = sparse_indices[position + 1 : position + 4]
        assert [next_index for next_index, _router in entries] == expected
        for next_index, router in entries:
            assert router is layers[next_index].mlp.router

    # The references live outside the module tree: never registered as a
    # child (Module is a dict of children) and absent from the parameter
    # tree, so sibling routers are not double-registered.
    for index in sparse_indices:
        mlp = layers[index].mlp
        assert "_mtplx_next_routers" not in mlp
        assert "_mtplx_next_routers" in vars(mlp)
    names = [name for name, _value in tree_flatten(model.parameters())]
    assert not any("_mtplx_next_routers" in name for name in names)
    router_weights = [name for name in names if name.endswith("router.gate.weight")]
    assert len(router_weights) == len(sparse_indices)


class _RuntimeStub:
    def __init__(
        self,
        *,
        prefetch_slots: int,
        islands=frozenset(),
        saturate_after: int | None = None,
    ) -> None:
        self.config = SimpleNamespace(prefetch_slots=prefetch_slots)
        self.island_layer_set = frozenset(islands)
        self.calls: list[tuple[int, list[int]]] = []
        self.speculation_saturated = False
        self._saturate_after = saturate_after

    def prefetch_experts(self, layer: int, expert_ids) -> int:
        ids = [int(expert) for expert in expert_ids]
        self.calls.append((layer, ids))
        if (
            self._saturate_after is not None
            and len(self.calls) >= self._saturate_after
        ):
            self.speculation_saturated = True
        return len(ids)


class _StubSwitch:
    """Callable switch seam carrying the runtime, like the streamed switch."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        return mx.zeros((*indices.shape, x.shape[-1]), dtype=x.dtype)


def test_sparse_mlp_decode_call_prefetches_next_streamed_layers() -> None:
    args = _model_args()
    model = Hy3Model(args)
    mlp = model.model.layers[2].mlp
    runtime = _RuntimeStub(prefetch_slots=4, islands={4})
    mlp.switch_mlp = _StubSwitch(runtime)

    x = mx.random.normal((1, 1, args.hidden_size))
    mx.eval(mlp(x))

    # Island layer 4 is skipped; streamed lookahead layers 3 and 5 get one
    # prediction each with in-range router ids.
    assert [layer for layer, _ids in runtime.calls] == [3, 5]
    for _layer, ids in runtime.calls:
        assert len(ids) == args.num_experts_per_tok
        assert all(0 <= expert < args.num_experts for expert in ids)

    # Prefill-shaped calls never predict.
    runtime.calls.clear()
    mx.eval(mlp(mx.random.normal((1, 3, args.hidden_size))))
    assert runtime.calls == []

    # Disabled prefetch keeps decode free of prediction work.
    runtime.calls.clear()
    runtime.config = SimpleNamespace(prefetch_slots=0)
    mx.eval(mlp(x))
    assert runtime.calls == []

    # The tail sparse layer has no lookahead targets and stays silent.
    tail = model.model.layers[5].mlp
    tail_runtime = _RuntimeStub(prefetch_slots=4)
    tail.switch_mlp = _StubSwitch(tail_runtime)
    mx.eval(tail(x))
    assert tail_runtime.calls == []


def test_lookahead_drops_far_layers_first_under_admission_pressure() -> None:
    """Overlap decays with lookahead depth (74.3% at L=1, 61% at L=3):
    predictions issue nearest-first, and once the speculative lane
    saturates, the farther (lower-value) layers are dropped."""

    args = _model_args()
    model = Hy3Model(args)
    mlp = model.model.layers[2].mlp

    # Saturated at entry: no prediction work at all.
    runtime = _RuntimeStub(prefetch_slots=4)
    runtime.speculation_saturated = True
    mlp.switch_mlp = _StubSwitch(runtime)
    x = mx.random.normal((1, 1, args.hidden_size))
    mx.eval(mlp(x))
    assert runtime.calls == []

    # Saturation after the first issue: only the nearest lookahead layer
    # (highest overlap) gets its prediction in; L>=2 are dropped.
    runtime = _RuntimeStub(prefetch_slots=4, saturate_after=1)
    mlp.switch_mlp = _StubSwitch(runtime)
    mx.eval(mlp(x))
    assert [layer for layer, _ids in runtime.calls] == [3]

    # Unsaturated: the full nearest-first order is preserved.
    runtime = _RuntimeStub(prefetch_slots=4)
    mlp.switch_mlp = _StubSwitch(runtime)
    mx.eval(mlp(x))
    assert [layer for layer, _ids in runtime.calls] == [3, 4, 5]


def test_lookahead_hook_is_inert_without_a_bound_runtime() -> None:
    args = _model_args()
    model = Hy3Model(args)
    mlp = model.model.layers[2].mlp
    # The unbound switch seam has no runtime attribute; the hook must be a
    # silent no-op (construction-time forward paths raise later, in the
    # switch itself, exactly as before).
    x = mx.random.normal((1, 1, args.hidden_size))
    indices, _scores = mlp.router(x)
    assert mlp._maybe_prefetch_lookahead(x, indices) is None
