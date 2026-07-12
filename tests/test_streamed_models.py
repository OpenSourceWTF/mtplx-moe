from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.activations import swiglu
from mlx_lm.models.deepseek_v32 import group_expert_select

import mtplx.models.expert_mlx as expert_mlx
from mtplx.expert_manifest import (
    build_expert_manifest,
    load_expert_manifest,
    save_expert_manifest,
)
from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingRuntime,
    RouteWave,
)
from mtplx.expert_slots import ExpertSlotBinding, ExpertSlotError
from mtplx.expert_streaming_models import ExpertStreamingModelSpec
from mtplx.models.expert_mlx import (
    HotExpertSwitchGLU,
    _run_q4_expert,
    make_mlx_component_bank_allocator,
    make_mlx_slot_buffer_allocator,
)
from mtplx.models.glm52_mlx import FP32MoEGate
from mtplx.models.glm52_mlx import Model as GlmModel
from mtplx.models.glm52_mlx import ModelArgs as GlmArgs
from mtplx.models.hy3_mlx import Model as Hy3Model
from mtplx.models.hy3_mlx import ModelArgs as Hy3Args
from mtplx.resident_loader import construct_resident_model


def _hy3_args() -> Hy3Args:
    return Hy3Args(
        model_type="hy_v3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        num_shared_experts=1,
        first_k_dense_replace=1,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=16,
        router_scaling_factor=2.0,
    )


def _glm_args(*, layers: int = 6, first_sparse: int = 1) -> GlmArgs:
    return GlmArgs(
        model_type="glm_moe_dsa",
        vocab_size=128,
        hidden_size=64,
        index_head_dim=8,
        index_n_heads=4,
        index_topk=4,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=24,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=first_sparse,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10_000.0},
        attention_bias=False,
        index_topk_pattern="FSFSFS" if layers == 6 else None,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )


def test_hy3_router_uses_unbiased_scores_for_weights() -> None:
    model = Hy3Model(_hy3_args())
    router = model.model.layers[1].mlp.router
    router.gate.weight = mx.array(
        [
            [1.0] + [0.0] * 63,
            [0.5] + [0.0] * 63,
        ],
        dtype=mx.float32,
    )
    router.expert_bias = mx.array([0.0, 10.0], dtype=mx.float32)
    hidden = mx.array([[[1.0] + [0.0] * 63]], dtype=mx.bfloat16)

    indices, weights = router(hidden)
    mx.eval(indices, weights)

    assert indices.item() == 1
    # Correction bias selects expert 1 but is not part of its returned weight.
    assert weights.item() == pytest.approx(2.0)


def test_hy3_non_fp32_combine_casts_routing_weights_to_activation_dtype() -> None:
    model = Hy3Model(_hy3_args())
    sparse_mlp = model.model.layers[1].mlp
    routed_values = [
        4.336133731530333,
        -3.7478681657261284,
        2.5238181218276177,
        4.55797767056557,
        -0.09106507450222434,
        -4.222918128185795,
        0.24504878855251455,
        3.055661890955095,
    ]
    routing_weights = [
        1.9165060684424067,
        0.1014428979156421,
        1.6294923886421544,
        1.2853930759868095,
        1.2728115717622752,
        0.5407021513938952,
        1.755765528776848,
        1.2906728107893897,
    ]

    class StubRouter:
        def __call__(self, _x):
            return (
                mx.zeros((1, 1, 8), dtype=mx.int32),
                mx.array([[routing_weights]], dtype=mx.float32),
            )

    class StubSwitch:
        def __call__(self, _x, _indices):
            rows = [[value] + [0.0] * 63 for value in routed_values]
            return mx.array([[rows]], dtype=mx.bfloat16)

    class StubShared:
        def __call__(self, x):
            return mx.zeros_like(x)

    sparse_mlp.router = StubRouter()
    sparse_mlp.switch_mlp = StubSwitch()
    sparse_mlp.shared_mlp = StubShared()
    hidden = mx.zeros((1, 1, 64), dtype=mx.bfloat16)

    output = sparse_mlp(hidden)
    mx.eval(output)

    # The pinned reference rounds routing weights to BF16 before the multiply.
    # Leaving them in FP32 produces 19.875 for this fixture instead of 20.0.
    assert output.dtype == mx.bfloat16
    assert output[0, 0, 0].item() == 20.0


def test_hy3_shared_mlp_uses_streamed_overlap_hook() -> None:
    model = Hy3Model(_hy3_args())
    sparse_mlp = model.model.layers[1].mlp
    events: list[str] = []

    class StubRouter:
        def __call__(self, x):
            return (
                mx.zeros((*x.shape[:-1], 1), dtype=mx.int32),
                mx.ones((*x.shape[:-1], 1), dtype=mx.bfloat16),
            )

    class OverlapSwitch:
        def __call__(self, _x, _indices):
            raise AssertionError("overlap-capable switch used its fallback path")

        def run_with_shared_overlap(self, x, indices, shared_work):
            events.append("switch")
            shared = shared_work()
            return mx.full((*indices.shape, x.shape[-1]), 2, dtype=x.dtype), shared

    class FallbackSwitch:
        def __call__(self, x, indices):
            return mx.full((*indices.shape, x.shape[-1]), 2, dtype=x.dtype)

    class StubShared:
        def __call__(self, x):
            events.append("shared")
            return mx.full(x.shape, 3, dtype=x.dtype)

    sparse_mlp.router = StubRouter()
    sparse_mlp.shared_mlp = StubShared()
    sparse_mlp.switch_mlp = FallbackSwitch()
    hidden = mx.zeros((1, 1, 64), dtype=mx.bfloat16)
    baseline = sparse_mlp(hidden)
    mx.eval(baseline)

    events.clear()
    sparse_mlp.switch_mlp = OverlapSwitch()
    output = sparse_mlp(hidden)
    mx.eval(output)

    assert events == ["switch", "shared"]
    assert mx.array_equal(output, baseline).item()
    assert mx.all(output == 5).item()


def test_glm_shared_mlp_uses_streamed_overlap_hook() -> None:
    model = GlmModel(_glm_args())
    streamed_moe = model.model.layers[1].mlp
    events: list[str] = []

    class StubGate:
        def __call__(self, x):
            return (
                mx.zeros((*x.shape[:-1], 2), dtype=mx.int32),
                mx.full((*x.shape[:-1], 2), 0.5, dtype=mx.bfloat16),
            )

    class OverlapSwitch:
        def __call__(self, _x, _indices):
            raise AssertionError("overlap-capable switch used its fallback path")

        def run_with_shared_overlap(self, x, indices, shared_work):
            events.append("switch")
            shared = shared_work()
            return mx.full((*indices.shape, x.shape[-1]), 2, dtype=x.dtype), shared

    class StubShared:
        def __call__(self, x):
            events.append("shared")
            return mx.full(x.shape, 3, dtype=x.dtype)

    streamed_moe.gate = StubGate()
    streamed_moe.switch_mlp = OverlapSwitch()
    streamed_moe.shared_experts = StubShared()
    output = streamed_moe(mx.zeros((1, 1, 64), dtype=mx.bfloat16))
    mx.eval(output)

    assert events == ["switch", "shared"]
    assert mx.all(output == 5).item()


class _OverlapPending:
    def __init__(
        self,
        events: list[str],
        *,
        assignment_count: int,
        misses_pending: bool = True,
    ) -> None:
        self.events = events
        self.plan = SimpleNamespace(hits=(), misses=(0,))
        self.hit_ready = None
        self.misses_pending = misses_pending
        binding = SimpleNamespace(expert=0)
        self._miss_ready = SimpleNamespace(
            bindings=(binding,) * assignment_count,
            plan=SimpleNamespace(experts=(0,)),
            release=lambda **_kwargs: None,
        )

    def release_hits(self) -> None:
        raise AssertionError("all-miss fixture has no hit route")

    def finish_misses(self):
        self.events.append("finish-misses")
        self.misses_pending = False
        return self._miss_ready

    def iter_ready_misses(self):
        ready = self.finish_misses()
        if ready is not None:
            yield ready

    def release_miss(self, ready) -> None:
        ready.release(synchronize=False)

    def close(self) -> None:
        self.events.append("close")


class _OverlapRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.spec = SimpleNamespace(top_k=1, hidden_size=2, quant_group_size=64)
        self.manifest = SimpleNamespace(sidecar=None)
        self.config = SimpleNamespace(slot_layout="direct-slots")

    def observe_route(self, *_args, **_kwargs) -> None:
        return None

    def prepare_prefill_seed(self, *_args, **_kwargs) -> tuple[int, ...]:
        self.events.append("prefill-seed")
        return ()

    def route_waves(self, expert_ids, **_kwargs):
        experts = tuple(expert_ids)
        return (
            RouteWave(
                positions=tuple(range(len(experts))),
                experts=experts,
            ),
        )

    def begin_split_route(self, _layer, experts, **_kwargs):
        self.events.append("begin-misses")
        return _OverlapPending(
            self.events,
            assignment_count=len(tuple(experts)),
        )


def test_streamed_decode_evaluates_shared_work_before_waiting_for_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    runtime = _OverlapRuntime(events)
    switch = HotExpertSwitchGLU(runtime, 1)

    def fake_q4(selected, _binding, *, group_size):
        assert group_size == 64
        events.append("miss-q4")
        return selected

    monkeypatch.setattr("mtplx.models.expert_mlx._run_q4_expert", fake_q4)

    def shared_work():
        events.append("shared")
        return mx.ones((1, 1, 2), dtype=mx.bfloat16)

    routed, shared = switch.run_with_shared_overlap(
        mx.zeros((1, 1, 2), dtype=mx.bfloat16),
        mx.zeros((1, 1, 1), dtype=mx.int32),
        shared_work,
    )
    mx.eval(routed, shared)

    assert events == [
        "begin-misses",
        "shared",
        "finish-misses",
        "miss-q4",
        "close",
    ]


def test_128k_prefill_preserves_bounded_routed_then_shared_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = 128 * 1024
    events: list[str] = []
    runtime = _OverlapRuntime(events)
    switch = HotExpertSwitchGLU(runtime, 1)

    def fake_q4(selected, _binding, *, group_size):
        assert group_size == 64
        events.append("routed-q4")
        return selected

    monkeypatch.setattr("mtplx.models.expert_mlx._run_q4_expert", fake_q4)

    def shared_work():
        events.append("shared")
        return mx.zeros((1, tokens, 2), dtype=mx.bfloat16)

    routed, shared = switch.run_with_shared_overlap(
        mx.zeros((1, tokens, 2), dtype=mx.bfloat16),
        mx.zeros((1, tokens, 1), dtype=mx.int32),
        shared_work,
    )
    mx.eval(routed, shared)

    assert routed.shape == (1, tokens, 1, 2)
    assert events == [
        "prefill-seed",
        "begin-misses",
        "finish-misses",
        "routed-q4",
        "close",
        "shared",
    ]


class _OwnedMissPart:
    def __init__(self) -> None:
        self.plan = SimpleNamespace(experts=(0,))
        self.bindings = (SimpleNamespace(expert=0),)
        self.releases = 0

    def release(self, *, synchronize: bool = True) -> None:
        assert synchronize is False
        self.releases += 1


class _OwnedMissPending:
    def __init__(self, part: _OwnedMissPart) -> None:
        self.plan = SimpleNamespace(hits=(), misses=(0,))
        self.hit_ready = None
        self.misses_pending = False
        self.part = part
        self.owner_releases = 0
        self.closed = False
        self.failure: BaseException | None = None
        self.owns_part = True

    def release_hits(self) -> None:
        raise AssertionError("all-miss fixture has no hit route")

    def iter_ready_misses(self):
        yield self.part

    def abort(self, error: BaseException) -> None:
        self.failure = error

    def release_miss(self, part: _OwnedMissPart) -> None:
        assert self.owns_part
        assert part is self.part
        self.owns_part = False
        self.owner_releases += 1
        part.release(synchronize=False)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.owns_part:
            self.release_miss(self.part)


class _OwnedMissRuntime(_OverlapRuntime):
    def __init__(self, pending: _OwnedMissPending) -> None:
        super().__init__([])
        self.pending = pending

    def begin_split_route(self, _layer, _experts, **_kwargs):
        return self.pending


def test_streamed_miss_parts_have_one_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    part = _OwnedMissPart()
    pending = _OwnedMissPending(part)
    switch = HotExpertSwitchGLU(_OwnedMissRuntime(pending), 1)
    monkeypatch.setattr(
        "mtplx.models.expert_mlx._run_q4_expert",
        lambda selected, _binding, **_kwargs: selected,
    )

    output = switch(
        mx.zeros((1, 1, 2), dtype=mx.bfloat16),
        mx.zeros((1, 1, 1), dtype=mx.int32),
    )
    mx.eval(output)

    assert pending.owner_releases == 1
    assert part.releases == 1


def test_streamed_miss_compute_failure_releases_current_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    part = _OwnedMissPart()
    pending = _OwnedMissPending(part)
    switch = HotExpertSwitchGLU(_OwnedMissRuntime(pending), 1)

    def fail_compute(*_args, **_kwargs):
        raise RuntimeError("injected streamed miss compute failure")

    monkeypatch.setattr("mtplx.models.expert_mlx._run_q4_expert", fail_compute)
    with pytest.raises(RuntimeError, match="streamed miss compute failure"):
        switch(
            mx.zeros((1, 1, 2), dtype=mx.bfloat16),
            mx.zeros((1, 1, 1), dtype=mx.int32),
        )

    assert pending.owner_releases == 1
    assert part.releases == 1


def test_glm_router_fp32_projection_changes_a_near_tie_route() -> None:
    x = mx.array(
        [
            [
                0.0205078125,
                -0.006805419921875,
                -0.05859375,
                0.080078125,
                -0.21484375,
                -0.1875,
                -0.01318359375,
                0.1767578125,
                -0.01556396484375,
                0.10498046875,
                0.212890625,
                0.06640625,
                0.04541015625,
                0.058349609375,
                0.044921875,
                -0.021484375,
            ]
        ],
        dtype=mx.bfloat16,
    )
    weight = mx.array(
        [
            [
                0.0267333984375,
                0.134765625,
                0.22265625,
                -0.043212890625,
                -0.1826171875,
                -0.109375,
                0.06298828125,
                0.1552734375,
                0.01214599609375,
                -0.06591796875,
                -0.09814453125,
                0.1787109375,
                -0.02294921875,
                -0.12353515625,
                -0.050048828125,
                -0.00023937225341796875,
            ],
            [
                0.002044677734375,
                -0.01202392578125,
                -0.1279296875,
                0.04296875,
                -0.166015625,
                0.0634765625,
                -0.173828125,
                -0.072265625,
                -0.171875,
                -0.0294189453125,
                -0.057373046875,
                0.2001953125,
                0.039306640625,
                -0.0859375,
                -0.029541015625,
                -0.052734375,
            ],
            [
                -0.0888671875,
                0.205078125,
                0.12060546875,
                0.031005859375,
                -0.1337890625,
                -0.0220947265625,
                0.006866455078125,
                -0.0166015625,
                0.05810546875,
                0.076171875,
                -0.006439208984375,
                0.103515625,
                -0.031494140625,
                0.134765625,
                0.01409912109375,
                -0.06005859375,
            ],
            [
                0.060791015625,
                -0.022216796875,
                -0.0181884765625,
                -0.049072265625,
                -0.1259765625,
                0.1298828125,
                0.09375,
                0.0126953125,
                -0.126953125,
                -0.111328125,
                -0.1494140625,
                0.0869140625,
                -0.0034027099609375,
                0.015869140625,
                0.11572265625,
                -0.038330078125,
            ],
        ],
        dtype=mx.bfloat16,
    )
    original = SimpleNamespace(
        top_k=1,
        norm_topk_prob=True,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        weight=weight,
        e_score_correction_bias=mx.zeros((4,), dtype=mx.float32),
    )
    fp32_router = FP32MoEGate(original)

    fp32_indices, _ = fp32_router(x)
    legacy_indices, _ = group_expert_select(
        x @ weight.T,
        original.e_score_correction_bias,
        1,
        1,
        1,
        1.0,
        True,
    )
    mx.eval(fp32_indices, legacy_indices)

    assert fp32_indices.item() == 2
    assert legacy_indices.item() == 0


def test_glm_indexshare_schedule_and_asymmetric_caches_execute() -> None:
    args = _glm_args(first_sparse=6)
    model = GlmModel(args)

    assert args.indexer_types == ["full", "shared", "full", "shared", "full", "shared"]
    assert [layer.self_attn.indexer is not None for layer in model.model.layers] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    cache = model.make_cache()
    assert [len(item.caches) for item in cache] == [2, 1, 2, 1, 2, 1]
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    logits = model(prompt, cache=cache)
    mx.eval(logits)
    assert logits.shape == (1, 4, args.vocab_size)
    assert mx.all(mx.isfinite(logits)).item()


def _raw_array(value: mx.array, dtype: str) -> bytes:
    mx.eval(value)
    if dtype == "U32":
        return np.array(value, copy=True).astype("<u4", copy=False).tobytes()
    return (
        np.array(value.view(mx.uint16), copy=True).astype("<u2", copy=False).tobytes()
    )


def test_portable_q4_slot_execution_matches_direct_quantized_matmul() -> None:
    mx.random.seed(7)
    hidden = 64
    intermediate = 64
    gate_source = mx.random.normal((intermediate, hidden)).astype(mx.bfloat16)
    up_source = mx.random.normal((intermediate, hidden)).astype(mx.bfloat16)
    down_source = mx.random.normal((hidden, intermediate)).astype(mx.bfloat16)
    gate = mx.quantize(gate_source, group_size=64, bits=4)
    up = mx.quantize(up_source, group_size=64, bits=4)
    down = mx.quantize(down_source, group_size=64, bits=4)
    arrays = {
        "gate_proj.weight": (gate[0], "U32"),
        "gate_proj.scales": (gate[1], "BF16"),
        "gate_proj.biases": (gate[2], "BF16"),
        "up_proj.weight": (up[0], "U32"),
        "up_proj.scales": (up[1], "BF16"),
        "up_proj.biases": (up[2], "BF16"),
        "down_proj.weight": (down[0], "U32"),
        "down_proj.scales": (down[1], "BF16"),
        "down_proj.biases": (down[2], "BF16"),
    }
    from mtplx.expert_manifest import ExpertRecord, TensorSegment

    payload = bytearray()
    segments = []
    for component, (value, dtype) in arrays.items():
        raw = _raw_array(value, dtype)
        segments.append(
            TensorSegment(
                component=component,
                tensor=component,
                shard="fixture",
                offset=len(payload),
                length=len(raw),
                dtype=dtype,
                shape=tuple(value.shape),
            )
        )
        payload.extend(raw)
    record = ExpertRecord(
        layer=1,
        expert=0,
        logical_bytes=len(payload),
        segments=tuple(segments),
    )
    binding = ExpertSlotBinding(1, 0, 0, 1, record, payload)
    x = mx.random.normal((3, hidden)).astype(mx.bfloat16)

    actual = _run_q4_expert(x, binding, group_size=64)
    reference_gate = mx.quantized_matmul(
        x,
        gate[0],
        scales=gate[1],
        biases=gate[2],
        group_size=64,
        bits=4,
    )
    reference_up = mx.quantized_matmul(
        x,
        up[0],
        scales=up[1],
        biases=up[2],
        group_size=64,
        bits=4,
    )
    reference = mx.quantized_matmul(
        swiglu(reference_gate, reference_up),
        down[0],
        scales=down[1],
        biases=down[2],
        group_size=64,
        bits=4,
    )
    mx.eval(actual, reference)
    assert mx.allclose(actual, reference, atol=1e-5, rtol=1e-5).item()


def _integrated_hy3_artifact(tmp_path: Path):
    args = _hy3_args()
    model = Hy3Model(args)
    weights = dict(tree_flatten(model.parameters()))
    expert_shapes = {
        "gate_proj.weight": (2, 64, 8),
        "gate_proj.scales": (2, 64, 1),
        "gate_proj.biases": (2, 64, 1),
        "up_proj.weight": (2, 64, 8),
        "up_proj.scales": (2, 64, 1),
        "up_proj.biases": (2, 64, 1),
        "down_proj.weight": (2, 64, 8),
        "down_proj.scales": (2, 64, 1),
        "down_proj.biases": (2, 64, 1),
    }
    for component, shape in expert_shapes.items():
        dtype = mx.uint32 if component.endswith("weight") else mx.bfloat16
        value = mx.zeros(shape, dtype=dtype)
        if component.endswith("scales"):
            value = mx.ones(shape, dtype=dtype)
        weights[f"model.layers.1.mlp.switch_mlp.{component}"] = value
    mx.eval(weights)
    root = tmp_path / "hy3"
    root.mkdir()
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    config = asdict(args)
    config["model_type"] = "hy_v3"
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    routed_bytes = 2 * 6_912
    total_bytes = sum(int(value.nbytes) for value in weights.values())
    spec = ExpertStreamingModelSpec(
        key="tiny-hy3-q4",
        display_name="Tiny Hy3 Q4",
        source_model="test/tiny-hy3",
        source_revision="source",
        quant_model="test/tiny-hy3-q4",
        quant_revision="quant",
        total_tensor_bytes=total_bytes,
        total_layers=2,
        routed_layer_start=1,
        routed_layer_count=1,
        expert_count=2,
        top_k=1,
        hidden_size=64,
        expert_hidden_size=64,
        quant_bits=4,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="float32",
        router_matmul_dtype="float32",
        router_bytes=2 * 64 * 4 + 2 * 4,
        kv_bytes_per_token=0,
        mtp_layer_index=2,
        mtp_included=False,
    )
    assert spec.routed_expert_bytes == routed_bytes
    manifest = build_expert_manifest(root, spec)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    return root, config, spec, manifest_path


def test_resident_loader_runs_hy3_without_materializing_routed_parameters(
    tmp_path: Path,
) -> None:
    root, config, spec, manifest_path = _integrated_hy3_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_slot_buffer_allocator(
            stream_config.memory_plan(spec), spec
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    try:
        resident = construct_resident_model(root, runtime, config=config)
        parameter_names = {
            name for name, _ in tree_flatten(resident.model.parameters())
        }
        assert not any("switch_mlp" in name for name in parameter_names)
        assert resident.report.raw_tensor_bytes == spec.resident_bytes
        assert resident.report.bound_sparse_layers == 1

        logits = resident.model(mx.array([[1]], dtype=mx.int32))
        mx.eval(logits)
        assert logits.shape == (1, 1, config["vocab_size"])
        assert mx.all(mx.isfinite(logits)).item()
        snapshot = runtime.snapshot(mx_module=mx)
        assert snapshot["cache"]["expert_requests"] == 1
        assert snapshot["slots"]["pins"] == 0
        assert snapshot["slots"]["buffer_backend"] == "mlx-metal-direct-slots"
        assert snapshot["slots"]["metrics"]["completion_fences"] >= 1
    finally:
        runtime.close()


def test_split_route_keeps_mlx_evaluation_on_generation_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config, spec, manifest_path = _integrated_hy3_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_slot_buffer_allocator(
            stream_config.memory_plan(spec), spec
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    original_async_eval = mx.async_eval
    async_calls = 0

    def tracked_async_eval(*values) -> None:
        nonlocal async_calls
        async_calls += 1
        original_async_eval(*values)

    monkeypatch.setattr(expert_mlx.mx, "async_eval", tracked_async_eval)
    try:
        resident = construct_resident_model(root, runtime, config=config)
        logits = resident.model(mx.array([[1]], dtype=mx.int32))
        mx.eval(logits)

        assert async_calls == 0
        assert runtime.snapshot(mx_module=mx)["slots"]["pins"] == 0
    finally:
        runtime.close()


def test_slot_fence_falls_back_to_synchronous_eval_without_async_mlx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config, spec, manifest_path = _integrated_hy3_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_slot_buffer_allocator(
            stream_config.memory_plan(spec), spec
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    monkeypatch.setattr(mx, "async_eval", None)
    try:
        resident = construct_resident_model(root, runtime, config=config)
        logits = resident.model(mx.array([[1]], dtype=mx.int32))
        mx.eval(logits)

        snapshot = runtime.snapshot(mx_module=mx)
        assert snapshot["slots"]["pins"] == 0
        assert snapshot["slots"]["metrics"]["completion_fences"] == 0
        assert snapshot["slots"]["metrics"]["completion_fence_fallbacks"] >= 1
    finally:
        runtime.close()


def test_slot_fence_synchronous_fallback_failure_blocks_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _config, spec, manifest_path = _integrated_hy3_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        cache_policy="lru",
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_slot_buffer_allocator(
            stream_config.memory_plan(spec), spec
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    switch = HotExpertSwitchGLU(runtime, 1)
    tokens = mx.zeros((1, 1, spec.hidden_size), dtype=mx.bfloat16)
    indices = mx.array([[0]], dtype=mx.int32)
    eval_error = RuntimeError("injected synchronous fallback eval failure")
    replacement_ready = None
    original_eval = expert_mlx.mx.eval
    monkeypatch.setattr(
        expert_mlx, "_run_q4_expert", lambda selected, *_a, **_k: selected
    )
    monkeypatch.setattr(expert_mlx.mx, "async_eval", None)

    try:
        warm = switch(tokens, indices)
        original_eval(warm)
        runtime.snapshot(mx_module=mx)
        slot = next(
            slot
            for slot in (*runtime.slots._persistent.values(), *runtime.slots._transient)
            if slot.expert == 0
        )
        with slot.condition:
            generation = slot.generation
        read_bytes = runtime.reader.metrics.as_dict()["read_bytes"]

        eval_calls = 0

        def fail_fence_eval(*values) -> None:
            nonlocal eval_calls
            eval_calls += 1
            if len(values) == 1 and values[0] is indices:
                original_eval(*values)
                return
            assert len(values) == 1
            assert isinstance(values[0], list)
            assert tuple(values[0][0].shape) == (1, spec.hidden_size)
            raise eval_error

        monkeypatch.setattr(expert_mlx.mx, "eval", fail_fence_eval)
        with pytest.raises(RuntimeError, match="fallback eval failure") as failed:
            switch(tokens, indices)
        monkeypatch.setattr(expert_mlx.mx, "eval", original_eval)
        assert failed.value is eval_error
        assert eval_calls == 2

        with slot.condition:
            assert slot.pins == 0
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0

        with pytest.raises(ExpertSlotError, match="completion fence failed") as blocked:
            replacement_ready = runtime.ensure_route(1, [1], phase="decode")
        assert blocked.value.__cause__ is eval_error
        with slot.condition:
            assert slot.expert == 0
            assert slot.generation == generation
        assert runtime.reader.metrics.as_dict()["read_bytes"] == read_bytes

        with pytest.raises(ExpertSlotError, match="completion fence failed") as visible:
            runtime.snapshot(mx_module=mx)
        assert visible.value.__cause__ is eval_error
        with pytest.raises(ExpertSlotError, match="completion fence failed") as closed:
            runtime.close(timeout=2)
        assert closed.value.__cause__ is eval_error
        assert runtime.reader._closed is True
    finally:
        monkeypatch.setattr(expert_mlx.mx, "eval", original_eval)
        if replacement_ready is not None:
            replacement_ready.release(synchronize=False)
        try:
            runtime.close(timeout=2)
        except ExpertSlotError:
            pass


def test_component_bank_hy3_executes_without_record_or_stack_copies(
    tmp_path: Path,
) -> None:
    root, config, spec, manifest_path = _integrated_hy3_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        slot_layout="component-banks",
    )
    plan = stream_config.memory_plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            plan,
            spec,
            load_expert_manifest(manifest_path),
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    try:
        resident = construct_resident_model(root, runtime, config=config)
        logits = resident.model(mx.array([[1, 2]], dtype=mx.int32))
        mx.eval(logits)
        assert logits.shape == (1, 2, config["vocab_size"])
        assert mx.all(mx.isfinite(logits)).item()
        snapshot = runtime.snapshot(mx_module=mx)
        assert snapshot["slots"]["buffer_backend"] == "mlx-metal-component-banks"
        assert snapshot["slots"]["pins"] == 0
        assert snapshot["slots"]["io"]["integrity_errors"] == 0
    finally:
        runtime.close()


def test_global_component_bank_allocator_reuses_one_persistent_bank_and_accounts_exactly(
    tmp_path: Path,
) -> None:
    root, _config, spec, manifest_path = _integrated_glm_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + 3 * spec.expert_record_bytes,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        cache_scope="global",
        slot_layout="component-banks",
    )
    plan = stream_config.memory_plan(spec)
    allocator = make_mlx_component_bank_allocator(
        plan,
        spec,
        load_expert_manifest(manifest_path),
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=allocator,
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    try:
        persistent_slots = tuple(runtime.slots._persistent.values())
        persistent_banks = {physical.buffer.bank for physical in persistent_slots}
        expected_slot_bytes = plan.persistent_cache_bytes + plan.transient_bytes
        physical_bank_bytes = sum(
            int(array.nbytes)
            for bank in allocator.banks.values()
            for array in bank.arrays.values()
        )

        assert plan.persistent_slots == 3
        assert len(persistent_slots) == plan.persistent_slots
        assert len(persistent_banks) == 1
        assert allocator.banks[("global-persistent", -1)].capacity == 3
        assert runtime.slots.allocated_bytes == expected_slot_bytes
        assert physical_bank_bytes == expected_slot_bytes
    finally:
        runtime.close()


@pytest.mark.parametrize("difference", ["order", "dtype", "shape", "length"])
def test_global_component_bank_allocator_rejects_non_exemplar_geometry(
    tmp_path: Path,
    difference: str,
) -> None:
    _root, _config, spec, manifest_path = _integrated_glm_artifact(tmp_path)
    manifest = load_expert_manifest(manifest_path)
    exemplar_index = next(
        index
        for index, record in enumerate(manifest.records)
        if record.layer == spec.routed_layer_indices[1] and record.expert == 1
    )
    exemplar = manifest.records[exemplar_index]
    segments = list(exemplar.segments)
    if difference == "order":
        segments[0], segments[1] = segments[1], segments[0]
    else:
        replacement = {
            "dtype": {"dtype": "BF16"},
            "shape": {"shape": (32, 16)},
            "length": {"length": segments[0].length + 4},
        }[difference]
        segments[0] = replace(segments[0], **replacement)
    records = list(manifest.records)
    records[exemplar_index] = replace(exemplar, segments=tuple(segments))
    heterogeneous = replace(manifest, records=tuple(records))
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    plan = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + 3 * spec.expert_record_bytes,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        cache_scope="global",
        slot_layout="component-banks",
    ).memory_plan(spec)

    with pytest.raises(ValueError, match="routed-layer component geometry differs"):
        make_mlx_component_bank_allocator(plan, spec, heterogeneous)


def test_global_component_bank_allocator_rejects_uniform_wrong_descriptor_geometry(
    tmp_path: Path,
) -> None:
    _root, _config, spec, manifest_path = _integrated_glm_artifact(tmp_path)
    manifest = load_expert_manifest(manifest_path)
    records = []
    for record in manifest.records:
        segments = list(record.segments)
        segments[0] = replace(segments[0], shape=tuple(reversed(segments[0].shape)))
        records.append(replace(record, segments=tuple(segments)))
    uniformly_wrong = replace(manifest, records=tuple(records))
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    plan = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + 3 * spec.expert_record_bytes,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        cache_scope="global",
        slot_layout="component-banks",
    ).memory_plan(spec)

    with pytest.raises(ValueError, match="model descriptor"):
        make_mlx_component_bank_allocator(plan, spec, uniformly_wrong)


def test_global_component_bank_allocator_requires_exact_routed_expert_keys(
    tmp_path: Path,
) -> None:
    _root, _config, spec, manifest_path = _integrated_glm_artifact(tmp_path)
    manifest = load_expert_manifest(manifest_path)
    records = list(manifest.records)
    record_index = next(
        index
        for index, record in enumerate(records)
        if record.layer == spec.routed_layer_indices[-1]
        and record.expert == spec.expert_count - 1
    )
    records[record_index] = replace(records[record_index], expert=spec.expert_count)
    shifted = replace(manifest, records=tuple(records))
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    plan = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + 3 * spec.expert_record_bytes,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        cache_scope="global",
        slot_layout="component-banks",
    ).memory_plan(spec)

    with pytest.raises(ValueError, match="routed expert keys differ"):
        make_mlx_component_bank_allocator(plan, spec, shifted)


@pytest.mark.parametrize(
    "label_template",
    [
        "global-persistent-junk-0",
        "global-persistent--1",
        "global-transient-junk-0",
        "global-transient--1",
        "layer-{layer}-persistent-junk-0",
        "layer-{layer}-persistent--0",
    ],
)
def test_global_component_bank_allocator_rejects_malformed_label_alias(
    tmp_path: Path,
    label_template: str,
) -> None:
    _root, _config, spec, manifest_path = _integrated_glm_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    plan = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=(fixed + spec.routed_layer_count * spec.expert_record_bytes),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        cache_scope="global",
        slot_layout="component-banks",
    ).memory_plan(spec)
    allocator = make_mlx_component_bank_allocator(
        plan,
        spec,
        load_expert_manifest(manifest_path),
    )
    label = label_template.format(layer=spec.routed_layer_indices[0])
    try:
        with pytest.raises(ValueError, match="unknown expert slot label"):
            allocator(spec.expert_record_bytes, label)
    finally:
        allocator.close()


@pytest.mark.parametrize(
    ("cache_scope", "label_template"),
    [
        ("global", "layer-{layer}-persistent-0"),
        ("layer", "global-persistent-0"),
    ],
)
def test_global_component_bank_allocator_rejects_wrong_cache_scope_persistent_label(
    tmp_path: Path,
    cache_scope: str,
    label_template: str,
) -> None:
    _root, _config, spec, manifest_path = _integrated_glm_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    plan = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=(fixed + spec.routed_layer_count * spec.expert_record_bytes),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        cache_scope=cache_scope,
        slot_layout="component-banks",
    ).memory_plan(spec)
    allocator = make_mlx_component_bank_allocator(
        plan,
        spec,
        load_expert_manifest(manifest_path),
    )
    label = label_template.format(layer=spec.routed_layer_indices[0])
    try:
        assert allocator.banks == {}
        assert allocator.slots == {}
        with pytest.raises(ValueError, match="cache scope"):
            allocator(spec.expert_record_bytes, label)
        assert allocator.banks == {}
        assert allocator.slots == {}
    finally:
        allocator.close()


def test_component_bank_all_hit_decode_keeps_router_order_without_split_route_ops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _config, spec, manifest_path = _integrated_glm_artifact(tmp_path)
    transient_slots = 4
    fixed = spec.resident_bytes + transient_slots * spec.expert_record_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(4),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        slot_layout="component-banks",
        transient_slots=transient_slots,
    )
    plan = stream_config.memory_plan(spec)
    assert plan.slots_per_layer == 4
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            plan,
            spec,
            load_expert_manifest(manifest_path),
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )

    def assignment_marker(
        selected: mx.array,
        bindings: tuple[ExpertSlotBinding, ...],
        *,
        group_size: int,
    ) -> mx.array:
        assert group_size == spec.quant_group_size
        expert_offsets = mx.array(
            [binding.expert * 100 for binding in bindings],
            dtype=selected.dtype,
        ).reshape((-1, 1))
        return selected + expert_offsets

    monkeypatch.setattr(expert_mlx, "_run_component_bank_q4", assignment_marker)
    switch = HotExpertSwitchGLU(runtime, 1)
    tokens = mx.zeros((2, 1, spec.hidden_size), dtype=mx.float32)
    tokens[:, 0, 0] = mx.array([1.0, 2.0])
    indices = mx.array([[2, 0], [2, 1]], dtype=mx.int32)
    try:
        # The cold call exercises the existing split/group/reorder path and
        # fills three persistent slots in one bounded route wave.
        reference = switch(tokens, indices)
        mx.eval(reference)
        assert reference[:, :, 0].tolist() == [[201.0, 1.0], [202.0, 102.0]]

        def unexpected(*_args, **_kwargs):
            raise AssertionError("all-hit decode used the split/group/reorder path")

        monkeypatch.setattr(runtime, "begin_split_route", unexpected)
        monkeypatch.setattr(runtime._split_executor, "submit", unexpected)
        monkeypatch.setattr(expert_mlx.mx, "take", unexpected)
        monkeypatch.setattr(expert_mlx.mx, "concatenate", unexpected)
        monkeypatch.setattr(expert_mlx.mx, "argsort", unexpected)

        actual = switch(tokens, indices)
        mx.eval(actual)

        assert mx.array_equal(actual, reference).item()
        assert actual[:, :, 0].tolist() == [[201.0, 1.0], [202.0, 102.0]]
        snapshot = runtime.snapshot(mx_module=mx)
        assert snapshot["cache"]["route_calls"] == 2
        assert snapshot["cache"]["expert_hits"] == 4
        assert snapshot["slots"]["pins"] == 0
    finally:
        runtime.close()


def test_global_component_bank_all_hit_decode_binds_without_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _config, spec, manifest_path = _integrated_glm_artifact(tmp_path)
    transient_slots = 4
    fixed = spec.resident_bytes + transient_slots * spec.expert_record_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + 4 * spec.expert_record_bytes,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        cache_scope="global",
        slot_layout="component-banks",
        transient_slots=transient_slots,
    )
    plan = stream_config.memory_plan(spec)
    assert plan.persistent_slots == 4
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            plan,
            spec,
            load_expert_manifest(manifest_path),
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    switch = HotExpertSwitchGLU(runtime, 1)
    tokens = mx.zeros((2, 1, spec.hidden_size), dtype=mx.float32)
    indices = mx.array([[2, 0], [2, 1]], dtype=mx.int32)
    observed: list[tuple[tuple[int, int, int], ...]] = []
    original_run = expert_mlx._run_component_bank_q4
    try:
        cold = switch(tokens, indices)
        mx.eval(cold)
        reads_before = runtime.reader.metrics.as_dict()["read_bytes"]

        def observe_component_bindings(
            selected: mx.array,
            bindings: tuple[ExpertSlotBinding, ...],
            *,
            group_size: int,
        ) -> mx.array:
            observed.append(
                tuple(
                    (binding.expert, binding.logical_slot, binding.generation)
                    for binding in bindings
                )
            )
            assert len({id(binding.buffer.bank) for binding in bindings}) == 1
            assert all(
                binding.buffer.bank_index == binding.logical_slot
                for binding in bindings
            )
            return original_run(selected, bindings, group_size=group_size)

        def unexpected_split(*_args, **_kwargs):
            raise AssertionError("global all-hit decode used the split route")

        monkeypatch.setattr(
            expert_mlx, "_run_component_bank_q4", observe_component_bindings
        )
        monkeypatch.setattr(runtime, "begin_split_route", unexpected_split)

        hot = switch(tokens, indices)
        mx.eval(hot)

        assert mx.array_equal(hot, cold).item()
        assert tuple(expert for expert, _, _ in observed[0]) == (2, 0, 2, 1)
        assert runtime.reader.metrics.as_dict()["read_bytes"] == reads_before
        assert runtime.snapshot(mx_module=mx)["slots"]["pins"] == 0
    finally:
        runtime.close()


def test_component_bank_all_hit_decode_preserves_route_waves_counters_and_shared_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _config, spec, manifest_path = _integrated_glm_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(3),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        transient_slots=2,
        slot_layout="component-banks",
        cache_policy="lru",
    )
    plan = stream_config.memory_plan(spec)
    assert plan.slots_per_layer == 3
    assert plan.transient_slots == 2
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            plan,
            spec,
            load_expert_manifest(manifest_path),
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    events: list[str] = []

    def assignment_marker(
        selected: mx.array,
        bindings: tuple[ExpertSlotBinding, ...],
        *,
        group_size: int,
    ) -> mx.array:
        assert group_size == spec.quant_group_size
        experts = tuple(binding.expert for binding in bindings)
        events.append(f"q4:{experts}")
        expert_offsets = mx.array(
            [expert * 100 for expert in experts],
            dtype=selected.dtype,
        ).reshape((-1, 1))
        return selected + expert_offsets

    monkeypatch.setattr(expert_mlx, "_run_component_bank_q4", assignment_marker)
    switch = HotExpertSwitchGLU(runtime, 1)
    tokens = mx.zeros((2, 1, spec.hidden_size), dtype=mx.float32)
    tokens[:, 0, 0] = mx.array([1.0, 2.0])
    indices = mx.array([[2, 0], [2, 1]], dtype=mx.int32)
    try:
        reference = switch(tokens, indices)
        mx.eval(reference)
        assert runtime._banks[1].resident_experts == (2, 0, 1)
        before = runtime.snapshot(mx_module=mx)["cache"]
        events.clear()
        fast_waves: list[tuple[int, ...]] = []
        original_try = runtime.try_all_hit_route

        def record_fast_wave(layer, expert_ids, **kwargs):
            experts = tuple(expert_ids)
            fast_waves.append(experts)
            return original_try(layer, experts, **kwargs)

        def unexpected_split(*_args, **_kwargs):
            raise AssertionError("all-hit wave used the split-route executor")

        monkeypatch.setattr(runtime, "try_all_hit_route", record_fast_wave)
        monkeypatch.setattr(runtime, "begin_split_route", unexpected_split)

        def shared_work():
            events.append("shared")
            return mx.full(tokens.shape, 7, dtype=tokens.dtype)

        actual, shared = switch.run_with_shared_overlap(
            tokens,
            indices,
            shared_work,
        )
        mx.eval(actual, shared)

        after = runtime.snapshot(mx_module=mx)
        assert mx.array_equal(actual, reference).item()
        assert mx.all(shared == 7).item()
        assert fast_waves == [(2, 0, 2), (1,)]
        assert events == ["q4:(2, 0, 2)", "q4:(1,)", "shared"]
        assert after["cache"]["route_calls"] - before["route_calls"] == 2
        assert after["cache"]["expert_requests"] - before["expert_requests"] == 4
        assert after["cache"]["expert_hits"] - before["expert_hits"] == 4
        assert after["slots"]["pins"] == 0
    finally:
        runtime.close()


def test_component_bank_all_hit_decode_releases_pins_on_q4_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _config, spec, manifest_path = _integrated_glm_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(3),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        transient_slots=2,
        slot_layout="component-banks",
        cache_policy="lru",
    )
    plan = stream_config.memory_plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            plan,
            spec,
            load_expert_manifest(manifest_path),
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    switch = HotExpertSwitchGLU(runtime, 1)
    tokens = mx.zeros((2, 1, spec.hidden_size), dtype=mx.float32)
    indices = mx.array([[2, 0], [2, 1]], dtype=mx.int32)
    try:
        warm = switch(tokens, indices)
        mx.eval(warm)
        q4_calls = 0

        def fail_second_wave(
            selected: mx.array,
            bindings: tuple[ExpertSlotBinding, ...],
            *,
            group_size: int,
        ) -> mx.array:
            nonlocal q4_calls
            assert group_size == spec.quant_group_size
            assert len(bindings) == int(selected.shape[0])
            q4_calls += 1
            if q4_calls == 2:
                raise RuntimeError("injected Q4 failure")
            return selected

        monkeypatch.setattr(expert_mlx, "_run_component_bank_q4", fail_second_wave)

        with pytest.raises(RuntimeError, match="injected Q4 failure"):
            switch(tokens, indices)

        assert q4_calls == 2
        assert runtime.snapshot(mx_module=mx)["slots"]["pins"] == 0
    finally:
        runtime.close()


def test_slot_fence_all_hit_synchronous_eval_failure_blocks_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _config, spec, manifest_path = _integrated_glm_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(3),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        transient_slots=2,
        slot_layout="component-banks",
        cache_policy="lru",
    )
    plan = stream_config.memory_plan(spec)
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_component_bank_allocator(
            plan,
            spec,
            load_expert_manifest(manifest_path),
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    switch = HotExpertSwitchGLU(runtime, 1)
    tokens = mx.zeros((2, 1, spec.hidden_size), dtype=mx.float32)
    indices = mx.array([[2, 0], [2, 1]], dtype=mx.int32)
    eval_error = RuntimeError("injected all-hit synchronous eval failure")
    replacement_ready = None
    original_eval = expert_mlx.mx.eval
    monkeypatch.setattr(
        expert_mlx,
        "_run_component_bank_q4",
        lambda selected, *_a, **_k: selected,
    )

    try:
        warm = switch(tokens, indices)
        original_eval(warm)
        runtime.snapshot(mx_module=mx)
        slots = (*runtime.slots._persistent.values(), *runtime.slots._transient)
        before = tuple(
            (slot.state, slot.layer, slot.expert, slot.generation) for slot in slots
        )
        read_bytes = runtime.reader.metrics.as_dict()["read_bytes"]

        eval_calls = 0

        def fail_all_hit_fence(*values) -> None:
            nonlocal eval_calls
            eval_calls += 1
            if len(values) == 1 and values[0] is indices:
                original_eval(*values)
                return
            assert len(values) == 1
            assert tuple(values[0].shape) == (3, spec.hidden_size)
            raise eval_error

        monkeypatch.setattr(expert_mlx.mx, "eval", fail_all_hit_fence)
        with pytest.raises(RuntimeError, match="all-hit synchronous eval") as failed:
            switch(tokens, indices)
        monkeypatch.setattr(expert_mlx.mx, "eval", original_eval)
        assert failed.value is eval_error
        assert eval_calls == 2
        assert runtime.slots.metrics.as_dict()["active_routes"] == 0
        assert sum(slot.pins for slot in slots) == 0
        assert (
            tuple(
                (slot.state, slot.layer, slot.expert, slot.generation) for slot in slots
            )
            == before
        )

        with pytest.raises(ExpertSlotError, match="completion fence failed") as blocked:
            replacement_ready = runtime.ensure_route(1, [3, 0], phase="decode")
        assert blocked.value.__cause__ is eval_error
        assert runtime.reader.metrics.as_dict()["read_bytes"] == read_bytes

        with pytest.raises(ExpertSlotError, match="completion fence failed") as visible:
            runtime.snapshot(mx_module=mx)
        assert visible.value.__cause__ is eval_error
        with pytest.raises(ExpertSlotError, match="completion fence failed") as closed:
            runtime.close(timeout=2)
        assert closed.value.__cause__ is eval_error
        assert runtime.reader._closed is True
    finally:
        monkeypatch.setattr(expert_mlx.mx, "eval", original_eval)
        if replacement_ready is not None:
            replacement_ready.release(synchronize=False)
        try:
            runtime.close(timeout=2)
        except ExpertSlotError:
            pass


def test_resident_loader_reads_extensionless_hugging_face_cache_blob(
    tmp_path: Path,
) -> None:
    source_parent = tmp_path / "source"
    source_parent.mkdir()
    source, config, spec, source_manifest = _integrated_hy3_artifact(source_parent)
    repository = tmp_path / "models--test--tiny-hy3"
    blobs = repository / "blobs"
    snapshot = repository / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / ("b" * 64)
    blob.hardlink_to(source / "model.safetensors")
    (snapshot / "model.safetensors").symlink_to(Path("..") / ".." / "blobs" / blob.name)
    (snapshot / "config.json").write_text(
        (source / "config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    manifest_path = snapshot / "expert-manifest.json"
    manifest_path.write_text(
        source_manifest.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    runtime = ExpertStreamingRuntime.open(
        snapshot,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_slot_buffer_allocator(
            stream_config.memory_plan(spec), spec
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    try:
        resident = construct_resident_model(snapshot, runtime, config=config)
        logits = resident.model(mx.array([[1]], dtype=mx.int32))
        mx.eval(logits)
        assert logits.shape == (1, 1, config["vocab_size"])
    finally:
        runtime.close()


def test_batched_single_token_decode_warms_persistent_expert_cache(
    tmp_path: Path,
) -> None:
    root, config, spec, manifest_path = _integrated_hy3_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed + spec.persistent_cache_bytes(1),
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        buffer_allocator=make_mlx_slot_buffer_allocator(
            stream_config.memory_plan(spec), spec
        ),
        device_synchronize=mx.synchronize,
        apply_memory_cap=False,
    )
    try:
        resident = construct_resident_model(root, runtime, config=config)
        logits = resident.model(mx.array([[1], [2]], dtype=mx.int32))
        mx.eval(logits)

        assert logits.shape == (2, 1, config["vocab_size"])
        assert runtime._banks[1].occupancy == 1
        assert runtime.snapshot(mx_module=mx)["cache"]["persistent_loads"] >= 1
    finally:
        runtime.close()


def _integrated_glm_artifact(tmp_path: Path):
    args = _glm_args(layers=6, first_sparse=1)
    model = GlmModel(args)
    weights = dict(tree_flatten(model.parameters()))
    expert_shapes = {
        "gate_proj.weight": (4, 64, 8),
        "gate_proj.scales": (4, 64, 1),
        "gate_proj.biases": (4, 64, 1),
        "up_proj.weight": (4, 64, 8),
        "up_proj.scales": (4, 64, 1),
        "up_proj.biases": (4, 64, 1),
        "down_proj.weight": (4, 64, 8),
        "down_proj.scales": (4, 64, 1),
        "down_proj.biases": (4, 64, 1),
    }
    for layer in range(1, 6):
        for component, shape in expert_shapes.items():
            dtype = mx.uint32 if component.endswith("weight") else mx.bfloat16
            value = mx.zeros(shape, dtype=dtype)
            if component.endswith("scales"):
                value = mx.ones(shape, dtype=dtype)
            weights[f"model.layers.{layer}.mlp.switch_mlp.{component}"] = value
    mx.eval(weights)
    root = tmp_path / "glm"
    root.mkdir()
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    config = asdict(args)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    total_bytes = sum(int(value.nbytes) for value in weights.values())
    spec = ExpertStreamingModelSpec(
        key="tiny-glm52-q4",
        display_name="Tiny GLM-5.2 Q4",
        source_model="test/tiny-glm",
        source_revision="source",
        quant_model="test/tiny-glm-q4",
        quant_revision="quant",
        total_tensor_bytes=total_bytes,
        total_layers=6,
        routed_layer_start=1,
        routed_layer_count=5,
        expert_count=4,
        top_k=2,
        hidden_size=64,
        expert_hidden_size=64,
        quant_bits=4,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="float32",
        router_matmul_dtype="float32",
        router_bytes=5 * (4 * 64 * 4 + 4 * 4),
        kv_bytes_per_token=0,
        mtp_layer_index=6,
        mtp_included=False,
        full_indexer_layers=(0, 2, 4),
    )
    manifest = build_expert_manifest(root, spec)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    return root, config, spec, manifest_path


def test_resident_loader_runs_indexshare_glm_with_streamed_sparse_layers(
    tmp_path: Path,
) -> None:
    root, config, spec, manifest_path = _integrated_glm_artifact(tmp_path)
    fixed = spec.resident_bytes + spec.transient_scratch_bytes
    stream_config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=fixed,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        stream_config,
        spec=spec,
        apply_memory_cap=False,
    )
    try:
        resident = construct_resident_model(root, runtime, config=config)
        parameter_names = {
            name for name, _ in tree_flatten(resident.model.parameters())
        }
        assert not any("switch_mlp" in name for name in parameter_names)
        assert [
            layer.self_attn.indexer is not None for layer in resident.model.model.layers
        ] == [True, False, True, False, True, False]

        logits = resident.model(mx.array([[1]], dtype=mx.int32))
        mx.eval(logits)
        assert logits.shape == (1, 1, config["vocab_size"])
        assert mx.all(mx.isfinite(logits)).item()
        assert runtime.snapshot(mx_module=mx)["cache"]["route_calls"] == 5
    finally:
        runtime.close()
