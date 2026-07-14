from __future__ import annotations

import contextlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mtplx.cache_state import restore_cache, snapshot_cache
from mtplx.models.glm52_mlx import Model as GlmModel
from mtplx.models.glm52_mlx import ModelArgs as GlmArgs
from mtplx.mtp_patch import MTPContract


TEST_REVISION = "test-glm52-revision"


def _args() -> GlmArgs:
    return GlmArgs(
        model_type="glm_moe_dsa",
        vocab_size=128,
        hidden_size=64,
        index_head_dim=8,
        index_n_heads=4,
        index_topk=4,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=6,
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
        first_k_dense_replace=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10_000.0},
        attention_bias=False,
        index_topk_pattern="FSFSFS",
        index_topk_freq=4,
        index_skip_topk_offset=3,
        num_nextn_predict_layers=1,
        index_share_for_mtp_iteration=True,
    )


def _raw_tensors(args: GlmArgs) -> dict[str, mx.array]:
    layer = args.num_hidden_layers
    prefix = f"model.layers.{layer}."
    h = args.hidden_size
    q = args.q_lora_rank
    kv = args.kv_lora_rank
    rope = args.qk_rope_head_dim
    nope = args.qk_nope_head_dim
    value = args.v_head_dim
    heads = args.num_attention_heads
    index_heads = args.index_n_heads
    index_dim = args.index_head_dim
    experts = args.n_routed_experts
    moe = args.moe_intermediate_size
    shared = moe * int(args.n_shared_experts or 0)
    tensors = {
        prefix + "eh_proj.weight": mx.zeros((h, 2 * h), dtype=mx.bfloat16),
        prefix + "enorm.weight": mx.ones((h,), dtype=mx.bfloat16),
        prefix + "hnorm.weight": mx.ones((h,), dtype=mx.bfloat16),
        prefix + "input_layernorm.weight": mx.ones((h,), dtype=mx.bfloat16),
        prefix + "mlp.gate.e_score_correction_bias": mx.zeros(
            (experts,), dtype=mx.float32
        ),
        prefix + "mlp.gate.weight": mx.zeros((experts, h), dtype=mx.bfloat16),
        prefix + "mlp.shared_experts.down_proj.weight": mx.zeros(
            (h, shared), dtype=mx.bfloat16
        ),
        prefix + "mlp.shared_experts.gate_proj.weight": mx.zeros(
            (shared, h), dtype=mx.bfloat16
        ),
        prefix + "mlp.shared_experts.up_proj.weight": mx.zeros(
            (shared, h), dtype=mx.bfloat16
        ),
        prefix + "post_attention_layernorm.weight": mx.ones((h,), dtype=mx.bfloat16),
        prefix + "self_attn.indexer.k_norm.bias": mx.zeros(
            (index_dim,), dtype=mx.bfloat16
        ),
        prefix + "self_attn.indexer.k_norm.weight": mx.ones(
            (index_dim,), dtype=mx.bfloat16
        ),
        prefix + "self_attn.indexer.weights_proj.weight": mx.zeros(
            (index_heads, h), dtype=mx.bfloat16
        ),
        prefix + "self_attn.indexer.wk.weight": mx.zeros(
            (index_dim, h), dtype=mx.bfloat16
        ),
        prefix + "self_attn.indexer.wq_b.weight": mx.zeros(
            (index_heads * index_dim, q), dtype=mx.bfloat16
        ),
        prefix + "self_attn.kv_a_layernorm.weight": mx.ones((kv,), dtype=mx.bfloat16),
        prefix + "self_attn.kv_a_proj_with_mqa.weight": mx.zeros(
            (kv + rope, h), dtype=mx.bfloat16
        ),
        prefix + "self_attn.kv_b_proj.weight": mx.zeros(
            (heads * (nope + value), kv), dtype=mx.bfloat16
        ),
        prefix + "self_attn.o_proj.weight": mx.zeros(
            (h, heads * value), dtype=mx.bfloat16
        ),
        prefix + "self_attn.q_a_layernorm.weight": mx.ones((q,), dtype=mx.bfloat16),
        prefix + "self_attn.q_a_proj.weight": mx.zeros((q, h), dtype=mx.bfloat16),
        prefix + "self_attn.q_b_proj.weight": mx.zeros(
            (heads * (nope + rope), q), dtype=mx.bfloat16
        ),
        prefix + "shared_head.norm.weight": mx.ones((h,), dtype=mx.bfloat16),
    }
    for expert in range(experts):
        for projection, shape in (
            ("gate_proj", (moe, h)),
            ("up_proj", (moe, h)),
            ("down_proj", (h, moe)),
        ):
            tensors[f"{prefix}mlp.experts.{expert}.{projection}.weight"] = mx.full(
                shape, expert + 1, dtype=mx.bfloat16
            )
    mx.eval(list(tensors.values()))
    return tensors


def _write_artifact(root: Path, tensors: dict[str, mx.array]) -> None:
    from mtplx.glm52_mtp_patch import GLM52_MTP_BF16_FILE

    root.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(root / GLM52_MTP_BF16_FILE), tensors)


def _trust_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    @contextlib.contextmanager
    def open_verified(root: Path, *, deep: bool = True):
        resolved_root = Path(root).resolve()
        with (resolved_root / "layer78-bf16.safetensors").open("rb") as artifact_file:
            yield SimpleNamespace(
                file=artifact_file,
                root=resolved_root,
                manifest={
                    "schema": "mtplx-glm52-mtp-layer78-v1",
                    "source_revision": TEST_REVISION,
                    "deep": deep,
                },
            )

    monkeypatch.setattr(
        "mtplx.glm52_mtp_patch.open_verified_glm52_mtp_layer78",
        open_verified,
    )


def test_glm52_bf16_loader_maps_exact_inventory_without_quantization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx.glm52_mtp_patch import (
        build_glm52_mtp_module,
        expected_glm52_mtp_inventory,
        load_glm52_mtp_bf16_weights,
    )

    args = _args()
    raw = _raw_tensors(args)
    assert len(raw) == 35
    assert len(expected_glm52_mtp_inventory(args)) == 35
    _write_artifact(tmp_path, raw)
    _trust_artifact(monkeypatch)

    mapped = load_glm52_mtp_bf16_weights(
        tmp_path,
        args,
        expected_revision=TEST_REVISION,
    )

    assert len(mapped) == 27
    assert not any(".experts." in name for name in mapped)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        name = f"layers.0.mtp_block.mlp.switch_mlp.{projection}.weight"
        stacked = mapped[name]
        assert stacked.shape[0] == args.n_routed_experts
        assert stacked.dtype == mx.bfloat16
        for expert in range(args.n_routed_experts):
            source = raw[f"model.layers.6.mlp.experts.{expert}.{projection}.weight"]
            assert mx.array_equal(stacked[expert], source).item()
    assert (
        mapped["layers.0.mtp_block.mlp.gate.e_score_correction_bias"].dtype
        == mx.float32
    )
    assert mapped["layers.0.mtp_block.self_attn.embed_q.weight"].shape == (4, 16, 8)
    assert mapped["layers.0.mtp_block.self_attn.unembed_out.weight"].shape == (
        4,
        16,
        16,
    )

    mtp = build_glm52_mtp_module(
        tmp_path,
        args,
        expected_revision=TEST_REVISION,
    )
    names = [name for name, _value in tree_flatten(mtp.parameters())]
    assert not any("shared_head_head" in name for name in names)
    assert not any(
        "UnboundExpertSwitch" in type(module).__name__ for module in _modules(mtp)
    )
    assert not any("Quantized" in type(module).__name__ for module in _modules(mtp))


def test_glm52_loader_uses_verified_handle_through_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtplx.glm52_mtp_patch as patch

    args = _args()
    original = _raw_tensors(args)
    _write_artifact(tmp_path, original)
    artifact_path = tmp_path / patch.GLM52_MTP_BF16_FILE
    replacement_path = tmp_path.parent / "replacement-layer78.safetensors"
    replacement = dict(original)
    expert_name = "model.layers.6.mlp.experts.0.gate_proj.weight"
    replacement[expert_name] = mx.full(
        original[expert_name].shape,
        99,
        dtype=mx.bfloat16,
    )
    mx.save_safetensors(str(replacement_path), replacement)
    receipt = {"source": {"revision": TEST_REVISION}}
    observed: dict[str, object] = {}
    replaced = False

    def replace_once() -> None:
        nonlocal replaced
        if not replaced:
            os.replace(replacement_path, artifact_path)
            replaced = True

    @contextlib.contextmanager
    def open_verified(_root: Path, *, deep: bool = True):
        assert deep is True
        with artifact_path.open("rb") as artifact_file:
            replace_once()
            yield SimpleNamespace(
                file=artifact_file,
                root=tmp_path.resolve(),
                manifest=receipt,
            )
        observed["closed_after_context"] = artifact_file.closed

    def legacy_verify(_root: Path, *, deep: bool = True):
        assert deep is True
        replace_once()
        return receipt

    class MXProxy:
        bfloat16 = mx.bfloat16
        float32 = mx.float32

        def __getattr__(self, name: str):
            return getattr(mx, name)

        def load(self, source, *, format: str):
            observed["load_received_path"] = isinstance(source, (str, Path))
            observed["artifact_file"] = source
            return mx.load(source, format=format)

        def eval(self, *values):
            artifact_file = observed["artifact_file"]
            observed["materialized_while_open"] = not artifact_file.closed
            return mx.eval(*values)

    monkeypatch.setattr(
        patch,
        "open_verified_glm52_mtp_layer78",
        open_verified,
        raising=False,
    )
    monkeypatch.setattr(
        patch,
        "verify_glm52_mtp_layer78",
        legacy_verify,
        raising=False,
    )

    mapped = patch.load_glm52_mtp_bf16_weights(
        tmp_path,
        args,
        expected_revision=TEST_REVISION,
        mx_module=MXProxy(),
    )

    assert observed["load_received_path"] is False
    assert observed["materialized_while_open"] is True
    assert observed["closed_after_context"] is True
    stacked = mapped["layers.0.mtp_block.mlp.switch_mlp.gate_proj.weight"]
    assert mx.array_equal(stacked[0], original[expert_name]).item()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing tensors"),
        ("extra", "unexpected tensors"),
        ("bf16_as_f32", "must be bfloat16"),
        ("correction_as_bf16", "must be float32"),
        ("wrong_shape", "expected shape"),
    ],
)
def test_glm52_loader_fails_closed_on_inventory_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    from mtplx.glm52_mtp_patch import Glm52MTPLoadError, load_glm52_mtp_bf16_weights

    args = _args()
    tensors = _raw_tensors(args)
    prefix = "model.layers.6."
    if mutation == "missing":
        tensors.pop(prefix + "enorm.weight")
    elif mutation == "extra":
        tensors[prefix + "unexpected.weight"] = mx.zeros((1,), dtype=mx.bfloat16)
    elif mutation == "bf16_as_f32":
        tensors[prefix + "enorm.weight"] = mx.zeros((64,), dtype=mx.float32)
    elif mutation == "correction_as_bf16":
        tensors[prefix + "mlp.gate.e_score_correction_bias"] = mx.zeros(
            (4,), dtype=mx.bfloat16
        )
    elif mutation == "wrong_shape":
        tensors[prefix + "enorm.weight"] = mx.zeros((63,), dtype=mx.bfloat16)
    _write_artifact(tmp_path, tensors)
    _trust_artifact(monkeypatch)

    with pytest.raises(Glm52MTPLoadError, match=message):
        load_glm52_mtp_bf16_weights(
            tmp_path,
            args,
            expected_revision=TEST_REVISION,
        )


def test_glm52_loader_validates_provenance_before_mlx_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx.glm52_mtp_patch import Glm52MTPLoadError, load_glm52_mtp_bf16_weights

    _write_artifact(tmp_path, _raw_tensors(_args()))

    @contextlib.contextmanager
    def reject(_root, *, deep=True):
        assert deep is True
        raise RuntimeError("manifest digest mismatch")
        yield

    monkeypatch.setattr(
        "mtplx.glm52_mtp_patch.open_verified_glm52_mtp_layer78",
        reject,
    )
    monkeypatch.setattr(mx, "load", lambda *_args, **_kwargs: pytest.fail("allocated"))

    with pytest.raises(Glm52MTPLoadError, match="manifest digest mismatch"):
        load_glm52_mtp_bf16_weights(
            tmp_path,
            _args(),
            expected_revision=TEST_REVISION,
        )


@pytest.mark.parametrize(
    "contract_update",
    [
        {"base_hidden_variant": "pre_norm"},
        {"hidden_variant": "pre_norm"},
        {"concat_order": "hidden_embedding"},
        {"mtp_position_mode": "local"},
        {"mtp_position_mode": "absolute"},
        {"mtp_quant_bits": 4},
        {"mtp_quant_policy": "all"},
        {"mtp_prequantized": True},
        {"mtp_prequantized_modules": ("layers.0.eh_proj",)},
        {
            "mtp_prequantized_module_specs": {
                "layers.0.eh_proj": {
                    "bits": 4,
                    "group_size": 64,
                    "mode": "affine",
                }
            }
        },
    ],
)
def test_glm52_injection_rejects_incompatible_contract_before_building_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_update: dict[str, object],
) -> None:
    from mtplx.glm52_mtp_patch import (
        Glm52MTPLoadError,
        inject_glm52_streamed_mtp_support,
    )

    monkeypatch.setattr(
        "mtplx.glm52_mtp_patch.build_glm52_mtp_module",
        lambda *_args, **_kwargs: pytest.fail("built head before contract validation"),
    )

    with pytest.raises(Glm52MTPLoadError, match="contract"):
        inject_glm52_streamed_mtp_support(
            GlmModel(_args()),
            tmp_path,
            {"model_type": "glm_moe_dsa"},
            replace(MTPContract(), **contract_update),
            expected_revision=TEST_REVISION,
        )


def test_glm52_injection_forwards_borrowed_verified_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx.glm52_mtp_patch import inject_glm52_streamed_mtp_support

    borrowed = SimpleNamespace(file=object(), manifest={})
    observed: dict[str, object] = {}

    def build(
        artifact_dir,
        args,
        *,
        expected_revision,
        verified_artifact=None,
    ):
        observed.update(
            artifact_dir=artifact_dir,
            args=args,
            expected_revision=expected_revision,
            verified_artifact=verified_artifact,
        )
        return SimpleNamespace()

    monkeypatch.setattr("mtplx.glm52_mtp_patch.build_glm52_mtp_module", build)
    args = _args()
    assert inject_glm52_streamed_mtp_support(
        GlmModel(args),
        tmp_path,
        {"model_type": "glm_moe_dsa"},
        MTPContract(),
        expected_revision=TEST_REVISION,
        verified_artifact=borrowed,
    )

    assert observed == {
        "artifact_dir": tmp_path,
        "args": args,
        "expected_revision": TEST_REVISION,
        "verified_artifact": borrowed,
    }


def test_glm52_injection_uses_prebuilt_head_without_reloading_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx.glm52_mtp_patch import inject_glm52_streamed_mtp_support

    monkeypatch.setattr(
        "mtplx.glm52_mtp_patch.build_glm52_mtp_module",
        lambda *_args, **_kwargs: pytest.fail("prebuilt head reloaded artifact"),
    )
    model = GlmModel(_args())
    prebuilt = SimpleNamespace(layers=[object()])

    assert inject_glm52_streamed_mtp_support(
        model,
        tmp_path,
        {"model_type": "glm_moe_dsa"},
        MTPContract(),
        expected_revision=TEST_REVISION,
        mtp_module=prebuilt,
    )
    assert model.mtp is prebuilt


def test_glm52_recurrent_depth_requires_a_persistent_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx.glm52_mtp_patch import (
        Glm52MTPLoadError,
        inject_glm52_streamed_mtp_support,
    )

    args = _args()
    _write_artifact(tmp_path, _raw_tensors(args))
    _trust_artifact(monkeypatch)
    model = GlmModel(args)
    assert inject_glm52_streamed_mtp_support(
        model,
        tmp_path,
        {"model_type": "glm_moe_dsa"},
        MTPContract(),
        expected_revision=TEST_REVISION,
    )
    assert model.mtp_recurrent_requires_persistent_cache is True

    with pytest.raises(Glm52MTPLoadError, match="persistent cache"):
        model.mtp_forward(
            mx.zeros((1, 1, args.hidden_size), dtype=mx.bfloat16),
            mx.array([[1]], dtype=mx.int32),
            mtp_cache=None,
            return_hidden=True,
            mtp_depth=2,
        )


def test_glm52_cache_keeps_only_d1_committed_and_snapshots_exclude_scratch() -> None:
    from mlx_lm.models.cache import KVCache

    from mtplx.glm52_mtp_patch import GLM52MTPCache

    cache = GLM52MTPCache(KVCache(), KVCache(), index_topk=4)
    cache.begin_cycle()
    _append(cache.main_kv, 1)
    _append(cache.indexer_kv, 1)
    cache.finish_d1(topk_indices=mx.array([[[[0]]]], dtype=mx.uint32))
    _append(cache.main_kv, 3)

    assert cache.offset == 4
    assert cache.indexer_kv.offset == 1
    assert cache.cycle is not None
    assert cache.cycle.d1_boundary == 1

    snapshot = snapshot_cache([cache])
    cache.finish_cycle()
    assert cache.main_kv.offset == 1
    assert cache.indexer_kv.offset == 1
    assert cache.cycle is None

    _append(cache.main_kv, 2)
    _append(cache.indexer_kv, 2)
    restore_cache([cache], snapshot)
    assert cache.main_kv.offset == 1
    assert cache.indexer_kv.offset == 1
    assert cache.cycle is None


@pytest.mark.parametrize(
    ("indexed_length", "should_reject"), [(2048, False), (2049, True)]
)
def test_glm52_cache_distinguishes_dense_prefix_from_missing_full_indexer_result(
    indexed_length: int,
    should_reject: bool,
) -> None:
    from mlx_lm.models.cache import KVCache

    from mtplx.glm52_mtp_patch import GLM52MTPCache, Glm52MTPLoadError

    cache = GLM52MTPCache(KVCache(), KVCache(), index_topk=2048)
    _append(cache.main_kv, indexed_length - 1)
    _append(cache.indexer_kv, indexed_length - 1)
    cache.begin_cycle()
    _append(cache.main_kv, 1)
    _append(cache.indexer_kv, 1)

    if should_reject:
        with pytest.raises(Glm52MTPLoadError, match="full indexer"):
            cache.finish_d1(topk_indices=None)
    else:
        cache.finish_d1(topk_indices=None)
        topk_indices, boundary = cache.recurrent_view()
        assert topk_indices is None
        assert boundary == indexed_length


def test_glm52_injection_shares_target_heads_and_reuses_indexer_across_d1_d5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx.glm52_mtp_patch import inject_glm52_streamed_mtp_support

    args = _args()
    _write_artifact(tmp_path, _raw_tensors(args))
    _trust_artifact(monkeypatch)
    model = GlmModel(args)
    embed_tokens = model.model.embed_tokens
    lm_head = model.lm_head

    assert inject_glm52_streamed_mtp_support(
        model,
        tmp_path,
        {"model_type": "glm_moe_dsa"},
        expected_revision=TEST_REVISION,
    )
    assert model.model.embed_tokens is embed_tokens
    assert model.lm_head is lm_head
    assert model.mtp_verify_width == 6

    cache = model.make_mtp_cache()
    model.mtp_update_cache(
        mx.zeros((1, args.index_topk, args.hidden_size), dtype=mx.bfloat16),
        mx.ones((1, args.index_topk), dtype=mx.int32),
        mtp_cache=cache,
    )
    assert cache[0].main_kv.offset == args.index_topk
    assert cache[0].indexer_kv.offset == args.index_topk

    indexer = model.mtp.layers[0].mtp_block.self_attn.indexer
    calls = 0

    class CountingIndexer:
        def __call__(self, *call_args, **call_kwargs):
            nonlocal calls
            calls += 1
            return indexer(*call_args, **call_kwargs)

    model.mtp.layers[0].mtp_block.self_attn.indexer = CountingIndexer()
    hidden = mx.zeros((1, 1, args.hidden_size), dtype=mx.bfloat16)
    token = mx.array([[1]], dtype=mx.int32)
    main_offsets = []
    indexer_offsets = []
    for depth in range(1, 6):
        logits, hidden = model.mtp_forward(
            hidden,
            token,
            mtp_cache=cache,
            return_hidden=True,
            mtp_depth=depth,
        )
        mx.eval(logits, hidden)
        main_offsets.append(cache[0].main_kv.offset)
        indexer_offsets.append(cache[0].indexer_kv.offset)

    assert calls == 1
    assert main_offsets == [5, 6, 7, 8, 9]
    assert indexer_offsets == [5, 5, 5, 5, 5]
    model.finish_mtp_cycle(cache)
    assert cache[0].main_kv.offset == 5
    assert cache[0].indexer_kv.offset == 5
    assert cache[0].cycle is None

    model.mtp_update_cache(
        mx.zeros((1, 2, args.hidden_size), dtype=mx.bfloat16),
        mx.array([[2, 3]], dtype=mx.int32),
        mtp_cache=cache,
    )
    assert cache[0].main_kv.offset == 7
    assert cache[0].indexer_kv.offset == 7
    assert cache[0].cycle is None


def _append(cache, count: int) -> None:
    values = mx.zeros((1, 1, count, 2), dtype=mx.bfloat16)
    cache.update_and_fetch(values, values)


def _modules(root: nn.Module) -> list[nn.Module]:
    found: list[nn.Module] = [root]
    for child in root.children().values():
        if isinstance(child, list):
            for item in child:
                if isinstance(item, nn.Module):
                    found.extend(_modules(item))
        elif isinstance(child, nn.Module):
            found.extend(_modules(child))
    return found
