from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from mtplx.expert_streaming_models import (
    MODEL_SPECS,
    QWEN38_FLASH_NEXT_Q4,
    get_model_spec,
)
from mtplx.models.qwen4_exp import ModelArgs

PINNED_REVISION = "43a82b3f0ff64fa417fd09ca046580f08d19b0d6"
CACHED_CONFIG = Path(
    "/Users/davidtai/.cache/huggingface/hub/"
    "models--Vontra--Qwen3.8-Flash-Next-MLX-oQ4-MTP/snapshots/"
    f"{PINNED_REVISION}/config.json"
)


def _text_config() -> dict[str, object]:
    layer_types = [
        "linear_attention" if layer % 4 != 3 else "full_attention"
        for layer in range(48)
    ]
    return {
        "model_type": "qwen4_exp_text",
        "dtype": "bfloat16",
        "hidden_act": "silu",
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "full_attention_interval": 4,
        "layer_types": layer_types,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "ple_layer_ids": [2],
        "ple_embed_dim": 2560,
        "ple_conv_kernel_size": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "hc_count": 4,
        "hc_lowrank": 320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 20_000_000,
        "make_ngram_vocab_size_divisible_by": 128,
        "split_ngram_parts": 128,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "mtp": {
            "hybrid": True,
            "layer_types": ["full_attention"],
            "mtp_use_hidden_state_from_layer": None,
            "num_hidden_layers": 1,
            "rope_theta": 10_000_000,
        },
        "vocab_size": 248_320,
        "eos_token_id": 248_044,
        "rms_norm_eps": 0.000001,
        "output_gate_type": "sigmoid",
        "tie_word_embeddings": False,
        "use_cache": True,
        "max_position_embeddings": 262_144,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "rope_type": "default",
        },
    }


def _root_config() -> dict[str, object]:
    return {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": _text_config(),
    }


def _assert_pinned_args(args: ModelArgs) -> None:
    expected_layers = tuple(
        "linear_attention" if layer % 4 != 3 else "qwen_sparse_attention"
        for layer in range(48)
    )
    assert args.model_type == "qwen4_exp_text"
    assert args.dtype == "bfloat16"
    assert args.hidden_act == "silu"
    assert args.hidden_size == 2560
    assert args.num_hidden_layers == 48
    assert args.full_attention_interval == 4
    assert args.layer_types == expected_layers
    assert args.attention_bias is False
    assert args.attention_dropout == 0.0
    assert args.ple_layer_ids == (2,)
    assert (args.ple_embed_dim, args.ple_conv_kernel_size) == (2560, 4)
    assert args.num_experts == 512
    assert args.num_experts_per_tok == 10
    assert args.moe_intermediate_size == 640
    assert args.shared_expert_intermediate_size == 640
    assert (args.hc_count, args.hc_lowrank) == (4, 320)
    assert (
        args.linear_num_key_heads,
        args.linear_num_value_heads,
        args.linear_key_head_dim,
        args.linear_value_head_dim,
        args.linear_conv_kernel_dim,
    ) == (16, 48, 128, 128, 4)
    assert args.mamba_ssm_dtype == "float32"
    assert (
        args.num_attention_heads,
        args.num_key_value_heads,
        args.head_dim,
    ) == (24, 2, 256)
    assert (
        args.indexer_n_heads,
        args.indexer_kv_heads,
        args.indexer_head_dim,
        args.indexer_budget,
        args.indexer_compress_ratio,
    ) == (4, 1, 128, 2048, 4)
    assert (
        args.ngram_size,
        args.heads_per_ngram,
        args.ngram_vocab_size_base,
        args.make_ngram_vocab_size_divisible_by,
        args.split_ngram_parts,
    ) == (3, 8, 20_000_000, 128, 128)
    assert args.mtp_num_hidden_layers == 1
    assert args.mtp_use_dedicated_embeddings is False
    assert args.mtp.hybrid is True
    assert args.mtp.layer_types == ("qwen_sparse_attention",)
    assert args.mtp.mtp_use_hidden_state_from_layer is None
    assert args.mtp.num_hidden_layers == 1
    assert args.mtp.rope_theta == 10_000_000
    assert (args.vocab_size, args.eos_token_id) == (248_320, 248_044)
    assert args.rms_norm_eps == 1e-6
    assert args.output_gate_type == "sigmoid"
    assert args.tie_word_embeddings is False
    assert args.use_cache is True
    assert args.max_position_embeddings == 262_144
    assert args.partial_rotary_factor == 0.25
    assert args.rope_parameters.mrope_interleaved is True
    assert args.rope_parameters.mrope_section == (11, 11, 10)
    assert args.rope_parameters.partial_rotary_factor == 0.25
    assert args.rope_parameters.rope_theta == 10_000_000
    assert args.rope_parameters.rope_type == "default"
    assert args.streamed_layer_ids == tuple(range(49))
    assert args.seed == 1234


def _set_path(
    config: dict[str, object], path: tuple[object, ...], value: object
) -> None:
    target: object = config
    for part in path[:-1]:
        if isinstance(part, int):
            assert isinstance(target, list)
            target = target[part]
        else:
            assert isinstance(target, dict)
            target = target[part]
    leaf = path[-1]
    if isinstance(leaf, int):
        assert isinstance(target, list)
        target[leaf] = value
    else:
        assert isinstance(target, dict)
        target[leaf] = value


def _pinned_scalar_mutations() -> list[tuple[tuple[object, ...], object]]:
    mutations: list[tuple[tuple[object, ...], object]] = []

    def visit(value: object, path: tuple[object, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, index))
        elif type(value) is bool:
            mutations.append((path, not value))
        elif type(value) is int:
            mutations.append((path, value + 1))
        elif type(value) is float:
            mutations.append((path, value + 0.5))
        elif type(value) is str:
            mutations.append((path, f"{value}-changed"))
        elif value is None:
            mutations.append((path, 0))
        else:
            raise AssertionError(f"uncovered fixture type at {path}: {type(value)}")

    visit(_text_config(), ())
    return mutations


@pytest.mark.parametrize("config", [_root_config(), _text_config()])
def test_from_dict_accepts_root_or_text_config_and_pins_geometry(
    config: dict[str, object],
) -> None:
    _assert_pinned_args(ModelArgs.from_dict(config))


@pytest.mark.parametrize(
    "root_fields",
    ({}, {"model_type": None}, {"model_type": "qwen4_exp_text"}),
)
def test_root_wrapper_requires_exact_qwen4_exp_model_type(
    root_fields: dict[str, object],
) -> None:
    wrapper = {**root_fields, "text_config": _text_config()}

    with pytest.raises(ValueError, match="root model_type must be qwen4_exp"):
        ModelArgs.from_dict(wrapper)


def test_cached_pinned_config_matches_repository_fixture_when_available() -> None:
    if not CACHED_CONFIG.exists():
        pytest.skip("pinned Hugging Face config is not cached on this machine")

    cached = json.loads(CACHED_CONFIG.read_text(encoding="utf-8"))
    _assert_pinned_args(ModelArgs.from_dict(cached))


def test_missing_rope_type_uses_transformers_default() -> None:
    config = _text_config()
    rope_parameters = config["rope_parameters"]
    assert isinstance(rope_parameters, dict)
    del rope_parameters["rope_type"]

    args = ModelArgs.from_dict(config)

    assert args.rope_parameters.rope_type == "default"


def test_model_args_are_frozen_and_reject_geometry_drift() -> None:
    args = ModelArgs.from_dict(_root_config())
    with pytest.raises(FrozenInstanceError):
        args.hidden_size = 1  # type: ignore[misc]

    changed = copy.deepcopy(_root_config())
    text_config = changed["text_config"]
    assert isinstance(text_config, dict)
    text_config["indexer_budget"] = 1024
    with pytest.raises(ValueError, match="pinned Qwen3.8 Flash-Next geometry"):
        ModelArgs.from_dict(changed)


@pytest.mark.parametrize(("path", "changed_value"), _pinned_scalar_mutations())
def test_every_pinned_config_scalar_rejects_drift(
    path: tuple[object, ...], changed_value: object
) -> None:
    changed = _text_config()
    _set_path(changed, path, changed_value)

    with pytest.raises(
        (TypeError, ValueError), match="pinned Qwen3.8 Flash-Next geometry"
    ):
        ModelArgs.from_dict(changed)


@pytest.mark.parametrize(
    ("path", "malformed"),
    (
        (("indexer_kv_heads",), True),
        (("mtp_num_hidden_layers",), True),
        (("mtp", "num_hidden_layers"), True),
        (("eos_token_id",), 248_044.0),
        (("mtp_use_dedicated_embeddings",), 0),
        (("mtp", "rope_theta"), 10_000_000.0),
        (("rope_parameters", "rope_theta"), 10_000_000.0),
        (("attention_bias",), 0),
        (("use_cache",), 1),
        (("mtp", "hybrid"), 1),
        (("rope_parameters", "mrope_interleaved"), 1),
        (("hidden_act",), 1),
        (("layer_types",), "linear_attention"),
        (("layer_types", 0), 0),
        (("ple_layer_ids", 0), True),
        (("mtp",), []),
        (("rope_parameters",), []),
        (("rope_parameters", "mrope_section", 0), 11.0),
        (("attention_dropout",), 0),
        (("rms_norm_eps",), 0),
        (("partial_rotary_factor",), 0),
        (("rope_parameters", "partial_rotary_factor"), 0),
    ),
)
def test_pinned_config_rejects_wrong_runtime_types(
    path: tuple[object, ...], malformed: object
) -> None:
    changed = _text_config()
    _set_path(changed, path, malformed)

    with pytest.raises(TypeError, match="exact"):
        ModelArgs.from_dict(changed)


def test_qwen38_flash_next_q4_streaming_geometry_is_exact() -> None:
    spec = QWEN38_FLASH_NEXT_Q4

    # Task 12 must publish the derivative and replace the construction digest
    # with its immutable HF commit before this candidate can enter the registry.
    assert spec.key not in MODEL_SPECS
    with pytest.raises(ValueError, match="unknown model"):
        get_model_spec(spec.key)
    assert spec.total_tensor_bytes == 81_325_121_012
    assert spec.external_backing_bytes == 32_000_153_600
    assert spec.disk_tensor_bytes == 113_325_274_612
    assert spec.total_layers == 49
    assert spec.routed_layer_indices == tuple(range(49))
    assert spec.expert_count == 512
    assert spec.top_k == 10
    assert spec.hidden_size == 2560
    assert spec.expert_hidden_size == 640
    assert spec.quant_bits == 4
    assert spec.quant_group_size == 32
    assert spec.quant_parameter_bytes == 2
    assert spec.expert_codec == "affine"
    assert spec.expert_activation == "swiglu"
    assert spec.expert_record_bytes == 3_072_000
    assert spec.routed_expert_bytes == 77_070_336_000
    assert spec.router_storage == "affine-q4-g32"
    assert spec.router_matmul_dtype == "activation_dtype"
    assert spec.router_bytes == 40_140_800
    assert spec.kv_bytes_per_token == 0
    assert spec.fixed_cache_bytes_per_batch == 0
    assert spec.mtp_layer_index is None
    assert spec.mtp_included is True
    assert spec.source_model == "Vontra/Qwen3.8-Flash-Next-MLX-oQ4-MTP"
    assert spec.source_revision == PINNED_REVISION
    assert spec.quant_model == "OpensourceWTF/Qwen3.8-Flash-Next-MTPLX-oQ4-MTP"
    assert spec.quant_revision == (
        "d873778fec4d66ad4cc5bf9785f5f2199f7b72c037abb5823d8bf1da689916f2"
    )
