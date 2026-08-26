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

PINNED_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
CACHED_CONFIG = Path(
    "/Users/davidtai/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.8-Flash-Next/snapshots/"
    f"{PINNED_REVISION}/config.json"
)


def _text_config() -> dict[str, object]:
    layer_types = [
        "linear_attention" if layer % 4 != 3 else "full_attention"
        for layer in range(48)
    ]
    return {
        "model_type": "qwen4_exp_text",
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "layer_types": layer_types,
        "ple_layer_ids": [2],
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
    assert args.hidden_size == 2560
    assert args.num_hidden_layers == 48
    assert args.layer_types == expected_layers
    assert args.ple_layer_ids == (2,)
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
    assert args.max_position_embeddings == 262_144
    assert args.partial_rotary_factor == 0.25
    assert args.rope_parameters.mrope_interleaved is True
    assert args.rope_parameters.mrope_section == (11, 11, 10)
    assert args.rope_parameters.partial_rotary_factor == 0.25
    assert args.rope_parameters.rope_theta == 10_000_000
    assert args.rope_parameters.rope_type == "default"
    assert args.seed == 1234


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


def test_qwen38_flash_next_q4_streaming_geometry_is_exact() -> None:
    spec = get_model_spec("qwen38-flash-next-q4")

    assert spec is QWEN38_FLASH_NEXT_Q4
    assert MODEL_SPECS[spec.key] is spec
    assert spec.total_tensor_bytes == 182_738_190_328
    assert spec.total_layers == 49
    assert spec.routed_layer_indices == tuple(range(49))
    assert spec.expert_count == 512
    assert spec.top_k == 10
    assert spec.hidden_size == 2560
    assert spec.expert_hidden_size == 640
    assert spec.quant_bits == 4
    assert spec.quant_group_size == 64
    assert spec.quant_parameter_bytes == 2
    assert spec.expert_codec == "affine"
    assert spec.expert_activation == "swiglu"
    assert spec.expert_record_bytes == 2_764_800
    assert spec.routed_expert_bytes == 69_363_302_400
    assert spec.router_storage == "bfloat16"
    assert spec.router_matmul_dtype == "float32"
    assert spec.router_bytes == 128_450_560
    assert spec.kv_bytes_per_token == 0
    assert spec.fixed_cache_bytes_per_batch == 0
    assert spec.mtp_layer_index is None
    assert spec.mtp_included is True
    assert spec.source_model == "Qwen/Qwen3.8-Flash-Next"
    assert spec.source_revision == PINNED_REVISION
    assert spec.quant_model == "OpensourceWTF/Qwen3.8-Flash-Next-MTPLX-Q4"
    assert spec.quant_revision == (
        "09dbf9fa47543c707eb50e6d27f929c989be54eb14cd386f0d0bb3c98564f2c6"
    )
