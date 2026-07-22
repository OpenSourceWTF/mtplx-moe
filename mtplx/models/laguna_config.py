"""Pure construction-time identity checks for the supported Laguna checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


LAGUNA_S_2_1_REPO_ID = "pipenetwork/Laguna-S-2.1-MLX-4bit"
LAGUNA_S_2_1_REVISION = "5544297f819d50330bc3616dd15cbc7edb598b2f"
LAGUNA_S_2_1_WEIGHT_BYTES = 66_147_556_864
LAGUNA_S_2_1_REPO_BYTES = 66_155_348_620
LAGUNA_S_2_1_DEFAULT_CONTEXT = 32_768
LAGUNA_S_2_1_FULL_KV_BYTES_PER_TOKEN = 49_152
LAGUNA_S_2_1_ROTATING_KV_BYTES = 75_497_472


def laguna_s_2_1_required_resident_bytes(context_tokens: int) -> int:
    return (
        LAGUNA_S_2_1_WEIGHT_BYTES
        + 8 * 1024**3
        + LAGUNA_S_2_1_ROTATING_KV_BYTES
        + max(1, int(context_tokens)) * LAGUNA_S_2_1_FULL_KV_BYTES_PER_TOKEN
    )


LAGUNA_S_2_1_MIN_RESIDENT_BYTES = laguna_s_2_1_required_resident_bytes(
    LAGUNA_S_2_1_DEFAULT_CONTEXT
)
LAGUNA_S_2_1_WEIGHT_SHARDS = tuple(
    f"model-{index:05d}-of-00013.safetensors" for index in range(1, 14)
)
LAGUNA_S_2_1_SHARD_SIZES = {
    "model-00001-of-00013.safetensors": 5_345_218_229,
    "model-00002-of-00013.safetensors": 5_189_210_514,
    "model-00003-of-00013.safetensors": 5_138_878_625,
    "model-00004-of-00013.safetensors": 5_107_862_084,
    "model-00005-of-00013.safetensors": 5_138_878_685,
    "model-00006-of-00013.safetensors": 5_138_878_659,
    "model-00007-of-00013.safetensors": 5_097_203_847,
    "model-00008-of-00013.safetensors": 5_138_878_795,
    "model-00009-of-00013.safetensors": 5_138_878_605,
    "model-00010-of-00013.safetensors": 5_097_203_801,
    "model-00011-of-00013.safetensors": 5_138_878_687,
    "model-00012-of-00013.safetensors": 5_138_878_641,
    "model-00013-of-00013.safetensors": 4_338_941_119,
}
LAGUNA_S_2_1_SIDECAR_SHA256 = {
    "config.json": "22ba23138b98e15d5452b5dc14cb88a96797bcd07fceab4fc84dcb1068c18d60",
    "generation_config.json": "2deeac08584c9177028e108a994e37dffd06acf61ca429dc064f76fee52e2bea",
    "chat_template.jinja": "2d3c724b3c2e9eb71fe9ccc5423ff268a370a8bfa89e9238b6de14fe000825c8",
    "model.safetensors.index.json": "f5926cd1cb7c1b5a928ec8d10e2691848785f907a0837f03c959d1bd01757c8d",
    "tokenizer.json": "ff04405d2d1e1b6c77a8be25f0fce9371003a558b055c23248d9e8ca1d956d92",
    "tokenizer_config.json": "fb4815e0e871cd4cb1cffa77722ce798df97db11da0481489b8da4d76142596e",
}
LAGUNA_S_2_1_REQUIRED_FILES = frozenset(
    (
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        *LAGUNA_S_2_1_WEIGHT_SHARDS,
    )
)


def laguna_s_2_1_artifact_integrity_errors(model_path: Path | str) -> tuple[str, ...]:
    """Return pinned-file mismatches before the 61.6 GiB weight load boundary."""

    root = Path(model_path)
    errors: list[str] = []
    for name, expected_size in LAGUNA_S_2_1_SHARD_SIZES.items():
        path = root / name
        try:
            if not path.is_file() or path.stat().st_size != expected_size:
                errors.append(name)
        except OSError:
            errors.append(name)
    for name, expected_sha256 in LAGUNA_S_2_1_SIDECAR_SHA256.items():
        path = root / name
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                errors.append(name)
        except OSError:
            errors.append(name)
    return tuple(sorted(set(errors)))


_LAYER_TYPES = (
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
) * 12
_ATTENTION_HEADS = tuple(
    48 if layer_type == "full_attention" else 72 for layer_type in _LAYER_TYPES
)
_MLP_LAYER_TYPES = ("dense",) + ("sparse",) * 47
_GATING_TYPES = ("per_head",) * 48
_ROUTER_QUANTIZATION_KEYS = tuple(
    f"model.layers.{layer}.mlp.gate" for layer in range(1, 48)
)
_QUANTIZATION_KEYS = frozenset(
    ("bits", "group_size", "mode", *_ROUTER_QUANTIZATION_KEYS)
)


def _is_exact_quantization_map(value: Any) -> bool:
    if not isinstance(value, dict) or frozenset(value) != _QUANTIZATION_KEYS:
        return False
    if (
        value.get("bits") != 4
        or value.get("group_size") != 64
        or value.get("mode") != "affine"
    ):
        return False
    return all(
        value.get(key) == {"bits": 8, "group_size": 64}
        for key in _ROUTER_QUANTIZATION_KEYS
    )


def is_laguna_s_2_1_mlx_4bit_config(config: dict[str, Any]) -> bool:
    """Match the exact arithmetic and storage geometry of the supported model."""

    if not isinstance(config, dict) or "model_file" in config:
        return False
    try:
        architectures = config.get("architectures") or []
        quantization = config.get("quantization")
        quantization_config = config.get("quantization_config")
        rope = config.get("rope_parameters") or {}
        full_rope = rope.get("full_attention") or {}
        sliding_rope = rope.get("sliding_attention") or {}
        return bool(
            architectures == ["LagunaForCausalLM"]
            and str(config.get("model_type") or "").lower() == "laguna"
            and int(config.get("hidden_size") or 0) == 3072
            and int(config.get("num_hidden_layers") or 0) == 48
            and int(config.get("intermediate_size") or 0) == 12288
            and int(config.get("num_attention_heads") or 0) == 48
            and tuple(config.get("num_attention_heads_per_layer") or ())
            == _ATTENTION_HEADS
            and config.get("attention_bias") is False
            and float(config.get("attention_dropout") or 0.0) == 0.0
            and int(config.get("num_key_value_heads") or 0) == 8
            and int(config.get("head_dim") or 0) == 128
            and int(config.get("vocab_size") or 0) == 100352
            and int(config.get("bos_token_id") or 0) == 2
            and config.get("eos_token_id") == [2, 24]
            and int(config.get("pad_token_id") or 0) == 9
            and float(config.get("rms_norm_eps") or 0.0) == 1e-6
            and int(config.get("num_experts") or 0) == 256
            and int(config.get("num_experts_per_tok") or 0) == 10
            and int(config.get("moe_intermediate_size") or 0) == 1024
            and int(config.get("shared_expert_intermediate_size") or 0) == 1024
            and int(config.get("decoder_sparse_step") or 0) == 1
            and config.get("norm_topk_prob") is True
            and float(config.get("moe_routed_scaling_factor") or 0.0) == 2.5
            and float(config.get("moe_router_logit_softcapping") or 0.0) == 0.0
            and config.get("moe_apply_router_weight_on_input") is False
            and float(config.get("router_aux_loss_coef") or 0.0) == 0.0
            and config.get("mlp_only_layers") == [0]
            and config.get("gating") == "per-head"
            and tuple(config.get("gating_types") or ()) == _GATING_TYPES
            and int(config.get("sliding_window") or 0) == 512
            and tuple(config.get("layer_types") or ()) == _LAYER_TYPES
            and tuple(config.get("mlp_layer_types") or ()) == _MLP_LAYER_TYPES
            and full_rope.get("rope_type") == "yarn"
            and float(full_rope.get("rope_theta") or 0.0) == 500_000.0
            and float(full_rope.get("factor") or 0.0) == 128.0
            and int(full_rope.get("original_max_position_embeddings") or 0)
            == 8192
            and float(full_rope.get("beta_slow") or 0.0) == 1.0
            and float(full_rope.get("beta_fast") or 0.0) == 32.0
            and float(full_rope.get("attention_factor") or 0.0)
            == 1.4852030263919618
            and float(full_rope.get("partial_rotary_factor") or 0.0) == 0.5
            and sliding_rope.get("rope_type") == "default"
            and float(sliding_rope.get("rope_theta") or 0.0) == 10_000.0
            and float(sliding_rope.get("partial_rotary_factor") or 0.0) == 1.0
            and int(config.get("max_position_embeddings") or 0) == 1_048_576
            and config.get("tie_word_embeddings") is False
            and config.get("torch_dtype") == "bfloat16"
            and config.get("use_cache") is True
            and quantization == quantization_config
            and _is_exact_quantization_map(quantization)
        )
    except (AttributeError, TypeError, ValueError):
        return False
