"""Pure construction-time identity checks for the supported Laguna checkpoint.

Exactly one Laguna artifact is admitted through the ``laguna_ar`` backend, by
exact repo id + pinned revision + artifact identity:

    mlx-community/Laguna-S-2.1-oQ4e @ 8e3f5cad513746264940c1c4195de48d7ea345a5

It is the mixed-precision imatrix export of Laguna-S-2.1 (LagunaForCausalLM, 48
layers, hidden 3072, 48/8 heads, head_dim 128, 256 experts top-10 + shared
expert, layer-0 dense, sliding (full,swa,swa,swa)x12 window 512, vocab 100352).
Every other Laguna variant — including the earlier uniform-4bit build — stays
blocked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


LAGUNA_S_2_1_REPO_ID = "mlx-community/Laguna-S-2.1-oQ4e"
LAGUNA_S_2_1_REVISION = "8e3f5cad513746264940c1c4195de48d7ea345a5"
# Tensor-storage total from the index metadata (the mixed-precision weights).
LAGUNA_S_2_1_WEIGHT_BYTES = 64_122_027_323
# Every downloadable file in the pinned snapshot.
LAGUNA_S_2_1_REPO_BYTES = 64_129_728_868
LAGUNA_S_2_1_DEFAULT_CONTEXT = 32_768
LAGUNA_S_2_1_FULL_KV_BYTES_PER_TOKEN = 49_152
LAGUNA_S_2_1_ROTATING_KV_BYTES = 75_497_472
# Every oQ4e tensor is exported under a ``language_model.`` wrapper; the runtime
# strips it (weights and the per-path quantization map alike) so the vendored
# module tree — which has no such wrapper — matches by path.
LAGUNA_S_2_1_WEIGHT_NAME_PREFIX = "language_model."


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
    "model-00001-of-00013.safetensors": 5_082_945_345,
    "model-00002-of-00013.safetensors": 5_133_833_178,
    "model-00003-of-00013.safetensors": 5_133_833_188,
    "model-00004-of-00013.safetensors": 5_133_833_222,
    "model-00005-of-00013.safetensors": 5_133_833_218,
    "model-00006-of-00013.safetensors": 5_133_833_226,
    "model-00007-of-00013.safetensors": 5_133_833_236,
    "model-00008-of-00013.safetensors": 5_133_833_208,
    "model-00009-of-00013.safetensors": 5_133_833_224,
    "model-00010-of-00013.safetensors": 5_133_833_218,
    "model-00011-of-00013.safetensors": 5_133_833_210,
    "model-00012-of-00013.safetensors": 5_133_833_230,
    "model-00013-of-00013.safetensors": 2_566_916_620,
}
LAGUNA_S_2_1_SIDECAR_SHA256 = {
    "config.json": "de530f3a85f0dbef0b22c6e7caff51d21b6b75d1f04024b07947d103e22a630c",
    "generation_config.json": "2deeac08584c9177028e108a994e37dffd06acf61ca429dc064f76fee52e2bea",
    "chat_template.jinja": "2d3c724b3c2e9eb71fe9ccc5423ff268a370a8bfa89e9238b6de14fe000825c8",
    "model.safetensors.index.json": "45709bf61be0398b4b34ed68845c80f8d2bab75f1f16d19e63c95e15571f0cc2",
    "tokenizer.json": "807c53a95141e77c14e45f68c51db3f84d2ea6b555a6ea832bc99c88dae6a279",
    "tokenizer_config.json": "ce5c24f821c92f73f1bf6d4d6a474636f9fb5ca1fabbbed149a8f466ccd18b56",
    "special_tokens_map.json": "70cd3459fde61761e9440751a590e89a108c09b1803cc7727f5ad1ed1ea6122b",
}
LAGUNA_S_2_1_REQUIRED_FILES = frozenset(
    (
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        *LAGUNA_S_2_1_WEIGHT_SHARDS,
    )
)


def laguna_s_2_1_artifact_integrity_errors(model_path: Path | str) -> tuple[str, ...]:
    """Return pinned-file mismatches before the 59.7 GiB weight load boundary."""

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

# Per-layer self-attention weight precision of the oQ4e imatrix bank, in
# (q,k,v,o,g)_proj order for layers 0..47. Every projection is 5- or 8-bit at
# group_size 64; layer 33 (``55585``) is the lone non-uniform row — its o_proj
# is promoted to 8-bit. This table plus the regular blocks below reproduce the
# checkpoint's exact quantization dict.
_OQ4E_ATTENTION_BITS = (
    "55555", "55555", "88888", "55555", "55555", "55555", "55555", "55555",
    "88888", "55555", "55555", "55555", "88888", "55555", "55555", "55555",
    "88888", "88888", "88888", "88888", "88888", "88888", "88888", "88888",
    "88888", "88888", "88888", "88888", "88888", "88888", "88888", "55555",
    "88888", "55585", "55555", "88888", "88888", "55555", "55555", "88888",
    "88888", "55555", "55555", "88888", "88888", "88888", "88888", "88888",
)


def _build_laguna_s_2_1_quantization() -> dict[str, Any]:
    """Reconstruct the pinned config['quantization'] map from its exact shape.

    Global default is 4-bit/gs128 (the MoE experts fall here); the MoE routers
    (``mlp.gate``) carry no entry and stay unquantized (BF16). Keys keep the
    ``language_model.`` export prefix so this matches the on-disk config dict
    exactly; the runtime strips the prefix before handing it to mlx-lm.
    """

    quantization: dict[str, Any] = {"group_size": 128, "bits": 4, "mode": "affine"}
    prefix = LAGUNA_S_2_1_WEIGHT_NAME_PREFIX
    quantization[f"{prefix}lm_head"] = {"bits": 8, "group_size": 64, "mode": "affine"}
    quantization[f"{prefix}model.embed_tokens"] = {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    }
    # Layer 0 is the lone dense MLP block (5/5/6-bit at gs64).
    quantization[f"{prefix}model.layers.0.mlp.gate_proj"] = {
        "bits": 5,
        "group_size": 64,
        "mode": "affine",
    }
    quantization[f"{prefix}model.layers.0.mlp.up_proj"] = {
        "bits": 5,
        "group_size": 64,
        "mode": "affine",
    }
    quantization[f"{prefix}model.layers.0.mlp.down_proj"] = {
        "bits": 6,
        "group_size": 64,
        "mode": "affine",
    }
    # Shared experts of the 47 sparse layers are uniform 8-bit/gs128.
    for layer in range(1, 48):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            quantization[
                f"{prefix}model.layers.{layer}.mlp.shared_expert.{projection}"
            ] = {"bits": 8, "group_size": 128, "mode": "affine"}
    # Self-attention projections follow the per-layer imatrix table.
    projections = ("q_proj", "k_proj", "v_proj", "o_proj", "g_proj")
    for layer, row in enumerate(_OQ4E_ATTENTION_BITS):
        for projection, bit in zip(projections, row):
            quantization[
                f"{prefix}model.layers.{layer}.self_attn.{projection}"
            ] = {"bits": int(bit), "group_size": 64, "mode": "affine"}
    return quantization


LAGUNA_S_2_1_QUANTIZATION = _build_laguna_s_2_1_quantization()


def _is_exact_quantization_map(value: Any) -> bool:
    """Match the oQ4e mixed-precision imatrix map by exact identity.

    Any change to a bit width, group size, key set, or the unquantized-router
    shape flips this to ``False``; the pinned scheme is the only admitted one.
    """

    return isinstance(value, dict) and value == LAGUNA_S_2_1_QUANTIZATION


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


def laguna_module_quantization(config: dict[str, Any]) -> dict[str, Any] | None:
    """Per-module quantization map keyed to the sanitized module tree.

    The pinned checkpoint ships its per-path quantization dict under the
    ``language_model.`` export prefix, which the vendored module tree does not
    carry. Strip the prefix so mlx-lm's config-driven quantizer matches each
    Linear/Embedding by path; the (unlisted) MoE routers stay unquantized.
    Returns ``None`` for a non-admitted config.
    """

    if not is_laguna_s_2_1_mlx_4bit_config(config):
        return None
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        return None
    prefix = LAGUNA_S_2_1_WEIGHT_NAME_PREFIX
    stripped: dict[str, Any] = {}
    for key, value in quantization.items():
        if key in ("group_size", "bits", "mode"):
            stripped[key] = value
        elif isinstance(key, str) and key.startswith(prefix):
            stripped[key[len(prefix) :]] = value
        else:
            stripped[key] = value
    return stripped
