"""Pinned, MLX-free configuration geometry for Qwen3.8 Flash-Next."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_SOURCE_FULL_ATTENTION = "full_attention"
_QWEN_SPARSE_ATTENTION = "qwen_sparse_attention"
_PINNED_LAYER_TYPES = tuple(
    "linear_attention" if layer % 4 != 3 else _QWEN_SPARSE_ATTENTION
    for layer in range(48)
)


def _normalize_layer_types(values: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    return tuple(
        _QWEN_SPARSE_ATTENTION if value == _SOURCE_FULL_ATTENTION else value
        for value in values
    )


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True)
class MTPArgs:
    hybrid: bool
    layer_types: tuple[str, ...]
    mtp_use_hidden_state_from_layer: int | None
    num_hidden_layers: int
    rope_theta: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_types",
            _normalize_layer_types(self.layer_types, name="mtp.layer_types"),
        )
        if (
            self.hybrid is not True
            or self.layer_types != (_QWEN_SPARSE_ATTENTION,)
            or self.mtp_use_hidden_state_from_layer is not None
            or self.num_hidden_layers != 1
            or self.rope_theta != 10_000_000
        ):
            raise ValueError(
                "configuration does not match pinned Qwen3.8 Flash-Next geometry"
            )

    @classmethod
    def from_dict(cls, params: object) -> MTPArgs:
        config = _mapping(params, name="mtp")
        try:
            return cls(
                hybrid=config["hybrid"],
                layer_types=config["layer_types"],
                mtp_use_hidden_state_from_layer=config[
                    "mtp_use_hidden_state_from_layer"
                ],
                num_hidden_layers=config["num_hidden_layers"],
                rope_theta=config["rope_theta"],
            )
        except KeyError as exc:
            raise ValueError(f"mtp is missing pinned field {exc.args[0]!r}") from exc


@dataclass(frozen=True)
class RopeParameters:
    mrope_interleaved: bool
    mrope_section: tuple[int, ...]
    partial_rotary_factor: float
    rope_theta: int
    rope_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "mrope_section", tuple(self.mrope_section))
        if (
            self.mrope_interleaved is not True
            or self.mrope_section != (11, 11, 10)
            or self.partial_rotary_factor != 0.25
            or self.rope_theta != 10_000_000
            or self.rope_type != "default"
        ):
            raise ValueError(
                "configuration does not match pinned Qwen3.8 Flash-Next geometry"
            )

    @classmethod
    def from_dict(cls, params: object) -> RopeParameters:
        config = _mapping(params, name="rope_parameters")
        try:
            return cls(
                mrope_interleaved=config["mrope_interleaved"],
                mrope_section=config["mrope_section"],
                partial_rotary_factor=config["partial_rotary_factor"],
                rope_theta=config["rope_theta"],
                rope_type=config["rope_type"],
            )
        except KeyError as exc:
            raise ValueError(
                f"rope_parameters is missing pinned field {exc.args[0]!r}"
            ) from exc


@dataclass(frozen=True)
class ModelArgs:
    """Construction-time proof of the pinned Qwen3.8 text geometry."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    ple_layer_ids: tuple[int, ...]
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    hc_count: int
    hc_lowrank: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    split_ngram_parts: int
    mtp_num_hidden_layers: int
    mtp_use_dedicated_embeddings: bool
    mtp: MTPArgs
    vocab_size: int
    eos_token_id: int
    max_position_embeddings: int
    partial_rotary_factor: float
    rope_parameters: RopeParameters
    seed: int = 1234

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_types",
            _normalize_layer_types(self.layer_types, name="layer_types"),
        )
        object.__setattr__(self, "ple_layer_ids", tuple(self.ple_layer_ids))
        pinned = {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "layer_types": _PINNED_LAYER_TYPES,
            "ple_layer_ids": (2,),
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
            "vocab_size": 248_320,
            "eos_token_id": 248_044,
            "max_position_embeddings": 262_144,
            "partial_rotary_factor": 0.25,
        }
        if any(getattr(self, name) != expected for name, expected in pinned.items()):
            raise ValueError(
                "configuration does not match pinned Qwen3.8 Flash-Next geometry"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

    @classmethod
    def from_dict(cls, params: Mapping[str, Any]) -> ModelArgs:
        """Accept a full conditional-generation config or its text config."""

        root = _mapping(params, name="config")
        if "text_config" in root:
            if root.get("model_type") not in (None, "qwen4_exp"):
                raise ValueError("root model_type must be qwen4_exp")
            config = _mapping(root["text_config"], name="text_config")
        else:
            config = root
        try:
            return cls(
                model_type=config["model_type"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_hidden_layers"],
                layer_types=_normalize_layer_types(
                    config["layer_types"], name="layer_types"
                ),
                ple_layer_ids=tuple(config["ple_layer_ids"]),
                num_experts=config["num_experts"],
                num_experts_per_tok=config["num_experts_per_tok"],
                moe_intermediate_size=config["moe_intermediate_size"],
                shared_expert_intermediate_size=config[
                    "shared_expert_intermediate_size"
                ],
                hc_count=config["hc_count"],
                hc_lowrank=config["hc_lowrank"],
                linear_num_key_heads=config["linear_num_key_heads"],
                linear_num_value_heads=config["linear_num_value_heads"],
                linear_key_head_dim=config["linear_key_head_dim"],
                linear_value_head_dim=config["linear_value_head_dim"],
                linear_conv_kernel_dim=config["linear_conv_kernel_dim"],
                num_attention_heads=config["num_attention_heads"],
                num_key_value_heads=config["num_key_value_heads"],
                head_dim=config["head_dim"],
                indexer_n_heads=config["indexer_n_heads"],
                indexer_kv_heads=config["indexer_kv_heads"],
                indexer_head_dim=config["indexer_head_dim"],
                indexer_budget=config["indexer_budget"],
                indexer_compress_ratio=config["indexer_compress_ratio"],
                ngram_size=config["ngram_size"],
                heads_per_ngram=config["heads_per_ngram"],
                ngram_vocab_size_base=config["ngram_vocab_size_base"],
                make_ngram_vocab_size_divisible_by=config[
                    "make_ngram_vocab_size_divisible_by"
                ],
                split_ngram_parts=config["split_ngram_parts"],
                mtp_num_hidden_layers=config["mtp_num_hidden_layers"],
                mtp_use_dedicated_embeddings=config["mtp_use_dedicated_embeddings"],
                mtp=MTPArgs.from_dict(config["mtp"]),
                vocab_size=config["vocab_size"],
                eos_token_id=config["eos_token_id"],
                max_position_embeddings=config["max_position_embeddings"],
                partial_rotary_factor=config["partial_rotary_factor"],
                rope_parameters=RopeParameters.from_dict(config["rope_parameters"]),
                seed=config.get("seed", root.get("seed", 1234)),
            )
        except KeyError as exc:
            raise ValueError(f"config is missing pinned field {exc.args[0]!r}") from exc
