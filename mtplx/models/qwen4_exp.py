"""Pinned, MLX-free configuration geometry for Qwen3.8 Flash-Next."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

_T = TypeVar("_T")
_SOURCE_FULL_ATTENTION = "full_attention"
_QWEN_SPARSE_ATTENTION = "qwen_sparse_attention"
_PINNED_ERROR = "pinned Qwen3.8 Flash-Next geometry"
_PINNED_LAYER_TYPES = tuple(
    "linear_attention" if layer % 4 != 3 else _QWEN_SPARSE_ATTENTION
    for layer in range(48)
)


def _exact_type(name: str, value: object, expected_type: type[_T]) -> _T:
    if type(value) is not expected_type:
        raise TypeError(
            f"{name} must have exact type {expected_type.__name__} for {_PINNED_ERROR}"
        )
    return value


def _require_pinned(name: str, value: object, expected: object) -> None:
    _exact_type(name, value, type(expected))
    if value != expected:
        raise ValueError(f"{name} does not match {_PINNED_ERROR}")


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{name} must have an exact string-keyed mapping shape for {_PINNED_ERROR}"
        )
    if any(type(key) is not str for key in value):
        raise TypeError(f"{name} keys must have exact type str for {_PINNED_ERROR}")
    return value


def _sequence(value: object, *, name: str) -> list[object] | tuple[object, ...]:
    if type(value) not in (list, tuple):
        raise TypeError(
            f"{name} must have exact sequence type list or tuple for {_PINNED_ERROR}"
        )
    return value


def _string_tuple(
    value: object, *, name: str, normalize_attention: bool = False
) -> tuple[str, ...]:
    values = _sequence(value, name=name)
    for index, item in enumerate(values):
        _exact_type(f"{name}[{index}]", item, str)
    if normalize_attention:
        return tuple(
            _QWEN_SPARSE_ATTENTION if item == _SOURCE_FULL_ATTENTION else item
            for item in values
        )
    return tuple(values)


def _int_tuple(value: object, *, name: str) -> tuple[int, ...]:
    values = _sequence(value, name=name)
    for index, item in enumerate(values):
        _exact_type(f"{name}[{index}]", item, int)
    return tuple(values)


@dataclass(frozen=True)
class MTPArgs:
    hybrid: bool
    layer_types: tuple[str, ...]
    mtp_use_hidden_state_from_layer: int | None
    num_hidden_layers: int
    rope_theta: int

    def __post_init__(self) -> None:
        normalized_layers = _string_tuple(
            self.layer_types,
            name="mtp.layer_types",
            normalize_attention=True,
        )
        object.__setattr__(self, "layer_types", normalized_layers)
        _require_pinned("mtp.hybrid", self.hybrid, True)
        _require_pinned("mtp.layer_types", self.layer_types, (_QWEN_SPARSE_ATTENTION,))
        _require_pinned(
            "mtp.mtp_use_hidden_state_from_layer",
            self.mtp_use_hidden_state_from_layer,
            None,
        )
        _require_pinned("mtp.num_hidden_layers", self.num_hidden_layers, 1)
        _require_pinned("mtp.rope_theta", self.rope_theta, 10_000_000)

    @classmethod
    def from_dict(cls, params: object) -> MTPArgs:
        config = _mapping(params, name="mtp")
        try:
            return cls(
                hybrid=_exact_type("mtp.hybrid", config["hybrid"], bool),
                layer_types=_string_tuple(
                    config["layer_types"],
                    name="mtp.layer_types",
                    normalize_attention=True,
                ),
                mtp_use_hidden_state_from_layer=config[
                    "mtp_use_hidden_state_from_layer"
                ],
                num_hidden_layers=_exact_type(
                    "mtp.num_hidden_layers", config["num_hidden_layers"], int
                ),
                rope_theta=_exact_type("mtp.rope_theta", config["rope_theta"], int),
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
        object.__setattr__(
            self,
            "mrope_section",
            _int_tuple(self.mrope_section, name="rope_parameters.mrope_section"),
        )
        _require_pinned(
            "rope_parameters.mrope_interleaved", self.mrope_interleaved, True
        )
        _require_pinned(
            "rope_parameters.mrope_section", self.mrope_section, (11, 11, 10)
        )
        _require_pinned(
            "rope_parameters.partial_rotary_factor",
            self.partial_rotary_factor,
            0.25,
        )
        _require_pinned("rope_parameters.rope_theta", self.rope_theta, 10_000_000)
        _require_pinned("rope_parameters.rope_type", self.rope_type, "default")

    @classmethod
    def from_dict(cls, params: object) -> RopeParameters:
        config = _mapping(params, name="rope_parameters")
        try:
            return cls(
                mrope_interleaved=_exact_type(
                    "rope_parameters.mrope_interleaved",
                    config["mrope_interleaved"],
                    bool,
                ),
                mrope_section=_int_tuple(
                    config["mrope_section"], name="rope_parameters.mrope_section"
                ),
                partial_rotary_factor=_exact_type(
                    "rope_parameters.partial_rotary_factor",
                    config["partial_rotary_factor"],
                    float,
                ),
                rope_theta=_exact_type(
                    "rope_parameters.rope_theta", config["rope_theta"], int
                ),
                rope_type=_exact_type(
                    "rope_parameters.rope_type", config["rope_type"], str
                ),
            )
        except KeyError as exc:
            raise ValueError(
                f"rope_parameters is missing pinned field {exc.args[0]!r}"
            ) from exc


_PINNED_MODEL_SCALARS: dict[str, object] = {
    "model_type": "qwen4_exp_text",
    "dtype": "bfloat16",
    "hidden_act": "silu",
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "full_attention_interval": 4,
    "attention_bias": False,
    "attention_dropout": 0.0,
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
    "vocab_size": 248_320,
    "eos_token_id": 248_044,
    "rms_norm_eps": 1e-6,
    "output_gate_type": "sigmoid",
    "tie_word_embeddings": False,
    "use_cache": True,
    "max_position_embeddings": 262_144,
    "partial_rotary_factor": 0.25,
}


@dataclass(frozen=True)
class ModelArgs:
    """Construction-time proof of the pinned Qwen3.8 text geometry."""

    model_type: str
    dtype: str
    hidden_act: str
    hidden_size: int
    num_hidden_layers: int
    full_attention_interval: int
    layer_types: tuple[str, ...]
    attention_bias: bool
    attention_dropout: float
    ple_layer_ids: tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
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
    mamba_ssm_dtype: str
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
    rms_norm_eps: float
    output_gate_type: str
    tie_word_embeddings: bool
    use_cache: bool
    max_position_embeddings: int
    partial_rotary_factor: float
    rope_parameters: RopeParameters
    seed: int = 1234
    streamed_layer_ids: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_types",
            _string_tuple(
                self.layer_types, name="layer_types", normalize_attention=True
            ),
        )
        object.__setattr__(
            self,
            "ple_layer_ids",
            _int_tuple(self.ple_layer_ids, name="ple_layer_ids"),
        )
        for name, expected in _PINNED_MODEL_SCALARS.items():
            _require_pinned(name, getattr(self, name), expected)
        _require_pinned("layer_types", self.layer_types, _PINNED_LAYER_TYPES)
        _require_pinned("ple_layer_ids", self.ple_layer_ids, (2,))
        _exact_type("mtp", self.mtp, MTPArgs)
        _exact_type("rope_parameters", self.rope_parameters, RopeParameters)
        _exact_type("seed", self.seed, int)
        object.__setattr__(self, "streamed_layer_ids", tuple(range(49)))

    @classmethod
    def from_dict(cls, params: Mapping[str, Any]) -> ModelArgs:
        """Accept a full conditional-generation config or its text config."""

        root = _mapping(params, name="config")
        if "text_config" in root:
            if root.get("model_type") != "qwen4_exp":
                raise ValueError("root model_type must be qwen4_exp")
            config = _mapping(root["text_config"], name="text_config")
        else:
            config = root
        try:
            scalars = {
                name: _exact_type(name, config[name], type(expected))
                for name, expected in _PINNED_MODEL_SCALARS.items()
            }
            return cls(
                **scalars,
                layer_types=_string_tuple(
                    config["layer_types"],
                    name="layer_types",
                    normalize_attention=True,
                ),
                ple_layer_ids=_int_tuple(config["ple_layer_ids"], name="ple_layer_ids"),
                mtp=MTPArgs.from_dict(config["mtp"]),
                rope_parameters=RopeParameters.from_dict(config["rope_parameters"]),
                seed=_exact_type(
                    "seed", config.get("seed", root.get("seed", 1234)), int
                ),
            )
        except KeyError as exc:
            raise ValueError(f"config is missing pinned field {exc.args[0]!r}") from exc


_RUNTIME_EXPORTS = frozenset(
    {
        "Qwen4Cache",
        "Qwen4GatedDeltaNet",
        "Qwen4GatedResidual",
        "sanitize_qwen4_weights",
    }
)


def __getattr__(name: str):
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import qwen4_exp_runtime

    value = getattr(qwen4_exp_runtime, name)
    globals()[name] = value
    return value
