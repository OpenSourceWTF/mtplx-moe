"""Pinned, MLX-free configuration geometry for Qwen3.8 Flash-Next."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
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


# MLX is imported only after the MLX-free configuration contract above has been
# declared.  The resident loader imports this model module only after the
# construction-time artifact checks have selected the Qwen4 execution lane.
import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_lm.models.gated_delta import gated_delta_kernel


def _config_int(config: object, name: str) -> int:
    value = getattr(config, name)
    if type(value) is not int:
        raise TypeError(f"{name} must have exact type int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _config_float(config: object, name: str) -> float:
    value = getattr(config, name)
    if type(value) is not float:
        raise TypeError(f"{name} must have exact type float")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _config_string(config: object, name: str, expected: str) -> str:
    value = getattr(config, name)
    if type(value) is not str:
        raise TypeError(f"{name} must have exact type str")
    if value != expected:
        raise ValueError(f"{name} must be {expected!r}")
    return value


class _Qwen4ZeroCenteredRMSNorm(nn.Module):
    """Official Qwen4 RMSNorm with checkpoint weights centered at zero."""

    def __init__(self, width: int, *, group_size: int | None, eps: float):
        super().__init__()
        if group_size is not None and width % group_size != 0:
            raise ValueError("RMSNorm width must be divisible by group_size")
        self.group_size = group_size
        self.eps = eps
        self.weight = mx.zeros((width,))

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        normalized = x.astype(mx.float32)
        if self.group_size is not None:
            normalized = normalized.reshape(*normalized.shape[:-1], -1, self.group_size)
        variance = mx.mean(normalized * normalized, axis=-1, keepdims=True)
        normalized = normalized * mx.rsqrt(variance + self.eps)
        if self.group_size is not None:
            normalized = normalized.reshape(*x.shape)
        normalized = normalized * (1.0 + self.weight.astype(mx.float32))
        return normalized.astype(input_dtype)


class _Qwen4GatedRMSNorm(nn.Module):
    """GDN output norm; unlike stream norms its weight is not zero-centered."""

    def __init__(self, width: int, *, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((width,))

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        input_dtype = x.dtype
        normalized = x.astype(mx.float32)
        variance = mx.mean(normalized * normalized, axis=-1, keepdims=True)
        normalized = normalized * mx.rsqrt(variance + self.eps)
        # Match the official ordering: cast the norm back before applying its
        # stored (ordinary, one-centered) weight, then apply the FP32 gate.
        normalized = normalized.astype(input_dtype) * self.weight
        gated = normalized.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))
        return gated.astype(input_dtype)


class Qwen4GatedResidual(nn.Module):
    """Four-stream Qwen4 Hyper-Connection read and write arithmetic."""

    def __init__(self, config: object, use_combine: bool = True):
        super().__init__()
        if type(use_combine) is not bool:
            raise TypeError("use_combine must have exact type bool")
        self.hc_count = _config_int(config, "hc_count")
        self.hidden_size = _config_int(config, "hidden_size")
        lowrank = _config_int(config, "hc_lowrank")
        eps = _config_float(config, "rms_norm_eps")
        hyper_width = self.hc_count * self.hidden_size
        self.hc_norm = _Qwen4ZeroCenteredRMSNorm(
            hyper_width,
            group_size=self.hidden_size,
            eps=eps,
        )
        self.input_mix_weight_down = nn.Linear(hyper_width, lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(lowrank, hyper_width, bias=False)
        if use_combine:
            self.block_inject_weight = nn.Linear(hyper_width, self.hc_count, bias=False)
        self._use_combine = use_combine

    def _mixed_input(self, hyper_input: mx.array) -> tuple[mx.array, mx.array]:
        normalized = self.hc_norm(hyper_input)
        mix_hidden = nn.silu(self.input_mix_weight_down(normalized) / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix_hidden)).reshape(
            *hyper_input.shape[:-1], self.hc_count, self.hidden_size
        )
        streams = normalized.reshape(
            *hyper_input.shape[:-1], self.hc_count, self.hidden_size
        )
        return mx.mean(mix * streams, axis=-2), normalized

    def read(self, hyper_input: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        mixed_input, normalized = self._mixed_input(hyper_input)
        injection = 2.0 * mx.sigmoid(
            self.block_inject_weight(normalized) / self.hc_count
        )
        return mixed_input, hyper_input, injection

    def write(
        self,
        residual: mx.array,
        block_output: mx.array,
        injection: mx.array,
    ) -> mx.array:
        streams = residual.reshape(
            *residual.shape[:-1], self.hc_count, self.hidden_size
        )
        updated = streams + block_output[..., None, :] * injection[..., :, None]
        return updated.reshape(*residual.shape)

    def __call__(self, x: mx.array):
        if self._use_combine:
            return self.read(x)
        mixed_input, _ = self._mixed_input(x)
        return mixed_input


@dataclass
class _Qwen4GDNCacheState:
    conv_state: mx.array | None = None
    recurrent_state: mx.array | None = None
    offset: int = 0


@dataclass(frozen=True)
class Qwen4GDNCacheSnapshot:
    conv_state: mx.array | None
    recurrent_state: mx.array | None
    offset: int


@dataclass(frozen=True)
class Qwen4CacheSnapshot:
    gdn: Mapping[int, Qwen4GDNCacheSnapshot]

    def arrays(self) -> tuple[mx.array, ...]:
        arrays: list[mx.array] = []
        for layer in sorted(self.gdn):
            state = self.gdn[layer]
            if state.conv_state is not None:
                arrays.append(state.conv_state)
            if state.recurrent_state is not None:
                arrays.append(state.recurrent_state)
        return tuple(arrays)

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"mtplx-qwen4-cache-v1\0")
        for layer in sorted(self.gdn):
            state = self.gdn[layer]
            digest.update(struct.pack("<qq", layer, state.offset))
            for value in (state.conv_state, state.recurrent_state):
                if value is None:
                    digest.update(b"none\0")
                    continue
                array = np.asarray(value)
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(b"\0")
                digest.update(struct.pack("<q", array.ndim))
                digest.update(struct.pack(f"<{array.ndim}q", *array.shape))
                digest.update(array.tobytes(order="C"))
        return digest.hexdigest()


def _layer_tuple(name: str, values: object) -> tuple[int, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{name} must have exact type tuple")
    if any(type(layer) is not int or layer < 0 for layer in values):
        raise ValueError(f"{name} must contain non-negative exact ints")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique layer ids")
    return values


class Qwen4Cache:
    """Explicit owner for Qwen4 recurrent state and future QSA state."""

    def __init__(
        self,
        *,
        gdn_layers: tuple[int, ...],
        qsa_layers: tuple[int, ...],
    ):
        gdn_layers = _layer_tuple("gdn_layers", gdn_layers)
        qsa_layers = _layer_tuple("qsa_layers", qsa_layers)
        if set(gdn_layers).intersection(qsa_layers):
            raise ValueError("gdn_layers and qsa_layers must be disjoint")
        self.gdn = {layer: _Qwen4GDNCacheState() for layer in gdn_layers}
        # Task 5 installs the construction-proven QSA owners in this mapping.
        self.qsa: dict[int, object] = {layer: None for layer in qsa_layers}

    @classmethod
    def tiny(
        cls,
        *,
        gdn_layers: tuple[int, ...],
        qsa_layers: tuple[int, ...],
    ) -> Qwen4Cache:
        return cls(gdn_layers=gdn_layers, qsa_layers=qsa_layers)

    def snapshot(self) -> Qwen4CacheSnapshot:
        return Qwen4CacheSnapshot(
            gdn=MappingProxyType(
                {
                    layer: Qwen4GDNCacheSnapshot(
                        conv_state=state.conv_state,
                        recurrent_state=state.recurrent_state,
                        offset=state.offset,
                    )
                    for layer, state in self.gdn.items()
                }
            )
        )

    def restore(self, snapshot: Qwen4CacheSnapshot) -> None:
        for layer, saved in snapshot.gdn.items():
            current = self.gdn[layer]
            current.conv_state = saved.conv_state
            current.recurrent_state = saved.recurrent_state
            current.offset = saved.offset

    def trim(self, count: int, *, snapshot: Qwen4CacheSnapshot) -> None:
        if type(count) is not int or count < 0:
            raise ValueError("trim count must be a non-negative exact int")
        for layer, current in self.gdn.items():
            if current.offset - snapshot.gdn[layer].offset != count:
                raise ValueError(
                    "snapshot does not identify the requested cache suffix"
                )
        self.restore(snapshot)


class Qwen4GatedDeltaNet(nn.Module):
    """Qwen4 Gated DeltaNet with a construction-bound recurrence lane."""

    def __init__(
        self,
        config: object,
        layer_idx: int,
        *,
        recurrent_lane: str | None = None,
    ):
        super().__init__()
        if type(layer_idx) is not int or layer_idx < 0:
            raise ValueError("layer_idx must be a non-negative exact int")
        self.layer_idx = layer_idx
        self.hidden_size = _config_int(config, "hidden_size")
        self.num_k_heads = _config_int(config, "linear_num_key_heads")
        self.num_v_heads = _config_int(config, "linear_num_value_heads")
        self.head_k_dim = _config_int(config, "linear_key_head_dim")
        self.head_v_dim = _config_int(config, "linear_value_head_dim")
        self.conv_kernel_size = _config_int(config, "linear_conv_kernel_dim")
        self.layer_norm_epsilon = _config_float(config, "rms_norm_eps")
        _config_string(config, "hidden_act", "silu")
        _config_string(config, "output_gate_type", "sigmoid")
        _config_string(config, "mamba_ssm_dtype", "float32")
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError("num_value_heads must be divisible by num_key_heads")

        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self._head_repeat = self.num_v_heads // self.num_k_heads

        if recurrent_lane is None:
            recurrent_lane = "metal" if type(config) is ModelArgs else "ops"
        elif type(recurrent_lane) is not str:
            raise TypeError("recurrent_lane must have exact type str or None")
        if recurrent_lane == "metal":
            if self.head_k_dim % 32 != 0:
                raise ValueError(
                    "Metal recurrence requires key head dim divisible by 32"
                )
            if not mx.metal.is_available():
                raise RuntimeError(
                    "Metal recurrence requires an available Metal device"
                )
            self._recurrent = self._metal_recurrence
        elif recurrent_lane == "ops":
            self._recurrent = self._ops_recurrence
        else:
            raise ValueError("recurrent_lane must be 'metal' or 'ops'")

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        self.A_log = mx.log(
            mx.random.uniform(low=0.01, high=16.0, shape=(self.num_v_heads,))
        )
        self.dt_bias = mx.ones((self.num_v_heads,))
        self.norm = _Qwen4GatedRMSNorm(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    @staticmethod
    def _l2_normalize(value: mx.array) -> mx.array:
        return value * mx.rsqrt(mx.sum(value * value, axis=-1, keepdims=True) + 1e-6)

    def _ops_recurrence(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        decay: mx.array,
        beta: mx.array,
        state: mx.array,
        attention_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        """Exact logical [K,V] reference lane used by small correctness fixtures."""

        query = mx.repeat(query, self._head_repeat, axis=2)
        key = mx.repeat(key, self._head_repeat, axis=2)
        outputs: list[mx.array] = []
        for token in range(query.shape[1]):
            old_state = state
            q_token = query[:, token].astype(mx.float32)
            k_token = key[:, token].astype(mx.float32)
            v_token = value[:, token].astype(mx.float32)
            state = decay[:, token, :, None, None] * state
            prediction = mx.sum(state * k_token[..., None], axis=-2)
            state = state + (
                beta[:, token, :, None, None].astype(mx.float32)
                * k_token[..., None]
                * (v_token - prediction)[..., None, :]
            )
            output = mx.sum(state * q_token[..., None], axis=-2)
            if attention_mask is not None:
                mask = attention_mask[:, token, None, None]
                state = mx.where(mask[..., None], state, old_state)
                output = mx.where(mask, output, 0)
            outputs.append(output)
        return mx.stack(outputs, axis=1).astype(query.dtype), state

    @staticmethod
    def _metal_recurrence(
        query: mx.array,
        key: mx.array,
        value: mx.array,
        decay: mx.array,
        beta: mx.array,
        state: mx.array,
        attention_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        """Direct Metal lane; the cache boundary remains logical [K,V]."""

        physical_state = mx.contiguous(mx.swapaxes(state, -1, -2))
        output, physical_state = gated_delta_kernel(
            query,
            key,
            value,
            decay,
            beta,
            physical_state,
            attention_mask,
        )
        logical_state = mx.contiguous(mx.swapaxes(physical_state, -1, -2))
        return output, logical_state

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        cache: Qwen4Cache | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states)
        if attention_mask is not None:
            mixed_qkv = mx.where(attention_mask[..., None], mixed_qkv, 0)

        if cache is None:
            conv_state = mx.zeros(
                (batch_size, self.conv_kernel_size - 1, self.conv_dim),
                dtype=hidden_states.dtype,
            )
            recurrent_state = mx.zeros(
                (
                    batch_size,
                    self.num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                ),
                dtype=mx.float32,
            )
            cache_state = None
        else:
            cache_state = cache.gdn[self.layer_idx]
            conv_state = cache_state.conv_state
            recurrent_state = cache_state.recurrent_state
            if conv_state is None:
                conv_state = mx.zeros(
                    (batch_size, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=hidden_states.dtype,
                )
                recurrent_state = mx.zeros(
                    (
                        batch_size,
                        self.num_v_heads,
                        self.head_k_dim,
                        self.head_v_dim,
                    ),
                    dtype=mx.float32,
                )

        conv_input = mx.concatenate((conv_state, mixed_qkv), axis=1)
        conv_out = nn.silu(self.conv1d(conv_input))
        next_conv_state = mx.contiguous(
            conv_input[:, -(self.conv_kernel_size - 1) :, :]
        )

        query, key, value = mx.split(
            conv_out,
            (self.key_dim, 2 * self.key_dim),
            axis=-1,
        )
        query = query.reshape(
            batch_size,
            sequence_length,
            self.num_k_heads,
            self.head_k_dim,
        )
        key = key.reshape(
            batch_size,
            sequence_length,
            self.num_k_heads,
            self.head_k_dim,
        )
        value = value.reshape(
            batch_size,
            sequence_length,
            self.num_v_heads,
            self.head_v_dim,
        )
        query = self._l2_normalize(query) * (self.head_k_dim**-0.5)
        key = self._l2_normalize(key)

        beta = mx.sigmoid(self.in_proj_b(hidden_states))
        a = self.in_proj_a(hidden_states).astype(mx.float32)
        decay = mx.exp(
            -mx.exp(self.A_log.astype(mx.float32))
            * nn.softplus(a + self.dt_bias.astype(mx.float32))
        )
        recurrent_out, state = self._recurrent(
            query,
            key,
            value,
            decay,
            beta,
            recurrent_state,
            attention_mask,
        )
        recurrent_out = recurrent_out.astype(hidden_states.dtype)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.num_v_heads,
            self.head_v_dim,
        )
        recurrent_out = self.norm(recurrent_out, z)

        if cache_state is not None:
            cache_state.conv_state = next_conv_state
            cache_state.recurrent_state = state
            cache_state.offset += sequence_length

        return self.out_proj(
            recurrent_out.reshape(batch_size, sequence_length, self.value_dim)
        )
