"""MLX runtime for the pinned Qwen3.8 Flash-Next configuration."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_lm.models.gated_delta import gated_delta_kernel

from .qwen4_exp import _QWEN_SPARSE_ATTENTION, ModelArgs


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


def sanitize_qwen4_weights(weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
    """Install source Qwen4 convolution weights at the sole layout boundary.

    The pinned checkpoint stores each GDN depthwise kernel as [C, 1, K], while
    MLX Conv1d consumes [C, K, 1]. Artifact converters must preserve the source
    layout and must not transpose these tensors; the future root Model.sanitize
    calls this function exactly once before strict loading.
    """

    sanitized: dict[str, mx.array] = {}
    for name, value in weights.items():
        is_gdn_conv = name == "conv1d.weight" or name.endswith(
            ".linear_attn.conv1d.weight"
        )
        if not is_gdn_conv:
            sanitized[name] = value
            continue
        if value.ndim != 3:
            raise ValueError("Qwen4 source conv weight must have rank 3")
        if value.shape[1] != 1:
            if value.shape[2] == 1:
                raise ValueError(
                    "Qwen4 source conv layout was already transposed; "
                    "artifact converter must not transpose it"
                )
            raise ValueError("Qwen4 source conv weight must use [C, 1, K] layout")
        if value.shape[2] <= 1:
            raise ValueError("Qwen4 source conv kernel must contain multiple taps")
        sanitized[name] = mx.transpose(value, (0, 2, 1))
    return sanitized


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
            self._execute = self.read
        else:
            self._execute = self._combine

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
        return self._execute(x)

    def _combine(self, x: mx.array) -> mx.array:
        mixed_input, _ = self._mixed_input(x)
        return mixed_input


@dataclass(frozen=True)
class _Qwen4GDNLayout:
    batch_size: int
    conv_tokens: int
    conv_width: int
    conv_dtype: str
    num_v_heads: int
    head_v_dim: int
    head_k_dim: int


@dataclass(frozen=True)
class _Qwen4GDNCacheState:
    conv_state: mx.array | None = None
    recurrent_state: mx.array | None = None
    offset: int = 0
    layout: _Qwen4GDNLayout | None = None


_CACHE_SCHEMA = "mtplx-qwen4-cache-v2"
_CACHE_CONSTRUCTOR = object()


@dataclass(frozen=True)
class _Qwen4CacheSnapshot:
    _owner: object
    _schema: str
    _gdn: Mapping[int, _Qwen4GDNCacheState]
    _qsa_layers: tuple[int, ...]
    _restore_bound: Callable[[], None]
    _trim_bound: Callable[[int], None]

    @property
    def gdn(self) -> Mapping[int, _Qwen4GDNCacheState]:
        return self._gdn

    def arrays(self) -> tuple[mx.array, ...]:
        arrays: list[mx.array] = []
        for layer in sorted(self._gdn):
            state = self._gdn[layer]
            if state.conv_state is not None:
                arrays.append(state.conv_state)
            if state.recurrent_state is not None:
                arrays.append(state.recurrent_state)
        return tuple(arrays)

    def restore(self) -> None:
        """Apply this snapshot only to the cache owner bound at capture."""

        self._restore_bound()

    def trim(self, count: int) -> None:
        """Restore an owner-bound suffix after its runtime length is checked."""

        self._trim_bound(count)

    def digest(self) -> str:
        """Hash complete raw cache bits for offline diagnostics only.

        This evaluates every array, makes it contiguous, and transfers the full
        cache to the host. It must never be called from model execution or any
        measured hot path. BF16 is viewed as uint16 before NumPy sees it, so no
        unsupported or lossy numerical BF16 conversion occurs.
        """

        digest = hashlib.sha256()
        digest.update(_CACHE_SCHEMA.encode("ascii") + b"\0")
        digest.update(struct.pack("<q", len(self._qsa_layers)))
        if self._qsa_layers:
            digest.update(struct.pack(f"<{len(self._qsa_layers)}q", *self._qsa_layers))
        for layer in sorted(self._gdn):
            state = self._gdn[layer]
            digest.update(struct.pack("<qq", layer, state.offset))
            for value in (state.conv_state, state.recurrent_state):
                if value is None:
                    digest.update(b"none\0")
                    continue
                contiguous = mx.contiguous(value)
                if value.dtype == mx.bfloat16:
                    raw = contiguous.view(mx.uint16)
                elif value.dtype == mx.float32:
                    raw = contiguous.view(mx.uint32)
                elif value.dtype == mx.float16:
                    raw = contiguous.view(mx.uint16)
                else:
                    raw = contiguous.view(mx.uint8)
                mx.eval(raw)
                host = np.asarray(raw)
                digest.update(str(value.dtype).encode("ascii") + b"\0")
                digest.update(struct.pack("<q", value.ndim))
                digest.update(struct.pack(f"<{value.ndim}q", *value.shape))
                digest.update(host.tobytes(order="C"))
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
    """Construction-proven Qwen4 request state with owner-bound rollback."""

    def __init__(
        self,
        *,
        gdn_layers: tuple[int, ...],
        qsa_layers: tuple[int, ...],
        model_args: ModelArgs | None,
        _constructor: object,
    ):
        if _constructor is not _CACHE_CONSTRUCTOR:
            raise TypeError("use Qwen4Cache.from_model_args or Qwen4Cache.tiny")
        gdn_layers = _layer_tuple("gdn_layers", gdn_layers)
        qsa_layers = _layer_tuple("qsa_layers", qsa_layers)
        if set(gdn_layers).intersection(qsa_layers):
            raise ValueError("gdn_layers and qsa_layers must be disjoint")
        self._owner = object()
        self._model_args = model_args
        self._request_installed = False
        self._gdn = {layer: _Qwen4GDNCacheState() for layer in gdn_layers}
        self._qsa: dict[int, object | None] = {layer: None for layer in qsa_layers}

    @classmethod
    def from_model_args(cls, args: ModelArgs) -> Qwen4Cache:
        if type(args) is not ModelArgs:
            raise TypeError("args must have exact type ModelArgs")
        gdn_layers = tuple(
            layer
            for layer, layer_type in enumerate(args.layer_types)
            if layer_type == "linear_attention"
        )
        qsa_layers = tuple(
            layer
            for layer, layer_type in enumerate(args.layer_types)
            if layer_type == _QWEN_SPARSE_ATTENTION
        )
        if len(gdn_layers) + len(qsa_layers) != args.num_hidden_layers:
            raise ValueError("ModelArgs contains an unsupported layer topology")
        return cls(
            gdn_layers=gdn_layers,
            qsa_layers=qsa_layers,
            model_args=args,
            _constructor=_CACHE_CONSTRUCTOR,
        )

    @classmethod
    def tiny(
        cls,
        *,
        gdn_layers: tuple[int, ...],
        qsa_layers: tuple[int, ...],
    ) -> Qwen4Cache:
        return cls(
            gdn_layers=gdn_layers,
            qsa_layers=qsa_layers,
            model_args=None,
            _constructor=_CACHE_CONSTRUCTOR,
        )

    @property
    def gdn(self) -> Mapping[int, None]:
        return MappingProxyType(dict.fromkeys(self._gdn))

    @property
    def qsa(self) -> Mapping[int, None]:
        return MappingProxyType(dict.fromkeys(self._qsa))

    def install_request(self, *, batch_size: int, activation_dtype: object) -> None:
        """Validate and allocate all production GDN state before execution."""

        if self._model_args is None:
            raise TypeError("tiny caches do not install production requests")
        if self._request_installed:
            raise RuntimeError("Qwen4 request state is already installed")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch size must be a positive exact int")
        if activation_dtype != mx.bfloat16:
            raise ValueError("pinned Qwen4 request activation dtype must be bfloat16")

        args = self._model_args
        key_dim = args.linear_num_key_heads * args.linear_key_head_dim
        value_dim = args.linear_num_value_heads * args.linear_value_head_dim
        conv_width = 2 * key_dim + value_dim
        layout = _Qwen4GDNLayout(
            batch_size=batch_size,
            conv_tokens=args.linear_conv_kernel_dim - 1,
            conv_width=conv_width,
            conv_dtype=str(activation_dtype),
            num_v_heads=args.linear_num_value_heads,
            head_v_dim=args.linear_value_head_dim,
            head_k_dim=args.linear_key_head_dim,
        )
        for layer in self._gdn:
            self._gdn[layer] = _Qwen4GDNCacheState(
                conv_state=mx.zeros(
                    (batch_size, layout.conv_tokens, conv_width),
                    dtype=activation_dtype,
                ),
                recurrent_state=mx.zeros(
                    (
                        batch_size,
                        layout.num_v_heads,
                        layout.head_v_dim,
                        layout.head_k_dim,
                    ),
                    dtype=mx.float32,
                ),
                offset=0,
                layout=layout,
            )
        self._request_installed = True

    def _gdn_state(self, layer: int) -> _Qwen4GDNCacheState:
        try:
            return self._gdn[layer]
        except KeyError as exc:
            raise ValueError(f"cache does not own GDN layer {layer}") from exc

    def _install_tiny_layer(
        self,
        layer: int,
        *,
        layout: _Qwen4GDNLayout,
        conv_dtype: object,
    ) -> None:
        if self._model_args is not None:
            raise TypeError("production caches must use install_request")
        current = self._gdn_state(layer)
        if current.layout is None:
            if current.conv_state is not None or current.recurrent_state is not None:
                raise ValueError("cache GDN state is only partially initialized")
            self._gdn[layer] = _Qwen4GDNCacheState(
                conv_state=mx.zeros(
                    (layout.batch_size, layout.conv_tokens, layout.conv_width),
                    dtype=conv_dtype,
                ),
                recurrent_state=mx.zeros(
                    (
                        layout.batch_size,
                        layout.num_v_heads,
                        layout.head_v_dim,
                        layout.head_k_dim,
                    ),
                    dtype=mx.float32,
                ),
                offset=0,
                layout=layout,
            )
        elif current.layout != layout:
            raise ValueError(
                "cache batch, dtype, or GDN state layout does not match input"
            )

    def _commit_gdn_direct(
        self,
        layer: int,
        *,
        conv_state: mx.array,
        recurrent_state: mx.array,
        length: int,
    ) -> None:
        current = self._gdn[layer]
        self._gdn[layer] = _Qwen4GDNCacheState(
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            offset=current.offset + length,
            layout=current.layout,
        )

    def _restore_prebound(
        self,
        saved: Mapping[int, _Qwen4GDNCacheState],
    ) -> None:
        self._gdn = dict(saved)

    def _trim_prebound(
        self,
        saved: Mapping[int, _Qwen4GDNCacheState],
        count: int,
    ) -> None:
        if type(count) is not int or count < 0:
            raise ValueError("trim count must be a non-negative exact int")
        if saved:
            reference = next(iter(saved))
            if self._gdn[reference].offset - saved[reference].offset != count:
                raise ValueError(
                    "snapshot does not identify the requested cache suffix"
                )
        self._restore_prebound(saved)

    def snapshot(self) -> _Qwen4CacheSnapshot:
        saved = MappingProxyType(self._gdn.copy())
        return _Qwen4CacheSnapshot(
            _owner=self._owner,
            _schema=_CACHE_SCHEMA,
            _gdn=saved,
            _qsa_layers=tuple(self._qsa),
            _restore_bound=lambda: self._restore_prebound(saved),
            _trim_bound=lambda count: self._trim_prebound(saved, count),
        )


class Qwen4GatedDeltaNet(nn.Module):
    """Qwen4 Gated DeltaNet with a construction-bound recurrence lane."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        if type(config) is not ModelArgs:
            raise TypeError(
                "production Qwen4GatedDeltaNet config must have exact type ModelArgs; "
                "tests must use Qwen4GatedDeltaNet.tiny"
            )
        if type(layer_idx) is not int or not 0 <= layer_idx < config.num_hidden_layers:
            raise ValueError("layer_idx is outside the ModelArgs layer range")
        if config.layer_types[layer_idx] != "linear_attention":
            raise ValueError("QSA layers cannot construct a GatedDeltaNet")
        topology = (
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
        )
        if topology != (16, 48, 128, 128, 4):
            raise ValueError("production GDN topology does not match pinned ModelArgs")
        self._install(
            config,
            layer_idx=layer_idx,
            recurrent_lane="metal",
            validate_inputs=False,
        )

    @classmethod
    def tiny(
        cls,
        config: object,
        layer_idx: int,
        *,
        recurrent_lane: str,
    ) -> Qwen4GatedDeltaNet:
        if type(config) is ModelArgs:
            raise TypeError("ModelArgs must use the production GDN constructor")
        if type(recurrent_lane) is not str:
            raise TypeError("recurrent_lane must have exact type str")
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module._install(
            config,
            layer_idx=layer_idx,
            recurrent_lane=recurrent_lane,
            validate_inputs=True,
        )
        return module

    def _install(
        self,
        config: object,
        *,
        layer_idx: int,
        recurrent_lane: str,
        validate_inputs: bool,
    ) -> None:
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
        self._execute = (
            self._validated_call if validate_inputs else self._direct_cached_call
        )

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
    ) -> tuple[mx.array, mx.array]:
        """Exact fixture lane using kernel-native physical [V,K] state."""

        query = mx.repeat(query, self._head_repeat, axis=2)
        key = mx.repeat(key, self._head_repeat, axis=2)
        outputs: list[mx.array] = []
        for token in range(query.shape[1]):
            q_token = query[:, token].astype(mx.float32)
            k_token = key[:, token].astype(mx.float32)
            v_token = value[:, token].astype(mx.float32)
            state = decay[:, token, :, None, None] * state
            prediction = mx.sum(state * k_token[..., None, :], axis=-1)
            state = state + (
                beta[:, token, :, None, None].astype(mx.float32)
                * (v_token - prediction)[..., :, None]
                * k_token[..., None, :]
            )
            outputs.append(mx.sum(state * q_token[..., None, :], axis=-1))
        return mx.stack(outputs, axis=1).astype(query.dtype), state

    @staticmethod
    def _metal_recurrence(
        query: mx.array,
        key: mx.array,
        value: mx.array,
        decay: mx.array,
        beta: mx.array,
        state: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Direct Metal lane over already-physical [B,Hv,Dv,Dk] state."""

        return gated_delta_kernel(query, key, value, decay, beta, state, None)

    def _validate_inputs(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None,
    ) -> tuple[int, int]:
        if not isinstance(hidden_states, mx.array) or hidden_states.ndim != 3:
            raise ValueError("hidden states must have rank 3")
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if sequence_length <= 0:
            raise ValueError("zero-length GDN inputs are not supported")
        if hidden_size != self.hidden_size:
            raise ValueError("hidden state width does not match GDN hidden size")
        if attention_mask is not None and (
            not isinstance(attention_mask, mx.array)
            or attention_mask.ndim != 2
            or attention_mask.shape != (batch_size, sequence_length)
            or attention_mask.dtype != mx.bool_
        ):
            raise ValueError(
                "attention mask must be a boolean [batch, sequence] MLX array"
            )
        return batch_size, sequence_length

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        cache: Qwen4Cache | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        return self._execute(
            hidden_states,
            cache=cache,
            attention_mask=attention_mask,
        )

    def _validated_call(
        self,
        hidden_states: mx.array,
        *,
        cache: Qwen4Cache | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, _ = self._validate_inputs(hidden_states, attention_mask)
        if cache is None:
            return self._stateless_call(
                hidden_states,
                attention_mask=attention_mask,
            )
        if type(cache) is not Qwen4Cache:
            raise TypeError("cache must have exact type Qwen4Cache")
        cache._install_tiny_layer(
            self.layer_idx,
            layout=_Qwen4GDNLayout(
                batch_size=batch_size,
                conv_tokens=self.conv_kernel_size - 1,
                conv_width=self.conv_dim,
                conv_dtype=str(hidden_states.dtype),
                num_v_heads=self.num_v_heads,
                head_v_dim=self.head_v_dim,
                head_k_dim=self.head_k_dim,
            ),
            conv_dtype=hidden_states.dtype,
        )
        return self._direct_cached_call(
            hidden_states,
            cache=cache,
            attention_mask=attention_mask,
        )

    def _stateless_call(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: mx.array | None,
    ) -> mx.array:
        batch_size = hidden_states.shape[0]
        output, _, _ = self._run_with_states(
            hidden_states,
            attention_mask=attention_mask,
            conv_state=mx.zeros(
                (batch_size, self.conv_kernel_size - 1, self.conv_dim),
                dtype=hidden_states.dtype,
            ),
            recurrent_state=mx.zeros(
                (
                    batch_size,
                    self.num_v_heads,
                    self.head_v_dim,
                    self.head_k_dim,
                ),
                dtype=mx.float32,
            ),
        )
        return output

    def _direct_cached_call(
        self,
        hidden_states: mx.array,
        *,
        cache: Qwen4Cache,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        cache_state = cache._gdn[self.layer_idx]
        output, conv_state, recurrent_state = self._run_with_states(
            hidden_states,
            attention_mask=attention_mask,
            conv_state=cache_state.conv_state,
            recurrent_state=cache_state.recurrent_state,
        )
        cache._commit_gdn_direct(
            self.layer_idx,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            length=hidden_states.shape[1],
        )
        return output

    def _run_with_states(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: mx.array | None,
        conv_state: mx.array,
        recurrent_state: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        batch_size, sequence_length, _ = hidden_states.shape
        if attention_mask is None:
            projection_input = hidden_states
        else:
            projection_input = mx.where(attention_mask[..., None], hidden_states, 0)

        mixed_qkv = self.in_proj_qkv(projection_input)
        z = self.in_proj_z(projection_input).reshape(
            batch_size,
            sequence_length,
            self.num_v_heads,
            self.head_v_dim,
        )
        beta = mx.sigmoid(self.in_proj_b(projection_input))
        a = self.in_proj_a(projection_input).astype(mx.float32)

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
        decay = mx.exp(
            -mx.exp(self.A_log.astype(mx.float32))
            * nn.softplus(a + self.dt_bias.astype(mx.float32))
        )
        recurrent_out, recurrent_state = self._recurrent(
            query,
            key,
            value,
            decay,
            beta,
            recurrent_state,
        )
        recurrent_out = self.norm(recurrent_out.astype(hidden_states.dtype), z)

        output = self.out_proj(
            recurrent_out.reshape(batch_size, sequence_length, self.value_dim)
        )
        return output, next_conv_state, recurrent_state
