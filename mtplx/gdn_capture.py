"""GDN state-capture verify helpers for Qwen3.5/Qwen3.6."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class QwenGDNVerifyConfig:
    """Construction-resolved operations for the fixed Qwen verify lane."""

    capture_backend: str
    projection_path: str
    linear_conv_path: str
    gdn_tail_path: str
    residual_path: str
    hidden_variant: str
    target_width: int
    attention_cache_type: str
    layer_eval_every: int
    layer_eval_schedule: tuple[tuple[int, int], ...]
    layer_eval_context_threshold: int
    layer_eval_max_q: int
    tape_replay_tgy: int
    project_inputs: Callable[[Any, Any], tuple[Any, Any, Any, Any]]
    capture_conv: Callable[[Any, Any, Any], tuple[Any, Any]]
    capture_delta: Callable[[Any, Any, Any, Any, Any], tuple[Any, Any, Any]]
    compute_g: Callable[[Any, Any, Any], Any]
    authoritative_state_path: str
    own_authoritative_state: Callable[[Any], Any]
    apply_gdn_tail: Callable[[Any, Any, Any], Any]
    apply_post_norm_residual: Callable[[Any, Any, Any], tuple[Any, Any]]
    embed_inputs: Callable[[Any], Any]
    create_fa_mask: Callable[[Any, Any], Any]
    create_ssm_mask: Callable[[Any, Any], Any]
    cache_context_length: Callable[[list[Any]], int]
    final_norm: Callable[[Any], Any]
    project_logits: Callable[[Any], Any]
    layer_routes: tuple[Callable[..., tuple[Any, dict[str, Any] | None]], ...] = ()

    @classmethod
    def stock(
        cls,
        *,
        capture_backend: str,
        hidden_variant: str,
    ) -> "QwenGDNVerifyConfig":
        from mlx_lm.models.gated_delta import compute_g

        return cls(
            capture_backend=capture_backend,
            projection_path="stock",
            linear_conv_path="stock",
            gdn_tail_path="stock",
            residual_path="stock",
            hidden_variant=hidden_variant,
            target_width=1,
            attention_cache_type="KVCache",
            layer_eval_every=0,
            layer_eval_schedule=(),
            layer_eval_context_threshold=0,
            layer_eval_max_q=8,
            tape_replay_tgy=8,
            project_inputs=_stock_gdn_input_projections,
            capture_conv=_stock_conv1d_capture_configured,
            capture_delta=partial(_kernel_tape_capture_configured, tgy=8),
            compute_g=compute_g,
            authoritative_state_path="identity",
            own_authoritative_state=_identity,
            apply_gdn_tail=_stock_gdn_tail,
            apply_post_norm_residual=_stock_post_norm_residual,
            embed_inputs=_identity,
            create_fa_mask=_return_none_mask,
            create_ssm_mask=_return_none_mask,
            cache_context_length=_zero_cache_context_length,
            final_norm=_identity,
            project_logits=_identity,
        )

    def eval_every(self, context_len: int) -> int:
        selected = self.layer_eval_every
        for threshold, every in self.layer_eval_schedule:
            if int(context_len) >= threshold:
                selected = every
        return selected


def _identity(value: Any) -> Any:
    return value


def _return_none_mask(_hidden_states: Any, _cache: Any) -> None:
    return None


def _zero_cache_context_length(_cache: list[Any]) -> int:
    return 0


def _configured_cache_context_length(
    cache: list[Any],
    *,
    layer_index: int,
    size: Callable[[Any], int],
) -> int:
    return int(size(cache[layer_index]))


def bind_qwen_cache_context_length(
    *,
    layer_index: int,
    target_width: int,
) -> tuple[str, Callable[[list[Any]], int]]:
    """Bind width-specific cache context lookup outside the target cycle."""
    from mlx_lm.models.cache import BatchKVCache, KVCache

    if target_width == 1:
        cache_type = KVCache
    elif target_width == 2:
        cache_type = BatchKVCache
    else:
        raise ValueError(f"configured Qwen target width must be 1 or 2, got {target_width}")
    return (
        cache_type.__name__,
        partial(
            _configured_cache_context_length,
            layer_index=layer_index,
            size=cache_type.size,
        ),
    )


def _configured_mask(
    hidden_states: Any,
    cache: list[Any],
    *,
    create: Callable[[Any, Any], Any],
    layer_index: int,
) -> Any:
    return create(hidden_states, cache[layer_index])


def _env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_GDN_POSTCONV_STATS: dict[str, Any] = {
    "enabled": False,
    "installed": False,
    "installation_status": "disabled",
    "installation_error": None,
    "gdn_layers": 0,
    "validated_contract": None,
    "implementation": "inline_g",
}
_A3B_GDN_POSTCONV_LAYER_TYPES = tuple(
    "linear_attention" if index % 4 != 3 else "full_attention"
    for index in range(40)
)


class A3BGDNPostconvConfigError(RuntimeError):
    """The exact A3B GDN post-conv lane could not be installed."""


@dataclass(frozen=True)
class A3BGDNPostconvInstallPlan:
    """Externally validated A3B GDN ownership awaiting its self-check."""

    gdns: tuple[Any, ...]


@dataclass(frozen=True)
class A3BGDNPostconvFactory:
    """Selfchecked, order-stable callables for the exact M1/M2/M3 traces.

    ``m3_implementations`` is the k=2 (3-row) verify recurrence; it defaults to
    empty so K1-only construction paths are unchanged and is populated whenever
    the postconv is installed.
    """

    m1_implementations: tuple[Callable[..., Any], ...]
    m2_implementations: tuple[Callable[..., Any], ...]
    m3_implementations: tuple[Callable[..., Any], ...] = ()


def _a3b_gdn_postconv_contract() -> dict[str, Any]:
    return {
        "batch": 1,
        "logical_m": [1, 2, 3],
        "routes": {
            "m1_correction": {
                "conv_shape": [1, 1, 8192],
                "gate_shapes": {"a": [1, 1, 32], "b": [1, 1, 32]},
                "output_shape": [1, 1, 32, 128],
                "captured_states_shape": [1, 1, 32, 128, 128],
            },
            "m2_verify": {
                "conv_shape": [1, 2, 8192],
                "gate_shapes": {"a": [1, 2, 32], "b": [1, 2, 32]},
                "output_shape": [1, 2, 32, 128],
                "captured_states_shape": [1, 2, 32, 128, 128],
            },
            "m3_verify": {
                "conv_shape": [1, 3, 8192],
                "gate_shapes": {"a": [1, 3, 32], "b": [1, 3, 32]},
                "output_shape": [1, 3, 32, 128],
                "captured_states_shape": [1, 3, 32, 128, 128],
            },
        },
        "state_shape": [1, 32, 128, 128],
        "input_dtype": "bfloat16",
        "state_dtype": "float32",
        "key_heads": 16,
        "value_heads": 32,
        "key_axis": 128,
        "value_axis": 128,
        "threadgroup": [32, 4, 1],
    }


def a3b_gdn_postconv_enabled() -> bool:
    return _env_enabled("MTPLX_FUSE_GDN_POST_CONV")


def _fail_a3b_gdn_postconv_configuration(message: str) -> None:
    _GDN_POSTCONV_STATS["installed"] = False
    _GDN_POSTCONV_STATS["installation_status"] = "configuration_error"
    _GDN_POSTCONV_STATS["installation_error"] = str(message)
    raise A3BGDNPostconvConfigError(message)


# Post-conv recurrence implementation selection.  ``inline_g`` (default) is the
# accepted TGY4 route; ``headquarter`` is the C1 redesigned-execution kernel.
_A3B_GDN_POSTCONV_IMPL_ENV = "MTPLX_A3B_GDN_POSTCONV_IMPL"
_A3B_GDN_POSTCONV_IMPL_DEFAULT = "inline_g"
_A3B_GDN_POSTCONV_IMPLS = ("inline_g", "headquarter")


def _a3b_gdn_postconv_impl_selection() -> str:
    """Resolve the requested post-conv implementation, fail-closed on unknown.

    Unset/empty selects the default ``inline_g`` route so the installed stack is
    byte-identical to the accepted baseline; any other value than the exact
    supported names hard-fails through the postconv configuration convention.
    """
    raw = os.environ.get(_A3B_GDN_POSTCONV_IMPL_ENV)
    value = (raw or "").strip().lower()
    if value == "":
        return _A3B_GDN_POSTCONV_IMPL_DEFAULT
    if value not in _A3B_GDN_POSTCONV_IMPLS:
        _fail_a3b_gdn_postconv_configuration(
            f"A3B GDN postconv {_A3B_GDN_POSTCONV_IMPL_ENV} must be one of "
            "'inline_g' or 'headquarter' (unset defaults to 'inline_g'); "
            f"got {raw!r}"
        )
    return value


def _a3b_gdn_postconv_headquarter_requested() -> bool:
    """Non-raising probe of whether the headquarter route is explicitly requested."""
    raw = os.environ.get(_A3B_GDN_POSTCONV_IMPL_ENV)
    return (raw or "").strip().lower() == "headquarter"


def _validate_a3b_quant_projection(
    gdn: Any,
    name: str,
    scales_shape: tuple[int, ...],
    layer_index: int,
) -> None:
    projection = getattr(gdn, name, None)
    scales = getattr(projection, "scales", None)
    if (
        int(getattr(projection, "bits", -1)) != 4
        or int(getattr(projection, "group_size", -1)) != 64
        or getattr(projection, "mode", None) != "affine"
        or tuple(getattr(scales, "shape", ())) != scales_shape
        or getattr(scales, "dtype", None) != mx.bfloat16
    ):
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv projection_quantization mismatch for "
            f"{name} at GDN layer {layer_index}"
        )


def prepare_a3b_gdn_postconv(
    model: Any,
    *,
    config: dict[str, Any],
) -> A3BGDNPostconvInstallPlan | None:
    """Validate checkpoint/model facts once for the exact A3B M1/M2 lanes."""
    _reset_gdn_postconv_stats_for_tests()
    if not a3b_gdn_postconv_enabled():
        return None
    _GDN_POSTCONV_STATS["enabled"] = True
    if not _env_enabled("MTPLX_COMPILED_TARGET_PREFIX"):
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv compiled_target_prefix_flag must be enabled"
        )
    if _env_enabled("MTPLX_NATIVE_GDN_TAIL"):
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv topology conflicts with MTPLX_NATIVE_GDN_TAIL"
        )

    text_config = config.get("text_config")
    if (
        config.get("model_type") != "qwen3_5_moe"
        or config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]
        or not isinstance(text_config, dict)
        or text_config.get("model_type") != "qwen3_5_moe_text"
        or int(text_config.get("hidden_size", -1)) != 2048
    ):
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv topology requires the exact A3B model"
        )
    if text_config.get("dtype") != "bfloat16":
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv config_dtype requires bfloat16"
        )

    text_model = getattr(model, "language_model", None)
    inner = getattr(text_model, "model", None)
    layers = list(getattr(inner, "layers", ()) or ())
    if len(layers) != 40 or int(text_config.get("num_hidden_layers", -1)) != 40:
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv layer_count requires exactly 40 layers"
        )
    actual_linear = [bool(getattr(layer, "is_linear", False)) for layer in layers]
    configured_types = tuple(text_config.get("layer_types", ()))
    expected_linear = [
        kind == "linear_attention" for kind in _A3B_GDN_POSTCONV_LAYER_TYPES
    ]
    if (
        actual_linear != expected_linear
        or configured_types != _A3B_GDN_POSTCONV_LAYER_TYPES
    ):
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv topology requires exact 30-layer ownership"
        )
    gdns = [
        getattr(layer, "linear_attn", None)
        for layer, is_linear in zip(layers, actual_linear)
        if is_linear
    ]
    if len(gdns) != 30 or any(gdn is None for gdn in gdns):
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv topology requires all 30 GDN modules"
        )

    config_geometry = {
        "linear_num_value_heads": 32,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    if any(
        int(text_config.get(name, -1)) != expected
        for name, expected in config_geometry.items()
    ) or float(text_config.get("rms_norm_eps", -1.0)) != 1e-6:
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv head_geometry mismatch in model config"
        )

    for index, gdn in enumerate(gdns):
        if getattr(gdn, "sharding_group", None) is not None:
            _fail_a3b_gdn_postconv_configuration(
                f"A3B GDN postconv sharding is forbidden at GDN layer {index}"
            )
        if (
            int(getattr(gdn, "conv_dim", -1)) != 8192
            or int(getattr(gdn, "key_dim", -1)) != 2048
            or int(getattr(gdn, "conv_kernel_size", -1)) != 4
        ):
            _fail_a3b_gdn_postconv_configuration(
                f"A3B GDN postconv conv_geometry mismatch at GDN layer {index}"
            )
        if (
            int(getattr(gdn, "num_k_heads", -1)) != 16
            or int(getattr(gdn, "num_v_heads", -1)) != 32
            or int(getattr(gdn, "head_k_dim", -1)) != 128
            or int(getattr(gdn, "head_v_dim", -1)) != 128
        ):
            _fail_a3b_gdn_postconv_configuration(
                f"A3B GDN postconv head_geometry mismatch at GDN layer {index}"
            )
        parameters = (
            ("A_log", (32,)),
            ("dt_bias", (32,)),
            ("conv1d.weight", (8192, 4, 1)),
        )
        for parameter_name, expected_shape in parameters:
            node = gdn
            for part in parameter_name.split("."):
                node = getattr(node, part, None)
            if tuple(getattr(node, "shape", ())) != expected_shape:
                _fail_a3b_gdn_postconv_configuration(
                    "A3B GDN postconv parameter_shape mismatch for "
                    f"{parameter_name} at GDN layer {index}"
                )
            if getattr(node, "dtype", None) != mx.bfloat16:
                _fail_a3b_gdn_postconv_configuration(
                    "A3B GDN postconv parameter_dtype requires BF16 for "
                    f"{parameter_name} at GDN layer {index}"
                )
        _validate_a3b_quant_projection(gdn, "in_proj_qkv", (8192, 32), index)
        _validate_a3b_quant_projection(gdn, "in_proj_a", (32, 32), index)
        _validate_a3b_quant_projection(gdn, "in_proj_b", (32, 32), index)

    _GDN_POSTCONV_STATS.update(
        {
            "installation_status": "awaiting_selfcheck",
            "installation_error": None,
            "gdn_layers": 30,
            "validated_contract": _a3b_gdn_postconv_contract(),
        }
    )
    return A3BGDNPostconvInstallPlan(gdns=tuple(gdns))


def install_a3b_gdn_postconv(
    plan: A3BGDNPostconvInstallPlan,
    selfcheck_report: dict[str, Any] | None,
) -> A3BGDNPostconvFactory:
    """Install the exact M1/M2 callables only after their combined self-check."""
    lanes = {} if selfcheck_report is None else selfcheck_report.get("lanes", {})
    implementation = _a3b_gdn_postconv_impl_selection()
    if implementation == "headquarter":
        required_lane = "gdn_postconv_headquarter"
        m1_apply = _apply_enabled_a3b_gdn_postconv_m1_headquarter
        m2_apply = _apply_enabled_a3b_gdn_postconv_m2_headquarter
        m3_apply = _apply_enabled_a3b_gdn_postconv_m3_headquarter
    else:
        required_lane = "gdn_postconv_inline_g"
        m1_apply = _apply_enabled_a3b_gdn_postconv_m1_tgy4
        m2_apply = _apply_enabled_a3b_gdn_postconv_m2_tgy4
        m3_apply = _apply_enabled_a3b_gdn_postconv_m3_tgy4
    if lanes.get(required_lane) != "ok":
        _fail_a3b_gdn_postconv_configuration(
            "A3B GDN postconv selfcheck did not validate the exact M1/M2 kernels"
            + (
                ""
                if implementation == _A3B_GDN_POSTCONV_IMPL_DEFAULT
                else f" for the {implementation} route"
            )
        )
    factory = A3BGDNPostconvFactory(
        m1_implementations=tuple(
            partial(
                m1_apply,
                A_log=gdn.A_log,
                dt_bias=gdn.dt_bias,
            )
            for gdn in plan.gdns
        ),
        m2_implementations=tuple(
            partial(
                m2_apply,
                A_log=gdn.A_log,
                dt_bias=gdn.dt_bias,
            )
            for gdn in plan.gdns
        ),
        m3_implementations=tuple(
            partial(
                m3_apply,
                A_log=gdn.A_log,
                dt_bias=gdn.dt_bias,
            )
            for gdn in plan.gdns
        ),
    )
    _GDN_POSTCONV_STATS["installed"] = True
    _GDN_POSTCONV_STATS["installation_status"] = "installed"
    _GDN_POSTCONV_STATS["implementation"] = implementation
    return factory


def gdn_postconv_stats() -> dict[str, Any]:
    """Report the immutable installation contract, never hot-path counters."""
    report = dict(_GDN_POSTCONV_STATS)
    contract = report.get("validated_contract")
    report["validated_contract"] = dict(contract) if isinstance(contract, dict) else None
    return report


def _reset_gdn_postconv_stats_for_tests() -> None:
    _GDN_POSTCONV_STATS.update(
        {
            "enabled": False,
            "installed": False,
            "installation_status": "disabled",
            "installation_error": None,
            "gdn_layers": 0,
            "validated_contract": None,
            "implementation": "inline_g",
        }
    )


def _cache_context_len(cache: Any) -> int:
    if cache is None:
        return 0
    best = 0
    for entry in cache:
        if entry is None:
            continue
        offset = getattr(entry, "offset", None)
        if isinstance(offset, mx.array):
            continue
        if offset is not None:
            best = max(best, int(offset or 0))
            continue
        size = getattr(entry, "size", None)
        if callable(size):
            try:
                best = max(best, int(size() or 0))
            except Exception:
                pass
    return best


def _target_layer_eval_every(context_len: int) -> int:
    schedule = os.environ.get("MTPLX_TARGET_LAYER_EVAL_SCHEDULE", "").strip()
    selected = 0
    if schedule:
        for part in schedule.replace(";", ",").split(","):
            item = part.strip()
            if not item:
                continue
            try:
                threshold_text, every_text = item.split(":", 1)
                threshold = int(threshold_text)
                every = int(every_text)
            except ValueError:
                continue
            if int(context_len) >= threshold:
                selected = max(0, every)
        return selected
    return int(os.environ.get("MTPLX_TARGET_LAYER_EVAL_EVERY", "0") or "0")


def _make_linear_conv1d_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto c_idx = thread_position_in_grid.x;
        auto b_idx = thread_position_in_grid.y;

        if (c_idx >= ConvDim) {
          return;
        }

        for (int t = 0; t < T; ++t) {
          auto parent_idx = t - 1;

          float acc = 0.0f;
          for (int k = 0; k < Keep; ++k) {
            float x;
            if (parent_idx < 0) {
              x = static_cast<float>(
                base_conv_state[(b_idx * Keep + k) * ConvDim + c_idx]
              );
            } else {
              x = static_cast<float>(
                conv_states[
                  (((b_idx * T + parent_idx) * Keep + k) * ConvDim) + c_idx
                ]
              );
            }
            auto w = static_cast<float>(conv_weight[c_idx * (Keep + 1) + k]);
            acc += x * w;
          }

          auto qkv_t = qkv + (b_idx * T + t) * ConvDim;
          acc += static_cast<float>(qkv_t[c_idx])
            * static_cast<float>(conv_weight[c_idx * (Keep + 1) + Keep]);

          conv_out[(b_idx * T + t) * ConvDim + c_idx] =
            static_cast<InT>(acc);

          for (int k = 0; k < Keep; ++k) {
            InT value;
            if (k + 1 < Keep) {
              if (parent_idx < 0) {
                value = base_conv_state[(b_idx * Keep + k + 1) * ConvDim + c_idx];
              } else {
                value = conv_states[
                  (((b_idx * T + parent_idx) * Keep + k + 1) * ConvDim) + c_idx
                ];
              }
            } else {
              value = qkv_t[c_idx];
            }
            conv_states[
              (((b_idx * T + t) * Keep + k) * ConvDim) + c_idx
            ] = value;
          }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_conv1d_capture",
        input_names=["qkv", "base_conv_state", "conv_weight", "T"],
        output_names=["conv_out", "conv_states"],
        source=source,
    )


def _make_linear_gated_delta_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        for (int t = 0; t < T; ++t) {
          auto parent_idx = t - 1;

          const device StT* parent_state;
          if (parent_idx < 0) {
            parent_state = state_in + (n * Dv + dv_idx) * Dk;
          } else {
            parent_state = states
              + (((b_idx * T + parent_idx) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          }

          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(parent_state[s_idx]);
          }

          auto q_t = q + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto k_t = k + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto v_t = v + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_t[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_t[dv_idx] - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_t[s_idx] * delta;
            out += state[i] * q_t[s_idx];
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          auto state_t = states
            + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state_t[s_idx] = static_cast<StT>(state[i]);
          }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_capture_v2",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


def _make_linear_gated_delta_final_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(state_ptr[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          auto q_t = q + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto k_t = k + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto v_t = v + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_t[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_t[dv_idx] - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_t[s_idx] * delta;
            out += state[i] * q_t[s_idx];
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }
        }

        auto state_out_ptr = state_out + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state_out_ptr[s_idx] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_final_v1",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float q_scale = inv_scale * inv_scale;
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float q_shared[Dk];
        threadgroup float k_shared[Dk];

        for (int t = 0; t < T; ++t) {
          auto parent_idx = t - 1;

          const device StT* parent_state;
          if (parent_idx < 0) {
            parent_state = state_in + (n * Dv + dv_idx) * Dk;
          } else {
            parent_state = states
              + (((b_idx * T + parent_idx) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          }

          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(parent_state[s_idx]);
          }

          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto q_t = conv_t + hk_idx * Dk;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          if (local_dv_idx == 0) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            float q_raw[n_per_t];
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              q_raw[i] = static_cast<float>(q_t[s_idx]);
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              q_sum += q_raw[i] * q_raw[i];
              k_sum += k_raw[i] * k_raw[i];
            }
            q_sum = simd_sum(q_sum);
            k_sum = simd_sum(k_sum);
            float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              q_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_val;
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            auto q_val = q_shared[s_idx];
            state[i] = state[i] + k_val * delta;
            out += state[i] * q_val;
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          auto state_t = states
            + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state_t[s_idx] = static_cast<StT>(state[i]);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_v1",
        input_names=["conv_out", "g", "beta", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_stream_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float q_scale = inv_scale * inv_scale;
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float q_shared[Dk];
        threadgroup float k_shared[Dk];

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(state_ptr[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto q_t = conv_t + hk_idx * Dk;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          if (local_dv_idx == 0) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            float q_raw[n_per_t];
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              q_raw[i] = static_cast<float>(q_t[s_idx]);
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              q_sum += q_raw[i] * q_raw[i];
              k_sum += k_raw[i] * k_raw[i];
            }
            q_sum = simd_sum(q_sum);
            k_sum = simd_sum(k_sum);
            float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              q_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_val;
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            auto q_val = q_shared[s_idx];
            state[i] = state[i] + k_val * delta;
            out += state[i] * q_val;
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          int capture_t = t - CaptureStart;
          if (capture_t >= 0) {
            auto state_t = states
              + (((b_idx * CaptureT + capture_t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto rounded = static_cast<StT>(state[i]);
              state_t[s_idx] = rounded;
              state[i] = static_cast<float>(rounded);
            }
          } else {
            for (int i = 0; i < n_per_t; ++i) {
              state[i] = static_cast<float>(static_cast<StT>(state[i]));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_stream_v1",
        input_names=["conv_out", "g", "beta", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_tape_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float q_scale = inv_scale * inv_scale;
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float q_shared[Dk];
        threadgroup float k_shared[Dk];

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(state_ptr[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto q_t = conv_t + hk_idx * Dk;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          if (local_dv_idx == 0) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            float q_raw[n_per_t];
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              q_raw[i] = static_cast<float>(q_t[s_idx]);
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              q_sum += q_raw[i] * q_raw[i];
              k_sum += k_raw[i] * k_raw[i];
            }
            q_sum = simd_sum(q_sum);
            k_sum = simd_sum(k_sum);
            float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              q_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_val;
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            auto q_val = q_shared[s_idx];
            state[i] = state[i] + k_val * delta;
            out += state[i] * q_val;
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
            tape[((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx] = delta;
          }

          for (int i = 0; i < n_per_t; ++i) {
            state[i] = static_cast<float>(static_cast<StT>(state[i]));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        auto state_t = final_state + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state_t[s_idx] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_tape_v1",
        input_names=["conv_out", "g", "beta", "state_in", "T"],
        output_names=["y", "final_state", "tape"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_tape_replay_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float k_shared[Dk];

        const device StT* state_ptr = state_in + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(state_ptr[s_idx]);
        }

        for (int t = 0; t < Steps; ++t) {
          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto g_t = g + (b_idx * T + t) * Hv;

          if (local_dv_idx == 0) {
            float k_sum = 0.0f;
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              k_sum += k_raw[i] * k_raw[i];
            }
            k_sum = simd_sum(k_sum);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          auto delta = tape[((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_t[hv_idx];
            state[i] = state[i] + k_shared[s_idx] * delta;
            state[i] = static_cast<float>(static_cast<StT>(state[i]));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        auto state_t = state_out + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state_t[s_idx] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_tape_replay_v1",
        input_names=["tape", "conv_out", "g", "state_in", "T"],
        output_names=["state_out"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_inline_g_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto local_dv_idx = thread_position_in_threadgroup.y;
        auto dv_idx = thread_position_in_grid.y;
        float inv_scale = 1.0f / metal::sqrt(float(Dk));
        float q_scale = inv_scale * inv_scale;
        float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
        threadgroup float q_shared[Dk];
        threadgroup float k_shared[Dk];
        threadgroup float g_shared;
        threadgroup float beta_shared;

        for (int t = 0; t < T; ++t) {
          auto parent_idx = t - 1;

          const device StT* parent_state;
          if (parent_idx < 0) {
            parent_state = state_in + (n * Dv + dv_idx) * Dk;
          } else {
            parent_state = states
              + (((b_idx * T + parent_idx) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          }

          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(parent_state[s_idx]);
          }

          auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
          auto q_t = conv_t + hk_idx * Dk;
          auto k_t = conv_t + KeyDim + hk_idx * Dk;
          auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
          auto a_t = a + (b_idx * T + t) * Hv;
          auto b_t = b + (b_idx * T + t) * Hv;

          if (dk_idx == 0 && local_dv_idx == 0) {
            InT b_val = b_t[hv_idx];
            auto beta_y = 1 / (1 + metal::exp(metal::abs(b_val)));
            InT beta_val = (b_val < InT(0)) ? beta_y : 1 - beta_y;

            InT a_val = a_t[hv_idx] + dt_bias[hv_idx];
            constexpr InT inf = metal::numeric_limits<InT>::infinity();
            InT maxval = metal::max(a_val, InT(0));
            InT minval = metal::min(a_val, InT(0));
            InT softplus_val = (minval == -inf || maxval == inf)
              ? maxval
              : (maxval + log1p(metal::exp(minval - maxval)));
            float decay_a = metal::exp(float(A_log[hv_idx]));
            beta_shared = static_cast<float>(beta_val);
            g_shared = metal::exp(-decay_a * float(softplus_val));
          }

          if (local_dv_idx == 0) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            float q_raw[n_per_t];
            float k_raw[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              q_raw[i] = static_cast<float>(q_t[s_idx]);
              k_raw[i] = static_cast<float>(k_t[s_idx]);
              q_sum += q_raw[i] * q_raw[i];
              k_sum += k_raw[i] * k_raw[i];
            }
            q_sum = simd_sum(q_sum);
            k_sum = simd_sum(k_sum);
            float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
            float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
              auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
              q_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
              k_shared[s_idx] =
                static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            state[i] = state[i] * g_shared;
            kv_mem += state[i] * k_val;
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem)
            * beta_shared;

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = k_shared[s_idx];
            auto q_val = q_shared[s_idx];
            state[i] = state[i] + k_val * delta;
            out += state[i] * q_val;
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          auto state_t = states
            + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state_t[s_idx] = static_cast<StT>(state[i]);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_inline_g_v1",
        input_names=["conv_out", "a", "b", "A_log", "dt_bias", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


def _make_linear_gated_delta_from_conv_headquarter_kernel():
    # C1 "headquarter" redesigned execution: one threadgroup per (head, Dv-quarter)
    # => grid (SIMDS*32, QUARTERS, B*Hv), threadgroup (SIMDS*32, 1, 1) = 8 simdgroups.
    # simd 0 computes the head's q/k rms-norm+scale + g/beta once into threadgroup
    # memory (redundancy 32x -> 4x), one producer->consumer barrier, then each
    # simdgroup drives RPS=(Dv/QUARTERS)/SIMDS=4 dv rows with fp32 state resident in
    # registers across the T loop.  Source verbatim from the G3a C1 bench candidate
    # (bit-exact vs inline_g: parity 0.0 on y and states at m1 and m2).
    if not mx.metal.is_available():
        return None

    source = """
    // --- geometry -----------------------------------------------------------
    auto n = thread_position_in_grid.z;          // b_idx*Hv + hv_idx
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    auto quarter = thread_position_in_grid.y;     // 0..QUARTERS-1
    uint tptg = thread_position_in_threadgroup.x; // 0..(SIMDS*32-1)
    uint simd_id = tptg / 32u;                     // 0..SIMDS-1
    uint dk_idx = thread_index_in_simdgroup;       // 0..31
    constexpr int n_per_t = Dk / 32;               // 4  (float4 per lane)
    constexpr int QSIZE = Dv / Quarters;           // dv rows per quarter (32)
    constexpr int RPS = QSIZE / Simds;             // dv rows per simdgroup (4)
    int base_dv = int(quarter) * QSIZE + int(simd_id) * RPS;

    float inv_scale = 1.0f / metal::sqrt(float(Dk));
    float q_scale = inv_scale * inv_scale;
    float k_scale = static_cast<float>(static_cast<InT>(inv_scale));

    threadgroup float q_shared[Dk];
    threadgroup float k_shared[Dk];
    threadgroup float g_shared;
    threadgroup float beta_shared;

    // running fp32 state for this simdgroup's RPS rows, resident in registers
    float S[RPS][n_per_t];
    for (int r = 0; r < RPS; ++r) {
      const device float4* s4 = reinterpret_cast<const device float4*>(
        state_in + (n * Dv + (base_dv + r)) * Dk);
      float4 sv = s4[dk_idx];
      S[r][0] = sv.x; S[r][1] = sv.y; S[r][2] = sv.z; S[r][3] = sv.w;
    }

    for (int t = 0; t < T; ++t) {
      auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
      auto q_t = conv_t + hk_idx * Dk;
      auto k_t = conv_t + KeyDim + hk_idx * Dk;
      auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
      auto a_t = a + (b_idx * T + t) * Hv;
      auto b_t = b + (b_idx * T + t) * Hv;

      // --- producer: simd 0 computes shared q/k (+ g/beta) once -------------
      if (simd_id == 0u) {
        if (dk_idx == 0u) {
          InT b_val = b_t[hv_idx];
          auto beta_y = 1 / (1 + metal::exp(metal::abs(b_val)));
          InT beta_val = (b_val < InT(0)) ? beta_y : 1 - beta_y;

          InT a_val = a_t[hv_idx] + dt_bias[hv_idx];
          constexpr InT inf = metal::numeric_limits<InT>::infinity();
          InT maxval = metal::max(a_val, InT(0));
          InT minval = metal::min(a_val, InT(0));
          InT softplus_val = (minval == -inf || maxval == inf)
            ? maxval
            : (maxval + log1p(metal::exp(minval - maxval)));
          float decay_a = metal::exp(float(A_log[hv_idx]));
          beta_shared = static_cast<float>(beta_val);
          g_shared = metal::exp(-decay_a * float(softplus_val));
        }

        float q_sum = 0.0f;
        float k_sum = 0.0f;
        float q_raw[n_per_t];
        float k_raw[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          q_raw[i] = static_cast<float>(q_t[s_idx]);
          k_raw[i] = static_cast<float>(k_t[s_idx]);
          q_sum += q_raw[i] * q_raw[i];
          k_sum += k_raw[i] * k_raw[i];
        }
        q_sum = simd_sum(q_sum);
        k_sum = simd_sum(k_sum);
        float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
        float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
          auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
          q_shared[s_idx] =
            static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
          k_shared[s_idx] =
            static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);   // BARRIER 1 (producer->consumer)

      // --- consumer: each simdgroup drives its RPS rows --------------------
      float g_local = g_shared;
      float beta_local = beta_shared;
      float qloc[n_per_t];
      float kloc[n_per_t];
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        qloc[i] = q_shared[s_idx];
        kloc[i] = k_shared[s_idx];
      }

      float kv[RPS];
      for (int r = 0; r < RPS; ++r) {
        float acc = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          S[r][i] = S[r][i] * g_local;
          acc += S[r][i] * kloc[i];
        }
        kv[r] = acc;
      }
      for (int r = 0; r < RPS; ++r) { kv[r] = simd_sum(kv[r]); }

      float delta[RPS];
      for (int r = 0; r < RPS; ++r) {
        delta[r] = (static_cast<float>(v_t[base_dv + r]) - kv[r]) * beta_local;
      }

      float out[RPS];
      for (int r = 0; r < RPS; ++r) {
        float acc = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          S[r][i] = S[r][i] + kloc[i] * delta[r];
          acc += S[r][i] * qloc[i];
        }
        out[r] = acc;
      }
      for (int r = 0; r < RPS; ++r) { out[r] = simd_sum(out[r]); }

      auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
      for (int r = 0; r < RPS; ++r) {
        int dv = base_dv + r;
        if (dk_idx == 0u) {
          y_t[dv] = static_cast<InT>(out[r]);
        }
        device float4* o4 = reinterpret_cast<device float4*>(
          states + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv) * Dk);
        o4[dk_idx] = float4(S[r][0], S[r][1], S[r][2], S[r][3]);
      }

      if (t + 1 < T) {
        threadgroup_barrier(mem_flags::mem_threadgroup);  // BARRIER 2 (WAR guard, T>1 only)
      }
    }
    """
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_headquarter_v1",
        input_names=["conv_out", "a", "b", "A_log", "dt_bias", "state_in", "T"],
        output_names=["y", "states"],
        source=source,
    )


_linear_conv1d_kernel = _make_linear_conv1d_kernel()
_linear_gated_delta_kernel = _make_linear_gated_delta_kernel()
_linear_gated_delta_final_kernel = _make_linear_gated_delta_final_kernel()
_linear_gated_delta_from_conv_kernel = _make_linear_gated_delta_from_conv_kernel()
_linear_gated_delta_from_conv_stream_kernel = (
    _make_linear_gated_delta_from_conv_stream_kernel()
)
_linear_gated_delta_from_conv_tape_kernel = (
    _make_linear_gated_delta_from_conv_tape_kernel()
)
_linear_gated_delta_from_conv_tape_replay_kernel = (
    _make_linear_gated_delta_from_conv_tape_replay_kernel()
)
_linear_gated_delta_from_conv_inline_g_kernel = (
    _make_linear_gated_delta_from_conv_inline_g_kernel()
)
_linear_gated_delta_from_conv_headquarter_kernel = (
    _make_linear_gated_delta_from_conv_headquarter_kernel()
)

_LINEAR_GDN_ALIASES = {"linear_gdn", "linear_gdn_len5"}
_LINEAR_GDN_FROM_CONV_ALIASES = {
    "linear_gdn_from_conv",
    "linear_gdn_from_conv_len5",
}
_LINEAR_GDN_FROM_CONV_STREAM_ALIASES = {
    "linear_gdn_from_conv_stream",
    "linear_gdn_from_conv_stream_len5",
}
_LINEAR_GDN_FROM_CONV_STREAM_SKIP0_ALIASES = {
    "linear_gdn_from_conv_stream_skip0",
    "linear_gdn_from_conv_stream_skip0_len5",
}
_LINEAR_GDN_FROM_CONV_TAPE_ALIASES = {
    "linear_gdn_from_conv_tape",
    "linear_gdn_from_conv_tape_len5",
}
_LINEAR_GDN_FROM_CONV_INLINE_G_ALIASES = {
    "linear_gdn_from_conv_inline_g",
    "linear_gdn_from_conv_inline_g_len5",
}
_LINEAR_GDN_FINAL_ALIASES = {"linear_gdn_final", "linear_gdn_final_len5"}
_DEMOTED_GDN_ALIASES = {
    "linear_gdn_conv",
    "linear_gdn_len6",
    "linear_gdn_mlp_gateup",
}
def _contiguous_recurrent_leaf(value: mx.array) -> mx.array:
    # Mirrors mlx-lm #1077's cache ownership fix: the authoritative recurrent
    # leaf must not retain the larger per-position capture buffer.
    return mx.contiguous(value)


def _maybe_contiguous_authoritative_gdn_leaf(value: mx.array) -> mx.array:
    if not _env_enabled("MTPLX_CAPTURE_CONTIGUOUS_GDN_STATE"):
        return value
    return _contiguous_recurrent_leaf(value)


def bind_qwen_authoritative_state_operation() -> tuple[str, Callable[[Any], Any]]:
    """Resolve the fixed Qwen authoritative-state operation at construction."""
    if _env_enabled("MTPLX_CAPTURE_CONTIGUOUS_GDN_STATE"):
        return "contiguous", _contiguous_recurrent_leaf
    return "identity", _identity


def _gdn_tape_meta(gdn: Any) -> dict[str, int]:
    return {
        "conv_dim": int(gdn.conv_dim),
        "head_k_dim": int(gdn.head_k_dim),
        "head_v_dim": int(gdn.head_v_dim),
        "num_k_heads": int(gdn.num_k_heads),
        "num_v_heads": int(gdn.num_v_heads),
        "key_dim": int(gdn.key_dim),
    }


def _gdn_meta_int(meta: Any, name: str) -> int:
    if hasattr(meta, name):
        return int(getattr(meta, name))
    if isinstance(meta, dict):
        return int(meta[name])
    return int(getattr(meta, name))


def resolve_gdn_capture_backend(backend: str | None = None) -> str:
    """Resolve the GDN capture backend with backwards-compatible env support."""
    if backend is None:
        env_value = os.environ.get("MTPLX_CAPTURE_CUSTOM_KERNEL")
        if env_value is None:
            return "stock"
        normalized_env = env_value.lower().replace("-", "_")
        if normalized_env in {"1", "true", "yes", "on"} | _LINEAR_GDN_ALIASES:
            return "linear_gdn"
        if normalized_env in _LINEAR_GDN_FROM_CONV_ALIASES:
            return "linear_gdn_from_conv"
        if normalized_env in _LINEAR_GDN_FROM_CONV_STREAM_ALIASES:
            return "linear_gdn_from_conv_stream"
        if normalized_env in _LINEAR_GDN_FROM_CONV_STREAM_SKIP0_ALIASES:
            return "linear_gdn_from_conv_stream_skip0"
        if normalized_env in _LINEAR_GDN_FROM_CONV_TAPE_ALIASES:
            return "linear_gdn_from_conv_tape"
        if normalized_env in _LINEAR_GDN_FROM_CONV_INLINE_G_ALIASES:
            return "linear_gdn_from_conv_inline_g"
        if normalized_env in _LINEAR_GDN_FINAL_ALIASES:
            return "linear_gdn_final"
        if normalized_env in {"0", "false", "no", "off", "stock"}:
            return "stock"
        if normalized_env in _DEMOTED_GDN_ALIASES:
            raise ValueError(
                f"MTPLX_CAPTURE_CUSTOM_KERNEL backend {env_value!r} is not promoted; "
                "use 'stock', 'linear-gdn', 'linear-gdn-len5', or "
                "'linear-gdn-from-conv'"
            )
        raise ValueError(
            "MTPLX_CAPTURE_CUSTOM_KERNEL must be one of 1/0, true/false, "
            "'linear-gdn', 'linear-gdn-len5', 'linear-gdn-from-conv', or 'stock'"
        )
    normalized = backend.replace("-", "_")
    if normalized == "stock":
        return "stock"
    if normalized in _LINEAR_GDN_ALIASES:
        return "linear_gdn"
    if normalized in _LINEAR_GDN_FROM_CONV_ALIASES:
        return "linear_gdn_from_conv"
    if normalized in _LINEAR_GDN_FROM_CONV_STREAM_ALIASES:
        return "linear_gdn_from_conv_stream"
    if normalized in _LINEAR_GDN_FROM_CONV_STREAM_SKIP0_ALIASES:
        return "linear_gdn_from_conv_stream_skip0"
    if normalized in _LINEAR_GDN_FROM_CONV_TAPE_ALIASES:
        return "linear_gdn_from_conv_tape"
    if normalized in _LINEAR_GDN_FROM_CONV_INLINE_G_ALIASES:
        return "linear_gdn_from_conv_inline_g"
    if normalized in _LINEAR_GDN_FINAL_ALIASES:
        return "linear_gdn_final"
    if normalized in _DEMOTED_GDN_ALIASES:
        raise ValueError(
            f"GDN capture backend {backend!r} is not promoted; use 'stock' or "
            "'linear-gdn-len5'"
        )
    raise ValueError(
        "GDN capture backend must be 'stock', 'linear-gdn', 'linear-gdn-len5', "
        "'linear-gdn-from-conv', or diagnostic 'linear-gdn-final'"
    )


def _linear_conv1d_capture(
    qkv: mx.array, base_conv_state: mx.array, conv_weight: mx.array
):
    if _linear_conv1d_kernel is None:
        return None
    B, T, conv_dim = qkv.shape
    keep = int(base_conv_state.shape[1])
    if (
        len(conv_weight.shape) != 3
        or int(conv_weight.shape[0]) != conv_dim
        or int(conv_weight.shape[1]) != keep + 1
        or int(conv_weight.shape[2]) != 1
    ):
        return None
    input_type = qkv.dtype
    raw_conv, conv_states = _linear_conv1d_kernel(
        inputs=[qkv, base_conv_state, conv_weight, T],
        template=[("InT", input_type), ("Keep", keep), ("ConvDim", conv_dim)],
        grid=(conv_dim, B, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, T, conv_dim), (B, T, keep, conv_dim)],
        output_dtypes=[input_type, input_type],
    )
    return nn.silu(raw_conv), conv_states


def _matching_quantized_linears(left: Any, right: Any) -> bool:
    if not isinstance(left, nn.QuantizedLinear) or not isinstance(
        right, nn.QuantizedLinear
    ):
        return False
    if "bias" in left or "bias" in right:
        return False
    return (
        int(left.bits) == int(right.bits)
        and int(left.group_size) == int(right.group_size)
        and str(left.mode) == str(right.mode)
        and tuple(left.weight.shape[1:]) == tuple(right.weight.shape[1:])
        and tuple(left.scales.shape[1:]) == tuple(right.scales.shape[1:])
        and tuple(left.biases.shape[1:]) == tuple(right.biases.shape[1:])
    )


def _fused_quantized_pair(
    owner: Any,
    cache_name: str,
    inputs: mx.array,
    left: nn.QuantizedLinear,
    right: nn.QuantizedLinear,
) -> tuple[mx.array, mx.array] | None:
    if not _matching_quantized_linears(left, right):
        return None
    cached = getattr(owner, cache_name, None)
    if cached is None:
        weight = mx.concatenate([left.weight, right.weight], axis=0)
        scales = mx.concatenate([left.scales, right.scales], axis=0)
        biases = mx.concatenate([left.biases, right.biases], axis=0)
        mx.eval(weight, scales, biases)
        cached = (weight, scales, biases, int(left.weight.shape[0]))
        setattr(owner, cache_name, cached)
    weight, scales, biases, split_at = cached
    out = mx.quantized_matmul(
        inputs,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=int(left.group_size),
        bits=int(left.bits),
        mode=str(left.mode),
    )
    left_out, right_out = mx.split(out, [int(split_at)], axis=-1)
    return left_out, right_out


def _fused_quantized_many(
    owner: Any,
    cache_name: str,
    inputs: mx.array,
    modules: tuple[nn.QuantizedLinear, ...],
) -> tuple[mx.array, ...] | None:
    if not modules:
        return None
    first = modules[0]
    if any(not _matching_quantized_linears(first, module) for module in modules[1:]):
        return None
    if "bias" in first:
        return None
    cached = getattr(owner, cache_name, None)
    if cached is None:
        weight = mx.concatenate([module.weight for module in modules], axis=0)
        scales = mx.concatenate([module.scales for module in modules], axis=0)
        biases = mx.concatenate([module.biases for module in modules], axis=0)
        mx.eval(weight, scales, biases)
        sizes = [int(module.weight.shape[0]) for module in modules]
        split_points = []
        running = 0
        for size in sizes[:-1]:
            running += size
            split_points.append(running)
        cached = (weight, scales, biases, tuple(split_points))
        setattr(owner, cache_name, cached)
    weight, scales, biases, split_points = cached
    out = mx.quantized_matmul(
        inputs,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=int(first.group_size),
        bits=int(first.bits),
        mode=str(first.mode),
    )
    return tuple(mx.split(out, list(split_points), axis=-1))


def _gdn_input_projections(
    gdn: Any, inputs: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    fuse_mode = os.environ.get("MTPLX_FUSE_GDN_PROJECTIONS", "").lower()
    if fuse_mode in {"all", "4to1", "one"}:
        fused = _fused_quantized_many(
            gdn,
            "_mtplx_fused_qkvzba",
            inputs,
            (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a),
        )
        if fused is not None:
            qkv, z, b, a = fused
            return qkv, z, b, a
    if fuse_mode in {"1", "true", "yes", "on"}:
        qkvz = _fused_quantized_pair(
            gdn,
            "_mtplx_fused_qkvz",
            inputs,
            gdn.in_proj_qkv,
            gdn.in_proj_z,
        )
        ba = _fused_quantized_pair(
            gdn,
            "_mtplx_fused_ba",
            inputs,
            gdn.in_proj_b,
            gdn.in_proj_a,
        )
        if qkvz is not None and ba is not None:
            qkv, z = qkvz
            b, a = ba
            return qkv, z, b, a
    return (
        gdn.in_proj_qkv(inputs),
        gdn.in_proj_z(inputs),
        gdn.in_proj_b(inputs),
        gdn.in_proj_a(inputs),
    )


def _stock_gdn_input_projections(
    gdn: Any,
    inputs: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    return (
        gdn.in_proj_qkv(inputs),
        gdn.in_proj_z(inputs),
        gdn.in_proj_b(inputs),
        gdn.in_proj_a(inputs),
    )


def _fused_pair_gdn_input_projections(
    gdn: Any,
    inputs: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    qkv, z = _fused_quantized_pair(
        gdn,
        "_mtplx_fused_qkvz",
        inputs,
        gdn.in_proj_qkv,
        gdn.in_proj_z,
    )
    b, a = _fused_quantized_pair(
        gdn,
        "_mtplx_fused_ba",
        inputs,
        gdn.in_proj_b,
        gdn.in_proj_a,
    )
    return qkv, z, b, a


def _fused_all_gdn_input_projections(
    gdn: Any,
    inputs: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    return _fused_quantized_many(
        gdn,
        "_mtplx_fused_qkvzba",
        inputs,
        (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a),
    )


def _stock_conv1d_capture(qkv: mx.array, base_conv_state: mx.array, gdn: Any):
    """Run the exact MLX Conv1d path and capture each linear-prefix state."""
    B, T, _ = qkv.shape
    keep = int(base_conv_state.shape[1])
    conv_input = mx.concatenate([base_conv_state, qkv], axis=1)
    conv_out = nn.silu(gdn.conv1d(conv_input))
    conv_states = mx.stack(
        [conv_input[:, i + 1 : i + 1 + keep, :] for i in range(T)],
        axis=1,
    )
    return conv_out, conv_states


def _stock_conv1d_capture_configured(
    qkv: mx.array,
    base_conv_state: mx.array,
    gdn: Any,
) -> tuple[mx.array, mx.array]:
    return _stock_conv1d_capture(qkv, base_conv_state, gdn)


def _linear_conv1d_capture_configured(
    qkv: mx.array,
    base_conv_state: mx.array,
    gdn: Any,
) -> tuple[mx.array, mx.array]:
    B, T, conv_dim = qkv.shape
    keep = int(base_conv_state.shape[1])
    input_type = qkv.dtype
    raw_conv, conv_states = _linear_conv1d_kernel(
        inputs=[qkv, base_conv_state, gdn.conv1d.weight, T],
        template=[("InT", input_type), ("Keep", keep), ("ConvDim", conv_dim)],
        grid=(conv_dim, B, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, T, conv_dim), (B, T, keep, conv_dim)],
        output_dtypes=[input_type, input_type],
    )
    return nn.silu(raw_conv), conv_states


def _linear_gated_delta_capture(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
):
    if _linear_gated_delta_kernel is None:
        return None
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk % 32 != 0:
        return None
    input_type = q.dtype
    state_type = state.dtype
    return _linear_gated_delta_kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _linear_gated_delta_final(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
):
    if _linear_gated_delta_final_kernel is None:
        return None
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk % 32 != 0:
        return None
    input_type = q.dtype
    state_type = state.dtype
    return _linear_gated_delta_final_kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _linear_gated_delta_from_conv_capture(
    conv_out: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    gdn: Any,
):
    if _linear_gated_delta_from_conv_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != int(gdn.conv_dim):
        return None
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    if Dk % 32 != 0:
        return None
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", "32"))
    except ValueError:
        tgy = 32
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    return _linear_gated_delta_from_conv_kernel(
        inputs=[conv_out, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _linear_gated_delta_from_conv_stream_capture(
    conv_out: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    gdn: Any,
    *,
    capture_start: int = 0,
):
    if _linear_gated_delta_from_conv_stream_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != int(gdn.conv_dim):
        return None
    capture_start = int(capture_start)
    if capture_start < 0 or capture_start >= int(T):
        return None
    capture_t = int(T) - capture_start
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    if Dk % 32 != 0:
        return None
    default_tgy = "8" if capture_start else "32"
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", default_tgy))
    except ValueError:
        tgy = 32
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    return _linear_gated_delta_from_conv_stream_kernel(
        inputs=[conv_out, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
            ("CaptureStart", capture_start),
            ("CaptureT", capture_t),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), (B, capture_t, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _linear_gated_delta_from_conv_tape_capture(
    conv_out: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    gdn: Any,
):
    # Alternative execution layout for the same contract (A3B C1 lineage).
    # Fail-closed: any ineligibility returns None from the wrapper and we
    # fall through to the incumbent TGY kernel below.
    if os.environ.get("MTPLX_LINEAR_GDN_TAPE_IMPL", "").strip().lower() == "headquarter":
        try:
            from .kernels.gdn_tape_headquarter import headquarter_tape_capture
        except Exception:
            headquarter_tape_capture = None
        if headquarter_tape_capture is not None:
            result = headquarter_tape_capture(conv_out, g, beta, state, gdn)
            if result is not None:
                return result
    if _linear_gated_delta_from_conv_tape_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != int(gdn.conv_dim):
        return None
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    if Dk % 32 != 0:
        return None
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", "8"))
    except ValueError:
        tgy = 8
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    return _linear_gated_delta_from_conv_tape_kernel(
        inputs=[conv_out, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk), (B, T, Hv, Dv)],
        output_dtypes=[input_type, state_type, mx.float32],
    )


def _kernel_tape_capture_configured(
    conv_out: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    gdn: Any,
    *,
    tgy: int,
):
    B, T, _ = conv_out.shape
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    input_type = conv_out.dtype
    state_type = state.dtype
    return _linear_gated_delta_from_conv_tape_kernel(
        inputs=[conv_out, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, int(tgy), 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk), (B, T, Hv, Dv)],
        output_dtypes=[input_type, state_type, mx.float32],
    )


def _headquarter_tape_capture_configured(
    conv_out: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    gdn: Any,
    *,
    kernel: Callable[..., Any],
    simds: int,
    quarters: int,
):
    B, T, _ = conv_out.shape
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    input_type = conv_out.dtype
    state_type = state.dtype
    return kernel(
        inputs=[conv_out, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
            ("Quarters", int(quarters)),
            ("Simds", int(simds)),
        ],
        grid=(int(simds) * 32, int(quarters), B * Hv),
        threadgroup=(int(simds) * 32, 1, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk), (B, T, Hv, Dv)],
        output_dtypes=[input_type, state_type, mx.float32],
    )


def _linear_gated_delta_from_conv_tape_capture_configured(
    conv_out: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    gdn: Any,
    *,
    implementation: str,
    tgy: int,
):
    """Construction helper; installed configs bind one direct backend callable."""
    if implementation == "headquarter":
        from .kernels.gdn_tape_headquarter import headquarter_tape_capture

        return headquarter_tape_capture(conv_out, g, beta, state, gdn)
    return _kernel_tape_capture_configured(
        conv_out,
        g,
        beta,
        state,
        gdn,
        tgy=tgy,
    )


def _linear_gated_delta_from_conv_tape_replay(
    tape: mx.array,
    conv_out: mx.array,
    g: mx.array,
    state: mx.array,
    gdn_meta: Any,
    *,
    steps: int,
    tgy: int | None = None,
):
    if _linear_gated_delta_from_conv_tape_replay_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != _gdn_meta_int(gdn_meta, "conv_dim"):
        return None
    steps = int(steps)
    if steps <= 0 or steps > int(T):
        return None
    Dk = _gdn_meta_int(gdn_meta, "head_k_dim")
    Dv = _gdn_meta_int(gdn_meta, "head_v_dim")
    Hk = _gdn_meta_int(gdn_meta, "num_k_heads")
    Hv = _gdn_meta_int(gdn_meta, "num_v_heads")
    if Dk % 32 != 0:
        return None
    if tgy is None:
        try:
            tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", "8"))
        except ValueError:
            tgy = 8
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    (state_out,) = _linear_gated_delta_from_conv_tape_replay_kernel(
        inputs=[tape, conv_out, g, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", _gdn_meta_int(gdn_meta, "key_dim")),
            ("ConvDim", _gdn_meta_int(gdn_meta, "conv_dim")),
            ("Steps", steps),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[state.shape],
        output_dtypes=[state_type],
    )
    return state_out


def _linear_gated_delta_from_conv_tape_replay_configured(
    capture: Mapping[str, mx.array],
    *,
    steps: int,
    kernel: Callable[..., Any],
    target_width: int,
    verified_tokens: int,
    tgy: int,
    head_k_dim: int,
    head_v_dim: int,
    num_k_heads: int,
    num_v_heads: int,
    key_dim: int,
    conv_dim: int,
) -> mx.array:
    """Replay one installed fixed-shape tape route without eligibility work."""
    conv_out = capture["conv_out"]
    state = capture["state_in"]
    (state_out,) = kernel(
        inputs=[
            capture["tape"],
            conv_out,
            capture["g"],
            state,
            verified_tokens,
        ],
        template=[
            ("InT", conv_out.dtype),
            ("StT", state.dtype),
            ("Dk", head_k_dim),
            ("Dv", head_v_dim),
            ("Hk", num_k_heads),
            ("Hv", num_v_heads),
            ("KeyDim", key_dim),
            ("ConvDim", conv_dim),
            ("Steps", int(steps)),
        ],
        grid=(32, head_v_dim, target_width * num_v_heads),
        threadgroup=(32, tgy, 1),
        output_shapes=[state.shape],
        output_dtypes=[state.dtype],
    )
    return state_out


def bind_qwen_tape_replay(
    *,
    gdn_layers: tuple[Any, ...],
    target_width: int,
    verified_tokens: int,
    tgy: int,
) -> Callable[..., mx.array]:
    """Validate once and bind the direct configured rejection replay."""
    if _linear_gated_delta_from_conv_tape_replay_kernel is None:
        raise RuntimeError("configured tape replay kernel is unavailable")
    geometries = {
        (
            _gdn_meta_int(gdn, "head_k_dim"),
            _gdn_meta_int(gdn, "head_v_dim"),
            _gdn_meta_int(gdn, "num_k_heads"),
            _gdn_meta_int(gdn, "num_v_heads"),
            _gdn_meta_int(gdn, "key_dim"),
            _gdn_meta_int(gdn, "conv_dim"),
        )
        for gdn in gdn_layers
    }
    if len(geometries) != 1:
        raise ValueError(
            "configured tape replay requires one exact GDN geometry"
        )
    head_k_dim, head_v_dim, num_k_heads, num_v_heads, key_dim, conv_dim = (
        geometries.pop()
    )
    if (
        int(target_width) not in {1, 2}
        or int(verified_tokens) != 3
        or head_k_dim % 32 != 0
        or int(tgy) not in {4, 8, 16, 32}
        or head_v_dim % int(tgy) != 0
    ):
        raise ValueError("configured tape replay geometry is invalid")
    return partial(
        _linear_gated_delta_from_conv_tape_replay_configured,
        kernel=_linear_gated_delta_from_conv_tape_replay_kernel,
        target_width=int(target_width),
        verified_tokens=int(verified_tokens),
        tgy=int(tgy),
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        key_dim=key_dim,
        conv_dim=conv_dim,
    )


def _commit_configured_recurrent_capture(
    entry: Any,
    capture: Mapping[str, mx.array],
    *,
    steps: int,
    row: int,
    replay: Callable[..., mx.array],
    own_authoritative_state: Callable[[Any], Any],
    install_state: Callable[[Any, list[mx.array]], None],
) -> None:
    conv_state = mx.contiguous(
        capture["conv_states"][row : row + 1, int(steps) - 1, :, :]
    )
    replayed = replay(capture, steps=steps)
    gdn_state = own_authoritative_state(
        replayed[row : row + 1, :, :, :]
    )
    install_state(entry, [conv_state, gdn_state])


def _commit_configured_recurrent_capture_memoized(
    entry: Any,
    capture: Mapping[str, mx.array],
    *,
    steps: int,
    row: int,
    layer_index: int,
    replay: Callable[..., mx.array],
    replay_memo: dict[tuple[int, int], mx.array],
    own_authoritative_state: Callable[[Any], Any],
    install_state: Callable[[Any, list[mx.array]], None],
) -> None:
    conv_state = mx.contiguous(
        capture["conv_states"][row : row + 1, int(steps) - 1, :, :]
    )
    replay_key = (int(layer_index), int(steps))
    try:
        replayed = replay_memo[replay_key]
    except KeyError:
        replayed = replay(capture, steps=steps)
        replay_memo[replay_key] = replayed
    gdn_state = own_authoritative_state(
        replayed[row : row + 1, :, :, :]
    )
    install_state(entry, [conv_state, gdn_state])


def _install_request_recurrent_state(
    entry: Any,
    state: list[mx.array],
) -> None:
    entry[0] = state[0]
    entry[1] = state[1]


def _install_owned_recurrent_state(
    entry: Any,
    state: list[mx.array],
) -> None:
    entry.replace_state(state)


def _install_owned_recurrent_state_lazy(
    entry: Any,
    state: list[mx.array],
) -> None:
    entry.replace_state_lazy_owned(state)


def _trim_configured_attention_capture(
    entry: Any,
    _capture: Any,
    *,
    steps: int,
    verified_tokens: int,
) -> None:
    entry.trim(verified_tokens - int(steps))


def _trim_configured_attention_capture_memoized(
    entry: Any,
    capture: Any,
    *,
    steps: int,
    verified_tokens: int,
    replay_memo: dict[tuple[int, int], mx.array],
) -> None:
    del replay_memo
    _trim_configured_attention_capture(
        entry,
        capture,
        steps=steps,
        verified_tokens=verified_tokens,
    )


def bind_qwen_capture_commit_route(
    *,
    config: QwenGDNVerifyConfig,
    cache_routes: tuple[Any, ...],
    layers: tuple[Any, ...],
    target_width: int,
    row: int,
    verified_tokens: int,
) -> Callable[..., list[Any]]:
    """Bind one fixed cohort-row rejection commit route."""
    del cache_routes
    gdn_layers = tuple(
        layer.linear_attn
        for layer in layers
        if bool(getattr(layer, "is_linear", False))
    )
    replay = bind_qwen_tape_replay(
        gdn_layers=gdn_layers,
        target_width=target_width,
        verified_tokens=verified_tokens,
        tgy=config.tape_replay_tgy,
    )
    install_recurrent_state = (
        _install_request_recurrent_state
        if target_width == 1
        else _install_owned_recurrent_state_lazy
    )
    layer_commits = tuple(
        partial(
            (
                _commit_configured_recurrent_capture
                if target_width == 1
                else _commit_configured_recurrent_capture_memoized
            ),
            row=row,
            replay=replay,
            own_authoritative_state=config.own_authoritative_state,
            install_state=install_recurrent_state,
            **(
                {}
                if target_width == 1
                else {"layer_index": layer_index}
            ),
        )
        if bool(getattr(layer, "is_linear", False))
        else partial(
            (
                _trim_configured_attention_capture
                if target_width == 1
                else _trim_configured_attention_capture_memoized
            ),
            verified_tokens=verified_tokens,
        )
        for layer_index, layer in enumerate(layers)
    )
    if target_width == 1:
        def commit_width1(
            cache: list[Any],
            captures: Mapping[int, Mapping[str, mx.array]],
            *,
            steps: int,
        ) -> list[Any]:
            for index, commit_layer in enumerate(layer_commits):
                commit_layer(cache[index], captures.get(index), steps=steps)
            return cache

        return commit_width1

    def commit_width2_request(
        request_cache: list[Any],
        captures: Mapping[int, Mapping[str, mx.array]],
        *,
        steps: int,
        replay_memo: dict[tuple[int, int], mx.array],
    ) -> list[Any]:
        for index, commit_layer in enumerate(layer_commits):
            commit_layer(
                request_cache[index],
                captures.get(index),
                steps=steps,
                replay_memo=replay_memo,
            )
        return request_cache

    return commit_width2_request


def _linear_gated_delta_from_conv_inline_g_capture(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    gdn: Any,
):
    if _linear_gated_delta_from_conv_inline_g_kernel is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != int(gdn.conv_dim):
        return None
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    if Dk % 32 != 0:
        return None
    try:
        tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", "32"))
    except ValueError:
        tgy = 32
    if tgy not in {4, 8, 16, 32} or Dv % tgy != 0:
        tgy = 8 if Dv % 8 == 0 else 4
    input_type = conv_out.dtype
    state_type = state.dtype
    return _linear_gated_delta_from_conv_inline_g_kernel(
        inputs=[conv_out, a, b, gdn.A_log, gdn.dt_bias, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, tgy, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def _a3b_compiled_target_gdn_postconv_m1_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the fixed A3B compiled-target M1 recurrence with TGY4."""
    return _linear_gated_delta_from_conv_inline_g_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 1],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
        ],
        grid=(32, 128, 32),
        threadgroup=(32, 4, 1),
        output_shapes=[(1, 1, 32, 128), (1, 1, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _a3b_compiled_target_gdn_postconv_m2_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the fixed A3B compiled-target M2 recurrence with TGY4."""
    return _linear_gated_delta_from_conv_inline_g_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 2],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
        ],
        grid=(32, 128, 32),
        threadgroup=(32, 4, 1),
        output_shapes=[(1, 2, 32, 128), (1, 2, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _apply_enabled_a3b_gdn_postconv_m1_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed exact A3B M1/TGY4 route."""
    return _a3b_compiled_target_gdn_postconv_m1_tgy4(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _apply_enabled_a3b_gdn_postconv_m2_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed exact A3B M2/TGY4 route."""
    return _a3b_compiled_target_gdn_postconv_m2_tgy4(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _a3b_compiled_target_gdn_postconv_m1_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the fixed A3B compiled-target M1 recurrence with the C1 headquarter kernel."""
    return _linear_gated_delta_from_conv_headquarter_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 1],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
            ("Quarters", 4),
            ("Simds", 8),
        ],
        grid=(256, 4, 32),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 1, 32, 128), (1, 1, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _a3b_compiled_target_gdn_postconv_m2_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the fixed A3B compiled-target M2 recurrence with the C1 headquarter kernel."""
    return _linear_gated_delta_from_conv_headquarter_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 2],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
            ("Quarters", 4),
            ("Simds", 8),
        ],
        grid=(256, 4, 32),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 2, 32, 128), (1, 2, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _apply_enabled_a3b_gdn_postconv_m1_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed exact A3B M1 headquarter route."""
    return _a3b_compiled_target_gdn_postconv_m1_headquarter(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _apply_enabled_a3b_gdn_postconv_m2_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed exact A3B M2 headquarter route."""
    return _a3b_compiled_target_gdn_postconv_m2_headquarter(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _a3b_compiled_target_gdn_postconv_m3_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the A3B compiled-target M3 (k=2, 3-row) recurrence with TGY4.

    Identical to the M2 launch except the logical sequence length is 3 -- the
    inline_g kernel scans ``logical_m`` positions, so the k=2 verify
    ``[primary, d1, d2]`` recurrence reuses the exact M1/M2 arithmetic per row.
    """
    return _linear_gated_delta_from_conv_inline_g_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 3],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
        ],
        grid=(32, 128, 32),
        threadgroup=(32, 4, 1),
        output_shapes=[(1, 3, 32, 128), (1, 3, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _a3b_compiled_target_gdn_postconv_m3_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Launch the A3B compiled-target M3 (k=2, 3-row) recurrence with headquarter."""
    return _linear_gated_delta_from_conv_headquarter_kernel(
        inputs=[conv_out, a, b, A_log, dt_bias, state, 3],
        template=[
            ("InT", mx.bfloat16),
            ("StT", mx.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 32),
            ("KeyDim", 2048),
            ("ConvDim", 8192),
            ("Quarters", 4),
            ("Simds", 8),
        ],
        grid=(256, 4, 32),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 3, 32, 128), (1, 3, 32, 128, 128)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )


def _apply_enabled_a3b_gdn_postconv_m3_tgy4(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed exact A3B M3/TGY4 route (k=2)."""
    return _a3b_compiled_target_gdn_postconv_m3_tgy4(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _apply_enabled_a3b_gdn_postconv_m3_headquarter(
    conv_out: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    *,
    A_log: mx.array,
    dt_bias: mx.array,
):
    """Execute the construction-installed exact A3B M3 headquarter route (k=2)."""
    return _a3b_compiled_target_gdn_postconv_m3_headquarter(
        conv_out,
        a,
        b,
        state,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def _stock_gated_delta_capture(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    state: mx.array,
    mask: Any,
    gdn: Any,
):
    """Capture per-position recurrent state through stock MLX single-token steps."""
    from mlx_lm.models.gated_delta import gated_delta_update

    T = int(q.shape[1])
    outs = []
    states = []
    current = state
    for idx in range(T):
        step_mask = None
        if mask is not None and not isinstance(mask, str):
            step_mask = mask[:, idx : idx + 1]
        out, current = gated_delta_update(
            q[:, idx : idx + 1, :, :],
            k[:, idx : idx + 1, :, :],
            v[:, idx : idx + 1, :, :],
            a[:, idx : idx + 1, :],
            b[:, idx : idx + 1, :],
            gdn.A_log,
            gdn.dt_bias,
            current,
            step_mask,
            use_kernel=not gdn.training,
        )
        outs.append(out)
        states.append(current)
    return mx.concatenate(outs, axis=1), mx.stack(states, axis=1)


def _stock_gdn_tail(gdn: Any, out: mx.array, z: mx.array) -> mx.array:
    B, S = out.shape[:2]
    return gdn.out_proj(gdn.norm(out, z).reshape(B, S, -1))


def _stock_qmv_gdn_tail(gdn: Any, out: mx.array, z: mx.array) -> mx.array:
    B, S = out.shape[:2]
    return _configured_qmv8_matmul(
        gdn.norm(out, z).reshape(B, S, -1),
        gdn.out_proj,
    )


def _fused_gdn_tail(gdn: Any, out: mx.array, z: mx.array) -> mx.array:
    from .kernels.fused_norm import fused_gdn_norm_gate

    B, S = out.shape[:2]
    normalized = fused_gdn_norm_gate(out, z, gdn.norm.weight, gdn.norm.eps)
    return gdn.out_proj(normalized.reshape(B, S, -1))


def _fused_qmv_gdn_tail(gdn: Any, out: mx.array, z: mx.array) -> mx.array:
    from .kernels.fused_norm import fused_gdn_norm_gate

    B, S = out.shape[:2]
    normalized = fused_gdn_norm_gate(out, z, gdn.norm.weight, gdn.norm.eps)
    return _configured_qmv8_matmul(
        normalized.reshape(B, S, -1),
        gdn.out_proj,
    )


def _configured_qmv8_matmul(x: mx.array, module: Any) -> mx.array:
    from .verify_qmv import _stocklike_qmv8_kernel

    leading = x.shape[:-1]
    m = 1
    for dim in leading:
        m *= int(dim)
    k = int(x.shape[-1])
    n = int(module.weight.shape[0])
    x2 = mx.contiguous(x.reshape(m, k))
    kernel = _stocklike_qmv8_kernel(int(module.group_size), x.dtype)
    (y,) = kernel(
        inputs=[x2, module.weight, module.scales, module.biases, k, n],
        template=[("T", x.dtype), ("GS", int(module.group_size))],
        grid=(32 * m, 2 * (n // 8), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*leading, n)


def _native_gdn_tail(
    gdn: Any,
    out: mx.array,
    z: mx.array,
    *,
    native: Any,
    num_simdgroups: int,
) -> mx.array:
    leading = out.shape[:-2]
    m = 1
    for dim in leading:
        m *= int(dim)
    hv = int(out.shape[-2])
    dv = int(out.shape[-1])
    out2 = mx.contiguous(out.reshape(m * hv, dv))
    z2 = mx.contiguous(z.reshape(m * hv, dv))
    projected = native.gdn_norm_gate_out_qmv8(
        out2,
        z2,
        gdn.norm.weight,
        gdn.out_proj.weight,
        gdn.out_proj.scales,
        gdn.out_proj.biases,
        hv,
        float(gdn.norm.eps),
        int(gdn.out_proj.group_size),
        int(num_simdgroups),
    )
    return projected.reshape(*leading, int(gdn.out_proj.weight.shape[0]))


def _stock_post_norm_residual(
    layer: Any,
    hidden_states: mx.array,
    residual: mx.array,
) -> tuple[mx.array, mx.array]:
    hidden = hidden_states + residual
    return hidden, layer.post_attention_layernorm(hidden)


def _fused_post_norm_residual(
    layer: Any,
    hidden_states: mx.array,
    residual: mx.array,
) -> tuple[mx.array, mx.array]:
    from .kernels.fused_norm import fused_add_rmsnorm

    return fused_add_rmsnorm(
        hidden_states,
        residual,
        layer.post_attention_layernorm.weight,
        layer.post_attention_layernorm.eps,
        threadgroup_size=512,
    )


def _configured_recurrent_layer_step(
    hidden_states: mx.array,
    fa_mask: Any,
    ssm_mask: Any,
    layer_cache: Any,
    *,
    layer: Any,
    config: QwenGDNVerifyConfig,
) -> tuple[mx.array, dict[str, Any]]:
    del fa_mask
    normed = layer.input_layernorm(hidden_states)
    residual, capture = gdn_forward_with_capture_configured(
        layer.linear_attn,
        normed,
        ssm_mask,
        layer_cache,
        config=config,
    )
    hidden, mlp_input = config.apply_post_norm_residual(
        layer,
        hidden_states,
        residual,
    )
    return hidden + layer.mlp(mlp_input), capture


def _configured_attention_layer_step(
    hidden_states: mx.array,
    fa_mask: Any,
    ssm_mask: Any,
    layer_cache: Any,
    *,
    layer: Any,
    config: QwenGDNVerifyConfig,
) -> tuple[mx.array, None]:
    del ssm_mask
    normed = layer.input_layernorm(hidden_states)
    residual = layer.self_attn(normed, mask=fa_mask, cache=layer_cache)
    hidden, mlp_input = config.apply_post_norm_residual(
        layer,
        hidden_states,
        residual,
    )
    return hidden + layer.mlp(mlp_input), None


def _bind_configured_layer_routes(
    config: QwenGDNVerifyConfig,
    layers: tuple[Any, ...],
) -> tuple[Callable[..., tuple[Any, dict[str, Any] | None]], ...]:
    routes = []
    for layer in layers:
        operation = (
            _configured_recurrent_layer_step
            if bool(getattr(layer, "is_linear", False))
            else _configured_attention_layer_step
        )
        routes.append(partial(operation, layer=layer, config=config))
    return tuple(routes)


def _parse_layer_eval_schedule(raw: str) -> tuple[tuple[int, int], ...]:
    schedule: list[tuple[int, int]] = []
    for part in raw.replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        threshold_text, every_text = item.split(":", 1)
        schedule.append((int(threshold_text), max(0, int(every_text))))
    return tuple(schedule)


def resolve_qwen_gdn_verify_config(
    *,
    capture_backend: str,
    hidden_variant: str,
    model: Any,
    target_width: int,
    layers: tuple[Any, ...] = (),
) -> QwenGDNVerifyConfig:
    """Resolve every Qwen verify choice once at lane construction."""
    backend = resolve_gdn_capture_backend(capture_backend)
    if backend != "linear_gdn_from_conv_tape":
        raise ValueError(
            "fixed Qwen GDN verify requires linear_gdn_from_conv_tape"
        )
    gdn_layers = tuple(
        layer.linear_attn
        for layer in layers
        if bool(getattr(layer, "is_linear", False))
    )
    text_model = getattr(model, "language_model", model)
    inner = text_model.model
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.models.gated_delta import compute_g
    attention_cache_index = int(inner.fa_idx)
    attention_cache_type, cache_context_length = bind_qwen_cache_context_length(
        layer_index=attention_cache_index,
        target_width=target_width,
    )
    project_logits = (
        inner.embed_tokens.as_linear
        if bool(text_model.args.tie_word_embeddings)
        else text_model.lm_head
    )

    projection_raw = os.environ.get("MTPLX_FUSE_GDN_PROJECTIONS", "").lower()
    if projection_raw in {"all", "4to1", "one", "1", "true", "yes", "on"}:
        raise ValueError(
            "fixed Qwen GDN verify does not install dynamically validated "
            "projection fusion"
        )
    projection_path = "stock"
    project_inputs = _stock_gdn_input_projections
    (
        authoritative_state_path,
        own_authoritative_state,
    ) = bind_qwen_authoritative_state_operation()

    if _env_enabled("MTPLX_LINEAR_CONV1D_CAPTURE"):
        if _linear_conv1d_kernel is None:
            raise RuntimeError("configured linear Conv1d capture kernel is unavailable")
        for gdn in gdn_layers:
            weight = gdn.conv1d.weight
            if (
                len(weight.shape) != 3
                or int(weight.shape[0]) != int(gdn.conv_dim)
                or int(weight.shape[1]) != int(gdn.conv_kernel_size)
                or int(weight.shape[2]) != 1
            ):
                raise ValueError("configured linear Conv1d geometry is invalid")
        linear_conv_path = "linear"
        capture_conv = _linear_conv1d_capture_configured
    else:
        linear_conv_path = "stock"
        capture_conv = _stock_conv1d_capture_configured

    tape_implementation = (
        os.environ.get("MTPLX_LINEAR_GDN_TAPE_IMPL", "").strip().lower()
        or "kernel"
    )
    tape_tgy = int(os.environ.get("MTPLX_LINEAR_GDN_FROM_CONV_TGY", "8") or "8")
    if tape_implementation == "headquarter":
        from .kernels import gdn_tape_headquarter

        if gdn_tape_headquarter._KERNEL is None:
            raise RuntimeError("configured headquarter tape kernel is unavailable")
        simds = int(gdn_tape_headquarter._SIMDS)
        quarters = int(gdn_tape_headquarter._QUARTERS)
        for gdn in gdn_layers:
            qsize = int(gdn.head_v_dim) // quarters
            if (
                int(gdn.head_k_dim) % 32 != 0
                or int(gdn.head_v_dim) % quarters != 0
                or qsize % simds != 0
                or int(gdn.num_v_heads) % int(gdn.num_k_heads) != 0
            ):
                raise ValueError("configured headquarter tape geometry is invalid")
        capture_delta = partial(
            _headquarter_tape_capture_configured,
            kernel=gdn_tape_headquarter._KERNEL,
            simds=simds,
            quarters=quarters,
        )
    elif tape_implementation == "kernel":
        if _linear_gated_delta_from_conv_tape_kernel is None:
            raise RuntimeError("configured tape capture kernel is unavailable")
        for gdn in gdn_layers:
            if (
                int(gdn.head_k_dim) % 32 != 0
                or tape_tgy not in {4, 8, 16, 32}
                or int(gdn.head_v_dim) % tape_tgy != 0
            ):
                raise ValueError("configured tape capture geometry is invalid")
        capture_delta = partial(_kernel_tape_capture_configured, tgy=tape_tgy)
    else:
        raise ValueError(
            f"unsupported configured Qwen tape implementation: {tape_implementation}"
        )

    from .kernel_selfcheck import lane_disabled

    qmv_enabled = _env_enabled("MTPLX_GDN_OUT_QMV8")
    if qmv_enabled:
        from .verify_qmv import is_stocklike_qmv8_eligible

        for gdn in gdn_layers:
            if (
                not is_stocklike_qmv8_eligible(gdn.out_proj)
                or "bias" in gdn.out_proj
                or gdn.out_proj.scales.dtype != gdn.norm.weight.dtype
                or gdn.out_proj.biases.dtype != gdn.norm.weight.dtype
            ):
                raise ValueError("configured QMV8 GDN output geometry is invalid")
    if _env_enabled("MTPLX_NATIVE_GDN_TAIL"):
        from .kernels import native_gdn_tail

        native = native_gdn_tail._native_module()
        if native is None:
            raise RuntimeError("configured native GDN tail is unavailable")
        for gdn in gdn_layers:
            out_proj = gdn.out_proj
            if (
                int(getattr(out_proj, "bits", 0) or 0) != 8
                or int(getattr(out_proj, "group_size", 0) or 0)
                not in {32, 64, 128}
                or str(getattr(out_proj, "mode", "affine")) != "affine"
                or "bias" in out_proj
                or int(out_proj.weight.shape[1]) * 4
                != int(gdn.num_v_heads) * int(gdn.head_v_dim)
                or out_proj.scales.dtype != gdn.norm.weight.dtype
                or out_proj.biases.dtype != gdn.norm.weight.dtype
            ):
                raise ValueError("configured native GDN tail geometry is invalid")
        simdgroups = int(os.environ.get("MTPLX_NATIVE_GDN_TAIL_SIMDGROUPS") or 2)
        gdn_tail_path = f"native_sg{simdgroups}"
        apply_gdn_tail = partial(
            _native_gdn_tail,
            native=native,
            num_simdgroups=simdgroups,
        )
    elif (
        _env_enabled("MTPLX_FUSE_GDN_NORM_GATE")
        and not lane_disabled("fused_gdn_norm_gate")
    ):
        gdn_tail_path = "fused_norm_qmv" if qmv_enabled else "fused_norm"
        apply_gdn_tail = _fused_qmv_gdn_tail if qmv_enabled else _fused_gdn_tail
    else:
        gdn_tail_path = "stock_qmv" if qmv_enabled else "stock"
        apply_gdn_tail = _stock_qmv_gdn_tail if qmv_enabled else _stock_gdn_tail

    if (
        _env_enabled("MTPLX_FUSE_POST_NORM_RESIDUAL")
        and not lane_disabled("fused_add_rmsnorm")
    ):
        residual_path = "fused"
        apply_post_norm_residual = _fused_post_norm_residual
    else:
        residual_path = "stock"
        apply_post_norm_residual = _stock_post_norm_residual

    config = QwenGDNVerifyConfig(
        capture_backend=backend,
        projection_path=projection_path,
        linear_conv_path=linear_conv_path,
        gdn_tail_path=gdn_tail_path,
        residual_path=residual_path,
        hidden_variant=hidden_variant,
        target_width=target_width,
        attention_cache_type=attention_cache_type,
        layer_eval_every=int(
            os.environ.get("MTPLX_TARGET_LAYER_EVAL_EVERY", "0") or "0"
        ),
        layer_eval_schedule=_parse_layer_eval_schedule(
            os.environ.get("MTPLX_TARGET_LAYER_EVAL_SCHEDULE", "")
        ),
        layer_eval_context_threshold=int(
            os.environ.get("MTPLX_TARGET_LAYER_EVAL_CONTEXT_THRESHOLD", "0")
            or "0"
        ),
        layer_eval_max_q=int(
            os.environ.get("MTPLX_TARGET_LAYER_EVAL_MAX_Q", "8") or "8"
        ),
        tape_replay_tgy=tape_tgy,
        project_inputs=project_inputs,
        capture_conv=capture_conv,
        capture_delta=capture_delta,
        compute_g=compute_g,
        authoritative_state_path=authoritative_state_path,
        own_authoritative_state=own_authoritative_state,
        apply_gdn_tail=apply_gdn_tail,
        apply_post_norm_residual=apply_post_norm_residual,
        embed_inputs=inner.embed_tokens,
        create_fa_mask=partial(
            _configured_mask,
            create=create_attention_mask,
            layer_index=attention_cache_index,
        ),
        create_ssm_mask=partial(
            _configured_mask,
            create=create_ssm_mask,
            layer_index=int(inner.ssm_idx),
        ),
        cache_context_length=cache_context_length,
        final_norm=inner.norm,
        project_logits=project_logits,
    )
    return replace(
        config,
        layer_routes=_bind_configured_layer_routes(config, layers),
    )


def gdn_forward_with_capture_configured(
    gdn: Any,
    inputs: mx.array,
    mask: Any,
    cache: Any,
    *,
    config: QwenGDNVerifyConfig,
) -> tuple[mx.array, dict[str, Any]]:
    """Run the installed tape capture route without dynamic eligibility checks."""
    del mask

    B, S, _ = inputs.shape
    qkv, z, b, a = config.project_inputs(gdn, inputs)
    z = z.reshape(B, S, gdn.num_v_heads, gdn.head_v_dim)
    conv_state = cache[0]
    conv_out, conv_states = config.capture_conv(qkv, conv_state, gdn)
    state = cache[1]
    beta = mx.sigmoid(b)
    g = config.compute_g(gdn.A_log, a, gdn.dt_bias)
    out, final_state, tape = config.capture_delta(
        conv_out,
        g,
        beta,
        state,
        gdn,
    )
    states = final_state[:, None, :, :, :]
    cache[0] = mx.contiguous(conv_states[:, -1, :, :])
    cache[1] = config.own_authoritative_state(
        states[:, -1, :, :, :]
    )
    cache.advance(S)
    out = config.apply_gdn_tail(gdn, out, z)
    return out, {
        "conv_states": conv_states,
        "conv_out": conv_out,
        "g": g,
        "state_in": state,
        "tape": tape,
        "gdn_meta": _gdn_tape_meta(gdn),
        "tape_replay_tgy": config.tape_replay_tgy,
    }


def gdn_forward_with_capture(
    gdn: Any,
    inputs: mx.array,
    mask: Any = None,
    cache: Any = None,
    *,
    capture_backend: str | None = None,
):
    if getattr(gdn, "sharding_group", None) is not None:
        return gdn(inputs, mask=mask, cache=cache), None

    from mlx_lm.models.gated_delta import compute_g

    B, S, _ = inputs.shape
    qkv, z, b, a = _gdn_input_projections(gdn, inputs)
    z = z.reshape(B, S, gdn.num_v_heads, gdn.head_v_dim)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (B, gdn.conv_kernel_size - 1, gdn.conv_dim),
            dtype=inputs.dtype,
        )

    conv_capture = None
    if _env_enabled("MTPLX_LINEAR_CONV1D_CAPTURE"):
        conv_capture = _linear_conv1d_capture(qkv, conv_state, gdn.conv1d.weight)
    if conv_capture is None:
        conv_capture = _stock_conv1d_capture(qkv, conv_state, gdn)
    conv_out, conv_states = conv_capture
    backend = resolve_gdn_capture_backend(capture_backend)

    state = cache[1] if cache and cache[1] is not None else None
    if state is None:
        state = mx.zeros(
            (B, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), dtype=mx.float32
        )

    final_only_capture = False
    capture_start = 0
    if backend == "linear_gdn_from_conv_inline_g":
        delta_result = _linear_gated_delta_from_conv_inline_g_capture(
            conv_out,
            a,
            b,
            state,
            gdn,
        )
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, states = delta_result
    elif backend == "linear_gdn_from_conv_tape":
        beta = mx.sigmoid(b)
        g = compute_g(gdn.A_log, a, gdn.dt_bias)
        delta_result = _linear_gated_delta_from_conv_tape_capture(
            conv_out,
            g,
            beta,
            state,
            gdn,
        )
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, final_state, tape = delta_result
        states = final_state[:, None, :, :, :]
    elif backend in {
        "linear_gdn_from_conv_stream",
        "linear_gdn_from_conv_stream_skip0",
    }:
        beta = mx.sigmoid(b)
        g = compute_g(gdn.A_log, a, gdn.dt_bias)
        capture_start = 1 if backend == "linear_gdn_from_conv_stream_skip0" else 0
        delta_result = _linear_gated_delta_from_conv_stream_capture(
            conv_out,
            g,
            beta,
            state,
            gdn,
            capture_start=capture_start,
        )
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, states = delta_result
    elif backend in {"linear_gdn", "linear_gdn_from_conv"}:
        use_from_conv = backend == "linear_gdn_from_conv" or os.environ.get(
            "MTPLX_LINEAR_GDN_FROM_CONV", ""
        ).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        beta = mx.sigmoid(b)
        g = compute_g(gdn.A_log, a, gdn.dt_bias)
        if use_from_conv:
            delta_result = _linear_gated_delta_from_conv_capture(
                conv_out, g, beta, state, gdn
            )
        else:
            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
                    [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                    [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
                )
            ]
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
            delta_result = _linear_gated_delta_capture(q, k, v, g, beta, state)
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, states = delta_result
    elif backend == "linear_gdn_final":
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
                [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        beta = mx.sigmoid(b)
        g = compute_g(gdn.A_log, a, gdn.dt_bias)
        delta_result = _linear_gated_delta_final(q, k, v, g, beta, state)
        if delta_result is None:
            return gdn(inputs, mask=mask, cache=cache), None
        out, final_state = delta_result
        states = final_state[:, None, :, :, :]
        final_only_capture = True
    else:
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
                [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
                [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, states = _stock_gated_delta_capture(q, k, v, a, b, state, mask, gdn)

    if cache is not None:
        cache[0] = mx.contiguous(conv_states[:, -1, :, :])
        cache[1] = _maybe_contiguous_authoritative_gdn_leaf(states[:, -1, :, :, :])
        cache.advance(S)

    tail_projected = False
    if os.environ.get("MTPLX_NATIVE_GDN_TAIL", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        from .kernels.native_gdn_tail import native_gdn_norm_gate_out_qmv8

        out = native_gdn_norm_gate_out_qmv8(
            out,
            z,
            gdn.norm.weight,
            gdn.norm.eps,
            gdn.out_proj,
            num_simdgroups=int(os.environ.get("MTPLX_NATIVE_GDN_TAIL_SIMDGROUPS") or 2),
        )
        tail_projected = True
    elif os.environ.get("MTPLX_FUSE_GDN_NORM_GATE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        from .kernel_selfcheck import lane_disabled
        from .kernels.fused_norm import fused_gdn_norm_gate

        if lane_disabled("fused_gdn_norm_gate"):
            out = gdn.norm(out, z)
        else:
            out = fused_gdn_norm_gate(out, z, gdn.norm.weight, gdn.norm.eps)
    else:
        out = gdn.norm(out, z)
    if not tail_projected:
        out = out.reshape(B, S, -1)
        if os.environ.get("MTPLX_GDN_OUT_QMV8", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            from .verify_qmv import stocklike_qmv8_matmul

            out = stocklike_qmv8_matmul(out, gdn.out_proj)
        else:
            out = gdn.out_proj(out)
    if final_only_capture:
        return out, {"final_only": True}
    if backend == "linear_gdn_from_conv_tape":
        return out, {
            "conv_states": conv_states,
            "conv_out": conv_out,
            "g": g,
            "state_in": state,
            "tape": tape,
            "gdn_meta": _gdn_tape_meta(gdn),
        }
    if capture_start:
        return out, {
            "conv_states": conv_states[:, capture_start:, :, :],
            "states": states,
            "capture_start": capture_start,
        }
    return out, {"conv_states": conv_states, "states": states}


def _a3b_gdn_forward_with_fixed_postconv(
    gdn: Any,
    inputs: mx.array,
    cache: Any,
    postconv_implementation: Callable[..., Any],
):
    """Build the unchecked exact A3B GDN graph with stock surroundings."""
    B, S, _ = inputs.shape
    qkv = gdn.in_proj_qkv(inputs)
    z = gdn.in_proj_z(inputs).reshape(B, S, 32, 128)
    b = gdn.in_proj_b(inputs)
    a = gdn.in_proj_a(inputs)
    conv_state = cache[0]
    conv_out, conv_states = _stock_conv1d_capture(qkv, conv_state, gdn)
    out, states = postconv_implementation(conv_out, a, b, cache[1])
    cache[0] = mx.contiguous(conv_states[:, -1, :, :])
    cache[1] = states[:, -1, :, :, :]
    out = gdn.norm(out, z)
    out = gdn.out_proj(out.reshape(B, S, -1))
    return out, {"conv_states": conv_states, "states": states}


def forward_with_a3b_gdn_postconv_capture(
    model: Any,
    inputs: mx.array,
    cache: list[Any],
    *,
    hidden_variant: str | None,
    postconv_implementations: tuple[Callable[..., Any], ...],
):
    """Build the unchecked exact 40-layer A3B target trace."""
    text_model = model.language_model
    inner = text_model.model
    hidden_states = inner.embed_tokens(inputs)

    from mlx_lm.models.base import create_attention_mask

    attention_mask = create_attention_mask(hidden_states, cache[3])
    captures: dict[int, dict[str, mx.array]] = {}
    implementation_iter = iter(postconv_implementations)
    for layer_idx, (layer, layer_cache, kind) in enumerate(
        zip(inner.layers, cache, _A3B_GDN_POSTCONV_LAYER_TYPES)
    ):
        normed = layer.input_layernorm(hidden_states)
        if kind == "linear_attention":
            r, capture = _a3b_gdn_forward_with_fixed_postconv(
                layer.linear_attn,
                normed,
                layer_cache,
                next(implementation_iter),
            )
            captures[layer_idx] = capture
        else:
            r = layer.self_attn(normed, mask=attention_mask, cache=layer_cache)
        h = hidden_states + r
        mlp_input = layer.post_attention_layernorm(h)
        hidden_states = h + layer.mlp(mlp_input)

    pre_norm = hidden_states
    post_norm = inner.norm(hidden_states)
    logits = (
        inner.embed_tokens.as_linear(post_norm)
        if text_model.args.tie_word_embeddings
        else text_model.lm_head(post_norm)
    )
    hidden = pre_norm if hidden_variant == "pre_norm" else post_norm
    return logits, hidden, captures


def forward_with_gdn_capture(
    model: Any,
    inputs: mx.array,
    cache=None,
    return_hidden: bool = False,
    *,
    hidden_variant: str | None = None,
    capture_backend: str | None = None,
):
    text_model = getattr(model, "language_model", model)
    inner = text_model.model
    layers = tuple(inner.layers)
    hybrid_metadata = hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx")
    if not hybrid_metadata:
        if any(bool(getattr(layer, "is_linear", False)) for layer in layers):
            raise RuntimeError(
                "hybrid capture target is missing fa_idx/ssm_idx metadata"
            )
        result = text_model(
            inputs,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
        )
        if return_hidden:
            logits, hidden = result
            return logits, hidden, {}
        return result, {}
    hidden_states = inner.embed_tokens(inputs)
    if cache is None:
        cache = [None] * len(layers)

    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
    captures: dict[int, dict[str, mx.array]] = {}
    backend = resolve_gdn_capture_backend(capture_backend)
    context_len = _cache_context_len(cache)
    layer_eval_every = _target_layer_eval_every(context_len)
    layer_eval_threshold = int(
        os.environ.get("MTPLX_TARGET_LAYER_EVAL_CONTEXT_THRESHOLD", "0") or "0"
    )
    layer_eval_max_q = int(os.environ.get("MTPLX_TARGET_LAYER_EVAL_MAX_Q", "8") or "8")
    layer_eval_enabled = (
        layer_eval_every > 0
        and int(inputs.shape[1]) <= max(1, layer_eval_max_q)
        and context_len >= max(0, layer_eval_threshold)
    )

    for layer_idx, (layer, layer_cache) in enumerate(zip(layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        normed = layer.input_layernorm(hidden_states)
        if layer.is_linear:
            if os.environ.get("MTPLX_ABLATE_LINEAR_ATTN", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                r = mx.zeros_like(normed)
            else:
                r, capture = gdn_forward_with_capture(
                    layer.linear_attn,
                    normed,
                    mask=mask,
                    cache=layer_cache,
                    capture_backend=backend,
                )
                if capture is not None:
                    if capture.get("final_only"):
                        captures["__final_only__"] = True
                    else:
                        captures[layer_idx] = capture
        else:
            r = layer.self_attn(normed, mask=mask, cache=layer_cache)
        if os.environ.get("MTPLX_FUSE_POST_NORM_RESIDUAL", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            from .kernel_selfcheck import lane_disabled
            from .kernels.fused_norm import fused_add_rmsnorm

            if lane_disabled("fused_add_rmsnorm"):
                h = hidden_states + r
                mlp_input = layer.post_attention_layernorm(h)
            else:
                h, mlp_input = fused_add_rmsnorm(
                    hidden_states,
                    r,
                    layer.post_attention_layernorm.weight,
                    layer.post_attention_layernorm.eps,
                    threadgroup_size=512,
                )
        else:
            h = hidden_states + r
            mlp_input = layer.post_attention_layernorm(h)
        hidden_states = h + layer.mlp(mlp_input)
        if layer_eval_enabled and (layer_idx + 1) % layer_eval_every == 0:
            mx.eval(hidden_states)

    pre_norm = hidden_states
    post_norm = inner.norm(hidden_states)
    logits = (
        inner.embed_tokens.as_linear(post_norm)
        if text_model.args.tie_word_embeddings
        else text_model.lm_head(post_norm)
    )
    if return_hidden:
        hidden = pre_norm if hidden_variant == "pre_norm" else post_norm
        return logits, hidden, captures
    return logits, captures


@dataclass(frozen=True)
class _StockCapturedLayerCommit:
    layer_index: int
    own_conv: Callable[[Any], Any]
    own_gdn: Callable[[Any], Any]
    replace_state: Callable[[Any, list[Any]], None]

    def __call__(
        self,
        cache: list[Any],
        captures: dict[int, dict[str, mx.array]],
        capture_index: int,
    ) -> None:
        capture = captures[self.layer_index]
        conv_state = self.own_conv(
            capture["conv_states"][:, capture_index, :, :]
        )
        gdn_state = self.own_gdn(
            capture["states"][:, capture_index, :, :, :]
        )
        self.replace_state(cache[self.layer_index], [conv_state, gdn_state])


@dataclass(frozen=True)
class CapturedPrefixCommitPlan:
    """Construction-qualified direct commit for stock GDN captures.

    Cache ownership, layer kinds, capture schema, shapes, dtypes, backend, and
    detach policy are proven before installation.  Runtime keep/verify widths
    genuinely vary (the final depth-3 cycle can be shorter), so commit uses
    those two values directly without rechecking the installed invariants.
    """

    max_verified_tokens: int
    _recurrent_commits: tuple[_StockCapturedLayerCommit, ...]
    trimmable_layer_indices: tuple[int, ...]

    @property
    def recurrent_layer_indices(self) -> tuple[int, ...]:
        return tuple(route.layer_index for route in self._recurrent_commits)

    def commit(
        self,
        cache: list[Any],
        captures: dict[int, dict[str, mx.array]],
        *,
        keep_tokens: int,
        verified_tokens: int,
    ) -> None:
        capture_index = keep_tokens - 1
        trim_tokens = verified_tokens - keep_tokens
        for route in self._recurrent_commits:
            route(cache, captures, capture_index)
        for layer_index in self.trimmable_layer_indices:
            cache[layer_index].trim(trim_tokens)


def qualify_captured_prefix_commit(
    cache: list[Any],
    captures: dict[int, dict[str, mx.array]],
    *,
    max_verified_tokens: int,
    capture_backend: str,
    detach_components: set[str],
) -> CapturedPrefixCommitPlan:
    """Fail closed before measurement and return a direct stock committer."""

    if capture_backend != "stock":
        raise RuntimeError("direct commit requires the stock capture backend")
    if detach_components:
        raise RuntimeError("direct commit does not permit capture detach components")
    if max_verified_tokens < 1:
        raise RuntimeError("direct commit max_verified_tokens must be positive")
    if captures.get("__final_only__"):
        raise RuntimeError("direct commit cannot install from a final-only capture")

    from .cache_state import replace_recurrent_cache_state

    recurrent: list[_StockCapturedLayerCommit] = []
    trimmable: list[int] = []
    for layer_index, entry in enumerate(cache):
        is_trimmable = getattr(entry, "is_trimmable", None)
        if callable(is_trimmable) and bool(is_trimmable()):
            if layer_index in captures:
                raise RuntimeError(
                    f"unexpected capture for trimmable layer {layer_index}"
                )
            if not callable(getattr(entry, "trim", None)):
                raise RuntimeError(
                    f"trimmable layer {layer_index} has no fixed trim operation"
                )
            trimmable.append(layer_index)
            continue

        state = getattr(entry, "state", None)
        if not isinstance(state, (list, tuple)) or len(state) != 2:
            raise RuntimeError(
                f"cache layer {layer_index} is neither trimmable nor recurrent"
            )
        capture = captures.get(layer_index)
        if capture is None:
            raise RuntimeError(f"missing capture for recurrent layer {layer_index}")
        if not isinstance(capture, dict) or set(capture) != {
            "conv_states",
            "states",
        }:
            raise RuntimeError(
                f"recurrent layer {layer_index} does not use the stock capture schema"
            )
        if not all(isinstance(value, mx.array) for value in state):
            raise RuntimeError(
                f"recurrent cache layer {layer_index} has non-array state"
            )
        conv_states = capture["conv_states"]
        gdn_states = capture["states"]
        if not isinstance(conv_states, mx.array) or not isinstance(gdn_states, mx.array):
            raise RuntimeError(
                f"recurrent layer {layer_index} capture leaves are not arrays"
            )
        if len(conv_states.shape) != 4 or len(gdn_states.shape) != 5:
            raise RuntimeError(
                f"recurrent layer {layer_index} capture ranks are invalid"
            )
        if (
            int(conv_states.shape[1]) != max_verified_tokens
            or int(gdn_states.shape[1]) != max_verified_tokens
        ):
            raise RuntimeError(
                f"recurrent layer {layer_index} capture width does not match "
                f"{max_verified_tokens}"
            )
        if (
            tuple(conv_states.shape[:1] + conv_states.shape[2:])
            != tuple(state[0].shape)
            or tuple(gdn_states.shape[:1] + gdn_states.shape[2:])
            != tuple(state[1].shape)
        ):
            raise RuntimeError(
                f"recurrent layer {layer_index} capture shapes do not match cache state"
            )
        if conv_states.dtype != state[0].dtype or gdn_states.dtype != state[1].dtype:
            raise RuntimeError(
                f"recurrent layer {layer_index} capture dtypes do not match cache state"
            )
        recurrent.append(
            _StockCapturedLayerCommit(
                layer_index=layer_index,
                own_conv=mx.contiguous,
                own_gdn=_contiguous_recurrent_leaf,
                replace_state=replace_recurrent_cache_state,
            )
        )

    capture_layers = {key for key in captures if isinstance(key, int)}
    recurrent_layers = {route.layer_index for route in recurrent}
    if capture_layers != recurrent_layers:
        unexpected = sorted(capture_layers - recurrent_layers)
        raise RuntimeError(f"unexpected recurrent capture layers: {unexpected}")
    if not recurrent:
        raise RuntimeError("direct commit requires at least one recurrent layer")
    return CapturedPrefixCommitPlan(
        max_verified_tokens=int(max_verified_tokens),
        _recurrent_commits=tuple(recurrent),
        trimmable_layer_indices=tuple(trimmable),
    )


def _forward_with_gdn_capture_impl(
    model: Any,
    inputs: mx.array,
    cache: list[Any],
    *,
    config: QwenGDNVerifyConfig,
) -> tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
    del model
    hidden_states = config.embed_inputs(inputs)
    fa_mask = config.create_fa_mask(hidden_states, cache)
    ssm_mask = config.create_ssm_mask(hidden_states, cache)
    captures: dict[int, dict[str, mx.array]] = {}
    context_len = config.cache_context_length(cache)
    layer_eval_every = config.eval_every(context_len)
    layer_eval_enabled = (
        layer_eval_every > 0
        and int(inputs.shape[1]) <= max(1, config.layer_eval_max_q)
        and context_len >= max(0, config.layer_eval_context_threshold)
    )

    for layer_idx, (route, layer_cache) in enumerate(
        zip(config.layer_routes, cache)
    ):
        hidden_states, capture = route(
            hidden_states,
            fa_mask,
            ssm_mask,
            layer_cache,
        )
        if capture is not None:
            captures[layer_idx] = capture
        if layer_eval_enabled and (layer_idx + 1) % layer_eval_every == 0:
            mx.eval(hidden_states)

    pre_norm = hidden_states
    post_norm = config.final_norm(hidden_states)
    logits = config.project_logits(post_norm)
    hidden = pre_norm if config.hidden_variant == "pre_norm" else post_norm
    return logits, hidden, captures


def forward_with_gdn_capture_configured(
    model: Any,
    inputs: mx.array,
    cache: list[Any],
    *,
    config: QwenGDNVerifyConfig,
) -> tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
    return _forward_with_gdn_capture_impl(
        model,
        inputs,
        cache,
        config=config,
    )


def _contains_captured_array(value: Any) -> bool:
    if isinstance(value, mx.array):
        return True
    if isinstance(value, Mapping):
        return any(_contains_captured_array(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_captured_array(item) for item in value)
    return False


def _extract_captured_value(value: Any, row: int) -> Any:
    if isinstance(value, mx.array):
        if value.ndim == 0:
            return value
        owned = mx.contiguous(value[row : row + 1])
        mx.eval(owned)
        return owned
    if isinstance(value, Mapping):
        if not _contains_captured_array(value):
            return value
        return {
            key: _extract_captured_value(item, row)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        if not _contains_captured_array(value):
            return value
        return tuple(_extract_captured_value(item, row) for item in value)
    if isinstance(value, list):
        if not _contains_captured_array(value):
            return value
        return [_extract_captured_value(item, row) for item in value]
    return value


def extract_captured_row(
    captures: Mapping[Any, Any],
    row: int,
) -> dict[Any, Any]:
    """Own every batch-array leaf before a request-local capture commit."""
    return {
        key: _extract_captured_value(value, row)
        for key, value in captures.items()
    }


def _extract_captured_value_lazy(value: Any, row: int) -> Any:
    if isinstance(value, mx.array):
        if value.ndim == 0:
            return value
        source = value[row : row + 1]
        owned = mx.zeros(source.shape, dtype=source.dtype)
        owned[tuple(slice(None) for _ in source.shape)] = source
        return owned
    if isinstance(value, Mapping):
        if not _contains_captured_array(value):
            return value
        return {
            key: _extract_captured_value_lazy(item, row)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        if not _contains_captured_array(value):
            return value
        return tuple(_extract_captured_value_lazy(item, row) for item in value)
    if isinstance(value, list):
        if not _contains_captured_array(value):
            return value
        return [_extract_captured_value_lazy(item, row) for item in value]
    return value


def extract_captured_row_lazy(
    captures: Mapping[Any, Any],
    row: int,
) -> dict[Any, Any]:
    """Own one capture row without evaluating its lazy copy graph."""
    return {
        key: _extract_captured_value_lazy(value, row)
        for key, value in captures.items()
    }


def commit_captured_prefix(
    cache: list[Any],
    captures: dict[int, dict[str, mx.array]],
    keep_tokens: int,
    verified_tokens: int,
    *,
    detach_components: set[str] | None = None,
    detach_mode: str = "selected_slice_contiguous_eval",
    detach_stats: dict[str, int] | None = None,
) -> bool:
    if keep_tokens <= 0 or keep_tokens > verified_tokens:
        return False
    if captures.get("__final_only__"):
        return False
    detach_requested = {
        item.strip().lower().replace("-", "_")
        for item in (detach_components or set())
        if item
    }
    trim_tokens = verified_tokens - keep_tokens
    capture_index = keep_tokens - 1
    for capture in captures.values():
        if isinstance(capture, dict):
            capture_start = int(capture.get("capture_start", 0))
            if capture_index - capture_start < 0:
                return False
    for layer_idx, entry in enumerate(cache):
        capture = captures.get(layer_idx)
        if capture is not None and hasattr(entry, "state"):
            capture_start = int(capture.get("capture_start", 0))
            adjusted_index = capture_index - capture_start
            conv_state = mx.contiguous(capture["conv_states"][:, adjusted_index, :, :])
            if "conv" in detach_requested:
                from .cache_state import detach_array_leaf

                conv_state = detach_array_leaf(conv_state, mode=detach_mode)
                if detach_stats is not None:
                    detach_stats["arrays"] = int(detach_stats.get("arrays", 0)) + 1
                    detach_stats["bytes"] = int(detach_stats.get("bytes", 0)) + int(
                        conv_state.nbytes
                    )
            if "tape" in capture:
                replayed_state = _linear_gated_delta_from_conv_tape_replay(
                    capture["tape"],
                    capture["conv_out"],
                    capture["g"],
                    capture["state_in"],
                    capture.get("gdn_meta", capture.get("gdn")),
                    steps=capture_index + 1,
                    tgy=capture.get("tape_replay_tgy"),
                )
                if replayed_state is None:
                    return False
                gdn_state = _maybe_contiguous_authoritative_gdn_leaf(replayed_state)
            else:
                gdn_state = _contiguous_recurrent_leaf(
                    capture["states"][:, adjusted_index, :, :, :]
                )
            if "gdn" in detach_requested:
                from .cache_state import detach_array_leaf

                gdn_state = detach_array_leaf(gdn_state, mode=detach_mode)
                if detach_stats is not None:
                    detach_stats["arrays"] = int(detach_stats.get("arrays", 0)) + 1
                    detach_stats["bytes"] = int(detach_stats.get("bytes", 0)) + int(
                        gdn_state.nbytes
                    )
            from .cache_state import replace_recurrent_cache_state

            replace_recurrent_cache_state(entry, [conv_state, gdn_state])
        elif trim_tokens and hasattr(entry, "is_trimmable") and entry.is_trimmable():
            entry.trim(trim_tokens)
    return True
