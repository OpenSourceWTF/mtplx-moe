"""Resident-only MLX model construction for streamed MoE checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .expert_manifest import (
    ExpertManifest,
    ExpertManifestError,
    resolve_artifact_member,
)
from .expert_runtime import ExpertStreamingRuntime
from .expert_streaming_models import proj_quant_covers


class ResidentLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResidentLoadReport:
    shard_count: int
    tensor_count: int
    raw_tensor_bytes: int
    evaluated_parameter_count: int
    bound_sparse_layers: int
    strict: bool
    proj_quant: str | None = None
    proj_quantized_modules: int = 0

    def as_dict(self) -> dict[str, int | bool | str | None]:
        return {
            "shard_count": self.shard_count,
            "tensor_count": self.tensor_count,
            "raw_tensor_bytes": self.raw_tensor_bytes,
            "evaluated_parameter_count": self.evaluated_parameter_count,
            "bound_sparse_layers": self.bound_sparse_layers,
            "strict": self.strict,
            "proj_quant": self.proj_quant,
            "proj_quantized_modules": self.proj_quantized_modules,
        }


@dataclass(frozen=True)
class ResidentModel:
    model: Any
    config: dict[str, Any]
    report: ResidentLoadReport


def _dtype_name(value: Any) -> str:
    text = str(getattr(value, "dtype", ""))
    name = text.rsplit(".", 1)[-1].upper()
    return {
        "BOOL_": "BOOL",
        "INT8": "I8",
        "UINT8": "U8",
        "INT16": "I16",
        "UINT16": "U16",
        "FLOAT16": "F16",
        "BFLOAT16": "BF16",
        "INT32": "I32",
        "UINT32": "U32",
        "FLOAT32": "F32",
        "INT64": "I64",
        "UINT64": "U64",
        "FLOAT64": "F64",
    }.get(name, name)


def load_resident_arrays(
    root: Path | str,
    manifest: ExpertManifest,
    *,
    mx_module: Any | None = None,
) -> dict[str, Any]:
    """Create lazy MLX arrays for only manifest-allowlisted resident tensors."""

    if mx_module is None:
        try:
            import mlx.core as mx
        except Exception as exc:
            raise ResidentLoadError(
                f"MLX is required for resident loading: {exc}"
            ) from exc
    else:
        mx = mx_module
    artifact_root = Path(root).resolve()
    by_shard: dict[str, list[Any]] = {}
    for tensor in manifest.resident_tensors:
        by_shard.setdefault(tensor.shard, []).append(tensor)
    selected: dict[str, Any] = {}
    for shard_name, expected_tensors in sorted(by_shard.items()):
        shard_path = resolve_artifact_member(artifact_root, shard_name)
        try:
            # Hugging Face cache blobs are content-addressed and extensionless,
            # so format inference fails after secure symlink resolution.  The
            # manifest admits safetensors shards only; make that contract
            # explicit to MLX.
            loaded = mx.load(str(shard_path), format="safetensors")
        except Exception as exc:
            raise ResidentLoadError(
                f"could not lazily load {shard_name}: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise ResidentLoadError(f"MLX returned a non-dictionary for {shard_name}")
        for expected in expected_tensors:
            try:
                value = loaded[expected.tensor]
            except KeyError as exc:
                raise ResidentLoadError(
                    f"resident tensor {expected.tensor} is missing from {shard_name}"
                ) from exc
            shape = tuple(int(dimension) for dimension in value.shape)
            if shape != expected.shape:
                raise ResidentLoadError(
                    f"resident tensor {expected.tensor} shape {shape} != {expected.shape}"
                )
            dtype = _dtype_name(value)
            if dtype != expected.dtype:
                raise ResidentLoadError(
                    f"resident tensor {expected.tensor} dtype {dtype} != {expected.dtype}"
                )
            if int(value.nbytes) != expected.length:
                raise ResidentLoadError(
                    f"resident tensor {expected.tensor} bytes {value.nbytes} != {expected.length}"
                )
            if expected.tensor in selected:
                raise ResidentLoadError(f"duplicate resident tensor {expected.tensor}")
            selected[expected.tensor] = value
        # Routed arrays returned by mx.load stay lazy and become unreachable
        # here; only the selected resident leaves survive to mx.eval.
        del loaded
    if len(selected) != len(manifest.resident_tensors):
        raise ResidentLoadError("resident allowlist was not loaded completely")
    return selected


def get_streaming_model_classes(config: dict[str, Any]) -> tuple[type, type]:
    model_type = str(config.get("model_type") or "")
    if model_type == "hy_v3":
        from .models.hy3_mlx import Model, ModelArgs

        return Model, ModelArgs
    if model_type == "glm_moe_dsa":
        from .models.glm52_mlx import Model, ModelArgs

        return Model, ModelArgs
    raise ResidentLoadError(f"no streamed model overlay for model_type={model_type!r}")


def _quantize_resident_model(
    model: Any,
    config: dict[str, Any],
    weights: dict[str, Any],
) -> None:
    try:
        import mlx.nn as nn
    except Exception as exc:
        raise ResidentLoadError(
            f"MLX NN is required for quantized loading: {exc}"
        ) from exc
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        return
    try:
        default_group_size = int(quantization["group_size"])
        default_bits = int(quantization["bits"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResidentLoadError("invalid affine quantization config") from exc
    default_mode = str(quantization.get("mode", "affine"))

    def predicate(path: str, module: Any) -> bool | dict[str, Any]:
        if not hasattr(module, "to_quantized"):
            return False
        override = quantization.get(path)
        if override is not None:
            if not isinstance(override, dict):
                raise ResidentLoadError(
                    f"quantization override {path!r} must be an object"
                )
            return dict(override)
        return f"{path}.scales" in weights

    nn.quantize(
        model,
        group_size=default_group_size,
        bits=default_bits,
        mode=default_mode,
        class_predicate=predicate,
    )


_PROJ_QUANT_BITS = {"q8": 8, "q4": 4}


def _runtime_quantize_projections(model: Any, mode: str) -> list[str]:
    """Quantize the bandwidth-dominant trunk ``*_proj`` Linears after a BF16 load.

    Scope comes from ``proj_quant_covers`` — the memory plan discounts the
    same tensors, so the two must never diverge. Router gates, embeddings,
    the LM head, norms, and MTP glue keep their loaded precision.
    """

    try:
        import mlx.nn as nn
    except Exception as exc:
        raise ResidentLoadError(
            f"MLX NN is required for projection quantization: {exc}"
        ) from exc
    bits = _PROJ_QUANT_BITS[mode]
    quantized: list[str] = []

    def predicate(path: str, module: Any) -> bool:
        if not isinstance(module, nn.Linear) or isinstance(
            module, nn.QuantizedLinear
        ):
            return False
        if proj_quant_covers(path):
            quantized.append(path)
            return True
        return False

    nn.quantize(
        model, group_size=64, bits=bits, mode="affine", class_predicate=predicate
    )
    if not quantized:
        raise ResidentLoadError(
            f"proj_quant={mode!r} matched no trunk *_proj Linear modules"
        )
    return quantized


def _verify_kv_quant_honored(model: Any, kv_quant: str) -> None:
    """Reject models whose make_cache silently ignores _mtplx_kv_quant.

    The attribute is honored per-model overlay; an ignored kv_quant would
    desync the memory plan's discounted KV pricing from reality.
    """

    try:
        from mlx_lm.models.cache import QuantizedKVCache

        probe_cache = model.make_cache()
    except Exception as exc:
        raise ResidentLoadError(
            f"kv_quant={kv_quant!r} probe failed: {exc}"
        ) from exc
    if not any(isinstance(entry, QuantizedKVCache) for entry in probe_cache):
        raise ResidentLoadError(
            f"kv_quant={kv_quant!r} requested but "
            f"{type(model).__name__}.make_cache ignores it"
        )


def construct_resident_model(
    root: Path | str,
    runtime: ExpertStreamingRuntime,
    *,
    config: dict[str, Any] | None = None,
    mx_module: Any | None = None,
    model_class_resolver: Callable[[dict[str, Any]], tuple[type, type]] | None = None,
    strict: bool = True,
) -> ResidentModel:
    """Instantiate, bind, strictly load, and evaluate only resident parameters."""

    artifact_root = Path(root).resolve()
    if config is None:
        try:
            from mlx_lm.utils import load_config

            config = load_config(artifact_root)
        except Exception as exc:
            raise ResidentLoadError(f"could not load model config: {exc}") from exc
    config = dict(config)
    if str(config.get("model_type") or "") not in {"hy_v3", "glm_moe_dsa"}:
        raise ResidentLoadError(
            "resident streaming supports only hy_v3 and glm_moe_dsa"
        )
    resolver = model_class_resolver or get_streaming_model_classes
    model_class, args_class = resolver(config)
    try:
        model_args = args_class.from_dict(config)
        model = model_class(model_args)
    except Exception as exc:
        raise ResidentLoadError(f"could not construct streamed model: {exc}") from exc
    from .models.expert_mlx import bind_streamed_switches

    try:
        bound = bind_streamed_switches(model, runtime)
    except Exception as exc:
        raise ResidentLoadError(
            f"could not bind streamed expert layers: {exc}"
        ) from exc
    if bound != runtime.spec.routed_layer_count:
        raise ResidentLoadError(
            f"bound {bound} sparse layers; expected {runtime.spec.routed_layer_count}"
        )
    weights = load_resident_arrays(artifact_root, runtime.manifest, mx_module=mx_module)
    try:
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)
        _quantize_resident_model(model, config, weights)
        model.eval()
        model.load_weights(list(weights.items()), strict=strict)
    except Exception as exc:
        raise ResidentLoadError(f"resident parameter validation failed: {exc}") from exc
    proj_quant = getattr(runtime.config, "proj_quant", None)
    quantized_paths: list[str] = []
    if proj_quant:
        # Drop the loader's references first so each replaced BF16 weight
        # frees as its quantized module lands, instead of at function exit.
        weights.clear()
        quantized_paths = _runtime_quantize_projections(model, proj_quant)
    if mx_module is None:
        import mlx.core as mx
    else:
        mx = mx_module
    try:
        parameters = model.parameters()
        mx.eval(parameters)
    except Exception as exc:
        raise ResidentLoadError(f"resident parameter evaluation failed: {exc}") from exc
    parameter_count = sum(1 for _name, _value in _flatten_tree(parameters))
    report = ResidentLoadReport(
        shard_count=len({tensor.shard for tensor in runtime.manifest.resident_tensors}),
        tensor_count=len(runtime.manifest.resident_tensors),
        raw_tensor_bytes=runtime.manifest.resident_tensor_bytes,
        evaluated_parameter_count=parameter_count,
        bound_sparse_layers=bound,
        strict=strict,
        proj_quant=proj_quant,
        proj_quantized_modules=len(quantized_paths),
    )
    setattr(model, "_mtplx_expert_runtime", runtime)
    setattr(model, "_mtplx_resident_load_report", report.as_dict())
    kv_quant = getattr(runtime.config, "kv_quant", None)
    if kv_quant:
        setattr(model, "_mtplx_kv_quant", kv_quant)
        _verify_kv_quant_honored(model, kv_quant)
    return ResidentModel(model=model, config=config, report=report)


def _flatten_tree(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_tree(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from _flatten_tree(child, child_prefix)
    else:
        yield prefix, value


def assert_no_routed_weights_loaded(
    weights: dict[str, Any],
    manifest: ExpertManifest,
) -> None:
    routed_names = {
        segment.tensor for record in manifest.records for segment in record.segments
    }
    overlap = routed_names & set(weights)
    if overlap:
        raise ExpertManifestError(
            f"resident loader materialized routed tensors: {sorted(overlap)[:4]}"
        )
