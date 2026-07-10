"""Model-independent orchestration for bounded SSD expert streaming."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .expert_io import PositionalExpertReader
from .expert_manifest import (
    ExpertManifest,
    ExpertManifestError,
    load_expert_manifest,
    verify_expert_manifest,
)
from .expert_slots import ExpertSlotError, ExpertSlotPool, ReadyRoute
from .expert_streaming import (
    CacheCounters,
    LayerExpertSlotBank,
    RoutingPhase,
)
from .expert_streaming_models import (
    ExpertMemoryPlan,
    ExpertStreamingModelSpec,
    get_model_spec,
    plan_expert_memory,
)


_MEMORY_RE = re.compile(r"^([0-9]+)([kmgt]i?b?|b)?$", re.IGNORECASE)


class ExpertStreamingConfigurationError(ValueError):
    pass


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def parse_memory_bytes(value: str | int) -> int:
    if isinstance(value, bool):
        raise ExpertStreamingConfigurationError("memory size must not be bool")
    if isinstance(value, int):
        if value <= 0:
            raise ExpertStreamingConfigurationError("memory size must be positive")
        return value
    if not isinstance(value, str):
        raise ExpertStreamingConfigurationError(
            "memory size must be bytes or a suffixed string"
        )
    normalized = value.strip().lower()
    match = _MEMORY_RE.fullmatch(normalized)
    if match is None:
        raise ExpertStreamingConfigurationError(f"invalid memory size {value!r}")
    number = int(match.group(1))
    suffix = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    result = number * multipliers[suffix]
    if result <= 0:
        raise ExpertStreamingConfigurationError("memory size must be positive")
    return result


@dataclass(frozen=True)
class ExpertStreamingConfig:
    model_key: str
    memory_limit_bytes: int
    max_live_kv_tokens: int
    runtime_reserve_bytes: int = 16 * 1024**3
    expert_cache_limit_bytes: int | None = None
    transient_slots: int | None = None
    io_staging_bytes: int = 0
    execution_workspace_bytes: int = 0
    max_inflight_io_bytes: int | None = None
    max_open_files: int = 16
    max_read_chunk_bytes: int = 8 * 1024 * 1024
    frequency_decay: float = 0.995
    prefer_sidecar: bool = True
    verify_record_hashes: bool = True
    verify_artifact_headers: bool = True
    prefill_admission: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_key, str) or not self.model_key:
            raise TypeError("model_key must be a non-empty string")
        for name, minimum in (
            ("memory_limit_bytes", 1),
            ("max_live_kv_tokens", 0),
            ("runtime_reserve_bytes", 0),
            ("io_staging_bytes", 0),
            ("execution_workspace_bytes", 0),
            ("max_open_files", 1),
            ("max_read_chunk_bytes", 1),
        ):
            object.__setattr__(
                self, name, _integer(name, getattr(self, name), minimum=minimum)
            )
        for name in (
            "expert_cache_limit_bytes",
            "transient_slots",
            "max_inflight_io_bytes",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _integer(name, value, minimum=0))
        if self.max_inflight_io_bytes == 0:
            raise ValueError("max_inflight_io_bytes must be positive when supplied")
        if isinstance(self.frequency_decay, bool):
            raise TypeError("frequency_decay must be numeric")
        decay = float(self.frequency_decay)
        if not 0.0 < decay <= 1.0:
            raise ValueError("frequency_decay must be in (0, 1]")
        object.__setattr__(self, "frequency_decay", decay)
        for name in (
            "prefer_sidecar",
            "verify_record_hashes",
            "verify_artifact_headers",
            "prefill_admission",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.prefill_admission:
            raise ValueError(
                "prefill admission is not implemented; prefill must use transient slots"
            )

    def memory_plan(self, spec: ExpertStreamingModelSpec) -> ExpertMemoryPlan:
        return plan_expert_memory(
            spec,
            total_limit_bytes=self.memory_limit_bytes,
            context_tokens=self.max_live_kv_tokens,
            runtime_reserve_bytes=self.runtime_reserve_bytes,
            expert_cache_limit_bytes=self.expert_cache_limit_bytes,
            transient_slots=self.transient_slots,
            io_staging_bytes=self.io_staging_bytes,
            execution_workspace_bytes=self.execution_workspace_bytes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class RouteWave:
    positions: tuple[int, ...]
    experts: tuple[int, ...]


def partition_route_waves(
    expert_ids: Iterable[int],
    *,
    max_unique_experts: int,
) -> tuple[RouteWave, ...]:
    """Greedily partition flattened assignments into bounded expert unions."""

    capacity = _integer("max_unique_experts", max_unique_experts, minimum=1)
    experts = tuple(expert_ids)
    ordered_unique: list[int] = []
    seen: set[int] = set()
    for expert in experts:
        if isinstance(expert, bool) or not isinstance(expert, int):
            raise TypeError("expert ids must be exact integers")
        if expert not in seen:
            seen.add(expert)
            ordered_unique.append(expert)
    waves: list[RouteWave] = []
    for start in range(0, len(ordered_unique), capacity):
        selected = set(ordered_unique[start : start + capacity])
        positions = tuple(
            position for position, expert in enumerate(experts) if expert in selected
        )
        waves.append(
            RouteWave(
                positions=positions,
                experts=tuple(experts[position] for position in positions),
            )
        )
    return tuple(waves)


@dataclass
class KVAdmission:
    runtime: ExpertStreamingRuntime
    tokens: int
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.runtime.release_kv_tokens(self.tokens)
        self.released = True

    def __enter__(self) -> KVAdmission:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def reconcile_mlx_memory_cap(
    plan: ExpertMemoryPlan,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    """Resolve the MLX-owned portion and reject a conflicting env cap."""

    mlx_limit = (
        plan.total_limit_bytes - plan.runtime_reserve_bytes - plan.io_staging_bytes
    )
    if mlx_limit <= 0:
        raise ExpertStreamingConfigurationError(
            "memory plan leaves no MLX allocation budget"
        )
    source = os.environ if env is None else env
    existing = source.get("MTPLX_MEMORY_LIMIT_BYTES")
    if existing:
        parsed = parse_memory_bytes(existing)
        if parsed != mlx_limit:
            raise ExpertStreamingConfigurationError(
                "MTPLX_MEMORY_LIMIT_BYTES conflicts with expert streaming plan: "
                f"env={parsed}, planned={mlx_limit}"
            )
    return mlx_limit


def apply_mlx_memory_cap(
    plan: ExpertMemoryPlan,
    *,
    mx_module: Any | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply the reconciled cap before resident or expert-slot allocation."""

    target_env = os.environ if env is None else env
    limit = reconcile_mlx_memory_cap(plan, env=target_env)
    target_env["MTPLX_MEMORY_LIMIT_BYTES"] = str(limit)
    if mx_module is None:
        try:
            import mlx.core as mx
        except Exception as exc:
            return {
                "applied": False,
                "reason": "mlx_unavailable",
                "error": repr(exc),
                "limit": limit,
            }
    else:
        mx = mx_module
    setter = getattr(mx, "set_memory_limit", None)
    if not callable(setter):
        metal = getattr(mx, "metal", None)
        setter = getattr(metal, "set_memory_limit", None)
    if not callable(setter):
        raise ExpertStreamingConfigurationError("MLX memory limit API is unavailable")
    setter(limit)
    return {"applied": True, "limit": limit}


def mlx_memory_telemetry(mx_module: Any | None = None) -> dict[str, int | str]:
    if mx_module is None:
        try:
            import mlx.core as mx
        except Exception as exc:
            return {"error": repr(exc)}
    else:
        mx = mx_module
    report: dict[str, int | str] = {}
    for name in ("get_active_memory", "get_peak_memory", "get_cache_memory"):
        getter = getattr(mx, name, None)
        if not callable(getter):
            getter = getattr(getattr(mx, "metal", None), name, None)
        if callable(getter):
            try:
                report[name.removeprefix("get_") + "_bytes"] = int(getter())
            except Exception as exc:
                report[name + "_error"] = repr(exc)
    return report


class ExpertStreamingRuntime:
    """Connect cache policy, checked I/O, fixed slots, and KV admission."""

    def __init__(
        self,
        root: Path,
        spec: ExpertStreamingModelSpec,
        config: ExpertStreamingConfig,
        manifest: ExpertManifest,
        plan: ExpertMemoryPlan,
        reader: PositionalExpertReader,
        slots: ExpertSlotPool,
        *,
        memory_cap_report: dict[str, Any] | None = None,
    ) -> None:
        self.root = root
        self.spec = spec
        self.config = config
        self.manifest = manifest
        self.plan = plan
        self.reader = reader
        self.slots = slots
        self.memory_cap_report = memory_cap_report
        self.counters = CacheCounters()
        self._banks = {
            layer: LayerExpertSlotBank(
                expert_count=spec.expert_count,
                persistent_slots=plan.slots_per_layer,
                transient_slots=plan.transient_slots,
                frequency_decay=config.frequency_decay,
            )
            for layer in spec.routed_layer_indices
        }
        self._layer_locks = {
            layer: threading.Lock() for layer in spec.routed_layer_indices
        }
        self._kv_lock = threading.Lock()
        self._live_kv_tokens = 0
        self._live_kv_peak = 0
        self._closed = False

    @classmethod
    def open(
        cls,
        root: Path | str,
        manifest_path: Path | str,
        config: ExpertStreamingConfig,
        *,
        spec: ExpertStreamingModelSpec | None = None,
        buffer_allocator: Callable[[int, str], Any] | None = None,
        device_synchronize: Callable[[], None] | None = None,
        apply_memory_cap: bool = True,
        mx_module: Any | None = None,
        env: dict[str, str] | None = None,
    ) -> ExpertStreamingRuntime:
        artifact_root = Path(root).resolve()
        model_spec = get_model_spec(config.model_key) if spec is None else spec
        if model_spec.key != config.model_key:
            raise ExpertStreamingConfigurationError("config and spec model keys differ")
        manifest = load_expert_manifest(manifest_path)
        cls._validate_manifest_identity(manifest, model_spec)
        if config.verify_artifact_headers:
            verify_expert_manifest(manifest, artifact_root)
        plan = config.memory_plan(model_spec)
        if not plan.fits_fixed:
            raise ExpertStreamingConfigurationError(
                f"fixed expert-streaming footprint exceeds limit by {-plan.unallocated_bytes} bytes"
            )
        cap_report = (
            apply_mlx_memory_cap(plan, mx_module=mx_module, env=env)
            if apply_memory_cap
            else None
        )
        reader = PositionalExpertReader(
            artifact_root,
            max_open_files=config.max_open_files,
            max_read_chunk_bytes=config.max_read_chunk_bytes,
        )
        try:
            slots = ExpertSlotPool(
                model_spec,
                plan,
                manifest,
                reader,
                buffer_allocator=buffer_allocator,
                max_inflight_io_bytes=config.max_inflight_io_bytes,
                prefer_sidecar=config.prefer_sidecar,
                verify_hashes=config.verify_record_hashes,
                device_synchronize=device_synchronize,
            )
        except Exception:
            reader.close()
            raise
        return cls(
            artifact_root,
            model_spec,
            config,
            manifest,
            plan,
            reader,
            slots,
            memory_cap_report=cap_report,
        )

    @staticmethod
    def _validate_manifest_identity(
        manifest: ExpertManifest,
        spec: ExpertStreamingModelSpec,
    ) -> None:
        errors: list[str] = []
        if manifest.model_key != spec.key:
            errors.append("model key")
        if manifest.source_repo != spec.quant_model:
            errors.append("source repository")
        if manifest.source_revision != spec.quant_revision:
            errors.append("source revision")
        if manifest.quant_bits != spec.quant_bits:
            errors.append("quantization bits")
        if manifest.quant_group_size != spec.quant_group_size:
            errors.append("quantization group size")
        if manifest.artifact_tensor_bytes != spec.total_tensor_bytes:
            errors.append("artifact tensor bytes")
        if errors:
            raise ExpertStreamingConfigurationError(
                "manifest does not match pinned model descriptor: " + ", ".join(errors)
            )

    def admit_kv_tokens(self, tokens: int) -> KVAdmission:
        count = _integer("tokens", tokens, minimum=1)
        with self._kv_lock:
            requested = self._live_kv_tokens + count
            if requested > self.config.max_live_kv_tokens:
                raise ExpertStreamingConfigurationError(
                    f"live KV admission {requested} exceeds planned "
                    f"{self.config.max_live_kv_tokens} tokens"
                )
            self._live_kv_tokens = requested
            self._live_kv_peak = max(self._live_kv_peak, requested)
        return KVAdmission(self, count)

    def release_kv_tokens(self, tokens: int) -> None:
        count = _integer("tokens", tokens, minimum=1)
        with self._kv_lock:
            if count > self._live_kv_tokens:
                raise RuntimeError("KV admission accounting underflow")
            self._live_kv_tokens -= count

    def ensure_route(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
    ) -> ReadyRoute:
        if self._closed:
            raise ExpertSlotError("expert streaming runtime is closed")
        try:
            bank = self._banks[layer]
            lock = self._layer_locks[layer]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} is not routed for {self.spec.key}"
            ) from exc
        with lock:
            route_plan = bank.plan(expert_ids, phase=phase)
            try:
                ready = self.slots.ensure_route(
                    layer,
                    route_plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                )
            except BaseException:
                for load in route_plan.loads:
                    if load.persistent:
                        bank.invalidate_expert(load.expert)
                    try:
                        self.slots.invalidate(layer, load.slot)
                    except ExpertSlotError:
                        pass
                raise
            self.counters.observe(
                route_plan,
                expert_record_bytes=self.spec.expert_record_bytes,
            )
            return ready

    def route_waves(self, expert_ids: Iterable[int]) -> tuple[RouteWave, ...]:
        return partition_route_waves(
            expert_ids,
            max_unique_experts=self.plan.transient_slots,
        )

    def reset(self) -> None:
        for lock in self._layer_locks.values():
            lock.acquire()
        try:
            self.slots.reset()
            for bank in self._banks.values():
                bank.reset()
        finally:
            for lock in reversed(tuple(self._layer_locks.values())):
                lock.release()

    def snapshot(self, *, mx_module: Any | None = None) -> dict[str, Any]:
        with self._kv_lock:
            live_kv = self._live_kv_tokens
            peak_kv = self._live_kv_peak
        return {
            "model_key": self.spec.key,
            "manifest_sha256": self.manifest.manifest_sha256,
            "memory_plan": {
                "total_limit_bytes": self.plan.total_limit_bytes,
                "fixed_bytes": self.plan.fixed_bytes,
                "persistent_cache_bytes": self.plan.persistent_cache_bytes,
                "slots_per_layer": self.plan.slots_per_layer,
                "transient_slots": self.plan.transient_slots,
                "allocated_bytes": self.plan.allocated_bytes,
                "unallocated_bytes": self.plan.unallocated_bytes,
            },
            "memory_cap": self.memory_cap_report,
            "mlx_memory": mlx_memory_telemetry(mx_module),
            "live_kv_tokens": live_kv,
            "live_kv_tokens_peak": peak_kv,
            "cache": self.counters.as_dict(),
            "slots": self.slots.snapshot(),
        }

    def close(self, *, timeout: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        self.slots.close(timeout=timeout)

    def __enter__(self) -> ExpertStreamingRuntime:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def load_configured_expert_runtime(
    root: Path | str,
    manifest_path: Path | str,
    config: ExpertStreamingConfig,
    **kwargs: Any,
) -> ExpertStreamingRuntime:
    try:
        return ExpertStreamingRuntime.open(root, manifest_path, config, **kwargs)
    except ExpertManifestError as exc:
        raise ExpertStreamingConfigurationError(str(exc)) from exc
