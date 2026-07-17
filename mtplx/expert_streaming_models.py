"""Pinned MoE layouts and memory budgeting for SSD expert streaming.

The routed expert tensors are intentionally excluded from the fixed resident
footprint.  At every sparse layer, a resident router chooses global expert
IDs, then :mod:`mtplx.expert_streaming` resolves those IDs into bounded Metal
slots.  This module converts a user-visible byte limit into a uniform number
of persistent slots per sparse layer while reserving resident model state, KV
cache, runtime headroom, and one globally reused transient service bank.

This is layout and policy scaffolding.  It does not load model weights or
allocate MLX arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index


def _integer(name: str, value: object, *, minimum: int | None = None) -> int:
    """Normalize an integer-like value without accepting lossy coercions."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        normalized = index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if minimum is not None and normalized < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


@dataclass(frozen=True)
class ExpertStreamingModelSpec:
    """Revision-pinned geometry needed to size a streamed quantized expert bank."""

    key: str
    display_name: str
    source_model: str
    source_revision: str
    quant_model: str
    quant_revision: str
    total_tensor_bytes: int
    total_layers: int
    routed_layer_start: int
    routed_layer_count: int
    expert_count: int
    top_k: int
    hidden_size: int
    expert_hidden_size: int
    quant_bits: int
    quant_group_size: int
    quant_parameter_bytes: int
    router_storage: str
    router_matmul_dtype: str
    router_bytes: int
    kv_bytes_per_token: int
    mtp_layer_index: int | None
    mtp_included: bool
    full_indexer_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        integer_fields = {
            "total_tensor_bytes": 1,
            "total_layers": 1,
            "routed_layer_start": 0,
            "routed_layer_count": 1,
            "expert_count": 1,
            "top_k": 1,
            "hidden_size": 1,
            "expert_hidden_size": 1,
            "quant_bits": 1,
            "quant_group_size": 1,
            "quant_parameter_bytes": 1,
            "router_bytes": 0,
            "kv_bytes_per_token": 0,
        }
        for name, minimum in integer_fields.items():
            normalized = _integer(name, getattr(self, name), minimum=minimum)
            object.__setattr__(self, name, normalized)
        if not isinstance(self.mtp_included, bool):
            raise TypeError("mtp_included must be bool")
        if self.mtp_layer_index is not None:
            object.__setattr__(
                self,
                "mtp_layer_index",
                _integer("mtp_layer_index", self.mtp_layer_index, minimum=0),
            )
        if not isinstance(self.full_indexer_layers, tuple):
            raise TypeError("full_indexer_layers must be a tuple")
        object.__setattr__(
            self,
            "full_indexer_layers",
            tuple(
                _integer("full_indexer_layer", layer, minimum=0)
                for layer in self.full_indexer_layers
            ),
        )
        if self.routed_layer_start + self.routed_layer_count > self.total_layers:
            raise ValueError("routed layers must be inside the target model")
        if self.top_k > self.expert_count:
            raise ValueError("top_k cannot exceed expert_count")
        if (
            self.hidden_size % self.quant_group_size
            or self.expert_hidden_size % self.quant_group_size
        ):
            raise ValueError("each expert projection input must divide into groups")
        projection_parameters = self.hidden_size * self.expert_hidden_size
        if projection_parameters * self.quant_bits % 8:
            raise ValueError("packed expert weights must occupy whole bytes")
        invalid_indexers = set(self.full_indexer_layers) - set(range(self.total_layers))
        if invalid_indexers:
            raise ValueError("full indexer layers must be inside the target model")
        if tuple(sorted(set(self.full_indexer_layers))) != self.full_indexer_layers:
            raise ValueError("full indexer layers must be sorted and unique")
        if self.routed_expert_bytes >= self.total_tensor_bytes:
            raise ValueError("resident model footprint must be positive")
        if self.router_bytes > self.resident_bytes:
            raise ValueError("router bytes cannot exceed the resident footprint")
        if (
            self.mtp_layer_index is not None
            and self.mtp_layer_index < self.total_layers
        ):
            raise ValueError("MTP layer must follow the target transformer layers")

    @property
    def routed_layer_indices(self) -> tuple[int, ...]:
        """Target layer indices whose routed experts are streamed."""

        stop = self.routed_layer_start + self.routed_layer_count
        return tuple(range(self.routed_layer_start, stop))

    @property
    def expert_source_parameters(self) -> int:
        """Unquantized gate, up, and down values in one routed expert."""

        return 3 * self.hidden_size * self.expert_hidden_size

    @property
    def packed_weight_bytes(self) -> int:
        return self.expert_source_parameters * self.quant_bits // 8

    @property
    def scale_bias_bytes(self) -> int:
        """Affine scales plus biases for all quantization groups."""

        group_count = self.expert_source_parameters // self.quant_group_size
        return group_count * 2 * self.quant_parameter_bytes

    @property
    def expert_record_bytes(self) -> int:
        """Packed quantized weights plus affine scale and bias leaves per expert."""

        return self.packed_weight_bytes + self.scale_bias_bytes

    @property
    def routed_expert_bytes(self) -> int:
        return self.routed_layer_count * self.expert_count * self.expert_record_bytes

    @property
    def resident_bytes(self) -> int:
        """All target tensors except routed experts, before runtime/KV reserve."""

        return self.total_tensor_bytes - self.routed_expert_bytes

    @property
    def cold_expert_bytes_per_token(self) -> int:
        return self.routed_layer_count * self.top_k * self.expert_record_bytes

    @property
    def transient_scratch_bytes(self) -> int:
        """One global top-k service bank, reused after each layer fence."""

        return self.top_k * self.expert_record_bytes

    def persistent_cache_bytes(self, slots_per_layer: int) -> int:
        """Bytes used by a uniform persistent bank across all sparse layers."""

        slots_per_layer = _integer("slots_per_layer", slots_per_layer, minimum=0)
        if not 0 <= slots_per_layer <= self.expert_count:
            raise ValueError(f"slots_per_layer must be inside [0, {self.expert_count}]")
        return self.routed_layer_count * slots_per_layer * self.expert_record_bytes


@dataclass(frozen=True)
class ExpertMemoryPlan:
    """Resolved memory allocation under a user-provided total byte limit."""

    model_key: str
    total_limit_bytes: int
    runtime_reserve_bytes: int
    io_staging_bytes: int
    execution_workspace_bytes: int
    context_tokens: int
    resident_bytes: int
    kv_bytes: int
    transient_slots: int
    transient_bytes: int
    expert_cache_limit_bytes: int | None
    persistent_budget_bytes: int
    cache_scope: str
    persistent_slots: int
    slots_per_layer: int
    persistent_cache_bytes: int
    unallocated_bytes: int
    fits_fixed: bool
    island_layer_count: int = 0
    island_bytes: int = 0
    mmap_island_layer_count: int = 0
    mmap_island_bytes: int = 0
    mmap_islands_wired: bool = True

    @property
    def fixed_bytes(self) -> int:
        # Wired mmap islands are registered in MLX's process-wide residency
        # set and count in MLX memory accounting, so they live on the fixed
        # side exactly like ordinary islands. Paged (unwired) bands live in
        # the page cache outside MLX and are charged by the cap reconciler
        # instead.
        wired_band = self.mmap_island_bytes if self.mmap_islands_wired else 0
        return (
            self.resident_bytes
            + self.kv_bytes
            + self.transient_bytes
            + self.io_staging_bytes
            + self.execution_workspace_bytes
            + self.runtime_reserve_bytes
            + self.island_bytes
            + wired_band
        )

    @property
    def allocated_bytes(self) -> int:
        return self.fixed_bytes + self.persistent_cache_bytes


HY3_Q4 = ExpertStreamingModelSpec(
    key="hy3-q4",
    display_name="Tencent Hy3 affine Q4",
    source_model="tencent/Hy3",
    source_revision="716aa7241bd6d95896be4ebfc761162a9c4d49ef",
    quant_model="pipenetwork/Hy3-4bit",
    quant_revision="160619d3f96c8470350b6dac0ef033a8381551e3",
    total_tensor_bytes=165_988_461_824,
    total_layers=80,
    routed_layer_start=1,
    routed_layer_count=79,
    expert_count=192,
    top_k=8,
    hidden_size=4096,
    expert_hidden_size=1536,
    quant_bits=4,
    quant_group_size=64,
    quant_parameter_bytes=2,
    router_storage="affine-q8 with fp32 correction bias",
    router_matmul_dtype="activation_dtype",
    router_bytes=66_071_808,
    kv_bytes_per_token=327_680,
    mtp_layer_index=80,
    mtp_included=False,
)


HY3_EXPERT_ONLY_Q4 = ExpertStreamingModelSpec(
    key="hy3-expert-only-q4",
    display_name="Local Tencent Hy3 expert-only affine Q4 control",
    source_model="tencent/Hy3",
    source_revision="716aa7241bd6d95896be4ebfc761162a9c4d49ef",
    quant_model="local/hy3-expert-only-mlx-q4",
    quant_revision="716aa7241bd6d95896be4ebfc761162a9c4d49ef",
    total_tensor_bytes=178_530_397_440,
    total_layers=80,
    routed_layer_start=1,
    routed_layer_count=79,
    expert_count=192,
    top_k=8,
    hidden_size=4096,
    expert_hidden_size=1536,
    quant_bits=4,
    quant_group_size=64,
    quant_parameter_bytes=2,
    router_storage="source bfloat16 with fp32 correction bias",
    router_matmul_dtype="float32",
    router_bytes=124_316_928,
    kv_bytes_per_token=327_680,
    mtp_layer_index=80,
    mtp_included=False,
)


HY3_EXPERT_Q2 = ExpertStreamingModelSpec(
    key="hy3-expert-q2",
    display_name="Experimental Tencent Hy3 expert-only affine Q2",
    source_model="tencent/Hy3",
    source_revision="716aa7241bd6d95896be4ebfc761162a9c4d49ef",
    quant_model="local/hy3-expert-only-mlx-q4",
    quant_revision="716aa7241bd6d95896be4ebfc761162a9c4d49ef",
    total_tensor_bytes=106_958_793_984,
    total_layers=80,
    routed_layer_start=1,
    routed_layer_count=79,
    expert_count=192,
    top_k=8,
    hidden_size=4096,
    expert_hidden_size=1536,
    quant_bits=2,
    quant_group_size=64,
    quant_parameter_bytes=2,
    router_storage="source bfloat16 with fp32 correction bias",
    router_matmul_dtype="float32",
    router_bytes=124_316_928,
    kv_bytes_per_token=327_680,
    mtp_layer_index=80,
    mtp_included=False,
)


GLM52_Q4 = ExpertStreamingModelSpec(
    key="glm52-q4",
    display_name="GLM-5.2 affine Q4",
    source_model="zai-org/GLM-5.2",
    source_revision="b4734de4facf877f85769a911abafc5283eab3d9",
    quant_model="mlx-community/GLM-5.2-4bit",
    quant_revision="6b347a6472d46bf55de65ee34032136a3929d778",
    total_tensor_bytes=418_320_895_488,
    total_layers=78,
    routed_layer_start=3,
    routed_layer_count=75,
    expert_count=256,
    top_k=8,
    hidden_size=6144,
    expert_hidden_size=2048,
    quant_bits=4,
    quant_group_size=64,
    quant_parameter_bytes=2,
    router_storage="bfloat16 with fp32 correction bias",
    router_matmul_dtype="float32",
    router_bytes=236_006_400,
    kv_bytes_per_token=95_232,
    mtp_layer_index=78,
    mtp_included=False,
    full_indexer_layers=(0, 1, 2, *range(6, 75, 4)),
)


GLM52_EXPERT_Q2 = ExpertStreamingModelSpec(
    key="glm52-expert-q2",
    display_name="GLM-5.2 expert-only affine Q2",
    source_model="zai-org/GLM-5.2",
    source_revision="b4734de4facf877f85769a911abafc5283eab3d9",
    quant_model="mlx-community/GLM-5.2-4bit",
    quant_revision="6b347a6472d46bf55de65ee34032136a3929d778",
    total_tensor_bytes=237_126_962_688,
    total_layers=78,
    routed_layer_start=3,
    routed_layer_count=75,
    expert_count=256,
    top_k=8,
    hidden_size=6144,
    expert_hidden_size=2048,
    quant_bits=2,
    quant_group_size=64,
    quant_parameter_bytes=2,
    router_storage="bfloat16 with fp32 correction bias",
    router_matmul_dtype="float32",
    router_bytes=236_006_400,
    kv_bytes_per_token=95_232,
    mtp_layer_index=78,
    mtp_included=False,
    full_indexer_layers=(0, 1, 2, *range(6, 75, 4)),
)


MODEL_SPECS: dict[str, ExpertStreamingModelSpec] = {
    spec.key: spec
    for spec in (
        HY3_Q4,
        HY3_EXPERT_ONLY_Q4,
        HY3_EXPERT_Q2,
        GLM52_Q4,
        GLM52_EXPERT_Q2,
    )
}


def get_model_spec(model: str) -> ExpertStreamingModelSpec:
    """Return a pinned streaming descriptor by CLI key."""

    try:
        return MODEL_SPECS[model]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"unknown model {model!r}; choose one of: {choices}") from exc


def plan_expert_memory(
    spec: ExpertStreamingModelSpec,
    *,
    total_limit_bytes: int,
    context_tokens: int,
    runtime_reserve_bytes: int = 0,
    expert_cache_limit_bytes: int | None = None,
    transient_slots: int | None = None,
    io_staging_bytes: int = 0,
    execution_workspace_bytes: int = 0,
    additional_resident_bytes: int = 0,
    cache_scope: str = "layer",
    island_layer_count: int = 0,
    mmap_island_layer_count: int = 0,
    mmap_islands_wired: bool = True,
    mmap_island_bytes_override: int | None = None,
) -> ExpertMemoryPlan:
    """Fit uniform persistent expert slots under an explicit memory ceiling.

    The total limit is never treated as an expert-cache-only setting: resident
    weights, KV state, runtime headroom, and transient miss service are removed
    first.  ``expert_cache_limit_bytes`` can impose a stricter secondary cap.
    The returned slot count is bounded by the model's expert count.

    ``island_layer_count`` routed layers hold every expert permanently
    resident in dense per-layer banks outside the uniform slot pool; their
    full cost lands on the fixed side and the uniform slot math spans only
    the remaining streamed layers.

    ``mmap_island_layer_count`` routed layers hold every expert in
    file-backed banks whose physical pages belong to the OS page cache, not
    to MLX: they leave the fixed budget untouched, contribute no uniform
    slots, and shrink the streamed layer count exactly like wired islands.
    ``mmap_island_bytes`` is advisory — the pager's working set, not an
    allocation.

    A plan with ``fits_fixed=False`` is diagnostic and must not be used to
    start inference.  Its negative ``unallocated_bytes`` is the fixed-footprint
    deficit.  ``context_tokens=0`` is an explicit load-only plan; an inference
    runtime must reject KV growth beyond the planned live-token total.
    """

    if not isinstance(spec, ExpertStreamingModelSpec):
        raise TypeError("spec must be an ExpertStreamingModelSpec")
    total_limit_bytes = _integer("total_limit_bytes", total_limit_bytes, minimum=1)
    context_tokens = _integer("context_tokens", context_tokens, minimum=0)
    runtime_reserve_bytes = _integer(
        "runtime_reserve_bytes", runtime_reserve_bytes, minimum=0
    )
    io_staging_bytes = _integer("io_staging_bytes", io_staging_bytes, minimum=0)
    execution_workspace_bytes = _integer(
        "execution_workspace_bytes", execution_workspace_bytes, minimum=0
    )
    additional_resident_bytes = _integer(
        "additional_resident_bytes", additional_resident_bytes, minimum=0
    )
    if expert_cache_limit_bytes is not None:
        expert_cache_limit_bytes = _integer(
            "expert_cache_limit_bytes", expert_cache_limit_bytes, minimum=0
        )
    if cache_scope not in {"layer", "global"}:
        raise ValueError("cache_scope must be 'layer' or 'global'")
    island_layer_count = _integer(
        "island_layer_count", island_layer_count, minimum=0
    )
    if island_layer_count > spec.routed_layer_count:
        raise ValueError(
            f"island_layer_count {island_layer_count} exceeds routed layer "
            f"count {spec.routed_layer_count}"
        )
    if island_layer_count and cache_scope != "layer":
        raise ValueError("dense island layers require cache_scope 'layer'")
    mmap_island_layer_count = _integer(
        "mmap_island_layer_count", mmap_island_layer_count, minimum=0
    )
    if island_layer_count + mmap_island_layer_count > spec.routed_layer_count:
        raise ValueError(
            f"island_layer_count {island_layer_count} and "
            f"mmap_island_layer_count {mmap_island_layer_count} together "
            f"exceed routed layer count {spec.routed_layer_count}"
        )
    if mmap_island_layer_count and cache_scope != "layer":
        raise ValueError("mmap island layers require cache_scope 'layer'")

    service_slots = spec.top_k if transient_slots is None else transient_slots
    service_slots = _integer("transient_slots", service_slots, minimum=0)
    if service_slots < spec.top_k:
        raise ValueError(f"transient_slots must be at least top_k ({spec.top_k})")

    kv_bytes = context_tokens * spec.kv_bytes_per_token
    transient_bytes = service_slots * spec.expert_record_bytes
    resident_bytes = spec.resident_bytes + additional_resident_bytes
    island_bytes = (
        island_layer_count * spec.expert_count * spec.expert_record_bytes
    )
    mmap_island_bytes = (
        mmap_island_layer_count * spec.expert_count * spec.expert_record_bytes
    )
    if mmap_island_bytes_override is not None:
        # Compressed banked bands occupy their true on-disk region bytes,
        # not the raw record footprint; the caller reads them from the
        # banked manifest.
        mmap_island_bytes = _integer(
            "mmap_island_bytes_override", mmap_island_bytes_override, minimum=0
        )
    streamed_layer_count = (
        spec.routed_layer_count - island_layer_count - mmap_island_layer_count
    )
    fixed_bytes = (
        resident_bytes
        + kv_bytes
        + runtime_reserve_bytes
        + transient_bytes
        + io_staging_bytes
        + execution_workspace_bytes
        + island_bytes
        + (mmap_island_bytes if mmap_islands_wired else 0)
    )
    available_bytes = max(0, total_limit_bytes - fixed_bytes)
    streamed_expert_bytes = (
        streamed_layer_count * spec.expert_count * spec.expert_record_bytes
    )
    persistent_budget_bytes = min(available_bytes, streamed_expert_bytes)
    if expert_cache_limit_bytes is not None:
        persistent_budget_bytes = min(persistent_budget_bytes, expert_cache_limit_bytes)

    bytes_per_uniform_slot = streamed_layer_count * spec.expert_record_bytes
    if cache_scope == "global":
        persistent_slots = min(
            spec.routed_layer_count * spec.expert_count,
            persistent_budget_bytes // spec.expert_record_bytes,
        )
        # This quota is used only to seed an empty global pool fairly during
        # prefill. Decode can lend every slot across layer boundaries.
        slots_per_layer = min(
            spec.expert_count, persistent_slots // spec.routed_layer_count
        )
        persistent_cache_bytes = persistent_slots * spec.expert_record_bytes
    elif streamed_layer_count == 0:
        slots_per_layer = 0
        persistent_slots = 0
        persistent_cache_bytes = 0
    else:
        slots_per_layer = min(
            spec.expert_count, persistent_budget_bytes // bytes_per_uniform_slot
        )
        persistent_slots = slots_per_layer * streamed_layer_count
        persistent_cache_bytes = persistent_slots * spec.expert_record_bytes
    unallocated_bytes = total_limit_bytes - fixed_bytes - persistent_cache_bytes

    return ExpertMemoryPlan(
        model_key=spec.key,
        total_limit_bytes=total_limit_bytes,
        runtime_reserve_bytes=runtime_reserve_bytes,
        io_staging_bytes=io_staging_bytes,
        execution_workspace_bytes=execution_workspace_bytes,
        context_tokens=context_tokens,
        resident_bytes=resident_bytes,
        kv_bytes=kv_bytes,
        transient_slots=service_slots,
        transient_bytes=transient_bytes,
        expert_cache_limit_bytes=expert_cache_limit_bytes,
        persistent_budget_bytes=persistent_budget_bytes,
        cache_scope=cache_scope,
        persistent_slots=persistent_slots,
        slots_per_layer=slots_per_layer,
        persistent_cache_bytes=persistent_cache_bytes,
        unallocated_bytes=unallocated_bytes,
        fits_fixed=fixed_bytes <= total_limit_bytes,
        island_layer_count=island_layer_count,
        island_bytes=island_bytes,
        mmap_island_layer_count=mmap_island_layer_count,
        mmap_island_bytes=mmap_island_bytes,
        mmap_islands_wired=bool(mmap_islands_wired),
    )
