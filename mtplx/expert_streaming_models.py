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

from dataclasses import dataclass, replace
from operator import index

from mtplx.expert_shadow import SHADOW_CODECS, shadow_record_bytes


# Load-time resident quantization (group 64 affine over BF16): kept bytes
# per source byte are packed bits plus one BF16 scale and bias per group:
# bits/16 + 1/32, i.e. 9/32 for q4 and 17/32 for q8.
AFFINE_QUANT_KEEP_NUMERATOR = {"q8": 17, "q4": 9}
AFFINE_QUANT_KEEP_DENOMINATOR = 32
_ATTENTION_PROJ_SUFFIXES = (".q_proj", ".k_proj", ".v_proj", ".o_proj", ".qkv_proj")
_MLP_PROJ_SUFFIXES = (".gate_proj", ".up_proj", ".down_proj", ".gate_up_proj")


def proj_quant_covers(path: str) -> bool:
    """Module/tensor paths quantized by the load-time proj-quant pass.

    Covers exactly the ``*_proj`` weights of the trunk: attention
    ``q/k/v/o_proj`` plus ``gate/up/down/gate_up_proj`` in ``mlp`` and
    ``shared_mlp`` blocks. Router gates, embeddings, the LM head, norms,
    and biases stay at loaded precision — the router picks experts, so its
    numerics stay exact.

    This is deliberately NARROWER than "everything resident"; the loader's
    module predicate and the memory plan's byte discount must agree, so
    both read scope from this single source.
    """

    if ".self_attn." in path and path.endswith(_ATTENTION_PROJ_SUFFIXES):
        return True
    if path.endswith(_MLP_PROJ_SUFFIXES):
        segments = path.split(".")
        return "mlp" in segments or "shared_mlp" in segments
    return False


def affine_quant_kept_bytes(length: int, mode: str) -> int:
    """Post-quantization byte size of a BF16 tensor, rounded up."""

    numerator = AFFINE_QUANT_KEEP_NUMERATOR[mode]
    return -(-length * numerator // AFFINE_QUANT_KEEP_DENOMINATOR)


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
    # Pin-first ordering for count-based island selection: layers ranked by
    # ascending top-147 routing coverage (worst streamed-cache layers first).
    # Empty when unmeasured; count-based selection then requires an explicit
    # layer list instead.
    island_pin_order: tuple[int, ...] = ()
    # Streamed record codec: "affine" (Q2/Q4 weight+scales+biases triples)
    # or a shadow codec ("b1", "t158") whose records store packed+scales
    # pairs (the q1 lane, issue #51). Shadow-codec specs are pricing and
    # registry entries: the streamed runtime cannot open their artifacts
    # until the affine assumptions catalogued in
    # research/streamed-q1-codec-gap-analysis.md are closed.
    expert_codec: str = "affine"

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
        if self.expert_codec != "affine" and self.expert_codec not in SHADOW_CODECS:
            choices = ", ".join(repr(codec) for codec in SHADOW_CODECS)
            raise ValueError(f"expert_codec must be 'affine', {choices}")
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
        """Bytes of one streamed expert record under the spec's codec.

        Affine: packed quantized weights plus scale and bias leaves.
        Shadow codecs (q1 lane): packed sign/trit words plus one bf16
        scale per group — no bias leaf.
        """

        if self.expert_codec != "affine":
            return shadow_record_bytes(
                self.expert_codec, self.expert_source_parameters
            )
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
    prefetch_slots_per_layer: int = 0
    prefetch_bytes: int = 0
    mmap_islands_wired: bool = True
    miss_shadow: str | None = None
    shadow_bytes: int = 0

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
            + self.prefetch_bytes
            + self.io_staging_bytes
            + self.execution_workspace_bytes
            + self.runtime_reserve_bytes
            + self.island_bytes
            + wired_band
            + self.shadow_bytes
        )

    @property
    def allocated_bytes(self) -> int:
        return self.fixed_bytes + self.persistent_cache_bytes


# Measured 2026-07-16: 511-step decode route trace (issue #63 heatmap), layers
# ranked by ascending top-147 expert coverage. Routing is a property of the
# shared bf16 router weights (tencent/Hy3@716aa724), not the expert codec, so
# every hy3 bank derived from that revision shares this order.
_HY3_ROUTED_PIN_ORDER = (
    1, 4, 5, 13, 2, 3, 12, 18, 14, 15, 11, 10, 16, 17, 22, 28, 24, 26, 6, 25, 66, 29, 71, 27, 31, 19, 7, 69, 67, 23, 33, 32, 74, 9, 8, 50, 73, 79, 65, 72, 75, 70, 20, 30, 57, 21, 60, 64, 35, 51, 49, 68, 48, 59, 61, 77, 54, 38, 47, 52, 78, 53, 45, 62, 34, 58, 76, 63, 46, 55, 56, 44, 36, 37, 41, 43, 39, 40, 42,
)


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
    # island_pin_order deliberately EMPTY: this bank has its own measured
    # census (<root>/island-placement.json, issue #98) whose layer_pin_order
    # differs from the q2-derived _HY3_ROUTED_PIN_ORDER — count-based island
    # selection must resolve from the census, not a borrowed static order.
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
    island_pin_order=_HY3_ROUTED_PIN_ORDER,
)


HY3_EXPERT_OQ2E = ExpertStreamingModelSpec(
    key="hy3-expert-oq2e",
    display_name="mlx-community Hy3 oQ2e (imatrix affine Q2, gs128 experts, q8 residents)",
    source_model="tencent/Hy3",
    source_revision="716aa7241bd6d95896be4ebfc761162a9c4d49ef",
    quant_model="mlx-community/Hy3-oQ2e",
    quant_revision="1979c306076e00eb40670335c9cb54b357824e29",
    # True header-inventory sum. The published index declares 89_871_151_524,
    # overstating by 345_252 bytes (publisher metadata bug); the local index's
    # total_size is corrected to this value (original kept alongside as
    # model.safetensors.index.json.orig-published).
    total_tensor_bytes=89_870_806_272,
    total_layers=80,
    routed_layer_start=1,
    routed_layer_count=79,
    expert_count=192,
    top_k=8,
    hidden_size=4096,
    expert_hidden_size=1536,
    quant_bits=2,
    quant_group_size=128,
    quant_parameter_bytes=2,
    router_storage="source bfloat16 with fp32 correction bias",
    router_matmul_dtype="float32",
    router_bytes=124_316_928,
    kv_bytes_per_token=327_680,
    mtp_layer_index=80,
    mtp_included=False,
    # Reused from HY3_EXPERT_Q2: island_pin_order is a routing property of the
    # shared bf16 router weights, not of the expert record codec. Residents here
    # are q8 rather than bf16, which could perturb the hidden states feeding the
    # router; treat this order as a starting point, not re-measured truth.
    island_pin_order=_HY3_ROUTED_PIN_ORDER,
)


HY3_EXPERT_OQ4E = ExpertStreamingModelSpec(
    key="hy3-expert-oq4e",
    display_name="unigilby Hy3 oQ4e (imatrix affine Q4 gs64 experts, 5-bit boosted residents)",
    source_model="tencent/Hy3",
    source_revision="716aa7241bd6d95896be4ebfc761162a9c4d49ef",
    quant_model="unigilby/Hy3-oQ4e",
    quant_revision="4b13f0ee5c9ae9b002cb7448f89d732654f3f50e",
    # True header-inventory sum. The published index declares 169_785_226_181,
    # overstating by 345_797 bytes — the same oMLX generator bug as oQ2e
    # (which overstated by 345_252); local index corrected, original kept as
    # model.safetensors.index.json.orig-published.
    total_tensor_bytes=169_784_880_384,
    total_layers=80,
    routed_layer_start=1,
    routed_layer_count=79,
    expert_count=192,
    top_k=8,
    hidden_size=4096,
    expert_hidden_size=1536,
    # Routed expert records are uniformly affine Q4/gs64 (header-verified across
    # sensitive layers 1-3/76-79); the imatrix 5-bit sensitivity boosts touch
    # only RESIDENT projections (config-driven loader passes per-tensor bits).
    quant_bits=4,
    quant_group_size=64,
    quant_parameter_bytes=2,
    router_storage="source bfloat16 with fp32 correction bias",
    router_matmul_dtype="float32",
    router_bytes=124_316_928,
    kv_bytes_per_token=327_680,
    mtp_layer_index=80,
    mtp_included=False,
    # island_pin_order deliberately EMPTY like HY3_EXPERT_ONLY_Q4: this bank
    # needs its own measured census before count-based island selection.
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
    # island_pin_order below: measured 2026-07-17 from 1023 decode steps /
    # 613,800 routed assignments (ctx 1024 deterministic AR, 44 GiB cache =
    # 53 slots/layer), ranked by ascending top-53 expert coverage.
    # Rank-stable across K (Spearman >= 0.956 for K in 32..128); mean
    # cov@53 0.499 vs 0.517 measured decode hit rate. Earliest routed
    # layers route flattest, like hy3.
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
    island_pin_order=(
        5, 3, 4, 9, 7, 11, 8, 10, 6, 16, 13, 26, 14, 25, 24,
        49, 65, 70, 23, 54, 67, 66, 12, 52, 59, 15, 47, 71, 69, 62,
        17, 48, 61, 68, 74, 53, 57, 76, 75, 73, 60, 55, 77, 30, 29,
        72, 56, 27, 46, 63, 51, 28, 50, 64, 22, 58, 44, 38, 45, 39,
        43, 31, 42, 21, 35, 41, 40, 34, 37, 19, 36, 32, 33, 18, 20,
    ),
)


# q1 lane (issue #51): the same GLM checkpoint with shadow-codec expert
# records produced by scripts/convert_expert_q1.py. Same resident tensors,
# routers, KV, and routing (island_pin_order carries over — routing is a
# model property, not a record-codec property); only the routed record
# bytes change. Probe (research/glm52-shadow-codec-probe-20260717.json):
# t158 combine-cosine 0.922 from Q2 sources; b1 0.017 (NO-GO from Q2 —
# priced here only for a potential bf16-source encode).
_GLM52_RESIDENT_BYTES = (
    GLM52_EXPERT_Q2.total_tensor_bytes - GLM52_EXPERT_Q2.routed_expert_bytes
)

GLM52_EXPERT_Q1T = replace(
    GLM52_EXPERT_Q2,
    key="glm52-expert-q1t",
    display_name="GLM-5.2 expert-only ternary t158 (q1 lane)",
    expert_codec="t158",
    total_tensor_bytes=_GLM52_RESIDENT_BYTES
    + GLM52_EXPERT_Q2.routed_layer_count
    * GLM52_EXPERT_Q2.expert_count
    * shadow_record_bytes("t158", GLM52_EXPERT_Q2.expert_source_parameters),
)

GLM52_EXPERT_Q1B1 = replace(
    GLM52_EXPERT_Q2,
    key="glm52-expert-q1b1",
    display_name="GLM-5.2 expert-only binary b1 (q1 lane)",
    expert_codec="b1",
    quant_bits=1,
    total_tensor_bytes=_GLM52_RESIDENT_BYTES
    + GLM52_EXPERT_Q2.routed_layer_count
    * GLM52_EXPERT_Q2.expert_count
    * shadow_record_bytes("b1", GLM52_EXPERT_Q2.expert_source_parameters),
)


MODEL_SPECS: dict[str, ExpertStreamingModelSpec] = {
    spec.key: spec
    for spec in (
        HY3_Q4,
        HY3_EXPERT_ONLY_Q4,
        HY3_EXPERT_Q2,
        HY3_EXPERT_OQ2E,
        HY3_EXPERT_OQ4E,
        GLM52_Q4,
        GLM52_EXPERT_Q2,
        GLM52_EXPERT_Q1T,
        GLM52_EXPERT_Q1B1,
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
    resident_discount_bytes: int = 0,
    kv_quant: str | None = None,
    cache_scope: str = "layer",
    island_layer_count: int = 0,
    mmap_island_layer_count: int = 0,
    mmap_islands_wired: bool = True,
    prefetch_slots_per_layer: int = 0,
    miss_shadow: str | None = None,
    miss_shadow_layers: int | None = None,
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

    ``miss_shadow`` prices one dense low-precision shadow bank per
    *streamed* layer (routed minus islands minus mmap band) so decode
    misses can be served in memory instead of from SSD; the full cost
    lands on the fixed side.

    A plan with ``fits_fixed=False`` is diagnostic and must not be used to
    start inference.  Its negative ``unallocated_bytes`` is the fixed-footprint
    deficit.  ``context_tokens=0`` is an explicit load-only plan; an inference
    runtime must reject KV growth beyond the planned live-token total unless
    it re-plans every admission boundary (the derived single-limit policy in
    :class:`mtplx.expert_runtime.ExpertStreamingConfig`).
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
    resident_discount_bytes = _integer(
        "resident_discount_bytes", resident_discount_bytes, minimum=0
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
    prefetch_slots_per_layer = _integer(
        "prefetch_slots_per_layer", prefetch_slots_per_layer, minimum=0
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
    if kv_quant is not None:
        if kv_quant not in AFFINE_QUANT_KEEP_NUMERATOR:
            raise ValueError("kv_quant must be None, 'q8', or 'q4'")
        # QuantizedKVCache uses the same group-64 affine layout as the
        # resident pass; the MTP layer's stock cache is not in
        # kv_bytes_per_token (mtp_included=False) and stays in reserve.
        kv_bytes = affine_quant_kept_bytes(kv_bytes, kv_quant)
    transient_bytes = service_slots * spec.expert_record_bytes
    resident_bytes = (
        spec.resident_bytes + additional_resident_bytes - resident_discount_bytes
    )
    if resident_bytes <= 0:
        raise ValueError(
            "resident_discount_bytes exceeds the resident footprint: "
            f"{resident_discount_bytes} vs {spec.resident_bytes}"
        )
    island_bytes = (
        island_layer_count * spec.expert_count * spec.expert_record_bytes
    )
    mmap_island_bytes = (
        mmap_island_layer_count * spec.expert_count * spec.expert_record_bytes
    )
    streamed_layer_count = (
        spec.routed_layer_count - island_layer_count - mmap_island_layer_count
    )
    prefetch_bytes = (
        streamed_layer_count * prefetch_slots_per_layer * spec.expert_record_bytes
    )
    if miss_shadow is not None and miss_shadow not in SHADOW_CODECS:
        choices = ", ".join(repr(codec) for codec in SHADOW_CODECS)
        raise ValueError(f"miss_shadow must be None, {choices}")
    if miss_shadow_layers is not None:
        miss_shadow_layers = _integer(
            "miss_shadow_layers", miss_shadow_layers, minimum=1
        )
        if miss_shadow is None:
            raise ValueError("miss_shadow_layers requires miss_shadow")
    shadow_layer_count = (
        min(miss_shadow_layers, streamed_layer_count)
        if miss_shadow is not None and miss_shadow_layers is not None
        else streamed_layer_count
    )
    shadow_bytes = (
        shadow_layer_count
        * spec.expert_count
        * shadow_record_bytes(miss_shadow, spec.expert_source_parameters)
        if miss_shadow is not None
        else 0
    )
    fixed_bytes = (
        resident_bytes
        + kv_bytes
        + runtime_reserve_bytes
        + transient_bytes
        + prefetch_bytes
        + io_staging_bytes
        + execution_workspace_bytes
        + island_bytes
        + (mmap_island_bytes if mmap_islands_wired else 0)
        + shadow_bytes
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
        prefetch_slots_per_layer=prefetch_slots_per_layer,
        prefetch_bytes=prefetch_bytes,
        miss_shadow=miss_shadow,
        shadow_bytes=shadow_bytes,
    )
