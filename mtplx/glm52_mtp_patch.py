"""Strict streamed GLM-5.2 layer-78 MTP support.

GLM-5.2 uses the DeepSeek-style NextN fusion sequence, but its trained head
is not a stock DeepSeek decoder: it owns GLM DSA indexer weights, uses FP32
MoE routing, and recurrently shares the D1 index/KV view through D5.  This
module mirrors the fail-closed external-artifact boundary used by Hy3 while
keeping the already-working streamed GLM target unchanged.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .glm52_mtp_artifact import (
    VerifiedGlm52MtpArtifact,
    open_verified_glm52_mtp_layer78,
)

logger = logging.getLogger(__name__)

GLM52_MTP_SOURCE_REPO = "zai-org/GLM-5.2"
GLM52_MTP_SOURCE_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
GLM52_MTP_BF16_FILE = "layer78-bf16.safetensors"

_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_HEAD_LOCAL_MODULES = ("eh_proj", "enorm", "hnorm")


class Glm52MTPLoadError(RuntimeError):
    """The external GLM-5.2 MTP artifact or runtime contract is invalid."""


@dataclass(frozen=True)
class GLM52MTPTensorSpec:
    dtype: str
    shape: tuple[int, ...]


@dataclass
class GLM52MTPCycleState:
    base_offset: int
    d1_boundary: int | None = None
    topk_indices: Any | None = None
    topk_computed: bool = False


class GLM52MTPCache:
    """One logical layer-78 cache with committed-only snapshot semantics."""

    def __init__(self, main_kv: Any, indexer_kv: Any, *, index_topk: int):
        if int(index_topk) < 1:
            raise ValueError("GLM MTP index_topk must be positive")
        self.main_kv = main_kv
        self.indexer_kv = indexer_kv
        self.index_topk = int(index_topk)
        self.cycle: GLM52MTPCycleState | None = None

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> Any:
        return (self.main_kv, self.indexer_kv)[index]

    @property
    def offset(self) -> int:
        return int(self.main_kv.offset)

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.offset == 0

    def is_trimmable(self) -> bool:
        return True

    @property
    def nbytes(self) -> int:
        return int(self.main_kv.nbytes) + int(self.indexer_kv.nbytes)

    @staticmethod
    def _state_at(cache: Any, offset: int) -> Any:
        if offset <= 0:
            return None
        if cache.keys is None or cache.values is None:
            raise Glm52MTPLoadError("non-empty GLM MTP cache has no KV state")
        return (
            cache.keys[..., :offset, :],
            cache.values[..., :offset, :],
        )

    @property
    def state(self) -> list[Any]:
        committed = self.offset
        if self.cycle is not None:
            committed = (
                self.cycle.d1_boundary
                if self.cycle.d1_boundary is not None
                else self.cycle.base_offset
            )
        return [
            self._state_at(self.main_kv, int(committed)),
            self._state_at(
                self.indexer_kv, min(int(committed), int(self.indexer_kv.offset))
            ),
        ]

    @state.setter
    def state(self, values: Any) -> None:
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError("GLM52MTPCache state must contain main and indexer KV")
        for child, value in zip((self.main_kv, self.indexer_kv), values):
            if value is None:
                child.keys = None
                child.values = None
                child.offset = 0
            else:
                child.state = value
        self.cycle = None

    def replace_state(self, values: Any) -> None:
        """Install a snapshot through cache_state's container-preserving path."""

        self.state = values

    @property
    def meta_state(self) -> tuple[str]:
        return ("mtplx-glm52-mtp-cache-v1",)

    @meta_state.setter
    def meta_state(self, value: Any) -> None:
        if value not in {None, (), ("mtplx-glm52-mtp-cache-v1",)}:
            raise ValueError(f"unsupported GLM52MTPCache meta state: {value!r}")
        self.cycle = None

    def rollback_to(self, target: int) -> None:
        target = max(0, int(target))
        for child in (self.main_kv, self.indexer_kv):
            excess = max(0, int(child.offset) - target)
            if excess:
                child.trim(excess)
        self.cycle = None

    def trim(self, count: int) -> int:
        before = self.offset
        self.rollback_to(max(0, before - max(0, int(count))))
        return before - self.offset

    def begin_cycle(self) -> None:
        if self.cycle is not None:
            self.finish_cycle()
        if int(self.indexer_kv.offset) != self.offset:
            raise Glm52MTPLoadError(
                "GLM MTP committed MLA/indexer offsets diverged before D1: "
                f"{self.offset} != {int(self.indexer_kv.offset)}"
            )
        self.cycle = GLM52MTPCycleState(base_offset=self.offset)

    def finish_d1(self, *, topk_indices: Any | None) -> None:
        state = self.cycle
        if state is None:
            raise Glm52MTPLoadError("GLM MTP D1 completed outside a draft cycle")
        expected = state.base_offset + 1
        if self.offset != expected or int(self.indexer_kv.offset) != expected:
            raise Glm52MTPLoadError(
                "GLM MTP D1 must advance both caches exactly once; "
                f"got MLA={self.offset}, indexer={int(self.indexer_kv.offset)}, "
                f"expected={expected}"
            )
        if topk_indices is None and expected > self.index_topk:
            raise Glm52MTPLoadError("GLM MTP D1 full indexer produced no top-k indices")
        state.d1_boundary = expected
        state.topk_indices = topk_indices
        state.topk_computed = True

    def recurrent_view(self) -> tuple[Any | None, int]:
        state = self.cycle
        if state is None or not state.topk_computed or state.d1_boundary is None:
            raise Glm52MTPLoadError("GLM MTP D2+ requires a completed D1 state")
        return state.topk_indices, state.d1_boundary

    def finish_cycle(self) -> None:
        state = self.cycle
        if state is None:
            return
        target = (
            state.d1_boundary if state.d1_boundary is not None else state.base_offset
        )
        for child in (self.main_kv, self.indexer_kv):
            excess = max(0, int(child.offset) - int(target))
            if excess:
                child.trim(excess)
        self.cycle = None


def _layer_prefix(args: Any) -> str:
    return f"model.layers.{int(args.num_hidden_layers)}."


def expected_glm52_mtp_inventory(args: Any) -> dict[str, GLM52MTPTensorSpec]:
    """Return the exact raw checkpoint tensor contract for the trained head."""

    prefix = _layer_prefix(args)
    h = int(args.hidden_size)
    q = int(args.q_lora_rank)
    kv = int(args.kv_lora_rank)
    rope = int(args.qk_rope_head_dim)
    nope = int(args.qk_nope_head_dim)
    value = int(args.v_head_dim)
    heads = int(args.num_attention_heads)
    index_heads = int(args.index_n_heads)
    index_dim = int(args.index_head_dim)
    experts = int(args.n_routed_experts)
    moe = int(args.moe_intermediate_size)
    shared = moe * int(args.n_shared_experts or 0)

    def spec(dtype: str, *shape: int) -> GLM52MTPTensorSpec:
        return GLM52MTPTensorSpec(dtype=dtype, shape=tuple(shape))

    inventory = {
        prefix + "eh_proj.weight": spec("BF16", h, 2 * h),
        prefix + "enorm.weight": spec("BF16", h),
        prefix + "hnorm.weight": spec("BF16", h),
        prefix + "input_layernorm.weight": spec("BF16", h),
        prefix + "mlp.gate.e_score_correction_bias": spec("F32", experts),
        prefix + "mlp.gate.weight": spec("BF16", experts, h),
        prefix + "mlp.shared_experts.down_proj.weight": spec("BF16", h, shared),
        prefix + "mlp.shared_experts.gate_proj.weight": spec("BF16", shared, h),
        prefix + "mlp.shared_experts.up_proj.weight": spec("BF16", shared, h),
        prefix + "post_attention_layernorm.weight": spec("BF16", h),
        prefix + "self_attn.indexer.k_norm.bias": spec("BF16", index_dim),
        prefix + "self_attn.indexer.k_norm.weight": spec("BF16", index_dim),
        prefix + "self_attn.indexer.weights_proj.weight": spec("BF16", index_heads, h),
        prefix + "self_attn.indexer.wk.weight": spec("BF16", index_dim, h),
        prefix + "self_attn.indexer.wq_b.weight": spec(
            "BF16", index_heads * index_dim, q
        ),
        prefix + "self_attn.kv_a_layernorm.weight": spec("BF16", kv),
        prefix + "self_attn.kv_a_proj_with_mqa.weight": spec("BF16", kv + rope, h),
        prefix + "self_attn.kv_b_proj.weight": spec("BF16", heads * (nope + value), kv),
        prefix + "self_attn.o_proj.weight": spec("BF16", h, heads * value),
        prefix + "self_attn.q_a_layernorm.weight": spec("BF16", q),
        prefix + "self_attn.q_a_proj.weight": spec("BF16", q, h),
        prefix + "self_attn.q_b_proj.weight": spec("BF16", heads * (nope + rope), q),
        prefix + "shared_head.norm.weight": spec("BF16", h),
    }
    for expert in range(experts):
        inventory[f"{prefix}mlp.experts.{expert}.gate_proj.weight"] = spec(
            "BF16", moe, h
        )
        inventory[f"{prefix}mlp.experts.{expert}.up_proj.weight"] = spec("BF16", moe, h)
        inventory[f"{prefix}mlp.experts.{expert}.down_proj.weight"] = spec(
            "BF16", h, moe
        )
    return inventory


def _receipt_revision(receipt: dict[str, Any]) -> str | None:
    direct = receipt.get("source_revision")
    if direct is not None:
        return str(direct)
    source = receipt.get("source")
    if isinstance(source, dict) and source.get("revision") is not None:
        return str(source["revision"])
    return None


def _mapped_target(suffix: str) -> str:
    if suffix.startswith("shared_head.norm."):
        return "layers.0.shared_head_norm." + suffix.removeprefix("shared_head.norm.")
    if suffix.split(".", 1)[0] in _HEAD_LOCAL_MODULES:
        return "layers.0." + suffix
    return "layers.0.mtp_block." + suffix


def _map_glm52_mtp_bf16_weights(
    tensors: dict[str, Any],
    args: Any,
    mx: Any,
) -> dict[str, Any]:
    expected = expected_glm52_mtp_inventory(args)
    missing = set(expected) - set(tensors)
    extra = set(tensors) - set(expected)
    if missing:
        raise Glm52MTPLoadError(
            f"{GLM52_MTP_BF16_FILE} is missing tensors: {sorted(missing)[:4]}"
        )
    if extra:
        raise Glm52MTPLoadError(
            f"{GLM52_MTP_BF16_FILE} has unexpected tensors: {sorted(extra)[:4]}"
        )
    for name, value in tensors.items():
        tensor_spec = expected[name]
        wanted_dtype = mx.bfloat16 if tensor_spec.dtype == "BF16" else mx.float32
        if value.dtype != wanted_dtype:
            label = "bfloat16" if tensor_spec.dtype == "BF16" else "float32"
            raise Glm52MTPLoadError(f"{name} must be {label}, found {value.dtype}")
        shape = tuple(int(dim) for dim in value.shape)
        if shape != tensor_spec.shape:
            raise Glm52MTPLoadError(
                f"{name} has shape {shape}; expected shape {tensor_spec.shape}"
            )

    prefix = _layer_prefix(args)
    mapped: dict[str, Any] = {}
    for name, value in tensors.items():
        suffix = name[len(prefix) :]
        if ".mlp.experts." in name or suffix == "self_attn.kv_b_proj.weight":
            continue
        mapped[_mapped_target(suffix)] = value

    kv_b = tensors[prefix + "self_attn.kv_b_proj.weight"]
    heads = int(args.num_attention_heads)
    nope = int(args.qk_nope_head_dim)
    value_dim = int(args.v_head_dim)
    kv_b = kv_b.reshape(heads, nope + value_dim, -1)
    mapped["layers.0.mtp_block.self_attn.embed_q.weight"] = mx.contiguous(
        kv_b[:, :nope, :].swapaxes(-1, -2)
    )
    mapped["layers.0.mtp_block.self_attn.unembed_out.weight"] = mx.contiguous(
        kv_b[:, nope:, :]
    )

    for projection in _EXPERT_PROJECTIONS:
        values = [
            tensors[f"{prefix}mlp.experts.{expert}.{projection}.weight"]
            for expert in range(int(args.n_routed_experts))
        ]
        mapped[f"layers.0.mtp_block.mlp.switch_mlp.{projection}.weight"] = mx.stack(
            values
        )
    if len(mapped) != 27:
        raise Glm52MTPLoadError(
            f"GLM-5.2 layer-78 mapped to {len(mapped)} leaves; expected 27"
        )
    return mapped


def load_glm52_mtp_bf16_weights(
    artifact_dir: Path | str,
    args: Any,
    *,
    expected_revision: str = GLM52_MTP_SOURCE_REVISION,
    mx_module: Any | None = None,
    verified_artifact: VerifiedGlm52MtpArtifact | None = None,
) -> dict[str, Any]:
    """Verify, load, validate, and map the exact BF16 layer-78 artifact."""

    artifact_dir = Path(artifact_dir).expanduser().resolve()
    path = artifact_dir / GLM52_MTP_BF16_FILE
    if not path.is_file():
        raise Glm52MTPLoadError(f"missing GLM-5.2 MTP artifact {path}")

    if mx_module is None:
        import mlx.core as mx
    else:
        mx = mx_module
    verification_context = (
        open_verified_glm52_mtp_layer78(artifact_dir, deep=True)
        if verified_artifact is None
        else contextlib.nullcontext(verified_artifact)
    )
    try:
        with verification_context as verified:
            if verified.file.closed:
                raise Glm52MTPLoadError(
                    "borrowed GLM-5.2 MTP artifact handle is closed"
                )
            verified_root = Path(verified.root).expanduser().resolve()
            if verified_root != artifact_dir:
                raise Glm52MTPLoadError(
                    "borrowed GLM-5.2 MTP artifact root does not match artifact_dir"
                )
            revision = _receipt_revision(verified.manifest)
            if revision != expected_revision:
                raise Glm52MTPLoadError(
                    f"GLM-5.2 MTP artifact revision {revision!r}; "
                    f"expected {expected_revision!r}"
                )
            tensors = dict(mx.load(verified.file, format="safetensors"))
            mapped = _map_glm52_mtp_bf16_weights(tensors, args, mx)
            mx.eval(mapped)
            return mapped
    except Glm52MTPLoadError:
        raise
    except Exception as exc:
        raise Glm52MTPLoadError(
            f"GLM-5.2 MTP artifact verification or load failed: {exc}"
        ) from exc


def build_glm52_mtp_module(
    artifact_dir: Path | str,
    args: Any,
    *,
    expected_revision: str = GLM52_MTP_SOURCE_REVISION,
    verified_artifact: VerifiedGlm52MtpArtifact | None = None,
) -> Any:
    """Construct and strictly load the resident BF16 GLM-5.2 MTP head."""

    import mlx.core as mx

    from .models.glm52_mlx import Glm52MTP

    weights = load_glm52_mtp_bf16_weights(
        artifact_dir,
        args,
        expected_revision=expected_revision,
        verified_artifact=verified_artifact,
    )
    mtp = Glm52MTP(args, num_mtp_layers=1)
    mtp.eval()
    try:
        mtp.load_weights(list(weights.items()), strict=True)
    except Exception as exc:
        raise Glm52MTPLoadError(
            f"GLM-5.2 MTP strict weight validation failed: {exc}"
        ) from exc
    mx.eval(mtp.parameters())
    logger.info(
        "[GLM-5.2 MTP] loaded %d BF16 leaves from %s",
        len(weights),
        Path(artifact_dir).expanduser(),
    )
    return mtp


def _validate_glm52_mtp_contract(contract: Any | None) -> None:
    expected = {
        "base_hidden_variant": "post_norm",
        "hidden_variant": "post_norm",
        "concat_order": "embedding_hidden",
        "mtp_position_mode": "cache",
    }
    mismatches = {
        name: getattr(contract, name, wanted) if contract is not None else wanted
        for name, wanted in expected.items()
        if (getattr(contract, name, wanted) if contract is not None else wanted)
        != wanted
    }
    quantized = contract is not None and (
        getattr(contract, "mtp_quant_bits", None) is not None
        or getattr(contract, "mtp_quant_policy", None) is not None
        or bool(getattr(contract, "mtp_prequantized", False))
        or bool(getattr(contract, "mtp_prequantized_modules", ()))
        or bool(getattr(contract, "mtp_prequantized_module_specs", {}))
    )
    if mismatches or quantized:
        details = [
            f"{name}={value!r} (expected {expected[name]!r})"
            for name, value in mismatches.items()
        ]
        if quantized:
            details.append("MTP quantization is unsupported")
        raise Glm52MTPLoadError(
            "GLM-5.2 MTP contract is incompatible: " + ", ".join(details)
        )


def inject_glm52_streamed_mtp_support(
    model: Any,
    artifact_dir: Path | str | None,
    config: dict[str, Any],
    contract: Any | None = None,
    *,
    expected_revision: str = GLM52_MTP_SOURCE_REVISION,
    verified_artifact: VerifiedGlm52MtpArtifact | None = None,
    mtp_module: Any | None = None,
) -> bool:
    """Attach the strict external layer-78 head to a streamed GLM target."""

    from mlx_lm.models.cache import KVCache

    if artifact_dir is None:
        raise Glm52MTPLoadError("streamed GLM-5.2 MTP requires an artifact directory")
    if str(config.get("model_type") or "") != "glm_moe_dsa":
        raise Glm52MTPLoadError("streamed GLM-5.2 MTP supports glm_moe_dsa only")
    _validate_glm52_mtp_contract(contract)
    args = getattr(model, "args", None)
    if args is None:
        raise Glm52MTPLoadError("streamed GLM-5.2 model exposes no ModelArgs")
    if int(getattr(args, "num_nextn_predict_layers", 0) or 0) != 1:
        raise Glm52MTPLoadError("GLM-5.2 must declare exactly one trained MTP layer")
    if not bool(getattr(args, "index_share_for_mtp_iteration", False)):
        raise Glm52MTPLoadError(
            "GLM-5.2 MTP requires index_share_for_mtp_iteration=true"
        )

    mtp = mtp_module
    if mtp is None:
        mtp = build_glm52_mtp_module(
            artifact_dir,
            args,
            expected_revision=expected_revision,
            verified_artifact=verified_artifact,
        )
    original_outer_class = model.__class__

    class _MTPLXStreamedGLM52Model(original_outer_class):
        mtp_verify_width = 6
        mtp_recurrent_requires_persistent_cache = True

        def __call__(
            self,
            inputs,
            cache=None,
            return_hidden: bool = False,
            input_embeddings=None,
            hidden_variant: str | None = None,
            emit_logits: bool = True,
            logits_keep: int | None = None,
            **kwargs,
        ):
            if input_embeddings is not None:
                raise ValueError("streamed GLM-5.2 does not support input_embeddings")
            if hidden_variant not in {None, "post_norm"}:
                raise ValueError("streamed GLM-5.2 MTP exposes post_norm hidden only")
            hidden = self.model(inputs, cache)
            logits = None
            if emit_logits:
                head_hidden = hidden
                if logits_keep is not None:
                    keep = int(logits_keep)
                    if keep < 1:
                        raise ValueError("logits_keep must be positive when supplied")
                    head_hidden = hidden[:, -keep:, :]
                logits = self.lm_head(head_hidden)
            if not return_hidden:
                return logits
            return logits, hidden

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            cache=None,
            mtp_cache=None,
            concat_order=None,
            return_hidden: bool = False,
            mtp_hidden_variant: str | None = "post_norm",
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            if concat_order not in {None, "embedding_hidden"}:
                raise ValueError(
                    "streamed GLM-5.2 MTP supports embedding_hidden order only"
                )
            if mtp_hidden_variant not in {None, "post_norm"}:
                raise ValueError("streamed GLM-5.2 MTP consumes post_norm hidden")
            depth = 1 if mtp_depth is None else int(mtp_depth)
            if depth < 1 or depth > 5:
                raise ValueError("GLM-5.2 MTP depth must be inside [1, 5]")
            if depth > 1 and mtp_cache is None:
                raise Glm52MTPLoadError(
                    "GLM-5.2 MTP recurrent depths require a persistent cache"
                )
            layer_cache = None
            if mtp_cache is not None:
                layer_cache = mtp_cache[0] if isinstance(mtp_cache, list) else mtp_cache
                if not isinstance(layer_cache, GLM52MTPCache):
                    raise TypeError("GLM-5.2 MTP requires GLM52MTPCache")

            topk_indices = None
            boundary = None
            compute_topk = True
            if layer_cache is not None:
                if depth == 1:
                    layer_cache.begin_cycle()
                else:
                    topk_indices, boundary = layer_cache.recurrent_view()
                    compute_topk = False
            try:
                logits, hidden, produced_topk = self.mtp.layers[0](
                    next_token_ids,
                    hidden_states,
                    embed_tokens=self.model.embed_tokens,
                    lm_head=self.lm_head,
                    cache=layer_cache,
                    prev_topk_indices=topk_indices,
                    compute_topk=compute_topk,
                    kv_read_boundary=boundary,
                )
                if layer_cache is not None and depth == 1:
                    layer_cache.finish_d1(topk_indices=produced_topk)
            except BaseException:
                if layer_cache is not None:
                    layer_cache.finish_cycle()
                raise
            if not return_hidden:
                return logits
            return logits, hidden

        def mtp_update_cache(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache=None,
            concat_order=None,
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            if concat_order not in {None, "embedding_hidden"}:
                raise ValueError(
                    "streamed GLM-5.2 MTP supports embedding_hidden order only"
                )
            layer_cache = None
            if mtp_cache is not None:
                layer_cache = mtp_cache[0] if isinstance(mtp_cache, list) else mtp_cache
                if not isinstance(layer_cache, GLM52MTPCache):
                    raise TypeError("GLM-5.2 MTP requires GLM52MTPCache")
                layer_cache.finish_cycle()
                if layer_cache.offset != int(layer_cache.indexer_kv.offset):
                    raise Glm52MTPLoadError(
                        "GLM MTP committed append requires equal cache offsets"
                    )
            _logits, hidden, _topk = self.mtp.layers[0](
                next_token_ids,
                hidden_states,
                embed_tokens=self.model.embed_tokens,
                lm_head=self.lm_head,
                cache=layer_cache,
                prev_topk_indices=None,
                compute_topk=True,
                kv_read_boundary=None,
            )
            return hidden

        def make_mtp_cache(self):
            return [
                GLM52MTPCache(
                    KVCache(),
                    KVCache(),
                    index_topk=int(self.args.index_topk),
                )
            ]

        def finish_mtp_cycle(self, mtp_cache) -> None:
            if mtp_cache is None:
                return
            entries = mtp_cache if isinstance(mtp_cache, list) else [mtp_cache]
            for entry in entries:
                if isinstance(entry, GLM52MTPCache):
                    entry.finish_cycle()

    model.mtp = mtp
    model.__class__ = _MTPLXStreamedGLM52Model
    logger.info("[GLM-5.2 MTP inject] strict layer-78 head attached")
    return True
