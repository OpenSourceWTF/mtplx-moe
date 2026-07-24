"""Load-time quantization of the bandwidth-dominant trunk ``*_proj`` Linears.

Decode throughput on large resident MoE models is bound by bytes read per
token, and the trunk projections (attention ``q/k/v/o_proj`` plus dense and
shared-expert ``gate/up/down_proj``) dominate the always-read set. These
passes shrink exactly that set at load time:

- ``quantize_projections`` converts BF16 ``nn.Linear`` projections to
  q4/q8 gs64 affine (checkpoints that ship residents in BF16);
- ``requantize_projections`` re-quantizes pre-quantized projections DOWN
  (e.g. a checkpoint's q8/gs64 residents to q4/gs64) via dequantize +
  ``QuantizedLinear.from_linear`` — the same canonical path a load-time
  quantization would take. The double quantization is deliberate and
  disclosed; deriving from BF16 sources remains the higher-quality route
  when a BF16 checkpoint is available.

Router gates, expert banks (SwitchLinear), embeddings, the LM head, norms,
and biases keep their loaded precision — the router picks experts, so its
numerics stay exact.

Measured on Hy3 oQ2e (2-bit experts, q8 residents, M-class 614 GB/s): q4
residents cut per-token reads ~11.9 GB -> ~7.9 GB and lifted AR decode
33.9 -> 43.1 tok/s short-context (36.6 at 1k-token real-code context), with
full-suite pass@1 statistically unchanged (McNemar p=1.0 on paired
HumanEval).
"""

from __future__ import annotations

from typing import Any

PROJ_QUANT_BITS = {"q8": 8, "q4": 4}
PROJ_REQUANT_BITS = {"q4": 4}
PROJ_QUANT_GROUP_SIZE = 64

_ATTENTION_PROJ_SUFFIXES = (".q_proj", ".k_proj", ".v_proj", ".o_proj", ".qkv_proj")
_MLP_PROJ_SUFFIXES = (".gate_proj", ".up_proj", ".down_proj", ".gate_up_proj")


class ProjQuantError(RuntimeError):
    pass


def proj_quant_covers(path: str) -> bool:
    """Module/tensor paths quantized by the load-time proj-quant pass.

    Covers exactly the ``*_proj`` weights of the trunk: attention
    ``q/k/v/o_proj`` plus ``gate/up/down/gate_up_proj`` in ``mlp`` and
    ``shared_mlp`` blocks. Deliberately narrower than "everything resident";
    expert banks are excluded by module type (SwitchLinear is never an
    ``nn.Linear``), routers/embeddings/head/norms by path.
    """

    if ".self_attn." in path and path.endswith(_ATTENTION_PROJ_SUFFIXES):
        return True
    if path.endswith(_MLP_PROJ_SUFFIXES):
        segments = path.split(".")
        return "mlp" in segments or "shared_mlp" in segments
    return False


def quantize_projections(model: Any, mode: str) -> list[str]:
    """Quantize BF16 trunk ``*_proj`` Linears to ``mode`` (q4/q8, gs64)."""

    import mlx.nn as nn

    if mode not in PROJ_QUANT_BITS:
        raise ProjQuantError(
            f"proj_quant mode must be one of {sorted(PROJ_QUANT_BITS)}, got {mode!r}"
        )
    bits = PROJ_QUANT_BITS[mode]
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
        model,
        group_size=PROJ_QUANT_GROUP_SIZE,
        bits=bits,
        mode="affine",
        class_predicate=predicate,
    )
    if not quantized:
        raise ProjQuantError(
            f"proj_quant={mode!r} matched no trunk *_proj Linear modules"
        )
    return quantized


def requantize_projections(model: Any, mode: str) -> list[str]:
    """Re-quantize pre-quantized trunk ``*_proj`` Linears down to ``mode``.

    Matches ``nn.QuantizedLinear`` modules in the ``proj_quant_covers`` scope
    whose bit width exceeds the target, dequantizes each via its own
    ``(group_size, bits, mode)`` triple, and rebuilds a standard q4/gs64
    affine module through ``QuantizedLinear.from_linear``. Idempotent: a
    module already at or below the target is never touched.
    """

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map_with_path

    if mode not in PROJ_REQUANT_BITS:
        raise ProjQuantError(
            f"proj_requant mode must be one of {sorted(PROJ_REQUANT_BITS)}, got {mode!r}"
        )
    target_bits = PROJ_REQUANT_BITS[mode]
    requantized: list[str] = []

    def rebuild(path: str, module: Any) -> Any:
        if not isinstance(module, nn.QuantizedLinear):
            return module
        if int(module.bits) <= target_bits:
            return module
        if not proj_quant_covers(path):
            return module
        weight = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
        )
        restored = nn.Linear(
            int(weight.shape[1]), int(weight.shape[0]), bias=("bias" in module)
        )
        restored.weight = weight
        if "bias" in module:
            restored.bias = module.bias
        requantized.append(path)
        return nn.QuantizedLinear.from_linear(
            restored,
            group_size=PROJ_QUANT_GROUP_SIZE,
            bits=target_bits,
            mode="affine",
        )

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(rebuild, leaves, is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    if not requantized:
        raise ProjQuantError(
            f"proj_requant={mode!r} matched no quantized trunk *_proj "
            f"modules above {target_bits} bits"
        )
    return requantized
