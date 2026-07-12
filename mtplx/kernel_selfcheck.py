"""Load-time exactness self-validation for the turbo kernel lanes.

Why this exists (2026-07-07, turbo-everywhere promotion): turbo became the
default profile for every dense catalog model, which means the MTPLX Metal
kernels now run on Apple GPU families nobody has physically measured (M1, M2,
M3, M3 Ultra). Every turbo lane is plain-SIMD and dtype-templated, so it
*should* be portable — but "should" is not a product guarantee. This module
turns it into one: at model load, each lane that can engage is run once on
tiny synthetic tensors in the model's actual dtype/quant format and compared
against the stock MLX reference. A lane that mismatches is disabled for the
process (serving falls back to the proven stock path for that lane only) and
the verdict is surfaced in ``/health`` as ``kernel_selfcheck``.

The check is a corruption tripwire, not a ULP certifier: thresholds sit ~10x
above the accumulation-order ULP band of each lane and ~10x below the
magnitude a broken kernel produces (wrong indexing, bad intrinsic, garbage
memory). Hot-shape ULP exactness is gated separately by the CI kernel matrix
and the release exactness gates.

Env:
- ``MTPLX_KERNEL_SELFCHECK=0`` disables the probe (default: runs whenever a
  turbo kernel env is active).
- ``MTPLX_FORCE_GPU_FAMILY_FALLBACK=1`` (handled in ``nax_verify``) forces the
  G17-gated m16 NAX lane off so newer machines can rehearse the exact
  M1-M4 code path.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_STATUS_OK = "ok"
_STATUS_FALLBACK = "fallback"
_STATUS_SKIPPED = "skipped"

# lane -> status ("ok" | "fallback" | "skipped"); empty until a run happens.
_LANE_STATUS: dict[str, str] = {}
_DISABLED_LANES: set[str] = set()
_LAST_REPORT: dict[str, Any] = {}

# Max |kernel - stock| tolerated per lane family at the fixture scales below
# (x ~ N(0, 0.5), w ~ N(0, 0.02), K = 1024). The qmm lanes' fp32-accumulate /
# lane-strided reduction differs from stock by low single-digit bf16 ULPs
# (~0.006 at these magnitudes); corruption lands at O(1) or NaN.
_QMM_TOLERANCE = 0.1
_SDPA_TOLERANCE = 0.02
_NORM_TOLERANCE = 0.02

_K = 1024  # satisfies every lane's K divisibility contract (%256 for m16)
_N = 1024  # satisfies N%32 (m16/msg) and N%4 (ksplit)


def _env_on(name: str, *, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "on", "yes"}


def selfcheck_enabled() -> bool:
    """Selfcheck runs by default whenever a turbo kernel lane is active."""
    raw = str(os.environ.get("MTPLX_KERNEL_SELFCHECK", "")).strip().lower()
    if raw in {"0", "false", "off", "no"}:
        return False
    if raw in {"1", "true", "on", "yes"}:
        return True
    return _env_on("MTPLX_NAX_VERIFY") or _env_on("MTPLX_GQA_PACKED_SDPA")


def lane_disabled(lane: str) -> bool:
    return lane in _DISABLED_LANES


def report_for_health() -> dict[str, Any]:
    """JSON-primitive-only payload for the ``/health`` endpoint."""
    payload: dict[str, Any] = {
        "ran": bool(_LAST_REPORT),
    }
    if _LAST_REPORT:
        payload["elapsed_ms"] = float(_LAST_REPORT.get("elapsed_ms", 0.0))
        payload["dtype"] = str(_LAST_REPORT.get("dtype", ""))
        payload["bits"] = int(_LAST_REPORT.get("bits", 0) or 0)
    for lane, status in sorted(_LANE_STATUS.items()):
        payload[lane] = status
    return payload


def _reset_for_tests() -> None:
    _LANE_STATUS.clear()
    _DISABLED_LANES.clear()
    _LAST_REPORT.clear()


def _quantized_fixture(mx, K: int, N: int, bits: int, group_size: int, dtype):
    mx.random.seed(7)
    w = (mx.random.normal((N, K), dtype=mx.float32) * 0.02).astype(dtype)
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(w_q, scales, biases)
    return w_q, scales, biases


def _qmm_reference(mx, x, w_q, scales, biases, *, bits: int, group_size: int):
    return mx.quantized_matmul(
        x,
        w_q,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )


def _max_abs_diff(mx, candidate, reference) -> float:
    diff = mx.abs(candidate.astype(mx.float32) - reference.astype(mx.float32))
    value = float(diff.max())
    if value != value:  # NaN
        return float("inf")
    return value


def _check_qmm_lane(mx, fn, m: int, bits: int, group_size: int, dtype) -> float:
    w_q, scales, biases = _quantized_fixture(mx, _K, _N, bits, group_size, dtype)
    x = (mx.random.normal((m, _K), dtype=mx.float32) * 0.5).astype(dtype)
    y = fn(x, w_q, scales, biases)
    ref = _qmm_reference(mx, x, w_q, scales, biases, bits=bits, group_size=group_size)
    if tuple(y.shape) != tuple(ref.shape):
        return float("inf")
    return _max_abs_diff(mx, y, ref)


def _check_gqa_packed(mx, dtype) -> float:
    from .kernels.sdpa_gqa_packed import sdpa_gqa_packed_tail

    hq, hk, d = 8, 2, 128
    capacity, offset, q_len = 512, 200, 4
    scale = d**-0.5
    mx.random.seed(11)
    queries = (mx.random.normal((1, hq, q_len, d), dtype=mx.float32) * 0.5).astype(dtype)
    keys = (mx.random.normal((1, hk, capacity, d), dtype=mx.float32) * 0.5).astype(dtype)
    values = (mx.random.normal((1, hk, capacity, d), dtype=mx.float32) * 0.5).astype(dtype)
    out = sdpa_gqa_packed_tail(
        queries=queries,
        keys=keys,
        values=values,
        offset=offset,
        scale=scale,
    )
    if out is None:
        return float("inf")
    # Tail-causal reference: query row j attends to rows n <= offset - q_len + j.
    rows = mx.arange(q_len).reshape(q_len, 1)
    cols = mx.arange(offset).reshape(1, offset)
    mask = cols <= (offset - q_len + rows)
    ref = mx.fast.scaled_dot_product_attention(
        queries,
        keys[:, :, :offset, :],
        values[:, :, :offset, :],
        scale=scale,
        mask=mask,
    )
    return _max_abs_diff(mx, out, ref)


def _check_fused_add_rmsnorm(mx, dtype) -> float:
    from .kernels.fused_norm import fused_add_rmsnorm

    mx.random.seed(13)
    rows, axis = 4, 512
    x = (mx.random.normal((rows, axis), dtype=mx.float32) * 0.5).astype(dtype)
    residual = (mx.random.normal((rows, axis), dtype=mx.float32) * 0.5).astype(dtype)
    weight = (mx.random.normal((axis,), dtype=mx.float32) * 0.1 + 1.0).astype(dtype)
    eps = 1e-6
    h, normed = fused_add_rmsnorm(x, residual, weight, eps, threadgroup_size=512)
    ref_h = x + residual
    ref_normed = mx.fast.rms_norm(ref_h, weight, eps).astype(dtype)
    return max(
        _max_abs_diff(mx, h, ref_h),
        _max_abs_diff(mx, normed, ref_normed),
    )


def _check_fused_gdn_norm_gate(mx, dtype) -> float:
    from .kernels.fused_norm import fused_gdn_norm_gate

    mx.random.seed(17)
    rows, axis = 4, 128
    x = (mx.random.normal((rows, axis), dtype=mx.float32) * 0.5).astype(dtype)
    gate = (mx.random.normal((rows, axis), dtype=mx.float32) * 0.5).astype(dtype)
    weight = (mx.random.normal((axis,), dtype=mx.float32) * 0.1 + 1.0).astype(dtype)
    eps = 1e-6
    y = fused_gdn_norm_gate(x, gate, weight, eps)
    normed = mx.fast.rms_norm(x, weight, eps)
    gate_f = gate.astype(mx.float32)
    ref = (gate_f * mx.sigmoid(gate_f) * normed.astype(mx.float32)).astype(dtype)
    return _max_abs_diff(mx, y, ref)


def run_kernel_selfcheck(dtype, bits: int, group_size: int) -> dict[str, Any]:
    """Probe every turbo lane that can engage for this model configuration.

    Returns ``{"lanes": {lane: status}, "dmax": {lane: float}, ...}`` and
    updates the process-wide disable registry: lanes reported ``fallback``
    stop engaging (their call sites route the stock path) until the process
    restarts. Idempotent — each run rebuilds the registry from scratch.
    """
    import mlx.core as mx

    from . import nax_verify

    bits = int(bits)
    group_size = int(group_size)
    started = time.perf_counter()

    lanes: dict[str, str] = {}
    dmax: dict[str, float] = {}

    def _record(lane: str, tolerance: float, probe) -> None:
        try:
            value = float(probe())
        except Exception as exc:  # any kernel-side failure means fallback
            logger.warning(
                "[mtplx] kernel selfcheck: %s raised (%s) — falling back to stock",
                lane,
                exc,
            )
            lanes[lane] = _STATUS_FALLBACK
            dmax[lane] = float("inf")
            return
        dmax[lane] = value
        if value <= tolerance:
            lanes[lane] = _STATUS_OK
        else:
            logger.warning(
                "[mtplx] kernel selfcheck: %s mismatched (dmax=%.4g) — "
                "falling back to stock",
                lane,
                value,
            )
            lanes[lane] = _STATUS_FALLBACK

    nax_on = _env_on("MTPLX_NAX_VERIFY")
    if nax_on and bits == 4:
        # 4-bit routes (nax_verify.patched): m4 -> env-selected impl (turbo
        # ships vk_k split-K), m5..6 -> legacy m6 ksplit, m7..16 -> NAX tile.
        _record(
            "qmm_m4",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: nax_verify.nax_qmm_m4(x, w, s, b, group_size=group_size),
                4,
                4,
                group_size,
                dtype,
            ),
        )
        _record(
            "qmm_m6",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: nax_verify.nax_qmm_m6(x, w, s, b, group_size=group_size),
                6,
                4,
                group_size,
                dtype,
            ),
        )
        lanes["qmm_m4_wide"] = _STATUS_SKIPPED  # single m4 impl covers all N at 4-bit
        lanes["qmm_m6_wide"] = _STATUS_SKIPPED
        if nax_verify.nax_available():
            _record(
                "qmm_m16_nax",
                _QMM_TOLERANCE,
                lambda: _check_qmm_lane(
                    mx,
                    lambda x, w, s, b: nax_verify.nax_qmm_m16(x, w, s, b, group_size=group_size),
                    16,
                    4,
                    group_size,
                    dtype,
                ),
            )
        else:
            lanes["qmm_m16_nax"] = _STATUS_SKIPPED
    elif nax_on and bits == 8:
        # 8-bit routes (nax_verify.patched): vk split-K for layer shapes,
        # vk msg wide tile for huge-N (lm_head-class) shapes.
        from .verify_kernels import (
            vk_qmm_m4,
            vk_qmm_m4_ksplit,
            vk_qmm_m6,
            vk_qmm_m6_ksplit,
        )

        _record(
            "qmm_m4",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: vk_qmm_m4_ksplit(x, w, s, b, bits=8, group_size=group_size),
                4,
                8,
                group_size,
                dtype,
            ),
        )
        _record(
            "qmm_m4_wide",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: vk_qmm_m4(x, w, s, b, bits=8, group_size=group_size),
                4,
                8,
                group_size,
                dtype,
            ),
        )
        _record(
            "qmm_m6",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: vk_qmm_m6_ksplit(x, w, s, b, bits=8, group_size=group_size),
                6,
                8,
                group_size,
                dtype,
            ),
        )
        _record(
            "qmm_m6_wide",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: vk_qmm_m6(x, w, s, b, bits=8, group_size=group_size),
                6,
                8,
                group_size,
                dtype,
            ),
        )
        lanes["qmm_m16_nax"] = _STATUS_SKIPPED  # 4-bit-only tile
    elif nax_on and bits == 6:
        # 6-bit routes (9B tier, 2026-07-07): split-K hexpack kernels only.
        from .verify_kernels import vk_qmm_m4_ksplit, vk_qmm_m6_ksplit

        _record(
            "qmm_m4",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: vk_qmm_m4_ksplit(x, w, s, b, bits=6, group_size=group_size),
                4,
                6,
                group_size,
                dtype,
            ),
        )
        _record(
            "qmm_m6",
            _QMM_TOLERANCE,
            lambda: _check_qmm_lane(
                mx,
                lambda x, w, s, b: vk_qmm_m6_ksplit(x, w, s, b, bits=6, group_size=group_size),
                6,
                6,
                group_size,
                dtype,
            ),
        )
        lanes["qmm_m4_wide"] = _STATUS_SKIPPED  # split-K only at 6-bit
        lanes["qmm_m6_wide"] = _STATUS_SKIPPED
        lanes["qmm_m16_nax"] = _STATUS_SKIPPED  # 4-bit-only tile
    else:
        for lane in ("qmm_m4", "qmm_m4_wide", "qmm_m6", "qmm_m6_wide", "qmm_m16_nax"):
            lanes[lane] = _STATUS_SKIPPED

    # Closed branch (2026-06-12): m8 ksplit is not routed by the dispatcher.
    lanes["qmm_m8_ksplit"] = _STATUS_SKIPPED
    # lm_head_topk kernels exist but are not routed on the serve path.
    lanes["lm_head_topk"] = _STATUS_SKIPPED

    if _env_on("MTPLX_GQA_PACKED_SDPA"):
        _record("gqa_packed_sdpa", _SDPA_TOLERANCE, lambda: _check_gqa_packed(mx, dtype))
    else:
        lanes["gqa_packed_sdpa"] = _STATUS_SKIPPED

    if _env_on("MTPLX_FUSE_POST_NORM_RESIDUAL"):
        _record(
            "fused_add_rmsnorm",
            _NORM_TOLERANCE,
            lambda: _check_fused_add_rmsnorm(mx, dtype),
        )
    else:
        lanes["fused_add_rmsnorm"] = _STATUS_SKIPPED

    if _env_on("MTPLX_FUSE_GDN_NORM_GATE"):
        _record(
            "fused_gdn_norm_gate",
            _NORM_TOLERANCE,
            lambda: _check_fused_gdn_norm_gate(mx, dtype),
        )
    else:
        lanes["fused_gdn_norm_gate"] = _STATUS_SKIPPED

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    _LANE_STATUS.clear()
    _LANE_STATUS.update(lanes)
    _DISABLED_LANES.clear()
    _DISABLED_LANES.update(
        lane for lane, status in lanes.items() if status == _STATUS_FALLBACK
    )
    dtype_tag = {mx.bfloat16: "bfloat16", mx.float16: "float16"}.get(dtype, str(dtype))
    report = {
        "lanes": dict(lanes),
        "dmax": {lane: float(value) for lane, value in dmax.items()},
        "dtype": dtype_tag,
        "bits": bits,
        "group_size": group_size,
        "elapsed_ms": elapsed_ms,
    }
    _LAST_REPORT.clear()
    _LAST_REPORT.update(report)
    return report


def _model_quant_signature(model: Any):
    """(dtype, bits, group_size) of the first quantized trunk projection."""
    import mlx.core as mx

    text_model = getattr(model, "language_model", model)
    inner = getattr(text_model, "model", text_model)
    for layer in getattr(inner, "layers", []) or []:
        for attr_path in (
            ("self_attn", "q_proj"),
            ("mlp", "gate_proj"),
            ("linear_attn", "in_proj_qkvz"),
        ):
            node = layer
            for name in attr_path:
                node = getattr(node, name, None)
                if node is None:
                    break
            bits = getattr(node, "bits", None)
            if bits is None:
                continue
            group_size = int(getattr(node, "group_size", 64) or 64)
            scales = None
            try:
                scales = node["scales"]
            except Exception:
                scales = getattr(node, "scales", None)
            dtype = getattr(scales, "dtype", None) or mx.bfloat16
            return dtype, int(bits), group_size
    return None


def maybe_run_model_selfcheck(model: Any) -> dict[str, Any] | None:
    """Run the selfcheck for a freshly loaded model if turbo lanes are active.

    Called once from ``runtime.load()`` before the runtime is returned; any
    failure inside the probe itself must never break model loading.
    """
    if not selfcheck_enabled():
        return None
    try:
        signature = _model_quant_signature(model)
        if signature is None:
            # Unquantized trunk: the qmm lanes never engage; still validate
            # the dtype-generic attention/norm lanes with the model dtype.
            import mlx.core as mx

            dtype = mx.bfloat16
            bits = 0
            group_size = 64
        else:
            dtype, bits, group_size = signature
        report = run_kernel_selfcheck(dtype, bits, group_size)
        fallbacks = sorted(
            lane for lane, status in report["lanes"].items() if status == _STATUS_FALLBACK
        )
        logger.info(
            "[mtplx] kernel selfcheck: %d lanes ok, %d fallback, %d skipped "
            "(%.0f ms, dtype=%s bits=%s)",
            sum(1 for s in report["lanes"].values() if s == _STATUS_OK),
            len(fallbacks),
            sum(1 for s in report["lanes"].values() if s == _STATUS_SKIPPED),
            report["elapsed_ms"],
            report["dtype"],
            report["bits"],
        )
        return report
    except Exception as exc:  # noqa: BLE001 - probe must never block serving
        logger.warning("[mtplx] kernel selfcheck failed to run: %s", exc)
        return None
