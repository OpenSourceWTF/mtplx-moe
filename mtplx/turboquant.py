"""Small MTPLX-side TurboQuant configuration helpers.

The vLLM-Metal reference implementation keeps its public constants behind a
``vllm`` import.  MTPLX only needs the stable cache-layout metadata to drive the
already-loaded Metal ops, so keep this dependency-free and deliberately narrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from collections.abc import Mapping
from statistics import NormalDist


KEY_QUANTS: dict[str, dict[str, int | bool | str]] = {
    "q8_0": {"bits": 8, "signed": True, "dtype": "int8"},
    "int8": {"bits": 8, "signed": True, "dtype": "int8"},
    "uint8": {"bits": 8, "signed": False, "dtype": "uint8"},
    "q5_0": {"bits": 5, "signed": False, "dtype": "uint8"},
    "q4_0": {"bits": 4, "signed": False, "dtype": "uint8"},
    "int4": {"bits": 4, "signed": False, "dtype": "uint8"},
    "uint4": {"bits": 4, "signed": False, "dtype": "uint8"},
    "int2": {"bits": 2, "signed": False, "dtype": "uint8"},
    "uint2": {"bits": 2, "signed": False, "dtype": "uint8"},
}

VALUE_QUANTS: dict[str, int] = {
    "q2_0": 2,
    "q3_0": 3,
    "q4_0": 4,
    "q5_0": 5,
    "q8_0": 8,
}

FWHT_SUPPORTED_HEAD_DIMS = {64, 128, 256, 512}
SCALE_GROUP_SIZE = 32

# vLLM-Metal's built-in 3-bit Lloyd-Max table.  This is the default and the
# only value quant we need for the first project-level diagnostic.
CENTROIDS_3BIT = (
    -2.15195,
    -1.34391,
    -0.75601,
    -0.24509,
    0.24509,
    0.75601,
    1.34391,
    2.15195,
)

_NORMAL = NormalDist()
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class TurboQuantConfig:
    key_quant: str = "q8_0"
    value_quant: str = "q3_0"

    @property
    def key_bits(self) -> int:
        return int(KEY_QUANTS[self.key_quant]["bits"])

    @property
    def value_bits(self) -> int:
        return int(VALUE_QUANTS[self.value_quant])

    @property
    def key_signed(self) -> bool:
        return bool(KEY_QUANTS[self.key_quant]["signed"])

    @property
    def key_dtype_name(self) -> str:
        return str(KEY_QUANTS[self.key_quant]["dtype"])


def env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def config_from_environment(
    environment: Mapping[str, str],
) -> TurboQuantConfig | None:
    enabled_raw = environment.get("MTPLX_VLLM_METAL_PAGED_TURBOQUANT")
    if enabled_raw is None or enabled_raw.strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    key_quant = (
        environment.get("MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT")
        or "q8_0"
    ).strip().lower()
    value_quant = (
        environment.get("MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT")
        or "q3_0"
    ).strip().lower()
    if key_quant not in KEY_QUANTS:
        raise ValueError(
            f"Unsupported TurboQuant key quant {key_quant!r}; "
            f"available={sorted(KEY_QUANTS)}"
        )
    if value_quant not in VALUE_QUANTS:
        raise ValueError(
            f"Unsupported TurboQuant value quant {value_quant!r}; "
            f"available={sorted(VALUE_QUANTS)}"
        )
    return TurboQuantConfig(key_quant=key_quant, value_quant=value_quant)


def config_from_env() -> TurboQuantConfig | None:
    return config_from_environment(os.environ)


def packed_dim(head_dim: int, bits: int) -> int:
    if (int(head_dim) * int(bits)) % 8 != 0:
        raise ValueError(
            f"TurboQuant packed dim is not byte-aligned: head_dim={head_dim}, bits={bits}"
        )
    return int(head_dim) * int(bits) // 8


def validate_head_dim(head_dim: int) -> None:
    if int(head_dim) % SCALE_GROUP_SIZE != 0:
        raise ValueError(
            f"TurboQuant requires head_dim divisible by {SCALE_GROUP_SIZE}, got {head_dim}"
        )
    if int(head_dim) not in FWHT_SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"TurboQuant FWHT supports head_dim in {sorted(FWHT_SUPPORTED_HEAD_DIMS)}, got {head_dim}"
        )


def _normal_pdf(x: float) -> float:
    if math.isinf(x):
        return 0.0
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _normal_cdf(x: float) -> float:
    if x == math.inf:
        return 1.0
    if x == -math.inf:
        return 0.0
    return _NORMAL.cdf(x)


def _truncated_normal_mean(lo: float, hi: float) -> float:
    mass = _normal_cdf(hi) - _normal_cdf(lo)
    if mass <= 0.0:
        if math.isinf(lo) and math.isinf(hi):
            return 0.0
        if math.isinf(lo):
            return hi
        if math.isinf(hi):
            return lo
        return 0.5 * (lo + hi)
    return (_normal_pdf(lo) - _normal_pdf(hi)) / mass


@lru_cache(maxsize=None)
def value_centroids(bits: int) -> tuple[float, ...]:
    """Return standard-normal Lloyd-Max V centroids for a TurboQuant bit width.

    vLLM-Metal's TurboQuant encode and attention kernels accept arbitrary
    ``v_bits`` in ``[1, 8]`` via a centroid buffer.  q3 keeps the upstream
    baked table for byte-for-byte continuity; other widths use the analytic
    Lloyd-Max update for a unit-normal source instead of an expensive sampled
    MLX pass at cache-allocation time.
    """

    bits = int(bits)
    if bits == 3:
        return CENTROIDS_3BIT
    if bits < 1 or bits > 8:
        raise ValueError(f"TurboQuant value bits must be in [1, 8], got {bits}")

    n = 1 << bits
    centroids: list[float] = []
    for i in range(n):
        lo = -math.inf if i == 0 else _NORMAL.inv_cdf(i / n)
        hi = math.inf if i == n - 1 else _NORMAL.inv_cdf((i + 1) / n)
        centroids.append(_truncated_normal_mean(lo, hi))

    tolerance = 5e-7 if bits >= 6 else 1e-8
    for _ in range(1000):
        boundaries = [
            0.5 * (centroids[i] + centroids[i + 1]) for i in range(n - 1)
        ]
        updated = []
        for i in range(n):
            lo = -math.inf if i == 0 else boundaries[i - 1]
            hi = math.inf if i == n - 1 else boundaries[i]
            updated.append(_truncated_normal_mean(lo, hi))
        max_delta = max(abs(a - b) for a, b in zip(updated, centroids, strict=True))
        centroids = updated
        if max_delta < tolerance:
            break

    return tuple(float(c) for c in centroids)


def compression_ratio(*, head_dim: int, key_bits: int, value_bits: int) -> float:
    fp16_bytes = 2 * int(head_dim) * 2
    quant_bytes = (
        packed_dim(head_dim, key_bits)
        + packed_dim(head_dim, value_bits)
        + 3 * (int(head_dim) // SCALE_GROUP_SIZE) * 2
    )
    return float(fp16_bytes) / float(quant_bytes)
