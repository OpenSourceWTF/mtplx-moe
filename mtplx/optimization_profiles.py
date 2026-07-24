"""Per-model optimization-profile registry.

Every serving optimization measured on large models is model-conditional:
the knob that buys one family double-digit throughput is meaningless (or
harmful) on another. This module records those MEASURED decisions as
reviewable in-repo data — a change to a default is a diff to this file, with
the measurement named in ``provenance``.

Profiles are not an autotuner. Entries carry one of four states:

- ``default_on``: measured win; the value is the recommended default.
- ``default_off``: applicable but measured harmful (or a null); leave off.
- ``not_applicable``: structurally meaningless or measured net-negative at
  the operating envelope — see the entry's provenance for which.
- ``unvalidated``: mechanism exists but no end-to-end measurement backs a
  default either way.

Consumers are read-only and advisory: ``resolve_profile_defaults`` lets
benchmark/serving surfaces print a "profile suggests" note when explicit
flags conflict with a measured default (warning only; no behavior change),
and ``not_applicable_violations`` lets a loader refuse a knob the registry
marks structurally wrong for the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

VALID_STATES = ("default_on", "default_off", "not_applicable", "unvalidated")

KNOB_NAMES = (
    "proj_quant",
    "proj_requant",
    "kv_quant",
    "mtp_history_policy",
    "draft_core",
    "verify_strategy",
    "speculative_depth",
    "compile_ar_forward",
)

ENFORCEABLE_KNOBS = frozenset({"proj_quant", "proj_requant", "kv_quant"})


@dataclass(frozen=True)
class KnobEntry:
    """One measured (or explicitly unmeasured) per-model knob decision."""

    state: str
    value: Any
    provenance: str

    def __post_init__(self) -> None:
        if self.state not in VALID_STATES:
            raise ValueError(
                f"state must be one of {VALID_STATES}, got {self.state!r}"
            )
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError(
                "provenance must name the measurement (date + config + "
                "delta) or say why none exists; empty provenance is a "
                "registry defect"
            )


@dataclass(frozen=True, eq=False)
class OptimizationProfile:
    """Complete knob decision set for one model_key."""

    model_key: str
    knobs: Mapping[str, KnobEntry]

    def __post_init__(self) -> None:
        if not isinstance(self.model_key, str) or not self.model_key:
            raise ValueError("model_key must be a non-empty string")
        if not isinstance(self.knobs, Mapping):
            raise ValueError("knobs must be a mapping")
        unknown = set(self.knobs) - set(KNOB_NAMES)
        if unknown:
            raise ValueError(f"unknown knob names: {sorted(unknown)}")


_HY3_OQ2E = OptimizationProfile(
    model_key="hy3-oq2e",
    knobs={
        "proj_requant": KnobEntry(
            state="default_on",
            value="q4",
            provenance=(
                "2026-07-24 M-class 614 GB/s, 1024-tok real-code prefill: "
                "q4 residents cut per-token reads ~11.9->~7.9 GB; AR 33.9->"
                "43.1 tok/s short-ctx (36.6 @ 1k ctx); paired HumanEval "
                "unchanged (McNemar p=1.0). Draft head excluded (stays q8)."
            ),
        ),
        "kv_quant": KnobEntry(
            state="not_applicable",
            value="off",
            provenance=(
                "2026-07-22 envelope matrix: MTP dead under kv4 at real "
                "prefill (AR 26.2 vs K3 10.8); bf16 KV locked."
            ),
        ),
        "mtp_history_policy": KnobEntry(
            state="default_on",
            value="committed",
            provenance=(
                "2026-07-24 mtpk d1, 1k-ctx real code: committed history "
                "accept 0.903 vs 0.262 unprimed; draft is context-starved "
                "without prompt priming."
            ),
        ),
        "draft_core": KnobEntry(
            state="default_on",
            value="device",
            provenance=(
                "2026-07-24 mtpk d1 committed, 1k-ctx real code: device "
                "core 37.9 tok/s vs stock core 35.3-36.5 (AR 35.9) at "
                "accept 0.903 — first configuration where K=1 beats AR."
            ),
        ),
        "verify_strategy": KnobEntry(
            state="default_on",
            value="batched",
            provenance=(
                "2026-07-24 d1 sweep: capture_commit 32.0 (accept drops "
                "0.903->0.776, off-lane numerics), graphbank 27.4, "
                "graphbank_capture_commit 26.5 — batched wins on this "
                "family."
            ),
        ),
        "speculative_depth": KnobEntry(
            state="default_on",
            value=1,
            provenance=(
                "2026-07-24 depth sweep @ accept ~0.9: d2 28.9, d3 23.5 "
                "vs d1 35.3-37.9 — each extra draft position multiplies "
                "MoE expert bytes in verify; depth 1 is the win."
            ),
        ),
        "compile_ar_forward": KnobEntry(
            state="default_off",
            value=False,
            provenance=(
                "2026-07-24 live A/B, engagement-proofed 256/256: null win "
                "(35.1 vs 35.9 eager) — the host graph-rebuild tax this "
                "targets is not present on the lean mlx-lm forward; greedy "
                "token parity also flipped (gather_qmm kernel selection)."
            ),
        ),
    },
)

_PROFILES: dict[str, OptimizationProfile] = {
    _HY3_OQ2E.model_key: _HY3_OQ2E,
}

_MODEL_KEY_ALIASES = {
    "mlx-community/hy3-oq2e": "hy3-oq2e",
    "hy3-oq2e-stock": "hy3-oq2e",
    "hy3-oq2e-stock-mtp": "hy3-oq2e",
    "hy3-oq2e-r4": "hy3-oq2e",
    "hy3-oq2e-r4-mtpbf16": "hy3-oq2e",
}


def canonical_model_key(model_key: str) -> str:
    lowered = model_key.strip().lower()
    return _MODEL_KEY_ALIASES.get(lowered, lowered)


def get_profile(model_key: str) -> OptimizationProfile | None:
    return _PROFILES.get(canonical_model_key(model_key))


def resolve_profile_defaults(model_key: str) -> dict[str, KnobEntry]:
    """Measured defaults for a model key; empty when unregistered."""

    profile = get_profile(model_key)
    if profile is None:
        return {}
    return dict(profile.knobs)


def profile_conflict_warnings(
    model_key: str, observed: Mapping[str, Any]
) -> list[str]:
    """Advisory-only: explicit values conflicting with measured defaults."""

    warnings: list[str] = []
    for name, entry in resolve_profile_defaults(model_key).items():
        if name not in observed or observed[name] is None:
            continue
        if entry.state not in {"default_on", "default_off"}:
            continue
        if observed[name] != entry.value:
            warnings.append(
                f"profile suggests {name}={entry.value!r} for "
                f"{canonical_model_key(model_key)!r} (got {observed[name]!r}): "
                f"{entry.provenance}"
            )
    return warnings


def not_applicable_violations(
    model_key: str, knob_values: Mapping[str, Any]
) -> list[str]:
    """Enforceable knobs explicitly set despite ``not_applicable`` state."""

    violations: list[str] = []
    for name, entry in resolve_profile_defaults(model_key).items():
        if entry.state != "not_applicable" or name not in ENFORCEABLE_KNOBS:
            continue
        value = knob_values.get(name)
        if value in (None, False, "off", "none", ""):
            continue
        violations.append(
            f"{name}={value!r} is marked not_applicable for "
            f"{canonical_model_key(model_key)!r}: {entry.provenance}"
        )
    return violations
