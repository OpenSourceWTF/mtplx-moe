"""Per-model optimization-profile registry (issue #99, v1).

Every execution optimization this campaign measured is model-conditional:
the same knob that buys hy3 +2.6 tok/s is meaningless (or harmful) on
GLM-5.2 or the resident qwen serving path. This module records those
MEASURED decisions as reviewable in-repo data — a change to a default is
a diff to this file, with the measurement named in ``provenance``.

Profiles are not an autotuner. Entries carry one of four states:

- ``default_on``: measured win; the value is the recommended default.
- ``default_off``: applicable but measured harmful; leave it off.
- ``not_applicable``: structurally meaningless for this model (or
  measured net-negative at the operating envelope — see the entry's
  provenance for which).
- ``unvalidated``: mechanism exists but no end-to-end measurement backs
  a default either way.

Consumers (v1, read-only):

- ``resolve_profile_defaults(model_key)`` — benchmark scripts print a
  "profile suggests" advisory when the user's flags conflict with a
  measured default (warning only; no behavior change).
- ``not_applicable_violations(model_key, knob_values)`` — the expert
  runtime raises when a knob marked ``not_applicable`` is explicitly
  forced.  Enforcement is limited to :data:`ENFORCEABLE_KNOBS`: knobs
  whose "cheap check" is pure config inspection and whose inapplicability
  is structural.  This generalizes the loud guards already in
  ``resident_loader`` (``_verify_kv_quant_honored``; the
  "matched no resident Linear modules" raise) to config time with
  provenance attached — no new runtime probes.

This module is MLX-free and imports nothing from the rest of mtplx so
that the runtime, benchmark scripts, and offline tools can all consult
it without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "VALID_STATES",
    "KNOB_NAMES",
    "ENFORCEABLE_KNOBS",
    "KnobEntry",
    "OptimizationProfile",
    "PROFILES",
    "get_profile",
    "resolve_profile_defaults",
    "profile_conflict_warnings",
    "not_applicable_violations",
]


VALID_STATES = ("default_on", "default_off", "not_applicable", "unvalidated")

# Canonical knob vocabulary. Every profile must carry an entry for every
# knob so "we never looked" (unvalidated) is always distinguishable from
# "the profile is silent".
KNOB_NAMES = (
    "proj_quant",
    "proj_requant",
    "kv_quant",
    "expert_integrity",
    "split_route_release",
    "prefetch_slots",
    "miss_shadow",
    "miss_shadow_layers",
    "submit_cadence_env",
    "island_source",
    "island_layer_count",
    "mtp_depth",
    "verify_strategy",
)

# Knobs whose not_applicable state is enforced (raise on force) by the
# expert runtime.  Membership requires BOTH: (a) forcing is detectable by
# pure ExpertStreamingConfig inspection, and (b) the inapplicability is
# structural — the runtime would fail loudly later anyway (kv_quant:
# glm make_cache ignores the overlay, guarded in resident_loader;
# proj_quant: load-time nn.quantize matches no resident Linears).
# island_layer_count is deliberately NOT here even where a profile marks
# it not_applicable: GLM's verdict is envelope-conditional (net negative
# at 96 GiB) and the census lane (#101) ships GLM as its first consumer —
# the benchmark advisory surfaces the conflict instead.
ENFORCEABLE_KNOBS = frozenset({"proj_quant", "kv_quant"})


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
            raise TypeError("knobs must be a mapping of knob name -> entry")
        names = set(self.knobs)
        missing = sorted(set(KNOB_NAMES) - names)
        if missing:
            raise ValueError(
                f"profile {self.model_key!r} is missing knob entries: "
                f"{missing}"
            )
        unknown = sorted(names - set(KNOB_NAMES))
        if unknown:
            raise ValueError(
                f"profile {self.model_key!r} has unknown knobs: {unknown} "
                f"(extend KNOB_NAMES if this is intentional)"
            )
        for name, entry in self.knobs.items():
            if not isinstance(entry, KnobEntry):
                raise TypeError(
                    f"profile {self.model_key!r} knob {name!r} must be a "
                    f"KnobEntry, got {type(entry).__name__}"
                )
        object.__setattr__(self, "knobs", MappingProxyType(dict(self.knobs)))


_HY3_EXPERT_Q2 = OptimizationProfile(
    model_key="hy3-expert-q2",
    knobs={
        "proj_quant": KnobEntry(
            state="default_on",
            value="q4",
            provenance=(
                "2026-07-17 #51 sub-96 ladder (commit d7a695a): load-time "
                "g64 affine quantize of attention/shared/dense-MLP "
                "Linears; K3 1024/1024 11.107 -> 13.44 +/-0.05 tok/s, "
                "wired peak 83.1 -> 71.3 GiB (acceptance 1.994 -> 1.813; "
                "bandwidth win dominates)"
            ),
        ),
        "proj_requant": KnobEntry(
            state="unvalidated",
            value=None,
            provenance=(
                "mechanism added 2026-07-21 for the oq2e resident "
                "experiment; no measurement for this model"
            ),
        ),
        "kv_quant": KnobEntry(
            state="unvalidated",
            value="q4",
            provenance=(
                "wired for the trunk KV, batched verify only "
                "(capture_commit adapts dense trunk KV state); no tok/s "
                "delta posted on #51; the #97 publication matrix (q4 KV) "
                "is the planned measurement"
            ),
        ),
        "expert_integrity": KnobEntry(
            state="default_on",
            value="headers-only",
            provenance=(
                "2026-07-17 #51 30-tps-at-90g ledger: +2.6 tok/s at the "
                "90 GiB knob, K2/2048 (per-read sha256 was ~10 ms/token, "
                "23% of the forward call)"
            ),
        ),
        "split_route_release": KnobEntry(
            state="default_on",
            value="deferred",
            provenance=(
                "2026-07-17 #51 30-tps-at-90g ledger: +0.26 tok/s with "
                "async part submission (without submission 8% SLOWER — "
                "per-part fences doubled as GPU submission points); "
                "requires deferred_pin_release"
            ),
        ),
        "prefetch_slots": KnobEntry(
            state="unvalidated",
            value=8,
            provenance=(
                "mechanism validated 2026-07-17 (#51): residual-stream "
                "lookahead router_(N+L)(h_N) top-8 overlap "
                "74.3/66.1/60.8% at L=1/2/3 vs 4.2% identity baseline "
                "(400 tokens, offline sanity 1.000); ~8 ring slots per "
                "streamed layer per design; race fix pending (#99); "
                "history/draft prefetch measured dead (4-24% recall); no "
                "end-to-end tok/s delta yet"
            ),
        ),
        "miss_shadow": KnobEntry(
            state="unvalidated",
            value="t158",
            provenance=(
                "2026-07-17 #103 probe on real Q2 records: t158 "
                "combine-cosine 0.929, b1 0.029 NO-GO from Q2 sources "
                "(the affine zero level carries ~50% of weight mass); "
                "shadow_gather_mm is parity-grade, no perf window (#102)"
            ),
        ),
        "miss_shadow_layers": KnobEntry(
            state="unvalidated",
            value=None,
            provenance=(
                "coverage cap exists (a37ff18, #103); no measured "
                "coverage sweep for hy3"
            ),
        ),
        "submit_cadence_env": KnobEntry(
            state="unvalidated",
            value=None,
            provenance=(
                "MTPLX_HY3_SUBMIT_CADENCE landed 8650cb0 (decode-lane "
                "async_eval checkpoints, bitwise-tested, off by default); "
                "#106 lists it as landed-this-campaign, but no standalone "
                "cadence value or tok/s delta is posted on #51"
            ),
        ),
        "island_source": KnobEntry(
            state="default_on",
            value="spec-pin-order",
            provenance=(
                "measured static order: 511-step decode route trace "
                "2026-07-16, ranked by ascending top-147 expert coverage "
                "(spec island_pin_order); near-uniform routing makes "
                "census-vs-spec agreement the #98 validation gate"
            ),
        ),
        "island_layer_count": KnobEntry(
            state="default_on",
            value=60,
            provenance=(
                "2026-07-17 #51 final ladder (af0c37f, 2eda9dc): islands "
                "52 -> 60 (hit 0.49 -> 0.73, ~60 -> ~15 miss-services/"
                "token) took 13.44 -> 22.72 +/-0.03 tok/s K3 sub-96 at "
                "79.7 GiB wired; envelope-exact at 96 GiB (MLX cap 83.34 "
                "+ paged band 12.66) — value is envelope-dependent"
            ),
        ),
        "mtp_depth": KnobEntry(
            state="default_on",
            value=3,
            provenance=(
                "K3 optimal at every fixed-cost level measured on #51: "
                "C5 sweep K2 9.764 / K3 10.332 / K4 9.908; mmap champion "
                "K4 9.98 / K5 9.41; final 22.72 at K3 (2026-07-17); the "
                "90g/2048 ledger ran K2 — depth is config-dependent "
                "(K2/K3)"
            ),
        ),
        "verify_strategy": KnobEntry(
            state="default_on",
            value="batched",
            provenance=(
                "every #51 champion config through the 22.7 ladder ran "
                "batched verify; kv_quant requires batched "
                "(capture_commit adapts dense trunk KV); "
                "compiled-verify-mode untested in champion configs (#106)"
            ),
        ),
    },
)


_GLM52_EXPERT_Q2 = OptimizationProfile(
    model_key="glm52-expert-q2",
    knobs={
        "proj_quant": KnobEntry(
            state="not_applicable",
            value=None,
            provenance=(
                "#99 2026-07-17: resident stack already ships quantized — "
                "load-time nn.quantize has no BF16 resident Linears to "
                "match (resident_loader raises 'matched no resident "
                "Linear modules'); the BF16 NextN head (18.5 GiB) is "
                "tracked separately (#100)"
            ),
        ),
        "proj_requant": KnobEntry(
            state="unvalidated",
            value=None,
            provenance=(
                "mechanism added 2026-07-21 for the oq2e resident "
                "experiment; no measurement for this model"
            ),
        ),
        "kv_quant": KnobEntry(
            state="not_applicable",
            value=None,
            provenance=(
                "#99 2026-07-17: MLA KV is tiny (95 KB/token vs hy3's "
                "320 KB) and glm make_cache ignores _mtplx_kv_quant — "
                "probe-guarded at load since 2fc8ff9 "
                "(_verify_kv_quant_honored)"
            ),
        ),
        "expert_integrity": KnobEntry(
            state="default_on",
            value="headers-only",
            provenance=(
                "2026-07-17 GLM window-1 session: headers-only 1.43x "
                "decode vs per-read sha256 (~300 miss-assignments/token; "
                "#99 filed it as 'expected large — pending'); session "
                "measurement, not yet posted to the tracker"
            ),
        ),
        "split_route_release": KnobEntry(
            state="default_on",
            value="deferred",
            provenance=(
                "2026-07-17 GLM window-1 session: deferred 1.05x decode; "
                "session measurement, not yet posted to the tracker; "
                "requires deferred_pin_release"
            ),
        ),
        "prefetch_slots": KnobEntry(
            state="unvalidated",
            value=None,
            provenance=(
                "#99: needs GLM wiring; concentrated routing (52-54% "
                "decode hit at 20.7% capacity, #98) changes the lookahead "
                "value — no GLM measurement"
            ),
        ),
        "miss_shadow": KnobEntry(
            state="unvalidated",
            value="t158",
            provenance=(
                "2026-07-17 #103 probe: t158 combine-cosine 0.922 from "
                "Q2 records (b1 0.017 NO-GO); full coverage does not fit "
                "(#102 pricing: t158 all-routed 158.2 GiB > 100 GiB wired "
                "limit) — needs a miss_shadow_layers cap; no perf window"
            ),
        ),
        "miss_shadow_layers": KnobEntry(
            state="unvalidated",
            value=None,
            provenance=(
                "required alongside miss_shadow on glm (#102 pricing: "
                "full shadow coverage exceeds the wired limit); no "
                "measured coverage sweep"
            ),
        ),
        "submit_cadence_env": KnobEntry(
            state="not_applicable",
            value=None,
            provenance=(
                "MTPLX_HY3_SUBMIT_CADENCE is read only by the hy3 model "
                "path (mtplx/models/hy3_mlx.py); glm decode never "
                "consults it"
            ),
        ),
        "island_source": KnobEntry(
            state="default_on",
            value="census",
            provenance=(
                "#98/#101 2026-07-17: concentrated routing rewards "
                "census placement; measured pin order a6417ca (window-1b "
                "trace: 1023 steps, 613,800 routed assignments, cov@53 "
                "0.499 vs 0.517 measured hit, Spearman >= 0.956 across K "
                "32..128) seeds and validates the auto-census"
            ),
        ),
        "island_layer_count": KnobEntry(
            state="not_applicable",
            value=None,
            provenance=(
                "2026-07-17 GLM window-1 session: whole-layer islands "
                "measured net NEGATIVE at the 96 GiB envelope (cache "
                "slots beat pins under concentrated routing); session "
                "measurement, not yet posted to the tracker; "
                "envelope-conditional — advisory only, not enforced, so "
                "other envelopes can re-measure"
            ),
        ),
        "mtp_depth": KnobEntry(
            state="default_on",
            value=3,
            provenance=(
                "#99: d3 baseline only; #106: conditional acceptance "
                "0.93/0.91/0.85 argues deeper depth amortizes verify "
                "better — d4/d5 row queued, unmeasured"
            ),
        ),
        "verify_strategy": KnobEntry(
            state="unvalidated",
            value="batched",
            provenance=(
                "no GLM verify-strategy comparison in the tracker; "
                "batched is the code default"
            ),
        ),
    },
)


_QWEN36_27B = OptimizationProfile(
    model_key="qwen36-27b",
    knobs={
        "proj_quant": KnobEntry(
            state="not_applicable",
            value=None,
            provenance=(
                "#99 2026-07-17: full model already quantized (4-bit "
                "serving build); resident serving path has no "
                "resident/streamed split for load-time quantization"
            ),
        ),
        "proj_requant": KnobEntry(
            state="unvalidated",
            value=None,
            provenance=(
                "mechanism added 2026-07-21 for the oq2e resident "
                "experiment; no measurement for this model"
            ),
        ),
        "kv_quant": KnobEntry(
            state="default_off",
            value=None,
            provenance=(
                "#99 2026-07-17: known regression, do not use; "
                "serving-side measurement 2026-07-07 (mtplx 2.0.1 scan "
                "workload): q4 KV collapsed decode 42 -> 11.5 tok/s — "
                "ops-notes measurement, not in this tracker"
            ),
        ),
        "expert_integrity": KnobEntry(
            state="not_applicable",
            value=None,
            provenance=(
                "resident serving: no expert streaming runtime, no "
                "record reads to verify"
            ),
        ),
        "split_route_release": KnobEntry(
            state="not_applicable",
            value=None,
            provenance=(
                "resident serving: no streamed split-route slots to "
                "release"
            ),
        ),
        "prefetch_slots": KnobEntry(
            state="not_applicable",
            value=None,
            provenance="resident serving: no expert cache to prefetch into",
        ),
        "miss_shadow": KnobEntry(
            state="not_applicable",
            value=None,
            provenance="resident serving: no miss economy to shadow",
        ),
        "miss_shadow_layers": KnobEntry(
            state="not_applicable",
            value=None,
            provenance=(
                "resident serving: miss_shadow is itself not applicable"
            ),
        ),
        "submit_cadence_env": KnobEntry(
            state="not_applicable",
            value=None,
            provenance=(
                "MTPLX_HY3_SUBMIT_CADENCE is hy3-model-scoped; the qwen "
                "serving path never consults it"
            ),
        ),
        "island_source": KnobEntry(
            state="not_applicable",
            value=None,
            provenance="resident serving: no island placement to derive",
        ),
        "island_layer_count": KnobEntry(
            state="not_applicable",
            value=None,
            provenance="resident serving: the whole model is resident",
        ),
        "mtp_depth": KnobEntry(
            state="default_on",
            value=3,
            provenance=(
                "2026-07-17 isolated depth matrix: K3 decode ceiling "
                "58-61 tok/s (serving+workload figure ~42 tok/s; MTP "
                "accept ~0.84); isolated-bench measurement recorded in "
                "ops notes and #99's table (K3, isolated matrix)"
            ),
        ),
        "verify_strategy": KnobEntry(
            state="unvalidated",
            value=None,
            provenance=(
                "resident serving via native MTP; no verify-strategy "
                "comparison recorded"
            ),
        ),
    },
)


_HY3_EXPERT_OQ2E = OptimizationProfile(
    model_key="hy3-expert-oq2e",
    knobs={
        "proj_quant": KnobEntry(
            state="not_applicable",
            value=None,
            provenance=(
                "2026-07-21: oQ2e residents ship pre-quantized q8-gs64 and "
                "load via the config-driven path; _runtime_quantize_projections "
                "matches zero BF16 Linears and raises. Serving directive for "
                "this checkpoint: stock bytes, never requantize."
            ),
        ),
        "proj_requant": KnobEntry(
            state="unvalidated",
            value="q4",
            provenance=(
                "EXPERIMENT arm: q8->q4 double-quantization of "
                "proj_quant_covers scope; quality gates = 20-task HumanEval "
                "vs 0.95 and WikiText-2 vs 6.443 (2026-07-21); pending"
            ),
        ),
        "kv_quant": KnobEntry(
            state="unvalidated",
            value=None,
            provenance=(
                "no oq2e measurement (2026-07-21 campaign ran default KV); "
                "no basis inherited — q2-lane KV history does not transfer "
                "automatically"
            ),
        ),
        "expert_integrity": KnobEntry(
            state="default_on",
            value="headers-only",
            provenance=(
                "2026-07-21 oq2e campaign ran headers-only throughout; "
                "measurement basis is the hy3-expert-q2 headers-only win "
                "(no oq2e-specific A/B)"
            ),
        ),
        "split_route_release": KnobEntry(
            state="default_on",
            value="deferred",
            provenance=(
                "inherited from hy3-expert-q2; NOTE: inert at "
                "island-layer-count 79 (DenseIslandSwitchGLU never consults "
                "it) — load-bearing only at reduced islands, e.g. the 88 GiB "
                "islands-74 configuration"
            ),
        ),
        "prefetch_slots": KnobEntry(
            state="unvalidated",
            value=None,
            provenance=(
                "no oq2e measurement; near-uniform hy3 routing made "
                "prefetch lose on the q2 bank — re-measure before enabling "
                "at any streamed operating point"
            ),
        ),
        "miss_shadow": KnobEntry(
            state="unvalidated",
            value=None,
            provenance="q2-bank-specific shadow machinery; no oq2e basis",
        ),
        "miss_shadow_layers": KnobEntry(
            state="unvalidated",
            value=None,
            provenance="see miss_shadow; no oq2e basis",
        ),
        "submit_cadence_env": KnobEntry(
            state="default_on",
            value="8",
            provenance=(
                "2026-07-21 campaign ran MTPLX_HY3_SUBMIT_CADENCE=8; "
                "confirmed load-bearing at full residency "
                "(mtplx/models/hy3_mlx.py cadence path)"
            ),
        ),
        "island_source": KnobEntry(
            state="default_on",
            value="spec-pin-order",
            provenance=(
                "spec reuses HY3_EXPERT_Q2's measured pin order (routing is "
                "a router-weight property); order is irrelevant at 79 "
                "islands and a starting point, not re-measured truth, at "
                "reduced counts"
            ),
        ),
        "island_layer_count": KnobEntry(
            state="default_on",
            value=79,
            provenance=(
                "2026-07-21: fully resident fits at memory-limit 103GiB "
                "(hard peak 91.9 GiB): AR 30.17 / K2 36.82 (1 unclassified "
                "divergence) / K3 31.81 bit-exact. At 95GiB limit the fixed "
                "footprint exceeds by 4.21 GiB; islands 74 measured K3 22.15"
            ),
        ),
        "mtp_depth": KnobEntry(
            state="default_on",
            value=3,
            provenance=(
                "2026-07-21: K3 31.81 tok/s bit-exact vs AR at both "
                "envelopes; K2 is faster (36.82) but carried one "
                "unclassified token divergence per run — K3 is the "
                "exact-parity default until K2's divergence is attributed"
            ),
        ),
        "verify_strategy": KnobEntry(
            state="default_on",
            value="batched",
            provenance=(
                "2026-07-21 campaign ran batched in every cell; basis "
                "inherited from hy3-expert-q2 (no oq2e-specific A/B)"
            ),
        ),
    },
)


PROFILES: Mapping[str, OptimizationProfile] = MappingProxyType(
    {
        profile.model_key: profile
        for profile in (
            _HY3_EXPERT_Q2,
            _HY3_EXPERT_OQ2E,
            _GLM52_EXPERT_Q2,
            _QWEN36_27B,
        )
    }
)


def get_profile(model_key: str) -> OptimizationProfile | None:
    """Return the profile for ``model_key``, or None when unregistered."""

    return PROFILES.get(model_key)


def resolve_profile_defaults(model_key: str) -> dict[str, KnobEntry]:
    """Return knob -> entry for ``model_key`` ({} when unregistered).

    Read-only consumer hook: callers compare their effective settings
    against the returned entries; nothing here changes behavior.
    """

    profile = get_profile(model_key)
    if profile is None:
        return {}
    return dict(profile.knobs)


def _is_forced(value: Any) -> bool:
    """True when a knob value is explicitly engaged (not off/unset)."""

    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, (str, tuple, list, set, frozenset)):
        return len(value) > 0
    return True


def _conflicts(entry: KnobEntry, observed: Any) -> bool:
    if entry.state == "not_applicable":
        return _is_forced(observed)
    if entry.state == "default_off":
        return _is_forced(observed) and observed != entry.value
    if entry.state == "default_on":
        if isinstance(observed, (tuple, list, set, frozenset)):
            # Sweep-style knobs (e.g. an mtp_depth matrix): the measured
            # default just has to be part of the sweep.
            return entry.value not in observed
        return observed != entry.value
    return False  # unvalidated entries never warn


def profile_conflict_warnings(
    model_key: str,
    observed: Mapping[str, Any],
) -> list[str]:
    """Advisory lines where ``observed`` conflicts with measured defaults.

    ``observed`` maps knob name -> the effective value the caller is
    about to run with; knobs absent from the mapping are skipped (a
    present ``None`` means "off/unset" and is compared). Purely
    informational — callers must only print the lines.
    """

    profile = get_profile(model_key)
    if profile is None:
        return []
    lines: list[str] = []
    for knob in KNOB_NAMES:
        if knob not in observed:
            continue
        entry = profile.knobs[knob]
        got = observed[knob]
        if not _conflicts(entry, got):
            continue
        if entry.state == "not_applicable":
            summary = f"{knob} is not_applicable for {model_key}"
        elif entry.state == "default_off":
            summary = f"profile suggests leaving {knob} off (default_off)"
        else:
            summary = (
                f"profile suggests {knob}={entry.value!r} (default_on)"
            )
        lines.append(
            f"[profile:{model_key}] {summary}; configured {got!r} — "
            f"{entry.provenance} [advisory only]"
        )
    return lines


def not_applicable_violations(
    model_key: str,
    knob_values: Mapping[str, Any],
) -> list[str]:
    """Messages for forced knobs the profile marks ``not_applicable``.

    Only knobs in :data:`ENFORCEABLE_KNOBS` are considered — the caller
    (the expert runtime) raises on a non-empty result. Unregistered
    model keys yield no violations: absence of a profile is not an
    error.
    """

    profile = get_profile(model_key)
    if profile is None:
        return []
    violations: list[str] = []
    for knob in sorted(set(knob_values) & ENFORCEABLE_KNOBS):
        entry = profile.knobs[knob]
        if entry.state != "not_applicable":
            continue
        value = knob_values[knob]
        if not _is_forced(value):
            continue
        violations.append(
            f"{knob}={value!r} is not_applicable for {model_key}: "
            f"{entry.provenance}"
        )
    return violations
