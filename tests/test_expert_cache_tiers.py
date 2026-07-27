"""Total-size accounting for the documented 32 GiB and 48 GiB cache tiers.

``--expert-cache-limit`` caps only the persistent expert cache. It does not
lower the profile's process ceiling, so the difference is left unallocated
rather than released. Readers (and agents) have repeatedly read the tier tables
as total-memory requirements, so these tests pin three things at once:

* the memory plan's parts sum exactly to the declared limit (no lost bytes);
* the slot/cache figures published in ``docs/advanced/ssd-streamed-moe.md``;
* the *total* footprint each tier actually reaches, which is the number the
  guide previously left implicit.

All of this is pure arithmetic: ``plan_expert_memory`` performs no model IO and
no GPU work.

The totals here are upper bounds. ``hy3-oq2e-64`` also sets ``proj_requant=q4``,
whose resident discount is computed from the on-disk manifest
(``proj_requant_plan_discount``) and so cannot be resolved without the model.
The realized footprint is that discount lower; no assertion here depends on it.
"""

from __future__ import annotations

import pytest

from mtplx.expert_profiles import load_expert_profiles, select_expert_profile
from mtplx.expert_streaming_models import get_model_spec, plan_expert_memory

GIB = 1024**3

HY3_KEY = "hy3-expert-oq2e"
HY3_CEILING = 71 * GIB
HY3_RESERVE = 7 * GIB
# What the shipped hy3-oq2e-64 profile actually configures: 49.9921875 GiB.
HY3_DEFAULT_CACHE = 53_678_702_592

GLM_KEY = "glm52-expert-q1t"
GLM_CEILING = 96 * GIB
GLM_RESERVE = 12 * GIB
GLM_TRANSIENT_SLOTS = 48
# The measured GLM plan's own cache bound, per the guide.
GLM_DEFAULT_CACHE = 72 * GIB

KV_TOKENS = 4096


def _plan(spec_key, *, ceiling, reserve, cache_cap=None, transient_slots=None):
    kwargs = {
        "total_limit_bytes": ceiling,
        "context_tokens": KV_TOKENS,
        "runtime_reserve_bytes": reserve,
    }
    if cache_cap is not None:
        kwargs["expert_cache_limit_bytes"] = cache_cap
    if transient_slots is not None:
        kwargs["transient_slots"] = transient_slots
    return plan_expert_memory(get_model_spec(spec_key), **kwargs)


def _hy3(cache_cap=None):
    return _plan(HY3_KEY, ceiling=HY3_CEILING, reserve=HY3_RESERVE, cache_cap=cache_cap)


def _glm(cache_cap=None):
    return _plan(
        GLM_KEY,
        ceiling=GLM_CEILING,
        reserve=GLM_RESERVE,
        cache_cap=cache_cap,
        transient_slots=GLM_TRANSIENT_SLOTS,
    )


def _parts(plan):
    """Every byte the plan claims, keyed by budget line."""

    return {
        "resident": plan.resident_bytes,
        "kv": plan.kv_bytes,
        "transient": plan.transient_bytes,
        "cache": plan.persistent_cache_bytes,
        "unallocated": plan.unallocated_bytes,
        "reserve": plan.runtime_reserve_bytes,
    }


def _used(plan):
    """Bytes the process actually reaches: everything except the unused tail."""

    return plan.total_limit_bytes - plan.unallocated_bytes


# --------------------------------------------------------------------------
# 1. The parts must sum to the limit -- "calculate the total sizes correctly"
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ceiling_gib", [20, 24, 28, 32, 40, 48, 56, 64, 71])
@pytest.mark.parametrize("cache_cap_gib", [None, 32, 48])
def test_hy3_plan_parts_sum_to_the_declared_limit(ceiling_gib, cache_cap_gib):
    plan = _plan(
        HY3_KEY,
        ceiling=ceiling_gib * GIB,
        reserve=HY3_RESERVE,
        cache_cap=None if cache_cap_gib is None else cache_cap_gib * GIB,
    )
    assert sum(_parts(plan).values()) == plan.total_limit_bytes


def test_glm_plan_parts_sum_to_the_declared_limit():
    for cache_cap in (None, 32 * GIB, 48 * GIB):
        plan = _glm(cache_cap)
        assert sum(_parts(plan).values()) == plan.total_limit_bytes


def test_no_budget_line_is_negative():
    for plan in (_hy3(), _hy3(32 * GIB), _glm(), _glm(32 * GIB)):
        for name, value in _parts(plan).items():
            assert value >= 0, name


# --------------------------------------------------------------------------
# 2. The slot/cache figures published in the guide
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cache_cap_gib", "expected_slots", "expected_cache_gib"),
    [(32, 81, 31.636), (40, 102, 39.838), (47, 120, 46.868)],
)
def test_hy3_documented_tier_geometry(cache_cap_gib, expected_slots, expected_cache_gib):
    plan = _hy3(cache_cap_gib * GIB)
    assert plan.slots_per_layer == expected_slots
    assert plan.persistent_cache_bytes / GIB == pytest.approx(expected_cache_gib, abs=5e-4)
    # A layer-scoped cache rounds down to whole slots, never up past the cap.
    assert plan.persistent_cache_bytes <= cache_cap_gib * GIB


@pytest.mark.parametrize(
    ("cache_cap_gib", "expected_slots", "expected_cache_gib"),
    [(32, 51, 31.517), (48, 77, 47.585)],
)
def test_glm_documented_tier_geometry(cache_cap_gib, expected_slots, expected_cache_gib):
    plan = _glm(cache_cap_gib * GIB)
    assert plan.slots_per_layer == expected_slots
    assert plan.persistent_cache_bytes / GIB == pytest.approx(expected_cache_gib, abs=5e-4)
    assert plan.persistent_cache_bytes <= cache_cap_gib * GIB


# --------------------------------------------------------------------------
# 3. The total footprint each tier actually reaches
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cache_cap_bytes", "expected_used_gib"),
    [
        (HY3_DEFAULT_CACHE, 66.992),
        (47 * GIB, 63.868),
        (40 * GIB, 56.838),
        (32 * GIB, 48.636),
    ],
)
def test_hy3_tier_total_footprint(cache_cap_bytes, expected_used_gib):
    plan = _hy3(cache_cap_bytes)
    assert _used(plan) / GIB == pytest.approx(expected_used_gib, abs=5e-4)


def test_documented_hy3_tiers_clear_the_installed_ram_check_on_a_64_gib_mac():
    """Both published sub-default tiers sit under 64 GiB of installed RAM.

    The 47 GiB row clears it by only 132 MiB, so it is the installed check it
    passes, not the available-memory gate a real 64 GiB machine applies.
    """

    for cap_gib, margin_gib in ((47, 0.13), (40, 7.16), (32, 15.36)):
        used = _used(_hy3(cap_gib * GIB))
        assert used < 64 * GIB
        assert (64 * GIB - used) / GIB == pytest.approx(margin_gib, abs=5e-3)


@pytest.mark.parametrize(
    ("cache_cap_bytes", "expected_used_gib"),
    [(GLM_DEFAULT_CACHE, 94.349), (48 * GIB, 70.248), (32 * GIB, 54.180)],
)
def test_glm_tier_total_footprint(cache_cap_bytes, expected_used_gib):
    plan = _glm(cache_cap_bytes)
    assert _used(plan) / GIB == pytest.approx(expected_used_gib, abs=5e-4)


def test_hy3_profile_default_cache_matches_the_documented_allowance():
    """The 49.9921875 GiB figure the guide quotes is the profile's own cap."""

    profile = load_expert_profiles()["hy3-oq2e-64"]
    assert profile.config["expert_cache_limit_bytes"] == HY3_DEFAULT_CACHE
    assert HY3_DEFAULT_CACHE / GIB == pytest.approx(49.9921875, abs=1e-7)
    assert _hy3(HY3_DEFAULT_CACHE).slots_per_layer == 128


def test_hy3_fixed_costs_do_not_move_with_the_cache_cap():
    """Only the cache line responds to the cap; the rest of the budget is fixed."""

    uncapped = _hy3()
    capped = _hy3(32 * GIB)
    for line in ("resident", "kv", "transient", "reserve"):
        assert _parts(capped)[line] == _parts(uncapped)[line]
    assert capped.persistent_cache_bytes < uncapped.persistent_cache_bytes


def test_cache_cap_strands_memory_instead_of_lowering_the_ceiling():
    """The reason a tier still *declares* 71 GiB while only using ~48.6 GiB."""

    uncapped = _hy3()
    capped = _hy3(32 * GIB)
    assert capped.total_limit_bytes == uncapped.total_limit_bytes == HY3_CEILING
    reclaimed = uncapped.persistent_cache_bytes - capped.persistent_cache_bytes
    assert capped.unallocated_bytes == uncapped.unallocated_bytes + reclaimed
    # ~22.4 GiB sits unused inside the declared ceiling.
    assert capped.unallocated_bytes / GIB == pytest.approx(22.364, abs=5e-4)


# --------------------------------------------------------------------------
# 4. Admission gates on the declared ceiling, before any override applies
# --------------------------------------------------------------------------


def test_profile_ceiling_is_envelope_plus_reserve_not_installed_ram():
    profile = load_expert_profiles()["hy3-oq2e-64"]
    assert profile.weight_envelope_bytes == 64 * GIB
    assert profile.process_ceiling_bytes == 71 * GIB
    # The "64" in the name is the envelope; the gate uses the ceiling.
    assert profile.process_ceiling_bytes > 64 * GIB


def test_named_profile_is_rejected_on_a_64_gib_machine():
    profile = load_expert_profiles()["hy3-oq2e-64"]
    with pytest.raises(ValueError, match="required"):
        select_expert_profile(
            "hy3-oq2e-64",
            model_key=profile.model_key,
            installed_ram_bytes=64 * GIB,
            available_bytes=64 * GIB,
        )


def test_no_hy3_profile_is_admitted_on_a_64_gib_machine():
    """`auto` cannot fall back to a smaller tier: none is promoted."""

    profile = load_expert_profiles()["hy3-oq2e-64"]
    with pytest.raises(ValueError, match="no promoted expert profile fits"):
        select_expert_profile(
            "auto",
            model_key=profile.model_key,
            installed_ram_bytes=64 * GIB,
            available_bytes=64 * GIB,
        )


def test_the_rejected_footprint_would_have_fit():
    """A 32 GiB-cache run needs ~48.6 GiB, well inside a 64 GiB machine.

    Admission compares the *declared* ceiling, so the cap cannot unlock the
    run even though the realized plan fits. This is the gap the guide has to
    state plainly.
    """

    capped = _hy3(32 * GIB)
    assert _used(capped) < 64 * GIB
    assert capped.fits_fixed
    assert capped.slots_per_layer >= 1
    assert load_expert_profiles()["hy3-oq2e-64"].process_ceiling_bytes > 64 * GIB


def test_expert_cli_select_wrapper_forwards_overrides(monkeypatch):
    """`expert_cli` wraps `select_expert_profile` to defer runtime imports.

    Tests that call `mtplx.expert_profiles` directly skip that indirection, so a
    stale signature here surfaces only when a server actually starts. Pin the
    forwarding instead of the outcome, which would depend on live memory.
    """

    import mtplx.expert_profiles as profiles_module
    from mtplx import expert_cli

    seen: dict[str, object] = {}

    def fake_select(requested, *, model_key, overrides=None, **kwargs):
        seen.update(requested=requested, model_key=model_key, overrides=overrides)
        return "sentinel-profile"

    monkeypatch.setattr(profiles_module, "select_expert_profile", fake_select)

    result = expert_cli.select_expert_profile(
        "hy3-oq2e-64",
        model_key="hy3-expert-oq2e",
        overrides={"expert_cache_limit_bytes": "32GiB"},
    )

    assert result == "sentinel-profile"
    assert seen["overrides"] == {"expert_cache_limit_bytes": "32GiB"}
    assert seen["requested"] == "hy3-oq2e-64"


def test_streaming_plan_floor_is_far_below_the_promoted_ceiling():
    """Bound the real floor so nobody reads 71 GiB as an architectural minimum."""

    floor = _plan(HY3_KEY, ceiling=18 * GIB, reserve=HY3_RESERVE)
    assert floor.fits_fixed
    assert floor.slots_per_layer >= 1
    too_small = _plan(HY3_KEY, ceiling=17 * GIB, reserve=HY3_RESERVE)
    assert not (too_small.fits_fixed and too_small.slots_per_layer >= 1)
