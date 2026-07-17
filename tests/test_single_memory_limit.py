"""Single unified-memory-limit policy over KV admission and the expert cache.

Issue #46: ``memory_limit_bytes`` is the only required memory knob.  When no
explicit ``expert_cache_limit_bytes`` is supplied, the runtime derives the
streamed expert-cache byte allowance from the total limit at every KV
boundary and synchronously evicts policy victims down to it before KV may
grow.  Supplying ``expert_cache_limit_bytes`` preserves the static
whole-context reservation exactly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from test_expert_slots_runtime import _artifact, _global_artifact, _spec

from mtplx.expert_manifest import save_expert_manifest
from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    ExpertStreamingRuntime,
    derived_expert_cache_allowance_bytes,
)
from mtplx.expert_streaming import (
    GlobalExpertSlotBank,
    LayerExpertSlotBank,
    SlotLoad,
)
from mtplx.expert_streaming_models import plan_expert_memory


def _fixed_bytes(spec) -> int:
    return spec.resident_bytes + spec.transient_scratch_bytes


def _slot_bytes(spec) -> int:
    return spec.expert_record_bytes


def _derived_config(spec, *, memory_limit_bytes: int, max_live_kv_tokens: int, **kwargs):
    return ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=memory_limit_bytes,
        max_live_kv_tokens=max_live_kv_tokens,
        runtime_reserve_bytes=0,
        verify_artifact_headers=False,
        cache_policy="lru",
        **kwargs,
    )


def _open_runtime(
    tmp_path: Path,
    *,
    expert_count: int = 2,
    slots: int,
    max_live_kv_tokens: int,
    additional_resident_bytes: int = 0,
    **config_kwargs,
):
    root, spec, manifest, expected = _artifact(tmp_path, expert_count=expert_count)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    config = _derived_config(
        spec,
        memory_limit_bytes=(
            _fixed_bytes(spec)
            + additional_resident_bytes
            + slots * _slot_bytes(spec)
        ),
        max_live_kv_tokens=max_live_kv_tokens,
        **config_kwargs,
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        config,
        spec=spec,
        apply_memory_cap=False,
        additional_resident_bytes=additional_resident_bytes,
    )
    return runtime, spec, expected


def _tokens_for_bytes(spec, size: int) -> int:
    assert size % spec.kv_bytes_per_token == 0
    return size // spec.kv_bytes_per_token


def _decode(runtime, layer: int, expert: int) -> None:
    ready = runtime.ensure_route(layer, [expert], phase="decode")
    ready.release(synchronize=False)


def _slot_states(runtime) -> dict[str, int]:
    return runtime.snapshot(mx_module=object())["slots"]["states"]


def _policy(runtime) -> dict[str, object]:
    return runtime.snapshot(mx_module=object())["expert_cache_policy"]


# ---------------------------------------------------------------------------
# Config: one knob selects the derived policy; the second knob restores the
# static plan exactly.
# ---------------------------------------------------------------------------


def test_omitting_expert_cache_limit_engages_derived_policy() -> None:
    spec = _spec()
    config = _derived_config(
        spec,
        memory_limit_bytes=_fixed_bytes(spec) + 2 * _slot_bytes(spec),
        max_live_kv_tokens=864,
    )
    assert config.derived_expert_cache_policy is True
    plan = config.memory_plan(spec)
    # The post-load boundary plan reserves no whole-context KV up front.
    assert plan.context_tokens == 0
    assert plan.kv_bytes == 0
    assert plan.slots_per_layer == 2
    boundary = config.memory_plan(spec, live_kv_tokens=432)
    assert boundary.context_tokens == 432
    assert boundary.kv_bytes == 432 * spec.kv_bytes_per_token
    assert boundary.slots_per_layer == 1


def test_supplying_expert_cache_limit_preserves_static_plan_exactly() -> None:
    spec = _spec()
    limit = _fixed_bytes(spec) + 2 * _slot_bytes(spec)
    config = _derived_config(
        spec,
        memory_limit_bytes=limit,
        max_live_kv_tokens=864,
        expert_cache_limit_bytes=2 * _slot_bytes(spec),
    )
    assert config.derived_expert_cache_policy is False
    assert config.memory_plan(spec) == plan_expert_memory(
        spec,
        total_limit_bytes=limit,
        context_tokens=864,
        runtime_reserve_bytes=0,
        expert_cache_limit_bytes=2 * _slot_bytes(spec),
    )
    with pytest.raises(ExpertStreamingConfigurationError, match="derived"):
        config.memory_plan(spec, live_kv_tokens=0)


def test_metal_mmap_layout_keeps_the_static_plan() -> None:
    spec = _spec()
    config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=_fixed_bytes(spec) + 2 * _slot_bytes(spec),
        max_live_kv_tokens=864,
        runtime_reserve_bytes=0,
        verify_artifact_headers=False,
        slot_layout="metal-mmap",
        verify_sidecar_hash_at_open=True,
    )
    assert config.derived_expert_cache_policy is False
    assert config.memory_plan(spec).context_tokens == 864


# ---------------------------------------------------------------------------
# Boundary 1: post-load initialization of the derived allowance.
# ---------------------------------------------------------------------------


def test_open_sizes_pool_to_the_post_load_allowance(tmp_path: Path) -> None:
    runtime, spec, _expected = _open_runtime(
        tmp_path,
        slots=2,
        max_live_kv_tokens=864,
    )
    try:
        assert runtime.plan.slots_per_layer == 2
        policy = _policy(runtime)
        assert policy == {
            "derived": True,
            "allowance_bytes": 2 * _slot_bytes(spec),
            "persistent_capacity": 2,
            "cached_bytes": 0,
        }
    finally:
        runtime.close()


def test_static_runtime_reports_no_derived_policy(tmp_path: Path) -> None:
    root, spec, manifest, _expected = _artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    kv_reserve = 864 * spec.kv_bytes_per_token
    config = _derived_config(
        spec,
        memory_limit_bytes=_fixed_bytes(spec) + kv_reserve + 2 * _slot_bytes(spec),
        max_live_kv_tokens=864,
        expert_cache_limit_bytes=2 * _slot_bytes(spec),
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        config,
        spec=spec,
        apply_memory_cap=False,
    )
    try:
        assert runtime.plan.slots_per_layer == 2
        assert runtime.plan.context_tokens == 864
        assert _policy(runtime) == {
            "derived": False,
            "allowance_bytes": None,
            "persistent_capacity": None,
            "cached_bytes": 0,
        }
        admission = runtime.admit_kv_tokens(864)
        # The static path never touches the cache: full capacity remains.
        assert runtime._banks[1].persistent_capacity == 2
        _decode(runtime, 1, 0)
        _decode(runtime, 1, 1)
        assert runtime._banks[1].occupancy == 2
        admission.release()
        assert runtime._banks[1].persistent_capacity == 2
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# Boundary 2/3: KV growth evicts policy victims down to the new allowance and
# fails closed when the request cannot fit at all.
# ---------------------------------------------------------------------------


def test_kv_growth_evicts_lru_entries_to_the_new_allowance(tmp_path: Path) -> None:
    runtime, spec, expected = _open_runtime(
        tmp_path,
        expert_count=4,
        slots=3,
        max_live_kv_tokens=2000,
    )
    try:
        bank = runtime._banks[1]
        for expert in (0, 1, 2):
            _decode(runtime, 1, expert)
        _decode(runtime, 1, 0)  # refresh expert 0: LRU order is now 1, 2, 0
        assert bank.occupancy == 3

        grow = _tokens_for_bytes(spec, 2 * _slot_bytes(spec))
        admission = runtime.admit_kv_tokens(grow)
        assert bank.occupancy == 1
        assert bank.resident_experts == (0,)
        assert bank.persistent_capacity == 1
        policy = _policy(runtime)
        assert policy["allowance_bytes"] == _slot_bytes(spec)
        assert policy["persistent_capacity"] == 1
        assert policy["cached_bytes"] == _slot_bytes(spec)
        states = _slot_states(runtime)
        assert states["ready"] == 1
        assert states["empty"] == 3

        # The surviving entry stays on the ordinary hit path.
        ready = runtime.ensure_route(1, [0], phase="decode")
        assert ready.plan.hits == (0,)
        assert ready.plan.loads == ()
        assert bytes(ready.bindings[0].buffer) == expected[0]
        ready.release(synchronize=False)

        # Between boundaries a miss may only replace the policy victim; the
        # cache never regrows past the allowance while KV is held.
        _decode(runtime, 1, 1)
        assert bank.occupancy == 1
        assert bank.resident_experts == (1,)
        admission.release()
    finally:
        runtime.close()


def test_kv_growth_fails_closed_when_it_cannot_fit(tmp_path: Path) -> None:
    runtime, spec, _expected = _open_runtime(
        tmp_path,
        slots=2,
        max_live_kv_tokens=10_000,
    )
    try:
        limit = _tokens_for_bytes(spec, 2 * _slot_bytes(spec))
        with pytest.raises(
            ExpertStreamingConfigurationError, match="oversubscribes"
        ):
            runtime.admit_kv_tokens(limit + 1)
        assert runtime._live_kv_tokens == 0
        assert runtime._banks[1].persistent_capacity == 2

        admission = runtime.admit_kv_tokens(limit)
        assert runtime._banks[1].persistent_capacity == 0
        admission.release()
        assert runtime._banks[1].persistent_capacity == 2
    finally:
        runtime.close()


def test_kv_growth_fails_closed_when_victims_are_pinned(tmp_path: Path) -> None:
    runtime, spec, _expected = _open_runtime(
        tmp_path,
        slots=1,
        max_live_kv_tokens=10_000,
    )
    try:
        grow = _tokens_for_bytes(spec, _slot_bytes(spec))
        pinned = runtime.ensure_route(1, [0], phase="decode")
        with pytest.raises(ExpertStreamingConfigurationError, match="pinned"):
            runtime.admit_kv_tokens(grow)
        assert runtime._live_kv_tokens == 0
        pinned.release(synchronize=False)

        admission = runtime.admit_kv_tokens(grow)
        assert runtime._banks[1].occupancy == 0
        assert _slot_states(runtime)["ready"] == 0
        admission.release()
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# Boundary 4: KV shrink recomputes the larger allowance without allocating.
# ---------------------------------------------------------------------------


def test_kv_shrink_raises_allowance_without_allocating(tmp_path: Path) -> None:
    runtime, spec, _expected = _open_runtime(
        tmp_path,
        expert_count=4,
        slots=3,
        max_live_kv_tokens=2000,
    )
    try:
        bank = runtime._banks[1]
        _decode(runtime, 1, 0)
        admission = runtime.admit_kv_tokens(
            _tokens_for_bytes(spec, 2 * _slot_bytes(spec))
        )
        assert bank.persistent_capacity == 1

        counters_before = runtime.counters.as_dict()
        admission.release()
        assert bank.persistent_capacity == 3
        # Nothing was eagerly loaded by the shrink.
        assert runtime.counters.as_dict() == counters_before
        assert bank.occupancy == 1
        states = _slot_states(runtime)
        assert states["ready"] == 1
        assert states["empty"] == 3

        # Later misses refill the cache naturally up to the new allowance.
        ready = runtime.ensure_route(1, [1], phase="decode")
        assert len(ready.plan.loads) == 1
        assert ready.plan.loads[0].persistent is True
        ready.release(synchronize=False)
        assert bank.occupancy == 2
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# The fixed footprint is never evictable; only the streamed slot cache is.
# ---------------------------------------------------------------------------


def test_transient_service_is_outside_the_evictable_set(tmp_path: Path) -> None:
    runtime, spec, _expected = _open_runtime(
        tmp_path,
        slots=1,
        max_live_kv_tokens=10_000,
    )
    try:
        # A prefill miss is served from the transient tier and holds no
        # persistent entry, so a maximal KV admission needs no eviction.
        ready = runtime.ensure_route(1, [1], phase="prefill")
        assert all(not load.persistent for load in ready.plan.loads)
        ready.release(synchronize=False)
        assert runtime._banks[1].occupancy == 0

        admission = runtime.admit_kv_tokens(
            _tokens_for_bytes(spec, _slot_bytes(spec))
        )
        assert _policy(runtime)["allowance_bytes"] == 0
        admission.release()
    finally:
        runtime.close()


def test_additional_resident_bytes_shrink_the_derived_allowance(
    tmp_path: Path,
) -> None:
    record_bytes = _slot_bytes(_spec())
    runtime, spec, _expected = _open_runtime(
        tmp_path,
        expert_count=4,
        slots=3,
        max_live_kv_tokens=2000,
        additional_resident_bytes=record_bytes,
    )
    try:
        assert runtime.plan.slots_per_layer == 3
        admission = runtime.admit_kv_tokens(
            _tokens_for_bytes(spec, _slot_bytes(spec))
        )
        policy = _policy(runtime)
        assert policy["allowance_bytes"] == 2 * _slot_bytes(spec)
        assert policy["persistent_capacity"] == 2
        admission.release()
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# Global cache scope: one shared byte-accounted LRU across layers.
# ---------------------------------------------------------------------------


def test_global_scope_evicts_least_recent_across_layers(tmp_path: Path) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    config = _derived_config(
        spec,
        memory_limit_bytes=_fixed_bytes(spec) + 3 * _slot_bytes(spec),
        max_live_kv_tokens=2000,
        cache_scope="global",
    )
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        config,
        spec=spec,
        apply_memory_cap=False,
    )
    try:
        bank = runtime._global_bank
        assert bank is not None
        assert runtime.plan.persistent_slots == 3
        _decode(runtime, 1, 0)
        _decode(runtime, 2, 0)
        _decode(runtime, 1, 1)
        assert bank.occupancy == 3

        admission = runtime.admit_kv_tokens(
            _tokens_for_bytes(spec, 2 * _slot_bytes(spec))
        )
        assert bank.occupancy == 1
        assert bank.persistent_capacity == 1
        assert bank.resident_experts_by_layer == {1: (1,), 2: ()}
        assert _policy(runtime)["persistent_capacity"] == 1
        admission.release()
        assert bank.persistent_capacity == 3
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# Allowance derivation is a pure, planner-invertible function of the plan and
# stays correct across the C6 mmap-band merge.
# ---------------------------------------------------------------------------


def _stub_plan(**overrides):
    values = {
        "total_limit_bytes": 100,
        "fixed_bytes": 60,
        "persistent_budget_bytes": 40,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_allowance_is_a_pure_function_of_the_plan() -> None:
    assert derived_expert_cache_allowance_bytes(_stub_plan()) == 40
    # The routed-bytes clamp inside the plan's budget is respected.
    assert (
        derived_expert_cache_allowance_bytes(
            _stub_plan(persistent_budget_bytes=25)
        )
        == 25
    )
    # A fixed-side deficit is reported as a negative allowance, never clamped.
    assert (
        derived_expert_cache_allowance_bytes(
            _stub_plan(fixed_bytes=130, persistent_budget_bytes=0)
        )
        == -30
    )


def test_allowance_respects_a_paged_mmap_band() -> None:
    # Wired bands already sit inside plan.fixed_bytes; only a paged band
    # additionally shrinks what the slot cache may hold under the MLX cap.
    wired = _stub_plan(mmap_islands_wired=True, mmap_island_bytes=30)
    assert derived_expert_cache_allowance_bytes(wired) == 40
    paged = _stub_plan(mmap_islands_wired=False, mmap_island_bytes=30)
    assert derived_expert_cache_allowance_bytes(paged) == 10
    drowned = _stub_plan(mmap_islands_wired=False, mmap_island_bytes=50)
    assert derived_expert_cache_allowance_bytes(drowned) == -10


def test_runtime_boundary_respects_a_paged_band_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, spec, _expected = _open_runtime(
        tmp_path,
        expert_count=4,
        slots=3,
        max_live_kv_tokens=10_000,
    )
    try:
        real_plan = runtime._derived_expert_plan

        def banded_plan(live_kv_tokens: int):
            plan = real_plan(live_kv_tokens)
            return SimpleNamespace(
                total_limit_bytes=plan.total_limit_bytes,
                fixed_bytes=plan.fixed_bytes,
                persistent_budget_bytes=plan.persistent_budget_bytes,
                mmap_islands_wired=False,
                mmap_island_bytes=_slot_bytes(spec),
            )

        monkeypatch.setattr(runtime, "_derived_expert_plan", banded_plan)
        _decode(runtime, 1, 0)
        _decode(runtime, 1, 1)
        admission = runtime.admit_kv_tokens(
            _tokens_for_bytes(spec, _slot_bytes(spec))
        )
        # 3 slots - 1 slot of KV - 1 slot of paged band = 1 slot allowance.
        assert runtime._banks[1].persistent_capacity == 1
        assert runtime._banks[1].occupancy == 1
        admission.release()

        with pytest.raises(
            ExpertStreamingConfigurationError, match="oversubscribes"
        ):
            runtime.admit_kv_tokens(
                _tokens_for_bytes(spec, 3 * _slot_bytes(spec))
            )
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# Smallest bank hooks: capacity cap and non-mutating victim peek.
# ---------------------------------------------------------------------------


def test_layer_bank_capacity_and_victim_hooks() -> None:
    bank = LayerExpertSlotBank(
        expert_count=4,
        persistent_slots=3,
        transient_slots=1,
        cache_policy="lru",
    )
    assert bank.persistent_capacity == 3
    for expert in (0, 1, 2):
        bank.plan([expert], phase="decode")
    bank.plan([0], phase="decode")
    assert bank.peek_victim() == (1, 1)
    assert bank.peek_victim(excluded={1}) == (2, 2)
    assert bank.peek_victim(excluded={0, 1, 2}) is None

    assert bank.set_persistent_capacity(1) == 1
    # Occupancy above the cap never grows: a new miss replaces the victim.
    plan = bank.plan([3], phase="decode")
    assert plan.loads == (SlotLoad(expert=3, slot=1, persistent=True),)
    assert bank.occupancy == 3
    assert bank.set_persistent_capacity(99) == 3
    with pytest.raises(TypeError):
        bank.set_persistent_capacity(True)
    with pytest.raises(ValueError):
        bank.set_persistent_capacity(-1)


def test_layer_bank_empty_slots_respect_the_capacity_cap() -> None:
    bank = LayerExpertSlotBank(
        expert_count=4,
        persistent_slots=3,
        transient_slots=1,
        cache_policy="lru",
    )
    bank.set_persistent_capacity(1)
    bank.plan([0], phase="decode")
    assert bank.occupancy == 1
    plan = bank.plan([1], phase="decode")
    # The second miss must not occupy a second persistent slot: it replaces
    # the policy victim in place.
    assert plan.loads == (SlotLoad(expert=1, slot=0, persistent=True),)
    assert bank.occupancy == 1
    assert bank.resident_experts == (1,)


def test_global_bank_capacity_and_victim_hooks() -> None:
    bank = GlobalExpertSlotBank(
        layer_indices=(1, 2),
        expert_count=2,
        persistent_slots=3,
        transient_slots=1,
        prefill_slots_per_layer=1,
        cache_policy="lru",
    )
    assert bank.persistent_capacity == 3
    bank.plan(1, [0], phase="decode")
    bank.plan(2, [0], phase="decode")
    bank.plan(1, [1], phase="decode")
    assert bank.peek_victim() == (1, 0, 0)
    assert bank.peek_victim(excluded={(1, 0)}) == (2, 0, 1)
    assert bank.set_persistent_capacity(2) == 2
    assert bank.set_persistent_capacity(99) == 3
    bank.set_persistent_capacity(3)
    bank.invalidate_expert(1, 0)
    assert bank.occupancy == 2
    bank.set_persistent_capacity(2)
    # A miss above the cap replaces a victim instead of taking a free slot.
    bank.plan(2, [1], phase="decode")
    assert bank.occupancy == 2
