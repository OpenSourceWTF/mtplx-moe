"""Auto-census island placement (issue #98): derivation, persistence, load.

The hy3 spec's measured ``island_pin_order`` is the validation reference:
it was produced by exactly the analysis ``derive_placement`` automates
(rank layers by ascending top-K routed-traffic coverage), so a synthetic
census over hy3's geometry must yield a stable, complete permutation of
the same routed layers. Concentrated routing (the GLM-5.2 shape) must pin
low-coverage layers first with a monotone coverage curve.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from mtplx.expert_manifest import save_expert_manifest
from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    ExpertStreamingRuntime,
    resolve_island_placement,
)
from mtplx.expert_streaming_models import get_model_spec
from mtplx.route_census import (
    CENSUS_SCHEMA_VERSION,
    DECAY_MERGE_THRESHOLD,
    ISLAND_PLACEMENT_FILENAME,
    PLACEMENT_SCHEMA_VERSION,
    ROUTE_CENSUS_FILENAME,
    Placement,
    RouteCensus,
    RouteCensusError,
    derive_placement,
    load_census,
    load_placement,
    save_census,
    save_placement,
)
from tests.test_expert_slots_runtime import _global_artifact, _plan


# --------------------------------------------------------------------------
# RouteCensus counting and merge


def test_census_observe_counts_and_total() -> None:
    census = RouteCensus("hy3-expert-q2")
    census.observe(1, [0, 1, 0])
    census.observe(2, [5])
    census.observe(1, [])
    assert census.counts() == {1: {0: 2, 1: 1}, 2: {5: 1}}
    assert census.total_routed_assignments == 4
    with pytest.raises(RouteCensusError):
        census.observe(1, [True])
    with pytest.raises(RouteCensusError):
        census.observe(-1, [0])


def test_census_merge_adds_below_decay_threshold() -> None:
    history = RouteCensus("m", counts={1: {0: 10}})
    session = RouteCensus("m", counts={1: {0: 1, 1: 2}, 2: {0: 3}})
    history.merge(session)
    assert history.counts() == {1: {0: 11, 1: 2}, 2: {0: 3}}
    assert history.total_routed_assignments == 16
    assert history.merge_count == 1


def test_census_merge_decays_history_beyond_threshold() -> None:
    history = RouteCensus(
        "m", counts={1: {0: DECAY_MERGE_THRESHOLD - 1, 1: 1}}
    )
    session = RouteCensus("m", counts={1: {0: 10}, 2: {0: 2}})
    history.merge(session)
    # History held exactly the threshold, so it halves (floor; the count-1
    # entry drops to zero and disappears) before the session lands whole.
    halved = (DECAY_MERGE_THRESHOLD - 1) // 2
    assert history.counts() == {1: {0: halved + 10}, 2: {0: 2}}
    assert history.total_routed_assignments == halved + 12
    assert history.merge_count == 1


def test_census_merge_rejects_model_mismatch() -> None:
    with pytest.raises(RouteCensusError, match="different models"):
        RouteCensus("a").merge(RouteCensus("b"))


# --------------------------------------------------------------------------
# Serialization round trips


def test_census_round_trip(tmp_path: Path) -> None:
    census = RouteCensus("hy3-expert-q2")
    census.observe(1, [0, 1, 0])
    census.observe(7, [190, 3])
    path = tmp_path / ROUTE_CENSUS_FILENAME
    save_census(census, path, updated_at="2026-07-17T00:00:00+00:00")
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == CENSUS_SCHEMA_VERSION
    assert payload["model_key"] == "hy3-expert-q2"
    assert payload["total_routed_assignments"] == 5
    assert payload["updated_at"] == "2026-07-17T00:00:00+00:00"
    loaded = load_census(path)
    assert loaded.model_key == census.model_key
    assert loaded.counts() == census.counts()
    assert loaded.total_routed_assignments == 5
    # A tampered total no longer matches the counts and must refuse.
    payload["total_routed_assignments"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(RouteCensusError, match="does not match"):
        load_census(path)


def test_placement_round_trip(tmp_path: Path) -> None:
    census = RouteCensus("m")
    census.observe(1, [0, 0, 0, 1])
    census.observe(2, [0, 1, 2, 3])
    placement = derive_placement(census, expert_count=4, slots_per_layer_hint=2)
    assert placement.advisory  # 8 assignments < ADVISORY threshold
    path = tmp_path / ISLAND_PLACEMENT_FILENAME
    save_placement(placement, path, updated_at="2026-07-17T00:00:00+00:00")
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == PLACEMENT_SCHEMA_VERSION
    assert payload["census_total_routed_assignments"] == 8
    loaded = load_placement(path)
    assert loaded == placement
    assert isinstance(loaded, Placement)


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    census = RouteCensus("m")
    census.observe(1, [0])
    path = tmp_path / ROUTE_CENSUS_FILENAME
    save_census(census, path, updated_at="t0")
    save_census(census, path, updated_at="t1")
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]
    assert json.loads(path.read_text())["updated_at"] == "t1"


# --------------------------------------------------------------------------
# Placement derivation


def _hy3_uniform_census() -> RouteCensus:
    """Near-uniform routing over hy3's real geometry (79 layers x 192)."""

    spec = get_model_spec("hy3-expert-q2")
    census = RouteCensus(spec.key)
    counts = {
        layer: {
            expert: 100 + (layer * 31 + expert * 7) % 5
            for expert in range(spec.expert_count)
        }
        for layer in spec.routed_layer_indices
    }
    return RouteCensus(spec.key, counts=counts)


def test_derive_placement_hy3_uniform_is_stable_and_complete() -> None:
    spec = get_model_spec("hy3-expert-q2")
    census = _hy3_uniform_census()
    first = derive_placement(
        census, expert_count=spec.expert_count, slots_per_layer_hint=147
    )
    second = derive_placement(
        census, expert_count=spec.expert_count, slots_per_layer_hint=147
    )
    assert first == second
    # Complete permutation of exactly the routed layers — the same set the
    # measured hy3 island_pin_order permutes.
    assert sorted(first.layer_pin_order) == sorted(spec.routed_layer_indices)
    assert set(first.layer_pin_order) == set(spec.island_pin_order)
    assert not first.advisory
    coverage = [first.coverage_by_layer[layer] for layer in first.layer_pin_order]
    assert coverage == sorted(coverage)
    assert all(0.0 < value <= 1.0 for value in coverage)
    for layer in spec.routed_layer_indices:
        ranking = first.expert_ranking_by_layer[layer]
        assert sorted(ranking) == list(range(spec.expert_count))


def test_derive_placement_concentrated_pins_low_coverage_first() -> None:
    expert_count = 64
    uniform_layers = (0, 1, 2, 3)
    concentrated_layers = (4, 5, 6, 7)
    counts: dict[int, dict[int, int]] = {}
    for layer in uniform_layers:
        counts[layer] = {expert: 10 for expert in range(expert_count)}
    for layer in concentrated_layers:
        counts[layer] = {
            expert: (1000 // (expert + 1)) if expert < 16 else 1
            for expert in range(expert_count)
        }
    census = RouteCensus("glm52-expert-q2", counts=counts)
    placement = derive_placement(
        census, expert_count=expert_count, slots_per_layer_hint=8
    )
    # Uniform layers have the LEAST top-8 coverage and must pin first.
    assert set(placement.layer_pin_order[:4]) == set(uniform_layers)
    assert set(placement.layer_pin_order[4:]) == set(concentrated_layers)
    curve = [
        placement.coverage_by_layer[layer]
        for layer in placement.layer_pin_order
    ]
    assert curve == sorted(curve)
    for layer in uniform_layers:
        assert placement.coverage_by_layer[layer] == pytest.approx(8 / 64)
    for layer in concentrated_layers:
        assert placement.coverage_by_layer[layer] > 0.5
        # Zipf head is the most popular expert.
        assert placement.expert_ranking_by_layer[layer][0] == 0


def test_derive_placement_rejects_foreign_expert_ids() -> None:
    census = RouteCensus("m")
    census.observe(1, [0, 9])
    with pytest.raises(RouteCensusError, match="different model geometry"):
        derive_placement(census, expert_count=4, slots_per_layer_hint=2)


# --------------------------------------------------------------------------
# Runtime collection hook


def _census_runtime(root, spec, manifest_path, **config_overrides):
    plan = _plan(spec)
    config = ExpertStreamingConfig(
        model_key=spec.key,
        memory_limit_bytes=plan.total_limit_bytes,
        max_live_kv_tokens=0,
        runtime_reserve_bytes=0,
        verify_artifact_headers=False,
        **config_overrides,
    )
    return ExpertStreamingRuntime.open(
        root,
        manifest_path,
        config,
        spec=spec,
        apply_memory_cap=False,
    )


def test_runtime_records_decode_census_and_flushes_on_close(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)

    runtime = _census_runtime(root, spec, manifest_path)
    assert runtime.config.route_census is True
    runtime.observe_route(1, "decode", [0, 1], token_count=2)
    runtime.observe_route(2, "decode", [1, 0], token_count=2)
    runtime.observe_route(1, "prefill", [0, 0], token_count=2)
    # No artifacts before close: the flush is not on the decode path.
    assert not (root / ROUTE_CENSUS_FILENAME).exists()
    runtime.close()

    census = load_census(root / ROUTE_CENSUS_FILENAME)
    assert census.model_key == spec.key
    # Prefill routes are excluded by design.
    assert census.counts() == {1: {0: 1, 1: 1}, 2: {0: 1, 1: 1}}
    assert census.total_routed_assignments == 4
    placement = load_placement(root / ISLAND_PLACEMENT_FILENAME)
    assert placement.model_key == spec.key
    assert sorted(placement.layer_pin_order) == [1, 2]
    assert placement.advisory

    # Second session merges into the same files.
    runtime = _census_runtime(root, spec, manifest_path)
    runtime.observe_route(1, "decode", [0, 0], token_count=2)
    runtime.close()
    merged = load_census(root / ROUTE_CENSUS_FILENAME)
    assert merged.counts() == {1: {0: 3, 1: 1}, 2: {0: 1, 1: 1}}
    assert merged.total_routed_assignments == 6
    assert merged.merge_count == 1


def test_runtime_census_opt_out_writes_nothing(tmp_path: Path) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    runtime = _census_runtime(root, spec, manifest_path, route_census=False)
    runtime.observe_route(1, "decode", [0, 1], token_count=2)
    runtime.close()
    assert not (root / ROUTE_CENSUS_FILENAME).exists()
    assert not (root / ISLAND_PLACEMENT_FILENAME).exists()


def test_close_survives_read_only_model_root(tmp_path: Path) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    runtime = _census_runtime(root, spec, manifest_path)
    runtime.observe_route(1, "decode", [0, 1], token_count=2)
    mode = stat.S_IMODE(os.stat(root).st_mode)
    os.chmod(root, 0o555)
    try:
        runtime.close()  # must not raise: census flush is best-effort
    finally:
        os.chmod(root, mode)
    assert not (root / ROUTE_CENSUS_FILENAME).exists()
    assert not (root / ISLAND_PLACEMENT_FILENAME).exists()


# --------------------------------------------------------------------------
# Load-path precedence: explicit island_layers > spec.island_pin_order >
# island-placement.json > error


def _write_placement(
    root: Path, spec, pin_order: tuple[int, ...], *, advisory: bool = True
) -> None:
    placement = Placement(
        model_key=spec.key,
        expert_count=spec.expert_count,
        slots_per_layer_hint=1,
        census_total_routed_assignments=100,
        advisory=advisory,
        layer_pin_order=pin_order,
        coverage_by_layer={layer: 0.5 for layer in pin_order},
        expert_ranking_by_layer={
            layer: tuple(range(spec.expert_count)) for layer in pin_order
        },
    )
    save_placement(
        placement, root / ISLAND_PLACEMENT_FILENAME, updated_at="t0"
    )


def _island_count_config(spec, **overrides) -> ExpertStreamingConfig:
    kwargs = {
        "model_key": spec.key,
        "memory_limit_bytes": 1 << 30,
        "max_live_kv_tokens": 16,
        "runtime_reserve_bytes": 0,
        "transient_slots": 2,
        "slot_layout": "component-banks",
        "island_layer_count": 1,
    }
    kwargs.update(overrides)
    return ExpertStreamingConfig(**kwargs)


def test_spec_without_pin_order_defers_count_resolution() -> None:
    # Registered spec without a measured order: construction succeeds with
    # the count pending instead of raising immediately (issue #98).
    # glm52-q4 remains unmeasured; glm52-expert-q2 gained a measured
    # order on 2026-07-17 and now resolves from the spec.
    config = _island_count_config(get_model_spec("glm52-q4"))
    assert config.island_layer_count == 1
    assert config.island_layers == ()


def test_resolve_prefers_explicit_layers_then_spec_order(
    tmp_path: Path,
) -> None:
    root, spec, _manifest, _expected = _global_artifact(tmp_path)
    _write_placement(root, spec, pin_order=(1, 2))
    # Explicit island_layers: nothing pending, config passes through.
    explicit = _island_count_config(
        spec, island_layer_count=None, island_layers=(2,)
    )
    assert resolve_island_placement(explicit, root, spec=spec) is explicit
    # A spec pin order beats the ADVISORY placement file (2026-07-21
    # census-first precedence: only a SOUND census outranks the spec).
    ordered_spec = replace(spec, island_pin_order=(2, 1))
    config = _island_count_config(spec)
    resolved = resolve_island_placement(config, root, spec=ordered_spec)
    assert resolved.island_layers == (2,)


def test_sound_census_outranks_spec_pin_order(tmp_path: Path) -> None:
    # 2026-07-21 (David): the bank's own measured census is per-bank truth;
    # the spec order is a static bootstrap default (often borrowed from a
    # sibling bank) and must not shadow a sound census.
    root, spec, _manifest, _expected = _global_artifact(tmp_path)
    _write_placement(root, spec, pin_order=(1, 2), advisory=False)
    ordered_spec = replace(spec, island_pin_order=(2, 1))
    config = _island_count_config(spec)
    resolved = resolve_island_placement(config, root, spec=ordered_spec)
    assert resolved.island_layers == (1,)
    assert resolved.island_layer_count is None


def test_open_resolves_island_count_from_placement_file(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    # Layer 2 has the worst top-K coverage in this synthetic placement, so
    # count=1 must pin layer 2.
    _write_placement(root, spec, pin_order=(2, 1))
    config = _island_count_config(spec, verify_artifact_headers=False)
    assert config.island_layers == ()  # deferred at construction
    runtime = ExpertStreamingRuntime.open(
        root,
        manifest_path,
        config,
        spec=spec,
        apply_memory_cap=False,
    )
    try:
        assert runtime.config.island_layers == (2,)
        assert runtime.config.island_layer_count is None
        assert runtime.island_layer_set == frozenset({2})
        assert runtime.plan.island_layer_count == 1
    finally:
        runtime.close()


def test_open_without_placement_raises_precedence_error(
    tmp_path: Path,
) -> None:
    root, spec, manifest, _expected = _global_artifact(tmp_path)
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    config = _island_count_config(spec, verify_artifact_headers=False)
    with pytest.raises(
        ExpertStreamingConfigurationError,
        # 2026-07-21 census-first precedence (David's call): the bank's own
        # census outranks the spec's static order; spec order is the
        # bootstrap default, advisory census demoted below it.
        match="island-placement.json",
    ):
        ExpertStreamingRuntime.open(
            root,
            manifest_path,
            config,
            spec=spec,
            apply_memory_cap=False,
        )


def test_resolve_rejects_bad_placements(tmp_path: Path) -> None:
    root, spec, _manifest, _expected = _global_artifact(tmp_path)
    config = _island_count_config(spec)
    placement_path = root / ISLAND_PLACEMENT_FILENAME

    # Wrong model.
    _write_placement(root, replace(spec, key="tiny-q4"), pin_order=(1, 2))
    with pytest.raises(
        ExpertStreamingConfigurationError, match="was derived for"
    ):
        resolve_island_placement(config, root, spec=spec)

    # Count larger than the ranked order.
    _write_placement(root, spec, pin_order=(1,))
    over = _island_count_config(spec, island_layer_count=2)
    with pytest.raises(
        ExpertStreamingConfigurationError, match="exceeds the"
    ):
        resolve_island_placement(over, root, spec=spec)

    # Corrupt JSON.
    placement_path.write_text("{not json")
    with pytest.raises(
        ExpertStreamingConfigurationError, match="cannot be resolved"
    ):
        resolve_island_placement(config, root, spec=spec)


def test_unresolved_count_blocks_memory_plan() -> None:
    spec = get_model_spec("glm52-q4")
    config = _island_count_config(spec)
    with pytest.raises(
        ExpertStreamingConfigurationError, match="unresolved"
    ):
        config.memory_plan(spec)
