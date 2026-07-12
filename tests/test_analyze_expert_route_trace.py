from __future__ import annotations

from scripts import analyze_expert_route_trace as route


def test_global_sequence_keys_records_by_layer_in_runtime_order() -> None:
    routes = {
        1: [(0, 1), (2, 3)],
        2: [(4, 5), (6, 7)],
    }

    assert route._global_sequence([1, 2], routes) == [
        (1, 0),
        (1, 1),
        (2, 4),
        (2, 5),
        (1, 2),
        (1, 3),
        (2, 6),
        (2, 7),
    ]


def test_global_belady_is_no_worse_than_global_lru() -> None:
    sequence = [
        (1, 0),
        (2, 0),
        (1, 1),
        (1, 0),
        (2, 0),
        (1, 1),
    ]

    lru_hits, lru_misses, _state = route._run_lru_sequence(sequence, [], 2)
    belady_hits, belady_misses = route._run_belady_sequence(sequence, [], 2)

    assert belady_hits >= lru_hits
    assert belady_misses <= lru_misses


def test_global_lru_can_lend_a_cold_layers_slot_to_a_hot_layer() -> None:
    global_sequence = [(2, 0), (1, 0), (1, 1), (1, 0), (1, 1)]

    global_hits, _global_misses, _state = route._run_lru_sequence(
        global_sequence, [], 2
    )
    layer_one_hits, _misses, _state = route._run_lru_sequence([0, 1, 0, 1], [], 1)
    layer_two_hits, _misses, _state = route._run_lru_sequence([0], [], 1)

    assert global_hits == 2
    assert layer_one_hits + layer_two_hits == 0


def test_trained_quota_moves_capacity_without_changing_total_budget() -> None:
    hit_curves = {
        1: [0, 5, 9],
        2: [0, 1, 1],
    }

    capacities, training = route._rebalance_trained_quotas(
        hit_curves,
        {1: 1, 2: 1},
        hysteresis_hits=0,
    )

    assert capacities == {1: 2, 2: 0}
    assert sum(capacities.values()) == 2
    assert training == {
        "reallocation_moves": 1,
        "training_hits_before": 6,
        "training_hits_after": 9,
        "training_hit_gain": 3,
        "hysteresis_hits": 0,
    }


def test_cluster_replay_accounts_for_useful_speculative_record() -> None:
    left = ((1, 0), (1, 1))
    right = ((1, 2), (1, 3))
    clusters = {key: cluster for cluster in (left, right) for key in cluster}

    metric = route._run_cluster_lru_sequence(
        [(1, 0), (1, 1), (1, 0)],
        [],
        2,
        clusters,
    )

    assert metric["hits"] == 2
    assert metric["misses"] == 1
    assert metric["physical_records_read"] == 2
    assert metric["demanded_records_read"] == 1
    assert metric["speculative_records_read"] == 1
    assert metric["speculative_records_used_before_eviction"] == 1
    assert metric["useful_prefetch_ratio"] == 1.0


def test_cluster_size_one_is_record_granularity_global_lru() -> None:
    sequence = [(1, 0), (2, 0), (1, 1), (1, 0), (2, 1)]
    clusters = {key: (key,) for key in set(sequence)}

    lru_hits, lru_misses, lru_state = route._run_lru_sequence(sequence, [], 3)
    clustered = route._run_cluster_lru_sequence(sequence, [], 3, clusters)

    assert (clustered["hits"], clustered["misses"]) == (lru_hits, lru_misses)
    assert clustered["physical_records_read"] == lru_misses
    assert clustered["speculative_records_read"] == 0
    assert clustered["final_lru_order"] == lru_state


def test_coactivation_clustering_groups_high_lift_pairs() -> None:
    training = {
        1: [(0, 1)] * 8 + [(2, 3)] * 6 + [(0, 2), (1, 3)],
    }

    clusters = route._build_coactivation_clusters(
        [1],
        training,
        expert_count=4,
        cluster_size=2,
    )

    assert set(clusters[(1, 0)]) == {(1, 0), (1, 1)}
    assert set(clusters[(1, 2)]) == {(1, 2), (1, 3)}
