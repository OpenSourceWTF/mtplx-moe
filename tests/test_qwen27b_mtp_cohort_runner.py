from __future__ import annotations

from dataclasses import dataclass, replace
import os
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import mtplx.qwen27b_mtp_cohort as cohort
from mtplx.mtp_k2_stepper import (
    MTPK2AcceptanceContext,
    MTPK2PrefillTicket,
    MTPK2RequestState,
    MTPK2VerifyResult,
    MTPK2VerifyTicket,
)
from mtplx.qwen27b_mtp_cohort import (
    LayerCacheRoute,
    MTPK2CohortRunner,
    Qwen27BK2DualLane,
    TargetForwardResult,
)


@dataclass
class _RequestCache:
    values: list[int]

    @property
    def state(self) -> list[int]:
        return self.values


@dataclass
class _CohortCache:
    rows: list[list[int]]

    @property
    def state(self) -> list[list[int]]:
        return self.rows


class _CancelEvent:
    def __init__(self, is_set: bool = False) -> None:
        self.value = is_set

    def is_set(self) -> bool:
        return self.value

    def set(self) -> None:
        self.value = True


@dataclass
class _Harness:
    lane: Qwen27BK2DualLane
    dependencies: SimpleNamespace
    target_calls: list[tuple[int, np.ndarray, list[Any]]]
    solo_ticket_calls: list[MTPK2VerifyTicket]
    materialization_calls: list[tuple[Any, ...]]
    events: list[str]
    merge_calls: list[tuple[_RequestCache, ...]]
    extract_calls: list[tuple[Any, int]]
    commit_calls: list[tuple[int, int, int, list[Any], Any]]
    commit_memos: list[dict[tuple[int, int], Any]]


def _harness(
    *,
    during_width2: Any | None = None,
    width2_error: BaseException | None = None,
) -> _Harness:
    target_calls: list[tuple[int, np.ndarray, list[Any]]] = []
    solo_ticket_calls: list[MTPK2VerifyTicket] = []
    materialization_calls: list[tuple[Any, ...]] = []
    events: list[str] = []
    merge_calls: list[tuple[_RequestCache, ...]] = []
    extract_calls: list[tuple[Any, int]] = []
    commit_calls: list[tuple[int, int, int, list[Any], Any]] = []
    commit_memos: list[dict[tuple[int, int], Any]] = []

    def merge(entries: tuple[_RequestCache, ...]) -> _CohortCache:
        merge_calls.append(entries)
        return _CohortCache([list(entry.values) for entry in entries])

    def extract(cache: _CohortCache, row: int) -> _RequestCache:
        extract_calls.append((cache, row))
        return _RequestCache(list(cache.rows[row]))

    route = LayerCacheRoute(
        layer_index=0,
        request_type=_RequestCache,
        cohort_type=_CohortCache,
        normalize_request=lambda entry: entry,
        merge=merge,
        extract=extract,
        own_request=lambda entry: _RequestCache(list(entry.values)),
    )

    def target(width: int, *, input_ids: np.ndarray, cache: list[Any]):
        events.append(f"target-{width}")
        target_calls.append((width, input_ids, cache))
        if width == 2 and during_width2 is not None:
            during_width2()
        if width == 2 and width2_error is not None:
            raise width2_error
        for row, entry in enumerate(cache):
            if isinstance(entry, _CohortCache):
                for cache_row, token_row in zip(
                    entry.rows, input_ids.tolist(), strict=True
                ):
                    cache_row.extend(int(token) for token in token_row)
            else:
                entry.values.extend(int(token) for token in input_ids[row])
        rows = int(input_ids.shape[0])
        return TargetForwardResult(
            logits=np.asarray(
                [
                    [[1000 * width + 100 * row + step] for step in range(3)]
                    for row in range(rows)
                ]
            ),
            hidden=np.asarray(
                [
                    [[2000 * width + 100 * row + step] for step in range(3)]
                    for row in range(rows)
                ]
            ),
            captures={"rows": np.asarray([[10 + row] for row in range(rows)])},
            cache=cache,
        )

    def commit_route(width: int, row: int):
        def commit(
            cache: list[Any],
            captures: Any,
            *,
            steps: int,
            replay_memo: dict[tuple[int, int], Any] | None = None,
        ) -> list[Any]:
            if width == 2:
                assert replay_memo is not None
                commit_memos.append(replay_memo)
            commit_calls.append((width, row, int(steps), cache, captures))
            request_cache = cache
            request_cache[0].values.append(9000 + 100 * row + int(steps))
            return request_cache

        return commit

    lane = Qwen27BK2DualLane(
        backend_id="qwen3_next",
        depth=2,
        bits=4,
        group_size=64,
        activation_dtype="bf16",
        hidden_variant="post_norm",
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        max_width=2,
        width1_target=lambda **kwargs: target(1, **kwargs),
        width2_target=lambda **kwargs: target(2, **kwargs),
        cache_routes=(route,),
        qlinear_routes=MappingProxyType({}),
        construction_receipt=MappingProxyType({}),
        capture_commit_routes=MappingProxyType(
            {
                (1, 0): commit_route(1, 0),
                (2, 0): commit_route(2, 0),
                (2, 1): commit_route(2, 1),
            }
        ),
    )

    def materialize(*values: Any) -> None:
        events.append("materialize")
        materialization_calls.append(values)

    def execute_width1(ticket: MTPK2VerifyTicket) -> MTPK2VerifyResult:
        events.append("solo-ticket")
        solo_ticket_calls.append(ticket)
        forward = target(
            1,
            input_ids=ticket.input_ids,
            cache=ticket.request_cache,
        )
        return MTPK2VerifyResult(
            logits=forward.logits,
            hidden=forward.hidden,
            captures=forward.captures,
            request_cache=forward.cache,
            commit_prefix=lambda steps: commit_route(1, 0)(
                forward.cache,
                forward.captures,
                steps=steps,
            ),
            forward_elapsed_s=0.125,
        )

    clock_values = iter((10.0, 10.25) * 20)
    dependencies = SimpleNamespace(
        stack_rows=lambda rows: np.concatenate(rows, axis=0),
        materialize=materialize,
        extract_captures=lambda captures, row: {
            "rows": np.array(captures["rows"][row : row + 1], copy=True)
        },
        clock=lambda: next(clock_values),
        execute_width1=execute_width1,
    )
    return _Harness(
        lane=lane,
        dependencies=dependencies,
        target_calls=target_calls,
        solo_ticket_calls=solo_ticket_calls,
        materialization_calls=materialization_calls,
        events=events,
        merge_calls=merge_calls,
        extract_calls=extract_calls,
        commit_calls=commit_calls,
        commit_memos=commit_memos,
    )


def _state(
    request_id: str,
    values: list[int],
    *,
    tokens: tuple[int, int, int] = (1, 2, 3),
    cancel_event: _CancelEvent | None = None,
    purpose: str = "verify",
) -> MTPK2RequestState:
    ticket = MTPK2VerifyTicket(
        request_id=request_id,
        input_ids=np.asarray([tokens]),
        request_cache=[_RequestCache(list(values))],
        draft_distributions=(object(), object()),
        acceptance_context=MTPK2AcceptanceContext(
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
            hidden_variant="post_norm",
            purpose=purpose,
        ),
    )
    return MTPK2RequestState(
        request_id=request_id,
        _machine=None,
        config=SimpleNamespace(),
        lane=None,
        width1_commit_route=lambda *_args, **_kwargs: [],
        target_cache=ticket.request_cache,
        mtp_cache=None,
        tokens=[],
        rng=np.random.default_rng(7),
        sampler=None,
        draft_sampler=None,
        constraint=None,
        stop_token_ids=set(),
        token_callback=None,
        prefill_callback=None,
        cancel_event=cancel_event,
        session_id=None,
        stats=SimpleNamespace(),
        pending_ticket=ticket,
        status="ready",
    )


def _runner(harness: _Harness) -> MTPK2CohortRunner:
    return MTPK2CohortRunner(
        harness.lane,
        dependencies=harness.dependencies,
    )


def test_width1_executes_prebound_solo_ticket_without_cohort_copy() -> None:
    harness = _harness()
    request = _state("solo", [41])
    authoritative = request.require_ticket().request_cache

    (result,) = _runner(harness).step((request,))

    assert len(harness.target_calls) == 1
    width, inputs, target_cache = harness.target_calls[0]
    assert width == 1
    assert inputs.shape == (1, 3)
    assert target_cache is authoritative
    assert target_cache[0] is authoritative[0]
    assert result.request_cache[0].values == [41, 1, 2, 3]
    assert harness.solo_ticket_calls == [request.require_ticket()]
    assert harness.merge_calls == []
    assert harness.extract_calls == []
    assert harness.materialization_calls == []
    assert harness.events == ["solo-ticket", "target-1"]
    assert result.forward_elapsed_s == 0.125
    assert result.commit_prefix(2)[0].values == [41, 1, 2, 3, 9002]
    assert [(width, row, steps) for width, row, steps, *_ in harness.commit_calls] == [
        (1, 0, 2)
    ]


def test_width2_stacks_t3_rows_and_preserves_request_order_in_one_call() -> None:
    harness = _harness()
    first = _state("first", [11], tokens=(1, 2, 3))
    second = _state("second", [22], tokens=(7, 8, 9))
    first_source = first.require_ticket().request_cache
    second_source = second.require_ticket().request_cache

    results = _runner(harness).step((first, second))

    assert len(harness.target_calls) == 1
    width, inputs, _target_cache = harness.target_calls[0]
    assert width == 2
    assert inputs.shape == (2, 3)
    assert inputs.tolist() == [[1, 2, 3], [7, 8, 9]]
    assert [result.logits[:, 0, :].item() for result in results] == [2000, 2100]
    assert [result.hidden[:, 0, :].item() for result in results] == [4000, 4100]
    assert results[0].request_cache[0].values == [11, 1, 2, 3]
    assert results[1].request_cache[0].values == [22, 7, 8, 9]
    assert first_source[0].values == [11]
    assert second_source[0].values == [22]
    assert len(harness.materialization_calls) == 1
    assert len(harness.materialization_calls[0]) == 7
    assert harness.events == ["target-2", "materialize"]
    assert {result.forward_elapsed_s for result in results} == {0.25}


def test_width2_rejection_and_full_accept_commit_independently() -> None:
    harness = _harness()
    results = _runner(harness).step(
        (_state("reject", [10]), _state("accept", [20]))
    )

    rejected_cache = results[0].commit_prefix(1)
    accepted_cache = results[1].commit_prefix(3)

    assert rejected_cache[0].values == [10, 1, 2, 3, 9001]
    assert accepted_cache[0].values == [20, 1, 2, 3, 9103]
    assert [(width, row, steps) for width, row, steps, *_ in harness.commit_calls] == [
        (2, 0, 1),
        (2, 1, 3),
    ]
    assert harness.commit_calls[0][3] is results[0].request_cache
    assert harness.commit_calls[1][3] is results[1].request_cache
    assert harness.commit_calls[0][3] is not harness.commit_calls[1][3]
    assert harness.commit_calls[0][4] is harness.commit_calls[1][4]
    assert results[0].captures is not results[1].captures
    assert len(harness.extract_calls) == 2
    assert harness.commit_memos[0] is harness.commit_memos[1]


def test_cohort_width_can_transition_one_two_one_without_padding() -> None:
    harness = _harness()
    runner = _runner(harness)
    first = _state("first", [1])
    second = _state("second", [2])

    runner.step((first,))
    runner.step((first, second))
    runner.step((second,))

    assert [width for width, *_ in harness.target_calls] == [1, 2, 1]
    assert [inputs.shape for _, inputs, _ in harness.target_calls] == [
        (1, 3),
        (2, 3),
        (1, 3),
    ]


def test_cancellation_before_selection_terminalizes_and_filters_request() -> None:
    harness = _harness()
    cancelled = _state("cancelled", [1], cancel_event=_CancelEvent(True))
    survivor = _state("survivor", [2])

    (result,) = _runner(harness).step((cancelled, survivor))

    assert cancelled.status == "cancelled"
    assert cancelled.pending_ticket is None
    assert [width for width, *_ in harness.target_calls] == [1]
    assert result.request_cache[0].values == [2, 1, 2, 3]


def test_cancellation_during_width2_discards_only_cancelled_extracted_row() -> None:
    cancelled_event = _CancelEvent()
    harness = _harness(during_width2=cancelled_event.set)
    cancelled = _state("cancelled", [1], cancel_event=cancelled_event)
    survivor = _state("survivor", [2])

    (result,) = _runner(harness).step((cancelled, survivor))

    assert cancelled.status == "cancelled"
    assert survivor.status == "ready"
    assert result.request_cache[0].values == [2, 1, 2, 3]
    assert len(harness.target_calls) == 1
    assert harness.target_calls[0][0] == 2
    assert [row for _, row in harness.extract_calls[-2:]] == [0, 1]


def test_width2_target_exception_propagates_once_without_width1_retry() -> None:
    failure = RuntimeError("paired target failed")
    harness = _harness(width2_error=failure)
    first = _state("first", [1])
    second = _state("second", [2])

    with pytest.raises(RuntimeError, match="paired target failed") as exc_info:
        _runner(harness).step((first, second))

    assert exc_info.value is failure
    assert [width for width, *_ in harness.target_calls] == [2]
    assert first.status == second.status == "ready"


@pytest.mark.parametrize("width", (0, 3))
def test_invalid_live_width_raises_before_target(width: int) -> None:
    harness = _harness()
    requests = tuple(_state(f"request-{index}", [index]) for index in range(width))

    with pytest.raises(
        ValueError,
        match=rf"cohort step requires one or two live requests, got {width}",
    ):
        _runner(harness).step(requests)

    assert harness.target_calls == []


def test_prefill_ticket_is_rejected_before_target_execution() -> None:
    harness = _harness()
    request = _state("prefill", [1])
    request.pending_ticket = MTPK2PrefillTicket(
        request_id="prefill",
        input_ids=np.asarray([[1, 2, 3]]),
        request_cache=[_RequestCache([1])],
        prompt_start=0,
        prompt_stop=3,
    )

    with pytest.raises(TypeError, match="verify ticket"):
        _runner(harness).step((request,))

    assert harness.target_calls == []


def test_final_commit_verify_ticket_uses_the_normal_cohort_route() -> None:
    harness = _harness()
    first = _state("first", [1], purpose="final_commit")
    second = _state("second", [2])

    results = _runner(harness).step((first, second))

    assert len(results) == 2
    assert [width for width, *_ in harness.target_calls] == [2]
    assert results[0].commit_prefix(1)[0].values[-1] == 9001


def test_step_uses_only_construction_bound_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness()
    runner = _runner(harness)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("step touched a dynamic environment or eligibility gate")

    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(os.environ, "get", forbidden)
    monkeypatch.setattr(cohort, "_construction_dependencies", forbidden)
    monkeypatch.setattr(cohort, "_cohort_dependencies", forbidden)
    monkeypatch.setattr(cohort, "normalize_target_cache", forbidden)
    monkeypatch.setattr(Qwen27BK2DualLane, "target_for_width", forbidden)
    monkeypatch.setattr(Qwen27BK2DualLane, "capture_commit_for", forbidden)

    results = runner.step((_state("first", [1]), _state("second", [2])))

    assert len(results) == 2
    assert [width for width, *_ in harness.target_calls] == [2]


def test_runner_construction_rejects_an_incomplete_commit_route_table() -> None:
    harness = _harness()
    incomplete = replace(
        harness.lane,
        capture_commit_routes=MappingProxyType(
            {
                (1, 0): harness.lane.capture_commit_for(1, 0),
                (2, 0): harness.lane.capture_commit_for(2, 0),
            }
        ),
    )

    with pytest.raises(ValueError, match=r"got \(2, 1\)"):
        MTPK2CohortRunner(incomplete, dependencies=harness.dependencies)

    assert harness.target_calls == []


def test_aggregate_materializer_deduplicates_roots_in_one_eval_call() -> None:
    class Array:
        pass

    calls: list[tuple[Array, ...]] = []
    mx = SimpleNamespace(
        array=Array,
        eval=lambda *roots: calls.append(roots),
    )
    first = Array()
    second = Array()

    cohort._materialize_cohort_trees(
        mx,
        {
            "forward": [first, second],
            "extracted": (first, {"capture": second}),
            "metadata": "unchanged",
        },
    )

    assert calls == [(first, second)]
