from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mtplx.hy3_router_last_arrival import (
    TaggedArrivalLayout,
    tagged_arrival_checksums,
    tagged_arrival_payload,
    tagged_arrival_tag,
)


_SCRIPT = Path(__file__).parents[1] / "benchmarks" / "hy3_router_last_arrival.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "hy3_router_last_arrival_benchmark",
        _SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_scratch(
    layout: TaggedArrivalLayout,
    *,
    base_event: int,
    seed: int,
) -> np.ndarray:
    scratch = np.zeros((layout.elections, layout.words_per_election), dtype=np.uint32)
    for election in range(layout.elections):
        event = base_event + election
        tag = tagged_arrival_tag(event)
        scratch[election, 0] = np.uint32((~tag) & 0xFFFFFFFF)
        scratch[election, 1 : layout.ready_words] = np.uint32(tag)
        scratch[
            election,
            layout.ready_words : layout.flag_words,
        ] = np.uint32((~tag) & 0xFFFFFFFF)
        for group in range(layout.threadgroups):
            scratch[election, layout.flag_words + group] = np.uint32(
                tagged_arrival_payload(event=event, group=group, seed=seed)
            )
        checksum_sum, checksum_xor = tagged_arrival_checksums(
            event=event,
            seed=seed,
            threadgroups=layout.threadgroups,
        )
        metadata = layout.flag_words + layout.payload_words
        scratch[election, metadata] = np.uint32(election % layout.threadgroups)
        scratch[election, metadata + 1] = np.uint32(checksum_sum)
        scratch[election, metadata + 2] = np.uint32(checksum_xor)
    return scratch.reshape(-1)


@pytest.mark.parametrize("threadgroups", (16, 48))
def test_litmus_validator_accepts_exact_no_init_elections(threadgroups: int) -> None:
    module = _load_script()
    layout = TaggedArrivalLayout(threadgroups=threadgroups, elections=4)
    scratch = _valid_scratch(layout, base_event=100, seed=51)

    observed = module.validate_litmus_scratch(
        scratch,
        layout=layout,
        base_event=100,
        seed=51,
    )

    assert observed["successful_elections"] == 4
    assert observed["failed_elections"] == 0
    assert observed["flag_failures"] == 0
    assert observed["payload_failures"] == 0
    assert observed["checksum_failures"] == 0
    assert observed["winner_failures"] == 0
    assert observed["first_failure"] is None


def test_litmus_validator_attributes_stale_payload_and_missing_claim() -> None:
    module = _load_script()
    layout = TaggedArrivalLayout(elections=3)
    scratch = _valid_scratch(layout, base_event=200, seed=52).reshape(
        layout.elections,
        layout.words_per_election,
    )
    scratch[0, 0] = np.uint32(tagged_arrival_tag(200))
    scratch[1, layout.flag_words + 7] ^= np.uint32(1)

    observed = module.validate_litmus_scratch(
        scratch.reshape(-1),
        layout=layout,
        base_event=200,
        seed=52,
    )

    assert observed["successful_elections"] == 1
    assert observed["failed_elections"] == 2
    assert observed["flag_failures"] == 1
    assert observed["payload_failures"] == 1
    assert observed["first_failure"]["event"] == 200


def test_litmus_validator_rejects_a_stale_complement_word() -> None:
    module = _load_script()
    layout = TaggedArrivalLayout(elections=2)
    scratch = _valid_scratch(layout, base_event=300, seed=53).reshape(
        layout.elections,
        layout.words_per_election,
    )
    scratch[1, layout.ready_words + 7] = np.uint32(tagged_arrival_tag(301))

    observed = module.validate_litmus_scratch(
        scratch.reshape(-1),
        layout=layout,
        base_event=300,
        seed=53,
    )

    assert observed["successful_elections"] == 1
    assert observed["failed_elections"] == 1
    assert observed["flag_failures"] == 1
    assert observed["first_failure"]["event"] == 301


def test_litmus_dispatch_omits_output_initialization(monkeypatch) -> None:
    module = _load_script()
    layout = TaggedArrivalLayout(elections=8)
    captured: dict[str, object] = {}

    class FakeKernel:
        def __call__(self, **kwargs: object):
            captured.update(kwargs)
            return (object(),)

    monkeypatch.setattr(module, "build_litmus_kernel", lambda _layout: FakeKernel())
    monkeypatch.setattr(module.mx, "array", lambda value, dtype: (value, dtype))

    module.dispatch_litmus(layout=layout, base_event=10, seed=53)

    assert "init_value" not in captured
    assert captured["grid"] == (16 * 8 * 32, 1, 1)
    assert captured["threadgroup"] == (32, 1, 1)
    assert captured["output_shapes"] == [(layout.total_words,)]


def test_litmus_cli_defaults_to_at_least_one_million_elections() -> None:
    module = _load_script()

    args = module._parser().parse_args([])

    assert args.total_elections >= 1_000_000
    assert args.total_elections % args.elections_per_dispatch == 0
    assert args.threadgroups == 16
    assert str(args.lock_path) == "/tmp/mtplx-gpu-exclusive.lock"


def test_litmus_cli_accepts_the_measured_t48_finalist() -> None:
    module = _load_script()

    args = module._parser().parse_args(["--threadgroups", "48"])

    assert args.threadgroups == 48
