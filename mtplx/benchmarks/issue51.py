"""Strict pure-data contracts for the Issue #51 benchmark campaign."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


ISSUE51_PRIORITY = (
    "compiled_whole_window",
    "mtp_hint_only_prediction",
    "q2_nax_grouping",
)
A1_CANDIDATES = (
    "batched-stock",
    "capture-eager",
    "capture-compiled-parity",
    "capture-compiled",
)
A1_PROCESS_CONFIG = {
    "batched-stock": ("off", "batched"),
    "capture-eager": ("off", "capture_commit"),
    "capture-compiled-parity": ("parity", "capture_commit"),
    "capture-compiled": ("on", "capture_commit"),
}
A1_ALLOWED_DEPTHS = frozenset({1, 2})
HY3_ROUTER_TOP_K = 8
ISSUE51_MAX_TARGET_ROWS = 4
ISSUE51_READER_CAPACITY = HY3_ROUTER_TOP_K * ISSUE51_MAX_TARGET_ROWS
REQUIRED_A1_GATES = frozenset(
    {
        "prompt_length_exact",
        "new_prefill_tokens_exact",
        "output_tokens_exact",
        "generated_count_consistent",
        "length_finish",
        "requested_depth_exact",
        "effective_depth_exact",
        "committed_history",
        "guards_disabled",
        "decode_expert_cache_metrics",
        "speculative_event_contract",
        "final_state_contract",
        "compiled_verify_evidence",
    }
)


@dataclass(frozen=True, order=True)
class CampaignCell:
    context_tokens: int
    depth: int


@dataclass(frozen=True)
class ScheduledRun:
    index: int
    block: int
    arm: str
    pair_slot: int


def build_abba_schedule(
    *, control: str, candidate: str, retained_pairs: int = 8
) -> tuple[ScheduledRun, ...]:
    """Return balanced ABBA rows containing two temporal pairs per block."""

    if retained_pairs <= 0 or retained_pairs % 2:
        raise ValueError("retained_pairs must be a positive even integer")
    if not control or not candidate or control == candidate:
        raise ValueError("control and candidate must be distinct nonempty arms")
    rows = []
    for block in range(retained_pairs // 2):
        for pair_slot, arm in enumerate((control, candidate, candidate, control)):
            rows.append(
                ScheduledRun(
                    index=len(rows),
                    block=block,
                    arm=arm,
                    pair_slot=pair_slot,
                )
            )
    return tuple(rows)


def pair_abba_rows(
    schedule: Sequence[ScheduledRun],
) -> tuple[tuple[ScheduledRun, ScheduledRun], ...]:
    """Pair each A with its adjacent B while retaining A as the left row."""

    if not schedule or len(schedule) % 4:
        raise ValueError("schedule must contain complete ABBA blocks")
    pairs: list[tuple[ScheduledRun, ScheduledRun]] = []
    for offset in range(0, len(schedule), 4):
        block = tuple(schedule[offset : offset + 4])
        expected_block = offset // 4
        if any(row.index != offset + index for index, row in enumerate(block)):
            raise ValueError("ABBA schedule indices must be contiguous")
        if any(row.block != expected_block for row in block):
            raise ValueError("ABBA schedule block identifiers disagree")
        if tuple(row.pair_slot for row in block) != (0, 1, 2, 3):
            raise ValueError("ABBA schedule pair slots disagree")
        control = block[0].arm
        candidate = block[1].arm
        if not control or not candidate or control == candidate:
            raise ValueError("ABBA schedule must contain distinct A and B arms")
        if tuple(row.arm for row in block) != (
            control,
            candidate,
            candidate,
            control,
        ):
            raise ValueError("schedule does not follow ABBA ordering")
        pairs.extend(((block[0], block[1]), (block[3], block[2])))
    return tuple(pairs)


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _sequence(value: object, *, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{context} must be an array")
    return value


def _finite_positive(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{context} must be a finite positive number")
    return result


def _exact_int(value: object, *, expected: int, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{context} must be exactly {expected}")
    return value


def _validate_compiled_evidence(row: Mapping[str, Any], *, arm: str) -> None:
    mode, _strategy = A1_PROCESS_CONFIG[arm]
    evidence_value = row.get("compiled_verify")
    if mode == "off":
        if evidence_value is None:
            return
        evidence = _mapping(evidence_value, context="compiled verifier evidence")
        compiled_calls = evidence.get("compiled_calls", 0)
        if isinstance(compiled_calls, bool) or not isinstance(compiled_calls, int):
            raise ValueError("compiled verifier compiled_calls must be an integer")
        if compiled_calls != 0:
            raise ValueError("compiled verifier ran while candidate mode was off")
        return
    evidence = _mapping(evidence_value, context="compiled verifier evidence")
    calls = evidence.get("calls")
    compiled_calls = evidence.get("compiled_calls")
    fallback_calls = evidence.get("fallback_calls")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls <= 0:
        raise ValueError("compiled verifier evidence must report positive calls")
    if compiled_calls != calls:
        raise ValueError("compiled verifier evidence is not fully compiled")
    if fallback_calls != 0:
        raise ValueError("compiled verifier evidence reports fallback calls")
    if evidence.get("mode") != mode:
        raise ValueError("compiled verifier evidence mode disagrees with candidate")


def _mean_active_readers(
    row: Mapping[str, Any], *, observation_index: int
) -> float | None:
    resource = row.get("expert_resource_telemetry")
    if resource is None:
        return None
    reader_pool = _mapping(
        _mapping(
            _mapping(resource, context="resource telemetry").get("decode"),
            context="decode resource telemetry",
        ).get("reader_pool"),
        context="decode reader_pool telemetry",
    )
    _exact_int(
        reader_pool.get("worker_capacity"),
        expected=ISSUE51_READER_CAPACITY,
        context=f"observation {observation_index} reader worker capacity",
    )
    return _finite_positive(
        reader_pool.get("mean_active_readers"),
        context=f"observation {observation_index} mean active readers",
    )


def _validate_terminal_mtp_cache_contract(
    contract: Mapping[str, Any], *, context_tokens: int, observation_index: int
) -> None:
    if contract.get("safe_to_commit") is not True:
        raise ValueError(
            f"observation {observation_index} final-state contract is unsafe"
        )
    if contract.get("generated_token_ids_match") is not True:
        raise ValueError(
            f"observation {observation_index} generated token IDs do not match"
        )
    if contract.get("finish_reason_match") is not True:
        raise ValueError(
            f"observation {observation_index} finish reason does not match"
        )
    _exact_int(
        contract.get("prompt_mtp_history_tokens"),
        expected=context_tokens - 1,
        context=f"observation {observation_index} prompt MTP history",
    )
    _exact_int(
        contract.get("mtp_history_position_base"),
        expected=0,
        context=f"observation {observation_index} MTP history position base",
    )

    def exact_offsets(field: str, *, expected: int, label: str) -> None:
        offsets = _sequence(
            contract.get(field), context=f"observation {observation_index} {label}"
        )
        if not offsets:
            raise ValueError(f"observation {observation_index} {label} is empty")
        for offset in offsets:
            _exact_int(
                offset,
                expected=expected,
                context=f"observation {observation_index} {label}",
            )

    exact_offsets(
        "target_cache_offsets",
        expected=context_tokens + 128,
        label="target cache offsets",
    )
    exact_offsets(
        "committed_mtp_cache_offsets",
        expected=context_tokens + 127,
        label="committed MTP cache offsets",
    )


def validate_a1_child(
    payload: Mapping[str, Any], *, arm: str, depths: Sequence[int] = (1,)
) -> dict[CampaignCell, dict[str, float]]:
    """Validate one process-isolated A1 artifact and return its fixed-cell metrics."""

    if arm not in A1_PROCESS_CONFIG:
        raise ValueError(f"unsupported A1 candidate arm: {arm!r}")
    expected_depths = tuple(depths)
    if len(expected_depths) != 1 or expected_depths[0] not in A1_ALLOWED_DEPTHS:
        raise ValueError("A1 validation requires exactly one depth, K=1 or K=2")
    root = _mapping(payload, context="A1 child payload")
    if root.get("schema") != "mtplx-q2-bf16-mtp-depth-matrix-v3":
        raise ValueError("A1 child has the wrong depth-matrix schema")
    if root.get("status") != "passed" or root.get("passed") is not True:
        raise ValueError("A1 child payload must be passed")
    configuration = _mapping(root.get("configuration"), context="configuration")
    if configuration.get("contexts") != [1024, 2048]:
        raise ValueError("A1 child contexts must be exactly [1024, 2048]")
    _exact_int(
        configuration.get("output_tokens"),
        expected=128,
        context="A1 child output token count",
    )
    mode, strategy = A1_PROCESS_CONFIG[arm]
    expected_candidate = {
        "verify_strategy": strategy,
        "compiled_verify_mode": mode,
        "trace_routes": False,
    }
    candidate = _mapping(configuration.get("candidate"), context="candidate")
    if dict(candidate) != expected_candidate:
        raise ValueError(f"A1 child candidate declaration disagrees with arm {arm!r}")

    models = _sequence(root.get("models"), context="models")
    if len(models) != 1:
        raise ValueError("A1 child must contain exactly one Hy3 model")
    model = _mapping(models[0], context="Hy3 model")
    if model.get("model") != "hy3-q2" or model.get("model_key") != "hy3-expert-q2":
        raise ValueError("A1 child must contain only hy3-expert-q2")
    if model.get("depths") != list(expected_depths):
        raise ValueError(f"A1 child depths must be exactly {list(expected_depths)}")
    if model.get("passed") is not True:
        raise ValueError("A1 Hy3 model must be passed")

    cells: dict[CampaignCell, dict[str, float]] = {}
    observations = _sequence(model.get("observations"), context="observations")
    for index, value in enumerate(observations):
        row = _mapping(value, context=f"observation {index}")
        depth = row.get("requested_depth")
        if depth not in {0, *expected_depths}:
            continue
        context_tokens = row.get("context_tokens")
        if isinstance(context_tokens, bool) or not isinstance(context_tokens, int):
            raise ValueError(f"observation {index} context must be an integer")
        cell = CampaignCell(context_tokens, depth)
        if cell in cells:
            raise ValueError(f"A1 child repeats cell {cell}")
        _exact_int(
            row.get("generated_tokens"),
            expected=128,
            context=f"observation {index} output token count",
        )
        gates = _mapping(row.get("gates"), context=f"observation {index} gates")
        if depth == 0:
            if gates.get("output_tokens_exact") is not True:
                raise ValueError(f"observation {index} AR output gate failed")
            cells[cell] = {
                "decode_tok_s": _finite_positive(
                    row.get("decode_tok_s"), context=f"observation {index} decode TPS"
                ),
                "end_to_end_tok_s": _finite_positive(
                    row.get("end_to_end_tok_s"),
                    context=f"observation {index} end-to-end TPS",
                ),
            }
            mean_active_readers = _mean_active_readers(row, observation_index=index)
            if mean_active_readers is not None:
                cells[cell]["mean_active_readers"] = mean_active_readers
            continue
        missing = sorted(REQUIRED_A1_GATES.difference(gates))
        if missing:
            raise ValueError(
                f"observation {index} is missing required gate(s): {', '.join(missing)}"
            )
        failed = sorted(
            name for name in REQUIRED_A1_GATES if gates.get(name) is not True
        )
        if failed:
            raise ValueError(
                f"observation {index} has failed gate(s): {', '.join(failed)}"
            )
        final_state = _mapping(
            row.get("final_state_contract"),
            context=f"observation {index} final-state contract",
        )
        _validate_terminal_mtp_cache_contract(
            final_state,
            context_tokens=context_tokens,
            observation_index=index,
        )
        _validate_compiled_evidence(row, arm=arm)
        cells[cell] = {
            "decode_tok_s": _finite_positive(
                row.get("decode_tok_s"), context=f"observation {index} decode TPS"
            ),
            "end_to_end_tok_s": _finite_positive(
                row.get("end_to_end_tok_s"),
                context=f"observation {index} end-to-end TPS",
            ),
        }
        mean_active_readers = _mean_active_readers(row, observation_index=index)
        if mean_active_readers is not None:
            cells[cell]["mean_active_readers"] = mean_active_readers
    observed = {(cell.context_tokens, cell.depth) for cell in cells}
    expected_cells = {
        (context, depth) for context in (1024, 2048) for depth in (0, *expected_depths)
    }
    if observed != expected_cells:
        raise ValueError(
            "A1 child retained matrix must contain exactly 1024/2048 x "
            f"K=0/K={expected_depths[0]}"
        )
    return cells


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_mean_interval(
    values: Sequence[float], *, samples: int = 10_000
) -> tuple[float, float]:
    rng = random.Random(0)
    count = len(values)
    means = [
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    ]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def paired_decode_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute deterministic paired decode and end-to-end fractional gains."""

    if not rows:
        raise ValueError("paired rows must not be empty")
    decode_gains: list[float] = []
    end_to_end_gains: list[float] = []
    details: list[dict[str, float | int]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, context=f"paired row {index}")
        control_decode = _finite_positive(
            row.get("control_decode_tok_s"), context="control decode TPS"
        )
        candidate_decode = _finite_positive(
            row.get("candidate_decode_tok_s"), context="candidate decode TPS"
        )
        control_end_to_end = _finite_positive(
            row.get("control_end_to_end_tok_s"), context="control end-to-end TPS"
        )
        candidate_end_to_end = _finite_positive(
            row.get("candidate_end_to_end_tok_s"),
            context="candidate end-to-end TPS",
        )
        decode_gain = candidate_decode / control_decode - 1.0
        end_to_end_gain = candidate_end_to_end / control_end_to_end - 1.0
        decode_gains.append(decode_gain)
        end_to_end_gains.append(end_to_end_gain)
        details.append(
            {
                "pair": index + 1,
                "control_decode_tok_s": control_decode,
                "candidate_decode_tok_s": candidate_decode,
                "fractional_decode_gain": decode_gain,
                "control_end_to_end_tok_s": control_end_to_end,
                "candidate_end_to_end_tok_s": candidate_end_to_end,
                "fractional_end_to_end_gain": end_to_end_gain,
            }
        )
    return {
        "samples": len(decode_gains),
        "bootstrap_samples": 10_000,
        "paired_fractional_decode_gains": decode_gains,
        "mean_fractional_decode_gain": statistics.fmean(decode_gains),
        "median_fractional_decode_gain": statistics.median(decode_gains),
        "p95_fractional_decode_gain": _percentile(decode_gains, 0.95),
        "bootstrap_95_interval": _bootstrap_mean_interval(decode_gains),
        "paired_fractional_end_to_end_gains": end_to_end_gains,
        "mean_fractional_end_to_end_gain": statistics.fmean(end_to_end_gains),
        "median_fractional_end_to_end_gain": statistics.median(end_to_end_gains),
        "p95_fractional_end_to_end_gain": _percentile(end_to_end_gains, 0.95),
        "end_to_end_bootstrap_95_interval": _bootstrap_mean_interval(end_to_end_gains),
        "pairs": details,
    }


def decide_performance(
    stats: Mapping[str, Any], *, default_threshold: float = 0.05
) -> dict[str, Any]:
    """Apply the Issue #51 end-to-end and five-percent promotion gates."""

    if not math.isfinite(default_threshold) or default_threshold < 0.0:
        raise ValueError("default_threshold must be finite and non-negative")
    decode_interval = _sequence(
        stats.get("bootstrap_95_interval"), context="decode bootstrap interval"
    )
    end_to_end_interval = _sequence(
        stats.get("end_to_end_bootstrap_95_interval"),
        context="end-to-end bootstrap interval",
    )
    if len(decode_interval) != 2 or len(end_to_end_interval) != 2:
        raise ValueError("bootstrap intervals must contain two bounds")
    decode_lower = float(decode_interval[0])
    end_to_end_lower = float(end_to_end_interval[0])
    mean = stats.get("mean_fractional_decode_gain")
    if (
        isinstance(mean, bool)
        or not isinstance(mean, (int, float))
        or not all(
            math.isfinite(value) for value in (decode_lower, end_to_end_lower, mean)
        )
    ):
        raise ValueError("performance decision inputs must be finite")
    positive_decode_interval = decode_lower > 0.0
    positive_end_to_end_interval = end_to_end_lower > 0.0
    meets_default_threshold = float(mean) >= default_threshold
    promote = (
        positive_decode_interval
        and positive_end_to_end_interval
        and meets_default_threshold
    )
    failed = [
        label
        for passed, label in (
            (positive_decode_interval, "decode interval is not positive"),
            (positive_end_to_end_interval, "end-to-end interval is not positive"),
            (meets_default_threshold, "mean decode gain is below threshold"),
        )
        if not passed
    ]
    return {
        "promote": promote,
        "default_threshold": default_threshold,
        "positive_decode_interval": positive_decode_interval,
        "positive_end_to_end_interval": positive_end_to_end_interval,
        "meets_default_threshold": meets_default_threshold,
        "reason": "all performance gates passed" if promote else "; ".join(failed),
    }
