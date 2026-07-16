#!/usr/bin/env python3
"""Join matched Issue 51 headline/resource matrices and render one evidence table."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "mtplx-q2-bf16-mtp-depth-matrix-v3"
HEADLINE_LANE = "headline-uninstrumented"
DIAGNOSTIC_LANE = "diagnostic-resource-instrumented"


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _sequence(value: object, *, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{context} must be an array")
    return value


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _lane(payload: Mapping[str, Any]) -> str:
    configuration = _mapping(payload.get("configuration"), context="configuration")
    return str(configuration.get("measurement_lane") or "")


def _validate_payload(payload: Mapping[str, Any], *, lane: str) -> Mapping[str, Any]:
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"payload must use schema {SCHEMA}")
    if payload.get("status") != "passed" or payload.get("passed") is not True:
        raise ValueError("payload must be a passed matrix")
    if _lane(payload) != lane:
        raise ValueError(f"payload must use {lane}")
    models = _sequence(payload.get("models"), context="models")
    if len(models) != 1:
        raise ValueError("Issue 51 table requires exactly one model per payload")
    return _mapping(models[0], context="model")


def _row_key(row: Mapping[str, Any]) -> tuple[int, int]:
    context = row.get("context_tokens")
    depth = row.get("requested_depth")
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise ValueError("observation context must be a positive integer")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("observation depth must be a non-negative integer")
    return context, depth


def _rows(model: Mapping[str, Any], *, diagnostic: bool) -> dict[tuple[int, int], Mapping[str, Any]]:
    indexed: dict[tuple[int, int], Mapping[str, Any]] = {}
    for value in _sequence(model.get("observations"), context="observations"):
        row = _mapping(value, context="observation")
        key = _row_key(row)
        if key in indexed:
            raise ValueError(f"duplicate observation cell {key}")
        resource = row.get("resource_telemetry")
        if diagnostic:
            report = _mapping(resource, context=f"resource telemetry for {key}")
            if report.get("schema") != "mtplx-resource-telemetry-v2":
                raise ValueError(f"resource telemetry schema disagrees for {key}")
        elif resource is not None:
            raise ValueError(f"headline row {key} unexpectedly contains telemetry")
        indexed[key] = row
    return indexed


def summarize(
    headline: Mapping[str, Any], diagnostic: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and join one telemetry-off matrix with its telemetry-on twin."""

    headline_model = _validate_payload(headline, lane=HEADLINE_LANE)
    diagnostic_model = _validate_payload(diagnostic, lane=DIAGNOSTIC_LANE)
    for name in ("model", "model_key"):
        if headline_model.get(name) != diagnostic_model.get(name):
            raise ValueError(f"model identity {name} disagrees")

    headline_configuration = _mapping(
        headline.get("configuration"), context="headline configuration"
    )
    diagnostic_configuration = _mapping(
        diagnostic.get("configuration"), context="diagnostic configuration"
    )
    for name in ("contexts", "output_tokens"):
        if headline_configuration.get(name) != diagnostic_configuration.get(name):
            raise ValueError(f"configuration {name} disagrees")

    headline_rows = _rows(headline_model, diagnostic=False)
    diagnostic_rows = _rows(diagnostic_model, diagnostic=True)
    if set(headline_rows) != set(diagnostic_rows):
        raise ValueError("headline and diagnostic cells disagree")

    joined: list[dict[str, Any]] = []
    prompt_policies: set[str] = set()
    prompt_formats: set[str] = set()
    for key in sorted(headline_rows):
        speed = headline_rows[key]
        resource = diagnostic_rows[key]
        speed_identity = dict(
            _mapping(speed.get("prompt_identity"), context=f"prompt identity {key}")
        )
        resource_identity = dict(
            _mapping(
                resource.get("prompt_identity"),
                context=f"diagnostic prompt identity {key}",
            )
        )
        if speed_identity != resource_identity:
            raise ValueError(f"prompt identity disagrees for cell {key}")
        policy = speed_identity.get("prompt_policy")
        prompt_format = speed_identity.get("prompt_format")
        if not isinstance(policy, str) or not policy:
            raise ValueError(f"prompt identity policy is missing for cell {key}")
        if not isinstance(prompt_format, str) or not prompt_format:
            raise ValueError(f"prompt identity format is missing for cell {key}")
        prompt_policies.add(policy)
        prompt_formats.add(prompt_format)
        joined.append(
            {
                "context_tokens": key[0],
                "depth": key[1],
                "headline": dict(speed),
                "diagnostic": dict(resource),
            }
        )
    if len(prompt_policies) != 1 or len(prompt_formats) != 1:
        raise ValueError("matrix must use one prompt policy and format")
    return {
        "schema": "mtplx-issue51-next-k-summary-v1",
        "model": headline_model.get("model"),
        "model_key": headline_model.get("model_key"),
        "output_tokens": headline_configuration.get("output_tokens"),
        "prompt_policy": next(iter(prompt_policies)),
        "prompt_format": next(iter(prompt_formats)),
        "rows": joined,
    }


def _percent(value: object) -> str:
    parsed = _number(value)
    return "n/a" if parsed is None else f"{parsed * 100:.1f}%"


def _fixed(value: object, digits: int = 3) -> str:
    parsed = _number(value)
    return "n/a" if parsed is None else f"{parsed:.{digits}f}"


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render the Issue 51 big table without mixing instrumented speed values."""

    rows = _sequence(summary.get("rows"), context="summary rows")
    controls: dict[int, float] = {}
    for value in rows:
        row = _mapping(value, context="summary row")
        if row.get("depth") == 0:
            headline = _mapping(row.get("headline"), context="headline row")
            decode = _number(headline.get("decode_tok_s"))
            if decode is not None:
                controls[int(row["context_tokens"])] = decode

    lines = [
        "# Issue 51 realistic-programming Next-K matrix",
        "",
        (
            f"Prompt policy: `{summary.get('prompt_policy')}` in "
            f"`{summary.get('prompt_format')}` format; output: "
            f"{summary.get('output_tokens')} tokens."
        ),
        "",
        (
            "Decode speed and timing come from the headline telemetry-off lane. "
            "Memory, CPU, SSD, GPU, and reader values come from the matched "
            "telemetry-on lane. Unavailable GPU or SSD coverage is shown as `n/a`, "
            "never as zero utilization."
        ),
        "",
        (
            "| Input | K | Prefill s | Decode s | Decode tok/s | Delta vs K0 | "
            "Cache hit | Accepted / evaluated | Yield | Verify calls | "
            "Memory GiB / % | CPU cores p/g | SSD GiB/s / % | GPU busy | "
            "Readers mean/peak | AR parity |"
        ),
        (
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | :--- |"
        ),
    ]
    for value in rows:
        row = _mapping(value, context="summary row")
        context = int(row["context_tokens"])
        depth = int(row["depth"])
        headline = _mapping(row.get("headline"), context="headline row")
        diagnostic = _mapping(row.get("diagnostic"), context="diagnostic row")
        report = _mapping(
            diagnostic.get("resource_telemetry"), context="resource report"
        )
        memory = _mapping(report.get("memory"), context="memory report")
        host = _mapping(report.get("host"), context="host report")
        storage = _mapping(report.get("storage"), context="storage report")
        readers = _mapping(report.get("reader_pool"), context="reader report")
        power = _mapping(report.get("powermetrics"), context="powermetrics report")

        decode = _number(headline.get("decode_tok_s"))
        control = controls.get(context)
        delta = (
            "control"
            if depth == 0
            else "n/a"
            if decode is None or control in {None, 0.0}
            else f"{(decode / control - 1.0) * 100:+.1f}%"
        )
        accepted = headline.get("accepted_drafts")
        evaluated = headline.get("evaluated_drafts")
        accepted_cell = (
            "n/a" if depth == 0 else f"{int(accepted or 0)} / {int(evaluated or 0)}"
        )
        yield_cell = "n/a" if depth == 0 else _percent(headline.get("conditional_hit_rate"))

        peak_bytes = _number(memory.get("peak_memory_bytes"))
        memory_gib = None if peak_bytes is None else peak_bytes / 1024**3
        memory_cell = (
            f"{_fixed(memory_gib, 2)} / {_percent(memory.get('utilization_of_limit'))}"
        )
        process_cpu = _number(power.get("process_cpu_ms_per_s_mean"))
        process_cores = None if process_cpu is None else process_cpu / 1000.0
        generation_core = host.get("generation_thread_core_fraction")
        cpu_cell = f"{_fixed(process_cores, 2)} / {_fixed(generation_core, 2)}"
        ssd_cell = (
            f"{_fixed(storage.get('mean_gib_per_second'), 2)} / "
            f"{_percent(storage.get('utilization_of_ceiling'))}"
        )
        gpu_cell = (
            _percent(power.get("process_gpu_busy_fraction"))
            if power.get("available") is True
            else "n/a"
        )
        reader_cell = (
            f"{_fixed(readers.get('mean_active_readers'), 2)} / "
            f"{int(readers.get('lifetime_active_readers_peak') or 0)}"
        )
        comparison = _mapping(headline.get("ar_comparison"), context="AR comparison")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(context),
                    str(depth),
                    _fixed(headline.get("prompt_target_prefill_time_s"), 3),
                    _fixed(headline.get("decode_elapsed_s"), 3),
                    _fixed(decode, 3),
                    delta,
                    _percent(headline.get("decode_expert_cache_hit_rate")),
                    accepted_cell,
                    yield_cell,
                    str(int(headline.get("verify_calls") or 0)),
                    memory_cell,
                    cpu_cell,
                    ssd_cell,
                    gpu_cell,
                    reader_cell,
                    str(comparison.get("status") or "unknown"),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        return _mapping(json.load(handle), context=str(path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline-json", type=Path, required=True)
    parser.add_argument("--resource-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args(argv)
    markdown = render_markdown(
        summarize(_load_json(args.headline_json), _load_json(args.resource_json))
    )
    if args.output_markdown is not None:
        target = args.output_markdown.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
