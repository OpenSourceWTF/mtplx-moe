#!/usr/bin/env python3
"""Compare Issue 64 stock/device-K artifacts only after identity gates pass."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "mtplx-q2-bf16-mtp-depth-matrix-v3"
_CONFIGURATION_FIELDS = (
    "contexts",
    "output_tokens",
    "retained_replicates",
    "warmup_output_tokens",
    "sampler",
    "runtime",
    "generation_environment",
    "measurement_lane",
    "mtp_resident",
)
_MODEL_FIELDS = (
    "model",
    "model_key",
    "model_root",
    "manifest",
    "mtp_artifacts",
    "mtp_precision",
    "runtime_config",
    "depths",
)
_ACCEPTANCE_FIELDS = (
    "accepted_drafts",
    "drafted_tokens",
    "evaluated_drafts",
    "acceptance_by_depth",
    "verify_calls",
)


class ArtifactComparisonError(ValueError):
    """Raised before a performance delta can be reported."""


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactComparisonError(f"{context} must be an object")
    return value


def _sequence(value: object, *, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ArtifactComparisonError(f"{context} must be an array")
    return value


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if payload.get("schema") != SCHEMA:
        raise ArtifactComparisonError(f"{label} artifact must use schema {SCHEMA}")
    if payload.get("status") != "passed" or payload.get("passed") is not True:
        raise ArtifactComparisonError(f"{label} artifact must be a passed matrix")
    configuration = _mapping(
        payload.get("configuration"),
        context=f"{label} configuration",
    )
    models = _sequence(payload.get("models"), context=f"{label} models")
    if len(models) != 1:
        raise ArtifactComparisonError(
            f"{label} artifact must contain exactly one model"
        )
    return configuration, _mapping(models[0], context=f"{label} model")


def _row_key(row: Mapping[str, Any], *, label: str) -> tuple[int, int, int]:
    values: list[int] = []
    for name, minimum in (
        ("context_tokens", 1),
        ("requested_depth", 0),
        ("replicate", 1),
    ):
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ArtifactComparisonError(
                f"{label} observation {name} must be an integer >= {minimum}"
            )
        values.append(value)
    return values[0], values[1], values[2]


def _rows(
    model: Mapping[str, Any],
    *,
    label: str,
) -> dict[tuple[int, int, int], Mapping[str, Any]]:
    indexed: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    observations = _sequence(
        model.get("observations"),
        context=f"{label} observations",
    )
    for value in observations:
        row = _mapping(value, context=f"{label} observation")
        key = _row_key(row, label=label)
        if key in indexed:
            raise ArtifactComparisonError(
                f"{label} artifact repeats context={key[0]} depth={key[1]} "
                f"replicate={key[2]}"
            )
        indexed[key] = row
    if not indexed:
        raise ArtifactComparisonError(f"{label} artifact has no observations")
    return indexed


def _prompt_identity(
    row: Mapping[str, Any],
    *,
    label: str,
    key: tuple[int, int, int],
) -> Mapping[str, Any]:
    identity = _mapping(
        row.get("prompt_identity"),
        context=(
            f"{label} prompt identity for context={key[0]} depth={key[1]} "
            f"replicate={key[2]}"
        ),
    )
    token_sha256 = identity.get("token_sha256")
    token_count = identity.get("token_count")
    if not isinstance(token_sha256, str) or not token_sha256:
        raise ArtifactComparisonError(
            f"{label} prompt identity for context={key[0]} depth={key[1]} "
            f"replicate={key[2]} has no token_sha256"
        )
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count <= 0
    ):
        raise ArtifactComparisonError(
            f"{label} prompt identity for context={key[0]} depth={key[1]} "
            f"replicate={key[2]} has invalid token_count"
        )
    return identity


def _draft_core(configuration: Mapping[str, Any], *, label: str) -> str:
    generation = _mapping(
        configuration.get("generation"),
        context=f"{label} configuration generation",
    )
    value = generation.get("draft_core")
    if not isinstance(value, str) or not value:
        raise ArtifactComparisonError(f"{label} arm draft_core is missing")
    return value


def _without_draft_core(value: object, *, context: str) -> dict[str, Any]:
    normalized = dict(_mapping(value, context=context))
    normalized.pop("draft_core", None)
    return normalized


def _number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactComparisonError(f"{context} must be numeric")
    parsed = float(value)
    if parsed <= 0.0:
        raise ArtifactComparisonError(f"{context} must be positive")
    return parsed


def compare_artifacts(
    stock: Mapping[str, Any],
    device: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on identity/config drift, then calculate per-cell deltas."""

    stock_configuration, stock_model = _validate_payload(stock, label="stock")
    device_configuration, device_model = _validate_payload(device, label="device-k")
    stock_rows = _rows(stock_model, label="stock")
    device_rows = _rows(device_model, label="device-k")

    # Prompt identity is the first cross-artifact comparison gate.  No
    # throughput value is parsed until every shared cell passes it.  This also
    # makes a partial historical K0-K3 artifact fail for its prompt confound
    # before the later K0-K7 shape gate.
    identities: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    shared_keys = set(stock_rows) & set(device_rows)
    if not shared_keys:
        raise ArtifactComparisonError("stock and device-k artifacts share no cells")
    for key in sorted(shared_keys):
        stock_identity = _prompt_identity(
            stock_rows[key],
            label="stock",
            key=key,
        )
        device_identity = _prompt_identity(
            device_rows[key],
            label="device-k",
            key=key,
        )
        if stock_identity != device_identity:
            raise ArtifactComparisonError(
                f"prompt identity mismatch for context={key[0]} depth={key[1]} "
                f"replicate={key[2]}: stock "
                f"token_sha256={stock_identity.get('token_sha256')}, device-k "
                f"token_sha256={device_identity.get('token_sha256')}; "
                "performance comparison refused"
            )
        identities[key] = stock_identity
    if set(stock_rows) != set(device_rows):
        raise ArtifactComparisonError("stock and device-k observation cells differ")

    for name in _CONFIGURATION_FIELDS:
        if stock_configuration.get(name) != device_configuration.get(name):
            raise ArtifactComparisonError(f"configuration {name} disagrees")
    for name in ("generation", "candidate"):
        stock_value = _without_draft_core(
            stock_configuration.get(name),
            context=f"stock configuration {name}",
        )
        device_value = _without_draft_core(
            device_configuration.get(name),
            context=f"device-k configuration {name}",
        )
        if stock_value != device_value:
            raise ArtifactComparisonError(f"configuration {name} disagrees")
    for name in _MODEL_FIELDS:
        if stock_model.get(name) != device_model.get(name):
            raise ArtifactComparisonError(f"model identity {name} disagrees")

    stock_core = _draft_core(stock_configuration, label="stock")
    device_core = _draft_core(device_configuration, label="device-k")
    if stock_core != "stock":
        raise ArtifactComparisonError(
            f"stock arm draft_core must be stock, got {stock_core}"
        )
    if device_core != "device-k":
        raise ArtifactComparisonError(
            f"device-k arm draft_core must be device-k, got {device_core}"
        )

    compared_rows: list[dict[str, Any]] = []
    for key in sorted(stock_rows):
        stock_row = stock_rows[key]
        device_row = device_rows[key]
        stock_tps = _number(
            stock_row.get("decode_tok_s"),
            context=f"stock decode_tok_s for {key}",
        )
        device_tps = _number(
            device_row.get("decode_tok_s"),
            context=f"device-k decode_tok_s for {key}",
        )
        acceptance_identical = all(
            stock_row.get(name) == device_row.get(name) for name in _ACCEPTANCE_FIELDS
        )
        compared_rows.append(
            {
                "context_tokens": key[0],
                "requested_depth": key[1],
                "replicate": key[2],
                "prompt_token_sha256": identities[key]["token_sha256"],
                "stock_decode_tok_s": stock_tps,
                "device_decode_tok_s": device_tps,
                "device_over_stock_ratio": round(device_tps / stock_tps, 12),
                "tokens_identical": stock_row.get("token_ids")
                == device_row.get("token_ids"),
                "acceptance_identical": acceptance_identical,
                "final_state_identical": stock_row.get("final_state_contract")
                == device_row.get("final_state_contract"),
                "event_contract_identical": stock_row.get("speculative_event_contract")
                == device_row.get("speculative_event_contract"),
            }
        )

    return {
        "schema": "mtplx-issue64-draft-artifact-comparison-v1",
        "status": "comparable",
        "prompt_identity_match": True,
        "stock_draft_core": stock_core,
        "device_draft_core": device_core,
        "all_tokens_identical": all(
            bool(row["tokens_identical"]) for row in compared_rows
        ),
        "all_acceptance_identical": all(
            bool(row["acceptance_identical"]) for row in compared_rows
        ),
        "all_final_state_identical": all(
            bool(row["final_state_identical"]) for row in compared_rows
        ),
        "rows": compared_rows,
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        return _mapping(json.load(handle), context=str(path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-json", type=Path, required=True)
    parser.add_argument("--device-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare_artifacts(
            _load_json(args.stock_json),
            _load_json(args.device_json),
        )
    except (ArtifactComparisonError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        target = args.output_json.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
