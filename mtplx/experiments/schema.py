"""Strict, no-MLX schema for experiment settings recipes."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class ExperimentStatus(str, Enum):
    ACTIVE = "active"
    RETAINED = "retained"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


@dataclass(frozen=True)
class ExperimentRecipe:
    id: str
    title: str
    status: ExperimentStatus
    owner: str
    tracking: str
    created: date
    review_after: date
    models: tuple[str, ...]
    purpose: str
    settings: Mapping[str, Any]
    source: Path
    replacement: str | None = None
    result: str | None = None

    def canonical_payload(self) -> dict[str, Any]:
        experiment: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "owner": self.owner,
            "tracking": self.tracking,
            "created": self.created.isoformat(),
            "review_after": self.review_after.isoformat(),
            "models": list(self.models),
            "purpose": self.purpose,
        }
        if self.replacement is not None:
            experiment["replacement"] = self.replacement
        if self.result is not None:
            experiment["result"] = self.result
        return {"experiment": experiment, "settings": dict(self.settings)}


_ALLOWED_TOP_LEVEL = frozenset({"experiment", "settings"})
_REQUIRED_METADATA = frozenset(
    {
        "id",
        "title",
        "status",
        "owner",
        "tracking",
        "created",
        "review_after",
        "models",
        "purpose",
    }
)
_OPTIONAL_METADATA = frozenset({"replacement", "result"})
_SCALAR_TYPES = (bool, int, float, str)
_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _required_text(metadata: Mapping[str, Any], name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"experiment.{name} must be a non-empty string")
    return value.strip()


def _optional_text(metadata: Mapping[str, Any], name: str) -> str | None:
    value = metadata.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"experiment.{name} must be a non-empty string")
    return value.strip()


def _date(metadata: Mapping[str, Any], name: str) -> date:
    value = metadata.get(name)
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"experiment.{name} must be an ISO date") from exc
    raise ValueError(f"experiment.{name} must be an ISO date")


def load_experiment(
    path: str | Path, *, today: date | None = None
) -> ExperimentRecipe:
    source = Path(path).expanduser()
    with source.open("rb") as handle:
        document = tomllib.load(handle)

    unsupported = sorted(set(document).difference(_ALLOWED_TOP_LEVEL))
    if unsupported:
        raise ValueError(f"unsupported top-level table: {unsupported[0]}")
    missing_tables = sorted(_ALLOWED_TOP_LEVEL.difference(document))
    if missing_tables:
        raise ValueError(f"missing top-level table: {missing_tables[0]}")

    metadata = document["experiment"]
    settings = document["settings"]
    if not isinstance(metadata, dict):
        raise ValueError("[experiment] must be a table")
    if not isinstance(settings, dict):
        raise ValueError("[settings] must be a table")

    missing = sorted(_REQUIRED_METADATA.difference(metadata))
    if missing:
        raise ValueError(f"missing experiment metadata: {missing[0]}")
    unknown = sorted(
        set(metadata).difference(_REQUIRED_METADATA | _OPTIONAL_METADATA)
    )
    if unknown:
        raise ValueError(f"unsupported experiment metadata: {unknown[0]}")

    experiment_id = _required_text(metadata, "id")
    if not _ID_RE.fullmatch(experiment_id):
        raise ValueError("experiment.id must be a lowercase kebab-case name")
    try:
        status = ExperimentStatus(_required_text(metadata, "status"))
    except ValueError as exc:
        choices = ", ".join(item.value for item in ExperimentStatus)
        raise ValueError(f"experiment.status must be one of {choices}") from exc

    models_value = metadata.get("models")
    if (
        not isinstance(models_value, list)
        or not models_value
        or any(not isinstance(item, str) or not item.strip() for item in models_value)
    ):
        raise ValueError("experiment.models must be a non-empty string array")
    models = tuple(item.strip() for item in models_value)

    parsed_settings: dict[str, Any] = {}
    for name, value in settings.items():
        if not isinstance(name, str) or not name:
            raise ValueError("settings must use non-empty string keys")
        if not isinstance(value, _SCALAR_TYPES):
            raise ValueError("settings must use scalar values")
        parsed_settings[name] = value
    if not parsed_settings:
        raise ValueError("experiment settings must not be empty")

    created = _date(metadata, "created")
    review_after = _date(metadata, "review_after")
    if review_after < created:
        raise ValueError("experiment review date precedes creation date")
    current_date = today or date.today()
    if status is ExperimentStatus.ACTIVE and review_after <= current_date:
        raise ValueError("active experiment review date has passed")

    replacement = _optional_text(metadata, "replacement")
    result = _optional_text(metadata, "result")
    if status is ExperimentStatus.ACTIVE and (replacement or result):
        raise ValueError("active experiments cannot declare replacement or result")

    return ExperimentRecipe(
        id=experiment_id,
        title=_required_text(metadata, "title"),
        status=status,
        owner=_required_text(metadata, "owner"),
        tracking=_required_text(metadata, "tracking"),
        created=created,
        review_after=review_after,
        models=models,
        purpose=_required_text(metadata, "purpose"),
        settings=MappingProxyType(parsed_settings),
        source=source.resolve(),
        replacement=replacement,
        result=result,
    )
