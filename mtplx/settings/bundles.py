from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


_SCALAR_TYPES = (bool, int, float, str)


@dataclass(frozen=True, eq=False)
class LoadedSettingsBundle(Mapping[str, Any]):
    settings: Mapping[str, Any]
    source: str
    sha256: str
    id: str | None = None
    experiment: Mapping[str, Any] | None = None

    def __getitem__(self, name: str) -> Any:
        return self.settings[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.settings)

    def __len__(self) -> int:
        return len(self.settings)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LoadedSettingsBundle):
            return (
                dict(self.settings) == dict(other.settings)
                and self.source == other.source
                and self.sha256 == other.sha256
                and self.id == other.id
                and self.experiment == other.experiment
            )
        if isinstance(other, Mapping):
            return dict(self.settings) == dict(other)
        return NotImplemented


def _settings_hash(settings: Mapping[str, Any]) -> str:
    normalized = json.dumps(
        dict(settings), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def load_settings_bundle(
    path: str | Path, *, model_family: str | None = None
) -> LoadedSettingsBundle:
    source = str(path)
    if source.startswith("lab:"):
        from mtplx.experiments.catalog import default_experiment_catalog

        resolved_experiment = default_experiment_catalog().resolve(source)
        recipe = resolved_experiment.recipe
        if model_family is not None and model_family not in recipe.models:
            supported = ", ".join(recipe.models)
            raise ValueError(
                f"experiment {recipe.id} supports {supported}, not {model_family}"
            )
        return LoadedSettingsBundle(
            settings=recipe.settings,
            source=source,
            sha256=resolved_experiment.sha256,
            id=recipe.id,
            experiment=MappingProxyType(
                recipe.canonical_payload()["experiment"]
            ),
        )

    resolved = Path(path).expanduser()
    with resolved.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {"settings"} or not isinstance(payload["settings"], dict):
        raise ValueError(f"{resolved}: settings bundles may contain only [settings]")
    settings: dict[str, Any] = {}
    for name, value in payload["settings"].items():
        if not isinstance(name, str) or not isinstance(value, _SCALAR_TYPES):
            raise ValueError(
                f"{resolved}: settings must use string keys and scalar values"
            )
        settings[name] = value
    return LoadedSettingsBundle(
        settings=MappingProxyType(settings),
        source=str(resolved.resolve()),
        sha256=_settings_hash(settings),
    )
