from __future__ import annotations

from difflib import get_close_matches
from typing import Iterable

from .schema import SettingSpec


class SettingCatalog:
    def __init__(self, specs: Iterable[SettingSpec]):
        self.by_name: dict[str, SettingSpec] = {}
        self.by_alias: dict[tuple[str, str], SettingSpec] = {}
        for spec in specs:
            if spec.name in self.by_name:
                raise ValueError(f"duplicate setting: {spec.name}")
            self.by_name[spec.name] = spec
            for alias in spec.aliases:
                key = (alias.source, alias.name)
                if key in self.by_alias:
                    raise ValueError(
                        f"duplicate setting alias: {alias.source}:{alias.name}"
                    )
                self.by_alias[key] = spec

    def require(self, name: str) -> SettingSpec:
        try:
            return self.by_name[name]
        except KeyError as exc:
            suggestions = self.suggest(name)
            suffix = f"; did you mean {', '.join(suggestions)}" if suggestions else ""
            raise ValueError(f"unknown setting: {name}{suffix}") from exc

    def resolve_alias(self, source: str, name: str) -> SettingSpec | None:
        return self.by_alias.get((source, name))

    def suggest(self, name: str) -> list[str]:
        return get_close_matches(name, self.by_name, n=3, cutoff=0.55)

    def list(
        self, *, domain: str | None = None, visibility: str | None = None
    ) -> tuple[SettingSpec, ...]:
        specs = self.by_name.values()
        return tuple(
            sorted(
                (
                    spec
                    for spec in specs
                    if (domain is None or spec.domain == domain)
                    and (
                        visibility is None
                        or spec.visibility.value == visibility
                    )
                ),
                key=lambda spec: spec.name,
            )
        )
