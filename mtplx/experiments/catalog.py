"""Discovery and lifecycle enforcement for data-only experiment recipes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from .schema import ExperimentRecipe, ExperimentStatus, load_experiment


@dataclass(frozen=True)
class ResolvedExperiment:
    recipe: ExperimentRecipe
    sha256: str
    source: Path


def _recipe_hash(recipe: ExperimentRecipe) -> str:
    normalized = json.dumps(
        recipe.canonical_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


class ExperimentCatalog:
    def __init__(self, recipes_root: str | Path):
        self.recipes_root = Path(recipes_root).expanduser()
        self.by_id: dict[str, ExperimentRecipe] = {}
        if not self.recipes_root.is_dir():
            raise ValueError(f"experiment recipes directory not found: {self.recipes_root}")
        for path in sorted(self.recipes_root.glob("*.toml")):
            recipe = load_experiment(path)
            if recipe.id in self.by_id:
                previous = self.by_id[recipe.id]
                raise ValueError(
                    f"duplicate experiment id {recipe.id}: "
                    f"{previous.source} and {recipe.source}"
                )
            self.by_id[recipe.id] = recipe

    def list(self, *, active_only: bool = False) -> tuple[ExperimentRecipe, ...]:
        recipes = self.by_id.values()
        if active_only:
            recipes = (
                recipe
                for recipe in recipes
                if recipe.status is ExperimentStatus.ACTIVE
            )
        return tuple(sorted(recipes, key=lambda recipe: recipe.id))

    def resolve(self, uri: str) -> ResolvedExperiment:
        if not uri.startswith("lab:"):
            raise ValueError("experiment references must use lab:ID")
        experiment_id = uri.removeprefix("lab:").strip()
        if not experiment_id:
            raise ValueError("experiment reference is missing an id")
        try:
            recipe = self.by_id[experiment_id]
        except KeyError as exc:
            raise ValueError(f"unknown experiment: {experiment_id}") from exc
        if recipe.status is not ExperimentStatus.ACTIVE:
            raise ValueError(
                f"experiment {recipe.id} is {recipe.status.value} and not executable"
            )
        return ResolvedExperiment(recipe, _recipe_hash(recipe), recipe.source)


def default_experiment_catalog() -> ExperimentCatalog:
    recipes = files("mtplx.experiments").joinpath("recipes")
    return ExperimentCatalog(Path(str(recipes)))
