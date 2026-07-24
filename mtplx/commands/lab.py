"""No-MLX experiment recipe inspection commands."""

from __future__ import annotations

import json
from typing import Any

from mtplx.experiments.catalog import default_experiment_catalog
from mtplx.experiments.schema import ExperimentRecipe, ExperimentStatus, load_experiment
from mtplx.settings.builtins import default_setting_catalog


def _validated_settings(recipe: ExperimentRecipe) -> dict[str, Any]:
    catalog = default_setting_catalog()
    settings: dict[str, Any] = {}
    for name, raw in recipe.settings.items():
        spec = catalog.require(name)
        value = spec.parse(raw)
        settings[name] = "[redacted]" if spec.secret and value else value
    return settings


def _metadata(recipe: ExperimentRecipe) -> dict[str, Any]:
    return {
        "id": recipe.id,
        "title": recipe.title,
        "status": recipe.status.value,
        "owner": recipe.owner,
        "tracking": recipe.tracking,
        "created": recipe.created.isoformat(),
        "review_after": recipe.review_after.isoformat(),
        "models": list(recipe.models),
        "purpose": recipe.purpose,
        "replacement": recipe.replacement,
        "result": recipe.result,
    }


def _emit(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        if value is not None:
            print(f"{key}: {json.dumps(value, default=str)}")


def _list(args: Any) -> int:
    catalog = default_experiment_catalog()
    recipes = catalog.list(active_only=not bool(args.all))
    payload = {"experiments": [_metadata(recipe) for recipe in recipes]}
    if args.json:
        _emit(payload, json_output=True)
    else:
        for item in payload["experiments"]:
            print(f"{item['id']}  {item['status']}  {item['owner']}  {item['title']}")
    return 0


def _show(args: Any) -> int:
    resolved = default_experiment_catalog().resolve(f"lab:{args.experiment_id}")
    payload = {
        **_metadata(resolved.recipe),
        "sha256": resolved.sha256,
        "source": f"lab:{resolved.recipe.id}",
        "settings": _validated_settings(resolved.recipe),
    }
    _emit(payload, json_output=bool(args.json))
    return 0


def _validate(args: Any) -> int:
    recipe = load_experiment(args.path)
    settings = _validated_settings(recipe)
    payload = {
        "ok": True,
        **_metadata(recipe),
        "executable": recipe.status is ExperimentStatus.ACTIVE,
        "settings": settings,
    }
    _emit(payload, json_output=bool(args.json))
    return 0


def cmd_lab(args: Any) -> int:
    try:
        if args.lab_action == "list":
            return _list(args)
        if args.lab_action == "show":
            return _show(args)
        if args.lab_action == "validate":
            return _validate(args)
        raise ValueError(f"unsupported lab action: {args.lab_action}")
    except (OSError, ValueError) as exc:
        if bool(getattr(args, "json", False)):
            _emit({"ok": False, "error": str(exc)}, json_output=True)
        else:
            print(f"error: {exc}")
        return 2
