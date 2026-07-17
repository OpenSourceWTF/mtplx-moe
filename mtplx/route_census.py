"""Decode-route census and automatic island placement (issue #98).

Island pin order used to be a per-model offline artifact: a manual analysis
of a dedicated route trace, hand-transcribed into
``ExpertStreamingModelSpec.island_pin_order``. This module turns that
analysis into a runtime by-product so a brand-new model reaches good island
placement with zero manual trace work, and every session after the first
starts warm from disk.

Lifecycle (v1, confirmed on issue #98):

1. **Collection** — ``ExpertStreamingRuntime.observe_route`` already sees
   every route; when ``ExpertStreamingConfig.route_census`` is enabled (the
   default) the runtime accumulates per-(layer, expert) decode-phase route
   counts in memory. No I/O is added to the decode path.
2. **Evaluation + persistence** — at ``ExpertStreamingRuntime.close()`` the
   in-memory counts merge into ``<model_root>/route-census.json`` (windowed
   decay, below) and the placement *ranking* is re-derived and written to
   ``<model_root>/island-placement.json``. Both writes are atomic
   (tempfile + rename, so concurrent processes and crashes never expose a
   partial file) and best-effort: a write failure logs and never fails
   ``close()``.
3. **Load** — when ``island_layer_count`` is requested for a model whose
   spec has no measured ``island_pin_order``, the runtime resolves the
   count from ``island-placement.json`` instead of refusing. Precedence:
   explicit ``island_layers`` > ``spec.island_pin_order`` > placement file
   > error. The placement file stores a full ranking rather than a
   budget-specific island list, so one file serves any memory limit.

Derivation is exactly the analysis both manual efforts (hy3's measured
order, GLM-5.2's window-1 trace) used: rank layers by **ascending top-K
routed-traffic coverage**, K = the planned slots per layer. The layers
whose top-K experts cover the least decode traffic are the worst
streamed-cache layers and pin first.

Guards:

- The census never changes routing or numerics — placement defaults only.
  Deleting both files reverts exactly to the pre-census behavior.
- A placement derived from fewer than ``ADVISORY_MIN_ROUTED_ASSIGNMENTS``
  routed assignments is marked ``advisory`` and logged as such at load.
- Windowed decay: when the historical census already holds at least
  ``DECAY_MERGE_THRESHOLD`` routed assignments at merge time, every
  historical count is halved (integer floor; emptied entries dropped)
  before the incoming run's counts are added at full weight. Old workload
  shape decays geometrically so drift can re-rank without a manual reset.

v2 (issue #98, deliberately not here): inter-query island refinement —
bounded bank-row swaps between queries so islands track workload drift
without a reload.

This module is intentionally MLX-free and import-safe from I/O workers.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

_LOGGER = logging.getLogger(__name__)

CENSUS_SCHEMA_VERSION = 1
PLACEMENT_SCHEMA_VERSION = 1
ROUTE_CENSUS_FILENAME = "route-census.json"
ISLAND_PLACEMENT_FILENAME = "island-placement.json"
# Below this many routed assignments a derived placement is usable but
# flagged advisory: rankings from a short run can still be noise-ordered.
ADVISORY_MIN_ROUTED_ASSIGNMENTS = 50_000
# Historical counts halve at merge time once the stored census reaches this
# many routed assignments, giving an effective forgetting window of roughly
# twice the threshold while small young censuses accumulate undecayed.
DECAY_MERGE_THRESHOLD = 1_000_000


class RouteCensusError(ValueError):
    """A census or placement artifact is structurally unusable."""


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RouteCensusError(f"{name} must be an exact integer")
    if value < minimum:
        raise RouteCensusError(f"{name} must be at least {minimum}")
    return value


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Publish ``payload`` at ``path`` without ever exposing a partial file.

    The temp file lives in the destination directory so ``os.replace`` is a
    same-filesystem atomic rename; a concurrent writer's rename simply wins
    or loses whole-file (last writer wins, readers never see a torn write).
    """

    directory = path.parent
    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


class RouteCensus:
    """Mergeable per-(layer, expert) decode-route assignment counts.

    ``total_routed_assignments`` always equals the sum of every stored
    count, so the census sample size is exact by construction.
    """

    def __init__(
        self,
        model_key: str,
        *,
        counts: Mapping[int, Mapping[int, int]] | None = None,
        merge_count: int = 0,
        updated_at: str | None = None,
    ) -> None:
        if not isinstance(model_key, str) or not model_key:
            raise RouteCensusError("model_key must be a non-empty string")
        self.model_key = model_key
        self.merge_count = _require_int("merge_count", merge_count)
        self.updated_at = updated_at
        self._counts: dict[int, dict[int, int]] = {}
        self.total_routed_assignments = 0
        if counts:
            for layer, experts in counts.items():
                layer = _require_int("layer", layer)
                for expert, count in experts.items():
                    expert = _require_int("expert", expert)
                    count = _require_int("count", count)
                    if count:
                        self._counts.setdefault(layer, {})[expert] = count
                        self.total_routed_assignments += count

    def observe(self, layer: int, expert_ids: Iterable[int]) -> None:
        """Count one decode route's flattened expert assignments."""

        layer = _require_int("layer", layer)
        experts = [_require_int("expert", expert) for expert in expert_ids]
        if not experts:
            return
        layer_counts = self._counts.setdefault(layer, {})
        for expert in experts:
            layer_counts[expert] = layer_counts.get(expert, 0) + 1
        self.total_routed_assignments += len(experts)

    def counts(self) -> dict[int, dict[int, int]]:
        return {layer: dict(experts) for layer, experts in self._counts.items()}

    def merge(self, incoming: "RouteCensus") -> None:
        """Fold a newer census into this (historical) one, in place.

        Windowed decay: when this census already holds at least
        ``DECAY_MERGE_THRESHOLD`` routed assignments, every historical
        count halves (integer floor, emptied entries dropped) before the
        incoming counts are added at full weight. New data always enters
        undecayed, so the merged census follows workload drift.
        """

        if incoming.model_key != self.model_key:
            raise RouteCensusError(
                "cannot merge censuses for different models: "
                f"{self.model_key!r} vs {incoming.model_key!r}"
            )
        if self.total_routed_assignments >= DECAY_MERGE_THRESHOLD:
            decayed: dict[int, dict[int, int]] = {}
            total = 0
            for layer, experts in self._counts.items():
                kept = {
                    expert: count // 2
                    for expert, count in experts.items()
                    if count // 2
                }
                if kept:
                    decayed[layer] = kept
                    total += sum(kept.values())
            self._counts = decayed
            self.total_routed_assignments = total
        for layer, experts in incoming._counts.items():
            layer_counts = self._counts.setdefault(layer, {})
            for expert, count in experts.items():
                layer_counts[expert] = layer_counts.get(expert, 0) + count
        self.total_routed_assignments += incoming.total_routed_assignments
        self.merge_count += 1

    def to_json_dict(self, *, updated_at: str) -> dict:
        return {
            "schema_version": CENSUS_SCHEMA_VERSION,
            "model_key": self.model_key,
            "total_routed_assignments": self.total_routed_assignments,
            "merge_count": self.merge_count,
            "updated_at": updated_at,
            "counts": {
                str(layer): {
                    str(expert): count
                    for expert, count in sorted(experts.items())
                }
                for layer, experts in sorted(self._counts.items())
            },
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping) -> "RouteCensus":
        if not isinstance(payload, Mapping):
            raise RouteCensusError("census payload must be a JSON object")
        version = payload.get("schema_version")
        if version != CENSUS_SCHEMA_VERSION:
            raise RouteCensusError(
                f"unsupported census schema_version {version!r}"
            )
        raw_counts = payload.get("counts")
        if not isinstance(raw_counts, Mapping):
            raise RouteCensusError("census counts must be a JSON object")
        try:
            counts = {
                int(layer): {
                    int(expert): count for expert, count in experts.items()
                }
                for layer, experts in raw_counts.items()
            }
        except (TypeError, ValueError, AttributeError) as exc:
            raise RouteCensusError(f"census counts are malformed: {exc}") from exc
        census = cls(
            payload.get("model_key", ""),
            counts=counts,
            merge_count=payload.get("merge_count", 0),
            updated_at=payload.get("updated_at"),
        )
        declared = payload.get("total_routed_assignments")
        if declared != census.total_routed_assignments:
            raise RouteCensusError(
                "census total_routed_assignments does not match its counts: "
                f"declared {declared!r}, counted {census.total_routed_assignments}"
            )
        return census


def load_census(path: Path | str) -> RouteCensus:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RouteCensusError(f"could not read census {path}: {exc}") from exc
    return RouteCensus.from_json_dict(payload)


def save_census(census: RouteCensus, path: Path | str, *, updated_at: str) -> None:
    _atomic_write_json(Path(path), census.to_json_dict(updated_at=updated_at))


@dataclass(frozen=True)
class Placement:
    """Derived placement ranking: budget-independent, memory-knob-free.

    ``layer_pin_order`` ranks layers by ascending top-K coverage (worst
    streamed-cache layers first); a count-based island request takes the
    first N entries. ``coverage_by_layer`` is the diagnostic coverage
    curve; ``expert_ranking_by_layer`` ranks each layer's observed experts
    by descending popularity (unobserved experts are omitted) for the v2
    expert-granular placement work.
    """

    model_key: str
    expert_count: int
    slots_per_layer_hint: int
    census_total_routed_assignments: int
    advisory: bool
    layer_pin_order: tuple[int, ...]
    coverage_by_layer: dict[int, float]
    expert_ranking_by_layer: dict[int, tuple[int, ...]]


def derive_placement(
    census: RouteCensus,
    *,
    expert_count: int,
    slots_per_layer_hint: int,
) -> Placement:
    """Rank layers by ascending top-K decode-traffic coverage.

    K is ``min(slots_per_layer_hint, expert_count)`` — the planned
    persistent slots per streamed layer. Ties break on ascending layer id
    (and ascending expert id inside each layer's popularity ranking), so
    the derivation is a pure deterministic function of the census.
    """

    expert_count = _require_int("expert_count", expert_count, minimum=1)
    slots_per_layer_hint = _require_int(
        "slots_per_layer_hint", slots_per_layer_hint, minimum=1
    )
    top_k = min(slots_per_layer_hint, expert_count)
    counts = census.counts()
    coverage: dict[int, float] = {}
    rankings: dict[int, tuple[int, ...]] = {}
    for layer in sorted(counts):
        experts = counts[layer]
        invalid = [expert for expert in experts if expert >= expert_count]
        if invalid:
            raise RouteCensusError(
                f"census layer {layer} routes expert ids {sorted(invalid)[:3]} "
                f"outside expert_count {expert_count}; the census belongs to "
                "a different model geometry"
            )
        total = sum(experts.values())
        if not total:
            continue
        ranking = tuple(
            expert
            for expert, _count in sorted(
                experts.items(), key=lambda item: (-item[1], item[0])
            )
        )
        rankings[layer] = ranking
        covered = sum(experts[expert] for expert in ranking[:top_k])
        coverage[layer] = covered / total
    pin_order = tuple(
        sorted(coverage, key=lambda layer: (coverage[layer], layer))
    )
    return Placement(
        model_key=census.model_key,
        expert_count=expert_count,
        slots_per_layer_hint=slots_per_layer_hint,
        census_total_routed_assignments=census.total_routed_assignments,
        advisory=(
            census.total_routed_assignments < ADVISORY_MIN_ROUTED_ASSIGNMENTS
        ),
        layer_pin_order=pin_order,
        coverage_by_layer=coverage,
        expert_ranking_by_layer=rankings,
    )


def save_placement(
    placement: Placement, path: Path | str, *, updated_at: str
) -> None:
    payload = {
        "schema_version": PLACEMENT_SCHEMA_VERSION,
        "model_key": placement.model_key,
        "expert_count": placement.expert_count,
        "slots_per_layer_hint": placement.slots_per_layer_hint,
        "census_total_routed_assignments": (
            placement.census_total_routed_assignments
        ),
        "advisory": placement.advisory,
        "updated_at": updated_at,
        "layer_pin_order": list(placement.layer_pin_order),
        "coverage_by_layer": {
            str(layer): fraction
            for layer, fraction in sorted(placement.coverage_by_layer.items())
        },
        "expert_ranking_by_layer": {
            str(layer): list(ranking)
            for layer, ranking in sorted(
                placement.expert_ranking_by_layer.items()
            )
        },
    }
    _atomic_write_json(Path(path), payload)


def load_placement(path: Path | str) -> Placement:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RouteCensusError(f"could not read placement {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RouteCensusError("placement payload must be a JSON object")
    version = payload.get("schema_version")
    if version != PLACEMENT_SCHEMA_VERSION:
        raise RouteCensusError(
            f"unsupported placement schema_version {version!r}"
        )
    try:
        pin_order = tuple(
            _require_int("pin order layer", layer)
            for layer in payload["layer_pin_order"]
        )
        coverage = {
            int(layer): float(fraction)
            for layer, fraction in payload["coverage_by_layer"].items()
        }
        rankings = {
            int(layer): tuple(
                _require_int("ranked expert", expert) for expert in ranking
            )
            for layer, ranking in payload["expert_ranking_by_layer"].items()
        }
        placement = Placement(
            model_key=payload["model_key"],
            expert_count=_require_int(
                "expert_count", payload["expert_count"], minimum=1
            ),
            slots_per_layer_hint=_require_int(
                "slots_per_layer_hint",
                payload["slots_per_layer_hint"],
                minimum=1,
            ),
            census_total_routed_assignments=_require_int(
                "census_total_routed_assignments",
                payload["census_total_routed_assignments"],
            ),
            advisory=bool(payload["advisory"]),
            layer_pin_order=pin_order,
            coverage_by_layer=coverage,
            expert_ranking_by_layer=rankings,
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RouteCensusError(f"placement {path} is malformed: {exc}") from exc
    if len(set(placement.layer_pin_order)) != len(placement.layer_pin_order):
        raise RouteCensusError(f"placement {path} repeats pin-order layers")
    return placement
